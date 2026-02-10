# STAC-BUILD: Alignment Manager
# Optimized Phase 3.7: RANSAC Auto-Leveling (FINAL GRAVITY FIX)
# Target Vector Inverted to account for main.py coordinate flip.

import sys
import gc
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from threading import Lock

# Add DA3 paths
DA3_ROOT = Path(__file__).parent.parent.parent / "Depth-Anything-3"
DA3_STREAMING = DA3_ROOT / "da3_streaming"
sys.path.insert(0, str(DA3_STREAMING))

try:
    from loop_utils.alignment_torch import robust_weighted_estimate_sim3_torch
    from loop_utils.sim3utils import precompute_scale_chunks_with_depth
    DA3_IMPORTS_OK = True
except ImportError:
    DA3_IMPORTS_OK = False

from config import cfg

def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    depth_tensor = torch.tensor(depth, dtype=torch.float32)
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
    extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)
    elif torch.cuda.is_available():
        depth_tensor = depth_tensor.cuda()
        intrinsics_tensor = intrinsics_tensor.cuda()
        extrinsics_tensor = extrinsics_tensor.cuda()

    N, H, W = depth_tensor.shape
    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    return point_cloud_world.float().cpu().numpy()

def weighted_align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold, precompute_scale=None):
    # Force float32 to avoid bfloat16 numpy incompatibility
    point_map1 = np.asarray(point_map1, dtype=np.float32)
    point_map2 = np.asarray(point_map2, dtype=np.float32)
    conf1 = np.asarray(conf1, dtype=np.float32)
    conf2 = np.asarray(conf2, dtype=np.float32)
    
    if precompute_scale is not None:
        point_map2 = point_map2 * precompute_scale
    
    b = min(point_map1.shape[0], point_map2.shape[0])
    aligned_points1, aligned_points2, confidence_weights = [], [], []

    for i in range(b):
        mask = (conf1[i] > conf_threshold) & (conf2[i] > conf_threshold)
        idx = np.where(mask)
        if len(idx[0]) == 0: continue
        aligned_points1.append(point_map1[i][idx])
        aligned_points2.append(point_map2[i][idx])
        confidence_weights.append(np.sqrt(conf1[i][idx] * conf2[i][idx]))

    if len(aligned_points1) == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    if DA3_IMPORTS_OK:
        align_method = cfg["alignment"]["method"]
        s_refine, R, t = robust_weighted_estimate_sim3_torch(
            all_pts2.astype(np.float32), all_pts1.astype(np.float32), all_weights.astype(np.float32),
            delta=cfg["alignment"]["ransac"]["delta"],
            max_iters=cfg["alignment"]["ransac"]["max_iters"],
            tol=cfg["alignment"]["ransac"]["tolerance"],
            align_method=align_method,
        )
        
        # Combine scales: final_s = s_pre * s_refine
        final_s = s_refine * (precompute_scale if precompute_scale is not None else 1.0)
        return final_s, R, t

    return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

@dataclass
class AlignedChunk:
    chunk_id: int
    point_cloud: Optional[np.ndarray] = None
    sample_indices: Optional[np.ndarray] = None  # Indices used during sampling - MUST be reused for segmentation

class AlignmentManager:
    def __init__(self, overlap_frames: int = 10):
        self.overlap_frames = overlap_frames
        self.aligned_chunks: List[AlignedChunk] = []
        self.lock = Lock()
        self.chunk_data_list: List[dict] = []
        self.sim3_list = []
        self.accumulated_transforms = []
        self.gravity_correction = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        self.next_global_id = 1
        self.last_chunk_last_frame_masks = None
    
    def reset(self):
        with self.lock:
            self.aligned_chunks.clear()
            self.chunk_data_list.clear()
            self.sim3_list.clear()
            self.accumulated_transforms.clear()
            self.gravity_correction = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            self.next_global_id = 1
            self.last_chunk_last_frame_masks = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[AlignmentManager] Reset")
    
    def add_chunk(self, chunk_result) -> Optional[AlignedChunk]:
        """
        Add a pre-aligned chunk from ChunkProcessor.
        ChunkProcessor now handles alignment via DA3_Streaming.
        AlignmentManager only applies gravity correction.
        """
        with self.lock:
            chunk_id = len(self.chunk_data_list)
            chunk_data = {
                'depths': chunk_result.depths, 'confs': chunk_result.confs,
                'intrinsics': chunk_result.intrinsics, 'extrinsics': chunk_result.extrinsics,
                'images': chunk_result.images, 'frame_count': chunk_result.frame_count,
                'masks': chunk_result.segmentation_masks
            }
            self.chunk_data_list.append(chunk_data)

            # ChunkProcessor provides pre-aligned point cloud with SIM3 transform
            # We just need to apply gravity correction on first chunk
            if chunk_id == 0:
                self._compute_initial_leveling(chunk_data)
                # Apply gravity correction to the pre-generated point cloud
                if chunk_result.point_cloud is not None:
                    point_cloud = self._apply_gravity_correction(chunk_result.point_cloud)
                else:
                    # Fallback if point cloud not yet generated
                    point_cloud = self._generate_point_cloud(
                        chunk_data, *self.gravity_correction, return_validity_info=False
                    )

                aligned = AlignedChunk(chunk_id=0, point_cloud=point_cloud, sample_indices=None)
                self.aligned_chunks.append(aligned)
                return aligned
            else:
                # Chunks after first: already aligned by ChunkProcessor
                # Just apply gravity correction
                if chunk_result.point_cloud is not None:
                    point_cloud = self._apply_gravity_correction(chunk_result.point_cloud)
                else:
                    # Fallback: generate with identity transform + gravity
                    s_g, R_g, t_g = self.gravity_correction
                    point_cloud = self._generate_point_cloud(
                        chunk_data, s_g, R_g, t_g, return_validity_info=False
                    )

                aligned = AlignedChunk(chunk_id=chunk_id, point_cloud=point_cloud, sample_indices=None)
                self.aligned_chunks.append(aligned)
                return aligned

    def _apply_gravity_correction(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Apply gravity correction to an already aligned point cloud.
        point_cloud: [N, 6] (XYZ + RGB)
        """
        if point_cloud is None or len(point_cloud) == 0:
            return point_cloud

        s_g, R_g, t_g = self.gravity_correction

        # Extract points
        points = point_cloud[:, :3]

        # Apply transform: P' = s * (R @ P) + t
        points_corrected = s_g * (points @ R_g.T) + t_g

        # Recombine with colors
        return np.concatenate([points_corrected, point_cloud[:, 3:]], axis=1).astype(np.float32)

    def _compute_initial_leveling(self, chunk_data):
        """
        Detecta el suelo usando RANSAC y lo alinea para que la gravedad sea (0, 1, 0) en Python.
        Al ser invertido por main.py (y=-y), la gravedad final en el visor será (0, -1, 0) [Correcto].
        """
        print("[AlignmentManager] Auto-leveling Chunk 0...")
        
        # Load Config - STRICT ACCESS (No defaults)
        # User requested elimination of hardcoded fallbacks.
        if "alignment" not in cfg or "auto_leveling" not in cfg["alignment"]:
            print("[AlignmentManager] ⚠️ Config missing 'alignment.auto_leveling'. Skipping.")
            return

        al_cfg = cfg["alignment"]["auto_leveling"]
        
        if not al_cfg["enabled"]:
             print("[AlignmentManager] Auto-leveling disabled in config.")
             return

        sample_ratio = al_cfg["sample_ratio"]
        min_points_ransac = al_cfg["min_points"]
        ransac_iters = al_cfg["ransac_iters"]
        ransac_thresh = al_cfg["ransac_threshold"]
        gravity_dot_thresh = al_cfg["gravity_check_threshold"]
        target_gravity = np.array(al_cfg["target_gravity"], dtype=np.float32)

        # 1. Generar nube temporal
        points_raw = self._generate_point_cloud(
            chunk_data, 1.0, np.eye(3), np.zeros(3), sample_ratio=sample_ratio, ignore_masks=True
        )[:, :3]
        
        if len(points_raw) < min_points_ransac:
            print("[Align] Not enough points for RANSAC")
            return

        # 2. Vector 'Abajo' de la cámara (para asegurar que no agarramos una pared)
        w2c = chunk_data['extrinsics']
        w2c_4x4 = np.eye(4, dtype=np.float32)[None, ...].repeat(len(w2c), axis=0)
        w2c_4x4[:, :3, :4] = w2c
        c2w = np.linalg.inv(w2c_4x4)
        avg_cam_down = np.mean(c2w[:, :3, 1], axis=0)
        avg_cam_down /= np.linalg.norm(avg_cam_down)

        # 3. RANSAC
        best_plane = None
        max_inliers = 0
        n_pts = len(points_raw)
        
        for _ in range(ransac_iters):
            idxs = np.random.choice(n_pts, 3, replace=False)
            p1, p2, p3 = points_raw[idxs]
            
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6: continue
            normal /= norm
            
            # Filtro de pared (debe ser paralelo a la gravedad de la cámara)
            if abs(np.dot(normal, avg_cam_down)) < gravity_dot_thresh:
                continue
                
            d = -np.dot(normal, p1)
            dists = np.abs(points_raw @ normal + d)
            inliers = np.sum(dists < ransac_thresh)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_plane = (normal, d)

        if best_plane is None:
            print("[Align] Could not find a valid floor plane.")
            return

        normal, d_val = best_plane
        
        # FIX: Ensure normal points UP (opposite to gravity/cam_down)
        # avg_cam_down points to the floor. We want normal to point AWAY from floor (UP).
        if np.dot(normal, avg_cam_down) > 0:
            normal = -normal
            d_val = -d_val
            
        print(f"[Align] Floor Normal (UP): {normal}")

        # 4. Calcular Rotación (LA CORRECCIÓN FINAL)
        # Queremos que la gravedad (normal) apunte a +Y (0, 1, 0) en Python.
        target_down = target_gravity
        
        v = np.cross(normal, target_down)
        s = np.linalg.norm(v)
        c = np.dot(normal, target_down)
        
        if s < 1e-6:
            R_align = np.eye(3, dtype=np.float32)
            if c < 0: R_align = np.diag([1, -1, 1]).astype(np.float32)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))
        
        R_align = R_align.astype(np.float32)

        # 5. Calcular Traslación (Llevar suelo a Y=0)
        dists = np.abs(points_raw @ normal + d_val)
        inliers_mask = dists < ransac_thresh
        if np.sum(inliers_mask) > 0:
            centroid = np.mean(points_raw[inliers_mask], axis=0)
            centroid_rot = R_align @ centroid
            # Ajustamos para que la altura del suelo sea 0
            t_align = np.array([0, -centroid_rot[1], 0], dtype=np.float32)
            # Centrar en XZ también
            t_align[0] = -centroid_rot[0]
            t_align[2] = -centroid_rot[2]
        else:
            t_align = np.zeros(3, dtype=np.float32)

        print(f"[Align] Gravity correction applied. Target: +Y (will be -Y in viewer).")
        self.gravity_correction = (1.0, R_align, t_align)

    def compute_leveling_from_points(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Standalone auto-leveling for point clouds (legacy session support).
        Computes Sim3 to align the largest plane (floor) to Y=0.
        Returns (s=1.0, R, t).
        """
        if points is None or len(points) < 100:
             return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        # 1. Subsample
        if len(points) > 5000:
             idx = np.random.choice(len(points), 5000, replace=False)
             pts_sub = points[idx, :3]
        else:
             pts_sub = points[:, :3]

        # 2. RANSAC for Plane
        best_plane = None
        max_inliers = 0
        # Config defaults if not available
        thresh = cfg.get("alignment", {}).get("auto_leveling", {}).get("ransac_threshold", 0.05)
        
        for _ in range(50): # Fast iterations
             idxs = np.random.choice(len(pts_sub), 3, replace=False)
             p1, p2, p3 = pts_sub[idxs]
             v1 = p2 - p1
             v2 = p3 - p1
             normal = np.cross(v1, v2)
             norm = np.linalg.norm(normal)
             if norm < 1e-6: continue
             normal /= norm
             
             d = -np.dot(normal, p1)
             dists = np.abs(pts_sub @ normal + d)
             inliers = np.sum(dists < thresh)
             if inliers > max_inliers:
                 max_inliers = inliers
                 best_plane = (normal, d)

        if not best_plane or max_inliers < len(pts_sub) * 0.1:
             print("[AlignStandalone] No floor plane found.")
             return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        normal, d = best_plane

        # 3. Orient UP
        # Assume most points are ABOVE the floor.
        signed_dists = pts_sub @ normal + d
        if np.mean(signed_dists) > 0:
             # Points are on positive side. Normal points towards points (UP).
             pass 
        else:
             # Points are on negative side. Normal points down. Flip.
             normal = -normal
             d = -d
        
        # 4. Compute Rotation to (0, 1, 0)
        target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c = np.dot(normal, target)
        
        if s < 1e-6:
             R = np.eye(3, dtype=np.float32) 
             if c < 0: # 180 deg flip
                 # Flip around X
                 R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        else:
             vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
             R = np.eye(3, dtype=np.float32) + vx + (vx @ vx) * ((1 - c) / (s**2))

        # 5. Translation t=(0, d, 0)
        t = np.array([0, d, 0], dtype=np.float32)
        
        print(f"[AlignStandalone] Correction found: d={d:.3f}")
        return 1.0, R, t

    def _add_first_chunk(self, chunk_data) -> AlignedChunk:
        s, R, t = self.gravity_correction
        # CRITICAL: Capture sample_indices to reuse for segmentation
        point_cloud, validity_mapping, sample_indices = self._generate_point_cloud(
            chunk_data, s, R, t, return_validity_info=True
        )
        aligned = AlignedChunk(chunk_id=0, point_cloud=point_cloud, sample_indices=sample_indices)
        self.aligned_chunks.append(aligned)
        return aligned
    
    def _align_to_previous(self, chunk_data, chunk_id) -> AlignedChunk:
        prev_data = self.chunk_data_list[chunk_id - 1]
        curr_data = chunk_data
        overlap = min(self.overlap_frames, prev_data['frame_count'], curr_data['frame_count'])
        
        point_map1 = depth_to_point_cloud_vectorized(prev_data['depths'][-overlap:], prev_data['intrinsics'][-overlap:], prev_data['extrinsics'][-overlap:])
        point_map2 = depth_to_point_cloud_vectorized(curr_data['depths'][:overlap], curr_data['intrinsics'][:overlap], curr_data['extrinsics'][:overlap])
        conf1 = prev_data['confs'][-overlap:]
        conf2 = curr_data['confs'][:overlap]
        
        try:
            print(f"[Align] aligning chunk {chunk_id} to {chunk_id-1} with overlap {overlap}")
            
            # 1. Dynamic Threshold Logic (DA3 Reference: MIN Logic)
            med1 = np.median(conf1) if conf1.size > 0 else 0
            med2 = np.median(conf2) if conf2.size > 0 else 0
            
            thresh_coef = cfg["point_cloud"]["generation"]["dynamic_threshold_coef"]
            thresh_min = cfg["point_cloud"]["generation"]["dynamic_threshold_min"]
            
            # Use min() so if EITHER chunk is blurry, we lower the bar to find points.
            dynamic_threshold = min(med1, med2) * thresh_coef
            dynamic_threshold = max(dynamic_threshold, thresh_min) # Safety floor
            print(f"[Align] Dynamic Threshold: {dynamic_threshold:.5f} (Medians: {med1:.4f}, {med2:.4f})")
            
            # 2. Robust Scale Precomputation (RANSAC on Depths)
            s_init = 1.0
            if DA3_IMPORTS_OK:
                # Prepare depths for precompute
                d1_sq = np.squeeze(prev_data['depths'][-overlap:])
                c1_sq = np.squeeze(prev_data['confs'][-overlap:])
                d2_sq = np.squeeze(curr_data['depths'][:overlap])
                c2_sq = np.squeeze(curr_data['confs'][:overlap])
                
                s_init, quality, method = precompute_scale_chunks_with_depth(
                    d1_sq, c1_sq, d2_sq, c2_sq, method="auto"
                )
                print(f"[Align] Precomputed Scale: {s_init:.4f} (Quality: {quality:.4f}, Method: {method})")
            
            # 3. Weighted Alignment with Pre-Scale
            s_rel, R_rel, t_rel = weighted_align_point_maps(
                point_map1, conf1, point_map2, conf2, 
                conf_threshold=dynamic_threshold,
                precompute_scale=s_init
            )
            
            print(f"[Align] Final Transform: s={s_rel:.4f}")
            
        except Exception as e:
            print(f"[Align] Alignment Failed for Chunk {chunk_id}: {e}")
            import traceback
            traceback.print_exc()
            s_rel, R_rel, t_rel = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        
        self.sim3_list.append((s_rel, R_rel, t_rel))
        
        acc_transforms = self._accumulate_sim3_transforms(self.sim3_list)
        s_acc, R_acc, t_acc = acc_transforms[-1]
        
        s_g, R_g, t_g = self.gravity_correction
        
        s_final = s_g * s_acc
        R_final = R_g @ R_acc
        t_final = s_g * (R_g @ t_acc) + t_g
        
        # CRITICAL: Capture sample_indices to reuse for segmentation
        point_cloud, validity_mapping, sample_indices = self._generate_point_cloud(
            chunk_data, s_final, R_final, t_final, return_validity_info=True
        )
        aligned = AlignedChunk(chunk_id=chunk_id, point_cloud=point_cloud, sample_indices=sample_indices)
        self.aligned_chunks.append(aligned)
        return aligned

    def update_chunk_masks(self, chunk_id: int, masks: dict) -> AlignedChunk:
        """
        Update the masks for a specific chunk and regenerate its point cloud.
        Used for retroactive prompting.
        """
        with self.lock:
            if chunk_id >= len(self.chunk_data_list):
                return None
            
            # Update data
            self.chunk_data_list[chunk_id]['masks'] = masks
            
            # Re-generate cloud using stored transforms
            # We need to find the transform used for this chunk.
            # Chunk 0 uses gravity_correction.
            # Chunk N uses gravity * stored_sim3_accumulator.
            
            s_g, R_g, t_g = self.gravity_correction
            
            if chunk_id == 0:
                s_final, R_final, t_final = s_g, R_g, t_g
            else:
                # We need to re-compute or access stored accumulated transform
                # self.accumulated_transforms stores accum from 1..N
                # Index 0 of accum corresponds to Chunk 1 (relative to 0?)
                # internal list: accumulated_transforms = _accumulate_sim3_transforms(self.sim3_list)
                # self.sim3_list has N-1 entries (Chunk 1..N).
                # chunk_id 1 is at index 0 of sim3_list.
                
                # Let's just re-calculate to be safe/stateless
                # But sim3_list is the source of truth.
                if self.sim3_list and (chunk_id - 1) < len(self.accumulated_transforms):
                    # accumulated_transforms[i] is the transform for Chunk i+1 ??
                    # _accumulate returns list [T1, T1*T2, etc]
                    # chunk 1 uses transforms[0]
                    # chunk k uses transforms[k-1]
                    s_acc, R_acc, t_acc = self.accumulated_transforms[chunk_id - 1]
                    
                    s_final = s_g * s_acc
                    R_final = R_g @ R_acc
                    t_final = s_g * (R_g @ t_acc) + t_g
                else:
                    # Should not happen if logic matches
                    s_final, R_final, t_final = s_g, R_g, t_g

            point_cloud = self._generate_point_cloud(
                self.chunk_data_list[chunk_id], s_final, R_final, t_final, 
                id_mapping=self._compute_id_mapping(masks)
            )
            
            # Update AlignedChunk
            self.aligned_chunks[chunk_id].point_cloud = point_cloud
            return self.aligned_chunks[chunk_id]

    def _compute_id_mapping(self, current_chunk_masks: dict) -> dict:
        """
        Computes Local ID -> Global ID mapping based on IoU with previous chunk.
        current_chunk_masks: dict {frame_idx: {'pred_masks': ...}}
        """
        # 1. Get first valid frame masks from current chunk
        sorted_frames = sorted(current_chunk_masks.keys())
        if not sorted_frames: return {}
        
        first_frame_idx = sorted_frames[0]
        last_frame_idx = sorted_frames[-1]
        
        # Extract binary masks [N, H, W]
        # Assumes inputs are tensors or numpy
        def get_masks_tensor(mask_entry):
            if 'pred_masks' not in mask_entry: return None
            m = mask_entry['pred_masks']
            if hasattr(m, 'cpu'): m = m.cpu().numpy()
            # If shape is [H, W], make it [1, H, W]
            if m.ndim == 2: m = m[None, ...]
            # Boolean
            return m > 0

        curr_first_masks = get_masks_tensor(current_chunk_masks[first_frame_idx])
        curr_last_masks = get_masks_tensor(current_chunk_masks[last_frame_idx])
        
        if curr_first_masks is None: return {}
        
        n_curr = curr_first_masks.shape[0]
        id_map = {} # local_idx -> global_id
        assigned_globals = set()
        
        # 2. Match with Previous Chunk
        if self.last_chunk_last_frame_masks is not None:
             prev_masks = self.last_chunk_last_frame_masks['masks'] # [M, H, W]
             prev_ids = self.last_chunk_last_frame_masks['ids']     # [M] IDs
             
             # Compute IoU matrix
             # simple implementation
             for i in range(n_curr):
                 curr_m = curr_first_masks[i]
                 best_iou = 0.0
                 best_match_idx = -1
                 
                 for j in range(len(prev_ids)):
                     prev_m = prev_masks[j]
                     # IoU
                     intersection = np.sum(curr_m & prev_m)
                     union = np.sum(curr_m | prev_m)
                     if union > 0:
                         iou = intersection / union
                         if iou > 0.5 and iou > best_iou:
                             best_iou = iou
                             best_match_idx = j
                             
                 if best_match_idx != -1:
                     gid = prev_ids[best_match_idx]
                     id_map[i] = gid
                     assigned_globals.add(gid)
        
        # 3. Assign new Global IDs to unmatched
        for i in range(n_curr):
            if i not in id_map:
                id_map[i] = self.next_global_id
                self.next_global_id += 1
        
        # 4. Store Last Frame for next iteration
        if curr_last_masks is not None:
            # We need to store global IDs corresponding to these masks.
            # Assuming SAM3 tracks consistency WITHIN the chunk, 
            # local ID 'i' at first frame is same as local ID 'i' at last frame.
            # So we reuse the id_map.
            
            # Filter only present objects? For now keep all.
            self.last_chunk_last_frame_masks = {
                'masks': curr_last_masks,
                'ids': [id_map.get(k, self.next_global_id) for k in range(curr_last_masks.shape[0])]
            }
            
        print(f"[Align] ID Mapping resolved: {id_map}")
        return id_map


    def _accumulate_sim3_transforms(self, transforms):
        if not transforms: return []
        cumulative = [transforms[0]]
        for i in range(1, len(transforms)):
            s_prev, R_prev, t_prev = cumulative[i - 1]
            s_next, R_next, t_next = transforms[i]
            R_cum = R_prev @ R_next
            s_cum = s_prev * s_next
            t_cum = s_prev * (R_prev @ t_next) + t_prev
            cumulative.append((s_cum, R_cum, t_cum))
        return cumulative

    def _generate_point_cloud(self, chunk_data, s, R, t, sample_ratio=0.1, ignore_masks=False, id_mapping=None, return_validity_info=False):
        # --- KEYFRAME FILTERING (match ChunkProcessor.generate_point_cloud) ---
        keyframe_interval = 5
        frame_count = chunk_data['frame_count']
        keyframe_indices = list(range(0, frame_count, keyframe_interval))  # [0, 5, 10, 15, 20, 25]
        
        # Filter all data to keyframes only
        depths_kf = chunk_data['depths'][keyframe_indices]
        confs_kf = chunk_data['confs'][keyframe_indices]
        intrinsics_kf = chunk_data['intrinsics'][keyframe_indices]
        extrinsics_kf = chunk_data['extrinsics'][keyframe_indices]
        
        if isinstance(chunk_data['images'], np.ndarray) and len(chunk_data['images'].shape) == 4:
            images_kf = chunk_data['images'][keyframe_indices]
        elif isinstance(chunk_data['images'], list):
            images_kf = [chunk_data['images'][i] for i in keyframe_indices]
        else:
            images_kf = chunk_data['images']  # fallback
        
        # Get masks keyed by ORIGINAL frame index
        masks_orig = chunk_data.get('masks', {}) if not ignore_masks else {}
        # -------------------------------------------------------------------
        
        point_map = depth_to_point_cloud_vectorized(depths_kf, intrinsics_kf, extrinsics_kf)
        shape = point_map.shape
        pts_flat = point_map.reshape(-1, 3)
        pts_transformed = s * (pts_flat @ R.T) + t 
        point_map_aligned = pts_transformed.reshape(shape)
        
        colors = images_kf
        confs = confs_kf
        
        all_points, all_colors = [], []
        all_point_indices = []  # Mapeo: cada punto → índice global (frame_local * H*W + pixel)
        validity_mapping = []  # For each keyframe: list of valid flat pixel indices
        conf_threshold = max(0.0, np.mean(confs[confs > 0]) * 0.5)
        
        # Iterate over keyframes (enumerate gives 0,1,2... but we also need original frame index)
        for kf_idx, orig_frame_idx in enumerate(keyframe_indices):
            pts = point_map_aligned[kf_idx].reshape(-1, 3)
            clrs = colors[kf_idx].reshape(-1, 3).astype(np.float32) / 255.0
            conf = confs[kf_idx].flatten()
            depth = depths_kf[kf_idx].flatten()
            
            mask = (conf >= conf_threshold) & (depth > 0.1) & (depth < 100)
            
            # --- Retroactive Masking / Highlighting ---
            # User wants "Highlight" effect, not isolation.
            # Strategy: Keep ALL points. If mask exists:
            # - Points IN mask: Keep original color (or boost?)
            # - Points OUT mask: Dim / Desaturate (make background)
            
            is_highlighting = False
            highlight_indices = None # Boolean mask for current batch 'pts'
            
            if masks_orig and orig_frame_idx in masks_orig:
                frame_mask_data = masks_orig[orig_frame_idx]
                # Key is 'out_binary_masks'
                if isinstance(frame_mask_data, dict) and 'out_binary_masks' in frame_mask_data:
                    raw_mask = frame_mask_data['out_binary_masks']
                    if hasattr(raw_mask, 'cpu'): raw_mask = raw_mask.cpu().numpy()
                    
                    # FIX 1: Resize Mask to match Depth Resolution if needed
                    # raw_mask: [N, H_orig, W_orig] or [H_orig, W_orig]
                    # depth: [H_depth, W_depth]
                    
                    # Get original 2D shape from keyframe-filtered depths
                    tgt_h, tgt_w = depths_kf[kf_idx].shape
                    
                    # Ensure raw_mask is [N, H, W] for consistent processing
                    if raw_mask.ndim == 2:
                        raw_mask = raw_mask[None, ...] # Add batch dim -> [1, H, W]
                        
                    # CHECK IF EMPTY
                    if raw_mask.size == 0 or raw_mask.shape[0] == 0:
                        # Skip this mask if it's empty
                        continue

                    # Resize if dimensions differ
                    if raw_mask.shape[1] != tgt_h or raw_mask.shape[2] != tgt_w:
                        # Iterate channels to resize (cv2.resize expects H,W image)
                        resized_masks = []
                        for m_channel in raw_mask:
                            # cv2.resize((width, height))
                            # m_channel is float/bool? It should be float for resizing or uint8
                            m_resized = cv2.resize(m_channel.astype(np.float32), (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
                            resized_masks.append(m_resized)
                        raw_mask = np.stack(resized_masks, axis=0)
                    
                    # Now raw_mask matches depth resolution [N, H_depth, W_depth]
                    
                    # Flatten for highlighting logic
                    # For binary "is object" check, we take MAX across objects
                    objects_combined = np.max(raw_mask, axis=0) # [H, W]
                    m_flat = objects_combined.flatten()
                    
                    if m_flat.shape[0] == mask.shape[0]:
                        is_highlighting = True
                        highlight_indices = (m_flat > 0)
                            
            # Filter Validity - keep original colors only
            valid_flat_indices = np.where(mask)[0]  # Pixel indices that became points
            valid_pts = pts[mask]
            valid_clrs = clrs[mask]
            
            all_points.append(valid_pts)
            all_colors.append(valid_clrs)
            
            # Calcular índices globales para mapeo de segmentación
            # Índice global = frame_local * (H * W) + pixel_index
            H, W = depths_kf[kf_idx].shape
            pixels_per_frame = H * W
            global_indices = kf_idx * pixels_per_frame + valid_flat_indices
            all_point_indices.append(global_indices)
            
            validity_mapping.append({
                'orig_frame_idx': orig_frame_idx,
                'valid_pixel_indices': valid_flat_indices.tolist()
            })
            
        all_points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3))
        all_colors = np.concatenate(all_colors, axis=0) if all_colors else np.empty((0, 3))
        all_point_indices = np.concatenate(all_point_indices, axis=0) if all_point_indices else np.empty((0,), dtype=np.int64)
        
        # sample_indices ahora contiene el mapeo global (punto PLY → índice en frame*H*W)
        sample_indices = all_point_indices
        
        if sample_ratio < 1.0 and len(all_points) > 0:
            n_samples = int(len(all_points) * sample_ratio)
            subsample_indices = np.sort(np.random.choice(len(all_points), n_samples, replace=False))
            all_points = all_points[subsample_indices]
            all_colors = all_colors[subsample_indices]
            # CRÍTICO: También subsamplueamos el mapeo de índices
            sample_indices = sample_indices[subsample_indices]
            
        # Output: XYZ (3) + RGB (3) = 6 channels, original colors
        point_cloud = np.concatenate([all_points, all_colors], axis=1).astype(np.float32)
        
        if return_validity_info:
            return point_cloud, validity_mapping, sample_indices
        return point_cloud

    def generate_validity_mapping(self, chunk_id: int, sample_ratio: float = 0.1) -> tuple:
        """
        Generate validity mapping for a chunk (which pixels became points).
        Returns tuple: (validity_mapping, sample_indices)
        - validity_mapping: list of dicts [{orig_frame_idx, valid_pixel_indices}, ...]
        - sample_indices: numpy array of indices kept after sampling, or None if no sampling
        """
        if chunk_id >= len(self.chunk_data_list):
            return [], None
        
        chunk_data = self.chunk_data_list[chunk_id]
        
        # Get appropriate transform for this chunk
        s_g, R_g, t_g = self.gravity_correction
        
        if chunk_id == 0:
            s, R, t = s_g, R_g, t_g
        else:
            if self.accumulated_transforms and (chunk_id - 1) < len(self.accumulated_transforms):
                s_acc, R_acc, t_acc = self.accumulated_transforms[chunk_id - 1]
                s = s_g * s_acc
                R = R_g @ R_acc
                t = s_g * (R_g @ t_acc) + t_g
            else:
                s, R, t = s_g, R_g, t_g
        
        # Generate point cloud WITH validity info (use same sample_ratio as PLY)
        _, validity_mapping, sample_indices = self._generate_point_cloud(
            chunk_data, s, R, t, 
            sample_ratio=sample_ratio,
            return_validity_info=True
        )
        
        return validity_mapping, sample_indices

    def get_unified_cloud(self): return None 
    def get_chunk_count(self): return len(self.aligned_chunks)
    def get_total_points(self): return sum(len(c.point_cloud) for c in self.aligned_chunks if c.point_cloud is not None)

_manager = None
def get_alignment_manager():
    global _manager
    if _manager is None: _manager = AlignmentManager()
    return _manager