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

# Centralised vendor path resolution
import vendor_paths

from config import cfg


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
        self._leveling_cache = None  # Cache for compute_leveling_from_points
    
    def reset(self):
        with self.lock:
            self.aligned_chunks.clear()
            self.chunk_data_list.clear()
            self.sim3_list.clear()
            self.accumulated_transforms.clear()
            self.gravity_correction = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            self.next_global_id = 1
            self.last_chunk_last_frame_masks = None
            self._leveling_cache = None
            torch.cuda.empty_cache()
            gc.collect()
            print("[AlignmentManager] Reset")
    


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
        
        # Return cached result if available (ensures OBB uses same transform as cloud)
        if self._leveling_cache is not None:
            print("[AlignStandalone] Using cached leveling transform")
            return self._leveling_cache

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
        result = (1.0, R, t)
        self._leveling_cache = result  # Cache for reuse
        return result

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