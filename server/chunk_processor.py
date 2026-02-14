# STAC-BUILD: Chunk Processor with DA3_Streaming
# Uses DA3_Streaming engine for 100% consistent alignment with offline mode

import os
import sys
import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from threading import Lock

# Centralised vendor path resolution (replaces hardcoded DA3 paths)
import vendor_paths
from config import cfg

# DA3 is optional - falls back to MASt3R-SLAM if not available
DA3_AVAILABLE = False
DA3_Streaming = None
depth_to_point_cloud_vectorized = None

try:
    from da3_streaming.da3_streaming import DA3_Streaming as _DA3_Streaming
    DA3_Streaming = _DA3_Streaming
    DA3_AVAILABLE = True
    print("[ChunkProcessor] DA3_Streaming loaded successfully")
    
    try:
        from da3_streaming.da3_streaming import depth_to_point_cloud_vectorized as _depth_to_point_cloud_vectorized
        depth_to_point_cloud_vectorized = _depth_to_point_cloud_vectorized
    except ImportError:
        pass
        
except ImportError as e:
    # Only warn if DA3 backend is selected (mast3r/hybrid use DA3 via subprocess)
    from config import cfg as _cfg
    if _cfg.get("slam_backend", "mast3r") == "da3":
        print(f"[ChunkProcessor] DA3_Streaming not available: {e}")
        print("[ChunkProcessor] Using MASt3R-SLAM backend only")

# Fallback implementation if DA3 not available
if depth_to_point_cloud_vectorized is None:
    def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
        import torch
        input_is_numpy = False
        if isinstance(depth, np.ndarray):
            input_is_numpy = True
            depth_tensor = torch.tensor(depth, dtype=torch.float32)
            intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
            extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
        else:
            depth_tensor = depth
            intrinsics_tensor = intrinsics
            extrinsics_tensor = extrinsics

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)

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


        if input_is_numpy:
            point_cloud_world = point_cloud_world.cpu().numpy()

        return point_cloud_world

# DA3 utility imports (optional)
accumulate_sim3_transforms = None
save_confident_pointcloud_batch = None
apply_sim3_direct_torch = None

try:
    from loop_utils.sim3utils import accumulate_sim3_transforms as _accum
    from loop_utils.alignment_torch import apply_sim3_direct_torch as _apply
    accumulate_sim3_transforms = _accum
    apply_sim3_direct_torch = _apply
    
    try:
        from loop_utils.sim3utils import save_confident_pointcloud_batch as _save
        save_confident_pointcloud_batch = _save
    except ImportError:
        pass
except ImportError as e:
    from config import cfg as _cfg
    if _cfg.get("slam_backend", "mast3r") == "da3":
        print(f"[ChunkProcessor] loop_utils not available: {e}")

from sam3_wrapper import get_sam3_wrapper
from config import cfg


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk_id: int
    frame_count: int

    # Per-frame data
    depths: np.ndarray  # [N, H, W] float32
    confs: np.ndarray  # [N, H, W] float32
    extrinsics: np.ndarray  # [N, 3, 4] float32 (w2c)
    intrinsics: np.ndarray  # [N, 3, 3] float32
    images: np.ndarray  # [N, H, W, 3] uint8

    # Point cloud (optional, generated after processing)
    point_cloud: Optional[np.ndarray] = None  # [M, 6] xyz + rgb

    # Alignment transform (set by alignment manager or this processor)
    sim3_scale: float = 1.0
    sim3_rotation: Optional[np.ndarray] = None  # [3, 3]
    sim3_translation: Optional[np.ndarray] = None  # [3]

    # SAM3 Segmentation Masks
    segmentation_masks: Optional[dict] = None


class ChunkProcessorStreaming:
    """
    Chunk processor that uses DA3_Streaming for alignment.
    Maintains compatibility with existing ChunkProcessor interface.
    """

    def __init__(self, device: str = None):
        self.device = device or cfg.get("models", {}).get("depth", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.lock = Lock()

        # DA3_Streaming instance (created per session)
        self.da3_streaming = None
        self.is_loaded = False

        # Chunk tracking
        self.chunk_count = 0
        self.previous_predictions = None
        self.sim3_list = []  # Accumulated transforms

        print(f"[ChunkProcessorStreaming] Initialized (device={self.device})")

    def load_model(self):
        """Lazy load - DA3_Streaming will be created when needed."""
        if not self.is_loaded:
            print("[ChunkProcessorStreaming] Model will be loaded on first chunk")
            self.is_loaded = True

    def unload_model(self):
        """Unload DA3_Streaming instance."""
        with self.lock:
            if self.da3_streaming is not None:
                # DA3_Streaming holds the model internally
                del self.da3_streaming.model
                del self.da3_streaming
                self.da3_streaming = None
                torch.cuda.empty_cache()
                gc.collect()
                print("[ChunkProcessorStreaming] Model unloaded")
                self.is_loaded = False

    def _init_da3_streaming(self, frames_dir: Path, output_dir: Path):
        """Initialize DA3_Streaming for a new session."""
        # Build config from config.yaml + HF cache (all from config.yaml, zero hardcoding)
        from da3_config_builder import build_da3_config
        da3_config = build_da3_config(cfg)

        # Create DA3_Streaming instance
        self.da3_streaming = DA3_Streaming(
            image_dir=str(frames_dir),
            save_dir=str(output_dir),
            config=da3_config
        )

        # Initialize chunk tracking
        self.chunk_count = 0
        self.previous_predictions = None
        self.sim3_list = []

        print(f"[ChunkProcessorStreaming] DA3_Streaming initialized for {frames_dir}")

    def process_chunk(self, frames_dir: Path, chunk_id: int = 0, prompt: Optional[str] = None) -> Optional[ChunkResult]:
        """
        Process a chunk using DA3_Streaming.

        Args:
            frames_dir: Directory containing chunk frames
            chunk_id: Chunk identifier
            prompt: Text prompt for SAM3 segmentation

        Returns:
            ChunkResult with aligned point cloud
        """
        if not self.is_loaded:
            self.load_model()

        # Initialize DA3_Streaming on first chunk
        if self.da3_streaming is None:
            output_dir = frames_dir.parent.parent / "output"
            self._init_da3_streaming(frames_dir.parent, output_dir)

        # Get frame paths
        import glob
        image_paths = sorted(
            glob.glob(str(frames_dir / "*.jpg")) +
            glob.glob(str(frames_dir / "*.png"))
        )

        if len(image_paths) == 0:
            print(f"[ChunkProcessorStreaming] No images found in {frames_dir}")
            return None

        n_frames = len(image_paths)
        print(f"[ChunkProcessorStreaming] Processing chunk {chunk_id}: {n_frames} frames")

        # Determine chunk range for DA3_Streaming
        # DA3_Streaming expects continuous frame indices
        chunk_range = (chunk_id * (cfg["server"]["chunk_size"] - cfg["server"]["chunk_overlap"]),
                       chunk_id * (cfg["server"]["chunk_size"] - cfg["server"]["chunk_overlap"]) + n_frames)

        # Override img_list to only include this chunk's frames
        self.da3_streaming.img_list = image_paths
        
        # FIX: Populate chunk_indices which is required by process_single_chunk
        # Use a dict to support arbitrary chunk_ids without sparse list
        self.da3_streaming.chunk_indices = {chunk_id: chunk_range}

        start = time.time()

        try:
            # Process chunk using DA3_Streaming
            predictions = self.da3_streaming.process_single_chunk(
                range_1=(0, n_frames),  # Process all frames in this directory
                chunk_idx=chunk_id,
                is_loop=False
            )

            elapsed = time.time() - start
            print(f"[ChunkProcessorStreaming] Inference done in {elapsed:.1f}s ({elapsed/n_frames:.3f}s/frame)")

            # Extract numpy arrays
            depths = np.squeeze(predictions.depth)
            confs = predictions.conf.copy()
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            images = predictions.processed_images

            # Handle uniform confidence
            if np.min(confs) == np.max(confs):
                confs = np.ones_like(confs)
            else:
                confs = confs - 1.0

            # Align with previous chunk if not first
            sim3_transform = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

            if chunk_id > 0 and self.previous_predictions is not None:
                print(f"[ChunkProcessorStreaming] Aligning chunk {chunk_id} with {chunk_id-1}")

                # Use DA3_Streaming's alignment logic
                overlap = cfg["server"]["chunk_overlap"]

                point_map1 = depth_to_point_cloud_vectorized(
                    self.previous_predictions.depth,
                    self.previous_predictions.intrinsics,
                    self.previous_predictions.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    predictions.depth, predictions.intrinsics, predictions.extrinsics
                )

                point_map1 = point_map1[-overlap:]
                point_map2 = point_map2[:overlap]
                conf1 = self.previous_predictions.conf[-overlap:]
                conf2 = predictions.conf[:overlap]

                # Prepare depth data for scale computation
                if cfg["alignment"]["method"] == "scale+se3":
                    chunk1_depth = np.squeeze(self.previous_predictions.depth[-overlap:])
                    chunk2_depth = np.squeeze(predictions.depth[:overlap])
                    chunk1_depth_conf = np.squeeze(self.previous_predictions.conf[-overlap:])
                    chunk2_depth_conf = np.squeeze(predictions.conf[:overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                # Align using DA3_Streaming's method
                s, R, t = self.da3_streaming.align_2pcds(
                    point_map1, conf1, point_map2, conf2,
                    chunk1_depth, chunk2_depth, chunk1_depth_conf, chunk2_depth_conf
                )

                self.sim3_list.append((s, R, t))
                
                # FIX: Sync with internal DA3_Streaming object so save_camera_poses works
                if hasattr(self.da3_streaming, 'sim3_list'):
                    self.da3_streaming.sim3_list.append((s, R, t))

                # Accumulate transforms
                accumulated = accumulate_sim3_transforms(self.sim3_list)
                sim3_transform = accumulated[-1]

            self.previous_predictions = predictions

            # --- SAM3 SEGMENTATION ---
            # NOTE: SAM3 is NO LONGER run inline with DA3 to avoid VRAM conflicts
            # Segmentation is queued and runs AFTER all DA3 chunks complete
            # See main.py _run_pending_retroactive() for the deferred segmentation logic
            sam3_masks = None
            if prompt:
                print(f"[ChunkProcessorStreaming] Prompt '{prompt}' received - SAM3 will run AFTER DA3 completes all chunks")

            # Create result
            result = ChunkResult(
                chunk_id=chunk_id,
                frame_count=n_frames,
                depths=depths.astype(np.float32),
                confs=confs.astype(np.float32),
                extrinsics=extrinsics.astype(np.float32),
                intrinsics=intrinsics.astype(np.float32),
                images=images,
                segmentation_masks=sam3_masks,
                sim3_scale=sim3_transform[0],
                sim3_rotation=sim3_transform[1],
                sim3_translation=sim3_transform[2]
            )

            self.chunk_count += 1
            return result

        except Exception as e:
            print(f"[ChunkProcessorStreaming] Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            return None

    def finalize_session(self):
        """Finalize the streaming session by saving camera poses and intrinsics."""
        if self.da3_streaming:
            print("[ChunkProcessor] Finalizing session: saving camera poses...")
            try:
                # This calling da3_streaming.save_camera_poses() which saves:
                # - camera_poses.ply
                # - camera_poses.txt
                # - intrinsic.txt
                self.da3_streaming.save_camera_poses()
            except Exception as e:
                print(f"[ChunkProcessor] Error finalising session: {e}")

    def generate_point_cloud(self, result: ChunkResult, sample_ratio: float = None, conf_threshold_coef: float = None) -> np.ndarray:
        """
        Generate aligned point cloud for a chunk.
        Uses DA3_Streaming's point cloud generation logic.
        """
        t0 = time.time()

        # Use config defaults if not provided
        if sample_ratio is None:
            # FIX: Key is 'models' not 'model'
            sample_ratio = cfg["models"]["da3"]["pointcloud_save"]["sample_ratio"]
        if conf_threshold_coef is None:
            conf_threshold_coef = cfg["models"]["da3"]["pointcloud_save"]["conf_threshold_coef"]

        # Generate point cloud in local frame
        point_map = depth_to_point_cloud_vectorized(
            result.depths, result.intrinsics, result.extrinsics
        )

        # Apply SIM3 transform
        if result.sim3_scale != 1.0 or not np.allclose(result.sim3_rotation, np.eye(3)):
            s, R, t = result.sim3_scale, result.sim3_rotation, result.sim3_translation

            # Convert to torch for efficient transformation
            point_map_torch = torch.from_numpy(point_map).float()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            point_map_torch = point_map_torch.to(device)

            # FIX: Convert all params to torch tensors on device
            R_torch = torch.from_numpy(R).float().to(device)
            t_torch = torch.from_numpy(t).float().to(device)
            s_torch = torch.tensor(float(s), dtype=torch.float32, device=device)

            # Correct import based on PYTHONPATH setup (da3_streaming dir is in path)
            try:
                from loop_utils.alignment_torch import apply_sim3_direct_torch
            except ImportError:
                # Fallback just in case
                from da3_streaming.loop_utils.alignment_torch import apply_sim3_direct_torch
            
            point_map_torch = apply_sim3_direct_torch(point_map_torch, s_torch, R_torch, t_torch)
            point_map = point_map_torch.cpu().numpy()

        # Flatten
        points_flat = point_map.reshape(-1, 3)
        colors_flat = result.images.reshape(-1, 3).astype(np.float32) / 255.0
        confs_flat = result.confs.reshape(-1)

        # Apply confidence filtering
        conf_threshold = max(0.0, np.mean(confs_flat[confs_flat > 0]) * conf_threshold_coef)
        mask = confs_flat >= conf_threshold

        valid_points = points_flat[mask]
        valid_colors = colors_flat[mask]

        # Sampling
        n_points = len(valid_points)
        if sample_ratio < 1.0 and n_points > 0:
            n_samples = int(n_points * sample_ratio)
            indices = np.random.choice(n_points, n_samples, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]

        # Combine
        if len(valid_points) > 0:
            point_cloud = np.concatenate([valid_points, valid_colors], axis=1).astype(np.float32)
        else:
            point_cloud = np.zeros((0, 6), dtype=np.float32)

        print(f"[ChunkProcessorStreaming] Generated {len(point_cloud)} points in {time.time()-t0:.3f}s")

        result.point_cloud = point_cloud
        return point_cloud

    def compute_segmentation_for_ply(self, ply_points: np.ndarray, meta: dict, masks: dict,
                                       sample_indices: np.ndarray = None, chunk_idx: int = 0) -> dict:
        """
        Compute segmentation for PLY points using direct 2D→3D mapping via sample_indices.
        Falls back to 3D→2D projection if sample_indices not available (legacy sessions).

        Args:
            ply_points: [N, 3] or [N, 6] array of world points
            meta: Metadata dict containing resolution info and cameras
            masks: Dictionary of {frame_idx: {mask_data}} from SAM3
            sample_indices: [N] array mapping each PLY point to flattened (N_frames * H * W) index
            chunk_idx: Which chunk this is (for frame_global calculation)

        Returns:
            Dict mapping segment_id -> list of point indices
        """
        if ply_points is None or len(ply_points) == 0:
            return {}

        n_points = len(ply_points)

        # Check if we have sample_indices for direct mapping (preferred method)
        if sample_indices is not None and len(sample_indices) == n_points:
            return self._compute_segmentation_direct(ply_points, meta, masks, sample_indices, chunk_idx)
        else:
            print(f"[SegCompute] sample_indices not available, falling back to projection method")
            return self._compute_segmentation_projection(ply_points, meta, masks)

    def _compute_segmentation_direct(self, ply_points: np.ndarray, meta: dict, masks: dict,
                                      sample_indices: np.ndarray, chunk_idx: int) -> dict:
        """
        Direct 2D→3D segmentation using sample_indices (ACCURATE method).
        Each PLY point has a sample_indices value that maps back to the original frame/pixel.
        """
        n_points = len(ply_points)

        # Get resolutions from metadata (NO hardcoding)
        scaled_res = meta.get("scaled_resolution")  # DA3 output resolution [H, W]
        original_res = meta.get("original_resolution")  # Original image resolution [H, W]

        if not scaled_res:
            print(f"[SegCompute-Direct] ERROR: scaled_resolution not in metadata!")
            return {}

        H_da3, W_da3 = scaled_res[0], scaled_res[1]

        # If original_resolution not in meta, infer from mask shape
        if not original_res and masks:
            first_mask_key = next(iter(masks))
            first_mask = masks[first_mask_key]
            if "out_binary_masks" in first_mask:
                bm = first_mask["out_binary_masks"]
                if hasattr(bm, "cpu"):
                    bm = bm.cpu().numpy()
                if bm.ndim == 3:
                    H_orig, W_orig = bm.shape[1], bm.shape[2]
                else:
                    H_orig, W_orig = bm.shape[0], bm.shape[1]
                original_res = [H_orig, W_orig]
                print(f"[SegCompute-Direct] Inferred original_resolution from mask: {original_res}")

        if not original_res:
            print(f"[SegCompute-Direct] ERROR: Cannot determine original_resolution!")
            return {}

        H_orig, W_orig = original_res[0], original_res[1]

        # Calculate scale factors
        scale_v = H_orig / H_da3
        scale_u = W_orig / W_da3

        # Get chunk step from config (NOT hardcoded)
        chunk_size = cfg["server"]["chunk_size"]
        chunk_overlap = cfg["server"]["chunk_overlap"]
        chunk_step = chunk_size - chunk_overlap

        # Frame count in this chunk
        frame_count = meta.get("frame_count", chunk_size)

        print(f"[SegCompute-Direct] Processing {n_points} points")
        print(f"[SegCompute-Direct] DA3 resolution: {H_da3}x{W_da3}, Original: {H_orig}x{W_orig}")
        print(f"[SegCompute-Direct] Scale factors: v={scale_v:.3f}, u={scale_u:.3f}")
        print(f"[SegCompute-Direct] Chunk {chunk_idx}, step={chunk_step}, frame_count={frame_count}")
        print(f"[SegCompute-Direct] Mask frames available: {sorted(masks.keys())}")

        # Initialize all points as unsegmented
        point_segment_ids = np.full(n_points, -1, dtype=np.int32)

        # Convert sample_indices to numpy if needed
        if isinstance(sample_indices, list):
            sample_indices = np.array(sample_indices, dtype=np.int64)

        # Precompute frame/pixel from sample_indices (vectorized)
        pixels_per_frame = H_da3 * W_da3
        frame_local = sample_indices // pixels_per_frame
        pixel_idx = sample_indices % pixels_per_frame
        v_scaled = pixel_idx // W_da3
        u_scaled = pixel_idx % W_da3

        # Scale to original resolution
        v_orig = (v_scaled * scale_v).astype(np.int32)
        u_orig = (u_scaled * scale_u).astype(np.int32)

        # Clamp to bounds (safety)
        v_orig = np.clip(v_orig, 0, H_orig - 1)
        u_orig = np.clip(u_orig, 0, W_orig - 1)

        # Convert frame_local to frame_global
        frame_global = chunk_idx * chunk_step + frame_local

        # Debug: show frame distribution
        unique_frames, frame_counts = np.unique(frame_global, return_counts=True)
        print(f"[SegCompute-Direct] Points distributed across frames {unique_frames.min()}-{unique_frames.max()}")

        # Process each mask frame
        points_segmented = 0
        for frame_idx, mask_entry in masks.items():
            if "out_binary_masks" not in mask_entry:
                continue

            binary_masks = mask_entry["out_binary_masks"]
            if hasattr(binary_masks, "cpu"):
                binary_masks = binary_masks.cpu().numpy()

            if binary_masks.ndim == 2:
                binary_masks = binary_masks[None, ...]

            if binary_masks.size == 0 or binary_masks.shape[0] == 0:
                continue

            # Find points that belong to this frame
            point_mask = (frame_global == frame_idx)
            n_points_in_frame = np.sum(point_mask)

            if n_points_in_frame == 0:
                continue

            # Get pixel coordinates for points in this frame
            v_frame = v_orig[point_mask]
            u_frame = u_orig[point_mask]
            point_indices_frame = np.where(point_mask)[0]

            # Check each object mask
            for obj_idx in range(binary_masks.shape[0]):
                mask_obj = binary_masks[obj_idx]

                # Resize mask if needed (shouldn't be necessary if original_res is correct)
                if mask_obj.shape[0] != H_orig or mask_obj.shape[1] != W_orig:
                    import cv2
                    mask_obj = cv2.resize(mask_obj.astype(np.float32), (W_orig, H_orig),
                                         interpolation=cv2.INTER_NEAREST)

                # Check which points hit the mask
                hits = mask_obj[v_frame, u_frame] > 0
                n_hits = np.sum(hits)

                if n_hits > 0:
                    # Assign segment ID (obj_idx + 1 to avoid 0)
                    hit_indices = point_indices_frame[hits]
                    # Only assign if not already segmented
                    unsegmented = point_segment_ids[hit_indices] == -1
                    point_segment_ids[hit_indices[unsegmented]] = obj_idx + 1
                    points_segmented += np.sum(unsegmented)

        print(f"[SegCompute-Direct] Segmented {points_segmented}/{n_points} points")

        # Build segments dict
        segments = {}
        unique_ids = np.unique(point_segment_ids)

        for seg_id in unique_ids:
            if seg_id == -1:
                continue
            indices = np.where(point_segment_ids == seg_id)[0].tolist()
            segments[str(seg_id)] = indices

        total_segmented = sum(len(v) for v in segments.values())
        print(f"[SegCompute-Direct] Found {len(segments)} segments with {total_segmented} total points")
        return segments

    def _compute_segmentation_projection(self, ply_points: np.ndarray, meta: dict, masks: dict) -> dict:
        """
        Legacy 3D→2D projection method (fallback for sessions without sample_indices).
        Less accurate but maintains backward compatibility.
        """
        n_points = len(ply_points)
        points_world = ply_points[:, :3].copy()

        # Check for alignment transform
        align_meta = meta.get("alignment_transform")
        ply_pre_aligned = meta.get("ply_pre_aligned", False)

        if align_meta and ply_pre_aligned:
            print(f"[SegCompute-Proj] PLY is pre-aligned. Applying inverse transform...")
            s = align_meta.get("scale", 1.0)
            R_align = np.array(align_meta.get("rotation", np.eye(3)), dtype=np.float32)
            t_align = np.array(align_meta.get("translation", np.zeros(3)), dtype=np.float32)
            points_world = (points_world - t_align) / s
            points_world = points_world @ R_align

        point_segment_ids = np.full(n_points, -1, dtype=np.int32)

        # Get cameras
        cameras = meta.get("cameras", {})
        if not cameras and "extrinsics" in meta and "intrinsics" in meta:
            for i in range(meta.get("frame_count", 0)):
                cameras[str(i)] = {
                    "extrinsics": meta["extrinsics"][i],
                    "intrinsics": meta["intrinsics"][i]
                }

        print(f"[SegCompute-Proj] Computing segments for {n_points} points using {len(masks)} mask frames...")

        frames_processed = 0
        for frame_idx_str, camera_info in cameras.items():
            frame_idx = int(frame_idx_str)

            if frame_idx not in masks:
                continue

            mask_entry = masks[frame_idx]
            if "out_binary_masks" not in mask_entry:
                continue

            binary_masks = mask_entry["out_binary_masks"]
            if hasattr(binary_masks, "cpu"):
                binary_masks = binary_masks.cpu().numpy()

            if binary_masks.ndim == 2:
                binary_masks = binary_masks[None, ...]

            if binary_masks.size == 0 or binary_masks.shape[0] == 0:
                continue

            frames_processed += 1
            H, W = binary_masks.shape[1], binary_masks.shape[2]

            w2c = np.array(camera_info["extrinsics"], dtype=np.float32)
            if w2c.shape == (3, 4):
                w2c = np.vstack([w2c, [0,0,0,1]])

            K = np.array(camera_info["intrinsics"], dtype=np.float32)

            # Scale intrinsics if needed
            scaled_res = meta.get("scaled_resolution")
            if scaled_res and len(scaled_res) == 2:
                scaled_h, scaled_w = scaled_res
                if scaled_h > 0 and scaled_w > 0 and (scaled_h != H or scaled_w != W):
                    scale_h = H / scaled_h
                    scale_w = W / scaled_w
                    K[0, 0] *= scale_w
                    K[1, 1] *= scale_h
                    K[0, 2] *= scale_w
                    K[1, 2] *= scale_h

            R = w2c[:3, :3]
            t = w2c[:3, 3]
            pts_cam = points_world @ R.T + t

            valid_z = pts_cam[:, 2] > 0.1
            pts_proj = pts_cam[valid_z]

            u = (pts_proj[:, 0] * K[0, 0] / pts_proj[:, 2]) + K[0, 2]
            v = (pts_proj[:, 1] * K[1, 1] / pts_proj[:, 2]) + K[1, 2]

            u = np.round(u).astype(np.int32)
            v = np.round(v).astype(np.int32)

            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            valid_indices = np.where(valid_z)[0][in_bounds]
            u_valid = u[in_bounds]
            v_valid = v[in_bounds]

            for obj_idx in range(binary_masks.shape[0]):
                mask_obj = binary_masks[obj_idx]
                hits = mask_obj[v_valid, u_valid] > 0
                hit_point_indices = valid_indices[hits]
                unsegmented_hits = point_segment_ids[hit_point_indices] == -1
                point_segment_ids[hit_point_indices[unsegmented_hits]] = obj_idx + 1

        print(f"[SegCompute-Proj] Processed {frames_processed} frames with masks.")

        segments = {}
        unique_ids = np.unique(point_segment_ids)

        for seg_id in unique_ids:
            if seg_id == -1:
                continue
            indices = np.where(point_segment_ids == seg_id)[0].tolist()
            segments[str(seg_id)] = indices

        total_segmented = sum(len(v) for v in segments.values())
        print(f"[SegCompute-Proj] Found {len(segments)} segments with {total_segmented} total points")
        return segments


# Alias for backward compatibility
ChunkProcessor = ChunkProcessorStreaming

# Singleton
_processor: Optional[ChunkProcessorStreaming] = None

def get_chunk_processor() -> ChunkProcessorStreaming:
    """Get or create the singleton ChunkProcessorStreaming."""
    global _processor
    if _processor is None:
        _processor = ChunkProcessorStreaming(
            device=cfg.get("models", {}).get("depth", {}).get("device", None)
        )
    return _processor
