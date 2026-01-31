# STAC-BUILD: Chunk Processor
# Wraps DA3-Streaming logic for processing individual chunks
# Optimized for RTX 3060 (12GB VRAM) with torch.compile and Vectorization

import os
import sys
import gc
import json
import glob
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from threading import Lock

# Add DA3 paths
DA3_ROOT = Path(__file__).parent.parent.parent / "Depth-Anything-3"
DA3_SRC = DA3_ROOT / "src"
DA3_STREAMING = DA3_ROOT / "da3_streaming"

sys.path.insert(0, str(DA3_SRC))
sys.path.insert(0, str(DA3_STREAMING))

# Lazy imports to avoid loading until needed
DepthAnything3 = None
load_file = None
# Lazy import SAM3 Wrapper
from sam3_wrapper import get_sam3_wrapper


def _lazy_import():
    """Lazy import DA3 modules."""
    global DepthAnything3, load_file
    if DepthAnything3 is None:
        from depth_anything_3.api import DepthAnything3 as _DA3
        from safetensors.torch import load_file as _load_file
        DepthAnything3 = _DA3
        load_file = _load_file


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
    
    # Alignment transform (set by alignment manager)
    sim3_scale: float = 1.0
    sim3_rotation: Optional[np.ndarray] = None  # [3, 3]
    sim3_translation: Optional[np.ndarray] = None  # [3]
    
    # SAM3 Segmentation Masks (frame_idx -> mask_data)
    # Only for keyframes (0, 5, 10...)
    segmentation_masks: Optional[dict] = None


class ChunkProcessor:
    """
    Processes chunks of frames using DA3-LARGE.
    Optimized: Uses torch.compile and vectorized point cloud generation.
    """
    
    # Model configuration - DA3-LARGE is optimized for RTX 3060
    MODEL_NAME = "depth-anything/DA3-LARGE"
    WEIGHTS_DIR = DA3_STREAMING / "weights"
    CONFIG_FILE = WEIGHTS_DIR / "config.json"
    WEIGHTS_FILE = WEIGHTS_DIR / "da3-large.safetensors"
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # Force float16 for RTX 3060 (Tensor Cores)
        self.dtype = torch.float16
        
        self.model = None
        self.is_loaded = False
        self.lock = Lock()
        
        # Check if weights exist locally
        self.use_hf_model = not self.WEIGHTS_FILE.exists()
        if self.use_hf_model:
            print(f"[ChunkProcessor] Local weights not found, will use HuggingFace: {self.MODEL_NAME}")
        else:
            print(f"[ChunkProcessor] Using local weights: {self.WEIGHTS_FILE}")
    
    def load_model(self):
        """Load the DA3 model to GPU and compile it."""
        if self.is_loaded:
            return
        
        with self.lock:
            if self.is_loaded:
                return
            
            _lazy_import()
            
            print(f"[ChunkProcessor] Loading model...")
            start = time.time()
            
            # 1. Load Model
            if self.use_hf_model:
                self.model = DepthAnything3.from_pretrained(self.MODEL_NAME)
                self.model = self.model.to(device=self.device)
            else:
                with open(self.CONFIG_FILE) as f:
                    config = json.load(f)
                self.model = DepthAnything3(**config)
                weight = load_file(str(self.WEIGHTS_FILE))
                self.model.load_state_dict(weight, strict=False)
                self.model.eval()
                self.model = self.model.to(self.device)
            
            # 2. TORCH COMPILE (Critical for RTX 3060 Speed)
            if torch.cuda.is_available():
                print("[ChunkProcessor] Compiling model with torch.compile (mode='reduce-overhead')...")
                print("[ChunkProcessor] First inference will be slower due to compilation.")
                try:
                    # reduce-overhead is best for small batches/real-time
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    print(f"[ChunkProcessor] Warning: torch.compile failed ({e}), using eager mode.")

            self.is_loaded = True
            elapsed = time.time() - start
            print(f"[ChunkProcessor] Model loaded in {elapsed:.1f}s")
    
    def unload_model(self):
        """Unload model from GPU."""
        with self.lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                torch.cuda.empty_cache()
                gc.collect()
                print("[ChunkProcessor] Model unloaded")
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Safely convert Tensor or Array to NumPy float32."""
        if hasattr(data, 'cpu'):
            # Is a Torch Tensor
            return data.float().cpu().numpy()
        elif isinstance(data, np.ndarray):
            # Is already Numpy
            return data.astype(np.float32)
        else:
            # Fallback
            return np.array(data, dtype=np.float32)

    def process_chunk(self, frames_dir: Path, chunk_id: int = 0, prompt: Optional[str] = None) -> Optional[ChunkResult]:
        """
        Process a chunk of frames using FP16 inference.
        """
        if not self.is_loaded:
            self.load_model()
        
        # Find all images
        image_paths = sorted(
            glob.glob(str(frames_dir / "*.jpg")) +
            glob.glob(str(frames_dir / "*.png"))
        )
        
        if len(image_paths) == 0:
            print(f"[ChunkProcessor] No images found in {frames_dir}")
            return None
        
        n_frames = len(image_paths)
        print(f"[ChunkProcessor] Processing chunk {chunk_id}: {n_frames} frames")
        
        start = time.time()
        
        try:
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                # Force AMP (Automatic Mixed Precision) for FP16 speedup
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    predictions = self.model.inference(
                        image_paths,
                        ref_view_strategy='saddle_balanced',
                    )
            
            elapsed = time.time() - start
            print(f"[ChunkProcessor] Inference done in {elapsed:.1f}s ({elapsed/n_frames:.3f}s/frame)")
            
            depths = np.squeeze(self._to_numpy(predictions.depth))  # [N, H, W]
            confs = self._to_numpy(predictions.conf).copy()
            extrinsics = self._to_numpy(predictions.extrinsics)
            intrinsics = self._to_numpy(predictions.intrinsics)
            images = predictions.processed_images
            
            if np.min(confs) == np.max(confs):
                confs = np.ones_like(confs)
            else:
                confs = confs - 1.0
            
            # --- SAM3 SEGMENTATION ---
            sam3_masks = None
            if prompt:
                print(f"\n[ChunkProcessor] 🟢 STARTING SAM3 SEGMENTATION for chunk {chunk_id}")
                print(f"[ChunkProcessor] Prompt: '{prompt}'")
                
                # MEMORY OPTIMIZATION: Unload DA3 before loading SAM3
                # This prevents both models from being in VRAM simultaneously
                if self.is_loaded:
                    print("[ChunkProcessor] ⏳ Unloading DA3 to free VRAM for SAM3...")
                    self.unload_model()
                
                t0_sam = time.time()
                try:
                    sam3 = get_sam3_wrapper()
                    # frames_dir is a Path, convert to str
                    sam3_masks = sam3.process_chunk(str(frames_dir), prompt, keyframe_interval=5)
                    print(f"[ChunkProcessor] 🏁 SAM3 DONE in {time.time()-t0_sam:.2f}s. Masks found: {len(sam3_masks) if sam3_masks else 0}")
                    
                    # Unload SAM3 after use to prepare for next DA3 load
                    sam3.unload_model()
                    print("[ChunkProcessor] ✅ SAM3 unloaded, VRAM cleared for next chunk")
                except Exception as e:
                    print(f"[ChunkProcessor] ❌ SAM3 Error: {e}")
            # -------------------------
            
            result = ChunkResult(
                chunk_id=chunk_id,
                frame_count=n_frames,
                depths=depths.astype(np.float32),
                confs=confs.astype(np.float32),
                extrinsics=extrinsics.astype(np.float32),
                intrinsics=intrinsics.astype(np.float32),
                images=images,
                segmentation_masks=sam3_masks,
            )
            return result
            
        except Exception as e:
            print(f"[ChunkProcessor] Error processing chunk: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_segmentation_for_ply(self, points: np.ndarray, metadata: dict, masks: dict) -> dict:
        """
        Compute segmentation indices for existing PLY from masks.
        
        Uses validity_mapping from metadata for accurate pixel-to-point correlation:
        - validity_mapping contains valid_pixel_indices for each keyframe
        - For each mask pixel that is True, check if that pixel index is in valid_pixel_indices
        - If yes, map to the corresponding point index
        """
        import cv2
        
        try:
            validity_mapping = metadata.get("validity_mapping")
            
            if not validity_mapping:
                print("[ChunkProcessor] No validity_mapping in metadata - cannot compute segmentation")
                return {}
            
            print(f"[Debug] Using validity_mapping with {len(validity_mapping)} keyframes")
            print(f"[Debug] Masks available for frames: {sorted(masks.keys())}")
            
            # Output: { "obj_id": [point_indices] }
            segments_sets = {}
            
            # Track cumulative point offset for global indexing
            cumulative_offset = 0
            
            for kf_entry in validity_mapping:
                orig_frame_idx = kf_entry['orig_frame_idx']
                valid_pixel_indices = kf_entry['valid_pixel_indices']  # List of flat pixel indices that became points
                
                points_in_frame = len(valid_pixel_indices)
                
                # Check if we have a mask for this keyframe
                if orig_frame_idx not in masks:
                    cumulative_offset += points_in_frame
                    continue
                
                frame_mask_data = masks[orig_frame_idx]
                if not isinstance(frame_mask_data, dict) or 'out_binary_masks' not in frame_mask_data:
                    cumulative_offset += points_in_frame
                    continue
                
                raw_mask = frame_mask_data['out_binary_masks']
                obj_ids = frame_mask_data.get('out_obj_ids', [1])
                
                if hasattr(raw_mask, 'cpu'):
                    raw_mask = raw_mask.cpu().numpy()
                
                if raw_mask.ndim == 2:
                    raw_mask = raw_mask[None, ...]
                
                if raw_mask.size == 0:
                    cumulative_offset += points_in_frame
                    continue
                
                H_mask, W_mask = raw_mask.shape[1], raw_mask.shape[2]
                mask_pixel_count = H_mask * W_mask
                
                # Estimate depth resolution from valid_pixel_indices
                # The max index tells us H*W approximately
                max_valid_idx = max(valid_pixel_indices) if valid_pixel_indices else 0
                estimated_hw = max_valid_idx + 1
                
                # If mask resolution differs from depth, we need to resize
                if mask_pixel_count != estimated_hw:
                    # Need to scale mask indices to depth resolution
                    scale_factor = estimated_hw / mask_pixel_count
                else:
                    scale_factor = 1.0
                
                # Build reverse lookup: pixel_idx -> local_point_idx
                pixel_to_local_point = {pixel_idx: local_idx for local_idx, pixel_idx in enumerate(valid_pixel_indices)}
                
                # Process each object mask
                for obj_idx, obj_id in enumerate(obj_ids):
                    if obj_idx >= len(raw_mask):
                        continue
                    if obj_id == 0:
                        continue
                    
                    obj_mask = raw_mask[obj_idx]  # [H_mask, W_mask]
                    
                    # Get flat indices where mask is True
                    mask_flat = obj_mask.flatten()
                    mask_hit_indices = np.where(mask_flat > 0)[0]
                    
                    if len(mask_hit_indices) == 0:
                        continue
                    
                    # Scale mask indices to depth resolution if needed
                    if scale_factor != 1.0:
                        scaled_mask_indices = (mask_hit_indices * scale_factor).astype(int)
                    else:
                        scaled_mask_indices = mask_hit_indices
                    
                    # Find which mask hits correspond to valid points
                    matched_global_indices = []
                    for mask_pixel_idx in scaled_mask_indices:
                        if mask_pixel_idx in pixel_to_local_point:
                            local_point_idx = pixel_to_local_point[mask_pixel_idx]
                            global_point_idx = cumulative_offset + local_point_idx
                            matched_global_indices.append(global_point_idx)
                    
                    if matched_global_indices:
                        if str(obj_id) not in segments_sets:
                            segments_sets[str(obj_id)] = set()
                        segments_sets[str(obj_id)].update(matched_global_indices)
                    
                    print(f"[Debug] KF {orig_frame_idx} obj {obj_id}: mask_hits={len(mask_hit_indices)}, matched_points={len(matched_global_indices)}")
                
                cumulative_offset += points_in_frame
            
            # Convert sets to lists
            result = {k: list(v) for k, v in segments_sets.items()}
            total_segmented = sum(len(v) for v in result.values())
            print(f"[Debug] Total segmented: {total_segmented} points in {len(result)} objects")
            return result

        except Exception as e:
            print(f"[ChunkProcessor] Error in compute_segmentation_for_ply: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_point_cloud(self, result: ChunkResult, 
                            sample_ratio: float = 0.1,
                            conf_threshold_coef: float = 0.75) -> np.ndarray:
        """
        Generate point cloud using VECTORIZED NumPy operations.
        KEYFRAME OPTIMIZATION: Processes only 1 out of every 5 frames.
        """
        t0 = time.time()
        
        # --- NUEVA LÓGICA DE KEYFRAMES ---
        # Seleccionamos los índices: 0, 5, 10, etc.
        indices_keyframes = np.arange(0, result.frame_count, 5)
        
        # Filtramos los datos del resultado para usar solo esos frames
        depths = result.depths[indices_keyframes]
        confs = result.confs[indices_keyframes]
        intrinsics = result.intrinsics[indices_keyframes]
        extrinsics = result.extrinsics[indices_keyframes]
        
        if len(result.images.shape) == 4:
            images = result.images[indices_keyframes]
        elif isinstance(result.images, list):
            images = [result.images[i] for i in indices_keyframes]
        else:
            images = result.images # Caso fallback
        # ---------------------------------

        N, H, W = depths.shape # Ahora N es la cantidad de keyframes
        
        # 1. Compute Confidence Threshold globally
        valid_confs = confs[confs > 0]
        if len(valid_confs) > 0:
            mean_conf = np.mean(valid_confs)
            conf_threshold = max(0.0, mean_conf * conf_threshold_coef)
        else:
            conf_threshold = 0.0
            
        # 2. Create Grid (Vectorized)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = np.tile(u[None, ...], (N, 1, 1)).astype(np.float32)
        v = np.tile(v[None, ...], (N, 1, 1)).astype(np.float32)
        
        # 3. Extract Intrinsics
        fx = intrinsics[:, 0, 0][:, None, None]
        fy = intrinsics[:, 1, 1][:, None, None]
        cx = intrinsics[:, 0, 2][:, None, None]
        cy = intrinsics[:, 1, 2][:, None, None]
        
        # 4. Unproject
        z = depths
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points_cam = np.stack([x, y, z], axis=-1)
        points_flat = points_cam.reshape(N, -1, 3)
        
        # 5. Transform to World
        w2c = np.eye(4, dtype=np.float32)[None, ...].repeat(N, axis=0)
        w2c[:, :3, :4] = extrinsics
        
        c2w = np.linalg.inv(w2c)
        R = c2w[:, :3, :3]
        t = c2w[:, :3, 3]
        
        points_world = np.einsum('nij,nlj->nli', R, points_flat) + t[:, None, :]
        
        # 6. Colors and Masking
        if isinstance(images, np.ndarray) and len(images.shape) == 4:
             colors_flat = images.reshape(N, -1, 3).astype(np.float32) / 255.0
        elif isinstance(images, list):
             colors_flat = np.array(images).reshape(N, -1, 3).astype(np.float32) / 255.0
        else:
             colors_flat = np.zeros_like(points_world)

        confs_flat = confs.reshape(N, -1)
        depths_flat = depths.reshape(N, -1)
        
        points_world_all = points_world.reshape(-1, 3)
        colors_all = colors_flat.reshape(-1, 3)
        confs_all = confs_flat.reshape(-1)
        depths_all = depths_flat.reshape(-1)
        
        # Filter
        mask = (confs_all >= conf_threshold) & (depths_all > 0.1) & (depths_all < 100)
        
        valid_points = points_world_all[mask]
        valid_colors = colors_all[mask]
        
        # 7. Sampling
        n_points = len(valid_points)
        if sample_ratio < 1.0 and n_points > 0:
            n_samples = int(n_points * sample_ratio)
            indices = np.random.choice(n_points, n_samples, replace=False)
            valid_points = valid_points[indices]
            valid_colors = valid_colors[indices]
        
        # 8. Combine
        if len(valid_points) > 0:
            point_cloud = np.concatenate([valid_points, valid_colors], axis=1).astype(np.float32)
        else:
            point_cloud = np.zeros((0, 6), dtype=np.float32)
            
        print(f"[ChunkProcessor] Generated {len(point_cloud)} points (from {N} keyframes) in {time.time()-t0:.3f}s")
        
        result.point_cloud = point_cloud
        return point_cloud


# Singleton instance
_processor: Optional[ChunkProcessor] = None


def get_chunk_processor() -> ChunkProcessor:
    """Get or create the singleton ChunkProcessor."""
    global _processor
    if _processor is None:
        _processor = ChunkProcessor()
    return _processor