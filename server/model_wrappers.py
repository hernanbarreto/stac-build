# STAC-BUILD: Model Wrappers
# Rewritten DA3 integration using batch processing for coherent poses
# Optimized for RTX 3060 (12GB VRAM)

import sys
import time
import numpy as np
import gc
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from collections import deque
from PIL import Image
from threading import Lock

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# Centralised vendor path resolution
import vendor_paths


def to_numpy(x):
    """Helper to convert tensor or array to numpy."""
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.asarray(x)


class DA3Wrapper:
    """
    Depth Anything 3 Wrapper with batch processing for coherent poses.
    Uses accumulated frames for joint depth/pose estimation.
    """
    
    MODEL_NAME = "depth-anything/DA3-LARGE"  # User specified
    PROCESS_RES = 504  # Resolution for processing
    MIN_BATCH_SIZE = 3  # Minimum frames for good pose estimation
    MAX_BATCH_SIZE = 10  # Maximum to avoid OOM
    
    def __init__(self, device: str = "cuda"):
        self.device = device if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.is_loaded = False
        self.use_mock = False
        
        # Frame buffer for batch processing
        self.frame_buffer: deque = deque(maxlen=self.MAX_BATCH_SIZE)
        self.frame_timestamps: deque = deque(maxlen=self.MAX_BATCH_SIZE)
        
        # Results cache
        self.last_prediction = None
        self.last_batch_frames = []
        
        # Global pose tracking (accumulated across batches)
        self.global_c2w = np.eye(4, dtype=np.float32)  # Current camera pose in world
        self.world_scale = 1.0  # Scale factor for metric consistency
        
        # Thread safety
        self.lock = Lock()
        
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load DA3."""
        try:
            from depth_anything_3.api import DepthAnything3
            self.model_class = DepthAnything3
            print("[DA3] Model class loaded successfully")
        except ImportError as e:
            print(f"[DA3] Import failed: {e}. Using MOCK mode.")
            self.use_mock = True
    
    def load_to_gpu(self):
        """Load the model to GPU memory."""
        if self.use_mock:
            self.is_loaded = True
            print("[DA3-MOCK] Simulating GPU load")
            return
        
        if not self.is_loaded and self.model is None:
            print(f"[DA3] Loading {self.MODEL_NAME}...")
            start = time.time()
            
            try:
                self.model = self.model_class.from_pretrained(self.MODEL_NAME)
                self.model = self.model.to(device=self.device)
                self.is_loaded = True
                print(f"[DA3] Model loaded to GPU in {time.time()-start:.1f}s")
            except Exception as e:
                print(f"[DA3] Failed to load: {e}")
                print("[DA3] Falling back to MOCK mode")
                self.use_mock = True
                self.is_loaded = True
    
    def unload_from_gpu(self):
        """Unload model from GPU to free VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            if HAS_TORCH:
                torch.cuda.empty_cache()
            gc.collect()
            print("[DA3] Model unloaded from GPU")
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """Add a frame to the buffer for batch processing."""
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            self.frame_buffer.append(frame.copy())
            self.frame_timestamps.append(timestamp)
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.frame_buffer)
    
    def clear_buffer(self):
        """Clear the frame buffer and reset state."""
        with self.lock:
            self.frame_buffer.clear()
            self.frame_timestamps.clear()
            self.last_prediction = None
            self.last_batch_frames = []
            # Don't reset global_c2w - maintain pose continuity
    
    def reset_pose(self):
        """Reset the global pose to origin."""
        self.global_c2w = np.eye(4, dtype=np.float32)
        self.world_scale = 1.0
        print("[DA3] Pose reset to origin")
    
    def process_batch(self) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Process all frames in buffer as a batch.
        
        Returns:
            List of (depth, pose_c2w, intrinsics) for each frame, or None if insufficient frames.
        """
        with self.lock:
            if len(self.frame_buffer) < self.MIN_BATCH_SIZE:
                print(f"[DA3] Insufficient frames: {len(self.frame_buffer)} < {self.MIN_BATCH_SIZE}")
                return None
            
            # Copy frames for processing
            frames = list(self.frame_buffer)
            timestamps = list(self.frame_timestamps)
        
        n_frames = len(frames)
        print(f"[DA3] Processing batch of {n_frames} frames")
        
        if self.use_mock:
            return self._mock_predict_batch(frames)
        
        if not self.is_loaded:
            self.load_to_gpu()
        
        try:
            # Convert to PIL images
            pil_images = [Image.fromarray(f) for f in frames]
            
            # Run inference on batch
            start = time.time()
            with torch.no_grad():
                prediction = self.model.inference(
                    pil_images,
                    ref_view_strategy='saddle_balanced',  # Best for multi-view
                )
            elapsed = time.time() - start
            print(f"[DA3] Batch inference done in {elapsed:.2f}s ({elapsed/n_frames:.2f}s/frame)")
            
            # Extract results
            results = []
            
            for i in range(n_frames):
                # Depth
                depth = to_numpy(prediction.depth[i]).squeeze().astype(np.float32)
                
                # Intrinsics
                if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                    K = to_numpy(prediction.intrinsics[i]).astype(np.float32)
                else:
                    h, w = depth.shape
                    K = self._default_intrinsics(w, h)
                
                # Extrinsics (w2c) -> Convert to c2w
                if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
                    w2c = to_numpy(prediction.extrinsics[i]).astype(np.float32)
                    if w2c.shape == (3, 4):
                        w2c = np.vstack([w2c, [0, 0, 0, 1]])
                    c2w_local = np.linalg.inv(w2c)
                else:
                    c2w_local = np.eye(4, dtype=np.float32)
                
                # First frame of batch = reference, subsequent frames relative to it
                if i == 0:
                    # First frame in batch - use global pose
                    c2w_world = self.global_c2w.copy()
                    ref_c2w_local = c2w_local
                else:
                    # Compute relative motion from reference frame
                    relative_motion = np.linalg.inv(ref_c2w_local) @ c2w_local
                    c2w_world = self.global_c2w @ relative_motion
                
                results.append((depth, c2w_world.astype(np.float32), K))
            
            # Update global pose for next batch (use last frame's pose)
            if len(results) > 0:
                self.global_c2w = results[-1][1].copy()
            
            # Cache results
            self.last_prediction = results
            self.last_batch_frames = frames
            
            # Print confidence info
            if hasattr(prediction, 'conf') and prediction.conf is not None:
                conf = to_numpy(prediction.conf)
                print(f"[DA3] Confidence: min={conf.min():.2f}, max={conf.max():.2f}, mean={conf.mean():.2f}")
            
            return results
            
        except Exception as e:
            print(f"[DA3] Batch processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_frames_direct(self, frames: list) -> Optional[list]:
        """
        Process a list of frames directly (no internal buffering).
        
        Args:
            frames: List of RGB images (H, W, 3) uint8
            
        Returns:
            List of (depth, pose_c2w, intrinsics) for each frame.
        """
        if len(frames) < 3:
            print(f"[DA3] Insufficient frames: {len(frames)} < 3")
            return None
        
        n_frames = len(frames)
        print(f"[DA3] Processing batch of {n_frames} frames")
        
        if self.use_mock:
            return self._mock_predict_batch(frames)
        
        if not self.is_loaded:
            self.load_to_gpu()
        
        try:
            # Convert to PIL images
            pil_images = [Image.fromarray(f) for f in frames]
            
            # Run inference on batch
            start = time.time()
            with torch.no_grad():
                prediction = self.model.inference(
                    pil_images,
                    ref_view_strategy='saddle_balanced',
                )
            elapsed = time.time() - start
            print(f"[DA3] Batch inference done in {elapsed:.2f}s ({elapsed/n_frames:.2f}s/frame)")
            
            # Extract results
            results = []
            ref_c2w_local = None
            
            for i in range(n_frames):
                # Depth
                depth = to_numpy(prediction.depth[i]).squeeze().astype(np.float32)
                
                # Intrinsics
                if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None:
                    K = to_numpy(prediction.intrinsics[i]).astype(np.float32)
                else:
                    h, w = depth.shape
                    K = self._default_intrinsics(w, h)
                
                # Extrinsics (w2c) -> Convert to c2w
                if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None:
                    w2c = to_numpy(prediction.extrinsics[i]).astype(np.float32)
                    if w2c.shape == (3, 4):
                        w2c = np.vstack([w2c, [0, 0, 0, 1]])
                    c2w_local = np.linalg.inv(w2c)
                else:
                    c2w_local = np.eye(4, dtype=np.float32)
                
                # First frame = reference, subsequent frames relative to global
                if i == 0:
                    c2w_world = self.global_c2w.copy()
                    ref_c2w_local = c2w_local
                else:
                    relative_motion = np.linalg.inv(ref_c2w_local) @ c2w_local
                    c2w_world = self.global_c2w @ relative_motion
                
                results.append((depth, c2w_world.astype(np.float32), K))
            
            # Update global pose for next batch
            if len(results) > 0:
                self.global_c2w = results[-1][1].copy()
            
            # Print confidence info
            if hasattr(prediction, 'conf') and prediction.conf is not None:
                conf = to_numpy(prediction.conf)
                print(f"[DA3] Confidence: min={conf.min():.2f}, max={conf.max():.2f}, mean={conf.mean():.2f}")
            
            return results
            
        except Exception as e:
            print(f"[DA3] Direct batch processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_single(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single frame (adds to buffer, processes if enough frames).
        
        For real-time use, call this repeatedly. When enough frames accumulate,
        it will process them as a batch and return results.
        
        Returns:
            (depth, pose_c2w, intrinsics) - may be from cached results if batch not ready.
        """
        self.add_frame(frame)
        
        # Check if we have enough frames
        if self.get_buffer_size() >= self.MIN_BATCH_SIZE:
            results = self.process_batch()
            if results:
                # Return the last frame's result
                self.clear_buffer()
                return results[-1]
        
        # Not enough frames yet - return mock/placeholder
        h, w = frame.shape[:2]
        if self.last_prediction:
            # Use last known result
            return self.last_prediction[-1]
        else:
            # Return placeholder
            return (
                np.ones((h, w), dtype=np.float32) * 5.0,  # 5m depth
                self.global_c2w.copy(),
                self._default_intrinsics(w, h)
            )
    
    def _default_intrinsics(self, width: int, height: int) -> np.ndarray:
        """Generate default intrinsics based on image size."""
        # Assume ~60 degree vertical FOV
        fov_rad = np.deg2rad(60.0)
        focal_length = height / (2 * np.tan(fov_rad / 2))
        cx, cy = width / 2.0, height / 2.0
        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _mock_predict_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate mock predictions for testing."""
        results = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            
            # Mock depth: gradient
            y_coords = np.linspace(1.0, 10.0, h).astype(np.float32)
            depth = np.tile(y_coords[:, None], (1, w))
            depth += np.random.randn(h, w).astype(np.float32) * 0.1
            
            # Mock pose: slight translation per frame
            pose = self.global_c2w.copy()
            pose[0, 3] += i * 0.1  # Move in X
            
            intrinsics = self._default_intrinsics(w, h)
            results.append((depth, pose.astype(np.float32), intrinsics))
        
        self.global_c2w = results[-1][1].copy()
        return results


class ModelManager:
    """
    Simplified model manager for DA3-only operation.
    SAM3 disabled for now.
    """
    
    def __init__(self):
        self.da3 = DA3Wrapper()
        self._ready = False
    
    def initialize(self):
        """Pre-load DA3 model."""
        print("[ModelManager] Initializing DA3...")
        self.da3.load_to_gpu()
        self._ready = True
        print("[ModelManager] Ready")
    
    def is_ready(self) -> bool:
        return self._ready
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """Add frame to processing buffer."""
        self.da3.add_frame(frame, timestamp)
    
    def get_buffer_size(self) -> int:
        return self.da3.get_buffer_size()
    
    def process_batch(self) -> Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Process accumulated frames."""
        return self.da3.process_batch()
    
    def process_single(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single frame.
        
        Returns:
            (depth, pose, intrinsics, class_ids)
        """
        depth, pose, intrinsics = self.da3.predict_single(frame)
        class_ids = np.zeros(depth.shape, dtype=np.int32)  # No segmentation for now
        return depth, pose, intrinsics, class_ids
    
    def reset(self):
        """Reset state for new scan."""
        self.da3.clear_buffer()
        self.da3.reset_pose()
        print("[ModelManager] Reset for new scan")


# Singleton instance
_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the singleton ModelManager."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
