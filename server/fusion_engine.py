# STAC-BUILD: Fusion Engine
# 3D Back-projection and Point Cloud Generation
# Optimized for real-time performance with NumPy/PyTorch vectorization

from __future__ import annotations

import numpy as np
import struct
from typing import Tuple, Optional, TYPE_CHECKING, Any
from dataclasses import dataclass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 640.0
    cy: float = 360.0
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 intrinsics matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @classmethod
    def from_resolution(cls, width: int, height: int, fx: float = 1000.0):
        """Create intrinsics centered on image."""
        return cls(
            fx=fx,
            fy=fx,  # Assume square pixels
            cx=width / 2.0,
            cy=height / 2.0
        )


class FusionEngine:
    """
    Real-time 3D fusion engine.
    Back-projects RGB-D frames to 3D point clouds with semantic labels.
    """
    
    # Binary output format: 7 floats per point
    # [x, y, z, r, g, b, class_id]
    POINT_SIZE = 7 * 4  # 7 float32 = 28 bytes
    
    def __init__(self, 
                 intrinsics: Optional[CameraIntrinsics] = None,
                 max_depth: float = 50.0,
                 min_depth: float = 0.1,
                 downsample_factor: int = 2):
        """
        Initialize the fusion engine.
        
        Args:
            intrinsics: Camera intrinsics (auto-detected if None)
            max_depth: Maximum depth to include (meters)
            min_depth: Minimum depth to include (meters)
            downsample_factor: Spatial downsampling for performance
        """
        self.intrinsics = intrinsics
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.downsample_factor = downsample_factor
        
        # Cache for pixel coordinates (computed once per resolution)
        self._pixel_cache = {}
        
        # Accumulation buffer (for SLAM-like behavior)
        self._accumulated_points = []
        self._max_accumulated_frames = 30
    
    def _get_pixel_coords(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cached pixel coordinate grids.
        
        Returns:
            u: (H, W) x-coordinates
            v: (H, W) y-coordinates
        """
        key = (height, width)
        if key not in self._pixel_cache:
            u = np.arange(width, dtype=np.float32)
            v = np.arange(height, dtype=np.float32)
            u, v = np.meshgrid(u, v)
            self._pixel_cache[key] = (u, v)
        return self._pixel_cache[key]
    
    def backproject(self,
                    rgb: np.ndarray,
                    depth: np.ndarray,
                    pose: np.ndarray,
                    intrinsics: np.ndarray,
                    class_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Back-project RGB-D to 3D point cloud.
        
        Args:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth map (H_d, W_d) float32 - may differ from RGB size
            pose: Camera pose (4, 4) or (3, 4) float32
            intrinsics: Camera intrinsics (3, 3) float32 - REQUIRED
            class_ids: Semantic labels (H, W) int32, optional
            
        Returns:
            points: (N, 7) array [x, y, z, r, g, b, class_id]
        """
        import cv2
        
        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth.shape[:2]
        
        # Resize depth to match RGB if different
        if (depth_h, depth_w) != (rgb_h, rgb_w):
            depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
            if class_ids is not None:
                class_ids = cv2.resize(class_ids.astype(np.float32), (rgb_w, rgb_h), 
                                       interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # Ensure pose is 4x4
        if pose.shape == (3, 4):
            pose_4x4 = np.eye(4, dtype=pose.dtype)
            pose_4x4[:3, :4] = pose
            pose = pose_4x4
        
        h, w = depth.shape[:2]
        
        # Use provided intrinsics
        K = intrinsics
        
        # Downsample for performance
        if self.downsample_factor > 1:
            step = self.downsample_factor
            depth = depth[::step, ::step]
            rgb = rgb[::step, ::step]
            if class_ids is not None:
                class_ids = class_ids[::step, ::step]
            h, w = depth.shape[:2]
            
            # Adjust intrinsics for downsampling (scale fx, fy, cx, cy)
            # Create a scaling matrix
            S = np.diag([1/step, 1/step, 1.0]).astype(K.dtype)
            K = S @ K
        
        # Get pixel coordinates
        u, v = self._get_pixel_coords(h, w)
        
        # Create valid depth mask
        valid = (depth > self.min_depth) & (depth < self.max_depth) & np.isfinite(depth)
        
        # Back-project to camera space
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # z = depth
        
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        
        # Stack to (N, 3) camera-space points
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Transform to world space using pose
        # P_world = R * P_cam + t
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (points_cam @ R.T) + t
        
        # Convert CV world points to glTF/Three.js world points
        # CV: X-right, Y-down, Z-forward
        # glTF/Three.js: X-right, Y-up, Z-backward
        points_world[:, 1] *= -1
        points_world[:, 2] *= -1
        
        # Get RGB values (normalized to 0-1 for shader)
        colors = rgb[valid].astype(np.float32) / 255.0
        
        # Get class IDs (0 = background)
        if class_ids is not None:
            classes = class_ids[valid].astype(np.float32)
        else:
            classes = np.zeros(len(points_world), dtype=np.float32)
        
        # Combine: [x, y, z, r, g, b, class_id]
        result = np.hstack([
            points_world,
            colors,
            classes[:, None]
        ]).astype(np.float32)
        
        return result
    
    def accumulate(self, points: np.ndarray):
        """
        Add points to accumulation buffer (SLAM-like).
        Maintains a sliding window of recent frames.
        """
        self._accumulated_points.append(points)
        
        # Keep only last N frames
        if len(self._accumulated_points) > self._max_accumulated_frames:
            self._accumulated_points.pop(0)
    
    def get_accumulated(self) -> np.ndarray:
        """Get all accumulated points."""
        if not self._accumulated_points:
            return np.zeros((0, 7), dtype=np.float32)
        return np.vstack(self._accumulated_points)
    
    def clear_accumulation(self):
        """Clear the accumulation buffer."""
        self._accumulated_points.clear()
    
    def to_binary(self, points: np.ndarray, compress: bool = False) -> bytes:
        """
        Convert point cloud to binary format for WebSocket transmission.
        
        Format: Flat array of Float32 values
        [x0, y0, z0, r0, g0, b0, class0, x1, y1, z1, ...]
        
        Args:
            points: (N, 7) point cloud array
            compress: Whether to apply compression (future)
            
        Returns:
            Binary buffer
        """
        if len(points) == 0:
            return b''
        
        # Ensure contiguous C-order float32 array
        data = np.ascontiguousarray(points, dtype=np.float32)
        return data.tobytes()
    
    def process_frame(self,
                      rgb: np.ndarray,
                      depth: np.ndarray,
                      pose: np.ndarray,
                      intrinsics: np.ndarray,
                      class_ids: Optional[np.ndarray] = None,
                      accumulate: bool = True) -> bytes:
        """
        Full pipeline: backproject + accumulate + serialize.
        
        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth map (H, W)
            pose: Camera pose (4, 4)
            intrinsics: Camera intrinsics (3, 3) - REQUIRED
            class_ids: Semantic labels (H, W)
            accumulate: Whether to add to sliding window
            
        Returns:
            Binary buffer ready for WebSocket transmission
        """
        # Back-project current frame
        points = self.backproject(rgb, depth, pose, intrinsics, class_ids)
        
        if accumulate:
            self.accumulate(points)
            # Return accumulated points
            all_points = self.get_accumulated()
        else:
            all_points = points
        
        return self.to_binary(all_points)


class FastFusionEngine(FusionEngine):
    """
    GPU-accelerated fusion using PyTorch when available.
    Falls back to NumPy implementation.
    """
    
    def __init__(self, *args, device: str = "cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device if HAS_TORCH and torch.cuda.is_available() else "cpu"
        self._torch_cache = {}
    
    def _get_pixel_coords_torch(self, height: int, width: int):
        """Get cached torch pixel coordinate grids."""
        key = (height, width, self.device)
        if key not in self._torch_cache:
            u = torch.arange(width, dtype=torch.float32, device=self.device)
            v = torch.arange(height, dtype=torch.float32, device=self.device)
            u, v = torch.meshgrid(u, v, indexing='xy')
            self._torch_cache[key] = (u, v)
        return self._torch_cache[key]
    
    def backproject_torch(self,
                          rgb: torch.Tensor,
                          depth: torch.Tensor,
                          pose: torch.Tensor,
                          intrinsics: torch.Tensor,
                          class_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GPU-accelerated back-projection using PyTorch.
        
        Args:
            rgb: (H, W, 3) uint8 tensor
            depth: (H, W) float tensor
            pose: (4, 4) pose tensor
            intrinsics: (3, 3) float tensor - REQUIRED
            class_ids: (H, W) int tensor, optional
            
        Returns:
            (N, 7) point cloud tensor
        """
        h, w = depth.shape[:2]
        
        # Use provided intrinsics
        K = intrinsics
        
        # Downsample
        if self.downsample_factor > 1:
            step = self.downsample_factor
            depth = depth[::step, ::step]
            rgb = rgb[::step, ::step]
            if class_ids is not None:
                class_ids = class_ids[::step, ::step]
            h, w = depth.shape[:2]
            
            # Adjust intrinsics for downsampling
            # S = torch.diag(torch.tensor([1/step, 1/step, 1.0], device=self.device))
            # K = S @ K
            # More efficient in torch to just clone and edit
            K = K.clone()
            K[0, 0] /= step # fx
            K[1, 1] /= step # fy
            K[0, 2] /= step # cx
            K[1, 2] /= step # cy
        
        u, v = self._get_pixel_coords_torch(h, w)
        
        # Valid mask
        valid = (depth > self.min_depth) & (depth < self.max_depth) & torch.isfinite(depth)
        
        # Back-project
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        z = depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy
        
        points_cam = torch.stack([x, y, z], dim=-1)
        
        # Transform to world
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (points_cam @ R.T) + t
        
        # Convert CV world points to glTF/Three.js world points
        points_world[:, 1] *= -1
        points_world[:, 2] *= -1
        
        # Colors
        colors = rgb[valid].float() / 255.0
        
        # Classes
        if class_ids is not None:
            classes = class_ids[valid].float().unsqueeze(-1)
        else:
            classes = torch.zeros(len(points_world), 1, device=self.device)
        
        result = torch.cat([points_world, colors, classes], dim=-1)
        return result


# Singleton for easy import
_engine: Optional[FusionEngine] = None

def get_fusion_engine(use_gpu: bool = True) -> FusionEngine:
    """Get or create the singleton FusionEngine."""
    global _engine
    if _engine is None:
        if use_gpu and HAS_TORCH and torch.cuda.is_available():
            _engine = FastFusionEngine()
        else:
            _engine = FusionEngine()
    return _engine
