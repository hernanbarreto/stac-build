# STAC-Builder: MASt3R-SLAM Wrapper
# Industrial-grade SLAM integration for construction quality control
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import sys
import os
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator, Any
from dataclasses import dataclass
import threading
import queue

# Add MASt3R-SLAM paths to sys.path BEFORE any mast3r imports
import _mast3r_paths  # noqa: F401 - configures sys.path

import lietorch
from mast3r_slam.config import load_config, config as mast3r_config, set_global_config
from mast3r_slam.frame import Mode, Frame, create_frame
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono, load_retriever, mast3r_match_asymmetric
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.retrieval_database import RetrievalDatabase


@dataclass
class SLAMResult:
    """Result from SLAM processing of a single frame."""
    frame_id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix (camera to world)
    pointmap: Optional[np.ndarray]  # [H*W, 3] points in camera frame
    colors: Optional[np.ndarray]  # [H*W, 3] RGB colors (0-1)
    confidence: Optional[np.ndarray]  # [H*W, 1] confidence values
    is_keyframe: bool
    mode: str  # "init", "tracking", "reloc", "lost"
    

@dataclass
class SLAMState:
    """Current state of the SLAM system."""
    num_keyframes: int
    num_frames_processed: int
    is_initialized: bool
    current_mode: str
    fps: float
    

class KeyFrameList(list):
    """
    Adapter for list to satisfy FrameTracker and FactorGraph interfaces.
    Implements methods required by FactorGraph:
    - __getitem__, __len__, append (inherited from list)
    - last_keyframe()
    - update_T_WCs(T_WCs, idx) - for global optimization
    """
    def last_keyframe(self):
        if not self:
            return None
        return self[-1]
    
    def update_T_WCs(self, T_WCs, idx):
        """Update poses after FactorGraph optimization (official pattern from frame.py:309-311)."""
        # Official impl: self.T_WC[idx] = T_WCs.data
        # T_WCs.data has shape [N, 1, 8] where N = len(idx)
        # We need to update each Frame's T_WC.data in-place
        idx_list = idx.tolist() if hasattr(idx, 'tolist') else list(idx)
        
        for i, kf_idx in enumerate(idx_list):
            if kf_idx < len(self):
                # Direct data assignment like official: self.T_WC[idx] = T_WCs.data
                # T_WCs.data[i] has shape [1, 8], frame.T_WC.data has shape [1, 8]
                self[kf_idx].T_WC.data = T_WCs.data[i:i+1].squeeze(1)  # [1, 8]


class MASt3RWrapper:
    """
    Wrapper for MASt3R-SLAM providing a clean API for STAC integration.
    
    Features:
    - Headless operation (no GUI dependency)
    - Single-process mode (suitable for WSL/server environments)
    - Frame-by-frame processing for real-time integration
    - Point cloud extraction with confidence filtering
    - Camera intrinsics estimation when not provided
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda:0",
        use_calibration: bool = False,
        intrinsics: Optional[Dict[str, float]] = None,
        retrieval_path: Optional[str] = None,
        img_size: int = 512,
    ):
        """
        Initialize MASt3R-SLAM wrapper.
        
        Args:
            config_path: Path to MASt3R config YAML
            checkpoint_dir: Directory containing model checkpoints
            device: CUDA device to use
            use_calibration: Whether to use provided camera calibration
            intrinsics: Camera intrinsics dict
            retrieval_path: Full path to retrieval model checkpoint
            img_size: Image size for processing
        """
        self.device = device
        self.use_calibration = use_calibration
        self.intrinsics = intrinsics
        self.retrieval_path = retrieval_path
        self.img_size = img_size
        
        # Set default paths
        if checkpoint_dir is None:
            checkpoint_dir = str(MAST3R_SLAM_PATH / "checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir)
        
        if config_path is None:
            config_path = str(MAST3R_SLAM_PATH / "config" / "base.yaml")
        
        # Load MASt3R configuration
        load_config(config_path)
        
        # Override for single-process headless mode
        # mast3r_config is the global config dict
        print(f"[MASt3R] Loading model on {device}...")
        
        # Disable gradient computation
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Load the model (uses most accurate checkpoint)
        self.model = load_mast3r(device=device)
        
        # Initialize state - use KeyFrameList adapter for FactorGraph compatibility
        self.keyframes = KeyFrameList()
        self.frame_count = 0
        self.mode = Mode.INIT
        self.tracker: Optional[FrameTrackerr] = None
        self.factor_graph: Optional[FactorGraphh] = None
        self.retrieval_database = None
        self.K: Optional[torch.Tensor] = None  # Intrinsics matrix
        
        # Image size (set from config)
        self.h = 0
        self.w = 0
        
        # FPS tracking
        self.fps_timer = time.time()
        self.last_frame_time = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Hybrid mode: optional densification callback
        # When set, keyframes are emitted for DA3 densification
        self.on_keyframe_callback = None  # Callable[[int, np.ndarray, np.ndarray], None]
        
        # Phase 3 hybrid: store ALL frame poses (not just keyframes)
        self.frame_poses = {}  # {frame_id: np.ndarray [4,4] c2w}
        self.frame_image_paths = {}  # {frame_id: str} image path for each frame
        
        
    def _initialize_from_first_frame(self, img: np.ndarray):
        """Initialize internal structures from first frame dimensions."""
        from mast3r_slam.mast3r_utils import resize_img
        
        # Get resized dimensions
        test_img = resize_img(img, self.img_size)
        true_shape = test_img["true_shape"][0]
        self.h, self.w = true_shape[0], true_shape[1]
        
        # Apply downsample if configured
        downsample = mast3r_config.get("dataset", {}).get("img_downsample", 1)
        if downsample > 1:
            self.h = self.h // downsample
            self.w = self.w // downsample
            
        print(f"[MASt3R] Frame size: {self.w}x{self.h}")
        
        # Setup intrinsics if provided
        if self.use_calibration and self.intrinsics:
            fx = self.intrinsics.get("fx", 1000.0)
            fy = self.intrinsics.get("fy", 1000.0)
            cx = self.intrinsics.get("cx", self.w / 2)
            cy = self.intrinsics.get("cy", self.h / 2)
            
            self.K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=self.device, dtype=torch.float32)
            
            mast3r_config["use_calib"] = True
            print(f"[MASt3R] Using calibration: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            mast3r_config["use_calib"] = False
            print("[MASt3R] No calibration - using ray-based optimization")
        
        # Create keyframe storage with adapter
        self.keyframes = KeyFrameList()
        # self.keyframe_T_WCs = []  # Not needed if frame.T_WC is accurate
        
        # Initialize tracker
        self._init_tracker()
        
    def _init_tracker(self):
        """Initialize the frame tracker."""
        # Use real MASt3R FrameTracker
        self.tracker = FrameTracker(self.model, self.keyframes, self.device)
        self.factor_graph = None  # Will init on demand
        self.retrieval_database = load_retriever(self.model, retriever_path=self.retrieval_path, device=self.device)
        
    def process_frame(
        self,
        img: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> SLAMResult:
        """
        Process a single frame through SLAM.
        
        Args:
            img: RGB image as numpy array [H, W, 3] uint8 or float32
            timestamp: Frame timestamp (uses current time if None)
            
        Returns:
            SLAMResult with pose, pointmap, and status
        """
        import traceback
        try:
            with self.lock:
                if timestamp is None:
                    timestamp = time.time()
                
            # Normalize image to float32 [0, 1]
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
                
            # Initialize on first frame
            if self.frame_count == 0:
                self._initialize_from_first_frame(img)
                
            # Get previous pose or identity
            if self.frame_count == 0:
                T_WC = lietorch.Sim3.Identity(1, device=self.device)
            else:
                T_WC = self.last_T_WC
                
            # Create frame
            frame = create_frame(
                self.frame_count,
                img,
                T_WC,
                img_size=self.img_size,
                device=self.device
            )
            
            is_keyframe = False
            mode_str = "tracking"
            
            # FP16 Disabled due to stability issues (IndexError in MASt3R internals)
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            if self.mode == Mode.INIT:
                # First frame - mono inference to initialize
                X_init, C_init = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X_init, C_init)
                
                self.keyframes.append(frame)
                self.mode = Mode.TRACKING
                is_keyframe = True
                mode_str = "init"
                
            elif self.mode == Mode.TRACKING:
                # Normal tracking mode
                # FrameTracker.track returns (new_kf, debug_list, try_reloc)
                add_new_kf, _, try_reloc = self.tracker.track(frame)
                
                # Force keyframe every N frames for denser reconstruction
                # (MASt3R's match_frac_thresh can be too conservative)
                last_kf = self.keyframes.last_keyframe()
                frames_since_kf = frame.frame_id - (last_kf.frame_id if last_kf else 0)
                force_kf = frames_since_kf >= 15
                
                if try_reloc:
                    self.mode = Mode.RELOC
                    mode_str = "lost"
                elif add_new_kf or force_kf:
                    # tracker.track already populates frame with pointmap via mast3r_match_asymmetric
                    # For forced KFs, we need to run mono inference since tracker might not have added pointmap
                    if force_kf and frame.X_canon is None:
                        # mast3r_inference_mono is imported at module level (line 24)
                        X, C = mast3r_inference_mono(self.model, frame)
                        frame.update_pointmap(X, C)
                    self.keyframes.append(frame)
                    is_keyframe = True
                    mode_str = "tracking_kf" if add_new_kf else "forced_kf"
                    
                    # Hybrid mode: emit keyframe for DA3 densification
                    if self.on_keyframe_callback is not None:
                        try:
                            # Get pose as 4x4 numpy matrix
                            pose_4x4 = self._sim3_to_matrix(frame.T_WC)
                            # Get image as numpy (frame.uimg is [H, W, 3])
                            img_np = frame.uimg.cpu().numpy() if torch.is_tensor(frame.uimg) else frame.uimg
                            self.on_keyframe_callback(
                                keyframe_id=len(self.keyframes) - 1,
                                image=img_np,
                                pose=pose_4x4
                            )
                        except Exception as e:
                            print(f"[MASt3R] Densification callback error: {e}")
                else:
                    mode_str = "tracking"
                    
            elif self.mode == Mode.RELOC:
                # Relocalization mode - try to recover tracking
                # Run mono inference for standalone 3D estimation
                X, C = mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X, C)
                
                # Simple recovery: add as keyframe and try tracking again
                # In full MASt3R-SLAM this would use retrieval + loop closure
                self.keyframes.append(frame)
                self.mode = Mode.TRACKING
                is_keyframe = True
                mode_str = "reloc_recover"
                
            # --- GLOBAL OPTIMIZATION (Official Pattern from mast3r_slam/main.py) ---
            if is_keyframe and len(self.keyframes) > 1:
                kf_idx = len(self.keyframes) - 1
                
                # Lazy init Factor Graph (official: line 214 of main.py)
                if self.factor_graph is None:
                    self.factor_graph = FactorGraph(
                        self.model, 
                        self.keyframes, 
                        K=self.K, 
                        device=self.device
                    )
                
                # Graph Construction (official pattern: run_backend lines 91-118)
                edges_to_add = []
                
                # 1. Add edge to previous consecutive keyframe
                if kf_idx > 0:
                    edges_to_add.append(kf_idx - 1)
                
                # 2. Add retrieval edges (loop closure candidates)
                if self.retrieval_database is not None:
                    try:
                        retrieval_inds = self.retrieval_database.update(
                            frame,
                            add_after_query=True,
                            k=3,  # config["retrieval"]["k"]
                            min_thresh=5e-3,  # config["retrieval"]["min_thresh"]
                        )
                        edges_to_add.extend(retrieval_inds)
                        lc_inds = set(retrieval_inds) - {kf_idx - 1}
                        if lc_inds:
                            print(f"[MASt3R] Loop closure candidates: {lc_inds}")
                    except Exception as e:
                        print(f"[MASt3R] Retrieval error: {e}")
                
                # Remove duplicates and self-references
                edges_to_add = list(set(edges_to_add) - {kf_idx})
                
                # Add factors
                if edges_to_add:
                    frame_idx = [kf_idx] * len(edges_to_add)
                    self.factor_graph.add_factors(
                        edges_to_add, frame_idx, 
                        min_match_frac=0.1
                    )
                
                # Optimize periodically (every 5 keyframes)
                if len(self.keyframes) % 5 == 0:
                    print(f"[MASt3R] Optimizing graph ({len(self.keyframes)} KFs)...")
                    try:
                        if self.use_calibration:
                            self.factor_graph.solve_GN_calib()
                        else:
                            self.factor_graph.solve_GN_rays()
                        print("[MASt3R] Optimization complete.")
                    except Exception as e:
                        print(f"[MASt3R] Optimization error: {e}")
            # --- END GLOBAL OPTIMIZATION ---
                
            # Store last pose
            self.last_T_WC = lietorch.Sim3(frame.T_WC.data.clone())
            
            # Extract pose as 4x4 numpy matrix
            pose_4x4 = self._sim3_to_matrix(frame.T_WC)
            
            # Phase 3 hybrid: save ALL frame poses
            self.frame_poses[self.frame_count] = pose_4x4.copy()
            
            # Extract pointmap if available
            pointmap = None
            colors = None
            confidence = None
            
            if frame.X_canon is not None:
                pointmap = frame.X_canon.cpu().numpy()
                
                # Extract colors from uimg (0-1 float RGB)
                if frame.uimg is not None:
                    # uimg is HxWx3, X_canon is (H*W)x3 or HxWx3 depending on mode
                    # But X_canon usually matches resolution of uimg (due to resize/downsample)
                    try:
                        # uimg is already on CPU (usually), so just view and numpy
                        if frame.uimg.device != torch.device("cpu"):
                            colors = frame.uimg.detach().cpu().view(-1, 3).numpy()
                        else:
                            colors = frame.uimg.view(-1, 3).numpy()
                    except:
                        pass
                        
            if frame.C is not None:
                confidence = frame.C.cpu().numpy()
                
            # Update counters
            self.frame_count += 1
            
            # Calculate FPS
            current_time = time.time()
            dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            if self.frame_count % 30 == 0:
                fps = self.frame_count / (current_time - self.fps_timer)
                print(f"[MASt3R] Frame {self.frame_count}, FPS: {fps:.1f}, "
                      f"Keyframes: {len(self.keyframes)}, Mode: {mode_str}")
                
            return SLAMResult(
                frame_id=self.frame_count - 1,
                timestamp=timestamp,
                pose=pose_4x4,
                pointmap=pointmap,
                colors=colors,
                confidence=confidence,
                is_keyframe=is_keyframe,
                mode=mode_str,
            )
        except Exception as e:
            print(f"❌ MASt3RWrapper Traceback:")
            traceback.print_exc()
            raise e
            
    def _sim3_to_matrix(self, T_WC: lietorch.Sim3) -> np.ndarray:
        """Convert Sim3 pose to 4x4 transformation matrix."""
        # Extract rotation, translation, and scale
        # T_WC.data can have shape [1, 8], [1, 1, 8], etc. - flatten to get [8]
        data = T_WC.data.cpu().numpy().flatten()[:8]  # [8] = [quat(4), trans(3), scale(1)]
        
        # Quaternion to rotation matrix
        quat = data[:4]  # [w, x, y, z] or [x, y, z, w] - need to check
        
        # Safeguard against zero norm quaternion
        if np.allclose(quat, 0):
            print("[MASt3R] ⚠️ Warning: Zero norm quaternion detected, using identity")
            return np.eye(4, dtype=np.float32)
            
        trans = data[4:7]
        scale = data[7]
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation
        # lietorch uses [qx, qy, qz, qw] format
        r = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
        R = r.as_matrix()
        
        # Build 4x4 matrix
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R * scale
        T[:3, 3] = trans
        
        return T
        
    def get_global_pointcloud(
        self,
        confidence_threshold: float = 0.5,
        sample_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the global point cloud from all keyframes.
        Uses official MASt3R-SLAM approach: T_WC.act(X_canon)
        
        Args:
            confidence_threshold: Minimum confidence to include point
            sample_ratio: Fraction of points to sample (for efficiency)
            
        Returns:
            Tuple of (points [N, 3], colors [N, 3])
        """
        all_points = []
        all_colors = []
        
        with self.lock:
            for kf in self.keyframes:
                if kf.X_canon is None:
                    continue
                
                # Get confidence filter
                C = kf.get_average_conf()  # Use averaged confidence like official
                if C is None:
                    C = torch.ones_like(kf.X_canon[..., :1])
                
                valid = C.cpu().numpy().flatten() > confidence_threshold
                
                # Transform to world frame using lietorch (official method)
                # pW = keyframe.T_WC.act(keyframe.X_canon) - from evaluate.py line 59
                pW = kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3)
                
                # Get colors from unnormalized image
                colors = (kf.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3) / 255.0
                
                all_points.append(pW[valid])
                all_colors.append(colors[valid])
                
        if not all_points:
            return np.empty((0, 3)), np.empty((0, 3))
            
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        
        # Subsample
        if sample_ratio < 1.0 and len(points) > 0:
            n_samples = int(len(points) * sample_ratio)
            indices = np.random.choice(len(points), n_samples, replace=False)
            points = points[indices]
            colors = colors[indices]
            
        return points, colors
        
    def get_state(self) -> SLAMState:
        """Get current SLAM state."""
        with self.lock:
            fps = self.frame_count / max(0.001, time.time() - self.fps_timer)
            return SLAMState(
                num_keyframes=len(self.keyframes),
                num_frames_processed=self.frame_count,
                is_initialized=self.mode != Mode.INIT,
                current_mode=self.mode.name.lower(),
                fps=fps,
            )
            
    def reset(self):
        """Reset SLAM state for a new session."""
        with self.lock:
            self.keyframes = KeyFrameList()
            self.frame_count = 0
            self.mode = Mode.INIT
            self.factor_graph = None  # Reset FactorGraph
            self.fps_timer = time.time()
            self.frame_poses = {}
            self.frame_image_paths = {}
            print("[MASt3R] SLAM state reset")
    
    def get_all_poses(self) -> np.ndarray:
        """Get all tracked frame poses as [N, 4, 4] c2w matrices."""
        with self.lock:
            if not self.frame_poses:
                return np.empty((0, 4, 4))
            max_id = max(self.frame_poses.keys())
            poses = []
            for i in range(max_id + 1):
                if i in self.frame_poses:
                    poses.append(self.frame_poses[i])
                else:
                    poses.append(np.eye(4))  # fallback identity
            return np.array(poses)
    
    def export_keyframe_data(self, output_path: str) -> int:
        """
        Export keyframe 3D data for DA3 depth calibration.
        
        Saves per-keyframe: world_points [H*W, 3], confidence [H*W],
        frame_id, c2w pose [4, 4], and image shape.
        
        Args:
            output_path: Path to save the .npz file
            
        Returns:
            Number of keyframes exported
        """
        keyframe_data = []
        
        with self.lock:
            for kf in self.keyframes:
                if kf.X_canon is None:
                    continue
                
                # Get confidence
                C = kf.get_average_conf()
                if C is None:
                    C = torch.ones_like(kf.X_canon[..., :1])
                conf = C.cpu().numpy().flatten().astype(np.float32)
                
                # Transform canonical points to world frame (official method)
                # pW = T_WC.act(X_canon) — from evaluate.py line 59
                pW = kf.T_WC.act(kf.X_canon).cpu().numpy().reshape(-1, 3).astype(np.float32)
                
                # Get c2w pose as 4x4 matrix
                c2w = self._sim3_to_matrix(kf.T_WC)
                
                # Get image shape
                img_shape = kf.img_shape.cpu().numpy().flatten()[:2]  # [H, W]
                
                keyframe_data.append({
                    'frame_id': kf.frame_id,
                    'world_points': pW,        # [H*W, 3]
                    'confidence': conf,         # [H*W]
                    'c2w': c2w,                 # [4, 4]
                    'img_shape': img_shape,     # [2] = [H, W]
                })
        
        # Save as npz
        np.savez(
            output_path,
            n_keyframes=len(keyframe_data),
            frame_ids=np.array([kd['frame_id'] for kd in keyframe_data]),
            img_shapes=np.array([kd['img_shape'] for kd in keyframe_data]),
            c2w_poses=np.array([kd['c2w'] for kd in keyframe_data]),
            # Variable-size arrays stored individually
            **{f'world_points_{i}': kd['world_points'] for i, kd in enumerate(keyframe_data)},
            **{f'confidence_{i}': kd['confidence'] for i, kd in enumerate(keyframe_data)},
        )
        
        print(f"[MASt3R] Exported {len(keyframe_data)} keyframes to {output_path}")
        for kd in keyframe_data:
            print(f"  KF {kd['frame_id']}: {kd['world_points'].shape[0]} pts, "
                  f"conf=[{kd['confidence'].min():.2f}, {kd['confidence'].max():.2f}]")
        
        return len(keyframe_data)
            
    def process_video(
        self,
        video_path: str,
        skip_frames: int = 1,
    ) -> Generator[SLAMResult, None, None]:
        """
        Process a video file frame by frame.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
            
        Yields:
            SLAMResult for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % skip_frames == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps
                
                result = self.process_frame(frame_rgb, timestamp)
                yield result
                
            frame_idx += 1
            
        cap.release()
        
    def process_frames_directory(
        self,
        frames_dir: str,
        extensions: Tuple[str, ...] = (".jpg", ".png", ".jpeg"),
    ) -> Generator[SLAMResult, None, None]:
        """
        Process frames from a directory.
        
        Args:
            frames_dir: Directory containing frame images
            extensions: Allowed file extensions
            
        Yields:
            SLAMResult for each processed frame
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            raise ValueError(f"Directory not found: {frames_dir}")
            
        # Get sorted list of frame files
        frame_files = sorted([
            f for f in frames_path.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
            
        print(f"[MASt3R] Processing {len(frame_files)} frames from {frames_dir}")
        
        for idx, frame_file in enumerate(frame_files):
            img = cv2.imread(str(frame_file))
            if img is None:
                print(f"[MASt3R] Warning: Could not read {frame_file}")
                continue
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = self.process_frame(img_rgb, timestamp=float(idx))
            yield result
            



def estimate_intrinsics_from_image(
    img: np.ndarray,
    fov_degrees: float = 60.0,
) -> Dict[str, float]:
    """
    Estimate camera intrinsics from image dimensions.
    
    This is a fallback when no calibration is provided.
    Uses a reasonable assumption for typical webcams/phones.
    
    Args:
        img: Input image
        fov_degrees: Assumed horizontal field of view
        
    Returns:
        Dict with fx, fy, cx, cy
    """
    h, w = img.shape[:2]
    
    # Estimate focal length from FOV
    fov_rad = np.radians(fov_degrees)
    fx = w / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    
    # Principal point at image cente
    cx = w / 2
    cy = h / 2
    
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
