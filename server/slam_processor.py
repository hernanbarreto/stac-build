# STAC-Builder: SLAM Processor
# Unified interface for SLAM backends (MASt3R-SLAM, DA3, and Hybrid)
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import asyncio
import sys
import time
import numpy as np
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Generator, Any, Callable
from dataclasses import dataclass, field
import threading
import queue
import cv2

from config import cfg


@dataclass
class SLAMFrame:
    """A single frame result from SLAM processing."""
    frame_id: int
    timestamp: float
    pose: Optional[np.ndarray] = None  # 4x4 transformation matrix
    points: Optional[np.ndarray] = None  # [N, 3] XYZ
    colors: Optional[np.ndarray] = None  # [N, 3] RGB (0-1)
    confidence: Optional[np.ndarray] = None  # [N] confidence values
    is_keyframe: bool = False
    status: str = "ok"  # "ok", "tracking", "lost", "init"
    

@dataclass 
class SLAMSession:
    """State for an active SLAM session."""
    session_id: str
    frames_dir: Path
    output_dir: Path
    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    keyframe_count: int = 0
    total_points: int = 0
    is_running: bool = False
    

class SLAMProcessor:
    """
    Unified SLAM processor that abstracts backend selection.
    
    Supports:
    - MASt3R-SLAM: High accuracy, industrial grade
    - DA3 (Depth Anything 3): Fast, memory efficient
    - Hybrid: MASt3R SLAM poses + DA3 dense depth (subprocess)
    
    Selection is via config.yaml: slam_backend: "mast3r" | "da3" | "hybrid"
    """
    
    def __init__(self):
        self.backend_name = cfg.get("slam_backend", "mast3r")
        self.hybrid_mode = self.backend_name == "hybrid"
        self._backend = None
        self._session: Optional[SLAMSession] = None
        self._lock = threading.Lock()
        self._on_frame_callback: Optional[Callable[[SLAMFrame], None]] = None
        self._is_initialized = False
        
        # Densification worker for hybrid mode
        self._densification_worker = None
        
        # Point cloud accumulator
        self._all_points: List[np.ndarray] = []
        self._all_colors: List[np.ndarray] = []
        
        # Dense points from DA3 (hybrid mode)
        self._dense_points: List[np.ndarray] = []
        self._dense_colors: List[np.ndarray] = []
        
        print(f"[SLAMProcessor] Backend selected: {self.backend_name}" + 
              (f" (MASt3R metric poses + DA3 dense depth)" if self.hybrid_mode else ""))
        
    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._is_initialized
        
    def initialize(self):
        """Initialize the SLAM backend."""
        if self._is_initialized:
            return
            
        print(f"[SLAMProcessor] Initializing {self.backend_name} backend...")
        
        if self.backend_name in ("mast3r", "hybrid"):
            self._init_mast3r()
        else:
            self._init_da3()
            
        self._is_initialized = True
        print(f"[SLAMProcessor] Backend initialized")
        
    def _init_mast3r(self):
        """Initialize MASt3R-SLAM backend."""
        import sys
        
        # MASt3R-SLAM requires specific paths to be at the START of sys.path
        # This is necessary because PYTHONPATH appends to the end
        mast3r_cfg = cfg.get("mast3r_slam", {})
        install_path = mast3r_cfg.get("install_path", "/home/hernan/mast3r_slam")
        
        # Required paths for MASt3R dependencies
        mast3r_paths = [
            install_path,  # mast3r_slam itself
            f"{install_path}/thirdparty/mast3r",  # mast3r
            f"{install_path}/thirdparty/mast3r/dust3r",  # dust3r
            f"{install_path}/thirdparty/mast3r/dust3r/croco",  # croco (contains 'models')
        ]
        
        for p in reversed(mast3r_paths):  # Insert in reverse so first path ends up first
            if p not in sys.path:
                sys.path.insert(0, p)
                print(f"[SLAMProcessor] Added to sys.path: {p}")
        
        from mast3r_wrapper import MASt3RWrapper
        
        config_file = str(Path(install_path) / mast3r_cfg.get("config_file", "config/base.yaml"))
        checkpoint_dir = str(Path(install_path) / "checkpoints")
        device = mast3r_cfg.get("device", "cuda:0")
        
        # Calibration settings
        calib_cfg = mast3r_cfg.get("calibration", {})
        use_calibration = calib_cfg.get("use_intrinsics", False)
        intrinsics = None
        if use_calibration:
            intrinsics = calib_cfg.get("default", {
                "fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 360.0
            })
            
        # Retrieval path
        ckpt_cfg = mast3r_cfg.get("checkpoint", {})
        retrieval_file = ckpt_cfg.get("retrieval_file", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
        retrieval_path = str(Path(checkpoint_dir) / retrieval_file)
        
        # Image settings
        img_cfg = mast3r_cfg.get("image", {})
        img_size = img_cfg.get("size", 512)
            
        self._backend = MASt3RWrapper(
            config_path=config_file,
            checkpoint_dir=checkpoint_dir,
            device=device,
            use_calibration=use_calibration,
            intrinsics=intrinsics,
            retrieval_path=retrieval_path,
            img_size=img_size,
        )
        
        # Setup hybrid mode: DA3 dense depth with MASt3R poses
        if self.hybrid_mode:
            self._setup_hybrid_phase3()
        
    def _setup_hybrid_phase3(self):
        """Setup hybrid: DA3 dense depth + MASt3R metric poses (post-processing)."""
        # DA3 runs as subprocess AFTER MASt3R finishes tracking all frames
        print("[SLAMProcessor] Hybrid mode: DA3 will densify after MASt3R tracking")
        
    def run_hybrid_densification(self, frames_dir: Path, output_dir: Path,
                                  on_chunk_callback=None) -> bool:
        """
        Run hybrid densification: MASt3R metric poses + DA3 dense depth.
        
        Architecture:
        1. Extract metric c2w poses from MASt3R (running in mast3r-slam env)
        2. Free MASt3R from GPU completely
        3. Launch DA3 as subprocess in its own conda env (da3)
        4. DA3 subprocess loads model, densifies with external poses, writes results
        5. Each chunk is saved incrementally and broadcast to viewers via callback
        6. When subprocess exits, DA3 VRAM is freed automatically
        
        Both models NEVER coexist on GPU — complete VRAM isolation.
        
        Args:
            frames_dir: Path to directory with frame images
            output_dir: Path to write output files
            on_chunk_callback: Optional callable(points, colors) called per-chunk
                              for real-time viewer streaming
        """
        if not self._backend:
            raise RuntimeError("[Hybrid] No MASt3R backend available")
        
        # ── Phase 1: Extract MASt3R poses ──────────────────────────────
        all_poses = self._backend.get_all_poses()  # [N, 4, 4] c2w
        if len(all_poses) == 0:
            raise RuntimeError("[Hybrid] MASt3R returned 0 poses")
        
        n_identity = sum(1 for p in all_poses if np.allclose(p, np.eye(4)))
        print(f"[Hybrid] ✅ Phase 1 complete: {len(all_poses)} MASt3R poses")
        print(f"[Hybrid]    Identity poses: {n_identity}/{len(all_poses)}")
        print(f"[Hybrid]    Trajectory span: {np.ptp(all_poses[:, :3, 3], axis=0)}")
        
        # Save poses to disk for the subprocess
        poses_path = output_dir / "mast3r_poses.npy"
        poses_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(poses_path), all_poses)
        
        # Verify image count matches
        import glob as _glob
        image_paths = sorted(
            _glob.glob(str(frames_dir / "*.jpg")) +
            _glob.glob(str(frames_dir / "*.png"))
        )
        n = min(len(image_paths), len(all_poses))
        if n == 0:
            raise RuntimeError(f"[Hybrid] No images found in {frames_dir}")
        print(f"[Hybrid] {n} frames matched to poses")
        
        # ── Phase 1.5: Free MASt3R from GPU ────────────────────────────
        print("[Hybrid] Releasing MASt3R from GPU...")
        try:
            import torch
            if hasattr(self._backend, 'model') and self._backend.model is not None:
                self._backend.model.cpu()
                del self._backend.model
                self._backend.model = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            free_mem = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"[Hybrid] ✅ MASt3R released — {free_mem:.1f} GB VRAM free")
        except Exception as e:
            print(f"[Hybrid] ⚠️ VRAM cleanup issue: {e}")
        
        # ── Phase 2: DA3 densification (subprocess in da3 env) ─────────
        import os as _os
        
        script_path = Path(__file__).parent / "da3_hybrid_densify.py"
        if not script_path.exists():
            raise FileNotFoundError(f"[Hybrid] da3_hybrid_densify.py not found at {script_path}")
        
        output_npz = output_dir / "hybrid_dense_cloud.npz"
        
        # Resolve da3 conda env Python
        conda_base = _os.path.expanduser("~/miniforge3")
        da3_python = f"{conda_base}/envs/da3/bin/python"
        if not _os.path.exists(da3_python):
            raise FileNotFoundError(
                f"[Hybrid] da3 conda env not found at {da3_python}. "
                f"The hybrid backend requires a 'da3' conda environment."
            )
        
        # Build PYTHONPATH for the DA3 subprocess
        da3_root = _os.path.expanduser("~/Depth-Anything-3")
        da3_pythonpath = f"{da3_root}/src:{da3_root}"
        
        # Serialize all relevant config to a temp JSON for the subprocess
        import json as _json
        config_json_path = output_dir / "da3_config.json"
        with open(str(config_json_path), "w") as f:
            _json.dump(cfg, f, indent=2)
        
        cmd = [
            da3_python, "-u", str(script_path),
            "--images-dir", str(frames_dir),
            "--poses", str(poses_path),
            "--output", str(output_npz),
            "--config-json", str(config_json_path),
        ]
        
        env = _os.environ.copy()
        env["PYTHONPATH"] = da3_pythonpath
        env["PYTHONUNBUFFERED"] = "1"
        # Clean env to avoid mast3r paths interfering
        env.pop("CONDA_PREFIX", None)
        
        chunk_size = cfg.get("server", {}).get("chunk_size", 30)
        chunk_overlap = cfg.get("server", {}).get("chunk_overlap", 5)
        n = len(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        print(f"[Hybrid] 🚀 Launching DA3 subprocess in da3 env...")
        print(f"[Hybrid]    Python: {da3_python}")
        print(f"[Hybrid]    Model: {cfg.get('models', {}).get('depth', {}).get('name', 'unknown')}")
        print(f"[Hybrid]    Frames: {n} | Chunks: size={chunk_size}, overlap={chunk_overlap}")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # Stream output and detect CHUNK_READY markers for real-time streaming
        chunk_files = []
        total_streamed = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            
            # Detect chunk ready markers from subprocess
            if line.startswith("CHUNK_READY:"):
                # Format: CHUNK_READY:<path>:<n_points>
                parts = line.split(":")
                if len(parts) >= 3:
                    chunk_path = parts[1]
                    n_pts = int(parts[2])
                    chunk_files.append(chunk_path)
                    total_streamed += n_pts
                    print(f"[Hybrid] 📤 Chunk ready: {n_pts:,} points (total: {total_streamed:,})")
                    
                    # Load and stream to viewer via callback
                    if on_chunk_callback and n_pts > 0:
                        try:
                            chunk_data = np.load(chunk_path)
                            on_chunk_callback(
                                chunk_data["points"],
                                chunk_data["colors"]
                            )
                        except Exception as e:
                            print(f"[Hybrid] ⚠️ Chunk stream error: {e}")
            else:
                print(f"[DA3] {line}")
        
        proc.wait()
        
        if proc.returncode != 0:
            raise RuntimeError(
                f"[Hybrid] DA3 subprocess failed with exit code {proc.returncode}. "
                f"Check the [DA3] log lines above for details."
            )
        
        # ── Phase 3: Load final results ────────────────────────────────
        if not output_npz.exists():
            raise FileNotFoundError(
                f"[Hybrid] DA3 subprocess completed but output not found: {output_npz}"
            )
        
        data = np.load(str(output_npz))
        points = data["points"]
        colors = data["colors"]
        
        if len(points) == 0:
            raise RuntimeError("[Hybrid] DA3 produced 0 points")
        
        self._dense_points = [points]
        self._dense_colors = [colors]
        
        # Cleanup incremental chunk files
        for cf in chunk_files:
            try:
                Path(cf).unlink(missing_ok=True)
            except Exception:
                pass
        
        print(f"[Hybrid] ✅ Phase 2 complete: {len(points):,} dense metric points loaded")
        print(f"[Hybrid] ✅ DA3 subprocess exited — VRAM automatically freed")
        return True
        
    def _init_da3(self):
        """Initialize DA3 backend (existing implementation)."""
        from da3_native_wrapper import RealtimeDA3
        self._backend = RealtimeDA3()
        
    def start_session(self, session_id: str, frames_dir: Path, output_dir: Path) -> SLAMSession:
        """Start a new SLAM session."""
        with self._lock:
            if self._session and self._session.is_running:
                self.stop_session()
                
            self._session = SLAMSession(
                session_id=session_id,
                frames_dir=frames_dir,
                output_dir=output_dir,
                is_running=True,
            )
            
            # Reset accumulators
            self._all_points = []
            self._all_colors = []
            self._dense_points = []
            self._dense_colors = []
            
            # Reset backend if MASt3R/Hybrid
            if self.backend_name in ("mast3r", "hybrid") and self._backend:
                self._backend.reset()
                
            print(f"[SLAMProcessor] Session started: {session_id}")
            return self._session
            
    def stop_session(self):
        """Stop current SLAM session."""
        with self._lock:
            if self._session:
                self._session.is_running = False
                print(f"[SLAMProcessor] Session stopped: {self._session.session_id}")
                self._session = None
            
            # Stop densification worker if running
            if self._densification_worker:
                self._densification_worker.stop()
                
    def process_frame(self, img: np.ndarray, timestamp: Optional[float] = None) -> SLAMFrame:
        """
        Process a single frame through SLAM.
        
        Args:
            img: RGB image [H, W, 3] as uint8 or float32
            timestamp: Frame timestamp (uses current time if None)
            
        Returns:
            SLAMFrame with pose and point data
        """
        if not self._is_initialized:
            self.initialize()
            
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            if self.backend_name in ("mast3r", "hybrid"):
                result = self._process_mast3r(img, timestamp)
            else:
                result = self._process_da3(img, timestamp)
                
            # Update session stats
            if self._session:
                self._session.frame_count += 1
                if result.is_keyframe:
                    self._session.keyframe_count += 1
                    
            # Accumulate points
            if result.points is not None and len(result.points) > 0:
                self._all_points.append(result.points)
                self._all_colors.append(result.colors if result.colors is not None else np.ones_like(result.points) * 0.5)
                
            # Callback
            if self._on_frame_callback:
                self._on_frame_callback(result)
                
            return result
            
    def _process_mast3r(self, img: np.ndarray, timestamp: float) -> SLAMFrame:
        """Process frame with MASt3R-SLAM."""
        result = self._backend.process_frame(img, timestamp)
        
        points = None
        colors = None
        
        if result.pointmap is not None and result.is_keyframe:
            # Extract point cloud from keyframe
            mast3r_cfg = cfg.get("mast3r_slam", {}).get("pointcloud", {})
            conf_thresh = mast3r_cfg.get("confidence_threshold", 0.5)
            sample_ratio = mast3r_cfg.get("sample_ratio", 0.15)
            
            # Filter by confidence
            if result.confidence is not None:
                mask = result.confidence.flatten() > conf_thresh
            else:
                mask = np.ones(len(result.pointmap), dtype=bool)
                
            pts_camera = result.pointmap[mask]
            
            # Transform to world frame using pose
            if result.pose is not None:
                R = result.pose[:3, :3]
                t = result.pose[:3, 3]
                points = (R @ pts_camera.T).T + t
            else:
                points = pts_camera
                
            # Sample for efficiency
            if sample_ratio < 1.0 and len(points) > 100:
                n_samples = max(100, int(len(points) * sample_ratio))
                indices = np.random.choice(len(points), n_samples, replace=False)
                points = points[indices]
                
                
            # Colors
            if result.colors is not None and len(result.colors) > 0:
                # If colors are flat (N,3) and pointmap is flat, they should match
                # Use mask
                colors = result.colors[mask]
                
                # Sample colors using SAME indices as points
                if sample_ratio < 1.0 and len(colors) > 100 and indices is not None:
                     colors = colors[indices]
                     
                # Verify alignment after sampling
                if len(colors) != len(points):
                     # Fallback if mismatch
                     colors = np.ones((len(points), 3), dtype=np.float32) * 0.7
            else:
                 # Fallback
                 colors = np.ones((len(points), 3), dtype=np.float32) * 0.7
            
        return SLAMFrame(
            frame_id=result.frame_id,
            timestamp=timestamp,
            pose=result.pose,
            points=points,
            colors=colors,
            confidence=result.confidence,
            is_keyframe=result.is_keyframe,
            status=result.mode,
        )
        
    def _process_da3(self, img: np.ndarray, timestamp: float) -> SLAMFrame:
        """Process frame with DA3 (placeholder - hooks into existing system)."""
        # DA3 uses chunk-based processing, not frame-by-frame
        # This is a compatibility shim
        frame_id = self._session.frame_count if self._session else 0
        
        return SLAMFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            pose=None,
            points=None,
            colors=None,
            is_keyframe=False,
            status="da3_chunk_mode",
        )
        
    def get_global_pointcloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get accumulated global point cloud.
        In hybrid mode, returns DA3 dense cloud (scaled to MASt3R's metric).
        
        Returns:
            Tuple of (points [N, 3], colors [N, 3])
        """
        with self._lock:
            if self.backend_name in ("mast3r", "hybrid") and self._backend:
                # Get from MASt3R with proper transforms
                mast3r_cfg = cfg.get("mast3r_slam", {}).get("pointcloud", {})
                mast3r_points, mast3r_colors = self._backend.get_global_pointcloud(
                    confidence_threshold=mast3r_cfg.get("confidence_threshold", 0.5),
                    sample_ratio=mast3r_cfg.get("sample_ratio", 0.15),
                )
                
                # In hybrid mode, use DA3 dense cloud
                if self.hybrid_mode and self._dense_points:
                    points = np.concatenate(self._dense_points, axis=0)
                    colors = np.concatenate(self._dense_colors, axis=0)
                    print(f"[SLAMProcessor] Hybrid cloud: {len(points):,} DA3 dense points")
                    return points, colors
                else:
                    return mast3r_points, mast3r_colors
            else:
                # Use accumulated points
                if not self._all_points:
                    return np.empty((0, 3)), np.empty((0, 3))
                    
                points = np.concatenate(self._all_points, axis=0)
                colors = np.concatenate(self._all_colors, axis=0)
                return points, colors
                
    def get_pointcloud_binary(self) -> bytes:
        """
        Get point cloud as binary data for WebSocket streaming.
        
        Format: [N, 7] floats = X, Y, Z, R, G, B, ClassID
        """
        points, colors = self.get_global_pointcloud()
        
        if len(points) == 0:
            return b""
            
        # Create [N, 7] array: XYZ + RGB + ClassID
        n_points = len(points)
        output = np.zeros((n_points, 7), dtype=np.float32)
        output[:, :3] = points
        output[:, 3:6] = colors
        output[:, 6] = 0.0  # Default class ID
        
        return output.tobytes()
        
    def process_frames_directory(
        self,
        frames_dir: Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[SLAMFrame, None, None]:
        """
        Process all frames in a directory.
        
        Args:
            frames_dir: Directory containing frame images
            on_progress: Callback(current_frame, total_frames)
            
        Yields:
            SLAMFrame for each processed frame
        """
        if not frames_dir.exists():
            print(f"[SLAMProcessor] Directory not found: {frames_dir}")
            return
            
        # Get sorted frame files
        extensions = (".jpg", ".png", ".jpeg")
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in extensions
        ])
        
        if not frame_files:
            print(f"[SLAMProcessor] No frames found in {frames_dir}")
            return
            
        total = len(frame_files)
        print(f"[SLAMProcessor] Processing {total} frames from {frames_dir}")
        
        for idx, frame_file in enumerate(frame_files):
            img = cv2.imread(str(frame_file))
            if img is None:
                print(f"[SLAMProcessor] Warning: Could not read {frame_file}")
                continue
                
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            result = self.process_frame(img_rgb, timestamp=float(idx))
            
            if on_progress:
                on_progress(idx + 1, total)
                
            yield result
            
    def process_video(
        self,
        video_path: Path,
        skip_frames: int = 1,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[SLAMFrame, None, None]:
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
            on_progress: Callback(current_frame, total_frames)
            
        Yields:
            SLAMFrame for each processed frame
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[SLAMProcessor] Could not open video: {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        processed = 0
        
        print(f"[SLAMProcessor] Processing video: {video_path} ({total_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % skip_frames == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
                
                result = self.process_frame(frame_rgb, timestamp)
                processed += 1
                
                if on_progress:
                    on_progress(frame_idx, total_frames)
                    
                yield result
                
            frame_idx += 1
            
        cap.release()
        print(f"[SLAMProcessor] Video complete: {processed} frames processed")
        
    def save_pointcloud_ply(self, output_path: Path) -> bool:
        """
        Save accumulated point cloud to PLY file (binary_little_endian).
        Uses vectorized numpy for fast writing — handles 30M+ points in seconds.
        
        Args:
            output_path: Path for output PLY file
            
        Returns:
            True if successful
        """
        points, colors = self.get_global_pointcloud()
        
        if len(points) == 0:
            print("[SLAMProcessor] No points to save")
            return False
            
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            n_points = len(points)
            header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".encode('ascii')
            
            # Vectorized binary packing — no Python loop
            pts = points.astype(np.float32)
            if colors.max() <= 1.0:
                rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(colors, 0, 255).astype(np.uint8)
            
            # Build structured array: 3 floats + 3 uint8 per vertex
            vertex_dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            vertices = np.empty(n_points, dtype=vertex_dtype)
            vertices['x'] = pts[:, 0]
            vertices['y'] = pts[:, 1]
            vertices['z'] = pts[:, 2]
            vertices['red'] = rgb[:, 0]
            vertices['green'] = rgb[:, 1]
            vertices['blue'] = rgb[:, 2]
            
            with open(output_path, 'wb') as f:
                f.write(header)
                vertices.tofile(f)
                
            print(f"[SLAMProcessor] Saved {n_points:,} points to {output_path}")
            return True
            
        except Exception as e:
            import traceback
            print(f"[SLAMProcessor] Error saving PLY: {e}")
            traceback.print_exc()
            return False
            
    def get_state(self) -> Dict[str, Any]:
        """Get current SLAM state."""
        state = {
            "backend": self.backend_name,
            "initialized": self._is_initialized,
            "session": None,
        }
        
        if self._session:
            state["session"] = {
                "id": self._session.session_id,
                "frame_count": self._session.frame_count,
                "keyframe_count": self._session.keyframe_count,
                "is_running": self._session.is_running,
                "duration": time.time() - self._session.start_time,
            }
            
        if self.backend_name in ("mast3r", "hybrid") and self._backend:
            backend_state = self._backend.get_state()
            state["backend_state"] = {
                "num_keyframes": backend_state.num_keyframes,
                "fps": backend_state.fps,
                "current_mode": backend_state.current_mode,
            }
            
        return state
        
    def set_on_frame_callback(self, callback: Callable[[SLAMFrame], None]):
        """Set callback for new frame processing."""
        self._on_frame_callback = callback


# Singleton instance
_slam_processor: Optional[SLAMProcessor] = None


def get_slam_processor() -> SLAMProcessor:
    """Get or create the singleton SLAM processor."""
    global _slam_processor
    if _slam_processor is None:
        _slam_processor = SLAMProcessor()
    return _slam_processor
