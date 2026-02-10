# STAC-Builder: Densification Worker (Subprocess Version)
# Async worker for DA3 dense point cloud generation using MASt3R poses
# Uses subprocess to run DA3 in its own conda environment
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import numpy as np
import threading
import queue
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
import time
import cv2


@dataclass
class DensificationTask:
    """A keyframe to densify."""
    keyframe_id: int
    image: np.ndarray  # [H, W, 3] RGB uint8
    pose: np.ndarray   # 4x4 T_WC transformation matrix
    timestamp: float
    intrinsics: Optional[np.ndarray] = None  # 3x3 camera intrinsics


@dataclass
class DensificationResult:
    """Result from densification."""
    keyframe_id: int
    points: np.ndarray   # [N, 3] world-space XYZ
    colors: np.ndarray   # [N, 3] RGB (0-1)
    confidence: np.ndarray  # [N]
    num_points: int


class SubprocessDA3Densifier:
    """
    Calls DA3 as subprocess in its own conda environment.
    Communication via temporary files.
    """
    
    def __init__(
        self,
        conda_env: str = "da3",
        script_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ):
        self.conda_env = conda_env
        self.confidence_threshold = confidence_threshold
        
        # Find da3_densify.py script
        if script_path:
            self.script_path = script_path
        else:
            self.script_path = str(Path(__file__).parent / "da3_densify.py")
        
        # Temp directory for IPC
        self.temp_dir = Path(tempfile.gettempdir()) / "stac_densify"
        self.temp_dir.mkdir(exist_ok=True)
        
    def densify_frame(
        self, 
        image: np.ndarray, 
        pose_T_WC: np.ndarray,
        keyframe_id: int,
        intrinsics: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Generate dense point cloud using subprocess.
        
        Returns:
            Tuple of (points [N, 3], colors [N, 3], confidence [N])
        """
        # Save inputs to temp files
        img_path = self.temp_dir / f"kf_{keyframe_id:04d}.jpg"
        pose_path = self.temp_dir / f"kf_{keyframe_id:04d}_pose.npy"
        output_path = self.temp_dir / f"kf_{keyframe_id:04d}_output.npz"
        
        # Save image (convert RGB to BGR for cv2, ensure uint8)
        img_save = image
        if img_save.dtype != np.uint8:
            # MASt3R's uimg may be float32 [0,1] or float32 [0,255]
            if img_save.max() <= 1.0:
                img_save = (img_save * 255).clip(0, 255).astype(np.uint8)
            else:
                img_save = img_save.clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(img_path), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
        np.save(str(pose_path), pose_T_WC)
        
        # Build command
        cmd = [
            "bash", "-c",
            f"source ~/miniforge3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"python {self.script_path} "
            f"--image {img_path} "
            f"--pose {pose_path} "
            f"--output {output_path} "
            f"--confidence {self.confidence_threshold}"
        ]
        
        if intrinsics is not None:
            intrinsics_path = self.temp_dir / f"kf_{keyframe_id:04d}_K.npy"
            np.save(str(intrinsics_path), intrinsics)
            cmd[-1] += f" --intrinsics {intrinsics_path}"
        
        try:
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout (first call loads DA3 model)
            )
            
            if result.returncode != 0:
                print(f"[Densifier] Subprocess error: {result.stderr}")
                return None, None, None
            
            # Load output
            if output_path.exists():
                data = np.load(str(output_path))
                points = data['points']
                colors = data['colors']
                confidence = data['confidence']
                
                # Cleanup temp files
                for f in [img_path, pose_path, output_path]:
                    if f.exists():
                        f.unlink()
                
                return points, colors, confidence
            else:
                print(f"[Densifier] Output file not found: {output_path}")
                return None, None, None
                
        except subprocess.TimeoutExpired:
            print(f"[Densifier] Subprocess timeout for keyframe {keyframe_id}")
            return None, None, None
        except Exception as e:
            print(f"[Densifier] Error: {e}")
            return None, None, None


class DensificationWorker:
    """
    Async worker that densifies keyframes in background using DA3 subprocess.
    Thread-safe queue for non-blocking operation.
    """
    
    def __init__(
        self,
        on_result: Optional[Callable[[DensificationResult], None]] = None,
        max_queue_size: int = 20,
        conda_env: str = "da3",
    ):
        self.on_result = on_result
        self.conda_env = conda_env
        
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._densifier: Optional[SubprocessDA3Densifier] = None
        
        # Statistics
        self.tasks_completed = 0
        self.total_points_generated = 0
        
    def start(self):
        """Start the worker thread."""
        if self._running:
            return
            
        self._running = True
        self._densifier = SubprocessDA3Densifier(conda_env=self.conda_env)
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        print("[DensificationWorker] Started (subprocess mode)")
        
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        # Put sentinel to unblock queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[DensificationWorker] Stopped")
        
    def submit(self, task: DensificationTask) -> bool:
        """
        Submit a keyframe for densification.
        Returns False if queue is full.
        """
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            print(f"[DensificationWorker] Queue full, dropping keyframe {task.keyframe_id}")
            return False
            
    def _worker_loop(self):
        """Main worker loop - runs in separate thread."""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                if task is None:
                    continue
                    
                start_time = time.time()
                
                # Densify the frame via subprocess
                points, colors, confidence = self._densifier.densify_frame(
                    image=task.image,
                    pose_T_WC=task.pose,
                    keyframe_id=task.keyframe_id,
                    intrinsics=task.intrinsics,
                )
                
                elapsed = time.time() - start_time
                
                if points is not None:
                    result = DensificationResult(
                        keyframe_id=task.keyframe_id,
                        points=points,
                        colors=colors,
                        confidence=confidence,
                        num_points=len(points),
                    )
                    
                    self.tasks_completed += 1
                    self.total_points_generated += len(points)
                    
                    print(f"[DensificationWorker] KF {task.keyframe_id}: {len(points):,} pts in {elapsed:.2f}s")
                    
                    # Callback with result
                    if self.on_result:
                        try:
                            self.on_result(result)
                        except Exception as e:
                            print(f"[DensificationWorker] Callback error: {e}")
                else:
                    print(f"[DensificationWorker] KF {task.keyframe_id}: Densification failed")
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DensificationWorker] Error processing task: {e}")
                import traceback
                traceback.print_exc()


# Singleton instance
_densification_worker: Optional[DensificationWorker] = None


def get_densification_worker(
    on_result: Optional[Callable[[DensificationResult], None]] = None
) -> DensificationWorker:
    """Get or create the densification worker singleton."""
    global _densification_worker
    if _densification_worker is None:
        _densification_worker = DensificationWorker(on_result=on_result)
    return _densification_worker
