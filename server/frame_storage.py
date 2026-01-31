# STAC-BUILD: Frame Storage System
# Saves incoming frames to disk organized by scan session
# Manages chunk organization with overlap for DA3-Streaming

import os
import cv2
import time
import shutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass, field
from threading import Lock
import pandas as pd # Optional if we use parquet, but PLY is text/binary


@dataclass
class ChunkInfo:
    """Information about a processing chunk."""
    chunk_id: int
    start_frame: int
    end_frame: int
    frame_count: int
    status: str = "pending"  # pending, processing, complete, failed
    output_ply: Optional[str] = None
    poses_file: Optional[str] = None
    prompt: Optional[str] = None  # Text prompt for SAM3


@dataclass
class ScanSession:
    """Active scanning session."""
    session_id: str
    start_time: float
    base_dir: Path
    frames_dir: Path
    chunks_dir: Path
    output_dir: Path
    frame_count: int = 0
    chunks: List[ChunkInfo] = field(default_factory=list)


class FrameStorage:
    """
    Manages frame storage and chunk organization for DA3-Streaming.
    
    Frames are saved to disk as they arrive, organized into overlapping
    chunks for processing.
    """
    
    # Configuration - reduced for RTX 3060
    CHUNK_SIZE = 30  # Frames per chunk (reduced from 120)
    CHUNK_OVERLAP = 10  # Increased from 5 for better stability
    SCANS_BASE_DIR = Path("./scans")
    
    def __init__(self, scans_dir: Optional[Path] = None):
        self.scans_dir = scans_dir or self.SCANS_BASE_DIR
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[ScanSession] = None
        self.current_session: Optional[ScanSession] = None
        self.lock = Lock()
        self.current_prompt: Optional[str] = None  # Current text prompt
        
        # Callbacks
        self.on_chunk_ready = None  # Called when a chunk is ready for processing
    
    def start_session(self) -> ScanSession:
        """Start a new scanning session."""
        with self.lock:
            # Create session ID from timestamp
            session_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            
            # Create directory structure
            base_dir = self.scans_dir / session_id
            frames_dir = base_dir / "frames"
            chunks_dir = base_dir / "chunks"
            output_dir = base_dir / "output"
            
            frames_dir.mkdir(parents=True, exist_ok=True)
            chunks_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.current_session = ScanSession(
                session_id=session_id,
                start_time=time.time(),
                base_dir=base_dir,
                frames_dir=frames_dir,
                chunks_dir=chunks_dir,
                output_dir=output_dir,
            )
            
            print(f"[FrameStorage] Started session: {session_id}")
            print(f"[FrameStorage] Frames dir: {frames_dir}")
            
            return self.current_session
    
    def stop_session(self) -> Optional[ScanSession]:
        """Stop the current session."""
        with self.lock:
            session = self.current_session
            if session:
                print(f"[FrameStorage] Stopped session: {session.session_id}")
                print(f"[FrameStorage] Total frames: {session.frame_count}")
                print(f"[FrameStorage] Total chunks: {len(session.chunks)}")
                self.current_session = None
            return session
    
    def set_prompt(self, prompt: str):
        """Set the current text prompt for segmentation."""
        with self.lock:
            self.current_prompt = prompt
            print(f"[FrameStorage] Segmentation prompt set to: '{prompt}'")

    def load_session_from_disk(self, session_id: str) -> Optional[ScanSession]:
        """Re-load an existing session metadata from disk (for offline processing)."""
        with self.lock:
            base_dir = self.scans_dir / session_id
            if not base_dir.exists():
                print(f"[FrameStorage] Session {session_id} not found.")
                return None
                
            frames_dir = base_dir / "frames"
            chunks_dir = base_dir / "chunks"
            output_dir = base_dir / "output"
            
            # Reconstruct chunks list
            # We assume chunk_000, chunk_001 etc exist
            chunks = []
            if chunks_dir.exists():
                chunk_dirs = sorted(chunks_dir.glob("chunk_*"))
                for cd in chunk_dirs:
                    try:
                        cid = int(cd.name.split("_")[-1])
                        # We don't know exact start/end frames without metadata file, 
                        # but we can guess or verify from files.
                        # For re-processing, we just need the ID and the path (which is standard).
                        # Let's count frames inside to be safe
                        c_frames = len(list(cd.glob("*.jpg")))
                        
                        ci = ChunkInfo(
                            chunk_id=cid,
                            start_frame=0, # Unknown/Irrelevant for simple re-process
                            end_frame=0,
                            frame_count=c_frames,
                            status="complete"
                        )
                        chunks.append(ci)
                    except: pass
            
            chunks.sort(key=lambda x: x.chunk_id)
            
            self.current_session = ScanSession(
                session_id=session_id,
                start_time=0, 
                base_dir=base_dir,
                frames_dir=frames_dir,
                chunks_dir=chunks_dir,
                output_dir=output_dir,
                chunks=chunks,
                frame_count=len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
            )
            print(f"[FrameStorage] Loaded session {session_id} from disk (Offline Mode).")
            return self.current_session

    def add_frame(self, frame: np.ndarray) -> Tuple[int, Optional[ChunkInfo]]:
        """
        Add a frame to the current session.
        
        Args:
            frame: RGB image (H, W, 3) uint8
            
        Returns:
            (frame_index, chunk_info) - chunk_info is set when a new chunk is ready
        """
        if self.current_session is None:
            self.start_session()
        
        session = self.current_session
        
        with self.lock:
            # Save frame to disk
            frame_idx = session.frame_count
            frame_path = session.frames_dir / f"{frame_idx:05d}.jpg"
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            session.frame_count += 1
            
            # Check if we have enough frames for a new chunk
            new_chunk = self._check_chunk_ready(session)
            
            return frame_idx, new_chunk
    
    def _check_chunk_ready(self, session: ScanSession) -> Optional[ChunkInfo]:
        """Check if a new chunk is ready for processing."""
        
        # Calculate expected frames for next chunk
        if len(session.chunks) == 0:
            # First chunk starts at frame 0
            chunk_start = 0
            chunk_end = self.CHUNK_SIZE - 1
        else:
            # Subsequent chunks have overlap
            last_chunk = session.chunks[-1]
            chunk_start = last_chunk.end_frame - self.CHUNK_OVERLAP + 1
            chunk_end = chunk_start + self.CHUNK_SIZE - 1
        
        # Check if we have enough frames
        if session.frame_count - 1 >= chunk_end:
            chunk_id = len(session.chunks)
            
            # Create chunk directory with symlinks to frames
            chunk_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
            chunk_dir.mkdir(exist_ok=True)
            
            # Create symlinks to frames
            for i, frame_idx in enumerate(range(chunk_start, chunk_end + 1)):
                src = session.frames_dir / f"{frame_idx:05d}.jpg"
                dst = chunk_dir / f"{i:05d}.jpg"
                if src.exists() and not dst.exists():
                    # Copy instead of symlink for cross-filesystem compatibility
                    shutil.copy2(src, dst)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_frame=chunk_start,
                end_frame=chunk_end,
                frame_count=chunk_end - chunk_start + 1,
                prompt=self.current_prompt,  # Associate current prompt with this chunk
            )
            
            session.chunks.append(chunk_info)
            
            print(f"[FrameStorage] Chunk {chunk_id} ready: frames {chunk_start}-{chunk_end}")
            
            # Trigger callback if set
            if self.on_chunk_ready:
                self.on_chunk_ready(session, chunk_info)
            
            return chunk_info
        
        return None
    
    def get_chunk_frames_dir(self, chunk_info: ChunkInfo) -> Path:
        """Get the directory containing frames for a chunk."""
        if self.current_session is None:
            raise ValueError("No active session")
        return self.current_session.chunks_dir / f"chunk_{chunk_info.chunk_id:03d}"
    
    def get_pending_chunks(self) -> List[ChunkInfo]:
        """Get list of chunks waiting to be processed."""
        if self.current_session is None:
            return []
        return [c for c in self.current_session.chunks if c.status == "pending"]
    
    def update_chunk_status(self, chunk_id: int, status: str, 
                           output_ply: Optional[str] = None,
                           poses_file: Optional[str] = None):
        """Update the status of a chunk."""
        if self.current_session is None:
            return
        
        for chunk in self.current_session.chunks:
            if chunk.chunk_id == chunk_id:
                chunk.status = status
                if output_ply:
                    chunk.output_ply = output_ply
                if poses_file:
                    chunk.poses_file = poses_file
                print(f"[FrameStorage] Chunk {chunk_id} status: {status}")
                break
    
    def get_session_info(self) -> dict:
        """Get current session information."""
        if self.current_session is None:
            return {"active": False}
        
        session = self.current_session
        return {
            "active": True,
            "session_id": session.session_id,
            "frame_count": session.frame_count,
            "chunks_total": len(session.chunks),
            "chunks_pending": len([c for c in session.chunks if c.status == "pending"]),
            "chunks_processing": len([c for c in session.chunks if c.status == "processing"]),
            "base_dir": str(session.base_dir),
        }
    
    def save_ply(self, session: ScanSession, chunk_id: int, point_cloud: np.ndarray) -> Optional[str]:
        """Save point cloud to PLY file in the session output directory."""
        if point_cloud is None or len(point_cloud) == 0:
            return None
            
        try:
            filename = f"chunk_{chunk_id:03d}.ply"
            filepath = session.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(point_cloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for p in point_cloud:
                    # p is [x, y, z, r, g, b] (floats 0-1 for color? No, usually 0-1 from DA3?)
                    # ChunkProcessor output:
                    # valid_colors = colors_all[mask] (from images, so 0-1)
                    # point_cloud = concatenate([valid_points, valid_colors])
                    
                    # Colors need to be 0-255 for PLY uchar
                    r = int(np.clip(p[3], 0, 1) * 255)
                    g = int(np.clip(p[4], 0, 1) * 255)
                    b = int(np.clip(p[5], 0, 1) * 255)
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
            
            print(f"[FrameStorage] Saved PLY: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[FrameStorage] Error saving PLY: {e}")
            return None

    def load_ply(self, session: ScanSession, chunk_id: int) -> Optional[np.ndarray]:
        """Load point cloud from ASCII PLY file."""
        try:
            filename = f"chunk_{chunk_id:03d}.ply"
            filepath = session.output_dir / filename
            if not filepath.exists(): return None
            
            points = []
            header_ended = False
            
            with open(filepath, 'r') as f:
                for line in f:
                    if not header_ended:
                        if line.strip() == "end_header":
                            header_ended = True
                        continue
                    
                    # Parse line: x y z r g b
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        # Normalize colors back to 0-1
                        r, g, b = float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0
                        points.append([x, y, z, r, g, b])
            
            return np.array(points, dtype=np.float32)
        except Exception as e:
            print(f"[FrameStorage] Error loading PLY: {e}")
            return None

    def save_chunk_metadata(self, session: ScanSession, chunk_id: int, result: Any, 
                            alignment_transform: tuple = None,
                            validity_mapping: list = None) -> Optional[str]:
        """
        Save chunk metadata (poses, intrinsics) to JSON for offline refinement.
        Avoids re-running DA3 if we just want to update masks.
        
        alignment_transform: (s, R, t) tuple used for point cloud alignment
        validity_mapping: list of {orig_frame_idx, valid_pixel_indices} for pixel-to-point mapping
        """
        try:
            filename = f"chunk_{chunk_id:03d}_meta.json"
            filepath = session.output_dir / filename
            
            data = {
                "chunk_id": chunk_id,
                "frame_count": result.frame_count,
                # Convert numpy arrays to lists
                "extrinsics": result.extrinsics.tolist() if isinstance(result.extrinsics, np.ndarray) else result.extrinsics,
                "intrinsics": result.intrinsics.tolist() if isinstance(result.intrinsics, np.ndarray) else result.intrinsics,
            }
            
            # Save alignment transform if provided (for retroactive segmentation projection)
            if alignment_transform:
                s, R, t = alignment_transform
                data["alignment_transform"] = {
                    "scale": float(s),
                    "rotation": R.tolist() if isinstance(R, np.ndarray) else R,
                    "translation": t.tolist() if isinstance(t, np.ndarray) else t
                }
            
            # Save validity mapping for pixel-to-point correlation
            if validity_mapping:
                data["validity_mapping"] = validity_mapping
            
            if result.segmentation_masks:
                 # Masks are dict {frame_idx: mask_array}
                 # We don't save full masks in JSON (too big). We rely on them being generated or re-generated.
                 # But we might want to save OBB if we computed it.
                 pass

            import json
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            print(f"[FrameStorage] Saved Metadata: {filepath}")
            
            # Save OBB if present
            if hasattr(result, 'obb') and result.obb:
                obb_path = session.output_dir / f"chunk_{chunk_id:03d}_obb.json"
                with open(obb_path, 'w') as f:
                    json.dump(result.obb, f)
                print(f"[FrameStorage] Saved OBB: {obb_path}")

            return str(filepath)
        except Exception as e:
             print(f"[FrameStorage] Error saving Metadata: {e}")
             return None

    def save_chunk_segmentation(self, session: ScanSession, chunk_id: int, 
                                 masks: dict, prompt: str,
                                 depth_shape: tuple, frame_count: int,
                                 depths: np.ndarray = None,
                                 confs: np.ndarray = None,
                                 sample_indices: np.ndarray = None,
                                 conf_mask: np.ndarray = None) -> Optional[str]:
        """
        Save segmentation data to JSON with enhanced format.
        
        Args:
            masks: Dict[frame_idx, mask_data] from SAM3
            prompt: Text label for the segmented object
            depth_shape: (H, W) shape of depth maps
            frame_count: Total frames in chunk
            depths: [N, H, W] depth maps for validity filtering (must match keyframes)
            confs: [N, H, W] confidence maps for validity filtering
            sample_indices: If point cloud was sampled, the indices that were kept
            conf_mask: Boolean mask of valid points (confidence filter)
        
        Format:
        {
            "chunk_id": 0,
            "objects": [
                {"id": 1, "label": "green trash can", "point_indices": [0, 1, 2...]}
            ]
        }
        """
        if not masks:
            return None
            
        try:
            import cv2
            
            # Match keyframe logic from _generate_point_cloud
            keyframe_interval = 5
            keyframe_indices = list(range(0, frame_count, keyframe_interval))
            
            H, W = depth_shape
            points_per_frame = H * W
            
            # Compute conf_threshold matching _generate_point_cloud
            if confs is not None:
                # Filter to keyframes
                confs_kf = confs[keyframe_indices] if len(confs) > len(keyframe_indices) else confs
                conf_threshold = max(0.0, np.mean(confs_kf[confs_kf > 0]) * 0.5)
            else:
                conf_threshold = 0.0
            
            # Filter depths to keyframes
            if depths is not None:
                depths_kf = depths[keyframe_indices] if len(depths) > len(keyframe_indices) else depths
            else:
                depths_kf = None
            
            # Will collect all point indices that are segmented
            all_segmented_indices = []
            
            # Track cumulative valid point count (running offset)
            cumulative_valid_count = 0
            
            for kf_idx, orig_frame_idx in enumerate(keyframe_indices):
                # Compute validity mask for this keyframe (matching _generate_point_cloud)
                if depths_kf is not None and confs_kf is not None:
                    depth_flat = depths_kf[kf_idx].flatten()
                    conf_flat = confs_kf[kf_idx].flatten()
                    validity_mask = (conf_flat >= conf_threshold) & (depth_flat > 0.1) & (depth_flat < 100)
                else:
                    # Fallback: assume all valid (will be wrong but better than nothing)
                    validity_mask = np.ones(points_per_frame, dtype=bool)
                
                # Create mapping: flat_pixel_idx -> point_cloud_idx (only for valid pixels)
                # valid_indices are the flat indices that are valid
                valid_flat_indices = np.where(validity_mask)[0]
                
                # Build reverse map: flat_idx -> point_idx (relative to this frame's valid points)
                flat_to_local_point = {}
                for local_pt_idx, flat_idx in enumerate(valid_flat_indices):
                    flat_to_local_point[flat_idx] = local_pt_idx
                
                # Get SAM3 mask for this frame
                if orig_frame_idx not in masks:
                    cumulative_valid_count += len(valid_flat_indices)
                    continue
                    
                frame_mask_data = masks[orig_frame_idx]
                if not isinstance(frame_mask_data, dict) or 'out_binary_masks' not in frame_mask_data:
                    cumulative_valid_count += len(valid_flat_indices)
                    continue
                    
                raw_mask = frame_mask_data['out_binary_masks']
                if hasattr(raw_mask, 'cpu'):
                    raw_mask = raw_mask.cpu().numpy()
                    
                if raw_mask.ndim == 2:
                    raw_mask = raw_mask[None, ...]
                    
                if raw_mask.size == 0:
                    cumulative_valid_count += len(valid_flat_indices)
                    continue
                    
                # Resize mask if needed
                if raw_mask.shape[1] != H or raw_mask.shape[2] != W:
                    resized = []
                    for m in raw_mask:
                        m_resized = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                        resized.append(m_resized)
                    raw_mask = np.stack(resized, axis=0)
                
                # Combine all object masks for this frame
                combined = np.max(raw_mask, axis=0) > 0  # [H, W] boolean
                
                # Get pixel indices where mask is True
                mask_flat = combined.flatten()
                mask_pixel_indices = np.where(mask_flat)[0]
                
                # Map mask pixels to point cloud indices
                for flat_idx in mask_pixel_indices:
                    if flat_idx in flat_to_local_point:
                        # This pixel is valid and in the mask
                        local_pt_idx = flat_to_local_point[flat_idx]
                        global_pt_idx = cumulative_valid_count + local_pt_idx
                        all_segmented_indices.append(global_pt_idx)
                
                # Advance cumulative counter
                cumulative_valid_count += len(valid_flat_indices)
            
            if not all_segmented_indices:
                return None
                
            # Convert to sorted list
            point_indices = sorted(all_segmented_indices)
            
            # If sampling was applied, map indices
            if sample_indices is not None:
                sample_set = set(sample_indices.tolist())
                sample_to_new = {old: new for new, old in enumerate(sample_indices)}
                point_indices = [sample_to_new[idx] for idx in point_indices if idx in sample_set]
            
            # Create enhanced format
            data = {
                "chunk_id": chunk_id,
                "objects": [
                    {
                        "id": 1,  # For now, single object per segmentation
                        "label": prompt,
                        "point_indices": point_indices
                    }
                ]
            }
            
            filename = f"chunk_{chunk_id:03d}_segments.json"
            filepath = session.output_dir / filename
            
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            print(f"[FrameStorage] Saved Segments: {len(point_indices)} points for '{prompt}'")
            return str(filepath)
            
        except Exception as e:
            print(f"[FrameStorage] Error saving Segments: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_segments_direct(self, session: ScanSession, chunk_id: int, segments: dict, prompt: str = "object") -> Optional[str]:
        """Save segmentation dictionary directly to JSON with consistent format."""
        try:
            filename = f"chunk_{chunk_id:03d}_segments.json"
            filepath = session.output_dir / filename
            
            # Convert from {obj_id: [indices]} to [{id, label, point_indices}] format
            objects = []
            for obj_id_str, indices in segments.items():
                try:
                    obj_id = int(obj_id_str)
                except:
                    obj_id = 1
                objects.append({
                    "id": obj_id,
                    "label": prompt,
                    "point_indices": indices
                })
            
            data = {
                "chunk_id": chunk_id,
                "objects": objects
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            total_points = sum(len(obj["point_indices"]) for obj in objects)
            print(f"[FrameStorage] Saved Segments: {total_points} points in {len(objects)} objects")
            return str(filepath)
        except Exception as e:
            print(f"[FrameStorage] Error saving segments direct: {e}")
            return None

    def load_chunk_metadata(self, session: ScanSession, chunk_id: int) -> Optional[dict]:
        """Load metadata (extrinsics/intrinsics) for a chunk."""
        try:
            filename = f"chunk_{chunk_id:03d}_meta.json"
            filepath = session.output_dir / filename
            if not filepath.exists(): return None
            
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[FrameStorage] Error loading metadata: {e}")
            return None
                
            if not segments:
                return None
                
            data = {
                "chunk_id": chunk_id,
                "segments": segments
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
                
            print(f"[FrameStorage] Saved Segmentation: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[FrameStorage] Error saving segmentation: {e}")
            return None


# Singleton instance
_storage: Optional[FrameStorage] = None


def get_frame_storage() -> FrameStorage:
    """Get or create the singleton FrameStorage."""
    global _storage
    if _storage is None:
        _storage = FrameStorage()
    return _storage
