# STAC-BUILD: Frame Storage System
# Saves incoming frames to disk organized by scan session
# Manages chunk organization with overlap for reconstruction pipeline

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

from segmentation_manager import SegmentationManager
from config import cfg


def filter_segment_outliers_3d(point_indices: List[int], point_cloud: np.ndarray, 
                                eps: float = None, min_samples: int = None,
                                sor_k: int = None, sor_std_ratio: float = None) -> List[int]:
    """
    Filter spurious points from a segment using DBSCAN + Statistical Outlier Removal.
    
    Args:
        point_indices: List of point indices that belong to the segment
        point_cloud: Full point cloud array [N, 6] where [:, :3] are XYZ coordinates
        eps: DBSCAN epsilon (if None, uses config)
        min_samples: DBSCAN minimum samples (if None, uses config)
        sor_k: Number of neighbors for SOR (if None, uses config)
        sor_std_ratio: Standard deviation multiplier for SOR (if None, uses config)
        
    Returns:
        Filtered list of point indices (outliers removed)
    """
    # Load defaults from config
    outlier_cfg = cfg.get("storage", {}).get("outlier_removal", {})
    if eps is None: eps = outlier_cfg.get("eps", 0.05)
    if min_samples is None: min_samples = outlier_cfg.get("min_samples", 10)
    if sor_k is None: sor_k = outlier_cfg.get("sor_k", 20)
    if sor_std_ratio is None: sor_std_ratio = outlier_cfg.get("sor_std_ratio", 2.0)

    if len(point_indices) < min_samples:
        return point_indices  # Not enough points to filter
    
    # Extract 3D positions for segment points
    indices_array = np.array(point_indices)
    valid_mask = indices_array < len(point_cloud)
    indices_array = indices_array[valid_mask]
    
    if len(indices_array) < min_samples:
        return point_indices
    
    points_3d = point_cloud[indices_array, :3]
    
    # Step 1: DBSCAN clustering
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_3d)
        labels = clustering.labels_
        
        # Keep only the largest cluster (main object)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            # All points are noise - return empty or original
            return point_indices
        
        largest_cluster = unique_labels[np.argmax(counts)]
        cluster_mask = labels == largest_cluster
        
        # Step 2: Statistical Outlier Removal on the main cluster
        cluster_points = points_3d[cluster_mask]
        cluster_indices = indices_array[cluster_mask]
        
        if len(cluster_points) < sor_k:
            return cluster_indices.tolist()
        
        nbrs = NearestNeighbors(n_neighbors=sor_k).fit(cluster_points)
        distances, _ = nbrs.kneighbors(cluster_points)
        mean_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (distance 0)
        
        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)
        threshold = mean_dist + sor_std_ratio * std_dist
        
        inlier_mask = mean_distances < threshold
        filtered_indices = cluster_indices[inlier_mask]
        
        removed_count = len(point_indices) - len(filtered_indices)
        if removed_count > 0:
            print(f"[Filter3D] Removed {removed_count} outliers ({100*removed_count/len(point_indices):.1f}%)")
        
        return filtered_indices.tolist()
        
    except ImportError:
        # sklearn not available - fallback to simple distance-based filtering
        print("[Filter3D] sklearn not available, using simple centroid filter")
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        threshold = np.mean(distances) + 2 * np.std(distances)
        inlier_mask = distances < threshold
        return indices_array[inlier_mask].tolist()



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
    Manages frame storage and chunk organization for reconstruction pipeline.
    
    Frames are saved to disk as they arrive, organized into overlapping
    chunks for processing.
    """
    
    def __init__(self, scans_dir: Optional[Path] = None):
        self.chunk_size = cfg["server"]["chunk_size"]
        self.chunk_overlap = cfg["server"]["chunk_overlap"]
        # Use strict path from config if not provided
        base_path = Path(cfg["paths"]["scans_dir"])
        self.scans_dir = scans_dir or base_path
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session: Optional[ScanSession] = None
        self.lock = Lock()
        self.current_prompt: Optional[str] = None  # Current text prompt
        self.segmentation_manager = None  # Will be set when session starts/loads
        
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
            
            # Initialize unified segmentation manager
            self.segmentation_manager = SegmentationManager(output_dir)
            
            print(f"[FrameStorage] Started session: {session_id}")
            print(f"[FrameStorage] Frames dir: {frames_dir}")
            
            return self.current_session
    
    def stop_session(self) -> Optional[ScanSession]:
        """Stop the current session."""
        with self.lock:
            session = self.current_session
            if session:
                print(f"[FrameStorage] Total chunks: {len(session.chunks)}")
                self.current_session = None
            
            # Reset segmentation manager to avoid state pollution
            self.segmentation_manager = None 
            return session
    
    def set_prompt(self, prompt: str):
        """Set the current text prompt for segmentation."""
        with self.lock:
            self.current_prompt = prompt
            print(f"[FrameStorage] Segmentation prompt set to: '{prompt}'")

    def load_session_from_disk(self, session_id: str) -> Optional[ScanSession]:
        """Re-load an existing session metadata from disk (for offline processing)."""
        with self.lock:
            # Use centralized path resolution (supports both projects/ and scans/)
            from project_paths import resolve_session
            import os
            server_dir = os.path.dirname(os.path.abspath(__file__))
            ctx = resolve_session(server_dir, session_id)
            base_dir = ctx.session_dir
            if not base_dir.exists():
                print(f"[FrameStorage] Session {session_id} not found.")
                return None
                
            frames_dir = ctx.frames_dir
            chunks_dir = base_dir / "chunks"
            output_dir = ctx.output_dir
            
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
            
            # Initialize unified segmentation manager (loads existing segments.json if present)
            self.segmentation_manager = SegmentationManager(output_dir)
            
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
            jpeg_quality = cfg["storage"]["jpeg_quality"]
            cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
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
            chunk_end = self.chunk_size - 1
        else:
            # Subsequent chunks have overlap
            last_chunk = session.chunks[-1]
            chunk_start = last_chunk.end_frame - self.chunk_overlap + 1
            chunk_end = chunk_start + self.chunk_size - 1
        
        # Check if we have enough frames
        if session.frame_count - 1 >= chunk_end:
            chunk_id = len(session.chunks)
            
            # NO COPIAR FRAMES - Solo crear directorio vacío para compatibilidad
            chunk_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
            chunk_dir.mkdir(exist_ok=True)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_frame=chunk_start,
                end_frame=chunk_end,
                frame_count=chunk_end - chunk_start + 1,
                prompt=self.current_prompt,
            )
            
            session.chunks.append(chunk_info)
            
            # Guardar manifiesto de chunks
            self._save_chunks_manifest(session)
            
            print(f"[FrameStorage] Chunk {chunk_id} ready: frames {chunk_start}-{chunk_end} (manifest only)")
            
            # Trigger callback if set
            if self.on_chunk_ready:
                self.on_chunk_ready(session, chunk_info)
            
            return chunk_info
        
        return None
    
    def _save_chunks_manifest(self, session: ScanSession):
        """Guardar chunks.json con la definición de todos los chunks."""
        manifest = {
            "version": "2.0",
            "chunk_size": self.chunk_size,
            "overlap": self.chunk_overlap,
            "total_frames": session.frame_count,
            "chunks": {}
        }
        
        for chunk in session.chunks:
            manifest["chunks"][str(chunk.chunk_id)] = {
                "start_frame": chunk.start_frame,
                "end_frame": chunk.end_frame,
                "frame_count": chunk.frame_count,
                "status": chunk.status
            }
        
        manifest_path = session.base_dir / "chunks.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def get_chunk_frame_paths(self, session: ScanSession, chunk_id: int) -> List[Path]:
        """Obtener lista de paths de frames para un chunk (desde frames/, no chunks/)."""
        # Buscar en manifiesto primero
        manifest_path = session.base_dir / "chunks.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            chunk_def = manifest.get("chunks", {}).get(str(chunk_id))
            if chunk_def:
                start = chunk_def["start_frame"]
                end = chunk_def["end_frame"]
                return [session.frames_dir / f"{i:05d}.jpg" for i in range(start, end + 1)]
        
        # Fallback: buscar en chunks_dir (sesiones antiguas)
        chunk_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
        if chunk_dir.exists():
            return sorted(chunk_dir.glob("*.jpg"))
        
        return []

    
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
        """Save point cloud to binary PLY with origin traceability fields.
        
        Automatically embeds frame_global, pixel_row, pixel_col from
        chunk_*_origins.npz if available, enabling segmentation to work
        on any filtered/reordered version of this cloud.
        """
        if point_cloud is None or len(point_cloud) == 0:
            return None
            
        try:
            filename = f"chunk_{chunk_id:03d}.ply"
            filepath = session.output_dir / filename
            n_pts = len(point_cloud)
            
            # Try to load origin data from reconstruction.s origins.npz
            origins_path = session.output_dir / f"chunk_{chunk_id:03d}_origins.npz"
            has_origins = False
            if origins_path.exists():
                try:
                    origins = np.load(origins_path)
                    fg = origins["frame_global"].astype(np.int32)
                    pr = origins["pixel_row"].astype(np.int16)
                    pc = origins["pixel_col"].astype(np.int16)
                    if len(fg) == n_pts:
                        has_origins = True
                    else:
                        print(f"[FrameStorage] ⚠️ Origins count mismatch: {len(fg)} vs {n_pts} points")
                except Exception as e:
                    print(f"[FrameStorage] ⚠️ Could not load origins: {e}")
            
            # Build binary PLY
            with open(filepath, 'wb') as f:
                header = "ply\n"
                header += "format binary_little_endian 1.0\n"
                header += f"element vertex {n_pts}\n"
                header += "property float x\n"
                header += "property float y\n"
                header += "property float z\n"
                header += "property uchar red\n"
                header += "property uchar green\n"
                header += "property uchar blue\n"
                if has_origins:
                    header += "property int frame_global\n"
                    header += "property short pixel_row\n"
                    header += "property short pixel_col\n"
                header += "end_header\n"
                f.write(header.encode('ascii'))
                
                # Pack vertex data
                if has_origins:
                    dtype = np.dtype([
                        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                        ('frame_global', '<i4'),
                        ('pixel_row', '<i2'), ('pixel_col', '<i2')
                    ])
                    packed = np.empty(n_pts, dtype=dtype)
                    packed['x'] = point_cloud[:, 0]
                    packed['y'] = point_cloud[:, 1]
                    packed['z'] = point_cloud[:, 2]
                    packed['r'] = np.clip(point_cloud[:, 3], 0, 1) * 255
                    packed['g'] = np.clip(point_cloud[:, 4], 0, 1) * 255
                    packed['b'] = np.clip(point_cloud[:, 5], 0, 1) * 255
                    packed['frame_global'] = fg
                    packed['pixel_row'] = pr
                    packed['pixel_col'] = pc
                else:
                    dtype = np.dtype([
                        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                    ])
                    packed = np.empty(n_pts, dtype=dtype)
                    packed['x'] = point_cloud[:, 0]
                    packed['y'] = point_cloud[:, 1]
                    packed['z'] = point_cloud[:, 2]
                    packed['r'] = np.clip(point_cloud[:, 3], 0, 1) * 255
                    packed['g'] = np.clip(point_cloud[:, 4], 0, 1) * 255
                    packed['b'] = np.clip(point_cloud[:, 5], 0, 1) * 255
                
                packed.tofile(f)
            
            origin_tag = " +origins" if has_origins else ""
            print(f"[FrameStorage] Saved PLY: {filepath} ({n_pts:,} pts, binary{origin_tag})")
            return str(filepath)
        except Exception as e:
            print(f"[FrameStorage] Error saving PLY: {e}")
            import traceback
            traceback.print_exc()
            return None


    def load_ply(self, session: ScanSession, chunk_id: int) -> Optional[np.ndarray]:
        """Load point cloud from PLY file (supporting ASCII and Binary, with or without origins)."""
        try:
            # Try legacy format first: chunk_000.ply in output root
            filename_legacy = f"chunk_{chunk_id:03d}.ply"
            filepath = session.output_dir / filename_legacy
            
            # Try new reconstruction format: {id}_pcd.ply in output/pcd
            if not filepath.exists():
                filename_pcd = f"{chunk_id}_pcd.ply"
                filepath = session.output_dir / "pcd" / filename_pcd
            
            if not filepath.exists(): return None
            
            with open(filepath, 'rb') as f:
                # Read Header
                properties = []
                num_points = 0
                is_binary = False
                has_origins = False
                
                while True:
                    line = f.readline().decode('ascii').strip()
                    if line.startswith('element vertex'):
                        num_points = int(line.split()[-1])
                    elif line.startswith('format'):
                        if 'binary_little_endian' in line:
                            is_binary = True
                    elif line.startswith('property'):
                        properties.append(line)
                        if 'frame_global' in line:
                            has_origins = True
                    elif line == 'end_header':
                        break
                
                if num_points == 0: return None

                if is_binary:
                    # Build dtype from properties
                    if has_origins:
                        dtype = np.dtype([
                            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                            ('frame_global', '<i4'),
                            ('pixel_row', '<i2'), ('pixel_col', '<i2')
                        ])
                    else:
                        dtype = np.dtype([
                            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                            ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                        ])
                    
                    data = np.frombuffer(f.read(), dtype=dtype)
                    
                    # Convert to float array (N, 6) - xyz + rgb normalized
                    result = np.zeros((len(data), 6), dtype=np.float32)
                    result[:, 0] = data['x']
                    result[:, 1] = data['y']
                    result[:, 2] = data['z']
                    result[:, 3] = data['r'] / 255.0
                    result[:, 4] = data['g'] / 255.0
                    result[:, 5] = data['b'] / 255.0
                    return result
                
                else:
                    # ASCII Loader
                    points = []
                    for line_bytes in f:
                        line = line_bytes.decode('ascii').strip()
                        parts = line.split()
                        if len(parts) >= 6:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            r, g, b = float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0
                            points.append([x, y, z, r, g, b])
                    
                    return np.array(points, dtype=np.float32)

        except Exception as e:
            print(f"[FrameStorage] Error loading PLY {filepath}: {e}")
            return None


    def save_chunk_metadata(self, session: ScanSession, chunk_id: int, result: Any,
                            alignment_transform: tuple = None,
                            validity_mapping: list = None,
                            sample_indices: np.ndarray = None) -> Optional[str]:
        """
        Save chunk metadata (poses, intrinsics) to JSON for offline refinement.
        Avoids re-running reconstruction if we just want to update masks.

        alignment_transform: (s, R, t) tuple used for point cloud alignment
        validity_mapping: list of {orig_frame_idx, valid_pixel_indices} for pixel-to-point mapping
        sample_indices: indices of points kept after sampling (for mapping pre-sample to post-sample indices)
        """
        try:
            filename = f"chunk_{chunk_id:03d}_meta.json"
            filepath = session.output_dir / filename

            # Format as 'cameras' dict {idx: {intrinsics: ..., extrinsics: ...}}
            # This matches compute_segmentation_for_ply and avoids ambiguity
            cameras = {}
            if result.extrinsics is not None and result.intrinsics is not None:
                for i in range(result.frame_count):
                    cameras[str(i)] = {
                        "extrinsics": result.extrinsics[i].tolist() if isinstance(result.extrinsics, np.ndarray) else result.extrinsics[i],
                        "intrinsics": result.intrinsics[i].tolist() if isinstance(result.intrinsics, np.ndarray) else result.intrinsics[i]
                    }

            # Get scaled resolution from depth output
            scaled_resolution = None
            if result.depths is not None and len(result.depths) > 0:
                scaled_resolution = list(result.depths[0].shape)

            # Get original resolution from first image in chunk (dynamic, not hardcoded)
            original_resolution = None
            chunk_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
            if chunk_dir.exists():
                img_files = sorted(chunk_dir.glob("*.jpg"))
                if img_files:
                    first_img = cv2.imread(str(img_files[0]))
                    if first_img is not None:
                        original_resolution = [first_img.shape[0], first_img.shape[1]]

            # Get chunk step from config (NOT hardcoded)
            chunk_size = cfg["server"]["chunk_size"]
            chunk_overlap = cfg["server"]["chunk_overlap"]
            chunk_step = chunk_size - chunk_overlap

            data = {
                "chunk_id": chunk_id,
                "frame_count": result.frame_count,
                "cameras": cameras,
                "scaled_resolution": scaled_resolution,
                "original_resolution": original_resolution,
                "chunk_step": chunk_step,
                "frame_global_start": chunk_id * chunk_step,
                "frame_global_end": chunk_id * chunk_step + result.frame_count - 1
            }

            # Save alignment transform if provided (for retroactive segmentation projection)
            if alignment_transform:
                s, R, t = alignment_transform
                data["alignment_transform"] = {
                    "scale": float(s),
                    "rotation": R.tolist() if isinstance(R, np.ndarray) else R,
                    "translation": t.tolist() if isinstance(t, np.ndarray) else t
                }
                # PLY was saved with alignment already applied (online mode saves aligned PLYs)
                data["ply_pre_aligned"] = True

            # Save sample_indices if provided (CRITICAL for segmentation index mapping)
            if sample_indices is not None:
                data["sample_indices"] = sample_indices.tolist() if isinstance(sample_indices, np.ndarray) else sample_indices

            with open(filepath, 'w') as f:
                json.dump(data, f)

            print(f"[FrameStorage] Saved Metadata: {filepath} (original_res={original_resolution}, scaled_res={scaled_resolution})")

            # Save OBB if present
            if hasattr(result, 'obb') and result.obb:
                obb_path = session.output_dir / f"chunk_{chunk_id:03d}_obb.json"
                with open(obb_path, 'w') as f:
                    json.dump(result.obb, f)
                print(f"[FrameStorage] Saved OBB: {obb_path}")

            return str(filepath)
        except Exception as e:
             print(f"[FrameStorage] Error saving Metadata: {e}")
             import traceback
             traceback.print_exc()
             return None

    def save_chunk_segmentation(self, session: ScanSession, chunk_id: int, 
                                 masks: dict, prompt: str,
                                 depth_shape: tuple, frame_count: int,
                                 depths: np.ndarray = None,
                                 confs: np.ndarray = None,
                                 sample_indices: np.ndarray = None,
                                 conf_mask: np.ndarray = None,
                                 point_cloud: np.ndarray = None) -> Optional[str]:
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
            point_cloud: [N, 6] point cloud for 3D outlier filtering
        
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
            print(f"[DEBUG save_chunk_segmentation] depth_shape: H={H}, W={W}, total={points_per_frame}")
            
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
                
                orig_mask_shape = raw_mask.shape
                    
                # Resize mask if needed
                if raw_mask.shape[1] != H or raw_mask.shape[2] != W:
                    print(f"[DEBUG] Frame {orig_frame_idx}: Resizing mask from {raw_mask.shape} to [{raw_mask.shape[0]}, {H}, {W}]")
                    resized = []
                    for m in raw_mask:
                        m_resized = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                        resized.append(m_resized)
                    raw_mask = np.stack(resized, axis=0)
                else:
                    print(f"[DEBUG] Frame {orig_frame_idx}: Mask already correct size {raw_mask.shape}")
                
                # Combine all object masks for this frame
                combined = np.max(raw_mask, axis=0) > 0  # [H, W] boolean
                
                # Get pixel indices where mask is True
                mask_flat = combined.flatten()
                mask_pixel_indices = np.where(mask_flat)[0]
                
                # Debug: Show mask pixel positions in 2D
                if len(mask_pixel_indices) > 0:
                    # Convert flat indices to (row, col)
                    rows = mask_pixel_indices // W
                    cols = mask_pixel_indices % W
                    print(f"[DEBUG] Frame {orig_frame_idx}: {len(mask_pixel_indices)} mask pixels")
                    print(f"  Mask 2D region: rows [{rows.min()}-{rows.max()}], cols [{cols.min()}-{cols.max()}]")
                    
                    # How many mask pixels are also valid depth pixels?
                    matches = sum(1 for idx in mask_pixel_indices if idx in flat_to_local_point)
                    print(f"  Matched {matches}/{len(mask_pixel_indices)} with valid depth ({100*matches/len(mask_pixel_indices):.1f}%)")
                
                # Map mask pixels to point cloud indices
                frame_matched = 0
                for flat_idx in mask_pixel_indices:
                    if flat_idx in flat_to_local_point:
                        # This pixel is valid and in the mask
                        local_pt_idx = flat_to_local_point[flat_idx]
                        global_pt_idx = cumulative_valid_count + local_pt_idx
                        all_segmented_indices.append(global_pt_idx)
                        frame_matched += 1
                
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
            
            # Apply 3D outlier filtering if point cloud is provided
            if point_cloud is not None and len(point_indices) > 20:
                point_indices = filter_segment_outliers_3d(point_indices, point_cloud)
            
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
            
            # Also update unified segments.json
            if self.segmentation_manager is not None:
                self.segmentation_manager.add_segment(
                    label=prompt,
                    chunk_id=chunk_id,
                    point_indices=point_indices,
                    instance_id=1  # For now, single instance per prompt
                )
            
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
            
            # Update unified segmentation manager
            if self.segmentation_manager:
                for obj in objects:
                    label = obj.get("label", "Unknown")
                    instance_id = obj.get("id", 1)
                    point_indices_data = obj.get("point_indices", [])
                    
                    actual_indices = []
                    if isinstance(point_indices_data, list):
                        if len(point_indices_data) > 0 and isinstance(point_indices_data[0], dict):
                            # Enhanced format (list of objects with 'indices' field)
                            for item in point_indices_data:
                                if "indices" in item:
                                    actual_indices.extend(item["indices"])
                        else:
                            # Legacy format (list of ints)
                            actual_indices = point_indices_data
                    
                    if actual_indices:
                        self.segmentation_manager.add_segment(
                            label=label,
                            chunk_id=chunk_id,
                            point_indices=actual_indices,
                            instance_id=instance_id
                        )
            
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
            if not filepath.exists():
                return None

            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[FrameStorage] Error loading metadata: {e}")
            return None


# Singleton instance
_storage: Optional[FrameStorage] = None


def get_frame_storage() -> FrameStorage:
    """Get or create the singleton FrameStorage."""
    global _storage
    if _storage is None:
        _storage = FrameStorage()
    return _storage
