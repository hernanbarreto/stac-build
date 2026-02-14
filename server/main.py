# STAC-BUILD: Main Server
# PHASE 3 COMPLETE: Incremental Streaming (Logic Fixed)

import asyncio
import json
import time
import gc
import os
import sys
import threading
import torch
import numpy as np
from pathlib import Path
from typing import Set, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from frame_storage import get_frame_storage, FrameStorage
from chunk_processor import get_chunk_processor, ChunkProcessor, ChunkResult
from alignment_manager import get_alignment_manager, AlignmentManager
from sam3_wrapper import get_sam3_wrapper
from config import cfg
from da3_native_wrapper import RealtimeDA3
from slam_processor import get_slam_processor, SLAMProcessor, SLAMFrame

# --- Helper for Chunking ---
def chunk_data(data, chunk_size=1048572): # approx 1MB, multiple of 28 bytes (7 floats * 4)
    # 28 bytes per point. 37449 * 28 = 1048572
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def cloud_to_binary(point_cloud: np.ndarray) -> bytes:
    """
    Convert point cloud [N, 6] (XYZRGB) to binary format for Viewer [N, 7] (XYZRGB + ClassId).
    """
    if point_cloud is None or len(point_cloud) == 0:
        return b""
        
    point_count = len(point_cloud)
    output_data = np.zeros((point_count, 7), dtype=np.float32)
    
    # XYZ
    output_data[:, 0] = point_cloud[:, 0]
    output_data[:, 1] = point_cloud[:, 1]
    output_data[:, 2] = point_cloud[:, 2]
    
    # RGB (Assumed float 0-1 or 0-255? DA3 returns 0-1 usually, or 0-255?)
    # AlignmentManagerr._generate_point_cloud uses images_kf which are 0-255, then divides by 255.0. 
    # So point_cloud[:, 3:] is 0-1 float.
    output_data[:, 3] = point_cloud[:, 3]
    output_data[:, 4] = point_cloud[:, 4]
    output_data[:, 5] = point_cloud[:, 5]
    
    # ClassID (Default 0)
    output_data[:, 6] = 0.0
    
    return output_data.tobytes()

# --- PLY Utilities for Offline Streaming ---
def load_ply_to_numpy(ply_path: Path) -> Optional[np.ndarray]:
    """Load PLY file (ASCII or BINARY) to numpy array [N, 6] (XYZ + RGB 0-1)."""
    try:
        import struct

        with open(ply_path, 'rb') as f:
            # Read header as binary, decode lines
            header_lines = []
            while True:
                line = f.readline()
                try:
                    line_str = line.decode('ascii').strip()
                except:
                    return None
                header_lines.append(line_str)
                if line_str == "end_header":
                    break

            # Parse header
            if not header_lines or header_lines[0] != "ply":
                return None

            vertex_count = 0
            is_binary = False

            for line in header_lines:
                if line.startswith("format"):
                    if "binary" in line:
                        is_binary = True
                elif line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])

            if vertex_count == 0:
                return None

            # Read vertex data
            if is_binary:
                # Binary format: float x, float y, float z, uchar r, uchar g, uchar b
                # Vectorized read using structured dtype
                vertex_dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
                ])
                vertices = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
                
                if len(vertices) == 0:
                    return None
                
                result = np.empty((len(vertices), 6), dtype=np.float32)
                result[:, 0] = vertices['x']
                result[:, 1] = vertices['y']
                result[:, 2] = vertices['z']
                result[:, 3] = vertices['red'].astype(np.float32) / 255.0
                result[:, 4] = vertices['green'].astype(np.float32) / 255.0
                result[:, 5] = vertices['blue'].astype(np.float32) / 255.0
                
                return result
            else:
                # ASCII format
                points = []
                for _ in range(vertex_count):
                    line = f.readline().decode('ascii').strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        r, g, b = float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0
                        points.append([x, y, z, r, g, b])

                return np.array(points, dtype=np.float32) if points else None

    except Exception as e:
        print(f"[PLY Utils] Error loading {ply_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- CloudCompPy Post-Processing ---
async def _run_cloudcompy_postprocess(session_id: str, postproc_config: dict, websocket=None):
    """Run CloudCompPy post-processing on DA3 chunk PLYs as subprocess."""
    import subprocess
    
    scans_dir = Path(__file__).parent / "scans" / session_id / "output"
    output_ply = scans_dir / "cleaned_cloud.ply"
    script_path = Path(__file__).parent / "run_cloudcompy.sh"
    
    voxel_size = postproc_config.get("voxel_size", 0.001)
    max_points = postproc_config.get("max_points", 0)
    
    if not script_path.exists():
        print(f"[PostProc] ⚠️ run_cloudcompy.sh not found, skipping")
        return
    
    # Check if chunks exist
    chunks = sorted(scans_dir.glob("chunk_*.ply"))
    if not chunks:
        print(f"[PostProc] ⚠️ No chunk PLYs found in {scans_dir}, skipping")
        return
    
    print(f"\n[PostProc] 🔧 Starting CloudCompPy professional cleaning ({len(chunks)} chunks, voxel={voxel_size*1000:.1f}mm)...")
    
    if websocket:
        try:
            await websocket.send_text(json.dumps({
                "type": "status",
                "message": f"Post-processing {len(chunks)} chunks with CloudCompPy..."
            }))
        except:
            pass
    
    cmd = [
        "bash", str(script_path),
        "--input-dir", str(scans_dir),
        "--output", str(output_ply),
        "--voxel-size", str(voxel_size),
        "--sor-knn", str(postproc_config.get("sor_knn", 6)),
        "--sor-sigma", str(postproc_config.get("sor_sigma", 1.0)),
        "--noise-radius", str(postproc_config.get("noise_radius", 0.01)),
        "--noise-sigma", str(postproc_config.get("noise_sigma", 1.0)),
    ]
    if max_points > 0:
        cmd.extend(["--max-points", str(max_points)])
    if postproc_config.get("skip_duplicates", False):
        cmd.append("--skip-duplicates")
    if postproc_config.get("skip_sor", False):
        cmd.append("--skip-sor")
    if postproc_config.get("skip_noise", False):
        cmd.append("--skip-noise")
    if postproc_config.get("skip_normals", False):
        cmd.append("--skip-normals")
    
    try:
        # Run as async subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        
        # Read output line by line
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='replace').strip()
            if line_str:
                print(f"  {line_str}")
        
        await process.wait()
        
        if process.returncode == 0 and output_ply.exists():
            file_size_mb = output_ply.stat().st_size / (1024 * 1024)
            print(f"[PostProc] ✅ Cleaned cloud saved: {output_ply} ({file_size_mb:.1f} MB)")
        else:
            print(f"[PostProc] ❌ CloudCompPy failed (exit code: {process.returncode})")
    
    except Exception as e:
        print(f"[PostProc] ❌ Error: {e}")
        import traceback
        traceback.print_exc()

def _align_cloud_to_floor(output_data: np.ndarray) -> np.ndarray:
    """
    Apply RANSAC floor alignment to the final cloud.
    Finds the floor plane and rotates so it sits at y=0 (XZ plane).
    input/output: [N, 7] float32 (x, y, z, r, g, b, classId)
    """
    if output_data is None or len(output_data) < 100:
        print("[FloorAlign] ⚠️ Not enough points for alignment")
        return output_data
    
    try:
        from alignment_manager import get_alignment_manager
        am = get_alignment_manager()
        
        # compute_leveling_from_points expects [N, 3+] with XYZ in first 3 cols
        s, R, t = am.compute_leveling_from_points(output_data[:, :3])
        
        # Check if alignment is identity (no floor found)
        if np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3)):
            print("[FloorAlign] ⚠️ No floor plane detected — sending unaligned")
            return output_data
        
        # Apply: P' = s * (R @ P) + t
        xyz = output_data[:, :3]
        xyz_aligned = s * (xyz @ R.T) + t
        
        result = output_data.copy()
        result[:, :3] = xyz_aligned
        
        # Log stats
        y_min = xyz_aligned[:, 1].min()
        y_max = xyz_aligned[:, 1].max()
        print(f"[FloorAlign] ✅ Floor aligned to y=0 (range: {y_min:.3f} to {y_max:.3f})")
        
        return result
    except Exception as e:
        print(f"[FloorAlign] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return output_data

async def _send_cleaned_cloud(websocket, session_id: str):
    """Load, align floor, and send point cloud to viewer.
    Prefers cleaned_cloud.ply; falls back to raw chunk_000.ply.
    Auto-detects origin fields in PLY header for correct binary parsing.
    """
    scans_dir = Path(__file__).parent / "scans" / session_id / "output"
    
    cleaned_ply = scans_dir / "cleaned_cloud.ply"
    raw_chunk = scans_dir / "chunk_000.ply"
    
    # Prefer cleaned cloud, fallback to raw chunk
    if cleaned_ply.exists():
        ply_path = cleaned_ply
        label = "cleaned_cloud"
    elif raw_chunk.exists():
        ply_path = raw_chunk
        label = "RAW chunk_000"
    else:
        print(f"[SendCloud] ⚠️ No PLY found for {session_id}")
        return False
    
    try:
        file_size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"[SendCloud] Loading {label} ({file_size_mb:.1f} MB)...")
        
        # Read PLY (detect ASCII vs binary, detect origin fields)
        with open(ply_path, "rb") as f:
            n_pts = 0
            is_binary = False
            has_origins = False
            while True:
                line = f.readline()
                if line.startswith(b"element vertex"):
                    n_pts = int(line.split()[-1])
                if line.startswith(b"format binary"):
                    is_binary = True
                if b"frame_global" in line:
                    has_origins = True
                if line.startswith(b"end_header"):
                    break
            
            if is_binary:
                # Build dtype matching the PLY properties
                if has_origins:
                    ply_dtype = np.dtype([
                        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                        ('frame_global', '<i4'),
                        ('pixel_row', '<i2'), ('pixel_col', '<i2')
                    ])
                else:
                    ply_dtype = np.dtype([
                        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                    ])
                raw_data = np.frombuffer(f.read(), dtype=ply_dtype)
                point_count = len(raw_data)
                output_data = np.zeros((point_count, 7), dtype=np.float32)
                output_data[:, 0] = raw_data['x']
                output_data[:, 1] = raw_data['y']
                output_data[:, 2] = raw_data['z']
                output_data[:, 3] = raw_data['r'] / 255.0
                output_data[:, 4] = raw_data['g'] / 255.0
                output_data[:, 5] = raw_data['b'] / 255.0
            else:
                # ASCII PLY
                points = []
                for line_bytes in f:
                    parts = line_bytes.decode('ascii').strip().split()
                    if len(parts) >= 6:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        r, g, b = float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0
                        points.append([x, y, z, r, g, b, 0.0])
                output_data = np.array(points, dtype=np.float32)
                point_count = len(output_data)
        
        if point_count == 0:
            print(f"[SendCloud] ⚠️ Empty cloud")
            return False
        
        if output_data.shape[1] == 7:
            pass  # Already has classId column
        else:
            tmp = np.zeros((point_count, 7), dtype=np.float32)
            tmp[:, :6] = output_data[:, :6]
            output_data = tmp
        
        # ── Floor Alignment ──
        output_data = _align_cloud_to_floor(output_data)
        
        binary_bytes = output_data.tobytes()
        
        # Send via proper viewer protocol
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": f"Sending {label} ({point_count:,} points, {file_size_mb:.1f} MB)..."
        }))
        await websocket.send_text(json.dumps({
            "type": "chunk_start",
            "chunk_id": 0,
            "point_count": point_count
        }))
        
        for sub in chunk_data(binary_bytes):
            await websocket.send_bytes(sub)
            await asyncio.sleep(0.001)
        
        origin_tag = " +origins" if has_origins else ""
        print(f"[SendCloud] ✅ Sent {point_count:,} points ({label}{origin_tag}, floor-aligned)")
        return True
    
    except Exception as e:
        print(f"[SendCloud] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def _send_cleaned_cloud_broadcast(session_id: str):
    """Load and broadcast cleaned_cloud.ply to ALL connected viewers."""
    scans_dir = Path(__file__).parent / "scans" / session_id / "output"
    cleaned_ply = scans_dir / "cleaned_cloud.ply"
    
    if not cleaned_ply.exists():
        print(f"[SendCloud] ⚠️ cleaned_cloud.ply not found for broadcast")
        return False
    
    try:
        file_size_mb = cleaned_ply.stat().st_size / (1024 * 1024)
        
        with open(cleaned_ply, "rb") as f:
            n_pts = 0
            has_origins = False
            while True:
                line = f.readline()
                if line.startswith(b"element vertex"):
                    n_pts = int(line.split()[-1])
                if b"frame_global" in line:
                    has_origins = True
                if line.startswith(b"end_header"):
                    break
            
            if has_origins:
                ply_dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                    ('frame_global', '<i4'),
                    ('pixel_row', '<i2'), ('pixel_col', '<i2')
                ])
            else:
                ply_dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                ])
            raw_data = np.frombuffer(f.read(), dtype=ply_dtype)
        
        point_count = len(raw_data)
        if point_count == 0:
            return False
        
        output_data = np.zeros((point_count, 7), dtype=np.float32)
        output_data[:, 0] = raw_data['x']
        output_data[:, 1] = raw_data['y']
        output_data[:, 2] = raw_data['z']
        output_data[:, 3] = raw_data['r'] / 255.0
        output_data[:, 4] = raw_data['g'] / 255.0
        output_data[:, 5] = raw_data['b'] / 255.0
        output_data[:, 6] = 0.0
        
        # ── Floor Alignment ──
        output_data = _align_cloud_to_floor(output_data)
        
        binary_bytes = output_data.tobytes()
        
        await viewer_manager.broadcast_text(json.dumps({
            "type": "chunk_start",
            "chunk_id": 0,
            "point_count": point_count
        }))
        for sub in chunk_data(binary_bytes):
            await viewer_manager.broadcast_binary(sub)
            await asyncio.sleep(0.001)
        
        origin_tag = " +origins" if has_origins else ""
        print(f"[SendCloud] ✅ Broadcast {point_count:,} points to all viewers (floor-aligned{origin_tag}, {file_size_mb:.1f} MB)")
        return True
    
    except Exception as e:
        print(f"[SendCloud] ❌ Broadcast error: {e}")
        import traceback
        traceback.print_exc()
        return False

def numpy_to_ply_bytes(points: np.ndarray) -> bytes:
    """Convert numpy [N, 6] to binary PLY bytes for streaming."""
    import struct

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
end_heade
""".encode('ascii')

    # Pack points: XYZ as float, RGB as ucha
    data = bytearray()
    for p in points:
        data += struct.pack('<fff', p[0], p[1], p[2])  # XYZ
        data += struct.pack('<BBB', int(p[3]*255), int(p[4]*255), int(p[5]*255))  # RGB

    return header + bytes(data)

def apply_gravity_correction(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply gravity correction transform to point cloud [N, 6]."""
    if points is None or len(points) == 0:
        return points

    xyz = points[:, :3]
    rgb = points[:, 3:]

    # Apply: P' = s * (R @ P) + t
    xyz_corrected = s * (xyz @ R.T) + t

    return np.concatenate([xyz_corrected, rgb], axis=1).astype(np.float32)

# --- Configuration ---
# Loaded from config.yaml via config.py
# Loaded from config.yaml via config.py
HOST = cfg["server"]["host"]
PORT = cfg["server"]["port"]
STATIC_DIR = Path(__file__).parent / cfg["server"]["static_dir"]
CHUNK_SIZE = cfg["server"]["chunk_size"]
CHUNK_OVERLAP = cfg["server"]["chunk_overlap"]


# --- Connection Managers ---
class CameraManager:
    def __init__(self):
        self.active_camera = None
        self.frame_count = 0
    
    async def connect(self, websocket: WebSocket):
        if self.active_camera:
            try:
                await self.active_camera.close()
            except Exception as e:
                print(f"[CameraManager] ⚠️ Warning: Error closing previous connection: {e}")
        self.active_camera = websocket
        print("[Camera] Client connected")
    
    def disconnect(self):
        self.active_camera = None
        print("[Camera] Client disconnected")

class ViewerManager:
    def __init__(self):
        self.viewers: Set[WebSocket] = set()
        self.locks: dict = {} # Map WebSocket -> asyncio.Lock
    
    async def connect_viewer(self, websocket: WebSocket):
        self.viewers.add(websocket)
        self.locks[websocket] = asyncio.Lock()
        print(f"[Viewer] Client connected. Total: {len(self.viewers)}")
    
    def disconnect_viewer(self, websocket: WebSocket):
        self.viewers.discard(websocket)
        if websocket in self.locks:
            del self.locks[websocket]
        print(f"[Viewer] Client disconnected. Total: {len(self.viewers)}")
    
    async def broadcast_binary(self, data: bytes):
        if not data: return
        for viewer in list(self.viewers):
            await self.send_bytes(viewer, data)

    async def broadcast_text(self, message: str):
        for viewer in list(self.viewers):
            await self.send_text(viewer, message)
            
    async def send_text(self, websocket: WebSocket, message: str):
        if websocket not in self.locks: return
        async with self.locks[websocket]:
            try:
                await websocket.send_text(message)
            except Exception as e:
                # print(f"Send Text Error: {e}")
                self.disconnect_viewer(websocket)

    async def send_bytes(self, websocket: WebSocket, data: bytes):
        if websocket not in self.locks: return
        async with self.locks[websocket]:
            try:
                await websocket.send_bytes(data)
            except Exception as e:
                # print(f"Send Bytes Error: {e}")
                self.disconnect_viewer(websocket)

# --- Global State ---
camera_manager = CameraManager()
viewer_manager = ViewerManager()
frame_storage = None
chunk_processor = None
alignment_manager = None
chunk_queue = asyncio.Queue()
last_chunk_result = None
is_processing_chunk = False
server_mode = "offline"  # "online" or "offline" - controls DA3 preloading
slam_processor = None  # Unified SLAM processor (MASt3R or DA3)

# --- WORKER: INCREMENTAL SENDING ---
async def chunk_processing_worker():
    global last_chunk_result, is_processing_chunk
    print("[ChunkWorker] Started")
    
    while True:
        try:
            session, chunk_info = await chunk_queue.get()
            is_processing_chunk = True
            chunk_id = chunk_info.chunk_id
            
            print(f"[ChunkWorker] Processing chunk {chunk_id}...")
            frames_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
            chunk_info.status = "processing"
            
            loop = asyncio.get_event_loop()
            
            def _heavy_lifting():
                if not frames_dir.exists(): return None
                # 1. AI Inference
                # KEY FIX: Fetch prompt dynamically from frame_storage to apply to PENDING chunks
                # This ensures that if user types "sofa", it applies to the next processed chunk immediately.
                current_live_prompt = frame_storage.current_prompt if frame_storage else None
                prompt = current_live_prompt if current_live_prompt else (chunk_info.prompt if hasattr(chunk_info, 'prompt') else None)
                
                result = chunk_processor.process_chunk(frames_dir, chunk_id, prompt=prompt)
                if result:
                    # 2. Point Cloud Gen
                    chunk_processor.generate_point_cloud(result)
                    # 3. Alignment
                    aligned_chunk = alignment_manager.add_chunk(result)
                    # 4. Save PLY for persistence/replay
                    if aligned_chunk:
                        frame_storage.save_ply(session, chunk_id, aligned_chunk.point_cloud)
                        # Use sample_indices from aligned_chunk (CRITICAL: same indices used for PLY)
                        align_transform = alignment_manager.gravity_correction if hasattr(alignment_manager, 'gravity_correction') else None
                        sample_indices = aligned_chunk.sample_indices
                        frame_storage.save_chunk_metadata(session, chunk_id, result,
                                                         alignment_transform=align_transform,
                                                         sample_indices=sample_indices)
                        # Save segmentation with new format (point_indices)
                        if result.segmentation_masks:
                            depth_shape = result.depths[0].shape  # (H, W)
                            frame_storage.save_chunk_segmentation(
                                session, chunk_id,
                                masks=result.segmentation_masks,
                                prompt=frame_storage.current_prompt or prompt or "",
                                depth_shape=depth_shape,
                                frame_count=result.frame_count,
                                depths=result.depths,  # Pass for validity mask
                                confs=result.confs,    # Pass for validity mask
                                sample_indices=sample_indices if sample_indices is not None else None,
                                point_cloud=aligned_chunk.point_cloud  # For 3D outlier filtering
                            )
                return result

            result = await loop.run_in_executor(None, _heavy_lifting)
            
            if result is None:
                print(f"[ChunkWorker] Chunk {chunk_id} failed")
                chunk_info.status = "failed"
            else:
                last_chunk_result = result
                chunk_info.status = "complete"
                
                # No real-time streaming — only log progress
                latest_chunk = alignment_manager.aligned_chunks[-1]
                if latest_chunk.point_cloud is not None and len(latest_chunk.point_cloud) > 0:
                    print(f"[ChunkWorker] Chunk {chunk_id} processed: {len(latest_chunk.point_cloud):,} points (not streaming)")
                    
                    # Send progress status to viewers
                    total_chunks = alignment_manager.get_chunk_count()
                    await viewer_manager.broadcast_text(json.dumps({
                        "type": "status",
                        "message": f"Processing chunk {chunk_id + 1}... ({total_chunks} chunks done)"
                    }))

            is_processing_chunk = False

            # Check if DA3 is done (queue empty) — trigger CloudCompPy + send cleaned cloud
            if chunk_queue.qsize() == 0:
                # Run CloudCompPy post-processing and send cleaned cloud
                session_id_current = frame_storage.current_session.session_id if frame_storage and frame_storage.current_session else None
                if session_id_current:
                    postproc_config = cfg.get("postprocessing", {})
                    if postproc_config.get("enabled", False):
                        await _run_cloudcompy_postprocess(session_id_current, postproc_config)
                    await _send_cleaned_cloud_broadcast(session_id_current)
                
                # Also check for pending retroactive segmentation
                if hasattr(frame_storage, '_pending_retroactive_prompt'):
                    pending_prompt = getattr(frame_storage, '_pending_retroactive_prompt', None)
                    if pending_prompt:
                        print(f"[ChunkWorker] DA3 complete. Running pending retroactive segmentation for '{pending_prompt}'...")
                        frame_storage._pending_retroactive_prompt = None
                        asyncio.create_task(_run_pending_retroactive(pending_prompt))

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ChunkWorker] Error: {e}")
            import traceback
            traceback.print_exc()
            is_processing_chunk = False


def _resolve_segmentation_prompt(prompt: str, frames_dir: str) -> str:
    """
    Resolve segmentation prompt: if 'auto' or empty, use InternVL3 scene analyzer.
    
    The VLM runs AFTER DA3 unload and BEFORE SAM3 load, so no VRAM conflict.
    """
    if prompt and prompt.lower() != "auto":
        return prompt
    
    print("[SceneAnalyzer] Auto-detecting categories with InternVL3...")
    try:
        from scene_analyzer import analyze_scene
        scene_cfg = cfg.get("scene_analysis", {})
        auto_prompt = analyze_scene(frames_dir, scene_cfg)
        if auto_prompt:
            print(f"[SceneAnalyzer] ✅ Auto-detected prompt: '{auto_prompt}'")
            return auto_prompt
        else:
            print("[SceneAnalyzer] ⚠️ No categories detected, falling back to generic")
            return "concrete wall;floor surface;ceiling;pipe;electrical panel;duct;cable tray;door"
    except Exception as e:
        print(f"[SceneAnalyzer] ❌ Error: {e}. Using fallback categories.")
        return "concrete wall;floor surface;ceiling;pipe;electrical panel;duct;cable tray;door"


async def _run_pending_retroactive(prompt: str):
    """Execute pending retroactive segmentation after DA3 completes ALL chunks."""
    try:
        session = frame_storage.current_session
        if not session:
            print("[Retro-Pending] No active session")
            return

        print(f"[Retro-Pending] DA3 complete. Starting segmentation with prompt: '{prompt}'")

        # Unload DA3 to free VRAM (CRITICAL: DA3 must be fully unloaded before SAM3)
        print("[Retro-Pending] Unloading DA3 to free VRAM...")
        chunk_processor.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Resolve prompt (auto-detect if needed, VLM runs here before SAM3)
        prompt = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

        # Run new segmentation pipeline
        from segmentation_pipeline import run_segmentation
        result = run_segmentation(
            frames_dir=str(session.frames_dir),
            output_dir=str(session.output_dir),
            prompt=prompt
        )

        if "error" in result:
            print(f"[Retro-Pending] Segmentation failed: {result['error']}")
        else:
            print(f"[Retro-Pending] ✅ Segmentation complete: {len(result['instances'])} instances")

        # Match masks against current cloud and broadcast
        from segmentation_pipeline import apply_segmentation_to_cloud
        seg_data = apply_segmentation_to_cloud(session.output_dir)
        if seg_data.get("instances"):
            await viewer_manager.broadcast_text(json.dumps(seg_data))
            print(f"[Retro-Pending] Broadcast segmentation ({len(seg_data['instances'])} instances)")
        
        await viewer_manager.broadcast_text(json.dumps({
            "type": "info",
            "message": f"Segmentation complete for '{prompt}': {len(result.get('instances', []))} instances"
        }))

    except Exception as e:
        print(f"[Retro-Pending] Error: {e}")
        import traceback
        traceback.print_exc()

def cloud_to_binary(cloud: np.ndarray) -> bytes:
    if cloud is None or len(cloud) == 0: return b""
    # Input cloud: can be N x 6 (XYZRGB) or N x 7 (XYZRGBC)
    n_points = len(cloud)
    n_cols = cloud.shape[1]
    
    data = np.zeros((n_points, 7), dtype=np.float32)
    
    if n_cols >= 7:
        data[:, :7] = cloud[:, :7]
    else:
        # Default case: Fill first 6, class=0
        data[:, :6] = cloud[:, :6]
    
    # CV to Three.js transform
    # REMOVED: Data is already aligned by AlignmentManagerr to Y-up (GL compatible)
    # data[:, 1] *= -1 
    # data[:, 2] *= -1 
    
    return data.tobytes()

def on_chunk_ready(session, chunk_info):
    asyncio.create_task(chunk_queue.put((session, chunk_info)))
    print(f"[FrameStorage] Chunk {chunk_info.chunk_id} queued")

# --- Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_storage, chunk_processor, alignment_manager, chunk_queue, slam_processor
    print("[Server] Starting STAC-BUILD...")
    
    frame_storage = get_frame_storage()
    frame_storage.on_chunk_ready = on_chunk_ready
    chunk_processor = get_chunk_processor()
    alignment_manager = get_alignment_manager()
    
    # Initialize SAM3 Wrapper (lazy load, but triggers init log)
    get_sam3_wrapper()
    
    # Initialize SLAM Processor (lazy loading of backend)
    slam_backend = cfg.get("slam_backend", "mast3r")
    slam_processor = get_slam_processor()
    print(f"[Server] SLAM backend configured: {slam_backend}")
    
    # NOTE: Models are NOT loaded here for memory efficiency.
    # They will be loaded lazily when needed:
    # - Online streaming: loaded when first frame arrives
    # - Offline: loaded when processing session
    print("[Server] Models will be loaded on-demand (lazy loading enabled)")
    
    worker_task = asyncio.create_task(chunk_processing_worker())
    
    yield
    
    worker_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return HTMLResponse('<html><head><meta http-equiv="refresh" content="0;url=/static/viewer.html"></head></html>')

@app.get("/health")
async def health():
    return {"status": "ok", "session": frame_storage.get_session_info() if frame_storage else {}}

@app.get("/status")
async def status():
    # Use get_total_points (Memory efficient)
    total_pts = alignment_manager.get_total_points() if alignment_manager else 0
    return {
        "chunks_aligned": alignment_manager.get_chunk_count() if alignment_manager else 0,
        "unified_cloud_points": total_pts, 
        "is_processing": is_processing_chunk
    }

@app.get("/sessions")
async def get_sessions():
    """List available scan sessions."""
    try:
        scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
        if not scans_dir.exists(): return []
        
        sessions = []
        for d in sorted(scans_dir.iterdir(), reverse=True):
            if d.is_dir():
                sessions.append({"id": d.name, "date": d.name})
        return sessions
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []

@app.get("/mode")
async def get_mode():
    """Get current server mode and DA3 readiness."""
    return {
        "mode": server_mode,
        "da3_loaded": chunk_processor.is_loaded if chunk_processor else False,
        "ready_for_streaming": server_mode == "online" and (chunk_processor.is_loaded if chunk_processor else False)
    }

@app.get("/segments/{session_id}")
async def get_segments(session_id: str):
    """Get unified segmentation data for a session."""
    try:
        scans_dir = Path(__file__).parent / "scans"
        output_dir = scans_dir / session_id / "output"
        
        # Use display-time matching (masks → cloud)
        masks_file = output_dir / "seg_masks.npz"
        seg_file = output_dir / "segmentation.json"
        if masks_file.exists() and seg_file.exists():
            from segmentation_pipeline import apply_segmentation_to_cloud
            return apply_segmentation_to_cloud(output_dir)
        
        # Fallback to legacy formats
        if seg_file.exists():
            with open(seg_file, 'r') as f:
                return json.load(f)
        
        return {"object_types": [], "instances": []}
    except Exception as e:
        print(f"Error serving segments: {e}")
        return {"error": str(e)}

@app.post("/mode/{new_mode}")
async def set_mode(new_mode: str):
    """
    Switch server mode between 'online' and 'offline'.
    - online: Pre-loads DA3 model for real-time streaming
    - offline: Unloads DA3 to save VRAM (will load on-demand if needed)
    """
    global server_mode
    
    if new_mode not in ["online", "offline"]:
        return {"error": "Invalid mode. Use 'online' or 'offline'"}
    
    old_mode = server_mode
    server_mode = new_mode
    
    if new_mode == "online":
        # Pre-load SLAM backend for streaming readiness
        if not slam_processor.is_initialized:
            print(f"[Server] 🔌 Switching to ONLINE mode - Loading {slam_processor.backend_name}...")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "info",
                "message": f"Loading {slam_processor.backend_name} model for streaming..."
            }))
            
            # Load in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, slam_processor.initialize)
            
            print(f"[Server] ✅ {slam_processor.backend_name} loaded - Ready for streaming!")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "mode_ready",
                "mode": "online",
                "message": "Ready for streaming"
            }))
        else:
            print(f"[Server] {slam_processor.backend_name} already loaded")
            
    elif new_mode == "offline":
        # Unload DA3 to free VRAM
        if chunk_processor.is_loaded:
            print("[Server] 📴 Switching to OFFLINE mode - Unloading DA3...")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "info", 
                "message": "Unloading DA3 to save memory..."
            }))
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, chunk_processor.unload_model)
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[Server] ✅ DA3 unloaded - VRAM freed")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "mode_ready",
                "mode": "offline",
                "message": "Offline mode active"
            }))
    
    return {
        "mode": server_mode,
        "da3_loaded": chunk_processor.is_loaded,
        "changed": old_mode != new_mode
    }

# --- SLAM Processing Endpoints ---

@app.get("/slam/status")
async def slam_status():
    """Get SLAM processor status."""
    if slam_processor is None:
        return {"initialized": False, "backend": cfg.get("slam_backend", "unknown")}
    return slam_processor.get_state()

@app.post("/slam/process/{session_id}")
async def process_session_with_slam(session_id: str):
    """
    Process an existing session's frames with SLAM.
    Uses configured backend (MASt3R or DA3).
    """
    if slam_processor is None:
        return {"error": "SLAM processor not initialized"}
    
    # Get session paths
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_id
    frames_dir = session_dir / "frames"
    output_dir = session_dir / "output"
    
    if not frames_dir.exists():
        return {"error": f"Session not found or no frames: {session_id}"}
    
    # Notify viewers
    await viewer_manager.broadcast_text(json.dumps({
        "type": "info",
        "message": f"Starting SLAM processing for {session_id}..."
    }))
    
    # Process in executor to not block
    loop = asyncio.get_event_loop()
    
    def _process_slam():
        # Initialize SLAM if not already
        slam_processor.initialize()
        
        # Start session
        slam_processor.start_session(session_id, frames_dir, output_dir)
        
        # Process all frames
        frame_count = 0
        keyframe_count = 0
        
        for result in slam_processor.process_frames_directory(frames_dir):
            frame_count += 1
            if result.is_keyframe:
                keyframe_count += 1
                
        # Get final point cloud
        points, colors = slam_processor.get_global_pointcloud()
        
        # Save PLY
        output_dir.mkdir(parents=True, exist_ok=True)
        ply_path = output_dir / "slam_reconstruction.ply"
        slam_processor.save_pointcloud_ply(ply_path)
        
        slam_processor.stop_session()
        
        return {
            "session_id": session_id,
            "frames_processed": frame_count,
            "keyframes": keyframe_count,
            "points": len(points),
            "output_ply": str(ply_path),
        }
    
    try:
        result = await loop.run_in_executor(None, _process_slam)
        
        # Send point cloud to viewers
        binary_data = slam_processor.get_pointcloud_binary()
        if binary_data:
            for sub_chunk in chunk_data(binary_data):
                await viewer_manager.broadcast_binary(sub_chunk)
        
        await viewer_manager.broadcast_text(json.dumps({
            "type": "slam_complete",
            "data": result
        }))
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.websocket("/ws/slam")
async def slam_websocket(websocket: WebSocket):
    """
    Real-time SLAM WebSocket for streaming frame processing.
    Client sends frames, receives incremental point cloud updates.
    """
    await websocket.accept()
    print("[SLAM WS] Client connected")
    
    if slam_processor is None:
        await websocket.close(code=1011, reason="SLAM not initialized")
        return
    
    # Initialize SLAM on first connection
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, slam_processor.initialize)
    
    # Start new session
    session_id = f"live_{int(time.time())}"
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_id
    frames_dir = session_dir / "frames"
    output_dir = session_dir / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slam_processor.start_session(session_id, frames_dir, output_dir)
    
    await websocket.send_text(json.dumps({
        "type": "session_started",
        "session_id": session_id
    }))
    
    frame_idx = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # Decode frame
            import cv2
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame)
            
            # Process with SLAM
            def _process_frame():
                return slam_processor.process_frame(frame_rgb, float(frame_idx))
            
            result = await loop.run_in_executor(None, _process_frame)
            frame_idx += 1
            
            # Send status update
            await websocket.send_text(json.dumps({
                "type": "frame_processed",
                "frame_id": result.frame_id,
                "is_keyframe": result.is_keyframe,
                "status": result.status,
            }))
            
            # If keyframe, send incremental point cloud
            if result.is_keyframe and result.points is not None:
                # Create binary data for this keyframe's points
                n_pts = len(result.points)
                if n_pts > 0:
                    output = np.zeros((n_pts, 7), dtype=np.float32)
                    output[:, :3] = result.points
                    output[:, 3:6] = result.colors if result.colors is not None else 0.7
                    output[:, 6] = 0.0
                    
                    # Send to this client
                    await websocket.send_bytes(output.tobytes())
                    
                    # Also broadcast to viewers
                    await viewer_manager.broadcast_binary(output.tobytes())
                    
    except WebSocketDisconnect:
        print("[SLAM WS] Client disconnected")
    except Exception as e:
        print(f"[SLAM WS] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final reconstruction
        output_ply = output_dir / "slam_reconstruction.ply"
        slam_processor.save_pointcloud_ply(output_ply)
        slam_processor.stop_session()
        print(f"[SLAM WS] Session saved: {session_id}")

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    Camera WebSocket - receives frames from camera.html client.
    Routes to appropriate SLAM backend (MASt3R or DA3 chunks).
    """
    await websocket.accept()
    
    # Get configured backend
    slam_backend = cfg.get("slam_backend", "mast3r")
    
    # For MASt3R, we can work in online or offline mode
    # For DA3, we require online mode (existing behavior)
    if slam_backend != "mast3r" and server_mode != "online":
        print(f"[Camera] ⛔ Connection rejected: Server is in {server_mode} mode")
        await websocket.close(code=1008, reason="Server is offline")
        return
        
    # Check if backend is ready (only if online mode is active/requested)
    if server_mode == "online" and not slam_processor.is_initialized:
        print("[Camera] ⏳ Connection rejected: Model loading...")
        await websocket.close(code=1013, reason="Server initializing")
        return

    await camera_manager.connect(websocket)
    
    # Initialize based on backend
    if slam_backend == "mast3r":
        await _camera_mast3r_flow(websocket)
    else:
        await _camera_da3_flow(websocket)


async def _camera_mast3r_flow(websocket: WebSocket):
    """Handle camera frames with MASt3R-SLAM backend."""
    import cv2
    
    # Initialize SLAM
    loop = asyncio.get_event_loop()
    
    print("[Camera/MASt3R] Initializing SLAM processor...")
    print("[Camera/MASt3R] Initializing SLAM processor...")
    # Use singleton accessor directly to avoid global state issues
    from slam_processor import get_slam_processor
    slam_processor = get_slam_processor()
    
    if slam_processor is None:
        print("❌ CRITICAL ERROR: slam_processor is still None after getter!")
        return

    await loop.run_in_executor(None, slam_processor.initialize)
    
    # Start session
    session_id = f"live_{int(time.time())}"
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_id
    frames_dir = session_dir / "frames"
    output_dir = session_dir / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slam_processor.start_session(session_id, frames_dir, output_dir)
    
    await viewer_manager.broadcast_text(json.dumps({
        "type": "session_started",
        "session_id": session_id,
        "backend": "mast3r"
    }))
    
    frame_idx = 0
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            cv2.imwrite(str(frames_dir / f"{frame_idx:06d}.jpg"), frame)
            
            # Process with SLAM in executo
            def _process():
                return slam_processor.process_frame(frame_rgb, float(frame_idx))
            
            result = await loop.run_in_executor(None, _process)
            frame_idx += 1
            camera_manager.frame_count = frame_idx
            
            # If keyframe, broadcast point cloud to viewers
            if result.is_keyframe and result.points is not None:
                n_pts = len(result.points)
                if n_pts > 0:
                    # Transform points from camera space to world space using pose
                    points_cam = result.points.reshape(-1, 3)
                    if result.pose is not None:
                        R = result.pose[:3, :3]
                        t = result.pose[:3, 3]
                        points_world = (R @ points_cam.T).T + t
                    else:
                        points_world = points_cam
                    
                    output = np.zeros((n_pts, 7), dtype=np.float32)
                    output[:, :3] = points_world
                    output[:, 3:6] = result.colors if result.colors is not None else 0.7
                    output[:, 6] = 0.0
                    
                    # Broadcast to all viewers
                    for sub_chunk in chunk_data(output.tobytes()):
                        await viewer_manager.broadcast_binary(sub_chunk)
                        
            # Status update every 30 frames
            if frame_idx % 30 == 0:
                state = slam_processor.get_state()
                await viewer_manager.broadcast_text(json.dumps({
                    "type": "slam_status",
                    "frames": frame_idx,
                    "keyframes": state.get("backend_state", {}).get("num_keyframes", 0),
                    "fps": state.get("backend_state", {}).get("fps", 0),
                }))
                    
    except Exception as e:
        print(f"[Camera/MASt3R] Connection closed: {e}")
    finally:
        camera_manager.disconnect()
        
        # Save final reconstruction
        print(f"[Camera/MASt3R] Saving session {session_id}...")
        output_ply = output_dir / "slam_reconstruction.ply"
        await loop.run_in_executor(None, lambda: slam_processor.save_pointcloud_ply(output_ply))
        slam_processor.stop_session()
        
        await viewer_manager.broadcast_text(json.dumps({
            "type": "session_complete",
            "session_id": session_id,
            "frames": frame_idx,
            "output": str(output_ply),
        }))


async def _camera_da3_flow(websocket: WebSocket):
    """Handle camera frames with DA3 chunks backend (original behavior)."""
    import cv2
    
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_storage.add_frame(frame)
                camera_manager.frame_count += 1
    except Exception as e:
        print(f"[Camera/DA3] Connection closed: {e}")
    finally:
        camera_manager.disconnect()
        # Finalize Session (Save Poses/Intrinsics)
        if chunk_processor and chunk_processor.is_loaded:
             print("[Camera/DA3] Finalizing session...")
             await asyncio.to_thread(chunk_processor.finalize_session)

@app.websocket("/ws/viewer")
async def viewer_websocket(websocket: WebSocket):
    """
    Viewer Socket.
    NEW LOGIC: Streams history chunk-by-chunk to new clients.
    """
    await websocket.accept()
    await viewer_manager.connect_viewer(websocket)
    
    try:
        await viewer_manager.send_text(websocket, json.dumps({"type": "status", "message": "Connected"}))
        
        # No auto-streaming on connect — viewer loads session explicitly
        # CloudCompPy cleaned cloud is sent only on load_session or after reconstruction
        
        while True:
            try:
                msg = await websocket.receive_text() # Keep alive
            except WebSocketDisconnect:
                print("[Viewer] Client disconnected.")
                break
            except RuntimeError as e:
                # Often "Need to call accept" if connection lost
                print(f"[Viewer] WebSocket Runtime Error: {e}")
                break
            except Exception as e:
                print(f"[Viewer] Receive Error: {e}")
                break

            try:
                cmd = json.loads(msg)
            except:
                continue

            if cmd.get("type") == "status":
                queue_size = chunk_queue.qsize()
                await viewer_manager.send_text(websocket, json.dumps({
                    "type": "status",
                    "camera_connected": camera_manager.active_camera is not None,
                    "chunk_queue": queue_size
                }))
            elif cmd.get("type") == "clear":
                alignment_manager.reset()
                frame_storage.stop_session()
                await viewer_manager.send_text(websocket, json.dumps({"type": "cleared"}))
            elif cmd.get("type") == "set_prompt":
                prompt = cmd.get("prompt", "").strip()
                if prompt:
                    print(f"[Viewer] Setting prompt: {prompt}")
                    frame_storage.set_prompt(prompt)
                
                await viewer_manager.send_text(websocket, json.dumps({"type": "prompt_set", "prompt": prompt}))

                # Retroactive Logic - SEQUENTIAL PROCESSING
                # SAM3 NEVER runs while DA3 is active to avoid VRAM conflicts
                if prompt and alignment_manager:
                    # Check if DA3 is currently processing chunks
                    da3_is_active = is_processing_chunk or chunk_queue.qsize() > 0

                    # Check if we have PLY files (either in memory OR on disk)
                    has_plys_in_memory = alignment_manager.get_chunk_count() > 0
                    has_plys_on_disk = False
                    if frame_storage and frame_storage.current_session:
                        output_dir = frame_storage.current_session.output_dir
                        ply_files = list(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                        has_plys_on_disk = len(ply_files) > 0

                    if da3_is_active:
                        # DA3 is working - queue the segmentation for late
                        print(f"[Viewer] DA3 is active (processing={is_processing_chunk}, queue={chunk_queue.qsize()}).")
                        print(f"[Viewer] Segmentation queued - will run AFTER DA3 completes ALL chunks.")
                        frame_storage._pending_retroactive_prompt = prompt
                        await viewer_manager.send_text(websocket, json.dumps({
                            "type": "info",
                            "message": "Segmentation queued. Will start after reconstruction completes."
                        }))
                        continue  # Don't run now

                    if has_plys_in_memory or has_plys_on_disk:
                         source = "Active Memory" if has_plys_in_memory else "Disk (Offline Session)"
                         chunk_count = alignment_manager.get_chunk_count() if has_plys_in_memory else len(ply_files)
                         print(f"[Viewer] 🔄 Retroactive Segmentation ({source}): {chunk_count} chunks.")
                         
                         # Run in background to not block socket
                         def _retro_process_active():
                             try:
                                 session = frame_storage.current_session
                                 if not session: return False

                                 # Unload DA3 if it's lingering from Online Capture
                                 print("[Retro] Unloading DA3 to free VRAM for SAM3...")
                                 chunk_processor.unload_model()
                                 import gc
                                 gc.collect()
                                 if torch.cuda.is_available():
                                     torch.cuda.empty_cache()

                                 # Resolve prompt (auto-detect if needed)
                                 prompt = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

                                 # Run new segmentation pipeline
                                 from segmentation_pipeline import run_segmentation
                                 result = run_segmentation(
                                     frames_dir=str(session.frames_dir),
                                     output_dir=str(session.output_dir),
                                     prompt=prompt
                                 )

                                 if "error" in result:
                                     print(f"[Retro] Segmentation failed: {result['error']}")
                                     return False

                                 print(f"[Retro] ✅ Segmentation complete: {len(result['instances'])} instances")
                                 return True
                             except Exception as e:
                                 print(f"[Retro] Error: {e}")
                                 import traceback
                                 traceback.print_exc()
                                 return False
                         
                         # Execute in thread
                         async def _run_retro_active():
                             loop = asyncio.get_running_loop()
                             success = await loop.run_in_executor(None, _retro_process_active)
                             if success:
                                 # Refresh viewer: resend cleaned cloud + segmentation
                                 session_id = frame_storage.current_session.session_id
                                 await websocket.send_text(json.dumps({"type": "cleared"}))
                                 
                                 # Resend the cleaned cloud
                                 sent = await _send_cleaned_cloud(websocket, session_id)
                                 
                                 # Apply segmentation against current cloud
                                 from segmentation_pipeline import apply_segmentation_to_cloud
                                 seg_data = apply_segmentation_to_cloud(frame_storage.current_session.output_dir)
                                 if seg_data.get("instances"):
                                     await websocket.send_text(json.dumps(seg_data))
                                 
                                 print(f"[Viewer] Refreshed view with segmented data (cloud={'✅' if sent else '❌'}).")

                         asyncio.create_task(_run_retro_active())
                    else:
                        # MEMORY EMPTY (Loaded Session) -> FULL RE-PROCESS MODE
                        print(f"[Viewer] ⚠️ Archived session detected. Starting FULL offline re-processing (DA3 + SAM3). This may take time.")
                        await websocket.send_text(json.dumps({
                            "type": "info", 
                            "message": "Starting Offline Segmentation (Re-running AI)... Please wait."
                        }))

                        # CAPTURE LOOP for thread-safe broadcasting
                        main_loop = asyncio.get_running_loop()

                        async def _retro_process_offline():
                            try:
                                session = frame_storage.current_session
                                if not session: return False

                                print(f"[Retro-Offline] 🟢 STARTING DA3 INCREMENTAL RECONSTRUCTION (RealtimeDA3)")

                                # 1. Prepare Paths
                                images_dir = session.frames_dir.resolve()
                                output_dir = session.output_dir.resolve()

                                # 1.5. Frame quality analysis (blur detection)
                                from frame_quality import analyze_frames, save_manifest
                                fq_result = analyze_frames(str(images_dir))
                                if "error" not in fq_result:
                                    save_manifest(str(images_dir), fq_result)

                                # 1.6. Visual novelty frame selection (H/F ratio keyframe filter)
                                frame_sel_cfg = cfg.get("frame_selection", {})
                                if frame_sel_cfg.get("enabled", False):
                                    try:
                                        from frame_selector import select_keyframes
                                        sel_result = select_keyframes(str(images_dir), frame_sel_cfg)
                                        print(f"[Retro-Offline] 🎯 Selected {sel_result['selected_count']}/{sel_result['total_frames']} keyframes")
                                    except Exception as e:
                                        print(f"[Retro-Offline] ⚠️ Frame selection failed, using stride fallback: {e}")

                                # 2. Create DA3 Config (all from config.yaml + HF cache)
                                from da3_config_builder import build_da3_config
                                da3_config = build_da3_config(cfg)

                                # 3. Create RealtimeDA3 instance with AlignmentManagerr for gravity correction
                                print(f"[Retro-Offline] Initializing RealtimeDA3...")
                                alignment_manager.reset()  # Reset to compute fresh gravity for this session
                                da3 = RealtimeDA3(
                                    image_dir=str(images_dir),
                                    save_dir=str(output_dir),
                                    config=da3_config,
                                    alignment_manager=alignment_manage
                                )

                                # 4. No-op callback (no streaming during processing)
                                async def on_chunk_complete(chunk_id, sim3_transform):
                                    print(f"[Retro-Offline] Chunk {chunk_id} saved (not streaming)")

                                # 5. Run DA3
                                print(f"[Retro-Offline] Processing {len(da3.img_list)} images...")
                                await da3.process_long_sequence_async(callback=on_chunk_complete)
                                print(f"[Retro-Offline] ✅ DA3 complete!")
                                return True

                            except Exception as e:
                                print(f"[Retro-Offline] Error: {e}")
                                import traceback
                                traceback.print_exc()
                                return False
                        
                        # Run DA3 then CloudCompPy then segment then send
                        success = await _retro_process_offline()
                        
                        if success:
                            session_id = frame_storage.current_session.session_id if frame_storage.current_session else None
                            if session_id:
                                # Run CloudCompPy post-processing
                                postproc_config = cfg.get("postprocessing", {})
                                if postproc_config.get("enabled", False):
                                    await _run_cloudcompy_postprocess(session_id, postproc_config, websocket)
                                
                                # Run segmentation pipeline if prompt was set
                                if prompt:
                                    session = frame_storage.current_session
                                    try:
                                        # Resolve prompt (auto-detect if needed)
                                        prompt = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

                                        from segmentation_pipeline import run_segmentation
                                        result = run_segmentation(
                                            frames_dir=str(session.frames_dir),
                                            output_dir=str(session.output_dir),
                                            prompt=prompt
                                        )
                                        print(f"[Retro-Offline] Segmentation: {len(result.get('instances', []))} instances")
                                    except Exception as e:
                                        print(f"[Retro-Offline] Segmentation error: {e}")
                                
                                # Send cleaned cloud to viewer
                                await _send_cleaned_cloud(websocket, session_id)
                                
                                # Apply segmentation against current cloud
                                output_dir = Path(__file__).parent / "scans" / session_id / "output"
                                from segmentation_pipeline import apply_segmentation_to_cloud
                                seg_data = apply_segmentation_to_cloud(output_dir)
                                if seg_data.get("instances"):
                                    await websocket.send_text(json.dumps(seg_data))
                            
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "Offline processing complete"
                            }))
                            print("[Viewer] Offline Processing Complete.")

            elif cmd.get("type") == "load_session":
                session_id = cmd.get("session_id")
                print(f"[Viewer] Loading session {session_id}...")
                
                # Clear current view
                alignment_manager.reset()
                frame_storage.stop_session()
                
                # Re-activate session in FrameStorage (Offline Mode)
                # This enables _retro_process to find files if needed
                if frame_storage:
                    frame_storage.load_session_from_disk(session_id)
                
                await websocket.send_text(json.dumps({"type": "cleared"}))
                
                # Get chunks to stream
                history = []
                if alignment_manager and alignment_manager.get_chunk_count() > 0:
                     history = alignment_manager.aligned_chunks
                elif frame_storage and frame_storage.current_session:
                     # Fallback if alignment manager reset but session loaded (though offline re-proc fills align-mgr)
                     # Actually offline re-proc fills it. But initial load might trigger this before re-proc?
                     # No, 'load_session' just loads PLY.
                     # But we need to know how many.
                     pass
                
                # If we just loaded from disk, alignment_manager is empty until re-proc OR untill we manually load PLYs into it?
                # Wait, 'load_session' below streams from 'frame_storage.save_ply' locations?
                # The original code streamed from `alignment_manager.aligned_chunks`.
                # If we use `load_session_from_disk`, `alignment_manager` is RESET.
                # So `history` is EMPTY.
                # But the code below (lines 352+) iterates `history`!
                # Logic Fix: We need to populate `history` from DISK if alignment_manager is empty.
                
                # Actually, the original 'load_session' loop (now at line ~415 in file) reads from disk?
                # No, it iterates `alignment_manager.aligned_chunks`. 
                # !!! If `alignment_manager` is reset, we stream NOTHING.
                # Use FrameStorage to get file list effectively.
                
                total_chunks = 0
                if frame_storage and frame_storage.current_session:
                    total_chunks = len(frame_storage.current_session.chunks)

                # Send Metadata
                await websocket.send_text(json.dumps({
                    "type": "session_info",
                    "session_id": session_id,
                    "mode": "Offline",
                    "total_chunks": total_chunks
                }))
                
                # Stream saved point clouds
                try:
                    scans_dir = Path(__file__).parent / "scans"
                    output_dir = scans_dir / session_id / "output"

                    # ── Send cleaned_cloud.ply (from CloudCompPy) ──
                    cleaned_ply = output_dir / "cleaned_cloud.ply"
                    
                    if cleaned_ply.exists():
                        # Direct send of cleaned cloud
                        sent = await _send_cleaned_cloud(websocket, session_id)
                        if sent:
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": f"Loaded cleaned cloud from {session_id}"
                            }))
                            # Apply segmentation against current cloud
                            from segmentation_pipeline import apply_segmentation_to_cloud
                            seg_data = apply_segmentation_to_cloud(output_dir)
                            if seg_data.get("instances"):
                                await websocket.send_text(json.dumps(seg_data))
                                print(f"[Viewer] Sent segmentation ({len(seg_data['instances'])} instances)")
                        else:
                            await websocket.send_text(json.dumps({"type": "error", "message": "Failed to load cleaned cloud"}))
                    else:
                        # No cleaned cloud yet — check if chunks exist to run CloudCompPy
                        chunk_plys = sorted(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                        if chunk_plys:
                            print(f"[Viewer] No cleaned_cloud.ply found. Running CloudCompPy on {len(chunk_plys)} chunks...")
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": f"Building cleaned cloud from {len(chunk_plys)} chunks..."
                            }))
                            postproc_config = cfg.get("postprocessing", {})
                            await _run_cloudcompy_postprocess(session_id, postproc_config, websocket)
                            
                            if cleaned_ply.exists():
                                sent = await _send_cleaned_cloud(websocket, session_id)
                                if sent:
                                    await websocket.send_text(json.dumps({
                                        "type": "status",
                                        "message": f"Loaded cleaned cloud from {session_id}"
                                    }))
                                    # Apply segmentation against current cloud
                                    from segmentation_pipeline import apply_segmentation_to_cloud as _apply_seg
                                    seg_data2 = _apply_seg(output_dir)
                                    if seg_data2.get("instances"):
                                        await websocket.send_text(json.dumps(seg_data2))
                            else:
                                await websocket.send_text(json.dumps({"type": "error", "message": "CloudCompPy failed to produce cleaned cloud"}))
                        else:
                            await websocket.send_text(json.dumps({"type": "error", "message": "No point clouds found for this session"}))

                except Exception as e:
                    print(f"Error loading session: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd.get("type") == "reconstruct_geometry":
                # Geometry-only reconstruction (no segmentation)
                session_id = cmd.get("session_id")
                print(f"[Viewer] 🔧 Reconstructing geometry ONLY for session {session_id}...")
                
                # Load session first
                alignment_manager.reset()
                frame_storage.stop_session()
                if frame_storage:
                    frame_storage.load_session_from_disk(session_id)
                
                await websocket.send_text(json.dumps({
                    "type": "info", 
                    "message": f"Starting geometry reconstruction for {session_id}... This may take several minutes."
                }))
                
                main_loop = asyncio.get_running_loop()
                
                async def _reconstruct_geometry_only():
                    try:
                        session = frame_storage.current_session
                        if not session: return False

                        # Check which backend to use
                        backend = cfg.get("slam_backend", "mast3r")
                        images_dir = session.frames_dir.resolve()
                        output_dir = session.output_dir.resolve()
                        
                        if backend in ("mast3r", "hybrid"):
                            # === MAST3R / HYBRID RECONSTRUCTION ===
                            mode_name = "HYBRID (MASt3R + DA3)" if backend == "hybrid" else "MAST3R"
                            print(f"[Reconstruct] 🟢 STARTING {mode_name} RECONSTRUCTION")
                            
                            # Initialize SLAM processor if needed
                            if not slam_processor.is_initialized:
                                print("[Reconstruct] Initializing MASt3R...")
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, slam_processor.initialize)
                            
                            # Process frames synchronously (simpler, no deadlock)
                            def _process_mast3r():
                                slam_processor.start_session(
                                    session.session_id, 
                                    images_dir, 
                                    output_dir
                                )
                                
                                frame_count = 0
                                keyframe_count = 0
                                
                                for result in slam_processor.process_frames_directory(images_dir):
                                    frame_count += 1
                                    if result.is_keyframe:
                                        keyframe_count += 1
                                        
                                return frame_count, keyframe_count
                            
                            loop = asyncio.get_event_loop()
                            frame_count, keyframe_count = await loop.run_in_executor(None, _process_mast3r)
                            
                            # In hybrid mode: DA3 dense depth with MASt3R metric poses
                            if backend == "hybrid":
                                print(f"[Reconstruct] 🔄 Phase 2: Running DA3 dense depth with MASt3R metric poses...")
                                
                                # Run DA3 densification (no streaming callback — CloudCompPy at end)
                                success_da3 = await loop.run_in_executor(
                                    None,
                                    lambda: slam_processor.run_hybrid_densification(
                                        images_dir, output_dir,
                                        on_chunk_callback=None
                                    )
                                )
                                if not success_da3:
                                    raise RuntimeError("[Hybrid] DA3 densification returned no points. Cannot proceed.")
                                print(f"[Reconstruct] ✅ DA3 hybrid densification complete")
                            
                            # Save PLY (needed for CloudCompPy input or as fallback)
                            points, colors = slam_processor.get_global_pointcloud()
                            if points is not None and len(points) > 0:
                                print(f"[Reconstruct] {len(points):,} points from {mode_name} (not streaming — CloudCompPy will process)")
                            
                            # Save PLY
                            ply_path = output_dir / "slam_reconstruction.ply"
                            await loop.run_in_executor(
                                None, 
                                lambda: slam_processor.save_pointcloud_ply(ply_path)
                            )
                            
                            slam_processor.stop_session()
                            
                            print(f"[Reconstruct] ✅ {mode_name} complete: {frame_count} frames, {keyframe_count} keyframes")
                            return True
                            
                        else:
                            # === DA3 RECONSTRUCTION ===
                            print(f"[Reconstruct] 🟢 STARTING DA3 INCREMENTAL RECONSTRUCTION (RealtimeDA3)")

                            # Frame quality analysis (blur detection)
                            from frame_quality import analyze_frames, save_manifest
                            fq_result = analyze_frames(str(images_dir))
                            if "error" not in fq_result:
                                save_manifest(str(images_dir), fq_result)

                            # Visual novelty frame selection (H/F ratio keyframe filter)
                            frame_sel_cfg = cfg.get("frame_selection", {})
                            if frame_sel_cfg.get("enabled", False):
                                try:
                                    from frame_selector import select_keyframes
                                    sel_result = select_keyframes(str(images_dir), frame_sel_cfg)
                                    print(f"[Reconstruct] 🎯 Selected {sel_result['selected_count']}/{sel_result['total_frames']} keyframes")
                                except Exception as e:
                                    print(f"[Reconstruct] ⚠️ Frame selection failed, using stride fallback: {e}")

                            # Create DA3 Config (all from config.yaml + HF cache)
                            from da3_config_builder import build_da3_config
                            da3_config = build_da3_config(cfg)

                            # 3. Create RealtimeDA3 instance
                            print(f"[Reconstruct] Initializing RealtimeDA3...")
                            da3 = RealtimeDA3(
                                image_dir=str(images_dir),
                                save_dir=str(output_dir),
                                config=da3_config,
                                alignment_manager=alignment_manager
                            )

                            # 4. No-op callback (no real-time streaming — only CloudCompPy at end)
                            async def on_chunk_complete(chunk_id, sim3_transform):
                                print(f"[Reconstruct] Chunk {chunk_id} saved (not streaming)")

                            # 5. Run DA3
                            print(f"[Reconstruct] Processing {len(da3.img_list)} images...")
                            await da3.process_long_sequence_async(callback=on_chunk_complete)

                            print(f"[Reconstruct] ✅ DA3 complete!")
                            return True

                    except Exception as e:
                        print(f"[Reconstruct] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        return False

                # Wrapper to run internally and handle completion
                async def _background_reconstruction():
                    success = await _reconstruct_geometry_only()
                    try:
                        if success:
                            # ── CloudCompPy Post-Processing ──
                            postproc_config = cfg.get("postprocessing", {})
                            if postproc_config.get("enabled", False):
                                await _run_cloudcompy_postprocess(session_id, postproc_config, websocket)
                            
                            # ── Send cleaned cloud to viewer ──
                            await _send_cleaned_cloud(websocket, session_id)
                            
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": f"Geometry reconstruction complete for {session_id}"
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": "Geometry reconstruction failed. Check server logs."
                            }))
                    except:
                         print("[Reconstruct] Failed to send completion message (client disconnected).")

                # Fire and forget (don't await)
                asyncio.create_task(_background_reconstruction())
                print(f"[Viewer] Background task started for session {session_id}")

    except Exception as e:
        print(f"[Viewer] ❌ WebSocket Error: {e}")
        import traceback
        traceback.print_exc()
        viewer_manager.disconnect_viewer(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")