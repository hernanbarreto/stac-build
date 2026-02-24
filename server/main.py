# STAC-BUILD: Main Server
# PHASE 3 COMPLETE: Incremental Streaming (Logic Fixed)

import asyncio
import json
import time
import logging
import gc
import os
import sys
import threading
import torch
import numpy as np
from pathlib import Path
from typing import Set, Optional
from contextlib import asynccontextmanager

# Suppress /health spam from access logs (applies to all uvicorn start modes)
class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

from frame_storage import get_frame_storage, FrameStorage
from chunk_processor import get_chunk_processor, ChunkProcessor, ChunkResult
from alignment_manager import get_alignment_manager, AlignmentManager
from sam3_wrapper import get_sam3_wrapper
from config import cfg
from da3_native_wrapper import RealtimeDA3
from slam_processor import get_slam_processor, SLAMProcessor, SLAMFrame
from pipeline_manager import PipelineManager, PipelineStage, StageId

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

def _align_cloud_to_floor(output_data: np.ndarray, session_dir: Path = None) -> np.ndarray:
    """
    Apply RANSAC floor alignment to the final cloud.
    Finds the floor plane and rotates so it sits at y=0 (XZ plane).
    input/output: [N, 7] float32 (x, y, z, r, g, b, classId)
    
    If session_dir is provided, persists the transform to disk so
    subsequent loads produce identical alignment.
    """
    if output_data is None or len(output_data) < 100:
        print("[FloorAlign] ⚠️ Not enough points for alignment")
        return output_data
    
    try:
        # ── Try loading cached transform from disk ──
        if session_dir:
            transform_path = Path(session_dir) / "output" / "floor_transform.npz"
            if transform_path.exists():
                data = np.load(transform_path)
                s_val = float(data['s'])
                R = data['R']
                t = data['t']
                print(f"[FloorAlign] ✅ Loaded cached floor transform from disk")
                
                xyz = output_data[:, :3]
                xyz_aligned = s_val * (xyz @ R.T) + t
                result = output_data.copy()
                result[:, :3] = xyz_aligned
                y_min = xyz_aligned[:, 1].min()
                y_max = xyz_aligned[:, 1].max()
                print(f"[FloorAlign] ✅ Floor aligned to y=0 (range: {y_min:.3f} to {y_max:.3f})")
                return result
        
        # ── Compute alignment (first time) ──
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
        
        # Save transform to disk for consistent future loads
        if session_dir:
            transform_path = Path(session_dir) / "output" / "floor_transform.npz"
            try:
                transform_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(transform_path, s=np.array(s), R=R, t=t)
                print(f"[FloorAlign] 💾 Saved floor transform to {transform_path}")
            except Exception as e:
                print(f"[FloorAlign] ⚠️ Could not save transform: {e}")
        
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
        
        # Offload heavy PLY parsing + floor alignment to thread pool
        session_dir = Path(__file__).parent / "scans" / session_id
        
        def _load_and_align():
            """Heavy sync: read PLY, parse, align floor."""
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
                return None, 0, False
            
            if output_data.shape[1] != 7:
                tmp = np.zeros((point_count, 7), dtype=np.float32)
                tmp[:, :6] = output_data[:, :6]
                output_data = tmp
            
            # Floor alignment
            output_data = _align_cloud_to_floor(output_data, session_dir=session_dir)
            return output_data.tobytes(), point_count, has_origins
        
        loop = asyncio.get_event_loop()
        binary_bytes, point_count, has_origins = await loop.run_in_executor(None, _load_and_align)
        
        if binary_bytes is None:
            print(f"[SendCloud] ⚠️ Empty cloud")
            return False
        
        # Send via proper viewer protocol (async, already non-blocking)
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
        session_dir = Path(__file__).parent / "scans" / session_id
        output_data = _align_cloud_to_floor(output_data, session_dir=session_dir)
        
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
pipeline_manager = PipelineManager()  # Pipeline orchestrator (subprocess workers)

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


def _resolve_segmentation_prompt(prompt: str, frames_dir: str) -> tuple:
    """
    Resolve segmentation prompt: if 'auto' or empty, use InternVL3 scene analyzer.
    
    The VLM runs AFTER DA3 unload and BEFORE SAM3 load, so no VRAM conflict.
    
    Returns:
        (prompt, frame_map) where frame_map maps category label → list of frame filenames
    """
    if prompt and prompt.lower() != "auto":
        return prompt, {}
    
    print("[SceneAnalyzer] Auto-detecting categories with InternVL3...")
    try:
        from scene_analyzer import analyze_scene
        scene_cfg = cfg.get("scene_analysis", {})
        auto_prompt, frame_map = analyze_scene(frames_dir, scene_cfg)
        if auto_prompt:
            print(f"[SceneAnalyzer] ✅ Auto-detected prompt: '{auto_prompt}'")
            return auto_prompt, frame_map
        else:
            print("[SceneAnalyzer] ⚠️ No categories detected, falling back to generic")
            return "floor;wall;ceiling;door;window;furniture;object", {}
    except Exception as e:
        print(f"[SceneAnalyzer] ❌ Error: {e}. Using fallback categories.")
        import traceback
        traceback.print_exc()
        return "floor;wall;ceiling;door;window;furniture;object", {}


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
        prompt, frame_map = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

        # Run new segmentation pipeline
        from segmentation_pipeline import run_segmentation
        result = run_segmentation(
            frames_dir=str(session.frames_dir),
            output_dir=str(session.output_dir),
            prompt=prompt,
            frame_map=frame_map
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

    # Initialize auth database
    from db import init_db
    await init_db()
    
    yield
    
    worker_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Potree octree file serving ──────────────────────────────────────────
# Serves pre-built LOD octree files (metadata.json, octree.bin, hierarchy.bin)
from potree_converter import convert_ply_to_potree, convert_ply_to_potree_async

SCANS_DIR = Path(__file__).parent / "scans"

@app.get("/potree/{session_id}/{file_path:path}")
async def serve_potree_files(session_id: str, file_path: str):
    """Serve Potree octree files for a session."""
    from fastapi.responses import FileResponse
    full_path = SCANS_DIR / session_id / "output" / "potree" / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    # Set proper content type for binary files
    content_type = "application/json" if file_path.endswith(".json") else "application/octet-stream"
    return FileResponse(
        str(full_path),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},  # Cache 1h
    )

# Mount auth routes
from routes_auth import router as auth_router
app.include_router(auth_router)

# Mount team routes
from routes_team import router as team_router
app.include_router(team_router)

@app.get("/")
async def root():
    return HTMLResponse('<html><head><meta http-equiv="refresh" content="0;url=/static/viewer.html"></head></html>')

@app.get("/health")
async def health():
    return {"status": "ok", "session": frame_storage.get_session_info() if frame_storage else {}}

# ─── Console Log Streaming via /ws/logs ─────────────────────────────────
class LogBroadcaster:
    """Manages WebSocket clients subscribed to server logs."""
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def _broadcast(self, message: str):
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def emit(self, level: str, message: str):
        """Thread-safe emit — schedule onto the event loop."""
        payload = json.dumps({
            "ts": time.strftime("%H:%M:%S"),
            "level": level,
            "msg": message.rstrip()
        })
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

log_broadcaster = LogBroadcaster()

class WebSocketLogHandler(logging.Handler):
    """Routes Python log records to connected console WebSockets."""
    LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }
    def emit(self, record: logging.LogRecord):
        try:
            level = self.LEVEL_MAP.get(record.levelno, "info")
            log_broadcaster.emit(level, self.format(record))
        except Exception:
            pass

class StreamCapture:
    """Captures print() / stderr output and forwards to LogBroadcaster."""
    def __init__(self, original, level: str):
        self._original = original
        self._level = level

    def write(self, text: str):
        self._original.write(text)
        if text.strip():
            log_broadcaster.emit(self._level, text)

    def flush(self):
        self._original.flush()

    def isatty(self):
        return False

# Install log handler + stream captures
_ws_log_handler = WebSocketLogHandler()
_ws_log_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logging.getLogger().addHandler(_ws_log_handler)
logging.getLogger("uvicorn").addHandler(_ws_log_handler)
logging.getLogger("uvicorn.access").addHandler(_ws_log_handler)
sys.stdout = StreamCapture(sys.__stdout__, "info")
sys.stderr = StreamCapture(sys.__stderr__, "error")

@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    log_broadcaster.set_loop(asyncio.get_event_loop())
    await log_broadcaster.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        await log_broadcaster.disconnect(websocket)

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
async def get_sessions(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """List available scan sessions filtered by team assignment.
    Admins see all sessions. Other roles see only sessions assigned to their teams."""
    from auth import decode_token
    from db import async_session_factory
    from db_team import TeamMember, SessionAssignment, Team
    from sqlalchemy import select

    # ── Resolve user ───────────────────────────────────────────
    user_info = None
    if credentials:
        try:
            payload = decode_token(credentials.credentials)
            user_info = {
                "id": int(payload.get("sub", 0)),
                "role": payload.get("role", "viewer"),
            }
        except Exception:
            pass

    # ── Determine allowed session IDs ──────────────────────────
    allowed_session_ids: set | None = None  # None = allow all

    if user_info and user_info["role"] != "admin":
        allowed_session_ids = set()
        async with async_session_factory() as session:
            # Get teams the user belongs to (as member or manager)
            member_q = await session.execute(
                select(TeamMember.team_id).where(TeamMember.user_id == user_info["id"])
            )
            managed_q = await session.execute(
                select(Team.id).where(Team.manager_id == user_info["id"])
            )
            team_ids = set(r[0] for r in member_q.all()) | set(r[0] for r in managed_q.all())

            if team_ids:
                sa_q = await session.execute(
                    select(SessionAssignment.session_id).where(
                        SessionAssignment.team_id.in_(team_ids)
                    )
                )
                allowed_session_ids = set(r[0] for r in sa_q.all())

    try:
        scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
        if not scans_dir.exists(): return []
        
        sessions = []
        for d in sorted(scans_dir.iterdir(), reverse=True):
            if d.is_dir():
                # Filter by team assignment (non-admin)
                if allowed_session_ids is not None and d.name not in allowed_session_ids:
                    continue

                # Count frames
                frames_dir = d / "frames"
                frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
                
                # Check output
                output_dir = d / "output"
                has_cloud = (output_dir / "cleaned_cloud.ply").exists() if output_dir.exists() else False
                has_segments = (output_dir / "segmentation.json").exists() if output_dir.exists() else False
                
                # Count PLY chunks
                ply_files = list(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                
                # Cloud size (MB)
                cloud_size_mb = 0
                if has_cloud:
                    cloud_size_mb = round((output_dir / "cleaned_cloud.ply").stat().st_size / (1024*1024), 1)
                
                sessions.append({
                    "id": d.name,
                    "date": d.name,
                    "frame_count": frame_count,
                    "chunk_count": len(ply_files),
                    "has_cloud": has_cloud,
                    "has_segments": has_segments,
                    "cloud_size_mb": cloud_size_mb,
                })
        return sessions
    except Exception as e:
        print(f"Error listing sessions: {e}")
        return []

@app.post("/sessions")
async def create_session(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Create a new empty session folder. Admin only."""
    from auth import decode_token

    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    body = await request.json()
    session_name = body.get("name", "").strip()
    if not session_name:
        raise HTTPException(status_code=400, detail="Session name is required")

    # Sanitize: only allow alphanumeric, dashes, underscores
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_name):
        raise HTTPException(status_code=400, detail="Session name can only contain letters, numbers, dashes, and underscores")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_name

    if session_dir.exists():
        raise HTTPException(status_code=409, detail=f"Session '{session_name}' already exists")

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "frames").mkdir(exist_ok=True)
    (session_dir / "output").mkdir(exist_ok=True)

    return {"ok": True, "session_id": session_name}

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

# --- Frame Image Serving ---

@app.get("/api/sessions/{session_id}/frames/{filename}")
async def serve_session_frame(session_id: str, filename: str):
    """Serve individual frame images from a session's frames directory."""
    from fastapi.responses import FileResponse

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    frame_path = scans_dir / session_id / "frames" / filename

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {filename}")

    return FileResponse(str(frame_path), media_type="image/jpeg")

@app.get("/api/sessions/{session_id}/keyframes")
async def get_session_keyframes(session_id: str):
    """Return the list of keyframe filenames from selected_frames.json."""
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    frames_dir = scans_dir / session_id / "frames"
    selected_json = frames_dir / "selected_frames.json"

    if not selected_json.exists():
        raise HTTPException(status_code=404, detail="selected_frames.json not found (DA3 must run first)")

    try:
        with open(selected_json) as f:
            data = json.load(f)
        selected_files = data.get("selected_files", [])
        # Filter to only files that exist on disk
        existing = [fn for fn in selected_files if (frames_dir / fn).exists()]
        return {"keyframes": sorted(existing), "total": len(existing)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Interactive Segmentation Endpoints ---

@app.post("/api/segmentation/start_session/{session_id}")
async def start_interactive_segmentation(session_id: str):
    """
    Initializes SAM3 video predictor for manual, interactive segmentation.
    Loads only keyframes (from selected_frames.json) for faster processing.
    """
    from sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    
    # Needs to run in executor
    loop = asyncio.get_event_loop()
    
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_id
    frames_dir = session_dir / "frames"
    
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames directory not found")

    # Read keyframes from selected_frames.json (same as the keyframes API endpoint)
    keyframes = None
    selected_frames_path = frames_dir / "selected_frames.json"
    if selected_frames_path.exists():
        with open(selected_frames_path) as f:
            sf_data = json.load(f)
        keyframes = sf_data.get("selected_files", [])
        # Filter to only files that exist
        keyframes = [fn for fn in keyframes if (frames_dir / fn).exists()]
        if keyframes:
            print(f"[InteractiveSeg] Using {len(keyframes)} keyframes from selected_frames.json")
    
    if not keyframes:
        # Fallback: use all frames
        keyframes = sorted([f.name for f in frames_dir.iterdir() 
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        print(f"[InteractiveSeg] No selected_frames.json, using all {len(keyframes)} frames")

    try:
        def _init():
            return sam3.init_interactive_session(str(frames_dir), keyframes=keyframes)
        
        state_id = await loop.run_in_executor(None, _init)
        
        # Build kf_index → original filename mapping for the frontend
        keyframes_sorted = sorted(keyframes)
        kf_mapping = {i: name for i, name in enumerate(keyframes_sorted)}
        
        return {
            "ok": True, 
            "state_id": state_id, 
            "session_id": session_id,
            "num_keyframes": len(keyframes_sorted),
            "kf_mapping": kf_mapping,  # SAM3 idx → original filename
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segmentation/add_prompt")
async def add_interactive_prompt(request: Request):
    """
    Apply a positive or negative click to a specific frame.
    Returns the predicted mask as base64 PNG for live overlay.
    """
    body = await request.json()
    state_id = body.get("state_id")
    session_id = body.get("session_id")  # needed to find segmentation.json for resize
    frame_idx = body.get("frame_idx")
    obj_id = body.get("obj_id", 1)
    points = body.get("points")      # list of [x, y]
    labels = body.get("labels")      # list of 1 (positive) or 0 (negative)
    
    if not all([state_id, frame_idx is not None, points, labels]):
        raise HTTPException(status_code=400, detail="Missing required parameters")
        
    from sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    loop = asyncio.get_event_loop()
    
    # Read original resolution from segmentation.json (so we can resize the mask)
    original_res = None
    if session_id:
        scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
        seg_path = scans_dir / session_id / "output" / "segmentation.json"
        if seg_path.exists():
            with open(seg_path) as f:
                seg_data = json.load(f)
            original_res = seg_data.get("resolution", {}).get("original")  # [H, W]
    
    try:
        def _prompt():
            result = sam3.add_interactive_prompt(
                state_id=state_id,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=np.array(points, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32)
            )
            
            # Encode mask as base64 PNG for frontend overlay
            mask_b64 = None
            if result.get("mask") is not None:
                import cv2
                import base64
                mask = result["mask"]
                
                # Resize mask to original resolution (SAM3 returns scaled resolution)
                if original_res and len(original_res) == 2:
                    orig_h, orig_w = original_res  # format is [H, W]
                    mask = cv2.resize(mask.astype(np.uint8),
                                      (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
                
                # Convert binary mask [H, W] to RGBA PNG with semi-transparent green
                h, w = mask.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                mask_bool = mask > 0
                rgba[mask_bool, 1] = 200  # Green
                rgba[mask_bool, 3] = 128  # 50% alpha
                _, png_buf = cv2.imencode('.png', rgba)
                mask_b64 = base64.b64encode(png_buf.tobytes()).decode('ascii')
            
            return result["success"], mask_b64
            
        success, mask_b64 = await loop.run_in_executor(None, _prompt)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add prompt")
        
        resp = {"ok": True}
        if mask_b64:
            resp["mask_png"] = mask_b64
        return resp
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/alignment")
async def save_alignment(session_id: str, request: Request):
    """Save a new floor alignment transform from the 3D viewer gizmo."""
    body = await request.json()
    transform = body.get("transform")

    if not transform or len(transform) != 16:
        raise HTTPException(status_code=400, detail="Missing or invalid transform (need 16 floats)")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    import numpy as np

    # Parse column-major 4x4 matrix (Three.js format)
    M = np.array(transform, dtype=np.float64).reshape(4, 4, order='F')

    # Decompose into s, R, t
    # M[:3,:3] = s * R, M[:3,3] = t
    upper = M[:3, :3]
    s = float(np.linalg.det(upper) ** (1.0/3.0))
    if abs(s) < 1e-10:
        raise HTTPException(status_code=400, detail="Degenerate transform (scale ≈ 0)")

    R = upper / s
    t = M[:3, 3]

    # Ensure R is a proper rotation (orthonormalize via SVD)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R

    # Save to floor_transform.npz
    transform_path = output_dir / "floor_transform.npz"
    np.savez(transform_path, s=np.array(s), R=R, t=t)
    print(f"[Alignment] ✅ Saved floor_transform.npz for {session_id}")
    print(f"[Alignment]   s={s:.6f}")
    print(f"[Alignment]   R=\n{R}")
    print(f"[Alignment]   t={t}")
    # Verify round-trip: reconstruct M and compare
    M_check = np.eye(4)
    M_check[:3, :3] = s * R
    M_check[:3, 3] = t
    print(f"[Alignment]   Original M:\n{M}")
    print(f"[Alignment]   Reconstructed M:\n{M_check}")
    print(f"[Alignment]   Max diff: {np.max(np.abs(M - M_check)):.8f}")

    # segmentation_result.json cache will auto-invalidate on next load
    # (apply_segmentation_to_cloud checks if floor_transform.npz is newer)
    seg_result_path = output_dir / "segmentation_result.json"
    if seg_result_path.exists():
        print(f"[Alignment] ℹ️ segmentation_result.json will auto-invalidate on next session load")

    return {"ok": True, "scale": s}


@app.post("/api/segmentation/propagate")
async def propagate_interactive_segmentation(request: Request):
    """
    SSE streaming propagation — sends per-frame progress events.
    Events:
      - progress: {frame, total, pct, mask_png}
      - saving: {status}
      - done: {instances}
      - error: {message}
    """
    from starlette.responses import StreamingResponse
    
    body = await request.json()
    state_id = body.get("state_id")
    session_id = body.get("session_id")
    label_name = body.get("label_name", "manual_object")
    
    if not state_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing state_id or session_id")
    
    from sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    
    def sse_generator():
        import base64, cv2
        all_masks = {}  # frame_idx → outputs
        
        try:
            for frame_idx, num_frames, outputs in sam3.propagate_interactive_stream(state_id):
                all_masks[frame_idx] = outputs
                pct = round((len(all_masks) / max(1, num_frames)) * 100)
                
                # Generate a small mask preview PNG for this frame
                mask_b64 = ""
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if mask.ndim == 3:
                        mask = np.max(mask, axis=0) if mask.shape[0] > 0 else np.zeros((1,1), dtype=np.uint8)
                    h, w = mask.shape[:2]
                    rgba = np.zeros((h, w, 4), dtype=np.uint8)
                    mask_bool = mask > 0
                    rgba[mask_bool, 1] = 200
                    rgba[mask_bool, 3] = 128
                    _, png_buf = cv2.imencode('.png', rgba)
                    mask_b64 = base64.b64encode(png_buf.tobytes()).decode('ascii')
                
                event = json.dumps({
                    "frame": int(frame_idx), "total": int(num_frames),
                    "pct": pct, "mask_png": mask_b64
                })
                yield f"event: progress\ndata: {event}\n\n"
            
            # All frames done — save results
            yield f"event: saving\ndata: {{\"status\": \"Saving masks...\"}}\n\n"
            
            from segmentation_pipeline import _save_masks, _parse_raw_masks, _match_and_save_result
            
            structured_masks = _parse_raw_masks(all_masks)
            
            from time import time
            temp_global_id = int(time() % 10000)
            
            remapped_masks = {}
            for f_idx, frame_masks in structured_masks.items():
                remapped_masks[f_idx] = {}
                for l_id, mask in frame_masks.items():
                    remapped_masks[f_idx][temp_global_id] = mask
            
            obj_labels = {temp_global_id: label_name}
            
            stale_result = output_dir / "segmentation_result.json"
            if stale_result.exists():
                stale_result.unlink()
            
            _save_masks(output_dir, remapped_masks, [label_name], obj_labels, cfg)
            
            yield f"event: saving\ndata: {{\"status\": \"Computing 3D matching...\"}}\n\n"
            
            result = _match_and_save_result(output_dir)
            
            done_event = json.dumps({
                "ok": True,
                "instances": result.get("instances", [])
            })
            yield f"event: done\ndata: {done_event}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"event: error\ndata: {{\"message\": \"{str(e)}\"}}\n\n"
    
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# --- Segmentation Manager Endpoints ---

@app.get("/api/sessions/{session_id}/segmentation")
async def get_segmentation_instances(session_id: str):
    """Return all existing segmentation instances for a session.
    Prefers segmentation_result.json (grouped by instance_id, with OBBs)
    over segmentation.json (raw obj_ids from SAM3).
    """
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    
    # Prefer the grouped result (has real instance count, OBBs, point_indices)
    result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"

    if not seg_path.exists():
        return {"instances": [], "prompts": []}

    try:
        # Load prompts from segmentation.json (always)
        with open(seg_path) as f:
            seg_data = json.load(f)
        prompts = seg_data.get("prompts", [])
        resolution = seg_data.get("resolution", {})
        
        # Load instances from segmentation_result.json if available
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            instances = result_data.get("instances", [])
        else:
            # Fallback to raw segmentation.json
            instances = seg_data.get("instances", [])
        
        return {
            "instances": instances,
            "prompts": prompts,
            "resolution": resolution,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/segmentation/mask/{instance_id}")
async def get_instance_mask(session_id: str, instance_id: int, frame: str = "", kf_index: int = -1):
    """Render a colored mask PNG for a specific instance on a specific frame.
    
    instance_id can be either:
      - A grouped instance_id from segmentation_result.json (groups obj_ids by instance_id)
      - A raw obj_id from NPZ
    
    Query params:
        frame: frame filename (e.g. '000042.jpg') — fallback for numeric index
        kf_index: sequential keyframe index (0-based) — preferred, matches NPZ frame indices
    """
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    masks_path = output_dir / "seg_masks.npz"
    seg_path = output_dir / "segmentation.json"

    if not masks_path.exists():
        # Return transparent 1x1 PNG instead of 404 to avoid console spam
        import io as _io
        buf = _io.BytesIO()
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")

    loop = asyncio.get_event_loop()

    try:
        def _render():
            npz = np.load(masks_path, allow_pickle=True)

            # Get color from segmentation_result.json or segmentation.json
            color_hex = "#00FF00"  # default green
            original_res = None
            
            # Try segmentation_result.json first (grouped instances)
            result_path = output_dir / "segmentation_result.json"
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                for inst in result_data.get("instances", []):
                    if inst.get("id") == instance_id:
                        color_hex = inst.get("color", color_hex)
                        break
            
            if seg_path.exists():
                with open(seg_path) as f:
                    seg_data = json.load(f)
                res_info = seg_data.get("resolution", {})
                original_res = res_info.get("original")  # [H, W]
                
                # Build mapping: instance_id → [obj_ids] from raw segmentation.json
                # Each entry has "id" (obj_id in NPZ) and "instance_id" (logical group)
                iid_to_obj_ids = {}
                for inst in seg_data.get("instances", []):
                    iid = inst.get("instance_id", inst.get("id"))
                    oid = inst.get("id")
                    if iid not in iid_to_obj_ids:
                        iid_to_obj_ids[iid] = []
                    iid_to_obj_ids[iid].append(oid)
            else:
                iid_to_obj_ids = {}

            # Parse color hex to RGB
            color_hex = color_hex.lstrip('#')
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16)
            b = int(color_hex[4:6], 16)

            # Determine frame index for NPZ lookup
            if kf_index >= 0:
                frame_idx = kf_index
            elif frame:
                import re
                nums = re.findall(r'\d+', frame.split('.')[0])
                frame_idx = int(nums[-1]) if nums else 0
            else:
                frame_idx = 0

            # Find all obj_ids for this instance_id
            obj_ids_to_check = iid_to_obj_ids.get(instance_id, [instance_id])
            
            # Try to find mask in NPZ (combine all obj_ids for this instance)
            mask = None
            for oid in obj_ids_to_check:
                key = f"f{frame_idx}_o{oid}"
                if key in npz:
                    m = npz[key]
                    if mask is None:
                        mask = m.copy()
                    else:
                        mask = np.maximum(mask, m)
                else:
                    # Try nearest frame
                    stored_frames = npz.get("frames", np.array([]))
                    if len(stored_frames) > 0:
                        diffs = np.abs(stored_frames.astype(int) - frame_idx)
                        closest_idx = stored_frames[np.argmin(diffs)]
                        alt_key = f"f{closest_idx}_o{oid}"
                        if alt_key in npz:
                            m = npz[alt_key]
                            if mask is None:
                                mask = m.copy()
                            else:
                                mask = np.maximum(mask, m)

            if mask is None:
                return None

            import cv2, base64

            # Resize mask to original resolution so it aligns with the displayed image
            if original_res and len(original_res) == 2:
                orig_h, orig_w = original_res  # format is [H, W] (confirmed by numpy shape)
                mask = cv2.resize(mask.astype(np.uint8),
                                  (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

            # Build RGBA image with instance color
            h, w = mask.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_bool = mask > 0
            rgba[mask_bool, 0] = b  # OpenCV uses BGR
            rgba[mask_bool, 1] = g
            rgba[mask_bool, 2] = r
            rgba[mask_bool, 3] = 140  # Semi-transparent

            _, png_buf = cv2.imencode('.png', rgba)
            return base64.b64encode(png_buf.tobytes()).decode('ascii')

        mask_b64 = await loop.run_in_executor(None, _render)

        if mask_b64 is None:
            return {"ok": True, "mask_png": None, "message": "No mask found for this frame"}

        return {"ok": True, "mask_png": mask_b64}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/rename")
async def rename_segmentation_instance(request: Request):
    """Rename an instance label and optionally toggle its exclusion flag.
    Updates both segmentation.json and segmentation_result.json."""
    body = await request.json()
    session_id = body.get("session_id")
    instance_id = body.get("instance_id")
    new_label = body.get("label")
    excluded = body.get("excluded")

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    seg_path = output_dir / "segmentation.json"
    result_path = output_dir / "segmentation_result.json"

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="No segmentation.json")

    try:
        # Update segmentation.json (raw instances)
        with open(seg_path) as f:
            data = json.load(f)

        updated = False
        for inst in data.get("instances", []):
            if inst.get("id") == instance_id or inst.get("instance_id") == instance_id:
                if new_label is not None:
                    inst["label"] = new_label
                if excluded is not None:
                    inst["excluded"] = excluded
                updated = True

        if updated:
            with open(seg_path, "w") as f:
                json.dump(data, f, indent=2)

        # Also update segmentation_result.json (grouped instances)
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            for inst in result_data.get("instances", []):
                if inst.get("id") == instance_id:
                    if new_label is not None:
                        inst["label"] = new_label
                    if excluded is not None:
                        inst["excluded"] = excluded
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/delete")
async def delete_segmentation_instance(request: Request):
    """Delete an instance from segmentation.json, segmentation_result.json, and seg_masks.npz."""
    body = await request.json()
    session_id = body.get("session_id")
    instance_id = body.get("instance_id")

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    seg_path = output_dir / "segmentation.json"
    result_path = output_dir / "segmentation_result.json"
    masks_path = output_dir / "seg_masks.npz"

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="No segmentation.json")

    try:
        loop = asyncio.get_event_loop()

        def _delete():
            # 1) Find all raw obj_ids for this instance_id in segmentation.json
            with open(seg_path) as f:
                seg_data = json.load(f)

            obj_ids_to_remove = set()
            remaining_instances = []
            for inst in seg_data.get("instances", []):
                if inst.get("instance_id") == instance_id or inst.get("id") == instance_id:
                    obj_ids_to_remove.add(inst.get("id"))
                else:
                    remaining_instances.append(inst)

            seg_data["instances"] = remaining_instances
            with open(seg_path, "w") as f:
                json.dump(seg_data, f, indent=2)

            # 2) Remove from segmentation_result.json
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                result_data["instances"] = [
                    inst for inst in result_data.get("instances", [])
                    if inst.get("id") != instance_id
                ]
                with open(result_path, "w") as f:
                    json.dump(result_data, f, indent=2)

            # 3) Remove masks from NPZ
            if masks_path.exists() and obj_ids_to_remove:
                npz = np.load(masks_path, allow_pickle=True)
                new_data = {}
                removed_keys = 0
                remaining_obj_ids = set()

                for key in npz.files:
                    # Check if this key belongs to a removed obj_id
                    skip = False
                    if key.startswith("f") and "_o" in key:
                        parts = key.split("_o")
                        try:
                            oid = int(parts[1])
                            if oid in obj_ids_to_remove:
                                skip = True
                                removed_keys += 1
                            else:
                                remaining_obj_ids.add(oid)
                        except (ValueError, IndexError):
                            pass

                    if not skip:
                        new_data[key] = npz[key]

                # Update obj_ids array
                if "obj_ids" in new_data:
                    old_ids = new_data["obj_ids"].tolist()
                    new_data["obj_ids"] = np.array(
                        [oid for oid in old_ids if oid not in obj_ids_to_remove],
                        dtype=np.int32
                    )

                import tempfile
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.npz', dir=str(masks_path.parent))
                os.close(tmp_fd)
                np.savez_compressed(tmp_path, **new_data)
                os.replace(tmp_path, str(masks_path))
                return {"removed_masks": removed_keys, "removed_obj_ids": list(obj_ids_to_remove)}

            return {"removed_masks": 0, "removed_obj_ids": list(obj_ids_to_remove)}

        result = await loop.run_in_executor(None, _delete)
        return {"ok": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/paint_mask")
async def paint_mask(request: Request):
    """Paint on an instance's mask for a specific frame.
    
    Body: {session_id, instance_id, kf_index, x, y, radius, action: 'add'|'remove'}
    x, y: normalized 0-1 coordinates
    radius: in pixels (in mask space)
    Returns: updated mask PNG
    """
    body = await request.json()
    session_id = body.get("session_id")
    instance_id = body.get("instance_id")
    kf_index = body.get("kf_index", 0)
    x = body.get("x", 0)  # normalized 0-1
    y = body.get("y", 0)  # normalized 0-1
    radius = body.get("radius", 15)
    action = body.get("action", "add")  # 'add' or 'remove'

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    masks_path = output_dir / "seg_masks.npz"
    seg_path = output_dir / "segmentation.json"

    if not masks_path.exists():
        raise HTTPException(status_code=404, detail="No mask data")

    loop = asyncio.get_event_loop()

    def _paint():
        import cv2, base64
        npz = dict(np.load(masks_path, allow_pickle=True))

        # Build instance_id → [obj_ids] mapping
        obj_ids_for_instance = [instance_id]
        if seg_path.exists():
            with open(seg_path) as f:
                seg_data = json.load(f)
            for inst in seg_data.get("instances", []):
                iid = inst.get("instance_id", inst.get("id"))
                if iid == instance_id and inst.get("id") != instance_id:
                    obj_ids_for_instance.append(inst.get("id"))

        # Find the NPZ key for this frame+instance
        target_key = None
        for oid in obj_ids_for_instance:
            key = f"f{kf_index}_o{oid}"
            if key in npz:
                target_key = key
                break

        if target_key is None:
            # No existing mask? Create one if action is 'add'
            if action != 'add':
                return None
            # Determine mask shape from any existing mask
            for k in npz:
                if k.startswith("f") and "_o" in k:
                    shape = npz[k].shape
                    break
            else:
                return None
            mask = np.zeros(shape, dtype=np.uint8)
            target_key = f"f{kf_index}_o{obj_ids_for_instance[0]}"
        else:
            mask = npz[target_key].astype(np.uint8).copy()

        h, w = mask.shape[:2]
        cx = int(x * w)
        cy = int(y * h)

        # Apply circular brush
        yy, xx = np.ogrid[:h, :w]
        circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

        if action == 'add':
            mask[circle] = 1
        else:
            mask[circle] = 0

        npz[target_key] = mask
        # Atomic write: save to temp file, then rename to prevent corruption
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.npz', dir=str(masks_path.parent))
        os.close(tmp_fd)
        np.savez_compressed(tmp_path, **npz)
        os.replace(tmp_path, str(masks_path))

        # Generate updated mask PNG
        # Get instance color
        color_hex = "#00FF00"
        result_path = output_dir / "segmentation_result.json"
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            for inst in result_data.get("instances", []):
                if inst.get("id") == instance_id:
                    color_hex = inst.get("color", color_hex)
                    break

        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_bool = mask > 0
        rgba[mask_bool, 0] = r
        rgba[mask_bool, 1] = g
        rgba[mask_bool, 2] = b
        rgba[mask_bool, 3] = 128

        # Resize to original resolution if needed
        if seg_path.exists():
            with open(seg_path) as f:
                sd = json.load(f)
            orig_res = sd.get("resolution", {}).get("original")
            if orig_res and (orig_res[0] != h or orig_res[1] != w):
                rgba = cv2.resize(rgba, (orig_res[1], orig_res[0]), interpolation=cv2.INTER_NEAREST)

        _, png_buf = cv2.imencode('.png', rgba)
        return base64.b64encode(png_buf.tobytes()).decode('ascii')

    result = await loop.run_in_executor(None, _paint)
    if result is None:
        return {"mask_png": None}
    return {"mask_png": result}


@app.post("/api/segmentation/text_prompt")
async def add_text_prompt_endpoint(request: Request):
    """Add a text-based prompt to a SAM3 interactive session. Returns mask preview."""
    body = await request.json()
    state_id = body.get("state_id")
    frame_idx = body.get("frame_idx")
    text = body.get("text", "")
    obj_id = body.get("obj_id", 1)

    if not state_id or frame_idx is None or not text:
        raise HTTPException(status_code=400, detail="Missing state_id, frame_idx, or text")

    from sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    loop = asyncio.get_event_loop()

    try:
        def _prompt():
            result = sam3.add_text_prompt(
                state_id=state_id, frame_idx=frame_idx, text=text, obj_id=obj_id,
            )
            mask_b64 = None
            if result.get("mask") is not None:
                import cv2, base64
                mask = result["mask"]
                h, w = mask.shape[:2]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                mask_bool = mask > 0
                rgba[mask_bool, 1] = 200
                rgba[mask_bool, 3] = 128
                _, png_buf = cv2.imencode('.png', rgba)
                mask_b64 = base64.b64encode(png_buf.tobytes()).decode('ascii')
            return result["success"], mask_b64

        success, mask_b64 = await loop.run_in_executor(None, _prompt)
        if not success:
            raise HTTPException(status_code=500, detail="Text prompt failed")
        resp = {"ok": True}
        if mask_b64:
            resp["mask_png"] = mask_b64
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/auto/{session_id}")
async def run_auto_segmentation(session_id: str):
    """Full auto-segmentation: VLM scene analysis → SAM3 pipeline."""
    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    session_dir = scans_dir / session_id
    frames_dir = session_dir / "frames"
    output_dir = session_dir / "output"

    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames directory not found")
    output_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()

    try:
        def _auto():
            from scene_analyzer import analyze_scene
            vlm_result = analyze_scene(str(frames_dir), str(output_dir))
            categories = vlm_result.get("categories", [])
            frame_map = vlm_result.get("frame_map", {})
            if not categories:
                return {"error": "VLM found no categories", "instances": []}
            labels = [c["label"] if isinstance(c, dict) else c for c in categories]
            prompt = ";".join(labels)
            from segmentation_pipeline import run_segmentation
            return run_segmentation(str(frames_dir), str(output_dir), prompt, frame_map=frame_map)

        result = await loop.run_in_executor(None, _auto)
        return {"ok": True, "instances": result.get("instances", []), "error": result.get("error")}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/clean_instance")
async def clean_segmentation_instance(request: Request):
    """Run DBSCAN on a single segmented instance to isolate the largest cluster."""
    body = await request.json()
    session_id = body.get("session_id")
    instance_id = body.get("instance_id")

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    scans_dir = Path(__file__).parent / cfg.get("paths", {}).get("scans_dir", "scans")
    output_dir = scans_dir / session_id / "output"
    seg_path = output_dir / "segmentation.json"
    cloud_path = output_dir / "cleaned_cloud.ply"

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="No segmentation data")
    if not cloud_path.exists():
        raise HTTPException(status_code=404, detail="No cleaned cloud")

    loop = asyncio.get_event_loop()

    try:
        def _clean():
            with open(seg_path) as f:
                seg_data = json.load(f)
            inst = next((i for i in seg_data.get("instances", []) if i.get("id") == instance_id), None)
            if not inst:
                return {"error": f"Instance {instance_id} not found"}
            indices = inst.get("point_indices", [])
            if not indices:
                return {"error": "Instance has no point_indices"}
            from workers.instance_cleaner_worker import _run_dbscan, _load_ply
            cloud_pts, cloud_colors = _load_ply(cloud_path)
            inst_pts = cloud_pts[indices]
            inst_cols = cloud_colors[indices] if cloud_colors is not None else np.zeros_like(inst_pts)
            inst_cloud = np.hstack((inst_pts, inst_cols))
            inst_cfg = cfg.get("instance_cleaning", {})
            cleaned = _run_dbscan(inst_cloud, eps=inst_cfg.get("dbscan_eps", 0.05),
                                  min_samples=inst_cfg.get("dbscan_min_samples", 10))
            return {"ok": True, "original_points": len(indices), "cleaned_points": len(cleaned),
                    "removed": len(indices) - len(cleaned)}

        result = await loop.run_in_executor(None, _clean)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
                                 prompt, frame_map = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

                                 # Run new segmentation pipeline
                                 from segmentation_pipeline import run_segmentation
                                 result = run_segmentation(
                                     frames_dir=str(session.frames_dir),
                                     output_dir=str(session.output_dir),
                                     prompt=prompt,
                                     frame_map=frame_map
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
                                 
                                 # Apply segmentation against current cloud (offloaded)
                                 from segmentation_pipeline import apply_segmentation_to_cloud
                                 seg_data = await loop.run_in_executor(None, apply_segmentation_to_cloud, frame_storage.current_session.output_dir)
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
                                _loop = asyncio.get_event_loop()

                                # 1. Prepare Paths
                                images_dir = session.frames_dir.resolve()
                                output_dir = session.output_dir.resolve()

                                # 1.5-2. Run sync init steps in thread pool to not block event loop
                                def _init_da3():
                                    from frame_quality import analyze_frames, save_manifest
                                    fq_result = analyze_frames(str(images_dir))
                                    if "error" not in fq_result:
                                        save_manifest(str(images_dir), fq_result)

                                    frame_sel_cfg = cfg.get("frame_selection", {})
                                    if frame_sel_cfg.get("enabled", False):
                                        try:
                                            from frame_selector import select_keyframes
                                            sel_result = select_keyframes(str(images_dir), frame_sel_cfg)
                                            print(f"[Retro-Offline] 🎯 Selected {sel_result['selected_count']}/{sel_result['total_frames']} keyframes")
                                        except Exception as e:
                                            print(f"[Retro-Offline] ⚠️ Frame selection failed, using stride fallback: {e}")

                                    from da3_config_builder import build_da3_config
                                    da3_config = build_da3_config(cfg)

                                    print(f"[Retro-Offline] Initializing RealtimeDA3...")
                                    alignment_manager.reset()
                                    da3 = RealtimeDA3(
                                        image_dir=str(images_dir),
                                        save_dir=str(output_dir),
                                        config=da3_config,
                                        alignment_manager=alignment_manage
                                    )
                                    return da3

                                da3 = await _loop.run_in_executor(None, _init_da3)

                                # 3. No-op callback (no streaming during processing)
                                async def on_chunk_complete(chunk_id, sim3_transform):
                                    print(f"[Retro-Offline] Chunk {chunk_id} saved (not streaming)")
                                    await asyncio.sleep(0)  # Yield for pings

                                # 4. Run DA3 (already uses asyncio.to_thread internally)
                                print(f"[Retro-Offline] Processing {len(da3.img_list)} images...")
                                await da3.process_long_sequence_async(callback=on_chunk_complete)
                                print(f"[Retro-Offline] ✅ DA3 complete!")
                                return True

                            except Exception as e:
                                print(f"[Retro-Offline] Error: {e}")
                                import traceback
                                traceback.print_exc()
                                return False
                        
                        # Run ENTIRE pipeline as background task so WS loop stays responsive
                        async def _run_offline_pipeline():
                            try:
                                success = await _retro_process_offline()
                                
                                if success:
                                    session_id = frame_storage.current_session.session_id if frame_storage.current_session else None
                                    if session_id:
                                        postproc_config = cfg.get("postprocessing", {})
                                        if postproc_config.get("enabled", False):
                                            await _run_cloudcompy_postprocess(session_id, postproc_config, websocket)
                                        
                                        if prompt:
                                            session = frame_storage.current_session
                                            try:
                                                _p, frame_map = _resolve_segmentation_prompt(prompt, str(session.frames_dir))
                                                from segmentation_pipeline import run_segmentation
                                                _seg_loop = asyncio.get_event_loop()
                                                result = await _seg_loop.run_in_executor(None, lambda: run_segmentation(
                                                    frames_dir=str(session.frames_dir),
                                                    output_dir=str(session.output_dir),
                                                    prompt=_p,
                                                    frame_map=frame_map
                                                ))
                                                print(f"[Retro-Offline] Segmentation: {len(result.get('instances', []))} instances")
                                            except Exception as e:
                                                print(f"[Retro-Offline] Segmentation error: {e}")
                                        
                                        await _send_cleaned_cloud(websocket, session_id)
                                        
                                        output_dir = Path(__file__).parent / "scans" / session_id / "output"
                                        from segmentation_pipeline import apply_segmentation_to_cloud
                                        _seg_loop2 = asyncio.get_event_loop()
                                        seg_data = await _seg_loop2.run_in_executor(None, apply_segmentation_to_cloud, output_dir)
                                        if seg_data.get("instances"):
                                            await websocket.send_text(json.dumps(seg_data))
                                    
                                    await websocket.send_text(json.dumps({
                                        "type": "status",
                                        "message": "Offline processing complete"
                                    }))
                                    print("[Viewer] Offline Processing Complete.")
                            except Exception as e:
                                print(f"[Retro-Offline] Pipeline error: {e}")
                                import traceback
                                traceback.print_exc()

                        asyncio.create_task(_run_offline_pipeline())

            elif cmd.get("type") == "load_session":
                session_id = cmd.get("session_id")
                print(f"[Viewer] Loading session {session_id}...")
                loop = asyncio.get_event_loop()
                
                # Clear current view (lightweight, no I/O)
                alignment_manager.reset()
                frame_storage.stop_session()
                
                # Offload disk I/O to thread pool
                if frame_storage:
                    await loop.run_in_executor(None, frame_storage.load_session_from_disk, session_id)
                
                await websocket.send_text(json.dumps({"type": "cleared"}))
                await asyncio.sleep(0)  # Yield to process pings
                
                total_chunks = 0
                if frame_storage and frame_storage.current_session:
                    total_chunks = len(frame_storage.current_session.chunks)

                await websocket.send_text(json.dumps({
                    "type": "session_info",
                    "session_id": session_id,
                    "mode": "Offline",
                    "total_chunks": total_chunks
                }))

                # Auto-send pipeline status if running
                status = pipeline_manager.get_status(session_id)
                if status and status.get("status") in ("queued", "running"):
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "pipeline_progress",
                            "session_id": session_id,
                            **status
                        }))
                    except Exception:
                        pass
                
                # Stream saved point clouds
                try:
                    scans_dir = Path(__file__).parent / "scans"
                    output_dir = scans_dir / session_id / "output"

                    # ── Potree LOD: always convert before serving ──
                    cleaned_ply = output_dir / "cleaned_cloud.ply"
                    potree_metadata = output_dir / "potree" / "metadata.json"
                    
                    if not cleaned_ply.exists():
                        chunk_plys = sorted(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                        if chunk_plys:
                            print(f"[Viewer] No cleaned_cloud.ply found. Running CloudCompPy on {len(chunk_plys)} chunks...")
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": f"Building cleaned cloud from {len(chunk_plys)} chunks..."
                            }))
                            postproc_config = cfg.get("postprocessing", {})
                            await _run_cloudcompy_postprocess(session_id, postproc_config, websocket)
                        else:
                            await websocket.send_text(json.dumps({"type": "error", "message": "No point clouds found for this session"}))
                            raise Exception("No PLY data")
                    
                    await asyncio.sleep(0)  # Yield to process pings
                    
                    if cleaned_ply.exists():
                        # Ensure Potree octree exists (convert if needed)
                        if not potree_metadata.exists():
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "Building LOD octree (first load)..."
                            }))
                            session_path = SCANS_DIR / session_id
                            success = await convert_ply_to_potree_async(
                                session_path,
                                on_progress=lambda msg: websocket.send_text(json.dumps({"type": "status", "message": msg}))
                            )
                            if not success:
                                await websocket.send_text(json.dumps({"type": "error", "message": "Potree conversion failed"}))
                                raise Exception("Potree conversion failed")
                        
                        await asyncio.sleep(0)  # Yield to process pings
                        
                        # Offload file reads to thread pool
                        def _load_metadata():
                            potree_meta = json.loads(potree_metadata.read_text())
                            floor_transform_4x4 = None
                            transform_path = output_dir / "floor_transform.npz"
                            if transform_path.exists():
                                try:
                                    data = np.load(transform_path)
                                    s_val = float(data['s'])
                                    R = data['R']
                                    t = data['t']
                                    M = np.eye(4)
                                    M[:3, :3] = s_val * R
                                    M[:3, 3] = t
                                    floor_transform_4x4 = M.T.flatten().tolist()
                                    print(f"[Viewer] Floor transform loaded for Potree cloud")
                                except Exception as e:
                                    print(f"[Viewer] ⚠️ Could not load floor_transform: {e}")
                            else:
                                # Fallback: compute from cleaned_cloud.ply if available
                                cleaned_path = output_dir / "cleaned_cloud.ply"
                                if cleaned_path.exists():
                                    try:
                                        from alignment_manager import get_alignment_manager
                                        from plyfile import PlyData
                                        plydata = PlyData.read(str(cleaned_path))
                                        vx = plydata['vertex']
                                        xyz = np.column_stack([
                                            np.array(vx['x'], dtype=np.float64),
                                            np.array(vx['y'], dtype=np.float64),
                                            np.array(vx['z'], dtype=np.float64),
                                        ])
                                        am = get_alignment_manager()
                                        s_val, R, t = am.compute_leveling_from_points(xyz)
                                        if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                                            np.savez(transform_path, s=np.array(s_val), R=R, t=t)
                                            M = np.eye(4)
                                            M[:3, :3] = s_val * R
                                            M[:3, 3] = t
                                            floor_transform_4x4 = M.T.flatten().tolist()
                                            print(f"[Viewer] Floor transform computed and saved")
                                        else:
                                            print(f"[Viewer] No floor plane detected")
                                    except Exception as e:
                                        print(f"[Viewer] ⚠️ Floor alignment fallback failed: {e}")
                            return potree_meta, floor_transform_4x4
                        
                        potree_meta, floor_transform_4x4 = await loop.run_in_executor(None, _load_metadata)
                        
                        msg = {
                            "type": "potree_ready",
                            "session_id": session_id,
                            "url": f"/potree/{session_id}/",
                            "points": potree_meta.get("points", 0),
                        }
                        if floor_transform_4x4:
                            msg["floorTransform"] = floor_transform_4x4
                        await websocket.send_text(json.dumps(msg))
                        print(f"[Viewer] ✅ Sent potree_ready for {session_id} ({potree_meta.get('points', 0):,} pts)")
                        
                        await asyncio.sleep(0)  # Yield to process pings
                        
                        # Offload segmentation loading to thread pool
                        from segmentation_pipeline import apply_segmentation_to_cloud
                        seg_data = await loop.run_in_executor(None, apply_segmentation_to_cloud, output_dir)
                        if seg_data.get("instances"):
                            await websocket.send_text(json.dumps(seg_data))
                            print(f"[Viewer] Sent segmentation ({len(seg_data['instances'])} instances)")
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "Failed to build cleaned cloud"}))

                except Exception as e:
                    print(f"Error loading session: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd.get("type") in ("reconstruct_geometry", "run_pipeline"):
                # Pipeline-based reconstruction (subprocess workers)
                session_id = cmd.get("session_id")
                print(f"[Pipeline] 🔧 Starting pipeline for session {session_id}")

                # Build stages from client config or defaults
                stages_config = cmd.get("stages", None)  # e.g. {"da3": true, "vlm": false}
                ordered_stages = cmd.get("ordered_stages", None) # e.g. ["vlm", "sam3", "da3", "cloudcompy"]
                
                from pipeline_manager import build_pipeline_stages
                stages = build_pipeline_stages(ordered_stages=ordered_stages, enabled=stages_config)

                # Progress callback: relay to this websocket + broadcast
                async def _on_pipeline_progress(sid, job_dict):
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "pipeline_progress",
                            "session_id": sid,
                            **job_dict,
                        }))
                    except Exception:
                        pass

                # Completion callback: send cloud + segmentation data
                async def _on_pipeline_complete(sid, success):
                    if not success:
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": f"Pipeline failed for {sid}. Check server logs."
                            }))
                        except Exception:
                            pass
                        return

                    # Convert to Potree octree and notify viewer
                    try:
                        session_path = SCANS_DIR / sid
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "Building LOD octree..."
                        }))
                        success = await convert_ply_to_potree_async(session_path, force=True)
                        if success:
                            # Offload file reads to thread pool
                            _pipe_loop = asyncio.get_event_loop()
                            def _load_pipe_metadata():
                                potree_meta_path = session_path / "output" / "potree" / "metadata.json"
                                potree_meta = json.loads(potree_meta_path.read_text())
                                floor_transform_4x4 = None
                                ft_path = session_path / "output" / "floor_transform.npz"
                                if ft_path.exists():
                                    try:
                                        data = np.load(ft_path)
                                        s_val = float(data['s'])
                                        R = data['R']
                                        t = data['t']
                                        M = np.eye(4)
                                        M[:3, :3] = s_val * R
                                        M[:3, 3] = t
                                        floor_transform_4x4 = M.T.flatten().tolist()
                                    except Exception:
                                        pass
                                return potree_meta, floor_transform_4x4
                            
                            potree_meta, floor_transform_4x4 = await _pipe_loop.run_in_executor(None, _load_pipe_metadata)
                            
                            pipe_msg = {
                                "type": "potree_ready",
                                "session_id": sid,
                                "url": f"/potree/{sid}/",
                                "points": potree_meta.get("points", 0),
                            }
                            if floor_transform_4x4:
                                pipe_msg["floorTransform"] = floor_transform_4x4
                            await websocket.send_text(json.dumps(pipe_msg))
                            print(f"[Pipeline] ✅ Potree ready for {sid}")
                        else:
                            print(f"[Pipeline] ⚠️ Potree conversion failed, sending raw cloud")
                            await _send_cleaned_cloud(websocket, sid)
                    except Exception as e:
                        print(f"[Pipeline] Send cloud error: {e}")

                    # Send segmentation result (with floor-aligned OBBs) if available
                    # Use apply_segmentation_to_cloud (same as session reload) to ensure
                    # proper cache invalidation and OBB recalculation
                    try:
                        from segmentation_pipeline import apply_segmentation_to_cloud
                        _pipe_loop = asyncio.get_event_loop()
                        _seg_output_dir = Path(__file__).parent / "scans" / sid / "output"
                        seg_data = await _pipe_loop.run_in_executor(
                            None, apply_segmentation_to_cloud, _seg_output_dir
                        )
                        if seg_data and seg_data.get("instances"):
                            await websocket.send_text(json.dumps(seg_data))
                            print(f"[Pipeline] Sent {len(seg_data['instances'])} segments")
                    except Exception as e:
                        print(f"[Pipeline] Broadcast error: {e}")

                    try:
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": f"Pipeline complete for {sid}"
                        }))
                    except Exception:
                        pass

                # Start pipeline
                replace = cmd.get("replace", True)
                await pipeline_manager.start_pipeline(
                    session_id=session_id,
                    stages=stages,
                    config=dict(cfg),
                    on_progress=_on_pipeline_progress,
                    on_complete=_on_pipeline_complete,
                    replace=replace,
                )

                await websocket.send_text(json.dumps({
                    "type": "info",
                    "message": f"Pipeline started for {session_id}"
                }))

            elif cmd.get("type") == "cancel_pipeline":
                session_id = cmd.get("session_id")
                await pipeline_manager.cancel_pipeline(session_id)
                await websocket.send_text(json.dumps({
                    "type": "info",
                    "message": f"Pipeline cancelled for {session_id}"
                }))

            elif cmd.get("type") == "get_pipeline_status":
                session_id = cmd.get("session_id")
                status = pipeline_manager.get_status(session_id)
                await websocket.send_text(json.dumps({
                    "type": "pipeline_status",
                    "session_id": session_id,
                    "status": status,
                }))

    except Exception as e:
        print(f"[Viewer] ❌ WebSocket Error: {e}")
        import traceback
        traceback.print_exc()
        viewer_manager.disconnect_viewer(websocket)

# ─── Team WebSocket: presence, messaging, WebRTC signaling ────

# Track connected team users: {user_id: {"ws": WebSocket, "username": str, "task": str}}
_team_connections: dict[int, dict] = {}


async def _broadcast_presence():
    """Broadcast online users list to all connected team WS clients."""
    online = [
        {"user_id": uid, "username": info["username"], "task": info.get("task", "")}
        for uid, info in _team_connections.items()
    ]
    msg = json.dumps({"type": "presence_update", "online": online})
    dead = []
    for uid, info in _team_connections.items():
        try:
            await info["ws"].send_text(msg)
        except Exception:
            dead.append(uid)
    for uid in dead:
        _team_connections.pop(uid, None)


@app.websocket("/ws/team")
async def ws_team(websocket: WebSocket):
    """
    Team WebSocket — handles:
    - presence (online/offline broadcast)
    - team_message (real-time chat relay)
    - WebRTC signaling (call_invite, rtc_offer, rtc_answer, rtc_ice, call_end)
    """
    await websocket.accept()
    user_id: int | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # ── Auth / connect ─────────────────────────────
            if msg_type == "team_auth":
                from auth import decode_token
                try:
                    payload = decode_token(msg.get("token", ""))
                    user_id = int(payload["sub"])
                    username = payload.get("username", "")
                    _team_connections[user_id] = {
                        "ws": websocket,
                        "username": username,
                        "task": "",
                    }
                    await websocket.send_text(json.dumps({
                        "type": "team_auth_ok", "user_id": user_id,
                    }))
                    await _broadcast_presence()
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "team_auth_fail", "error": str(e),
                    }))
                continue

            if not user_id:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": "Not authenticated. Send team_auth first.",
                }))
                continue

            # ── Presence: update task ──────────────────────
            if msg_type == "update_task":
                if user_id in _team_connections:
                    _team_connections[user_id]["task"] = msg.get("task", "")
                    await _broadcast_presence()

            # ── Team message (chat) ────────────────────────
            elif msg_type == "team_message":
                team_id = msg.get("team_id")
                content = msg.get("content", "")
                if team_id and content and user_id in _team_connections:
                    username = _team_connections[user_id]["username"]
                    # Persist
                    from db_team import TeamMessage
                    from db import async_session_factory as _asf
                    async with _asf() as session:
                        tm = TeamMessage(
                            team_id=team_id,
                            user_id=user_id,
                            username=username,
                            content=content,
                        )
                        session.add(tm)
                        await session.commit()
                        await session.refresh(tm)

                    # Broadcast to all connected users
                    relay = json.dumps({
                        "type": "team_message",
                        "message": tm.to_dict(),
                    })
                    for uid, info in list(_team_connections.items()):
                        try:
                            await info["ws"].send_text(relay)
                        except Exception:
                            pass

            # ── WebRTC signaling ───────────────────────────
            elif msg_type in ("call_invite", "call_accept", "call_decline",
                              "call_end", "rtc_offer", "rtc_answer", "rtc_ice"):
                target_id = msg.get("to")
                if target_id and target_id in _team_connections:
                    # Forward to target, adding 'from' field
                    relay = {**msg, "from": user_id}
                    try:
                        await _team_connections[target_id]["ws"].send_text(
                            json.dumps(relay)
                        )
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Team WS] Error: {e}")
    finally:
        if user_id and user_id in _team_connections:
            _team_connections.pop(user_id, None)
            await _broadcast_presence()


if __name__ == "__main__":
    import uvicorn

    # Suppress /health spam from access logs
    class HealthCheckFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "GET /health" in msg:
                return False
            return True

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    uvicorn.run(
        "main:app", host=HOST, port=PORT, log_level="info",
        ws_ping_interval=30,    # Send ping every 30s
        ws_ping_timeout=300,    # Allow 5min without pong (for long reconstructions)
    )