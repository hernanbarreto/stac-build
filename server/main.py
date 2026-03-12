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
        msg = record.getMessage()
        return "GET /health" not in msg and "GET /api/tasks/" not in msg

logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

from frame_storage import get_frame_storage, FrameStorage
from alignment_manager import get_alignment_manager, AlignmentManager
from sam3_wrapper import get_sam3_wrapper
from config import cfg, DATA_DIR
from pipeline_manager import PipelineManager, PipelineStage, StageId
from project_paths import resolve_session

# --- Centralized path resolution ---
SERVER_DIR = str(Path(__file__).parent)

def _ctx(session_id: str):
    """Resolve a session_id to a path context (new-style or legacy)."""
    return resolve_session(SERVER_DIR, session_id)

def _audit_log(action: str, session_id: str, user: str = "system", role: str = "", detail: str = ""):
    """Append an entry to the project audit log (survives project deletion)."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "action": action,
        "session_id": session_id,
        "user": user,
        "role": role,
        "detail": detail,
    }
    with open(log_dir / "project_audit.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

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
    
    # RGB (Assumed float 0-1 or 0-255)
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
    """Run CloudCompPy post-processing on reconstruction chunk PLYs as subprocess."""
    import subprocess
    
    ctx = _ctx(session_id)
    scans_dir = ctx.output_dir
    output_ply = ctx.merged_cloud
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
    ctx = _ctx(session_id)
    
    cleaned_ply = ctx.merged_cloud
    raw_chunk = ctx.output_dir / "chunk_000.ply"
    
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
        session_dir = ctx.session_dir
        
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


async def _send_sabana_cloud(websocket, session_id: str):
    """Stream sábana PLY via same binary protocol as cleaned_cloud.
    The sábana PLY is already in registered BIM space with deviation colors.
    No floor alignment needed.
    """
    ctx = _ctx(session_id)
    ply_path = ctx.bim_comparison_dir / "sabana_cloud.ply"
    if not ply_path.exists():
        ply_path = ctx.session_dir / "sabana_cloud.ply"
    
    if not ply_path.exists():
        print(f"[SendSabana] ⚠️ sabana_cloud.ply not found for {session_id}")
        return False
    
    try:
        file_size_mb = ply_path.stat().st_size / (1024 * 1024)
        print(f"[SendSabana] Loading sábana ({file_size_mb:.1f} MB)...")
        
        def _load():
            """Read the sábana PLY (binary, XYZRGB)."""
            with open(ply_path, "rb") as f:
                n_pts = 0
                while True:
                    line = f.readline()
                    if line.startswith(b"element vertex"):
                        n_pts = int(line.split()[-1])
                    if line.startswith(b"end_header"):
                        break
                
                ply_dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                ])
                raw_data = np.frombuffer(f.read(), dtype=ply_dtype)
                point_count = len(raw_data)
                # Convert to N×7 float32 (same format as cleaned cloud)
                output_data = np.zeros((point_count, 7), dtype=np.float32)
                output_data[:, 0] = raw_data['x']
                output_data[:, 1] = raw_data['y']
                output_data[:, 2] = raw_data['z']
                output_data[:, 3] = raw_data['r'] / 255.0
                output_data[:, 4] = raw_data['g'] / 255.0
                output_data[:, 5] = raw_data['b'] / 255.0
            
            if point_count == 0:
                return None, 0
            
            return output_data.tobytes(), point_count
        
        loop = asyncio.get_event_loop()
        binary_bytes, point_count = await loop.run_in_executor(None, _load)
        
        if binary_bytes is None:
            print("[SendSabana] ⚠️ Empty sábana cloud")
            return False
        
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": f"Sending sábana ({point_count:,} points, {file_size_mb:.1f} MB)..."
        }))
        # Use sabana_start so frontend knows to apply transparency
        await websocket.send_text(json.dumps({
            "type": "sabana_start",
            "chunk_id": 0,
            "point_count": point_count
        }))
        
        for sub in chunk_data(binary_bytes):
            await websocket.send_bytes(sub)
            await asyncio.sleep(0.001)
        
        # Signal completion
        await websocket.send_text(json.dumps({
            "type": "sabana_loaded",
            "point_count": point_count
        }))
        
        print(f"[SendSabana] ✅ Sent {point_count:,} sábana points")
        return True
    
    except Exception as e:
        print(f"[SendSabana] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def _send_cleaned_cloud_broadcast(session_id: str):
    """Load and broadcast cleaned_cloud.ply to ALL connected viewers."""
    ctx = _ctx(session_id)
    cleaned_ply = ctx.merged_cloud
    
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
        output_data = _align_cloud_to_floor(output_data, session_dir=ctx.session_dir)
        
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
CHUNK_SIZE = cfg["mapanything"]["chunk_size"]
CHUNK_OVERLAP = cfg["mapanything"]["chunk_overlap"]


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
alignment_manager = None
pipeline_manager = PipelineManager()  # Pipeline orchestrator (subprocess workers)



def _resolve_segmentation_prompt(prompt: str, frames_dir: str) -> tuple:
    """
    Resolve segmentation prompt: if 'auto' or empty, use InternVL3 scene analyzer.
    
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


# --- Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_storage, alignment_manager
    print("[Server] Starting STAC-BUILD...")
    
    frame_storage = get_frame_storage()
    alignment_manager = get_alignment_manager()
    
    # Initialize SAM3 Wrapper (lazy load, but triggers init log)
    get_sam3_wrapper()
    
    print("[Server] Reconstruction backend: MapAnything (VGGT-Long)")
    print("[Server] Models will be loaded on-demand (lazy loading enabled)")

    # Initialize auth database
    from db import init_db
    await init_db()
    
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Potree octree file serving ──────────────────────────────────────────
# Serves pre-built LOD octree files (metadata.json, octree.bin, hierarchy.bin)
from potree_converter import convert_ply_to_potree, convert_ply_to_potree_async

# SCANS_DIR removed — use _ctx(session_id) for all path resolution

@app.get("/potree/{session_id}/{file_path:path}")
async def serve_potree_files(session_id: str, file_path: str):
    """Serve Potree octree files for a session."""
    from fastapi.responses import FileResponse
    ctx = _ctx(session_id)
    full_path = ctx.merged_potree / file_path
    if not full_path.exists():
        # Fallback: check output/potree/ (for data not yet in merged/)
        full_path = ctx.output_dir / "potree" / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    # Set proper content type for binary files
    content_type = "application/json" if file_path.endswith(".json") else "application/octet-stream"
    return FileResponse(
        str(full_path),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},  # Cache 1h
    )

@app.get("/potree_sabana/{session_id}/{file_path:path}")
async def serve_sabana_potree_files(session_id: str, file_path: str):
    """Serve sábana Potree octree files for a session."""
    from fastapi.responses import FileResponse
    ctx = _ctx(session_id)
    full_path = ctx.bim_comparison_dir / "sabana_potree" / file_path
    if not full_path.exists():
        # Fallback: check session_dir (legacy location)
        full_path = ctx.session_dir / "sabana_potree" / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    content_type = "application/json" if file_path.endswith(".json") else "application/octet-stream"
    return FileResponse(
        str(full_path),
        media_type=content_type,
        headers={"Cache-Control": "no-cache"},  # No cache — may be regenerated
    )

@app.get("/api/sessions/{session_id}/bim/{filename}")
async def serve_bim_file(session_id: str, filename: str):
    """Serve BIM GLB/JSON files for a session."""
    from fastapi.responses import FileResponse
    ctx = _ctx(session_id)
    # Check ifcs_dir first (migrated IFCs), then project_dir, then bim_comparison_dir
    full_path = ctx.ifcs_dir / filename
    if not full_path.exists():
        full_path = ctx.project_dir / filename
    if not full_path.exists():
        full_path = ctx.bim_comparison_dir / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"BIM file not found: {filename}")
    if filename.endswith(".glb"):
        content_type = "model/gltf-binary"
    elif filename.endswith(".ifc"):
        content_type = "application/octet-stream"
    else:
        content_type = "application/json"
    return FileResponse(
        str(full_path),
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
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
        "is_processing": False
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
        scans_dir = Path(__file__).parent / "scans"
        projects_dir = DATA_DIR / "projects"
        
        # Collect sessions from both legacy scans/ and new projects/
        all_dirs = []
        if scans_dir.exists():
            all_dirs.extend(sorted(scans_dir.iterdir(), reverse=True))
        if projects_dir.exists():
            all_dirs.extend(sorted(projects_dir.iterdir(), reverse=True))
        if not all_dirs: return []
        
        sessions = []
        for d in all_dirs:
            if d.is_dir():
                # Filter by team assignment (non-admin)
                if allowed_session_ids is not None and d.name not in allowed_session_ids:
                    continue

                # Use centralized path resolution
                try:
                    ctx = _ctx(d.name)
                except Exception:
                    continue

                # Count frames
                frames_dir = ctx.frames_dir
                frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
                
                # Check output
                output_dir = ctx.output_dir
                has_cloud = ctx.merged_cloud.exists()
                has_segments = (ctx.segmentation_dir / "segmentation.json").exists() if hasattr(ctx, 'segmentation_dir') else (output_dir / "segmentation.json").exists()
                
                # Count PLY chunks
                ply_files = list(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                
                # Cloud size (MB)
                cloud_size_mb = 0
                if has_cloud:
                    cloud_size_mb = round(ctx.merged_cloud.stat().st_size / (1024*1024), 1)
                
                # BIM / IFC files
                ifc_files = list(ctx.ifcs_dir.glob("*.ifc")) if ctx.ifcs_dir.exists() else []
                has_bim = len(ifc_files) > 0
                has_sabana = (ctx.bim_comparison_dir / "sabana.npz").exists() if hasattr(ctx, 'bim_comparison_dir') else False
                
                sessions.append({
                    "id": d.name,
                    "date": d.name,
                    "frame_count": frame_count,
                    "chunk_count": len(ply_files),
                    "has_cloud": has_cloud,
                    "has_segments": has_segments,
                    "cloud_size_mb": cloud_size_mb,
                    "has_bim": has_bim,
                    "bim_count": len(ifc_files),
                    "has_sabana": has_sabana,
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
    """Create a new empty session folder. Admin and Project Manager only."""
    from auth import decode_token

    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Admin or Manager only")

    body = await request.json()
    session_name = body.get("name", "").strip()
    if not session_name:
        raise HTTPException(status_code=400, detail="Session name is required")

    # Sanitize: only allow alphanumeric, dashes, underscores
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_name):
        raise HTTPException(status_code=400, detail="Session name can only contain letters, numbers, dashes, and underscores")

    # Create as new-style project
    from project_paths import ProjectPaths
    import time as _time
    projects_dir = DATA_DIR / "projects"
    paths = ProjectPaths(str(projects_dir), session_name)
    
    if paths.project_dir.exists():
        raise HTTPException(status_code=409, detail=f"Session '{session_name}' already exists")
    
    # Also check legacy scans/ for conflict
    legacy_dir = Path(__file__).parent / "scans" / session_name
    if legacy_dir.exists():
        raise HTTPException(status_code=409, detail=f"Session '{session_name}' already exists (legacy)")

    paths.ensure_dirs()
    today = _time.strftime("%Y-%m-%d")
    paths.ensure_source_dirs(today, "default")
    paths.init_project_meta(session_name)

    # Audit log
    username = payload.get("username", payload.get("sub", "unknown"))
    _audit_log("created", session_name, user=username, role=payload.get("role", ""))

    return {"ok": True, "session_id": session_name}


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Delete a project permanently. Admin only. Logged to project_audit.jsonl."""
    from auth import decode_token
    import shutil

    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    # Locate the project/session folder
    projects_dir = DATA_DIR / "projects"
    project_dir = projects_dir / session_id
    legacy_dir = Path(__file__).parent / "scans" / session_id

    target = None
    if project_dir.exists():
        target = project_dir
    elif legacy_dir.exists():
        target = legacy_dir
    else:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Unload if currently active
    if frame_storage and frame_storage.current_session and frame_storage.current_session.session_id == session_id:
        frame_storage.current_session = None

    # Delete
    username = payload.get("username", payload.get("sub", "unknown"))
    size_mb = round(sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) / (1024 * 1024), 1)
    shutil.rmtree(target)

    # Audit log (independent of project, survives deletion)
    _audit_log("deleted", session_id, user=username, role="admin",
               detail=f"size={size_mb}MB")

    print(f"[Server] 🗑️ Project '{session_id}' deleted by {username} ({size_mb}MB)")
    return {"ok": True, "session_id": session_id, "size_mb": size_mb}


@app.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Rename a project. Admin and Manager only."""
    from auth import decode_token
    import re

    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Admin or Manager only")

    body = await request.json()
    new_name = body.get("name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name is required")
    if not re.match(r'^[a-zA-Z0-9_-]+$', new_name):
        raise HTTPException(status_code=400, detail="Name can only contain letters, numbers, dashes, and underscores")
    if new_name == session_id:
        return {"ok": True, "old_id": session_id, "new_id": new_name}

    # Locate current directory
    projects_dir = DATA_DIR / "projects"
    legacy_scans = Path(__file__).parent / "scans"
    old_dir = None
    is_legacy = False
    if (projects_dir / session_id).exists():
        old_dir = projects_dir / session_id
    elif (legacy_scans / session_id).exists():
        old_dir = legacy_scans / session_id
        is_legacy = True
    else:
        raise HTTPException(status_code=404, detail=f"Project '{session_id}' not found")

    # Check new name doesn't conflict
    new_dir = (legacy_scans if is_legacy else projects_dir) / new_name
    if new_dir.exists():
        raise HTTPException(status_code=409, detail=f"Project '{new_name}' already exists")
    if (projects_dir / new_name).exists() or (legacy_scans / new_name).exists():
        raise HTTPException(status_code=409, detail=f"Project '{new_name}' already exists")

    # 1. Rename directory
    old_dir.rename(new_dir)

    # 2. Update DB references
    try:
        from db_team import SessionAssignment, ActivityLog
        from db_project import Project
        from sqlalchemy import update
        async with async_session_factory() as db:
            # SessionAssignment
            await db.execute(
                update(SessionAssignment)
                .where(SessionAssignment.session_id == session_id)
                .values(session_id=new_name)
            )
            # ActivityLog
            await db.execute(
                update(ActivityLog)
                .where(ActivityLog.session_id == session_id)
                .values(session_id=new_name)
            )
            # Project table
            await db.execute(
                update(Project)
                .where(Project.slug == session_id)
                .values(slug=new_name, name=new_name)
            )
            await db.commit()
    except Exception as e:
        print(f"[Server] ⚠️ DB update during rename failed: {e}")
        # Directory already renamed, log but don't fail

    # 3. Update loaded session reference if applicable
    if frame_storage and frame_storage.current_session and frame_storage.current_session.session_id == session_id:
        frame_storage.current_session.session_id = new_name

    # Audit
    username = payload.get("username", payload.get("sub", "unknown"))
    _audit_log("renamed", new_name, user=username, role=payload.get("role", ""),
               detail=f"old={session_id}")
    print(f"[Server] ✏️ Project '{session_id}' renamed to '{new_name}' by {username}")
    return {"ok": True, "old_id": session_id, "new_id": new_name}

@app.get("/sessions/{session_id}/scans")
async def get_session_scans(session_id: str):
    """List all scan days and sources for a project.
    Returns scan_key identifiers used by the pipeline to target specific scans.
    """
    from project_paths import ProjectPaths

    projects_dir = DATA_DIR / "projects"
    project_json = projects_dir / session_id / "project.json"

    if not project_json.exists():
        # Legacy session: single implicit scan
        ctx = _ctx(session_id)
        frame_count = len(list(ctx.frames_dir.glob("*.jpg"))) if ctx.frames_dir.exists() else 0
        has_output = ctx.merged_cloud.exists()
        return {
            "session_id": session_id,
            "scans": [{
                "date": "legacy",
                "source": "default",
                "key": "legacy/default",
                "frame_count": frame_count,
                "has_output": has_output,
            }]
        }

    paths = ProjectPaths(str(projects_dir), session_id)
    scans = []
    for date in paths.list_scan_days():
        for source in paths.list_sources(date):
            ctx = paths.for_source(date, source)
            frame_count = len(list(ctx.frames_dir.glob("*.jpg"))) if ctx.frames_dir.exists() else 0
            has_output = ctx.output_dir.exists() and any(ctx.output_dir.glob("chunk_*.ply"))
            scans.append({
                "date": date,
                "source": source,
                "key": f"{date}/{source}",
                "frame_count": frame_count,
                "has_output": has_output,
            })

    return {"session_id": session_id, "scans": scans}

@app.get("/segments/{session_id}")
async def get_segments(session_id: str):
    """Get unified segmentation data for a session."""
    try:
        ctx = _ctx(session_id)
        output_dir = ctx.output_dir
        
        # Use display-time matching (masks → cloud)
        masks_file = output_dir / "seg_masks.npz"
        seg_file = output_dir / "segmentation.json"
        if masks_file.exists() and seg_file.exists():
            from segmentation_pipeline import apply_segmentation_to_cloud
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, apply_segmentation_to_cloud, output_dir)
        
        # Fallback to legacy formats
        if seg_file.exists():
            with open(seg_file, 'r') as f:
                return json.load(f)
        
        return {"object_types": [], "instances": []}
    except Exception as e:
        print(f"Error serving segments: {e}")
        return {"error": str(e)}



@app.get("/api/sessions/{session_id}/frames/{filename}")
async def serve_session_frame(session_id: str, filename: str):
    """Serve individual frame images from a session's frames directory."""
    from fastapi.responses import FileResponse

    ctx = _ctx(session_id)
    frame_path = ctx.frames_dir / filename

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {filename}")

    return FileResponse(str(frame_path), media_type="image/jpeg")

@app.get("/api/sessions/{session_id}/keyframes")
async def get_session_keyframes(session_id: str):
    """Return the list of keyframe filenames from selected_frames.json."""
    ctx = _ctx(session_id)
    frames_dir = ctx.frames_dir
    selected_json = frames_dir / "selected_frames.json"

    if not selected_json.exists():
        raise HTTPException(status_code=404, detail="selected_frames.json not found (reconstruction must run first)")

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
    
    ctx = _ctx(session_id)
    frames_dir = ctx.frames_dir
    
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
        ctx_seg = _ctx(session_id)
        seg_path = ctx_seg.segmentation_dir / "segmentation.json"
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

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
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

    # Touch segmentation_result.json so its mtime is newer than floor_transform.npz.
    # This prevents unnecessary DBSCAN recomputation on next session load.
    # (The gizmo alignment only changes display transform, not the underlying segmentation)
    seg_result_path = output_dir / "segmentation_result.json"
    if seg_result_path.exists():
        seg_result_path.touch()
        print(f"[Alignment] ✅ Touched segmentation_result.json to preserve cache")

    return {"ok": True, "scale": s}


# ── BIM / IFC file management ──────────────────────────────────

@app.post("/api/sessions/{session_id}/bim/upload")
async def upload_bim(
    session_id: str,
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Upload an IFC file to a session. Admin/manager only."""
    from auth import decode_token
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    role = payload.get("role", "viewer")
    if role not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if not file.filename or not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files are accepted")

    ctx = _ctx(session_id)
    if not ctx.ifcs_dir.exists():
        ctx.ifcs_dir.mkdir(parents=True, exist_ok=True)
    if not ctx.session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Save file
    dest = ctx.ifcs_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    print(f"[BIM] ✅ Uploaded {file.filename} ({len(content)/1024:.0f} KB) to {session_id}")

    # Return updated list of IFC files
    ifc_files = [f.name for f in sorted(ctx.ifcs_dir.glob("*.ifc"))]
    return {"ok": True, "filename": file.filename, "ifc_files": ifc_files}


@app.delete("/api/sessions/{session_id}/bim/{filename}")
async def delete_bim(
    session_id: str,
    filename: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Delete an IFC file from a session. Admin/manager only."""
    from auth import decode_token
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    role = payload.get("role", "viewer")
    if role not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if not filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files can be deleted")

    ctx = _ctx(session_id)
    filepath = ctx.ifcs_dir / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    filepath.unlink()
    print(f"[BIM] 🗑️ Deleted {filename} from {session_id}")

    # Return updated list of IFC files
    ifc_files = [f.name for f in sorted(ctx.ifcs_dir.glob("*.ifc"))]
    return {"ok": True, "ifc_files": ifc_files}
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
    from task_manager import task_manager
    
    body = await request.json()
    state_id = body.get("state_id")
    session_id = body.get("session_id")
    label_name = body.get("label_name", "manual_object")
    
    if not state_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing state_id or session_id")
    
    from sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    
    tid = task_manager.start(session_id, "propagation", f'Propagating "{label_name}"')
    
    def sse_generator():
        import base64, cv2
        all_masks = {}  # frame_idx → outputs
        
        try:
            for frame_idx, num_frames, outputs in sam3.propagate_interactive_stream(state_id):
                all_masks[frame_idx] = outputs
                pct = round((len(all_masks) / max(1, num_frames)) * 100)
                task_manager.update(tid, pct=pct, detail=f"Frame {len(all_masks)}/{num_frames}")
                
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
            task_manager.update(tid, pct=100, detail="Saving masks...")
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
            
            task_manager.update(tid, detail="Computing 3D matching...")
            yield f"event: saving\ndata: {{\"status\": \"Computing 3D matching...\"}}\n\n"
            
            result = _match_and_save_result(output_dir)
            
            task_manager.finish(tid)
            done_event = json.dumps({
                "ok": True,
                "instances": result.get("instances", [])
            })
            yield f"event: done\ndata: {done_event}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            task_manager.fail(tid, str(e))
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
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    
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
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
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

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
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

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    seg_path = output_dir / "segmentation.json"
    result_path = output_dir / "segmentation_result.json"
    masks_path = output_dir / "seg_masks.npz"

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="No segmentation.json")

    try:
        loop = asyncio.get_event_loop()

        def _delete():
            print(f"[SegDelete] Deleting instance_id={instance_id} from session {session_id}")

            # 1) Collect ALL obj_ids to remove from segmentation.json
            with open(seg_path) as f:
                seg_data = json.load(f)

            obj_ids_to_remove = set()
            instance_ids_to_remove = set()
            remaining_instances = []
            for inst in seg_data.get("instances", []):
                iid = inst.get("instance_id")
                oid = inst.get("id")
                if iid == instance_id or oid == instance_id:
                    obj_ids_to_remove.add(oid)
                    if iid is not None:
                        instance_ids_to_remove.add(iid)
                else:
                    remaining_instances.append(inst)

            print(f"[SegDelete]   segmentation.json: removing {len(seg_data.get('instances', [])) - len(remaining_instances)} entries, "
                  f"obj_ids={obj_ids_to_remove}, instance_ids={instance_ids_to_remove}")
            seg_data["instances"] = remaining_instances
            with open(seg_path, "w") as f:
                json.dump(seg_data, f, indent=2)

            # 2) Remove from segmentation_result.json (just delete the entry)
            fresh_instances = []
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                fresh_instances = [
                    inst for inst in result_data.get("instances", [])
                    if inst.get("id") not in instance_ids_to_remove
                       and inst.get("instance_id") not in instance_ids_to_remove
                ]
                result_data["instances"] = fresh_instances
                with open(result_path, "w") as f:
                    json.dump(result_data, f)
                print(f"[SegDelete]   segmentation_result.json: {len(fresh_instances)} instances remaining")

            # 3) Remove masks from NPZ
            if masks_path.exists() and obj_ids_to_remove:
                npz = np.load(masks_path, allow_pickle=True)
                new_data = {}
                removed_keys = 0

                for key in npz.files:
                    skip = False
                    if key.startswith("f") and "_o" in key:
                        parts = key.split("_o")
                        try:
                            oid = int(parts[1])
                            if oid in obj_ids_to_remove:
                                skip = True
                                removed_keys += 1
                        except (ValueError, IndexError):
                            pass

                    if not skip:
                        new_data[key] = npz[key]

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
                print(f"[SegDelete]   seg_masks.npz: removed {removed_keys} mask keys")

            print(f"[SegDelete]   ✅ Done: {len(fresh_instances)} instances remaining")

            return {
                "removed_obj_ids": list(obj_ids_to_remove),
                "instances": fresh_instances,
            }

        result = await loop.run_in_executor(None, _delete)
        return {"ok": True, **result}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/refresh")
async def refresh_segmentation(body: dict):
    """Delete segmentation_result.json and regenerate with full DBSCAN + matching."""
    from task_manager import task_manager
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(400, "session_id required")
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    result_path = output_dir / "segmentation_result.json"

    loop = asyncio.get_event_loop()
    tid = task_manager.start(session_id, "dbscan_refresh", "Refreshing segmentation (DBSCAN)")

    def _refresh():
        try:
            import time as _time
            # Freshness guard: if result was just computed (< 120s), skip rebuild
            if result_path.exists():
                age = _time.time() - result_path.stat().st_mtime
                if age < 120:
                    with open(result_path) as f:
                        result = json.load(f)
                    print(f"[SegRefresh] Result is fresh ({age:.0f}s old) — skipping rebuild")
                    task_manager.finish(tid)
                    return {"instances": result.get("instances", [])}
                result_path.unlink()
                print(f"[SegRefresh] Deleted stale segmentation_result.json for {session_id} ({age:.0f}s old)")
            from segmentation_pipeline import _match_and_save_result
            task_manager.update(tid, pct=10, detail="Running DBSCAN + cloud matching...")
            print(f"[SegRefresh] Regenerating segmentation_result.json...")
            result = _match_and_save_result(output_dir)
            instances = result.get("instances", [])
            print(f"[SegRefresh] ✅ Done: {len(instances)} instances")
            task_manager.finish(tid)
            return {"instances": instances}
        except Exception as e:
            task_manager.fail(tid, str(e))
            raise

    result = await loop.run_in_executor(None, _refresh)
    return {"ok": True, **result}


@app.get("/api/tasks/{session_id}")
async def get_active_tasks(session_id: str):
    """Return all active (running) tasks for a session."""
    from task_manager import task_manager
    tasks = task_manager.get_active(session_id)
    return {"tasks": tasks}


@app.get("/api/pipelines/active")
async def get_active_pipelines():
    """Return all active pipeline jobs (for UI state recovery after refresh)."""
    jobs = pipeline_manager.get_all_jobs()
    # Filter to only running/queued jobs
    active = {sid: job for sid, job in jobs.items()
              if job.get("status") in ("running", "queued")}
    return {"pipelines": active}


@app.get("/api/bim/auto_match/{session_id}")
async def bim_auto_match(session_id: str):
    """
    Automatically match segment labels to IFC element name suffixes.
    Returns all discovered matches for this session.
    """
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    
    # Load segmentation
    seg_result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"
    if seg_result_path.exists():
        seg_data = json.loads(seg_result_path.read_text())
    elif seg_path.exists():
        seg_data = json.loads(seg_path.read_text())
    else:
        return {"matches": [], "error": "No segmentation data"}
    
    seg_labels = {str(inst.get("label", "")) for inst in seg_data.get("instances", [])}
    
    # Find IFC file and build name index
    ifc_files = list(ctx.ifcs_dir.glob("*.ifc"))
    if not ifc_files:
        return {"matches": [], "error": "No IFC file found"}
    
    import ifcopenshell
    ifc_file = ifcopenshell.open(str(ifc_files[0]))
    
    # Build name suffix → element info map
    ifc_elements = {}
    for el in ifc_file.by_type('IfcProduct'):
        name = getattr(el, 'Name', None)
        if not name:
            continue
        name = str(name)
        parts = name.split(':')
        suffix = parts[-1].strip()
        if suffix.isdigit():
            ifc_elements[suffix] = {
                "element_key": suffix,
                "ifc_type": el.is_a(),
                "ifc_name": name,
                "ifc_id": el.id(),
            }
    
    # Find matches
    matches = []
    for label in seg_labels:
        if label in ifc_elements:
            matches.append({
                "segment_label": label,
                **ifc_elements[label],
            })
    
    return {
        "matches": matches,
        "total_segments": len(seg_labels),
        "total_ifc_elements": len(ifc_elements),
    }


@app.post("/api/bim/compare")
async def bim_compare(request: Request):
    """
    Run Cloud-to-Mesh deviation analysis.
    Body: { session_id, matches: [{ segment_label, element_key }], tolerance_mm?: number }
    """
    from task_manager import task_manager
    body = await request.json()
    session_id = body.get("session_id")
    matches = body.get("matches", [])
    # Tolerance from config.yaml bim.deviation.tolerance_mm (mm)
    bim_cfg = cfg.get("bim", {}).get("deviation", {})
    tolerance_mm = bim_cfg.get("tolerance_mm", 50.0)
    
    if not session_id or not matches:
        raise HTTPException(400, "session_id and matches required")
    
    ctx = _ctx(session_id)
    session_dir = str(ctx.session_dir)
    
    loop = asyncio.get_event_loop()
    tid = task_manager.start(session_id, "bim_compare", "BIM vs Scan Comparison")
    
    try:
        def _compare():
            try:
                from bim_comparison import run_comparison
                
                def on_progress(pct, detail=""):
                    task_manager.update(tid, pct=pct, detail=detail)
                
                result = run_comparison(
                    session_dir,
                    matches,
                    tolerance=tolerance_mm / 1000.0,
                    progress_callback=on_progress,
                )
                task_manager.finish(tid)
                return result
            except Exception as e:
                task_manager.fail(tid, str(e))
                raise
        
        result = await loop.run_in_executor(None, _compare)
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/sessions/{session_id}/sabana/exists")
async def sabana_exists(session_id: str):
    """Check if a sábana has been generated for this session."""
    ctx = _ctx(session_id)
    # Check both bim_comparison_dir (migrated) and session_dir (newly generated)
    npz_path = ctx.bim_comparison_dir / "sabana.npz"
    if not npz_path.exists():
        npz_path = ctx.session_dir / "sabana.npz"
    meta_path = ctx.bim_comparison_dir / "sabana_meta.json"
    if not meta_path.exists():
        meta_path = ctx.session_dir / "sabana_meta.json"
    
    if not npz_path.exists():
        return {"exists": False}
    
    # Return metadata if available
    meta = None
    if meta_path.exists():
        import json as _json
        meta = _json.loads(meta_path.read_text())
    
    return {"exists": True, "meta": meta}


@app.get("/api/sessions/{session_id}/sabana/meta")
async def sabana_meta(session_id: str):
    """Serve full sabana_meta.json for the BIM Analysis Panel."""
    ctx = _ctx(session_id)
    meta_path = ctx.bim_comparison_dir / "sabana_meta.json"
    if not meta_path.exists():
        meta_path = ctx.session_dir / "sabana_meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No sábana metadata found")
    import json as _json
    return _json.loads(meta_path.read_text())


@app.get("/api/sessions/{session_id}/sabana")
async def sabana_load(session_id: str):
    """Serve saved sábana as raw binary (positions + colors as Float32)."""
    from starlette.responses import Response
    ctx = _ctx(session_id)
    # Check bim_comparison dir first (migrated), then session_dir (newly generated)
    pos_path = ctx.bim_comparison_dir / "sabana_positions.bin"
    col_path = ctx.bim_comparison_dir / "sabana_colors.bin"
    if not pos_path.exists():
        pos_path = ctx.session_dir / "sabana_positions.bin"
        col_path = ctx.session_dir / "sabana_colors.bin"
    
    if not pos_path.exists() or not col_path.exists():
        raise HTTPException(404, "No sábana found for this session")
    
    pos_data = pos_path.read_bytes()
    col_data = col_path.read_bytes()
    n_points = len(pos_data) // (3 * 4)  # 3 floats × 4 bytes
    
    # Header: 4 bytes uint32 nPoints, then positions, then colors
    import struct
    header = struct.pack('<I', n_points)
    
    return Response(
        content=header + pos_data + col_data,
        media_type="application/octet-stream",
        headers={"X-Sabana-Points": str(n_points)},
    )

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

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
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
    from task_manager import task_manager
    ctx = _ctx(session_id)
    frames_dir = ctx.frames_dir
    output_dir = ctx.output_dir

    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames directory not found")
    output_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_event_loop()
    tid = task_manager.start(session_id, "auto_seg", "Auto-segmentation (VLM + SAM3)")

    try:
        def _auto():
            try:
                task_manager.update(tid, pct=10, detail="Analyzing scene with VLM...")
                from scene_analyzer import analyze_scene
                vlm_result = analyze_scene(str(frames_dir), str(output_dir))
                categories = vlm_result.get("categories", [])
                frame_map = vlm_result.get("frame_map", {})
                if not categories:
                    task_manager.finish(tid)
                    return {"error": "VLM found no categories", "instances": []}
                labels = [c["label"] if isinstance(c, dict) else c for c in categories]
                prompt = ";".join(labels)
                task_manager.update(tid, pct=40, detail=f"Segmenting {len(labels)} categories...")
                from segmentation_pipeline import run_segmentation
                result = run_segmentation(str(frames_dir), str(output_dir), prompt, frame_map=frame_map)
                task_manager.finish(tid)
                return result
            except Exception as e:
                task_manager.fail(tid, str(e))
                raise

        result = await loop.run_in_executor(None, _auto)
        return {"ok": True, "instances": result.get("instances", []), "error": result.get("error")}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation/clean_instance")
async def clean_segmentation_instance(request: Request):
    """Run DBSCAN on a single segmented instance to isolate the largest cluster."""
    from task_manager import task_manager
    body = await request.json()
    session_id = body.get("session_id")
    instance_id = body.get("instance_id")

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    seg_path = output_dir / "segmentation.json"
    cloud_path = ctx.merged_cloud

    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="No segmentation data")
    if not cloud_path.exists():
        raise HTTPException(status_code=404, detail="No cleaned cloud")

    loop = asyncio.get_event_loop()
    tid = task_manager.start(session_id, "instance_clean", f"Cleaning instance {instance_id}")

    try:
        def _clean():
            try:
                with open(seg_path) as f:
                    seg_data = json.load(f)
                inst = next((i for i in seg_data.get("instances", []) if i.get("id") == instance_id), None)
                if not inst:
                    task_manager.finish(tid)
                    return {"error": f"Instance {instance_id} not found"}
                indices = inst.get("point_indices", [])
                if not indices:
                    task_manager.finish(tid)
                    return {"error": "Instance has no point_indices"}
                task_manager.update(tid, pct=20, detail="Running DBSCAN...")
                from workers.instance_cleaner_worker import _run_dbscan, _load_ply
                cloud_pts, cloud_colors = _load_ply(cloud_path)
                inst_pts = cloud_pts[indices]
                inst_cols = cloud_colors[indices] if cloud_colors is not None else np.zeros_like(inst_pts)
                inst_cloud = np.hstack((inst_pts, inst_cols))
                inst_cfg = cfg.get("instance_cleaning", {})
                cleaned = _run_dbscan(inst_cloud, eps=inst_cfg.get("dbscan_eps", 0.05),
                                      min_samples=inst_cfg.get("dbscan_min_samples", 10))
                task_manager.finish(tid)
                return {"ok": True, "original_points": len(indices), "cleaned_points": len(cleaned),
                        "removed": len(indices) - len(cleaned)}
            except Exception as e:
                task_manager.fail(tid, str(e))
                raise

        result = await loop.run_in_executor(None, _clean)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """
    Camera WebSocket - receives frames from camera.html client.
    Stores frames via frame_storage for later offline pipeline processing.
    """
    await websocket.accept()
    await camera_manager.connect(websocket)
    await _camera_frame_capture(websocket)


async def _camera_frame_capture(websocket: WebSocket):
    """Capture camera frames and store them for offline pipeline processing."""
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
        print(f"[Camera] Connection closed: {e}")
    finally:
        camera_manager.disconnect()

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
                await viewer_manager.send_text(websocket, json.dumps({
                    "type": "status",
                    "camera_connected": camera_manager.active_camera is not None,
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

                # Retroactive Segmentation — runs segmentation on existing reconstruction
                if prompt and alignment_manager:
                    # Check if we have PLY files (either in memory OR on disk)
                    has_plys_in_memory = alignment_manager.get_chunk_count() > 0
                    has_plys_on_disk = False
                    if frame_storage and frame_storage.current_session:
                        output_dir = frame_storage.current_session.output_dir
                        ply_files = list(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []
                        has_plys_on_disk = len(ply_files) > 0

                    if has_plys_in_memory or has_plys_on_disk:
                         source = "Active Memory" if has_plys_in_memory else "Disk (Offline Session)"
                         chunk_count = alignment_manager.get_chunk_count() if has_plys_in_memory else len(ply_files)
                         print(f"[Viewer] 🔄 Retroactive Segmentation ({source}): {chunk_count} chunks.")
                         
                         # Run in background to not block socket
                         def _retro_process_active():
                             try:
                                 session = frame_storage.current_session
                                 if not session: return False

                                 # Resolve prompt (auto-detect if needed)
                                 resolved_prompt, frame_map = _resolve_segmentation_prompt(prompt, str(session.frames_dir))

                                 # Run segmentation pipeline
                                 from segmentation_pipeline import run_segmentation
                                 result = run_segmentation(
                                     frames_dir=str(session.frames_dir),
                                     output_dir=str(session.output_dir),
                                     prompt=resolved_prompt,
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
                        # No reconstruction available — inform user to run pipeline first
                        print(f"[Viewer] ⚠️ No reconstruction found. Run pipeline first.")
                        await websocket.send_text(json.dumps({
                            "type": "info", 
                            "message": "No reconstruction found. Please run the pipeline first."
                        }))

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
                    _load_ctx = _ctx(session_id)
                    output_dir = _load_ctx.output_dir

                    # ── Potree LOD: always convert before serving ──
                    cleaned_ply = output_dir / "cleaned_cloud.ply"
                    # Check both merged/potree (migrated) and output/potree (newly converted)
                    potree_metadata = _load_ctx.merged_potree / "metadata.json"
                    if not potree_metadata.exists():
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
                            session_path = _load_ctx.session_dir
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
                            # Detect IFC files in session directory
                            ifc_files = [f.name for f in sorted(_load_ctx.ifcs_dir.glob('*.ifc'))] if _load_ctx.ifcs_dir.exists() else []
                            return potree_meta, floor_transform_4x4, ifc_files
                        
                        potree_meta, floor_transform_4x4, ifc_files = await loop.run_in_executor(None, _load_metadata)
                        
                        msg = {
                            "type": "potree_ready",
                            "session_id": session_id,
                            "url": f"/potree/{session_id}/",
                            "points": potree_meta.get("points", 0),
                        }
                        if floor_transform_4x4:
                            msg["floorTransform"] = floor_transform_4x4
                        if ifc_files:
                            msg["bimFiles"] = ifc_files
                        await websocket.send_text(json.dumps(msg))
                        print(f"[Viewer] ✅ Sent potree_ready for {session_id} ({potree_meta.get('points', 0):,} pts)")
                        
                        await asyncio.sleep(0)  # Yield to process pings
                        
                        # Offload segmentation loading to thread pool
                        from segmentation_pipeline import apply_segmentation_to_cloud
                        seg_data = await loop.run_in_executor(None, apply_segmentation_to_cloud, output_dir)
                        if seg_data.get("instances"):
                            await websocket.send_text(json.dumps(seg_data))
                            print(f"[Viewer] Sent segmentation ({len(seg_data['instances'])} instances)")
                        
                        # Notify frontend about BIM files (raw IFC — parsed on frontend by web-ifc)
                        if ifc_files:
                            bim_models = [
                                {'name': name, 'url': f'/api/sessions/{session_id}/bim/{name}'}
                                for name in ifc_files
                            ]
                            await websocket.send_text(json.dumps({
                                'type': 'bim_ready',
                                'session_id': session_id,
                                'models': bim_models,
                            }))
                            print(f"[Viewer] ✅ Sent bim_ready ({len(bim_models)} IFC files)")
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "message": "Failed to build cleaned cloud"}))

                except Exception as e:
                    print(f"Error loading session: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif cmd.get("type") == "load_sabana":
                session_id = cmd.get("session_id")
                print(f"[Viewer] Loading sábana for {session_id}...")
                _sabana_ctx = _ctx(session_id)
                sabana_dir = _sabana_ctx.bim_comparison_dir
                # Fallback to session_dir if sabana files aren't in bim_comparison
                if not (sabana_dir / "sabana_cloud.ply").exists():
                    sabana_dir = _sabana_ctx.session_dir

                # Step 1: Convert sábana PLY → Potree octree (cached)
                from potree_converter import convert_sabana_to_potree_async
                success = await convert_sabana_to_potree_async(
                    sabana_dir,
                    on_progress=lambda msg: websocket.send_text(json.dumps({"type": "status", "message": msg})),
                )
                if not success:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Sábana Potree conversion failed"}))
                else:
                    # Read metadata
                    potree_meta_path = sabana_dir / "sabana_potree" / "metadata.json"
                    potree_meta = json.loads(potree_meta_path.read_text()) if potree_meta_path.exists() else {}
                    n_pts = potree_meta.get("points", 0)

                    # Send sabana_potree_ready (NOT cleared — keeps BIM + OBBs)
                    await websocket.send_text(json.dumps({
                        "type": "sabana_potree_ready",
                        "session_id": session_id,
                        "url": f"/potree_sabana/{session_id}/",
                        "points": n_pts,
                    }))
                    print(f"[Viewer] ✅ Sábana Potree ready ({n_pts:,} pts)")

            elif cmd.get("type") in ("reconstruct_geometry", "run_pipeline"):
                # Pipeline-based reconstruction (subprocess workers)
                session_id = cmd.get("session_id")
                print(f"[Pipeline] 🔧 Starting pipeline for session {session_id}")

                # Build stages from client config or defaults
                stages_config = cmd.get("stages", None)  # e.g. {"reconstruction": true, "vlm": false}
                ordered_stages = cmd.get("ordered_stages", None) # e.g. ["vlm", "sam3", "reconstruction", "cloudcompy"]
                
                from pipeline_manager import build_pipeline_stages
                stages = build_pipeline_stages(ordered_stages=ordered_stages, enabled=stages_config)

                # Progress callback: relay to this websocket + broadcast
                from task_manager import task_manager as _tm
                _pipeline_tid = _tm.start(session_id, "pipeline", "Running Pipeline")

                async def _on_pipeline_progress(sid, job_dict):
                    try:
                        # Update task manager with pipeline progress
                        stage = job_dict.get("current_stage", "")
                        pct = job_dict.get("pct", 0)
                        _tm.update(_pipeline_tid, pct=pct, detail=f"Stage: {stage}")
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
                        _tm.fail(_pipeline_tid, "Pipeline failed")
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
                        session_path = _ctx(sid).session_dir
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "Building LOD octree..."
                        }))
                        success = await convert_ply_to_potree_async(session_path, force=True)
                        if success:
                            # Offload file reads to thread pool
                            _pipe_loop = asyncio.get_event_loop()
                            def _load_pipe_metadata():
                                _pipe_ctx = _ctx(sid)
                                potree_meta_path = _pipe_ctx.merged_potree / "metadata.json"
                                if not potree_meta_path.exists():
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
                        _seg_ctx = _ctx(sid)
                        _seg_output_dir = _seg_ctx.output_dir
                        seg_data = await _pipe_loop.run_in_executor(
                            None, apply_segmentation_to_cloud, _seg_output_dir
                        )
                        if seg_data and seg_data.get("instances"):
                            await websocket.send_text(json.dumps(seg_data))
                            print(f"[Pipeline] Sent {len(seg_data['instances'])} segments")
                    except Exception as e:
                        print(f"[Pipeline] Broadcast error: {e}")

                    try:
                        _tm.finish(_pipeline_tid)
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": f"Pipeline complete for {sid}"
                        }))
                    except Exception:
                        pass

                # Start pipeline — support sequential multi-scan
                replace = cmd.get("replace", True)
                scan_keys = cmd.get("scans", [])  # e.g. ["2026-03-07/legacy", "2026-03-08/default"]

                if len(scan_keys) <= 1:
                    # Single scan or auto-resolve
                    single_key = scan_keys[0] if scan_keys else None
                    await pipeline_manager.start_pipeline(
                        session_id=session_id,
                        stages=stages,
                        config=dict(cfg),
                        on_progress=_on_pipeline_progress,
                        on_complete=_on_pipeline_complete,
                        replace=replace,
                        scan_key=single_key,
                    )
                    label = single_key or "auto"
                    await websocket.send_text(json.dumps({
                        "type": "info",
                        "message": f"Pipeline started for {session_id} (scan: {label})"
                    }))
                else:
                    # Sequential multi-scan: run each scan one at a time
                    async def _run_multi_scan():
                        total = len(scan_keys)
                        for i, sk in enumerate(scan_keys):
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "info",
                                    "message": f"Starting scan {i+1}/{total}: {sk}"
                                }))
                            except Exception:
                                pass

                            # Use a future to await sequential completion
                            done_event = asyncio.Event()
                            scan_success = [True]

                            async def _on_scan_complete(sid, success, _ev=done_event, _ss=scan_success):
                                _ss[0] = success
                                await _on_pipeline_complete(sid, success)
                                _ev.set()

                            await pipeline_manager.start_pipeline(
                                session_id=session_id,
                                stages=stages,
                                config=dict(cfg),
                                on_progress=_on_pipeline_progress,
                                on_complete=_on_scan_complete,
                                replace=replace,
                                scan_key=sk,
                            )
                            await done_event.wait()

                            if not scan_success[0]:
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "error",
                                        "message": f"Scan {sk} failed. Stopping multi-scan pipeline."
                                    }))
                                except Exception:
                                    pass
                                return

                        try:
                            await websocket.send_text(json.dumps({
                                "type": "info",
                                "message": f"All {total} scans completed for {session_id}"
                            }))
                        except Exception:
                            pass

                    asyncio.create_task(_run_multi_scan())
                    await websocket.send_text(json.dumps({
                        "type": "info",
                        "message": f"Multi-scan pipeline started: {len(scan_keys)} scans for {session_id}"
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
    finally:
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

# ═══════════════════════════════════════════════════════════════
#  WebXR SCAN CAPTURE — receives frames + pose + intrinsics
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/scan")
async def scan_websocket(websocket: WebSocket):
    """
    WebXR Scan WebSocket — receives frames + camera data from camera.html.
    Saves JPEG frames and camera metadata (pose, intrinsics) to disk.
    """
    await websocket.accept()
    print("[Scan WS] Client connected")

    project_id = None
    scan_date = None
    source_name = "webxr"
    frames_dir = None
    camera_data_dir = None
    frame_count = 0

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "init_scan":
                # ── Initialize scan session ──
                project_id = msg.get("project_id", "").strip()
                if not project_id:
                    await websocket.send_json({"type": "error", "message": "Missing project_id"})
                    continue

                # Verify project exists
                try:
                    ctx = _ctx(project_id)
                except Exception:
                    await websocket.send_json({"type": "error", "message": f"Project '{project_id}' not found"})
                    continue

                # Create scan directory for today
                scan_date = time.strftime("%Y-%m-%d")
                from project_paths import ProjectPaths
                projects_dir = DATA_DIR / "projects"
                paths = ProjectPaths(str(projects_dir), project_id)
                paths.ensure_source_dirs(scan_date, source_name)

                # Resolve frame and camera_data directories
                source_dir = paths.source_dir(scan_date, source_name)
                frames_dir = source_dir / "frames"
                camera_data_dir = source_dir / "camera_data"
                frames_dir.mkdir(parents=True, exist_ok=True)
                camera_data_dir.mkdir(parents=True, exist_ok=True)

                # Create scan_meta.json with capture info
                scan_meta = {
                    "capture_method": "webxr",
                    "has_ar": msg.get("has_ar", False),
                    "capture_fps": msg.get("capture_fps", 3),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "zoom_level": 1.0,
                    "source": "webxr_capture",
                }
                meta_path = source_dir / "scan_meta.json"
                with open(meta_path, "w") as f:
                    json.dump(scan_meta, f, indent=2)

                frame_count = 0
                print(f"[Scan WS] Initialized: project={project_id}, scan={scan_date}/{source_name}")
                print(f"[Scan WS]   frames → {frames_dir}")
                print(f"[Scan WS]   camera → {camera_data_dir}")

                await websocket.send_json({
                    "type": "scan_ready",
                    "scan_path": f"scans/{scan_date}/{source_name}",
                })

            elif msg_type == "frame":
                if not frames_dir:
                    await websocket.send_json({"type": "error", "message": "Call init_scan first"})
                    continue

                idx = msg.get("frame_index", frame_count)
                filename = f"{idx:06d}"

                # ── Save JPEG frame ──
                image_b64 = msg.get("image_base64")
                if image_b64:
                    import base64
                    jpg_bytes = base64.b64decode(image_b64)
                    jpg_path = frames_dir / f"{filename}.jpg"
                    with open(jpg_path, "wb") as f:
                        f.write(jpg_bytes)

                # ── Save camera data (pose + intrinsics) ──
                camera_info = {
                    "frame_index": idx,
                    "timestamp": msg.get("timestamp"),
                    "has_ar": msg.get("has_ar", False),
                    "intrinsics": msg.get("intrinsics"),     # {fx, fy, cx, cy, width, height}
                    "pose_matrix": msg.get("pose_matrix"),   # [16 floats] or null
                    "image_width": msg.get("image_width"),
                    "image_height": msg.get("image_height"),
                }
                cam_path = camera_data_dir / f"{filename}.json"
                with open(cam_path, "w") as f:
                    json.dump(camera_info, f)

                frame_count += 1

                # Ack every 10th frame to avoid flooding
                if frame_count % 10 == 0:
                    await websocket.send_json({
                        "type": "frame_ack",
                        "count": frame_count,
                    })

            elif msg_type == "stop_scan":
                total = msg.get("total_frames", frame_count)
                print(f"[Scan WS] Scan stopped: {total} frames saved to {frames_dir}")
                await websocket.send_json({
                    "type": "scan_complete",
                    "total_frames": total,
                    "project_id": project_id,
                    "scan_date": scan_date,
                })

    except WebSocketDisconnect:
        print(f"[Scan WS] Client disconnected ({frame_count} frames saved)")
    except Exception as e:
        print(f"[Scan WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


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