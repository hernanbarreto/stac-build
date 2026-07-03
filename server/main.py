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
from typing import Set, Optional, Dict, List, Any
from contextlib import asynccontextmanager

# Suppress polling spam from access logs (applies to all uvicorn start modes).
# These endpoints are hit on a tight interval by the UI and bury useful logs.
_POLLING_NOISE = (
    "GET /health",
    "GET /api/tasks/",
    "GET /api/segmentation/shape/progress/",
    "GET /api/segmentation/reconstruct/progress/",
    "GET /api/segmentation/tsdf/progress/",
)
class _HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(p in msg for p in _POLLING_NOISE):
            return False
        # 206 Partial Content range-request spam (Potree octree nodes, video).
        # uvicorn adds the phrase "Partial Content" in the FORMATTER, not the
        # message, so we match the STATUS CODE 206 (last positional arg of the
        # access record), not the text. Fallback: the raw '" 206' in the message.
        try:
            args = record.args
            if isinstance(args, (tuple, list)) and args and int(args[-1]) == 206:
                return False
        except (ValueError, TypeError, IndexError):
            pass
        return '" 206' not in msg

logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())

from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)

from frame_storage import get_frame_storage, FrameStorage
from alignment_manager import get_alignment_manager, AlignmentManager
from segmentation.sam3_wrapper import get_sam3_wrapper
from config import cfg, DATA_DIR, PROJECTS_DIR
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
    import logging as _logging
    _olog = _logging.getLogger("onload-cloudcompy")  # → server.log (root RotatingFileHandler)

    ctx = _ctx(session_id)
    scans_dir = ctx.output_dir
    output_ply = ctx.merged_cloud
    script_path = Path(__file__).parent / "run_cloudcompy.sh"

    voxel_size = postproc_config.get("voxel_size", 0.001)
    max_points = postproc_config.get("max_points", 0)

    if not script_path.exists():
        _olog.warning("[on-load PostProc] run_cloudcompy.sh not found, skipping")
        return

    # Check if chunks exist
    chunks = sorted(scans_dir.glob("chunk_*.ply"))
    if not chunks:
        _olog.warning(f"[on-load PostProc] No chunk PLYs found in {scans_dir}, skipping")
        return

    _olog.info(f"[on-load PostProc] Starting CloudCompPy on {len(chunks)} chunks, "
               f"voxel={voxel_size*1000:.1f}mm (session={session_id})")
    
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
        process = _track_worker(await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=_die_with_parent_sigkill,
        ))
        
        # Read output line by line
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='replace').strip()
            if line_str:
                print(f"  [on-load cloudcompy] {line_str}", flush=True)  # → console
                _olog.info(f"[on-load cloudcompy] {line_str}")           # → server.log
        
        await process.wait()
        
        if process.returncode == 0 and output_ply.exists():
            file_size_mb = output_ply.stat().st_size / (1024 * 1024)
            print(f"[PostProc] ✅ Cleaned cloud saved: {output_ply} ({file_size_mb:.1f} MB)")
            # Cascade cleanup (same as the pipeline cloudcompy worker): once cleaned_cloud
            # exists, the per-chunk PLYs are baked in → delete them so they don't linger
            # and don't re-trigger a rebuild on the next open.
            _removed = 0
            for _pat in ("chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json"):
                for _f in scans_dir.glob(_pat):
                    try:
                        _f.unlink(); _removed += 1
                    except Exception:
                        pass
            if _removed:
                print(f"[PostProc] [cleanup] removed {_removed} chunk files (baked into cleaned_cloud)")
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
                has_confidence = False
                while True:
                    line = f.readline()
                    if line.startswith(b"element vertex"):
                        n_pts = int(line.split()[-1])
                    if line.startswith(b"format binary"):
                        is_binary = True
                    if b"confidence" in line:
                        has_confidence = True
                    if b"frame_global" in line:
                        has_origins = True
                    if line.startswith(b"end_header"):
                        break
                
                if is_binary:
                    # Dynamically build numpy dtype based on detected fields
                    dtype_fields = [
                        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                        ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                    ]
                    if has_confidence:
                        dtype_fields.append(('confidence', '<f4'))
                    if has_origins:
                        dtype_fields.extend([
                            ('frame_global', '<i4'),
                            ('pixel_row', '<i2'), ('pixel_col', '<i2')
                        ])
                    ply_dtype = np.dtype(dtype_fields)
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


# Load libc ONCE in the parent (at import) so the preexec_fn below never does a
# dlopen inside the post-fork child — dlopen there can deadlock on the dynamic-linker
# lock if another thread held it at fork time. The child only calls the cached fn ptr.
try:
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6", use_errno=True)
except Exception:
    _LIBC = None


def _die_with_parent_sigkill():
    """preexec_fn for spawned worker subprocesses: ask the kernel to SIGKILL this
    child the instant its parent (this server) dies. Without it, restarting/killing
    the server ORPHANS its heavy workers (TSDF/Poisson/ShapeR/CloudCompPy) — they
    reparent to init and keep pegging CPU/GPU forever (the 'zombie Poisson at 2300%
    CPU after a server restart' incident). Linux-only; silently no-ops elsewhere.

    Belt-and-suspenders with the _ACTIVE_WORKERS registry + lifespan shutdown kill:
    PR_SET_PDEATHSIG covers a HARD parent death (SIGKILL / crash); the shutdown kill
    covers a graceful Ctrl+C; together no worker is ever orphaned."""
    if _LIBC is None:
        return
    try:
        import signal as _sig
        _PR_SET_PDEATHSIG = 1
        _LIBC.prctl(_PR_SET_PDEATHSIG, _sig.SIGKILL)
    except Exception:
        pass


# Every spawned worker subprocess registers here so a graceful shutdown (Ctrl+C →
# uvicorn → lifespan shutdown) can kill any still running. PR_SET_PDEATHSIG handles
# the hard-kill path; this handles the clean one.
_ACTIVE_WORKERS: set = set()


def _track_worker(proc):
    """Register a spawned worker and auto-deregister it when it exits."""
    _ACTIVE_WORKERS.add(proc)
    try:
        import asyncio as _aio
        _aio.ensure_future(_reap_worker(proc))
    except Exception:
        pass
    return proc


async def _reap_worker(proc):
    try:
        await proc.wait()
    except Exception:
        pass
    finally:
        _ACTIVE_WORKERS.discard(proc)


def _kill_active_workers(reason: str = "shutdown"):
    """SIGKILL every still-running tracked worker. Called from the lifespan shutdown
    so Ctrl+C never leaves an orphan pegging CPU/GPU."""
    for p in list(_ACTIVE_WORKERS):
        try:
            if p.returncode is None:
                p.kill()
                print(f"[Server] killed worker pid={getattr(p, 'pid', '?')} ({reason})")
        except Exception:
            pass
    _ACTIVE_WORKERS.clear()
# Read chunk settings from active reconstruction backend
_recon = cfg.get("reconstruction", {})
_backend = _recon.get("backend", "mapanything")
_backend_cfg = _recon.get(_backend, cfg.get("mapanything", {}))
CHUNK_SIZE = _backend_cfg.get("chunk_size", 120 if _backend == "da3" else 60)
CHUNK_OVERLAP = _backend_cfg.get("chunk_overlap", _backend_cfg.get("overlap", 60 if _backend == "da3" else 30))


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
    
    print(f"[Server] Reconstruction backend: {_backend.upper()} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("[Server] Models will be loaded on-demand (lazy loading enabled)")

    # Initialize auth database
    from db import init_db
    await init_db()

    # Re-apply the access-log noise filter AFTER uvicorn configured its logging.
    # uvicorn's startup dictConfig can drop the module-import-time filter (line ~33),
    # which is why "Partial Content" (206 range requests for Potree/video) kept
    # printing. Re-adding here (post-config) makes it stick.
    _ua = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _HealthCheckFilter) for f in _ua.filters):
        _ua.addFilter(_HealthCheckFilter())
        print("[Server] access-log noise filter re-applied (206/health/polling)")

    yield

    # ── Shutdown (Ctrl+C → uvicorn → here): kill any worker still running so none
    #    is orphaned. PR_SET_PDEATHSIG is the hard-kill backstop; this is the clean one.
    if _ACTIVE_WORKERS:
        print(f"[Server] shutdown — killing {len(_ACTIVE_WORKERS)} active worker(s)")
        _kill_active_workers("server shutdown")

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


@app.get("/api/sessions/{session_id}/video")
async def serve_session_video(session_id: str):
    """Serve the session's source video (for the synced video↔3D flythrough)."""
    from fastapi.responses import FileResponse
    ctx = _ctx(session_id)
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
        p = ctx.source_dir / f"source_video{ext}"
        if p.exists():
            return FileResponse(str(p), media_type="video/mp4",
                                headers={"Accept-Ranges": "bytes"})
    raise HTTPException(status_code=404, detail="source video not found")


@app.get("/api/sessions/{session_id}/flythrough")
async def get_flythrough(session_id: str):
    """Per-frame camera poses for the synced flythrough. Reads camera_poses.txt (+
    camera_frames.txt for the REAL frame numbers). Poses are already in the cloud's
    metric frame → scale 1.0. Source: mapanything (maplong_run) / DA3 / lidar."""
    import numpy as np
    ctx = _ctx(session_id)
    poses_txt = next((c for c in (ctx.output_dir / "maplong_run" / "camera_poses.txt",
                                  ctx.output_dir / "camera_poses.txt",
                                  ctx.output_dir / "da3_run" / "camera_poses.txt")
                      if c.exists()), None)
    if poses_txt is None:
        raise HTTPException(status_code=404, detail="no camera poses for this session")
    mats = []
    with open(poses_txt) as f:
        for line in f:
            v = line.split()
            if len(v) >= 16:
                mats.append([float(x) for x in v[:16]])
    frames = list(range(len(mats)))
    fr_txt = poses_txt.parent / "camera_frames.txt"
    if not fr_txt.exists():
        fr_txt = ctx.output_dir / "camera_frames.txt"
    if fr_txt.exists():
        try:
            nums = [int(x) for x in open(fr_txt).read().split()]
            if len(nums) == len(mats):
                frames = nums
        except Exception:
            pass
    K = None
    for ip in (poses_txt.parent / "intrinsic.txt", ctx.output_dir / "intrinsic.txt"):
        if ip.exists():
            try:
                arr = np.loadtxt(str(ip))
                row = arr if arr.ndim == 1 else arr[len(arr) // 2]
                K = [float(x) for x in np.asarray(row).reshape(-1)[:4]]
            except Exception:
                pass
            break

    # Video fps + total frame count — the flythrough has poses only at KEYFRAMES
    # (frames[]), spaced non-uniformly. The client maps the video's real playback
    # time → real frame number (via these) → the surrounding keyframes, then
    # interpolates. Without the total count it can't place a pose at the right
    # time and the camera drifts out of sync with the footage. Best-effort: cv2
    # reads container metadata (no decode), so this is cheap.
    fps, video_n_frames = None, None
    try:
        import cv2
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
            vp = ctx.source_dir / f"source_video{ext}"
            if vp.exists():
                cap = cv2.VideoCapture(str(vp))
                f = cap.get(cv2.CAP_PROP_FPS)
                nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                fps = float(f) if f and f > 0 else None
                video_n_frames = int(nf) if nf and nf > 0 else None
                break
    except Exception:
        pass

    return {"n_frames": len(frames), "frames": frames, "poses": mats,
            "scale": 1.0, "intrinsics": K, "fps": fps,
            "video_n_frames": video_n_frames,
            "video_url": f"/api/sessions/{session_id}/video"}

# Mount auth routes
from auth.routes_auth import router as auth_router
app.include_router(auth_router)

# Mount team routes
from auth.routes_team import router as team_router
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

# Persist EVERYTHING to disk so runs are auditable after the fact (server +
# pipeline + relayed worker logs: DA3, MapAnything, cloudcompy, TSDF, errors).
# pipeline_manager relays each worker `send_log` via logger.info → captured here.
from logging.handlers import RotatingFileHandler as _RFH
_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
_file_log_handler = _RFH(_log_dir / "server.log",
                         maxBytes=100 * 1024 * 1024, backupCount=5, encoding="utf-8")
_file_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_file_log_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_file_log_handler)
logging.getLogger().setLevel(logging.INFO)
# (No console StreamHandler here: pipeline_manager already print()s its relayed
# worker logs to stdout, and the on-load cloudcompy print()s too — a root
# StreamHandler would DOUBLE every pipeline line on screen.)

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
    from db.team import TeamMember, SessionAssignment, Team
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
        projects_dir = PROJECTS_DIR
        
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
    projects_dir = PROJECTS_DIR
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


# ── Video upload → frame extraction ─────────────────────────────────
# A freshly-created project has no frames, so reconstruction can't run yet.
# The user uploads a video, we extract ALL frames into ctx.frames_dir as
# {idx:06d}.jpg (same naming the frame selector / reconstruction expect),
# then the UI swaps its "+" (upload) button for the "🔨" (reconstruct) one.
# Extraction can take a while, so it runs as a background task; the UI polls
# GET /api/sessions/{id}/video/extract_progress.
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
_video_extract_state: Dict[str, Dict[str, Any]] = {}  # session_id -> {phase, pct, saved, total, error}


def _video_extract_set(session_id: str, **kw):
    """Synchronous, GIL-safe — callable from the executor thread."""
    st = _video_extract_state.setdefault(session_id, {"phase": "idle"})
    st.update(kw)


def _extract_video_frames_sync(video_path: str, frames_dir: Path, session_id: str):
    """Decode every frame of the video to frames_dir/{idx:06d}.jpg. Runs in an executor."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _video_extract_set(session_id, phase="error", error="could not open video")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    idx = 0
    saved = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"{idx:06d}.jpg"), frame, params)
            idx += 1
            saved += 1
            if saved % 25 == 0:
                pct = round(100.0 * idx / total, 1) if total else 0.0
                _video_extract_set(session_id, pct=pct, saved=saved, total=total)
    finally:
        cap.release()
    _video_extract_set(session_id, phase="done", pct=100.0, saved=saved,
                       total=total or saved, error=None, finished_at=time.time())
    print(f"[Video] ✅ Extracted {saved} frames → {frames_dir}", flush=True)


@app.post("/api/sessions/{session_id}/video/upload")
async def upload_video(
    session_id: str,
    file: UploadFile = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Upload a video and extract all its frames into the session. Admin/manager only.
    Returns immediately; poll /api/sessions/{id}/video/extract_progress for status."""
    from auth import decode_token
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload.get("role") not in ("admin", "manager"):
        raise HTTPException(status_code=403, detail="Admin or Manager only")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported video type. Accepted: {', '.join(_VIDEO_EXTS)}")

    ctx = _ctx(session_id)
    if not ctx.session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    st = _video_extract_state.get(session_id)
    if st and st.get("phase") == "extracting":
        raise HTTPException(status_code=409, detail="A video is already being processed for this session")

    # Stream the upload to disk (videos can be large — avoid loading into RAM)
    ctx.source_dir.mkdir(parents=True, exist_ok=True)
    video_path = ctx.source_dir / f"source_video{ext}"
    with video_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"[Video] ⬆️ Received {file.filename} → {video_path}", flush=True)

    _video_extract_set(session_id, phase="extracting", pct=0.0, saved=0, total=0,
                       error=None, started_at=time.time())

    loop = asyncio.get_event_loop()

    async def _job():
        try:
            await loop.run_in_executor(
                None, _extract_video_frames_sync, str(video_path), ctx.frames_dir, session_id
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            _video_extract_set(session_id, phase="error", error=str(e))

    asyncio.create_task(_job())
    return {"ok": True, "started": True, "filename": file.filename}


@app.get("/api/sessions/{session_id}/video/extract_progress")
async def video_extract_progress(session_id: str):
    """Poll frame-extraction progress. When idle, reports the on-disk frame count."""
    st = _video_extract_state.get(session_id)
    if not st:
        ctx = _ctx(session_id)
        fc = len(list(ctx.frames_dir.glob("*.jpg"))) if ctx.frames_dir.exists() else 0
        return {"phase": "idle", "frame_count": fc}
    return st


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
    projects_dir = PROJECTS_DIR
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
    projects_dir = PROJECTS_DIR
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
        from db.team import SessionAssignment, ActivityLog
        from db.project import Project
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

def _detect_recon_state(output_dir: Path) -> dict:
    """Detect reconstruction completeness for a scan's output directory.

    - 'complete': final chunk_*.ply already exist
    - 'partial':  DA3-streaming chunk cache exists (resumable) but no final output
    - 'none':     nothing reconstructed yet

    The UI uses this to default "Replace existing outputs" to OFF (resume) when a
    partial run can be completed instead of re-run from scratch.
    """
    try:
        if output_dir.exists() and any(output_dir.glob("chunk_*.ply")):
            return {"recon_state": "complete", "cached_chunks": 0}
        cache = output_dir / "da3_run" / "_tmp_results_unaligned"
        if cache.exists():
            n = len(list(cache.glob("chunk_*.npy")))
            if n > 0:
                return {"recon_state": "partial", "cached_chunks": n}
    except Exception:
        pass
    return {"recon_state": "none", "cached_chunks": 0}


@app.get("/sessions/{session_id}/scans")
async def get_session_scans(session_id: str):
    """List all scan days and sources for a project.
    Returns scan_key identifiers used by the pipeline to target specific scans.
    """
    from project_paths import ProjectPaths

    projects_dir = PROJECTS_DIR
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
                **_detect_recon_state(ctx.output_dir),
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
                **_detect_recon_state(ctx.output_dir),
            })

    return {"session_id": session_id, "scans": scans}


@app.get("/api/sessions/{session_id}/available_backends")
async def get_available_backends(session_id: str, scan_key: str = None):
    """Return which reconstruction backends are available for this session.

    Checks for Stray Scanner data (depth/, odometry.csv, camera_matrix.csv)
    in the source directory. If present, hybrid and lidar modes are available.

    Query params:
        scan_key: Optional "date/source" to check a specific scan (e.g. "2026-04-09/ipad_hernan").
                  If omitted, checks the latest/default scan.

    Returns:
        {
            "session_id": "...",
            "backends": ["da3", "hybrid", "lidar", "mapanything"],
            "stray_data": { "has_lidar": true, "has_arkit": true, ... },
            "recommended": "hybrid"
        }
    """
    from ingestors.stray_detector import detect_stray_data
    from project_paths import ProjectPaths

    # Resolve session dir
    if scan_key:
        parts = scan_key.split("/", 1)
        date = parts[0]
        source = parts[1] if len(parts) > 1 else "default"
        projects_dir = PROJECTS_DIR
        if (projects_dir / session_id / "project.json").exists():
            paths = ProjectPaths(str(projects_dir), session_id)
            ctx = paths.for_source(date, source)
            session_dir = str(ctx.session_dir)
        else:
            ctx = _ctx(session_id)
            session_dir = str(ctx.session_dir)
    else:
        ctx = _ctx(session_id)
        session_dir = str(ctx.session_dir)

    # Detect Stray Scanner data
    stray_info = detect_stray_data(session_dir)

    # Always available
    backends = ["da3", "mapanything"]

    # Stray Scanner backends (streaming)
    if stray_info["is_stray_session"]:
        backends.insert(1, "hybrid")
        backends.insert(2, "lidar")

    # GauS-SLAM backends
    backends.append("gaus_slam_da3")                       # always available (no LiDAR needed)
    if stray_info["has_lidar"] and stray_info["has_arkit"]:
        backends.append("gaus_slam_lidar")                 # LiDAR + ARKit
        backends.append("gaus_slam_hybrid")                # LiDAR-calibrated DA3 + ARKit

    # Recommendation
    if stray_info["has_lidar"] and stray_info["has_arkit"]:
        recommended = "gaus_slam_lidar"
    elif stray_info["is_stray_session"]:
        recommended = "hybrid"
    else:
        recommended = "da3"

    return {
        "session_id": session_id,
        "scan_key": scan_key,
        "backends": backends,
        "stray_data": {
            "has_lidar": stray_info["has_lidar"],
            "has_arkit": stray_info["has_arkit"],
            "has_intrinsics": stray_info["has_intrinsics"],
            "is_stray_session": stray_info["is_stray_session"],
            "depth_count": stray_info["depth_count"],
        },
        "recommended": recommended,
    }

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

# Global SAM3 init status (keyed by session_id)
_sam3_init_status: dict = {}

@app.post("/api/segmentation/start_session/{session_id}")
async def start_interactive_segmentation(session_id: str):
    """
    Kicks off SAM3 initialization as a background task.
    Returns 202 immediately. Frontend polls /api/segmentation/init_status/{session_id}.
    """
    # If already loading for this session, skip
    current = _sam3_init_status.get(session_id, {})
    if current.get("status") == "loading":
        return JSONResponse({"ok": True, "status": "loading"}, status_code=202)

    from segmentation.sam3_wrapper import get_sam3_wrapper

    ctx = _ctx(session_id)
    frames_dir = ctx.frames_dir
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Frames directory not found")

    # Read keyframes
    keyframes = None
    selected_frames_path = frames_dir / "selected_frames.json"
    if selected_frames_path.exists():
        with open(selected_frames_path) as f:
            sf_data = json.load(f)
        keyframes = sf_data.get("selected_files", [])
        keyframes = [fn for fn in keyframes if (frames_dir / fn).exists()]
        if keyframes:
            print(f"[InteractiveSeg] Using {len(keyframes)} keyframes from selected_frames.json")

    if not keyframes:
        keyframes = sorted([f.name for f in frames_dir.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        print(f"[InteractiveSeg] No selected_frames.json, using all {len(keyframes)} frames")

    _sam3_init_status[session_id] = {"status": "loading", "message": "Loading SAM3 model..."}

    async def _background_init():
        try:
            sam3 = get_sam3_wrapper()
            # Clean up existing sessions
            for sid in list(sam3._interactive_sessions.keys()):
                try:
                    sam3.predictor.handle_request(request=dict(type="reset_session", session_id=sid))
                    session_info = sam3._interactive_sessions.pop(sid, None)
                    if session_info and isinstance(session_info, dict):
                        import shutil
                        temp_dir = session_info.get("keyframe_temp_dir")
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass

            loop = asyncio.get_event_loop()
            _sam3_init_status[session_id] = {"status": "loading", "message": "Initializing SAM3 session..."}

            def _init():
                return sam3.init_interactive_session(str(frames_dir), keyframes=keyframes)

            state_id = await loop.run_in_executor(None, _init)

            keyframes_sorted = sorted(keyframes)
            kf_mapping = {i: name for i, name in enumerate(keyframes_sorted)}

            _sam3_init_status[session_id] = {
                "status": "ready",
                "state_id": state_id,
                "session_id": session_id,
                "num_keyframes": len(keyframes_sorted),
                "kf_mapping": kf_mapping,
            }
            print(f"[InteractiveSeg] ✅ SAM3 ready for {session_id} (state_id={state_id})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _sam3_init_status[session_id] = {"status": "error", "message": str(e)}

    asyncio.create_task(_background_init())
    return JSONResponse({"ok": True, "status": "loading"}, status_code=202)


@app.get("/api/segmentation/init_status/{session_id}")
async def get_sam3_init_status(session_id: str):
    """Lightweight polling endpoint — returns current SAM3 init status."""
    status = _sam3_init_status.get(session_id, {"status": "idle"})
    return status

@app.post("/api/segmentation/reset_session")
async def reset_interactive_segmentation(request: Request):
    """
    Reset/close an interactive SAM3 session to free VRAM.
    Called when the SegmentationManager is closed.
    """
    from segmentation.sam3_wrapper import get_sam3_wrapper
    body = await request.json()
    state_id = body.get("state_id")
    if not state_id:
        raise HTTPException(status_code=400, detail="Missing state_id")

    sam3 = get_sam3_wrapper()
    loop = asyncio.get_event_loop()

    def _reset():
        try:
            sam3.predictor.handle_request(
                request=dict(type="reset_session", session_id=state_id)
            )
            # Cleanup temp keyframe dir if it exists
            session_info = sam3._interactive_sessions.pop(state_id, None)
            if session_info and isinstance(session_info, dict):
                import shutil
                temp_dir = session_info.get("keyframe_temp_dir")
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[InteractiveSeg] Session {state_id} reset, VRAM freed")
        except Exception as e:
            print(f"[InteractiveSeg] Reset session error (non-fatal): {e}")

    await loop.run_in_executor(None, _reset)
    return {"ok": True}

@app.post("/api/segmentation/clear_prompts")
async def clear_interactive_prompts(request: Request):
    """
    Clear prompts from the current interactive session WITHOUT destroying it.
    Frames stay loaded, model stays loaded.

    Body: { state_id, obj_id?: int }
      - with obj_id: remove ONLY that object's prompts/tracking (the UI's
        per-context "Clear" — other queued objects survive).
      - without: reset every tracked object ("Clear all").
    """
    from segmentation.sam3_wrapper import get_sam3_wrapper
    body = await request.json()
    state_id = body.get("state_id")
    obj_id = body.get("obj_id")
    if not state_id:
        raise HTTPException(status_code=400, detail="Missing state_id")

    sam3 = get_sam3_wrapper()
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(
        None, lambda: sam3.clear_interactive_prompts(state_id, obj_id=obj_id))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to clear prompts")
    return {"ok": True}
_last_prompt = {"key": None, "time": 0, "result": None}
_prompt_in_flight = {"key": None, "event": None}  # Track currently-executing prompt + completion signal

@app.post("/api/segmentation/add_prompt")
async def add_interactive_prompt(request: Request):
    """
    Apply a positive or negative click to a specific frame.
    Returns the predicted mask as base64 PNG for live overlay.
    
    Handles browser timeout retries: if the same prompt is still being
    processed, the retry will WAIT for completion and return the result.
    """
    import time as _time
    import threading as _threading
    body = await request.json()
    state_id = body.get("state_id")
    session_id = body.get("session_id")
    frame_idx = body.get("frame_idx")
    obj_id = body.get("obj_id", 1)
    points = body.get("points")
    labels = body.get("labels")
    
    if not all([state_id, frame_idx is not None, points, labels]):
        raise HTTPException(status_code=400, detail="Missing required parameters")
    
    prompt_key = f"{state_id}:{frame_idx}:{points}:{labels}"
    now = _time.time()

    # ── Guard 1: Same prompt still in-flight → WAIT for result (don't reject)
    if prompt_key == _prompt_in_flight["key"] and _prompt_in_flight["event"] is not None:
        print(f"[add_prompt] ⏳ Duplicate detected — waiting for in-flight result...")
        evt = _prompt_in_flight["event"]
        loop = asyncio.get_event_loop()
        # Wait for the event in a thread to not block the event loop (up to 10 min)
        got_result = await loop.run_in_executor(None, lambda: evt.wait(timeout=600))
        if got_result and _last_prompt.get("result") and _last_prompt["key"] == prompt_key:
            print(f"[add_prompt] ✅ Returning result from in-flight prompt")
            return _last_prompt["result"]
        print(f"[add_prompt] ⚠️ In-flight prompt finished but no result available")
        return {"ok": True}

    # ── Guard 2: Short debounce for rapid double-clicks (< 2 seconds)
    if prompt_key == _last_prompt["key"] and (now - _last_prompt["time"]) < 2.0:
        print(f"[add_prompt] Debounced duplicate call (Δ={now - _last_prompt['time']:.2f}s)")
        return _last_prompt.get("result") or {"ok": True}

    # ── Guard 3: If result is already cached for this exact prompt, return it
    if prompt_key == _last_prompt["key"] and _last_prompt.get("result"):
        print(f"[add_prompt] Returning cached result for identical prompt")
        return _last_prompt["result"]

    # Set up in-flight tracking with completion event
    completion_event = _threading.Event()
    _last_prompt["key"] = prompt_key
    _last_prompt["time"] = now
    _last_prompt["result"] = None
    _prompt_in_flight["key"] = prompt_key
    _prompt_in_flight["event"] = completion_event
        
    from segmentation.sam3_wrapper import get_sam3_wrapper
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
        _last_prompt["result"] = resp
        return resp
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Signal completion so any waiting retry gets the result
        completion_event.set()
        # Clear in-flight flag
        if _prompt_in_flight["key"] == prompt_key:
            _prompt_in_flight["key"] = None
            _prompt_in_flight["event"] = None

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

    # Save to floor_transform.npz (capture the OLD transform first — the OBB
    # re-projection below needs the old→new delta)
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

    # Re-project segmentation_result OBBs (stored in the display frame) with
    # the old→new transform delta instead of deleting the file: deleting it
    # forced a full DBSCAN + cloud-matching rerun (~minutes) on the next
    # refresh just because the user nudged the gizmo. DBSCAN should only run
    # on Segmentation Manager exit when masks actually changed.
    seg_result_path = output_dir / "segmentation_result.json"
    if seg_result_path.exists():
        try:
            with open(seg_result_path) as f:
                result_data = json.load(f)
            n_obb = _recompute_result_obbs(output_dir, result_data, s, R, t)
            with open(seg_result_path, "w") as f:
                json.dump(result_data, f)
            print(f"[Alignment] ✅ Recomputed {n_obb} OBBs under the new frame "
                  "(no DBSCAN rerun needed)")
        except Exception as e:
            # fallback: stale OBBs are worse than a recompute — invalidate
            print(f"[Alignment] OBB re-projection failed ({e}) — deleting "
                  "segmentation_result.json for recompute")
            seg_result_path.unlink()

    return {"ok": True, "scale": s}


# ── Floor leveling (segmented floor → y=0 on the XZ display plane) ──
# The display frame is Y-up (xyz_display = s·R·xyz + t, see
# segmentation/pipeline.py). The selected FLOOR instance's fitted plane is
# rotated to +Y and translated to y=0 with a MINIMAL delta over the current
# transform. segmentation_result OBBs (stored in display frame) are
# re-projected with the same delta so nothing needs a DBSCAN rerun and the
# viewer updates instantly.

def _load_floor_transform_srt(output_dir):
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    p = output_dir / "floor_transform.npz"
    if p.exists():
        try:
            d = np.load(p)
            s, R, t = float(d["s"]), d["R"], d["t"]
        except Exception:
            pass
    return s, R, t


def _floor_candidates_from_result(result_data):
    """Floor instances with their current display height (obb centre Y)."""
    out = []
    for inst in result_data.get("instances", []):
        if "floor" not in str(inst.get("label", "")).lower():
            continue
        obb = inst.get("obb") or {}
        c = obb.get("center")
        out.append({
            "instance_id": inst.get("instance_id", inst.get("id")),
            "label": inst.get("label", ""),
            "height_m": (float(c[1]) if c else None),
            "n_points": inst.get("total_points", 0),
        })
    return out


def _srt_to_threejs_col_major(s, R, t):
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M.flatten(order="F").tolist()


def _recompute_result_obbs(output_dir, result_data, s, R, t):
    """Recompute EVERY instance's OBB from the cloud under the CURRENT
    transform. Delta re-projection is not enough: historical rebuilds left
    result OBBs out of sync with the npz (test3's floor OBB sat 105 mm off
    the actual plane), and a delta faithfully preserves that stale offset.
    globalIndices → no DBSCAN, no matching; just per-instance OBB fits."""
    import open3d as o3d
    from segmentation.pipeline import _compute_obb
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        return 0
    pts = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    n = 0
    for inst in result_data.get("instances", []):
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < len(pts))]
        if len(gi) < 4:
            continue
        seg = pts[gi]
        if len(seg) > 150_000:
            seg = seg[np.random.default_rng(0).choice(len(seg), 150_000,
                                                      replace=False)]
        xyz_display = s * (seg @ R.T) + t
        try:
            inst["obb"] = _compute_obb(xyz_display)
            n += 1
        except Exception:
            pass
    return n


@app.get("/api/segmentation/floor_level/{session_id}")
async def get_floor_level(session_id: str):
    """Floor candidates + current selection for the SEGMENTS combobox."""
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    result_path = output_dir / "segmentation_result.json"
    if not result_path.exists():
        return {"ok": True, "candidates": [], "selected": None}
    with open(result_path) as f:
        result_data = json.load(f)
    candidates = _floor_candidates_from_result(result_data)
    selected = None
    fl = output_dir / "floor_level.json"
    if fl.exists():
        try:
            selected = json.loads(fl.read_text()).get("selected_instance_id")
        except Exception:
            pass
    if selected not in {c["instance_id"] for c in candidates}:
        selected = None
    return {"ok": True, "candidates": candidates, "selected": selected}


@app.post("/api/segmentation/level_floor")
async def level_floor(request: Request):
    """Level the selected (or lowest) segmented floor to y=0.

    Body: { session_id, instance_id?: int,
            mode?: "auto" | "explicit" | "auto_if_needed" }
      - auto: previously selected floor if it still exists, else the LOWEST
      - explicit: the given instance_id (combobox change)
      - auto_if_needed: no-op when the floor already sits at y=0 (within 1 cm
        and 0.5° of horizontal) — used on session load
    """
    body = await request.json()
    session_id = body.get("session_id")
    req_iid = body.get("instance_id")
    mode = body.get("mode", "explicit" if req_iid is not None else "auto")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    result_path = output_dir / "segmentation_result.json"
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not result_path.exists() or not cloud_path.exists():
        return {"ok": True, "leveled": False,
                "reason": "no segmentation/cloud yet"}

    loop = asyncio.get_event_loop()

    def _level():
        import open3d as o3d
        from reconstruction.geometry.primitives import fit_plane_ransac

        with open(result_path) as f:
            result_data = json.load(f)
        candidates = _floor_candidates_from_result(result_data)
        if not candidates:
            return {"ok": True, "leveled": False, "reason": "no floor instances",
                    "candidates": []}

        # selection: explicit > remembered > lowest
        fl_path = output_dir / "floor_level.json"
        remembered = None
        if fl_path.exists():
            try:
                remembered = json.loads(fl_path.read_text()).get("selected_instance_id")
            except Exception:
                pass
        cand_ids = {c["instance_id"] for c in candidates}
        if mode == "explicit" and req_iid is not None:
            selected = int(req_iid)
            if selected not in cand_ids:
                return {"ok": False, "leveled": False,
                        "reason": f"instance {selected} is not a floor",
                        "candidates": candidates}
        elif remembered in cand_ids:
            selected = remembered
        else:
            with_h = [c for c in candidates if c["height_m"] is not None]
            selected = (min(with_h, key=lambda c: c["height_m"])
                        if with_h else candidates[0])["instance_id"]

        inst = next(i for i in result_data["instances"]
                    if i.get("instance_id", i.get("id")) == selected)
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        if len(gi) < 100:
            return {"ok": False, "leveled": False,
                    "reason": "selected floor has too few points",
                    "candidates": candidates}

        pts = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
        gi = gi[(gi >= 0) & (gi < len(pts))]
        seg = pts[gi]
        if len(seg) > 200_000:
            seg = seg[np.random.default_rng(0).choice(len(seg), 200_000,
                                                      replace=False)]

        s, R, t = _load_floor_transform_srt(output_dir)
        pf = fit_plane_ransac(seg, dist_thresh=0.02, iters=400,
                              min_inlier_frac=0.2, measure_curvature=False)
        if pf is None:
            return {"ok": False, "leveled": False,
                    "reason": "floor plane fit failed", "candidates": candidates}
        n_raw = pf.normal
        n_disp = R @ n_raw
        if n_disp[1] < 0:
            n_raw, n_disp = -n_raw, -n_disp
        c_raw = seg[pf.inliers].mean(0)
        c_disp = s * (R @ c_raw) + t

        tilt_deg = float(np.degrees(np.arccos(np.clip(n_disp[1], -1, 1))))
        height = float(c_disp[1])
        already = tilt_deg < 0.5 and abs(height) < 0.01
        if mode == "auto_if_needed" and already:
            return {"ok": True, "leveled": True, "changed": False,
                    "selected": selected, "candidates": candidates,
                    "residual_mm": round(height * 1000, 1),
                    "matrix": _srt_to_threejs_col_major(s, R, t)}

        # minimal delta rotation in DISPLAY space: n_disp → +Y
        up = np.array([0.0, 1.0, 0.0])
        v = n_disp / max(np.linalg.norm(n_disp), 1e-12)
        axis = np.cross(v, up)
        ln = np.linalg.norm(axis)
        if ln < 1e-9:
            R_delta = np.eye(3)
        else:
            axis /= ln
            ang = float(np.arccos(np.clip(v @ up, -1, 1)))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R_delta = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

        R_new = R_delta @ R
        t_new = t.copy()
        # keep lateral placement; put the fitted plane exactly at y=0
        c_disp_new = R_delta @ (c_disp - t) + t
        t_new[1] = t[1] - c_disp_new[1]
        np.savez(output_dir / "floor_transform.npz",
                 s=np.array(s), R=R_new, t=np.array([t[0], t_new[1], t[2]]))

        # Recompute EVERY OBB from the cloud under the new transform (delta
        # re-projection preserves any historical drift between result OBBs
        # and the npz — see _recompute_result_obbs). No DBSCAN rerun; the
        # rewrite also keeps the result-cache mtime valid vs the new npz.
        t_final = np.array([t[0], t_new[1], t[2]])
        n_obb = _recompute_result_obbs(output_dir, result_data, s, R_new, t_final)
        print(f"[FloorLevel]   recomputed {n_obb} OBBs under the new frame")
        with open(result_path, "w") as f:
            json.dump(result_data, f)

        fl_path.write_text(json.dumps({
            "selected_instance_id": selected,
            "candidates": candidates,
            "leveled_height_before_mm": round(height * 1000, 1),
            "tilt_before_deg": round(tilt_deg, 3),
        }, indent=2))

        print(f"[FloorLevel] {session_id}: floor inst {selected} → y=0 "
              f"(was {height*1000:+.1f} mm, tilt {tilt_deg:.2f}°)")
        return {"ok": True, "leveled": True, "changed": True,
                "selected": selected, "candidates": candidates,
                "residual_before_mm": round(height * 1000, 1),
                "matrix": _srt_to_threejs_col_major(
                    s, R_new, np.array([t[0], t_new[1], t[2]]))}

    result = await loop.run_in_executor(None, _level)
    return result


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

@app.post("/api/segmentation/evaluate_frames")
async def evaluate_frames(request: Request):
    """
    Use DINOv2 (via DINOScout) to evaluate which KEYFRAMES contain the
    prompted object.  Only processes the keyframes loaded in SAM3 (not all
    session frames), making this 5-10× faster on CPU.

    Returns a list of booleans (one per keyframe) indicating presence.
    """
    body = await request.json()
    state_id = body.get("state_id")
    session_id = body.get("session_id")
    frame_idx = body.get("frame_idx", 0)
    obj_id = body.get("obj_id", 1)
    threshold = body.get("threshold", 0.25)

    if not state_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing state_id or session_id")

    from segmentation.sam3_wrapper import get_sam3_wrapper

    sam3 = get_sam3_wrapper()
    mask = sam3.get_mask_for_frame(state_id, frame_idx, obj_id)
    if mask is None:
        raise HTTPException(status_code=400, detail="No initial mask found for this object. Prompt first.")

    # Build keyframe path list from init status (only the frames SAM3 loaded)
    init_status = _sam3_init_status.get(session_id, {})
    kf_mapping = init_status.get("kf_mapping", {})
    if not kf_mapping:
        raise HTTPException(status_code=400, detail="SAM3 session not initialized (no kf_mapping)")

    ctx = _ctx(session_id)
    frames_dir = ctx.frames_dir
    sorted_keys = sorted(kf_mapping.keys(), key=lambda k: int(k))
    kf_paths = [str(frames_dir / kf_mapping[k]) for k in sorted_keys]
    source_path = kf_paths[frame_idx] if frame_idx < len(kf_paths) else kf_paths[0]

    print(f"[Evaluate] Using DINOScout on {len(kf_paths)} keyframes (threshold={threshold})")

    loop = asyncio.get_event_loop()

    def _eval():
        from segmentation.dino_scout import DINOScout
        scout = DINOScout()
        try:
            # Precompute features (cached to output dir on disk)
            output_dir = ctx.output_dir if hasattr(ctx, 'output_dir') else None
            save_dir = str(output_dir) if output_dir and output_dir.exists() else None
            precomputed = scout.precompute_keyframe_features(kf_paths, save_dir=save_dir)

            result_paths = scout.find_object_frames(
                mask=mask,
                source_frame_path=source_path,
                all_keyframe_paths=kf_paths,
                threshold=threshold,
                precomputed_features=precomputed,
                min_frames=3,
            )
        finally:
            scout.unload()

        # Convert matched paths → boolean array aligned with keyframe order
        result_set = set(result_paths)
        return [p in result_set for p in kf_paths]

    valid_frames = await loop.run_in_executor(None, _eval)
    n_selected = sum(valid_frames)
    print(f"[Evaluate] ✅ {n_selected}/{len(valid_frames)} keyframes selected")
    return {"ok": True, "valid_frames": valid_frames}

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
    # Multi-object: {obj_id: name} so each tracked object keeps its own label.
    # Falls back to label_name for any obj_id not listed (and for the legacy
    # single-object UI that sends only label_name).
    obj_labels_req = body.get("obj_labels", None)
    selected_frames = body.get("selected_frames", None)
    
    if not state_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing state_id or session_id")
        
    # Unload any DINOv2 models before running SAM3 propagation (free CPU RAM)
    from segmentation.dino_evaluator import get_dino_evaluator
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_dino_evaluator().unload_model)
    
    # Prevent concurrent propagations on the same session
    active_tasks = task_manager.get_active(session_id)
    if any(task["task_type"] == "propagation" for task in active_tasks):
        raise HTTPException(status_code=409, detail="A propagation is already running for this session. Please wait or cancel the existing task.")
    
    from segmentation.sam3_wrapper import get_sam3_wrapper
    sam3 = get_sam3_wrapper()
    
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    
    # Get kf_mapping to translate SAM3 sequential indices → real frame numbers
    kf_mapping = _sam3_init_status.get(session_id, {}).get("kf_mapping", {})
    
    tid = task_manager.start(session_id, "propagation", f'Propagating "{label_name}"')
    
    # Shared state for save logic (accessible from finally block)
    _all_masks = {}
    _save_done = False
    
    def _save_results():
        """Save masks + 3D matching. Called from finally if propagation produced masks."""
        nonlocal _save_done
        if _save_done or not _all_masks:
            return
        _save_done = True
        try:
            from segmentation_pipeline import _save_masks, _parse_raw_masks, _match_and_save_result
            structured_masks = _parse_raw_masks(_all_masks)
            
            # ── Translate SAM3 sequential indices → real frame_global ──
            # SAM3 uses sequential indices (0,1,2,...) for its temp dir,
            # but the PLY has frame_global from the original filenames.
            # kf_mapping: {seq_idx: "001188.jpg"} → extract numeric part
            translated_masks = {}
            for f_idx, frame_masks in structured_masks.items():
                if kf_mapping:
                    kf_name = kf_mapping.get(f_idx, kf_mapping.get(str(f_idx)))
                    if kf_name:
                        # Extract numeric frame ID from filename (e.g., "001188.jpg" → 1188)
                        import re
                        m = re.search(r'(\d+)', kf_name)
                        real_frame = int(m.group(1)) if m else f_idx
                    else:
                        real_frame = f_idx
                else:
                    real_frame = f_idx
                translated_masks[real_frame] = frame_masks
            
            if kf_mapping:
                sample_in = list(structured_masks.keys())[:3]
                sample_out = list(translated_masks.keys())[:3]
                print(f"[SegPipeline] Frame translation: SAM3 {sample_in} → real {sample_out}")
            
            # Preserve SAM3's per-object ids (out_obj_ids) so each tracked object
            # keeps its OWN label, instead of collapsing them all into one. The UI
            # sends obj_labels={obj_id: name}; any id without an explicit name falls
            # back to label_name (covers the legacy single-object flow).
            obj_label_map = {}
            for k, v in (obj_labels_req or {}).items():
                try:
                    obj_label_map[int(k)] = str(v)
                except (TypeError, ValueError):
                    pass
            all_obj_ids = set()
            for frame_masks in translated_masks.values():
                all_obj_ids.update(frame_masks.keys())
            obj_labels = {oid: obj_label_map.get(oid, label_name) for oid in all_obj_ids}
            categories = sorted(set(obj_labels.values())) or [label_name]
            saved_seg = _save_masks(output_dir, translated_masks, categories, obj_labels, cfg)
            print(f"[Segmentation] Saved {len(all_obj_ids)} object(s): "
                  f"{', '.join(f'{oid}={obj_labels[oid]}' for oid in sorted(all_obj_ids))}")
            
            # NOTE: We intentionally do NOT run _match_and_save_result here.
            # Cloud matching + Potree rebuild is expensive (~minutes) and should
            # only run ONCE when the user finishes segmenting all objects
            # (i.e., when closing the Segmentation Manager or pressing Finalize).
            
            task_manager.finish(tid)
            print(f"[Segmentation] ✅ Saved {len(_all_masks)} frames for '{label_name}'")
        except Exception as e:
            print(f"[Segmentation] Error saving results: {e}")
            import traceback
            traceback.print_exc()
            task_manager.fail(tid, str(e))
    
    def sse_generator():
        import base64, cv2
        actual_total = len(selected_frames) if selected_frames else None
        
        try:
            for frame_idx, num_frames, outputs in sam3.propagate_interactive_stream(state_id, selected_frames=selected_frames):
                _all_masks[frame_idx] = outputs
                effective_total = actual_total if actual_total else num_frames
                pct = min(99, round((len(_all_masks) / max(1, effective_total)) * 100))
                task_manager.update(tid, pct=pct, detail=f"Frame {len(_all_masks)}/{effective_total}")
                
                # Generate a small mask preview PNG for this frame
                mask_b64 = ""
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
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
                    "frame": int(frame_idx), "total": int(effective_total),
                    "pct": pct, "mask_png": mask_b64
                })
                yield f"event: progress\ndata: {event}\n\n"
            
            # Propagation completed normally — save and emit done
            if not _all_masks:
                task_manager.fail(tid, "No masks generated")
                yield f"event: error\ndata: {{\"message\": \"No masks generated\"}}\n\n"
                return

            task_manager.update(tid, pct=100, detail="Saving masks...")
            yield f"event: saving\ndata: {{\"status\": \"Saving masks & matching to 3D cloud...\"}}\n\n"
            
            _save_results()
            
            done_event = json.dumps({"ok": True})
            yield f"event: done\ndata: {done_event}\n\n"
            
        except GeneratorExit:
            # Client disconnected — save partial results without yielding
            print(f"[Segmentation] Client disconnected. Saving {len(_all_masks)} frames in background...")
            _save_results()
            return  # MUST return, not re-raise, after GeneratorExit in sync generators
        except Exception as e:
            print(f"[Segmentation] Error during propagation: {e}")
            import traceback
            traceback.print_exc()
            task_manager.fail(tid, str(e))
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    
    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# --- User Viewer Preferences ---

@app.get("/api/sessions/{session_id}/prefs")
async def get_viewer_prefs(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Load per-user viewer preferences for a session."""
    from auth import decode_token
    if not credentials:
        return {}
    try:
        payload = decode_token(credentials.credentials)
        username = payload.get("username", "")
    except Exception:
        return {}
    
    ctx = _ctx(session_id)
    prefs_file = ctx.session_dir / "user_prefs" / f"{username}.json"
    if prefs_file.exists():
        return json.loads(prefs_file.read_text())
    return {}

@app.post("/api/sessions/{session_id}/prefs")
async def save_viewer_prefs(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Save per-user viewer preferences for a session."""
    from auth import decode_token
    if not credentials:
        return {"error": "Not authenticated"}
    try:
        payload = decode_token(credentials.credentials)
        username = payload.get("username", "")
    except Exception:
        return {"error": "Invalid token"}
    
    prefs = await request.json()
    ctx = _ctx(session_id)
    prefs_dir = ctx.session_dir / "user_prefs"
    prefs_dir.mkdir(parents=True, exist_ok=True)
    prefs_file = prefs_dir / f"{username}.json"
    prefs_file.write_text(json.dumps(prefs))
    return {"ok": True}

# --- Segmentation Manager Endpoints ---

@app.get("/api/sessions/{session_id}/segmentation")
async def get_segmentation_instances(session_id: str):
    """Return all existing segmentation instances for a session.

    Source of truth for **which instances exist** is ``segmentation.json``,
    which is updated by every interactive propagation (``_save_masks``).
    The grouped result file ``segmentation_result.json`` only gets refreshed
    by the expensive cloud-matching pass that runs on finalize — using IT as
    the list source means freshly-propagated instances stay invisible until
    the user closes the manager (the original bug).

    So: take the list from ``segmentation.json`` and *enrich* each entry with
    OBB / globalIndices / total_points from ``segmentation_result.json`` when
    available. Best of both: live list + enriched metadata.
    """
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir

    result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"

    if not seg_path.exists():
        return {"instances": [], "prompts": []}

    try:
        with open(seg_path) as f:
            seg_data = json.load(f)
        prompts = seg_data.get("prompts", [])
        resolution = seg_data.get("resolution", {})
        raw_instances = seg_data.get("instances", [])

        enriched_by_id: Dict[int, Dict[str, Any]] = {}
        if result_path.exists():
            try:
                with open(result_path) as f:
                    result_data = json.load(f)
                for inst in result_data.get("instances", []):
                    iid = inst.get("instance_id", inst.get("id"))
                    if iid is not None:
                        enriched_by_id[int(iid)] = inst
            except Exception:
                pass  # if it's malformed, just skip enrichment

        merged: List[Dict[str, Any]] = []
        for inst in raw_instances:
            # Cross-reference by instance_id, NOT id: segmentation.json uses
            # 0-based ``id`` while segmentation_result.json uses 1-based ``id``;
            # only ``instance_id`` is consistent across both. Matching on ``id``
            # mis-aligns the enrichment by one (door inherits the ladder's label
            # + points, ladder shows 0). See enriched_by_id above (also by iid).
            iid = inst.get("instance_id", inst.get("id"))
            if iid is not None and int(iid) in enriched_by_id:
                # Enriched fields (OBB, globalIndices, total_points) override
                # bare fields from segmentation.json; raw entries that don't
                # have a match yet (just-propagated, no cloud-match yet) pass
                # through unchanged so the UI list still includes them.
                merged.append({**inst, **enriched_by_id[int(iid)]})
            else:
                merged.append(inst)

        return {
            "instances": merged,
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
    # instance_id alone can collide across writers; the UI sends the row's
    # current label so the match is exact (None keeps old behavior)
    old_label_hint = body.get("old_label")

    if not session_id or instance_id is None:
        raise HTTPException(status_code=400, detail="Missing session_id or instance_id")

    def _match(inst, id_field):
        if inst.get(id_field) != instance_id:
            return False
        return old_label_hint is None or inst.get("label", "") == old_label_hint

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

        # Save old label BEFORE updating (for shape folder rename)
        old_label = None
        for inst in data.get("instances", []):
            if _match(inst, "instance_id"):
                old_label = inst.get("label")
                break

        updated = False
        for inst in data.get("instances", []):
            if _match(inst, "instance_id"):
                if new_label is not None:
                    inst["label"] = new_label
                if excluded is not None:
                    inst["excluded"] = excluded
                updated = True
                break  # Only rename ONE instance

        if updated:
            with open(seg_path, "w") as f:
                json.dump(data, f, indent=2)

        # Also update segmentation_result.json (grouped instances)
        if result_path.exists():
            with open(result_path) as f:
                result_data = json.load(f)
            for inst in result_data.get("instances", []):
                if _match(inst, "id") or _match(inst, "instance_id"):
                    if new_label is not None:
                        inst["label"] = new_label
                    if excluded is not None:
                        inst["excluded"] = excluded
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

        # Sync shape folder if label changed
        if new_label is not None and old_label and old_label != new_label:
            from segmentation.session_io import rename_shape_folder
            rename_shape_folder(output_dir, old_label, new_label, instance_id)

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
    # instance_id is NOT globally unique (obj_id vs instance_id spaces, and
    # shape/tsdf writers can collide) — the UI also sends the row's label so
    # the match is exact: (label, instance_id). label=None keeps old behavior.
    target_label = body.get("label")

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

            seg_data_original_instances = list(seg_data.get("instances", []))  # save for shape cleanup

            obj_ids_to_remove = set()
            instance_ids_to_remove = set()
            remaining_instances = []
            for inst in seg_data.get("instances", []):
                iid = inst.get("instance_id")
                oid = inst.get("id")
                # The UI lists and sends each entry's "id" (obj_id, with instance_id
                # as fallback — see GET /segmentation). Match ONLY that identifier.
                # Matching the OTHER field too cross-deletes once the obj_id and
                # instance_id spaces overlap (e.g. door obj0/inst1 + ladder obj1/inst2:
                # deleting "1" hit ladder's obj_id AND door's instance_id → both gone).
                list_id = oid if oid is not None else iid
                if target_label is not None:
                    # label pins the exact row, so the id may match EITHER
                    # field: the UI's seg.id is the RESULT-space instance_id,
                    # while this file's list_id prefers the SAM3 obj id —
                    # requiring list_id equality alone missed the entry
                    # entirely (wall2 id=3/instance_id=4: nothing deleted,
                    # and the GET — whose source of truth is THIS file —
                    # resurrected the row and its OBB in the viewer).
                    hit = (inst.get("label", "") == target_label
                           and instance_id in (oid, iid))
                else:
                    hit = (list_id == instance_id)
                if hit:
                    if oid is not None:
                        obj_ids_to_remove.add(oid)
                    if iid is not None:
                        instance_ids_to_remove.add(iid)
                else:
                    remaining_instances.append(inst)

            # Fallback: a caller-provided label that matches NOTHING (e.g. a
            # UI placeholder like "Object 3" for an unlabeled entry) must not
            # turn the delete into a silent no-op — retry with id-only
            # matching, which is unambiguous when the label failed to pin.
            if target_label is not None and not obj_ids_to_remove \
                    and not instance_ids_to_remove:
                print(f"[SegDelete]   label '{target_label}' matched nothing — "
                      "retrying by id only")
                remaining_instances = []
                for inst in seg_data.get("instances", []):
                    iid = inst.get("instance_id")
                    oid = inst.get("id")
                    if instance_id in (oid, iid):
                        if oid is not None:
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

            # 2) Remove from segmentation_result.json. Entries are joined to
            # segmentation.json by the INSTANCE id space only (canonical key:
            # instance_id, id as fallback) — matching the `id` field against
            # instance_ids cross-deletes unrelated entries once the obj_id and
            # instance_id spaces overlap (the "deleted A, B vanished" bug).
            def _result_key(inst):
                v = inst.get("instance_id")
                return v if v is not None else inst.get("id")

            def _is_target(inst):
                return (_result_key(inst) in remove_keys
                        and (target_label is None
                             or inst.get("label", "") == target_label))

            remove_keys = instance_ids_to_remove or {instance_id}
            fresh_instances = []
            if result_path.exists():
                with open(result_path) as f:
                    result_data = json.load(f)
                fresh_instances = [
                    inst for inst in result_data.get("instances", [])
                    if not _is_target(inst)
                ]
                result_data["instances"] = fresh_instances
                with open(result_path, "w") as f:
                    json.dump(result_data, f)
                print(f"[SegDelete]   segmentation_result.json: {len(fresh_instances)} instances remaining")

            # 2b) seg_broadcast.json is a snapshot of segmentation.json written
            # by the SAM3 worker; anything that replays it (viewer reload,
            # instance cleaner) RESURRECTS deleted instances unless it is
            # updated in lockstep.
            bc_path = output_dir / "seg_broadcast.json"
            if bc_path.exists():
                try:
                    with open(bc_path) as f:
                        bc = json.load(f)
                    n_before = len(bc.get("instances", []))
                    bc["instances"] = [
                        inst for inst in bc.get("instances", [])
                        if not (((inst.get("id") if inst.get("id") is not None
                                  else inst.get("instance_id")) == instance_id
                                 or _result_key(inst) in remove_keys)
                                and (target_label is None
                                     or inst.get("label", "") == target_label))
                    ]
                    with open(bc_path, "w") as f:
                        json.dump(bc, f)
                    print(f"[SegDelete]   seg_broadcast.json: "
                          f"{n_before - len(bc['instances'])} entries removed")
                except Exception as e:
                    print(f"[SegDelete]   seg_broadcast cleanup failed: {e}")

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

            # 4) Delete shape folder if it exists
            try:
                from segmentation.session_io import delete_shape_folder
                for iid in instance_ids_to_remove:
                    # Try to find label from the removed instances
                    for inst in seg_data_original_instances:
                        if inst.get("instance_id") == iid or inst.get("id") == iid:
                            delete_shape_folder(output_dir, inst.get("label", ""), iid)
                            break
            except Exception as e:
                print(f"[SegDelete]   Shape folder cleanup failed: {e}")

            # 5) Delete the per-instance TSDF crop folder — its GLB is what
            # the viewer auto-loads as the instance mesh/BBOX, so leaving it
            # keeps the "ghost" visible in the 3D scene after deletion.
            try:
                from segmentation.tsdf_export import delete_tsdf_folder
                for iid in instance_ids_to_remove:
                    for inst in seg_data_original_instances:
                        if inst.get("instance_id") == iid or inst.get("id") == iid:
                            delete_tsdf_folder(output_dir, inst.get("label", ""), iid)
                            break
            except Exception as e:
                print(f"[SegDelete]   TSDF folder cleanup failed: {e}")

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


# ── Shape progress tracking ─────────────────────────────────────
# Per-session, per-instance state for the Shape pipeline. Read by
# /api/segmentation/shape/progress/:session_id.
#
# Phases: pending → captioning → exporting_pkl → reconstructing → done | error
_shape_progress: Dict[str, Dict[int, Dict[str, Any]]] = {}
_shape_progress_lock = asyncio.Lock()

# In-flight dedup for /shape/export. Browsers retry the POST after a 30 s
# silence even though the server is still working — without dedup, we end up
# captioning + exporting PKLs twice in parallel, doubling GPU/CPU usage and
# producing the duplicate "[Shape] ━━━ /shape/export request ━━━" pattern in
# logs. Keyed by (session_id, sorted(instance_ids)). Mirrors the pattern in
# /api/segmentation/add_prompt.
_shape_in_flight: Dict[str, Any] = {
    "key": None, "event": None, "result": None, "completed_at": None,
}

# Global single-flight slot for the MeshFlow subprocess (inherited from the
# retired ShapeR path, where spawning a second batch OOM-killed the first on
# 2026-05-09). One generative batch at a time, regardless of session: the
# pipeline holds ~5 GB of weights + inference tensors on the GPU and a
# parallel spawn buys no throughput, only risk.
_meshflow_subprocess: Optional["asyncio.subprocess.Process"] = None
_meshflow_subprocess_lock = asyncio.Lock()
# Strong references to fire-and-forget background tasks. The event loop only keeps
# weak refs, so a bare asyncio.create_task() can be garbage-collected before it
# runs. Keep them alive here until they finish.
_bg_tasks: set = set()


def _shape_set(session_id: str, inst_id: int, **kw):
    """Synchronous helper — safe to call from worker threads."""
    state = _shape_progress.setdefault(session_id, {})
    entry = state.setdefault(int(inst_id), {"id": int(inst_id), "phase": "pending"})
    entry.update(kw)


def _shape_set_overall(session_id: str, **kw):
    state = _shape_progress.setdefault(session_id, {})
    overall = state.setdefault("__overall__", {})
    overall.update(kw)


async def _run_meshflow_subprocess(session_id: str, output_dir: Path,
                                   ply_paths: List[Path],
                                   inst_id_by_pkl: Dict[str, int]):
    """Spawn run_meshflow.sh and parse [BATCH] events for live UI progress.

    Single-flight globally (see _meshflow_subprocess above). Output GLBs are
    GENERATIVE visual assets: ``<stem>_visual.glb`` + ``metric: false`` in
    meta.json — never metric deliverables.
    """
    global _meshflow_subprocess

    if not ply_paths:
        print("[Shape] no segment PLYs to generate — skipping subprocess")
        _shape_set_overall(session_id, phase="done", reason="no_plys")
        return

    # Single-flight guard. We hold the lock only across the check + assignment
    # so the read loop below doesn't block other code paths.
    async with _meshflow_subprocess_lock:
        if (_meshflow_subprocess is not None
                and _meshflow_subprocess.returncode is None):
            existing_pid = _meshflow_subprocess.pid
            msg = (f"Another MeshFlow subprocess is already running "
                   f"(pid={existing_pid}). Refusing to spawn a second one — "
                   f"one generative batch at a time.")
            print(f"[Shape] ⚠ {msg}")
            _shape_set_overall(session_id, phase="error", error=msg)
            return

    server_dir = Path(__file__).resolve().parent
    script = server_dir / "run_meshflow.sh"

    mcfg = (cfg or {}).get("meshflow", {}) or {}
    cmd = [
        "bash", str(script),
        "--plys", *[str(p) for p in ply_paths],
        "--steps", str(mcfg.get("steps", 28)),
        "--guidance_scale", str(mcfg.get("guidance_scale", 2.5)),
        "--seed", str(mcfg.get("seed", 42)),
        "--dtype", str(mcfg.get("dtype", "fp16")),
    ]
    if mcfg.get("num_verts"):
        cmd += ["--num_verts", str(mcfg["num_verts"])]

    print(f"[Shape] ▶ starting MeshFlow generation of {len(ply_paths)} object(s)")
    print(f"[Shape]   PLYs: {[p.name for p in ply_paths]}")
    print(f"[Shape]   PLY→instance map: {inst_id_by_pkl}")
    print(f"[Shape]   cmd: {' '.join(cmd[:6])} ... (+{len(cmd)-6} args)")
    _shape_set_overall(session_id, phase="reconstructing",
                       total=len(ply_paths), done=0, started_at=time.time())

    try:
        proc = _track_worker(await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},  # inherit GPU visibility (A100 80GB); run_meshflow_batch.py auto-detects CUDA
            preexec_fn=_die_with_parent_sigkill,
        ))
        print(f"[Shape]   subprocess pid={proc.pid}")
    except Exception as e:
        print(f"[Shape] ❌ failed to spawn subprocess: {e}")
        import traceback; traceback.print_exc()
        _shape_set_overall(session_id, phase="error", error=str(e))
        return

    async with _meshflow_subprocess_lock:
        _meshflow_subprocess = proc

    done_count = 0
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode(errors="replace").rstrip()
        if not text:
            continue

        # Print to console (StreamCapture also forwards to WS)
        print(f"[Shape] {text}")

        # tqdm progress uses carriage returns; a [BATCH] line can arrive glued
        # after \r-fragments on the same buffered line — find it anywhere.
        if not text.startswith("[BATCH]"):
            seg = next((sg for sg in text.split("\r")
                        if sg.strip().startswith("[BATCH]")), None)
            if seg is None:
                continue
            text = seg.strip()

        # Parse "[BATCH] key=value key=value"
        body = text[len("[BATCH]"):].strip()
        kv = {}
        # Manual parser since values may contain spaces (e.g. msg=...)
        # Strategy: tokens that look like k=v split on the first '=' until we hit msg= or tb= which slurps the rest.
        i = 0
        tokens = body.split(" ")
        while i < len(tokens):
            tok = tokens[i]
            if "=" in tok:
                k, _, v = tok.partition("=")
                if k in ("msg", "tb"):
                    v = " ".join([v] + tokens[i+1:])
                    kv[k] = v
                    break
                kv[k] = v
            else:
                kv.setdefault("_event", tok)
            i += 1
        if "_event" not in kv:
            kv["_event"] = body.split(" ", 1)[0] if body else ""

        ev = kv.get("_event") or ""
        name = kv.get("name")
        inst_id = inst_id_by_pkl.get(name)

        if ev == "item" and inst_id is not None:
            status = kv.get("status", "")
            update = {"phase": status if status != "done" else "done"}
            if status == "starting":
                update["phase"] = "reconstructing"
                update["started_at"] = time.time()
            elif status == "done":
                update["elapsed"] = float(kv.get("elapsed", 0))
                update["mesh"] = kv.get("out", "")
                done_count += 1
                _shape_set_overall(session_id, done=done_count)
            elif status == "error":
                update["phase"] = "error"
                update["error"] = kv.get("msg", "unknown")
            _shape_set(session_id, inst_id, **update)

    rc = await proc.wait()
    # Release the single-flight slot so the next /shape/export can spawn.
    async with _meshflow_subprocess_lock:
        if _meshflow_subprocess is proc:
            _meshflow_subprocess = None
    overall_phase = "done" if rc == 0 else "error"
    _shape_set_overall(session_id, phase=overall_phase, finished_at=time.time(),
                       returncode=rc)
    icon = "✅" if rc == 0 else "❌"
    print(f"[Shape] {icon} generation finished rc={rc} "
          f"({done_count}/{len(ply_paths)} succeeded)")


@app.post("/api/segmentation/shape/export")
async def export_shape_inputs(request: Request):
    """Export per-instance segment PLYs and (optionally) chain MeshFlow
    generation. Replaces the retired ShapeR PKL exporter — MeshFlow consumes
    the segment geometry directly, so no captions/multi-view renders.

    Body:
        session_id: str
        instance_ids: Optional[list[int]] — filter to these IDs only
        auto_reconstruct: bool (default True) — chain MeshFlow inference
        (captions / auto_caption are accepted but ignored — deprecated with
        ShapeR; MeshFlow has no text conditioning)

    Routing: architectural classes and oversized segments are SKIPPED here
    (they belong to the metric surface_fit / TSDF paths) — see the response's
    "skipped" list. Outputs are GENERATIVE visual assets (metric: false).

    A browser retry after timeout will hit the in-flight guard below and wait
    for the original request's result instead of spawning a parallel run.
    """
    import threading as _threading
    body = await request.json()
    session_id = body.get("session_id")
    instance_ids = body.get("instance_ids")
    auto_reconstruct = body.get("auto_reconstruct", True)

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    # ── In-flight dedup ───────────────────────────────────────────────
    # Stable key: session + sorted instance ids. Two requests with the same
    # set are equivalent regardless of body order or auto_caption flag (the
    # captioning is part of the same job).
    inflight_key = (
        f"{session_id}:"
        f"{','.join(str(i) for i in sorted(instance_ids))}"
        if instance_ids else f"{session_id}:ALL"
    )
    if (inflight_key == _shape_in_flight["key"]
            and _shape_in_flight["event"] is not None):
        evt = _shape_in_flight["event"]
        loop = asyncio.get_event_loop()
        if not evt.is_set():
            # Original is still running — wait for it to finish.
            print(f"[Shape] ⏳ duplicate request for {inflight_key} — waiting for in-flight result")
            # 30 min cap — PKL export with 3 large objects can take ~25 min.
            await loop.run_in_executor(None, lambda: evt.wait(timeout=1800))
        # Either we just waited, or it had already completed when we arrived.
        # Return the cached response if it's both present and fresh — a stale
        # result from an hour-old run shouldn't satisfy a new explicit request.
        completed_at = _shape_in_flight.get("completed_at")
        fresh = completed_at is not None and (time.time() - completed_at) < 60
        if (evt.is_set()
                and _shape_in_flight.get("result") is not None
                and _shape_in_flight["key"] == inflight_key
                and fresh):
            age = time.time() - completed_at
            print(f"[Shape] ✅ returning cached result ({age:.0f}s old) for {inflight_key}")
            return _shape_in_flight["result"]
        print(f"[Shape] ⚠ in-flight wait expired without fresh result — falling through")

    # Claim the in-flight slot. The completion_event is signaled in `finally`
    # so any duplicate retry that arrived during processing wakes up and reads
    # the cached result.
    completion_event = _threading.Event()
    _shape_in_flight["key"] = inflight_key
    _shape_in_flight["event"] = completion_event
    _shape_in_flight["result"] = None
    _claimed_key = inflight_key

    try:
        ctx = _ctx(session_id)
        output_dir = ctx.output_dir
        frames_dir = ctx.frames_dir
        result_path = output_dir / "segmentation_result.json"

        if not result_path.exists():
            raise HTTPException(status_code=404,
                                detail="No segmentation_result.json — run propagation first")

        with open(result_path) as f:
            segments_result = json.load(f)

        print(f"[Shape] ━━━ /shape/export request ━━━")
        print(f"[Shape]   session={session_id}  instance_ids={instance_ids}")
        print(f"[Shape]   auto_reconstruct={auto_reconstruct}")

        # Reset state for the instances we're about to touch
        target_ids = set(instance_ids or [
            inst.get("id", inst.get("instance_id"))
            for inst in segments_result.get("instances", [])
        ])
        async with _shape_progress_lock:
            sess_state = _shape_progress.setdefault(session_id, {})
            for tid in target_ids:
                sess_state[int(tid)] = {"id": int(tid), "phase": "exporting_ply"}
            sess_state["__overall__"] = {
                "phase": "exporting_ply",
                "total": len(target_ids),
                "done": 0,
                "started_at": time.time(),
            }

        loop = asyncio.get_event_loop()
        mcfg = (cfg or {}).get("meshflow", {}) or {}

        def _export():
            from segmentation.mesh_export import export_segment_plys
            return export_segment_plys(
                output_dir=output_dir,
                segments_result=segments_result,
                obj_ids=instance_ids,
                max_extent_m=float(mcfg.get("max_extent_m", 15.0)),
                exclude_architectural=bool(mcfg.get("exclude_architectural", False)),
                frames_dir=frames_dir,
                require_ref_image=bool(mcfg.get("require_ref_image", True)),
            )

        try:
            exported, skipped = await loop.run_in_executor(None, _export)
        except Exception as e:
            import traceback
            traceback.print_exc()
            async with _shape_progress_lock:
                _shape_set_overall(session_id, phase="error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

        # Map folder name → instance_id (folder name = "<safe_label>_<id>";
        # the PLY stem equals the folder name, so the [BATCH] stdout matcher
        # resolves instances by stem directly).
        inst_by_folder = {}
        for inst in segments_result.get("instances", []):
            iid = inst.get("id", inst.get("instance_id"))
            label = inst.get("label", f"object_{iid}")
            safe = label.replace(" ", "_").replace("/", "_")[:30]
            inst_by_folder[f"{safe}_{iid}"] = int(iid)

        inst_id_by_stem = {}
        for p in exported:
            folder = p.parent.name
            if folder in inst_by_folder:
                iid = inst_by_folder[folder]
                inst_id_by_stem[p.stem] = iid
                async with _shape_progress_lock:
                    _shape_set(session_id, iid, phase="ply_ready",
                               ply=str(p.relative_to(output_dir)))
        # Skipped instances are terminal for this run — surface the reason
        async with _shape_progress_lock:
            for s in skipped:
                _shape_set(session_id, int(s["instance_id"]), phase="skipped",
                           reason=s["reason"])

        response = {
            "ok": True,
            "exported": [str(p.relative_to(output_dir)) for p in exported],
            "skipped": skipped,
            "count": len(exported),
            "reconstructing": False,
        }

        print(f"[Shape] PLY stage complete: {len(exported)} file(s), "
              f"{len(skipped)} skipped:")
        for s_ in skipped:
            print(f"[Shape]   skipped {s_['label']}_{s_['instance_id']}: {s_['reason']}")

        if auto_reconstruct and exported:
            print(f"[Shape] auto_reconstruct=True — scheduling background MeshFlow task", flush=True)
            async def _bg():
                print(f"[Shape] _bg ENTER — about to call _run_meshflow_subprocess", flush=True)
                try:
                    await _run_meshflow_subprocess(
                        session_id, output_dir, exported,
                        inst_id_by_stem,
                    )
                except Exception as e:
                    import traceback
                    print(f"[Shape] ❌ reconstruction crashed: {e}", flush=True)
                    traceback.print_exc()
                    async with _shape_progress_lock:
                        _shape_set_overall(session_id, phase="error", error=str(e))
            # Keep a STRONG reference: the loop only holds a weak ref to the task,
            # so a bare create_task() can be GC'd before it runs. Also surface any
            # exception that kills the task before it can log it itself.
            t = asyncio.create_task(_bg())
            _bg_tasks.add(t)
            def _on_bg_done(_t):
                _bg_tasks.discard(_t)
                if not _t.cancelled() and _t.exception() is not None:
                    print(f"[Shape] ❌ _bg task died before logging: {_t.exception()!r}", flush=True)
            t.add_done_callback(_on_bg_done)
            response["reconstructing"] = True
        elif not auto_reconstruct:
            print(f"[Shape] auto_reconstruct=False — stopping after PLY export")
            async with _shape_progress_lock:
                _shape_set_overall(session_id, phase="done")
                for p in exported:
                    folder = p.parent.name
                    if folder in inst_by_folder:
                        _shape_set(session_id, inst_by_folder[folder], phase="ply_ready")

        # Stash result + timestamp BEFORE the finally so the cached result is
        # visible the moment the event fires. Keep key/event/result populated
        # after completion: waiters that wake up after `event.set()` need to
        # read them, and a same-key retry within 60 s is a browser duplicate
        # that should also get the cached response. The slot is overwritten
        # by the next request with a different key, or aged out via the 60 s
        # freshness check at the top of the handler.
        _shape_in_flight["result"] = response
        _shape_in_flight["completed_at"] = time.time()
        return response

    finally:
        # Always wake up waiters. Do NOT clear key/event — that would make the
        # waiters' freshness check fail. Stale entries are filtered by the
        # `< 60s` freshness gate, not by clearing.
        completion_event.set()


@app.get("/api/segmentation/shape/progress/{session_id}")
async def shape_progress(session_id: str):
    """Live state for the shape pipeline. UI polls this every ~1s during a run."""
    async with _shape_progress_lock:
        state = _shape_progress.get(session_id, {})
        # Shallow copy to avoid mutation under our feet
        per_instance = [v for k, v in state.items() if k != "__overall__"]
        overall = state.get("__overall__", {"phase": "idle"})
    return {"ok": True, "overall": overall, "instances": per_instance}


@app.get("/api/segmentation/shape/status/{session_id}")
async def shape_status(session_id: str):
    """Check which segmented instances already have Shape PKLs and/or meshes."""
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    result_path = output_dir / "segmentation_result.json"
    shape_dir = output_dir / "shape"

    if not result_path.exists():
        return {"ok": True, "instances": []}

    with open(result_path) as f:
        result_data = json.load(f)

    statuses = []
    for inst in result_data.get("instances", []):
        iid = inst.get("id", inst.get("instance_id"))
        label = inst.get("label", f"object_{iid}")
        safe_label = label.replace(" ", "_").replace("/", "_")[:30]
        obj_folder = shape_dir / f"{safe_label}_{iid}"

        # "has_pkl" key kept for UI compat; it now means "has exported input"
        # (segment .ply for MeshFlow; legacy .pkl folders also count)
        has_input = (any(obj_folder.glob("*.ply")) or any(obj_folder.glob("*.pkl"))) \
            if obj_folder.exists() else False
        has_mesh = any(obj_folder.glob("*.glb")) if obj_folder.exists() else False

        statuses.append({
            "id": iid,
            "label": label,
            "has_pkl": has_input,
            "has_mesh": has_mesh,
            "folder": str(obj_folder.relative_to(output_dir)) if obj_folder.exists() else None,
        })

    return {"ok": True, "instances": statuses}


@app.get("/api/segmentation/shape/list/{session_id}")
async def shape_list(session_id: str):
    """List all reconstructed meshes for a session (for viewer auto-load).

    Each entry includes the URL to fetch the .glb plus the metadata sidecar
    so the viewer can know placement, label, ICP residual, etc. without
    a second roundtrip per object.
    """
    ctx = _ctx(session_id)
    shape_dir = ctx.output_dir / "shape"
    if not shape_dir.exists():
        return {"ok": True, "shapes": []}

    shapes = []
    for obj_folder in sorted(shape_dir.iterdir()):
        if not obj_folder.is_dir():
            continue
        glb_files = sorted(obj_folder.glob("*.glb"))
        if not glb_files:
            continue
        # prefer the MeshFlow output (<folder>_visual.glb) over legacy names
        glb_file = next((g for g in glb_files if g.stem.endswith("_visual")),
                        glb_files[0])
        # sidecar: legacy <glb>.meta.json, else the folder-level meta.json
        meta = None
        for meta_file in (glb_file.with_suffix(".meta.json"),
                          obj_folder / "meta.json"):
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    break
                except Exception as e:
                    print(f"[ShapeList] failed to read {meta_file}: {e}")

        shapes.append({
            "folder": obj_folder.name,
            "glb_url": f"/api/segmentation/shape/file/{session_id}/{obj_folder.name}/{glb_file.name}",
            "meta": meta,
        })

    return {"ok": True, "shapes": shapes}


@app.get("/api/segmentation/shape/file/{session_id}/{folder}/{filename}")
async def shape_file(session_id: str, folder: str, filename: str):
    """Serve a .glb (or its .meta.json) for the given session/object."""
    from fastapi.responses import FileResponse
    # Sanitize: prevent path traversal — folder + filename must be plain names
    if "/" in folder or ".." in folder or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path component")

    ctx = _ctx(session_id)
    full_path = ctx.output_dir / "shape" / folder / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        str(full_path),
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},  # may be regenerated
    )


# ── surface_fit (measurement-backed smooth surfaces; ShapeR replacement for
#    architectural classes) ────────────────────────────────────────────────
# Fits the lowest-DOF smooth model per SAM3 segment (plane → quadric → swept
# profile → B-spline), regularizes planes scene-wide, and keeps the residuals
# as deliverable (deviation PLY + heatmap PNG + stats JSON). Runs in a
# SEPARATE PROCESS (run_surface_fit.py, same da3 env as the server) with the
# same single-flight discipline as ShapeR: one run at a time, any session.

_surface_fit_progress: Dict[str, Dict[str, Any]] = {}
_surface_fit_progress_lock = asyncio.Lock()
_surface_fit_subprocess: Optional["asyncio.subprocess.Process"] = None
_surface_fit_lock = asyncio.Lock()


@app.post("/api/segmentation/surface_fit/run")
async def surface_fit_run(request: Request):
    """Launch surface_fit for a session. Body: {session_id, instance_ids?: [int],
    params?: {fit_segment kwargs}}. Without instance_ids fits every
    architectural instance + scene regularization (hybrid mode)."""
    global _surface_fit_subprocess
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    instance_ids = body.get("instance_ids") or None
    params = body.get("params") or {}

    ctx = _ctx(session_id)
    session_dir = ctx.frames_dir.parent
    if not (ctx.output_dir / "segmentation_result.json").exists():
        raise HTTPException(status_code=409,
                            detail="No segmentation_result.json — run SAM3 first")
    if not (ctx.output_dir / "cleaned_cloud.ply").exists():
        raise HTTPException(status_code=409,
                            detail="No cleaned_cloud.ply — run CloudComPy first")

    async with _surface_fit_lock:
        if (_surface_fit_subprocess is not None
                and _surface_fit_subprocess.returncode is None):
            raise HTTPException(status_code=429,
                                detail="A surface_fit run is already in progress")

    worker = Path(SERVER_DIR) / "run_surface_fit.py"
    cmd = [sys.executable, str(worker), "--session-dir", str(session_dir),
           "--params", json.dumps(params)]
    if instance_ids:
        for iid in instance_ids:
            cmd += ["--instance-id", str(int(iid))]
    else:
        cmd.append("--all")

    async with _surface_fit_progress_lock:
        _surface_fit_progress[session_id] = {
            "overall": {"phase": "starting", "started_at": time.time()},
            "instances": {},
        }
    print(f"[SurfaceFit] ▶ run session={session_id} "
          f"instances={'ALL' if not instance_ids else instance_ids}")

    async def _bg():
        global _surface_fit_subprocess
        result = None
        try:
            proc = _track_worker(await asyncio.create_subprocess_exec(
                *cmd, cwd=str(SERVER_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                preexec_fn=_die_with_parent_sigkill,
            ))
            async with _surface_fit_lock:
                _surface_fit_subprocess = proc
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "ignore").rstrip()
                if not line:
                    continue
                if line.startswith("[SFIT-PROGRESS]"):
                    try:
                        upd = json.loads(line[len("[SFIT-PROGRESS]"):])
                        async with _surface_fit_progress_lock:
                            st = _surface_fit_progress.setdefault(session_id, {
                                "overall": {}, "instances": {}})
                            iid = upd.get("instance_id")
                            if iid is not None:
                                st["instances"][str(iid)] = upd
                            st["overall"]["phase"] = upd.get("phase", "running")
                            st["overall"]["elapsed"] = upd.get("elapsed")
                    except Exception:
                        pass
                elif line.startswith("[SFIT-RESULT]"):
                    try:
                        result = json.loads(line[len("[SFIT-RESULT]"):])
                    except Exception:
                        result = None
                else:
                    try:
                        print(f"[SurfaceFit] {line}", flush=True)
                    except (BrokenPipeError, OSError):
                        pass
            await proc.wait()
            ok = bool(result and result.get("ok"))
            async with _surface_fit_progress_lock:
                _surface_fit_progress.setdefault(session_id, {})["overall"] = {
                    "phase": "done" if ok else "error",
                    "result": result,
                    "finished_at": time.time(),
                }
            print(f"[SurfaceFit] {'✅ done' if ok else '❌ failed'}: "
                  f"{len((result or {}).get('ok', []))} fitted")
        except Exception as e:
            import traceback as _tb
            _tb.print_exc()
            async with _surface_fit_progress_lock:
                _surface_fit_progress.setdefault(session_id, {})["overall"] = {
                    "phase": "error", "error": str(e), "finished_at": time.time(),
                }
        finally:
            async with _surface_fit_lock:
                _surface_fit_subprocess = None

    task = asyncio.create_task(_bg())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return {"ok": True, "started": True,
            "mode": "scene" if not instance_ids else "instances"}


@app.get("/api/segmentation/surface_fit/progress/{session_id}")
async def surface_fit_progress(session_id: str):
    async with _surface_fit_progress_lock:
        st = _surface_fit_progress.get(session_id,
                                       {"overall": {"phase": "idle"}, "instances": {}})
        return {"ok": True, "overall": dict(st.get("overall", {})),
                "instances": list(st.get("instances", {}).values())}


@app.get("/api/segmentation/surface_fit/list/{session_id}")
async def surface_fit_list(session_id: str):
    """Fitted surfaces of a session (viewer auto-load, parity with shape/list)."""
    ctx = _ctx(session_id)
    base = ctx.output_dir / "surface_fit"
    out = []
    if base.exists():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            meta_p = d / "meta.json"
            if not meta_p.exists():
                continue
            try:
                meta = json.loads(meta_p.read_text())
            except Exception:
                continue
            entry = {"folder": d.name, "meta": meta}
            for fname, key in (("surface.glb", "glb_url"),
                               ("surface.ply", "ply_url"),
                               ("deviation.ply", "deviation_url"),
                               ("heatmap.png", "heatmap_url"),
                               ("residuals.json", "residuals_url")):
                if (d / fname).exists():
                    entry[key] = (f"/api/segmentation/surface_fit/file/"
                                  f"{session_id}/{d.name}/{fname}")
            out.append(entry)
    scene_rep = base / "scene_report.json"
    report = None
    if scene_rep.exists():
        try:
            report = json.loads(scene_rep.read_text())
        except Exception:
            report = None
    return {"ok": True, "surfaces": out, "scene_report": report}


@app.get("/api/segmentation/surface_fit/file/{session_id}/{folder}/{filename}")
async def surface_fit_file(session_id: str, folder: str, filename: str):
    """Serve a surface_fit artifact (GLB/PLY/PNG/JSON)."""
    from fastapi.responses import FileResponse
    if "/" in folder or ".." in folder or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path component")
    ctx = _ctx(session_id)
    full_path = ctx.output_dir / "surface_fit" / folder / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    media = {"glb": "model/gltf-binary", "json": "application/json",
             "png": "image/png", "ply": "application/octet-stream"}.get(
        filename.rsplit(".", 1)[-1], "application/octet-stream")
    return FileResponse(str(full_path), media_type=media,
                        headers={"Cache-Control": "no-cache"})


# ── Reconstruction v2 (semantic-geometric, neighbor-aware) ─────────────
# Additive to /shape/export: classifies every instance, reconstructs parametric
# surfaces/swept solids/boxes/linear-repeats directly (no subprocess) and uses
# the ShapeR GLBs only for genuinely free-form blobs, then assembles a coherent
# scene (structural shell, opening detection, object clipping, snapping). Output:
# output/scene/<label>_<id>.glb + .meta.json and output/scene/scene.json.

# ── Reconstruction-v2 progress + single-flight ──────────────────────
# run_reconstruction loads the full point cloud, builds Open3D raycasting
# scenes and rasterises silhouettes per camera. One run is heavy; two in
# parallel OOM-killed the host (2026-05-12 — two `[ReconV2] start` lines in
# the log, then a hard reboot). So: GLOBAL single-flight (one recon at a time,
# any session), run as a background task, expose progress for the UI to poll.
_recon_state: Dict[str, Dict[str, Any]] = {}      # session_id -> {phase, ...}
_recon_lock = asyncio.Lock()                       # guards the running-task slot
_recon_running: Dict[str, Any] = {"session_id": None, "task": None, "started_at": None}


def _recon_set(session_id: str, **kw):
    """Synchronous, GIL-safe — callable from the worker thread (no asyncio lock)."""
    st = _recon_state.setdefault(session_id, {"phase": "idle"})
    st.update(kw)


async def _run_reconstruction_job(session_id: str, ctx, only_ids, use_shaper: bool,
                                  max_views: int):
    loop = asyncio.get_event_loop()

    def _cb(d: dict):
        _recon_set(session_id, **d)

    def _run():
        from reconstruction_runner import run_reconstruction
        return run_reconstruction(session_id, ctx.output_dir, ctx.frames_dir,
                                  session_dir=ctx.session_dir, only_obj_ids=only_ids,
                                  use_shaper_glbs=use_shaper, max_views=max_views,
                                  progress_cb=_cb)
    try:
        summary = await loop.run_in_executor(None, _run)
        if summary.get("ok"):
            _recon_set(session_id, phase="done", finished_at=time.time(), summary=summary,
                       n_elements=summary.get("n_elements"), by_class=summary.get("by_class"),
                       n_adjacency=summary.get("n_adjacency"), elapsed_s=summary.get("elapsed_s"),
                       error=None)
        else:
            _recon_set(session_id, phase="error", finished_at=time.time(), summary=summary,
                       error=summary.get("error", "reconstruction failed"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        _recon_set(session_id, phase="error", finished_at=time.time(), error=str(e))
    finally:
        async with _recon_lock:
            if _recon_running.get("session_id") == session_id:
                _recon_running.update({"session_id": None, "task": None, "started_at": None})


@app.post("/api/segmentation/reconstruct/{session_id}")
async def reconstruct_v2(session_id: str, request: Request):
    """Kick off reconstruction v2 as a background job. Returns immediately;
    poll GET /api/segmentation/reconstruct/progress/{session_id} for status.

    Single-flight globally — two parallel runs OOM the CPU-only host.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    ctx = _ctx(session_id)
    if not (ctx.output_dir / "segmentation_result.json").exists():
        raise HTTPException(status_code=400, detail="No segmentation_result.json — run segmentation first")
    only_ids = body.get("instance_ids")
    use_shaper = bool(body.get("use_shaper_glbs", True))
    try:
        max_views = int(body.get("max_views") or 150)
    except (TypeError, ValueError):
        max_views = 150
    max_views = max(1, min(max_views, 500))

    async with _recon_lock:
        cur = _recon_running.get("task")
        if cur is not None and not cur.done():
            other = _recon_running.get("session_id")
            raise HTTPException(
                status_code=409,
                detail=(f"A reconstruction is already running (session={other}). "
                        f"Wait for it to finish — running two at once OOMs the host."))
        _recon_set(session_id, phase="starting", started_at=time.time(), finished_at=None,
                   error=None, summary=None, n_elements=None, by_class=None, n_views=None,
                   n_points=None, done=0, total=None, current=None)
        task = asyncio.create_task(
            _run_reconstruction_job(session_id, ctx, only_ids, use_shaper, max_views))
        _recon_running.update({"session_id": session_id, "task": task, "started_at": time.time()})
    return {"ok": True, "started": True, "session_id": session_id}


@app.get("/api/segmentation/reconstruct/progress/{session_id}")
async def reconstruct_progress(session_id: str):
    """Live state for a reconstruction-v2 job. The UI polls this every ~1.5 s."""
    st = dict(_recon_state.get(session_id, {"phase": "idle"}))
    busy_session = _recon_running.get("session_id")
    st["busy"] = busy_session is not None
    st["busy_session"] = busy_session
    return {"ok": True, "session_id": session_id, **st}


def _load_scene_payload(output_dir: Path, session_id: str) -> Optional[Dict[str, Any]]:
    """Read ``output/scene/scene.json``, resolve a ``glb_url`` per element, return
    the payload the UI consumes. ``None`` if scene.json is missing or unreadable
    (a bad JSON is logged but treated as "no scene" so a load can still succeed)."""
    scene_path = Path(output_dir) / "scene" / "scene.json"
    if not scene_path.exists():
        return None
    try:
        scene = json.loads(scene_path.read_text())
        from reconstruction.elements import json_safe
        scene = json_safe(scene)
    except Exception as e:
        print(f"[Scene] bad scene.json for {session_id}: {e}")
        return None
    scene_dir = Path(output_dir) / "scene"
    for el in scene.get("elements", []):
        iid = el.get("instance_id")
        label = str(el.get("label", f"object_{iid}")).replace(" ", "_").replace("/", "_")[:30]
        folder = f"{label}_{iid}"
        glb = scene_dir / f"{folder}.glb"
        el["glb_url"] = (f"/api/segmentation/scene/file/{session_id}/{folder}.glb"
                         if glb.exists() else None)
    return {"session_id": scene.get("session_id", session_id),
            "elements": scene.get("elements", []),
            "adjacency": scene.get("adjacency", [])}


@app.get("/api/segmentation/scene/{session_id}")
async def get_scene(session_id: str):
    """Return the assembled scene (output/scene/scene.json) + GLB URLs per element."""
    ctx = _ctx(session_id)
    payload = _load_scene_payload(ctx.output_dir, session_id)
    if payload is None:
        return {"ok": True, "exists": False, "elements": [], "adjacency": []}
    return {"ok": True, "exists": True, **payload}


@app.get("/api/segmentation/scene/file/{session_id}/{filename}")
async def scene_file(session_id: str, filename: str):
    """Serve a reconstruction-v2 element GLB / meta.json."""
    from fastapi.responses import FileResponse
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path component")
    ctx = _ctx(session_id)
    full_path = ctx.output_dir / "scene" / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    media_type = ("model/gltf-binary" if filename.endswith(".glb")
                  else "application/json" if filename.endswith(".json")
                  else "application/octet-stream")
    return FileResponse(str(full_path), media_type=media_type,
                        headers={"Cache-Control": "no-cache"})


# ── TSDF reconstruction endpoints ───────────────────────────────────
# Sibling pipeline to ShapeR (above) — same I/O contract per session, output
# folder is ``output/tsdf/`` instead of ``output/shape/``. Runs in-process
# (Open3D, no torch model) so no subprocess shell-out is needed.

_tsdf_progress: Dict[str, Dict[int, Dict[str, Any]]] = {}
_tsdf_progress_lock = asyncio.Lock()


def _tsdf_set(session_id: str, inst_id: int, **kw):
    state = _tsdf_progress.setdefault(session_id, {})
    entry = state.setdefault(int(inst_id), {"id": int(inst_id), "phase": "pending"})
    entry.update(kw)


def _tsdf_set_overall(session_id: str, **kw):
    state = _tsdf_progress.setdefault(session_id, {})
    overall = state.setdefault("__overall__", {})
    overall.update(kw)


@app.post("/api/segmentation/tsdf/export")
async def export_tsdf_endpoint(request: Request):
    """Reconstruct TSDF meshes for selected instances.

    Body:
        session_id: str
        instance_ids: Optional[list[int]]
        voxel_length: float (m, default 0.015)
        sdf_trunc:    float (m, default 0.04)
        depth_trunc:  float (m, default 5.0)
        dilate_radius: int (px, default 3)
    """
    body = await request.json()
    session_id = body.get("session_id")
    instance_ids = body.get("instance_ids")
    voxel_length = float(body.get("voxel_length", 0.015))
    sdf_trunc = float(body.get("sdf_trunc", 0.04))
    depth_trunc = float(body.get("depth_trunc", 5.0))
    dilate_radius = int(body.get("dilate_radius", 3))

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    frames_dir = ctx.frames_dir
    result_path = output_dir / "segmentation_result.json"

    if not result_path.exists():
        raise HTTPException(status_code=404,
                            detail="No segmentation_result.json — run propagation first")

    with open(result_path) as f:
        segments_result = json.load(f)

    target_ids = set(instance_ids or [
        inst.get("id", inst.get("instance_id"))
        for inst in segments_result.get("instances", [])
    ])

    print(f"[TSDF] ━━━ /tsdf/export request ━━━")
    print(f"[TSDF]   session={session_id}  instance_ids={sorted(target_ids)}")
    print(f"[TSDF]   voxel={voxel_length}m  trunc={sdf_trunc}m  "
          f"depth_max={depth_trunc}m  dilate={dilate_radius}px")

    async with _tsdf_progress_lock:
        sess_state = _tsdf_progress.setdefault(session_id, {})
        for tid in target_ids:
            sess_state[int(tid)] = {"id": int(tid), "phase": "pending"}
        sess_state["__overall__"] = {
            "phase": "integrating",
            "total": len(target_ids),
            "done": 0,
            "started_at": time.time(),
        }

    loop = asyncio.get_event_loop()
    done_counter = {"n": 0}

    def _progress_cb(inst_id: int, phase: str, elapsed, mesh):
        # Called from worker thread → fire-and-forget into the loop.
        update: Dict[str, Any] = {"phase": phase}
        if elapsed is not None:
            update["elapsed"] = float(elapsed)
        if mesh:
            update["mesh"] = mesh
        if phase == "done":
            done_counter["n"] += 1
        try:
            asyncio.run_coroutine_threadsafe(
                _tsdf_apply_progress(session_id, inst_id, update,
                                     done_counter["n"] if phase == "done" else None),
                loop,
            )
        except Exception:
            pass  # progress is best-effort

    def _run_tsdf():
        from segmentation.tsdf_export import (
            export_tsdf_meshes,
            crop_scene_mesh_to_instances,
        )
        # Prefer carving each instance out of the already-reconstructed scene mesh
        # (tsdf/scene/scene.glb). It is cloud-consistent and works for EVERY
        # backend — including those whose per-frame depth lives under a run dir the
        # per-object re-integration doesn't recognise (e.g. VGGTOMEGA → omega_run/),
        # which is exactly what made the per-object TSDF report "no depth source".
        scene_dir = output_dir / "tsdf" / "scene"
        has_scene = ((scene_dir / "scene.glb.orig").exists()
                     or (scene_dir / "scene.glb").exists())
        if has_scene:
            print("[TSDF] scene mesh present → carving instances from it (crop)")
            return crop_scene_mesh_to_instances(
                output_dir=output_dir,
                segments_result=segments_result,
                obj_ids=list(target_ids) or None,
                progress_cb=_progress_cb,
            )
        print("[TSDF] no scene mesh → per-object depth re-integration (legacy)")
        return export_tsdf_meshes(
            output_dir=output_dir,
            frames_dir=frames_dir,
            segments_result=segments_result,
            session_dir=frames_dir.parent,
            obj_ids=list(target_ids) or None,
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
            dilate_radius=dilate_radius,
            progress_cb=_progress_cb,
        )

    async def _bg():
        try:
            written = await loop.run_in_executor(None, _run_tsdf)
            async with _tsdf_progress_lock:
                _tsdf_set_overall(session_id, phase="done",
                                  done=len(written), finished_at=time.time())
            print(f"[TSDF] ✅ wrote {len(written)} mesh(es)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            async with _tsdf_progress_lock:
                _tsdf_set_overall(session_id, phase="error", error=str(e))

    asyncio.create_task(_bg())

    return {"ok": True, "started": True, "count": len(target_ids)}


async def _tsdf_apply_progress(session_id: str, inst_id: int,
                               update: Dict[str, Any],
                               done_count: Optional[int]):
    async with _tsdf_progress_lock:
        _tsdf_set(session_id, inst_id, **update)
        if done_count is not None:
            _tsdf_set_overall(session_id, done=done_count)


# ── Whole-scene TSDF (single mesh from all frames; no instance filtering) ──

@app.post("/api/segmentation/tsdf/scene_export")
async def export_tsdf_scene_endpoint(request: Request):
    """Reconstruct ONE TSDF mesh from the entire scan (all poses + depth).

    Body:
        session_id: str
        voxel_length: float (m, default 0.015)
        sdf_trunc:    float (m, default 0.05)
        depth_trunc:  float (m, default 5.0)
        depth_min:    float (m, default 0.15)
        edge_thresh:  float (m, default 0.08) — discontinuity drop threshold

    Progress is tracked under the ``__scene__`` key of the same TSDF state
    dict, so a single progress endpoint can surface both per-instance and
    whole-scene jobs.
    """
    body = await request.json()
    session_id = body.get("session_id")
    # Defaults come from config.yaml `tsdf:` so the manual TSDF button honours the
    # SAME tuning as the pipeline worker. Previously this endpoint used hardcoded
    # defaults (depth_trunc=5m, da3_conf_percentile=50, no mask/weight passthrough)
    # and ignored config entirely → far walls truncated at 5 m + half the depth
    # dropped by confidence → "the TSDF doesn't cover the whole cloud". Body keys
    # still override per-request.
    # Single source of truth (shared with workers/tsdf_worker.py): the manual
    # /tsdf/scene_export button forwards the SAME config keys to export_tsdf_scene
    # as the in-pipeline TSDF stage, so the two reconstruct identically. Per-request
    # body keys override config. Derived from export_tsdf_scene's signature → cannot
    # drift when a kwarg is added/removed.
    from segmentation.tsdf_export import build_tsdf_scene_kwargs
    params = build_tsdf_scene_kwargs(cfg, overrides=body)

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    frames_dir = ctx.frames_dir

    print(f"[TSDF-scene] ━━━ /tsdf/scene_export request ━━━")
    print(f"[TSDF-scene]   session={session_id}")
    print(f"[TSDF-scene]   params (config tsdf + body overrides): {params}")

    async with _tsdf_progress_lock:
        sess_state = _tsdf_progress.setdefault(session_id, {})
        sess_state["__scene__"] = {
            "phase": "starting",
            "started_at": time.time(),
        }

    # Run the heavy GPU/Open3D work in a SEPARATE PROCESS (not run_in_executor):
    # a thread still saturates this process + the GIL and blocks the event loop,
    # so /health times out and the UI falsely reports "server down" during a
    # reconstruction. A subprocess read with async streams keeps the loop free.
    # params already built above from config.yaml `tsdf:` + body overrides
    worker = Path(SERVER_DIR) / "run_tsdf_scene.py"

    async def _bg():
        result_path = None
        try:
            proc = _track_worker(await asyncio.create_subprocess_exec(
                sys.executable, str(worker),
                "--output-dir", str(output_dir),
                "--frames-dir", str(frames_dir),
                "--session-dir", str(frames_dir.parent),
                "--params", json.dumps(params),
                cwd=str(SERVER_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"},
                preexec_fn=_die_with_parent_sigkill,
            ))
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "ignore").rstrip()
                if not line:
                    continue
                if line.startswith("[TSDF-PROGRESS]"):
                    try:
                        upd = json.loads(line[len("[TSDF-PROGRESS]"):])
                        await _tsdf_scene_apply_progress(session_id, upd)
                        print(f"[TSDF-scene] phase={upd.get('phase')}"
                              + (f" {upd.get('elapsed'):.0f}s" if upd.get('elapsed') else ""),
                              flush=True)  # store phase transitions in the server log
                    except Exception:
                        pass
                elif line.startswith("[TSDF-RESULT]"):
                    r = line[len("[TSDF-RESULT]"):].strip()
                    result_path = None if r == "NONE" else r
                else:
                    try:
                        print(line, flush=True)  # forward worker logs to server console
                    except (BrokenPipeError, OSError):
                        pass  # dead stdout must NOT abort the task or mark the scene "error"
            await proc.wait()
            async with _tsdf_progress_lock:
                _tsdf_progress.setdefault(session_id, {})["__scene__"] = {
                    "phase": "done" if result_path else "error",
                    "mesh": result_path,
                    "finished_at": time.time(),
                }
            print(f"[TSDF-scene] ✅ wrote {result_path}" if result_path
                  else "[TSDF-scene] ❌ failed")
        except Exception as e:
            import traceback
            traceback.print_exc()
            async with _tsdf_progress_lock:
                _tsdf_progress.setdefault(session_id, {})["__scene__"] = {
                    "phase": "error", "error": str(e),
                }

    asyncio.create_task(_bg())

    return {"ok": True, "started": True}


async def _tsdf_scene_apply_progress(session_id: str, update: Dict[str, Any]):
    async with _tsdf_progress_lock:
        state = _tsdf_progress.setdefault(session_id, {})
        scene = state.setdefault("__scene__", {})
        scene.update(update)


# ── Whole-scene Poisson (Option B — surface straight from the cleaned cloud) ──

@app.post("/api/segmentation/poisson/scene_export")
async def export_poisson_scene_endpoint(request: Request):
    """Reconstruct ONE screened-Poisson mesh from the cleaned cloud.

    Sibling of /tsdf/scene_export: meshes the ALREADY-cleaned point cloud directly
    (denser than the neural depth the TSDF integrates) and writes
    output/tsdf/scene_poisson/scene_poisson.glb — which surfaces as its own
    toggleable row next to the TSDF mesh. Progress lives under the
    ``__poisson_scene__`` state key and is exposed as ``poisson`` on /tsdf/progress.

    Body: session_id (+ optional per-request overrides of config.yaml `poisson:`).
    """
    body = await request.json()
    session_id = body.get("session_id")
    # Config-driven, same pattern as the TSDF endpoint: defaults from config.yaml
    # `poisson:` (so this button honours the project tuning), body keys override.
    from segmentation.tsdf_export import build_poisson_scene_kwargs
    params = build_poisson_scene_kwargs(cfg, overrides=body)

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    frames_dir = ctx.frames_dir

    print(f"[Poisson-scene] ━━━ /poisson/scene_export request ━━━")
    print(f"[Poisson-scene]   session={session_id}")
    print(f"[Poisson-scene]   params (config poisson + body overrides): {params}")

    async with _tsdf_progress_lock:
        sess_state = _tsdf_progress.setdefault(session_id, {})
        sess_state["__poisson_scene__"] = {
            "phase": "starting",
            "started_at": time.time(),
        }

    worker = Path(SERVER_DIR) / "run_poisson_scene.py"

    async def _bg():
        result_path = None
        try:
            proc = _track_worker(await asyncio.create_subprocess_exec(
                sys.executable, str(worker),
                "--output-dir", str(output_dir),
                "--frames-dir", str(frames_dir),
                "--session-dir", str(frames_dir.parent),
                "--params", json.dumps(params),
                cwd=str(SERVER_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1"},
                preexec_fn=_die_with_parent_sigkill,
            ))
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "ignore").rstrip()
                if not line:
                    continue
                if line.startswith("[TSDF-PROGRESS]"):
                    try:
                        upd = json.loads(line[len("[TSDF-PROGRESS]"):])
                        await _poisson_scene_apply_progress(session_id, upd)
                        print(f"[Poisson-scene] phase={upd.get('phase')}"
                              + (f" {upd.get('elapsed'):.0f}s" if upd.get('elapsed') else ""),
                              flush=True)
                    except Exception:
                        pass
                elif line.startswith("[TSDF-RESULT]"):
                    r = line[len("[TSDF-RESULT]"):].strip()
                    result_path = None if r == "NONE" else r
                else:
                    try:
                        print(line, flush=True)  # forward worker logs to server console
                    except (BrokenPipeError, OSError):
                        pass  # dead stdout must NOT abort the task or mark the scene "error"
            await proc.wait()
            async with _tsdf_progress_lock:
                _tsdf_progress.setdefault(session_id, {})["__poisson_scene__"] = {
                    "phase": "done" if result_path else "error",
                    "mesh": result_path,
                    "finished_at": time.time(),
                }
            print(f"[Poisson-scene] ✅ wrote {result_path}" if result_path
                  else "[Poisson-scene] ❌ failed")
        except Exception as e:
            import traceback
            traceback.print_exc()
            async with _tsdf_progress_lock:
                _tsdf_progress.setdefault(session_id, {})["__poisson_scene__"] = {
                    "phase": "error", "error": str(e),
                }

    asyncio.create_task(_bg())

    return {"ok": True, "started": True}


async def _poisson_scene_apply_progress(session_id: str, update: Dict[str, Any]):
    async with _tsdf_progress_lock:
        state = _tsdf_progress.setdefault(session_id, {})
        scene = state.setdefault("__poisson_scene__", {})
        scene.update(update)


@app.get("/api/segmentation/tsdf/progress/{session_id}")
async def tsdf_progress(session_id: str):
    """Live state for the TSDF pipeline. UI polls this every ~1s during a run."""
    async with _tsdf_progress_lock:
        state = _tsdf_progress.get(session_id, {})
        per_instance = [
            v for k, v in state.items()
            if k not in ("__overall__", "__scene__", "__poisson_scene__")
        ]
        overall = state.get("__overall__", {"phase": "idle"})
        scene = state.get("__scene__", {"phase": "idle"})
        poisson = state.get("__poisson_scene__", {"phase": "idle"})
    return {"ok": True, "overall": overall, "instances": per_instance,
            "scene": scene, "poisson": poisson}


@app.get("/api/segmentation/tsdf/status/{session_id}")
async def tsdf_status(session_id: str):
    """Check which segmented instances already have TSDF meshes."""
    ctx = _ctx(session_id)
    output_dir = ctx.output_dir
    result_path = output_dir / "segmentation_result.json"
    tsdf_dir = output_dir / "tsdf"

    if not result_path.exists():
        return {"ok": True, "instances": []}

    with open(result_path) as f:
        result_data = json.load(f)

    statuses = []
    for inst in result_data.get("instances", []):
        iid = inst.get("id", inst.get("instance_id"))
        label = inst.get("label", f"object_{iid}")
        safe_label = label.replace(" ", "_").replace("/", "_")[:30]
        obj_folder = tsdf_dir / f"{safe_label}_{iid}"
        has_mesh = any(obj_folder.glob("*.glb")) if obj_folder.exists() else False
        statuses.append({
            "id": iid,
            "label": label,
            "has_mesh": has_mesh,
            "folder": str(obj_folder.relative_to(output_dir)) if obj_folder.exists() else None,
        })
    return {"ok": True, "instances": statuses}


@app.get("/api/segmentation/tsdf/list/{session_id}")
async def tsdf_list(session_id: str):
    """List all TSDF meshes for a session (viewer auto-load)."""
    ctx = _ctx(session_id)
    tsdf_dir = ctx.output_dir / "tsdf"
    if not tsdf_dir.exists():
        return {"ok": True, "shapes": []}

    shapes = []
    for obj_folder in sorted(tsdf_dir.iterdir()):
        if not obj_folder.is_dir():
            continue
        glb_files = sorted(obj_folder.glob("*.glb"))
        if not glb_files:
            continue
        glb_file = glb_files[0]
        meta_file = glb_file.with_suffix(".meta.json")
        meta = None
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[TSDFList] failed to read {meta_file}: {e}")
        shapes.append({
            "folder": obj_folder.name,
            "glb_url": f"/api/segmentation/tsdf/file/{session_id}/{obj_folder.name}/{glb_file.name}",
            "meta": meta,
        })
    return {"ok": True, "shapes": shapes}


@app.get("/api/segmentation/tsdf/file/{session_id}/{folder}/{filename}")
async def tsdf_file(session_id: str, folder: str, filename: str):
    """Serve a TSDF .glb (or its .meta.json)."""
    from fastapi.responses import FileResponse
    if "/" in folder or ".." in folder or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path component")

    ctx = _ctx(session_id)
    full_path = ctx.output_dir / "tsdf" / folder / filename
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    if filename.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif filename.endswith(".json"):
        media_type = "application/json"
    else:
        media_type = "application/octet-stream"
    return FileResponse(
        str(full_path),
        media_type=media_type,
        headers={"Cache-Control": "no-cache"},
    )


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
            # Content-based freshness: only rebuild if masks have changed
            masks_file = output_dir / "seg_masks.npz"
            if result_path.exists():
                if masks_file.exists():
                    masks_mtime = masks_file.stat().st_mtime
                    result_mtime = result_path.stat().st_mtime
                    if result_mtime >= masks_mtime:
                        # Result was generated AFTER the last mask change — still valid
                        with open(result_path) as f:
                            result = json.load(f)
                        age = _time.time() - result_path.stat().st_mtime
                        print(f"[SegRefresh] Result is up-to-date (masks unchanged, {age:.0f}s old) — skipping rebuild")
                        task_manager.finish(tid)
                        return {"instances": result.get("instances", [])}
                result_path.unlink()
                print(f"[SegRefresh] Masks changed — deleted stale segmentation_result.json for {session_id}")
            from segmentation_pipeline import _match_and_save_result
            task_manager.update(tid, pct=10, detail="Running DBSCAN + cloud matching...")
            print(f"[SegRefresh] Regenerating segmentation_result.json...")
            result = _match_and_save_result(output_dir)
            instances = result.get("instances", [])
            print(f"[SegRefresh] ✅ Done: {len(instances)} instances")

            # Carve per-object TSDF meshes — same faithful deliverable the
            # pipeline SAM3 stage produces (sam3_worker.py). The interactive
            # Segmentation Manager flow ends HERE (finalize/close), so without
            # this the interactive segments never got their cropped meshes.
            # Best-effort: never fail the refresh over a crop issue.
            try:
                scene_dir = output_dir / "tsdf" / "scene"
                if (scene_dir / "scene.glb.orig").exists() or (scene_dir / "scene.glb").exists():
                    from segmentation.tsdf_export import crop_scene_mesh_to_instances
                    task_manager.update(tid, pct=70,
                                        detail="Carving per-object TSDF meshes...")
                    written = crop_scene_mesh_to_instances(
                        output_dir=output_dir, segments_result=result)
                    print(f"[SegRefresh] Per-object TSDF: wrote {len(written)} mesh(es)")
                else:
                    print("[SegRefresh] No scene TSDF — skipping per-object crop")
            except Exception as e:
                print(f"[SegRefresh] Per-object TSDF crop failed (non-fatal): {e}")

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
    skip_registration = body.get("skip_registration", False)
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
                    skip_registration=skip_registration,
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

    from segmentation.sam3_wrapper import get_sam3_wrapper
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
                
                # Unload SAM3 model if loaded — free VRAM for new session
                try:
                    from segmentation.sam3_wrapper import get_sam3_wrapper
                    _sam3 = get_sam3_wrapper()
                    if _sam3.is_loaded:
                        print("[Viewer] Unloading SAM3 model to free VRAM...")
                        await loop.run_in_executor(None, _sam3.unload_model)
                except Exception:
                    pass
                
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

                    # Auto-cleanup: stale display-space data from previous code version
                    display_marker = output_dir / ".display_space"
                    if display_marker.exists():
                        print("[Viewer] ⚠️ Stale .display_space marker found — cleaning cached data")
                        display_marker.unlink()
                        for stale in ["corrected_cloud.ply", "segmentation_result.json", "classification.npy"]:
                            p = output_dir / stale
                            if p.exists():
                                p.unlink()
                        stale_potree = output_dir / "potree"
                        if stale_potree.exists():
                            import shutil
                            shutil.rmtree(stale_potree)

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
                            # Check if cleaned_cloud.ply has confidence data
                            has_confidence = False
                            cleaned_path = output_dir / "cleaned_cloud.ply"
                            if cleaned_path.exists():
                                with open(cleaned_path, 'rb') as fp:
                                    for hl in fp:
                                        if b'confidence' in hl:
                                            has_confidence = True
                                        if hl.startswith(b'end_header'):
                                            break
                            # Load camera poses (4x4 extrinsic matrices from VGGT-Long)
                            # Prefer maplong_run/ (keyframe-only) over root (may have all frames)
                            # Transform to floor-aligned space using floor_transform (s*R*p+t)
                            camera_poses_list = None
                            frame_names = None
                            
                            # Load keyframe names first (for tooltip labels + filtering)
                            sel_json = _load_ctx.frames_dir / "selected_frames.json"
                            if sel_json.exists():
                                with open(sel_json) as sf:
                                    sel_data = json.loads(sf.read())
                                    frame_names = sel_data if isinstance(sel_data, list) else sel_data.get("selected_files", [])
                            
                            # Find poses file (prefer maplong_run/ or da3_run/)
                            poses_path = output_dir / "maplong_run" / "camera_poses.txt"
                            if not poses_path.exists():
                                poses_path = output_dir / "da3_run" / "camera_poses.txt"
                            if not poses_path.exists():
                                poses_path = output_dir / "camera_poses.txt"
                            
                            if poses_path.exists():
                                try:
                                    # Load floor transform for pose transformation
                                    ft_s, ft_R, ft_t = 1.0, np.eye(3), np.zeros(3)
                                    transform_path = output_dir / "floor_transform.npz"
                                    if transform_path.exists():
                                        ft_data = np.load(transform_path)
                                        ft_s = float(ft_data['s'])
                                        ft_R = ft_data['R']
                                        ft_t = ft_data['t']
                                    
                                    c2w_list = []
                                    with open(poses_path) as pf:
                                        for line in pf:
                                            vals = line.strip().split()
                                            if len(vals) >= 16:
                                                c2w_scan = np.array([float(v) for v in vals[:16]]).reshape(4, 4)
                                                c2w_aligned = np.eye(4)
                                                c2w_aligned[:3, :3] = ft_R @ c2w_scan[:3, :3]
                                                c2w_aligned[:3, 3] = ft_s * ft_R @ c2w_scan[:3, 3] + ft_t
                                                c2w_list.append(c2w_aligned.tolist())
                                    
                                    # Filter to keyframe indices if poses > keyframes
                                    n_kf = len(frame_names) if frame_names else 0
                                    if n_kf > 0 and len(c2w_list) > n_kf:
                                        all_jpg = sorted(f.name for f in _load_ctx.frames_dir.glob("*.jpg"))
                                        fn2idx = {fn: i for i, fn in enumerate(all_jpg)}
                                        kf_indices = [fn2idx[kf] for kf in frame_names if kf in fn2idx]
                                        valid = [i for i in kf_indices if i < len(c2w_list)]
                                        if valid:
                                            c2w_list = [c2w_list[i] for i in valid]
                                            print(f"[Viewer] Filtered {len(valid)} keyframe poses from {len(all_jpg)} total")
                                    
                                    if c2w_list:
                                        camera_poses_list = c2w_list
                                        # Load intrinsics for FOV matching
                                        intrinsics_list = []
                                        intr_path = poses_path.parent / "intrinsic.txt"
                                        if intr_path.exists():
                                            with open(intr_path) as inf:
                                                for line in inf:
                                                    parts = line.strip().split()
                                                    if len(parts) >= 4:
                                                        intrinsics_list.append({
                                                            "fx": float(parts[0]),
                                                            "fy": float(parts[1]),
                                                            "cx": float(parts[2]),
                                                            "cy": float(parts[3]),
                                                        })
                                        # Bundle: (poses, frame_names, intrinsics)
                                        camera_poses_list = (
                                            c2w_list,
                                            frame_names[:len(c2w_list)] if frame_names else None,
                                            intrinsics_list[:len(c2w_list)] if intrinsics_list else None,
                                        )
                                        print(f"[Viewer] Loaded {len(c2w_list)} camera poses from {poses_path.name}")
                                except Exception as e:
                                    print(f"[Viewer] ⚠️ Could not load camera poses: {e}")
                            scene_payload = _load_scene_payload(output_dir, session_id)
                            return potree_meta, floor_transform_4x4, ifc_files, has_confidence, camera_poses_list, scene_payload

                        potree_meta, floor_transform_4x4, ifc_files, has_confidence, camera_poses_list, scene_payload = await loop.run_in_executor(None, _load_metadata)

                        msg = {
                            "type": "potree_ready",
                            "session_id": session_id,
                            "url": f"/potree/{session_id}/",
                            "points": potree_meta.get("points", 0),
                            "hasConfidence": has_confidence,
                        }
                        if floor_transform_4x4:
                            msg["floorTransform"] = floor_transform_4x4
                        if ifc_files:
                            msg["bimFiles"] = ifc_files
                        if scene_payload is not None:
                            msg["scene"] = scene_payload
                        # Camera-pose markers: sent on normal load (UI has a toggle).
                        # The flythrough hides them via the viewport (setFlythroughActive).
                        if camera_poses_list:
                            if isinstance(camera_poses_list, tuple):
                                msg["cameraPoses"] = camera_poses_list[0]
                                if camera_poses_list[1]:
                                    msg["cameraFrameNames"] = camera_poses_list[1]
                                if camera_poses_list[2]:
                                    msg["cameraIntrinsics"] = camera_poses_list[2]
                            else:
                                msg["cameraPoses"] = camera_poses_list
                        await websocket.send_text(json.dumps(msg))
                        print(f"[Viewer] ✅ Sent potree_ready for {session_id} ({potree_meta.get('points', 0):,} pts)")
                        
                        await asyncio.sleep(0)  # Yield to process pings
                        
                        # Offload segmentation loading to thread pool
                        from segmentation_pipeline import apply_segmentation_to_cloud
                        seg_data = await loop.run_in_executor(None, apply_segmentation_to_cloud, output_dir)
                        if seg_data.get("instances"):
                            should_reload_potree = seg_data.pop("reload_potree", False)
                            await websocket.send_text(json.dumps(seg_data))
                            print(f"[Viewer] Sent segmentation ({len(seg_data['instances'])} instances)")
                            
                            # Reload Potree with corrected (projected) cloud
                            if should_reload_potree:
                                potree_meta_path = _load_ctx.merged_potree / "metadata.json"
                                if not potree_meta_path.exists():
                                    potree_meta_path = output_dir / "potree" / "metadata.json"
                                if potree_meta_path.exists():
                                    reload_meta = json.loads(potree_meta_path.read_text())
                                    reload_msg = {
                                        "type": "potree_ready",
                                        "session_id": session_id,
                                        "url": f"/potree/{session_id}/",
                                        "points": reload_meta.get("points", 0),
                                    }
                                    if floor_transform_4x4 is not None:
                                        reload_msg["floorTransform"] = floor_transform_4x4
                                    if has_confidence:
                                        reload_msg["hasConfidence"] = True
                                    await websocket.send_text(json.dumps(reload_msg))
                                    print(f"[Viewer] 🔄 Sent potree_ready reload (corrected cloud)")

                        
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
                    print(f"Error loading session:")
                    print(f"[Viewer] ❌ WebSocket Error: {e}")
                    try:
                        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass  # Client already disconnected

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

                # The pipeline ALWAYS runs end-to-end (reconstruction → cloudcompy
                # → tsdf) — any stage selection the client might still send is
                # deliberately ignored (see build_pipeline_stages).
                from pipeline_manager import build_pipeline_stages
                _recon_backend = cfg.get("reconstruction", {}).get("backend", "da3")
                stages = build_pipeline_stages(backend=_recon_backend)

                # Progress callback: relay to this websocket + broadcast
                from task_manager import task_manager as _tm
                _pipeline_tid = _tm.start(session_id, "pipeline", "Running Pipeline")

                async def _on_pipeline_progress(sid, job_dict):
                    # BROADCAST to all live viewers, NOT the socket captured at run start: a
                    # pipeline runs for hours and outlives the WS (ping timeout → reconnect),
                    # so the captured socket dies and progress/'done'/potree_ready were sent to
                    # a dead socket → UI stuck on "pipeline running", cloud never loaded.
                    try:
                        stage = job_dict.get("current_stage", "")
                        pct = job_dict.get("pct", 0)
                        _tm.update(_pipeline_tid, pct=pct, detail=f"Stage: {stage}")
                        await viewer_manager.broadcast_text(json.dumps({
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
                            await viewer_manager.broadcast_text(json.dumps({
                                "type": "error",
                                "message": f"Pipeline failed for {sid}. Check server logs."
                            }))
                        except Exception:
                            pass
                        return

                    # Convert to Potree octree and notify viewer
                    try:
                        session_path = _ctx(sid).session_dir
                        await viewer_manager.broadcast_text(json.dumps({
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
                                # Check confidence in PLY
                                has_confidence = False
                                cp = session_path / "output" / "cleaned_cloud.ply"
                                if cp.exists():
                                    with open(cp, 'rb') as fp:
                                        for hl in fp:
                                            if b'confidence' in hl:
                                                has_confidence = True
                                            if hl.startswith(b'end_header'):
                                                break
                                return potree_meta, floor_transform_4x4, has_confidence
                            
                            potree_meta, floor_transform_4x4, has_confidence = await _pipe_loop.run_in_executor(None, _load_pipe_metadata)
                            
                            pipe_msg = {
                                "type": "potree_ready",
                                "session_id": sid,
                                "url": f"/potree/{sid}/",
                                "points": potree_meta.get("points", 0),
                                "hasConfidence": has_confidence,
                            }
                            if floor_transform_4x4:
                                pipe_msg["floorTransform"] = floor_transform_4x4
                            await viewer_manager.broadcast_text(json.dumps(pipe_msg))
                            print(f"[Pipeline] ✅ Potree ready for {sid}")
                        else:
                            print(f"[Pipeline] ⚠️ Potree conversion failed, sending raw cloud")
                            await _send_cleaned_cloud_broadcast(sid)
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
                            await viewer_manager.broadcast_text(json.dumps(seg_data))
                            print(f"[Pipeline] Sent {len(seg_data['instances'])} segments")
                    except Exception as e:
                        print(f"[Pipeline] Broadcast error: {e}")

                    try:
                        _tm.finish(_pipeline_tid)
                        await viewer_manager.broadcast_text(json.dumps({
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
                    from db.team import TeamMessage
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
                projects_dir = PROJECTS_DIR
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