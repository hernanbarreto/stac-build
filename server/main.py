# STAC-BUILD: Main Server
# PHASE 3 COMPLETE: Incremental Streaming (Logic Fixed)

import asyncio
import json
import time
import gc
import os
import sys
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
    # AlignmentManager._generate_point_cloud uses images_kf which are 0-255, then divides by 255.0. 
    # So point_cloud[:, 3:] is 0-1 float.
    output_data[:, 3] = point_cloud[:, 3]
    output_data[:, 4] = point_cloud[:, 4]
    output_data[:, 5] = point_cloud[:, 5]
    
    # ClassID (Default 0)
    output_data[:, 6] = 0.0
    
    return output_data.tobytes()

# --- PLY Utilities for Offline Streaming ---
def load_ply_to_numpy(ply_path: Path) -> Optional[np.ndarray]:
    """Load PLY file (ASCII or BINARY) to numpy array [N, 6] (XYZ + RGB)."""
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
                points = []
                for _ in range(vertex_count):
                    data = f.read(15)  # 3 floats (12 bytes) + 3 uchars (3 bytes)
                    if len(data) < 15:
                        break
                    x, y, z = struct.unpack('<fff', data[0:12])
                    r, g, b = struct.unpack('<BBB', data[12:15])
                    points.append([x, y, z, r/255.0, g/255.0, b/255.0])

                return np.array(points, dtype=np.float32) if points else None
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
end_header
""".encode('ascii')

    # Pack points: XYZ as float, RGB as uchar
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
        if self.active_camera: await self.active_camera.close()
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
                
                # --- KEY FIX: INCREMENTAL BROADCAST ---
                # Enviamos SOLO el último chunk procesado.
                # El JS (FusionRenderer) se encarga de acumularlo.
                latest_chunk = alignment_manager.aligned_chunks[-1]
                if latest_chunk.point_cloud is not None and len(latest_chunk.point_cloud) > 0:
                    # 1. Send Chunk Start
                    await viewer_manager.broadcast_text(json.dumps({
                        "type": "chunk_start",
                        "chunk_id": chunk_id,
                        "point_count": len(latest_chunk.point_cloud)
                    }))
                    
                    # 2. Send Binary
                    binary = cloud_to_binary(latest_chunk.point_cloud)
                    await viewer_manager.broadcast_binary(binary)
                    
                    # 3. Send Segmentation
                    # The file was just saved in _heavy_lifting at save_chunk_segmentation
                    seg_path = session.chunks_dir.parent / "output" / f"chunk_{chunk_id:03d}_segments.json"
                    
                    # Check if file exists, if not maybe frame_storage has different path logic
                    # frame_storage.save_chunk_segmentation uses self.output_dir which is session.output_path
                    # session.chunks_dir is usually .../frames/chunk_XXX. 
                    # Output is .../output/chunk_XXX.
                    # Let's rely on constructing path or check frame_storage.
                    
                    if not seg_path.exists():
                         # Fallback: try to construct path from session root
                         seg_path = session.output_dir / f"chunk_{chunk_id:03d}_segments.json"

                    if seg_path.exists():
                        try:
                            with open(seg_path, 'r') as f:
                                seg_data = json.load(f)
                            seg_data["type"] = "segmentation"
                            await viewer_manager.broadcast_text(json.dumps(seg_data))
                            print(f"[ChunkWorker] Sent segmentation for Chunk {chunk_id}")
                        except Exception as e:
                            print(f"[ChunkWorker] Failed to send seg: {e}")
                    
                    print(f"[ChunkWorker] Sent INCREMENTAL update: {len(latest_chunk.point_cloud)} points")

            is_processing_chunk = False

            # Check if DA3 is done (queue empty) and there's a pending retroactive prompt
            if chunk_queue.qsize() == 0 and hasattr(frame_storage, '_pending_retroactive_prompt'):
                pending_prompt = getattr(frame_storage, '_pending_retroactive_prompt', None)
                if pending_prompt:
                    print(f"[ChunkWorker] DA3 complete. Running pending retroactive segmentation for '{pending_prompt}'...")
                    frame_storage._pending_retroactive_prompt = None  # Clear the flag

                    # Run retroactive segmentation in background
                    asyncio.create_task(_run_pending_retroactive(pending_prompt))

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ChunkWorker] Error: {e}")
            import traceback
            traceback.print_exc()
            is_processing_chunk = False


async def _run_pending_retroactive(prompt: str):
    """Execute pending retroactive segmentation after DA3 completes ALL chunks."""
    try:
        session = frame_storage.current_session
        if not session:
            print("[Retro-Pending] No active session")
            return

        print(f"[Retro-Pending] DA3 complete. Starting segmentation for ALL chunks with prompt: '{prompt}'")

        # Unload DA3 to free VRAM (CRITICAL: DA3 must be fully unloaded before SAM3)
        print("[Retro-Pending] Unloading DA3 to free VRAM...")
        chunk_processor.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Retro-Pending] VRAM freed. Available: {torch.cuda.memory_allocated()/1e9:.2f}GB used")

        sam3 = get_sam3_wrapper()

        # Discover ALL chunks (not just those without segmentation)
        # This ensures consistent segmentation across the entire session
        output_dir = session.output_dir
        ply_files = sorted(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []

        chunk_ids = []
        for ply_path in ply_files:
            try:
                cid = int(ply_path.stem.split('_')[1])
                chunk_ids.append(cid)
            except:
                pass

        chunk_ids = sorted(set(chunk_ids))

        if not chunk_ids:
            print("[Retro-Pending] No chunks found to segment.")
            sam3.unload_model()
            return

        print(f"[Retro-Pending] Processing {len(chunk_ids)} chunks: {chunk_ids}")

        for chunk_id in sorted(chunk_ids):
            frames_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"
            if not frames_dir.exists():
                continue

            print(f"[Retro-Pending] Segmenting Chunk {chunk_id}...")
            try:
                masks = sam3.process_chunk(str(frames_dir), prompt, keyframe_interval=None)

                ply_points = frame_storage.load_ply(session, chunk_id)
                meta = frame_storage.load_chunk_metadata(session, chunk_id)

                if ply_points is not None and meta:
                    sample_indices = meta.get("sample_indices")
                    if sample_indices is not None:
                        sample_indices = np.array(sample_indices, dtype=np.int64)

                    segments = chunk_processor.compute_segmentation_for_ply(
                        ply_points, meta, masks,
                        sample_indices=sample_indices,
                        chunk_idx=chunk_id
                    )
                    frame_storage.save_segments_direct(session, chunk_id, segments, prompt=prompt)
            except Exception as e:
                print(f"[Retro-Pending] Error segmenting chunk {chunk_id}: {e}")

        # Unload SAM3
        print("[Retro-Pending] Unloading SAM3...")
        sam3.unload_model()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Rebuild unified segments.json
        if frame_storage.segmentation_manager:
            frame_storage.segmentation_manager.rebuild_from_chunks()

        # Notify viewers
        await viewer_manager.broadcast_text(json.dumps({
            "type": "info",
            "message": f"Retroactive segmentation complete for '{prompt}'"
        }))

        print("[Retro-Pending] Retroactive segmentation complete.")

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
    # REMOVED: Data is already aligned by AlignmentManager to Y-up (GL compatible)
    # data[:, 1] *= -1 
    # data[:, 2] *= -1 
    
    return data.tobytes()

def on_chunk_ready(session, chunk_info):
    asyncio.create_task(chunk_queue.put((session, chunk_info)))
    print(f"[FrameStorage] Chunk {chunk_info.chunk_id} queued")

# --- Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global frame_storage, chunk_processor, alignment_manager, chunk_queue
    print("[Server] Starting STAC-BUILD...")
    
    frame_storage = get_frame_storage()
    frame_storage.on_chunk_ready = on_chunk_ready
    chunk_processor = get_chunk_processor()
    alignment_manager = get_alignment_manager()
    
    # Initialize SAM3 Wrapper (lazy load, but triggers init log)
    get_sam3_wrapper()
    
    # NOTE: DA3 model is NOT loaded here anymore for memory efficiency.
    # It will be loaded lazily when needed:
    # - Online streaming: loaded when first chunk arrives
    # - Offline: only loaded if PLYs don't exist
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
        segments_file = scans_dir / session_id / "output" / "segments.json"
        
        if segments_file.exists():
            with open(segments_file, 'r') as f:
                return json.load(f)
        return {"object_types": []}
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
        # Pre-load DA3 for streaming readiness
        if not chunk_processor.is_loaded:
            print("[Server] 🔌 Switching to ONLINE mode - Loading DA3...")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "info",
                "message": "Loading DA3 model for streaming..."
            }))
            
            # Load in executor to not block
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, chunk_processor.load_model)
            
            print("[Server] ✅ DA3 loaded - Ready for streaming!")
            await viewer_manager.broadcast_text(json.dumps({
                "type": "mode_ready",
                "mode": "online",
                "message": "Ready for streaming"
            }))
        else:
            print("[Server] DA3 already loaded")
            
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

@app.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # 1. Enforce Online Mode
    if server_mode != "online":
        print(f"[Camera] ⛔ Connection rejected: Server is in {server_mode} mode")
        await websocket.close(code=1008, reason="Server is offline")
        return

    await camera_manager.connect(websocket)
    # frame_storage.start_session() # REMOVED: Lazy init in add_frame to avoid empty folders
    try:
        while True:
            data = await websocket.receive_bytes()
            import cv2
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_storage.add_frame(frame)
                camera_manager.frame_count += 1
    except Exception as e:
        print(f"[Camera] Connection closed: {e}")
    finally:
        camera_manager.disconnect()
        # 2. Finalize Session (Save Poses/Intrinsics)
        if chunk_processor and chunk_processor.is_loaded:
             print("[Camera] Finalizing session...")
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
        
        # --- STREAMING HISTORY ---
        # Instead of sending one giant file, we send the movie frame-by-frame
        if alignment_manager:
            history = alignment_manager.aligned_chunks
            if history:
                print(f"[Viewer] Streaming history: {len(history)} chunks...")
                for chunk in history:
                    if chunk.point_cloud is not None and len(chunk.point_cloud) > 0:
                        binary = cloud_to_binary(chunk.point_cloud)
                        # Send in small chunks to avoid disconnection
                        for sub_chunk in chunk_data(binary):
                            await viewer_manager.send_bytes(websocket, sub_chunk)
                            await asyncio.sleep(0.001) # Micro-sleep
                print("[Viewer] History stream complete.")
        
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
                        # DA3 is working - queue the segmentation for later
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
                                 # Iterate all chunks that are already finished
                                 session = frame_storage.current_session
                                 if not session: return False

                                 # Unload DA3 if it's lingering from Online Capture
                                 print("[Retro] Unloading DA3 to free VRAM for SAM3...")
                                 chunk_processor.unload_model()
                                 import gc
                                 gc.collect()
                                 if torch.cuda.is_available():
                                     torch.cuda.empty_cache()

                                 sam3 = get_sam3_wrapper()

                                 # Discover chunks from PLY files on disk (more reliable than session.chunks for offline)
                                 output_dir = session.output_dir
                                 ply_files = sorted(output_dir.glob("chunk_*.ply")) if output_dir.exists() else []

                                 # Extract chunk IDs from PLY filenames
                                 chunk_ids = []
                                 for ply_path in ply_files:
                                     try:
                                         # chunk_XXX.ply -> XXX
                                         cid = int(ply_path.stem.split('_')[1])
                                         chunk_ids.append(cid)
                                     except:
                                         pass

                                 chunk_ids = sorted(set(chunk_ids))
                                 print(f"[Retro] Found {len(chunk_ids)} chunks with PLY files: {chunk_ids}")

                                 for chunk_id in chunk_ids:
                                     # Build frames directory path
                                     frames_dir = session.chunks_dir / f"chunk_{chunk_id:03d}"

                                     # Ensure Frames Exist
                                     if not frames_dir.exists():
                                         print(f"[Retro] Skipped Chunk {chunk_id}: Frames directory not found.")
                                         continue

                                     print(f"[Retro] Segmenting Chunk {chunk_id}...")
                                     try:
                                         # Run SAM3
                                         masks = sam3.process_chunk(str(frames_dir), prompt, keyframe_interval=None)

                                         # PROJECT to Existing PLY (Do NOT regenerate PLY)
                                         # Load PLY & Meta
                                         ply_points = frame_storage.load_ply(session, chunk_id)
                                         meta = frame_storage.load_chunk_metadata(session, chunk_id)

                                         if ply_points is not None and meta:
                                             # Get sample_indices from metadata for direct 2D→3D mapping
                                             sample_indices = meta.get("sample_indices")
                                             if sample_indices is not None:
                                                 sample_indices = np.array(sample_indices, dtype=np.int64)

                                             segments = chunk_processor.compute_segmentation_for_ply(
                                                 ply_points, meta, masks,
                                                 sample_indices=sample_indices,
                                                 chunk_idx=chunk_id
                                             )
                                             seg_path = frame_storage.save_segments_direct(session, chunk_id, segments, prompt=prompt)

                                             # Broadcast Segments IMMEDIATELY via main loop
                                             if seg_path and os.path.exists(seg_path):
                                                  pass # Viewer refresh at end will handle it
                                         else:
                                             print(f"[Retro] Skipped Chunk {chunk_id}: Missing PLY/Meta.")

                                     except Exception as e:
                                         print(f"[Retro] Error Segmenting Chunk {chunk_id}: {e}")

                                 print("[Retro] Done re-segmenting history.")

                                 # Unload SAM3 to free VRAM
                                 print("[Retro] Unloading SAM3 to free VRAM...")
                                 sam3.unload_model()
                                 gc.collect()
                                 if torch.cuda.is_available():
                                     torch.cuda.empty_cache()
                                 print("[Retro] SAM3 unloaded, resources freed.")

                                 # Rebuild unified segments.json
                                 if frame_storage.segmentation_manager:
                                     print("[Retro] Rebuilding unified segments.json...")
                                     frame_storage.segmentation_manager.rebuild_from_chunks()

                                 return True
                             except Exception as e:
                                 print(f"[Retro] Error: {e}")
                                 import traceback
                                 traceback.print_exc()
                                 # Ensure SAM3 is unloaded even on error
                                 try:
                                     sam3.unload_model()
                                 except:
                                     pass
                                 return False
                         
                         # Execute in thread
                         async def _run_retro_active():
                             loop = asyncio.get_running_loop()
                             success = await loop.run_in_executor(None, _retro_process_active)
                             if success:
                                 # Refresh viewer
                                 await websocket.send_text(json.dumps({"type": "cleared"}))
                                 history = alignment_manager.aligned_chunks
                                 for chunk in history:
                                     if chunk.point_cloud is not None and len(chunk.point_cloud) > 0:
                                         binary = cloud_to_binary(chunk.point_cloud)
                                         await viewer_manager.send_bytes(websocket, binary)
                                         await asyncio.sleep(0.01)
                                         
                                         # Send Segments if available
                                         seg_path = frame_storage.current_session.output_dir / f"chunk_{chunk.chunk_id:03d}_segments.json"
                                         if seg_path.exists():
                                             with open(seg_path, 'r') as f:
                                                 seg_data = json.load(f)
                                             seg_data["type"] = "segmentation"
                                             await viewer_manager.send_text(websocket, json.dumps(seg_data))

                                 print("[Viewer] Refreshed view with segmented data.")

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
                                weights_dir = Path(cfg["paths"]["da3_weights_dir"]).resolve()

                                # Auto-select correct config.json based on model name
                                model_name = cfg["models"]["depth"]["name"]
                                if "GIANT" in model_name.upper() or "NESTED" in model_name.upper():
                                    config_source = weights_dir / "config_giant.json"
                                    if config_source.exists():
                                        import shutil
                                        shutil.copy(str(config_source), str(weights_dir / "config.json"))
                                        print(f"[Retro-Offline] Using GIANT config (CPU mode)")
                                else:
                                    config_source = weights_dir / "config_large.json"
                                    if config_source.exists():
                                        import shutil
                                        shutil.copy(str(config_source), str(weights_dir / "config.json"))
                                        print(f"[Retro-Offline] Using LARGE config (GPU mode)")

                                # 2. Create DA3 Config
                                da3_config = {
                                    "Weights": {
                                        "DA3_CONFIG": str(weights_dir / cfg["models"]["depth"]["config_file"]),
                                        "DA3": str(weights_dir / cfg["models"]["depth"]["weights_file"]),
                                        "SALAD": str(weights_dir / "dino_salad.ckpt")
                                    },
                                    "Model": {
                                        "chunk_size": cfg["server"]["chunk_size"],
                                        "overlap": cfg["server"]["chunk_overlap"],
                                        "loop_chunk_size": cfg["models"]["da3"]["loop_chunk_size"],
                                        "loop_enable": False,
                                        "useDBoW": False,
                                        "delete_temp_files": True,
                                        "align_method": cfg["alignment"]["method"],
                                        "align_lib": "torch",
                                        "scale_compute_method": "auto",
                                        "align_type": "dense",
                                        "ref_view_strategy": cfg["models"]["depth"]["ref_view_strategy"],
                                        "ref_view_strategy_loop": cfg["models"]["depth"]["ref_view_strategy"],
                                        "depth_threshold": 15.0,
                                        "save_depth_conf_result": False,
                                        "save_debug_info": False,
                                        "Sparse_Align": {
                                            "keypoint_select": "orb",
                                            "keypoint_num": 5000
                                        },
                                        "IRLS": {
                                            "delta": cfg["alignment"]["ransac"]["delta"],
                                            "max_iters": cfg["alignment"]["ransac"]["max_iters"],
                                            "tol": cfg["alignment"]["ransac"]["tolerance"]
                                        },
                                        "Pointcloud_Save": {
                                            "conf_threshold_coef": cfg["models"]["da3"]["pointcloud_save"]["conf_threshold_coef"],
                                            "sample_ratio": cfg["models"]["da3"]["pointcloud_save"]["sample_ratio"]
                                        }
                                    },
                                    "Loop": {
                                        "SALAD": {
                                            "image_size": [336, 336],
                                            "batch_size": 32,
                                            "similarity_threshold": 0.85,
                                            "top_k": 5,
                                            "use_nms": True,
                                            "nms_threshold": 25
                                        },
                                        "SIM3_Optimizer": {
                                            "lang_version": "python",
                                            "max_iterations": 30,
                                            "lambda_init": 1e-6
                                        }
                                    }
                                }

                                # 3. Create RealtimeDA3 instance with AlignmentManager for gravity correction
                                print(f"[Retro-Offline] Initializing RealtimeDA3...")
                                alignment_manager.reset()  # Reset to compute fresh gravity for this session
                                da3 = RealtimeDA3(
                                    image_dir=str(images_dir),
                                    save_dir=str(output_dir),
                                    config=da3_config,
                                    alignment_manager=alignment_manager
                                )

                                # 4. Setup incremental streaming callback
                                sent_chunks = set()

                                async def on_chunk_complete(chunk_id, sim3_transform):
                                    """Callback: Send PLY to viewer as soon as it's generated."""
                                    try:
                                        # Look for chunk_XXX.ply in output directory
                                        ply_file = output_dir / f"chunk_{chunk_id:03d}.ply"

                                        # Wait briefly for file to be written
                                        await asyncio.sleep(0.1)

                                        if ply_file.exists() and chunk_id not in sent_chunks:
                                            file_size = ply_file.stat().st_size
                                            if file_size > 0:
                                                # Load and count points
                                                points = load_ply_to_numpy(ply_file)
                                                point_count = len(points) if points is not None else 0

                                                # Read binary PLY for sending
                                                with open(ply_file, "rb") as f:
                                                    ply_data = f.read()

                                                await viewer_manager.broadcast_text(json.dumps({
                                                    "type": "chunk_start",
                                                    "chunk_id": chunk_id,
                                                    "point_count": point_count
                                                }))
                                                await viewer_manager.broadcast_binary(ply_data)
                                                print(f"[Retro-Offline] ✅ Sent Chunk {chunk_id} ({point_count:,} points, {file_size/1024/1024:.1f}MB)")
                                                sent_chunks.add(chunk_id)
                                    except Exception as e:
                                        print(f"[Retro-Offline] ⚠️  Error sending chunk {chunk_id}: {e}")

                                # 5. Run DA3 with incremental streaming
                                print(f"[Retro-Offline] Processing {len(da3.img_list)} images...")
                                await da3.process_long_sequence_async(callback=on_chunk_complete)

                                print(f"[Retro-Offline] ✅ Complete! Sent {len(sent_chunks)} chunks")
                                return True

                            except Exception as e:
                                print(f"[Retro-Offline] Error: {e}")
                                import traceback
                                traceback.print_exc()
                                return False
                        
                        # Direct await since it is now async
                        success = await _retro_process_offline()


                        
                        if success:
                            # Send completion message - chunks were already broadcast during Phase 3
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
                
                # Stream saved PLYs
                try:
                    scans_dir = Path(__file__).parent / "scans"
                    output_dir = scans_dir / session_id / "output"

                    # Look for PLY files in both pcd/ subdirectory and output/ directory
                    pcd_dir = output_dir / "pcd"
                    ply_files = []

                    if pcd_dir.exists():
                        ply_files.extend(pcd_dir.glob("*.ply"))

                    # Also check output directory itself
                    if output_dir.exists():
                        ply_files.extend(output_dir.glob("*.ply"))

                    # Remove duplicates and sort
                    ply_files = sorted(set(ply_files))
                    
                    if not ply_files:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No point clouds found for this session"}))
                    else:
                        print(f"[Viewer] Found {len(ply_files)} PLY files in {pcd_dir}")
                        valid_chunks = 0
                        
                        # Legacy Alignment Support
                        offline_transform = None
                        
                        for ply_path in ply_files:
                            try:
                                # Skip non-chunk files
                                if "camera_poses" in ply_path.name: continue
                                
                                # Parse Chunk ID
                                cid_str = ""
                                if "_pcd" in ply_path.name:
                                    cid_str = ply_path.name.split('_')[0]
                                elif "chunk_" in ply_path.name:
                                    cid_str = ply_path.stem.split('_')[1]
                                else:
                                    continue # Unknown format
                                    
                                chunk_id = int(cid_str)
                                
                                # Use unified loader
                                if frame_storage.current_session:
                                    # Use frame_storage.load_ply which now handles ASCII/Binary and multiple paths
                                    data = frame_storage.load_ply(frame_storage.current_session, chunk_id)
                                    
                                    if data is None: 
                                        print(f"[Viewer] Failed to load Chunk {chunk_id} (None returned)")
                                        continue
                                    
                                    # Check for metadata to decide on auto-leveling
                                    # If chunk 0 and no meta, compute leveling
                                    if chunk_id == 0:
                                        meta = frame_storage.load_chunk_metadata(frame_storage.current_session, 0)
                                        # ALWAYS compute auto-leveling on chunk 0 for gravity correction
                                        # ply_pre_aligned means chunks aligned to EACH OTHER, not gravity
                                        print(f"[Viewer] Computing Auto-Leveling on Chunk 0...")
                                        s_l, R_l, t_l = alignment_manager.compute_leveling_from_points(data)
                                        offline_transform = (s_l, R_l, t_l)

                                    # Apply Offline Transform if computed
                                    if offline_transform is not None:
                                        s_l, R_l, t_l = offline_transform
                                        # Apply: p' = s * (p @ R.T) + t
                                        # Data is (N, 6) [x,y,z,r,g,b]
                                        xyz = data[:, :3]
                                        data[:, :3] = s_l * (xyz @ R_l.T) + t_l
                                    
                                    # 1. Send Chunk Start
                                    # 1. Send Chunk Start
                                    await viewer_manager.send_text(websocket, json.dumps({
                                        "type": "chunk_start",
                                        "chunk_id": chunk_id,
                                        "point_count": len(data)
                                    }))
                                    
                                    # 2. Send Binary Cloud
                                    binary = cloud_to_binary(data)
                                    # Send small chunks
                                    for sub_chunk in chunk_data(binary):
                                        await viewer_manager.send_bytes(websocket, sub_chunk)
                                        await asyncio.sleep(0.001)

                                    valid_chunks += 1
                                    
                                    # 3. Send Segmentation (if exists) -> Look in output dir (not pcd)
                                    seg_path = output_dir / f"chunk_{chunk_id:03d}_segments.json"
                                    if seg_path.exists():
                                        try:
                                            with open(seg_path, 'r') as f:
                                                seg_data = json.load(f)
                                            seg_data["type"] = "segmentation"
                                            await viewer_manager.send_text(websocket, json.dumps(seg_data))
                                        except Exception as e:
                                            print(f"[Viewer] Seg send error: {e}")
                                    
                                    await asyncio.sleep(0.01)
                                    
                            except Exception as e:
                                print(f"[Viewer] Error loading chunk {ply_path}: {e}")
                        
                        await websocket.send_text(json.dumps({"type": "status", "message": f"Loaded {len(ply_files)} chunks from {session_id}"}))

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

                        print(f"[Reconstruct] 🟢 STARTING DA3 INCREMENTAL RECONSTRUCTION (RealtimeDA3)")

                        # 1. Prepare Paths
                        images_dir = session.frames_dir.resolve()
                        output_dir = session.output_dir.resolve()
                        weights_dir = Path(cfg["paths"]["da3_weights_dir"]).resolve()

                        # Auto-select correct config.json based on model name
                        model_name = cfg["models"]["depth"]["name"]
                        if "GIANT" in model_name.upper() or "NESTED" in model_name.upper():
                            config_source = weights_dir / "config_giant.json"
                            if config_source.exists():
                                import shutil
                                shutil.copy(str(config_source), str(weights_dir / "config.json"))
                                print(f"[Reconstruct] Using GIANT config (CPU mode)")
                        else:
                            config_source = weights_dir / "config_large.json"
                            if config_source.exists():
                                import shutil
                                shutil.copy(str(config_source), str(weights_dir / "config.json"))
                                print(f"[Reconstruct] Using LARGE config (GPU mode)")

                        # 2. Create DA3 Config
                        da3_config = {
                            "Weights": {
                                "DA3_CONFIG": str(weights_dir / cfg["models"]["depth"]["config_file"]),
                                "DA3": str(weights_dir / cfg["models"]["depth"]["weights_file"]),
                                "SALAD": str(weights_dir / "dino_salad.ckpt")
                            },
                            "Model": {
                                "chunk_size": cfg["server"]["chunk_size"],
                                "overlap": cfg["server"]["chunk_overlap"],
                                "loop_chunk_size": cfg["models"]["da3"]["loop_chunk_size"],
                                "loop_enable": False,
                                "useDBoW": False,
                                "delete_temp_files": True,
                                "align_method": cfg["alignment"]["method"],
                                "align_lib": "torch",
                                "scale_compute_method": "auto",
                                "align_type": "dense",
                                "ref_view_strategy": cfg["models"]["depth"]["ref_view_strategy"],
                                "ref_view_strategy_loop": cfg["models"]["depth"]["ref_view_strategy"],
                                "depth_threshold": 15.0,
                                "save_depth_conf_result": False,
                                "save_debug_info": False,
                                "Sparse_Align": {
                                    "keypoint_select": "orb",
                                    "keypoint_num": 5000
                                },
                                "IRLS": {
                                    "delta": cfg["alignment"]["ransac"]["delta"],
                                    "max_iters": cfg["alignment"]["ransac"]["max_iters"],
                                    "tol": cfg["alignment"]["ransac"]["tolerance"]
                                },
                                "Pointcloud_Save": {
                                    "conf_threshold_coef": cfg["point_cloud"]["generation"]["conf_threshold_coef"],
                                    "sample_ratio": cfg["point_cloud"]["generation"]["sample_ratio"]
                                }
                            },
                            "Loop": {
                                "SALAD": {
                                    "image_size": [336, 336],
                                    "batch_size": 32,
                                    "similarity_threshold": 0.85,
                                    "top_k": 5,
                                    "use_nms": True,
                                    "nms_threshold": 25
                                },
                                "SIM3_Optimizer": {
                                    "lang_version": "python",
                                    "max_iterations": 30,
                                    "lambda_init": 1e-6
                                }
                            }
                        }

                        # 3. Create RealtimeDA3 instance
                        print(f"[Reconstruct] Initializing RealtimeDA3...")
                        da3 = RealtimeDA3(
                            image_dir=str(images_dir),
                            save_dir=str(output_dir),
                            config=da3_config,
                            alignment_manager=alignment_manager
                        )

                        # 4. Setup incremental streaming callback
                        sent_chunks = set()

                        async def on_chunk_complete(chunk_id, sim3_transform):
                            """Callback: Send PLY to viewer as soon as it's generated."""
                            print(f"[Main] Callback triggered for Chunk {chunk_id}")
                            try:
                                # Look for chunk_XXX.ply in output directory
                                ply_file = output_dir / f"chunk_{chunk_id:03d}.ply"

                                # Wait briefly for file to be written
                                await asyncio.sleep(0.5) # Increased wait time

                                if ply_file.exists(): 
                                    file_size = ply_file.stat().st_size
                                    print(f"[Main] Found {ply_file.name} ({file_size} bytes)")
                                    
                                    if chunk_id not in sent_chunks and file_size > 0:
                                        # Parse PLY and convert to FusionRenderer format (N, 7) float32
                                        with open(ply_file, "rb") as f:
                                            # Skip header
                                            while True:
                                                line = f.readline()
                                                if line.startswith(b"end_header"):
                                                    break
                                            
                                            # Read binary body
                                            # PLY format from sim3utils: x, y, z (f4), r, g, b (u1)
                                            ply_dtype = np.dtype([
                                                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                                            ])
                                            raw_data = np.frombuffer(f.read(), dtype=ply_dtype)
                                        
                                        point_count = len(raw_data)
                                        
                                        if point_count > 0:
                                            # Convert to (N, 7) float32: x, y, z, r, g, b, classId
                                            # r,g,b need to be normalized 0-1 for FusionRenderer if it expects floats
                                            # (Actually FusionRenderer reads them as floats. If we send 255.0 it might be too bright, 
                                            # but let's stick to standard 0-1 for normalized float colors)
                                            output_data = np.zeros((point_count, 7), dtype=np.float32)
                                            output_data[:, 0] = raw_data['x']
                                            output_data[:, 1] = raw_data['y']
                                            output_data[:, 2] = raw_data['z']
                                            output_data[:, 3] = raw_data['r'] / 255.0
                                            output_data[:, 4] = raw_data['g'] / 255.0
                                            output_data[:, 5] = raw_data['b'] / 255.0
                                            output_data[:, 6] = 0.0 # classId
                                            
                                            ply_data_bytes = output_data.tobytes()

                                            print(f"[Main] Broadcasting Chunk {chunk_id} to viewer... ({len(ply_data_bytes)} bytes)")
                                            await viewer_manager.broadcast_text(json.dumps({
                                                "type": "chunk_start",
                                                "chunk_id": chunk_id,
                                                "point_count": point_count
                                            }))
                                            
                                            # Send in multiple small chunks (1MB) to prevent disconnection
                                            for sub_chunk in chunk_data(ply_data_bytes):
                                                await viewer_manager.broadcast_binary(sub_chunk)
                                                await asyncio.sleep(0.001)

                                            print(f"[Reconstruct] ✅ Sent Chunk {chunk_id} ({point_count:,} points)")
                                            sent_chunks.add(chunk_id)
                                        else:
                                            print(f"[Main] Skipping Chunk {chunk_id}: No points in PLY.")
                                    else:
                                        print(f"[Main] Skipping Chunk {chunk_id}: Already sent or empty.")
                                else:
                                    print(f"[Main] File NOT found: {ply_file}")
                            except Exception as e:
                                print(f"[Reconstruct] ⚠️  Error sending chunk {chunk_id}: {e}")

                        # 5. Run DA3 with incremental streaming
                        print(f"[Reconstruct] Processing {len(da3.img_list)} images...")
                        await da3.process_long_sequence_async(callback=on_chunk_complete)

                        print(f"[Reconstruct] ✅ Complete! Sent {len(sent_chunks)} chunks")
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