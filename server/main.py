# STAC-BUILD: Main Server
# PHASE 3 COMPLETE: Incremental Streaming (Logic Fixed)

import asyncio
import json
import time
import gc
import os
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


# --- Configuration ---
class Config:
    HOST = "0.0.0.0"
    PORT = 8765
    STATIC_DIR = Path(__file__).parent.parent / "static"
    CHUNK_SIZE = 30
    CHUNK_OVERLAP = 10


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
    
    async def connect_viewer(self, websocket: WebSocket):
        self.viewers.add(websocket)
        print(f"[Viewer] Client connected. Total: {len(self.viewers)}")
    
    def disconnect_viewer(self, websocket: WebSocket):
        self.viewers.discard(websocket)
        print(f"[Viewer] Client disconnected. Total: {len(self.viewers)}")
    
    async def broadcast_binary(self, data: bytes):
        if not data: return
        for viewer in list(self.viewers):
            try:
                await viewer.send_bytes(data)
            except:
                self.viewers.discard(viewer)

    async def broadcast_text(self, message: str):
        for viewer in list(self.viewers):
            try:
                await viewer.send_text(message)
            except:
                self.viewers.discard(viewer)

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
                        # Get alignment transform and validity mapping for retroactive segmentation
                        align_transform = alignment_manager.gravity_correction if hasattr(alignment_manager, 'gravity_correction') else None
                        validity_map = alignment_manager.generate_validity_mapping(chunk_id)
                        frame_storage.save_chunk_metadata(session, chunk_id, result, 
                                                         alignment_transform=align_transform,
                                                         validity_mapping=validity_map)
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
                                confs=result.confs     # Pass for validity mask
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
                         seg_path = session.output_path / f"chunk_{chunk_id:03d}_segments.json"

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
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[ChunkWorker] Error: {e}")
            import traceback
            traceback.print_exc()
            is_processing_chunk = False

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
    data[:, 1] *= -1 
    data[:, 2] *= -1 
    
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
app.mount("/static", StaticFiles(directory=str(Config.STATIC_DIR)), name="static")

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
        scans_dir = Path(__file__).parent / "scans"
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
    except:
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
        await websocket.send_text(json.dumps({"type": "status", "message": "Connected"}))
        
        # --- STREAMING HISTORY ---
        # Instead of sending one giant file, we send the movie frame-by-frame
        if alignment_manager:
            history = alignment_manager.aligned_chunks
            if history:
                print(f"[Viewer] Streaming history: {len(history)} chunks...")
                for chunk in history:
                    if chunk.point_cloud is not None and len(chunk.point_cloud) > 0:
                        binary = cloud_to_binary(chunk.point_cloud)
                        await websocket.send_bytes(binary)
                        # Tiny sleep to let network breathe
                        await asyncio.sleep(0.01)
                print("[Viewer] History stream complete.")
        
        while True:
            msg = await websocket.receive_text() # Keep alive
            cmd = json.loads(msg)
            if cmd.get("type") == "status":
                queue_size = chunk_queue.qsize()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "camera_connected": camera_manager.active_camera is not None,
                    "chunk_queue": queue_size
                }))
            elif cmd.get("type") == "clear":
                alignment_manager.reset()
                frame_storage.stop_session()
                await websocket.send_text(json.dumps({"type": "cleared"}))
            elif cmd.get("type") == "set_prompt":
                prompt = cmd.get("prompt")
                print(f"[Viewer] Received prompt request: '{prompt}'")
                if frame_storage:
                    frame_storage.set_prompt(prompt)
                await websocket.send_text(json.dumps({"type": "prompt_set", "prompt": prompt}))
                # --- Retroactive Processing (User Request) ---
                # "I apply the prompt at the end... it should propagate through frames"
                if prompt and alignment_manager:
                    if alignment_manager.get_chunk_count() > 0:
                        print(f"[Viewer] 🔄 Starting Retroactive Segmentation for {alignment_manager.get_chunk_count()} existing chunks...")
                        
                        # Run in background to not block socket
                        def _retro_process():
                            try:
                                # Iterate all chunks that are already finished
                                session = frame_storage.current_session
                                if not session: return

                                # Unload DA3 if it's lingering from Online Capture
                                print("[Retro] Unloading DA3 to free VRAM for SAM3...")
                                chunk_processor.unload_model()
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                
                                sam3 = get_sam3_wrapper()
                                
                                # Use all chunks from session object to ensure we get the latest ones
                                all_chunks = session.chunks
                                sam3_masks_cache = {}

                                for chunk_info in all_chunks:
                                    chunk_id = chunk_info.chunk_id
                                    frames_dir = frame_storage.get_chunk_frames_dir(chunk_info)
                                    
                                    # Ensure Frames Exist
                                    if not frames_dir.exists(): continue

                                    print(f"[Retro] Segmenting Chunk {chunk_id}...")
                                    try:
                                        # Run SAM3
                                        masks = sam3.process_chunk(str(frames_dir), prompt, keyframe_interval=5)
                                        
                                        # PROJECT to Existing PLY (Do NOT regenerate PLY)
                                        # Load PLY & Meta
                                        ply_points = frame_storage.load_ply(session, chunk_id)
                                        meta = frame_storage.load_chunk_metadata(session, chunk_id)
                                        
                                        if ply_points is not None and meta:
                                            segments = chunk_processor.compute_segmentation_for_ply(ply_points, meta, masks)
                                            seg_path = frame_storage.save_segments_direct(session, chunk_id, segments, prompt=prompt)
                                            
                                            # Broadcast Segments IMMEDIATELY
                                            if seg_path and os.path.exists(seg_path):
                                                pass # Viewer refresh at end will handle it, or we could broadcast here.
                                        else:
                                            print(f"[Retro] Skipped Chunk {chunk_id}: Missing PLY/Meta.")
                                            
                                    except Exception as e:
                                        print(f"[Retro] Error Segmenting Chunk {chunk_id}: {e}")

                                print("[Retro] Done re-segmenting history.")
                                return True
                            except Exception as e:
                                print(f"[Retro] Error: {e}")
                                import traceback
                                traceback.print_exc()
                                return False

                        # Execute
                        success = await asyncio.get_event_loop().run_in_executor(None, _retro_process)
                        
                        if success:
                            # Clear viewer and re-stream everything to show clean segmented result
                            await websocket.send_text(json.dumps({"type": "cleared"}))
                            history = alignment_manager.aligned_chunks
                            for chunk in history:
                                if chunk.point_cloud is not None and len(chunk.point_cloud) > 0:
                                    binary = cloud_to_binary(chunk.point_cloud)
                                    await websocket.send_bytes(binary)
                                    await asyncio.sleep(0.01)
                            print("[Viewer] Refreshed view with segmented data.")
                            
                            print("[Viewer] Refreshed view with segmented data.")
                            
                    else:
                        # MEMORY EMPTY (Loaded Session) -> FULL RE-PROCESS MODE
                        print(f"[Viewer] ⚠️ Archived session detected. Starting FULL offline re-processing (DA3 + SAM3). This may take time.")
                        await websocket.send_text(json.dumps({
                            "type": "info", 
                            "message": "Starting Offline Segmentation (Re-running AI)... Please wait."
                        }))

                        # CAPTURE LOOP for thread-safe broadcasting
                        main_loop = asyncio.get_running_loop()

                        def _retro_process_offline():
                            try:
                                session = frame_storage.current_session
                                if not session: return False
                                
                                chunks = session.chunks
                                print(f"[Retro-Offline] Found {len(chunks)} chunks. Starting Staged Processing.")
                                
                                # Store intermediate results: {chunk_id: ChunkResult} or {chunk_id: masks}
                                chunk_results = {} 
                                sam3_masks_cache = {} # {chunk_id: masks}

                                # CHECK REFINEMENT MODE
                                # If all chunks have PLY and Meta, we skip DA3 and Geometry gen.
                                can_refine = True
                                for c in chunks:
                                    ply_exists = (frame_storage.current_session.output_dir / f"chunk_{c.chunk_id:03d}.ply").exists()
                                    meta_exists = (frame_storage.current_session.output_dir / f"chunk_{c.chunk_id:03d}_meta.json").exists()
                                    if not (ply_exists and meta_exists):
                                        can_refine = False
                                        break
                                
                                refine_mode = can_refine
                                if refine_mode:
                                     print(f"[Retro-Offline] 🟢 REFINEMENT MODE DETECTED. Preserving Geometry, Recalculating Segmentation.")
                                else:
                                     print(f"[Retro-Offline] 🟠 FULL RE-PROCESS MODE. Regenerating Depth & Geometry (Use with caution).")

                                # --- PASS 1: DEPTH ESTIMATION (DA3) ---
                                if not refine_mode:
                                    print("\n[Retro-Offline] === PHASE 1: Depth Estimation (DA3) ===")
                                    chunk_processor.load_model() # Load ONCE
                                    
                                    for chunk_info in chunks:
                                        chunk_id = chunk_info.chunk_id
                                        frames_dir = frame_storage.get_chunk_frames_dir(chunk_info)
                                        print(f"[Retro-Offline] Depth Pass: Chunk {chunk_id}...")
                                        
                                        res = chunk_processor.process_chunk(frames_dir, chunk_id, prompt=None)
                                        
                                        if res:
                                            # Restore poses if partial re-run?
                                            # If full re-process, we might want to restore original poses anyway if available
                                            # to avoid "Stacking" if alignment fails.
                                            old_meta = frame_storage.load_chunk_metadata(session, chunk_id)
                                            if old_meta and "extrinsics" in old_meta:
                                                try:
                                                    res.extrinsics = np.array(old_meta["extrinsics"], dtype=np.float32)
                                                    res.intrinsics = np.array(old_meta["intrinsics"], dtype=np.float32)
                                                    print(f"[Retro-Offline] Restored original poses for Chunk {chunk_id}")
                                                except: pass

                                            chunk_results[chunk_id] = res
                                        else:
                                            print(f"[Retro-Offline] Failed to process depth for Chunk {chunk_id}")
                                    
                                    # Unload DA3 completely
                                    chunk_processor.unload_model()
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    print("[Retro-Offline] DA3 Unloaded. VRAM Cleared.")
                                else:
                                    print("[Retro-Offline] Skipping Phase 1 (Geometry exists).")
                                
                                # --- PASS 2: SEGMENTATION (SAM3) ---
                                if prompt:
                                    print("\n[Retro-Offline] === PHASE 2: Segmentation (SAM3) ===")
                                    sam3 = get_sam3_wrapper() # Load ONCE
                                    
                                    for chunk_info in chunks:
                                        chunk_id = chunk_info.chunk_id
                                        
                                        # In refinement mode, we process all chunks.
                                        # In full mode, only successful depth chunks.
                                        if not refine_mode and chunk_id not in chunk_results: continue
                                        
                                        frames_dir = frame_storage.get_chunk_frames_dir(chunk_info)
                                        print(f"[Retro-Offline] Segmentation Pass: Chunk {chunk_id}...")
                                        
                                        try:
                                            sam3_masks = sam3.process_chunk(str(frames_dir), prompt, keyframe_interval=5)
                                            if not refine_mode:
                                                chunk_results[chunk_id].segmentation_masks = sam3_masks
                                            else:
                                                sam3_masks_cache[chunk_id] = sam3_masks
                                        except Exception as e:
                                            print(f"[Retro-Offline] SAM3 Error for Chunk {chunk_id}: {e}")
                                    
                                    # Unload SAM3 completely
                                    sam3.unload_model()
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    print("[Retro-Offline] SAM3 Unloaded. VRAM Cleared.")
                                
                                # --- PASS 3: FUSION / PROJECTION & BROADCAST ---
                                print("\n[Retro-Offline] === PHASE 3: Fusion & Alignment ===")
                                
                                if not refine_mode:
                                    alignment_manager.reset()
                                
                                for chunk_info in chunks:
                                    chunk_id = chunk_info.chunk_id
                                    
                                    if refine_mode:
                                        # REFINEMENT LOGIC
                                        if chunk_id not in sam3_masks_cache: continue
                                        print(f"[Retro-Offline] Projecting Masks for Chunk {chunk_id}...")
                                        
                                        # Load PLY & Meta
                                        ply_points = frame_storage.load_ply(session, chunk_id)
                                        meta = frame_storage.load_chunk_metadata(session, chunk_id)
                                        masks = sam3_masks_cache[chunk_id]
                                        
                                        if ply_points is not None and meta:
                                            segments = chunk_processor.compute_segmentation_for_ply(ply_points, meta, masks)
                                            seg_path = frame_storage.save_segments_direct(session, chunk_id, segments, prompt=prompt)
                                            
                                            # Broadcast Segments ONLY
                                            if seg_path and os.path.exists(seg_path):
                                                try:
                                                    with open(seg_path, 'r') as f:
                                                        seg_data = json.load(f)
                                                    seg_data["type"] = "segmentation"
                                                    asyncio.run_coroutine_threadsafe(
                                                        viewer_manager.broadcast_text(json.dumps(seg_data)), main_loop
                                                    )
                                                except Exception as e:
                                                    print(f"[Retro-Offline] Broadcast Error: {e}")
                                    else:
                                        # FULL RE-PROCESS LOGIC
                                        if chunk_id not in chunk_results: continue
                                        result = chunk_results[chunk_id]
                                        print(f"[Retro-Offline] Fusing Chunk {chunk_id}...")
                                        
                                        # Generate Point Cloud
                                        chunk_processor.generate_point_cloud(result)
                                        
                                        # Align
                                        aligned_chunk = alignment_manager.add_chunk(result)
                                        
                                        # Save & Broadcast
                                        if aligned_chunk:
                                            frame_storage.save_ply(session, chunk_id, aligned_chunk.point_cloud)
                                            # Get alignment transform and validity mapping for retroactive segmentation
                                            align_transform = alignment_manager.gravity_correction if hasattr(alignment_manager, 'gravity_correction') else None
                                            validity_map = alignment_manager.generate_validity_mapping(chunk_id)
                                            frame_storage.save_chunk_metadata(session, chunk_id, result, 
                                                                             alignment_transform=align_transform,
                                                                             validity_mapping=validity_map)
                                            # Save segmentation with new format (point_indices)
                                            # Masks can be in sam3_masks_cache (refine) or result.segmentation_masks (full)
                                            seg_path = None
                                            masks_for_save = sam3_masks_cache.get(chunk_id) or result.segmentation_masks
                                            if masks_for_save:
                                                depth_shape = result.depths[0].shape  # (H, W)
                                                seg_path = frame_storage.save_chunk_segmentation(
                                                    session, chunk_id,
                                                    masks=masks_for_save,
                                                    prompt=prompt,
                                                    depth_shape=depth_shape,
                                                    frame_count=result.frame_count,
                                                    depths=result.depths,  # Pass for validity mask
                                                    confs=result.confs     # Pass for validity mask
                                                )
                                            
                                            # Live Broadcast (with proper await via .result())
                                            if aligned_chunk.point_cloud is not None:
                                                try:
                                                    # Chunk Start
                                                    future1 = asyncio.run_coroutine_threadsafe(
                                                        viewer_manager.broadcast_text(json.dumps({
                                                            "type": "chunk_start",
                                                            "chunk_id": chunk_id,
                                                            "point_count": len(aligned_chunk.point_cloud)
                                                        })), main_loop
                                                    )
                                                    future1.result(timeout=5)  # Wait for completion
                                                    
                                                    # Binary
                                                    binary = cloud_to_binary(aligned_chunk.point_cloud)
                                                    future2 = asyncio.run_coroutine_threadsafe(
                                                        viewer_manager.broadcast_binary(binary), main_loop
                                                    )
                                                    future2.result(timeout=5)  # Wait for completion
                                                    
                                                    # Segmentation
                                                    if seg_path and os.path.exists(seg_path):
                                                        with open(seg_path, 'r') as f:
                                                            seg_data = json.load(f)
                                                        seg_data["type"] = "segmentation"
                                                        future3 = asyncio.run_coroutine_threadsafe(
                                                            viewer_manager.broadcast_text(json.dumps(seg_data)), main_loop
                                                        )
                                                        future3.result(timeout=5)  # Wait for completion
                                                except Exception as e:
                                                    print(f"[Retro-Offline] Broadcast Error: {e}")
                                        
                                        # Cleanup Result Object to free RAM
                                        del result
                                
                                chunk_results.clear()
                                sam3_masks_cache.clear()
                                gc.collect()
                                print("[Retro-Offline] DONE.")
                                return True
                                
                            except Exception as e:
                                print(f"[Retro-Offline] Error: {e}")
                                import traceback
                                traceback.print_exc()
                                return False

                        success = await asyncio.get_event_loop().run_in_executor(None, _retro_process_offline)
                        
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
                    ply_files = sorted(output_dir.glob("*.ply"))
                    
                    if not ply_files:
                        await websocket.send_text(json.dumps({"type": "error", "message": "No point clouds found for this session"}))
                    else:
                        print(f"[Viewer] Found {len(ply_files)} PLY files")
                        for ply_path in ply_files:
                            try:
                                cid_str = ply_path.stem.split('_')[1]
                                chunk_id = int(cid_str)
                                
                                # Use unified loader to ensure alignment/parsing match Refinement Mode
                                if frame_storage.current_session:
                                    data = frame_storage.load_ply(frame_storage.current_session, chunk_id)
                                    if data is None: continue
                                    
                                    # 1. Send Chunk Start
                                    await websocket.send_text(json.dumps({
                                        "type": "chunk_start",
                                        "chunk_id": chunk_id,
                                        "point_count": len(data)
                                    }))
                                    
                                    # 2. Send Binary Cloud
                                    binary = cloud_to_binary(data)
                                    await websocket.send_bytes(binary)
                                    
                                    # 3. Send Segmentation (if exists)
                                    seg_path = output_dir / f"chunk_{chunk_id:03d}_segments.json"
                                    if seg_path.exists():
                                        try:
                                            with open(seg_path, 'r') as f:
                                                seg_data = json.load(f)
                                            seg_data["type"] = "segmentation"
                                            await websocket.send_text(json.dumps(seg_data))
                                        except Exception as e:
                                            print(f"[Viewer] Seg send error: {e}")
                                    
                                    await asyncio.sleep(0.01)
                                    
                            except Exception as e:
                                print(f"[Viewer] Error loading chunk {ply_path}: {e}")
                        
                        await websocket.send_text(json.dumps({"type": "status", "message": f"Loaded {len(ply_files)} chunks from {session_id}"}))

                except Exception as e:
                    print(f"Error loading session: {e}")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

    except:
        viewer_manager.disconnect_viewer(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=Config.HOST, port=Config.PORT, log_level="info")