# STAC-Builder: DA3 Worker (Subprocess)
# Runs DA3 dense 3D reconstruction in its own process.
# Reads frames from session directory, writes chunk PLYs.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import json
from pathlib import Path
from multiprocessing.connection import Connection

from workers.base import WorkerPipe, run_worker_safe


def _da3_work(pipe: WorkerPipe, session_dir: str, config: dict):
    """DA3 reconstruction — runs inside a dedicated subprocess."""

    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = config.get("slam_backend", "da3")

    pipe.send_log(f"Starting reconstruction (backend={backend})")
    pipe.send_progress(0, "Initializing...", stage="da3")

    if backend in ("mast3r", "hybrid"):
        # ── MASt3R / HYBRID path ──
        _run_mast3r(pipe, frames_dir, output_dir, backend, config)
    else:
        # ── Pure DA3 path ──
        _run_da3(pipe, frames_dir, output_dir, config, session_path)

    pipe.send_progress(100, "Reconstruction complete", stage="da3")


def _run_mast3r(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
                backend: str, config: dict):
    """MASt3R or Hybrid (MASt3R + DA3) reconstruction."""

    # Add server dir to path for imports
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    from slam_processor import SlamProcessor

    mode_name = "HYBRID" if backend == "hybrid" else "MAST3R"
    pipe.send_log(f"Mode: {mode_name}")

    slam = SlamProcessor(config)
    try:
        if not slam.is_initialized:
            pipe.send_progress(5, "Loading MASt3R model...", stage="da3")
            slam.initialize()

        slam.start_session("worker", frames_dir, output_dir)

        # Count total frames for progress
        frame_files = sorted(frames_dir.glob("frame_*.jpg")) + sorted(frames_dir.glob("frame_*.png"))
        total = max(len(frame_files), 1)

        pipe.send_progress(10, f"Processing {total} frames...", stage="da3")

        frame_count = 0
        keyframe_count = 0
        for result in slam.process_frames_directory(frames_dir):
            if pipe.check_cancel():
                pipe.send_log("Cancelled by user", level="warning")
                return
            frame_count += 1
            if result.is_keyframe:
                keyframe_count += 1
            pct = 10 + (frame_count / total) * 60
            pipe.send_progress(pct, f"Frame {frame_count}/{total} ({keyframe_count} kf)", stage="da3")

        if backend == "hybrid":
            pipe.send_progress(75, "Phase 2: DA3 dense depth...", stage="da3")
            success = slam.run_hybrid_densification(frames_dir, output_dir, on_chunk_callback=None)
            if not success:
                raise RuntimeError("DA3 hybrid densification returned no points")

        # Save PLY
        pipe.send_progress(90, "Saving point cloud...", stage="da3")
        ply_path = output_dir / "slam_reconstruction.ply"
        slam.save_pointcloud_ply(ply_path)
        slam.stop_session()

        pipe.send_log(f"{mode_name} complete: {frame_count} frames, {keyframe_count} keyframes")
    finally:
        del slam
        import gc; gc.collect()


def _load_scan_zoom(session_path: Path) -> float:
    """Load zoom_level from scan_meta.json. Returns 1.0 if not found."""
    scan_meta_path = session_path / "scan_meta.json"
    if scan_meta_path.exists():
        try:
            with open(scan_meta_path) as f:
                meta = json.load(f)
            zoom = float(meta.get("zoom_level", 1.0))
            return zoom
        except Exception:
            pass
    return 1.0


def _run_da3(pipe: WorkerPipe, frames_dir: Path, output_dir: Path,
             config: dict, session_path: Path):
    """Pure DA3 single-sequence reconstruction with session-level zoom correction."""

    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # ── Step 1: Frame quality analysis ──
    pipe.send_progress(2, "Analyzing frame quality...", stage="da3")
    try:
        from frame_quality import analyze_frames, save_manifest
        fq = analyze_frames(str(frames_dir))
        if "error" not in fq:
            save_manifest(str(frames_dir), fq)
    except Exception as e:
        pipe.send_log(f"Frame quality analysis skipped: {e}", level="warning")

    # ── Step 2: Load session zoom level ──
    zoom_level = _load_scan_zoom(session_path)
    if zoom_level != 1.0:
        pipe.send_log(f"Session zoom level: {zoom_level}x (from scan_meta.json)")
    else:
        pipe.send_log(f"Session zoom level: 1.0x (default)")

    # ── Step 3: Frame selection (visual novelty filter) ──
    from frame_selector import _load_valid_frame_list, select_keyframes
    valid_frames = _load_valid_frame_list(frames_dir)

    frame_sel_cfg = config.get("frame_selection", {})
    if frame_sel_cfg.get("enabled", False):
        try:
            pipe.send_progress(5, "Selecting keyframes...", stage="da3")
            sel = select_keyframes(str(frames_dir), frame_sel_cfg)
            pipe.send_log(f"Selected {sel['selected_count']}/{sel['total_frames']} keyframes")
        except Exception as e:
            pipe.send_log(f"Frame selection failed: {e}", level="warning")

    # ── Step 4: DA3 reconstruction (single sequence) ──
    from da3_config_builder import build_da3_config
    da3_config = build_da3_config(config)

    pipe.send_progress(10, "Initializing RealtimeDA3...", stage="da3")
    from da3_native_wrapper import RealtimeDA3
    from alignment_manager import AlignmentManager

    alignment = AlignmentManager()
    da3 = RealtimeDA3(
        image_dir=str(frames_dir),
        save_dir=str(output_dir),
        config=da3_config,
        alignment_manager=alignment,
        zoom_level=zoom_level,
    )

    total = max(len(da3.img_list), 1)
    chunk_size = config.get("server", {}).get("chunk_size", 20)
    chunk_overlap = config.get("server", {}).get("chunk_overlap", 4)
    chunk_step = max(chunk_size - chunk_overlap, 1)
    num_chunks = max((total - chunk_overlap) // chunk_step, 1)

    pipe.send_progress(15, f"Processing {total} images in ~{num_chunks} chunks...", stage="da3")

    import asyncio
    async def _on_chunk(chunk_id, sim3):
        images_done = min((chunk_id + 1) * chunk_step + chunk_overlap, total)
        pct = 15 + (images_done / total) * 80
        pipe.send_progress(pct, f"Chunk {chunk_id} complete ({images_done}/{total} images)", stage="da3")
        if pipe.check_cancel():
            raise KeyboardInterrupt("Cancelled")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(da3.process_long_sequence_async(callback=_on_chunk))
    finally:
        loop.close()

    pipe.send_log(f"DA3 complete: {total} images processed, zoom={zoom_level}x")
    del da3, alignment
    import gc; gc.collect()


# ── Process entry point ──────────────────────────────────────

def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager as multiprocessing target."""
    run_worker_safe(_da3_work, conn, session_dir, config)
