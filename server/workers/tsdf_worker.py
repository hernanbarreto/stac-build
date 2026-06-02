"""
Whole-scene TSDF stage — final pipeline step, runs AFTER CloudCompy (cleaned_cloud
+ Potree) so one pipeline run produces cloud → Potree → TSDF mesh in one go.

Integrates every posed frame's depth into a single TSDF volume on the GPU and bakes
photo colour, writing output/tsdf/scene/scene.glb (+ scene.meta.json). Depth + poses
are resolved automatically inside export_tsdf_scene per backend (mapanything chunk
.npy / DA3 / LiDAR; poses keyed by real frame number via camera_frames.txt).

Runs in the SERVER env (Open3D-CUDA), spawned by PipelineManager like the other
workers via run(conn, session_dir, config).
"""
from multiprocessing.connection import Connection
from pathlib import Path

from workers.base import WorkerPipe, run_worker_safe

# export_tsdf_scene kwargs we forward from config["tsdf"] when present.
_PASSTHROUGH = (
    "voxel_length", "sdf_trunc", "depth_trunc", "depth_min", "edge_thresh",
    "conf_min", "da3_conf_percentile", "mask_to_cleaned_cloud",
    "cleaned_cloud_dilate", "smooth_iterations", "smooth_method",
    "decimate_target", "texture", "texture_mode", "tsdf_block_count",
    "tsdf_weight_thresh",
)

# Coarse phase → percentage mapping for the UI progress bar.
_PHASE_PCT = {
    "starting": 2, "loading": 5, "masking": 12, "integrating": 45,
    "extracting": 70, "smoothing": 80, "texturing": 90, "writing": 96,
    "done": 100,
}


def _tsdf_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir)
    frames_dir = (session_path / "frames").resolve()
    output_dir = (session_path / "output").resolve()

    if not output_dir.exists():
        pipe.send_log("No output/ dir, skipping TSDF", level="warning")
        return
    # The whole-scene TSDF needs the cleaned cloud (mask) from CloudCompy.
    if not (output_dir / "cleaned_cloud.ply").exists():
        pipe.send_log("cleaned_cloud.ply not found (CloudCompy stage missing?), "
                      "skipping TSDF", level="warning")
        return

    tcfg = config.get("tsdf", {}) or {}
    kwargs = {k: tcfg[k] for k in _PASSTHROUGH if k in tcfg}

    def _cb(phase, elapsed=None, mesh=None):
        pct = _PHASE_PCT.get(str(phase), None)
        msg = f"TSDF: {phase}" + (f" ({elapsed:.0f}s)" if elapsed else "")
        if pct is not None:
            pipe.send_progress(pct, msg, stage="tsdf")
        else:
            pipe.send_log(msg)

    pipe.send_progress(0, "Starting whole-scene TSDF...", stage="tsdf")
    pipe.send_log(f"TSDF kwargs (overrides): {kwargs}" if kwargs
                  else "TSDF using export defaults")

    # Forward the export_tsdf_scene logger ("TSDFExport") to server.log via the pipe so
    # its coverage diagnostics (n_integrated / skipped / mask points / depth source /
    # "integrating unmasked") are visible — otherwise they only hit the worker console.
    import logging as _logging
    _tsdf_logger = _logging.getLogger("TSDFExport")

    class _PipeLogHandler(_logging.Handler):
        def emit(self, record):
            try:
                pipe.send_log(f"[TSDF] {record.getMessage()}",
                              level="warning" if record.levelno >= _logging.WARNING else "info")
            except Exception:
                pass

    _ph = _PipeLogHandler()
    _ph.setLevel(_logging.INFO)
    if not any(isinstance(h, _PipeLogHandler) for h in _tsdf_logger.handlers):
        _tsdf_logger.addHandler(_ph)
    _tsdf_logger.setLevel(_logging.INFO)

    from segmentation.tsdf_export import export_tsdf_scene
    path = export_tsdf_scene(
        output_dir=output_dir,
        frames_dir=frames_dir,
        session_dir=session_path,
        progress_cb=_cb,
        **kwargs,
    )
    if not path:
        raise RuntimeError("export_tsdf_scene produced no mesh (no depth source / "
                           "no integrable frames)")
    pipe.send_progress(100, f"TSDF mesh written: {Path(path).name}", stage="tsdf")
    pipe.send_log(f"TSDF scene mesh: {path}")

    # Cascade cleanup (step 3/3): the mesh is built, so the aligned chunk depth
    # (maplong_run/_tmp_results_aligned, ~tens of GB) is no longer needed. Default ON.
    # Set tsdf.keep_aligned_chunks: true to retain it for TSDF param re-tuning.
    if not tcfg.get("keep_aligned_chunks", False):
        import shutil
        aligned = output_dir / "maplong_run" / "_tmp_results_aligned"
        if aligned.exists():
            size_mb = sum(f.stat().st_size for f in aligned.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(aligned, ignore_errors=True)
            pipe.send_log(f"[cleanup] removed _tmp_results_aligned/ ({size_mb:.0f} MB freed) "
                          f"— set tsdf.keep_aligned_chunks: true to keep for re-tuning")


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_tsdf_work, conn, session_dir, config)
