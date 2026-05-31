"""
Whole-scene TSDF stage — final pipeline step, runs AFTER CloudCompy (cleaned_cloud
+ Potree) so one pipeline run produces cloud → Potree → TSDF mesh in one go.

Integrates every posed frame's depth into a single TSDF volume on the GPU and bakes
photo colour, writing output/tsdf/scene/scene.glb (+ scene.meta.json). For the
pose_source=vipe pipeline the depth source is DA3 metric depth (da3_depth/, used
directly) and the poses are ViPE's, scaled by the single global g — resolved
automatically inside export_tsdf_scene (_resolve_vipe_depth + camera_frames.txt).

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


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_tsdf_work, conn, session_dir, config)
