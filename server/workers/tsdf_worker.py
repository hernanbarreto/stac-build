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

    # NO FALLBACK: the TSDF mesh is a required output — fail if its inputs are missing.
    if not output_dir.exists():
        raise FileNotFoundError(f"No output/ dir at {output_dir} — TSDF cannot run")
    # The whole-scene TSDF needs the cleaned cloud (mask) from CloudCompy.
    if not (output_dir / "cleaned_cloud.ply").exists():
        raise FileNotFoundError("cleaned_cloud.ply not found — the CloudCompy stage did not "
                                "produce a cloud; TSDF cannot run")

    tcfg = config.get("tsdf", {}) or {}
    # Single source of truth (shared with main.py /tsdf/scene_export) → both run
    # paths forward the SAME config keys to export_tsdf_scene, never diverging.
    from segmentation.tsdf_export import build_tsdf_scene_kwargs
    kwargs = build_tsdf_scene_kwargs(config)

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
    pipe.send_log(f"TSDF scene mesh: {path}")

    # ── Per-object textured meshes (the segmentation "cut") ──
    # In the anchored order SAM3 runs BEFORE this stage, so its own crop attempt
    # finds no scene mesh and skips ("No scene TSDF mesh"). Now the scene mesh
    # exists → carve each instance's textured mesh out of it, the same faithful
    # deliverable the manual Segmentation Manager flow produces via
    # /api/segmentation/refresh. Best-effort: never fail the TSDF stage.
    try:
        import json as _json
        result_path = output_dir / "segmentation_result.json"
        if result_path.exists():
            segments_result = _json.loads(result_path.read_text())
            n_tot = len(segments_result.get("instances", []))
            if n_tot:
                pipe.send_progress(97, f"Carving {n_tot} per-object TSDF meshes...",
                                   stage="tsdf")
                from segmentation.tsdf_export import crop_scene_mesh_to_instances

                def _crop_progress(inst_id, phase, elapsed, mesh_path):
                    if phase == "done":
                        _crop_progress.done += 1
                        pipe.send_progress(min(97 + (_crop_progress.done / n_tot) * 3, 99),
                                           f"Object mesh {_crop_progress.done}/{n_tot}",
                                           stage="tsdf")
                _crop_progress.done = 0

                written = crop_scene_mesh_to_instances(
                    output_dir=output_dir,
                    segments_result=segments_result,
                    progress_cb=_crop_progress,
                )
                pipe.send_log(f"Per-object TSDF: wrote {len(written)} textured mesh(es)")
        else:
            pipe.send_log("No segmentation_result.json — per-object TSDF crop skipped "
                          "(session without segmentation)")
    except Exception as e:
        pipe.send_log(f"Per-object TSDF crop failed (non-fatal): {e}", level="warning")

    pipe.send_progress(100, f"TSDF mesh written: {Path(path).name}", stage="tsdf")

    # Cascade cleanup (step 3/3): the aligned chunk depth (maplong_run/_tmp_results_aligned)
    # is the MapAnything dense depth+K that MATCHES cleaned_cloud.ply — it is REQUIRED to
    # re-run the TSDF stage (param tuning, dense integration) without re-running MapAnything
    # (hours). Default KEEP. Set tsdf.keep_aligned_chunks: false ONLY to reclaim disk when
    # you are certain you will never re-mesh this scan.
    if not tcfg.get("keep_aligned_chunks", True):
        import shutil
        aligned = output_dir / "maplong_run" / "_tmp_results_aligned"
        if aligned.exists():
            size_mb = sum(f.stat().st_size for f in aligned.rglob("*") if f.is_file()) / (1024 * 1024)
            shutil.rmtree(aligned, ignore_errors=True)
            pipe.send_log(f"[cleanup] removed _tmp_results_aligned/ ({size_mb:.0f} MB freed) "
                          f"— TSDF re-runs will now require re-running MapAnything")
    else:
        pipe.send_log("[cleanup] keeping _tmp_results_aligned/ (MapAnything dense depth) "
                      "so the TSDF stage can be re-run without re-running MapAnything")


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_tsdf_work, conn, session_dir, config)
