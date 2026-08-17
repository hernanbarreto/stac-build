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

    # ONE execution path (user mandate 2026-08-17: "debe funcionar exactamente
    # igual"). The stage spawns the SAME run_tsdf_scene.py subprocess the UI
    # button uses — same logger capture (TSDFExport, CloudMesh, MeshFusion,
    # TextureBake into run.log + stream), same faulthandler, same OMP cap.
    # The previous in-process call ran the mesher BLIND (its loggers were not
    # forwarded) and WITHOUT those protections — a silent death mid-stage left
    # no trace. Never let the two paths diverge again.
    import json as _json
    import subprocess as _sp
    import sys as _sys
    _script = Path(__file__).resolve().parent.parent / "run_tsdf_scene.py"
    _cmd = [_sys.executable, str(_script),
            "--output-dir", str(output_dir),
            "--frames-dir", str(frames_dir),
            "--session-dir", str(session_path),
            "--params", _json.dumps(kwargs)]
    import os as _os
    _proc = _sp.Popen(_cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                      cwd=str(Path(__file__).resolve().parent.parent),
                      env={**_os.environ, "PYTHONUNBUFFERED": "1",
                           "PYTHONFAULTHANDLER": "1"})
    path = None
    assert _proc.stdout is not None
    for _line in _proc.stdout:
        _line = _line.rstrip()
        if not _line:
            continue
        if _line.startswith("[TSDF-PROGRESS]"):
            try:
                _upd = _json.loads(_line[len("[TSDF-PROGRESS]"):])
                _cb(_upd.get("phase"), _upd.get("elapsed"))
            except Exception:
                pass
        elif _line.startswith("[TSDF-RESULT]"):
            _r = _line[len("[TSDF-RESULT]"):].strip()
            path = None if _r == "NONE" else _r
        else:
            pipe.send_log(_line)
    _proc.wait()
    if _proc.returncode != 0:
        raise RuntimeError(f"TSDF subprocess died (exit {_proc.returncode}) — "
                           f"see run.log / fault log in output/tsdf/scene/")
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
