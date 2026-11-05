# STAC-Builder: Floor Worker (Subprocess)
# USER 2026-11-05 ("el piso lo acomodaba la certificación? hay que agregar al pipeline eso, la corrección de piso
# después — toda esa mierda del acta no, pero lo del piso sí, porque está inclinada la nube"; "sí, implementalo,
# adelante"): the certification's floor correction ALONE, after the mask projection.
#
# correction.run.run_floor with the model `session` (correction/floor.py solve_floor_session): the segmentation's
# support surfaces over the whole walk → the level under the most cameras; its normal → up about its centroid when the
# tilt clears its own error, the walk's start at y = 0 when the height does. ONE rigid motion for every keyframe: it
# straightens the cloud, it never deforms it. No closures, no depth per chunk, no acta — and none of the epoch's other
# work: the mask filter and the reconsolidation are OFF in this transaction (a rigid motion leaves nothing for them).
#
# Published the way the certification published its floor: one transactional epoch (cloud, poses, segmentation with
# its boxes recomputed, octree, instance store); the epoch it replaced is discarded right after — the session keeps
# ONE cloud, the final one. The projection before this stage builds no octree (pipeline_manager.certify_follows): the
# epoch brings it. A floor already level within its own error publishes no epoch, and the octree of the live cloud is
# built here (segmentation.pipeline.ensure_live_octree) — one octree either way.

import json
import sys
from dataclasses import replace
from pathlib import Path
from multiprocessing.connection import Connection
from typing import Optional, Tuple

from workers.base import WorkerPipe, run_worker_safe

STAGE = "floor"
RECORD = "floor_session.json"


def _frozen_config(session_path: Path, handed: dict, log) -> tuple:
    """(the configuration this stage runs with, its sha256): the job's frozen output/run_config.yaml with the
    stage's overrides, verified against the dict the manager handed this process (point 139)."""
    from intake.run_config import canonical_yaml, effective_config, load_run_config
    frozen, sha = load_run_config(session_path)
    raw = effective_config(frozen, STAGE)
    mine = {k: v for k, v in handed.items() if not str(k).startswith("_")}
    for key in sorted(set(mine) | set(raw)):
        if canonical_yaml({key: mine.get(key)}) != canonical_yaml({key: raw.get(key)}):
            raise RuntimeError(f"the configuration handed to the floor stage differs from the job's frozen "
                               f"output/run_config.yaml at '{key}' — a stage never runs on a configuration "
                               f"nobody froze (point 139)")
    log(f"[floor] configuration: the job's frozen run_config.yaml (sha256 {sha[:12]})")
    return raw, sha


def _live_epoch(output_dir: Path) -> dict:
    """The live geometry: geometry_epoch.json's epoch and correction (None/None at epoch 0 without a record)."""
    try:
        d = json.loads((Path(output_dir) / "geometry_epoch.json").read_text())
        return {"epoch": d.get("epoch"), "correction_id": d.get("correction_id")}
    except (OSError, ValueError):
        return {"epoch": None, "correction_id": None}


def _segmentation_sha(output_dir: Path) -> Optional[str]:
    from repro import sha256_file
    p = Path(output_dir) / "segmentation_result.json"
    return sha256_file(p) if p.is_file() else None


def floor_is_current(output_dir) -> Tuple[bool, str]:
    """(complete, reason) — the stage's resume probe: its record names the live geometry it left and the
    segmentation it was found on, and both are still the session's."""
    out = Path(output_dir)
    p = out / RECORD
    if not p.is_file():
        return False, "no floor record"
    try:
        rec = json.loads(p.read_text())
    except ValueError:
        return False, "the floor record is unreadable"
    if rec.get("live_epoch") != _live_epoch(out):
        return False, "the live geometry is not the one the floor stage left"
    if rec.get("segmentation_sha256") != _segmentation_sha(out):
        return False, "the segmentation was projected again after the floor stage"
    return True, f"floor stage done on this geometry ({rec.get('status')})"


def _write_record(out: Path, **fields) -> dict:
    from atomic_io import atomic_write_json
    doc = {"version": 1, "stage": STAGE, **fields,
           "live_epoch": _live_epoch(out), "segmentation_sha256": _segmentation_sha(out)}
    atomic_write_json(out / RECORD, doc, indent=1, sort_keys=True)
    return doc


def _floor_work(pipe: WorkerPipe, session_dir: str, config: dict):
    session_path = Path(session_dir)
    out = (session_path / "output").resolve()
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    config, config_sha = _frozen_config(session_path, config, pipe.send_log)

    def log(m):
        pipe.send_log(f"[floor] {m}")

    if not (out / "cleaned_cloud.ply").exists():
        raise RuntimeError(f"No cleaned_cloud.ply in {out} — the floor stage levels the published cloud")
    if not (out / "camera_poses.txt").exists():
        raise RuntimeError(f"No camera_poses.txt in {out} — the reconstruction left no poses")
    from segmentation.pipeline import ensure_live_octree
    if not (out / "segmentation_result.json").exists():
        log("no segmentation_result.json — the floor is found on the segmented support: nothing is moved "
            "(declared)")
        ensure_live_octree(out, log=pipe.send_log)
        _write_record(out, status="no segmentation", run_config_sha256=config_sha)
        pipe.send_progress(100, "Floor: no segmentation to find it on", stage=STAGE)
        return

    import numpy as np
    from correction.config import load_correction_config
    from correction.floor import solve_floor_session
    from correction.session import load_session
    cfg = load_correction_config(config)
    # the floor's own transaction: the rigid motion and nothing else (the mask filter deletes points, the
    # reconsolidation moves them — neither belongs to a floor correction)
    cfg = replace(cfg, apply=replace(cfg.apply, mask_filter=False, reconsolidate=False))

    pipe.send_progress(5, "Floor: finding the session's floor on the segmentation...", stage=STAGE)
    session = load_session(out)
    sol = solve_floor_session(session, cfg, log=log)
    del session
    R = np.asarray(sol["R_kf"][0], np.float64)
    t = np.asarray(sol["t_kf"][0], np.float64)
    if not (np.allclose(sol["R_kf"], R) and np.allclose(sol["t_kf"], t)):
        raise RuntimeError("the floor model `session` returned more than one motion — the floor stage applies "
                           "ONE rigid motion for the session")
    params = sol.get("model_params") or {}
    if np.array_equal(R, np.eye(3)) and not np.any(t):
        log("the floor is level within its own error (tilt and height) — no epoch is published")
        ensure_live_octree(out, log=pipe.send_log)
        _write_record(out, status="level", model_params=params, run_config_sha256=config_sha)
        pipe.send_progress(100, "Floor already level", stage=STAGE)
        return

    from correction.run import run_floor
    report = run_floor(out, "session", None, "pipeline:floor", log=log,
                       progress=lambda p, m: pipe.send_progress(10 + int(p * 0.85), m, stage=STAGE),
                       cfg=cfg, record_extra={"run_config_sha256": config_sha, "stage": STAGE})
    if report.get("status") != "applied":
        raise RuntimeError(f"the floor correction was not applied: {report.get('rejection_reason')}")
    # ONE cloud: the epoch the floor replaced goes (the precision core's pose-only epochs already went)
    from workers.map_worker import _discard_previous_epochs
    freed = _discard_previous_epochs(out)
    ensure_live_octree(out, log=pipe.send_log)      # the epoch carries its octree: verified, never rebuilt
    _write_record(out, status="applied", correction_id=report.get("correction_id"),
                  epoch_to=report.get("epoch_to"), model_params=params, run_config_sha256=config_sha)
    tilt = params.get("tilt_deg") if params.get("tilt_applied") else 0.0
    dy = params.get("offset_m") if params.get("height_applied") else 0.0
    log(f"applied: tilt {float(tilt or 0):.2f}°, height {float(dy or 0) * 100:+.1f} cm — ONE rigid motion, epoch "
        f"{report.get('epoch_to')}; the previous epoch discarded ({freed / 1048576:.0f} MB)")
    pipe.send_progress(100, f"Floor levelled ({float(tilt or 0):.2f}°, {float(dy or 0) * 100:+.1f} cm)", stage=STAGE)


def run(conn: Connection, session_dir: str, config: dict):
    """Entry point called by PipelineManager."""
    run_worker_safe(_floor_work, conn, session_dir, config)
