"""A geometry epoch of POSES ONLY — what F2 (gauge) and F5 (refine) publish.

USER 2026-09-29: the core needs no cloud until F7 — F0 reads Omega's K, F2 reads the
DA3 windows and Omega's depth, F4 the frames, F5 the tracks, F6 Omega's depth × the
measured scale with F5's poses, and F7 builds the cloud from F6. The previous path
(``correction.visit_drift_run.apply_transform_epoch``) warped a working cloud merged
from Omega's chunks and rebuilt its octree, twice, for a cloud F7 discards — and needed
that cloud merged and FILTERED before the core ran ("¿por qué filtramos una nube que no
vamos a usar todavía?").

What an epoch of poses carries, through the same journaled swap as every epoch:
``camera_poses.txt`` (+ the copies that exist), the cumulative depth-correction sidecar
(the per-keyframe k of the transform, the rule of ``stage_transaction`` step 5),
``scale_diagnostics.json`` regenerated, ``geometry_epoch.json`` and the exact
per-keyframe transform ``corrections/epoch_<N>.npz`` (replayable). No cloud, no
octree, no instance store.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


class PosesEpochError(RuntimeError):
    """The poses cannot be published — with the exact reason."""


def load_poses(output_dir: Path) -> Tuple[List[int], np.ndarray]:
    """(keyframe frame numbers, (N,4,4) c2w) from camera_frames.txt / frame_list.json
    and camera_poses.txt — the two files a reconstruction always leaves; consistent
    or an error naming the mismatch."""
    from correction.session import _load_frame_index_map, read_poses
    out = Path(output_dir)
    frames = _load_frame_index_map(out)
    if not frames:
        raise PosesEpochError(f"{out}/camera_frames.txt (or frame_list.json) is missing — "
                              f"the keyframe↔frame mapping is mandatory")
    poses_path = out / "camera_poses.txt"
    if not poses_path.exists():
        raise PosesEpochError(f"{poses_path} does not exist — no camera poses")
    poses = read_poses(poses_path)
    if len(poses) != len(frames):
        raise PosesEpochError(f"camera_poses.txt has {len(poses)} rows but camera_frames.txt "
                              f"lists {len(frames)} keyframes — the session is inconsistent")
    return [int(f) for f in frames], poses


def apply_pose_epoch(output_dir: Path, R_kf: np.ndarray, t_kf: np.ndarray,
                     k_kf: np.ndarray, kind: str, diagnosis: Sequence[dict],
                     log: Callable[[str], None] = print) -> dict:
    """Publish the per-keyframe transform (R, t about the world; k = depth scale about
    the camera, recorded) as the next geometry epoch — poses and records only."""
    from correction import diagnose as diag_mod
    from correction import ledger
    from correction.apply import (TX_PREFIX, assert_no_interrupted_swap, next_epoch,
                                  swap_transaction, transform_poses)
    from correction.epoch import EPOCH_FILE, current_epoch, make_epoch_record
    from correction.ledger import EPOCH_NPZ_DIR, save_epoch_npz
    from correction.session import POSE_COPY_RELPATHS, read_poses, write_poses
    from segmentation.session_io import DEPTH_CORRECTION_NAME, load_depth_affine

    out = Path(output_dir)
    assert_no_interrupted_swap(out)
    frames, poses = load_poses(out)
    n = len(frames)
    R_kf = np.asarray(R_kf, np.float64)
    t_kf = np.asarray(t_kf, np.float64)
    k_kf = np.asarray(k_kf, np.float64)
    if R_kf.shape != (n, 3, 3) or t_kf.shape != (n, 3) or k_kf.shape != (n,):
        raise PosesEpochError(f"{kind}: transform shapes {R_kf.shape} / {t_kf.shape} / "
                              f"{k_kf.shape} do not match the {n} keyframes")
    cid = ledger.new_correction_id()
    epoch_from, epoch_to = int(current_epoch(out)), int(next_epoch(out))
    tx = out / f"{TX_PREFIX}{epoch_to}"
    if tx.exists():
        shutil.rmtree(tx)
    tx.mkdir(parents=True)
    arts: List[dict] = []

    def art(rel: str) -> None:
        arts.append({"rel": rel, "existed_before": (out / rel).exists()})

    # 1) poses: the canonical file and every copy that exists with the same rows
    poses_new = transform_poses(poses, R_kf, t_kf)
    write_poses(tx / "camera_poses.txt", poses_new)
    art("camera_poses.txt")
    skipped: List[str] = []
    for rel in POSE_COPY_RELPATHS:
        if rel == "camera_poses.txt" or not (out / rel).exists():
            continue
        rows = read_poses(out / rel)
        if len(rows) != n:
            skipped.append(f"{rel} ({len(rows)} rows)")
            continue
        (tx / rel).parent.mkdir(parents=True, exist_ok=True)
        write_poses(tx / rel, transform_poses(rows, R_kf, t_kf))
        art(rel)
    if skipped:
        log(f"  pose copies skipped (declared): {skipped}")

    # 2) depth-correction sidecar: cumulative k per keyframe (stage_transaction step 5)
    old_kb = load_depth_affine(out) or {}
    new_kb = dict(old_kb)
    for i in range(n):
        kv = float(k_kf[i])
        if kv != 1.0:
            k0, b0 = new_kb.get(frames[i], (1.0, 0.0))
            new_kb[frames[i]] = (k0 * kv, kv * b0)
    if new_kb or old_kb:
        (tx / DEPTH_CORRECTION_NAME).write_text(json.dumps(
            {"version": 2, "epoch": epoch_to,
             "k": {str(f): round(v[0], 6) for f, v in sorted(new_kb.items())},
             "b": {str(f): round(v[1], 6) for f, v in sorted(new_kb.items())}}, indent=1))
        art(DEPTH_CORRECTION_NAME)

    # 3) scale diagnostics regenerated for the epoch
    diag = diag_mod.regenerate_scale_diagnostics(out, {}, epoch_to, cid)
    if diag is not None:
        (tx / "scale_diagnostics.json").write_text(json.dumps(diag, indent=2))
        art("scale_diagnostics.json")

    # 4) the epoch record and its exact, replayable transform
    (tx / EPOCH_FILE).write_text(json.dumps(make_epoch_record(epoch_to, cid, epoch_from),
                                            indent=1))
    art(EPOCH_FILE)
    save_epoch_npz(out, epoch_to, R_kf, t_kf, k_kf, frames, dir_override=tx / EPOCH_NPZ_DIR)
    art(f"{EPOCH_NPZ_DIR}/epoch_{epoch_to}.npz")

    swap_transaction(out, {"tx_dir": str(tx), "epoch_from": epoch_from,
                           "epoch_to": epoch_to, "artifacts": arts}, log=log)
    moved = float(np.linalg.norm(poses_new[:, :3, 3] - poses[:, :3, 3], axis=1).max()) if n else 0.0
    rec = {"epoch": epoch_to, "corrected_by": kind, "poses_only": True,
           "cameras_moved_max_m": moved, "diagnosis": list(diagnosis),
           "provenance": "tool_measured"}
    rp = out / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1, default=float))
    ledger.record_run(out, correction_id=cid, epoch_from=epoch_from, epoch_to=epoch_to,
                      kind=kind, operator="auto", instance_ids=[], visits=[],
                      observability=[], anchors=[], diagnosis=list(diagnosis), gates=[],
                      overrides={}, verdict="applied",
                      report_path=str(rp.relative_to(out)))
    log(f"[{kind}] poses published → epoch {epoch_to} (cameras moved ≤ {moved * 100:.1f} cm; "
        f"no cloud: F7 builds it)")
    return {"correction_id": cid, "epoch_to": epoch_to, "cameras_moved_max_m": moved}
