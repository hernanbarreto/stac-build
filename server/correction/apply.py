"""Transactional application of a distributed correction.

Contract (prompt §7): every artifact of the new epoch is staged under
``output/_tx_epoch_<N>/`` — cloud, raw cloud, poses (canonical + copies),
segmentation OBBs, depth-correction sidecar, regenerated scale diagnostics,
geometry epoch, exact per-keyframe transform, and the Potree octree built
right there. Integrity is verified on the staged files, and only then the
journaled atomic swap runs: current versions move to ``output/_epoch_<N-1>/``
(kept until approve), staged versions move into place. ANY failure before the
swap discards the transaction and the session is byte-identical to before;
a failure mid-swap rolls the completed renames back. There is no state where
the cloud changed but Potree did not.

Undo = the inverse swap from ``_epoch_<N-1>/``. Approve = delete
``_epoch_<N-1>/``. The ledger records both verdicts; history is never erased.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from correction.config import CorrectionConfig
from correction.epoch import EPOCH_FILE, current_epoch, make_epoch_record
from correction.ledger import EPOCH_NPZ_DIR, save_epoch_npz
from correction.session import (CorrectionSession, read_poses, write_ply,
                                write_poses)

TX_PREFIX = "_tx_epoch_"
PREV_PREFIX = "_epoch_"
SWAP_JOURNAL = "_tx_swap_journal.json"
MANIFEST_NAME = "_manifest.json"

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)


def prev_dir_for(output_dir: Path) -> Optional[Path]:
    """The pending previous-epoch directory (None when nothing is pending)."""
    cur = current_epoch(output_dir)
    p = Path(output_dir) / f"{PREV_PREFIX}{cur - 1}"
    return p if cur > 0 and p.exists() else None


def assert_no_interrupted_swap(output_dir: Path) -> None:
    j = Path(output_dir) / SWAP_JOURNAL
    if j.exists():
        raise RuntimeError(
            f"an interrupted correction swap left {j} behind — the session "
            f"may be half-swapped; inspect the journal and the _epoch_*/ "
            f"directories before running anything else on this session")


# ── warp ─────────────────────────────────────────────────────────────────

def warp_full_cloud(session: CorrectionSession, R_kf: np.ndarray,
                    t_kf: np.ndarray, k_kf: np.ndarray,
                    log=print, b_kf: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, int]:
    """New coordinates for the whole cloud (per-keyframe grouping; depth
    z' = k·z + b along each point's own camera ray first, then the rigid
    transform). ``b_kf`` (metres, default 0) is the affine offset of the
    depth-by-correspondences stage (claude_stac.txt §6.4). Returns
    (xyz_new, n_moved)."""
    xyz_new = session.xyz.copy()
    ks = session.ks
    n_kf = session.n_kf
    order = np.argsort(ks, kind="stable")
    bounds = np.searchsorted(ks[order], np.arange(-1, n_kf + 1))
    n_moved = 0
    identity = (np.eye(3), np.zeros(3))
    for k in range(n_kf):
        sel = order[bounds[k + 1]:bounds[k + 2]]
        if not len(sel):
            continue
        kv = float(k_kf[k])
        bv = float(b_kf[k]) if b_kf is not None else 0.0
        R, t = R_kf[k], t_kf[k]
        is_id = (kv == 1.0 and bv == 0.0 and np.allclose(R, identity[0])
                 and np.allclose(t, identity[1]))
        if is_id:
            continue
        if kv != 1.0 or bv != 0.0:
            frame = session.frames[k]
            cam = session.cam_center[frame]
            if bv != 0.0:
                axis = session.poses[k][:3, 2]        # camera z in world
                z = (xyz_new[sel] - cam) @ axis
                zc = np.where(np.abs(z) > 1e-9, z, 1e-9)
                xyz_new[sel] = cam + (xyz_new[sel] - cam) * ((kv * z + bv) / zc)[:, None]
            else:
                xyz_new[sel] = cam + (xyz_new[sel] - cam) * kv
        xyz_new[sel] = xyz_new[sel] @ R.T + t
        n_moved += len(sel)
    return xyz_new, n_moved


def transform_poses(poses: np.ndarray, R_kf: np.ndarray,
                    t_kf: np.ndarray) -> np.ndarray:
    out = poses.copy()
    for k in range(len(out)):
        R, t = R_kf[k], t_kf[k]
        out[k][:3, :3] = R @ out[k][:3, :3]
        out[k][:3, 3] = R @ out[k][:3, 3] + t
    return out


def _recompute_obbs(output_dir: Path, xyz_new: np.ndarray,
                    result: dict, log=print,
                    floor_npz: Optional[dict] = None) -> dict:
    """OBBs refit from the moved points, in DISPLAY space via the session's
    floor transform — the NEW one when this correction replaces it
    (globalIndices untouched — point order is preserved)."""
    ft = output_dir / "floor_transform.npz"
    if floor_npz is not None:
        s_, R_, t_ = float(floor_npz["s"]), floor_npz["R"], floor_npz["t"]
    elif ft.exists():
        d = np.load(ft)
        s_, R_, t_ = float(d["s"]), d["R"], d["t"]
    else:
        s_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
    from segmentation.pipeline import _compute_obb
    n_done = 0
    for inst in result.get("instances", []):
        g = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        g = g[(g >= 0) & (g < len(xyz_new))]
        if len(g) < 10:
            continue
        disp = (s_ * (xyz_new[g] @ R_.T)) + t_
        inst["obb"] = _compute_obb(disp)
        n_done += 1
    log(f"  segmentation_result.json OBBs recomputed ({n_done} instance(s))")
    return result


# ── transaction ──────────────────────────────────────────────────────────

def stage_transaction(session: CorrectionSession, cfg: CorrectionConfig,
                      R_kf: np.ndarray, t_kf: np.ndarray, k_kf: np.ndarray,
                      *, correction_id: str, scale_diag_new: Optional[dict],
                      floor_npz: Optional[dict] = None,
                      log=print, progress=None,
                      b_kf: Optional[np.ndarray] = None) -> dict:
    """Build every new-epoch artifact under output/_tx_epoch_<N>/. Returns
    {"tx_dir", "epoch_from", "epoch_to", "artifacts": [...],
     "pose_copies_skipped": [...], "points_moved": int}. Raises on any
    problem — nothing outside the tx dir is touched."""
    output_dir = session.output_dir
    assert_no_interrupted_swap(output_dir)
    epoch_from = current_epoch(output_dir)
    epoch_to = epoch_from + 1
    tx = output_dir / f"{TX_PREFIX}{epoch_to}"
    if tx.exists():
        log(f"  removing stale transaction dir {tx.name} (crashed run)")
        shutil.rmtree(tx)
    tx.mkdir(parents=True)
    artifacts: List[dict] = []

    def _art(rel: str):
        artifacts.append({"rel": rel,
                          "existed_before": (output_dir / rel).exists()})

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    # 1) cloud ------------------------------------------------------------
    _p(55, "tx: warping cloud per keyframe...")
    xyz_new, n_moved = warp_full_cloud(session, R_kf, t_kf, k_kf, log=log, b_kf=b_kf)
    data_new = session.data.copy()
    data_new["x"] = xyz_new[:, 0].astype(session.data.dtype["x"])
    data_new["y"] = xyz_new[:, 1].astype(session.data.dtype["y"])
    data_new["z"] = xyz_new[:, 2].astype(session.data.dtype["z"])
    write_ply(tx / "cleaned_cloud.ply", session.header, data_new)
    _art("cleaned_cloud.ply")
    log(f"  cleaned_cloud.ply staged ({n_moved:,} pts warped)")

    # 2) raw cloud (same per-keyframe transform — same order+provenance) ---
    if session.raw_data is not None:
        raw_xyz = np.stack([session.raw_data["x"], session.raw_data["y"],
                            session.raw_data["z"]], axis=1).astype(np.float64)
        raw_fg = session.raw_data["frame_global"].astype(np.int64)
        raw_sess = session.__class__(
            output_dir=session.output_dir, header=session.raw_header,
            data=session.raw_data, xyz=raw_xyz, fg=raw_fg,
            ks=session.ks, frames=session.frames, kf_index=session.kf_index,
            poses=session.poses, cam_center=session.cam_center,
            raw_header=None, raw_data=None, pose_copies=[])
        raw_new, _ = warp_full_cloud(raw_sess, R_kf, t_kf, k_kf, log=log, b_kf=b_kf)
        raw_data_new = session.raw_data.copy()
        raw_data_new["x"] = raw_new[:, 0].astype(session.raw_data.dtype["x"])
        raw_data_new["y"] = raw_new[:, 1].astype(session.raw_data.dtype["y"])
        raw_data_new["z"] = raw_new[:, 2].astype(session.raw_data.dtype["z"])
        write_ply(tx / "cleaned_cloud_raw.ply", session.raw_header,
                  raw_data_new)
        _art("cleaned_cloud_raw.ply")
        log("  cleaned_cloud_raw.ply staged (same transform)")

    # 3) poses: canonical + every matching copy ---------------------------
    _p(62, "tx: transforming camera poses (canonical + copies)...")
    poses_new = transform_poses(session.poses, R_kf, t_kf)
    write_poses(tx / "camera_poses.txt", poses_new)
    _art("camera_poses.txt")
    pose_copies_skipped: List[dict] = []
    for copy_path in session.pose_copies:
        rel = str(copy_path.relative_to(output_dir))
        if rel == "camera_poses.txt":
            continue
        rows = read_poses(copy_path)
        if len(rows) != session.n_kf:
            pose_copies_skipped.append({
                "path": rel, "rows": len(rows),
                "reason": f"row count {len(rows)} ≠ {session.n_kf} keyframes "
                          f"— copy predates the current keyframe set"})
            continue
        out = tx / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        write_poses(out, transform_poses(rows, R_kf, t_kf))
        _art(rel)
    if pose_copies_skipped:
        log(f"  pose copies skipped (declared): {pose_copies_skipped}")

    # 4) segmentation OBBs ------------------------------------------------
    res_path = output_dir / "segmentation_result.json"
    if res_path.exists():
        result = json.loads(res_path.read_text())
        result = _recompute_obbs(output_dir, xyz_new, result, log=log,
                                 floor_npz=floor_npz)
        (tx / "segmentation_result.json").write_text(json.dumps(result))
        _art("segmentation_result.json")

    # 5) depth-correction sidecar (cumulative affine z'' = k·(k₀z + b₀) + b) --
    from segmentation.session_io import DEPTH_CORRECTION_NAME, \
        load_depth_affine
    old_kb = load_depth_affine(output_dir) or {}
    new_kb = dict(old_kb)
    for kf in range(session.n_kf):
        kv = float(k_kf[kf])
        bv = float(b_kf[kf]) if b_kf is not None else 0.0
        if kv != 1.0 or bv != 0.0:
            frame = session.frames[kf]
            k0, b0 = new_kb.get(frame, (1.0, 0.0))
            new_kb[frame] = (k0 * kv, kv * b0 + bv)
    if new_kb or old_kb:
        (tx / DEPTH_CORRECTION_NAME).write_text(json.dumps(
            {"version": 2, "epoch": epoch_to,
             "k": {str(f): round(v[0], 6) for f, v in sorted(new_kb.items())},
             "b": {str(f): round(v[1], 6) for f, v in sorted(new_kb.items())}},
            indent=1))
        _art(DEPTH_CORRECTION_NAME)

    # 6) scale diagnostics regenerated per epoch --------------------------
    if scale_diag_new is not None:
        (tx / "scale_diagnostics.json").write_text(
            json.dumps(scale_diag_new, indent=2))
        _art("scale_diagnostics.json")

    # 7) geometry epoch ---------------------------------------------------
    (tx / EPOCH_FILE).write_text(json.dumps(
        make_epoch_record(epoch_to, correction_id, epoch_from), indent=1))
    _art(EPOCH_FILE)

    # 8) exact per-keyframe transform -------------------------------------
    save_epoch_npz(output_dir, epoch_to, R_kf, t_kf, k_kf, session.frames,
                   dir_override=tx / EPOCH_NPZ_DIR, b_kf=b_kf)
    _art(f"{EPOCH_NPZ_DIR}/epoch_{epoch_to}.npz")

    # 9) floor transform (floor-align kind only) --------------------------
    if floor_npz is not None:
        np.savez(tx / "floor_transform.npz", **floor_npz)
        _art("floor_transform.npz")

    # 9b) re-consolidate the WARPED cloud ---------------------------------
    # The warp moves every point by its own keyframe's correction and nothing
    # cleans up afterwards. Where a duplicate finally closes, the two copies
    # land on top of each other and stay TWO point sets: the geometry is right
    # and the user still sees double density on the object he was promised
    # would become one. Consolidation moves points without adding or removing
    # any — same count, same order — so globalIndices, colours and per-point
    # provenance survive it, which is why it is safe here and a re-run of the
    # SOR would not be. Runs BEFORE the octree so the build carries it. Never
    # fatal: a transaction that could not consolidate is still a valid epoch.
    if getattr(cfg.apply, "reconsolidate", False):
        _p(68, "tx: re-consolidating the warped cloud...")
        try:
            from reconstruction.surface_fit.consolidate import scene_consolidate
            rep = scene_consolidate(tx)
            if rep:
                log(f"  re-consolidated {rep.get('n_points', 0):,} pts, "
                    f"mean move {rep.get('mean_move_mm', 0):.2f} mm "
                    f"(p95 {rep.get('p95_move_mm', 0):.2f} mm)")
            else:
                log("  re-consolidation returned nothing — cloud left as warped")
        except Exception as e:  # noqa: BLE001 — declared, never fatal
            log(f"  re-consolidation failed ({e}) — cloud left as warped")

    # 10) Potree inside the transaction -----------------------------------
    if cfg.apply.potree_rebuild:
        _p(70, "tx: building Potree octree inside the transaction...")
        from potree_converter import convert_ply_to_potree
        ok = convert_ply_to_potree(output_dir.parent, force=True,
                                   ply_override=tx / "cleaned_cloud.ply",
                                   potree_dir_override=tx / "potree")
        if not ok or not (tx / "potree" / "metadata.json").exists():
            shutil.rmtree(tx)
            raise RuntimeError(
                "Potree build FAILED inside the transaction — nothing was "
                "applied, the session is untouched (check the converter log; "
                "disk space is the usual cause)")
        _art("potree")

    # 11) integrity verification on the STAGED files ----------------------
    _p(85, "tx: verifying staged artifacts...")
    from correction.session import read_ply
    _, staged = read_ply(tx / "cleaned_cloud.ply")
    if len(staged) != session.n_points:
        shutil.rmtree(tx)
        raise RuntimeError(
            f"staged cloud has {len(staged)} points, expected "
            f"{session.n_points} — transaction discarded")
    for fld in ("frame_global", "pixel_row", "pixel_col"):
        if not np.array_equal(staged[fld], session.data[fld]):
            shutil.rmtree(tx)
            raise RuntimeError(
                f"staged cloud provenance field '{fld}' differs from the "
                f"source — transaction discarded")
    if len(read_poses(tx / "camera_poses.txt")) != session.n_kf:
        shutil.rmtree(tx)
        raise RuntimeError("staged camera_poses.txt row count mismatch — "
                           "transaction discarded")

    return {"tx_dir": str(tx), "epoch_from": epoch_from,
            "epoch_to": epoch_to, "artifacts": artifacts,
            "pose_copies_skipped": pose_copies_skipped,
            "points_moved": int(n_moved)}


def swap_transaction(output_dir: Path, tx_info: dict, log=print) -> None:
    """Journaled atomic swap: current → _epoch_<from>/, staged → current.
    Rolls completed renames back on any mid-swap failure."""
    output_dir = Path(output_dir)
    tx = Path(tx_info["tx_dir"])
    prev = output_dir / f"{PREV_PREFIX}{tx_info['epoch_from']}"
    if prev.exists():
        raise RuntimeError(
            f"{prev} already exists — a previous correction is still "
            f"pending; approve or undo it first")
    prev.mkdir(parents=True)
    journal_path = output_dir / SWAP_JOURNAL
    journal = {"epoch_from": tx_info["epoch_from"],
               "epoch_to": tx_info["epoch_to"],
               "artifacts": tx_info["artifacts"]}
    journal_path.write_text(json.dumps(journal, indent=1))
    done: List[dict] = []
    try:
        for art in tx_info["artifacts"]:
            rel = art["rel"]
            cur = output_dir / rel
            if art["existed_before"]:
                target = prev / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                cur.rename(target)
            staged = tx / rel
            cur.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(cur)
            done.append(art)
    except BaseException:
        # roll back what moved, restore the original state exactly
        for art in reversed(done):
            rel = art["rel"]
            cur = output_dir / rel
            back = tx / rel
            back.parent.mkdir(parents=True, exist_ok=True)
            if cur.exists():
                cur.rename(back)
            if art["existed_before"]:
                (prev / rel).rename(cur)
        shutil.rmtree(prev, ignore_errors=True)
        journal_path.unlink(missing_ok=True)
        raise
    (prev / MANIFEST_NAME).write_text(json.dumps(journal, indent=1))
    journal_path.unlink()
    shutil.rmtree(tx, ignore_errors=True)
    log(f"  swap complete: epoch {tx_info['epoch_from']} → "
        f"{tx_info['epoch_to']} ({len(tx_info['artifacts'])} artifact(s); "
        f"previous epoch kept in {prev.name}/ until approve)")


def undo_swap(output_dir: Path, log=print) -> dict:
    """Inverse swap: restore the previous epoch exactly. The undone epoch's
    files are discarded (the ledger keeps its record and verdict)."""
    output_dir = Path(output_dir)
    assert_no_interrupted_swap(output_dir)
    prev = prev_dir_for(output_dir)
    if prev is None:
        raise RuntimeError("no pending correction to undo — there is no "
                           "previous-epoch directory")
    manifest = json.loads((prev / MANIFEST_NAME).read_text())
    discard = output_dir / f"_undo_discard_{manifest['epoch_to']}"
    if discard.exists():
        shutil.rmtree(discard)
    discard.mkdir()
    journal_path = output_dir / SWAP_JOURNAL
    journal_path.write_text(json.dumps({"undo_of": manifest}, indent=1))
    done: List[dict] = []
    try:
        for art in reversed(manifest["artifacts"]):
            rel = art["rel"]
            cur = output_dir / rel
            if cur.exists():
                target = discard / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                cur.rename(target)
            if art["existed_before"]:
                cur.parent.mkdir(parents=True, exist_ok=True)
                (prev / rel).rename(cur)
            done.append(art)
    except BaseException:
        for art in reversed(done):
            rel = art["rel"]
            cur = output_dir / rel
            if art["existed_before"] and cur.exists():
                (prev / rel).parent.mkdir(parents=True, exist_ok=True)
                cur.rename(prev / rel)
            if (discard / rel).exists():
                (discard / rel).rename(cur)
        journal_path.unlink(missing_ok=True)
        raise
    journal_path.unlink()
    shutil.rmtree(discard, ignore_errors=True)
    shutil.rmtree(prev, ignore_errors=True)
    log(f"  undo complete: epoch {manifest['epoch_to']} discarded, epoch "
        f"{manifest['epoch_from']} restored")
    return manifest


def pending_prev_dirs(output_dir: Path) -> List[Path]:
    """The chain of pending previous-epoch directories, newest first
    (_epoch_<cur-1>, _epoch_<cur-2>, … while they exist)."""
    output_dir = Path(output_dir)
    cur = current_epoch(output_dir)
    out = []
    e = cur - 1
    while e >= 0:
        p = output_dir / f"{PREV_PREFIX}{e}"
        if not p.exists():
            break
        out.append(p)
        e -= 1
    return out


def approve_swap(output_dir: Path, log=print) -> dict:
    """Approve: the current epoch IS the session; every pending previous
    epoch of the chain is removed (USER: approval leaves no remains — the
    ledger keeps the record). Returns the newest manifest."""
    output_dir = Path(output_dir)
    assert_no_interrupted_swap(output_dir)
    chain = pending_prev_dirs(output_dir)
    if not chain:
        raise RuntimeError("no pending correction to approve")
    manifest = json.loads((chain[0] / MANIFEST_NAME).read_text())
    for prev in chain:
        shutil.rmtree(prev)
    log(f"  approved: epoch {manifest['epoch_to']} is now the session; "
        f"{len(chain)} pending epoch dir(s) removed")
    return manifest
