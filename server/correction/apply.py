"""Transactional application of a distributed correction.

Contract (prompt §7): every artifact of the new epoch is staged under
``output/_tx_epoch_<N>/`` — cloud, raw cloud, poses (canonical + copies),
segmentation OBBs, depth-correction sidecar, regenerated scale diagnostics,
geometry epoch, exact per-keyframe transform, and the Potree octree built
right there. Integrity is verified on the staged files, and only then the
journaled atomic swap runs: current versions move to ``output/_epoch_<N-1>/``
(kept and selectable), staged versions move into place. ANY failure before the
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
from correction.epoch import (EPOCH_DIR_PREFIX, EPOCH_FILE, current_epoch,
                              make_epoch_record)
from correction.ledger import EPOCH_NPZ_DIR, save_epoch_npz
from correction.session import (CorrectionSession, read_poses, write_ply,
                                write_poses)

TX_PREFIX = "_tx_epoch_"
PREV_PREFIX = EPOCH_DIR_PREFIX
SWAP_JOURNAL = "_tx_swap_journal.json"
MANIFEST_NAME = "_manifest.json"

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)


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
    epoch_to = next_epoch(output_dir)
    tx = output_dir / f"{TX_PREFIX}{epoch_to}"
    if tx.exists():
        log(f"  removing stale transaction dir {tx.name} (crashed run)")
        shutil.rmtree(tx)
    tx.mkdir(parents=True)
    artifacts: List[dict] = []

    _art_seen = set()

    def _art(rel: str):
        """Register a staged artifact for the swap. Idempotent: two steps can
        legitimately touch the same file (step 4 stages the segmentation and
        step 9c republishes it), and the swap moves the manifest ROW BY ROW —
        a second row for the same path renames a file that is already gone and
        aborts the whole swap."""
        if rel in _art_seen:
            return
        _art_seen.add(rel)
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
    raw_data_new = None
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

    # 4) segmentation staged as it stands ---------------------------------
    # The OBBs are NOT fitted here. The mask filter (9a) has not run yet, so
    # `xyz_new` still holds points this transaction is about to delete, and
    # `_reindex` then remaps every globalIndex without refitting — an OBB
    # computed from points the same transaction removes (pccr 2026-09-20).
    # The fit happens at 9c, on the geometry that actually ships.
    res_path = output_dir / "segmentation_result.json"
    if res_path.exists():
        (tx / "segmentation_result.json").write_text(res_path.read_text())
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

    # 9a) THE MASK FILTER — last, and INSIDE this transaction ------------
    # USER 2026-09-19: *"al final de todo como último paso antes de la
    # consolidación y del octree"* … *"no vamos a hacer un octree atrás de
    # otro"*. Here the pose is already corrected, so a point is judged against
    # its own mask where the correction actually left it; and the ONE
    # consolidation and the ONE octree below carry the result.
    #
    # It is FATAL when it is configured ON and cannot run. It used to be
    # "never fatal", and on 2026-09-21 a NameError inside it degraded to one
    # line of log: the epoch published WITHOUT filtering a single point, and
    # the geometric cleanup then skipped itself because "the mask filter
    # already ran inside the epoch". A step that cannot do its job must stop
    # the transaction, not let the next one inherit a false premise.
    kept_mask = None
    if getattr(cfg.apply, "mask_filter", False):
        _p(66, "tx: mask filter (masklets, last before consolidation)...")
        try:
            from correction.visit_drift_run import filter_staged_cloud
            out = filter_staged_cloud(
                tx, session, data_new, xyz_new, poses_new,
                raw_data_new,
                cfg, log=log)
            frep, kept_mask = out if out is not None else (None, None)
            if frep is not None:
                (tx / "mask_filter.json").write_text(json.dumps(frep, indent=1,
                                                                default=float))
                _art("mask_filter.json")
                # the epoch npz was written at step 8, before this filter knew
                # what it would delete. An epoch that moved 22 M points AND
                # deleted a quarter million is not reproduced by the motion
                # alone, so it is rewritten with what left (correction.replay
                # honours `dropped`).
                save_epoch_npz(output_dir, epoch_to, R_kf, t_kf, k_kf,
                               session.frames, dir_override=tx / EPOCH_NPZ_DIR,
                               b_kf=b_kf,
                               dropped=np.flatnonzero(~kept_mask))
        except Exception as e:  # noqa: BLE001 — declared, never silent
            shutil.rmtree(tx)
            raise RuntimeError(
                f"the mask filter is ON and could not run ({e}) — transaction "
                f"discarded rather than publishing an epoch that skipped it "
                f"and letting the cleanup believe the filtering happened "
                f"(pccr 2026-09-21)") from e

    # 9c) the cloud's siblings, republished from the FINAL staged geometry --
    # A cloud never travels alone: `classification.npy` (the per-point class
    # the Potree converter bakes into the octree), `out_of_place.npy` (the
    # cleanup marks) and the result's own census are all indexed BY ROW. Until
    # now the epoch staged the cloud and left them behind, and nothing said so
    # — the converter compares the two lengths, writes NOTHING and logs one
    # line, so pccr's epoch-1 octree came out with every class byte at zero
    # and `geometric_cleanup` refused to run at all. This is deliberately
    # FATAL: an epoch whose siblings do not describe its cloud is the silent
    # breakage this step exists to end.
    _p(67, "tx: republishing the segmentation siblings...")
    xyz_final = xyz_new[kept_mask] if kept_mask is not None else xyz_new
    res_tx = tx / "segmentation_result.json"
    if res_tx.exists():
        result = _recompute_obbs(output_dir, xyz_final,
                                 json.loads(res_tx.read_text()),
                                 log=log, floor_npz=floor_npz)
        res_tx.write_text(json.dumps(result))
    try:
        from segmentation.republish import republish_membership
        rep = republish_membership(tx, n_points=int(len(xyz_final)),
                                   keep=kept_mask, source_dir=output_dir,
                                   log=log)
        for rel in rep["files"]:
            _art(rel)
    except Exception as e:
        shutil.rmtree(tx)
        raise RuntimeError(
            f"the epoch could not republish what hangs off its cloud ({e}) — "
            f"transaction discarded rather than shipping an octree with no "
            f"classes and per-point arrays that describe the previous epoch")

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
            # The MLS asks for ~7.7 GB in one allocation and this runs inside a
            # certification, which started vLLM itself to classify the loops.
            # pccr 2026-09-17: both consolidations of the day died with 859 MB
            # free. The loop classification is long done by now, and any later
            # consumer restarts the service.
            from workers.base import stop_semantic_service
            stop_semantic_service(stage="epoch re-consolidation", log=log)
            from reconstruction.surface_fit.consolidate import scene_consolidate
            rep = scene_consolidate(tx, artifacts_dir=output_dir)
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
    # after the mask filter the expectation is the FILTERED one — the invariant
    # is that every surviving row keeps its provenance and its ORDER, not that
    # no row was ever removed (USER 2026-09-19: the filter deletes for real)
    n_expect = int(kept_mask.sum()) if kept_mask is not None else session.n_points
    if len(staged) != n_expect:
        shutil.rmtree(tx)
        raise RuntimeError(
            f"staged cloud has {len(staged)} points, expected "
            f"{n_expect} — transaction discarded")
    for fld in ("frame_global", "pixel_row", "pixel_col"):
        src_fld = (session.data[fld][kept_mask] if kept_mask is not None
                   else session.data[fld])
        if not np.array_equal(staged[fld], src_fld):
            shutil.rmtree(tx)
            raise RuntimeError(
                f"staged cloud provenance field '{fld}' differs from the "
                f"source — transaction discarded")
    for sib in ("classification.npy", "out_of_place.npy"):
        sp = tx / sib
        if sp.exists() and len(np.load(sp)) != n_expect:
            shutil.rmtree(tx)
            raise RuntimeError(
                f"staged {sib} has {len(np.load(sp))} values against "
                f"{n_expect} points — transaction discarded")
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
            f"{prev} already exists — epoch {tx_info['epoch_from']} is already "
            f"stored; the session is inconsistent")
    prev.mkdir(parents=True)
    journal_path = output_dir / SWAP_JOURNAL
    journal = {"epoch": tx_info["epoch_from"],      # the state kept in prev/
               "epoch_from": tx_info["epoch_from"],
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
        f"epoch {tx_info['epoch_from']} kept in {prev.name}/, selectable)")


def next_epoch(output_dir: Path) -> int:
    """The number the next correction gets: one past the HIGHEST epoch this
    session has ever held, not one past the live one.

    Since the epochs are selected (USER 2026-09-16), a correction can run on
    top of an older epoch while newer ones sit on disk. `current + 1` would
    then re-use a number that already names a different state — colliding
    directory `_epoch_<N>/`, colliding `corrections/epoch_<N>.npz`, and a
    ledger that says two things about the same epoch. Numbers are never
    recycled; the branch point is recorded as `parent_epoch` in
    ``geometry_epoch.json``.
    """
    output_dir = Path(output_dir)
    highest = current_epoch(output_dir)
    for d in output_dir.glob(f"{PREV_PREFIX}*"):
        if not d.is_dir():
            continue
        try:
            highest = max(highest, int(d.name[len(PREV_PREFIX):]))
        except ValueError:
            continue
    npz_dir = output_dir / EPOCH_NPZ_DIR
    for q in npz_dir.glob("epoch_*.npz") if npz_dir.is_dir() else ():
        try:
            highest = max(highest, int(q.stem.split("_")[-1]))
        except ValueError:
            continue
    return highest + 1


def session_artifacts(output_dir: Path) -> List[dict]:
    """Every artifact rel any epoch of this session ever staged, in a stable
    order, WITHOUT the per-epoch transforms.

    The manifest kept in ``_epoch_<N>/`` is the journal of the swap that
    produced epoch N+1, so the union over the stored manifests is the complete
    file set the session has moved through. Selecting an epoch has to move all
    of them: the artifacts of ONE manifest leave behind whatever a later epoch
    introduced (`depth_correction.json`, `floor_transform.npz`).

    `corrections/epoch_<N>.npz` is excluded on purpose — the transforms are
    HISTORY, not state: they are what lets the instance store follow the
    geometry and `correction.replay` reproduce any epoch, so they stay live
    whichever epoch is shown.
    """
    output_dir = Path(output_dir)
    seen, arts = set(), []
    for d in sorted(output_dir.glob(f"{PREV_PREFIX}*"),
                    key=lambda q: (len(q.name), q.name)):
        man = d / MANIFEST_NAME
        if not man.is_file():
            continue
        try:
            entries = json.loads(man.read_text()).get("artifacts", [])
        except (OSError, ValueError):
            continue
        for a in entries:
            rel = str(a.get("rel", ""))
            if not rel or rel in seen or rel.startswith(EPOCH_NPZ_DIR + "/"):
                continue
            seen.add(rel)
            arts.append({"rel": rel})
    return arts


def available_epochs(output_dir: Path) -> List[dict]:
    """Every epoch this session holds, oldest first, and which one is live.

    USER 2026-09-16: *"todas viven, solo se seleccionan y la que se selecciona
    se muestra"*. There is no approving and no undoing: an epoch is a state the
    session can be shown in, and every one of them stays on disk until a new
    reconstruction replaces the session.
    """
    output_dir = Path(output_dir)
    cur = current_epoch(output_dir)
    out = []
    # EVERY _epoch_* directory, not a contiguous walk down from the current
    # one: once an epoch can be SELECTED the stored ones stop being a chain
    # below the live state. Showing epoch 1 while 2 and 3 exist leaves them
    # above it, and a walk from cur-1 downwards would hide them.
    for d in sorted(output_dir.glob(f"{PREV_PREFIX}*")):
        if not d.is_dir() or not (d / MANIFEST_NAME).is_file():
            continue
        try:
            e = int(d.name[len(PREV_PREFIX):])
        except ValueError:
            continue
        if e == cur:
            continue
        out.append({"epoch": e, "live": False, "dir": d.name,
                    "potree": (d / "potree" / "metadata.json").exists()})
    out.append({"epoch": cur, "live": True, "dir": None,
                "potree": (output_dir / "potree" / "metadata.json").exists()})
    return sorted(out, key=lambda r: r["epoch"])


def select_epoch(output_dir: Path, epoch: int, log=print) -> dict:
    """Show the session in the state of ``epoch``. Nothing is destroyed.

    The live artifacts are filed under their own epoch's directory and the
    chosen epoch's are moved into place — the same journaled rename dance as
    the swap, so an interruption rolls back to exactly where it started.

    This REPLACES approve/undo (USER 2026-09-16: *"el accept y undo no sirven
    para nada, en realidad deben quedar épocas que deben ser seleccionables
    para verificación visual, nada más"*). Approve used to delete every
    previous epoch and Undo used to delete the current one; between them a
    session could only ever hold two states, and choosing wrong destroyed the
    other. Now every epoch survives and selecting is free.
    """
    output_dir = Path(output_dir)
    assert_no_interrupted_swap(output_dir)
    epoch = int(epoch)
    cur = current_epoch(output_dir)
    if epoch == cur:
        return {"epoch": cur, "changed": False}
    src = output_dir / f"{PREV_PREFIX}{epoch}"
    if not src.is_dir() or not (src / MANIFEST_NAME).is_file():
        raise RuntimeError(f"epoch {epoch} is not in this session "
                           f"({[e['epoch'] for e in available_epochs(output_dir)]})")
    # Every artifact any epoch of this session ever staged, not just the ones
    # in the chosen epoch's manifest: an epoch that came later may have created
    # a file the chosen one never had (`depth_correction.json` of a depth
    # correction, `floor_transform.npz` of a floor alignment). Swapping only
    # the destination's list left that file live over older geometry — the
    # cloud of epoch 1 with the depth sidecar of epoch 3.
    arts = session_artifacts(output_dir)
    if not arts:
        raise RuntimeError(f"epoch {epoch} has no artifact manifest to restore")
    dst = output_dir / f"{PREV_PREFIX}{cur}"
    if dst.exists():
        raise RuntimeError(f"{dst} already exists — the session is inconsistent")
    dst.mkdir(parents=True)
    journal_path = output_dir / SWAP_JOURNAL
    journal_path.write_text(json.dumps(
        {"select": epoch, "from": cur, "artifacts": arts}, indent=1))
    done: List[dict] = []
    stored: List[dict] = []
    try:
        for art in arts:
            rel = art["rel"]
            live = output_dir / rel
            if live.exists():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                live.rename(dst / rel)
                stored.append({"rel": rel, "existed_before": True})
            keep = src / rel
            if keep.exists():
                live.parent.mkdir(parents=True, exist_ok=True)
                keep.rename(live)
            done.append(art)
    except BaseException:
        for art in reversed(done):
            rel = art["rel"]
            live = output_dir / rel
            if live.exists():
                (src / rel).parent.mkdir(parents=True, exist_ok=True)
                live.rename(src / rel)
            back = dst / rel
            if back.exists():
                back.rename(live)
        shutil.rmtree(dst, ignore_errors=True)
        journal_path.unlink(missing_ok=True)
        raise
    # the directory that held the chosen epoch now holds the one we left, and
    # its manifest describes what actually landed there
    (dst / MANIFEST_NAME).write_text(json.dumps(
        {"epoch": cur, "artifacts": stored}, indent=1))
    shutil.rmtree(src, ignore_errors=True)
    journal_path.unlink()
    log(f"  showing epoch {epoch} (was {cur}); every epoch of this session is "
        f"still on disk")
    return {"epoch": epoch, "previous": cur, "changed": True,
            "available": [e["epoch"] for e in available_epochs(output_dir)]}
