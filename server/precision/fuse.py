"""F7 — witness fusion with tiers, full provenance and the rejected set
(claude_stac.txt §4-F7).

The cloud is rebuilt from F6's native-resolution depth (``output/depth_native``)
with F5's camera and poses: every pixel of tier 0 (plane sweep) — and of tier 1
(prior fill) when ``prior_fill: keep`` — is back-projected, and a point ENTERS
only with at least ``min_witness_views`` consistent views (tier 1:
``prior_fill_min_views``). Points that fall in the same ``voxel_m`` cell are ONE
point: the one with the best evidence (views that confirm it, then its ZNCC,
then the earliest frame and pixel — a total order, so the result does not depend
on the order anything was read), never an average. SOR (the GPU statistical
outlier removal of ``reconstruction.gpu_cloud_clean``) is selectable and OFF by
default; CloudComPy's noise filter is not available here (it runs in its own env)
and selecting it fails with that reason.

Memory: pass 1 keeps 20 bytes per entering pixel (voxel key, evidence, frame,
pixel); the winners are chosen by one global sort; pass 2 rebuilds only the
winners (double precision), with colour and provenance.

Every candidate — a pixel F6 gave a depth hypothesis — ends either in the cloud
or in ``rejected_points.npz`` with its reason (the two add up to the candidates).
Provenance v2 (``precision.provenance``) is written into the PLY (the v1 fields
plus the v2 columns) and row-aligned into ``output/origins.npz``. The viewer's
existing live filters read two channels: ``confidence`` = the ZNCC for tier 0 and
0 for tier 1 (the confidence slider above 0 hides exactly tier 1), ``mv_votes`` =
the consistent views; ``status`` = verified (every point is, by construction).

Publication: a new geometry epoch through the correction module's transaction
(``_tx_epoch_<N>/``, Potree built inside, atomic journaled swap; epoch 0 stays
selectable). The epoch-0 segmentation indexes another cloud: the new epoch carries
an empty one marked pending (F8 rebuilds the semantics on it).

CLI: ``python -m precision.fuse --session <dir> [--no-publish]``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from precision import provenance as PV

FUSE_REPORT_NAME = "fuse_report.json"
FUSE_VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[fuse]"
KEY_BITS = 21                                  # per axis of the packed voxel key
KEY_HALF = 1 << (KEY_BITS - 1)


class FuseError(RuntimeError):
    """The fusion cannot run — with the exact reason."""


def pack_keys(ijk: np.ndarray) -> np.ndarray:
    """(N,3) int64 voxel indices → one int64 per voxel (21 bits per axis)."""
    if len(ijk) and (np.abs(ijk).max() >= KEY_HALF):
        raise FuseError(f"a voxel index reaches {int(np.abs(ijk).max())} ≥ {KEY_HALF} — the "
                        f"scene exceeds the packed key's range at this voxel size")
    u = (ijk + KEY_HALF).astype(np.int64)
    return (u[:, 0] << (2 * KEY_BITS)) | (u[:, 1] << KEY_BITS) | u[:, 2]


def evidence(n_consistent: np.ndarray, ncc: np.ndarray) -> np.ndarray:
    """Sortable evidence: the confirming views first, then the ZNCC (in [-1, 1],
    so it never reaches the next view count)."""
    return np.asarray(n_consistent, np.float64) * 4 + (np.nan_to_num(ncc, nan=-1.0) + 1.0)


def backproject(depth: np.ndarray, rows: np.ndarray, cols: np.ndarray, K: np.ndarray,
                c2w: np.ndarray) -> np.ndarray:
    z = depth[rows, cols].astype(np.float64)
    P = np.stack([(cols - K[0, 2]) / K[0, 0] * z, (rows - K[1, 2]) / K[1, 1] * z, z], 1)
    return P @ c2w[:3, :3].T + c2w[:3, 3]


def _load_inputs(session_dir: Path, pcfg):
    from intake.quality import read_session_epochs
    from precision.camera import load_camera_json, undistort_maps
    from precision.depth_sweep import DEPTH_DIRNAME, REPORT_NAME, _read_poses
    out = session_dir / "output"
    rep_p = out / DEPTH_DIRNAME / REPORT_NAME
    if not rep_p.exists():
        raise FuseError(f"{rep_p} is missing — F7 fuses F6's depth "
                        f"(python -m precision.depth_sweep --session <dir>)")
    rep = json.loads(rep_p.read_text())
    epochs = read_session_epochs(session_dir)
    if any(rep.get(k) != v for k, v in epochs.items()):
        raise FuseError(f"depth_native was measured on {({k: rep.get(k) for k in epochs})}, the "
                        f"session is at {epochs} — re-run F6 on the current epoch")
    cam = load_camera_json(out / "camera.json")
    kf, c2w = _read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
    m1, m2, K = undistort_maps(cam)
    tags = {}
    try:
        from intake.content import load_content
        tags = {int(f): t for f, t in (load_content(session_dir).get("frames") or {}).items()}
    except Exception:                                   # I2 absent: no weights, declared
        tags = {}
    return rep, epochs, cam, kf, c2w, (m1, m2), K, tags


def _rgb_undistorted(frames_dir: Path, frame: int, maps) -> np.ndarray:
    import cv2
    from intake.content import frame_file
    bgr = cv2.imread(str(frame_file(frames_dir, frame)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FuseError(f"cannot read frame {frame}")
    und = cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    return und[..., ::-1]


def _candidate_codes(src: np.ndarray, n_cons: np.ndarray, fcfg, dcfg):
    """(entering mask, reject reason per pixel (0 = none / not a candidate))."""
    from precision.depth_sweep import (DISCARD_EXCLUDED, DISCARD_INCONSISTENT,
                                       DISCARD_PRIOR_FILL_DROPPED, SOURCE_PRIOR_FILL,
                                       SOURCE_SWEEP)
    R = PV.REJECT_REASONS
    t0 = src == SOURCE_SWEEP
    t1 = src == SOURCE_PRIOR_FILL
    enter = (t0 & (n_cons >= int(fcfg.min_witness_views))) | \
        (t1 & (n_cons >= int(dcfg.prior_fill_min_views)))
    reason = np.zeros(src.shape, np.uint8)
    reason[(t0 | t1) & ~enter] = R["insufficient_witnesses"]
    reason[src == DISCARD_INCONSISTENT] = R["inconsistent"]
    reason[src == DISCARD_PRIOR_FILL_DROPPED] = R["prior_fill_dropped"]
    reason[src == DISCARD_EXCLUDED] = R["excluded_mask"]
    return enter, reason


def _frame_codes(src, ncons, c2w, fcfg, dcfg):
    """The one rule both passes read: entering mask and reject reasons of a frame; a
    keyframe without a finite pose testifies for nothing (unlocalized_frame)."""
    enter, reason = _candidate_codes(src, ncons, fcfg, dcfg)
    if not np.all(np.isfinite(c2w)):
        reason[enter] = PV.REJECT_REASONS["unlocalized_frame"]
        enter[:] = False
    return enter, reason


def fuse(session_dir: Path, pcfg, log: Callable = print) -> Dict[str, Any]:
    """Build the fused cloud; returns {'data': structured PLY array, 'header':
    PLY header lines, 'origins': v2 columns, 'rejected': columns, 'report': dict}."""
    from precision.depth_sweep import DEPTH_DIRNAME, SOURCE_PRIOR_FILL, SOURCE_SWEEP
    fcfg, dcfg = pcfg.fuse, pcfg.depth
    if fcfg.noise_filter:
        raise FuseError("precision.fuse.noise_filter: CloudComPy's noise filter runs in its "
                        "own env (CloudComPy310) and is not available to the fusion — set it "
                        "false (SOR is available: precision.fuse.sor)")
    session_dir = Path(session_dir)
    out = session_dir / "output"
    ddir = out / DEPTH_DIRNAME
    rep, epochs, cam, kf, c2w, maps, K, tags = _load_inputs(session_dir, pcfg)
    t0 = time.time()
    voxel = float(fcfg.voxel_m)
    # ── pass 1: who enters, its voxel and its evidence ───────────────────
    keys, score, fidx, plin = [], [], [], []
    n_cand = 0
    rej_counts = {k: 0 for k in PV.REJECT_REASONS}
    frames_used = []
    for i, f in enumerate(kf):
        p = ddir / f"frame_{f}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            depth, src, ncons, ncc = z["depth"], z["source"], z["n_consistent"], z["ncc"]
        enter, reason = _frame_codes(src, ncons, c2w[i], fcfg, dcfg)
        n_cand += int(enter.sum() + (reason > 0).sum())
        rr, cc = np.nonzero(enter)
        if rr.size:
            X = backproject(depth, rr, cc, K, c2w[i])
            keys.append(pack_keys(np.floor(X / voxel).astype(np.int64)))
            score.append(evidence(ncons[rr, cc], ncc[rr, cc]).astype(np.float32))
            fidx.append(np.full(rr.size, i, np.int32))
            plin.append((rr * depth.shape[1] + cc).astype(np.int32))
        frames_used.append(f)
    if not keys:
        raise FuseError("no pixel of depth_native passes the witness rule — nothing to fuse")
    keys = np.concatenate(keys)
    score = np.concatenate(score)
    fidx = np.concatenate(fidx)
    plin = np.concatenate(plin)
    # total order: voxel, then best evidence, then earliest frame, then pixel
    order = np.lexsort((plin, fidx, -score.astype(np.float64), keys))
    first = np.ones(order.size, bool)
    first[1:] = keys[order[1:]] != keys[order[:-1]]
    win = np.zeros(keys.size, bool)
    win[order[first]] = True
    n_entered = int(keys.size)
    del keys, score, order, first
    log(f"{LOG_TAG} {n_entered:,} pixel(s) pass the witness rule over {len(frames_used)} "
        f"keyframe(s); {int(win.sum()):,} voxel winner(s) at {voxel * 1000:.1f} mm")

    # ── pass 2: rebuild the winners; every other candidate gets its reason ─
    cols: Dict[str, List[np.ndarray]] = {k: [] for k in PV.V2_FIELDS}
    xyz_l, rgb_l = [], []
    rej: Dict[str, List[np.ndarray]] = {k: [] for k in PV.V2_FIELDS + ("reason",)}
    starts = np.flatnonzero(np.r_[True, fidx[1:] != fidx[:-1]])
    bounds = dict(zip(fidx[starts].tolist(), zip(starts.tolist(), np.r_[starts[1:], fidx.size].tolist())))
    ge, ce = int(epochs["geometry_epoch"]), int(epochs["camera_epoch"])

    def _cols(dst, f, rows, cc, ncons, ncc, src, res, flags):
        dst["frame_global"].append(np.full(rows.size, f, np.int32))
        dst["pixel_row"].append(np.rint(maps[1][rows, cc]).astype(np.int32))
        dst["pixel_col"].append(np.rint(maps[0][rows, cc]).astype(np.int32))
        dst["pixel_v_und"].append(rows.astype(np.int32))
        dst["pixel_u_und"].append(cc.astype(np.int32))
        dst["n_consistent"].append(ncons[rows, cc].astype(np.uint8))
        dst["ncc"].append(ncc[rows, cc].astype(np.float32))
        dst["source"].append(src[rows, cc].astype(np.uint8))
        dst["residual_rel"].append(res[rows, cc].astype(np.float32))
        dst["content_flags"].append(np.full(rows.size, flags, np.uint8))
        dst["geometry_epoch"].append(np.full(rows.size, ge, np.int16))
        dst["camera_epoch"].append(np.full(rows.size, ce, np.int16))

    for i, f in enumerate(kf):
        p = ddir / f"frame_{f}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            depth, src, ncons, ncc = z["depth"], z["source"], z["n_consistent"], z["ncc"]
            res = z["residual_rel"].astype(np.float32)
        enter, reason = _frame_codes(src, ncons, c2w[i], fcfg, dcfg)
        flags = PV.content_flags_of(tags.get(int(f)))
        W = depth.shape[1]
        if i in bounds:
            a, b = bounds[i]
            lin = plin[a:b]
            w = win[a:b]
            rows, cc = lin[w] // W, lin[w] % W
            X = backproject(depth, rows, cc, K, c2w[i])
            rgb = _rgb_undistorted(session_dir / "frames", f, maps)[rows, cc]
            xyz_l.append(X)
            rgb_l.append(rgb.astype(np.uint8))
            _cols(cols, f, rows, cc, ncons, ncc, src, res, flags)
            lr, lc = lin[~w] // W, lin[~w] % W
            if lr.size:
                _cols(rej, f, lr, lc, ncons, ncc, src, res, flags)
                rej["reason"].append(np.full(lr.size, PV.REJECT_REASONS["dedup"], np.uint8))
                rej_counts["dedup"] += int(lr.size)
        rr, rc = np.nonzero(reason > 0)
        if rr.size:
            _cols(rej, f, rr, rc, ncons, ncc, src, res, flags)
            rej["reason"].append(reason[rr, rc])
            for code in np.unique(reason[rr, rc]):
                rej_counts[PV.REJECT_NAMES[int(code)]] += int(np.sum(reason[rr, rc] == code))
    xyz = np.concatenate(xyz_l)
    rgb = np.concatenate(rgb_l)
    origins = {k: np.concatenate(v) for k, v in cols.items()}
    rejected = {k: (np.concatenate(v) if v else np.zeros(0, PV.V2_DTYPES.get(k, np.uint8)))
                for k, v in rej.items()}

    if fcfg.sor:
        from reconstruction.gpu_cloud_clean import _sor_keep
        keep, mu, sd = _sor_keep(xyz, int(fcfg.sor_knn), float(fcfg.sor_std),
                                 float(fcfg.sor_cell_m))
        drop = ~keep
        for k in PV.V2_FIELDS:
            rejected[k] = np.concatenate([rejected[k], origins[k][drop]])
            origins[k] = origins[k][keep]
        rejected["reason"] = np.concatenate([rejected["reason"],
                                             np.full(int(drop.sum()), PV.REJECT_REASONS["sor"], np.uint8)])
        rej_counts["sor"] += int(drop.sum())
        xyz, rgb = xyz[keep], rgb[keep]
        log(f"{LOG_TAG} SOR (k {fcfg.sor_knn}, {fcfg.sor_std} σ) removed {int(drop.sum()):,} point(s)")

    data = cloud_array(xyz, rgb, origins)
    n_rej = int(len(rejected["reason"]))
    if len(data) + n_rej != n_cand:
        raise FuseError(f"{len(data)} points + {n_rej} rejected ≠ {n_cand} candidates — the "
                        f"accounting does not close")
    tiers = origins["source"]
    report = {"version": FUSE_VERSION, "provenance": PROVENANCE, **epochs,
              "params": {"fuse": {k: getattr(fcfg, k) for k in fcfg.__dataclass_fields__},
                         "prior_fill": dcfg.prior_fill,
                         "prior_fill_min_views": dcfg.prior_fill_min_views},
              "n_keyframes": len(frames_used), "n_candidates": n_cand,
              "n_points": int(len(data)), "n_rejected": n_rej,
              "rejected_by_reason": rej_counts,
              "points_by_tier": {"tier0": int(np.sum(tiers == SOURCE_SWEEP)),
                                 "tier1_prior_fill": int(np.sum(tiers == SOURCE_PRIOR_FILL))},
              "witness_histogram": {str(int(v)): int(c) for v, c in
                                    zip(*np.unique(origins["n_consistent"], return_counts=True))},
              "seconds": round(time.time() - t0, 1),
              "viewer_channels": {"confidence": "ZNCC for tier 0, 0 for tier 1 (slider > 0 "
                                                "hides tier 1)",
                                  "mv_votes": "consistent views", "status": "verified"}}
    return {"data": data, "origins": origins, "rejected": rejected, "report": report}


PLY_FIELDS = [("x", "<f8", "double"), ("y", "<f8", "double"), ("z", "<f8", "double"),
              ("red", "u1", "uchar"), ("green", "u1", "uchar"), ("blue", "u1", "uchar"),
              ("frame_global", "<i4", "int"), ("pixel_row", "<i4", "int"),
              ("pixel_col", "<i4", "int"), ("pixel_u_und", "<i4", "int"),
              ("pixel_v_und", "<i4", "int"), ("n_consistent", "u1", "uchar"),
              ("ncc", "<f4", "float"), ("source", "u1", "uchar"),
              ("residual_rel", "<f4", "float"), ("content_flags", "u1", "uchar"),
              ("confidence", "<f4", "float"), ("mv_votes", "u1", "uchar"),
              ("status", "u1", "uchar")]


def cloud_array(xyz: np.ndarray, rgb: np.ndarray, o: Dict[str, np.ndarray]) -> np.ndarray:
    from precision.depth_sweep import SOURCE_SWEEP
    from reconstruction.witness.status import STATUS_CODES
    data = np.empty(len(xyz), dtype=[(n, t) for n, t, _ in PLY_FIELDS])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for k in ("frame_global", "pixel_row", "pixel_col", "pixel_u_und", "pixel_v_und",
              "n_consistent", "ncc", "source", "residual_rel", "content_flags"):
        data[k] = o[k]
    data["confidence"] = np.where(o["source"] == SOURCE_SWEEP,
                                  np.clip(np.nan_to_num(o["ncc"], nan=0.0), 0.0, 1.0), 0.0)
    data["mv_votes"] = o["n_consistent"]
    data["status"] = STATUS_CODES["verified"]
    return data


def ply_header(n: int) -> List[bytes]:
    lines = ["ply", "format binary_little_endian 1.0",
             "comment STAC F7 witness fusion (precision/fuse.py) - provenance v2",
             f"element vertex {n}"]
    lines += [f"property {ply} {name}" for name, _t, ply in PLY_FIELDS]
    lines.append("end_header")
    return [(ln + "\n").encode("ascii") for ln in lines]


# ── publication ──────────────────────────────────────────────────────────

def publish(session_dir: Path, result: Dict[str, Any], log: Callable = print) -> Dict[str, Any]:
    """Stage the fused cloud as a new geometry epoch and swap it in (epoch N stays
    selectable)."""
    from correction import ledger
    from correction.apply import (TX_PREFIX, assert_no_interrupted_swap, next_epoch,
                                  swap_transaction)
    from correction.epoch import EPOCH_FILE, current_epoch, make_epoch_record
    from correction.session import write_ply
    out = Path(session_dir) / "output"
    assert_no_interrupted_swap(out)
    epoch_from, epoch_to = current_epoch(out), next_epoch(out)
    tx = out / f"{TX_PREFIX}{epoch_to}"
    if tx.exists():
        shutil.rmtree(tx)
    tx.mkdir(parents=True)
    arts: List[dict] = []

    def art(rel):
        arts.append({"rel": rel, "existed_before": (out / rel).exists()})

    data = result["data"]
    header = ply_header(len(data))
    write_ply(tx / "cleaned_cloud.ply", header, data)
    art("cleaned_cloud.ply")
    # the raw twin: nothing consolidates the fused cloud — it IS the measurement
    write_ply(tx / "cleaned_cloud_raw.ply", header, data)
    art("cleaned_cloud_raw.ply")
    cid = ledger.new_correction_id()
    meta = {"stage": "fuse", "correction_id": cid, "epoch": epoch_to}
    o = dict(result["origins"])
    o["geometry_epoch"] = np.full(len(data), epoch_to, np.int16)
    PV.write_origins(tx / PV.ORIGINS_NAME, o, meta)
    art(PV.ORIGINS_NAME)
    r = dict(result["rejected"])
    np.savez(tx / PV.REJECTED_NAME, version=np.int64(PV.ORIGINS_VERSION),
             reasons=np.array(json.dumps(PV.REJECT_REASONS)),
             **{k: np.asarray(v) for k, v in r.items()})
    art(PV.REJECTED_NAME)
    rep = dict(result["report"], epoch_to=epoch_to, epoch_from=epoch_from, correction_id=cid)
    (tx / FUSE_REPORT_NAME).write_text(json.dumps(rep, indent=1, default=float))
    art(FUSE_REPORT_NAME)
    # the epoch-0 semantics index another cloud: pending until F8 rebuilds them
    if (out / "segmentation_result.json").exists():
        (tx / "segmentation_result.json").write_text(json.dumps(
            {"instances": [], "pending": "the semantics are rebuilt on this epoch by F8 "
                                         "(claude_stac.txt §4-F8) — the fused cloud is new",
             "geometry_epoch": epoch_to}))
        art("segmentation_result.json")
    for sib, dt in (("classification.npy", np.uint8), ("out_of_place.npy", bool)):
        if (out / sib).exists():
            np.save(tx / sib, np.zeros(len(data), dt))
            art(sib)
    (tx / EPOCH_FILE).write_text(json.dumps(make_epoch_record(epoch_to, cid, epoch_from),
                                            indent=1))
    art(EPOCH_FILE)
    from potree_converter import convert_ply_to_potree
    ok = convert_ply_to_potree(Path(session_dir), force=True,
                               ply_override=tx / "cleaned_cloud.ply",
                               potree_dir_override=tx / "potree")
    if not ok or not (tx / "potree" / "metadata.json").exists():
        shutil.rmtree(tx)
        raise FuseError("the Potree build failed inside the transaction — nothing was "
                        "published, the session is untouched")
    art("potree")
    swap_transaction(out, {"tx_dir": str(tx), "epoch_from": epoch_from,
                           "epoch_to": epoch_to, "artifacts": arts}, log=log)
    ledger.record_run(out, correction_id=cid, epoch_from=epoch_from, epoch_to=epoch_to,
                      kind="fuse", operator="auto", instance_ids=[], visits=[],
                      observability=[], anchors=[], diagnosis=[], gates=[], overrides={},
                      verdict="applied", report_path=FUSE_REPORT_NAME)
    log(f"{LOG_TAG} published epoch {epoch_to} ({len(data):,} points; epoch {epoch_from} "
        f"kept, selectable)")
    return rep


def run_fuse(session_dir: Path, pcfg, *, publish_epoch: bool = True,
             log: Callable = print) -> Dict[str, Any]:
    res = fuse(session_dir, pcfg, log=log)
    rep = res["report"]
    log(f"{LOG_TAG} {rep['n_points']:,} point(s) (tier 0 {rep['points_by_tier']['tier0']:,}, "
        f"tier 1 {rep['points_by_tier']['tier1_prior_fill']:,}), {rep['n_rejected']:,} rejected "
        f"{ {k: v for k, v in rep['rejected_by_reason'].items() if v} }")
    if publish_epoch:
        return publish(session_dir, res, log=log)
    out = Path(session_dir) / "output" / "precision"
    out.mkdir(parents=True, exist_ok=True)
    (out / FUSE_REPORT_NAME).write_text(json.dumps(rep, indent=1, default=float))
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.fuse",
                                 description="Witness fusion + provenance v2 (F7).")
    ap.add_argument("--session", required=True)
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args(argv)
    run_fuse(Path(args.session), load_precision_config(), publish_epoch=not args.no_publish)
    return 0


if __name__ == "__main__":
    sys.exit(main())
