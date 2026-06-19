"""
Runner: refine MapAnything poses with the COLMAP/Ceres pose-prior BA over VGGSfM tracks.
================================================================================
Wires the learned correspondences (reconstruction.vggt_tracks → tracks.npz) + the
MapAnything metric poses + per-frame intrinsics/depth into pycolmap's pose-prior
bundle adjuster (reconstruction.colmap_ba), then writes the refined poses back.

This is the MapAnything-backend pose refinement (the strong baseline to compare
against the VGGT-Omega path). MapAnything is already metric, so no scale step — the
pose priors keep it metric while the tracks fix relative drift.

Run (mapanything env), after vggt_tracks produced tracks.npz:
    python -m reconstruction.run_colmap_ba --output-dir <out> [--smoke] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from reconstruction.run_bundle_adjust import (
    _load_poses, _npz_dir, _load_frame_KD, _sample_depth)
from reconstruction.colmap_ba import refine_poses_ba

logger = logging.getLogger("RunColmapBA")


def run(output_dir: Path, smoke: bool = False, dry_run: bool = False,
        prior_stddev_m: float = 0.10, min_views: int = 2) -> Dict:
    tracks_path = output_dir / "ba_run" / ("tracks_smoke.npz" if smoke else "tracks.npz")
    if not tracks_path.exists():
        logger.error(f"no tracks at {tracks_path} — run reconstruction.vggt_tracks first")
        return {}
    tr = np.load(tracks_path)
    res_w, res_h = int(tr["res_w"]), int(tr["res_h"])
    obs_frame = tr["obs_frame"]; obs_uv = tr["obs_uv"]; obs_track = tr["obs_track"]
    obs_score = tr["obs_score"]
    tq_id = tr["track_query_id"]; tq_frame = tr["track_query_frame"]; tq_uv = tr["track_query_uv"]
    qmap = {int(tq_id[k]): (int(tq_frame[k]), tq_uv[k]) for k in range(len(tq_id))}
    logger.info(f"tracks: {len(obs_frame):,} obs, {len(set(obs_track.tolist())):,} tracks, "
                f"res {res_w}×{res_h}")

    poses_c2w, _ = _load_poses(output_dir)
    npz_dir = _npz_dir(output_dir)
    if npz_dir is None:
        logger.error("no backbone results_output/*.npz (depth+K)")
        return {}

    # usable frames: in tracks, with a pose AND an npz (K+depth)
    KD: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    use_frames: List[int] = []
    for fn in sorted(set(obs_frame.tolist())):
        if fn not in poses_c2w:
            continue
        kd = _load_frame_KD(npz_dir, fn, res_w, res_h)
        if kd is None:
            continue
        KD[fn] = kd; use_frames.append(fn)
    if len(use_frames) < 3:
        logger.error(f"only {len(use_frames)} usable frames — abort")
        return {}
    fidx = {fn: i for i, fn in enumerate(use_frames)}
    N = len(use_frames)
    poses_w2c = np.stack([np.linalg.inv(poses_c2w[fn]) for fn in use_frames])
    K_arr = np.stack([KD[fn][0] for fn in use_frames])

    keep = np.array([fn in fidx for fn in obs_frame])
    o_tr = obs_track[keep]; o_fn = obs_frame[keep]; o_uv = obs_uv[keep]; o_sc = obs_score[keep]
    obs_img = np.array([fidx[fn] for fn in o_fn], np.int64)
    # compact track ids; keep tracks with ≥min_views obs
    uniq, inv = np.unique(o_tr, return_inverse=True)
    good = np.bincount(inv) >= min_views
    km = good[inv]
    inv = inv[km]; obs_img = obs_img[km]; o_uv = o_uv[km]; o_sc = o_sc[km]
    tid_used, obs_pt = np.unique(inv, return_inverse=True)
    M = len(tid_used)
    orig_tid = uniq[tid_used]
    logger.info(f"BA problem: N={N} poses, M={M:,} landmarks, P={len(obs_pt):,} obs")

    # landmark init: unproject each track's query observation through depth+pose
    pts_w = np.zeros((M, 3), np.float64); n_bad = 0
    for j in range(M):
        q = qmap.get(int(orig_tid[j])); X = None
        if q is not None and q[0] in KD:
            qf, quv = q
            d = _sample_depth(KD[qf][1], quv[None, :])[0]
            if d > 1e-3:
                Kf = KD[qf][0]
                x = (quv[0] - Kf[0, 2]) * d / Kf[0, 0]
                y = (quv[1] - Kf[1, 2]) * d / Kf[1, 1]
                X = (poses_c2w[qf] @ np.array([x, y, d, 1.0]))[:3]
        if X is None:
            sel = np.where(obs_pt == j)[0]
            if len(sel):
                ci = use_frames[obs_img[sel[0]]]; d = _sample_depth(KD[ci][1], o_uv[sel[0]][None, :])[0]
                if d > 1e-3:
                    Kf = KD[ci][0]; uv = o_uv[sel[0]]
                    x = (uv[0] - Kf[0, 2]) * d / Kf[0, 0]; y = (uv[1] - Kf[1, 2]) * d / Kf[1, 1]
                    X = (poses_c2w[ci] @ np.array([x, y, d, 1.0]))[:3]
        if X is None:
            X = np.array([0.0, 0.0, 1.0]); n_bad += 1
        pts_w[j] = X
    if n_bad:
        logger.warning(f"{n_bad}/{M} landmarks had no depth → placeholder")

    ref_w2c, info = refine_poses_ba(
        poses_w2c, K_arr, (res_w, res_h), obs_img, obs_pt, o_uv.astype(np.float64), pts_w,
        prior_stddev_m=prior_stddev_m)
    logger.info(f"COLMAP pose-prior BA: reproj {info['reproj_before_px']:.3f}→"
                f"{info['reproj_after_px']:.3f}px  ({info['n_points']:,} pts, {info['n_obs']:,} obs)")

    refined_c2w = {use_frames[i]: np.linalg.inv(ref_w2c[i]) for i in range(N)}
    moved = np.array([np.linalg.norm(refined_c2w[fn][:3, 3] - poses_c2w[fn][:3, 3])
                      for fn in use_frames])
    logger.info(f"pose centre shift: median {np.median(moved):.3f} m, max {moved.max():.3f} m")

    if dry_run:
        logger.info("[dry-run] not writing poses")
        return {"info": info, "moved_median": float(np.median(moved)), "n_opt": N}
    _writeback(output_dir, refined_c2w)
    logger.info(f"✅ refined {N} keyframe poses written back")
    return {"info": info, "moved_median": float(np.median(moved)), "n_opt": N}


def _writeback(output_dir: Path, refined_c2w: Dict[int, np.ndarray]) -> None:
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if not pp.exists() or not fp.exists():
            continue
        plines = pp.read_text().splitlines()
        nums = [int(float(x)) for x in fp.read_text().split()]
        if len([l for l in plines if len(l.split()) == 16]) != len(nums):
            logger.warning(f"{pp}: poses/frames not aligned — skip"); continue
        bak = pp.with_suffix(".txt.precolmapba")
        if not bak.exists():
            shutil.copy(pp, bak)
        out, pi = [], 0
        for ln in plines:
            if len(ln.split()) == 16:
                fn = nums[pi]; pi += 1
                if fn in refined_c2w:
                    ln = " ".join(f"{v:.8g}" for v in refined_c2w[fn].reshape(-1))
            out.append(ln)
        pp.write_text("\n".join(out) + "\n")
        logger.info(f"  wrote {pp} (backup {bak.name})")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prior-stddev-m", type=float, default=0.10)
    a = ap.parse_args()
    res = run(Path(a.output_dir), smoke=a.smoke, dry_run=a.dry_run, prior_stddev_m=a.prior_stddev_m)
    if not res:                           # NO FALLBACK: the BA did not run → fail
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
