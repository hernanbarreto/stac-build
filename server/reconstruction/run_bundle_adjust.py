"""
Global bundle-adjustment runner — wires the VGGSfM tracks + backbone depth/intrinsics
+ initial poses into the full landmark BA, then writes the refined keyframe poses back.
================================================================================
Pipeline (Tier-1 pose refinement):
  1. load tracks.npz (learned correspondences)            ← reconstruction.vggt_tracks
  2. load initial poses (camera_poses.txt + camera_frames.txt)
  3. load per-frame intrinsics K + metric depth (backbone npz, at the track resolution)
  4. init each landmark by unprojecting its query observation through depth+pose
  5. run the Schur-complement BA (reprojection + depth-prior scale anchor)
                                                          ← reconstruction.bundle_adjust
  6. write the refined poses back into every camera_poses.txt copy, row-aligned with
     camera_frames.txt; FILLER lines (frames not optimised) are left untouched.

Only frames that appear in the tracks (the keyframes the tracker windowed over) are
optimised; the original is backed up to camera_poses.txt.preba first.

Run (mapanything env), after vggt_tracks produced tracks.npz:
    python -m reconstruction.run_bundle_adjust --output-dir <out> [--smoke] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reconstruction.bundle_adjust import bundle_adjust

logger = logging.getLogger("RunBA")


def _load_poses(output_dir: Path) -> Tuple[Dict[int, np.ndarray], Path]:
    """Return ({frame_num: c2w 4x4}, the camera_poses.txt used). Reads camera_frames.txt
    for the real numbers; requires row-aligned files."""
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if not pp.exists():
            continue
        mats = [np.array([float(v) for v in ln.split()], np.float64).reshape(4, 4)
                for ln in pp.read_text().splitlines() if len(ln.split()) == 16]
        if fp.exists():
            nums = [int(float(x)) for x in fp.read_text().split()]
        else:
            nums = list(range(len(mats)))
        if len(nums) == len(mats) and mats:
            return {nums[i]: mats[i] for i in range(len(mats))}, pp
    raise FileNotFoundError("no row-aligned camera_poses.txt/camera_frames.txt found")


def _npz_dir(output_dir: Path) -> Optional[Path]:
    for sub in ("da3_run/results_output", "results_output", "gaus_slam_run/results_output"):
        d = output_dir / sub
        if d.exists() and any(d.glob("frame_*.npz")):
            return d
    return None


def _load_frame_KD(npz_dir: Path, num: int, res_w: int, res_h: int
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (K 3x3 scaled to res, depth (res_h,res_w)) for frame `num`, or None."""
    p = npz_dir / f"frame_{num}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        depth = np.asarray(z["depth"], np.float32)
        K = np.asarray(z["intrinsics"], np.float64).copy()
        h, w = depth.shape[:2]
        if (w, h) != (res_w, res_h):
            # scale K and nearest-resize depth to the track resolution
            sx, sy = res_w / w, res_h / h
            K[0, :] *= sx; K[1, :] *= sy
            yi = (np.arange(res_h) * h / res_h).astype(int).clip(0, h - 1)
            xi = (np.arange(res_w) * w / res_w).astype(int).clip(0, w - 1)
            depth = depth[yi][:, xi]
        return K, depth
    except Exception as e:
        logger.warning(f"frame {num} npz load failed: {e}")
        return None


def _sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel depth at pixel coords uv (N,2) x=col,y=row. 0 where out of range."""
    h, w = depth.shape
    x = np.round(uv[:, 0]).astype(int)
    y = np.round(uv[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros(len(uv), np.float32)
    out[ok] = depth[y[ok], x[ok]]
    return out


def run(output_dir: Path, smoke: bool = False, dry_run: bool = False,
        depth_w: float = 0.1, iters: int = 12, min_views: int = 2) -> Optional[dict]:
    t0 = time.time()
    tracks_path = output_dir / "ba_run" / ("tracks_smoke.npz" if smoke else "tracks.npz")
    if not tracks_path.exists():
        logger.error(f"no tracks at {tracks_path} — run reconstruction.vggt_tracks first")
        return None
    tr = np.load(tracks_path)
    res_w, res_h = int(tr["res_w"]), int(tr["res_h"])
    obs_track = tr["obs_track"]; obs_frame = tr["obs_frame"]; obs_uv = tr["obs_uv"]
    obs_score = tr["obs_score"]
    tq_id = tr["track_query_id"]; tq_frame = tr["track_query_frame"]; tq_uv = tr["track_query_uv"]
    qmap = {int(tq_id[k]): (int(tq_frame[k]), tq_uv[k]) for k in range(len(tq_id))}
    logger.info(f"tracks: {len(obs_track):,} obs, {len(set(obs_track.tolist())):,} tracks, "
                f"res {res_w}×{res_h}")

    poses_c2w, _src = _load_poses(output_dir)
    npz_dir = _npz_dir(output_dir)
    if npz_dir is None:
        logger.error("no backbone results_output/*.npz (depth+K) found")
        return None

    # frames we can optimise: appear in tracks, have a pose AND an npz (K+depth)
    track_frames = sorted(set(obs_frame.tolist()))
    KD: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    use_frames: List[int] = []
    for fn in track_frames:
        if fn not in poses_c2w:
            continue
        kd = _load_frame_KD(npz_dir, fn, res_w, res_h)
        if kd is None:
            continue
        KD[fn] = kd
        use_frames.append(fn)
    if len(use_frames) < 3:
        logger.error(f"only {len(use_frames)} usable frames — abort")
        return None
    fidx = {fn: i for i, fn in enumerate(use_frames)}
    N = len(use_frames)

    poses_w2c = np.stack([np.linalg.inv(poses_c2w[fn]) for fn in use_frames])  # (N,4,4)
    K_arr = np.stack([KD[fn][0] for fn in use_frames])                          # (N,3,3)

    # ── observations (keep only those on usable frames) ──
    keep = np.array([fn in fidx for fn in obs_frame])
    o_tr = obs_track[keep]; o_fn = obs_frame[keep]; o_uv = obs_uv[keep]; o_sc = obs_score[keep]
    cam_idx = np.array([fidx[fn] for fn in o_fn], np.int64)
    # depth at each observation (for the scale-anchoring depth prior)
    o_dep = np.zeros(len(o_tr), np.float32)
    for fn in use_frames:
        m = o_fn == fn
        if m.any():
            o_dep[m] = _sample_depth(KD[fn][1], o_uv[m])

    # remap track ids → contiguous, keep only tracks with ≥min_views usable obs
    uniq, inv = np.unique(o_tr, return_inverse=True)
    counts = np.bincount(inv)
    good_t = counts >= min_views
    keep2 = good_t[inv]
    inv = inv[keep2]
    cam_idx = cam_idx[keep2]; o_uv = o_uv[keep2]; o_sc = o_sc[keep2]; o_dep = o_dep[keep2]
    # compact track ids
    tid_used, pt_idx = np.unique(inv, return_inverse=True)
    M = len(tid_used)
    P = len(pt_idx)
    logger.info(f"BA problem: N={N} poses, M={M:,} landmarks, P={P:,} observations")

    # ── landmark init: unproject each track's query observation through depth+pose ──
    orig_tid = uniq[tid_used]                          # original track id per compact pt
    pts_w = np.zeros((M, 3), np.float64)
    n_bad = 0
    for j in range(M):
        ot = int(orig_tid[j])
        q = qmap.get(ot)
        qf, quv = (q[0], q[1]) if q is not None else (-1, None)
        X = None
        if qf in KD and quv is not None:
            d = _sample_depth(KD[qf][1], quv[None, :])[0]
            if d > 1e-3:
                K = KD[qf][0]
                x = (quv[0] - K[0, 2]) * d / K[0, 0]
                y = (quv[1] - K[1, 2]) * d / K[1, 1]
                Xc = np.array([x, y, d, 1.0])
                X = (poses_c2w[qf] @ Xc)[:3]
        if X is None:
            # fall back: unproject the first observation of this track
            sel = np.where(pt_idx == j)[0]
            if len(sel):
                ci = use_frames[cam_idx[sel[0]]]
                d = o_dep[sel[0]]
                if d > 1e-3:
                    K = KD[ci][0]; uv = o_uv[sel[0]]
                    x = (uv[0] - K[0, 2]) * d / K[0, 0]
                    y = (uv[1] - K[1, 2]) * d / K[1, 1]
                    X = (poses_c2w[ci] @ np.array([x, y, d, 1.0]))[:3]
        if X is None:
            X = np.array([0.0, 0.0, 1.0]); n_bad += 1
        pts_w[j] = X
    if n_bad:
        logger.warning(f"{n_bad}/{M} landmarks had no depth → placeholder init")

    # ── run BA ──
    ref_w2c, _pts, info = bundle_adjust(
        poses_w2c, K_arr, cam_idx, pt_idx, o_uv.astype(np.float64), pts_w,
        obs_w=o_sc.astype(np.float64), depth_obs=o_dep.astype(np.float64),
        depth_w=depth_w, iters=iters, huber_px=2.0, fix_first=True,
        verbose=True)
    logger.info(f"BA done: reproj_rms={info['reproj_rms_px']:.3f}px "
                f"({info['n_obs']:,} obs) in {time.time()-t0:.0f}s")

    refined_c2w = {use_frames[i]: np.linalg.inv(ref_w2c[i]) for i in range(N)}

    # report how much poses moved (sanity)
    moved = np.array([np.linalg.norm(refined_c2w[fn][:3, 3] - poses_c2w[fn][:3, 3])
                      for fn in use_frames])
    logger.info(f"pose centre shift: median {np.median(moved):.3f} m, max {moved.max():.3f} m")

    if dry_run:
        logger.info("[dry-run] not writing poses back")
        return {"info": info, "moved_median": float(np.median(moved)), "n_opt": N}

    _writeback(output_dir, refined_c2w)
    logger.info(f"✅ refined {N} keyframe poses written back in {time.time()-t0:.0f}s")
    return {"info": info, "moved_median": float(np.median(moved)), "n_opt": N}


def _writeback(output_dir: Path, refined_c2w: Dict[int, np.ndarray]) -> None:
    """Overwrite the refined frames' lines in every camera_poses.txt copy (row-aligned
    with camera_frames.txt). Filler/un-optimised lines are kept verbatim. Backs up first."""
    for base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if not pp.exists() or not fp.exists():
            continue
        plines = pp.read_text().splitlines()
        nums = [int(float(x)) for x in fp.read_text().split()]
        if len([l for l in plines if len(l.split()) == 16]) != len(nums):
            logger.warning(f"{pp}: poses/frames not aligned — skipping writeback there")
            continue
        bak = pp.with_suffix(".txt.preba")
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
    ap.add_argument("--dry-run", action="store_true", help="run BA but don't write poses")
    ap.add_argument("--depth-w", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    run(Path(args.output_dir), smoke=args.smoke, dry_run=args.dry_run,
        depth_w=args.depth_w, iters=args.iters)


if __name__ == "__main__":
    main()
