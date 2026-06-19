"""
Densification from the two-pass BA filler poses.
================================================================================
Replaces the old ICP dense-fusion. The two-pass bundle adjustment (run_colmap_ba)
localised the FILLER frames against the fixed keyframe map (→ ba_run/filler_poses.npz,
globally consistent with the BA-refined keyframes). Here we back-project each filler's
DA3 metric depth at its BA pose → extra cloud points with inter-keyframe coverage,
written as a chunk PLY + origins that CloudCompPy merges. Same confidence filter as the
backbone cloud, so the densified points are denoised the same way.

    python -m reconstruction.densify_fillers --output-dir <out>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("DensifyFillers")

CHUNK = "chunk_997_densefusion"


def _npz_dir(output_dir: Path):
    for sub in ("da3_run/results_output", "results_output"):
        d = output_dir / sub
        if d.exists() and any(d.glob("frame_*.npz")):
            return d
    return None


def run(output_dir: Path, stride: int = 2, conf_coef: float = 0.75,
        depth_min: float = 0.1, depth_max: float = 8.0) -> int:
    fp = output_dir / "ba_run" / "filler_poses.npz"
    if not fp.exists():
        logger.warning("no ba_run/filler_poses.npz — nothing to densify (BA localised no fillers)")
        return 0
    z = np.load(fp)
    frames = z["frames"].astype(np.int64); c2ws = z["c2w"].astype(np.float64)
    npz_dir = _npz_dir(output_dir)
    if npz_dir is None:
        raise FileNotFoundError("no DA3 results_output/*.npz for densification")

    all_xyz, all_rgb, all_fg, all_pr, all_pc, all_cf = [], [], [], [], [], []
    n_used = 0
    for fi in range(len(frames)):
        fn = int(frames[fi]); c2w = c2ws[fi]
        p = npz_dir / f"frame_{fn}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        depth = np.asarray(d["depth"], np.float32)
        K = np.asarray(d["intrinsics"], np.float64)
        conf = np.asarray(d["conf"], np.float32) if "conf" in d else np.ones_like(depth)
        img = np.asarray(d["image"], np.uint8) if "image" in d else None
        H, W = depth.shape
        thr = float(conf[conf > 1e-5].mean()) * conf_coef if (conf > 1e-5).any() else 0.0
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        dd = depth[::stride, ::stride]; cc = conf[::stride, ::stride]
        m = (dd > depth_min) & (dd < depth_max) & np.isfinite(dd) & (cc >= thr)
        if not m.any():
            continue
        u, v, dv = uu[m], vv[m], dd[m]
        x = (u - K[0, 2]) * dv / K[0, 0]; y = (v - K[1, 2]) * dv / K[1, 1]
        cam = np.stack([x, y, dv], 1)
        world = cam @ c2w[:3, :3].T + c2w[:3, 3]
        all_xyz.append(world.astype(np.float32))
        if img is not None:
            all_rgb.append(img[v, u][:, :3].astype(np.uint8))
        else:
            all_rgb.append(np.full((len(u), 3), 200, np.uint8))
        all_fg.append(np.full(len(u), fn, np.int32))
        all_pr.append(v.astype(np.int16)); all_pc.append(u.astype(np.int16))
        all_cf.append(cc[m].astype(np.float32))
        n_used += 1

    if not all_xyz:
        logger.warning("densification produced no points")
        return 0
    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    fg = np.concatenate(all_fg); pr = np.concatenate(all_pr); pc = np.concatenate(all_pc)
    cf = np.concatenate(all_cf)

    np.savez_compressed(output_dir / f"{CHUNK}_origins.npz",
                        frame_global=fg, pixel_row=pr, pixel_col=pc, confidence=cf)
    # binary PLY (x,y,z + rgb) matching the chunk format CloudCompPy merges
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vd = np.empty(len(xyz), dt)
    vd["x"], vd["y"], vd["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vd["r"], vd["g"], vd["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(output_dir / f"{CHUNK}.ply", "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode("ascii"))
        vd.tofile(f)
    logger.info(f"✅ densified {n_used} fillers → {CHUNK}.ply ({len(xyz):,} pts) "
                f"(stride {stride}, conf×{conf_coef})")
    return len(xyz)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    run(Path(a.output_dir), stride=a.stride)


if __name__ == "__main__":
    main()
