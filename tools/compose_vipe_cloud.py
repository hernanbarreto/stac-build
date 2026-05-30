#!/usr/bin/env python3
"""
Stage-1 validation for the ViPE pose-source migration (read-only on pipeline
inputs). Recomposes the cloud by back-projecting the depth-mode's depth with
ViPE poses, and quantifies coherence at the chunk-handoff junction (kf79→80 on
test1) vs the current DA3 poses.

It does NOT touch the pipeline or overwrite chunk_*.ply — it writes a separate
``vipe_cloud.ply`` for visual inspection and prints a junction metric.

The thesis: with DA3 poses, frames either side of a chunk boundary are placed
metres apart → the same wall appears doubled. With ViPE's continuous poses they
overlap → a single coherent surface. We measure this as the median
nearest-neighbour distance between consecutive keyframes' back-projected points.

Usage:
    python tools/compose_vipe_cloud.py \
        --vipe-out  <session>/output/vipe_run \
        --output-dir <session>/output \
        --selected  <session>/frames/selected_frames.json \
        --mode da3 [--junction 79 80] [--max-kf-full 706] [--subsample 12]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ── ViPE pose loading + keyframe mapping ────────────────────────────

def load_vipe_poses(vipe_out: Path):
    d = np.load(vipe_out / "pose" / "frames.npz")
    inds = np.asarray(d["inds"]).astype(np.int64)
    data = np.asarray(d["data"], dtype=np.float64)        # (N,4,4) c2w
    return inds, data


def keyframe_original_numbers(selected_json: Path):
    sf = json.load(open(selected_json))
    files = sf if isinstance(sf, list) else sf.get("selected_files", [])
    return [int(Path(f).stem) for f in sorted(files)]     # kf_seq_idx -> original frame #


def build_pose_map(vipe_out: Path, selected_json: Path):
    """kf_seq_idx -> ViPE c2w. ViPE inds are POSITIONAL in the sorted file list
    ViPE was given; on test1 the frames dir is contiguous so ind == original #.
    A slam_index.json (pipeline) would generalise this; for the standalone test
    we assume the contiguous case and assert coverage."""
    inds, mats = load_vipe_poses(vipe_out)
    vidx = {int(fi): row for row, fi in enumerate(inds)}  # original # -> vipe row
    kf_orig = keyframe_original_numbers(selected_json)
    pose_map, missing = {}, []
    for i, on in enumerate(kf_orig):
        if on in vidx:
            pose_map[i] = mats[vidx[on]]
        else:
            missing.append(i)
    return pose_map, kf_orig, missing


def load_da3_poses(camera_poses_txt: Path):
    mats = []
    for line in open(camera_poses_txt):
        v = line.split()
        if len(v) >= 16:
            mats.append(np.array([float(x) for x in v[:16]], np.float64).reshape(4, 4))
    return {i: m for i, m in enumerate(mats)}


# ── global scale (ViPE → metric depth units) ───────────────────────

def umeyama_scale(src, dst):
    """Scale s mapping src→dst point sets (Nx3). Robust over many keyframes;
    a global scalar is unaffected by a couple of DA3 chunk jumps."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(Xd.T @ Xs / len(src))
    return float(np.trace(np.diag(D)) / ((Xs ** 2).sum() / len(src)))


# ── back-projection (mirrors tsdf_export integrate exactly) ─────────

def load_da3_frame(npz_dir: Path, kf: int, conf_percentile=50.0):
    p = npz_dir / f"frame_{kf}.npz"
    if not p.exists():
        return None
    z = np.load(str(p))
    depth = z["depth"].astype(np.float32)
    H, W = depth.shape
    K = np.asarray(z["intrinsics"], np.float64).reshape(3, 3)
    rgb = z["image"].astype(np.uint8) if "image" in z else None
    valid = np.isfinite(depth) & (depth > 0.15) & (depth < 5.0)
    if "conf" in z and conf_percentile is not None:
        conf = z["conf"].astype(np.float32)
        if conf.shape == depth.shape:
            valid &= conf >= np.percentile(conf, conf_percentile)
    return depth, valid, K, rgb


def backproject(depth, valid, K, c2w, rgb=None, edge_thresh=0.04, subsample=1):
    H, W = depth.shape
    m = valid.copy()
    # flying-pixel cull at depth discontinuities (same as TSDF)
    d = np.where(m, depth, 0.0)
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1, :]))
    m &= (gx < edge_thresh) & (gy < edge_thresh)
    if subsample > 1:
        sub = np.zeros_like(m); sub[::subsample, ::subsample] = True; m &= sub
    vv, uu = np.nonzero(m)
    z = depth[vv, uu].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Xc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], axis=1)   # (N,3)
    Xw = (c2w[:3, :3] @ Xc.T).T + c2w[:3, 3]
    col = (rgb[vv, uu].astype(np.float64) / 255.0) if rgb is not None else None
    return Xw, col


def junction_nn_median(npz_dir, pose_map, a, b, scale):
    """Median NN distance between kf a and kf b back-projected point sets —
    small ⇒ coherent overlap, large ⇒ doubled wall."""
    fa, fb = load_da3_frame(npz_dir, a), load_da3_frame(npz_dir, b)
    if fa is None or fb is None or a not in pose_map or b not in pose_map:
        return None
    ca = pose_map[a].copy(); ca[:3, 3] *= scale
    cb = pose_map[b].copy(); cb[:3, 3] *= scale
    Xa, _ = backproject(*fa[:3], ca, subsample=2)
    Xb, _ = backproject(*fb[:3], cb, subsample=2)
    if len(Xa) < 100 or len(Xb) < 100:
        return None
    pa = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xa))
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Xb))
    dists = np.asarray(pb.compute_point_cloud_distance(pa))   # b→nearest in a
    return float(np.median(dists)), float(np.percentile(dists, 90)), len(Xa), len(Xb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe-out", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--selected", required=True, type=Path)
    ap.add_argument("--mode", default="da3")
    ap.add_argument("--junction", nargs=2, type=int, default=[79, 80])
    ap.add_argument("--subsample", type=int, default=12, help="pixel stride for the full cloud")
    ap.add_argument("--out-ply", type=Path, default=None)
    args = ap.parse_args()

    npz_dir = args.output_dir / "da3_run" / "results_output"
    if not npz_dir.exists():
        npz_dir = args.output_dir / "results_output"
    print(f"[compose] depth npz dir: {npz_dir}")

    pose_map, kf_orig, missing = build_pose_map(args.vipe_out, args.selected)
    print(f"[compose] keyframes={len(kf_orig)}  vipe-mapped={len(pose_map)}  missing={len(missing)}")

    # global scale: ViPE → DA3 metric (robust over all matched keyframes)
    da3_poses = load_da3_poses(args.output_dir / "da3_run" / "camera_poses.txt")
    common = sorted(set(pose_map) & set(da3_poses))
    vpos = np.array([pose_map[i][:3, 3] for i in common])
    dpos = np.array([da3_poses[i][:3, 3] for i in common])
    scale = umeyama_scale(vpos, dpos)
    print(f"[compose] global scale ViPE→DA3 = {scale:.4f}")

    # ── junction coherence: ViPE poses vs DA3 poses ──
    a, b = args.junction
    print(f"\n=== Junction coherence kf{a}→kf{b} "
          f"(median NN dist between the two frames' points; small = coherent) ===")
    rv = junction_nn_median(npz_dir, pose_map, a, b, scale)
    rd = junction_nn_median(npz_dir, da3_poses, a, b, 1.0)
    if rv:
        print(f"  ViPE poses : median={rv[0]*1000:6.0f}mm  p90={rv[1]*1000:6.0f}mm  "
              f"({rv[2]}+{rv[3]} pts)")
    if rd:
        print(f"  DA3  poses : median={rd[0]*1000:6.0f}mm  p90={rd[1]*1000:6.0f}mm  "
              f"({rd[2]}+{rd[3]} pts)")
    if rv and rd:
        print(f"  → ViPE/DA3 median ratio = {rv[0]/max(rd[0],1e-9):.2f}  "
              f"({'✅ ViPE más coherente' if rv[0] < rd[0] else '❌ no mejora'})")

    # ── full subsampled cloud (ViPE poses) for visual inspection ──
    out_ply = args.out_ply or (args.output_dir / "tsdf" / "vipe_cloud.ply")
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    allX, allC = [], []
    for i in sorted(pose_map):
        fr = load_da3_frame(npz_dir, i)
        if fr is None:
            continue
        c2w = pose_map[i].copy(); c2w[:3, 3] *= scale
        X, C = backproject(fr[0], fr[1], fr[2], c2w, rgb=fr[3], subsample=args.subsample)
        allX.append(X)
        if C is not None:
            allC.append(C)
    X = np.concatenate(allX);
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    if allC:
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(allC))
    o3d.io.write_point_cloud(str(out_ply), pc)
    print(f"\n[compose] wrote {out_ply}  ({len(X):,} pts, subsample={args.subsample})")
    print(f"[compose] bbox size = {np.round(X.max(0)-X.min(0),2)}")


if __name__ == "__main__":
    main()
