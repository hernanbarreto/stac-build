#!/usr/bin/env python3
"""
Global metric scale for PURE ViPE + export of ViPE's (now scaled) depth.

ViPE runs PURE (no priors) → consistent geometry but only up-to-scale (~metric but
off by a factor). We anchor the metric with ONE global scale g = robust median of
(DA3_depth / ViPE_depth) over co-located pixels across a sample of frames — a single
number, so it cannot distort the geometry (no per-pixel/per-chunk poisoning that
wrecked the BA when we injected DA3 depth as a prior).

Then export every fusion frame's ViPE EXR depth × g (+ K) → vipe_depth/{frame}.npz,
and write pose_scale.json = g (vipe_poses scales the pose translations by it too).

Runs in ViPE's .venv (reads the ViPE EXR). DA3 depth = vipe_priors/depth/{frame}.npy.

    vendor/vipe/.venv/bin/python server/reconstruction/vipe_calibrate.py \
        --vipe-out <out>/vipe_run --da3-depth <out>/vipe_priors/depth \
        --depth-out <out>/vipe_depth --out <out>/vipe_run/pose_scale.json
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import OpenEXR


def read_vipe_exr(zf: zipfile.ZipFile, orig_frame: int):
    name = f"{orig_frame:05d}.exr"
    if name not in zf.namelist():
        return None
    with zf.open(name) as f:
        exr = OpenEXR.InputFile(f)
        dw = exr.header()["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        z = np.frombuffer(exr.channels(["Z"])[0], np.float16).reshape(H, W).astype(np.float32)
    return z


def _k_for_res(fxfycxcy, H_exr, W_exr):
    fx, fy, cx, cy = [float(v) for v in fxfycxcy]
    W_v, H_v = max(1.0, 2.0 * cx), max(1.0, 2.0 * cy)
    sx, sy = W_exr / W_v, H_exr / H_v
    if abs(sx - 1) > 0.02 or abs(sy - 1) > 0.02:
        fx *= sx; cx *= sx; fy *= sy; cy *= sy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe-out", required=True, type=Path)
    ap.add_argument("--da3-depth", required=True, type=Path)   # vipe_priors/depth (DA3 metric)
    ap.add_argument("--depth-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-scale-frames", type=int, default=80)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=6.0)
    args = ap.parse_args()

    intr = np.load(args.vipe_out / "intrinsics" / "frames.npz")
    intr_map = {int(fi): intr["data"][i] for i, fi in enumerate(np.asarray(intr["inds"]).astype(int))}
    zdepth = zipfile.ZipFile(args.vipe_out / "depth" / "frames.zip")
    exr_frames = sorted(int(n[:-4]) for n in zdepth.namelist() if n.endswith(".exr"))

    # ── 1. global scale g = robust median of DA3/ViPE depth ratio ──
    da3_frames = sorted(int(p.stem) for p in args.da3_depth.glob("[0-9]*.npy")) \
        if args.da3_depth.exists() else []
    ratios = []
    step = max(1, len(da3_frames) // args.max_scale_frames) if da3_frames else 1
    for fr in da3_frames[::step]:
        if fr not in set(exr_frames):
            continue
        dv = read_vipe_exr(zdepth, fr)
        if dv is None:
            continue
        try:
            dd = np.load(args.da3_depth / f"{fr:06d}.npy").astype(np.float32)
        except Exception:
            continue
        Hd, Wd = dd.shape
        ys = np.linspace(0, dv.shape[0] - 1, Hd).astype(int)
        xs = np.linspace(0, dv.shape[1] - 1, Wd).astype(int)
        dvr = dv[np.ix_(ys, xs)]
        m = ((dd > args.lo) & (dd < args.hi) & (dvr > args.lo) & (dvr < args.hi) &
             np.isfinite(dd) & np.isfinite(dvr))
        if m.sum() >= 500:
            ratios.append(float(np.median(dd[m] / dvr[m])))
    g = float(np.median(ratios)) if ratios else 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pose_scale": g, "n_frames": len(ratios)}, open(args.out, "w"))
    print(f"[vipe-scale] global g = {g:.4f} (median DA3/ViPE over {len(ratios)} frames)", flush=True)

    # ── 2. export ViPE depth × g + K for every fusion frame ──
    args.depth_out.mkdir(parents=True, exist_ok=True)
    n = 0; last = None
    for fr in exr_frames:
        if fr not in intr_map:
            continue
        dv = read_vipe_exr(zdepth, fr)
        if dv is None:
            continue
        depth = (dv * g).astype(np.float32)
        K = _k_for_res(intr_map[fr], dv.shape[0], dv.shape[1])
        np.savez_compressed(args.depth_out / f"{fr:06d}.npz",
                            depth=depth, intrinsics=K.astype(np.float64))
        n += 1; last = dv.shape
    print(f"[vipe-scale] exported {n} ViPE depth maps × g "
          f"({last[1]}x{last[0]} px) → {args.depth_out.name}/", flush=True)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        print("[vipe-scale] FATAL — traceback follows:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
