#!/usr/bin/env python3
"""
Compute the SINGLE global scale g that maps ViPE pose translations into the DA3
depth metric, so DA3 depth (used DIRECTLY, no per-frame calibration) is coherent
with ViPE poses.

ViPE pose+depth are self-consistent at ViPE's scale; DA3 depth is metric, with
DA3_depth ≈ g · ViPE_depth. So g = robust median of (DA3_depth / ViPE_depth) over
co-located pixels across a sample of frames. Apply g to ViPE pose translations.

(We deliberately do NOT do per-frame affine calibration anymore — it injected
~6cm median depth error + ~10% garbage-scale frames → noisy TSDF. DA3 depth is
clean and used as-is; only the global pose scale is needed.)

Runs in ViPE's .venv (reads ViPE EXR depth). Writes vipe_run/pose_scale.json.

    vendor/vipe/.venv/bin/python server/reconstruction/vipe_calibrate.py \
        --vipe-out <session>/output/vipe_run \
        --da3-depth <session>/output/da3_depth \
        --out <session>/output/vipe_run/pose_scale.json [--max-frames 60]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe-out", required=True, type=Path)
    ap.add_argument("--da3-depth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--depth-lo", type=float, default=0.3)
    ap.add_argument("--depth-hi", type=float, default=6.0)
    args = ap.parse_args()

    inds = np.asarray(np.load(args.vipe_out / "pose" / "frames.npz")["inds"]).astype(int)
    zdepth = zipfile.ZipFile(args.vipe_out / "depth" / "frames.zip")
    da3_files = sorted(int(p.stem) for p in args.da3_depth.glob("[0-9]*.npz"))
    if not da3_files:
        raise RuntimeError(f"[scale] no DA3 depth in {args.da3_depth}")

    # sample evenly across the DA3-covered frames
    step = max(1, len(da3_files) // args.max_frames)
    sample = da3_files[::step]
    inds_set = set(int(x) for x in inds)

    ratios = []
    for fr in sample:
        if fr not in inds_set:
            continue
        dv = read_vipe_exr(zdepth, fr)
        if dv is None:
            continue
        try:
            dd = np.load(args.da3_depth / f"{fr:06d}.npz")["depth"].astype(np.float32)
        except Exception:
            continue
        Hd, Wd = dd.shape
        ys = np.linspace(0, dv.shape[0] - 1, Hd).astype(int)
        xs = np.linspace(0, dv.shape[1] - 1, Wd).astype(int)
        dvr = dv[np.ix_(ys, xs)]
        m = ((dd > args.depth_lo) & (dd < args.depth_hi) &
             (dvr > args.depth_lo) & (dvr < args.depth_hi) &
             np.isfinite(dd) & np.isfinite(dvr))
        if m.sum() < 500:
            continue
        # per-frame median ratio; aggregate medians across frames (robust)
        ratios.append(float(np.median(dd[m] / dvr[m])))

    if not ratios:
        g = 1.0
        print("[scale] WARNING: no overlap to estimate scale — using g=1.0", flush=True)
    else:
        g = float(np.median(ratios))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pose_scale": g, "n_frames": len(ratios),
                   "ratio_min": float(np.min(ratios)) if ratios else None,
                   "ratio_max": float(np.max(ratios)) if ratios else None}, f)
    print(f"[scale] global ViPE→DA3-metric pose scale g={g:.4f} "
          f"(median over {len(ratios)} frames)", flush=True)


if __name__ == "__main__":
    main()
