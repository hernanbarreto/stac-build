#!/usr/bin/env python3
"""
Export ViPE's REFINED depth (EXR → per-frame npz) for the fusion keyframes.

In the prior-driven architecture ViPE runs WITH metric-depth + pose priors, so its
output depth is already metric and multi-view consistent — there is NO global scale
to estimate anymore (g = 1). This step only transcodes the ViPE EXR depth + ViPE
intrinsics into plain npz that the composer (da3 env) and TSDF (server env) read
without OpenEXR. Exported only for the fusion keyframes to bound disk.

Runs in ViPE's .venv (reads the ViPE EXR). Writes vipe_depth/{frame:06d}.npz
{depth (metric), intrinsics} + pose_scale.json (always 1.0, for the pose writer).

    vendor/vipe/.venv/bin/python server/reconstruction/vipe_calibrate.py \
        --vipe-out <out>/vipe_run --depth-out <out>/vipe_depth \
        --out <out>/vipe_run/pose_scale.json [--selected-frames <sf>.json]
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
    ap.add_argument("--depth-out", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--selected-frames", default=None,
                    help="selected_frames.json — restrict export to fusion keyframes")
    args = ap.parse_args()

    intr = np.load(args.vipe_out / "intrinsics" / "frames.npz")
    intr_map = {int(fi): intr["data"][i] for i, fi in enumerate(np.asarray(intr["inds"]).astype(int))}
    zdepth = zipfile.ZipFile(args.vipe_out / "depth" / "frames.zip")
    exr_frames = sorted(int(n[:-4]) for n in zdepth.namelist() if n.endswith(".exr"))

    keep = None
    if args.selected_frames and Path(args.selected_frames).exists():
        sf = json.load(open(args.selected_frames))
        names = sf if isinstance(sf, list) else sf.get("selected_files", sf.get("selected", []))
        keep = {int(Path(n).stem) for n in names}

    args.depth_out.mkdir(parents=True, exist_ok=True)
    n = 0; last = None
    for fr in exr_frames:
        if keep is not None and fr not in keep:
            continue
        if fr not in intr_map:
            continue
        dv = read_vipe_exr(zdepth, fr)
        if dv is None:
            continue
        K = _k_for_res(intr_map[fr], dv.shape[0], dv.shape[1])
        np.savez_compressed(args.depth_out / f"{fr:06d}.npz",
                            depth=dv.astype(np.float32), intrinsics=K.astype(np.float64))
        n += 1; last = dv.shape

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"pose_scale": 1.0}, open(args.out, "w"))   # ViPE output is metric (priors)
    print(f"[vipe-depth] exported {n} ViPE depth maps "
          f"({last[1]}x{last[0]} px) → {args.depth_out.name}/ (metric, g=1)", flush=True)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        print("[vipe-depth] FATAL — traceback follows:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
