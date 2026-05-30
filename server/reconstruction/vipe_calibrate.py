#!/usr/bin/env python3
"""
Per-frame ViPE→DA3 depth calibration (mirror of the hybrid DA3↔LiDAR affine
calibration in stray_da3_streaming.py). ViPE provides dense, pose-consistent
depth for ALL frames; DA3 provides the metric reference. A per-frame affine
``depth_cal = s·vipe + o`` (least-squares on the overlap, on a common grid) puts
ViPE depth into DA3 metric while keeping it consistent with ViPE poses.

Validated on test1: raw DA3/ViPE ratio varies 1.06–1.68 per frame; after the
per-frame affine the median residual vs DA3 is ~3–6 cm across all frames.

Runs in ViPE's .venv (needs OpenEXR to read ViPE depth). Emits, per frame, a
calibrated-depth npz the composer (da3 env) reads — no OpenEXR dependency
downstream. Layout mirrors da3_full/: one file per original frame number.

    vendor/vipe/.venv/bin/python server/reconstruction/vipe_calibrate.py \
        --vipe-out  <session>/output/vipe_run \
        --da3-depth <session>/output/da3_all_run/results_output \
        --out       <session>/output/vipe_run/depth_calibrated \
        [--min-overlap 200] [--depth-lo 0.3] [--depth-hi 6.0]
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import OpenEXR


def read_vipe_exr(zf: zipfile.ZipFile, orig_frame: int):
    """ViPE metric depth: EXR channel 'Z', float16 (see vipe/utils/io.py)."""
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


def read_da3_depth(da3_dir: Path, frame_idx: int):
    """DA3 streaming per-frame depth+conf+intrinsics (results_output/frame_<i>.npz)."""
    p = da3_dir / f"frame_{frame_idx}.npz"
    if not p.exists():
        return None, None
    z = np.load(str(p))
    d = z["depth"].astype(np.float32) if "depth" in z else None
    c = z["conf"].astype(np.float32) if "conf" in z else None
    return d, c


def affine_calibrate(vipe_d, da3_d, depth_lo, depth_hi, min_overlap):
    """Fit da3 ≈ s·vipe + o on the valid overlap (resampled to DA3 grid); return
    (calibrated full-res ViPE depth, s, o, residual_m, n_overlap)."""
    Hd, Wd = da3_d.shape
    ys = np.linspace(0, vipe_d.shape[0] - 1, Hd).astype(int)
    xs = np.linspace(0, vipe_d.shape[1] - 1, Wd).astype(int)
    vr = vipe_d[np.ix_(ys, xs)]
    m = ((da3_d > depth_lo) & (da3_d < depth_hi) &
         (vr > depth_lo) & (vr < depth_hi) &
         np.isfinite(da3_d) & np.isfinite(vr))
    n = int(m.sum())
    if n < min_overlap:
        return None
    s, o = np.polyfit(vr[m], da3_d[m], 1)
    res = float(np.median(np.abs(da3_d[m] - (vr[m] * s + o))))
    return (vipe_d * s + o).astype(np.float32), float(s), float(o), res, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vipe-out", required=True, type=Path)
    ap.add_argument("--da3-depth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-overlap", type=int, default=200)
    ap.add_argument("--depth-lo", type=float, default=0.3)
    ap.add_argument("--depth-hi", type=float, default=6.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ViPE poses + intrinsics give the frame list (inds = positional → original #
    # via the slam list; here we read the inds directly as original frame numbers
    # for the contiguous case and key everything by them).
    pose = np.load(args.vipe_out / "pose" / "frames.npz")
    intr = np.load(args.vipe_out / "intrinsics" / "frames.npz")
    inds = np.asarray(pose["inds"]).astype(int)
    Kdata = np.asarray(intr["data"], np.float64)  # (N,4) fx fy cx cy (ViPE full res)
    zdepth = zipfile.ZipFile(args.vipe_out / "depth" / "frames.zip")

    manifest = {}
    fitted = skipped = 0
    s_list = []
    for row, orig in enumerate(inds):
        vd = read_vipe_exr(zdepth, int(orig))
        dd, dc = read_da3_depth(args.da3_depth, int(orig))
        if vd is None or dd is None:
            skipped += 1
            continue
        res = affine_calibrate(vd, dd, args.depth_lo, args.depth_hi, args.min_overlap)
        if res is None:
            skipped += 1
            continue
        depth_cal, s, o, residual, n = res
        s_list.append(s)
        # carry DA3 confidence resampled to the ViPE grid (for TSDF weighting)
        conf_v = None
        if dc is not None:
            ys = np.linspace(0, dc.shape[0] - 1, vd.shape[0]).astype(int)
            xs = np.linspace(0, dc.shape[1] - 1, vd.shape[1]).astype(int)
            conf_v = dc[np.ix_(ys, xs)].astype(np.float32)
        fx, fy, cx, cy = Kdata[row]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
        out_npz = args.out / f"{int(orig):06d}.npz"
        save = dict(depth=depth_cal, intrinsics=K, scale=np.float32(s),
                    offset=np.float32(o), residual=np.float32(residual))
        if conf_v is not None:
            save["conf"] = conf_v
        np.savez_compressed(out_npz, **save)
        manifest[int(orig)] = {"scale": s, "offset": o, "residual_m": residual, "n": n}
        fitted += 1
        if fitted % 200 == 0:
            print(f"[calib] {fitted} frames calibrated…", flush=True)

    with open(args.out / "calibration_manifest.json", "w") as f:
        json.dump(manifest, f)
    s_arr = np.array(s_list) if s_list else np.array([1.0])
    print(f"[calib] done: fitted={fitted} skipped={skipped} "
          f"| affine scale median={np.median(s_arr):.3f} "
          f"min={s_arr.min():.3f} max={s_arr.max():.3f}")
    print(f"[calib] wrote {args.out}")


if __name__ == "__main__":
    main()
