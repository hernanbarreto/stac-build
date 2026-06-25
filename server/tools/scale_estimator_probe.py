#!/usr/bin/env python3
"""READ-ONLY probe: compare the current scale_align estimator against agnostic
variants (near-depth band / DA3-confidence weighting) on a finished output, WITHOUT
touching production. Prints each variant's s and the trocha it would imply, given the
trocha already measured for the applied s.

Usage:
    python tools/scale_estimator_probe.py <output_dir> [--applied-s 9.9819] [--measured-mm 1328] [--target-mm 1435]

Needs BOTH da3_run/results_output/frame_*.npz (depth+conf) and
omega_run/results_output/frame_*.npz (depth) alive in <output_dir>.
"""
import sys, glob, argparse
from pathlib import Path
import numpy as np


def _resize_to(a, shape):
    if a.shape == shape:
        return a
    H, W = shape
    yi = (np.arange(H) * a.shape[0] / H).astype(int).clip(0, a.shape[0] - 1)
    xi = (np.arange(W) * a.shape[1] / W).astype(int).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


def _trimmed_median(ratios):
    """Match production: trim to 10–90 pct, then median."""
    r = np.asarray([x for x in ratios if x is not None and np.isfinite(x) and x > 0])
    if len(r) < 3:
        return None, len(r)
    lo, hi = np.percentile(r, [10, 90])
    return float(np.median(r[(r >= lo) & (r <= hi)])), len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--applied-s", type=float, default=None,
                    help="the s that was actually applied (anchor for trocha prediction)")
    ap.add_argument("--measured-mm", type=float, default=None,
                    help="trocha measured in the cloud built with --applied-s")
    ap.add_argument("--target-mm", type=float, default=1435.0)
    a = ap.parse_args()

    out = Path(a.output_dir)
    da3_dir = out / "da3_run" / "results_output"
    om_dir = out / "omega_run" / "results_output"
    if not da3_dir.is_dir() or not om_dir.is_dir():
        sys.exit(f"FAIL: need both {da3_dir} and {om_dir} (da3_run may have been freed)")

    om_files = {int(Path(p).stem.split("_")[1]): p
                for p in glob.glob(str(om_dir / "frame_*.npz"))}
    n_pairs = 0
    # per-frame ratio lists for each variant
    R = {k: [] for k in ("all", "near40", "near25", "conf_hi", "near40_conf")}

    for n, dp in {int(Path(p).stem.split("_")[1]): p
                  for p in glob.glob(str(da3_dir / "frame_*.npz"))}.items():
        if n not in om_files:
            continue
        try:
            dz = np.load(dp)
            da3 = dz["depth"].astype(np.float32)
            conf = dz["conf"].astype(np.float32) if "conf" in dz.files else None
            om = np.load(om_files[n])["depth"].astype(np.float32)
        except Exception:
            continue
        da3 = _resize_to(da3, om.shape)
        if conf is not None:
            conf = _resize_to(conf, om.shape)
        valid = np.isfinite(da3) & np.isfinite(om) & (da3 > 1e-3) & (om > 1e-3)
        if valid.sum() < 100:
            continue
        n_pairs += 1
        rr = da3[valid] / om[valid]
        od = om[valid]                       # omega depth = our distance proxy (near = small)
        cf = conf[valid] if conf is not None else None

        # variant: all pixels (== production)
        R["all"].append(float(np.median(rr)))
        # variant: near band — closest 40% / 25% by omega depth (rails are near)
        for key, pct in (("near40", 40), ("near25", 25)):
            thr = np.percentile(od, pct)
            sel = od <= thr
            if sel.sum() >= 50:
                R[key].append(float(np.median(rr[sel])))
        # variant: high-confidence pixels (top 40% conf)
        if cf is not None:
            cthr = np.percentile(cf, 60)
            sel = cf >= cthr
            if sel.sum() >= 50:
                R["conf_hi"].append(float(np.median(rr[sel])))
            # combo: near 40% AND high conf
            sel2 = (od <= np.percentile(od, 40)) & (cf >= cthr)
            if sel2.sum() >= 50:
                R["near40_conf"].append(float(np.median(rr[sel2])))

    print(f"matched frame pairs: {n_pairs}\n")
    base_s = a.applied_s
    print(f"{'variant':<14} {'s':>10} {'vs_applied':>11} {'pred_trocha_mm':>15}")
    print("-" * 54)
    for key in ("all", "near40", "near25", "conf_hi", "near40_conf"):
        s, ncnt = _trimmed_median(R[key])
        if s is None:
            print(f"{key:<14} {'(insufficient)':>10}")
            continue
        ratio = (s / base_s) if base_s else float("nan")
        pred = (a.measured_mm * ratio) if (a.measured_mm and base_s) else float("nan")
        flag = ""
        if a.target_mm and not np.isnan(pred):
            flag = f"  ({100*(pred-a.target_mm)/a.target_mm:+.1f}% vs {a.target_mm:.0f})"
        print(f"{key:<14} {s:>10.4f} {ratio:>10.4f}x {pred:>15.0f}{flag}  (n={ncnt})")


if __name__ == "__main__":
    main()
