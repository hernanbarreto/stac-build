#!/usr/bin/env python3
"""
Stage-1 cloud: ViPE poses + ViPE depth CALIBRATED per-frame to DA3 metric
(mirror of the hybrid DA3↔LiDAR affine calibration). Runs in ViPE's .venv
(needs OpenEXR). Writes a binary PLY (numpy, no open3d dep) for visual review
and reports a wall-doubling metric across the kf79→80 chunk junction.

    vendor/vipe/.venv/bin/python tools/compose_vipe_calibrated.py \
        --session <session_src_default> [--subsample 10]
"""
import argparse, json, zipfile, struct
from pathlib import Path
import numpy as np
import OpenEXR
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path)
    ap.add_argument("--subsample", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    S = args.session
    vipe = S / "output" / "vipe_run"

    zdepth = zipfile.ZipFile(vipe / "depth" / "frames.zip")
    vp = np.load(vipe / "pose" / "frames.npz"); vmats = vp["data"]; vrow = {int(f): r for r, f in enumerate(vp["inds"])}
    Kd = np.load(vipe / "intrinsics" / "frames.npz")["data"]
    sf = json.load(open(S / "frames" / "selected_frames.json"))
    files = sorted(sf if isinstance(sf, list) else sf["selected_files"])
    kf_orig = [int(Path(x).stem) for x in files]

    def vipe_depth(o):
        with zdepth.open(f"{o:05d}.exr") as f:
            e = OpenEXR.InputFile(f); dw = e.header()["dataWindow"]
            W = dw.max.x - dw.min.x + 1; H = dw.max.y - dw.min.y + 1
            return np.frombuffer(e.channels(["Z"])[0], np.float16).reshape(H, W).astype(np.float32)

    def da3_depth(kf):
        p = S / f"output/da3_run/results_output/frame_{kf}.npz"
        return np.load(p)["depth"].astype(np.float32) if p.exists() else None

    def calibrated(kf):
        """ViPE depth → DA3 metric via per-frame affine (polyfit on overlap)."""
        dv = vipe_depth(kf_orig[kf]); dd = da3_depth(kf)
        if dd is None:
            return dv, 1.0, 0.0
        Hd, Wd = dd.shape
        ys = np.linspace(0, dv.shape[0] - 1, Hd).astype(int); xs = np.linspace(0, dv.shape[1] - 1, Wd).astype(int)
        dvr = dv[np.ix_(ys, xs)]
        m = (dd > 0.3) & (dd < 6) & (dvr > 0.3) & (dvr < 6) & np.isfinite(dd) & np.isfinite(dvr)
        if m.sum() < 200:
            return dv, 1.0, 0.0
        s, o = np.polyfit(dvr[m], dd[m], 1)
        return dv * s + o, float(s), float(o)

    def backproject(kf, sub):
        d, _, _ = calibrated(kf); o = kf_orig[kf]
        if o not in vrow:
            return None
        K = Kd[vrow[o]]; c2w = vmats[vrow[o]]
        H, W = d.shape
        m = (d > 0.3) & (d < 8) & np.isfinite(d)
        if sub > 1:
            s = np.zeros_like(m); s[::sub, ::sub] = True; m &= s
        vv, uu = np.nonzero(m); z = d[vv, uu].astype(np.float64)
        fx, fy, cx, cy = K
        Xc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z], 1)
        Xw = (c2w[:3, :3] @ Xc.T).T + c2w[:3, 3]
        # color from original frame jpg at depth res
        jpg = S / "frames" / f"{o:06d}.jpg"
        col = None
        if jpg.exists():
            im = cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (W, H))
            col = im[vv, uu].astype(np.uint8)
        return Xw, col

    # ── wall-doubling metric across the kf79→80 junction ──
    # frames either side of the junction that view the same wall; if poses are
    # coherent their points overlap (small inter-set distance), if doubled, large.
    def junction_thickness(side_a, side_b):
        Xa = [backproject(k, 4)[0] for k in side_a if backproject(k, 4)]
        Xb = [backproject(k, 4)[0] for k in side_b if backproject(k, 4)]
        Xa = np.concatenate(Xa); Xb = np.concatenate(Xb)
        # crude NN: for a sample of Xb, nearest in Xa via grid
        idx = np.random.RandomState(0).choice(len(Xb), min(4000, len(Xb)), replace=False)
        from scipy.spatial import cKDTree
        t = cKDTree(Xa); dist, _ = t.query(Xb[idx], k=1)
        return float(np.median(dist)) * 1000, float(np.percentile(dist, 90)) * 1000

    print("=== wall-doubling across kf79→80 (median NN between pre- and post-junction frames) ===")
    try:
        med, p90 = junction_thickness([74, 75, 76, 77, 78, 79], [80, 81, 82, 83, 84, 85])
        print(f"  ViPE+calib: median={med:.0f}mm  p90={p90:.0f}mm  "
              f"({'✅ pared única' if med < 150 else '⚠ revisar'})")
    except Exception as e:
        print(f"  (metric skipped: {e})")

    # ── full cloud ──
    out = args.out or (S / "output" / "tsdf" / "vipe_cloud_calibrated.ply")
    out.parent.mkdir(parents=True, exist_ok=True)
    allX, allC = [], []
    for i in range(len(kf_orig)):
        r = backproject(i, args.subsample)
        if r is None:
            continue
        allX.append(r[0])
        if r[1] is not None:
            allC.append(r[1])
    X = np.concatenate(allX).astype(np.float32)
    C = np.concatenate(allC).astype(np.uint8) if allC else np.full((len(X), 3), 180, np.uint8)
    with open(out, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(X)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        buf = np.empty(len(X), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
        buf["x"], buf["y"], buf["z"] = X[:, 0], X[:, 1], X[:, 2]
        buf["r"], buf["g"], buf["b"] = C[:, 0], C[:, 1], C[:, 2]
        f.write(buf.tobytes())
    print(f"\n[compose] wrote {out}  ({len(X):,} pts, subsample={args.subsample})")
    print(f"[compose] bbox size = {np.round(X.max(0)-X.min(0),2)}")


if __name__ == "__main__":
    main()
