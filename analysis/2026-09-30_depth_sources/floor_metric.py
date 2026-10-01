"""Floor quality of a cloud (read-only). Up = minus the mean camera y-axis (cameras held upright).
Global floor = RANSAC plane (normal within 15 deg of up) among the lowest 20 % of points; band =
points within 0.30 m of it. Per 1 m cell (>= 300 band pts): local plane fit ->
  thickness  = p90 - p10 of residuals to the cell's own plane   (layers inside the cell)
  height     = the cell plane's offset at the cell centre vs the global plane
undulation   = p90 - p10 of the cell heights                    (waves between cells)
usage: floor_metric.py <label> <cloud.ply> <camera_poses.txt> ..."""
import sys
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
from reconstruction.gpu_cloud_clean import _read_ply

rng = np.random.default_rng(0)
args = sys.argv[1:]
for k in range(0, len(args), 3):
    label, ply, poses = args[k:k + 3]
    rng = np.random.default_rng(0)                            # same draws for every cloud
    v = _read_ply(ply)
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if len(X) > 6_000_000:
        X = X[rng.choice(len(X), 6_000_000, replace=False)]
    P = np.loadtxt(poses).reshape(-1, 4, 4)
    up = -P[:, :3, 1].mean(0); up /= np.linalg.norm(up)
    C = P[:, :3, 3]
    best, bn, bd = 0, None, None
    for _ in range(600):
        smp = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(smp[1] - smp[0], smp[2] - smp[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn
        if n @ up < 0: n = -n
        if n @ up < np.cos(np.radians(45)): continue          # roughly horizontal
        d = -n @ smp[0]
        if np.mean(C @ n + d > 0.3) < 0.9: continue              # the cameras stand ABOVE the floor
        c = int((np.abs(X[::10] @ n + d) < 0.03).sum())
        if c > best: best, bn, bd = c, n, d
    inl = X[np.abs(X @ bn + bd) < 0.03]                        # refine on its inliers
    ctr = inl.mean(0); _, _, vt = np.linalg.svd(inl - ctr, full_matrices=False)
    bn = vt[2] if vt[2] @ up > 0 else -vt[2]; bd = -bn @ ctr
    cam_h = C @ bn + bd
    print(f"{label:>9}: floor normal vs camera-up {np.degrees(np.arccos(np.clip(bn @ up, -1, 1))):.1f} deg, "
          f"cameras above floor median {np.median(cam_h):.2f} m (p10 {np.percentile(cam_h,10):.2f}, p90 {np.percentile(cam_h,90):.2f})", flush=True)
    dist = X @ bn + bd
    B = X[np.abs(dist) < 0.30]; bdist = B @ bn + bd
    e1 = np.cross(bn, [1, 0, 0] if abs(bn[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(bn, e1)
    uv = np.stack([B @ e1, B @ e2], 1); cell = np.floor(uv).astype(np.int64)
    key = cell[:, 0] * 100000 + cell[:, 1]
    order = np.argsort(key); key, uv, bdist = key[order], uv[order], bdist[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1, len(key)]
    th, ht, cuv = [], [], []
    bins = np.arange(-0.30, 0.3001, 0.01)
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 300: continue
        hist, _ = np.histogram(bdist[a:b], bins)
        hist = np.convolve(hist, np.ones(3), "same")
        mode = bins[np.argmax(hist)] + 0.005                    # the cell's dominant floor layer
        m = np.abs(bdist[a:b] - mode) < 0.10
        if m.sum() < 200: continue
        A = np.c_[uv[a:b][m], np.ones(m.sum())]
        coef = np.linalg.lstsq(A, bdist[a:b][m], rcond=None)[0]
        res = bdist[a:b][m] - A @ coef
        th.append(np.percentile(res, 90) - np.percentile(res, 10))
        ht.append(mode); cuv.append(np.floor(uv[a]) + 0.5)
    th, ht = np.array(th), np.array(ht)
    cu = np.array(cuv)
    A = np.c_[cu, np.ones(len(cu))]; ht = ht - A @ np.linalg.lstsq(A, ht, rcond=None)[0]   # tilt out: undulation only
    print(f"{label:>9}: {len(th)} cells | thickness median {np.median(th)*100:.1f} cm p90 {np.percentile(th,90)*100:.1f} cm "
          f"| undulation (p90-p10 of cell heights) {(np.percentile(ht,90)-np.percentile(ht,10))*100:.1f} cm "
          f"| band {len(B)/len(X)*100:.1f} % of pts", flush=True)
