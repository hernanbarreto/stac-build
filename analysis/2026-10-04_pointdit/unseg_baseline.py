"""pccr live epoch: share of cloud points whose SOURCE PIXEL carries no SAM3 mask (before any growth)."""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
O = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output")
from correction.session import read_ply
from precision.silhouette_filter import masks_by_keyframe
t0 = time.time()
_, data = read_ply(O / "cleaned_cloud.ply")
fg = np.asarray(data["frame_global"]).astype(np.int64); pr = np.asarray(data["pixel_row"]).astype(np.int64); pc = np.asarray(data["pixel_col"]).astype(np.int64)
kfs = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
pos_of = {f: i for i, f in enumerate(kfs)}
masks = np.load(O / "seg_masks.npz"); by_kf = masks_by_keyframe(O, masks)
H, W = masks[masks.files[0]].shape
n = len(fg); covered = np.zeros(n, bool); has_pos = np.zeros(n, bool)
per_kf = []
for f in np.unique(fg):
    i = pos_of.get(int(f))
    sel = np.nonzero(fg == f)[0]
    if i is None:
        continue
    has_pos[sel] = True
    U = np.zeros((H, W), bool)
    for oid, key in by_kf.get(i, []):
        U |= np.asarray(masks[key]) > 0
    ok = (pr[sel] >= 0) & (pr[sel] < H) & (pc[sel] >= 0) & (pc[sel] < W)
    c = np.zeros(len(sel), bool); c[ok] = U[pr[sel][ok], pc[sel][ok]]
    covered[sel] = c
    per_kf.append((i, int(f), len(sel), int((~c).sum())))
un = has_pos & ~covered
print(f"points {n:,}; with a keyframe {has_pos.sum():,}; source pixel UNMASKED {un.sum():,} = {un.sum() / n * 100:.1f} %  ({time.time() - t0:.0f} s)")
per_kf.sort(key=lambda t: -t[3])
print("keyframes with most unmasked points (pos, frame, pts, unmasked):", per_kf[:10])
u = np.array([t[3] / max(t[2], 1) for t in per_kf]); print(f"per-keyframe unmasked share: median {np.median(u) * 100:.1f} %, p90 {np.percentile(u, 90) * 100:.1f} %, max {u.max() * 100:.1f} %")
