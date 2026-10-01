"""How well does each DA3 keyframe agree with its neighbours? (read-only, on output/da3_posed_depth)"""
import numpy as np
from pathlib import Path
D = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output/da3_posed_depth")
kf = {int(p.stem[3:]): np.load(p) for p in sorted(D.glob("kf_*.npz"))}
N = len(kf)
def rel_err(i, j):
    a, b = kf[i], kf[j]
    d, v, K, c2w_i = a["depth"], a["valid"], a["K"], a["c2w"]
    H, W = d.shape
    rr, cc = np.nonzero(v)
    z = d[rr, cc].astype(np.float64)
    X = np.stack([(cc - K[0, 2]) / K[0, 0] * z, (rr - K[1, 2]) / K[1, 1] * z, z], 1) @ c2w_i[:3, :3].T + c2w_i[:3, 3]
    w2c = np.linalg.inv(b["c2w"]); Xj = X @ w2c[:3, :3].T + w2c[:3, 3]
    Kj = b["K"]; zj = Xj[:, 2]
    ok = zj > 0.05
    u = np.round(Kj[0, 0] * Xj[:, 0] / np.where(ok, zj, 1) + Kj[0, 2]).astype(int)
    r = np.round(Kj[1, 1] * Xj[:, 1] / np.where(ok, zj, 1) + Kj[1, 2]).astype(int)
    ok &= (u >= 0) & (u < W) & (r >= 0) & (r < H)
    e = np.full(len(z), np.nan)
    dj = b["depth"][r[ok], u[ok]]
    e[ok] = (zj[ok] - dj) / dj            # + : point lies BEHIND j's surface (j sees in front of it)
    return e, a["conf"][rr, cc].astype(np.float32), z, rr
allE, allC, allZ, allR, agree = [], [], [], [], []
for i in range(0, N, 4):
    es = []
    for j in (i - 6, i - 3, i + 3, i + 6):
        if 0 <= j < N:
            e, c, z, rr = rel_err(i, j); es.append(e)
    E = np.vstack(es)
    seen = np.isfinite(E).sum(0)
    ok2 = (np.abs(np.nan_to_num(E, nan=9)) < 0.02).sum(0)
    agree.append(((seen >= 2) & (ok2 >= 2)).mean())
    med = np.nanmedian(np.abs(E), 0)
    allE.append(med); allC.append(c); allZ.append(z); allR.append(rr / kf[i]["depth"].shape[0])
E = np.concatenate(allE); C = np.concatenate(allC); Z = np.concatenate(allZ); R = np.concatenate(allR)
f = np.isfinite(E); E, C, Z, R = E[f], C[f], Z[f], R[f]
print(f"pixels judged {len(E):,} (every 4th keyframe vs neighbours ±3, ±6)")
print(f"|rel err| median {np.median(E)*100:.2f} %  p75 {np.percentile(E,75)*100:.2f} %  p90 {np.percentile(E,90)*100:.2f} %")
for t in (0.01, 0.02, 0.05, 0.10):
    print(f"  > {t*100:.0f} %: {(E > t).mean()*100:.1f} % of pixels")
print(f"kept by 'agrees within 2 % with >= 2 neighbours': {np.mean(agree)*100:.1f} % of pixels")
q = np.percentile(C, [20, 40, 60, 80])
print("by DA3 confidence quintile: median |err| / share > 5 %")
for lo, hi, n in zip([-np.inf, *q], [*q, np.inf], range(5)):
    m = (C >= lo) & (C < hi)
    print(f"  q{n+1}: {np.median(E[m])*100:.2f} % / {(E[m] > 0.05).mean()*100:.1f} %")
print("by depth: median |err|")
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
    m = (Z >= lo) & (Z < hi)
    if m.any(): print(f"  {lo}-{hi} m: {np.median(E[m])*100:.2f} %  ({m.mean()*100:.0f} % of pixels)")
m = R > 0.75
print(f"bottom quarter of the image (floor mostly): median |err| {np.median(E[m])*100:.2f} %, > 5 %: {(E[m]>0.05).mean()*100:.1f} %")
