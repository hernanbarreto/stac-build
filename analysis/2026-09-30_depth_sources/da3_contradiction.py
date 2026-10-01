"""Read-only: how much a CONTRADICTION rule drops vs a CONFIRMATION rule, on output/da3_posed_depth.
Per pixel of keyframe i, each neighbour j that sees it votes:
  agree      |z_ij - d_j| <= tol * d_j
  contradict z_ij < d_j * (1 - tol)   -> j sees PAST the point: it floats in space j saw empty
  no vote    z_ij > d_j * (1 + tol)   -> something sits in front of it in j (occlusion), no evidence
confirmation rule: keep if >= 2 agree.  contradiction rule: drop only if contradict > agree."""
import numpy as np
from pathlib import Path
D = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default/output/da3_posed_depth")
kf = {int(p.stem[3:]): np.load(p) for p in sorted(D.glob("kf_*.npz"))}
N = len(kf); NB = (-12, -6, -3, 3, 6, 12)
def proj(i, j):
    a, b = kf[i], kf[j]; d, v, K = a["depth"], a["valid"], a["K"]; H, W = d.shape
    rr, cc = np.nonzero(v); z = d[rr, cc].astype(np.float64)
    X = np.stack([(cc - K[0, 2]) / K[0, 0] * z, (rr - K[1, 2]) / K[1, 1] * z, z], 1) @ a["c2w"][:3, :3].T + a["c2w"][:3, 3]
    w2c = np.linalg.inv(b["c2w"]); Xj = X @ w2c[:3, :3].T + w2c[:3, 3]; zj = Xj[:, 2]; Kj = b["K"]
    ok = zj > 0.05; zs = np.where(ok, zj, 1)
    u = np.round(Kj[0, 0] * Xj[:, 0] / zs + Kj[0, 2]).astype(int); r = np.round(Kj[1, 1] * Xj[:, 1] / zs + Kj[1, 2]).astype(int)
    ok &= (u >= 0) & (u < W) & (r >= 0) & (r < H)
    dj = np.full(len(z), np.nan); dj[ok] = b["depth"][r[ok], u[ok]]
    ok &= b["valid"][np.clip(r, 0, H - 1), np.clip(u, 0, W - 1)]
    return zj, dj, ok
res = {t: {"conf2": 0, "contra": 0, "unjudged": 0} for t in (0.02, 0.03, 0.05)}; n = 0
for i in range(0, N, 6):
    P = [proj(i, j) for j in (i + k for k in NB) if 0 <= j < N]
    n += len(P[0][0])
    for t in res:
        ag = np.zeros(len(P[0][0]), int); co = np.zeros_like(ag)
        for zj, dj, ok in P:
            e = (zj - dj) / dj
            ag += ok & (np.abs(e) <= t); co += ok & (e < -t)
        res[t]["conf2"] += int((ag < 2).sum()); res[t]["contra"] += int((co > ag).sum())
        res[t]["unjudged"] += int(((ag == 0) & (co == 0)).sum())
print(f"pixels {n:,} (every 6th keyframe, neighbours ±3 ±6 ±12)")
for t, r in res.items():
    print(f"tol {t*100:.0f} %: confirmation (>=2 agree) drops {r['conf2']/n*100:.1f} %  |  "
          f"contradiction (contra > agree) drops {r['contra']/n*100:.1f} %  (no neighbour could judge: {r['unjudged']/n*100:.1f} %, kept)")
