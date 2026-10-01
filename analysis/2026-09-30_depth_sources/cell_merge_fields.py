"""Epoch 4 (DA3 on R0) + epoch 5 (Omega on R0), SAME frame, merged cell by cell -> epoch 7.
USER 2026-09-30: DA3 is much better in many places the landmark judge never saw. The judge here is
what the user sees: how THICK the surface is in a cell (a duplicated / wavy surface is thick):
  per cell, per cloud: local plane by PCA, thickness = p90 - p10 of the distances to it.
The thinner one wins the cell; a cell only one cloud covers goes to that cloud.
Reported for cells of 0.25 / 0.5 / 1.0 m; published at PUBLISH_CELL. Also reported: where both clouds
cover a cell, how far apart their two planes are (the step a cell border can show).
"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
EPOCH = 7
DST, TMP = O / f"_epoch_{EPOCH}", O / "_tx_cell_merge"
CELLS = (0.25, 0.5, 1.0)
PUBLISH_CELL = 0.5
MIN_PTS = 50


def log(m):
    print(f"[cells {time.strftime('%H:%M:%S')}] {m}", flush=True)


def where(e):
    live = json.loads((O / "geometry_epoch.json").read_text())["epoch"]
    return (O if live == e else O / f"_epoch_{e}")


from reconstruction.gpu_cloud_clean import _read_ply
t0 = time.time()
cl = {}
full = {}
for name, e in (("DA3", 4), ("Omega", 5)):
    v = _read_ply(str(where(e) / "cleaned_cloud.ply"))
    full[name] = v
    cl[name] = (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32),
                np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8))
    log(f"{name} (epoch {e}): {len(cl[name][0]):,} pts")
lo = np.minimum(cl["DA3"][0].min(0), cl["Omega"][0].min(0))
rng = np.random.default_rng(0)


def cell_stats(xyz, size):
    k = np.floor((xyz - lo) / size).astype(np.int64)
    key = (k[:, 0] * 100000 + k[:, 1]) * 100000 + k[:, 2]
    order = np.argsort(key, kind="stable"); ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1, len(ks)]
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < MIN_PTS: continue
        idx = order[a:b]
        if len(idx) > 3000: idx = rng.choice(idx, 3000, replace=False)
        P = xyz[idx].astype(np.float64); c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2]
        dd = (P - c) @ n
        out[int(ks[a])] = (float(np.percentile(dd, 90) - np.percentile(dd, 10)), c, n)
    return key, out



cell_stats(cl["DA3"][0], 0.25); cell_stats(cl["Omega"][0], 0.25)          # same RNG sequence as the published run
kd, sd = cell_stats(cl["DA3"][0], PUBLISH_CELL); ko, so = cell_stats(cl["Omega"][0], PUBLISH_CELL)
win = {}
for k in set(sd) | set(so):
    win[k] = ("DA3" if sd[k][0] < so[k][0] else "Omega") if (k in sd and k in so) else ("DA3" if k in sd else "Omega")
keep_d = np.array([win.get(int(k)) == "DA3" for k in kd]); keep_o = np.array([win.get(int(k)) == "Omega" for k in ko])
claimed = set(win)
keep_d |= np.array([int(k) not in claimed for k in kd]) & ~np.isin(kd, ko)
keep_o |= np.array([int(k) not in claimed for k in ko]) & ~np.isin(ko, kd)
data = np.concatenate([full["DA3"][keep_d], full["Omega"][keep_o]])
live = O / "cleaned_cloud.ply"
from correction.session import read_ply
_, cur = read_ply(live)
assert len(cur) == len(data) and np.allclose(cur["x"], data["x"]) and np.allclose(cur["z"], data["z"]), "selection differs"
hdr = ["ply", "format binary_little_endian 1.0", "comment epoch 7: epoch 4 + epoch 5 per 0.5 m cell, provenance carried",
       f"element vertex {len(data)}"]
tn = {"f4": "float", "u1": "uchar", "i4": "int", "i2": "short", "f8": "double", "u2": "ushort", "i8": "int64"}
for n in data.dtype.names:
    hdr.append(f"property {tn[data.dtype[n].str[1:]]} {n}")
hdr.append("end_header")
tmp = live.with_suffix(".tmp")
with open(tmp, "wb") as f:
    f.write(("\n".join(hdr) + "\n").encode()); f.write(data.tobytes())
tmp.replace(live)
print("rewritten with fields", data.dtype.names, len(data))
