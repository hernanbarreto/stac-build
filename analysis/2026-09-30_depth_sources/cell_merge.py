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
for name, e in (("DA3", 4), ("Omega", 5)):
    v = _read_ply(str(where(e) / "cleaned_cloud.ply"))
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


results = {}
for size in CELLS:
    kd, sd = cell_stats(cl["DA3"][0], size); ko, so = cell_stats(cl["Omega"][0], size)
    win = {}; steps = []
    for k in set(sd) | set(so):
        if k in sd and k in so:
            win[k] = "DA3" if sd[k][0] < so[k][0] else "Omega"
            (_, c1, n1), (_, c2, _) = sd[k], so[k]
            steps.append(abs((c2 - c1) @ n1))
        else:
            win[k] = "DA3" if k in sd else "Omega"
    both = [k for k in win if k in sd and k in so]
    nd = sum(1 for k in both if win[k] == "DA3")
    thick = np.array([min(sd[k][0], so[k][0]) for k in both])
    log(f"cell {size:g} m: {len(win):,} cells ({len(both):,} covered by both) — DA3 wins {nd/len(both)*100:.1f} %, "
        f"Omega {100-nd/len(both)*100:.1f} % | thickness median DA3 {np.median([sd[k][0] for k in both])*100:.1f} cm, "
        f"Omega {np.median([so[k][0] for k in both])*100:.1f} cm, merged {np.median(thick)*100:.1f} cm | "
        f"plane-to-plane step median {np.median(steps)*100:.1f} cm p90 {np.percentile(steps,90)*100:.1f} cm")
    results[size] = (kd, ko, win)

kd, ko, win = results[PUBLISH_CELL]
keep_d = np.array([win.get(int(k)) == "DA3" for k in kd]); keep_o = np.array([win.get(int(k)) == "Omega" for k in ko])
# a point of a cell too sparse to judge (< MIN_PTS) goes with its cloud if the other cloud does not claim that cell
claimed = set(win)
keep_d |= np.array([int(k) not in claimed for k in kd]) & ~np.isin(kd, ko)
keep_o |= np.array([int(k) not in claimed for k in ko]) & ~np.isin(ko, kd)
xyz = np.vstack([cl["DA3"][0][keep_d], cl["Omega"][0][keep_o]]); rgb = np.vstack([cl["DA3"][1][keep_d], cl["Omega"][1][keep_o]])
log(f"published at {PUBLISH_CELL} m: {len(xyz):,} pts (DA3 {keep_d.sum():,}, Omega {keep_o.sum():,})")
for p in (TMP, DST):
    if p.exists(): shutil.rmtree(p)
TMP.mkdir()
from precision.epoch0_cloud import _write_ply_xyzrgb
_write_ply_xyzrgb(TMP / "cleaned_cloud.ply", xyz, rgb)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=TMP / "cleaned_cloud.ply", potree_dir_override=TMP / "potree"):
    log("octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(TMP / "cleaned_cloud.ply"), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_r0_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": f"epoch 4 (DA3) + epoch 5 (Omega), same R0 frame, merged per {PUBLISH_CELL} m cell by the thinner surface; outside the pipeline 2026-09-30",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch {EPOCH} in {(time.time()-t0)/60:.1f} min")
