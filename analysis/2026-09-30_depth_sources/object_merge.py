"""Epoch 4 (DA3 on R0) + epoch 5 (Omega on R0), SAME frame, merged PER OBJECT -> epoch 7 (redo).
USER 2026-09-30: the per-cell merge "está mal mezclada" — cells of one surface came from two sources
3.4 cm (median) / 14.7 cm (p90) apart, so every cell border cut a step through the surface.

Each point already knows its keyframe and pixel (frame_global, pixel_row, pixel_col on the native
832x464 grid), so its SAM3 object comes straight from seg_masks.npz (key f<kf index>_o<id>) — nothing
is re-projected. Per object, per cloud: the median over the object's 0.25 m cells (>= MIN_PTS points in
BOTH clouds) of the local surface thickness (PCA plane, p90 - p10). The whole object comes from the
thinner cloud; the unsegmented remainder comes, whole, from the cloud thinner over it.
"""
import json, re, shutil, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
EPOCH = 7
DST, TMP = O / f"_epoch_{EPOCH}", O / "_tx_object_merge"
CELL, MIN_PTS = 0.25, 30


def log(m):
    print(f"[objmerge {time.strftime('%H:%M:%S')}] {m}", flush=True)


def where(e):
    live = json.loads((O / "geometry_epoch.json").read_text())["epoch"]
    return O if live == e else O / f"_epoch_{e}"


t0 = time.time()
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
kf_of = {f: k for k, f in enumerate(frames)}
seg = json.loads((O / "segmentation.json").read_text())
label_of = {int(i["id"]): i["label"] for i in seg["instances"]}
z = np.load(O / "seg_masks.npz")
H, W = 832, 464
lab = np.full((len(frames), H, W), -1, np.int32)
area = {}
keys = [k for k in z.files if re.match(r"f(\d+)_o(\d+)$", k)]
# smaller masks painted last so a small object on a large one keeps its pixels
for k in sorted(keys, key=lambda k: -int(z[k].sum())):
    f, o = map(int, re.match(r"f(\d+)_o(\d+)$", k).groups())
    m = z[k] > 0
    lab[f][m] = o
log(f"{len(keys)} masks over {len(frames)} keyframes, {len(label_of)} objects")

from reconstruction.gpu_cloud_clean import _read_ply
cl = {}
for name, e in (("DA3", 4), ("Omega", 5)):
    v = _read_ply(str(where(e) / "cleaned_cloud.ply"))
    fg = v["frame_global"].astype(np.int64)
    lut = np.full(max(int(fg.max()), max(frames)) + 1, -1, np.int64); lut[frames] = np.arange(len(frames))
    k_ = np.where(fg >= 0, lut[np.clip(fg, 0, len(lut) - 1)], -1)
    r = np.clip(v["pixel_row"].astype(np.int64), 0, H - 1); c = np.clip(v["pixel_col"].astype(np.int64), 0, W - 1)
    obj = np.where(k_ >= 0, lab[np.clip(k_, 0, len(frames) - 1), r, c], -1)
    cl[name] = (v, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), obj)
    log(f"{name} (epoch {e}): {len(v):,} pts, {np.mean(obj >= 0) * 100:.1f} % inside an object mask")

lo = np.minimum(cl["DA3"][1].min(0), cl["Omega"][1].min(0))
rng = np.random.default_rng(0)


def cell_thick(xyz):
    k = np.floor((xyz - lo) / CELL).astype(np.int64)
    key = (k[:, 0] * 100000 + k[:, 1]) * 100000 + k[:, 2]
    order = np.argsort(key, kind="stable"); ks = key[order]
    st = np.r_[0, np.nonzero(np.diff(ks))[0] + 1, len(ks)]
    out = {}
    for a, b in zip(st[:-1], st[1:]):
        if b - a < MIN_PTS: continue
        idx = order[a:b]
        if len(idx) > 2000: idx = rng.choice(idx, 2000, replace=False)
        P = xyz[idx] - xyz[idx].mean(0)
        n = np.linalg.svd(P, full_matrices=False)[2][2]
        d = P @ n
        out[int(ks[a])] = float(np.percentile(d, 90) - np.percentile(d, 10))
    return out


def judge(md, mo):
    td, to = cell_thick(cl["DA3"][1][md]), cell_thick(cl["Omega"][1][mo])
    both = sorted(set(td) & set(to))
    if not both:
        return ("DA3" if md.sum() >= mo.sum() else "Omega"), None, None, 0
    a, b = np.median([td[k] for k in both]), np.median([to[k] for k in both])
    return ("DA3" if a < b else "Omega"), a, b, len(both)


choice, rows = {}, []
ids = sorted(set(np.unique(cl["DA3"][2])) | set(np.unique(cl["Omega"][2])))
for o in ids:
    md, mo = cl["DA3"][2] == o, cl["Omega"][2] == o
    w, a, b, n = judge(md, mo)
    choice[o] = w
    rows.append((o, label_of.get(o, "unsegmented"), w, a, b, n, int(md.sum()), int(mo.sum())))
# per class summary
by = {}
for o, l, w, a, b, n, nd, no in rows:
    d = by.setdefault(l, {"DA3": 0, "Omega": 0, "pts": 0}); d[w] += 1; d["pts"] += nd + no
log("per class — objects won (DA3 / Omega):")
for l, d in sorted(by.items(), key=lambda x: -x[1]["pts"]):
    log(f"   {l:18s} DA3 {d['DA3']:3d} / Omega {d['Omega']:3d}")
big = sorted(rows, key=lambda r: -(r[6] + r[7]))[:12]
log("largest objects: label, winner, median thickness DA3 / Omega (cm), cells judged")
for o, l, w, a, b, n, nd, no in big:
    log(f"   {l:18s} #{o:<4d} -> {w:5s}  {'' if a is None else f'{a*100:.1f} / {b*100:.1f}'}  ({n} cells)")

keep_d = np.isin(cl["DA3"][2], [o for o in ids if choice[o] == "DA3"])
keep_o = np.isin(cl["Omega"][2], [o for o in ids if choice[o] == "Omega"])
data = np.concatenate([cl["DA3"][0][keep_d], cl["Omega"][0][keep_o]])
log(f"merged: {len(data):,} pts (DA3 {keep_d.sum():,}, Omega {keep_o.sum():,})")
for p in (TMP, DST):
    if p.exists(): shutil.rmtree(p)
TMP.mkdir()
tn = {"f4": "float", "u1": "uchar", "i4": "int", "i2": "short", "f8": "double", "u2": "ushort"}
hdr = ["ply", "format binary_little_endian 1.0", "comment epoch 7: epochs 4 + 5 merged per SAM3 object by thinner surface",
       f"element vertex {len(data)}"] + [f"property {tn[data.dtype[n].str[1:]]} {n}" for n in data.dtype.names] + ["end_header"]
with open(TMP / "cleaned_cloud.ply", "wb") as f:
    f.write(("\n".join(hdr) + "\n").encode()); f.write(data.tobytes())
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=TMP / "cleaned_cloud.ply", potree_dir_override=TMP / "potree"):
    log("octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(TMP / "cleaned_cloud.ply"), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_r0_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": "epoch 4 (DA3) + epoch 5 (Omega), same R0 frame, merged PER SAM3 OBJECT by the thinner surface; outside the pipeline 2026-09-30",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
(DST / "object_choice.json").write_text(json.dumps([{"id": int(o), "label": l, "winner": w, "thick_da3_m": a, "thick_omega_m": b,
    "cells": n, "pts_da3": nd, "pts_omega": no} for o, l, w, a, b, n, nd, no in rows]))
shutil.rmtree(TMP, ignore_errors=True)
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch {EPOCH} in {(time.time()-t0)/60:.1f} min")
