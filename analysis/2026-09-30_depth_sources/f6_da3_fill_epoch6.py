"""PROTOTYPE of F6 tier 2 = DA3 conditioned on F5 (USER 2026-09-30: "pensá bien dónde debería correr … como
parte de F6") -> epoch 6. Per keyframe, per pixel, the depth decision F6 would make:
  tier 0 / tier 1  F6's own (sweep-confirmed / Omega × s_k), as stored in output/depth_native
  repair           a contradicted pixel takes the depth its F6 neighbours agree on (precision.corrected_cloud)
  tier 2 (NEW)     a pixel still empty takes DA3's depth (DA3-streaming conditioned on F5 R1 poses + camera,
                   output/da3_stream_f5) when >= 2 DA3 neighbours agree with it and fewer contradict it;
                   DA3's final pose (its chunk Sim(3) o F5) carries it onto F5's camera of that keyframe
Sky (Omega's SKY_CONF) never enters. Tolerances = p75 of each source's own neighbour disagreement.
Then the cloud (Omega chunks as units), SAM3 masks (co-visible split with the free-space test), certify."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
TMP, DST = O / "_tx_f6_da3", O / "_epoch_6"
NB = (-12, -6, -3, 3, 6, 12)


def log(m, level="info"):
    print(f"[f6-da3 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


t0 = time.time()
from config import cfg
from precision.config import load_precision_config
from precision import corrected_cloud as CC
from precision.depth_sweep import SOURCE_SWEEP, SOURCE_PRIOR_FILL, DISCARD_CONTRADICTED
from precision.epoch0_cloud import SKY_CONF, _write_ply_xyzrgb
from correction.run import run_select
from PIL import Image
pcfg = load_precision_config(); cc = pcfg.cloud
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
F5 = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
p = json.loads((O / "precision" / "camera_f5_r1.json").read_text())["params"]
K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], np.float64)
H, W = 832, 464
c2w = {f: F5[i] for i, f in enumerate(frames)}

# ── F6 + repair ──
f6 = {f: CC.f6_frame(O / "depth_native", f, (W, H)) for f in frames}
dep = {f: f6[f]["depth"] for f in frames}; ent = {f: f6[f]["enter"] for f in frames}
bad = {f: f6[f]["source"] == DISCARD_CONTRADICTED for f in frames}
tau6 = CC.measured_tau(dep, ent, K, c2w, frames, cc.repair_neighbors, cc.repair_tau_quantile)
rep = CC.repair_contradicted(dep, ent, bad, K, c2w, frames, cc.repair_neighbors, tau6, cc.repair_min_views)
final = {}
for f in frames:
    z = dep[f].copy(); src = np.where(ent[f], f6[f]["source"], 255).astype(np.uint8)
    if f in rep:
        r, c, zz = rep[f]; z[r, c] = zz; src[r, c] = 20                     # 20 = repaired
    final[f] = (z, src)
n_px = len(frames) * H * W
log(f"F6: tier 0/1 {sum(e.sum() for e in ent.values()) / n_px * 100:.1f} %, repaired "
    f"{sum(len(v[0]) for v in rep.values()) / n_px * 100:.1f} % (tau {tau6 * 100:.2f} %)")
del f6, bad

# ── DA3 conditioned on F5: final pose per keyframe, confirmation by its own neighbours ──
RUN = O / "da3_stream_f5" / "results_output"
dd, dK, dc2w = {}, None, {}
for f in frames:
    zz = np.load(RUN / f"frame_{f}.npz")
    e = np.eye(4); e[:3, :4] = zz["extrinsics"][:3, :4]; cl = np.linalg.inv(e)
    s_, R_, T_ = float(zz["s"]), np.asarray(zz["R"]).reshape(3, 3), np.asarray(zz["T"]).reshape(3)
    P = np.eye(4); P[:3, :3] = R_ @ cl[:3, :3]; P[:3, 3] = s_ * R_ @ cl[:3, 3] + T_
    dd[f] = zz["depth"].astype(np.float32); dc2w[f] = P; dK = zz["intrinsics"].astype(np.float64)
dent = {f: (dd[f] > 0) & np.isfinite(dd[f]) for f in frames}
tauD = CC.measured_tau(dd, dent, dK, dc2w, frames, NB, cc.repair_tau_quantile)
dw2c = {f: np.linalg.inv(dc2w[f]) for f in frames}
dH, dW = dd[frames[0]].shape
conf_da3 = {}
for i, f in enumerate(frames):
    agree = np.zeros((dH, dW), np.int16); contra = np.zeros((dH, dW), np.int16)
    zi = dd[f].astype(np.float64)
    for d in NB:
        j = i + d
        if not 0 <= j < len(frames):
            continue
        g = frames[j]
        sg = CC._splat(dd[g], dent[g], dK, dc2w[g], dw2c[f], (dH, dW))         # g's surface seen from f
        m2 = np.isfinite(sg) & dent[f]
        rel = np.zeros((dH, dW)); rel[m2] = (zi[m2] - sg[m2]) / sg[m2]
        agree += (m2 & (np.abs(rel) <= tauD)).astype(np.int16)
        contra += (m2 & (rel > tauD)).astype(np.int16)                          # f's point BEHIND g's surface:
    conf_da3[f] = dent[f] & (agree >= cc.repair_min_views) & (contra < agree)  # g saw free space there
log(f"DA3: tau {tauD * 100:.2f} %, confirmed {sum(v.sum() for v in conf_da3.values()) / (len(frames) * dH * dW) * 100:.1f} % of its pixels")

# ── tier 2: fill what is still empty — DA3 aligned to Omega per keyframe, judged by Omega's geometry ──
# USER 2026-09-30 (first prototype): the door came out duplicated — DA3 differs from Omega's depth by
# 0.93-1.03 per keyframe and 3-5 % per pixel (22 cm at 5 m). Now: (1) per keyframe DA3 is fitted onto
# F6/Omega (robust ratio c0 + c1 u + c2 v on the pixels both see), (2) a filled pixel enters only when
# NO Omega neighbour sees free space through it and, where an Omega neighbour sees a surface there, at
# least one agrees within Omega's own tau (tau6). Where no Omega neighbour sees anything it is a hole.
def _irls(A, r):
    w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c; sc = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(e), 1e-12)))
    return c


n_fill = n_cand = 0
rec_dir = O / "omega_run" / "results_output"
w2c5 = {f: np.linalg.inv(c2w[f]) for f in frames}
vv, uu = np.mgrid[0:H, 0:W]
align = []
for i, f in enumerate(frames):
    z, src = final[f]
    r, c = np.nonzero(conf_da3[f]); zz = dd[f][r, c].astype(np.float64)
    X = np.stack([(c - dK[0, 2]) / dK[0, 0] * zz, (r - dK[1, 2]) / dK[1, 1] * zz, zz], 1)
    Xw = X @ dc2w[f][:3, :3].T + dc2w[f][:3, 3]; Xc = Xw @ w2c5[f][:3, :3].T + w2c5[f][:3, 3]
    ok = Xc[:, 2] > 0
    u = np.rint(K[0, 0] * Xc[ok, 0] / Xc[ok, 2] + K[0, 2]).astype(int); v = np.rint(K[1, 1] * Xc[ok, 1] / Xc[ok, 2] + K[1, 2]).astype(int)
    zc = Xc[ok, 2]; inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    buf = np.full(H * W, np.inf); np.minimum.at(buf, v[inb] * W + u[inb], zc[inb]); buf = buf.reshape(H, W)
    both = (z > 0) & np.isfinite(buf)
    if both.sum() >= 1000:
        A = np.c_[np.ones(both.sum()), (uu[both] - W / 2) / W, (vv[both] - H / 2) / H]
        cf = _irls(A, z[both] / buf[both])
        kmap = cf[0] + cf[1] * (uu - W / 2) / W + cf[2] * (vv - H / 2) / H
        buf = buf * kmap; align.append(cf[0])
    with np.load(rec_dir / f"frame_{f}.npz") as rz:
        sky = np.asarray(rz["conf"], np.float32) <= SKY_CONF
    cand = (z <= 0) & np.isfinite(buf) & ~sky
    rr, cc = np.nonzero(cand); n_cand += len(rr)
    if not len(rr):
        continue
    zq = buf[rr, cc]
    Xq = np.stack([(cc - K[0, 2]) / K[0, 0] * zq, (rr - K[1, 2]) / K[1, 1] * zq, zq], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
    agree = np.zeros(len(rr), np.int16); contra = np.zeros(len(rr), np.int16); seen = np.zeros(len(rr), np.int16)
    for d in NB:
        j = i + d
        if not 0 <= j < len(frames):
            continue
        g = frames[j]; zg = final[g][0]
        Y = Xq @ w2c5[g][:3, :3].T + w2c5[g][:3, 3]; yz = Y[:, 2]
        okj = yz > 0; yzs = np.where(okj, yz, 1)
        uj = np.rint(K[0, 0] * Y[:, 0] / yzs + K[0, 2]).astype(int); vj = np.rint(K[1, 1] * Y[:, 1] / yzs + K[1, 2]).astype(int)
        okj &= (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
        dj = np.zeros(len(rr)); dj[okj] = zg[vj[okj], uj[okj]]
        has = okj & (dj > 0)
        e = np.zeros(len(rr)); e[has] = (yz[has] - dj[has]) / dj[has]
        seen += has; agree += has & (np.abs(e) <= tau6); contra += has & (e < -tau6)
    take = (contra == 0) & ((agree >= 1) | (seen == 0))
    z[rr[take], cc[take]] = zq[take]; src[rr[take], cc[take]] = 30
    n_fill += int(take.sum())
log(f"DA3 aligned to Omega per keyframe: scale c0 median {np.median(align):.4f} [{np.percentile(align, 5):.4f}, "
    f"{np.percentile(align, 95):.4f}]")
log(f"tier 2 (DA3): {n_cand / n_px * 100:.1f} % candidates, {n_fill / n_px * 100:.1f} % of all pixels filled "
    f"after Omega's judge (tau {tau6 * 100:.2f} %)")

# ── cloud: Omega chunks as units, the cleaner, octree, epoch 6 ──
for pth in (TMP, DST):
    shutil.rmtree(pth, ignore_errors=True)
(TMP / "chunks").mkdir(parents=True)
by_chunk = {}
for i, f in enumerate(frames):
    with np.load(rec_dir / f"frame_{f}.npz") as rz:
        by_chunk.setdefault(int(rz["chunk"]), []).append(f)
tot = {0: 0, 1: 0, 20: 0, 30: 0}
for k, fl in sorted(by_chunk.items()):
    L = {x: [] for x in ("xyz", "rgb", "fg", "pr", "pc", "cf")}
    for f in fl:
        z, src = final[f]; m = z > 0
        for key in tot: tot[key] += int((src[m] == key).sum())
        r, c = vv[m], uu[m]; zz = z[m].astype(np.float64)
        X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
        img = np.asarray(Image.open(S / "frames" / f"{f:06d}.jpg").convert("RGB").resize((W, H)))
        L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[r, c]); L["fg"].append(np.full(len(r), f, np.int32))
        L["pr"].append(r.astype(np.int16)); L["pc"].append(c.astype(np.int16)); L["cf"].append(src[m].astype(np.float32))
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{k:03d}.ply", np.concatenate(L["xyz"]), np.concatenate(L["rgb"]))
    np.savez(TMP / "chunks" / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(L["fg"]),
             pixel_row=np.concatenate(L["pr"]), pixel_col=np.concatenate(L["pc"]), confidence=np.concatenate(L["cf"]))
log(f"raw points by source: tier0 {tot[0]:,}, tier1 {tot[1]:,}, repaired {tot[20]:,}, DA3 {tot[30]:,}")
cleaned = TMP / "cleaned_cloud.ply"
CC.clean(cfg, TMP / "chunks", cleaned, log)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("❌ octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": 6, "epoch_from": 6, "epoch_to": 6, "kind": "new_cloud",
    "note": "PROTOTYPE F6 tier 2: F6 tiers + repair + DA3 (conditioned on F5) filling what stays empty",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"cloud built ({(time.time() - t0) / 60:.1f} min) — epoch 6 live")
run_select(O, 6, "auto", log=log)
seg = O / "segmentation_result.json"
if seg.exists():
    seg.rename(O / "precision" / "segmentation_result.before_epoch6.json")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
log(f"  {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time() - t0) / 60:.1f} min)")
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
e_fin = live()
if e_fin != 6:
    shutil.rmtree(O / "_epoch_6", ignore_errors=True)
    recp = O / "geometry_epoch.json"; r_ = json.loads(recp.read_text()); r_.update(epoch=6, renumbered_from=e_fin)
    recp.write_text(json.dumps(r_, indent=2))
log(f"DONE in {(time.time() - t0) / 60:.1f} min — live {live()}, stored {sorted(x.name for x in O.glob('_epoch_*'))}")
