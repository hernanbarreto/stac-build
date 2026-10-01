"""Epoch 7 — Omega's depth BENT to F5 before the vote (USER 2026-10-01, option 1: "corregir la profundidad de
Omega antes de F6"). F6 contradicted 13 % (floor) to 49 % (walls) of Omega's pixels: Omega's depth, measured
with Omega's own poses and camera, put on F5's. Here, per keyframe:
  1. landmarks: F5's FIT tracks triangulated with F5 (R1) poses + camera; HELD-OUT tracks only judge
  2. Omega's depth × k(u,v) = c0 + c1 u + c2 v, fitted robustly on the landmarks of keyframes i-w..i+w;
     w chosen by the held-out (median |dz|/z)
  3. the vote (same rule as the DA3 fusion): a pixel leaves when more neighbours (±3, ±6, ±12) see free space
     through it than agree; else its depth is the median of the agreeing views. tau = p75 of the session's own
     neighbour disagreement. The ONE confidence floor (conf_min_norm, min-max per Omega chunk); sky out.
  4. holes: DA3 (conditioned on F5) aligned per keyframe and judged by this geometry (epoch 6's rule)
  5. cloud (Omega chunks as units), voxel + SOR, octree, masks (today's code), certify -> renumbered 7."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
TMP, DST, EPOCH = O / "_tx_omega_bent", O / "_epoch_7", 7
NB = (-12, -6, -3, 3, 6, 12)
WINDOWS = (0, 1, 2, 4, 8)


def log(m, level="info"):
    print(f"[bent {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


t0 = time.time()
from config import cfg
from precision.config import load_precision_config
from precision.camera import undistort_solver
from precision.tracks import load_tracks_v2
from precision import refine as RF
from precision import corrected_cloud as CC
from precision.epoch0_cloud import SKY_CONF, _write_ply_xyzrgb
from correction.run import run_select
from PIL import Image
pcfg = load_precision_config(); solver = undistort_solver(pcfg.camera)
p = json.loads((O / "precision" / "camera_f5_r1.json").read_text())["params"]
K = RF.K_of(p)
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
N = len(frames); H, W = 832, 464
F5 = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4); w2c = np.linalg.inv(F5)
c2w = {f: F5[i] for i, f in enumerate(frames)}

# 1. landmarks
tr = load_tracks_v2(S)
split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
obs = {}
for sp in (0, 1):
    m = split == sp
    g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
    X = RF.triangulate_tracks(g, w2c, p, solver, pcfg.refine.min_tri_deg)
    rows = {i: [] for i in range(N)}
    for t, lst in g.items():
        if t in X:
            for i, uv in lst:
                z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                if z > 0: rows[i].append((uv[0], uv[1], z))
    obs[sp] = {i: np.array(v, float).reshape(-1, 3) for i, v in rows.items()}
    log(f"{'fit' if sp == 0 else 'held-out'}: {len(X):,} landmarks with F5 R1")

# Omega's depth per keyframe (record grid = native here), conf floor per Omega chunk, sky
rec = O / "omega_run" / "results_output"
zo, conf, chunk = {}, {}, {}
for f in frames:
    with np.load(rec / f"frame_{f}.npz") as z:
        zo[f] = np.asarray(z["depth"], np.float32); conf[f] = np.asarray(z["conf"], np.float32); chunk[f] = int(z["chunk"])
floor_norm = float(cfg["reconstruction"]["simple"]["conf_min_norm"])
thr = {}
for k in set(chunk.values()):
    v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel() for f in frames if chunk[f] == k])
    thr[k] = v.min() + floor_norm * (v.max() - v.min())


def bil(img, u, v):
    u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int); du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv) + img[v0 + 1, u0] * (1 - du) * dv
            + img[v0 + 1, u0 + 1] * du * dv)


def des(u, v):
    return np.c_[np.ones(len(u)), (u - W / 2) / W, (v - H / 2) / H]


def irls(A, r):
    w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c; sc = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(e), 1e-12)))
    return c


# 2. the bend, window chosen by half A of the held-out
fitrows = {}
for i, f in enumerate(frames):
    o = obs[0][i]; z_ = bil(zo[f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0); ok = z_ > 0.05
    fitrows[i] = (des(o[ok, 0], o[ok, 1]), o[ok, 2] / z_[ok])
score, coefs = {}, {}
for wnd in WINDOWS:
    cw, errs = {}, []
    for i, f in enumerate(frames):
        js = range(max(0, i - wnd), min(N, i + wnd + 1))
        A = np.vstack([fitrows[j][0] for j in js]); r = np.concatenate([fitrows[j][1] for j in js])
        cw[i] = irls(A, r) if len(r) >= 20 else np.array([1.0, 0, 0])
        h = obs[1][i]; h = h[(np.arange(len(h)) % 2) == 0]
        if len(h):
            z_ = bil(zo[f], h[:, 0], h[:, 1]); ok = z_ > 0.05
            errs.append(np.abs(z_[ok] * (des(h[ok, 0], h[ok, 1]) @ cw[i]) - h[ok, 2]) / h[ok, 2])
    coefs[wnd] = cw; score[wnd] = float(np.median(np.concatenate(errs)))
wb = min(score, key=score.get)
raw = []
for i, f in enumerate(frames):
    h = obs[1][i]
    if len(h):
        z_ = bil(zo[f], h[:, 0], h[:, 1]); ok = z_ > 0.05; raw.append(np.abs(z_[ok] - h[ok, 2]) / h[ok, 2])
log("held-out |dz|/z: unbent " + f"{np.median(np.concatenate(raw)) * 100:.2f} %, " +
    ", ".join(f"±{w_} {score[w_] * 100:.2f} %" for w_ in WINDOWS) + f" -> ±{wb}")
sc_ = np.array([coefs[wb][i][0] for i in range(N)])
log(f"scale along the walk: median {np.median(sc_):.4f} [{sc_.min():.4f}, {sc_.max():.4f}], largest jump {np.abs(np.diff(sc_)).max():.3f}")
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
dep, ent = {}, {}
for i, f in enumerate(frames):
    k = (des(uu.ravel(), vv.ravel()) @ coefs[wb][i]).reshape(H, W)
    valid = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF) & (conf[f] >= thr[chunk[f]])
    dep[f] = np.where(valid, zo[f] * k, 0).astype(np.float32); ent[f] = valid

# 3. the vote
tau = CC.measured_tau(dep, ent, K, c2w, frames, NB, 75.0)
w2cd = {f: np.linalg.inv(c2w[f]) for f in frames}
final = {}
n_in = n_out = 0
for i, f in enumerate(frames):
    rr, cc = np.nonzero(ent[f]); z = dep[f][rr, cc].astype(np.float64)
    agree = np.zeros(len(z), np.int32); contra = np.zeros(len(z), np.int32); cand = [z]
    rw = np.stack([(cc - K[0, 2]) / K[0, 0], (rr - K[1, 2]) / K[1, 1], np.ones(len(rr))], 1) @ c2w[f][:3, :3].T
    Cf = c2w[f][:3, 3]
    for d in NB:
        j = i + d
        if not 0 <= j < N: continue
        g = frames[j]
        a = w2cd[g][2, :3] @ Cf + w2cd[g][2, 3]; b = rw @ w2cd[g][2, :3]
        Xw = Cf + z[:, None] * rw; Xj = Xw @ w2cd[g][:3, :3].T + w2cd[g][:3, 3]; zj = Xj[:, 2]
        okj = zj > 0; zs = np.where(okj, zj, 1)
        u = np.rint(K[0, 0] * Xj[:, 0] / zs + K[0, 2]).astype(int); v = np.rint(K[1, 1] * Xj[:, 1] / zs + K[1, 2]).astype(int)
        okj &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        dj = np.zeros(len(z)); dj[okj] = dep[g][v[okj], u[okj]]; okj &= dj > 0
        e = np.zeros(len(z)); e[okj] = (zj[okj] - dj[okj]) / dj[okj]
        ag = okj & (np.abs(e) <= tau) & (np.abs(b) > 0)
        agree += ag; contra += okj & (e < -tau)
        cand.append(np.where(ag, (dj - a) / np.where(np.abs(b) > 0, b, 1), np.nan))
    keep = contra <= agree
    zf = np.nanmedian(np.vstack(cand), 0)
    zz = np.zeros((H, W), np.float32); zz[rr[keep], cc[keep]] = zf[keep]
    src = np.full((H, W), 255, np.uint8); src[rr[keep], cc[keep]] = 1
    final[f] = (zz, src); n_in += len(z); n_out += int((~keep).sum())
log(f"vote: tau {tau * 100:.2f} %, {n_out / max(n_in, 1) * 100:.1f} % of the valid pixels contradicted (F6 on the "
    f"unbent depth: 32 %); coverage {sum((final[f][0] > 0).sum() for f in frames) / (N * H * W) * 100:.1f} % of all pixels")

# 4. DA3 holes: OFF — USER 2026-10-01 ("veo varias capas duplicadas y puntos voladores, probablemente de DA3")
log("DA3 holes: off (epoch 7 = Omega bent to F5 + the vote, nothing else)")

# 5. cloud
for pth in (TMP, DST):
    shutil.rmtree(pth, ignore_errors=True)
(TMP / "chunks").mkdir(parents=True)
by_chunk = {}
for f in frames: by_chunk.setdefault(chunk[f], []).append(f)
for k, fl in sorted(by_chunk.items()):
    L = {x: [] for x in ("xyz", "rgb", "fg", "pr", "pc", "cf")}
    for f in fl:
        z, src = final[f]; m = z > 0; r, c = vv[m], uu[m]; zz = z[m].astype(np.float64)
        X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
        img = np.asarray(Image.open(S / "frames" / f"{f:06d}.jpg").convert("RGB").resize((W, H)))
        L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[r, c]); L["fg"].append(np.full(len(r), f, np.int32))
        L["pr"].append(r.astype(np.int16)); L["pc"].append(c.astype(np.int16)); L["cf"].append(src[m].astype(np.float32))
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{k:03d}.ply", np.concatenate(L["xyz"]), np.concatenate(L["rgb"]))
    np.savez(TMP / "chunks" / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(L["fg"]),
             pixel_row=np.concatenate(L["pr"]), pixel_col=np.concatenate(L["pc"]), confidence=np.concatenate(L["cf"]))
cleaned = TMP / "cleaned_cloud.ply"
CC.clean(cfg, TMP / "chunks", cleaned, log)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("❌ octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": f"Omega bent to F5 per keyframe (±{wb}) before the vote; no DA3; floor per chunk blended across overlaps",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"cloud built ({(time.time() - t0) / 60:.1f} min) — epoch {EPOCH} live")
run_select(O, EPOCH, "auto", log=log)
E6 = O / "_epoch_6"                                           # epoch 6's segmentation travels with it
m6 = json.loads((E6 / "_manifest.json").read_text()); have = {a["rel"] for a in m6["artifacts"]}
for r_ in ("segmentation_result.json", "scene_r.db", "classification.npy", "class_map.json"):
    if (O / r_).exists():
        shutil.copy2(O / r_, E6 / r_)
        if r_ not in have: m6["artifacts"].append({"rel": r_, "existed_before": True})
(E6 / "_manifest.json").write_text(json.dumps(m6))
for r_ in ("segmentation_result.json", "classification.npy", "class_map.json"):
    if (O / r_).exists():
        (O / r_).rename(O / "precision" / f"before_e7_{r_}")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
log(f"  {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time() - t0) / 60:.1f} min)")
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
e_fin = live()
if e_fin != EPOCH:
    shutil.rmtree(O / f"_epoch_{EPOCH}", ignore_errors=True)
    recp = O / "geometry_epoch.json"; r_ = json.loads(recp.read_text()); r_.update(epoch=EPOCH, renumbered_from=e_fin)
    recp.write_text(json.dumps(r_, indent=2))
log(f"DONE in {(time.time() - t0) / 60:.1f} min — live {live()}, stored {sorted(x.name for x in O.glob('_epoch_*'))}")
