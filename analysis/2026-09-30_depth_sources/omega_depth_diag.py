"""Omega's depth, per keyframe, bent only as smoothly as F5's landmarks require, on F5-R0's
poses and camera (= Omega's camera, fixed) -> epoch 5. USER 2026-09-30: "que omega engulla las
poses y focal de F5"; Omega takes no camera input, R0 showed the focal change bought nothing.

1. Landmarks: F5's FIT tracks triangulated with R0's poses + camera (precision.refine helpers).
   The HELD-OUT tracks, triangulated the same way, are only used to judge.
2. Per keyframe, the landmarks' depth in that camera vs Omega's depth at the observed pixel.
   Correction models of the depth RATIO z_F5 / z_omega, fitted robustly per keyframe:
     none      1
     scale     c0
     gradient  c0 + c1 u + c2 v          (u, v normalised image coords)
   The model is chosen by the held-out landmarks (median |z - z_F5| / z_F5 over all keyframes),
   never by a number chosen here.
3. Points: c2w_R0 . (k(u,v) z_omega K_R0^-1 [u v 1]); Omega's own confidence gate per chunk
   (conf_percentile + conf_min_norm floor); each keyframe from the chunk where it is most central.
4. The cloud stage's voxel + SOR, octree, output/_epoch_5.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
EPOCH = 5
DST, TMP = O / f"_epoch_{EPOCH}", O / "_tx_omega_depth_r0"
CH = O / "maplong_run" / "_tmp_results_aligned"


def log(m):
    print(f"[omega-d {time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
from config import cfg
from precision.config import load_precision_config
from precision.camera import undistort_solver
from precision.tracks import load_tracks_v2
from precision import refine as RF
pcfg = load_precision_config()
solver = undistort_solver(pcfg.camera)
params0 = list(json.loads((O / "precision" / "refine.json").read_text())["camera"]["before"])
KR = RF.K_of(params0)
R0 = np.loadtxt(O / "precision" / "f5_r0_camera_poses.txt").reshape(-1, 4, 4); w2c = np.linalg.inv(R0)
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
N = len(frames)
tr = load_tracks_v2(S)
split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
groups = {}
for sp in (0, 1):
    m = split == sp
    g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
    X = RF.triangulate_tracks(g, w2c, params0, solver, pcfg.refine.min_tri_deg)
    groups[sp] = (g, X)
    log(f"{'fit' if sp == 0 else 'held-out'} tracks: {len(g):,} grouped, {len(X):,} triangulated with R0")

# per keyframe: (uv, z_landmark) for fit and held-out
obs = {sp: {i: [] for i in range(N)} for sp in (0, 1)}
for sp, (g, X) in groups.items():
    for t, lst in g.items():
        if t not in X: continue
        Xt = np.asarray(X[t], float)
        for i, p in lst:
            z = w2c[i][2, :3] @ Xt + w2c[i][2, 3]
            if z > 0.05: obs[sp][i].append((p[0], p[1], z))
obs = {sp: {i: np.array(v, float).reshape(-1, 3) for i, v in d.items()} for sp, d in obs.items()}

plan = json.loads((O / "chunk_plan.json").read_text())["chunk_ranges"]
owner = {i: min((c for c, (a, b) in enumerate(plan) if a <= i < b), key=lambda c: abs(i - (plan[c][0] + plan[c][1] - 1) / 2))
         for i in range(N)}
MODELS = ("none", "scale", "gradient")


def design(u, v, W, H, model):
    un, vn = (u - W / 2) / W, (v - H / 2) / H
    if model == "none": return None
    if model == "scale": return np.ones((len(u), 1))
    return np.c_[np.ones(len(u)), un, vn]


def fit(A, r):
    """robust least squares (IRLS, Huber at 1.345 x the MAD scale of the residual)"""
    w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c; s = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * s / np.maximum(np.abs(e), 1e-12)))
    return c


def bil(img, u, v):
    H, W = img.shape; u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int); du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


from config import cfg as _c
simple = _c["reconstruction"]["simple"]; pct, floor_norm = float(simple["conf_percentile"]), float(simple["conf_min_norm"])
coef = {m: {} for m in MODELS}; held_err = {m: [] for m in MODELS}; nfit = []
depth_cache = {}
for c, (a, b) in enumerate(plan):
    d = np.load(CH / f"chunk_{c}.npy", allow_pickle=True).item()
    conf = d["world_points_conf"]; v_ = conf[np.isfinite(conf) & (conf > 0)]
    thr = max(np.percentile(v_, pct), v_.min() + floor_norm * (v_.max() - v_.min()))
    for jj, i in enumerate(range(a, b)):
        if owner[i] != c: continue
        zo = d["depth"][jj].astype(np.float64); H, W = zo.shape
        depth_cache[i] = (zo.astype(np.float32), (np.isfinite(conf[jj]) & (conf[jj] >= thr) & (zo > 0)),
                          conf[jj].astype(np.float32), (np.clip(np.moveaxis(d["images"][jj], 0, -1), 0, 1) * 255).astype(np.uint8))
        f_, h_ = obs[0][i], obs[1][i]
        zf = bil(zo, f_[:, 0], f_[:, 1]) if len(f_) else np.zeros(0)
        ok = zf > 0.05; f_, zf = f_[ok], zf[ok]; nfit.append(len(f_))
        zh = bil(zo, h_[:, 0], h_[:, 1]) if len(h_) else np.zeros(0)
        okh = zh > 0.05; h_, zh = h_[okh], zh[okh]
        for mdl in MODELS:
            if mdl == "none" or len(f_) < 20:
                cc = None
            else:
                cc = fit(design(f_[:, 0], f_[:, 1], W, H, mdl), f_[:, 2] / zf)
            coef[mdl][i] = cc
            if len(h_):
                k = np.ones(len(h_)) if cc is None else design(h_[:, 0], h_[:, 1], W, H, mdl) @ cc
                held_err[mdl].append(np.abs(k * zh - h_[:, 2]) / h_[:, 2])
    del d
    log(f"chunk {c} read")
res = {m: float(np.median(np.concatenate(held_err[m]))) for m in MODELS}
best = min(res, key=res.get)
log(f"landmarks per keyframe (fit): median {np.median(nfit):.0f}, min {min(nfit)}")
log("held-out |z - z_F5| / z_F5 median: " + ", ".join(f"{m} {res[m]*100:.2f} %" for m in MODELS) + f" -> {best}")
sc = np.array([coef[best][i][0] if coef[best][i] is not None else 1.0 for i in range(N)])
log(f"per-keyframe depth scale ({best}): median {np.median(sc):.4f}, p5 {np.percentile(sc,5):.4f}, p95 {np.percentile(sc,95):.4f}")


FIN = np.loadtxt(O / "maplong_run" / "camera_poses.txt").reshape(-1, 4, 4)
mv = np.linalg.norm(R0[:, :3, 3] - FIN[:, :3, 3], axis=1)
rows = []
for i in range(N):
    cc = coef[best][i]; h_ = obs[1][i]
    rows.append((i, owner[i], len(obs[0][i]), mv[i], (cc[0] if cc is not None else np.nan),
                 (np.hypot(cc[1], cc[2]) if cc is not None and len(cc) > 2 else np.nan)))
rows = np.array(rows, float)
per_kf_err = {}
print("chunk | kf range | R0 camera move vs omega (median/max cm) | depth scale median [min,max] | gradient |c1,c2| median/max")
for c in range(len(plan)):
    r = rows[rows[:, 1] == c]
    print(f"  {c} | {int(r[0,0])}-{int(r[-1,0])} | {np.median(r[:,3])*100:.1f}/{r[:,3].max()*100:.1f} | "
          f"{np.nanmedian(r[:,4]):.3f} [{np.nanmin(r[:,4]):.3f},{np.nanmax(r[:,4]):.3f}] | {np.nanmedian(r[:,5]):.3f}/{np.nanmax(r[:,5]):.3f}")
jump = np.abs(np.diff(rows[:, 4]))
top = np.argsort(-jump)[:8]
print("largest keyframe-to-keyframe jumps of the depth scale:", [(int(rows[k,0]), int(rows[k+1,0]), round(float(jump[k]),3)) for k in top])
low = np.argsort(rows[:, 4])[:8]
print("lowest scales:", [(int(rows[k,0]), round(float(rows[k,4]),3), int(rows[k,2])) for k in low])
