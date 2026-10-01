"""Omega + DA3 per pixel, each where it is measurably better -> epoch 6. USER 2026-09-30: "DA3 falla a la
distancia pero de cerca es muy preciso; omega tiene mejor consistencia a la distancia y de costado".

Common frame: F5-R0 poses + Omega's camera (params fixed), Omega's native grid (832x464).
  Omega: its own depth (owner chunk) x a SMOOTH correction along the walk: the held-out-chosen
         'gradient' ratio model fitted per keyframe on the F5 landmarks of keyframes i-w..i+w;
         w chosen by held-out half A.
  DA3:   epoch 4's depth (conditioned on R0), carried to the R0 camera through its own final pose
         and z-buffered onto the same grid; its bottom-10 % confidence left out as in epoch 4.
Judge: F5-R0 HELD-OUT landmarks, split by track in two halves. Half A: w, and each source's error
  (median |dz|/z) per cell of (landmark distance x incidence angle), cells at the quantiles of half A.
  Half B: the report only.
Per pixel: both agree within z * 1.96 * sqrt(s_o^2 + s_d^2) (the project's declared 95 % confidence)
  -> inverse-variance mean; otherwise the lower-error source of that cell; one source -> that one.
Cloud: the cloud stage's voxel + SOR, octree, output/_epoch_6.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
EPOCH = 6
DST, TMP = O / f"_epoch_{EPOCH}", O / "_tx_combined"
CH = O / "maplong_run" / "_tmp_results_aligned"
DA3DIR = O / "da3_stream_r0" / "results_output"
Z95 = 1.959964
WINDOWS = (0, 1, 2, 4, 8, 16)
NQ = 5                                   # quantile cells per axis (5 x 5)


def log(m):
    print(f"[combo {time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
from config import cfg
from precision.config import load_precision_config
from precision.camera import undistort_solver
from precision.tracks import load_tracks_v2
from precision import refine as RF
pcfg = load_precision_config(); solver = undistort_solver(pcfg.camera)
params0 = list(json.loads((O / "precision" / "refine.json").read_text())["camera"]["before"]); KR = RF.K_of(params0)
R0 = np.loadtxt(O / "precision" / "f5_r0_camera_poses.txt").reshape(-1, 4, 4); w2c = np.linalg.inv(R0)
e4 = O / "_epoch_4" / "camera_poses.txt"
if not e4.exists():
    e4 = O / "camera_poses.txt"
    assert json.loads((O / "geometry_epoch.json").read_text())["epoch"] == 4, "epoch 4's poses not found"
D3P = np.loadtxt(e4).reshape(-1, 4, 4)                          # DA3's final c2w (epoch 4)
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
N = len(frames)
tr = load_tracks_v2(S)
split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
obs = {}
for sp in (0, 1):
    m = split == sp
    g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
    X = RF.triangulate_tracks(g, w2c, params0, solver, pcfg.refine.min_tri_deg)
    rows = {i: [] for i in range(N)}
    for t, lst in g.items():
        if t in X:
            for i, p in lst:
                z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                if z > 0.05: rows[i].append((p[0], p[1], z, t % 2))      # half A = even track id
    obs[sp] = {i: np.array(v, float).reshape(-1, 4) for i, v in rows.items()}
    log(f"{'fit' if sp == 0 else 'held-out'}: {len(X):,} landmarks triangulated with R0")

plan = json.loads((O / "chunk_plan.json").read_text())["chunk_ranges"]
owner = {i: min((c for c, (a, b) in enumerate(plan) if a <= i < b), key=lambda c: abs(i - (plan[c][0] + plan[c][1] - 1) / 2))
         for i in range(N)}
simple = cfg["reconstruction"]["simple"]; pct, floor_norm = float(simple["conf_percentile"]), float(simple["conf_min_norm"])


def bil(img, u, v):
    H, W = img.shape; u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int); du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def near(img, u, v):
    H, W = img.shape
    return img[np.clip(np.round(v).astype(int), 0, H - 1), np.clip(np.round(u).astype(int), 0, W - 1)]


# ── Omega per keyframe ──
om = {}
for c, (a, b) in enumerate(plan):
    d = np.load(CH / f"chunk_{c}.npy", allow_pickle=True).item()
    conf = d["world_points_conf"]; v_ = conf[np.isfinite(conf) & (conf > 0)]
    thr = max(np.percentile(v_, pct), v_.min() + floor_norm * (v_.max() - v_.min()))
    for jj, i in enumerate(range(a, b)):
        if owner[i] != c: continue
        zo = d["depth"][jj].astype(np.float32)
        om[i] = (zo, np.isfinite(conf[jj]) & (conf[jj] >= thr) & (zo > 0),
                 (np.clip(np.moveaxis(d["images"][jj], 0, -1), 0, 1) * 255).astype(np.uint8))
    del d
H, W = om[0][0].shape
log(f"Omega depth {W}x{H} for {len(om)} keyframes")

# ── DA3 onto the same grid, in R0's camera ──
allc = []
for f in frames:
    allc.append(np.load(DA3DIR / f"frame_{f}.npz")["conf"].ravel()[::50])
cthr = float(np.percentile(np.concatenate(allc), 10))
da = {}
for i, f in enumerate(frames):
    z = np.load(DA3DIR / f"frame_{f}.npz"); Dd, Kd = z["depth"], z["intrinsics"]
    ok = (Dd > 0) & np.isfinite(Dd) & (z["conf"] >= cthr)
    rr, cc = np.nonzero(ok); zz = Dd[rr, cc].astype(np.float64)
    Xc = np.stack([(cc - Kd[0, 2]) / Kd[0, 0] * zz, (rr - Kd[1, 2]) / Kd[1, 1] * zz, zz], 1)
    Xw = Xc @ D3P[i][:3, :3].T + D3P[i][:3, 3]
    Xr = Xw @ w2c[i][:3, :3].T + w2c[i][:3, 3]; zr = Xr[:, 2]
    u = np.round(KR[0, 0] * Xr[:, 0] / zr + KR[0, 2]).astype(int); v = np.round(KR[1, 1] * Xr[:, 1] / zr + KR[1, 2]).astype(int)
    m = (zr > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img = np.full(H * W, np.inf, np.float32)
    np.minimum.at(img, v[m] * W + u[m], zr[m].astype(np.float32))
    img[~np.isfinite(img)] = 0
    da[i] = img.reshape(H, W)
log(f"DA3 carried onto R0's grid (its conf gate {cthr:.3f})")

# ── smooth Omega correction: pooled gradient ratio model, window chosen by held-out half A ──
def des(u, v):
    return np.c_[np.ones(len(u)), (u - W / 2) / W, (v - H / 2) / H]


def fit(A, r):
    w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c; s = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * s / np.maximum(np.abs(e), 1e-12)))
    return c


fitrows = {}
for i in range(N):
    f_ = obs[0][i]; zo = bil(om[i][0], f_[:, 0], f_[:, 1]) if len(f_) else np.zeros(0)
    ok = zo > 0.05
    fitrows[i] = (des(f_[ok, 0], f_[ok, 1]), f_[ok, 2] / zo[ok])
heldA = {i: obs[1][i][obs[1][i][:, 3] == 0] for i in range(N)}
coefs, scoreW = {}, {}
for w in WINDOWS:
    cw, errs = {}, []
    for i in range(N):
        js = [j for j in range(max(0, i - w), min(N, i + w + 1))]
        A = np.vstack([fitrows[j][0] for j in js]); r = np.concatenate([fitrows[j][1] for j in js])
        cw[i] = fit(A, r) if len(r) >= 20 else np.array([1.0, 0, 0])
        h = heldA[i]
        if len(h):
            zo = bil(om[i][0], h[:, 0], h[:, 1]); ok = zo > 0.05
            errs.append(np.abs(zo[ok] * (des(h[ok, 0], h[ok, 1]) @ cw[i]) - h[ok, 2]) / h[ok, 2])
    coefs[w] = cw; scoreW[w] = float(np.median(np.concatenate(errs)))
bw = min(scoreW, key=scoreW.get)
log("smooth window (held-out A median |dz|/z): " + ", ".join(f"±{w} {scoreW[w]*100:.2f} %" for w in WINDOWS) + f" -> ±{bw}")
sc = np.array([coefs[bw][i][0] for i in range(N)])
log(f"   scale along the walk: largest keyframe-to-keyframe jump {np.abs(np.diff(sc)).max():.3f} (per-frame fit had 0.163)")
omc = {}
for i in range(N):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    k = (des(uu.ravel(), vv.ravel()) @ coefs[bw][i]).reshape(H, W)
    omc[i] = np.where(om[i][1], om[i][0] * k, 0).astype(np.float32)


# ── incidence angle from the corrected Omega depth (camera frame) ──
def incidence(z, step=3):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(uu - KR[0, 2]) / KR[0, 0] * z, (vv - KR[1, 2]) / KR[1, 1] * z, z], -1)
    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[:, step:-step] = P[:, 2 * step:] - P[:, :-2 * step]; dv[step:-step] = P[2 * step:] - P[:-2 * step]
    n = np.cross(du, dv); nn = np.linalg.norm(n, axis=-1) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cosang = np.abs((n * ray).sum(-1)) / nn
    bad = (z <= 0) | (nn < 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1))); ang[bad] = np.nan
    return ang


inc = {i: incidence(omc[i]) for i in range(N)}

# ── per-source error table on held-out half A ──
def samples(half):
    L = []
    for i in range(N):
        h = obs[1][i]; h = h[h[:, 3] == half]
        if not len(h): continue
        L.append(np.c_[np.full(len(h), i), h[:, :3], near(omc[i], h[:, 0], h[:, 1]), near(da[i], h[:, 0], h[:, 1]),
                       near(inc[i], h[:, 0], h[:, 1])])
    return np.vstack(L)                    # kf, u, v, zL, z_omega, z_da3, incidence


A_, B_ = samples(0), samples(1)
okA = np.isfinite(A_[:, 6])
dq = np.quantile(A_[:, 3], np.linspace(0, 1, NQ + 1)); aq = np.quantile(A_[okA, 6], np.linspace(0, 1, NQ + 1))


def cell(z, ang):
    di = np.clip(np.searchsorted(dq, z, side="right") - 1, 0, NQ - 1)
    ai = np.clip(np.searchsorted(aq, np.nan_to_num(ang, nan=aq[-1]), side="right") - 1, 0, NQ - 1)
    return di * NQ + ai


sig = {}
for name, col in (("omega", 4), ("da3", 5)):
    s = np.full(NQ * NQ, np.nan)
    ca = cell(A_[:, 3], A_[:, 6])
    for k in range(NQ * NQ):
        m = (ca == k) & (A_[:, col] > 0)
        if m.sum() >= 50:
            s[k] = np.median(np.abs(A_[m, col] - A_[m, 3]) / A_[m, 3])
    sig[name] = s
log("error per cell (half A, median |dz|/z %) — rows: distance, cols: incidence angle")
log(f"   distance edges m {np.round(dq, 2).tolist()} | incidence edges deg {np.round(aq, 1).tolist()}")
for di in range(NQ):
    log("   " + " | ".join(f"O {sig['omega'][di*NQ+ai]*100:4.1f} D {sig['da3'][di*NQ+ai]*100:4.1f}" for ai in range(NQ)))


def combine(zo, zd, ang, zref):
    k = cell(zref, ang); so, sd = sig["omega"][k], sig["da3"][k]
    so = np.where(np.isfinite(so), so, np.nanmedian(sig["omega"])); sd = np.where(np.isfinite(sd), sd, np.nanmedian(sig["da3"]))
    both = (zo > 0) & (zd > 0)
    agree = both & (np.abs(zo - zd) <= Z95 * zref * np.sqrt(so ** 2 + sd ** 2))
    wo, wd = 1 / so ** 2, 1 / sd ** 2
    out = np.where(zo > 0, zo, zd)
    out = np.where(both & ~agree, np.where(so <= sd, zo, zd), out)
    out = np.where(agree, (wo * zo + wd * zd) / (wo + wd), out)
    src = np.where(agree, 2, np.where(both, np.where(so <= sd, 0, 1), np.where(zo > 0, 0, 1)))
    return out, src


zc, _ = combine(B_[:, 4], B_[:, 5], B_[:, 6], np.where(B_[:, 4] > 0, B_[:, 4], B_[:, 5]))
for name, col in (("Omega corrected", 4), ("DA3", 5)):
    m = B_[:, col] > 0
    log(f"half B: {name:15s} median |dz|/z {np.median(np.abs(B_[m, col] - B_[m, 3]) / B_[m, 3])*100:.2f} % over {m.sum():,}")
m = zc > 0
log(f"half B: {'combined':15s} median |dz|/z {np.median(np.abs(zc[m] - B_[m, 3]) / B_[m, 3])*100:.2f} % over {m.sum():,}")

# ── cloud ──
for p in (TMP, DST):
    if p.exists(): shutil.rmtree(p)
(TMP / "chunks").mkdir(parents=True)
from precision.epoch0_cloud import _write_ply_xyzrgb
buf = {k: [] for k in ("xyz", "rgb", "fg", "pr", "pc", "cf")}; ci = 0; ntot = 0; srcn = np.zeros(3, np.int64)
for i in range(N):
    zo, zd = omc[i], da[i]
    zref = np.where(zo > 0, zo, zd)
    z, src = combine(zo.ravel(), zd.ravel(), inc[i].ravel(), zref.ravel())
    z = z.reshape(H, W); src = src.reshape(H, W)
    r_, c_ = np.nonzero(z > 0); zz = z[r_, c_].astype(np.float64)
    srcn += np.bincount(src[r_, c_].astype(int), minlength=3)
    Xc = np.stack([(c_ - KR[0, 2]) / KR[0, 0] * zz, (r_ - KR[1, 2]) / KR[1, 1] * zz, zz], 1)
    X = Xc @ R0[i][:3, :3].T + R0[i][:3, 3]
    buf["xyz"].append(X.astype(np.float32)); buf["rgb"].append(om[i][2][r_, c_])
    buf["fg"].append(np.full(len(r_), frames[i], np.int32)); buf["pr"].append(r_.astype(np.int16))
    buf["pc"].append(c_.astype(np.int16)); buf["cf"].append(src[r_, c_].astype(np.float32)); ntot += len(r_)
    if (i + 1) % 24 == 0 or i == N - 1:
        _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{ci:03d}.ply", np.concatenate(buf["xyz"]), np.concatenate(buf["rgb"]))
        np.savez(TMP / "chunks" / f"chunk_{ci:03d}_origins.npz", frame_global=np.concatenate(buf["fg"]),
                 pixel_row=np.concatenate(buf["pr"]), pixel_col=np.concatenate(buf["pc"]), confidence=np.concatenate(buf["cf"]))
        for kk in buf: buf[kk].clear()
        ci += 1
log(f"{ntot:,} raw points: Omega {srcn[0]/ntot*100:.1f} %, DA3 {srcn[1]/ntot*100:.1f} %, weighted mean of both {srcn[2]/ntot*100:.1f} %")
from precision.epoch0_cloud import clean_cmd
cleaned = TMP / "cleaned_cloud.ply"
r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd="/workspace/stac-build/server", capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "✅" in ln and ("→" in ln or "Merged" in ln): log(ln.strip())
if r.returncode or not cleaned.exists():
    log("cleaner failed " + r.stderr[-800:]); sys.exit(1)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
np.savetxt(DST / "camera_poses.txt", R0.reshape(N, -1))
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": f"Omega (smooth correction ±{bw} kf) + DA3 (epoch 4) per pixel by measured error per distance x incidence cell, "
            f"on F5-R0 poses + Omega camera; outside the pipeline 2026-09-30",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch {EPOCH} in {(time.time()-t0)/60:.1f} min")
