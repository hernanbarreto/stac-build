"""DA3-streaming 120/60 conditioned on F5 (camera + poses) -> epoch 1. USER 2026-09-30: "última
oportunidad a da3 streaming, 120/60 … la nueva va a ser epoch1 … filtra el 10% peor".

1. DA3-streaming (the pipeline's config builder, map_worker._build_da3_config; overrides: 120/60,
   loop on, sky removed, per-frame Sim(3) saved, no gaussians), every chunk conditioned on F5's
   w2c + K through the vendor hook `_stac_chunk_camera_priors`, native resolution (process_res =
   the frames' long side). Chunks are Sim(3)-aligned on their overlap points, SALAD loop closure.
2. Per frame: depth already x s of its chunk, final pose = (s, R, T) o chunk-local pose.
3. Bottom 10 % of DA3 confidence (session-wide) out — they neither vote nor enter.
4. The epoch-10 fusion: contradiction-majority drop + median of agreeing views, tau = p75 measured.
5. The cloud stage's voxel + SOR, octree, output/_epoch_1 (selectable).
"""
import gc, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

SERVER = Path("/workspace/stac-build/server")
DA3S = Path("/workspace/stac-build/vendor/depth-anything-3/da3_streaming")
sys.path.insert(0, str(SERVER)); sys.path.insert(0, str(DA3S))
os.chdir(DA3S)                                           # loop_utils imports, as run_da3.sh does
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
EPOCH = int(os.environ.get("STREAM_EPOCH", "1"))
POSE_SRC = os.environ.get("STREAM_POSES", "f5")       # f5 | omega (epoch 0: its poses AND its per-frame K)
RUN, DST, TMP = O / f"da3_stream_{POSE_SRC}", O / f"_epoch_{EPOCH}", O / f"_tx_da3_stream_{POSE_SRC}"
CONF_DROP_PCT = float(os.environ.get("STREAM_CONF_DROP_PCT", "10"))   # 0 = no confidence gate (USER 2026-09-30)
NB = (-12, -6, -3, 3, 6, 12)
TAU_Q = 75


def log(m):
    print(f"[stream {time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
F5 = np.loadtxt({"f5": O / "precision" / "f5_camera_poses.txt", "f5da3": O / "precision" / "f5_camera_poses.txt", "omega": O / "maplong_run" / "camera_poses.txt",
                 "r0": O / "precision" / "f5_r0_camera_poses.txt"}[POSE_SRC]).reshape(-1, 4, 4)
cam = json.loads((O / "camera.json").read_text())
fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
W0, H0 = int(cam["width"]), int(cam["height"])
K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
names = [f"{f:06d}.jpg" for f in frames]
if POSE_SRC == "omega":
    _k = np.loadtxt(O / "maplong_run" / "intrinsic.txt").reshape(-1, 4)       # fx fy cx cy per keyframe, native grid
    Kf = [np.array([[a, 0, c], [0, b, d], [0, 0, 1]], np.float32) for a, b, c, d in _k]
elif POSE_SRC in ("f5", "f5da3"):
    if POSE_SRC == "f5":                                 # F5 rung R1 camera (fx 392), kept aside when camera.json went R0
        _p = json.loads((O / "precision" / "camera_f5_r1.json").read_text())["params"]
    else:                                                # DA3's own camera: session median of its per-frame estimate
        import glob as _g
        _ks = []
        for _f in sorted(_g.glob(str(O / "da3_run" / "results_output" / "frame_*.npz"))):
            _z = np.load(_f); _k = np.asarray(_z["intrinsics"]).reshape(3, 3); _h, _w = _z["depth"].shape
            _ks.append([_k[0, 0] * W0 / _w, _k[1, 1] * H0 / _h, _k[0, 2] * W0 / _w, _k[1, 2] * H0 / _h])
        _p = list(np.median(np.array(_ks), 0))
    Kf = [np.array([[_p[0], 0, _p[2]], [0, _p[1], _p[3]], [0, 0, 1]], np.float32)] * len(names)
    print(f"[stream] camera fx {_p[0]:.1f} fy {_p[1]:.1f} cx {_p[2]:.1f} cy {_p[3]:.1f}", flush=True)
elif POSE_SRC == "r0":                                    # R0 = Omega's camera FIXED (refine.json camera.before)
    _p = json.loads((O / "precision" / "refine.json").read_text())["camera"]["before"]
    Kf = [np.array([[_p[0], 0, _p[2]], [0, _p[1], _p[3]], [0, 0, 1]], np.float32)] * len(names)
else:
    Kf = [K0] * len(names)
prior = {n: (np.linalg.inv(F5[i]).astype(np.float32), Kf[i]) for i, n in enumerate(names)}
for p in (RUN, TMP, DST):
    if p.exists(): shutil.rmtree(p)
RUN.mkdir(parents=True)
(RUN / "selected_frames.json").write_text(json.dumps({"selected_files": names}))

from config import cfg
from workers.map_worker import _build_da3_config
dc = _build_da3_config(cfg["reconstruction"])
dc["Model"].update(chunk_size=120, overlap=60, loop_enable=True, remove_sky=True,
                   save_depth_conf_result=True, save_debug_info=True, infer_gs=False)
import yaml
(RUN / "da3_streaming_config.yaml").write_text(yaml.dump(dc, default_flow_style=False))
m = dc["Model"]
log(f"{len(frames)} keyframes; chunk {m['chunk_size']}/{m['overlap']}, loop {m['loop_enable']}, align "
    f"{m['align_method']}/{m['align_lib']}, SALAD thr {dc['Loop']['SALAD']['similarity_threshold']}, "
    f"process_res {max(W0, H0)}")

from da3_streaming import warmup_numba
if m["align_lib"] == "numba":
    warmup_numba()
from stray_da3_streaming import StrayDA3Streaming


class F5Streaming(StrayDA3Streaming):
    """Pure DA3-streaming whose every chunk (and loop chunk) is conditioned on F5."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        _inf = self.model.inference

        def _native(*aa, **kk):
            kk.setdefault("process_res", max(W0, H0))
            return _inf(*aa, **kk)
        self.model.inference = _native

    def _stac_chunk_camera_priors(self, chunk_image_paths):
        e, k = zip(*(prior[os.path.basename(p)] for p in chunk_image_paths))
        return np.stack(e), np.stack(k)


ds = F5Streaming(str(S / "frames"), str(RUN), dc, stray_data=None)
ds.run(selected_frames_path=str(RUN / "selected_frames.json"))
res_dir = Path(ds.result_output_dir)
ds.close(); del ds; gc.collect()
import torch
torch.cuda.empty_cache()
log(f"streaming done in {(time.time() - t0) / 60:.1f} min")
lc = RUN / "loop_closures.txt"
if lc.exists():
    log(f"loop closures: {sum(1 for l in lc.read_text().splitlines() if l.strip())} line(s) in {lc.name}")

# ── per-frame final depth + pose ──
dep, conf, Ks, c2w = [], [], [], []
for i, f in enumerate(frames):
    z = np.load(res_dir / f"frame_{f}.npz")
    e = np.eye(4); e[:3, :4] = z["extrinsics"][:3, :4]
    cl = np.linalg.inv(e)
    s, R, T = float(z["s"]), np.asarray(z["R"]).reshape(3, 3), np.asarray(z["T"]).reshape(3)
    cf = np.eye(4); cf[:3, :3] = R @ cl[:3, :3]; cf[:3, 3] = s * R @ cl[:3, 3] + T
    dep.append(z["depth"].astype(np.float32)); conf.append(z["conf"].astype(np.float32))
    Ks.append(z["intrinsics"].astype(np.float64)); c2w.append(cf)
N = len(frames); H, W = dep[0].shape
C = np.stack([c[:3, 3] for c in c2w]); dC = np.linalg.norm(C - F5[:, :3, 3], axis=1)
ang = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", np.stack([c[:3, :3] for c in c2w]), F5[:, :3, :3]) - 1) / 2, -1, 1)))
log(f"depth {W}x{H}, K fx {Ks[0][0,0]:.1f} cx {Ks[0][0,2]:.1f}; final poses vs the input ({POSE_SRC}): |dt| median {np.median(dC)*100:.1f} cm "
    f"max {dC.max()*100:.1f} cm, rot median {np.median(ang):.2f} deg max {ang.max():.2f}")
allc = np.concatenate([c[(d > 0) & np.isfinite(d)] for c, d in zip(conf, dep)])
cthr = float(np.percentile(allc, CONF_DROP_PCT)) if CONF_DROP_PCT > 0 else -np.inf
val = [(d > 0) & np.isfinite(d) & (c >= cthr) for d, c in zip(dep, conf)]
log(f"confidence gate: bottom {CONF_DROP_PCT} % (conf < {cthr:.3f}) out")
w2c = [np.linalg.inv(c) for c in c2w]


def rays(i):
    rr, cc = np.nonzero(val[i]); K = Ks[i]
    r_c = np.stack([(cc - K[0, 2]) / K[0, 0], (rr - K[1, 2]) / K[1, 1], np.ones(len(rr))], 1)
    return rr, cc, r_c @ c2w[i][:3, :3].T


def project(i, j, rr, cc, rw):
    z = dep[i][rr, cc].astype(np.float64); Cw = c2w[i][:3, 3]
    a = w2c[j][2, :3] @ Cw + w2c[j][2, 3]; b = rw @ w2c[j][2, :3]
    X = Cw + z[:, None] * rw
    Xj = X @ w2c[j][:3, :3].T + w2c[j][:3, 3]; zj = Xj[:, 2]; Kj = Ks[j]
    ok = zj > 0.05; zs = np.where(ok, zj, 1)
    u = np.round(Kj[0, 0] * Xj[:, 0] / zs + Kj[0, 2]).astype(np.int64)
    v = np.round(Kj[1, 1] * Xj[:, 1] / zs + Kj[1, 2]).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc, vc = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    ok &= val[j][vc, uc]
    return zj, dep[j][vc, uc].astype(np.float64), ok, a, b


samp = []
for i in range(0, N, 6):
    rr, cc, rw = rays(i)
    for d in NB:
        j = i + d
        if 0 <= j < N:
            zj, dj, ok, _, _ = project(i, j, rr, cc, rw)
            samp.append(np.abs((zj[ok] - dj[ok]) / dj[ok])[::7])
samp = np.concatenate(samp); tau = float(np.percentile(samp, TAU_Q))
log(f"measured disagreement: median {np.median(samp)*100:.2f} %, p75 {tau*100:.2f} % -> tau")

(TMP / "chunks").mkdir(parents=True)
from precision.epoch0_cloud import _write_ply_xyzrgb
from PIL import Image
buf = {k: [] for k in ("xyz", "rgb", "fg", "pr", "pc", "cf")}; ci = [0]
n_in = n_drop = n_fused = 0; shift = []


def flush():
    if not buf["xyz"]:
        return
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{ci[0]:03d}.ply", np.concatenate(buf["xyz"]), np.concatenate(buf["rgb"]))
    np.savez(TMP / "chunks" / f"chunk_{ci[0]:03d}_origins.npz", frame_global=np.concatenate(buf["fg"]),
             pixel_row=np.concatenate(buf["pr"]), pixel_col=np.concatenate(buf["pc"]), confidence=np.concatenate(buf["cf"]))
    for k in buf: buf[k].clear()
    ci[0] += 1


for i in range(N):
    rr, cc, rw = rays(i)
    z = dep[i][rr, cc].astype(np.float64)
    agree = np.zeros(len(z), np.int32); contra = np.zeros(len(z), np.int32); cand = [z]
    for d in NB:
        j = i + d
        if not 0 <= j < N:
            continue
        zj, dj, ok, a, b = project(i, j, rr, cc, rw)
        e = (zj - dj) / np.where(ok, dj, 1)
        good_b = np.abs(b) > 1e-6
        ag = ok & (np.abs(e) <= tau) & good_b
        contra += ok & (e < -tau); agree += ag
        cand.append(np.where(ag, (dj - a) / np.where(good_b, b, 1), np.nan))
    keep = contra <= agree
    zf = np.nanmedian(np.vstack(cand), 0)
    fused = keep & (agree > 0)
    n_in += len(z); n_drop += int((~keep).sum()); n_fused += int(fused.sum())
    shift.append(np.abs(zf[fused] - z[fused])[::20])
    rr, cc, rw, zf = rr[keep], cc[keep], rw[keep], zf[keep]
    X = c2w[i][:3, 3] + zf[:, None] * rw
    img = np.asarray(Image.open(S / "frames" / f"{frames[i]:06d}.jpg").convert("RGB").resize((W, H)))
    buf["xyz"].append(X.astype(np.float32)); buf["rgb"].append(img[rr, cc])
    buf["fg"].append(np.full(len(rr), frames[i], np.int32))
    buf["pr"].append(np.round((rr + 0.5) * H0 / H - 0.5).astype(np.int16))
    buf["pc"].append(np.round((cc + 0.5) * W0 / W - 0.5).astype(np.int16))
    buf["cf"].append(conf[i][rr, cc].astype(np.float32))
    if (i + 1) % 24 == 0:
        flush()
flush()
shift = np.concatenate(shift)
log(f"{n_in:,} px after the conf gate: contradicted out {n_drop/n_in*100:.1f} %, fused {n_fused/n_in*100:.1f} %, "
    f"kept as measured {(n_in-n_drop-n_fused)/n_in*100:.1f} %; depth moved median {np.median(shift)*1000:.1f} mm "
    f"p90 {np.percentile(shift,90)*1000:.1f} mm")

from precision.epoch0_cloud import clean_cmd
cleaned = TMP / "cleaned_cloud.ply"
r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd=str(SERVER), capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "✅" in ln and ("→" in ln or "Merged" in ln): log(ln.strip())
if r.returncode or not cleaned.exists():
    log("cleaner failed " + r.stderr[-800:]); sys.exit(1)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
np.savetxt(DST / "camera_poses.txt", np.stack(c2w).reshape(N, -1))
(DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
    "note": f"DA3-streaming 120/60 conditioned on {POSE_SRC} poses + K (native res, loop on), bottom {CONF_DROP_PCT} % conf out, "
            f"multi-view fused (tau {tau*100:.2f} % = p{TAU_Q} measured); outside the pipeline 2026-09-30",
    "artifacts": [{"rel": r, "existed_before": True} for r in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
for _d in ("_tmp_results_aligned", "_tmp_results_unaligned", "_tmp_results_loop", "pcd"):
    shutil.rmtree(RUN / _d, ignore_errors=True)            # disk: keep only the per-frame depth
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch {EPOCH} in "
    f"{(time.time() - t0) / 60:.1f} min")
