"""pccr 2026-09-30 — USER: "sacamos todas las épocas, dejamos la 4 como época 1, sin el filtrado de confianza
del 10 %, luego de aplicarle las correcciones del pipeline completas".
 A. epoch 0 live; epochs 1, 4, 5, 7 and da3_stream_f5 deleted
 B. camera.json -> F5 rung R0 (Omega's camera, fx 364) — the cloud's own camera; the R1 file kept as backup
 C. epoch 4's DA3 maps (output/da3_stream_r0/results_output, conditioned on R0) re-fused WITHOUT a confidence
    gate (contradiction vote + median of agreeing views, tau = measured p75), voxel + SOR -> new-cloud epoch 1, live
 D. SAM3 masks projected on it (segmentation.pipeline.map_segmentation_to_cloud; the stale result set aside)
 E. the certify stage (reconstruction.certify.run.certify_session): closures, depth per chunk, floor per
    chunk, mask filter, chunk check -> a corrected epoch
 F. compacted: epoch 0 + the corrected one as epoch 1
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(Path(__file__).resolve().parent))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
RUN, TMP = O / "da3_stream_r0", O / "_tx_final"
NB = (-12, -6, -3, 3, 6, 12); TAU_Q = 75


def log(m, level="info"):
    print(f"[final {time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
from config import cfg
from correction.run import run_select


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


log("A. epoch 0 live, the other epochs deleted")
if live() != 0:
    run_select(O, 0, "auto", log=log)
assert live() == 0
for d in [O / f"_epoch_{e}" for e in (1, 4, 5, 7)] + [O / "da3_stream_f5"]:
    shutil.rmtree(d, ignore_errors=True)
log(f"   stored epochs now: {sorted(p.name for p in O.glob('_epoch_*'))}")

log("B. camera.json -> F5 rung R0 (Omega's camera)")
camp = O / "camera.json"; cam = json.loads(camp.read_text())
bk = O / "precision" / "camera_f5_r1.json"
if not bk.exists(): shutil.copy(camp, bk)
ref = json.loads((O / "precision" / "refine.json").read_text())
cam["params"] = list(ref["camera"]["before"]); cam["source"] = "refine_R0"; cam["report"] = {"rung": "R0", "blocks": 1}
camp.write_text(json.dumps(cam, indent=1))
log(f"   params {cam['params'][:4]} (R1 kept in precision/camera_f5_r1.json)")

log("C. epoch 4's DA3 maps re-fused without a confidence gate")
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
W0, H0 = int(cam["width"]), int(cam["height"])
res_dir = RUN / "results_output"
F5 = np.loadtxt(O / "precision" / "f5_r0_camera_poses.txt").reshape(-1, 4, 4)
POSE_SRC = "r0"
shutil.rmtree(TMP, ignore_errors=True)
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
cthr = -np.inf                                  # USER 2026-09-30: no confidence gate
val = [(d > 0) & np.isfinite(d) & (c >= cthr) for d, c in zip(dep, conf)]
log("confidence gate: none (every pixel with a depth enters; the contradiction vote removes flyers)")
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
log(f"{n_in:,} px (no conf gate): contradicted out {n_drop/n_in*100:.1f} %, fused {n_fused/n_in*100:.1f} %, "
    f"kept as measured {(n_in-n_drop-n_fused)/n_in*100:.1f} %; depth moved median {np.median(shift)*1000:.1f} mm "
    f"p90 {np.percentile(shift,90)*1000:.1f} mm")

from precision.epoch0_cloud import clean_cmd
cleaned = TMP / "cleaned_cloud.ply"
r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd=str(SERVER), capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "✅" in ln and ("→" in ln or "Merged" in ln): log(ln.strip())
if r.returncode or not cleaned.exists():
    log("❌ cleaner failed " + r.stderr[-800:]); sys.exit(1)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("❌ octree failed"); sys.exit(1)
DST = O / "_epoch_1"
shutil.rmtree(DST, ignore_errors=True); DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
np.savetxt(DST / "camera_poses.txt", np.stack(c2w).reshape(N, -1))
(DST / "_manifest.json").write_text(json.dumps({"epoch": 1, "epoch_from": 1, "epoch_to": 1, "kind": "new_cloud",
    "note": "DA3-streaming 120/60 conditioned on F5-R0, native res, NO confidence gate, multi-view fused",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
run_select(O, 1, "auto", log=log)
assert live() == 1
log(f"   epoch 1 live: {json.loads((O / 'potree' / 'metadata.json').read_text())['points']:,} pts ({(time.time()-t0)/60:.1f} min)")

log("D. SAM3 masks projected on the new cloud")
seg = O / "segmentation_result.json"
if seg.exists():
    seg.rename(O / "precision" / "segmentation_result.stale.json")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
if d.get("error"):
    log(f"❌ projection: {d['error']}"); sys.exit(1)
log(f"   {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time()-t0)/60:.1f} min)")
(O / "precision" / "segmentation_result.stale.json").unlink(missing_ok=True)

log("E. certify stage: the correction")
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
log(f"   {acta.get('stop_reason')} — epoch {acta.get('epoch_initial')} -> {acta.get('epoch_final')} ({(time.time()-t0)/60:.1f} min)")

log("F. compact to epoch 0 + the corrected one as epoch 1")
import rebuild_nowitness as RB
RB.log = log; RB.O = O
RB.compact()
log(f"DONE in {(time.time()-t0)/60:.1f} min — live epoch {live()}, stored {sorted(p.name for p in O.glob('_epoch_*'))}")
