"""Multi-view fusion of the F5-conditioned DA3 depth (output/da3_posed_depth) -> epoch 10.
USER 2026-09-30: flyers out, floor flattened, without the aggressive confirmation filter.

Per pixel of keyframe i, every neighbour j in i+NB that sees it votes (occlusion-aware):
  agree      |z_ij - d_j| <= tau * d_j
  contradict z_ij <  d_j * (1 - tau)   j saw PAST the point: it floats in space j saw empty
  no vote    z_ij >  d_j * (1 + tau)   something sits in front of it in j (occlusion)
DROP when contradict > agree.  DEPTH = median of {own, each agreeing j's surface along i's ray}
(exact: the point C_i + z r has depth a + b z in j, so j's surface is at z = (d_j - a) / b).
tau = p75 of the session's own |relative disagreement| between neighbours (measured here).
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
D, DST, TMP = O / "da3_posed_depth", O / "_epoch_10", O / "_tx_da3_fuse"
NB = (-12, -6, -3, 3, 6, 12)
TAU_Q = 75                                    # tau = this percentile of the measured disagreement


def log(m):
    print(f"[fuse {time.strftime('%H:%M:%S')}] {m}", flush=True)


t0 = time.time()
kf = [np.load(p) for p in sorted(D.glob("kf_*.npz"))]
N = len(kf)
dep = [k["depth"] for k in kf]; val = [k["valid"] for k in kf]; Ks = [k["K"] for k in kf]
c2w = [k["c2w"] for k in kf]; w2c = [np.linalg.inv(c) for c in c2w]; frames = [int(k["frame"]) for k in kf]
conf = [k["conf"] for k in kf]
H, W = dep[0].shape
log(f"{N} keyframes {W}x{H}, neighbours {NB}")


def rays(i):
    rr, cc = np.nonzero(val[i]); K = Ks[i]
    r_c = np.stack([(cc - K[0, 2]) / K[0, 0], (rr - K[1, 2]) / K[1, 1], np.ones(len(rr))], 1)
    return rr, cc, r_c @ c2w[i][:3, :3].T               # world direction per unit depth


def project(i, j, rr, cc, rw):
    z = dep[i][rr, cc].astype(np.float64)
    C = c2w[i][:3, 3]
    a = w2c[j][2, :3] @ C + w2c[j][2, 3]; b = rw @ w2c[j][2, :3]      # depth in j = a + b z
    X = C + z[:, None] * rw
    Xj = X @ w2c[j][:3, :3].T + w2c[j][:3, 3]; zj = Xj[:, 2]; Kj = Ks[j]
    ok = zj > 0.05; zs = np.where(ok, zj, 1)
    u = np.round(Kj[0, 0] * Xj[:, 0] / zs + Kj[0, 2]).astype(np.int64)
    v = np.round(Kj[1, 1] * Xj[:, 1] / zs + Kj[1, 2]).astype(np.int64)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uc, vc = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    ok &= val[j][vc, uc]
    dj = dep[j][vc, uc].astype(np.float64)
    return zj, dj, ok, a, b


# tau from the session's own disagreement (sample: every 6th keyframe)
samp = []
for i in range(0, N, 6):
    rr, cc, rw = rays(i)
    for d in NB:
        j = i + d
        if 0 <= j < N:
            zj, dj, ok, _, _ = project(i, j, rr, cc, rw)
            samp.append(np.abs((zj[ok] - dj[ok]) / dj[ok])[::7])
samp = np.concatenate(samp)
tau = float(np.percentile(samp, TAU_Q))
log(f"measured disagreement: median {np.median(samp)*100:.2f} %, p75 {tau*100:.2f} % -> tau {tau*100:.2f} %")

for p in (TMP, DST):
    if p.exists(): shutil.rmtree(p)
(TMP / "chunks").mkdir(parents=True)
from precision.epoch0_cloud import _write_ply_xyzrgb
from PIL import Image
n_in = n_drop = n_fused = 0; shift = []
buf = {k: [] for k in ("xyz", "rgb", "fg", "pr", "pc", "cf")}; ci = 0
H0, W0 = 832, 464


def flush():
    global ci
    if not buf["xyz"]:
        return
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{ci:03d}.ply", np.concatenate(buf["xyz"]), np.concatenate(buf["rgb"]))
    np.savez(TMP / "chunks" / f"chunk_{ci:03d}_origins.npz", frame_global=np.concatenate(buf["fg"]),
             pixel_row=np.concatenate(buf["pr"]), pixel_col=np.concatenate(buf["pc"]), confidence=np.concatenate(buf["cf"]))
    for k in buf: buf[k].clear()
    ci += 1


for i in range(N):
    rr, cc, rw = rays(i)
    z = dep[i][rr, cc].astype(np.float64)
    agree = np.zeros(len(z), np.int32); contra = np.zeros(len(z), np.int32)
    cand = [z]
    for d in NB:
        j = i + d
        if not 0 <= j < N:
            continue
        zj, dj, ok, a, b = project(i, j, rr, cc, rw)
        e = (zj - dj) / np.where(ok, dj, 1)
        ag = ok & (np.abs(e) <= tau) & (np.abs(b) > 1e-6)
        contra += ok & (e < -tau); agree += ag
        cand.append(np.where(ag, (dj - a) / np.where(np.abs(b) > 1e-6, b, 1), np.nan))
    keep = contra <= agree
    zf = np.nanmedian(np.vstack(cand), 0)
    n_in += len(z); n_drop += int((~keep).sum()); n_fused += int((keep & (agree > 0)).sum())
    shift.append(np.abs(zf[keep & (agree > 0)] - z[keep & (agree > 0)])[::20])
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
        log(f"  {i + 1}/{N} keyframes")
flush()
shift = np.concatenate(shift)
log(f"{n_in:,} px: dropped as contradicted {n_drop / n_in * 100:.1f} %, fused with >=1 agreeing view "
    f"{n_fused / n_in * 100:.1f} %, kept as measured {(n_in - n_drop - n_fused) / n_in * 100:.1f} %; "
    f"fusion moved depth median {np.median(shift) * 1000:.1f} mm p90 {np.percentile(shift, 90) * 1000:.1f} mm")

from config import cfg
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
shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": 10, "epoch_from": 10, "epoch_to": 10, "kind": "new_cloud",
    "note": f"epoch 8's DA3 depth, multi-view fused: contradiction-majority drop + median of agreeing views, tau {tau*100:.2f} % (p{TAU_Q} measured), outside the pipeline 2026-09-30",
    "artifacts": [{"rel": r, "existed_before": True} for r in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"DONE {json.loads((DST / 'potree' / 'metadata.json').read_text())['points']:,} pts -> epoch 10 in {(time.time() - t0) / 60:.1f} min")
