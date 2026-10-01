"""Omega with F5 + certification (with floor correction) -> epoch 4. USER 2026-09-30:
"generame omega con f5, y la certificación y corrección de piso, como época 4".

The pipeline's own f7_cloud recipe (precision/corrected_cloud.py, source omega_corrected), run outside
the pipeline because its epoch/camera guards refuse the session's current state:
  point = c2w_F5 · (z_F6 · K_F5⁻¹ [u v 1]),  z_F6 = F6's depth_native (tier 0 sweep / tier 1 Omega × s_k)
  confidence gate per Omega chunk (conf_percentile + conf_min_norm on Omega's own conf), voxel + SOR,
  NO witness filter (USER 2026-09-30 morning). F5 = rung R1 (fx 392) + its poses.
Then: epoch 4 live, camera.json = F5 R1 (the cloud's camera), SAM3 masks projected (with the co-visible
split), the certify stage (closures, depth per chunk, floor per chunk, mask filter, chunk check); the
corrected epoch is renumbered 4 and the uncorrected one removed.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
TMP, DST = O / "_tx_omega_f5", O / "_epoch_4"


def log(m, level="info"):
    print(f"[omega-f5 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


t0 = time.time()
from config import cfg
from precision.epoch0_cloud import _write_ply_xyzrgb, conf_threshold, clean_cmd
from correction.run import run_select
from PIL import Image
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
P = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
camR1 = json.loads((O / "precision" / "camera_f5_r1.json").read_text())
fx, fy, cx, cy = camR1["params"][:4]
simple = cfg["reconstruction"]["simple"]
pct, floor = float(simple["conf_percentile"]), float(simple["conf_min_norm"] or 0.0)
log(f"{len(frames)} keyframes; F5 R1 camera fx {fx:.1f} fy {fy:.1f}; gate p{pct:g} + min-max {floor:g} per Omega chunk")

rec = {f: np.load(O / "omega_run" / "results_output" / f"frame_{f}.npz") for f in frames}
by_chunk = {}
for i, f in enumerate(frames):
    by_chunk.setdefault(int(rec[f]["chunk"]), []).append(i)
for p in (TMP, DST):
    shutil.rmtree(p, ignore_errors=True)
(TMP / "chunks").mkdir(parents=True)
H, W = 832, 464
vv, uu = np.mgrid[0:H, 0:W]
n_tot = 0
for c, members in sorted(by_chunk.items()):
    thr = conf_threshold(np.concatenate([np.asarray(rec[frames[i]]["conf"], np.float32).ravel() for i in members]), pct, floor)
    xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
    for i in members:
        f = frames[i]
        d = np.load(O / "depth_native" / f"frame_{f}.npz")
        z = d["depth"].astype(np.float64); conf = np.asarray(rec[f]["conf"], np.float32)
        m = (z > 0) & np.isfinite(z) & (conf >= thr)
        r, cc = vv[m], uu[m]; zz = z[m]
        Xc = np.stack([(cc - cx) / fx * zz, (r - cy) / fy * zz, zz], 1)
        X = Xc @ P[i][:3, :3].T + P[i][:3, 3]
        img = np.asarray(Image.open(S / "frames" / f"{f:06d}.jpg").convert("RGB").resize((W, H)))
        xyz_l.append(X.astype(np.float32)); rgb_l.append(img[r, cc]); fg_l.append(np.full(len(r), f, np.int32))
        pr_l.append(r.astype(np.int16)); pc_l.append(cc.astype(np.int16)); cf_l.append(conf[m])
    xyz = np.concatenate(xyz_l); n_tot += len(xyz)
    _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{c:03d}.ply", xyz, np.concatenate(rgb_l))
    np.savez(TMP / "chunks" / f"chunk_{c:03d}_origins.npz", frame_global=np.concatenate(fg_l),
             pixel_row=np.concatenate(pr_l), pixel_col=np.concatenate(pc_l), confidence=np.concatenate(cf_l))
    log(f"  Omega chunk {c}: {len(members)} kf, gate {thr:.3f}, {len(xyz):,} pts")
log(f"{n_tot:,} raw points")
cleaned = TMP / "cleaned_cloud.ply"
r = subprocess.run(clean_cmd(cfg, TMP / "chunks", cleaned), cwd=str(SERVER), capture_output=True, text=True)
for ln in r.stdout.splitlines():
    if "✅" in ln and ("→" in ln or "Merged" in ln): log(ln.strip())
if r.returncode or not cleaned.exists():
    log("❌ cleaner failed " + r.stderr[-800:]); sys.exit(1)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("❌ octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": 4, "epoch_from": 4, "epoch_to": 4, "kind": "new_cloud",
    "note": "Omega depth (F6 tiers) re-projected with F5 R1 camera + poses, no witness filter",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"cloud built ({(time.time()-t0)/60:.1f} min) — making epoch 4 live")
run_select(O, 4, "auto", log=log)
assert live() == 4

camp = O / "camera.json"; cam = json.loads(camp.read_text())
if not (O / "precision" / "camera_r0.json").exists():
    shutil.copy(camp, O / "precision" / "camera_r0.json")
cam["params"] = camR1["params"]; cam["source"] = camR1.get("source", "refine"); cam["report"] = camR1.get("report")
camp.write_text(json.dumps(cam, indent=1))
log("camera.json = F5 R1 (the R0 one kept in precision/camera_r0.json)")

log("SAM3 masks projected on epoch 4 (co-visible split on)")
seg = O / "segmentation_result.json"
if seg.exists():
    seg.rename(O / "precision" / "segmentation_result.epoch1.json")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
if d.get("error"):
    log(f"❌ projection: {d['error']}"); sys.exit(1)
log(f"  {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time()-t0)/60:.1f} min)")

log("certify stage")
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
e_fin = live()
log(f"  {acta.get('stop_reason')} — epoch 4 -> {e_fin} ({(time.time()-t0)/60:.1f} min)")

if e_fin != 4:
    shutil.rmtree(O / "_epoch_4", ignore_errors=True)            # the uncorrected cloud
    recp = O / "geometry_epoch.json"; rec_ = json.loads(recp.read_text())
    rec_.update(epoch=4, renumbered_from=e_fin); recp.write_text(json.dumps(rec_, indent=2))
    log(f"  corrected epoch {e_fin} renumbered 4; the uncorrected cloud removed")
log(f"DONE in {(time.time()-t0)/60:.1f} min — live epoch {live()}, stored {sorted(p.name for p in O.glob('_epoch_*'))}")
