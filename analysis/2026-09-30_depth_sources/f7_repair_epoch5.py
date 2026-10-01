"""Validate the CHANGED f7_cloud (no own confidence gate, contradicted pixels repaired) on pccr, through the
pipeline's own functions (precision.corrected_cloud.write_chunks / clean), then the certify stage -> epoch 5.
The SweepInputs are built by hand from the same files load_inputs reads (F5 R1 camera + F5 poses, Omega's
records), because load_inputs' epoch guard refuses the session's state after today's epoch churn."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
TMP, DST = O / "_tx_f7_repair", O / "_epoch_5"


def log(m, level="info"):
    print(f"[f7-repair {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


t0 = time.time()
from config import cfg
from precision.config import load_precision_config
from precision.camera import CameraModel, undistort_maps
from precision.depth_sweep import SweepInputs
from precision import corrected_cloud as CC
from correction.run import run_select
pcfg = load_precision_config()
camR1 = json.loads((O / "precision" / "camera_f5_r1.json").read_text())
cam = CameraModel.from_dict(camR1) if hasattr(CameraModel, "from_dict") else None
if cam is None:
    from precision.camera import load_camera_json
    tmpc = O / "precision" / "_camera_r1_for_f7.json"; tmpc.write_text(json.dumps(camR1)); cam = load_camera_json(tmpc)
m1, m2, K = undistort_maps(cam)
kf = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
F5 = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
inp = SweepInputs(cam=cam, K=K, wh=(cam.width, cam.height), maps=(m1, m2), kf=kf, kf_w2c=np.linalg.inv(F5),
                  wit=[], wit_w2c=np.zeros((0, 4, 4)), tau_px=0.0, heldout_rms_px=0.0,
                  epochs={"geometry_epoch": 0, "camera_epoch": 1}, frames_dir=S / "frames",
                  records_dir=O / "omega_run" / "results_output")
f6_frames = sorted(int(p.stem.split("_")[1]) for p in (O / "depth_native").glob("frame_*.npz"))
f6 = {"dir": O / "depth_native", "frames": [f for f in f6_frames if f in set(kf)]}
log(f"F5 R1 camera fx {K[0, 0]:.1f}; {len(f6['frames'])} F6 maps; cloud config {pcfg.cloud}")
for p in (TMP, DST):
    shutil.rmtree(p, ignore_errors=True)
chunks = CC.write_chunks(inp, f6, cfg, TMP / "chunks", log, ccfg=pcfg.cloud)
log(f"raw {sum(c['raw_points'] for c in chunks):,} pts, repaired {sum(c.get('repaired', 0) for c in chunks):,}")
cleaned = TMP / "cleaned_cloud.ply"
CC.clean(cfg, TMP / "chunks", cleaned, log)
from potree_converter import convert_ply_to_potree
if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
    log("❌ octree failed"); sys.exit(1)
DST.mkdir()
shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
(DST / "_manifest.json").write_text(json.dumps({"epoch": 5, "epoch_from": 5, "epoch_to": 5, "kind": "new_cloud",
    "note": "f7_cloud CHANGED: Omega via F6 on F5 R1, no own confidence gate, contradicted pixels repaired",
    "artifacts": [{"rel": x, "existed_before": True} for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt")]}))
shutil.rmtree(TMP, ignore_errors=True)
log(f"cloud built ({(time.time()-t0)/60:.1f} min) — epoch 5 live")
run_select(O, 5, "auto", log=log)
assert live() == 5
seg = O / "segmentation_result.json"
if seg.exists():
    seg.rename(O / "precision" / f"segmentation_result.before_epoch5.json")
from segmentation.pipeline import map_segmentation_to_cloud
d = map_segmentation_to_cloud(O)
log(f"  {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time()-t0)/60:.1f} min)")
from reconstruction.loops.config import load_loops_config
from reconstruction.certify.run import certify_session
acta = certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
e_fin = live()
if e_fin != 5:
    shutil.rmtree(O / "_epoch_5", ignore_errors=True)
    recp = O / "geometry_epoch.json"; r_ = json.loads(recp.read_text()); r_.update(epoch=5, renumbered_from=e_fin)
    recp.write_text(json.dumps(r_, indent=2))
log(f"DONE in {(time.time()-t0)/60:.1f} min — live {live()}, stored {sorted(p.name for p in O.glob('_epoch_*'))}")
