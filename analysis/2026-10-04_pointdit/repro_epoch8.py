"""Reproduce pccr epoch 8 with the PRODUCT code (precision.depth_on_f5.run_depth_on_f5), fed F5's camera and
poses as the hand recipe read them, stopped right after the vote. Writes NOTHING into the session.
Reference (docs/pipeline_final.md, epoch 8): bend ±0, held-out 2.80 %, tau 1.93 %, kept 63.1 %,
contradicted 15.1 %, repaired 8.5 %, admitted 0.4 %, coverage 71.0 %."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"

def log(m):
    print(f"[repro {time.strftime('%H:%M:%S')}] {m}", flush=True)

def mem_gb():
    return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()) / 1e9

from precision import depth_sweep as DS, depth_on_f5 as DF
from precision.camera import load_camera_json, undistort_maps
from precision.config import load_precision_config

def fake_load_inputs(session_dir, pcfg):
    cam = load_camera_json(O / "precision" / "camera_f5_r1.json")
    kf, c2w = DS._read_poses(O / "precision" / "f5_camera_poses.txt", O / "camera_frames.txt")
    m1, m2, K = undistort_maps(cam)
    return DS.SweepInputs(cam=cam, K=K, wh=(cam.width, cam.height), maps=(m1, m2), kf=kf,
                          kf_w2c=np.linalg.inv(c2w), wit=[], wit_w2c=np.zeros((0, 4, 4)), tau_px=0.0,
                          heldout_rms_px=0.0, epochs={}, frames_dir=S / "frames",
                          records_dir=O / "omega_run" / "results_output")

class StopAfterVote(Exception):
    pass

_orig_vote = DF.edge_keeping_vote
def vote_and_stop(*a, **k):
    final, tau, tot = _orig_vote(*a, **k)
    nv = max(tot["valid"], 1); N = len(a[0]); H, W = next(iter(a[1].values())).shape
    res = {"tau_pct": tau * 100, "kept_pct": tot["kept"] / nv * 100, "contradicted_pct": tot["contradicted"] / nv * 100,
           "repaired_pct": tot["repaired"] / nv * 100, "admitted_pct": tot["admitted"] / nv * 100,
           "coverage_pct": tot["out"] / float(N * H * W) * 100, "mixed": tot["mixed"], "snapped": tot["snapped"],
           "valid": tot["valid"], "edge_pct": tot["edge"] / nv * 100}
    log("RESULT " + json.dumps(res))
    log(f"reference: tau 1.93, kept 63.1, contradicted 15.1, repaired 8.5, admitted 0.4, coverage 71.0; cgroup {mem_gb():.1f} GB")
    raise StopAfterVote()

DS.load_inputs = fake_load_inputs
DF.edge_keeping_vote = vote_and_stop
t0 = time.time()
log(f"start, cgroup {mem_gb():.1f} GB")
try:
    DF.run_depth_on_f5(S, load_precision_config(), log=log, progress=lambda p, m: log(f"{p:5.1f} % {m}; cgroup {mem_gb():.1f} GB"))
except StopAfterVote:
    log(f"stopped after the vote as intended, {(time.time() - t0) / 60:.1f} min, nothing written")
