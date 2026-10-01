"""F5 with Omega's camera FIXED (rung R0 only), from epoch 0's poses — the same tracks, split,
gauge sigma and solver settings run_refine uses (precision/refine.py). Nothing is applied:
the poses go to output/precision/f5_r0_camera_poses.txt. USER 2026-09-30: Omega with F5's poses
without breaking its geometry (R1's focal 364 -> 392 bought 0.004 px held-out and bent every ray)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); out = S / "output"
t0 = time.time()
from precision.config import load_precision_config
from precision.camera import load_camera_json, undistort_solver
from precision.tracks import load_tracks_v2
from precision import refine as RF
pcfg = load_precision_config(); cfg = pcfg.refine
cam = load_camera_json(out / "camera.json")                       # for width/height only
ref = json.loads((out / "precision" / "refine.json").read_text())
params0 = list(ref["camera"]["before"])                           # Omega's camera, as F5 received it
solver = undistort_solver(pcfg.camera)
poses = np.loadtxt(out / "maplong_run" / "camera_poses.txt").reshape(-1, 4, 4)   # epoch 0 (== its live poses, verified)
kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
w2c0 = np.linalg.inv(poses)
tr = load_tracks_v2(S)
split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
sigma_rel, inst = RF._gauge_sigma(out)
print(f"[R0] {len(kf)} kf, {len(tr['track_query_id']):,} tracks, camera {params0[:4]}, gauge σ {sigma_rel:.4f} ({inst})", flush=True)
fit_g = RF.group_tracks(tr["obs_track"][split == 0], tr["obs_frame"][split == 0], tr["obs_uv_native"][split == 0], kf)
held_g = RF.group_tracks(tr["obs_track"][split == 1], tr["obs_frame"][split == 1], tr["obs_uv_native"][split == 1], kf)
X0 = RF.triangulate_tracks(fit_g, w2c0, params0, solver, cfg.min_tri_deg)
sig = RF.prior_sigmas(np.array([-T[:3, :3].T @ T[:3, 3] for T in w2c0]), sigma_rel)
base = RF.RungResult("init", w2c0, [list(params0)], np.zeros(len(w2c0), int), float("nan"), 0, "")
h0 = RF._heldout_of(base, held_g, solver, cfg)
r = RF.run_rung("R0", w2c0, params0, (cam.width, cam.height), fit_g, X0, sig, cfg)
h1 = RF._heldout_of(r, held_g, solver, cfg)
v0, v1 = np.array(list(h0.values())), np.array(list(h1.values()))
c2w = np.linalg.inv(r.w2c)
np.savetxt(out / "precision" / "f5_r0_camera_poses.txt", c2w.reshape(len(c2w), -1))
f5 = np.loadtxt(out / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
d0 = np.linalg.norm(c2w[:, :3, 3] - poses[:, :3, 3], axis=1)
print(f"[R0] held-out median {np.median(v0):.4f} px (epoch 0) -> {np.median(v1):.4f} px (R0); camera kept {r.params_by_block[0][:4]}", flush=True)
print(f"[R0] cameras moved from epoch 0: median {np.median(d0)*100:.1f} cm, max {d0.max()*100:.1f} cm; "
      f"{r.termination}; {(time.time()-t0)/60:.1f} min -> precision/f5_r0_camera_poses.txt", flush=True)
