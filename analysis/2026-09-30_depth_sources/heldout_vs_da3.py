"""Read-only: DA3-on-R0 depth (epoch 4, output/da3_stream_r0) vs F5-R0 held-out landmarks — same yardstick
as the epoch-5 model choice, no correction fitted."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"); O = S / "output"
from precision.config import load_precision_config
from precision.camera import undistort_solver
from precision.tracks import load_tracks_v2
from precision import refine as RF
pcfg = load_precision_config(); solver = undistort_solver(pcfg.camera)
params0 = list(json.loads((O / "precision" / "refine.json").read_text())["camera"]["before"]); K0 = RF.K_of(params0)
R0 = np.loadtxt(O / "precision" / "f5_r0_camera_poses.txt").reshape(-1, 4, 4); w2c = np.linalg.inv(R0)
frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
tr = load_tracks_v2(S)
split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8); m = split == 1
g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
X = RF.triangulate_tracks(g, w2c, params0, solver, pcfg.refine.min_tri_deg)
err = []
per = {i: [] for i in range(len(frames))}
for t, lst in g.items():
    if t in X:
        for i, p in lst: per[i].append((p[0], p[1], w2c[i][2, :3] @ X[t] + w2c[i][2, 3]))
for i, f in enumerate(frames):
    z = np.load(O / "da3_stream_r0" / "results_output" / f"frame_{f}.npz")
    D, K = z["depth"], z["intrinsics"]; H, W = D.shape
    a = np.array(per[i]).reshape(-1, 3)
    if not len(a): continue
    xn = (a[:, 0] - K0[0, 2]) / K0[0, 0]; yn = (a[:, 1] - K0[1, 2]) / K0[1, 1]     # native pixel -> ray -> DA3 grid
    u = np.round(xn * K[0, 0] + K[0, 2]).astype(int); v = np.round(yn * K[1, 1] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (a[:, 2] > 0.05)
    d = D[v[ok], u[ok]]; ok2 = d > 0
    err.append(np.abs(d[ok2] - a[ok, 2][ok2]) / a[ok, 2][ok2])
e = np.concatenate(err)
print(f"DA3 on R0 (epoch 4), no correction: held-out |z - z_F5| / z_F5 median {np.median(e)*100:.2f} %, p75 {np.percentile(e,75)*100:.2f} %, n {len(e):,}")
