"""OpenMVS trial (USER 2026-10-17: "probar el pipeline completo en su mejor versión de OpenMVS para darle la
oportunidad de competir contra GenRecon"): the session's CERTIFIED cloud, keyframes and poses as a COLMAP text
model for InterfaceCOLMAP — RAW frame, PINHOLE camera (the session's lens terms are under a pixel), and every
point with the keyframes that SEE it as its track (visibility by the cloud's own z-buffer per keyframe, the
measured keyframe first), because OpenMVS picks each image's neighbour views from the points they share.
    python export_colmap.py <session src_default> <work dir> [max_points 4000000] [max_views 8]
"""
import sys, json, shutil, time, numpy as np, torch
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from scipy.spatial.transform import Rotation
from correction.session import load_session
from segmentation.perfect_object import _read_ply_fields

SES = Path(sys.argv[1]); WORK = Path(sys.argv[2])
N_MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
MAXV = int(sys.argv[4]) if len(sys.argv) > 4 else 8
t0 = time.time(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = SES / "output"; s = load_session(out)
cam = json.loads((out / "camera.json").read_text())
W, H = int(cam["width"]), int(cam["height"]); fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]
col = WORK / "colmap"; img = WORK / "images"
if WORK.exists():
    shutil.rmtree(WORK)
col.mkdir(parents=True); img.mkdir()
(col / "cameras.txt").write_text("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
frames = [int(f) for f in s.frames]; poses = np.asarray(s.poses, np.float64)
keep = [i for i, f in enumerate(frames) if (SES / "frames" / f"{f:06d}.jpg").exists()]
frames = [frames[i] for i in keep]; poses = poses[keep]
kf_index = {f: i + 1 for i, f in enumerate(frames)}
lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
for i, (f, c2w) in enumerate(zip(frames, poses)):
    src = SES / "frames" / f"{f:06d}.jpg"
    (img / src.name).symlink_to(src)
    w2c = np.linalg.inv(c2w); q = Rotation.from_matrix(w2c[:3, :3]).as_quat(); t = w2c[:3, 3]
    lines.append(f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {src.name}\n\n")
(col / "images.txt").write_text("".join(lines))
f_ = _read_ply_fields(out / "cleaned_cloud.ply")
xyz = np.column_stack([f_["x"], f_["y"], f_["z"]]).astype(np.float32)
rgb = np.column_stack([f_["red"], f_["green"], f_["blue"]]).astype(int)
fg = np.asarray(f_["frame_global"]).astype(int)
# the z-buffer of the WHOLE cloud per keyframe (min-filtered), the measured truth of what each view sees
P = torch.from_numpy(xyz).to(dev); w2c_t = torch.from_numpy(np.linalg.inv(poses).astype(np.float32)).to(dev)
FAR = 1.0e6; zbuf = torch.full((len(frames), H * W), FAR, device=dev)
for k in range(len(frames)):
    pc = P @ w2c_t[k, :3, :3].T + w2c_t[k, :3, 3]; z = pc[:, 2]; fr = z > 0.2
    u = (fx * pc[fr, 0] / z[fr] + cx).long(); v = (fy * pc[fr, 1] / z[fr] + cy).long()
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    zbuf[k].scatter_reduce_(0, (v[inb] * W + u[inb]), z[fr][inb], reduce="amin")
zbuf = -torch.nn.functional.max_pool2d(-zbuf.view(len(frames), 1, H, W), kernel_size=5, stride=1, padding=2).view(len(frames), H, W)
del P
sel = np.linspace(0, len(xyz) - 1, min(len(xyz), N_MAX)).astype(np.int64)         # a fixed stride, never a draw
Ps = torch.from_numpy(xyz[sel]).to(dev)
origin = np.array([kf_index.get(int(v), 0) for v in fg[sel]])
tracks = [[] for _ in range(len(sel))]
n_vis = np.zeros(len(sel), int)
B = 500_000
for b0 in range(0, len(sel), B):
    Pb = Ps[b0:b0 + B]
    vis = torch.zeros((len(frames), len(Pb)), dtype=torch.bool, device=dev)
    for k in range(len(frames)):
        pc = Pb @ w2c_t[k, :3, :3].T + w2c_t[k, :3, 3]; z = pc[:, 2]; fr = z > 0.2
        zs = torch.where(fr, z, torch.ones_like(z))
        u = (fx * pc[:, 0] / zs + cx).long(); v = (fy * pc[:, 1] / zs + cy).long()
        inb = fr & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zb = zbuf[k][v.clamp(0, H - 1), u.clamp(0, W - 1)]
        vis[k] = inb & (z <= zb + (0.03 + 0.02 * z))
    vis_np = vis.cpu().numpy()
    for j in range(len(Pb)):
        ks = np.flatnonzero(vis_np[:, j]) + 1
        o = origin[b0 + j]
        order = ([o] if o else []) + [int(k) for k in ks if k != o]
        tracks[b0 + j] = order[:MAXV]
        n_vis[b0 + j] = len(ks)
with (col / "points3D.txt").open("w") as fh:
    fh.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for j, k in enumerate(sel):
        tr = "".join(f" {v} 0" for v in tracks[j])
        fh.write(f"{j + 1} {xyz[k, 0]:.5f} {xyz[k, 1]:.5f} {xyz[k, 2]:.5f} {rgb[k, 0]} {rgb[k, 1]} {rgb[k, 2]} 0.0{tr}\n")
rec = {"session": str(SES), "frame": "raw", "camera": "PINHOLE", "images": len(frames), "points": int(len(sel)),
       "cloud_points": int(len(xyz)), "max_views": MAXV, "views_per_point_p10_p50_p90": [int(v) for v in np.percentile(n_vis, [10, 50, 90])],
       "points_without_view": int((n_vis == 0).sum()), "seconds": round(time.time() - t0)}
json.dump(rec, open(WORK / "export.json", "w"), indent=1)
print(f"{len(frames)} images, {len(sel):,} points of {len(xyz):,}; views per point p10/p50/p90 {rec['views_per_point_p10_p50_p90']}, "
      f"{rec['points_without_view']:,} with none; {time.time() - t0:.0f} s → {col}")
