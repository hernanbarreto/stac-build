"""Our session → GenRecon's iPhone/COLMAP-text input: the keyframes, the CERTIFIED poses (epoch 4,
c2w, OpenCV camera) as COLMAP w2c, the camera.json OPENCV intrinsics, and the published cloud as
points3D.txt — the world turned Y-up → Z-up (GenRecon's chunker reads the height along z)."""
import sys, json, shutil, numpy as np
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from scipy.spatial.transform import Rotation
from correction.session import load_session
from segmentation.perfect_object import _read_ply_fields

SES = Path(sys.argv[1]); WORK = Path(sys.argv[2]); N_PTS = int(sys.argv[3]) if len(sys.argv) > 3 else 600_000
out = SES / "output"
R_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])     # (x, y, z) → (x, -z, y): y-up → z-up
s = load_session(out)
cam = json.loads((out / "camera.json").read_text())
W, H = int(cam["width"]), int(cam["height"]); fx, fy, cx, cy, k1, k2, p1, p2 = [float(v) for v in cam["params"][:8]]
col = WORK / "colmap_mast3r"; rgb = WORK / "rgb"
if WORK.exists(): shutil.rmtree(WORK)
col.mkdir(parents=True); rgb.mkdir()
(col / "cameras.txt").write_text("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                                 f"1 OPENCV {W} {H} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")
lines = ["# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
R4 = np.eye(4); R4[:3, :3] = R_UP
n = 0
for i, (frame, c2w) in enumerate(zip(s.frames, np.asarray(s.poses, np.float64))):
    src = SES / "frames" / f"{int(frame):06d}.jpg"
    if not src.exists():
        continue
    name = src.name
    (rgb / name).symlink_to(src)
    c2w_z = R4 @ c2w
    w2c = np.linalg.inv(c2w_z)
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()           # x, y, z, w
    t = w2c[:3, 3]
    lines.append(f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n\n")
    n += 1
(col / "images.txt").write_text("".join(lines))
f = _read_ply_fields(out / "cleaned_cloud.ply")
xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
rgbv = np.column_stack([f.get(k, np.full(len(xyz), 128)) for k in ("red", "green", "blue")]) if "red" in f else np.full((len(xyz), 3), 128)
sel = np.linspace(0, len(xyz) - 1, min(len(xyz), N_PTS)).astype(np.int64)        # a fixed stride, never a draw
P = xyz[sel] @ R_UP.T; C = np.asarray(rgbv[sel]); C = (C * 255).astype(int) if C.max() <= 1.0 else C.astype(int)
with (col / "points3D.txt").open("w") as fh:
    fh.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for j in range(len(P)):
        fh.write(f"{j + 1} {P[j, 0]:.5f} {P[j, 1]:.5f} {P[j, 2]:.5f} {int(C[j, 0])} {int(C[j, 1])} {int(C[j, 2])} 0.0\n")
cams_z = (R4 @ np.asarray(s.poses, np.float64))[:, :3, 3]
print(f"{n} images, {len(P):,} points; z range points p2/p98 {np.percentile(P[:, 2], 2):.2f}/{np.percentile(P[:, 2], 98):.2f}, cameras z med {np.median(cams_z[:, 2]):.2f}; xy extent {np.round(P[:, :2].max(0) - P[:, :2].min(0), 1).tolist()}")
np.save(WORK / "R_up.npy", R_UP)
