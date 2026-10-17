"""Our session → GenRecon's iPhone/COLMAP-text input: the keyframes, the CERTIFIED poses (the selected
epoch, c2w, OpenCV camera) as COLMAP w2c, the camera.json OPENCV intrinsics, and the published cloud as
points3D.txt. The scene is handed over in the session's DISPLAY frame (floor_transform.npz applied, so
up is up even when the raw frame lies on its side — bufferStop) and turned Y-up → Z-up (GenRecon's
chunker reads the height along z). The transform back to the raw frame is saved for publish.py."""
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
# the display frame: display = s_f · R_f · raw + t_f (identity on a baked session)
s_f, R_f, t_f = 1.0, np.eye(3), np.zeros(3)
ftp = out / "floor_transform.npz"
if ftp.exists():
    z = np.load(ftp); s_f, R_f, t_f = float(z["s"]), np.asarray(z["R"], np.float64), np.asarray(z["t"], np.float64)
M = np.eye(4); M[:3, :3] = R_UP @ (s_f * R_f); M[:3, 3] = R_UP @ t_f          # raw → GenRecon world
cam = json.loads((out / "camera.json").read_text())
W, H = int(cam["width"]), int(cam["height"]); fx, fy, cx, cy, k1, k2, p1, p2 = [float(v) for v in cam["params"][:8]]
col = WORK / "colmap_mast3r"; rgb = WORK / "rgb"
if WORK.exists(): shutil.rmtree(WORK)
col.mkdir(parents=True); rgb.mkdir()
(col / "cameras.txt").write_text("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                                 f"1 OPENCV {W} {H} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n")
lines = ["# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
n = 0
for i, (frame, c2w) in enumerate(zip(s.frames, np.asarray(s.poses, np.float64))):
    src = SES / "frames" / f"{int(frame):06d}.jpg"
    if not src.exists():
        continue
    (rgb / src.name).symlink_to(src)
    c2w_g = M @ c2w
    c2w_g[:3, :3] /= s_f                                   # the camera's rotation stays orthonormal
    w2c = np.linalg.inv(c2w_g)
    q = Rotation.from_matrix(w2c[:3, :3]).as_quat()           # x, y, z, w
    t = w2c[:3, 3]
    lines.append(f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {src.name}\n\n")
    n += 1
(col / "images.txt").write_text("".join(lines))
f = _read_ply_fields(out / "cleaned_cloud.ply")
xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
rgbv = np.column_stack([f[k] for k in ("red", "green", "blue")]) if "red" in f else np.full((len(xyz), 3), 128)
sel = np.linspace(0, len(xyz) - 1, min(len(xyz), N_PTS)).astype(np.int64)        # a fixed stride, never a draw
P = xyz[sel] @ M[:3, :3].T + M[:3, 3]; C = np.asarray(rgbv[sel]); C = (C * 255).astype(int) if C.max() <= 1.0 else C.astype(int)
with (col / "points3D.txt").open("w") as fh:
    fh.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for j in range(len(P)):
        fh.write(f"{j + 1} {P[j, 0]:.5f} {P[j, 1]:.5f} {P[j, 2]:.5f} {int(C[j, 0])} {int(C[j, 1])} {int(C[j, 2])} 0.0\n")
cams_g = (np.asarray(s.poses, np.float64)[:, :3, 3] @ M[:3, :3].T) + M[:3, 3]
# STAC 2026-10-16: the WHOLE published cloud in GenRecon's frame (the active voxels of every chunk come from it,
# reconstruct_scene --coords_from_points) and the session's floor and ceiling LEVELS for the chunk cubes
# (levels.json: the cube's base is the lowest support level, its top the highest ceiling level — the model
# paints floor and ceiling at the cube's faces, so a percentile of a subsample must not decide them).
np.save(WORK / "cloud_genrecon.npy", (xyz @ M[:3, :3].T + M[:3, 3]).astype(np.float32))
import yaml
from correction.floor import head_noun
_cfg = yaml.safe_load(open("/workspace/stac-build/server/config.yaml"))
support = set(_cfg["autoprompt"]["support_surfaces"])
y_d = (xyz @ (s_f * R_f).T + t_f)[:, 1]                      # heights in the display frame = GenRecon's z
seg = json.loads((out / "segmentation_result.json").read_text()) if (out / "segmentation_result.json").exists() else {"instances": []}
sup_idx = np.concatenate([np.asarray(i["globalIndices"], np.int64) for i in seg["instances"] if head_noun(i["label"]) in support] or [np.zeros(0, np.int64)])
ceil_idx = [np.asarray(i["globalIndices"], np.int64) for i in seg["instances"] if head_noun(i["label"]) == "ceiling"]
levels = {"bin_m": 0.05, "level_min_share": 0.01}
if len(sup_idx) > 1000:
    ys = y_d[sup_idx]; h, e = np.histogram(ys, bins=np.arange(np.floor(ys.min() * 20) / 20, ys.max() + 0.05, 0.05))
    lv = [float(e[i]) for i in range(len(h)) if h[i] / h.sum() >= levels["level_min_share"]]
    levels.update(z_floor=min(lv), support_levels=lv, source="segmentation support surfaces")
else:
    levels.update(z_floor=float(np.percentile(y_d, 0.2)), source="percentile 0.2 of the cloud (no support instance)")
if ceil_idx:
    levels.update(z_ceiling=max(float(np.percentile(y_d[g], 95)) for g in ceil_idx), ceiling_source="segmentation ceiling instances p95")
else:
    levels.update(z_ceiling=float(np.percentile(y_d, 99.97)), ceiling_source="percentile 99.97 of the cloud (no ceiling instance)")
(WORK / "levels.json").write_text(json.dumps(levels, indent=2))
# STAC 2026-10-17 (USER: "debería completar lo que no ve o no tenía puntos … las patas de una mesa … pero solo eso"):
# the boxes of the segmented OBJECTS in GenRecon's frame — the model may complete only inside them; a support
# surface, a wall, a ceiling, a pane is never an object here (that is where the phantom floors and rooms came from)
import re
STAND_MAX_M = 1.2      # an object's lowest points this close to the support under it stand on it (declared)
_STRUCTURAL = re.compile(r"\b(wall|walls|drywall|ceiling|ceilings|floor|floors|roof|slab|slabs|partition|partitions|panel|panels|sheeting|glass|window|windows|opening|openings|facade|façade)\b", re.I)
objs = []
for inst in seg["instances"]:
    obb = inst.get("obb") or {}
    if not obb.get("center") or head_noun(inst["label"]) in support or _STRUCTURAL.search(inst["label"].replace("_", " ")):
        continue                                   # labels carry underscores: "drywall_panel" must read as words
    c = np.asarray(obb["center"], np.float64); he = np.asarray(obb["half_extents"], np.float64); Rb = np.asarray(obb["rotation"], np.float64)
    iid = int(inst.get("instance_id", inst.get("id", 0)))
    objs.append({"id": iid, "label": inst["label"],
                 "center": (M[:3, :3] @ c + M[:3, 3]).tolist(), "half_extents": (he * s_f).tolist(),
                 "axes": (R_UP @ R_f @ Rb).tolist()})            # columns = the box axes in GenRecon's frame
    # STANDS ON THE FLOOR (USER: "las patas que tienen pocos puntos o no tienen ninguno … la geometría real exige
    # que sí tenga"): an object whose lowest points sit within STAND_MAX_M of the support level under it gets a
    # second, axis-aligned box from that level up to its lowest points over its footprint, where the model may
    # put the legs; a duct, a lamp, a cable hanging higher keeps its own box only
    gi = np.asarray(inst["globalIndices"], np.int64)
    if len(gi) >= 50:
        pd = xyz[gi] @ (s_f * R_f).T + t_f                              # display frame: y up
        y_lo = float(np.percentile(pd[:, 1], 1)); below = [l for l in levels.get("support_levels", [levels["z_floor"]]) if l <= y_lo + 0.05]
        if below:
            floor_y = max(below)
            if 0.03 < (y_lo - floor_y) <= STAND_MAX_M:
                x0, x1 = np.percentile(pd[:, 0], [1, 99]); z0, z1 = np.percentile(pd[:, 2], [1, 99])
                cd = np.array([(x0 + x1) / 2, (floor_y + y_lo) / 2, (z0 + z1) / 2]); hd = np.array([(x1 - x0) / 2, (y_lo - floor_y) / 2, (z1 - z0) / 2])
                objs.append({"id": iid, "label": inst["label"] + " (stand)", "center": (R_UP @ cd).tolist(),
                             "half_extents": hd[[0, 2, 1]].tolist(), "axes": np.eye(3).tolist()})   # y-up → z-up: extents reorder
(WORK / "objects.json").write_text(json.dumps({"frame": "genrecon (M_raw_to_genrecon applied)", "axes": "columns", "objects": objs}, indent=1))
print(f"objects.json: {len(objs)} object boxes of {len(seg['instances'])} instances (structural and support excluded)")
print(f"levels: z_floor {levels['z_floor']:+.3f} ({levels['source']}), z_ceiling {levels['z_ceiling']:+.3f} ({levels['ceiling_source']}); cloud_genrecon.npy {len(xyz):,} pts")
np.save(WORK / "M_raw_to_genrecon.npy", M); np.save(WORK / "R_up.npy", R_UP)
print(f"{n} images, {len(P):,} points; display transform {'identity' if np.allclose(R_f, np.eye(3)) and np.allclose(t_f, 0) else 'applied'}; "
      f"z range points p2/p98 {np.percentile(P[:, 2], 2):.2f}/{np.percentile(P[:, 2], 98):.2f}, cameras z med {np.median(cams_g[:, 2]):.2f}; "
      f"xy extent {np.round(P[:, :2].max(0) - P[:, :2].min(0), 1).tolist()}")
