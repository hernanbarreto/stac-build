"""Our session → GenRecon's iPhone/COLMAP-text input: the keyframes, the CERTIFIED poses (the selected
epoch, c2w, OpenCV camera) as COLMAP w2c, the camera.json OPENCV intrinsics, and the published cloud as
points3D.txt. The scene is handed over in the session's DISPLAY frame (floor_transform.npz applied, so
up is up even when the raw frame lies on its side — bufferStop) and turned Y-up → Z-up (GenRecon's
chunker reads the height along z). The transform back to the raw frame is saved for publish.py."""
import sys, json, shutil, yaml, numpy as np
from pathlib import Path
SERVER = Path(__file__).resolve().parents[2]      # the server tree this adapter belongs to (its modules, config.yaml)
sys.path.insert(0, str(SERVER))
from scipy.spatial.transform import Rotation
from correction.session import load_session
from segmentation.perfect_object import _read_ply_fields

SES = Path(sys.argv[1]); WORK = Path(sys.argv[2]); N_PTS = int(sys.argv[3]) if len(sys.argv) > 3 else 600_000
CUBE_M = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0     # genrecon.cube_size_m: stacked cubes of this side (0 = one layer)
CONF = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0       # the viewer's confidence slider, 0..1 (0 = every point)
MIN_PTS = int(sys.argv[6]) if len(sys.argv) > 6 else 10       # --coords_min_points: a voxel is active from this many points
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
# USER 2026-11-05: "usé el filtro de confianza del panel … al 25 % sale bien … que ajuste el usuario al valor que deja la
# geometría limpia, eso depende de cómo se capturó, no puede generalizarse": the points GenRecon receives are the ones the
# viewer SHOWS at the user's slider — the same arithmetic (PotreeLoader.ts: the raw `confidence` normalised by the octree
# metadata's min/max; Viewport.tsx hides a point whose normalised confidence is under the slider). 0 = every point.
n_all = len(xyz)
keep_all = np.ones(n_all, bool)
if CONF > 0:
    conf = np.asarray(f["confidence"], np.float64)
    meta = out / "potree" / "metadata.json"
    a = next((a for a in json.loads(meta.read_text())["attributes"] if a["name"] == "confidence"), None) if meta.exists() else None
    cmin, cmax = (float(a["min"][0]), float(a["max"][0])) if a else (float(conf.min()), float(conf.max()))
    keep_all = (conf - cmin) / max(cmax - cmin, 1e-6) >= CONF
    print(f"confidence filter {CONF:.2f} (the viewer's: raw confidence normalised by [{cmin:g}, {cmax:g}] "
          f"{'from the octree metadata' if a else 'of the cloud itself (no octree metadata)'}): "
          f"{int(keep_all.sum()):,} of {n_all:,} points kept ({100.0 * keep_all.sum() / max(n_all, 1):.1f} %)")
# USER 2026-11-05 ("que decidan las fotos, no la cuenta de puntos … el límite es la nube"): the keyframes vote on every
# point that passed the slider — the depth each keyframe measured is rebuilt from the cloud (each point's birth
# keyframe and pixel); a point stays when enough keyframes measure it and fewer see through it (presence.py). The
# votes only remove points; the cubes and their active voxels are laid over the WHOLE of what stays.
LAYOUT = None
_g = (yaml.safe_load(open(SERVER / "config.yaml")).get("genrecon") or {})
VOTE = bool(_g.get("presence_vote", False))      # USER 2026-11-05: "hay que volver a como estaba antes" — off
# USER 2026-11-11 ("todo lo que me marcaste que puede pasarse a GPU hay que hacerlo en GPU … Nada debe hacerse en
# cpu"): the arithmetic over the cloud's points runs on genrecon.adapter_device — numpy is THE RECORD (the lines
# below as they always were), cpu / cuda are layout_gpu.py's twins of them
import layout_gpu as LG
DEV = LG.adapter_device(_g)
if CUBE_M > 0 and VOTE:
    from presence import vote_points
    votes = vote_points(xyz[keep_all], np.asarray(f["frame_global"])[keep_all], np.asarray(f["pixel_row"])[keep_all],
                        np.asarray(f["pixel_col"])[keep_all], [int(fr) for fr in s.frames], np.asarray(s.poses, np.float64),
                        [fx, fy, cx, cy, k1, k2, p1, p2], W, H, CUBE_M / 32.0,
                        min_views=int(_g.get("presence_min_views", 2)), device=str(_g.get("presence_device", "cuda")))
    sub = np.flatnonzero(keep_all)
    keep_all[sub[~votes["keep"]]] = False
    (WORK / "presence.json").write_text(json.dumps({k: v for k, v in votes.items() if k not in ("keep", "support", "through")}
                                                   | {"points_published": int(n_all), "points_kept": int(keep_all.sum()),
                                                      "confidence_threshold": CONF}, indent=2))
old2new = None
if not keep_all.all():
    old2new = np.full(n_all, -1, np.int64); old2new[keep_all] = np.arange(int(keep_all.sum()))
    xyz, rgbv = xyz[keep_all], rgbv[keep_all]


def _remap(idx):
    """Indices of the published cloud (the segmentation's globalIndices) → indices of the points kept."""
    idx = np.asarray(idx, np.int64)
    if old2new is None:
        return idx
    r = old2new[idx]
    return r[r >= 0]
sel = np.linspace(0, len(xyz) - 1, min(len(xyz), N_PTS)).astype(np.int64)        # a fixed stride, never a draw
if DEV == "numpy":
    P = xyz[sel] @ M[:3, :3].T + M[:3, 3]
else:
    import torch
    torch.use_deterministic_algorithms(True)
    xyz_t = LG.as_points(xyz, DEV)                         # the points kept, on the device once
    P_t = LG.to_frame(xyz_t, M[:3, :3], M[:3, 3])          # … in GenRecon's frame
    P = P_t[torch.from_numpy(sel).to(DEV)].cpu().numpy()
C = np.asarray(rgbv[sel]); C = (C * 255).astype(int) if C.max() <= 1.0 else C.astype(int)
with (col / "points3D.txt").open("w") as fh:
    fh.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for j in range(len(P)):
        fh.write(f"{j + 1} {P[j, 0]:.5f} {P[j, 1]:.5f} {P[j, 2]:.5f} {int(C[j, 0])} {int(C[j, 1])} {int(C[j, 2])} 0.0\n")
cams_g = (np.asarray(s.poses, np.float64)[:, :3, 3] @ M[:3, :3].T) + M[:3, 3]
# STAC 2026-10-16: the WHOLE published cloud in GenRecon's frame (the active voxels of every chunk come from it,
# reconstruct_scene --coords_from_points) and the session's floor and ceiling LEVELS for the chunk cubes
# (levels.json: the cube's base is the lowest level — of the support surfaces or of the whole cloud —, its top
# the highest ceiling level; the model paints floor and ceiling at the cube's faces, so a percentile of a
# subsample must not decide them).
if DEV == "numpy":
    P_g = xyz @ M[:3, :3].T + M[:3, 3]                     # the points kept, in GenRecon's frame
    np.save(WORK / "cloud_genrecon.npy", P_g.astype(np.float32))
else:
    P_g = P_t
    np.save(WORK / "cloud_genrecon.npy", P_t.to(torch.float32).cpu().numpy())
import yaml
from correction.floor import head_noun
_cfg = yaml.safe_load(open(SERVER / "config.yaml"))
support = set(_cfg["autoprompt"]["support_surfaces"])
if DEV == "numpy":
    y_d = (xyz @ (s_f * R_f).T + t_f)[:, 1]                  # heights in the display frame = GenRecon's z
else:
    D_t = LG.to_frame(xyz_t, s_f * R_f, t_f)               # the display frame (y up), on the device
seg =json.loads((out / "segmentation_result.json").read_text()) if (out / "segmentation_result.json").exists() else {"instances": []}
sup_idx = np.concatenate([_remap(i["globalIndices"]) for i in seg["instances"] if head_noun(i["label"]) in support] or [np.zeros(0, np.int64)])
ceil_idx = [_remap(i["globalIndices"]) for i in seg["instances"] if head_noun(i["label"]) == "ceiling"]
# the cube's base on the LOWEST level, of the support surfaces or of the whole cloud (genrecon_levels.py:
# station's tracks, 1.15 m under the platform and no support noun, sat under every cube — USER 2026-11-02)
from genrecon_levels import anchored_centers, chunk_levels, stack_layers
if DEV == "numpy":
    levels = chunk_levels(y_d, y_d[sup_idx], [y_d[g] for g in ceil_idx], bin_m=0.05, min_share=0.01)
    z_lo, z_hi = float(P_g[:, 2].min()), float(P_g[:, 2].max())
    xy_extent = [float(P_g[:, 0].min()), float(P_g[:, 0].max()), float(P_g[:, 1].min()), float(P_g[:, 1].max())]
else:
    def _heights(idx):
        """The display heights of these points of the cloud, sorted on the device."""
        return LG.sorted_heights(D_t[torch.from_numpy(np.asarray(idx, np.int64)).to(DEV), 1])
    levels = chunk_levels(LG.sorted_heights(D_t[:, 1]), _heights(sup_idx), [_heights(g) for g in ceil_idx],
                          bin_m=0.05, min_share=0.01, levels_of=LG.levels_of_sorted,
                          percentile_of=LG.percentile_of_sorted)
    g_lo, g_hi = LG.extent(P_t)
    z_lo, z_hi = float(g_lo[2]), float(g_hi[2])
    xy_extent = [float(g_lo[0]), float(g_hi[0]), float(g_lo[1]), float(g_hi[1])]
levels["floor_level"], levels["floor_level_source"] = levels["z_floor"], levels["source"]
levels.update(z_top=z_hi, coverage_z=[z_lo, z_hi], xy_extent=xy_extent,
              coords_min_points=MIN_PTS, coords_resolution=32,     # GenRecon's shape SLat grid (11.6 cm at 3.7 m)
              confidence_threshold=CONF, points_kept=int(len(xyz)), points_published=int(n_all))
centers = None
if CUBE_M > 0:
    # USER 2026-11-08 ("grilla anclada en y = 0 … suelo sólido … nada más cambia"): one cube base sits EXACTLY on
    # the lowest measured support level (z_floor as chunk_levels measured it — the model paints the floor at a
    # cube's base, as it was trained), the layers rise as before and continue DOWN only where the cloud reaches
    # lower, so the WHOLE cloud stays covered (USER 2026-11-05: "debe abarcar la totalidad de la nube … no me
    # importa dónde está ubicada ni cuánto mide") over its whole floor plan (xy_extent, the exact min/max — never
    # a percentile of a subsample). The support levels' columns are filled SOLID down to the lattice base
    # (presence.plan_layout: "el suelo es sólido"), so no floor keeps air below it in any cube or layer.
    anchor = float(levels["z_floor"])
    support_z = [float(l) for l in (levels.get("support_levels") or [anchor])]
    centers = anchored_centers(anchor, z_lo, z_hi, CUBE_M)
    levels.update(z_anchor=anchor, anchor_source=levels["source"])
    from presence import plan_layout
    # a voxel active from MIN_PTS points — GenRecon's own presence rule, as always — unless the keyframes' vote
    # already kept only confirmed points (then from one)
    LAYOUT = plan_layout(P_g, CUBE_M, min_points=(1 if VOTE else MIN_PTS),
                         z_centers=centers, support_levels=support_z, level_bin=0.05, device=DEV)
    levels["ground_filled_voxels"] = LAYOUT["ground_filled"]
    np.save(WORK / "active_cells.npy", LAYOUT.pop("cells"))
    (WORK / "layout.json").write_text(json.dumps(LAYOUT, indent=1))
else:
    # GenRecon's own single floor-to-ceiling layer: full coverage keeps the 2026-11-05 rule unchanged
    levels.update(z_floor=z_lo, source="the whole cloud's lowest point (USER 2026-11-05)")
levels = stack_layers(levels, CUBE_M, centers)
(WORK / "levels.json").write_text(json.dumps(levels, indent=2))
# STAC 2026-10-17 (USER: "debería completar lo que no ve o no tenía puntos … las patas de una mesa … pero solo eso"):
# the boxes of the segmented OBJECTS in GenRecon's frame — the model may complete only inside them; a support
# surface, a wall, a ceiling, a pane is never an object here (that is where the phantom floors and rooms came from)
import re
STAND_MAX_M = 1.2      # an object's lowest points this close to the support under it stand on it (declared)
STAND_MIN_PTS = 50     # an object's lowest points are read from this many points (as always)
_STRUCTURAL = re.compile(r"\b(wall|walls|drywall|ceiling|ceilings|floor|floors|roof|slab|slabs|partition|partitions|panel|panels|sheeting|glass|window|windows|opening|openings|facade|façade)\b", re.I)


def _no_object(inst):
    """An instance with no box, a support surface, a wall, a ceiling, a pane: never an object here."""
    obb = inst.get("obb") or {}
    return bool(not obb.get("center") or head_noun(inst["label"]) in support
                or _STRUCTURAL.search(inst["label"].replace("_", " ")))   # labels carry underscores: "drywall_panel" must read as words


# on the device: the lowest points and the footprint of EVERY object ranked together, in one pass per axis over
# their points (the record reads them object by object, inside the loop)
STANDS = {}
if DEV != "numpy":
    held = [(k, _remap(inst["globalIndices"])) for k, inst in enumerate(seg["instances"]) if not _no_object(inst)]
    held = [(k, gi) for k, gi in held if len(gi) >= STAND_MIN_PTS]
    if held:
        pd_t = D_t[torch.from_numpy(np.concatenate([gi for _, gi in held])).to(DEV)]      # display frame: y up
        sizes = [len(gi) for _, gi in held]
        lows = LG.group_percentiles(pd_t[:, 1], sizes, [1])
        span_x, span_z = LG.group_percentiles(pd_t[:, 0], sizes, [1, 99]), LG.group_percentiles(pd_t[:, 2], sizes, [1, 99])
        STANDS = {k: (lows[j, 0], span_x[j, 0], span_x[j, 1], span_z[j, 0], span_z[j, 1]) for j, (k, _) in enumerate(held)}
objs = []
for k, inst in enumerate(seg["instances"]):
    obb = inst.get("obb") or {}
    if _no_object(inst):
        continue
    c =np.asarray(obb["center"], np.float64); he = np.asarray(obb["half_extents"], np.float64); Rb = np.asarray(obb["rotation"], np.float64)
    iid = int(inst.get("instance_id", inst.get("id", 0)))
    objs.append({"id": iid, "label": inst["label"],
                 "center": (M[:3, :3] @ c + M[:3, 3]).tolist(), "half_extents": (he * s_f).tolist(),
                 "axes": (R_UP @ R_f @ Rb).tolist()})            # columns = the box axes in GenRecon's frame
    # STANDS ON THE FLOOR (USER: "las patas que tienen pocos puntos o no tienen ninguno … la geometría real exige
    # que sí tenga"): an object whose lowest points sit within STAND_MAX_M of the support level under it gets a
    # second, axis-aligned box from that level up to its lowest points over its footprint, where the model may
    # put the legs; a duct, a lamp, a cable hanging higher keeps its own box only
    if DEV == "numpy":
        gi = _remap(inst["globalIndices"])
        reads = len(gi) >= STAND_MIN_PTS
    else:
        reads = k in STANDS
    if reads:
        if DEV == "numpy":
            pd = xyz[gi] @ (s_f * R_f).T + t_f                          # display frame: y up
            y_lo = float(np.percentile(pd[:, 1], 1))
        else:
            y_lo = float(STANDS[k][0])
        below = [l for l in levels.get("support_levels", [levels["z_floor"]]) if l <= y_lo + 0.05]
        if below:
            floor_y = max(below)
            if 0.03 < (y_lo - floor_y) <= STAND_MAX_M:
                if DEV == "numpy":
                    x0, x1 = np.percentile(pd[:, 0], [1, 99]); z0, z1 = np.percentile(pd[:, 2], [1, 99])
                else:
                    x0, x1, z0, z1 = STANDS[k][1:]
                cd =np.array([(x0 + x1) / 2, (floor_y + y_lo) / 2, (z0 + z1) / 2]); hd = np.array([(x1 - x0) / 2, (y_lo - floor_y) / 2, (z1 - z0) / 2])
                objs.append({"id": iid, "label": inst["label"] + " (stand)", "center": (R_UP @ cd).tolist(),
                             "half_extents": hd[[0, 2, 1]].tolist(), "axes": np.eye(3).tolist()})   # y-up → z-up: extents reorder
(WORK / "objects.json").write_text(json.dumps({"frame": "genrecon (M_raw_to_genrecon applied)", "axes": "columns", "objects": objs}, indent=1))
print(f"objects.json: {len(objs)} object boxes of {len(seg['instances'])} instances (structural and support excluded)")
print(f"levels: z_floor {levels['z_floor']:+.3f} ({levels['source']}), z_ceiling {levels['z_ceiling']:+.3f} ({levels['ceiling_source']}), "
      f"top {levels['z_top']:+.3f}; " + (f"{len(levels['chunk_z_centers'])} layer(s) of {CUBE_M} m cubes centred at "
      f"{[round(z, 3) for z in levels['chunk_z_centers']]}; " if 'chunk_z_centers' in levels else "one layer, floor to ceiling; ")
      + (f"one base anchored at {levels['z_anchor']:+.3f}, {levels.get('ground_filled_voxels', 0):,} solid-ground "
         f"voxels filled; " if 'z_anchor' in levels else "")
      + f"cloud_genrecon.npy {len(xyz):,} pts")
np.save(WORK / "M_raw_to_genrecon.npy", M); np.save(WORK / "R_up.npy", R_UP)
print(f"{n} images, {len(P):,} points; display transform {'identity' if np.allclose(R_f, np.eye(3)) and np.allclose(t_f, 0) else 'applied'}; "
      f"z range points p2/p98 {np.percentile(P[:, 2], 2):.2f}/{np.percentile(P[:, 2], 98):.2f}, cameras z med {np.median(cams_g[:, 2]):.2f}; "
      f"xy extent {np.round(P[:, :2].max(0) - P[:, :2].min(0), 1).tolist()}")
