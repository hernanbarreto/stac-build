"""Completion presence FROM THE MASKS (USER 2026-10-17: "las patas que tienen pocos puntos o ninguno … la parte
trasera de elementos … la geometría real exige que sí tenga" / on the model's own presence: "duplicó las figuras").

Inside each segmented OBJECT's box (never a floor, a wall, a ceiling, a pane) and, for an object standing on a
support level, the box from that level up to its lowest points over its footprint, a 5 cm voxel EXISTS when it
projects inside the object's SAM3 mask in the keyframes that frame it (the visual hull at voxel level) and it is
not within one voxel of the measured surface (never a second copy of what the cloud already has). No model
presence anywhere: GenRecon only shapes the fine surface inside these voxels. Writes hull_points.npy (GenRecon's
frame) for reconstruct_scene --coords_extra_points and hull_points.json (the record, per object).
    python hull_points.py <session src_default> <work dir> [inside_share 0.9] [min_views 3] [near_m 0.10]
"""
import sys, json, re, time, numpy as np, torch
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from scipy.spatial import cKDTree
import yaml
from correction.session import load_session
from correction.floor import head_noun
from segmentation.perfect_object import _read_ply_fields
from segmentation.object_captioner import _oids_by_instance

SES = Path(sys.argv[1]); WORK = Path(sys.argv[2])
INSIDE_SHARE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9     # share of the framing views the voxel must be inside the mask
MIN_VIEWS = int(sys.argv[4]) if len(sys.argv) > 4 else 3            # framing views needed to judge at all
NEAR_M = float(sys.argv[5]) if len(sys.argv) > 5 else 0.10          # closer than this to a cloud point = already measured
STEP = 0.05                                                          # candidate spacing
DILATE_PX = 3                                                        # the mask grown this much (its own edge noise)
STAND_MAX_M = 1.2                                                    # the adapter's rule for "stands on the support"
STRUCTURAL = re.compile(r"\b(wall|walls|drywall|ceiling|ceilings|floor|floors|roof|slab|slabs|partition|partitions|panel|panels|sheeting|glass|window|windows|opening|openings|facade|façade)\b", re.I)
out = SES / "output"; dev = torch.device("cuda")
t0 = time.time()
s = load_session(out); poses = np.asarray(s.poses, np.float64); n_kf = len(s.frames)
cam = json.loads((out / "camera.json").read_text()); fx, fy, cx, cy = [float(v) for v in cam["params"][:4]]; W, H = int(cam["width"]), int(cam["height"])
cfg = yaml.safe_load(open("/workspace/stac-build/server/config.yaml")); support = set(cfg["autoprompt"]["support_surfaces"])
s_f, R_f, t_f = 1.0, np.eye(3), np.zeros(3)
if (out / "floor_transform.npz").exists():
    z = np.load(out / "floor_transform.npz"); s_f, R_f, t_f = float(z["s"]), np.asarray(z["R"], np.float64), np.asarray(z["t"], np.float64)
M = np.load(WORK / "M_raw_to_genrecon.npy"); levels = json.loads((WORK / "levels.json").read_text())
seg = json.loads((out / "segmentation_result.json").read_text()); parent = json.loads((out / "segmentation.json").read_text())
oids_of = _oids_by_instance(parent, seg.get("absorbed") or {})
masks = np.load(out / "seg_masks.npz", allow_pickle=True); mask_keys = set(masks.keys())
f = _read_ply_fields(out / "cleaned_cloud.ply"); xyz = np.column_stack([f["x"], f["y"], f["z"]]).astype(np.float64)
cloud_tree = cKDTree(xyz[np.linspace(0, len(xyz) - 1, min(len(xyz), 8_000_000)).astype(np.int64)])
w2c = torch.from_numpy(np.linalg.inv(poses).astype(np.float32)).to(dev)            # [n_kf, 4, 4]

def to_raw_display(p_disp):          # display frame (y up) → raw
    return ((p_disp - t_f) @ np.linalg.inv(s_f * R_f).T)

def project_votes(P: np.ndarray, frames: list, mask_of: dict):
    """P (N, 3) raw frame. For every frame that has a mask of this object: framed (inside the image, in front)
    and inside the (dilated) mask. Returns (framed count, inside count) per point."""
    Pt = torch.from_numpy(P.astype(np.float32)).to(dev)
    framed = torch.zeros(len(P), dtype=torch.int32, device=dev); inside = torch.zeros(len(P), dtype=torch.int32, device=dev)
    for k in frames:
        Mk = w2c[k]; pc = Pt @ Mk[:3, :3].T + Mk[:3, 3]; zc = pc[:, 2]; front = zc > 0.2
        zs = torch.where(front, zc, torch.ones_like(zc))
        u = (fx * pc[:, 0] / zs + cx); v = (fy * pc[:, 1] / zs + cy)
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not bool(inb.any()):
            continue
        m = mask_of[k]
        ui = u.clamp(0, W - 1).long(); vi = v.clamp(0, H - 1).long()
        hit = m[vi, ui] > 0
        framed += inb.int(); inside += (inb & hit).int()
    return framed.cpu().numpy(), inside.cpu().numpy()

def dilate(m: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(m.astype(np.float32)).to(dev)[None, None]
    return torch.nn.functional.max_pool2d(t, kernel_size=2 * DILATE_PX + 1, stride=1, padding=DILATE_PX)[0, 0]

kept_all, record = [], {"inside_share": INSIDE_SHARE, "min_views": MIN_VIEWS, "near_m": NEAR_M, "step_m": STEP, "dilate_px": DILATE_PX, "objects": []}
for inst in seg["instances"]:
    obb = inst.get("obb") or {}
    label = str(inst.get("label", "")); iid = int(inst.get("instance_id", inst.get("id", 0)))
    if not obb.get("center") or head_noun(label) in support or STRUCTURAL.search(label.replace("_", " ")):
        continue
    oids = oids_of.get(iid, [])
    frames = sorted({k for k in range(n_kf) for o in oids if f"f{k}_o{o}" in mask_keys})
    if len(frames) < MIN_VIEWS:
        continue
    c = np.asarray(obb["center"], np.float64); he = np.asarray(obb["half_extents"], np.float64); Rb = np.asarray(obb["rotation"], np.float64)
    # candidates: the OBB (grown by one step) on a 5 cm lattice in its own axes
    axes = [np.arange(-h - STEP, h + STEP + 1e-9, STEP) for h in he]
    shape_obb = tuple(len(a) for a in axes)
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    cand = g @ Rb.T + c
    shape_stand = None
    # the stand box, in the display frame (y up), down to the support level under the object
    gi = np.asarray(inst["globalIndices"], np.int64)
    if len(gi) >= 50:
        pd = xyz[gi] @ (s_f * R_f).T + t_f
        y_lo = float(np.percentile(pd[:, 1], 1)); below = [l for l in levels.get("support_levels", [levels["z_floor"]]) if l <= y_lo + 0.05]
        if below and 0.03 < (y_lo - max(below)) <= STAND_MAX_M:
            x0, x1 = np.percentile(pd[:, 0], [1, 99]); z0, z1 = np.percentile(pd[:, 2], [1, 99]); fy_ = max(below)
            ax_s = (np.arange(x0, x1 + 1e-9, STEP), np.arange(fy_ + STEP / 2, y_lo, STEP), np.arange(z0, z1 + 1e-9, STEP))
            gs = np.stack(np.meshgrid(*ax_s, indexing="ij"), -1).reshape(-1, 3)
            if len(gs):
                shape_stand = tuple(len(a) for a in ax_s)
                cand = np.concatenate([cand, to_raw_display(gs)])
    if len(cand) > 3_000_000:
        record["objects"].append({"id": iid, "label": label, "skipped": f"{len(cand)} candidates"}); continue
    # the masks of this object per frame (its masklets ORed), dilated
    mask_of = {}
    for k in frames:
        m = None
        for o in oids:
            key = f"f{k}_o{o}"
            if key in mask_keys:
                a = masks[key]; m = a if m is None else np.maximum(m, a)
        if m is not None and m.shape == (H, W):
            mask_of[k] = dilate(m)
    frames = [k for k in frames if k in mask_of]
    if len(frames) < MIN_VIEWS:
        continue
    framed, inside = project_votes(cand, frames, mask_of)
    judged = framed >= MIN_VIEWS
    hull = judged & (inside >= INSIDE_SHARE * np.maximum(framed, 1))
    # the SHELL of the hull only: a voxel with a 6-neighbour outside it (the generator models surfaces; a solid
    # interior would ask it for nested ones). Each lattice is regular, so erosion on its grid gives the shell.
    from scipy.ndimage import binary_erosion
    n_obb = int(np.prod(shape_obb)); shell = np.zeros_like(hull)
    h_o = hull[:n_obb].reshape(shape_obb); shell[:n_obb] = (h_o & ~binary_erosion(h_o)).ravel()
    if shape_stand is not None:
        h_s = hull[n_obb:].reshape(shape_stand); shell[n_obb:] = (h_s & ~binary_erosion(h_s)).ravel()
    n_solid = int(hull.sum()); hull = shell
    pts = cand[hull]
    n_hull = int(hull.sum())
    if n_hull:
        d, _ = cloud_tree.query(pts, k=1, workers=8)
        pts = pts[d > NEAR_M]
    n_new = len(pts)
    if n_new:
        kept_all.append(pts)
        yd = (pts @ (s_f * R_f).T + t_f)[:, 1]
        rec_y = [round(float(np.percentile(yd, q)), 2) for q in (1, 50, 99)]
    else:
        rec_y = None
    record["objects"].append({"id": iid, "label": label, "mask_frames": len(frames), "candidates": int(len(cand)), "judged": int(judged.sum()),
                              "in_hull_solid": n_solid, "in_hull_shell": n_hull, "new_beyond_cloud": n_new, "y_p1_p50_p99": rec_y})
    del mask_of
K = np.concatenate(kept_all) if kept_all else np.zeros((0, 3))
Kg = (K @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
np.save(WORK / "hull_points.npy", Kg)
record["total_points"] = int(len(Kg)); record["objects_with_completion"] = int(sum(1 for o in record["objects"] if o.get("new_beyond_cloud")))
record["seconds"] = round(time.time() - t0)
(WORK / "hull_points.json").write_text(json.dumps(record, indent=1))
top = sorted([o for o in record["objects"] if "new_beyond_cloud" in o], key=lambda o: -o["new_beyond_cloud"])[:8]
print(f"hull_points: {len(Kg):,} completion voxel centres (5 cm) beyond the cloud in {record['objects_with_completion']} of {len(record['objects'])} objects, {record['seconds']} s; top: "
      + "; ".join(f"{o['label']}#{o['id']} {o['new_beyond_cloud']} (shell {o['in_hull_shell']} of solid {o['in_hull_solid']}, y {o['y_p1_p50_p99']})" for o in top))
