"""ShapeR input packages (segmentation/shaper_export.py) on a precision session
(USER 2026-09-29: ShapeR replaces MeshFlow behind the generative button).

A synthetic box seen by keyframes on a circle: camera.json (F5 camera, OPENCV,
no lens distortion, a half-resolution traceability grid), camera_poses.txt /
camera_frames.txt, native frames, a cleaned cloud with provenance, one segmented
instance, orientation baked (+Y up). Checks the package the vendor consumes:
- one PKL per object, named by its folder (the runner reports by PKL stem);
- Fisheye624 whose radial polynomial IS the pinhole (the vendor rectifies every
  view Fisheye624 → pinhole; k = 0 is equidistant and warped 27 % at the corner);
- the object's projections land where the pinhole camera puts its points;
- Z-up model frame from the baked +Y, metric scale kept (bounds = half extents);
- USER 2026-10-01: EVERY posed keyframe that SEES the object is a view (not only
  the frames its points were born in), the description is the VLM's stored
  `shape_caption` (manual > object > concept > on-demand > label), the published
  GLB is the one with its own .meta.json, `simplify_faces` reaches the batch.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precision.camera import CameraModel, GridMap, save_camera_json
from segmentation import shaper_export as SE

W, H, F = 232, 416, 196.0          # native frame (portrait, like the videos)
GRID_W, GRID_H = 116, 208           # the traceability (Omega record) grid: half resolution
SEEN_ONLY_FRAME = 99                # a posed keyframe that SEES the box but birthed none of its points
N_BIRTH_FRAMES = 8


def _write_ply(path: Path, xyz, fg, pr, pc):
    n = len(xyz)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                   ("blue", "u1"), ("frame_global", "<i4"), ("pixel_row", "<i4"), ("pixel_col", "<i4")])
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    a["red"] = a["green"] = a["blue"] = 128
    a["frame_global"], a["pixel_row"], a["pixel_col"] = fg, pr, pc
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
           "property float x", "property float y", "property float z",
           "property uchar red", "property uchar green", "property uchar blue",
           "property int frame_global", "property int pixel_row", "property int pixel_col",
           "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(a.tobytes())


def _look_at(c, target):
    z = target - c
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 1.0, 0.0]))       # +Y up world, OpenCV camera (y down)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = c
    return T


def _write_poses(out: Path, frames, poses):
    (out / "camera_frames.txt").write_text("\n".join(map(str, frames)) + "\n")
    np.savetxt(out / "camera_poses.txt", np.array(poses).reshape(len(poses), 16))


@pytest.fixture()
def session(tmp_path):
    out = tmp_path / "output"
    fr = tmp_path / "frames"
    out.mkdir()
    fr.mkdir()
    grid = GridMap(name="omega", w=GRID_W, h=GRID_H, content_w=GRID_W, content_h=GRID_H,
                   pad_left=0, pad_top=0, crop_x=0, crop_y=0, crop_w=W, crop_h=H,
                   native_w=W, native_h=H)
    cam = CameraModel(width=W, height=H, params=(F, F, W / 2, H / 2, 0, 0, 0, 0),
                      source="synthetic", camera_epoch=1, omega_grid=grid)
    save_camera_json(out / "camera.json", cam, geometry_epoch=0)
    rng = np.random.default_rng(0)
    box = rng.uniform([-0.2, 0.0, -0.15], [0.2, 0.5, 0.15], (4000, 3))   # a 40×50×30 cm box on the floor
    target = np.array([0.0, 0.25, 0.0])
    frames, poses, xyz, fg, pr, pc = [], [], [], [], [], []
    for i, ang in enumerate(np.linspace(0, np.pi, N_BIRTH_FRAMES)):
        f = 10 * i + 3
        T = _look_at(np.array([1.6 * np.cos(ang), 1.2, 1.6 * np.sin(ang)]), target)
        w2c = np.linalg.inv(T)
        q = box @ w2c[:3, :3].T + w2c[:3, 3]
        u, v = F * q[:, 0] / q[:, 2] + W / 2, F * q[:, 1] / q[:, 2] + H / 2
        ok = (q[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        sel = np.flatnonzero(ok)[:400]
        # provenance on the half-resolution grid (native = (g + 0.5)·2 − 0.5)
        gu = np.clip(np.round((u[sel] + 0.5) / 2 - 0.5), 0, GRID_W - 1).astype(int)
        gv = np.clip(np.round((v[sel] + 0.5) / 2 - 0.5), 0, GRID_H - 1).astype(int)
        xyz.append(box[sel]); fg.append(np.full(len(sel), f)); pr.append(gv); pc.append(gu)
        img = np.full((H, W, 3), 200, np.uint8)
        img[np.clip(v[sel].astype(int), 0, H - 1), np.clip(u[sel].astype(int), 0, W - 1)] = 40
        Image.fromarray(img).save(fr / f"{f:06d}.jpg")
        frames.append(f)
        poses.append(T)
    # a posed keyframe on the other side of the circle: it SEES the whole box but
    # not one of the cloud's points was born in it (filtered away downstream)
    T = _look_at(np.array([1.6 * np.cos(1.3 * np.pi), 1.2, 1.6 * np.sin(1.3 * np.pi)]), target)
    Image.fromarray(np.full((H, W, 3), 200, np.uint8)).save(fr / f"{SEEN_ONLY_FRAME:06d}.jpg")
    frames.append(SEEN_ONLY_FRAME)
    poses.append(T)
    _write_poses(out, frames, poses)
    (out / ".orientation_applied").write_text("baked\n")
    xyz = np.concatenate(xyz)
    _write_ply(out / "cleaned_cloud.ply", xyz, np.concatenate(fg), np.concatenate(pr), np.concatenate(pc))
    seg = {"instances": [{"id": 7, "instance_id": 7, "label": "wooden box",
                          "globalIndices": list(range(len(xyz)))}]}
    return tmp_path, out, fr, seg, xyz


def _export(root, out, fr, seg, **kw):
    kw.setdefault("max_views", 8)
    kw.setdefault("min_view_points", 20)
    pkls = SE.export_shaper_pkls(out, fr, seg, session_dir=root, **kw)
    assert len(pkls) == 1
    return pkls[0], pickle.loads(pkls[0].read_bytes())


def test_the_package_is_what_the_vendor_consumes(session):
    root, out, fr, seg, xyz = session
    pkl, d = _export(root, out, fr, seg, captions={7: "a small wooden box with a lid"})
    assert pkl.name == "wooden_box_7.pkl" and pkl.parent.name == "wooden_box_7"
    assert d["caption"] == "a small wooden box with a lid" and d["instance_id"] == 7
    assert d["caption_source"] == "manual"
    n_v = d["n_views"]
    assert 2 <= n_v <= 8 and len(d["image_data"]) == n_v == d["camera_params"].shape[0]
    # Z-up model frame from the baked +Y, metric: the box is 50 cm tall → half-extent 0.25 on Z
    b = d["bounds"].numpy()
    assert abs(b[2] - 0.25) < 0.02 and abs(b[0] - 0.20) < 0.02


def test_fisheye624_is_the_pinhole_and_the_projections_land_on_it(session):
    root, out, fr, seg, xyz = session
    _, d = _export(root, out, fr, seg, captions={7: "box"})
    for v in range(d["n_views"]):
        p = d["camera_params"][v].numpy().astype(np.float64)
        fx, fy, cx, cy = p[:4]
        assert np.allclose(p[4:10], SE._TAN_SERIES_K) and not np.any(p[10:])
        # the Fisheye624 radial model reproduces r = f·tanθ over the frame
        pts = d["visible_points_model"][v].numpy().astype(np.float64)
        T = d["Ts_camera_model"][v].numpy().astype(np.float64)
        q = pts @ T[:3, :3].T + T[:3, 3]
        r_n = np.hypot(q[:, 0], q[:, 1]) / q[:, 2]
        th = np.arctan(r_n)
        rd = th * (1 + sum(k * th ** (2 * (i + 1)) for i, k in enumerate(SE._TAN_SERIES_K)))
        u = fx * rd * (q[:, 0] / q[:, 2]) / np.maximum(r_n, 1e-12) + cx
        vv = fy * rd * (q[:, 1] / q[:, 2]) / np.maximum(r_n, 1e-12) + cy
        uv = d["object_point_projections"][v].numpy()
        # the stored projections ARE the pinhole camera's (all of the object's points
        # the view sees, projected — not the birth pixels of a subset)
        assert np.median(np.hypot(uv[:, 0] - u, uv[:, 1] - vv)) < 1.0
        img = Image.open(__import__("io").BytesIO(d["image_data"][v]))
        assert img.size == (W, H) and img.mode == "L"          # native, grayscale


def test_no_camera_json_falls_back_without_the_precision_grid(session):
    root, out, fr, seg, xyz = session
    src = SE._load_precision_source(out)
    assert src is not None and src.backend == "precision" and src.grid is not None
    (out / "camera.json").unlink()
    assert SE._load_precision_source(out) is None


# ── USER 2026-10-01: every posed keyframe that SEES the object is a view ──

def test_a_keyframe_that_sees_the_object_but_birthed_no_point_is_a_view(session):
    root, out, fr, seg, xyz = session
    _, d = _export(root, out, fr, seg, captions={7: "box"}, max_views=16)
    # the view pool is every posed keyframe that sees ≥ min_view_points of the object:
    # the 8 birth frames AND the one that birthed nothing
    assert d["n_posed_frames"] == N_BIRTH_FRAMES + 1
    assert d["n_candidate_frames"] == N_BIRTH_FRAMES + 1
    assert d["n_views"] == N_BIRTH_FRAMES + 1
    assert SEEN_ONLY_FRAME in d["source_frames"]
    v = d["source_frames"].index(SEEN_ONLY_FRAME)
    # that view carries the whole box (every point projects inside the frame)
    assert len(d["visible_points_model"][v]) == len(xyz)
    assert len(d["object_point_projections"][v]) == len(xyz)


def test_a_keyframe_looking_away_is_not_a_view(session):
    root, out, fr, seg, xyz = session
    frames = [int(x) for x in (out / "camera_frames.txt").read_text().split()]
    poses = list(np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4))
    # a posed keyframe with the box BEHIND it: posed, on disk, sees nothing of it
    away = 123
    poses.append(_look_at(np.array([3.0, 1.2, 0.0]), np.array([6.0, 1.2, 0.0])))
    frames.append(away)
    Image.fromarray(np.full((H, W, 3), 200, np.uint8)).save(fr / f"{away:06d}.jpg")
    _write_poses(out, frames, poses)
    _, d = _export(root, out, fr, seg, captions={7: "box"}, max_views=16)
    assert d["n_posed_frames"] == N_BIRTH_FRAMES + 2
    assert d["n_candidate_frames"] == N_BIRTH_FRAMES + 1
    assert away not in d["source_frames"] and SEEN_ONLY_FRAME in d["source_frames"]


def test_max_views_still_picks_a_diverse_subset_of_the_pool(session):
    root, out, fr, seg, xyz = session
    _, d = _export(root, out, fr, seg, captions={7: "box"}, max_views=4)
    assert d["n_views"] == 4 and d["n_candidate_frames"] == N_BIRTH_FRAMES + 1
    assert len(set(d["source_frames"])) == 4


def test_per_instance_phases_are_reported_to_the_ui(session):
    root, out, fr, seg, xyz = session
    phases = []
    _export(root, out, fr, seg, captions={7: "box"}, on_phase=lambda iid, ph: phases.append((iid, ph)))
    assert phases == [(7, "exporting_pkl")]          # no captioner ran: nothing to describe


# ── USER 2026-10-01: the description is the VLM's ──

_STORED_OBJECT = {"caption": "A rectangular wooden crate with a hinged lid.", "category": "crate",
                  "shape": "rectangular box", "material": "wood", "detail": "hinged lid",
                  "provenance": "vlm_proposed", "source": "object", "generated": "2026-10-01"}
_STORED_CONCEPT = {"caption": "A wooden storage box.", "category": "box", "shape": "box",
                   "material": "wood", "provenance": "vlm_proposed", "source": "concept"}


def test_caption_precedence_manual_over_stored_over_on_demand_over_label():
    inst = {"id": 7, "label": "wooden box", "shape_caption": _STORED_OBJECT}
    calls = []

    def captioner(frames, masks, label):
        calls.append((frames, masks, label))
        return {"caption": "An on-demand description.", "category": "thing", "shape": "",
                "material": "", "detail": ""}

    # 1. manual (human_validated) wins over everything, and its fields are parsed
    text, fields, src = SE.resolve_caption(7, "wooden box", inst, {7: "crate, cubic, pine"},
                                           captioner, ["f.jpg"])
    assert (text, src) == ("crate, cubic, pine", "manual")
    assert fields == {"category": "crate", "shape": "cubic", "material": "pine", "detail": ""}
    assert calls == []
    # 2. the stored description beats the on-demand captioner, which is NOT called
    text, fields, src = SE.resolve_caption(7, "wooden box", inst, {}, captioner, ["f.jpg"])
    assert (text, src) == (_STORED_OBJECT["caption"], "vlm_object")
    assert fields == {"category": "crate", "shape": "rectangular box", "material": "wood",
                      "detail": "hinged lid"}
    assert calls == []
    # 3. nothing stored: the on-demand captioner, with the masks it is handed, and the
    #    "captioning" phase for the dialog
    phases = []
    text, fields, src = SE.resolve_caption(
        7, "wooden box", {"id": 7, "label": "wooden box"}, None, captioner, ["f.jpg"],
        masks_fn=lambda: {"f.jpg": np.ones((2, 2), bool)},
        on_phase=lambda iid, ph: phases.append((iid, ph)))
    assert (text, src) == ("An on-demand description.", "vlm_on_demand")
    assert fields["category"] == "thing" and phases == [(7, "captioning")]
    assert len(calls) == 1 and calls[0][2] == "wooden box" and "f.jpg" in calls[0][1]
    # 4. nothing at all: the SAM3 label, no fields (nothing was described)
    assert SE.resolve_caption(7, "wooden box", {"id": 7}, None, None, []) == ("wooden box", None, "label")


def test_stored_object_description_beats_the_concept_one_and_fields_are_never_none():
    # a list of candidates: the object's own description over the concept's
    inst = {"id": 7, "shape_caption": [_STORED_CONCEPT, _STORED_OBJECT]}
    text, fields, src = SE.resolve_caption(7, "wooden box", inst, None, None, [])
    assert (text, src) == (_STORED_OBJECT["caption"], "vlm_object")
    # only the concept's: used, with the fields it lacks as empty strings
    inst = {"id": 7, "shape_caption": _STORED_CONCEPT}
    text, fields, src = SE.resolve_caption(7, "wooden box", inst, None, None, [])
    assert (text, src) == ("A wooden storage box.", "vlm_concept")
    assert fields == {"category": "box", "shape": "box", "material": "wood", "detail": ""}
    # an empty stored caption is no description
    assert SE.resolve_caption(7, "wooden box", {"shape_caption": {"caption": "  ", "source": "object"}},
                              None, None, [])[2] == "label"
    # a failing on-demand captioner leaves the label (declared, never a crash)
    def broken(*_a):
        raise RuntimeError("vLLM down")
    assert SE.resolve_caption(7, "wooden box", {}, None, broken, ["f.jpg"])[2] == "label"


def test_the_pkl_carries_the_stored_description_and_its_source(session):
    root, out, fr, seg, xyz = session
    seg["instances"][0]["shape_caption"] = _STORED_OBJECT
    called = []
    _, d = _export(root, out, fr, seg, caption_fn=lambda *a: called.append(a) or {"caption": "x"})
    assert d["caption"] == _STORED_OBJECT["caption"] and d["caption_source"] == "vlm_object"
    assert d["caption_fields"]["material"] == "wood" and d["category"] == "wooden box"
    assert called == []                                  # the stored description was enough


def test_the_on_demand_captioner_gets_the_sam3_masks_of_the_kept_views(session):
    root, out, fr, seg, xyz = session
    # a mask store in VIDEO frame space: instance 7 = segmentation.json masklets id 3
    # and 4 (fused), masked in the birth frames; the store declares its space
    from segmentation import mask_space
    (out / "segmentation.json").write_text(json.dumps({"instances": [
        {"id": 3, "instance_id": 7, "label": "wooden box"},
        {"id": 4, "instance_id": 7, "label": "wooden box"},
        {"id": 5, "instance_id": 8, "label": "other"}]}))
    store = {mask_space.NPZ_KEY: np.array(mask_space.SPACE_VIDEO)}
    for i in range(N_BIRTH_FRAMES):
        f = 10 * i + 3
        m = np.zeros((H // 2, W // 2), np.uint8)
        m[20:40, 10:30] = 1
        store[f"f{f}_o3"] = m
        m2 = np.zeros((H // 2, W // 2), np.uint8)
        m2[40:60, 10:30] = 1
        store[f"f{f}_o4"] = m2
        store[f"f{f}_o5"] = np.ones((H // 2, W // 2), np.uint8)
    np.savez_compressed(out / "seg_masks.npz", **store)
    mask_space.invalidate(out)
    got = {}

    def captioner(frames, masks, label):
        got["frames"], got["masks"] = frames, masks
        return {"caption": "Described from the masks.", "category": "box"}

    _, d = _export(root, out, fr, seg, caption_fn=captioner, max_views=16)
    assert d["caption_source"] == "vlm_on_demand"
    # every kept BIRTH view got the union of the instance's own masklets (not the other
    # object's full-frame mask, not the projected-point speckle); the view with no SAM3
    # mask (SEEN_ONLY_FRAME) simply has none
    birth = {f"{10 * i + 3:06d}.jpg" for i in range(N_BIRTH_FRAMES)}
    assert set(got["masks"]) == birth
    m = got["masks"][next(iter(birth))]
    assert m.shape == (H // 2, W // 2) and int(m.sum()) == 2 * 20 * 20
    assert f"{SEEN_ONLY_FRAME:06d}.jpg" not in got["masks"]


# ── the published GLB of a shape folder ──

def test_shape_list_prefers_the_glb_with_its_own_meta_sidecar(tmp_path):
    folder = tmp_path / "wooden_box_7"
    folder.mkdir()
    shaper = folder / "wooden_box_7.glb"
    visual = folder / "wooden_box_7_visual.glb"
    legacy = folder / "mesh.glb"
    for g in (legacy, visual, shaper):
        g.write_bytes(b"glb")
    (folder / "meta.json").write_text("{}")              # MeshFlow's folder-level sidecar
    files = sorted(folder.glob("*.glb"))
    # no <stem>.meta.json anywhere: the legacy MeshFlow choice
    assert SE.pick_shape_glb(files) == visual
    assert SE.pick_shape_glb([legacy]) == legacy
    # ShapeR's pair <stem>.glb + <stem>.meta.json wins over the _visual next to it
    (folder / "wooden_box_7.meta.json").write_text("{}")
    assert SE.pick_shape_glb(files) == shaper
    assert SE.pick_shape_glb([]) is None


# ── run_shaper_batch: the decimation budget is a config key, not a hardcode ──

def test_run_shaper_batch_takes_simplify_faces_from_the_command_line():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import run_shaper_batch as RB
    ap = RB.build_arg_parser()
    ns = ap.parse_args(["--pkls", "a.pkl", "--output_dir", "o", "--simplify_faces", "0"])
    assert ns.simplify_faces == 0
    ns = ap.parse_args(["--pkls", "a.pkl", "--output_dir", "o", "--simplify_faces", "250000"])
    assert ns.simplify_faces == 250000
    # a bare command line keeps the value the script hardcoded until 2026-10-01
    assert ap.parse_args(["--pkls", "a.pkl", "--output_dir", "o"]).simplify_faces == 125000
    with pytest.raises(SystemExit):                      # the old switch is gone
        ap.parse_args(["--pkls", "a.pkl", "--output_dir", "o", "--no_simplify"])
    assert "max" in RB.PRESETS and RB.PRESETS["max"][0] == 32


def test_a_camera_behind_a_wall_is_not_a_view_of_the_object():
    """Every posed keyframe is a candidate view (2026-10-01), so the occlusion test must hold:
    points behind the keyframe's own surface (a wall at 2 m, the object at 4 m) are not seen."""
    from segmentation.shaper_export import _project_pinhole, occluded_points
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    c2w = np.eye(4)
    pts = np.array([[0.0, 0.0, 4.0], [0.1, 0.1, 4.0], [0.2, -0.1, 1.5]])   # two behind, one in front
    u, v, inside = _project_pinhole(pts, c2w, K, 100, 80)
    D = np.full((80, 100), 2.0, np.float32)                               # the wall the camera sees
    occ = occluded_points(pts, c2w, u, v, inside, 100, 80, D, 0.05)
    assert inside.all() and occ.tolist() == [True, True, False]
    D0 = np.zeros((80, 100), np.float32)                                  # no depth: says nothing
    assert not occluded_points(pts, c2w, u, v, inside, 100, 80, D0, 0.05).any()
    assert not occluded_points(pts, c2w, u, v, inside, 100, 80, None, 0.05).any()
    D2 = np.full((40, 50), 2.0, np.float32)                               # a coarser depth grid maps by scale
    assert occluded_points(pts, c2w, u, v, inside, 100, 80, D2, 0.05).tolist() == [True, True, False]


def test_stray_data_is_taken_from_the_scans_own_directory_only(tmp_path):
    """Plan point 35 (the same fix as precision.camera and session_io): a sibling scan's Stray
    calibration is another recording's — never read; the scan dir, then its stray/ subdir."""
    scan = tmp_path / "scans" / "src_default"
    sib = tmp_path / "scans" / "src_other"
    for d in (scan, sib):
        d.mkdir(parents=True)
    (sib / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
    (sib / "camera_matrix.csv").write_text("1,0,0\n0,1,0\n0,0,1\n")
    assert SE._find_stray_dir(scan) is None
    (scan / "stray").mkdir()
    (scan / "stray" / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
    (scan / "stray" / "camera_matrix.csv").write_text("1,0,0\n0,1,0\n0,0,1\n")
    assert SE._find_stray_dir(scan) == scan / "stray"
    (scan / "odometry.csv").write_text("timestamp,frame,x,y,z,qx,qy,qz,qw\n")
    (scan / "camera_matrix.csv").write_text("1,0,0\n0,1,0\n0,0,1\n")
    assert SE._find_stray_dir(scan) == scan
