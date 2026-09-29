"""ShapeR input packages (segmentation/shaper_export.py) on a precision session
(USER 2026-09-29: ShapeR replaces MeshFlow behind the generative button).

A synthetic box seen by keyframes on a circle: camera.json (F5 camera, OPENCV,
no lens distortion, a half-resolution traceability grid), camera_poses.txt /
camera_frames.txt, native frames, a cleaned cloud with provenance, one segmented
instance, orientation baked (+Y up). Checks the package the vendor consumes:
- one PKL per object, named by its folder (the runner reports by PKL stem);
- Fisheye624 whose radial polynomial IS the pinhole (the vendor rectifies every
  view Fisheye624 → pinhole; k = 0 is equidistant and warped 27 % at the corner);
- the object's projections land where the pinhole camera puts its points (exact
  traceability-grid → native mapping, no resolution guess);
- Z-up model frame from the baked +Y, metric scale kept (bounds = half extents).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precision.camera import CameraModel, GridMap, save_camera_json
from segmentation import shaper_export as SE

W, H, F = 232, 416, 196.0          # native frame (portrait, like the videos)
GRID_W, GRID_H = 116, 208           # the traceability (Omega record) grid: half resolution


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
    for i, ang in enumerate(np.linspace(0, np.pi, 8)):
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
    (out / "camera_frames.txt").write_text("\n".join(map(str, frames)) + "\n")
    np.savetxt(out / "camera_poses.txt", np.array(poses).reshape(len(poses), 16))
    (out / ".orientation_applied").write_text("baked\n")
    xyz = np.concatenate(xyz)
    _write_ply(out / "cleaned_cloud.ply", xyz, np.concatenate(fg), np.concatenate(pr), np.concatenate(pc))
    seg = {"instances": [{"id": 7, "instance_id": 7, "label": "wooden box",
                          "globalIndices": list(range(len(xyz)))}]}
    return tmp_path, out, fr, seg, xyz


def test_the_package_is_what_the_vendor_consumes(session):
    root, out, fr, seg, xyz = session
    pkls = SE.export_shaper_pkls(out, fr, seg, session_dir=root, max_views=8, min_view_points=20,
                                 captions={7: "a small wooden box with a lid"})
    assert len(pkls) == 1
    pkl = pkls[0]
    assert pkl.name == "wooden_box_7.pkl" and pkl.parent.name == "wooden_box_7"
    d = pickle.loads(pkl.read_bytes())
    assert d["caption"] == "a small wooden box with a lid" and d["instance_id"] == 7
    n_v = d["n_views"]
    assert 2 <= n_v <= 8 and len(d["image_data"]) == n_v == d["camera_params"].shape[0]
    # Z-up model frame from the baked +Y, metric: the box is 50 cm tall → half-extent 0.25 on Z
    b = d["bounds"].numpy()
    assert abs(b[2] - 0.25) < 0.02 and abs(b[0] - 0.20) < 0.02


def test_fisheye624_is_the_pinhole_and_the_projections_land_on_it(session):
    root, out, fr, seg, xyz = session
    d = pickle.loads(SE.export_shaper_pkls(out, fr, seg, session_dir=root, max_views=8,
                                           min_view_points=20, captions={7: "box"})[0].read_bytes())
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
        # the stored projections (traceability grid → native, exact) sit where the camera
        # model puts the points — within the half-resolution grid's quantisation
        assert np.median(np.hypot(uv[:, 0] - u, uv[:, 1] - vv)) < 1.0
        img = Image.open(__import__("io").BytesIO(d["image_data"][v]))
        assert img.size == (W, H) and img.mode == "L"          # native, grayscale


def test_no_camera_json_falls_back_without_the_precision_grid(session):
    root, out, fr, seg, xyz = session
    src = SE._load_precision_source(out)
    assert src is not None and src.backend == "precision" and src.grid is not None
    (out / "camera.json").unlink()
    assert SE._load_precision_source(out) is None
