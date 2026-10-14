"""The floor UNDER THE CAMERA in the chunk check and the floor metric (USER 2026-10-08: a session can
hold several floors; the chunk check and the floor metric use the certification's own choice of the
floor of every chunk, correction.floor.chunk_floor_levels, measured on the published cloud).

A synthetic platform (y = 0, x <= 0.6 m) beside a track bed one metre below (x > 0.6 m); the walk
looks down the platform and sees the track bed in the right of every frame.

    walk on the platform        → the floor is the platform: camera 1.5 m over it — never the 2.5 m
                                  over the track bed, the lowest surface every frame sees
    last chunk on the track bed → two floor surfaces, each measured on its own plane
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from precision import chunk_check as CC                       # noqa: E402

W, H, F = 64, 96, 60.0
CX, CY = 32.0, 48.0
N_KF, PER_CHUNK, STEP_M = 30, 10, 0.1
R0 = np.diag([1.0, -1.0, -1.0])                               # OpenCV camera → world (+Y up), looks along −Z
EDGE_X, TRACK_Y, CAM_OVER_FLOOR = 0.6, -1.0, 1.5
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])


def render(cam: np.ndarray, ceil_y: float = 3.0, wall_ahead_m: float = 6.0) -> np.ndarray:
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u)], -1) @ R0.T     # world directions
    t = np.full((H, W), wall_ahead_m)
    down = d[..., 1] < -1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        tp = np.where(down, (0.0 - cam[1]) / d[..., 1], np.inf)               # the platform's top
        tt = np.where(down, (TRACK_Y - cam[1]) / d[..., 1], np.inf)           # the track bed
        tf = np.where(np.abs(d[..., 0]) > 1e-9, (EDGE_X - cam[0]) / d[..., 0], np.inf)   # the edge face
    xp, xt = cam[0] + tp * d[..., 0], cam[0] + tt * d[..., 0]
    yf = cam[1] + tf * d[..., 1]
    on_platform = down & (tp > 0) & (xp <= EDGE_X)
    on_face = down & ~on_platform & (xt < EDGE_X) & (tf > 0) & (yf < 0.0) & (yf > TRACK_Y)
    on_track = down & ~on_platform & ~on_face & (tt > 0)
    hit = np.where(on_platform, tp, np.where(on_face, tf, np.where(on_track, tt, np.inf)))
    t = np.minimum(t, hit)
    up = d[..., 1] > 1e-9
    t[up] = np.minimum(t[up], (ceil_y - cam[1]) / d[..., 1][up])
    return t                                                  # camera z = t (d_cam z = 1)


def _write_cloud(path: Path, xyz: np.ndarray, frame: np.ndarray) -> None:
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4")])
    rec["x"], rec["y"], rec["z"], rec["frame_global"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], frame
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(rec)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property int frame_global\nend_header\n").encode("ascii")
    path.write_bytes(head + rec.tobytes())


def build_session(tmp_path: Path, track_chunk: int = -1) -> np.ndarray:
    """Three Omega chunks of ten keyframes walking down the platform at x = 0; with
    ``track_chunk`` that chunk walks on the track bed instead (x = 2, the same 1.5 m over it)."""
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    (out / "da3_run" / "results_output").mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = [10 * i + 1 for i in range(N_KF)]
    poses, chain, pts, fids = [], [], [], []
    for i, f in enumerate(frames):
        chunk = i // PER_CHUNK
        on_track = chunk == track_chunk
        cam = np.array([2.0 if on_track else 0.0, (TRACK_Y if on_track else 0.0) + CAM_OVER_FLOOR,
                        -STEP_M * i])
        depth = render(cam)
        omega = depth * (1.0 + 0.005 * rng.standard_normal(depth.shape))
        da3 = depth * 0.92 * (1.0 + 0.01 * rng.standard_normal(depth.shape))
        np.savez(out / "omega_run" / "results_output" / f"frame_{f}.npz", depth=omega.astype(np.float32),
                 conf=np.ones_like(omega, np.float32), chunk=np.int64(chunk))
        np.savez(out / "da3_run" / "results_output" / f"frame_{f}.npz", depth=da3.astype(np.float32),
                 conf=np.ones_like(da3, np.float32), intrinsics=K)
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = cam
        poses.append(T); chain.append(STEP_M * i)
        X = CC.unproject(depth, K, T, 1)
        pts.append(X); fids.append(np.full(len(X), f, np.int32))
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    np.savetxt(out / "camera_poses.txt", np.array(poses).reshape(N_KF, 16))
    from precision.camera import grid_full_frame_resize
    (out / "camera.json").write_text(json.dumps({"width": W, "height": H, "model": "OPENCV",
                                                  "params": [F, F, CX, CY, 0, 0, 0, 0], "camera_epoch": 1,
                                                  "source": "omega",
                                                  "omega_grid": grid_full_frame_resize(W, H, W, H, "omega").to_dict()}))
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 2}))
    _write_cloud(out / "cleaned_cloud.ply", np.concatenate(pts), np.concatenate(fids))
    return np.array(chain)


@pytest.fixture(scope="module")
def pcfg():
    from precision.config import load_precision_config
    p = load_precision_config()
    return replace(p, chunk_check=replace(p.chunk_check, pixel_stride=1, min_points=50, bootstrap=200))


def test_the_floor_is_the_one_under_the_camera_not_the_lowest_surface_seen(tmp_path, pcfg):
    chain = build_session(tmp_path)
    rep = CC.run_check(tmp_path, pcfg, log=lambda m: None, chainage=chain)
    fl = rep["floor_levels"]
    assert set(fl["surfaces"].values()) == {fl["surfaces"]["0"]} and len(fl["surfaces"]) == 3
    for c in fl["per_chunk"]:
        chosen = [lv for lv in c["levels"] if lv["rank"] == c["chosen_rank"]][0]
        assert abs(chosen["floor_y_m"]) < 0.02, c                     # the platform, never the track bed
        real_y = [lv["floor_y_m"] for lv in c["levels"] if lv["real"]]
        assert any(abs(y - TRACK_Y) < 0.02 for y in real_y), c        # the track bed is a real level too
    assert abs(rep["session"]["cf_omega_m"] - CAM_OVER_FLOOR) < 0.05, rep["session"]
    assert "under the camera" in rep["plane"]["floor_band"]
    # without the published cloud the chunk floors are not measured (declared): each keyframe's
    # lowest band is its floor — the track bed, 2.5 m under the camera
    (tmp_path / "output" / "cleaned_cloud.ply").unlink()
    old = CC.run_check(tmp_path, pcfg, log=lambda m: None, chainage=chain)
    assert old["floor_levels"] is None and "lowest band" in old["plane"]["floor_band"]
    assert abs(old["session"]["cf_omega_m"] - (CAM_OVER_FLOOR - TRACK_Y)) < 0.05, old["session"]


def test_two_floor_surfaces_are_each_measured_on_their_own_plane(tmp_path, pcfg):
    build_session(tmp_path, track_chunk=2)
    from precision.cloud_metrics import run_cloud_metrics
    rep = run_cloud_metrics(tmp_path, pcfg, stage="test", log=lambda m: None)
    fs = rep["floor"]["floor_surfaces"]
    assert fs["n"] == 2, rep["floor"]
    assert fs["chunks"]["0"] == fs["chunks"]["1"] != fs["chunks"]["2"]
    for s, m in fs["per_surface"].items():
        assert m["cells"] > 0, (s, m)
        assert abs(m["cameras_above_floor_m"]["median"] - CAM_OVER_FLOOR) < 0.05, (s, m)
    assert rep["floor"]["cells"] == max(m["cells"] for m in fs["per_surface"].values())
