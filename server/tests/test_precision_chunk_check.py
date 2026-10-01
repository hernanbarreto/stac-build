"""precision/chunk_check.py on a synthetic room — floor y = 0, ceiling y = 3, a far wall,
30 keyframes along a walk at 1.3 m in three Omega chunks; Omega = the rendered depth
(+1 % noise, the order of Omega's per-frame depth noise on pccr), DA3 = the same ×0.92 (its own global bias) +3 % noise.

    clean                         → every chunk ok, nothing to correct
    chunk 1 Omega depth × 1.15    → depth, factor ≈ 1/1.15 (route: scale about the camera)
    chunk 2 poses +14 cm          → pose (floor AND ceiling jump together at the seam)
    a real 14 cm step under chunk 2 → level (the floor jumps, the ceiling does not)
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


def render(cam: np.ndarray, floor_y: float, ceil_y: float = 3.0, wall_ahead_m: float = 6.0) -> np.ndarray:
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - CX) / F, (v - CY) / F, np.ones_like(u)], -1) @ R0.T     # world directions
    t = np.full((H, W), wall_ahead_m)
    down = d[..., 1] < -1e-9
    t[down] = np.minimum(t[down], (floor_y - cam[1]) / d[..., 1][down])
    up = d[..., 1] > 1e-9
    t[up] = np.minimum(t[up], (ceil_y - cam[1]) / d[..., 1][up])
    return t                                                  # camera z = t (d_cam z = 1)


def build_session(tmp_path: Path, case: str) -> tuple:
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    (out / "da3_run" / "results_output").mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = [10 * i + 1 for i in range(N_KF)]
    poses, chain = [], []
    K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])
    for i, f in enumerate(frames):
        chunk = i // PER_CHUNK
        floor_y, cam_y = 0.0, 1.3
        if case == "level" and chunk == 2:
            floor_y, cam_y = -0.14, 1.16                      # the person keeps their height over a lower floor
        cam = np.array([0.0, cam_y, -STEP_M * i])
        depth = render(cam, floor_y)
        omega = depth * (1.0 + 0.01 * rng.standard_normal(depth.shape))
        if case == "depth" and chunk == 1:
            omega = omega * 1.15
        da3 = depth * 0.92 * (1.0 + 0.03 * rng.standard_normal(depth.shape))
        np.savez(out / "omega_run" / "results_output" / f"frame_{f}.npz", depth=omega.astype(np.float32),
                 conf=np.ones_like(omega, np.float32), chunk=np.int64(chunk))
        np.savez(out / "da3_run" / "results_output" / f"frame_{f}.npz", depth=da3.astype(np.float32),
                 conf=np.ones_like(da3, np.float32), intrinsics=K)
        pose_cam = cam.copy()
        if case == "pose" and chunk == 2:
            pose_cam[1] += 0.14                               # the pose file says higher than the truth
        T = np.eye(4); T[:3, :3] = R0; T[:3, 3] = pose_cam
        poses.append(T); chain.append(STEP_M * i)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    np.savetxt(out / "camera_poses.txt", np.array(poses).reshape(N_KF, 16))
    (out / "camera.json").write_text(json.dumps({"width": W, "height": H, "model": "OPENCV",
                                                  "params": [F, F, CX, CY, 0, 0, 0, 0], "camera_epoch": 1,
                                                  "omega_grid": {"scale_x": 1.0, "crop_x": 0, "pad_left": 0}}))
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 2}))
    return frames, np.array(chain)


@pytest.fixture(scope="module")
def pcfg():
    from precision.config import load_precision_config
    p = load_precision_config()
    return replace(p, chunk_check=replace(p.chunk_check, pixel_stride=1, min_points=50, bootstrap=200))


def _run(tmp_path, pcfg, case):
    _, chain = build_session(tmp_path, case)
    rep = CC.run_check(tmp_path, pcfg, log=lambda m: None, chainage=chain)
    by = {c["chunk"]: c for c in rep["chunks"]}
    seams = {(s["left_chunk"], s["right_chunk"]): s for s in rep["seams"]}
    return rep, by, seams


def test_clean_room_is_ok_everywhere(tmp_path, pcfg):
    rep, by, seams = _run(tmp_path, pcfg, "clean")
    assert [by[k]["verdict"] for k in range(3)] == ["ok", "ok", "ok"], [by[k]["why"] for k in range(3)]
    assert rep["to_correct"] == []
    assert all(s["verdict"] == "ok" for s in seams.values())
    assert abs(rep["session"]["cf_omega_m"] - 1.3) < 0.02
    assert (tmp_path / "output" / "precision" / CC.CHECK_NAME).exists()


def test_long_depth_in_one_chunk_is_depth_with_its_factor(tmp_path, pcfg):
    rep, by, _ = _run(tmp_path, pcfg, "depth")
    assert by[1]["verdict"] == "depth", by[1]["why"]
    assert by[0]["verdict"] != "depth" and by[2]["verdict"] != "depth"
    fix = [c for c in rep["to_correct"] if c["chunk"] == 1]
    assert len(fix) == 1 and fix[0]["kind"] == "scale_about_camera"
    assert abs(fix[0]["factor"] - 1 / 1.15) < 0.03 * (1 / 1.15)


def test_cameras_off_vertically_is_pose_the_ceiling_moves_with_the_floor(tmp_path, pcfg):
    rep, by, seams = _run(tmp_path, pcfg, "pose")
    assert seams[(1, 2)]["verdict"] == "pose", seams[(1, 2)]
    assert abs(seams[(1, 2)]["floor_jump_m"] - 0.14) < 0.03
    assert abs(seams[(1, 2)]["ceiling_jump_m"] - 0.14) < 0.03
    assert by[2]["verdict"] == "pose", by[2]["why"]
    fix = [c for c in rep["to_correct"] if c["chunk"] == 2]
    assert len(fix) == 1 and fix[0]["kind"] == "vertical_pose" and abs(fix[0]["delta_m"] + 0.14) < 0.03


def test_a_real_step_is_level_the_ceiling_stays(tmp_path, pcfg):
    rep, by, seams = _run(tmp_path, pcfg, "level")
    assert seams[(1, 2)]["verdict"] == "level", seams[(1, 2)]
    assert abs(seams[(1, 2)]["floor_jump_m"] + 0.14) < 0.03
    assert by[2]["verdict"] == "level", by[2]["why"]
    assert rep["to_correct"] == []


def test_config_and_runner_step():
    from precision.config import load_precision_config
    from precision import runner as RN
    cc = load_precision_config().chunk_check
    assert 0 < cc.low_pct < 50 < cc.high_pct <= 100 and 0.5 <= cc.confidence < 1
    keys = [s.key for s in RN.STEPS]
    # after F5 comes the depth stage: f6_bend (the product since 2026-10-01) or f6_sweep (+ f7)
    assert keys[-1] == "f6_check" and keys.index("f6_bend") == keys.index("f5_refine") + 1
    assert keys.index("f6_sweep") == keys.index("f6_bend") + 1
    assert keys.index("f7_cloud") > keys.index("f6_sweep")
    assert not next(s for s in RN.STEPS if s.key == "f6_check").gpu
    pcfg = load_precision_config()
    chain = [s.key for s in RN.chain_steps(pcfg)]
    assert chain[-1] == "f6_check"
    if pcfg.cloud.source == "omega_bent":
        assert "f6_bend" in chain and not {"f6_sweep", "f7_cloud", "f7_fuse"} & set(chain)
    else:
        assert "f6_bend" not in chain and ("f7_cloud" in chain) != ("f7_fuse" in chain)
        if "f7_cloud" in chain:
            assert chain.index("f7_cloud") == chain.index("f6_sweep") + 1
