"""precision.poses_epoch: F2 / F5 publish poses only — no cloud exists before F7
(USER 2026-09-29: "¿por qué filtramos una nube que no vamos a usar todavía?")."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _session(tmp_path: Path, n: int = 5) -> Path:
    out = tmp_path / "sess" / "output"
    (out / "omega_run").mkdir(parents=True)
    frames = [10 * i for i in range(n)]
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n, dtype=float)          # cameras along x
    rows = [" ".join(f"{v:.10g}" for v in p.ravel()) for p in poses]
    (out / "camera_poses.txt").write_text("\n".join(rows) + "\n")
    (out / "omega_run" / "camera_poses.txt").write_text("\n".join(rows) + "\n")   # a copy
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    return out


def test_publishes_poses_without_any_cloud(tmp_path):
    from precision.poses_epoch import apply_pose_epoch, load_poses
    out = _session(tmp_path)
    assert not (out / "cleaned_cloud.ply").exists()
    frames, poses = load_poses(out)
    n = len(frames)
    R = np.tile(np.eye(3), (n, 1, 1))
    t = np.zeros((n, 3)); t[:, 2] = 0.5                  # everything 50 cm along z
    k = np.full(n, 1.02)
    res = apply_pose_epoch(out, R, t, k, "gauge", [{"stage": "gauge"}], log=lambda *a: None)
    assert res["epoch_to"] == 1
    assert json.loads((out / "geometry_epoch.json").read_text())["epoch"] == 1
    _, new = load_poses(out)
    assert np.allclose(new[:, :3, 3] - poses[:, :3, 3], [0.0, 0.0, 0.5])
    # the copy moved with the canonical file
    from correction.session import read_poses
    assert np.allclose(read_poses(out / "omega_run" / "camera_poses.txt"), new)
    # the exact transform is replayable, the depth k recorded, nothing about a cloud
    assert (out / "corrections" / "epoch_1.npz").exists()
    dc = json.loads((out / "depth_correction.json").read_text())
    assert dc["epoch"] == 1 and dc["k"][str(frames[0])] == pytest.approx(1.02)
    assert not (out / "cleaned_cloud.ply").exists() and not (out / "potree").exists()
    assert abs(res["cameras_moved_max_m"] - 0.5) < 1e-9


def test_two_epochs_compose_and_the_previous_one_is_kept_for_the_discard(tmp_path):
    from precision.poses_epoch import apply_pose_epoch, load_poses
    out = _session(tmp_path)
    n = len(load_poses(out)[0])
    R = np.tile(np.eye(3), (n, 1, 1))
    apply_pose_epoch(out, R, np.zeros((n, 3)), np.full(n, 1.1), "gauge", [], log=lambda *a: None)
    apply_pose_epoch(out, R, np.zeros((n, 3)), np.full(n, 1.1), "refine", [], log=lambda *a: None)
    assert json.loads((out / "geometry_epoch.json").read_text())["epoch"] == 2
    dc = json.loads((out / "depth_correction.json").read_text())
    assert list(dc["k"].values())[0] == pytest.approx(1.21)      # 1.1 × 1.1, cumulative
    assert (out / "_epoch_1").is_dir()                            # what map_worker discards


def test_refuses_inconsistent_shapes(tmp_path):
    from precision.poses_epoch import PosesEpochError, apply_pose_epoch
    out = _session(tmp_path, n=3)
    with pytest.raises(PosesEpochError, match="do not match the 3 keyframes"):
        apply_pose_epoch(out, np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 3)), np.ones(2),
                         "gauge", [], log=lambda *a: None)
