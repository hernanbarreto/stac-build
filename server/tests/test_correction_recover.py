"""Recovery: the correction returns the copies to the reference within
tolerance and recovers an injected depth compression k."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_objects, run_verdict          # noqa: E402
from correction.session import read_ply, read_poses          # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _revisit_obj_mask(scene):
    m = np.zeros(len(scene.xyz_true), bool)
    rev = set(scene.gt["revisit_kfs"])
    m[np.isin(scene.ks, list(rev)) & ~scene.floor_mask] = True
    return m


def _median_err(scene):
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    m = _revisit_obj_mask(scene)
    return float(np.median(np.linalg.norm(xyz[m] - scene.xyz_true[m],
                                          axis=1)))


def test_rigid_recovery(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=2.0, drift_t=(0.25, 0.0, 0.10))
    before = _median_err(scene)
    assert before > 0.10, "injection sanity"
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    after = _median_err(scene)
    assert after < 0.03, f"copies not recovered: {before:.3f} → {after:.3f} m"
    # cameras follow: revisit camera centres back at the true trajectory
    poses = read_poses(scene.output_dir / "camera_poses.txt")
    for k in scene.gt["revisit_kfs"]:
        err = np.linalg.norm(poses[k][:3, 3] - scene.poses_true[k][:3, 3])
        assert err < 0.05, f"kf {k} camera off by {err:.3f} m"
    run_verdict(scene.output_dir, "approved", "test")


def test_depth_recovery(tmp_path):
    c = 0.90
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=c)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    diag = rep["diagnosis"][0]
    assert diag["depth_needed"], diag
    assert abs(diag["k"] - 1.0 / c) < 0.03, diag
    assert _median_err(scene) < 0.03


def test_both_recovery(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5,
                        drift_t=(0.20, 0.0, 0.08), depth_c=0.92)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    assert _median_err(scene) < 0.035
    # continuity: reported max step within the gate
    assert rep["distribution"]["max_step_between_keyframes_mm"] <= 60.0
