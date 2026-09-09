"""Observability: the solve never touches a DOF the evidence does not
observe, and declares the unrestrained ones."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.ledger import load_epoch_npz                 # noqa: E402
from correction.run import run_objects                       # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def test_planar_only_solves_normal_translation(tmp_path):
    """Only the wall marked: translation along its normal (z) is observable;
    yaw and in-plane translation stay identity and are DECLARED."""
    # drift purely along the wall normal (z) so the observable part fully
    # explains the copies
    scene = build_scene(tmp_path, drift_yaw_deg=0.0,
                        drift_t=(0.0, 0.0, 0.12))
    # bounded rule disabled (tolerance ~0): the strict planar DOF applies
    rep = run_objects(scene.output_dir, [1], "test",
                      cfg=make_correction_cfg(
                          **{"observability.bounded_extent_tol": 1e-9}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    obs = rep["observability"][0]
    assert obs["dof"] == ["t_normal"]
    assert "yaw" in obs["unrestrained"]
    sol = rep["solutions"][0]
    assert sol["rot_deg"] == 0.0
    npz = load_epoch_npz(scene.output_dir, 1)
    rev = scene.gt["revisit_kfs"]
    t = npz["t_kf"][rev[-1]]
    # translation is (anti)parallel to the wall normal z: x/y components ~0
    assert abs(t[0]) < 0.02 and abs(t[1]) < 0.02, t
    assert abs(t[2] + 0.12) < 0.03, t
    R = npz["R_kf"][rev[-1]]
    assert np.allclose(R, np.eye(3)), "yaw must stay identity"


def test_cylinder_only_no_along_axis(tmp_path):
    """Only the column marked (vertical axis): the along-axis (y) component
    is unobservable and stays 0 even though the injected drift has one.
    (The unobservable y part is kept small: a LARGE unobservable drift is
    correctly VETOED by the scene exam — the floor witness stays broken —
    which test_correction_gates covers.)"""
    scene = build_scene(tmp_path, drift_yaw_deg=0.0,
                        drift_t=(0.10, 0.04, 0.05))
    rep = run_objects(scene.output_dir, [3], "test",
                      cfg=make_correction_cfg(
                          **{"observability.bounded_extent_tol": 1e-9}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    obs = rep["observability"][0]
    assert obs["dof"] == ["t_perp_axis(2)"]
    assert "t_along_axis" in obs["unrestrained"]
    npz = load_epoch_npz(scene.output_dir, 1)
    t = npz["t_kf"][scene.gt["revisit_kfs"][-1]]
    assert abs(t[1]) < 0.02, f"along-axis (y) component must be 0, got {t}"
    assert abs(t[0] + 0.10) < 0.04 and abs(t[2] + 0.05) < 0.04, t


def test_no_depth_without_evidence(tmp_path):
    """A single object cannot propose k, whatever the compression."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.0,
                        drift_t=(0.0, 0.0, 0.05), depth_c=0.9)
    rep = run_objects(scene.output_dir, [1], "test",
                      cfg=make_correction_cfg())
    # whatever the verdict, no k may be proposed
    for d in rep.get("diagnosis") or []:
        assert not d["depth_needed"], d
    if rep["status"] == "pending":
        assert rep["distribution"]["depth_keyframes"] == 0


def test_bounded_planar_object_observes_full_translation(tmp_path):
    """USER 2026-09-08 (pccr desk1): a lone desk was solved along its
    normal only. Two copies with the SAME supported extents are a bounded,
    equally-covered object → full translation (yaw stays free)."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.0,
                        drift_t=(0.30, 0.0, 0.12))
    rep = run_objects(scene.output_dir, [1], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    obs = rep["observability"][0]
    assert obs["dof"] == ["tx", "ty", "tz"] and obs["unrestrained"] == ["yaw"]
    npz = load_epoch_npz(scene.output_dir, 1)
    t = npz["t_kf"][scene.gt["revisit_kfs"][-1]]
    # the IN-PLANE (x) drift is recovered too, not only the normal (z)
    assert abs(t[0] + 0.30) < 0.04 and abs(t[2] + 0.12) < 0.04, t
