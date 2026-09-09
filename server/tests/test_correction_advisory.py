"""Advisory gates (USER 2026-09-09): gates are measured and reported, the
correction is applied anyway; the visual Approve/Undo is the verdict."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_objects, run_floor                # noqa: E402
from correction.session import read_ply                          # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _floor_y_err(scene):
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    y = data["y"].astype(np.float64)
    rev = np.isin(scene.ks, list(scene.gt["revisit_kfs"])) & scene.floor_mask
    return float(np.median(np.abs(y[rev] - scene.xyz_true[rev, 1])))


def test_advisory_mode_applies_with_warnings(tmp_path):
    """USER 2026-09-09: gates never block — they are reported. A run whose
    continuity gate fails in veto mode is APPLIED in advisory mode with the
    failed gates flagged."""
    scene = build_scene(tmp_path, drift_yaw_deg=1.0,
                        drift_t=(0.30, 0.0, 0.10))
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg(**{"gates.mode": "advisory",
                                                 "gates.max_step_mm": 5.0}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    assert any("continuity" in w for w in rep["warnings"]), rep["warnings"]
    g = next(g for g in rep["gates"] if g["name"] == "continuity")
    assert g["advisory"] is True and g["passed"] is False
    assert (scene.output_dir / "geometry_epoch.json").exists()


def test_floor_alignment_advisory_applies(tmp_path):
    scene = build_scene(tmp_path, floor="ramp", floor_slope=0.04)
    rep = run_floor(scene.output_dir, "level", None, "test",
                    cfg=make_correction_cfg(**{"gates.mode": "advisory",
                                               "gates.max_step_mm": 1.0}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    assert rep["warnings"]
