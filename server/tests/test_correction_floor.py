"""Floor models: plane keeps a real slope; level flattens it (declared);
a real step between anchors is preserved (step-demote)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_floor                         # noqa: E402
from correction.session import read_ply                      # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _floor_xy(scene):
    _, data = read_ply(scene.output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    m = scene.floor_mask
    return xyz[m, 0], xyz[m, 1]


def _fit_slope(x, y):
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])


def test_plane_keeps_ramp_slope_and_removes_drift(tmp_path):
    slope = 0.04
    scene = build_scene(tmp_path, floor="ramp", floor_slope=slope,
                        drift_yaw_deg=0.0, drift_t=(0.0, 0.10, 0.0))
    rep = run_floor(scene.output_dir, "plane", None, "test",
                    cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    x, y = _floor_xy(scene)
    got = _fit_slope(x, y)
    # the reference plane is fitted on the DRIFTED floor (the y-drift ramps
    # linearly along the walk → a small extra tilt is inherent); the test
    # asserts the slope SURVIVES — level would return ~0
    assert abs(got - slope) < 0.015, f"slope {slope} → {got} (flattened?)"
    assert got > 0.025, f"slope must survive, got {got}"
    # drift removed: residual of the floor vs its own fitted line is small
    resid = y - (_fit_slope(x, y) * x + np.median(y - got * x))
    assert float(np.median(np.abs(resid))) < 0.02


def test_level_flattens_by_design(tmp_path):
    slope = 0.04
    scene = build_scene(tmp_path, floor="ramp", floor_slope=slope)
    rep = run_floor(scene.output_dir, "level", None, "test",
                    cfg=make_correction_cfg(**{"gates.max_step_mm": 120.0}))
    assert rep["status"] == "applied", rep.get("rejection_reason")
    x, y = _floor_xy(scene)
    got = _fit_slope(x, y)
    assert abs(got) < 0.01, f"level must flatten: slope {got}"
    assert abs(float(np.median(y))) < 0.03, "floor must sit at y=0"
    assert rep["diagnosis"][0]["model"] == "level"


def test_step_is_preserved_by_demotion(tmp_path):
    scene = build_scene(tmp_path, floor="step",
                        drift_t=(0.0, 0.0, 0.0))
    rep = run_floor(scene.output_dir, "plane", None, "test",
                    cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    # platform keyframes demoted as a REAL level change
    per_kf = rep["solutions"][0]["per_kf_report"]
    demoted_step = [e for e in per_kf
                    if e.get("role") == "demoted"
                    and "step" in str(e.get("why", ""))]
    assert demoted_step, "platform anchors must be step-demoted"
    x, y = _floor_xy(scene)
    main = y[x < 5.0]
    plat = y[x > 8.0]
    gap = float(np.median(plat) - np.median(main))
    assert abs(gap - 0.5) < 0.05, f"real step must survive: {gap:.3f} m"


def test_the_tilt_bar_is_measured_not_a_constant():
    src = (Path(__file__).resolve().parents[1] / "correction" / "floor.py").read_text()
    assert "sigma_tilt" in src, "the session's own normal scatter must be measured"
    i = src.index("sigma_tilt = ")
    j = src.index("_tilt_bar = ")
    assert i < j, "the scatter must be measured before the bar is built"
    assert "max(float(cfg.floor.min_tilt_deg)" in src, \
        "min_tilt_deg must be the FLOOR of the bar, not the bar"
