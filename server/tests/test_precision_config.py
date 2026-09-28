"""reconstruction.precision: a missing/out-of-range key fails at load naming
it; the real config.yaml loads; no decision literal lives in the precision or
intake packages outside their config.py (static scan, the
test_correction_config.py pattern)."""

import ast
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.config import PrecisionConfigError, load_precision_config   # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
PACKAGES = {"precision": SERVER / "precision", "intake": SERVER / "intake"}


def _raw():
    with open(SERVER / "config.yaml") as f:
        return yaml.safe_load(f)


def test_real_config_loads():
    cfg = load_precision_config(_raw())
    assert cfg.camera.model == "OPENCV"
    assert cfg.camera.init_from in ("auto", "omega", "stray")
    assert cfg.runner.heartbeat_s > 0


def test_missing_key_names_the_key():
    raw = _raw()
    del raw["reconstruction"]["precision"]["camera"]["fx_spread_warn_pct"]
    with pytest.raises(PrecisionConfigError, match="camera.fx_spread_warn_pct"):
        load_precision_config(raw)


@pytest.mark.parametrize("key", ["undistort_max_iter", "undistort_eps_px",
                                 "undistort_roundtrip_ulps"])
def test_undistort_solver_keys_are_mandatory_and_bounded(key):
    raw = _raw()
    cam = raw["reconstruction"]["precision"]["camera"]
    assert key in cam
    missing = copy.deepcopy(raw)
    del missing["reconstruction"]["precision"]["camera"][key]
    with pytest.raises(PrecisionConfigError, match=f"camera.{key}"):
        load_precision_config(missing)
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["camera"][key] = 0
    with pytest.raises(PrecisionConfigError, match=f"camera.{key}"):
        load_precision_config(bad)
    cfg = load_precision_config(raw).camera
    assert cfg.undistort_max_iter == cam["undistort_max_iter"]
    assert isinstance(cfg.undistort_max_iter, int)
    assert cfg.undistort_eps_px == cam["undistort_eps_px"]
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["camera"]["undistort_max_iter"] = 2.5
    with pytest.raises(PrecisionConfigError, match="camera.undistort_max_iter"):
        load_precision_config(bad)


def test_missing_section_fails():
    with pytest.raises(PrecisionConfigError, match="reconstruction.precision"):
        load_precision_config({"reconstruction": {}})


def test_out_of_range_and_enum_name_the_key():
    raw = _raw()
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["camera"]["init_from"] = "banana"
    with pytest.raises(PrecisionConfigError, match="camera.init_from"):
        load_precision_config(bad)
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["runner"]["heartbeat_s"] = 0
    with pytest.raises(PrecisionConfigError, match="runner.heartbeat_s"):
        load_precision_config(bad)
    bad = copy.deepcopy(raw)
    bad["reconstruction"]["precision"]["enabled"] = "yes"
    with pytest.raises(PrecisionConfigError, match="enabled"):
        load_precision_config(bad)


# ── static literal scan ──────────────────────────────────────────────────
# Decision thresholds live in config.yaml only. Flagged: FLOAT literals in the
# packages (outside config.py) that are not identity / epsilon / unit
# conversion values. 0.6744897501960817 is Φ⁻¹(3/4) (MAD → σ), a property of
# the normal distribution. Integers are indices, counts and shapes.
_FLOAT_WHITELIST = {0.0, 1.0, -1.0, 2.0, 0.5, 1e-9, 1e-6, 1e-12, 100.0, 1000.0,
                    0.6744897501960817}


def _float_literals(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value not in _FLOAT_WHITELIST:
                yield node.lineno, node.value


@pytest.mark.parametrize("pkg", sorted(PACKAGES))
def test_no_decision_literals_outside_config(pkg):
    offenders = []
    for py in sorted(PACKAGES[pkg].rglob("*.py")):
        if py.name == "config.py":
            continue
        for lineno, val in _float_literals(py):
            offenders.append(f"{py.relative_to(SERVER)}:{lineno} = {val}")
    assert not offenders, ("decision-looking float literals outside "
                           f"{pkg}/config.py: {offenders}")


@pytest.mark.parametrize("key", ["window_frames", "window_overlap_frac", "process_res",
                                 "model_id", "knot_walk_m", "heldout_confidence",
                                 "huber_tol", "huber_max_iter", "instruments"])
def test_gauge_keys_are_mandatory(key):
    raw = _raw()
    del raw["reconstruction"]["precision"]["gauge"][key]
    with pytest.raises(PrecisionConfigError, match=rf"gauge\.{key}"):
        load_precision_config(raw)


def test_gauge_windows_must_chain():
    raw = copy.deepcopy(_raw())
    raw["reconstruction"]["precision"]["gauge"]["window_overlap_frac"] = 0.02
    with pytest.raises(PrecisionConfigError, match="cannot be chained"):
        load_precision_config(raw)
    raw = copy.deepcopy(_raw())
    raw["reconstruction"]["precision"]["gauge"]["instruments"] = ["da3_windows", "lidar9000"]
    with pytest.raises(PrecisionConfigError, match="instruments"):
        load_precision_config(raw)


@pytest.mark.parametrize("key", ["enabled", "resolutions", "mode", "window_frames",
                                 "pair_samples"])
def test_resolution_probe_keys_are_mandatory(key):
    raw = _raw()
    del raw["reconstruction"]["precision"]["omega"]["resolution_probe"][key]
    with pytest.raises(PrecisionConfigError, match=rf"resolution_probe\.{key}"):
        load_precision_config(raw)


def test_tracks_keys_are_mandatory_and_the_dense_matcher_stays_off():
    raw = _raw()
    del raw["reconstruction"]["precision"]["tracks"]["heldout_frac"]
    with pytest.raises(PrecisionConfigError, match=r"tracks\.heldout_frac"):
        load_precision_config(raw)
    raw = copy.deepcopy(_raw())
    raw["reconstruction"]["precision"]["tracks"]["dense_matcher"] = "roma"
    with pytest.raises(PrecisionConfigError, match="VENDORS.lock.md"):
        load_precision_config(raw)


def test_tracker_weights_are_pinned_by_content():
    t = load_precision_config(_raw()).tracks
    assert "/resolve/main/" not in t.tracker_weights_url           # a revision, not a branch
    assert len(t.tracker_weights_sha256) == 64
    for key in ("tracker_weights_url", "tracker_weights_sha256"):
        raw = _raw()
        del raw["reconstruction"]["precision"]["tracks"][key]
        with pytest.raises(PrecisionConfigError, match=rf"tracks\.{key}"):
            load_precision_config(raw)
    raw = _raw()
    raw["reconstruction"]["precision"]["tracks"]["tracker_weights_sha256"] = "deadbeef"
    with pytest.raises(PrecisionConfigError, match="64 hex"):
        load_precision_config(raw)


def test_ceres_threads_must_stay_one():
    """More than one Ceres thread sums in a run-dependent order: the refinement would
    not be bit-identical run to run (measured, see config.yaml) — refused at load."""
    raw = _raw()
    assert raw["reconstruction"]["precision"]["refine"]["ceres_threads"] == 1
    raw["reconstruction"]["precision"]["refine"]["ceres_threads"] = 8
    with pytest.raises(PrecisionConfigError, match="ceres_threads.*bit-identical"):
        load_precision_config(raw)
