"""docs/plan_determinismo.md point 71 (DECIDIDO): the SALAD revisit bars (distance = the scene's
median depth, angle = half the FOV) carry the error MEASURED by a fixed-key bootstrap of the
median over the I3 windows; a keyframe pair within error_factor x that error of either bar is in
NEITHER class of the Youden calibration; the error factor is the user's (correction_graph.graph.
improvement_error_factor) read from the frozen run configuration. These tests run once
point71.patch is applied (calibration.py, intake/walk.py); until then they are skipped, saying so."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"))

from LoopModels import calibration as CAL                        # noqa: E402
from intake import walk as W                                     # noqa: E402
from tests.test_intake_walk import _trajectory, _write_windows   # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
PATCHED = "dist_bar_err_m" in inspect.getsource(CAL.calibrate_threshold) \
    and W.REVISIT_REFERENCE_VERSION >= 3
pytestmark = pytest.mark.skipif(not PATCHED, reason="point71.patch not applied yet "
                                                     "(scratchpad/point71.patch)")


def _ref(n=60, dist_err=0.0, cos_err=0.0, factor=2.0):
    """A walk that comes back: keyframes k and n-1-k stand at the same place."""
    c = np.array([[min(k, n - 1 - k) * 2.0, 0.0, 0.0] for k in range(n)])
    f = np.tile([0.0, 0.0, 1.0], (n, 1))
    return {"frames": list(range(n)), "centres": c.tolist(), "forward": f.tolist(),
            "dist_bar_m": 1.0, "cos_bar": float(np.cos(np.radians(30))),
            "dist_bar_err_m": dist_err, "cos_bar_err": cos_err, "error_factor": factor}


def _sims(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    local = np.abs(idx[:, None] - idx[None]) < 5
    sims = rng.uniform(0.2, 0.4, (n, n))
    for k in range(n):
        sims[k, n - 1 - k] = sims[n - 1 - k, k] = rng.uniform(0.5, 0.6)
    return (sims + sims.T) / 2, local


def test_pairs_within_the_bars_error_are_in_neither_class():
    n = 60
    sims, local = _sims(n)
    thr0, rep0 = CAL.calibrate_threshold(sims, local, list(range(n)), _ref(n))
    assert rep0["n_undecided_pairs"] == 0 and rep0["error_factor"] == 2.0
    # the revisit partners stand at D = 0 and the others at >= 2 m: a distance error of 0.6 m
    # (2 x 0.6 = 1.2 m around the 1 m bar) leaves the partners (|0 - 1| = 1 < 1.2) undecided
    thr, rep = CAL.calibrate_threshold(sims, local, list(range(n)), _ref(n, dist_err=0.6))
    assert thr is None and rep["n_revisit_pairs"] == 0 and rep["n_undecided_pairs"] > 0
    # a smaller error (2 x 0.3 = 0.6 m): partners at D = 0 are clear of the bar → calibrated
    thr2, rep2 = CAL.calibrate_threshold(sims, local, list(range(n)), _ref(n, dist_err=0.3))
    assert thr2 == thr0 and rep2["n_revisit_pairs"] == rep0["n_revisit_pairs"]
    assert rep2["dist_bar_err_m"] == 0.3 and rep2["n_undecided_pairs"] == 0
    # the angle bar: every pair looks the same way (F·F = 1, cos_bar 0.866): an error of 0.1
    # (2 x 0.1 = 0.2 > 1 - 0.866) puts every pair within the bar's error → none decided
    thr3, rep3 = CAL.calibrate_threshold(sims, local, list(range(n)), _ref(n, cos_err=0.1))
    assert thr3 is None and rep3["n_undecided_pairs"] == int(np.triu(~local, 1).sum())
    # the factor is the reference's (the user's rule), not a constant of the module
    thr4, rep4 = CAL.calibrate_threshold(sims, local, list(range(n)),
                                         _ref(n, dist_err=0.3, factor=4.0))
    assert thr4 is None and rep4["error_factor"] == 4.0


def test_a_reference_without_measured_errors_is_refused():
    n = 20
    sims, local = _sims(n)
    ref = _ref(n)
    for key in ("dist_bar_err_m", "cos_bar_err", "error_factor"):
        bad = {k: v for k, v in ref.items() if k != key}
        with pytest.raises(KeyError, match="revisit reference carries no measured bar error"):
            CAL.calibrate_threshold(sims, local, list(range(n)), bad)


def test_the_reference_measures_each_bars_error_by_a_fixed_key_bootstrap(tmp_path):
    from intake import run_config as RC
    with open(SERVER / "config.yaml") as f:
        raw = yaml.safe_load(f)
    RC.freeze_run_config(tmp_path, raw, log=lambda m: None)          # the user's factor lives there
    truth = _trajectory(64)
    _write_windows(tmp_path, truth, noise_m=0.0)
    doc = W.revisit_reference(tmp_path)
    assert doc["version"] == 3 and doc["bar_bootstrap"] == {"n_windows": doc["bar_bootstrap"]["n_windows"],
                                                             "n_boot": 2000, "seed": 0}
    assert doc["error_factor"] == raw["correction_graph"]["graph"]["improvement_error_factor"]
    # the synthetic windows' depth of window i is i and window 0 (depth 0) holds no valid depth:
    # the median over windows and its bootstrap error are those of the integers 1..n_w
    n_w = doc["bar_bootstrap"]["n_windows"]
    md = np.arange(1, n_w + 1, dtype=np.float64)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, n_w, size=(2000, n_w))
    assert doc["dist_bar_m"] == float(np.median(md))
    assert doc["dist_bar_err_m"] == pytest.approx(float(np.std(np.median(md[pick], axis=1))), rel=1e-12)
    assert doc["cos_bar_err"] == pytest.approx(0.0, abs=1e-15)          # K is the same in every window
    # the same twice, bit for bit (fixed key)
    doc2 = W.revisit_reference(tmp_path)
    assert doc2 == doc
    on_disk = json.loads((tmp_path / "output" / W.REVISIT_REFERENCE_NAME).read_text())
    assert on_disk == doc
