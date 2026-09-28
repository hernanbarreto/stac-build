"""SALAD's appearance bar calibrated on the session's own geometric revisits
(2026-09-28): the DA3-window walk says which keyframe pairs see the same place; the
threshold is the similarity that best separates those pairs from the rest
(Youden's J), with the configured value only as the fallback."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"))


from LoopModels.calibration import calibrate_threshold            # noqa: E402
from intake import walk as W                                     # noqa: E402
from tests.test_intake_walk import G, _trajectory, _write_windows   # noqa: E402


def _ref(n=60):
    """A walk that comes back: keyframes k and n-1-k stand at the same place,
    looking the same way."""
    c = np.array([[min(k, n - 1 - k) * 2.0, 0.0, 0.0] for k in range(n)])    # 2 m apart
    f = np.tile([0.0, 0.0, 1.0], (n, 1))
    return {"frames": list(range(n)), "centres": c.tolist(), "forward": f.tolist(),
            "dist_bar_m": 1.0, "cos_bar": float(np.cos(np.radians(30)))}


def test_the_bar_separates_the_revisits_from_the_rest():
    n = 60
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    local = np.abs(idx[:, None] - idx[None]) < 5
    sims = rng.uniform(0.2, 0.4, (n, n))
    for k in range(n):                      # the revisit partner looks alike
        sims[k, n - 1 - k] = sims[n - 1 - k, k] = rng.uniform(0.5, 0.6)
    sims = (sims + sims.T) / 2
    thr, rep = calibrate_threshold(sims, local, list(range(n)), _ref(n))
    partners = [sims[k, n - 1 - k] for k in range(n) if abs(2 * k - n + 1) >= 5]
    # any bar above every other pair and at most the weakest revisit separates them
    assert 0.4 < thr <= min(partners) + 1e-12
    assert rep["tpr"] > 0.9 and rep["fpr"] < 0.01 and rep["youden_j"] > 0.9
    assert rep["n_revisit_pairs"] > 0 and rep["revisit_similarity_median"] > rep["other_similarity_median"]
    json.dumps(rep)


def test_no_revisit_leaves_the_configured_bar():
    n = 20
    ref = _ref(n)
    ref["centres"] = [[k * 5.0, 0.0, 0.0] for k in range(n)]      # a straight walk away
    idx = np.arange(n)
    thr, rep = calibrate_threshold(np.full((n, n), 0.3), np.abs(idx[:, None] - idx[None]) < 3,
                                   list(range(n)), ref)
    assert thr is None and rep["n_revisit_pairs"] == 0


def test_the_reference_bars_come_from_the_windows(tmp_path):
    truth = _trajectory(64)
    _write_windows(tmp_path, truth, noise_m=0.0)
    doc = W.revisit_reference(tmp_path)
    assert len(doc["frames"]) == 64 and len(doc["centres"]) == 64
    # the synthetic windows' depth of window i is i (index as depth): its median
    assert doc["dist_bar_m"] > 0
    # K = identity on a 6-px-wide depth: hfov = 2·atan(6/2)
    assert abs(doc["hfov_rad"] - 2 * np.arctan(3.0)) < 1e-9
    assert abs(doc["cos_bar"] - np.cos(np.arctan(3.0))) < 1e-9
    assert (tmp_path / "output" / W.REVISIT_REFERENCE_NAME).exists()
