"""Nothing is rejected for being BIG; σ is measured (USER 2026-09-16).

On pccr the only two bridges able to close a 44 m walk were dropped at 17 cm
by `max_residual_m: 0.10`, and the pose graph fell back to IDENTITY — *"no
debes rechazar correcciones por umbrales arbitrarios"*. What decides now:

  * a fit must EXIST (starvation is the only geometric rejection);
  * σ = the SPLIT-HALF held-out residual — fit on half the correspondences,
    measure on the other half;
  * σ is widened to the disagreement between independent bridges of the same
    chunk pair, which is an error bar nobody had to invent;
  * the residual is DECLARED against what the session itself achieves on its
    own seams, never against a constant.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_VENDOR = Path("/workspace/stac-build/vendor/VGGT-Long")
sys.path.insert(0, str(_VENDOR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_utils.loop_bridges import (split_half_residual,  # noqa: E402
                                     verify_loop)
from tests.synth_metric import fork_loops_cfg                # noqa: E402


def _rigid(seed=0, n=40000, noise=0.0, R_deg=3.0, t=(0.4, -0.2, 0.1)):
    """Correspondences related by a real rigid transform (+ optional noise)."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, 3)) * 2.0
    a = np.deg2rad(R_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    q = p @ R.T + np.asarray(t)
    if noise:
        q = q + rng.normal(scale=noise, size=q.shape)
    return p, q


class TestSplitHalf:
    def test_a_real_transform_predicts_what_it_never_saw(self):
        p, q = _rigid(noise=0.01)
        sh = split_half_residual(p, q, rigid=True, sample=200000, min_pts=1000)
        assert sh is not None
        # held-out ≈ fit: the fit describes the geometry, not the sample
        assert sh["ratio"] < 1.2, sh
        assert sh["holdout_residual_m"] < 0.03, sh

    def test_a_fit_on_unrelated_points_does_not_generalize(self):
        """No transform relates the two clouds: whatever is fitted on one half
        says nothing about the other, and the held-out error says so."""
        rng = np.random.default_rng(3)
        p = rng.normal(size=(40000, 3)) * 2.0
        q = rng.normal(size=(40000, 3)) * 2.0
        sh = split_half_residual(p, q, rigid=True, sample=200000, min_pts=1000)
        assert sh is not None
        assert sh["holdout_residual_m"] > 10 * 0.03, sh

    def test_it_declines_when_there_is_not_enough_to_split(self):
        p, q = _rigid(n=1500)
        assert split_half_residual(p, q, rigid=True, sample=200000, min_pts=1000) is None


def _meas(residual, holdout=None, n_corr=400000, s_ab=1.0):
    m = {"ok": True, "residual_m": residual, "n_corr": n_corr, "s_ab": s_ab,
         "R_ab": np.eye(3).tolist(), "t_ab": [0.0, 0.0, 0.0]}
    if holdout is not None:
        m["holdout_residual_m"] = holdout
        m["holdout_ratio"] = holdout / residual
    return m


class TestVerdict:
    def test_the_pccr_bridge_is_no_longer_dropped(self):
        """17 cm against a 10 cm constant: the edge that closed the walk."""
        cfg = fork_loops_cfg()
        v = verify_loop(_meas(0.17, holdout=0.17), cfg, reference_m=0.051)
        assert v["status"] == "accepted", v
        assert v["sigma_m"] == pytest.approx(0.17)
        assert v["evidence"] == "weak"          # declared, not vetoed
        assert any("weak evidence" in r for r in v["reasons"])

    def test_sigma_is_the_holdout_error_not_the_fit_residual(self):
        cfg = fork_loops_cfg()
        v = verify_loop(_meas(0.02, holdout=0.09), cfg, reference_m=0.05)
        assert v["sigma_m"] == pytest.approx(0.09)

    def test_a_bridge_as_good_as_the_session_is_not_flagged(self):
        cfg = fork_loops_cfg()
        v = verify_loop(_meas(0.03, holdout=0.03), cfg, reference_m=0.051)
        assert v["status"] == "accepted"
        assert v.get("evidence") != "weak", v
        assert not any("weak evidence" in r for r in v.get("reasons", []))

    def test_the_session_reference_beats_the_config_constant(self):
        cfg = fork_loops_cfg()
        got = verify_loop(_meas(0.08, holdout=0.08), cfg, reference_m=0.051)
        assert got["checks"]["geometric"]["reference_source"] == "session_seams"
        fallback = verify_loop(_meas(0.08, holdout=0.08), cfg)
        assert fallback["checks"]["geometric"]["reference_source"] == "config_fallback"
        assert fallback["checks"]["geometric"]["reference_m"] == cfg["max_residual_m"]

    def test_starvation_is_the_only_geometric_rejection(self):
        cfg = fork_loops_cfg()
        v = verify_loop(_meas(0.01, holdout=0.01, n_corr=5), cfg, reference_m=0.05)
        assert v["status"] == "rejected"
        assert any("starved" in r for r in v["reasons"])
        assert "sigma_m" not in v        # never measured → never printed as nan

    def test_no_residual_magnitude_rejects_anything(self):
        cfg = fork_loops_cfg()
        for res in (0.05, 0.17, 0.8, 3.0):
            v = verify_loop(_meas(res, holdout=res), cfg, reference_m=0.05)
            assert v["status"] in ("accepted", "scale_break"), (res, v)
            assert v["sigma_m"] == pytest.approx(res)
