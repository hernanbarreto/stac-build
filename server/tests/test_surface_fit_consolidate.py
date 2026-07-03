"""
Stage-1 consolidation tests: the double-layer ("onion") failure mode must
collapse to a thin sheet BEFORE fitting, so the escalation sees shape instead
of registration bias — while the stage-4 report still measures against the
original layered cloud.

The WLOP test needs the CloudComPy310 conda env (CGAL bindings); it skips
cleanly when the satellite launcher can't run.
"""
import numpy as np
import pytest

from reconstruction.surface_fit.consolidate import (consolidate_mls,
                                                    consolidate_wlop)
from reconstruction.surface_fit.runner import fit_segment

SIGMA = 0.002


def make_double_layer(n=40_000, sep=0.008, sigma=SIGMA, seed=21):
    """Horizontal plane at z=0 with the x>0 half duplicated `sep` above —
    the exact TSDF 'onion layer' pathology."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(0, sigma, n)
    pts = np.column_stack([x, y, z])
    ghost = pts[pts[:, 0] > 0].copy()
    ghost[:, 2] += sep
    return np.vstack([pts, ghost])


def thickness_mm(pts, region):
    """std (mm) of z inside a boolean region — layer collapse metric."""
    return float(pts[region][:, 2].std() * 1000.0)


class TestMLS:
    def test_double_layer_collapses(self):
        cloud = make_double_layer()
        out = consolidate_mls(cloud, radius=0.06, iterations=2)
        # right half held two sheets 8 mm apart (std ≈ 4 mm); must end thin
        assert thickness_mm(cloud, cloud[:, 0] > 0.5) > 3.0
        assert thickness_mm(out, out[:, 0] > 0.5) < 1.5

    def test_left_half_undamaged(self):
        """Where there was no ghost layer, MLS must only denoise, not move."""
        cloud = make_double_layer()
        out = consolidate_mls(cloud, radius=0.06, iterations=2)
        left = out[out[:, 0] < -0.5]
        assert abs(left[:, 2].mean()) * 1000.0 < 0.5    # plane stays at z≈0


@pytest.mark.skipif(
    consolidate_wlop(np.random.default_rng(0).uniform(0, 1, (300, 3))) is None,
    reason="CGAL WLOP satellite (CloudComPy310 env) not available")
class TestWLOP:
    def test_double_layer_collapses(self):
        cloud = make_double_layer(n=25_000)
        out = consolidate_wlop(cloud, neighbor_radius_m=0.06,
                               select_percentage=25.0, iterations=25)
        assert out is not None and len(out) > 1000
        assert thickness_mm(out, out[:, 0] > 0.5) < 1.0


def make_full_double_layer(n=40_000, sep=0.008, sigma=SIGMA, seed=22):
    """The whole plane seen twice, `sep` apart (two chunks, registration
    bias) — collapses to the mid-plane with no step."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.normal(0, sigma, n)
    pts = np.column_stack([x, y, z])
    ghost = pts.copy()
    ghost[:, 2] += sep
    return np.vstack([pts, ghost])


class TestEndToEnd:
    def test_consolidated_fit_accepts_plane(self):
        """With consolidation ON, the fully-doubled segment must fit as a
        PLANE whose escalation residual is clean, while the stage-4 report
        (vs the ORIGINAL layered cloud) honestly keeps the bimodal deviation."""
        cloud = make_full_double_layer()
        fs = fit_segment(cloud, consolidate_method="mls", label="floor",
                         models=["plane", "bspline"])
        assert fs is not None and fs.kind == "plane"
        # report vs original: the two layers straddle the fitted plane → the
        # residual spread must reflect the ±half-separation, not be hidden
        assert fs.report.stats.p95_mm > 2.0
        assert fs.n_input_points == len(cloud)

    def test_partial_ghost_keeps_step_visible(self):
        """Ghost on HALF the area: consolidation leaves a genuine step at the
        coverage boundary — the fit must NOT silently call that a plane."""
        cloud = make_double_layer()
        fs = fit_segment(cloud, consolidate_method="mls", label="floor",
                         models=["plane"])
        assert fs.report.stats.moran.relevant or fs.report.stats.p95_mm > 2.0

    def test_without_consolidation_escalates(self):
        """Same cloud, stage 1 off → the plane leaves relevant structure."""
        cloud = make_double_layer()
        fs = fit_segment(cloud, consolidate_method="none", label="floor",
                         models=["plane"])
        assert fs.report.stats.moran.relevant
