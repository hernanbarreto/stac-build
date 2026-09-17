"""A correction is judged by PROPORTION, not by size (USER 2026-09-16).

*"podría haber una corrección de más de 30 cm y ser perfectamente correcta"* —
and it can. What `elastic_max_t_m: 0.30` really caught is what its own comment
recorded: *seam 7->8 fitted up to 1.59 m to close an 8 cm gap*. A fit that moves
the points further than the disagreement it removes is degenerate whatever its
size; one that moves 50 cm to remove 48 cm is right, and the cap shrank it.

Also here: chunk health stops being a cliff at IQR/median 0.30 and becomes a
weight measured against the session's own median spread.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_VENDOR = Path("/workspace/stac-build/vendor/VGGT-Long")
sys.path.insert(0, str(_VENDOR))

from loop_utils.metric_lock import (chunk_trust,  # noqa: E402
                                    demote_disproportionate_fits,
                                    elastic_corrections)


def _fit(t, R=None):
    return (np.eye(3) if R is None else R, np.asarray(t, np.float64))


def _report(entries):
    """entries: {seam: {frame: (before_m, residual_m, motion_m)}}"""
    seams = {}
    for j, d in entries.items():
        seams[str(j)] = {str(g): {"before_m": b, "residual_m": r, "motion_m": m}
                         for g, (b, r, m) in d.items()}
    return {"seams": seams}


class TestProportion:
    def test_a_big_correction_that_explains_itself_is_kept(self):
        """50 cm removing 48 cm of disagreement: the old cap shrank this."""
        fits = {0: {10: _fit([0.50, 0.0, 0.0])}}
        rep = _report({0: {10: (0.48, 0.02, 0.50)}})
        out, n, stats = demote_disproportionate_fits(fits, rep)
        assert n == 0
        assert np.allclose(out[0][10][1], [0.50, 0.0, 0.0])

    def test_the_real_pathology_is_demoted(self):
        """seam 7->8: 1.59 m of motion to close 8 cm."""
        fits = {7: {3: _fit([1.59, 0.0, 0.0])}}
        rep = _report({7: {3: (0.08, 0.06, 1.59)}})
        out, n, stats = demote_disproportionate_fits(fits, rep)
        assert n == 1
        moved = float(np.linalg.norm(out[7][3][1]))
        assert moved == pytest.approx(0.08 + 0.06, abs=1e-6)   # its own allowance
        assert stats["worst_ratio"] > 10

    def test_the_allowance_is_per_frame_not_global(self):
        fits = {0: {1: _fit([0.9, 0, 0]), 2: _fit([0.9, 0, 0])}}
        rep = _report({0: {1: (0.85, 0.05, 0.90),     # proportionate
                           2: (0.05, 0.02, 0.90)}})   # not
        out, n, _ = demote_disproportionate_fits(fits, rep)
        assert n == 1
        assert float(np.linalg.norm(out[0][1][1])) == pytest.approx(0.9)
        assert float(np.linalg.norm(out[0][2][1])) == pytest.approx(0.07, abs=1e-6)

    def test_a_fit_without_the_measurement_is_left_alone(self):
        """An elastic_seams.json written before the measurement existed."""
        fits = {0: {5: _fit([2.0, 0, 0])}}
        out, n, _ = demote_disproportionate_fits(fits, {"seams": {"0": {"5": {}}}})
        assert n == 0 and np.allclose(out[0][5][1], [2.0, 0, 0])

    def test_no_report_at_all_changes_nothing(self):
        fits = {0: {5: _fit([2.0, 0, 0])}}
        out, n, _ = demote_disproportionate_fits(fits, None)
        assert n == 0 and np.allclose(out[0][5][1], [2.0, 0, 0])


class TestTrust:
    def test_equally_spread_chunks_weigh_the_same(self):
        t = chunk_trust({0: 0.2, 1: 0.2, 2: 0.2})
        assert len(set(round(v, 9) for v in t.values())) == 1

    def test_trust_is_continuous_around_the_old_cliff(self):
        """0.299 and 0.311 used to be different worlds."""
        below = chunk_trust({0: 0.10, 1: 0.299})[1]
        above = chunk_trust({0: 0.10, 1: 0.311})[1]
        assert abs(below - above) < 0.01, (below, above)

    def test_a_shakier_chunk_argues_more_quietly(self):
        t = chunk_trust({0: 0.10, 1: 0.40})
        assert t[1] < t[0]

    def test_a_chunk_without_a_measurement_keeps_the_median(self):
        t = chunk_trust({0: 0.10, 1: 0.20, 2: None})
        assert t[2] == pytest.approx(float(np.median([t[0], t[1]])))


class TestConsensusStillCoincides:
    """Whatever the weighting, the two copies of a shared frame must land on
    the same point — that is what the stage exists for."""

    def test_the_two_copies_coincide_under_measured_trust(self):
        chunk_indices = [(0, 10), (5, 15)]
        rng = np.random.default_rng(0)
        fits = {0: {g: _fit(rng.normal(scale=0.05, size=3)) for g in range(5, 10)}}
        trust = chunk_trust({0: 0.10, 1: 0.35})
        c0 = elastic_corrections(chunk_indices, 0, fits, trust=trust)
        c1 = elastic_corrections(chunk_indices, 1, fits, trust=trust)
        pts = rng.normal(size=(200, 3))          # chunk 1's copy of the frame
        for g in range(5, 10):
            R, t = fits[0][g]                    # T_g: chunk 1's copy -> chunk 0's
            A = c0[g - 0]                        # chunk 0 is the DST side of seam 0
            B = c1[g - 5]                        # chunk 1 is the SRC side
            chunk0_copy = pts @ np.asarray(R).T + np.asarray(t)
            dst = chunk0_copy @ A[:3, :3].T + A[:3, 3]
            src = pts @ B[:3, :3].T + B[:3, 3]
            assert np.allclose(dst, src, atol=1e-9), g
