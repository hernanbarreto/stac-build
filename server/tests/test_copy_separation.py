"""USER 2026-09-17/18: the measure has to see what the eye sees.

The separation of two copies was the median NEAREST-NEIGHBOUR distance from one
to the other. That cannot see a copy slid ALONG its own surface — two desks side
by side overlapping 50% have near neighbours everywhere they overlap. pccr
desk#201: the loop reported "copies 60.4 -> 2.7 cm" and called the duplicate
closed while the two centroids stood 52.6 cm apart, which is exactly what the
user was looking at. Between two of its own epochs the desk slid 15 cm FURTHER
and the measure read 2.7 -> 4.2 cm, i.e. nothing.

Worse than a reporting defect: the ICP minimises the same quantity, so its cost
has a flat valley along the surface and the fits land on slid solutions — the
63.9 deg rotations with |t| of metres that a pure translation matched exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify.loops_posthoc import measure_copy      # noqa: E402
from reconstruction.loops.config import load_loops_config          # noqa: E402

SCFG = load_loops_config().certify.scale


def _slab(n=4000, size=(1.40, 0.70, 0.04), seed=0):
    """A desk top: wide, deep, thin — the shape a slide hides in."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) - 0.5) * np.asarray(size)


def test_a_slide_along_the_surface_is_reported_not_hidden():
    """The regression, in one assertion: two copies of a desk slid 60 cm along
    their own length overlap heavily, so nearest neighbours are millimetres —
    and the separation must still report the slide."""
    b = _slab()
    a = b + np.array([0.60, 0.0, 0.0])          # slid along the desk's length
    m = measure_copy(a, b, SCFG)
    assert m is not None
    assert m["nn_before_m"] < 0.10, "nearest neighbour cannot see the slide — that is the point"
    assert m["centroid_before_m"] > 0.50, m["centroid_before_m"]
    assert m["offset_before_m"] > 0.50, \
        f"the reported separation hid a 60 cm slide: {m['offset_before_m'] * 100:.1f} cm"


def test_the_fit_closes_the_slide_instead_of_sliding_further():
    b = _slab()
    a = b + np.array([0.60, 0.0, 0.0])
    m = measure_copy(a, b, SCFG)
    assert m["offset_after_m"] < 0.05, m["offset_after_m"]
    assert np.linalg.norm(m["t"]) == pytest.approx(0.60, abs=0.05), \
        "the closure must be the displacement the bodies demand"


def test_a_genuine_gap_is_still_measured():
    """The surface term must keep working: copies pulled APART along the normal
    have no near neighbours and no centroid coincidence either."""
    b = _slab()
    a = b + np.array([0.0, 0.0, 0.25])
    m = measure_copy(a, b, SCFG)
    assert m["nn_before_m"] > 0.20 and m["centroid_before_m"] > 0.20
    assert m["offset_before_m"] > 0.20
    assert m["offset_after_m"] < 0.05


def test_neither_term_can_hide_the_other():
    """The separation is the WORSE of the two views, so a fit cannot win by
    being good at one and bad at the other."""
    b = _slab()
    a = b + np.array([0.60, 0.0, 0.0])
    m = measure_copy(a, b, SCFG)
    assert m["offset_after_m"] >= m["nn_after_m"] - 1e-9
    assert m["offset_after_m"] >= m["centroid_after_m"] - 1e-9


def test_identical_copies_report_no_separation():
    b = _slab()
    m = measure_copy(b.copy(), b, SCFG)
    assert m["offset_before_m"] < 0.02 and m["offset_after_m"] < 0.02


def test_the_competing_fits_are_all_recorded():
    """Nothing silent: which fit won and what each scored lands in the record."""
    b = _slab()
    a = b + np.array([0.60, 0.0, 0.0])
    m = measure_copy(a, b, SCFG)
    assert m["fit_chosen"] in ("rotated", "translated", "centroid")
    assert set(m["fit_scores_m"]) == {"rotated", "translated", "centroid"}
    assert m["fit_scores_m"][m["fit_chosen"]] == pytest.approx(m["offset_after_m"], abs=1e-3)


def test_a_rotation_that_earns_nothing_is_not_kept():
    b = _slab()
    a = b + np.array([0.60, 0.0, 0.0])
    m = measure_copy(a, b, SCFG)
    assert not m["rotation_earned"], m["rot_deg_fitted"]
    assert np.allclose(np.asarray(m["R"]), np.eye(3))


def test_a_real_rotation_survives():
    """A copy genuinely turned must keep its rotation — the gate removes
    invented DOF, not measured ones."""
    from scipy.spatial.transform import Rotation
    b = _slab(size=(1.40, 0.70, 0.30))          # give it depth so yaw is observable
    R = Rotation.from_rotvec([0.0, np.deg2rad(25.0), 0.0]).as_matrix()
    a = b @ R.T + np.array([0.30, 0.0, 0.0])
    m = measure_copy(a, b, SCFG)
    assert m["offset_after_m"] < 0.06, m["offset_after_m"]
