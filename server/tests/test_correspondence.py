"""The four invariants, on geometry that belongs to no scene.

Every case here is a shape, not an object: a plane, a cylinder, a compact
body. Nothing in the module may need to know what the thing IS — if a rule
only works because someone knew it was a floor or a duct, it fails here.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction import correspondence as co  # noqa: E402

FLOOR = 0.01          # a session repeatability, in metres


def _plane(n=6000, size=20.0, seed=0):
    """A flat patch in the z = 0 plane: normals all along z.

    Large on purpose. What makes a slide along a surface invisible is that the
    two observations still land ON the same surface; a patch smaller than the
    slide stops overlapping, and then the displacement IS detectable — the
    module is right to say so, so a fixture must not test the opposite."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-size / 2, size / 2, size=(n, 2))
    return np.column_stack([xy, np.zeros(n)])


def _cylinder(n=6000, length=30.0, radius=0.3, seed=0):
    """A duct/beam run along x: normals span y-z, x is free.

    Long, for the same reason as _plane: two observations of one duct run are
    two stretches of the SAME duct, so a slide along it still lands on it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-length / 2, length / 2, n)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([x, radius * np.cos(th), radius * np.sin(th)])


def _compact(n=4000, r=0.5, seed=0):
    """A closed body: normals point everywhere."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True) * r


def _measure(a, b, t, R=None, cov=(0.8, 0.8), n=4000):
    return co.measure(a, b, R if R is not None else np.eye(3), t,
                      cov[0], cov[1], n, FLOOR)


# ── invariant 3: nothing is corrected in a direction nobody observed ─────

def test_a_plane_observes_only_its_normal():
    a = _plane()
    m = _measure(a, a + np.array([0.0, 0.0, 0.7]), np.array([0.0, 0.0, 0.7]))
    assert int(np.count_nonzero(m.observed)) == 1, (m.energy, m.observed)
    # the displacement along the normal is real and survives whole
    assert np.linalg.norm(m.t) == pytest.approx(0.7, abs=1e-6)
    assert m.dropped_m == pytest.approx(0.0, abs=1e-9)
    assert m.is_duplicate


def test_a_plane_sliding_on_itself_claims_nothing():
    """The in-plane component is free: moving there changes no measured
    distance, so it is not a displacement of the world."""
    a = _plane()
    m = _measure(a, a + np.array([3.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert np.linalg.norm(m.t) == pytest.approx(0.0, abs=1e-6)
    assert m.dropped_m == pytest.approx(3.0, abs=1e-6)
    assert not m.is_duplicate
    assert "slides" in m.reason
    assert m.notes["gain_x"] <= FLOOR, m.notes


def test_a_cylinder_leaves_its_axis_free():
    """7 m along a duct is not a duplicate 7 m away — it is the same duct,
    further along. A sampled surface never gives its axis exactly, so a
    residue leaks onto the observed directions; it must stay below what the
    session can tell apart from zero."""
    a = _cylinder()
    m = _measure(a, a + np.array([7.0, 0.0, 0.0]), np.array([7.0, 0.0, 0.0]))
    # across the axis the surface is stiff; along it, free
    assert m.notes["stiff_y"] > 0.3 and m.notes["stiff_z"] > 0.3, m.notes
    assert m.notes["stiff_x"] < 0.1, m.notes
    # and the 7 m along it bought nothing, so nothing is claimed
    assert m.notes["gain_x"] <= FLOOR, m.notes
    assert np.linalg.norm(m.t) == 0.0
    assert m.dropped_m == pytest.approx(7.0, rel=1e-3)
    assert not m.is_duplicate


def test_a_cylinder_observes_everything_across_its_axis():
    a = _cylinder()
    m = _measure(a, a + np.array([0.0, 0.6, 0.0]), np.array([0.0, 0.6, 0.0]))
    assert np.linalg.norm(m.t) == pytest.approx(0.6, abs=1e-5)
    assert m.is_duplicate


def test_a_compact_body_observes_all_three():
    a = _compact()
    t = np.array([0.4, -0.3, 0.2])
    m = _measure(a, a + t, t)
    assert int(np.count_nonzero(m.observed)) == 3, (m.energy, m.observed)
    assert np.linalg.norm(m.t) == pytest.approx(np.linalg.norm(t), rel=1e-4)


def test_only_the_unobserved_part_is_dropped():
    """A mixed displacement keeps what the surface sees and drops the rest."""
    a = _plane()
    t = np.array([2.0, 0.0, 0.5])
    m = _measure(a, a + t, t)
    assert np.linalg.norm(m.t) == pytest.approx(0.5, abs=1e-5)
    assert m.dropped_m == pytest.approx(2.0, abs=1e-5)
    assert m.is_duplicate            # what survives is a real displacement


# ── invariant 3, rotations ──────────────────────────────────────────────

def test_a_single_plane_cannot_earn_a_rotation():
    """One plane spins about its own normal at no measured cost — a fitted
    angle there is an artifact of the fit."""
    a = _plane()
    th = np.radians(20.7)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    m = _measure(a, a, np.zeros(3), R=R)
    assert np.allclose(m.R, np.eye(3))
    assert m.rot_deg == 0.0


def test_a_compact_body_keeps_its_rotation():
    a = _compact()
    th = np.radians(12.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    m = _measure(a, a, np.zeros(3), R=R)
    assert m.rot_deg == pytest.approx(12.0, abs=1e-6)


# ── invariant 4: confidence is evidence, never the leftover error ────────

def test_no_coverage_earns_no_confidence():
    """The failure this exists for: an edge measured against nothing reached
    the pose graph as the most trusted of the run."""
    a = _compact()
    t = np.array([0.0, 0.0, 0.3])
    strong = _measure(a, a + t, t, cov=(0.87, 0.85), n=4000)
    empty = _measure(a, a + t, t, cov=(0.87, 0.0), n=4000)
    assert np.all(np.isinf(empty.sigma_t))
    assert np.nanmin(strong.sigma_t[strong.observed]) < 1.0


def test_more_evidence_is_more_confidence():
    """Coverage and independent views are the evidence; the raw point count is
    not — one surface sampled a million times is still one observation."""
    a = _compact()
    t = np.array([0.0, 0.0, 0.3])
    few = _measure(a, a + t, t, cov=(0.3, 0.3), n=200)
    many = _measure(a, a + t, t, cov=(0.9, 0.9), n=8000)
    assert many.sigma_t[many.observed].max() < few.sigma_t[few.observed].max()


def test_dense_sampling_alone_buys_nothing():
    a = _compact()
    t = np.array([0.0, 0.0, 0.3])
    sparse = _measure(a, a + t, t, cov=(0.5, 0.5), n=500)
    dense = _measure(a, a + t, t, cov=(0.5, 0.5), n=500_000)
    assert np.allclose(sparse.sigma_t[sparse.observed],
                       dense.sigma_t[dense.observed])


def test_sigma_never_beats_the_sessions_own_repeatability():
    a = _compact()
    t = np.array([0.0, 0.0, 0.3])
    m = _measure(a, a + t, t, cov=(1.0, 1.0), n=10 ** 6)
    assert m.sigma_t[m.observed].min() >= FLOOR


def test_an_unobserved_axis_carries_no_information():
    a = _plane()
    m = _measure(a, a + np.array([0.0, 0.0, 0.7]), np.array([0.0, 0.0, 0.7]))
    info = m.info_translation()
    # the information matrix has rank 1 — the plane's normal, nothing else
    assert np.linalg.matrix_rank(info, tol=1e-9) == 1


# ── invariant 2: a witness must have been able to see ────────────────────

def _cam(t_world):
    c2w = np.eye(4)
    c2w[:3, 3] = t_world
    return c2w


def test_a_point_out_of_frame_is_not_a_witness():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 2.0], [50.0, 0.0, 2.0]])   # second far off-axis
    w = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                         None, (100, 100))
    assert w[0].tolist() == [True, False]


def test_a_point_behind_the_camera_is_not_a_witness():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    w = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                         None, (100, 100))
    assert w[0].tolist() == [True, False]


def test_a_hidden_point_is_not_a_witness():
    """The frame measured something closer along the same ray."""
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 5.0]])
    depth = np.full((100, 100), 2.0)          # a wall at 2 m in front of it
    w = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                         lambda f: depth, (100, 100))
    assert w[0].tolist() == [False]


def test_a_ray_with_no_measurement_says_nothing():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 5.0]])
    depth = np.zeros((100, 100))              # nothing measured there
    w = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                         lambda f: depth, (100, 100))
    assert w[0].tolist() == [False]


def test_a_visible_point_at_its_own_depth_is_a_witness():
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 5.0]])
    depth = np.full((100, 100), 5.0)
    w = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                         lambda f: depth, (100, 100))
    assert w[0].tolist() == [True]


def test_the_witness_set_does_not_depend_on_a_trial():
    """It is computed on the reference geometry, so a correction cannot score
    by pushing the scene out of view."""
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    pts = np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]])
    ref = co.witness_pairs(pts, [0], lambda f: _cam([0, 0, 0]), lambda f: K,
                           None, (100, 100))
    moved = co.witness_pairs(pts + np.array([0.0, 0.0, 50.0]), [0],
                             lambda f: _cam([0, 0, 0]), lambda f: K, None, (100, 100))
    assert ref[0].sum() == 2 and moved[0].sum() == 2     # both still project
    # the point: the caller uses `ref` for both, never recomputing per trial


# ── the module must not need to know what the object is ─────────────────

def test_nothing_depends_on_the_world_frame():
    """The same shape, rotated arbitrarily in the world, decides the same."""
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 2] = -Q[:, 2]
    a = _cylinder()
    t = np.array([7.0, 0.0, 0.0])
    m0 = _measure(a, a + t, t)
    m1 = _measure(a @ Q.T, (a + t) @ Q.T, Q @ t)
    assert m0.is_duplicate == m1.is_duplicate
    assert np.linalg.norm(m0.t) == pytest.approx(np.linalg.norm(m1.t), abs=1e-5)
    assert np.count_nonzero(m0.observed) == np.count_nonzero(m1.observed)
