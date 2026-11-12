"""The oriented bounding box on the device end to end (segmentation/obb_gpu.py, USER 2026-11-11: "todo lo
que me marcaste que puede pasarse a GPU hay que hacerlo en GPU ... Nada debe hacerse en cpu").

The exact parts against the record (segmentation.pipeline): the stable keys bit for bit, the stable sample,
the flyer components; the hull against scipy's qhull and the rotating calipers against the record's loop;
the drawn parts (the RANSAC triplets, the bootstrap flags — torch's generator) deterministic, the footprint
error a function of the point SET; the box itself still finds a table's yaw."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import obb_gpu as G  # noqa: E402
from segmentation import pipeline as P  # noqa: E402


def _cloud(seed, n=3000):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)) * np.array([2.0, 0.5, 1.0]) + np.array([-3.2, 1.1, 7.5])
    pts[:20] += 40.0                                                     # a flyer blob far away
    return pts


@pytest.mark.parametrize("seed", [0, 1])
def test_the_stable_keys_and_sample_are_the_records(seed):
    pts = _cloud(seed)
    ref = P._stable_point_keys(pts).view(np.int64)
    new = G.stable_point_keys_t(torch.as_tensor(pts)).numpy()
    assert np.array_equal(ref, new)
    assert np.array_equal(P._stable_sample(pts, 500), G.stable_sample_t(torch.as_tensor(pts), 500).numpy())


def test_the_core_points_are_the_records():
    from config import cfg
    rng = np.random.default_rng(3)
    pts = np.r_[rng.uniform(0.0, 0.6, size=(4000, 3)),                  # the element: dense at 5 cm
                rng.uniform(0.0, 0.02, size=(25, 3)) + 3.0]               # a flyer blob 3 m off
    ocfg = dict(cfg["segmentation"]["obb_core"])
    factor = P._decision_params(cfg)[0]
    c_ref, n_ref, r_ref = P._obb_core_points(pts, ocfg, factor)
    c_new, n_new, r_new = G.obb_core_points_t(torch.as_tensor(pts), ocfg, factor)
    assert n_ref == n_new > 0 and r_ref == r_new
    assert np.array_equal(c_ref, c_new.numpy())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_hull_is_qhulls_and_the_calipers_the_records(seed):
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(seed)
    xz = rng.normal(size=(800, 2)) * np.array([3.0, 1.0])
    H = G.convex_hull_t(torch.as_tensor(xz)).numpy()
    ref = xz[ConvexHull(xz).vertices]
    assert {tuple(p) for p in H} == {tuple(p) for p in ref}
    yaw, area = G.calipers_t(torch.as_tensor(H))
    # the record's loop over qhull's polygon
    best_area, best_angle = float("inf"), 0.0
    for i in range(len(ref)):
        e = ref[(i + 1) % len(ref)] - ref[i]
        a = P._fold_yaw(np.arctan2(e[1], e[0]))
        c, s = np.cos(-a), np.sin(-a)
        r = ref @ np.array([[c, -s], [s, c]]).T
        ar = (r[:, 0].max() - r[:, 0].min()) * (r[:, 1].max() - r[:, 1].min())
        if ar < best_area or (ar == best_area and a < best_angle):
            best_area, best_angle = ar, a
    assert abs(area - best_area) <= 1e-9 * best_area and abs(yaw - best_angle) <= 1e-9


def test_the_draws_are_deterministic_and_the_footprint_error_a_function_of_the_set():
    pts = _cloud(5)
    a, b = P._compute_obb(pts), P._compute_obb(pts)
    assert a == b
    xz = torch.as_tensor(pts[:, [0, 2]])
    keys_u = G.unsigned_order(G.stable_point_keys_t(torch.stack([xz[:, 0], torch.zeros(len(xz), dtype=xz.dtype),
                                                                 xz[:, 1]], 1)))
    perm = torch.as_tensor(np.random.default_rng(1).permutation(len(xz)))
    r = G._rotate(xz, torch.tensor(0.3, dtype=torch.float64))
    e1 = G.footprint_error_g(r, keys_u, 1234, 200, float(np.exp(-1.0)), 64)
    e2 = G.footprint_error_g(r[perm], keys_u[perm], 1234, 200, float(np.exp(-1.0)), 64)
    assert e1 == e2 and e1 > 0


def _table(yaw_deg, rng, n_front=4000, n_side=2000):
    """A table: two long vertical faces and two short ones, turned by ``yaw_deg``."""
    w, d, h = 2.0, 0.8, 0.75
    fx = rng.uniform(-w / 2, w / 2, n_front)
    front = np.c_[fx, rng.uniform(0, h, n_front), np.full(n_front, d / 2) + rng.normal(0, 0.003, n_front)]
    back = front * np.array([1, 1, -1])
    sz = rng.uniform(-d / 2, d / 2, n_side)
    side = np.c_[np.full(n_side, w / 2) + rng.normal(0, 0.003, n_side), rng.uniform(0, h, n_side), sz]
    other = side * np.array([-1, 1, 1])
    pts = np.r_[front, back, side, other]
    t = np.radians(yaw_deg)
    R = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
    return pts @ R.T


@pytest.mark.parametrize("yaw", [0.0, 17.0, 30.0, 61.0])
def test_the_box_finds_the_tables_yaw_and_extent(yaw):
    obb = P._compute_obb(_table(yaw, np.random.default_rng(2)))
    got = np.degrees(np.arctan2(-np.asarray(obb["rotation"])[0][2], np.asarray(obb["rotation"])[0][0])) % 90.0
    want = yaw % 90.0
    assert min(abs(got - want), 90.0 - abs(got - want)) < 0.5
    he = sorted(obb["half_extents"][::2])
    assert abs(he[0] - 0.4) < 0.02 and abs(he[1] - 1.0) < 0.02
