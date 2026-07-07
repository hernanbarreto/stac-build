# STAC-Builder — Phase R geometry + store unit tests (synthetic).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.geometry import (  # noqa: E402
    fit_gravity_aligned_obb, filter_outliers_knn, signed_distances_along_axis,
)
from phase_r.onion import detect_onion  # noqa: E402
from phase_r.instance_store import InstanceStore  # noqa: E402

rng = np.random.default_rng(0)


def test_obb_recovers_box_extents():
    # axis-aligned box 2 x 1 x 0.5 (x,y,z), y up
    pts = rng.uniform(-1, 1, (2000, 3)) * np.array([1.0, 0.5, 0.25])
    out = fit_gravity_aligned_obb(pts, gravity=np.array([0, -1, 0]))
    assert out is not None
    T, aabb, pos = out
    ex, ey, ez = aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]
    extents = sorted([ex, ey, ez])
    expected = sorted([2.0, 1.0, 0.5])
    assert np.allclose(extents, expected, atol=0.1), (extents, expected)
    assert np.allclose(pos, 0, atol=0.1)


def test_knn_removes_outliers():
    cluster = rng.normal(0, 0.05, (500, 3))
    outliers = rng.uniform(-5, 5, (30, 3))
    pts = np.vstack([cluster, outliers])
    keep = filter_outliers_knn(pts, k=6, factor=3.0)
    # most cluster kept, most outliers removed
    assert keep[:500].mean() > 0.9
    assert keep[500:].mean() < 0.5


def test_onion_detects_double_surface():
    # two parallel planes (a doubled wall) separated by 0.20 m along z
    n = 800
    xy = rng.uniform(-1, 1, (n, 2))
    z = np.where(rng.random(n) < 0.5, -0.10, 0.10) + rng.normal(0, 0.01, n)
    pts = np.column_stack([xy[:, 0], xy[:, 1] * 0.5, z])
    out = fit_gravity_aligned_obb(pts, gravity=np.array([0, -1, 0]))
    T, aabb, pos = out
    res = detect_onion(pts, T, aabb, min_points=40)
    assert res.bimodal, res
    assert abs(res.separation_m - 0.20) < 0.05, res.separation_m


def test_onion_single_surface_not_bimodal():
    n = 800
    xy = rng.uniform(-1, 1, (n, 2))
    z = rng.normal(0, 0.01, n)
    pts = np.column_stack([xy[:, 0], xy[:, 1] * 0.5, z])
    out = fit_gravity_aligned_obb(pts, gravity=np.array([0, -1, 0]))
    T, aabb, pos = out
    res = detect_onion(pts, T, aabb, min_points=40)
    assert not res.bimodal, res


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = InstanceStore(os.path.join(d, "scene.db"))
        store.upsert_instance(1, "column", known=True, confidence=0.9, n_views=5,
                              status="proposed", scene_type="depot")
        pts = rng.normal(0, 1, (100, 3)).astype(np.float32)
        store.set_points(1, pts)
        T = np.eye(4); aabb = np.array([-1, 1, -1, 1, -1, 1.0]); pos = np.zeros(3)
        store.set_obb(1, T, aabb, pos, window_id="w0", n_points=100)
        store.add_masklet_ref(1, 42, "f42_o1", np.array([0.1, 0.1, 0.2, 0.3]), 0.8)
        store.set_vote_metrics(1, 0.3, 0.9, 100)
        store.set_onion_metric(1, True, 0.15, 12.0)

        got = store.get_points(1)
        assert got.shape == (100, 3) and np.allclose(got, pts, atol=1e-5)
        insts = store.list_instances()
        assert len(insts) == 1 and insts[0]["label"] == "column"
        obb = store.get_obb(1, "w0")
        assert obb is not None and np.allclose(obb[1], aabb)
        m = store.get_metrics(1)
        assert m["onion"]["bimodal"] and abs(m["onion"]["separation_m"] - 0.15) < 1e-6
        assert store.get_masklets(1)[0]["mask_key"] == "f42_o1"
        store.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all phase_r geometry/store tests passed")
