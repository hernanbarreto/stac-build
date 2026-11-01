"""The rival search of the closures on the device (correction.visit_drift_torch.CellIndex behind
visit_drift.ambiguity, `correction.visit_drift.device` cpu | cuda) against the k-d tree of record: the same
bounded nearest distances point for point (inf beyond the bar), the same rivals, the same margins."""
import os

import numpy as np
import pytest
from scipy.spatial import cKDTree

from correction import visit_drift as VD

torch = pytest.importorskip("torch")
from correction import visit_drift_torch as VT  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _clouds(seed):
    rng = np.random.default_rng(seed)
    A = np.column_stack([rng.uniform(0, 2, 4000), rng.normal(0, 0.01, 4000), rng.uniform(0, 1, 4000)])
    Q = np.column_stack([rng.uniform(-1, 3, 30000), rng.uniform(-0.5, 0.5, 30000), rng.uniform(-0.5, 1.5, 30000)])
    Q[:50] = A[:50]                                            # exact hits at distance 0
    return A, Q


@pytest.mark.parametrize("bound", [0.05, 0.2, 0.7])
def test_the_cell_index_equals_the_bounded_kd_tree_query(bound):
    for seed in range(3):
        A, Q = _clouds(seed)
        want, _ = cKDTree(A).query(Q, k=1, workers=1, distance_upper_bound=bound)
        got = VT.CellIndex(A, bound, DEV, block_entries=50_000).nearest_within(Q, block=7_000)
        assert got.shape == want.shape
        fin = np.isfinite(want)
        assert np.array_equal(fin, np.isfinite(got))
        assert fin.sum() > 100 and (~fin).sum() > 100
        assert np.allclose(got[fin], want[fin], rtol=1e-12, atol=0)


def test_ambiguity_on_the_device_equals_numpy():
    rng = np.random.default_rng(11)
    xyz = rng.uniform(-5, 5, (120_000, 3))
    points_by_oid = {}
    label_of = {}
    for o in range(12):
        c = rng.uniform(-4, 4, 3)
        idx = np.flatnonzero(np.linalg.norm(xyz - c, axis=1) < rng.uniform(0.5, 1.5))
        points_by_oid[o] = idx
        label_of[o] = "wall" if o % 3 else "desk"
    copy_a = xyz[points_by_oid[0]][: len(points_by_oid[0]) // 2]
    for mag in (0.1, 0.8, 2.5):
        VD.use_device("numpy")
        try:
            a = VD.ambiguity(copy_a, 0, label_of[0], points_by_oid, label_of, xyz, mag)
        finally:
            VD.use_device("cpu")
        VD.use_device(DEV)
        try:
            b = VD.ambiguity(copy_a, 0, label_of[0], points_by_oid, label_of, xyz, mag)
        finally:
            VD.use_device("cpu")
        assert a[0] == b[0] and a[1] == b[1], mag
        assert a[2].keys() == b[2].keys()
        for k in a[2]:
            assert np.isclose(a[2][k], b[2][k], rtol=1e-12, atol=0)
