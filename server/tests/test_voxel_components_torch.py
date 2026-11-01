"""The voxel connected components on the device (segmentation/voxel_components_torch.py) against the three
numpy recipes they replace: `segmentation.pipeline._voxel_components` (the OBB's core), `_label_occupied_cells`
(the co-visible split) and `reconstruction.loops.instance_loops.disjoint_clusters` — label for label, group
for group, on blobs with chains, touching corners (26-connectivity) and isolated strays."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from segmentation import pipeline as SP  # noqa: E402
from segmentation import voxel_components_torch as VC  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _blobs(seed=0, n_strays=25):
    """Three dense blobs, two joined by a thin diagonal chain of points (corner contact), plus strays."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 0.4, (3000, 3))
    b = rng.uniform(0, 0.4, (2000, 3)) + [1.0, 0.0, 0.0]
    c = rng.uniform(0, 0.3, (1500, 3)) + [0.0, 2.0, 0.0]
    t = np.linspace(0, 1, 60)
    chain = np.column_stack([0.4 + 0.6 * t, 0.4 * (1 - t), 0.4 * (1 - t)]) + rng.normal(0, 0.002, (60, 3))
    strays = rng.uniform(-3, 5, (n_strays, 3))
    return np.concatenate([a, b, c, chain, strays])


@pytest.mark.parametrize("voxel", [0.05, 0.1, 0.021])
def test_voxel_components_equal_the_union_find(voxel):
    for seed in range(3):
        P = _blobs(seed)
        want = SP._voxel_components(P, voxel)
        got = VC.voxel_components(P, voxel, DEV)
        assert np.array_equal(want, got)
        assert len(np.unique(want)) >= 3


def test_label_occupied_cells_equal_scipy_relabelled():
    for seed in range(3):
        P = _blobs(seed)
        for gap in (0.05, 0.12):
            k = np.floor(P / gap).astype(np.int64)
            ki = k - k.min(0)
            want, n_w = SP._label_occupied_cells(ki)
            got, n_g = VC.label_occupied_cells(ki, DEV)
            assert n_w == n_g and np.array_equal(want, got)


def test_single_voxel_and_single_point_edge_cases():
    P = np.tile([[0.01, 0.02, 0.03]], (10, 1))
    assert np.array_equal(VC.voxel_components(P, 0.05, DEV), SP._voxel_components(P, 0.05))
    got, n = VC.label_occupied_cells(np.zeros((10, 3), np.int64), DEV)
    assert n == 1 and np.array_equal(got, np.zeros(10, np.int64))
    one = np.array([[0.3, 0.3, 0.3]])
    assert np.array_equal(VC.voxel_components(one, 0.05, DEV), SP._voxel_components(one, 0.05))


def test_disjoint_clusters_equal_the_numpy_recipe():
    from types import SimpleNamespace
    from reconstruction.loops import instance_loops as IL
    cfg = SimpleNamespace(cluster_min_points=40, dbscan_eps_m=0.06)
    for seed in range(3):
        P = _blobs(seed, n_strays=60)
        want = IL._disjoint_clusters_numpy(P, cfg)
        got = IL.disjoint_clusters(P, cfg)
        assert len(want) == len(got) >= 2
        for w, g in zip(want, got):
            assert np.array_equal(w, g)
