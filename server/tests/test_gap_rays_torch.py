"""The ray judges of the co-visible split on the card equal the cKDTree recipe, judge for judge
(USER 2026-10-15: observatorio's split ran for hours on a 99 M-point cKDTree)."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from segmentation import pipeline as P
from segmentation.gap_rays_torch import FreeSpaceIndex, gap_rays_t


def _scene():
    rng = np.random.default_rng(5)
    floor = np.column_stack([rng.uniform(-4, 4, 60000), rng.uniform(-3, 3, 60000), rng.normal(0, 0.004, 60000)])
    wall = np.column_stack([rng.uniform(-4, 4, 30000), np.full(30000, 3.0) + rng.normal(0, 0.004, 30000), rng.uniform(0, 2.5, 30000)])
    desk_a = np.column_stack([rng.uniform(-2.0, -1.0, 15000), rng.uniform(0, 0.6, 15000), np.full(15000, 0.75) + rng.normal(0, 0.004, 15000)])
    desk_b = np.column_stack([rng.uniform(0.5, 1.5, 15000), rng.uniform(0, 0.6, 15000), np.full(15000, 0.75) + rng.normal(0, 0.004, 15000)])
    cloud = np.vstack([floor, wall, desk_a, desk_b])
    cams = np.array([[-3.0, -2.5, 1.6], [0.0, -2.5, 1.6], [2.5, -2.0, 1.6], [-0.5, 1.5, 1.7]])
    return cloud, desk_a, desk_b, cams


def test_any_within_is_exact_against_brute_force():
    cloud, da, db, cams = _scene()
    idx = FreeSpaceIndex(cloud, 0.05, device="cpu")
    rng = np.random.default_rng(9)
    S = np.vstack([rng.uniform(-4.5, 4.5, (3000, 3)) * np.array([1, 0.8, 0.6]) + np.array([0, 0, 0.8]),
                   cloud[rng.choice(len(cloud), 500)] + rng.normal(0, 0.03, (500, 3))])
    got = idx.any_within(S, 0.05)
    d, _ = cKDTree(cloud).query(S, k=1, distance_upper_bound=0.05)
    ref = np.isfinite(d)
    assert np.array_equal(got, ref)
    assert 0.1 < got.mean() < 0.9


def test_the_torch_judges_equal_the_ckdtree_judges():
    cloud, da, db, cams = _scene()
    tree = cKDTree(cloud)
    gap, reach = 0.10, float(np.linalg.norm(cloud.max(0) - cloud.min(0)))
    ref = P._gap_rays(da, db, tree, cams, gap, reach)
    idx = FreeSpaceIndex(cloud, gap / 2.0, device="cpu")
    got = gap_rays_t(da, db, idx, cams, gap, reach)
    assert len(ref) > 10 and got == ref, (len(ref), len(got))
    assert P._gap_rays(da, db, idx, cams, gap, reach) == ref, "the pipeline's own entry dispatches to the card"
    # the decision too
    d_ref = P._gap_is_free_space(da, db, tree, cams, gap, reach, factor=2.0, confidence=0.95, min_judges=5)
    d_got = P._gap_is_free_space(da, db, idx, cams, gap, reach, factor=2.0, confidence=0.95, min_judges=5)
    assert d_got == d_ref and d_ref["n_rays"] == len(ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_on_cuda_the_judges_equal_the_cpu_ones():
    cloud, da, db, cams = _scene()
    gap, reach = 0.10, float(np.linalg.norm(cloud.max(0) - cloud.min(0)))
    cpu = gap_rays_t(da, db, FreeSpaceIndex(cloud, gap / 2.0, device="cpu"), cams, gap, reach)
    gpu = [gap_rays_t(da, db, FreeSpaceIndex(cloud, gap / 2.0, device="cuda"), cams, gap, reach) for _ in range(2)]
    assert gpu[0] == cpu and gpu[1] == cpu
