"""Deterministic cloud cleaning (2026-09-28): the voxel pick and the SOR of
reconstruction.gpu_cloud_clean give the SAME surviving set whatever the card's
free VRAM and whatever the memory bounds, and the SOR statistic is the exact
kNN mean distance — including for the isolated floater the old 27-cell box
averaged over "the few it had" (or called +inf).

Runs on the CPU device: the torch code is the same on CUDA."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reconstruction import gpu_cloud_clean as G              # noqa: E402
from reconstruction.grid_knn import grid_knn                 # noqa: E402

DEV = "cpu"


def _cloud(seed=0, n_surf=6000, n_fly=40):
    """Two noisy planes at 1-5 mm spacing + a handful of floaters far away
    (up to 0.5 m) — the points the old box search could not see."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([rng.uniform(0, 1.2, n_surf), rng.uniform(0, 0.8, n_surf),
                         rng.normal(0, 0.002, n_surf)])
    b = np.column_stack([rng.uniform(0, 1.2, n_surf), rng.normal(0.9, 0.002, n_surf),
                         rng.uniform(0, 0.6, n_surf)])
    fly = rng.uniform(-0.5, 1.7, (n_fly, 3))
    return np.vstack([a, b, fly]).astype(np.float32)


def _brute_mean_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """Exact mean distance to the k nearest (self excluded), same float64
    arithmetic: (dx·dx + dy·dy) + dz·dz, sorted, sqrt, summed in order."""
    p = xyz.astype(np.float64)
    out = np.empty(len(p))
    for i in range(len(p)):
        d = p - p[i]
        d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        d2[i] = np.inf
        sel = np.sqrt(np.sort(d2)[:k])
        s = sel[0]
        for j in range(1, k):
            s = s + sel[j]
        out[i] = s / k
    return out


def test_voxel_pick_independent_of_tiling():
    xyz = _cloud(1)
    one = G._voxel_keep(xyz, 0.005, tile_points=len(xyz), device=DEV)
    for tp in (len(xyz) // 3, 997, 101):
        many = G._voxel_keep(xyz, 0.005, tile_points=tp, device=DEV)
        assert np.array_equal(one, many), f"tile_points={tp} changed the survivors"


def test_voxel_pick_is_the_nearest_to_each_cell_centre():
    xyz = _cloud(2, n_surf=800, n_fly=5)
    voxel = 0.01
    keep = G._voxel_keep(xyz, voxel, tile_points=100, device=DEV)
    p = xyz.astype(np.float64)
    o = p.min(0)
    c = np.floor((p - o) / voxel).astype(np.int64)
    ctr = o + (c + 0.5) * voxel
    d2 = ((p - ctr) ** 2).sum(1)
    best = {}
    for i, key in enumerate(map(tuple, c)):
        if key not in best or d2[i] < d2[best[key]]:
            best[key] = i
    assert np.array_equal(keep, np.sort(np.array(list(best.values()))))


def test_same_cloud_two_vram_budgets_bit_identical(monkeypatch):
    """The free VRAM is not an input any more: two simulated cards give the
    same cloud (and the functions never ask)."""
    xyz = _cloud(3)
    results = []
    for free in (48 * 2 ** 30, 300 * 2 ** 20):
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a, _f=free: (_f, 48 * 2 ** 30))
        vk = G._voxel_keep(xyz, 0.004, tile_points=5000, device=DEV)
        sub = xyz[vk]
        keep, mu, sd = G._sor_keep(sub, 8, 1.0, 0.015, device=DEV,
                                   query_block=4096, candidate_budget=1 << 20)
        results.append((vk, sub[keep], mu, sd))
    (a, ca, mua, sda), (b, cb, mub, sdb) = results
    assert np.array_equal(a, b)
    assert ca.tobytes() == cb.tobytes()
    assert (mua, sda) == (mub, sdb)


def test_sor_statistic_is_the_exact_knn_mean():
    xyz = _cloud(4, n_surf=700, n_fly=12)
    k = 8
    ref = _brute_mean_knn(xyz, k)
    mean_d = np.empty(len(xyz))
    pts = torch.from_numpy(xyz)
    d2_all = np.empty((len(xyz), k))
    for q, idx, d2 in grid_knn(pts, k, 0.01, query_block=333, candidate_budget=5000):
        d2_all[q.numpy()] = d2.numpy()
    d = np.sqrt(d2_all)
    s = d[:, 0].copy()
    for j in range(1, k):
        s += d[:, j]
    mean_d = s / k
    assert np.array_equal(mean_d, ref)
    # and the floaters — far beyond one box of 1 cm — carry their true distance
    assert np.isfinite(mean_d).all() and mean_d[-12:].min() > 0.05
    keep, mu, sd = G._sor_keep(xyz, k, 1.0, 0.01, device=DEV,
                               query_block=333, candidate_budget=5000)
    assert mu == float(ref.mean()) and sd == float(ref.std())
    assert np.array_equal(keep, ref <= mu + sd)


def test_sor_independent_of_blocks():
    xyz = _cloud(5)
    a = G._sor_keep(xyz, 8, 2.0, 0.015, device=DEV, query_block=1 << 20,
                    candidate_budget=1 << 30)
    b = G._sor_keep(xyz, 8, 2.0, 0.015, device=DEV, query_block=77,
                    candidate_budget=900)
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_radius_knn_matches_brute_force():
    xyz = _cloud(6, n_surf=500, n_fly=8).astype(np.float64)
    k, r = 10, 0.03
    pts = torch.from_numpy(xyz)
    got_i = np.full((len(xyz), k), -1)
    got_d = np.full((len(xyz), k), np.inf)
    for q, idx, d2 in grid_knn(pts, k, None, radius=r, query_block=200,
                               candidate_budget=3000):
        got_i[q.numpy()] = idx.numpy()
        got_d[q.numpy()] = d2.numpy()
    for i in range(len(xyz)):
        d = xyz - xyz[i]
        d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        d2[i] = np.inf
        within = np.flatnonzero(d2 <= r * r)
        order = within[np.argsort(d2[within], kind="stable")][:k]
        m = len(order)
        assert np.array_equal(got_d[i][:m], d2[order])          # exact, ascending
        assert np.array_equal(got_i[i][:m], order)
        assert (got_i[i][m:] == -1).all() and np.isinf(got_d[i][m:]).all()


def test_bounds_come_from_config():
    b = G._clean_bounds()
    assert set(b) == {"tile_points", "knn_query_block", "knn_candidate_budget"}
    assert all(isinstance(v, int) and v > 0 for v in b.values())
