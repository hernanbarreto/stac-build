"""Deterministic cloud cleaning (2026-09-28): the voxel pick and the SOR of
reconstruction.gpu_cloud_clean give the SAME surviving set whatever the card's
free VRAM and whatever the memory bounds, and the SOR statistic is the exact
kNN mean distance. A point with no neighbour within the reach is isolated:
dropped and kept out of mean/std, as the old 27-cell box did (+inf) — far
floaters never set the cut-off.

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


def _brute_nearest_d2(xyz: np.ndarray) -> np.ndarray:
    """Squared distance to the nearest other point, same float64 arithmetic."""
    p = xyz.astype(np.float64)
    out = np.empty(len(p))
    for i in range(len(p)):
        d = p - p[i]
        d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
        d2[i] = np.inf
        out[i] = d2.min()
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
    # the grid is anchored at the WORLD origin (plan point 102): cell = floor(x / voxel)
    c = np.floor(p / voxel).astype(np.int64)
    ctr = (c + 0.5) * voxel
    d2 = ((p - ctr) ** 2).sum(1)
    best = {}
    for i, key in enumerate(map(tuple, c)):
        if key not in best or d2[i] < d2[best[key]]:
            best[key] = i
    assert np.array_equal(keep, np.sort(np.array(list(best.values()))))


def test_same_cloud_two_bound_sets_bit_identical(monkeypatch):
    """voxel pick → SOR under two very different sets of memory bounds gives
    the same bytes — and neither run asks the card for its free VRAM."""
    def _no_vram_probe(*a, **k):
        raise AssertionError("the cleaning must not read the card's free VRAM")
    monkeypatch.setattr(torch.cuda, "mem_get_info", _no_vram_probe)
    xyz = _cloud(3)
    results = []
    for tp, qb, cb in ((len(xyz), 1 << 20, 1 << 30), (997, 61, 2000)):
        vk = G._voxel_keep(xyz, 0.004, tile_points=tp, device=DEV)
        sub = xyz[vk]
        keep, mu, sd = G._sor_keep(sub, 8, 1.0, 0.015, device=DEV,
                                   query_block=qb, candidate_budget=cb)
        results.append((vk, sub[keep], mu, sd))
    (a, ca, mua, sda), (b, cb_, mub, sdb) = results
    assert np.array_equal(a, b)
    assert ca.tobytes() == cb_.tobytes()
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
    # the SOR: isolated = no neighbour within the reach (brute force), dropped
    # and out of mu/sd; the others judged on their exact kNN mean
    reach = 0.01
    iso = _brute_nearest_d2(xyz) > reach * reach
    assert 0 < iso.sum() < len(xyz) and iso[-12:].all()
    keep, mu, sd = G._sor_keep(xyz, k, 1.0, reach, device=DEV,
                               query_block=333, candidate_budget=5000)
    assert mu == float(ref[~iso].mean()) and sd == float(ref[~iso].std())
    assert np.array_equal(keep, ~iso & (ref <= mu + sd))


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


def test_far_floaters_do_not_set_the_sor_cutoff():
    """Review 2026-09-28: with the exact kNN and no isolation reach, twenty
    floaters 10-30 m away grew sd until the cut-off kept EVERY near-surface
    noise point. Isolated points are out of mean/std, so adding them changes
    nothing for the rest — and the noise is still removed."""
    rng = np.random.default_rng(11)
    n_plane, n_noise, side = 20_000, 300, 1.4                 # ~1 pt/cm²
    plane = np.column_stack([rng.uniform(0, side, n_plane), rng.uniform(0, side, n_plane),
                             rng.normal(0, 0.001, n_plane)])
    off = rng.uniform(0.01, 0.04, n_noise) * rng.choice([-1, 1], n_noise)
    noise = np.column_stack([rng.uniform(0, side, n_noise),
                             rng.uniform(0, side, n_noise), off])
    u = rng.normal(size=(20, 3))
    fly = side / 2 + u / np.linalg.norm(u, axis=1)[:, None] * rng.uniform(10, 30, 20)[:, None]
    base = np.vstack([plane, noise]).astype(np.float32)
    kw = dict(device=DEV, query_block=1 << 16, candidate_budget=1 << 22)
    k0, mu0, sd0 = G._sor_keep(base, 8, 3.0, 0.015, **kw)
    k1, mu1, sd1 = G._sor_keep(np.vstack([base, fly.astype(np.float32)]), 8, 3.0, 0.015, **kw)
    assert (mu1, sd1) == (mu0, sd0)                            # the cut-off did not move
    assert np.array_equal(k1[:len(base)], k0)                  # nor any survivor
    assert not k1[len(base):].any()                            # every floater dropped
    assert k0[n_plane:].mean() < 0.25                          # near-surface noise removed
    assert k0[:n_plane].mean() > 0.99                          # the surface kept


def test_bounds_come_from_config():
    b = G._clean_bounds()
    assert set(b) == {"tile_points", "knn_query_block", "knn_candidate_budget"}
    assert all(isinstance(v, int) and v > 0 for v in b.values())
    assert G._sor_reach(0.005) == pytest.approx(0.015)         # max(3 × 5 mm, 1 cm)
    assert G._sor_reach(0.001) == pytest.approx(0.01)


@pytest.mark.parametrize("section,key", [
    (("postprocessing", "clean_bounds"), "tile_points"),
    (("postprocessing", "clean_bounds"), "knn_query_block"),
    (("postprocessing", "clean_bounds"), "knn_candidate_budget"),
    (("postprocessing",), "sor_reach_voxels"),
    (("postprocessing",), "sor_reach_min_m"),
])
def test_missing_key_fails_naming_it(tmp_path, monkeypatch, section, key):
    import yaml
    from reconstruction import grid_knn
    raw = yaml.safe_load(grid_knn._CONFIG.read_text())
    sec = raw
    for p in section:
        sec = sec[p]
    del sec[key]
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(grid_knn, "_CONFIG", cfg)
    dotted = ".".join(section + (key,))
    with pytest.raises(KeyError, match=dotted.replace(".", r"\.")):
        G._clean_bounds() if section[-1] == "clean_bounds" else G._sor_reach(0.005)


def test_voxel_grid_is_world_anchored_an_extra_point_changes_only_its_cell(capsys):
    """docs/plan_determinismo.md point 102 (DECIDIDO): the grid used to start at the cloud's
    minimum, so one flyer 1.3 mm below min-x moved every cell boundary and 59.6 % of the survivors
    (measured). Anchored at the world origin, the extra point changes only its own cell — on a
    cloud with negative coordinates and over several slabs — and the old grid's offset (the
    minimum modulo the voxel) is reported below one voxel."""
    voxel = 0.005
    xyz = _cloud(3, n_surf=4000, n_fly=20) - np.array([0.7, 0.3, 0.2])      # negative coordinates too
    keep = G._voxel_keep(xyz, voxel, tile_points=900, device=DEV)
    out = capsys.readouterr().out
    assert "anchored at the world origin" in out and "below one voxel" in out
    extra = xyz.min(0) - np.array([0.0013, 0.0, 0.0])                        # 1.3 mm below min-x
    xyz2 = np.vstack([xyz, extra[None]])
    keep2 = G._voxel_keep(xyz2, voxel, tile_points=900, device=DEV)
    cell = lambda p: tuple(np.floor(np.asarray(p, np.float64) / voxel).astype(np.int64))   # noqa: E731
    touched = cell(extra)
    old = {i for i in keep if cell(xyz[i]) != touched}
    new = {i for i in keep2 if i < len(xyz) and cell(xyz[i]) != touched}
    assert old == new, "survivors outside the extra point's cell must not change"
    assert len(keep2) in (len(keep), len(keep) + 1)
    # the SAME survivors whatever the slab count, with negative cells
    assert np.array_equal(keep, G._voxel_keep(xyz, voxel, tile_points=len(xyz), device=DEV))
    assert np.array_equal(keep, G._voxel_keep(xyz, voxel, tile_points=137, device=DEV))
    # the old grid's boundaries sat (min mod voxel) past the world grid's: below one voxel
    shift = xyz.min(0) - np.floor(xyz.min(0) / voxel) * voxel
    assert np.all(shift >= 0) and np.all(shift < voxel)
