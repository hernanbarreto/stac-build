"""reconstruction/gpu_cloud_clean after the 2026-11-09 speed work (USER: "no podemos pasar tantos pasos sin saber qué
está pasando"; the load on one core): the voxel slabs found by ONE sort must leave the SAME survivors, index for index,
as the per-slab scan of the version before (read from git) — several slabs, voxel ties included. CPU (the card is the
running job's)."""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parents[1]
REPO = SERVER.parent


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_voxel_survivors_equal_the_previous_version(tmp_path):
    old_src = subprocess.run(["git", "show", "HEAD:server/reconstruction/gpu_cloud_clean.py"], cwd=str(REPO),
                             capture_output=True, text=True, check=True).stdout
    (tmp_path / "gcc_old.py").write_text(old_src)
    OLD = _module(tmp_path / "gcc_old.py", "gcc_old")
    NEW = _module(SERVER / "reconstruction" / "gpu_cloud_clean.py", "gcc_new")
    rng = np.random.default_rng(11)
    xyz = rng.uniform([0, 0, 0], [3.0, 1.0, 1.0], size=(120000, 3)).astype(np.float32)
    xyz[:12000] = (np.round(xyz[:12000] / 0.02) * 0.02 + 0.01).astype(np.float32)    # exact cell centres: ties
    xyz[12000:14000] = xyz[:2000]                                                      # duplicates: exact ties
    for tile in (10_000, 37_000, 10**9):                                               # many slabs, few, one
        a = OLD._voxel_keep(xyz, 0.02, tile_points=tile, device="cpu")
        b = NEW._voxel_keep(xyz, 0.02, tile_points=tile, device="cpu")
        assert np.array_equal(a, b), f"tile_points {tile}: the survivors changed"


def test_one_sort_slabs_equal_the_per_slab_scan():
    """The slab members found by one stable sort equal np.flatnonzero per slab — set AND order."""
    G = _module(SERVER / "reconstruction" / "gpu_cloud_clean.py", "gcc")
    rng = np.random.default_rng(3)
    ca = rng.integers(-50, 50, size=200000).astype(np.int64)
    bounds = G._slab_bounds(ca - ca.min(), 7) + ca.min()
    order = np.argsort(ca, kind="stable")
    cs = ca[order]
    st, en = np.searchsorted(cs, bounds[:-1], side="left"), np.searchsorted(cs, bounds[1:], side="left")
    for t in range(len(bounds) - 1):
        ref = np.flatnonzero((ca >= bounds[t]) & (ca < bounds[t + 1]))
        assert np.array_equal(np.sort(order[st[t]:en[t]]), ref)
