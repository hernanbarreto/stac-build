"""reconstruction/cubes.py — the cut's two hard rules of USER 2026-11-08: cubes under
min_chunk_keyframes FUSE into their neighbours ("los chunks pequeños deben fusionarse, no debemos
tener menos de 32 kf"), and consecutive cubes ALWAYS share at least half of the smaller one's
keyframes ("siempre deben tener el 50 % de solape" — the 99.8 cm seam stood on 5 shared frames)."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "cubes", Path(__file__).resolve().parent.parent / "reconstruction" / "cubes.py")
CB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CB)


def _walk(n=30, step=0.2):
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = step * np.arange(n)
    units = [{"keyframes": [i], "views": 1, "turn": []} for i in range(n)]
    return c2w, units


def _plan(min_kf, cube_m=1.0):
    c2w, units = _walk()
    return CB.plan_cubes(c2w, units, metres_per_unit=1.0, cube_m=cube_m, max_views=100,
                         device="cpu", cut="metres", min_kf=min_kf)


def test_small_cubes_fuse_to_the_minimum():
    p0 = _plan(min_kf=0)
    sizes0 = [len(c["keyframes"]) for c in p0["cubes"]]
    assert min(sizes0) < 8, "the raw metres cut makes small cubes on this walk"
    p = _plan(min_kf=8)
    sizes = [len(c["keyframes"]) for c in p["cubes"]]
    assert min(sizes) >= 8, f"every cube holds at least min_kf keyframes: {sizes}"
    assert len(p["cubes"]) < len(p0["cubes"])
    covered = sorted({k for c in p["cubes"] for k in c["keyframes"]})
    assert covered == list(range(30)), "no keyframe is lost by the fusion"


def test_neighbours_always_share_half_of_the_smaller():
    for min_kf in (0, 8):
        p = _plan(min_kf=min_kf)
        cubes = [set(c["keyframes"]) for c in p["cubes"]]
        for a, b in zip(cubes, cubes[1:]):
            shared = len(a & b)
            assert shared * 2 >= min(len(a), len(b)), \
                f"min_kf={min_kf}: shared {shared} of sizes {len(a)}/{len(b)} — under 50 % of the smaller"


def test_the_cut_is_deterministic():
    a, b = _plan(min_kf=8), _plan(min_kf=8)
    assert [c["keyframes"] for c in a["cubes"]] == [c["keyframes"] for c in b["cubes"]]
    assert a["owner"] == b["owner"]
