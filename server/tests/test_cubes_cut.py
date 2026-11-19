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


def test_the_overlap_is_a_parameter_and_one_half_is_the_plan_of_before():
    """USER 2026-11-19, on the cubes that guessed across a hole of the walk: "lo que hay que hacer … es que
    compartan muchas mas imagenes entre cubos". The share of its keyframes a cube has in common with the next is
    ``overlap`` (reconstruction.vggtomega.structure.cube_overlap): one half gives EXACTLY the plan of before; three
    quarters makes every pair of neighbours share at least three quarters of the smaller, puts a keyframe of the
    middle of the walk in four cubes, and loses none."""
    c2w = np.tile(np.eye(4), (400, 1, 1))
    c2w[:, 0, 3] = 0.05 * np.arange(400)
    units = [{"keyframes": [i], "views": 1, "turn": []} for i in range(400)]

    def plan(**kw):
        return CB.plan_cubes(c2w, units, metres_per_unit=1.0, cube_m=1000.0, max_views=64, device="cpu", cut="metres",
                             min_kf=32, **kw)
    before, half = plan(), plan(overlap=0.5)
    assert [c["keyframes"] for c in before["cubes"]] == [c["keyframes"] for c in half["cubes"]]
    assert before["owner"] == half["owner"] and half["overlap"] == 0.5
    wide = plan(overlap=0.75)
    cubes = [set(c["keyframes"]) for c in wide["cubes"]]
    assert len(cubes) > len(before["cubes"]) * 1.7, (len(cubes), len(before["cubes"]))
    for a, b in zip(cubes, cubes[1:]):
        assert len(a & b) * 4 >= 3 * min(len(a), len(b)), (len(a & b), len(a), len(b))
    assert sorted({k for c in cubes for k in c}) == list(range(400)) and max(len(c) for c in cubes) <= 64
    memberships = [sum(k in c for c in cubes) for k in range(120, 280)]
    assert min(memberships) >= 3 and max(memberships) == 4, (min(memberships), max(memberships))
    assert len(set(wide["owner"])) == len(cubes), "every cube owns keyframes"
    for bad in (0, 1, 1.5, True, "0.75"):
        try:
            plan(overlap=bad)
        except CB.CubePlanError:
            continue
        raise AssertionError(f"overlap {bad!r} was accepted")

