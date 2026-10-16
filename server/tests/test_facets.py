"""Plane SEGMENTS (USER 2026-10-16: "segmentos de plano, no único plano que describe todo"): a flat
wall is one segment even when the cloud doubles it, a curved wall a chain of facets, a step two
parallel segments with their exact offset, a ramp its own tilted segment, a blob the residue."""
import numpy as np

from reconstruction.surface_fit.facets import grow_plane_segments, object_cell


def _rng(seed=0):
    return np.random.default_rng(seed)


def _wall(rng, n, x0, x1, y0, y1, z, noise=0.003):
    P = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), np.full(n, z)])
    P[:, 2] += rng.normal(0, noise, n)
    return P


def test_a_doubled_flat_wall_is_one_segment_at_the_resolution():
    rng = _rng(1)
    a = _wall(rng, 30000, 0, 2, 0, 1.5, 0.0)
    b = _wall(rng, 30000, 0, 2, 0, 1.5, 0.03)            # the revisit copy, 3 cm behind
    P = np.vstack([a, b])
    cell = 0.10
    reg, regions = grow_plane_segments(P, cell, tol=cell / 2.0)
    assert regions[0]["n"] > 0.98 * len(P) and (reg >= 0).mean() > 0.98
    assert abs(abs(float(regions[0]["normal"][2])) - 1.0) < 1e-3
    # at the cloud's own noise the copies come apart: the tolerance has to be the resolution's
    reg2, regions2 = grow_plane_segments(P, cell, tol=0.009)
    assert not regions2 or regions2[0]["n"] < 0.6 * len(P)      # no cell is a surface at 9 mm, or pieces


def test_a_curved_wall_is_a_chain_of_facets_as_wide_as_its_curvature_allows():
    rng = _rng(2)
    r = 3.0
    th = rng.uniform(0, np.pi / 2, 80000)
    P = np.column_stack([r * np.cos(th), r * np.sin(th), rng.uniform(0, 2.0, 80000)])
    P[:, :2] += rng.normal(0, 0.003, (80000, 2))
    cell, tol = 0.10, 0.05
    reg, regions = grow_plane_segments(P, cell, tol=tol)
    big = [q for q in regions if q["n"] > 0.02 * len(P)]
    chord = 2.0 * np.sqrt(2.0 * r * tol)                     # ≈ 1.1 m of arc per facet
    expected = (np.pi / 2 * r) / chord
    assert 0.5 * expected <= len(big) <= 3 * expected, (len(big), expected)
    assert (reg >= 0).mean() > 0.9
    # every facet is vertical and its normal points radially (a tangent plane of the cylinder)
    for q in big:
        assert abs(float(q["normal"][2])) < 0.05
        c = q["centroid"]
        radial = c[:2] / np.linalg.norm(c[:2])
        assert abs(abs(float(q["normal"][:2] @ radial)) - 1.0) < 0.05


def test_a_step_is_two_parallel_segments_with_the_exact_offset():
    rng = _rng(3)
    lo = _wall(rng, 30000, 0, 2, 0, 2, 0.0)
    hi = _wall(rng, 30000, 2, 4, 0, 2, 0.18)                 # 18 cm up
    riser = np.column_stack([np.full(6000, 2.0) + rng.normal(0, 0.003, 6000), rng.uniform(0, 2, 6000),
                             rng.uniform(0, 0.18, 6000)])
    P = np.vstack([lo, hi, riser])
    reg, regions = grow_plane_segments(P, 0.10, tol=0.02)
    flats = [q for q in regions if abs(float(q["normal"][2])) > 0.99]
    assert len(flats) == 2 and all(q["n"] > 0.9 * 30000 for q in flats)
    z = sorted(float(q["centroid"][2]) for q in flats)
    assert abs((z[1] - z[0]) - 0.18) < 0.005                 # the step's exact height
    vert = [q for q in regions if abs(float(q["normal"][0])) > 0.99]
    # the riser is its own segment (its foot and top, within the tolerance of the treads, go
    # with them — the planes meet on their intersection lines when meshed)
    assert vert and vert[0]["n"] > 0.6 * 6000, [(q["n"], q["normal"].round(2).tolist()) for q in regions[:5]]


def test_a_ramp_next_to_the_floor_keeps_its_slope():
    rng = _rng(4)
    floor = _wall(rng, 40000, 0, 4, 0, 2, 0.0)
    x = rng.uniform(4, 6, 4000)                              # a 2 m ramp at 5°, sparse
    ramp = np.column_stack([x, rng.uniform(0, 2, 4000), (x - 4) * np.tan(np.radians(5)) + rng.normal(0, 0.003, 4000)])
    P = np.vstack([floor, ramp])
    reg, regions = grow_plane_segments(P, 0.10, tol=0.05)
    assert regions[0]["n"] > 0.95 * 40000 and abs(float(regions[0]["normal"][2])) > 0.999
    tilted = [q for q in regions if 3.0 < np.degrees(np.arccos(abs(float(q["normal"][2])))) < 7.0]
    # the ramp's first half-metre lies within the tolerance of the floor and goes with it; the
    # rest is the ramp, at its measured slope
    assert tilted and tilted[0]["n"] > 0.6 * 4000, [np.degrees(np.arccos(abs(float(q["normal"][2])))) for q in regions[:5]]
    assert abs(np.degrees(np.arccos(abs(float(tilted[0]["normal"][2])))) - 5.0) < 0.3


def test_a_blob_is_residue_not_a_plane():
    rng = _rng(5)
    blob = rng.normal(0, 0.15, (20000, 3))
    reg, regions = grow_plane_segments(blob, 0.10, tol=0.02)
    assert (reg < 0).mean() > 0.9 and all(q["n"] < 0.05 * len(blob) for q in regions), (len(regions), (reg < 0).mean())


def test_the_objects_resolution_follows_its_smallest_extent():
    assert abs(object_cell(np.array([2.0, 0.30, 0.29]), 0.01, 0.10, 4) - 0.0725) < 1e-9
    assert object_cell(np.array([10.0, 8.0, 0.1]), 0.02, 0.10, 4) == 0.08             # four sampling gaps
    assert object_cell(np.array([10.0, 8.0, 3.0]), 0.01, 0.10, 4) == 0.10             # the user's cap
