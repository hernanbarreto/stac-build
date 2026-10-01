"""The edge-keeping vote of the epoch-8 recipe (USER 2026-10-01: *"lo más importante es que los objetos
deben tener mucha definición, corte en los filos, las aristas"*): mixed pixels at depth steps snapped to
the side their SAM3 mask says, the edge band measured from where the confidence floor erodes, the
two-sided vote, the repair from the neighbours' splats, and the consecutive-keyframe agreement that
chooses the bend window. Synthetic scenes, no GPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.corrected_cloud import (agreeing_median, consecutive_ratio, depth_steps,  # noqa: E402
                                       edge_band, edge_vote_decision, mixed_pixels, ring_distance,
                                       ring_histogram, snap_mixed, two_sided_vote, window_extremes)

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])
TAU = 0.02


def _cam(x):
    c2w = np.eye(4)
    c2w[0, 3] = x                                        # cameras side by side, looking along +z
    return c2w


def _step_scene():
    """Left of column 40 a surface at 2 m, right of it a wall at 4 m, column 40 the ramp (3 m)."""
    z = np.full((H, W), 4.0, np.float32)
    z[:, :40] = 2.0
    z[:, 40] = 3.0
    return z, np.ones((H, W), bool)


# ── mixed pixels, steps, snapping ───────────────────────────────────────────

def test_window_extremes_ignore_invalid_pixels():
    z, valid = _step_scene()
    valid[:, 41] = False
    mn, mx = window_extremes(z, valid)
    assert mn[10, 40] == 2.0 and mx[10, 40] == 3.0      # the 4 m column next to it is not valid
    assert mx[10, 42] == 4.0 and mn[10, 42] == 4.0


def test_the_ramp_is_mixed_and_a_smooth_slope_is_not():
    z, valid = _step_scene()
    mixed = mixed_pixels(z, valid, TAU)
    assert mixed[:, 40].all() and mixed.sum() == H
    assert depth_steps(z, valid, TAU)[:, 39:42].all() and not depth_steps(z, valid, TAU)[:, :38].any()
    slope = np.tile(np.linspace(3.0, 3.3, W, dtype=np.float32), (H, 1))   # 0.13 % per pixel
    assert not mixed_pixels(slope, valid, TAU).any() and not depth_steps(slope, valid, TAU).any()


def test_a_mixed_pixel_goes_to_the_side_its_mask_says():
    z, valid = _step_scene()
    lab = np.zeros((H, W), np.int64)
    lab[:, :41] = 5                                      # the object's mask covers the ramp
    out, mixed, snapped = snap_mixed(z, valid, lab, TAU)
    assert snapped[:, 40].all() and np.allclose(out[:, 40], 2.0)
    lab[:, 40] = 0                                       # the mask stops before the ramp: background
    out, _, snapped = snap_mixed(z, valid, lab, TAU)
    assert snapped[:, 40].all() and np.allclose(out[:, 40], 4.0)
    assert np.array_equal(out[:, :40], z[:, :40]) and np.array_equal(out[:, 41:], z[:, 41:])


def test_a_mask_that_does_not_follow_the_step_or_overlaps_decides_nothing():
    z, valid = _step_scene()
    out, _, snapped = snap_mixed(z, valid, np.full((H, W), 5, np.int64), TAU)   # one label both sides
    assert not snapped.any() and np.array_equal(out, z)
    lab = np.zeros((H, W), np.int64)
    lab[:, :41] = 5
    lab[:, 40] = -1                                      # two masks overlap on the ramp
    out, _, snapped = snap_mixed(z, valid, lab, TAU)
    assert not snapped.any()


# ── the edge band, measured ─────────────────────────────────────────────────

def test_ring_distance_counts_3x3_dilations():
    seed = np.zeros((H, W), bool)
    seed[:, 40] = True
    r = ring_distance(seed)
    assert r[5, 40] == 0 and r[5, 43] == 3 and r[5, 37] == 3
    assert np.isinf(ring_distance(np.zeros((H, W), bool))).all()


def test_edge_band_is_where_the_floor_stops_eroding():
    seed = np.zeros((H, W), bool)
    seed[:, 40] = True
    rings = ring_distance(seed)
    valid = np.ones((H, W), bool)
    passed = rings >= 3                                  # the floor removes rings 0, 1, 2
    nv, npass = ring_histogram(rings, valid, passed)
    band, rate = edge_band(nv, npass)
    assert band == 3 and rate[0] == 0.0 and rate[3] == 1.0
    nv, npass = ring_histogram(rings, valid, np.ones((H, W), bool))
    assert edge_band(nv, npass)[0] == 0                  # nothing eroded at the step: no edge band


# ── consecutive agreement (bend window) ─────────────────────────────────────

def test_consecutive_ratio_reads_a_scale_jump_between_two_keyframes():
    plane = np.full((H, W), 3.0, np.float32)
    ok = np.ones((H, W), bool)
    a, b = _cam(-0.05), _cam(0.05)
    wb = np.linalg.inv(b)
    assert abs(consecutive_ratio(plane, ok, plane, ok, K, a, wb, 2) - 1.0) < 1e-6
    r = consecutive_ratio(plane, ok, plane * 1.1, ok, K, a, wb, 2)
    assert abs(r - 1 / 1.1) < 1e-3
    assert np.isnan(consecutive_ratio(plane, ok, plane, np.zeros((H, W), bool), K, a, wb, 2))


# ── the two-sided vote, the repair, the decision ────────────────────────────

def _three_views():
    order = [0, 1, 2]
    c2w = {0: _cam(-0.2), 1: _cam(0.0), 2: _cam(0.2)}
    w2c = {f: np.linalg.inv(c2w[f]) for f in order}
    depth = {f: np.full((H, W), 3.0, np.float32) for f in order}
    judge = {f: np.ones((H, W), bool) for f in order}
    return order, c2w, w2c, depth, judge


def test_a_pixel_behind_the_neighbours_surface_is_contradicted_only_by_the_second_side():
    order, c2w, w2c, depth, judge = _three_views()
    depth[1][20:30, 30:40] = 3.5                         # skirt: the pixel claims free space through the plane
    depth[1][40:45, 30:40] = 2.5                         # flyer in front of the plane
    v = two_sided_vote(1, order, depth, judge, np.ones((H, W), bool), K, c2w, w2c, (-1, 1), TAU)
    at = {(r, c): k for k, (r, c) in enumerate(zip(v["rr"], v["cc"]))}
    behind, flyer, good = at[(25, 35)], at[(42, 35)], at[(10, 35)]
    assert v["contra_fwd"][behind] == 0 and v["contra"][behind] == 2 and v["agree"][behind] == 0
    assert v["contra_fwd"][flyer] == 2 and v["contra"][flyer] == 2
    assert v["agree"][good] == 2 and v["contra"][good] == 0 and abs(v["zmed"][good] - 3.0) < 1e-3
    ok, z, n = agreeing_median(v["splats"][:, [behind]], TAU, 2)
    assert ok[0] and abs(z[0] - 3.0) < 1e-3 and n[0] == 2   # repaired onto the plane


def test_only_floor_passing_neighbours_judge():
    order, c2w, w2c, depth, judge = _three_views()
    judge[0][:] = False
    judge[2][:] = False                                  # the neighbours are all below the floor
    v = two_sided_vote(1, order, depth, judge, np.ones((H, W), bool), K, c2w, w2c, (-1, 1), TAU)
    assert not v["agree"].any() and not v["contra"].any()
    assert np.isnan(v["splats"]).all()


def test_agreeing_median_needs_min_views_that_agree():
    C = np.array([[3.0, 3.0, 3.0], [3.01, 4.5, np.nan], [2.99, 6.0, np.nan], [9.0, np.nan, np.nan]])
    ok, z, n = agreeing_median(C, TAU, 2)
    assert ok[0] and abs(z[0] - 3.0) < 0.011 and n[0] == 3
    assert not ok[1] and not ok[2]                       # disagreeing views / a single view
    ok, z, n = agreeing_median(np.zeros((0, 4)), TAU, 2)
    assert not ok.any() and len(z) == 4


def test_the_floor_keeps_acting_on_interior_pixels_and_lets_confirmed_edges_in():
    passed = np.array([True, True, True, True, False, False, False, False])
    edge = np.array([False, False, True, False, True, True, False, True])
    agree = np.array([0, 2, 1, 0, 1, 0, 3, 1])
    contra = np.array([0, 1, 3, 3, 1, 0, 0, 2])
    repairable = np.array([False, False, True, False, True, True, True, True])
    keep, repair, admit = edge_vote_decision(passed, edge, agree, contra, repairable)
    assert keep.tolist() == [True, True, False, False, False, False, False, False]   # 0/0 stays: it passed
    assert repair.tolist() == [False, False, True, False, False, False, False, False]
    assert admit.tolist() == [False, False, False, False, True, False, False, False]
    # below the floor: 0/0 never enters, an interior pixel never enters, a contested edge never enters
