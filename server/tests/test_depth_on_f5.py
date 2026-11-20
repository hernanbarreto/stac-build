"""The depth on F5 (USER 2026-10-01, pccr epoch 7): Omega's depth at the session's one factor, then a vote.
Synthetic, no GPU: a plane seen by three cameras."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.depth_on_f5 import (DepthOnF5Error, camera_travels, edge_keeping_vote,  # noqa: E402
                                   heldout_error, heldout_rows, session_factor)

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


def _fac() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


FAC = _fac()


def _cams():
    out = {}
    for f, x in zip((0, 1, 2), (-0.2, 0.0, 0.2)):
        c = np.eye(4); c[0, 3] = x; out[f] = c
    return out


def test_the_edge_keeping_vote_removes_a_flyer_and_keeps_the_surface():
    """pccr epoch 8's vote (the pipeline's since 2026-10-01): a flyer the neighbours see through leaves,
    the wall stays at its own depth."""
    cams = _cams(); order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    dep[1][10:20, 30:40] = 1.5                       # a flyer in front of the wall the others see
    ok = {f: np.ones((H, W), bool) for f in order}
    final, tau, tot = edge_keeping_vote(order, dep, ok, {f: ok[f].copy() for f in order}, K, cams, None,
                                        (-1, 1), 75.0, 2, False, log=lambda *a: None)
    z1, a1 = final[1]
    # the flyer is gone: its pixels see the wall behind it, so they are REPAIRED to the depth the two
    # neighbours agree on (epoch 8: a contradicted pixel still sees a surface) — never left at 1.5 m
    assert np.allclose(z1[12:18, 32:38], 3.0, atol=1e-3), "the flyer's pixels take the wall the neighbours see"
    assert np.allclose(z1[40:50, 30:40], 3.0, atol=1e-3) and (a1[40:50, 30:40] >= 1).all()
    assert tot["contradicted"] > 0 and tot["repaired"] > 0 and 0 <= tau


def test_a_pixel_under_the_floor_is_rescued_with_the_surface_two_views_agree_on():
    """USER 2026-11-04 (night): a below-floor pixel takes the depth its floor-passing neighbours agree
    on along its ray, never its own; one whose own depth lies in front of that surface (a thin
    occluder no view measured) stays out; off = the floor alone decides, as before."""
    cams = _cams(); order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    dep[1][10:20, 30:40] = 3.3                       # under the floor, its own depth noisy behind the wall
    dep[1][40:50, 30:40] = 1.5                       # under the floor, its own depth IN FRONT of the wall
    ok = {f: np.ones((H, W), bool) for f in order}
    passed = {f: ok[f].copy() for f in order}
    passed[1][10:20, 30:40] = False
    passed[1][40:50, 30:40] = False
    args = (order, {f: d.copy() for f, d in dep.items()}, ok, passed, K, cams, None, (-1, 1), 75.0, 2)
    off, _, tot_off = edge_keeping_vote(*args, False, log=lambda *a: None)
    assert (off[1][0][10:20, 30:40] == 0).all() and (off[1][0][40:50, 30:40] == 0).all()
    assert tot_off["rescued"] == 0 and tot_off["below_floor"] == 200
    args = (order, {f: d.copy() for f, d in dep.items()}, ok, passed, K, cams, None, (-1, 1), 75.0, 2)
    on, _, tot = edge_keeping_vote(*args, True, log=lambda *a: None)
    z1, a1 = on[1]
    assert np.allclose(z1[10:20, 30:40], 3.0, atol=1e-3), "the wall the two neighbours agree on, never 3.3"
    assert (a1[10:20, 30:40] == 2).all()
    assert (z1[40:50, 30:40] == 0).all(), "its own depth in front: it may see something no view measured"
    assert tot["rescued"] == 100 and tot["rescue_refused_nearer"] == 100
    # every pixel the floor passed is decided exactly as without the rescue
    keep = passed[1]
    assert np.array_equal(z1[keep], off[1][0][keep]) and np.array_equal(a1[keep], off[1][1][keep])


def test_strongest_consensus_moves_the_loser_and_drops_the_orphan():
    """USER 2026-11-08 ("no es un filtro, es un ajuste … elegir la posición más fuerte"): the strongest
    cluster along the ray decides the position; the own camp losing is MOVED (own_in False); a ray with
    no cluster of min_views dies; ties take the nearest depth (deterministic)."""
    import torch
    from precision.f6_torch import strongest_consensus
    nan = float("nan")
    # columns: [own, w1, w2, w3, w4] on a 1x5 image
    C = torch.tensor([
        [3.40, 3.00, 1.50, 3.00, nan],      # own
        [3.00, 3.01, 3.00, 3.40, nan],      # w1
        [3.01, 2.99, 3.01, 3.40, nan],      # w2
        [2.99, 3.00, 2.99, nan,  nan],      # w3
        [3.41, nan,  nan,  nan,  nan],      # w4
    ], dtype=torch.float64).reshape(5, 1, 5)
    z, n, own_in = strongest_consensus(C, tau=0.02, min_views=3)
    z, n, own_in = z[0], n[0], own_in.reshape(-1)
    assert abs(z[0] - 3.00) < 0.02 and n[0] == 3 and not own_in[0]      # own camp (3.4 x2) loses, MOVED
    assert abs(z[1] - 3.00) < 0.02 and n[1] == 4 and own_in[1]          # own inside the winner
    assert abs(z[2] - 3.00) < 0.02 and not own_in[2]                    # the flyer at 1.5 moved onto the wall
    assert n[3] == 0 and torch.isnan(z[3])                              # 3.0 x1 + 3.4 x2: no cluster of 3
    assert n[4] == 0 and torch.isnan(z[4])                              # nothing on the ray


def test_the_adjust_vote_collapses_a_displaced_camp_and_the_distance_cap_silences_far_witnesses():
    """USER 2026-11-08: two camps of keyframes put the same wall at 3.0 m (4 frames) and 3.4 m (2 frames) —
    the displaced camp's pixels are MOVED onto the strong wall (one surface, no duplicate); with
    witness_max_m below the wall's range no witness votes and the pixels die instead."""
    rng = np.random.default_rng(3)
    order = [0, 1, 2, 3, 4, 5]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.3 * f                 # 0.3 m baseline: ~5.7° at 3 m, inside [3°, 30°]
        cams[f] = c
    jit = {f: (1.0 + 0.004 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: (3.0 if f < 4 else 3.4) * jit[f] for f in order}
    ok = {f: np.ones((H, W), bool) for f in order}
    fusion = {"max_reproj_px": 6.0, "max_depth_rel": 0.0, "tri_min_deg": 3.0, "tri_max_deg": 30.0,
              "mode": "adjust", "witness_max_m": 10.0}
    args = lambda fu: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                       K, cams, None, (-3, -2, -1, 1, 2, 3), 75.0, 2)
    final, tau, tot = edge_keeping_vote(*args(fusion), False, log=lambda *a: None,
                                        min_parallax_deg=3.0, min_support=3, fusion=dict(fusion))
    z4 = final[4][0]
    mid = z4[H // 3: 2 * H // 3, W // 3: 2 * W // 3]
    # with only 3 same-camp witnesses and synthetic jitter ~τ, some rays miss a 3-cluster — the real maps
    # carry dozens of witnesses; the semantics under test is the MOVE, not the synthetic survival rate
    assert (mid > 0).mean() > 0.6, "the displaced keyframe keeps most pixels (adjust, not filter)"
    assert abs(np.median(mid[mid > 0]) - 3.0) < 0.1, "…MOVED onto the strong wall at 3.0, not 3.4"
    assert tot["repaired"] > 0, "moved pixels are counted"
    z1 = final[1][0]
    assert abs(np.median(z1[z1 > 0]) - 3.0) < 0.05, "the strong camp stays on its wall"
    near = dict(fusion, witness_max_m=2.0)               # the wall is ~3 m away: nothing lies within the range
    from precision.corrected_cloud import CorrectedCloudError
    with pytest.raises(CorrectedCloudError, match="share a surface"):
        # USER 2026-11-09: nothing beyond the range is considered anywhere — not even to measure τ
        edge_keeping_vote(*args(near), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=3, fusion=near)


def test_fusion_repair_publishes_the_short_pixel_at_its_witnesses_median():
    """USER 2026-11-08 evening: a COLMAP-mode pixel short of min_support whose witnesses' surfaces agree
    among themselves on its ray is published at THEIR median; with the flag off it is dropped (the
    validated recipe bit for bit)."""
    rng = np.random.default_rng(5)
    order = [0, 1, 2, 3]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.3 * f
        cams[f] = c
    jit = {f: (1.0 + 0.003 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: 3.0 * jit[f] for f in order}
    dep[1][10:20, 30:40] = 1.2            # a patch whose own depth agrees with nobody (short of support)
    ok = {f: np.ones((H, W), bool) for f in order}
    base = {"max_reproj_px": 6.0, "max_depth_rel": 0.0, "tri_min_deg": 3.0, "tri_max_deg": 30.0, "mode": "keep"}
    args = lambda: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                    K, cams, None, (-2, -1, 1, 2), 75.0, 2)
    off, _, t_off = edge_keeping_vote(*args(), False, log=lambda *a: None,
                                      min_parallax_deg=3.0, min_support=3, fusion=dict(base, repair=False))
    assert (off[1][0][12:18, 32:38] == 0).all(), "short of support and no repair: dropped"
    on, _, t_on = edge_keeping_vote(*args(), False, log=lambda *a: None,
                                    min_parallax_deg=3.0, min_support=3, fusion=dict(base, repair=True))
    z = on[1][0][12:18, 32:38]
    # with only 3 witnesses and synthetic jitter ~τ some rays miss the agreeing set — the semantics under
    # test is WHERE the repaired pixel lands (the witnesses' wall), not the synthetic survival rate
    assert (z > 0).mean() > 0.5 and abs(np.median(z[z > 0]) - 3.0) < 0.1, \
        "repair ON: published at the witnesses' agreed median (the wall at 3.0), never its own 1.2"
    assert t_on["repaired"] > 0 and t_off["repaired"] == 0


def test_fusion_occlusion_veto_removes_the_copy_behind_a_stronger_surface():
    """The Merrell duel (ICCV07, researched 2026-11-08): a pixel confirmed only by its own camp but
    OCCLUDED by more witnesses than confirm it (their surfaces in front on its ray) leaves when the
    veto is on; with it off (the validated recipe) it stays."""
    rng = np.random.default_rng(9)
    order = [0, 1, 2, 3, 4, 5, 6]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.3 * f
        cams[f] = c
    jit = {f: (1.0 + 0.003 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: (3.0 if f < 4 else 3.5) * jit[f] for f in order}   # camp A (0-3) in front, camp B (4-6) behind
    ok = {f: np.ones((H, W), bool) for f in order}
    base = {"max_reproj_px": 6.0, "max_depth_rel": 0.0, "tri_min_deg": 3.0, "tri_max_deg": 30.0, "mode": "keep"}
    args = lambda: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                    K, cams, None, (-4, -3, -2, -1, 1, 2, 3, 4), 75.0, 2)
    off, _, t_off = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0,
                                      min_support=2, fusion=dict(base, occlusion_veto=False))
    z5_off = off[5][0]
    mid = (slice(H // 3, 2 * H // 3), slice(W // 3, 2 * W // 3))
    assert (z5_off[mid] > 0).mean() > 0.5, "veto off: the behind camp self-confirms and stays"
    on, _, t_on = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0,
                                    min_support=2, fusion=dict(base, occlusion_veto=True))
    z5_on, z1_on = on[5][0], on[1][0]
    assert (z5_on[mid] > 0).mean() < 0.3, "veto on: the copy behind the stronger surface leaves"
    assert (z1_on[mid] > 0).mean() > 0.5, "the FRONT surface is never occluded and stays"
    assert t_on["occlusion_veto"] > 0 and t_off.get("occlusion_veto", 0) == 0


def test_the_range_births_nothing_beyond_it_and_silences_far_witnesses():
    """USER 2026-11-09 ("nadie debe aportar puntos a más de 10m de distancia" / "si están a más de 10m ya no votan
    aunque lo vean"): a wall 3 m away is kept with a range of 10 m; with a range of 2 m no keyframe gives birth
    to it (counted as beyond_range) and no witness confirms it."""
    rng = np.random.default_rng(4)
    order = [0, 1, 2, 3, 4, 5]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.3 * f
        cams[f] = c
    jit = {f: (1.0 + 0.003 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: 3.0 * jit[f] for f in order}
    ok = {f: np.ones((H, W), bool) for f in order}
    base = {"max_reproj_px": 6.0, "max_depth_rel": 0.0, "tri_min_deg": 3.0, "tri_max_deg": 30.0, "mode": "keep"}
    args = lambda: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                    K, cams, None, (-3, -2, -1, 1, 2, 3), 75.0, 2)
    mid = (slice(H // 3, 2 * H // 3), slice(W // 3, 2 * W // 3))
    far, _, t_far = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                                      fusion=dict(base, witness_max_m=10.0))
    assert (far[2][0][mid] > 0).mean() > 0.5 and t_far["beyond_range"] == 0, "3 m inside a 10 m range: kept"
    from precision.corrected_cloud import CorrectedCloudError
    with pytest.raises(CorrectedCloudError, match="share a surface"):
        # nothing within the range: not even τ can be measured (USER 2026-11-09: "ni de nada")
        edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                          fusion=dict(base, witness_max_m=2.0))
    import torch
    from precision import f6_torch as FT
    d = {f: dd.copy() for f, dd in dep.items()}
    w2c = {f: np.linalg.inv(cams[f]) for f in order}
    v = FT.two_sided_vote(2, order, d, ok, np.ones((H, W), bool), K, cams, w2c, (-2, -1, 1, 2), 0.01,
                          fusion=dict(base, max_depth_rel=0.01, witness_max_m=2.0))
    assert int(v["agree_q"].sum()) == 0 and int(v["thru_q"].sum()) == 0, "no witness beyond the range votes"
    assert not bool(v["in_range"].any())


def test_the_majority_drops_the_point_more_witnesses_see_through():
    """USER 2026-11-09 ("si tenemos muchas vistas del mismo punto 5 puede significar poco, pero si tiene pocas vistas
    puede significar mucho"): a floater 2 m away confirmed by its own two neighbours but seen THROUGH by the four
    witnesses that measure the wall behind it at 3 m — with the majority ON it leaves; OFF (the recipe before) it
    stays on its two confirmations; the wall itself is never touched."""
    rng = np.random.default_rng(6)
    order = [0, 1, 2, 3, 4, 5, 6]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.3 * f
        cams[f] = c
    jit = {f: (1.0 + 0.003 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: 3.0 * jit[f] for f in order}
    for f in (2, 3, 4):                                   # the floater's camp: three keyframes put a patch at 2 m
        dep[f][20:40, 20:60] = 2.0 * jit[f][20:40, 20:60]
    ok = {f: np.ones((H, W), bool) for f in order}
    base = {"max_reproj_px": 6.0, "max_depth_rel": 0.01, "tri_min_deg": 3.0, "tri_max_deg": 30.0, "mode": "keep",
            "witness_max_m": 10.0}
    args = lambda: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                    K, cams, None, (-3, -2, -1, 1, 2, 3), 75.0, 2)
    patch = (slice(25, 35), slice(35, 45))                # ±1 land inside their own patch, ±2/±3 on the wall
    wall = (slice(5, 12), slice(10, 70))
    off, _, t_off = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                                      fusion=dict(base, majority=False))
    assert (off[3][0][patch] > 0).mean() > 0.5, "majority OFF: the floater self-confirms with its two neighbours"
    on, _, t_on = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                                    fusion=dict(base, majority=True))
    assert (on[3][0][patch] > 0).mean() < 0.2, "majority ON: seen through by more than confirm it — it leaves"
    assert t_on["minority"] > 0 and t_off["minority"] == 0
    assert (on[3][0][wall] > 0).mean() > 0.5, "the wall around it stays"


def test_in_place_publishes_a_surface_its_witnesses_agree_on_without_moving_it():
    """USER 2026-11-09 ("los recuperados que están cerca de la cámara, ése es el problema" / "aplicá esta nueva
    regla"): a wall every keyframe sees from almost the same place (1 cm apart — no witness in the triangulation
    band) gets no COLMAP confirmation; IN PLACE publishes it AT ITS OWN DEPTH because its witnesses' surfaces agree
    with it. A patch one keyframe puts 10 % nearer is NOT moved onto the wall: it stays out."""
    rng = np.random.default_rng(11)
    order = [0, 1, 2, 3, 4, 5, 6]
    cams = {}
    for f in order:
        c = np.eye(4); c[0, 3] = 0.01 * f
        cams[f] = c
    jit = {f: (1.0 + 0.002 * rng.standard_normal((H, W))).astype(np.float32) for f in order}
    dep = {f: 3.0 * jit[f] for f in order}
    dep[3][20:40, 20:60] = 2.7 * jit[3][20:40, 20:60]          # one keyframe's patch 10 % nearer
    ok = {f: np.ones((H, W), bool) for f in order}
    base = {"max_reproj_px": 6.0, "max_depth_rel": 0.01, "tri_min_deg": 3.0, "tri_max_deg": 30.0, "mode": "keep",
            "witness_max_m": 10.0, "majority": True, "occlusion_veto": True}
    args = lambda: (order, {f: d.copy() for f, d in dep.items()}, ok, {f: ok[f].copy() for f in order},
                    K, cams, None, (-3, -2, -1, 1, 2, 3), 75.0, 2)
    wall = (slice(5, 15), slice(10, 70))
    patch = (slice(25, 35), slice(30, 50))
    off, _, t_off = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                                      fusion=dict(base, in_place=False))
    assert (off[3][0][wall] > 0).mean() < 0.05, "no witness in the band: COLMAP confirms nothing"
    on, _, t_on = edge_keeping_vote(*args(), False, log=lambda *a: None, min_parallax_deg=3.0, min_support=2,
                                    fusion=dict(base, in_place=True))
    z3 = on[3][0]
    pub = z3[wall] > 0
    assert pub.mean() > 0.5, "in place: the wall its witnesses agree on is published"
    assert np.allclose(z3[wall][pub], dep[3][wall][pub]), "published AT ITS OWN DEPTH, never moved"
    assert (z3[patch] > 0).mean() < 0.05, "the 10 % nearer patch is not moved onto the wall: it stays out"
    assert t_on["in_place"] > 0 and t_off["in_place"] == 0


def test_the_camera_travels_with_the_cloud(tmp_path):
    """Written into the transaction (one row per keyframe); publish registers it as an epoch artifact."""
    import inspect
    from precision import corrected_cloud as CC
    p = camera_travels(tmp_path, [391.9, 388.7, 234.8, 414.1, 0, 0, 0, 0], 3, log=lambda *a: None)
    rows = p.read_text().splitlines()
    assert p == tmp_path / "intrinsic.txt" and len(rows) == 3 and rows[0].startswith("391.9 388.7")
    src = inspect.getsource(CC.publish)
    assert 'art("intrinsic.txt")' in src, "publish must file the camera with the epoch (pccr 2026-10-01)"


def test_chunk_check_applies_the_published_bend():
    """f6_check measures the cloud that was PUBLISHED: Omega's depth × k(u, v) of the bend report."""
    from precision.chunk_check import scale_map
    c = np.array([0.95, 0.04, -0.03])
    k = scale_map(c[0], (c[1], c[2]), H, W)
    rng = np.random.default_rng(1)
    u = rng.integers(0, W, 50); v = rng.integers(0, H, 50)
    assert np.allclose(k[v, u], c[0] + c[1] * (u - W / 2.0) / W + c[2] * (v - H / 2.0) / H)
    assert scale_map(1.03, (0.0, 0.0), H, W) == 1.03


# ── the depth's factor: ONE for the session (USER 2026-11-20) ────────────────

def test_the_depth_factor_is_one_for_the_session_and_the_odd_keyframes_do_not_move_it():
    rng = np.random.default_rng(0)
    ratios = [0.95 + rng.normal(0, 0.004, 300) for _ in range(40)]      # z_landmark / z_Omega, keyframe by keyframe
    ratios[10] = 0.86 + rng.normal(0, 0.004, 300)       # a stretch whose own landmarks ask for another factor
    ratios[0] = np.full(300, 0.10)                      # a keyframe whose landmarks are nonsense
    ratios.append(np.zeros(0))                          # a keyframe without landmarks takes the same factor
    f, info = session_factor(ratios, 1.345, 10)
    assert abs(f - 0.95) < 0.003, f
    assert info["factor"] == f and info["rows"] == 40 * 300 and info["keyframes_with_rows"] == 40
    assert info["sigma"] > 0 and abs(info["median_of_the_rows"] - 0.95) < 0.003
    with pytest.raises(DepthOnF5Error, match="no landmark row"):
        session_factor([np.zeros(0), np.zeros(0)], 1.345, 10)


def test_the_held_out_landmarks_report_what_the_factor_leaves_in_per_cent_and_in_metres():
    zmap = np.full((H, W), 2.0)
    obs = np.array([[10.0, 10.0, 1.9], [20.0, 20.0, 1.9], [30.5, 12.25, 2.1]])       # u, v, z of the landmark
    z_lm, z_om = heldout_rows(obs, zmap, 0.05)
    assert z_lm.tolist() == [1.9, 1.9, 2.1] and np.allclose(z_om, 2.0)
    rep = heldout_error([(z_lm, z_om), (np.zeros(0), np.zeros(0))], 1.0)             # |2.0 - z| = 0.1 on the three
    assert rep["rows"] == 3 and abs(rep["median_m"] - 0.1) < 1e-12 and rep["range_m"] == 1.9
    assert abs(rep["median_rel"] - 0.1 / 1.9) < 1e-12
    assert abs(heldout_error([(z_lm, z_om)], 0.95)["median_m"]) < 1e-12              # 2.0 x 0.95 = 1.9: two of three
    assert heldout_error([], 1.0) == {"median_rel": None, "median_m": None, "range_m": None, "rows": 0}
    assert heldout_rows(np.zeros((0, 3)), zmap, 0.05)[0].size == 0
    assert heldout_rows(obs, np.full((H, W), 0.01), 0.05)[0].size == 0               # Omega's depth under the minimum


# ── the cloud's provenance column and the card ───────────────────────────────

def test_every_published_point_is_omegas_own_pixel_and_f6_takes_the_card():
    from precision.config import load_precision_config
    from precision.depth_on_f5 import SRC_OMEGA_BENT, source_column
    data = np.zeros(5, dtype=[("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])   # the cleaned
    col = source_column(data)                                                                    # cloud's rows
    assert col.dtype == np.uint8 and col.tolist() == [SRC_OMEGA_BENT] * 5 == [1, 1, 1, 1, 1]
    assert source_column(data[:0]).tobytes() == b""
    # F6 runs its maps and its vote on the card: the chain hands it the card whatever the configuration
    from precision.runner import chain_steps
    steps = {s.key: s.gpu for s in chain_steps(load_precision_config())}
    assert steps["f6_bend"] is True and "f6_sweep" not in steps


