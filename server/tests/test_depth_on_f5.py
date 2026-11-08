"""The depth on F5 (USER 2026-10-01, pccr epoch 7): Omega's depth bent to F5's landmarks, then a vote.
Synthetic, no GPU: a plane seen by three cameras."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.depth_on_f5 import (bend_coefficients, camera_travels, design, edge_keeping_vote,  # noqa: E402
                                   heldout_metres, rel_errors)

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


def test_bend_recovers_a_scale_error_and_keeps_omega_without_landmarks():
    """Epoch 7's fit: a keyframe with too few rows keeps Omega's depth (k = 1)."""
    rng = np.random.default_rng(0)
    rows = {}
    for i, true in enumerate((0.9, 0.9, 1.1)):
        u = rng.uniform(0, W, 200); v = rng.uniform(0, H, 200)
        rows[i] = (design(u, v, W, H), np.full(200, true) + rng.normal(0, 1e-3, 200))
    rows[3] = (design(np.zeros(0), np.zeros(0), W, H), np.zeros(0))      # no landmark at all
    c, info = bend_coefficients(rows, 4, 0, 20, 1.345, 10, FAC)
    assert abs(c[0][0] - 0.9) < 1e-2 and abs(c[2][0] - 1.1) < 1e-2
    assert np.allclose(c[3], [1.0, 0.0, 0.0]) and info[3]["fallback"] == "identity"
    # a flat ratio field: the tilt terms c1, c2 are not significant and are dropped exactly (point 48)
    assert c[0][1] == 0.0 and c[0][2] == 0.0 and info[0]["kept"] == {"c1": False, "c2": False}


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
    near = dict(fusion, witness_max_m=2.0)               # the wall is ~3 m away: nobody may vote
    final2, _, tot2 = edge_keeping_vote(*args(near), False, log=lambda *a: None,
                                        min_parallax_deg=3.0, min_support=3, fusion=near)
    assert (final2[4][0] == 0).all(), "witnesses beyond witness_max_m have no vote (USER: 10 m, no más)"


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
    assert np.allclose(k[v, u], design(u.astype(float), v.astype(float), W, H) @ c)
    assert scale_map(1.03, (0.0, 0.0), H, W) == 1.03


# ── the mono-detail hook (claude_stac.txt 2026-10-04) ────────────────────────

def test_mono_detail_off_leaves_the_bend_untouched_and_the_source_is_omega():
    from dataclasses import replace
    from precision.config import load_precision_config
    from precision.depth_on_f5 import apply_mono_detail, confidence_weight, source_column
    pcfg = load_precision_config()
    pcfg = replace(pcfg, mono_detail=replace(pcfg.mono_detail, enabled=False))   # the flag's value is the user's
    dep = {0: np.ones((4, 4), np.float32)}; valid = {0: np.ones((4, 4), bool)}; passed = {0: np.ones((4, 4), bool)}
    d2, v2, p2, src, rep = apply_mono_detail(pcfg, [0], dep, valid, passed, {}, None, K, {}, Path("."),
                                             lambda m: None, lambda a, b: None)
    assert d2 is dep and v2 is valid and p2 is passed and src is None and rep is None
    data = {"frame_global": np.zeros(3, np.int64), "pixel_row": np.zeros(3, np.int64), "pixel_col": np.arange(3)}
    assert source_column(data, None).tolist() == [1, 1, 1]
    maps = {0: np.array([[0, 1, 3]], np.uint8)}
    assert source_column(data, maps).tolist() == [1, 2, 4]
    w = confidence_weight(np.array([0.0, 0.5, 1.0, 2.0]), 0.5, 1.5, np.array([False, True, True, True]))
    assert np.allclose(w, [0.0, 0.0, 0.5, 1.0])
    # the chain gives f6_bend the card only when the stage is on
    from precision.runner import chain_steps
    off = {s.key: s.gpu for s in chain_steps(pcfg)}
    on = {s.key: s.gpu for s in chain_steps(replace(pcfg, mono_detail=replace(pcfg.mono_detail, enabled=True)))}
    assert off["f6_bend"] is False and on["f6_bend"] is True and "f6_sweep" not in on


def test_the_half_b_held_out_is_also_measured_in_metres():
    """USER 2026-10-16: the certification's σ floor reads F6's held-out as a DISTANCE along the
    ray (the relative score is the bend's judge, not a repeatability). Same rows as the relative
    score, pooled over the keyframes, with their landmark range and count; a keyframe without a
    half-B row contributes nothing, a session without any reports None."""
    rng = np.random.default_rng(1)
    rows_B, coefs = {}, {}
    for i, (k, z0) in enumerate(((0.9, 2.0), (1.1, 4.0))):
        n = 300
        u, v = rng.uniform(0, W, n), rng.uniform(0, H, n)
        z_lm = np.full(n, z0) + rng.normal(0, 0.02, n)      # the landmarks, 2 cm of scatter
        z_om = z_lm / k                                       # Omega's depth, off by the scale k
        rows_B[i] = (u, v, z_lm, z_om)
        coefs[i] = np.array([k, 0.0, 0.0])                    # the bend recovers exactly k
    rows_B[2] = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    coefs[2] = np.array([1.0, 0.0, 0.0])
    got = heldout_metres(rows_B, coefs, 3, W, H)
    assert got["rows"] == 600 and got["median_m"] == pytest.approx(0.0, abs=1e-9)
    assert got["range_m"] == pytest.approx(3.0, abs=0.1)
    # an unbent keyframe: its error in metres is the relative error times its own depth
    coefs[0] = np.array([1.0, 0.0, 0.0])
    got = heldout_metres({0: rows_B[0]}, {0: coefs[0]}, 1, W, H)
    rel = rel_errors(*rows_B[0], coefs[0], W, H)
    assert got["median_m"] == pytest.approx(float(np.median(rel * rows_B[0][2])), rel=1e-9)
    assert got["median_m"] == pytest.approx(2.0 / 0.9 - 2.0, abs=0.01)   # |z/0.9 - z| at z = 2 m
    assert heldout_metres({0: rows_B[2]}, {0: coefs[2]}, 1, W, H) == {"median_m": None, "range_m": None, "rows": 0}
