"""The coherence measure on a scene whose answer is known: a textured wall 3 m away seen by two passes of six
keyframes each."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import coherence as CO                        # noqa: E402

H, W, N = 120, 160, 12
K = (140.0, 140.0, 79.5, 59.5)
LENS = (0.03, -0.01, 0.001, -0.0005)
WALL_Z = 3.0
CFG = CO.Params(device="cpu", pixel_stride=2, search_px=6, blur_sigma_px=1.0, min_pixels=500, same_surface_sigmas=3.0,
                neighbour_offsets=(1, 2), pass_min_gap=3, range_m=15.0, conf_floor_norm=0.1, confidence=0.95,
                error_factor=1.1, min_judges=5, bootstrap=400, seed=0)


def _texture(x, y):
    rng = np.random.default_rng(5)
    g = np.zeros_like(x)
    for _ in range(60):
        lam = rng.uniform(0.08, 0.6)                          # wavelengths of 8 – 60 cm on the wall (>= 4 px)
        th = rng.uniform(0, math.pi)
        g += rng.uniform(0.3, 1.0) * np.sin(2 * math.pi * (x * math.cos(th) + y * math.sin(th)) / lam + rng.uniform(0, 6.28))
    return 128 + 14 * g


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _arrays():
    """12 cameras: keyframes 0–5 step 5 cm along x with a small yaw, keyframes 6–11 the same places 4 cm higher (a
    second pass). Each one's image and depth map of the wall z = 3, through the lens. -> (depth, rgb, c2w)."""
    c2w = np.tile(np.eye(4), (N, 1, 1))
    for k in range(N):
        c2w[k, :3, :3] = _rot_y(0.8 * math.sin(1.3 * k))
        c2w[k, :3, 3] = (0.05 * (k % 6), 0.04 * (k // 6), 0.0)
    vv, uu = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    xn, yn = CO.unproject(torch.tensor(uu.ravel()), torch.tensor(vv.ravel()), K, LENS)
    rays = np.stack([xn.numpy(), yn.numpy(), np.ones(H * W)], 1)
    depth, rgb = np.zeros((N, H, W), np.float32), np.zeros((N, H, W, 3), np.uint8)
    for k in range(N):
        d = rays @ c2w[k, :3, :3].T
        t = (WALL_Z - c2w[k, 2, 3]) / d[:, 2]                 # the depth along the optical axis (the ray's z is 1)
        P = c2w[k, :3, 3] + t[:, None] * d
        depth[k] = t.reshape(H, W)
        rgb[k] = np.clip(_texture(P[:, 0], P[:, 1]), 0, 255).reshape(H, W, 1).round().astype(np.uint8)
    return depth, rgb, c2w


def _scene():
    depth, rgb, c2w = _arrays()
    conf = np.full((N, H, W), 5.0, np.float32)
    maps = CO.make_maps(depth, conf, [0] * 6 + [1] * 6, CO.grey_of(torch.tensor(rgb), CFG), list(range(100, 100 + N)), CFG)
    return maps, c2w


def _state(label, c2w, factor=None):
    f = np.ones(N) if factor is None else np.asarray(factor, float)
    return CO.State(label, torch.tensor(c2w), K, LENS, torch.tensor(f))


@pytest.fixture(scope="module")
def world():
    maps, c2w = _scene()
    nb = CO.neighbour_pairs(N, CFG.neighbour_offsets)
    covis = [[100 + k, 100 + k + 6, 0.8, 0.7] for k in range(6)] + [[100, 109, 0.05, 0.05], [100, 101, 0.9, 0.9]]
    ps = CO.pass_pairs(covis, maps.frames, CFG.pass_min_gap, CFG.min_judges)
    pairs, kinds = nb + ps, [CO.KIND_NEIGHBOUR] * len(nb) + [CO.KIND_PASS] * len(ps)
    return maps, c2w, pairs, kinds, [0] * 6 + [1] * 6


def test_the_rule_gives_five_judges_at_95_and_the_partners_come_from_the_map(world):
    maps, _c2w, pairs, kinds, _z = world
    assert CFG.min_judges == 5
    ps = [p for p, k in zip(pairs, kinds) if k == CO.KIND_PASS]
    assert (0, 6) in ps and (5, 11) in ps and (0, 9) in ps, "the most co-visible partners of other passes"
    assert (0, 1) not in ps, "a neighbour in the walk is no partner of another pass, whatever the map says"


def test_a_coherent_state_reads_zero(world):
    maps, c2w, pairs, kinds, _z = world
    m = CO.measure(maps, _state("truth", c2w), pairs, kinds, CFG)
    assert m["measured"].all() and len(m["i"]) == 2 * len(pairs)
    s = CO.summary(m)
    for kind in ("neighbours", "different passes"):
        assert s[kind]["along_cm"]["p90"] < 0.05, s
        assert s[kind]["across_cm"]["p90"] < 0.35, s            # a sixth of a pixel at 3 m (1 px = 2.1 cm)


def test_a_pass_displaced_sideways_is_read_across_the_ray_and_only_between_passes(world):
    maps, c2w, pairs, kinds, zones = world
    moved = c2w.copy(); moved[6:, 0, 3] += 0.03                 # the second pass 3 cm to the side, as one piece
    a = CO.measure(maps, _state("truth", c2w), pairs, kinds, CFG)
    b = CO.measure(maps, _state("moved", moved), pairs, kinds, CFG)
    s = CO.summary(b)
    assert abs(s["different passes"]["across_cm"]["median"] - 3.0) < 0.4, s
    assert s["different passes"]["along_cm"]["p90"] < 0.3, s    # a wall facing the cameras: sideways is not depth
    inside = (b["kind"] == CO.KIND_NEIGHBOUR) & ((b["i"] < 6) == (b["j"] < 6))      # neighbours of ONE pass
    assert b["across_cm"][inside].max() < 0.35, "inside a pass that moved as one piece nothing changed"
    assert np.all(np.abs(b["across_cm"][(b["kind"] == CO.KIND_NEIGHBOUR) & ~inside] - 3.0) < 0.4)
    v = CO.judge_zones(a, b, zones, CFG)
    assert v["worse"] == [0, 1], v                              # a displacement BETWEEN two passes is of both zones
    for z in (0, 1):
        assert v["zones"][z]["series"]["different passes / across_cm"]["verdict"] == "worse"
        assert v["zones"][z]["series"]["neighbours / across_cm"]["verdict"] == "same"
    back = CO.judge_zones(b, a, zones, CFG)
    assert back["better"] == [0, 1] and back["worse"] == []


def test_a_stretch_at_another_scale_agrees_with_itself_and_not_with_the_other_pass(world):
    """Lemma 1 of the audit: keyframes with a common wrong depth factor confirm each other — only another pass
    tells."""
    maps, c2w, pairs, kinds, zones = world
    a = CO.measure(maps, _state("truth", c2w), pairs, kinds, CFG)
    c = CO.measure(maps, _state("scaled", c2w, [1.05] * 6 + [1.0] * 6), pairs, kinds, CFG)
    nb = (c["kind"] == CO.KIND_NEIGHBOUR)
    inside0 = nb & (c["i"] < 6) & (c["j"] < 6)
    assert c["along_cm"][inside0].max() < 0.6, "the stretch agrees with itself"
    ps = c["kind"] == CO.KIND_PASS
    assert np.all(np.abs(c["along_cm"][ps] - 15.0) < 1.5), c["along_cm"][ps]      # 5 % of 3 m
    v = CO.judge_zones(a, c, zones, CFG)
    assert v["worse"] == [0, 1] and v["zones"][0]["series"]["different passes / along_cm"]["verdict"] == "worse"


def test_a_state_against_itself_is_the_same_everywhere(world):
    maps, c2w, pairs, kinds, zones = world
    a = CO.measure(maps, _state("truth", c2w), pairs, kinds, CFG)
    v = CO.judge_zones(a, a, zones, CFG)
    assert v["worse"] == [] and v["better"] == [] and v["same"] == [0, 1]


def test_the_lens_is_removed_and_verified():
    u = torch.linspace(0, W - 1, 50, dtype=torch.float64); v = torch.linspace(0, H - 1, 50, dtype=torch.float64)
    x, y = CO.unproject(u, v, K, LENS)
    ub, vb = CO.project(x, y, K, LENS)
    assert float((ub - u).abs().max()) < 1e-9 and float((vb - v).abs().max()) < 1e-9
    with pytest.raises(CO.CoherenceError, match="lens could not be removed"):
        CO.unproject(u, v, K, (5.0, 0.0, 0.0, 0.0), iterations=3)


def test_the_parameters_come_from_the_typed_configuration_and_a_missing_key_is_named():
    import copy
    from config import cfg as raw_cfg
    from precision.config import PrecisionConfigError, load_precision_config
    pcfg = load_precision_config()
    P = CO.params(pcfg, raw_cfg, device="cpu")
    assert P.device == "cpu" and P.min_judges == 5 and P.confidence == pcfg.coherence.heldout_confidence
    assert P.neighbour_offsets == tuple(sorted(set(abs(int(o)) for o in pcfg.bend.neighbors))) and min(P.neighbour_offsets) > 0
    assert P.pass_min_gap == pcfg.tracks.window_frames and P.range_m == pcfg.view_range_m
    assert P.error_factor == raw_cfg["correction_graph"]["graph"]["improvement_error_factor"]
    assert P.conf_floor_norm == raw_cfg["reconstruction"]["simple"]["conf_min_norm"]
    bad = copy.deepcopy(dict(raw_cfg))
    del bad["reconstruction"]["precision"]["coherence"]["search_px"]
    with pytest.raises(PrecisionConfigError, match="coherence.search_px"):
        load_precision_config(bad)
