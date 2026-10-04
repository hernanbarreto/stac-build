"""The depth on F5 (USER 2026-10-01, pccr epoch 7): Omega's depth bent to F5's landmarks, then a vote.
Synthetic, no GPU: a plane seen by three cameras."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.depth_on_f5 import bend_coefficients, camera_travels, design, edge_keeping_vote  # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


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
    c = bend_coefficients(rows, 4, 0, 20, 1.345, 10)
    assert abs(c[0][0] - 0.9) < 1e-2 and abs(c[2][0] - 1.1) < 1e-2
    assert np.allclose(c[3], [1.0, 0.0, 0.0])


def test_the_edge_keeping_vote_removes_a_flyer_and_keeps_the_surface():
    """pccr epoch 8's vote (the pipeline's since 2026-10-01): a flyer the neighbours see through leaves,
    the wall stays at its own depth."""
    cams = _cams(); order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    dep[1][10:20, 30:40] = 1.5                       # a flyer in front of the wall the others see
    ok = {f: np.ones((H, W), bool) for f in order}
    final, tau, tot = edge_keeping_vote(order, dep, ok, {f: ok[f].copy() for f in order}, K, cams, None,
                                        (-1, 1), 75.0, 2, log=lambda *a: None)
    z1, a1 = final[1]
    # the flyer is gone: its pixels see the wall behind it, so they are REPAIRED to the depth the two
    # neighbours agree on (epoch 8: a contradicted pixel still sees a surface) — never left at 1.5 m
    assert np.allclose(z1[12:18, 32:38], 3.0, atol=1e-3), "the flyer's pixels take the wall the neighbours see"
    assert np.allclose(z1[40:50, 30:40], 3.0, atol=1e-3) and (a1[40:50, 30:40] >= 1).all()
    assert tot["contradicted"] > 0 and tot["repaired"] > 0 and 0 <= tau


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
    assert pcfg.mono_detail.enabled is False
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
