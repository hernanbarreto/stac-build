"""Phases 2-4 and 6 of the mono-detail work (claude_stac.txt 2026-10-04): tiles, per-tile affine, detail and
discontinuity band, mixed-pixel resolution. Synthetic, no GPU, PointDiT mocked."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import mono_detail as MD  # noqa: E402
from precision.config import load_precision_config  # noqa: E402


def _mcfg(**kw):
    return replace(load_precision_config().mono_detail, **kw)


# ── Phase 2 ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H,W,ctx", [(832, 464, 1.0), (832, 464, 2.0), (300, 1000, 1.0), (64, 64, 1.0)])
def test_tiles_cover_the_frame_borders_included_and_weights_sum_to_one(H, W, ctx):
    tiles = MD.plan_tiles(H, W, 512, 0.25, ctx)
    cov = np.zeros((H, W), int)
    for t in tiles:
        assert t.h % 16 == 0 and t.w % 16 == 0 and t.run_h % 16 == 0 and t.run_w % 16 == 0
        assert t.y0 + t.h <= H and t.x0 + t.w <= W
        cov[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] += 1
    assert cov.min() >= 1                                     # every pixel, the borders included
    ws = MD.feather_weights(H, W, tiles)
    total = np.sum(ws, axis=0)
    assert np.allclose(total, 1.0)
    if len(tiles) > 1:                                        # the overlap is shared, not a hard seam
        inner = [w for w in ws if (w > 0).sum() > 0]
        assert any(((w > 0) & (w < 1)).any() for w in inner)


def test_context_scale_two_runs_a_larger_window_at_the_tile_size():
    t1 = MD.plan_tiles(832, 464, 512, 0.25, 1.0)[0]
    t2 = MD.plan_tiles(832, 464, 512, 0.25, 2.0)[0]
    assert (t1.h, t1.w, t1.run_h, t1.run_w) == (512, 464, 512, 464)
    assert (t2.h, t2.w) == (832, 464) and (t2.run_h, t2.run_w) == (416, 224)
    assert MD.band_half_width_px(1.0) == 1 and MD.band_half_width_px(2.0) == 2


# ── Phase 3 ──────────────────────────────────────────────────────────────────

def _ramp(H=64, W=96):
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    return (2.0 + 0.01 * uu + 0.005 * vv).astype(np.float32)


def test_affine_recovers_s_and_b_with_30_percent_outliers():
    rng = np.random.default_rng(0)
    z_cal = _ramp()
    z_mono = ((z_cal - 1.7) / 0.6).astype(np.float32)        # z_cal = 0.6 z_mono + 1.7
    noisy = z_cal.copy()
    out = rng.random(z_cal.shape) < 0.30
    noisy[out] += rng.normal(0, 0.8, out.sum())               # gross outliers on 30 %
    sup = np.ones_like(out); w = np.ones(z_cal.shape)
    z_al, s, b, res = MD.align_tile(noisy, z_mono, sup, w, "depth", 1.345, 10)
    assert abs(s - 0.6) < 0.03 and abs(b - 1.7) < 0.06
    assert np.abs(z_al[~out] - z_cal[~out]).max() < 0.05
    z_al2, s2, b2, _ = MD.align_tile(noisy, z_mono, sup, w, "inverse", 1.345, 10)
    assert np.median(np.abs(z_al2[~out] - z_cal[~out]) / z_cal[~out]) < 0.03


def test_tile_without_support_is_rejected_and_seams_show_no_step():
    rng = np.random.default_rng(1)
    H, W = 64, 160
    z_cal = _ramp(H, W)
    # PointDiT "sees" the same ramp but each tile returns it in ITS OWN affine space
    def runner_depth(img, size):
        h, w = size
        tile = img[..., 0].astype(np.float32)                # the image carries z_cal in channel 0 (test trick)
        a, b = rng.uniform(0.5, 2.0), rng.uniform(-3, 3)
        return (a * tile + b).astype(np.float32), np.ones((h, w), bool)
    class R:
        depth = staticmethod(runner_depth)
    img_of = lambda f: np.repeat(z_cal[:, :, None], 3, 2)
    valid = {0: np.ones((H, W), bool)}
    weight = {0: np.ones((H, W))}
    weight[0][:, :48] = 0.0                                   # the left part carries no confidence at all
    mcfg = _mcfg(tile_px=64, tile_overlap_frac=0.5, context_scale=1.0, min_support_frac=0.5,
                 tile_residual_quantile=100.0, lowpass_patch_fraction=0.5, side_reach_px=3)
    dep, src, rep = MD.run_stage([0], {0: z_cal.copy()}, valid, weight, img_of, R(), mcfg, tau=0.02,
                                 native_px_per_omega_px=1.0, huber_k=1.345, log=lambda m: None)
    assert rep.rejected_support >= 1 and rep.accepted >= 2
    d = dep[0]
    # where tiles were accepted the refined ramp equals the ramp (no seam step)
    m = src[0] == MD.SRC_DETAIL
    assert m.sum() > 0.4 * H * W
    assert np.abs(d[m] - z_cal[m]).max() < 0.01
    steps = np.abs(np.diff(d, axis=1))[m[:, 1:] & m[:, :-1]]
    assert steps.max() < 0.02


# ── Phase 4 ──────────────────────────────────────────────────────────────────

def test_plane_gives_no_detail_and_a_step_gives_a_localised_band():
    H, W = 48, 64
    z = _ramp(H, W); v = np.ones((H, W), bool)
    lp, ok = MD.masked_lowpass(z, v, 3.0)
    assert np.abs((z - lp)[8:-8, 8:-8]).max() < 2e-3 and ok.all()   # a ramp is its own lowpass (away from the border)
    step = np.where(np.arange(W)[None, :] < 32, 1.5, 3.0).astype(np.float32) * np.ones((H, 1), np.float32)
    disc, band = MD.discontinuity_band(step, v, 0.02, 2)
    cols = np.nonzero(disc.any(0))[0]
    assert set(cols.tolist()) == {31, 32}                     # the jump sits between columns 31 and 32
    bcols = np.nonzero(band.any(0))[0]
    assert bcols.min() == 29 and bcols.max() == 34            # dilated by band_px = 2 on each side
    assert MD.lowpass_sigma_px(16, 1.0, 0.5) == 8.0


# ── Phase 6 ──────────────────────────────────────────────────────────────────

def test_mixed_pixels_go_to_a_side_never_in_between_and_without_support_stay_unresolved():
    H, W = 40, 64
    front, back = 1.5, 3.0
    z_cal = np.where(np.arange(W)[None, :] < 32, front, back).astype(np.float32) * np.ones((H, 1), np.float32)
    z_cal[:, 31] = 2.1; z_cal[:, 32] = 2.6                  # Omega's mixed pixels on the edge
    v = np.ones((H, W), bool)
    # PointDiT (aligned): the sharp edge, column 31 belongs to the front, 32 to the back;
    # on rows >= 30 PointDiT is itself intermediate there (no evidence)
    z_al = np.where(np.arange(W)[None, :] < 32, front, back).astype(np.float32) * np.ones((H, 1), np.float32)
    z_al[30:, 31] = 2.2; z_al[30:, 32] = 2.3
    fr = MD.refine_frame(z_cal, v, z_al, v, tau=0.02, band_px=1, sigma_px=3.0, reach_px=3, align_err=0.0)
    d = fr.depth; s = fr.source
    assert (s[:30, 31] == MD.SRC_BAND_FRONT).all() and np.allclose(d[:30, 31], front)
    assert (s[:30, 32] == MD.SRC_BAND_BACK).all() and np.allclose(d[:30, 32], back)
    assert (s[30:, 31] == MD.SRC_UNRESOLVED).all() and (d[30:, 31] == 0).all()
    assert (s[30:, 32] == MD.SRC_UNRESOLVED).all()
    # no band pixel in the measurement tier (depth > 0) is intermediate
    band_pts = fr.band & (d > 0)
    assert not ((d[band_pts] > front * 1.02) & (d[band_pts] < back * 0.98)).any()
    # surfaces far from the edge: the detail term is ~0 on flat surfaces
    assert np.abs(d[:, 5:20] - front).max() < 1e-3 and np.abs(d[:, 44:60] - back).max() < 1e-3
    assert fr.counts["mixed"] == 2 * H and fr.counts["mixed_unresolved"] == 20


def test_production_config_declares_the_phase_keys():
    m = load_precision_config().mono_detail
    assert m.enabled is False and m.tile_px >= 16 and m.fit_space in ("depth", "inverse")
