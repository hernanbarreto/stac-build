"""Phase C (precision task) — native-resolution depth refinement, synthetic tests.

Edge preservation is THE requirement (the legacy upsampler lost its A/B on
smeared edges): a depth step must stay a step after ×2 guided upsampling, flats
must stay flat, the DA3 detail transfer must graft ONLY high frequencies at the
metric scale, and the cap must bound hallucination."""
import numpy as np

from reconstruction import native_depth as nd

H = W = 48
F = 2


def _step_scene():
    """Left half depth 2 m, right half 4 m; guide has the same vertical edge."""
    d = np.full((H, W), 2.0, np.float32)
    d[:, W // 2:] = 4.0
    g = np.full((H * F, W * F), 40, np.uint8)
    g[:, W * F // 2:] = 200
    return d, g


def test_guided_upsample_preserves_step_edge():
    d, g = _step_scene()
    up, valid = nd.guided_upsample(d, None, g, F, device="cpu")
    assert up.shape == (H * F, W * F) and valid.all()
    inner = up[8:-8, :]
    # NO intermediate depths: every pixel stays on one side of the step
    mid = (inner > 2.2) & (inner < 3.8)
    assert mid.mean() < 0.01, f"{100 * mid.mean():.1f}% smeared pixels across the edge"
    assert abs(float(np.median(inner[:, : W * F // 2 - 2])) - 2.0) < 0.02
    assert abs(float(np.median(inner[:, W * F // 2 + 2:])) - 4.0) < 0.02


def test_guided_upsample_smooths_flat_noise():
    rng = np.random.default_rng(0)
    d = (2.0 + 0.02 * rng.standard_normal((H, W))).astype(np.float32)
    g = np.full((H * F, W * F), 128, np.uint8)
    up, _ = nd.guided_upsample(d, None, g, F, device="cpu")
    assert np.std(up[4:-4, 4:-4]) < np.std(d[2:-2, 2:-2])   # smoother, not noisier


def test_guided_upsample_no_hallucination_on_invalid():
    d, g = _step_scene()
    v = np.ones_like(d, bool)
    v[:, : W // 4] = False                                   # invalid strip
    up, valid = nd.guided_upsample(d, v, g, F, device="cpu")
    assert not valid[:, : W * F // 4 - F].any()              # stays invalid


def test_detail_transfer_grafts_high_freq_at_scale():
    base = np.full((H * F, W * F), 2.0, np.float32)
    valid = np.ones_like(base, bool)
    yy = np.arange(H * F)[:, None]
    ripple = 0.01 * np.sin(2 * np.pi * yy / 6.0)             # high-freq detail
    da3 = (1.0 + ripple / 2.0).astype(np.float32) * np.float32(1.0)
    # da3 lives at HALF the metric scale (gain must recover ×2)
    out, mean_rel = nd.detail_transfer(base, valid, da3, None, F, device="cpu")
    got = out[12:-12, 12:-12] - 2.0
    assert got.std() > 0.005                                 # ripple transferred
    # amplitude ≈ metric ripple (0.01·2/2 = 0.01 → std ≈ 0.007)
    assert 0.004 < got.std() < 0.012
    assert mean_rel < 0.02


def test_detail_transfer_cap_bounds_hallucination():
    base = np.full((H * F, W * F), 2.0, np.float32)
    valid = np.ones_like(base, bool)
    da3 = base.copy()
    da3[40:60, 40:60] = 10.0                                 # absurd DA3 structure
    out, _ = nd.detail_transfer(base, valid, da3, None, F,
                                detail_cap_rel=0.10, device="cpu")
    assert float(np.max(np.abs(out - base) / base)) <= 0.10 + 1e-5


def test_run_writes_npz_and_report(tmp_path):
    d, g = _step_scene()
    frames = {0: {"depth": d, "valid": np.ones_like(d, bool), "guide": g},
              10: {"depth": d, "valid": np.ones_like(d, bool), "guide": g}}
    rep = nd.run(tmp_path, tmp_path, method="guided_filter", factor=F,
                 _frames_override=frames, device="cpu")
    assert rep["n_keyframes"] == 2
    npz = np.load(tmp_path / nd.ND_DIRNAME / "frame_0.npz")
    assert npz["depth"].shape == (H * F, W * F)
    # resume: identical params → skip (no override needed)
    rep2 = nd.run(tmp_path, tmp_path, method="guided_filter", factor=F,
                  _frames_override=None)
    assert rep2["n_keyframes"] == 2
