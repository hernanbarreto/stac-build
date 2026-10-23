"""The bend's depth model (precision/bend_model.py, audit 2026-10-23): the pixel-ratio family of epoch 7
curves every plane seen obliquely as soon as a tilt term enters; the inverse-depth family takes planes
to planes exactly, recovers monocular depth's scale-and-shift error, and the held-out judge chooses
between them. Synthetic, no GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import bend_model as BM                       # noqa: E402
from precision import depth_on_f5 as D                       # noqa: E402

# pccr's camera: 464x832 portrait, f ≈ 388 px
W, H = 464, 832
K = np.array([[388.0, 0, W / 2], [0, 388.0, H / 2], [0, 0, 1]])
UU, VV = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))


def _fac() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


FAC = _fac()


def _floor(height_m: float = 1.5, far_m: float = 8.0):
    """A floor ``height_m`` under the camera (camera y down), seen below the horizon, up to far_m."""
    yt = (VV - K[1, 2]) / K[1, 1]
    z = np.where(yt > 0, height_m / np.maximum(yt, 1e-9), np.nan)
    valid = np.isfinite(z) & (z < far_m)
    return np.where(valid, z, 0.0), valid


def test_the_pixel_ratio_family_bends_an_oblique_plane_and_the_inverse_depth_family_does_not():
    """A floor plane under the pixel family with a 3 % left-right tilt leaves its best plane by
    centimetres (8.8 cm at 10 %); under the inverse-depth family with tilt terms of the same order
    it stays a plane to numerical precision."""
    z, valid = _floor()
    rms0, _ = BM.plane_rms_m(z, K, valid)
    assert rms0 < 1e-6
    for c1, bar_cm in ((0.03, 0.5), (0.10, 1.8)):
        zb = BM.apply_map(BM.AFFINE_PIXEL, np.array([1.0, c1, 0.0]), z, W, H)
        rms, mx = BM.plane_rms_m(zb, K, valid)
        assert rms > bar_cm / 100, f"c1 {c1}: the pixel family left the plane by rms {rms * 100:.2f} cm"
    zb = BM.apply_map(BM.AFFINE_PIXEL, np.array([1.0, 0.10, 0.0]), z, W, H)
    assert BM.plane_rms_m(zb, K, valid)[1] > 0.08                  # 8.8 cm at the image's edge
    for coef in ([1.05, 0.0, 0.0, 0.0], [1.0, 0.02, 0.0, 0.0], [1.0, 0.0, 0.02, 0.02], [0.97, 0.01, -0.02, 0.03]):
        zi = BM.apply_map(BM.INVERSE_DEPTH, np.array(coef), z, W, H)
        rms, mx = BM.plane_rms_m(zi, K, valid & (zi > 0))
        assert mx < 1e-6, f"inverse_depth {coef}: a plane came out bent by {mx * 100:.4f} cm"


def test_the_inverse_depth_family_recovers_scale_and_shift_and_drops_what_is_not_there():
    """Monocular depth's classic error — an affine map of the inverse depth — is recovered exactly;
    on a pure scale error the shift and tilt terms are dropped to 0 exactly (point 48), on a pure
    scale error the pixel family drops its tilts too."""
    rng = np.random.default_rng(0)
    n = 400
    u = rng.uniform(0, W, n); v = rng.uniform(0, H, n)
    z_true = rng.uniform(1.0, 6.0, n)
    z_lm = z_true * (1 + rng.normal(0, 0.002, n))                     # the landmarks, 0.2 % of scatter
    a, b0 = 1.08, -0.02
    z_om = 1.0 / ((1.0 / z_true - b0) / a)                            # 1/z = a/z_om + b0 → Omega's depth
    rows = {0: BM.fit_rows(BM.INVERSE_DEPTH, u, v, z_lm, z_om, W, H)}
    c, info = D.bend_coefficients(rows, 1, 0, 20, 1.345, 10, FAC, BM.INVERSE_DEPTH)
    assert abs(c[0][0] - a) < 2e-3 and abs(c[0][1] - b0) < 2e-3
    # a tilt that is not there enters only at the 2σ bar's own false-positive rate and then stays
    # negligible (|b| < 1e-3 in inverse depth: under a millimetre at 3 m); the shift is kept
    assert abs(c[0][2]) < 1e-3 and abs(c[0][3]) < 1e-3
    assert info[0]["model"] == BM.INVERSE_DEPTH and info[0]["kept"]["b0"] is True
    assert np.median(D.rel_errors(u, v, z_lm, z_om, c[0], W, H, BM.INVERSE_DEPTH)) < 0.003
    # a pure scale error: only the scale enters, in both families
    z_om = z_true / 0.9
    for model in BM.MODELS:
        rows = {0: BM.fit_rows(model, u, v, z_lm, z_om, W, H)}
        c, info = D.bend_coefficients(rows, 1, 0, 20, 1.345, 10, FAC, model)
        assert abs(BM.scale_equivalent(model, c[0]) - 0.9) < 2e-3
        assert all(abs(x) < 1e-3 for x in BM.tilt_terms(model, c[0])), (model, c[0])
    # the identity of each family leaves Omega's depth alone
    for model in BM.MODELS:
        assert np.allclose(BM.apply(model, BM.identity(model), z_om, u, v, W, H), z_om)
    with pytest.raises(BM.BendModelError):
        BM.apply(BM.AFFINE_PIXEL, np.zeros(4), z_om, u, v, W, H)
    # a pixel the inverse model cannot place (non-positive inverse depth) has no depth, never a negative one
    out = BM.apply(BM.INVERSE_DEPTH, np.array([1.0, -2.0, 0.0, 0.0]), np.array([1.0, 0.4]), np.zeros(2), np.zeros(2), W, H)
    assert out[0] == 0.0 and out[1] > 0


def test_the_judge_prefers_the_plane_preserving_family_within_the_bar_and_the_better_one_beyond_it():
    """choose_model (bend.model: judged): the plane-preserving family wins a tie by the window rule's
    own construction; a family that is better beyond the bar wins outright, declared."""
    rng = np.random.default_rng(1)
    errs_eq = {w: {i: rng.normal(0.03, 0.002, 60) for i in range(8)} for w in (0, 1)}
    se = D.cluster_bootstrap_se(list(errs_eq[0].values()), 300, 0)
    tie = 0.5 * FAC * se                                           # well inside the bar: best + FAC x se
    score = {BM.AFFINE_PIXEL: {0: 0.0300, 1: 0.0300 + tie}, BM.INVERSE_DEPTH: {0: 0.0300 + tie, 1: 0.0300 + 1.5 * tie}}
    errs = {BM.AFFINE_PIXEL: errs_eq, BM.INVERSE_DEPTH: errs_eq}
    model, w, rule = D.choose_model(score, errs, FAC, 300, 0)
    assert model == BM.INVERSE_DEPTH and rule["best_model"] == BM.AFFINE_PIXEL and rule["model"] == BM.INVERSE_DEPTH
    assert w == 1                                                  # the widest window within the bar
    score2 = {BM.AFFINE_PIXEL: {0: 0.010, 1: 0.011}, BM.INVERSE_DEPTH: {0: 0.030, 1: 0.030}}
    model, w, rule = D.choose_model(score2, errs, FAC, 300, 0)
    assert model == BM.AFFINE_PIXEL and rule["curve"][BM.INVERSE_DEPTH]["0"]["within"] is False


def test_the_chunk_check_measures_the_published_depth_under_either_model():
    """f6_check applies to Omega's record exactly what the cloud applied: the pixel ratio (s_k + bend,
    epochs folded in) or the family named in the report, times the epochs' factor, plus the offset."""
    from precision.chunk_check import published_depth, scale_map
    z, valid = _floor()
    d_aff = published_depth(z, 0.95, (0.04, -0.03), 0.0)
    assert np.allclose(d_aff, z * scale_map(0.95, (0.04, -0.03), H, W))
    coef = np.array([1.06, -0.01, 0.02, 0.0])
    d_inv = published_depth(z, 1.0, (0.0, 0.0), 0.05, depth_model=(BM.INVERSE_DEPTH, coef), k_fac=1.02)
    ref = BM.apply_map(BM.INVERSE_DEPTH, coef, z, W, H) * 1.02
    ref = np.where(np.isfinite(z) & (z > 0), ref + 0.05, ref)
    assert np.allclose(d_inv, ref)


def test_disabled_bend_and_verification_carry_the_model():
    wb, rule, coefs, prov = D.disabled_bend(2, BM.INVERSE_DEPTH)
    assert wb is None and np.array_equal(coefs[0], BM.identity(BM.INVERSE_DEPTH)) and prov[1]["status"] == "disabled"
    rows = {i: (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)) for i in range(2)}
    final, prov = D.verify_keyframes((0,), 0, {0: {i: BM.identity(BM.INVERSE_DEPTH) for i in range(2)}},
                                     {0: {i: {"fallback": "identity"} for i in range(2)}}, rows, 2, W, H, FAC,
                                     0.95, 50, 0, 20, BM.INVERSE_DEPTH)
    assert prov[0]["model"] == BM.INVERSE_DEPTH and np.array_equal(final[0], BM.identity(BM.INVERSE_DEPTH))
