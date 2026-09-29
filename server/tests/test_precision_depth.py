"""F6 — prior-guided plane sweep, tiers, confidence calibration, COLMAP reference I/O.

A synthetic textured scene with ground truth: a textured back wall, a textured
box face in front of it and an UNTEXTURED patch. The sweep must beat its prior
on texture, per distance, and never produce a tier-0 depth where the images carry
no signal. The calibration must recover an injected confidence → error curve.
CPU only (torch on CPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from precision import confidence as CAL                                   # noqa: E402
from precision import depth_colmap as DC                                  # noqa: E402
from precision import depth_sweep as DS                                   # noqa: E402

W, H = 128, 96
K = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])
WALL_Z, BOX_Z, FLAT_Z = 4.0, 2.5, 3.2
BOX = (-0.9, -0.1, -0.35, 0.35)          # x0, x1, y0, y1 of the box face
FLAT = (0.35, 1.0, -0.45, 0.25)          # the untextured patch


def _noise_grid(seed, n=256):
    return np.random.default_rng(seed).random((n, n))


GRIDS = [_noise_grid(1), _noise_grid(2), _noise_grid(3)]


def _value_noise(a, b, grid, cell):
    """Bilinear value noise of world coordinates (a, b) at ``cell`` metres."""
    n = grid.shape[0]
    x, y = a / cell + n / 2, b / cell + n / 2
    x0, y0 = np.floor(x).astype(int) % (n - 1), np.floor(y).astype(int) % (n - 1)
    fx, fy = x - np.floor(x), y - np.floor(y)
    g = grid
    return ((1 - fx) * (1 - fy) * g[y0, x0] + fx * (1 - fy) * g[y0, x0 + 1]
            + (1 - fx) * fy * g[y0 + 1, x0] + fx * fy * g[y0 + 1, x0 + 1])


def _texture(X, which):
    a, b = X[..., 0], X[..., 1]
    g = GRIDS[which]
    return 0.15 + 0.5 * _value_noise(a, b, g, 0.05) + 0.3 * _value_noise(a + 3.1, b - 1.7, g, 0.021)


def render(c2w, K_=K, wh=(W, H), ss=2):
    """(gray [0,1] quantised to 8 bits, z-depth) of the scene from camera c2w."""
    w, h = wh[0] * ss, wh[1] * ss
    Ks = K_.copy()
    Ks[:2] *= ss
    Ks[0, 2] = (K_[0, 2] + 0.5) * ss - 0.5
    Ks[1, 2] = (K_[1, 2] + 0.5) * ss - 0.5
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    d = np.stack([(u - Ks[0, 2]) / Ks[0, 0], (v - Ks[1, 2]) / Ks[1, 1], np.ones_like(u)], -1)
    R, C = c2w[:3, :3], c2w[:3, 3]
    dw = d @ R.T
    best_t = np.full((h, w), np.inf)
    img = np.zeros((h, w))
    for zp, rect, kind in ((WALL_Z, None, 0), (BOX_Z, BOX, 1), (FLAT_Z, FLAT, None)):
        t = (zp - C[2]) / dw[..., 2]
        X = C + t[..., None] * dw
        ok = t > 0
        if rect is not None:
            ok &= (X[..., 0] >= rect[0]) & (X[..., 0] <= rect[1]) & (X[..., 1] >= rect[2]) \
                & (X[..., 1] <= rect[3])
        ok &= t < best_t
        best_t = np.where(ok, t, best_t)
        val = np.full((h, w), 0.5) if kind is None else _texture(X, kind)
        img = np.where(ok, val, img)
    z = best_t * d[..., 2]                          # depth along the optical axis (ref frame)
    img = img.reshape(wh[1], ss, wh[0], ss).mean((1, 3))
    Xs = C + best_t[..., None] * dw
    zc = ((Xs - C) @ R)[..., 2].reshape(wh[1], ss, wh[0], ss)[:, ss // 2, :, ss // 2]
    return np.round(img * 255) / 255, zc.astype(np.float64)


def cam(tx, ty, tz=0.0, yaw_deg=0.0):
    c = np.eye(4)
    a = np.radians(yaw_deg)
    c[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c[:3, 3] = [tx, ty, tz]
    return c


REF = cam(0, 0)
VIEWS = [cam(0.25, 0.0), cam(-0.25, 0.02), cam(0.0, 0.2), cam(0.12, -0.18), cam(-0.18, -0.12)]


def prior_error(shape, amp=0.06, phase=0.0):
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    return amp * np.sin(u / 17.0 + phase) * np.cos(v / 13.0 - phase) + 0.02


@pytest.fixture(scope="module")
def scene():
    ref, gt = render(REF)
    views = [render(c) for c in VIEWS]
    return ref, gt, views


def _frame(ref, views, c2w_ref=REF, c2ws=VIEWS, best_k=3):
    return DS._Frame(ref.astype(np.float32), K, np.linalg.inv(c2w_ref),
                     [v[0].astype(np.float32) for v in views],
                     [np.ones_like(v[0], bool) for v in views],
                     [np.linalg.inv(c) for c in c2ws], 7, best_k, "cpu")


def _regions(gt):
    v, u = np.mgrid[0:H, 0:W]
    inner = (u >= 12) & (u < W - 12) & (v >= 12) & (v < H - 12)
    # stay clear of the depth edges: a pixel whose 9×9 neighbourhood has one depth
    z = gt
    edge = np.zeros_like(z, bool)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            edge |= np.abs(np.roll(np.roll(z, dy, 0), dx, 1) - z) > 0.05
    Xr = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    flat = (np.abs(z - FLAT_Z) < 1e-6) & (Xr[..., 0] >= FLAT[0]) & (Xr[..., 0] <= FLAT[1])
    box = np.abs(z - BOX_Z) < 1e-6
    wall = np.abs(z - WALL_Z) < 1e-6
    return inner & ~edge, box, wall, flat


def test_sweep_beats_prior_on_texture_per_distance_and_never_scores_flat(scene):
    ref, gt, views = scene
    z0 = gt * (1 + prior_error(gt.shape))
    fr = _frame(ref, views)
    r = DS.sweep_frame(fr, z0.astype(np.float32), np.full(gt.shape, 0.15, np.float32), 24, 2,
                       DS.contrast_sigma(ref, np.ones_like(ref, bool)))
    ok, box, wall, flat = _regions(gt)
    for name, m in (("box 2.5 m", ok & box), ("wall 4 m", ok & wall)):
        e_sw = np.median(np.abs(r["depth"][m] / gt[m] - 1))
        e_pr = np.median(np.abs(z0[m] / gt[m] - 1))
        assert m.sum() > 200, name
        assert e_sw < 0.25 * e_pr, f"{name}: sweep {e_sw:.4f} vs prior {e_pr:.4f}"
        assert e_sw < 0.01, name
    # the untextured patch: every window fully inside it has no signal at all
    core = flat.copy()
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            core &= np.roll(np.roll(flat, dy, 0), dx, 1)
    assert core.sum() > 100
    assert np.all(r["score"][core] == -1.0)


def test_null_floor_separates_real_views_from_shuffled(scene):
    ref, gt, views = scene
    z0 = (gt * (1 + prior_error(gt.shape))).astype(np.float32)
    beta = np.full(gt.shape, 0.15, np.float32)
    cs = DS.contrast_sigma(ref, np.ones_like(ref, bool))
    # null: every view's image replaced by an image of a DIFFERENT scene (other textures)
    rng = np.random.default_rng(7)
    fake = [(np.round(np.clip(cv2.GaussianBlur(rng.random((H, W)), (0, 0), 1.2) * 1.4 - 0.2, 0, 1)
                      * 255) / 255, None) for _ in VIEWS]
    rn = DS.sweep_frame(_frame(ref, fake), z0, beta, 24, 2, cs)
    m = z0 > 0
    table = DS.floor_table(rn["score"][m], rn["ref_std"][m], 0.95, 4)
    rr = DS.sweep_frame(_frame(ref, views), z0, beta, 24, 2, cs)
    ok, box, wall, _ = _regions(gt)
    sig = rr["score"] > DS.floor_of(table, rr["ref_std"])
    assert sig[ok & (box | wall)].mean() > 0.9
    assert (rn["score"] > DS.floor_of(table, rn["ref_std"]))[m].mean() <= 0.06


def test_consistency_counts_agreeing_views_and_rejects_a_wrong_depth(scene):
    _, gt, views = scene
    nb = [v[1] for v in views[:3]]
    nw = [np.linalg.inv(c) for c in VIEWS[:3]]
    n, res, bad = DS.consistency(gt, K, np.linalg.inv(REF), nb, nw, tau_rel=0.01, tau_px=1.0)
    ok, box, wall, _ = _regions(gt)
    assert np.median(n[ok & wall]) == 3
    assert np.nanmedian(res[ok & wall]) < 0.005
    assert bad[ok & wall].max() == 0                       # nobody sees through a true surface
    n2, _, bad2 = DS.consistency(gt * 1.1, K, np.linalg.inv(REF), nb, nw, tau_rel=0.01, tau_px=1.0)
    assert n2[ok].max() == 0
    # a point placed 10 % IN FRONT of the true surface: the other views measure the
    # surface FARTHER along its ray — they see free space through it → contradicted
    n3, _, bad3 = DS.consistency(gt * 0.9, K, np.linalg.inv(REF), nb, nw, tau_rel=0.01, tau_px=1.0)
    assert n3[ok].max() == 0 and np.median(bad3[ok & wall]) >= 2


def test_tiers_textureless_is_prior_fill_or_discarded_never_tier0(scene):
    """prior_requires_confirmation true: the rule until 2026-09-29, unchanged."""
    ref, gt, views = scene
    ok, box, wall, flat = _regions(gt)
    sig = np.zeros(gt.shape, bool)
    sig[ok & (box | wall)] = True
    n_s = np.where(sig, 3, 0).astype(np.uint8)
    z0 = gt.astype(np.float32)
    n_p = np.full(gt.shape, 2, np.uint8)
    conf = dict(prior_requires_confirmation=True)
    src = DS.assign_tiers(sig, n_s, z0, n_p, None, min_consistent_views=2,
                          prior_fill_min_views=2, prior_fill="keep", **conf)
    assert np.all(src[sig] == DS.SOURCE_SWEEP)
    assert np.all(src[flat & ~sig] == DS.SOURCE_PRIOR_FILL)
    src2 = DS.assign_tiers(sig, n_s, z0, np.zeros_like(n_p), None, 2, 2, "keep", **conf)
    assert np.all(src2[flat & ~sig] == DS.DISCARD_NO_SIGNAL)
    src3 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 2, "drop", **conf)
    assert np.all(src3[flat & ~sig] == DS.DISCARD_PRIOR_FILL_DROPPED)
    src4 = DS.assign_tiers(sig, np.zeros_like(n_s), z0, n_p, flat, 2, 2, "keep", **conf)
    assert np.all(src4[sig] == DS.DISCARD_INCONSISTENT)
    assert np.all(src4[flat] == DS.DISCARD_EXCLUDED)
    assert not np.any(src4 == DS.SOURCE_SWEEP)
    # contradictions: a prior any view sees through is out; a measured depth stands
    # while its confirming views outnumber the contradicting ones
    one = np.ones(gt.shape, np.uint8)
    src5 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 2, "keep", bad_s=one, bad_p=one, **conf)
    assert np.all(src5[flat & ~sig] == DS.DISCARD_CONTRADICTED)
    assert np.all(src5[sig] == DS.SOURCE_SWEEP)            # 3 views for, 1 against
    src6 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 2, "keep", bad_s=one * 3, **conf)
    assert np.all(src6[sig] == DS.DISCARD_CONTRADICTED)     # 3 for, 3 against
    # the confidence floor: a low-confidence prior is out, a measured depth is not
    low = np.ones(gt.shape, bool)
    src7 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 2, "keep", low_conf=low, **conf)
    assert np.all(src7[flat & ~sig] == DS.DISCARD_LOW_CONF) and np.all(src7[sig] == DS.SOURCE_SWEEP)


def test_max_coverage_keeps_unconfirmed_priors_and_still_drops_contradicted_ones(scene):
    """USER 2026-09-29 (prior_requires_confirmation false): every prior pixel that is not
    tier 0 is tier 1 — the sweep's unconfirmed pixels fall back to the prior, no view
    count is asked — unless contradicted, under the confidence floor or excluded."""
    _, gt, _ = scene
    ok, box, wall, flat = _regions(gt)
    H_, W_ = gt.shape
    z0 = gt.astype(np.float32)
    z0[:, :4] = 0                                            # no prior there
    sig = np.zeros(gt.shape, bool)
    sig[ok & (box | wall)] = True
    n_s = np.where(sig, 3, 0).astype(np.uint8)
    unconfirmed = sig & (np.arange(W_)[None, :] % 2 == 0)    # signal, but no view agrees
    n_s[unconfirmed] = 0
    n_p = np.zeros(gt.shape, np.uint8)                       # NO view confirms any prior
    assert unconfirmed.sum() > 100 and (flat & ~sig & (z0 > 0)).sum() > 100
    mc = dict(prior_requires_confirmation=False)
    src = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 3, "keep", **mc)
    assert np.all(src[sig & ~unconfirmed] == DS.SOURCE_SWEEP)            # tier 0 unchanged
    assert np.all(src[unconfirmed] == DS.SOURCE_PRIOR_FILL)              # fell back to the prior
    assert np.all(src[flat & ~sig & (z0 > 0)] == DS.SOURCE_PRIOR_FILL)   # no view count asked
    assert np.all(src[z0 <= 0] == DS.DISCARD_NO_PRIOR)
    assert not np.isin(src, [DS.DISCARD_NO_SIGNAL, DS.DISCARD_INCONSISTENT]).any()
    # the same pixels under the confirmation rule are discarded — the switch is the difference
    src_c = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 3, "keep", prior_requires_confirmation=True)
    assert np.all(src_c[unconfirmed] == DS.DISCARD_INCONSISTENT)
    assert np.all(src_c[flat & ~sig & (z0 > 0)] == DS.DISCARD_NO_SIGNAL)
    # contradicted priors still leave (the majority own-error vote is the caller's bad_p)
    bad_p = np.zeros(gt.shape, np.uint8)
    contra = (np.arange(H_)[:, None] % 3 == 0) & (z0 > 0)
    bad_p[contra] = 2
    src2 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 3, "keep", bad_p=bad_p, **mc)
    assert np.all(src2[contra & ~(sig & ~unconfirmed)] == DS.DISCARD_CONTRADICTED)
    assert np.all(src2[contra & sig & ~unconfirmed] == DS.SOURCE_SWEEP)  # a tier-0 depth is not the prior
    # a sweep outvoted by its contradictions falls back to the prior, judged on its own
    bad_s = np.where(sig, 5, 0).astype(np.uint8)
    src3 = DS.assign_tiers(sig, n_s, z0, n_p, None, 2, 3, "keep", bad_s=bad_s, **mc)
    assert np.all(src3[sig] == DS.SOURCE_PRIOR_FILL)
    # the confidence floor and the exclusion mask still hold
    low = flat.copy()
    src4 = DS.assign_tiers(sig, n_s, z0, n_p, box, 2, 3, "keep", low_conf=low, **mc)
    assert np.all(src4[flat & ~sig & (z0 > 0)] == DS.DISCARD_LOW_CONF)
    assert np.all(src4[box] == DS.DISCARD_EXCLUDED)


def test_a_view_contradicts_only_beyond_its_own_error(scene):
    """pccr 2026-09-29: judged at τ_rel against priors 5-27 % off, ~100 views contradicted
    almost every point. A view's farther surface contradicts only beyond ITS error."""
    _, gt, views = scene
    nb = [v[1] for v in views[:3]]
    nw = [np.linalg.inv(c) for c in VIEWS[:3]]
    ok, box, wall, _ = _regions(gt)
    # the views' depths 5 % too far — a prior's typical error
    far = [d * 1.05 for d in nb]
    _, _, bad_tight = DS.consistency(gt, K, np.linalg.inv(REF), far, nw, tau_rel=0.01, tau_px=1.0)
    assert np.median(bad_tight[ok & wall]) == 3               # τ_rel alone: all contradict
    _, _, bad_own = DS.consistency(gt, K, np.linalg.inv(REF), far, nw, tau_rel=0.01, tau_px=1.0,
                                   nbr_bad_margin=[0.10, 0.10, 0.10])
    assert bad_own[ok & wall].max() == 0                     # within their own 10 %: none


def test_conf_floor_is_the_min_max_fraction_of_the_frame():
    conf = np.array([[0.0, 1.0, 2.0, 10.0]])
    valid = np.ones_like(conf, bool)
    assert DS.conf_floor_mask(conf, valid, 0.10).tolist() == [[True, False, False, False]]
    assert DS.conf_floor_mask(conf, valid, 0.25).tolist() == [[True, True, True, False]]
    assert not DS.conf_floor_mask(conf, valid, 0.0).any()
    assert not DS.conf_floor_mask(np.full((1, 4), 3.0), valid, 0.5).any()   # no range: nothing under


def test_parabola_vertex_and_normals():
    x = np.array([1.0, 2.5, 3.0])
    y = -(x - 2.2) ** 2 + 5
    xv, ok = DS.parabola_vertex(x[:1], x[1:2], x[2:], y[:1], y[1:2], y[2:])
    assert ok[0] and abs(xv[0] - 2.2) < 1e-12
    # a plane z = 3 + 0.5 x (camera frame): normal ∝ (0.5, 0, -1), facing the camera
    v, u = np.mgrid[0:40, 0:50].astype(np.float64)
    a = (u - K[0, 2]) / K[0, 0]
    z = 3.0 / (1 - 0.5 * a)
    n = DS.normals_from_depth(z, K)
    ref = np.array([0.5, 0.0, -1.0]) / np.linalg.norm([0.5, 0.0, -1.0])
    assert np.nanmax(np.abs(n[5:-5, 5:-5] - ref)) < 1e-6


def test_calibration_recovers_injected_confidence_curve():
    rng = np.random.default_rng(0)
    n = 200_000
    conf = rng.random(n)
    dist = rng.uniform(1, 5, n)
    sigma = 0.01 + 0.08 * (1 - conf)
    err = rng.normal(0, sigma)
    t = CAL.calibrate(err, conf, dist, conf_bins=8, dist_bins=2, quantile=0.95,
                      min_bin_samples=200)
    edges = np.asarray(t["conf_edges"])
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows = np.asarray(t["row_abs_err_quantile"])
    # |N(0,σ)| q95 = 1.96 σ, σ varying linearly within the bin: compare with the
    # quantile of the bin's own mixture
    for i, c in enumerate(centres):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        true_q = np.quantile(np.abs(rng.normal(0, sigma[m])), 0.95)
        assert abs(rows[i] - true_q) / true_q < 0.05
    q = CAL.lookup(t, np.array([0.05, 0.5, 0.95]), np.array([2.0, 2.0, 2.0]))
    assert q[0] > q[1] > q[2]
    assert abs(q[2] / (1.96 * (0.01 + 0.08 * 0.05)) - 1) < 0.15
    # a starved cell falls back to its confidence row, a starved row to the global
    t2 = CAL.calibrate(err[:1000], conf[:1000], dist[:1000], conf_bins=8, dist_bins=4,
                       quantile=0.95, min_bin_samples=400)
    q2 = CAL.lookup(t2, np.array([0.5]), np.array([2.0]))
    assert q2[0] == pytest.approx(t2["global_abs_err_quantile"])


def test_keyframe_scales_pool_neighbours():
    samples = {0: {"z_tri": np.full(30, 2.0), "z_rec": np.full(30, 1.0), "conf": np.ones(30)},
               2: {"z_tri": np.full(30, 3.0), "z_rec": np.full(30, 1.0), "conf": np.ones(30)}}
    s, n = DS.keyframe_scales(samples, 4, 40)
    assert s[0] in (2.0, 2.5) and n[0] >= 40
    assert np.isfinite(s).all()
    s2, _ = DS.keyframe_scales(samples, 4, 100)
    assert np.isnan(s2).all()


def test_keyframe_scales_pool_along_the_walk_not_by_index():
    # keyframes 0-3 at chainage 0, 0.3, 0.6, 5.0 m; the last one is far away
    samples = {k: {"z_tri": np.full(60, v), "z_rec": np.full(60, 1.0), "conf": np.ones(60)}
               for k, v in enumerate((1.0, 1.1, 1.2, 3.0))}
    chain = np.array([0.0, 0.3, 0.6, 5.0])
    s, n = DS.keyframe_scales(samples, 4, 10, chainage=chain, window_m=2.0)
    assert s[0] == s[1] == s[2] == 1.1 and n[0] == 180         # the three within ±1 m pooled
    assert s[3] == 3.0 and n[3] == 60                          # the far one alone
    s_idx, _ = DS.keyframe_scales(samples, 4, 10)               # by index: each on its own
    assert s_idx.tolist() == [1.0, 1.1, 1.2, 3.0]


def test_colmap_io_roundtrips(tmp_path):
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = DC.qvec_to_rotmat(q)
        assert np.allclose(DC.qvec_to_rotmat(DC.rotmat_to_qvec(R)), R, atol=1e-10)
    a = rng.random((7, 5)).astype(np.float32)
    DC.write_colmap_array(tmp_path / "d.bin", a)
    assert np.array_equal(DC.read_colmap_array(tmp_path / "d.bin"), a)
    w2c = np.stack([np.linalg.inv(REF), np.linalg.inv(VIEWS[0])])
    DC.write_sparse(tmp_path / "sparse", K, (W, H), [5, 9], w2c,
                    [{"X": np.array([0.0, 0.0, 4.0]), "obs": [(0, (63.5, 47.5)), (1, (40.0, 47.5))]}])
    cams = (tmp_path / "sparse" / "cameras.txt").read_text().split()
    assert cams[1] == "PINHOLE" and float(cams[6]) == pytest.approx(64.0) \
        and float(cams[7]) == pytest.approx(48.0)
    lines = (tmp_path / "sparse" / "images.txt").read_text().splitlines()
    assert lines[0].endswith("000005.png") and lines[2].endswith("000009.png")
    assert lines[1].split() == ["64.0000", "48.0000", "1"]
    DC.write_patch_match_cfg(tmp_path / "stereo", {5: [9]})
    assert (tmp_path / "stereo" / "patch-match.cfg").read_text().split("\n")[:2] == \
        ["000005.png", "000009.png"]


def test_compare_counts_common_pixels(tmp_path):
    d = np.full((10, 10), 2.0, np.float32)
    src = np.full((10, 10), DS.SOURCE_SWEEP, np.uint8)
    src[:, 5:] = DS.DISCARD_NO_SIGNAL
    np.savez(tmp_path / "f.npz", depth=d, source=src)
    c = np.full((10, 10), 2.02, np.float32)
    c[0] = 0
    r = DC.compare(tmp_path / "f.npz", c)
    assert r["tier0_n_common"] == 45
    assert r["tier0_median_rel"] == pytest.approx(abs(2.0 / 2.02 - 1), rel=1e-5)
    assert r["colmap_coverage"] == pytest.approx(0.9)


# ── the whole run on a synthetic session ─────────────────────────────────

def _session(tmp_path, pcfg):
    """Frames, camera, poses, Omega records, F4 tracks and an APPLIED F5 on the scene."""
    from precision.camera import CameraModel, GridMap, save_camera_json
    from precision.refine import REFINE_NAME, RESIDUALS_NAME, WITNESS_FRAMES_NAME, WITNESS_POSES_NAME
    from precision.tracks import TRACKS_NAME
    s = tmp_path / "sess"
    fr, out = s / "frames", s / "output"
    rec = out / "omega_run" / "results_output"
    pdir = out / "precision"
    for d in (fr, rec, pdir):
        d.mkdir(parents=True)
    kf_c2w = [REF] + VIEWS[:3]
    wit_c2w = VIEWS[3:]
    kf = [10, 20, 30, 40]
    wit = [15, 25]
    # a far keyframe that sees another wall (the null's non-covisible images)
    far = cam(40.0, 0.0)
    kf_c2w.append(far)
    kf.append(50)
    gh, gw = H // 2, W // 2
    grid = GridMap("omega", gw, gh, gw, gh, 0, 0, 0, 0, W, H, W, H)
    camm = CameraModel(W, H, (K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0, 0, 0, 0), "refine", 1, grid)
    save_camera_json(out / "camera.json", camm, 1)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 1}))
    omega_units = 0.8                                     # the records' own units
    for f, c in list(zip(kf, kf_c2w)) + list(zip(wit, wit_c2w)):
        img, z = render(c)
        cv2.imwrite(str(fr / f"{f:06d}.png"), np.round(img * 255).astype(np.uint8))
        if f in kf:
            zg = cv2.resize(z, (gw, gh), interpolation=cv2.INTER_AREA)
            zg = zg * (1 + prior_error(zg.shape, 0.04, phase=f)) * omega_units
            np.savez(rec / f"frame_{f}.npz", depth=zg.astype(np.float32),
                     conf=np.full((gh, gw), 2.0, np.float32), pose_c2w=c)
    np.savetxt(out / "camera_poses.txt", np.stack([c.ravel() for c in kf_c2w]))
    (out / "camera_frames.txt").write_text(" ".join(map(str, kf)))
    # the DA3 walk (I3): chainage per keyframe — the sweep pools s_k along it
    cen = np.array([c[:3, 3] for c in kf_c2w])
    chain = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cen, axis=0), axis=1))])
    (s / "intake").mkdir(exist_ok=True)
    (s / "intake" / "walk.json").write_text(json.dumps(
        {"version": 1, "walk_length_m": float(chain[-1]), "n_keyframes": len(kf),
         "chainage": [{"frame": int(f), "chainage_m": float(c)} for f, c in zip(kf, chain)],
         "windows": []}))
    np.savetxt(pdir / WITNESS_POSES_NAME, np.stack([c.ravel() for c in wit_c2w]))
    (pdir / WITNESS_FRAMES_NAME).write_text(" ".join(map(str, wit)))
    (pdir / REFINE_NAME).write_text(json.dumps({"applied": True, "epoch_to": 1}))
    np.savez(pdir / RESIDUALS_NAME, heldout_rms_px=np.full(50, 0.5, np.float32))
    # F4 tracks: wall + box points seen by the first four keyframes
    rng = np.random.default_rng(5)
    Xs = np.concatenate([np.stack([rng.uniform(-1.2, 1.2, 150), rng.uniform(-0.8, 0.8, 150),
                                   np.full(150, WALL_Z)], 1),
                         np.stack([rng.uniform(BOX[0], BOX[1], 80), rng.uniform(BOX[2], BOX[3], 80),
                                   np.full(80, BOX_Z)], 1)])
    ot, of, ouv = [], [], []
    zmaps = {f: render(c)[1] for f, c in zip(kf[:4], kf_c2w[:4])}
    for t, X in enumerate(Xs):
        for f, c in zip(kf[:4], kf_c2w[:4]):
            w2c = np.linalg.inv(c)
            p = K @ (w2c[:3, :3] @ X + w2c[:3, 3])
            u, v = p[:2] / p[2]
            iu, iv = int(round(u)), int(round(v))
            # a track exists only where the point is SEEN (not behind the box)
            if not (0 <= iu < W and 0 <= iv < H) or abs(zmaps[f][iv, iu] - p[2]) > 0.01:
                continue
            ot.append(t)
            of.append(f)
            ouv.append((u, v))
    np.savez(pdir / TRACKS_NAME, obs_track=np.array(ot), obs_frame=np.array(of),
             obs_uv_native=np.array(ouv))
    return s, omega_units


@pytest.fixture(scope="module")
def pcfg():
    from dataclasses import replace
    from precision.config import load_precision_config
    p = load_precision_config()
    return replace(p, depth=replace(p.depth, n_views=3, best_k=2, n_hyp=16, null_frames=2,
                                    min_scale_samples=50, calib_min_bin_samples=50,
                                    calib_samples_per_frame=2000, view_samples=128,
                                    propagation_iters=1, min_consistent_views=2))


def test_run_sweep_end_to_end_on_a_synthetic_session(tmp_path, pcfg):
    s, omega_units = _session(tmp_path, pcfg)
    logs = []
    doc = DS.run_sweep(s, pcfg, log=logs.append, device="cpu")
    out = s / "output"
    rep = json.loads((out / DS.DEPTH_DIRNAME / DS.REPORT_NAME).read_text())
    assert rep["geometry_epoch"] == 1 and rep["camera_epoch"] == 1
    # the measured s_k carried the records' units to the session's metres
    # (the records also carry the prior's own error, −2…+6 %, which s_k absorbs:
    # without the units carried the product would read 0.8)
    assert 1 / 1.06 <= rep["s_k"]["median"] * omega_units <= 1 / 0.98
    assert rep["beta_source"] == "landmarks"
    assert not (out / DS.DEPTH_DIRNAME / DS.WORK_DIRNAME).exists()
    with np.load(out / DS.DEPTH_DIRNAME / "frame_10.npz") as z:
        depth, src, ncc = z["depth"], z["source"], z["ncc"]
        assert z["normal"].dtype == np.float32 and z["residual_rel"].dtype == np.float32
    _, gt = render(REF)
    ok, box, wall, flat = _regions(gt)
    t0 = (src == DS.SOURCE_SWEEP) & ok
    assert t0[ok & (box | wall)].mean() > 0.9
    assert np.median(np.abs(depth[t0] / gt[t0] - 1)) < 0.01
    assert not np.any(src[flat & ok] == DS.SOURCE_SWEEP)
    assert doc["counts"]["tier0"] > 0
    # maximum coverage (production): a prior pixel is tier 0 or tier 1 unless contradicted,
    # under the confidence floor or excluded — nothing is left 'no_signal' / 'inconsistent',
    # and the retired independence rule writes nothing
    assert rep["tier1_rule"] == "max_coverage" and rep["contradiction"] is True
    assert rep["counts"]["no_signal"] == 0 and rep["counts"]["inconsistent"] == 0
    assert rep["counts"]["prior_not_independent"] == 0
    assert set(rep["s_k_by_frame"]) == {str(f) for f in (10, 20, 30, 40, 50)}   # every keyframe's s_k
    assert any("MAXIMUM COVERAGE" in str(m) for m in logs)
    # the untextured patch has no ZNCC signal: its pixels fall back to Omega's prior
    assert (src[flat & ok] == DS.SOURCE_PRIOR_FILL).mean() > 0.5
    assert np.all(depth[src == DS.SOURCE_PRIOR_FILL] > 0)
    cal = CAL.load_calibration(out, {"geometry_epoch": 1, "camera_epoch": 1}, reference="tier0")
    assert cal is not None and "abs_err_quantile" in cal["models"]["omega"]
    # re-running reads β from the tier-0 calibration of this epoch
    doc2 = DS.run_sweep(s, pcfg, log=logs.append, device="cpu")
    assert doc2["beta_source"] == "tier0"


def test_identical_sessions_give_bit_identical_depth(tmp_path, pcfg):
    """Two sessions built from the same inputs, swept one after the other (torch's RNG
    and allocation history differ between the two): every array of every frame, and
    the calibration, are the same bits."""
    import torch
    mode_before = torch.are_deterministic_algorithms_enabled()
    runs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        s, _ = _session(tmp_path / name, pcfg)
        torch.rand(1000 + len(runs))                     # someone used the RNG in between
        DS.run_sweep(s, pcfg, log=lambda *a: None, device="cpu")
        out = s / "output"
        arrays = {}
        for p in sorted((out / DS.DEPTH_DIRNAME).glob("frame_*.npz")):
            with np.load(p) as z:
                arrays.update({(p.name, k): (z[k].dtype.str, z[k].tobytes()) for k in z.files})
        cal = json.loads((out / "precision" / CAL.CALIBRATION_NAME).read_text())["models"]
        runs.append((arrays, cal))
    assert runs[0][0].keys() == runs[1][0].keys() and len(runs[0][0]) >= 6 * 4
    for k in runs[0][0]:
        assert runs[0][0][k] == runs[1][0][k], k
    assert runs[0][1] == runs[1][1]
    # the mode is scoped to the run: whatever the process had before, it has after
    assert torch.are_deterministic_algorithms_enabled() == mode_before


def test_the_sweep_core_is_bit_identical_and_nanmedian_is_its_own(scene):
    ref, gt, views = scene
    z0 = (gt * (1 + prior_error(gt.shape))).astype(np.float32)
    beta = np.full(gt.shape, 0.15, np.float32)
    cs = DS.contrast_sigma(ref, np.ones_like(ref, bool))
    from precision.tracks import deterministic_torch
    with deterministic_torch(0):
        a = DS.sweep_frame(_frame(ref, views), z0, beta, 16, 2, cs)
        b = DS.sweep_frame(_frame(ref, views), z0, beta, 16, 2, cs)
    for k in a:
        assert a[k].tobytes() == b[k].tobytes(), k
    # the deterministic replacement of torch.nanmedian(·, 0) picks the same element
    rng = np.random.default_rng(1)
    R = torch.as_tensor(rng.random((5, 400)))
    R[torch.as_tensor(rng.random((5, 400)) < 0.4)] = float("nan")
    ref_med = torch.nanmedian(R, 0).values
    got = DS._nanmedian0(R)
    assert torch.equal(torch.isnan(got), torch.isnan(ref_med))
    assert torch.equal(got[~torch.isnan(got)], ref_med[~torch.isnan(ref_med)])


def test_run_sweep_refuses_an_unapplied_refinement(tmp_path, pcfg):
    s, _ = _session(tmp_path, pcfg)
    p = s / "output" / "precision" / "refine.json"
    p.write_text(json.dumps({"applied": False, "epoch_to": None}))
    with pytest.raises(DS.DepthSweepError, match="no-apply"):
        DS.run_sweep(s, pcfg, device="cpu")
    p.write_text(json.dumps({"applied": True, "epoch_to": 0}))
    with pytest.raises(DS.DepthSweepError, match="epoch"):
        DS.run_sweep(s, pcfg, device="cpu")


def test_depth_config_loads_from_the_repo_yaml():
    from precision.config import load_precision_config
    d = load_precision_config().depth
    assert d.patch_px % 2 == 1 and d.best_k <= d.n_views and d.prior_fill in ("keep", "drop")
    assert Path(d.colmap.binary).name == "colmap"
