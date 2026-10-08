"""F6's judged bend (docs/plan_determinismo.md points 48, 49, 51, 52, 53, 60 — USER 2026-10-07):
c1 / c2 enter only when significant against their measured fit error, c0 is verified on the
keyframe's own held-out rows by the user's rule (else the pooled fit of its neighbours), the window
is the smoothest within error of the best, SAM3 masks enter only when stamped for this
reconstruction, the ONE confidence floor's arithmetic is pinned, every bar's margins are recorded.
CPU, synthetic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import depth_on_f5 as D                              # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])
QUIET = lambda *a: None  # noqa: E731


def _fac() -> float:
    from config import cfg as raw_cfg
    from reconstruction.loops.config import improvement_error_factor
    return float(improvement_error_factor(raw_cfg))


FAC = _fac()


def _rows(n, k0, k1=0.0, k2=0.0, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, W, n); v = rng.uniform(0, H, n)
    A = D.design(u, v, W, H)
    return A, A @ np.array([k0, k1, k2]) + rng.normal(0, noise, n), u, v


# ── point 48: the terms ────────────────────────────────────────────────────

def test_insignificant_tilt_terms_are_dropped_exactly_and_a_real_tilt_is_kept():
    A, r, _, _ = _rows(300, 0.9)
    c, info = D.significant_bend(A, r, 1.345, 10, FAC)
    assert c[1] == 0.0 and c[2] == 0.0 and info["kept"] == {"c1": False, "c2": False}
    assert info["margins"]["c1"] < 0 and info["margins"]["c2"] < 0 and abs(c[0] - 0.9) < 1e-3
    assert all(np.isfinite(info["sigma_full"])) and info["sigma"][0] > 0
    A, r, _, _ = _rows(300, 0.9, k1=0.2)
    c, info = D.significant_bend(A, r, 1.345, 10, FAC)
    assert info["kept"]["c1"] is True and info["margins"]["c1"] > 0 and abs(c[1] - 0.2) < 2e-2
    assert info["kept"]["c2"] is False and c[2] == 0.0
    # the coefficients of the full model are epoch 7's fit, bit for bit
    full, _ = D.irls_huber_fit(A, r, 1.345, 10)
    assert np.array_equal(full, D.irls_huber(A, r, 1.345, 10))


def test_sigma_is_the_measured_fit_error_it_shrinks_with_rows_and_grows_with_noise():
    _, s_small = D.irls_huber_fit(*_rows(50, 1.0, noise=1e-2)[:2], 1.345, 10)
    _, s_big = D.irls_huber_fit(*_rows(2000, 1.0, noise=1e-2)[:2], 1.345, 10)
    _, s_quiet = D.irls_huber_fit(*_rows(2000, 1.0, noise=1e-3)[:2], 1.345, 10)
    assert s_big[0] < s_small[0] and s_quiet[0] < s_big[0]
    # a degenerate design (one row repeated) has no measurable error: nothing is significant
    A = np.tile(D.design(np.array([3.0]), np.array([4.0]), W, H), (30, 1))
    c, info = D.significant_bend(A, np.full(30, 1.2), 1.345, 10, FAC)
    assert info["kept"] == {"c1": False, "c2": False}


# ── point 49: the window ───────────────────────────────────────────────────

def test_window_is_the_smoothest_within_error_factor_times_the_best_standard_error():
    rng = np.random.default_rng(1)
    # per window the per-keyframe error arrays; the best (±0) has a measurable standard error
    errs = {w: {i: np.abs(rng.normal(0.028 + 0.004 * k, 0.01, 40)) for i, k in enumerate(np.linspace(-1, 1, 20))}
            for w in (0, 1, 2)}
    score = {0: 0.0285, 1: 0.0288, 2: 0.060}
    wb, rule = D.choose_window(score, errs, FAC, 300, 0)
    se = D.cluster_bootstrap_se(list(errs[0].values()), 300, 0)
    assert se > 0 and rule["heldout_best_se"] == se and rule["bar"] == 0.0285 + FAC * se
    assert wb == 1 and rule["window_best"] == 0 and rule["curve"]["2"]["within"] is False
    assert rule["margin"] >= 0 and rule["curve"]["1"]["margin_to_bar"] == rule["margin"]
    # with no measurable error the strict minimum stays
    one = {w: {0: np.array([0.03])} for w in (0, 1)}
    assert D.choose_window({0: 0.03, 1: 0.031}, one, FAC, 50, 0)[0] == 0
    # the same input gives the same window and bar (fixed seed)
    assert D.choose_window(score, errs, FAC, 300, 0) == (wb, rule)


# ── point 48: c0 verified per keyframe, the pooled fit otherwise ──────────

def _keyframes(n_rows_bad=24):
    """Five keyframes at k = 0.9 with many landmark rows; keyframe 2 holds a handful of rows that
    say k = 2.0 (a degenerate own fit) while its held-out rows say 0.9 (pccr 720 / 723)."""
    rows, held = {}, {}
    for i in range(5):
        if i == 2:
            A, r, _, _ = _rows(n_rows_bad, 2.0, noise=1e-3, seed=10 + i)
        else:
            A, r, _, _ = _rows(400, 0.9, noise=1e-3, seed=10 + i)
        rows[i] = (A, r)
        rng = np.random.default_rng(100 + i)
        u = rng.uniform(0, W, 60); v = rng.uniform(0, H, 60)
        zo = rng.uniform(1.0, 5.0, 60)
        z_lm = zo * 0.9 * (1 + rng.normal(0, 1e-3, 60))
        held[i] = (u, v, z_lm, zo)
    return rows, held


def test_a_degenerate_own_fit_fails_its_heldout_rows_and_takes_the_pooled_fit():
    rows, held = _keyframes()
    windows = (0, 1, 2)
    coefs, infos = {}, {}
    for w in windows:
        coefs[w], infos[w] = D.bend_coefficients(rows, 5, w, 20, 1.345, 10, FAC)
    assert abs(coefs[0][2][0] - 2.0) < 0.05                             # the own fit is degenerate
    final, prov = D.verify_keyframes(windows, 0, coefs, infos, held, 5, W, H, FAC, 0.95, 400, 0, 20)
    assert prov[0]["status"] == "verified" and prov[0]["window"] == 0 and abs(final[0][0] - 0.9) < 1e-2
    assert prov[2]["status"] == "verified" and prov[2]["window"] == 1 and abs(final[2][0] - 0.9) < 2e-2
    t = prov[2]["trials"]
    assert t[0]["window"] == 0 and t[0]["verdict"]["improves"] is False
    assert t[1]["window"] == 1 and t[1]["verdict"]["improves"] is True
    assert t[1]["verdict"]["error"] == infos[1][2]["sigma"][0] and t[1]["verdict"]["min_judges"] == 5
    assert prov[2]["heldout_rows_A"] == 60 and prov[2]["fit"]["window"] == 1


def test_too_few_heldout_rows_is_unverifiable_and_takes_the_neighbours_pooled_fit():
    rows, held = _keyframes()
    held[1] = tuple(x[:3] for x in held[1])                                # 3 rows: fewer than 5 judges
    windows = (0, 1, 2)
    coefs, infos = {}, {}
    for w in windows:
        coefs[w], infos[w] = D.bend_coefficients(rows, 5, w, 20, 1.345, 10, FAC)
    final, prov = D.verify_keyframes(windows, 0, coefs, infos, held, 5, W, H, FAC, 0.95, 400, 0, 20)
    assert prov[1]["status"] == "unverifiable_pooled" and prov[1]["window"] == 1
    assert "3 held-out row(s)" in prov[1]["reason"] and np.array_equal(final[1], coefs[1][1])
    # when the chosen window is the widest, an unverifiable keyframe keeps its own fit, declared
    final2, prov2 = D.verify_keyframes(windows, 2, coefs, infos, held, 5, W, H, FAC, 0.95, 400, 0, 20)
    assert prov2[1]["status"] == "unverifiable_own" and prov2[1]["window"] == 2


def test_no_window_verifying_leaves_the_widest_pooled_fit_declared_and_no_rows_is_identity():
    rows, held = _keyframes()
    # keyframe 0's held-out rows contradict every fit (they say k = 1.6): nothing verifies
    u, v, z_lm, zo = held[0]
    held[0] = (u, v, zo * 1.6, zo)
    windows = (0, 1)
    coefs, infos = {}, {}
    for w in windows:
        coefs[w], infos[w] = D.bend_coefficients(rows, 5, w, 20, 1.345, 10, FAC)
    final, prov = D.verify_keyframes(windows, 0, coefs, infos, held, 5, W, H, FAC, 0.95, 400, 0, 20)
    assert prov[0]["status"] == "unverified_pooled" and prov[0]["window"] == 1
    assert len(prov[0]["trials"]) == 2 and all(t["verdict"]["improves"] is False for t in prov[0]["trials"])
    # a keyframe with no landmark rows in any window: Omega's depth as it is (epoch 7)
    rows2 = {i: rows[i] for i in (0, 1)}
    rows2[4] = (D.design(np.zeros(0), np.zeros(0), W, H), np.zeros(0))
    c2, i2 = {}, {}
    for w in (0,):
        c2[w], i2[w] = D.bend_coefficients(rows2, 5, w, 20, 1.345, 10, FAC)
    f2, p2 = D.verify_keyframes((0,), 0, c2, i2, held, 5, W, H, FAC, 0.95, 400, 0, 20)
    assert p2[4]["status"] == "identity" and np.array_equal(f2[4], D.IDENTITY)


def test_heldout_rows_split_into_the_selection_and_the_report_halves():
    obs = np.c_[np.arange(10.0) * 5 + 2, np.full(10, 7.0), np.arange(10.0) + 1]
    zmap = np.full((H, W), 2.0, np.float32)
    uA = D.heldout_rows(obs, zmap, 0.05, 0)[0]
    uB = D.heldout_rows(obs, zmap, 0.05, 1)[0]
    assert list(uA) == list(obs[0::2, 0]) and list(uB) == list(obs[1::2, 0])
    assert len(D.heldout_rows(obs, zmap, 0.05, None)[0]) == 10
    assert len(D.heldout_rows(obs, np.zeros((H, W), np.float32), 0.05, None)[0]) == 0   # under min depth


# ── point 52: THE ONE confidence floor, pinned ─────────────────────────────

def test_the_confidence_floor_is_min_plus_norm_times_the_span_and_the_weight_its_ramp():
    v = np.array([0.2, 0.5, 1.0, 4.0])
    thr, cmax = D.confidence_floor(v, 0.10)
    assert thr == 0.2 + 0.10 * (4.0 - 0.2) and cmax == 4.0
    assert D.confidence_floor(np.array([3.0]), 0.5) == (3.0, 3.0)
    w = D.confidence_weight(v, thr, cmax, np.array([True, True, True, True]))
    assert np.allclose(w, np.clip((v - thr) / (cmax - thr), 0, 1)) and w[0] == 0.0 and w[-1] == 1.0
    with pytest.raises(D.DepthOnF5Error):
        D.confidence_floor(np.zeros(0), 0.1)
    # the key it reads is the pipeline's ONE floor, shared with the viewer's slider arithmetic
    import inspect
    from config import cfg as raw_cfg
    assert float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"]) == 0.10
    src = inspect.getsource(D.compute)
    assert 'raw_cfg["reconstruction"]["simple"]["conf_min_norm"]' in src and "confidence_floor(" in src


# ── point 51: masks only when stamped for this reconstruction ─────────────

def _session_with_reconstruction(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    np.savez(out / "omega_run" / "results_output" / "frame_1.npz", depth=np.ones((4, 4), np.float32))
    (out / "camera_frames.txt").write_text("1\n")
    return out


def test_masks_without_a_stamp_or_from_another_reconstruction_are_ignored(tmp_path):
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    out = _session_with_reconstruction(tmp_path)
    rid = reconstruction_id(out)
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8))
    masks = np.load(out / "seg_masks.npz")
    taken, why = D.masks_stamp_check(out, masks)
    assert not taken and RECONSTRUCTION_ID_KEY in why
    msgs = []
    assert D.mask_labels(out, H, W, log=msgs.append) is None
    assert any("IGNORED" in m and "point 51" in m for m in msgs)
    # another reconstruction's stamp: refused with the reason
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: "0" * 64})
    taken, why = D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))
    assert not taken and "another reconstruction" in why
    # this reconstruction's stamp in the store itself, or in segmentation.json next to it: taken
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: rid})
    assert D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))[0] is True
    np.savez(out / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8))
    (out / "segmentation.json").write_text(json.dumps({RECONSTRUCTION_ID_KEY: rid}))
    assert D.masks_stamp_check(out, np.load(out / "seg_masks.npz"))[0] is True
    # a session with no reconstruction to name takes no masks at all
    bare = tmp_path / "bare" / "output"
    bare.mkdir(parents=True)
    np.savez(bare / "seg_masks.npz", f0_o0=np.ones((H, W), np.uint8), **{RECONSTRUCTION_ID_KEY: rid})
    assert D.masks_stamp_check(bare, np.load(bare / "seg_masks.npz"))[0] is False


# ── point 53: the vote's bar and its margins ───────────────────────────────

def test_the_vote_reports_tau_and_every_frames_margins_to_it():
    cams = {}
    for f, x in zip((0, 1, 2), (-0.2, 0.0, 0.2)):
        c = np.eye(4); c[0, 3] = x; cams[f] = c
    order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    dep[1][10:20, 30:40] = 1.5
    ok = {f: np.ones((H, W), bool) for f in order}
    bars = {}
    _, tau, tot = D.edge_keeping_vote(order, dep, ok, {f: ok[f].copy() for f in order}, K, cams, None,
                                      (-1, 1), 75.0, 2, log=QUIET, bars_out=bars)
    assert all(isinstance(v, (int, float, np.integer, np.floating)) for v in tot.values())   # numeric for its readers
    assert bars["tau"] == tau and bars["tau_quantile"] == 75.0
    pf = bars["tau_margin_rel"]["per_frame"]
    assert set(pf) == {"0", "1", "2"} and pf["1"]["n"] > 0
    assert set(pf["1"]) == {"n", "p05", "p25", "p50", "p75", "p95", "share_beyond"}
    assert 0.0 <= pf["1"]["share_beyond"] <= 1.0 and bars["tau_margin_rel"]["median_of_frame_medians"] is not None


def test_margin_quantiles_report_the_share_beyond_the_bar():
    from precision.mono_detail import margin_quantiles
    q = margin_quantiles(np.array([1.0, 0.5, -0.5, np.nan]))
    assert q["n"] == 3 and abs(q["share_beyond"] - 1 / 3) < 1e-12 and q["p50"] == 0.5
    assert margin_quantiles(np.array([])) is None


# ── point 45 / 56: the camera file is exact ───────────────────────────────

def test_the_camera_file_is_float64_round_trip_exact(tmp_path):
    params = [391.123456789012345, 388.7, 234.8, 414.1, 0, 0, 0, 0]
    p = D.camera_travels(tmp_path, params, 2, log=QUIET)
    rows = np.loadtxt(p).reshape(-1, 4)
    assert rows.shape == (2, 4) and rows[0, 0] == params[0] and rows[1, 1] == 388.7
