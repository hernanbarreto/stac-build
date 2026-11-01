"""The floor solve's torch path (correction.floor_torch, `correction.floor.device` cpu | cuda) against the
numpy path of record on the synthetic sessions of the floor tests: the same levels, the same inliers, the
same bootstrap σ, the same verdicts, the same rigid motion — and the kernels one by one: numpy's linear
percentile, the per-keyframe medians, the k-d tree's nearest distance, the Poisson bootstrap with the same
draws, the Tukey IRLS plane."""
import os

import numpy as np
import pytest
from scipy.spatial import cKDTree

from correction import floor as FL
from tests.test_correction_floor_chunk import _cfg as _cfg_chunk, _session as _chunk_session
from tests.test_correction_floor_session import _cfg as _cfg_session, _scene

torch = pytest.importorskip("torch")
from correction import floor_torch as FT  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")      # the card when a GPU run asks for it


def _on(dev, fn):
    """``fn()`` with the floor solve on ``dev``; the suite's override (cpu) restored after."""
    FL.use_device(dev)
    try:
        return fn()
    finally:
        FL.use_device("cpu")


def _fit_sans_iterations(fit):
    """The IRLS record without its iteration count: the 1e-9 rad convergence test fires when the dot product
    of two successive normals ROUNDS to 1.0 exactly (arccos(1 − 1e-16) = 1.4e-8 rad), so the iteration it
    fires at depends on the order of the float64 sums — the planes agree to 1e-9 all the same."""
    return {k: v for k, v in fit.items() if k != "iterations"}


def _same_candidates(a, b):
    assert len(a) == len(b)
    for qa, qb in zip(a, b):
        assert qa["rank"] == qb["rank"] and qa["real"] == qb["real"]
        assert qa["support"] == qb["support"]
        assert np.allclose(qa["normal"], qb["normal"], atol=1e-9, rtol=0)
        assert np.allclose(qa["centroid"], qb["centroid"], atol=1e-9, rtol=0)
        assert qa["points"].shape == qb["points"].shape
        assert np.allclose(qa["points"], qb["points"], atol=1e-9, rtol=0)
        assert np.isclose(qa["sigma_tilt_deg"], qb["sigma_tilt_deg"], rtol=1e-6, atol=1e-12)
        assert np.isclose(qa["sigma_height_m"], qb["sigma_height_m"], rtol=1e-6, atol=1e-12)
        assert _fit_sans_iterations(qa["fit"]) == _fit_sans_iterations(qb["fit"])
        if qa["judge"] is None:
            assert qb["judge"] is None
        else:
            assert qa["judge"]["improves"] == qb["judge"]["improves"]
            assert np.isclose(qa["judge"]["median_delta"], qb["judge"]["median_delta"], rtol=1e-6, atol=1e-12)


def test_the_session_floor_on_the_device_equals_numpy(tmp_path):
    s, _ = _scene(tmp_path, tilt_deg=3.0, offset=0.12, frame_rot_deg=25.0, platform=True)
    cfg = _cfg_session()
    sol_np = _on("numpy", lambda: FL.solve_floor_session(s, cfg, log=lambda m: None))
    sol_t = _on(DEV, lambda: FL.solve_floor_session(s, cfg, log=lambda m: None))
    assert np.allclose(sol_np["R_kf"], sol_t["R_kf"], atol=1e-9, rtol=0)
    assert np.allclose(sol_np["t_kf"], sol_t["t_kf"], atol=1e-9, rtol=0)
    ra, rb = sol_np["model_params"], sol_t["model_params"]
    for k in ("role", "chosen_rank", "under_cameras", "inliers", "cameras_under", "tilt_applied", "height_applied",
              "n_support", "support_instances"):
        assert ra[k] == rb[k], k
    assert len(ra["levels"]) == len(rb["levels"]) == 2
    for la, lb in zip(ra["levels"], rb["levels"]):
        for k in ("rank", "real", "support", "cameras_under", "footprint_support", "chosen"):
            assert la[k] == lb[k], k
        assert _fit_sans_iterations(la["fit"]) == _fit_sans_iterations(lb["fit"])
    # the raw candidates: the same planes, inliers and errors (the records above are rounded)
    cams = np.asarray(s.poses, np.float64)[:, :3, 3]
    rows, _ = FL.support_rows(s, cfg.floor.support_labels)
    fac, conf = FL._judge_of(cfg)
    rep = FL._session_repeatability(s, cfg)
    lv_np = _on("numpy", lambda: FL.session_floor_levels(s.xyz, s.ks, cams, rows, cfg.floor, rep, fac, conf))
    lv_t = _on(DEV, lambda: FL.session_floor_levels(s.xyz, s.ks, cams, rows, cfg.floor, rep, fac, conf))
    assert lv_np["chosen"] == lv_t["chosen"]
    assert np.allclose(lv_np["up0"], lv_t["up0"], atol=1e-9)
    _same_candidates(lv_np["cands"], lv_t["cands"])


def test_the_chunk_floor_on_the_device_equals_numpy(tmp_path):
    s = _chunk_session(tmp_path)
    cfg = _cfg_chunk()
    sol_np = _on("numpy", lambda: FL.solve_floor_by_chunk(s, cfg, np.random.default_rng(0), log=lambda m: None))
    sol_t = _on(DEV, lambda: FL.solve_floor_by_chunk(s, cfg, np.random.default_rng(0), log=lambda m: None))
    assert np.allclose(sol_np["R_kf"], sol_t["R_kf"], atol=1e-9, rtol=0)
    assert np.allclose(sol_np["t_kf"], sol_t["t_kf"], atol=1e-9, rtol=0)
    assert [r.get("role") for r in sol_np["per_kf_report"]] == [r.get("role") for r in sol_t["per_kf_report"]]
    cams = np.asarray(s.poses, np.float64)[:, :3, 3]
    ck = FL._omega_chunk_of_keyframes(s)
    fac, conf = FL._judge_of(cfg)
    rep = FL._session_repeatability(s, cfg)
    lv_np = _on("numpy", lambda: FL.chunk_floor_levels(s.xyz, s.ks, cams, ck, cfg.floor, rep, fac, conf))
    lv_t = _on(DEV, lambda: FL.chunk_floor_levels(s.xyz, s.ks, cams, ck, cfg.floor, rep, fac, conf))
    assert lv_np["chunks"] == lv_t["chunks"]
    for c in lv_np["chunks"]:
        _same_candidates(lv_np["cands_of"][c], lv_t["cands_of"][c])
        assert [q["footprint_support"] for q in lv_np["cands_of"][c]] == \
               [q["footprint_support"] for q in lv_t["cands_of"][c]]


# ── the kernels ───────────────────────────────────────────────────────────────────────────────────────

def test_percentile_is_numpys_linear_percentile_bit_for_bit():
    rng = np.random.default_rng(1)
    for n in (3, 4, 7, 20, 21, 1000, 1001, 12345):
        v = rng.normal(0, 1, n)
        for pct in (0.0, 5.0, 12.5, 50.0, 95.0, 100.0):
            assert FT.percentile_t(torch.as_tensor(v, device=DEV), pct) == np.percentile(v, pct), (n, pct)


def test_segment_medians_are_numpys_per_key():
    rng = np.random.default_rng(2)
    ks = rng.integers(0, 17, 5000)
    v = rng.normal(0, 1, 5000)
    uniq, med = FT.segment_medians_t(torch.as_tensor(v, device=DEV), torch.as_tensor(ks, device=DEV))
    assert np.array_equal(uniq, np.unique(ks))
    want = np.array([np.median(v[ks == k]) for k in np.unique(ks)])
    assert np.array_equal(med, want)


def test_nearest_xz_is_the_kd_trees_distance():
    rng = np.random.default_rng(3)
    A = rng.uniform(-5, 5, (300, 3))
    B = rng.uniform(-5, 5, (20000, 3))
    d = FT.nearest_xz(A, B, DEV)
    want, _ = cKDTree(B[:, [0, 2]]).query(A[:, [0, 2]], k=1)
    assert np.allclose(d, want, atol=1e-12, rtol=0)
    d2 = FT.nearest_xz(B, A, DEV)
    want2, _ = cKDTree(A[:, [0, 2]]).query(B[:, [0, 2]], k=1)
    assert np.allclose(d2, want2, atol=1e-12, rtol=0)


def test_bootstrap_plane_draws_the_same_resamples():
    rng = np.random.default_rng(4)
    n = 7000
    P = np.column_stack([rng.uniform(-3, 3, n), rng.normal(0.0, 0.004, n), rng.uniform(-3, 3, n)])
    P = P @ np.array([[np.cos(0.05), np.sin(0.05), 0], [-np.sin(0.05), np.cos(0.05), 0], [0, 0, 1]]) + [0.3, 0.12, -0.4]
    s_np = FL._bootstrap_plane(P, 123, 400)
    s_t = FT.bootstrap_plane_t(torch.as_tensor(P, device=DEV), 123, 400, FL._POISSON1_CDF, FL.IRLS_TOL, block=96)
    assert np.isclose(s_np[0], s_t[0], rtol=1e-7, atol=0) and np.isclose(s_np[1], s_t[1], rtol=1e-7, atol=0)
    assert s_np[0] > 0 and s_np[1] > 0


def test_irls_plane_converges_to_the_same_plane():
    rng = np.random.default_rng(5)
    n = 20000
    P = np.column_stack([rng.uniform(-4, 4, n), rng.normal(0.0, 0.01, n), rng.uniform(-4, 4, n)])
    P[: n // 10, 1] += rng.uniform(0.1, 0.6, n // 10)             # furniture in the band
    th = np.radians(2.0)
    P = P @ np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]]) + [0, 0.2, 0]
    n_np, c_np, inl_np, rec_np = FL._irls_plane(P, 0.5, 0.03)
    n_t, c_t, inl_t, rec_t = FT.irls_plane_t(torch.as_tensor(P, device=DEV), 0.5, 0.03, FL.IRLS_MAX_ITER, FL.IRLS_TOL)
    assert np.allclose(n_np, n_t, atol=1e-9, rtol=0) and np.allclose(c_np, c_t, atol=1e-9, rtol=0)
    assert np.array_equal(inl_np, inl_t.cpu().numpy())
    assert rec_np["stages"] == rec_t["stages"] and rec_np["converged"] == rec_t["converged"]
    assert inl_np.sum() > 0.8 * n
