"""Phase 3f (audit 2026-10-23): rolling shutter from declared to corrected. test_precision_refine's
observable scene, the observations made with a ROLLING shutter: row v of keyframe i exposed at
the keyframe's pose moved by (v − c_y)/H · τ_f/Δf of its neighbours' motion (τ_f = 0.9 frame
periods, keyframes 10 frames apart). The global-shutter ladder leaves the systematic residual;
the stage measures τ_f, rectifies, re-solves and is taken by the user's rule."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pycolmap = pytest.importorskip("pycolmap")
if not hasattr(pycolmap.Camera, "create_from_model_name"):
    pytest.skip("pycolmap 4 (the mapanything env) is required", allow_module_level=True)

from precision import refine as R                                 # noqa: E402
from precision import rolling_shutter as RS                        # noqa: E402
import test_precision_refine as T                                  # noqa: E402

TAU_F = 0.9            # readout in frame periods
STEP = 10              # frames between keyframes


def _observe_rs(X, c2w, params, noise, seed, frames, tau_f):
    """test_precision_refine._observe with the rolling shutter: the pixel is found by fixed-point
    iteration — the row decides the time, the time decides the pose, the pose decides the row."""
    rng = np.random.default_rng(seed)
    w2c = np.linalg.inv(c2w)
    motion = RS.keyframe_motion(w2c, np.asarray(frames, np.float64))
    omega, dC, df = motion
    H = T.WH[1]; cy = float(params[3])
    track, frame, uv = [], [], []
    for i, Wi in enumerate(w2c):
        p = R.project(X, Wi, params)
        for _ in range(4):
            s = (p[:, 1] - cy) / H / df[i]
            p_new = np.array([R.project(X[j:j + 1], RS.pose_at(Wi, omega[i], dC[i], float(s[j]) * tau_f), params)[0]
                              for j in range(len(X))])
            p = p_new
        z = (X @ Wi[:3, :3].T + Wi[:3, 3])[:, 2]
        ok = (z > 0) & (p[:, 0] >= 0) & (p[:, 0] < T.WH[0]) & (p[:, 1] >= 0) & (p[:, 1] < T.WH[1]) & np.isfinite(p).all(1)
        for j in np.flatnonzero(ok):
            track.append(j); frame.append(frames[i]); uv.append(p[j] + rng.normal(0, noise, 2))
    return np.array(track), np.array(frame), np.array(uv)


def test_keyframe_motion_and_pose_at_are_consistent():
    X, c2w = T._scene(seed=2)
    w2c = np.linalg.inv(c2w)
    omega, dC, df = RS.keyframe_motion(w2c, np.arange(len(c2w)) * STEP)
    assert np.allclose(df[1:-1], 2 * STEP) and df[0] == STEP
    # s = 1 moves keyframe i by its neighbours' whole motion: the centre lands on C_{i+1} − C_{i−1} + C_i
    i = 5
    P1 = RS.pose_at(w2c[i], omega[i], dC[i], 1.0)
    C1 = -P1[:3, :3].T @ P1[:3, 3]
    assert np.allclose(C1, c2w[i, :3, 3] + (c2w[i + 1, :3, 3] - c2w[i - 1, :3, 3]))
    assert np.allclose(RS.pose_at(w2c[i], omega[i], dC[i], 0.0), w2c[i])


def test_the_readout_is_measured_and_the_rectified_solve_is_taken():
    X, c2w = T._scene(seed=4, n_kf=16)
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = [STEP * k for k in range(len(c2w))]
    track, frame, uv = _observe_rs(X, c2w, gt, 0.3, 9, kf, TAU_F)
    split = T._split(track)
    cfg = replace(T.CFG, permutations=300, rolling_shutter="judged", rs_passes=2)
    core = R.refine_core(np.linalg.inv(T._perturb(c2w, 0.1, 0.01, 3)), gt, T.WH, track, frame, uv, split, kf, 0.01,
                         cfg, T.SOLVER, error_factor=T.FAC, log=lambda *a: None)
    rs = core["rolling_shutter"]
    m = rs["measured_on_global_shutter_residuals"]
    # the residuals' own reading sees only the fraction the solve did not absorb; the SEARCH finds
    # the injected readout (grid 0.5, refined twice: 0.125 resolution)
    assert m["significant"] and 0.0 < m["tau_frames"] < TAU_F, m
    assert abs(rs["tau_frames"] - TAU_F) <= 0.125 + 1e-9, (rs["tau_frames"], rs["search"])
    assert rs["spearman_before"] is not None and rs["spearman_before"] > 0.1
    assert rs["taken"] and core["best"].name == "RS", rs
    gs_med = float(np.median(list(core["held"]["R0" if "R0" in core["held"] else list(core["held"])[1]].values())))
    rs_med = float(np.median(list(core["held"]["RS"].values())))
    assert rs_med < 0.7 * gs_med, (gs_med, rs_med)
    assert rs["spearman_after"] is not None and rs["spearman_after"] < rs["spearman_before"]
    assert core["rungs"]["RS"]["readout_frames"] == rs["tau_frames"]
    assert rs["heldout_median_px_rolling_shutter"] < rs["heldout_median_px_global_shutter"]


def test_without_rolling_shutter_nothing_is_measured_as_significant():
    X, c2w = T._scene(seed=5, n_kf=14)
    gt = [500.0, 500.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
    kf = [STEP * k for k in range(len(c2w))]
    track, frame, uv = T._observe(X, c2w, gt, 0.3, 6, kf)
    cfg = replace(T.CFG, permutations=300, rolling_shutter="judged")
    core = R.refine_core(np.linalg.inv(c2w), gt, T.WH, track, frame, uv, T._split(track), kf, 0.01, cfg, T.SOLVER,
                         error_factor=T.FAC, log=lambda *a: None)
    rs = core["rolling_shutter"]
    assert not rs["taken"] and core["best"].name != "RS"
    m = rs["measured_on_global_shutter_residuals"]
    if m["significant"]:
        # a chance significance must still lose the search or fail the held-out judge
        assert rs.get("tau_frames", 0.0) == 0.0 or rs["verdict_vs_best"]["improves"] is False
    else:
        assert abs(m["tau_frames"]) < 0.3


def test_a_negative_readout_is_searched_and_taken():
    """Regression (pccr_audit 2026-10-31): the residuals read the readout with its sign, the search
    walks the grid on that side — and every candidate there was short-circuited to the global
    shutter (`key <= 0`), so a negative readout produced seven candidates with one identical held-out
    median and 'not corrected'. The same scene as the positive test, the rows read the other way."""
    X, c2w = T._scene(seed=4, n_kf=16)
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = [STEP * k for k in range(len(c2w))]
    track, frame, uv = _observe_rs(X, c2w, gt, 0.3, 9, kf, -TAU_F)
    split = T._split(track)
    cfg = replace(T.CFG, permutations=300, rolling_shutter="judged", rs_passes=2)
    core = R.refine_core(np.linalg.inv(T._perturb(c2w, 0.1, 0.01, 3)), gt, T.WH, track, frame, uv, split, kf, 0.01,
                         cfg, T.SOLVER, error_factor=T.FAC, log=lambda *a: None)
    rs = core["rolling_shutter"]
    m = rs["measured_on_global_shutter_residuals"]
    assert m["significant"] and -TAU_F < m["tau_frames"] < 0.0, m
    searched = [e for e in rs["search"] if e["tau_frames"] < 0.0]
    assert len(searched) >= 4, rs["search"]
    # the candidates were SOLVED: their held-out medians are not the global shutter's
    gs = rs["heldout_median_px_global_shutter"]
    assert any(e["heldout_median_px"] != gs for e in searched), rs["search"]
    assert abs(rs["tau_frames"] + TAU_F) <= 0.125 + 1e-9, (rs["tau_frames"], rs["search"])
    assert rs["taken"] and core["best"].name == "RS", rs
    assert rs["heldout_median_px_rolling_shutter"] < gs
    assert core["rungs"]["RS"]["readout_frames"] == rs["tau_frames"] < 0.0
