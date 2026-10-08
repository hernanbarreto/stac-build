"""F5's judge (docs/plan_determinismo.md points 46, 55, 59, 60, 61, 62 — USER 2026-10-07): the
rungs are warm-started and nested-cost-checked, every rung carries its continuation solve and its
solver error, a more complex rung enters only by THE USER'S RULE with keyframe clusters, held-out
tracks that do not round-trip are dropped and counted, R3 is measured and never applied.
Needs pycolmap 4 (the mapanything env); the synthetic scene is test_precision_refine's."""

import os
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

from precision import refine as R                                # noqa: E402
import test_precision_refine as T                                # noqa: E402

CFG, SOLVER, WH, FAC = T.CFG, T.SOLVER, T.WH, T.FAC
QUIET = lambda *a: None  # noqa: E731


def _problem(seed=0, n_kf=14):
    X, c2w = T._scene(seed=seed, n_kf=n_kf)
    gt = [500.0, 500.0, 319.5, 239.5, -0.05, 0.01, 0.0, 0.0]
    kf = list(range(len(c2w)))
    track, frame, uv = T._observe(X, c2w, gt, 0.3, seed + 1, kf)
    init = [515.0, 515.0, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0]
    w2c0 = np.linalg.inv(T._perturb(c2w, 0.2, 0.02, seed + 2))
    return w2c0, init, track, frame, uv, T._split(track), kf


def test_rungs_are_warm_started_nested_and_carry_their_continuation():
    w2c0, init, track, frame, uv, split, kf = _problem()
    core = R.refine_core(w2c0, init, WH, track, frame, uv, split, kf, 0.01, CFG, SOLVER,
                         error_factor=FAC, log=QUIET)
    rungs = core["rungs"]
    costs = [rungs[n]["ceres"]["final_cost"] for n in ("R0", "R1", "R2")]
    # point 61: R1 starts at R0's solution and R2 at R1's — the cost never climbs along the ladder
    assert costs[1] <= costs[0] and costs[2] <= costs[1], costs
    chk = rungs["R1"]["nested_cost_check"]
    assert chk["parent"] == "R0" and chk["parent_final_cost"] == costs[0] and chk["final_cost"] == costs[1]
    assert chk["passed"] is True and chk["margin"] >= 0 and chk["cost_tolerance"] >= 0
    assert rungs["R2"]["nested_cost_check"]["parent"] == "R1"
    # a warm start lands: R1 begins far below the gauge start's cost (the adjuster's prior alignment
    # perturbs it a little before solving, so it is not exactly R0's final cost)
    assert rungs["R1"]["ceres"]["initial_cost"] < 0.5 * rungs["R0"]["ceres"]["initial_cost"]
    # point 59: a continuation solve from the end state, its held-out change = the solver error
    for n in ("R0", "R1", "R2"):
        c = rungs[n]["continuation"]
        assert np.isfinite(c["error"]) and c["error"] >= 0.0 and c["n_tracks"] > 0
        # the continuation re-converges to the same optimum: the adjuster's prior alignment perturbs
        # its start, so its final cost sits within the solver's own resolution (measured 3e-7 relative)
        assert abs(c["final_cost"] - rungs[n]["ceres"]["final_cost"]) <= 1e-5 * rungs[n]["ceres"]["final_cost"]
        assert core["solver_errors"][n]["error"] == c["error"]
    # point 46: the verdict is decide_change's, clustered by keyframe, with the solver error as the bar
    v = rungs["R2"]["verdict_vs_best"]
    assert v["clustered"] is True and v["n_judges"] <= len(kf) and v["min_judges"] == 5
    assert v["error"] == max(core["solver_errors"]["R1"]["error"], core["solver_errors"]["R2"]["error"]) \
        or not rungs["R1"]["taken"]
    assert v["error_factor"] == FAC and set(v["failed"]) <= {"judges", "significance", "error"}
    assert v["n_tracks"] == v["n_obs"] and v["median_best_px"] is not None
    # a real lens still enters: significant, enough keyframes, far beyond the solver error
    assert core["best"].name == "R2" and v["improves"] and v["error_margin"] > 0
    # the landmarks handed on are the ones the best rung solved, not X0
    assert len(core["X"]) == len(core["best"].X) and core["X"] is core["best"].X
    assert core["X0"] is not core["X"]
    assert core["ladder"] == ["R0", "R1", "R2"] and core["ladder_restricted_by"] is None
    assert core["dropped"] == {"fit_init": {"tracks": 0, "observations": 0},
                               "heldout_init": {"tracks": 0, "observations": 0}}


def test_judge_rungs_is_the_users_rule():
    """Significant AND >= 5 keyframes AND median improvement >= factor x the solver error."""
    rng = np.random.default_rng(0)
    n = 400
    before = {t: 1.0 + 0.01 * rng.standard_normal() for t in range(n)}
    after = {t: before[t] - 0.001 for t in range(n)}                 # every track 0.001 px better
    clusters = {t: t % 10 for t in range(n)}                          # 10 keyframes
    cfg = replace(CFG, permutations=400)
    v = R.judge_rungs(before, after, clusters, error=0.0001, error_factor=2.0, cfg=cfg)
    assert v["improves"] and v["significant"] and v["enough_judges"] and v["beyond_error"]
    # the same change against a solver error of 0.001: not beyond 2 x the error
    v2 = R.judge_rungs(before, after, clusters, error=0.001, error_factor=2.0, cfg=cfg)
    assert not v2["improves"] and v2["failed"] == ["error"] and v2["error_margin"] < 0
    # four keyframes cannot testify at 0.95, however clear the change
    v3 = R.judge_rungs(before, after, {t: t % 4 for t in range(n)}, error=0.0001, error_factor=2.0, cfg=cfg)
    assert not v3["improves"] and "judges" in v3["failed"] and v3["n_judges"] == 4
    # only the tracks valid in BOTH states are compared (point 60)
    v4 = R.judge_rungs(before, {t: after[t] for t in range(0, n, 2)}, clusters, 0.0001, 2.0, cfg)
    assert v4["n_tracks"] == n // 2


def test_solver_error_is_the_median_absolute_paired_change():
    end = {1: 0.5, 2: 0.6, 3: 0.7, 9: 1.0}
    cont = {1: 0.5, 2: 0.59, 3: 0.72, 4: 2.0}
    e = R.solver_error(end, cont)
    assert e["n_tracks"] == 3 and abs(e["error"] - 0.01) < 1e-12
    assert abs(e["median_signed_change"] - 0.0) < 1e-12
    assert R.solver_error({}, cont)["n_tracks"] == 0 and np.isnan(R.solver_error({}, cont)["error"])


def test_a_rung_that_ends_above_its_parent_is_not_taken_and_the_next_starts_from_the_parent(monkeypatch):
    w2c0, init, track, frame, uv, split, kf = _problem(seed=3)
    real = R.run_rung

    def inflated(name, *a, **kw):
        r = real(name, *a, **kw)
        if name == "R1" and not getattr(inflated, "done", False):
            # the FIRST R1 solve (not its continuation) reports a cost above its parent's
            inflated.done = True
            r.final_cost = r.final_cost * 10.0
        return r
    monkeypatch.setattr(R, "run_rung", inflated)
    core = R.refine_core(w2c0, init, WH, track, frame, uv, split, kf, 0.01, CFG, SOLVER,
                         error_factor=FAC, log=QUIET)
    r1 = core["rungs"]["R1"]
    assert r1["taken"] is False and "above R0" in r1["rejected"] and r1["nested_cost_check"]["passed"] is False
    assert core["rungs"]["R2"]["nested_cost_check"]["parent"] == "R0"
    # the bar is the PARENT's cost resolution (its continuation change), never the rung's own
    assert r1["nested_cost_check"]["cost_tolerance"] == abs(core["rungs"]["R0"]["continuation"]["final_cost"]
                                                           - core["rungs"]["R0"]["ceres"]["final_cost"])
    assert core["best"].name in ("R0", "R2")


def test_heldout_tracks_that_do_not_round_trip_are_dropped_and_counted():
    """Point 60: one held-out observation that cannot be undistorted (NaN never converges) drops
    its track from that rung's held-out set — counted, the ladder goes on."""
    w2c0, init, track, frame, uv, split, kf = _problem(seed=5)
    held_tracks = np.unique(track[split == 1])
    bad = held_tracks[0]
    uv = uv.copy()
    uv[np.flatnonzero(track == bad)[0]] = [np.nan, np.nan]
    core = R.refine_core(w2c0, init, WH, track, frame, uv, split, kf, 0.01, CFG, SOLVER,
                         error_factor=FAC, log=QUIET)
    assert core["dropped"]["heldout_init"]["tracks"] == 1
    for n in ("R0", "R1", "R2"):
        assert core["rungs"][n]["heldout_dropped"]["tracks"] == 1
        assert bad not in core["held"][n]
    assert core["best"].name in ("R1", "R2")


def test_r3_is_measured_from_retriangulated_landmarks_and_never_applied(monkeypatch):
    w2c0, init, track, frame, uv, split, kf = _problem(seed=6)
    cfg = replace(CFG, focal_block_frames=7)                         # two temporal blocks
    monkeypatch.setattr(R, "_systematic_in_time", lambda pf, b, c: {"systematic": True, "p": 0.0, "stat": 1.0})
    seen = {}
    real = R.run_rung

    def spy(name, w2c_start, params_start, wh, fit_groups, X_start, sigmas, cfg_, **kw):
        if name == "R3" and "R3" not in seen:
            seen["R3"] = {"X_start": {t: np.array(x) for t, x in X_start.items()},
                          "prior": np.array(kw["prior_w2c"]), "start": np.array(w2c_start)}
        return real(name, w2c_start, params_start, wh, fit_groups, X_start, sigmas, cfg_, **kw)
    monkeypatch.setattr(R, "run_rung", spy)
    core = R.refine_core(w2c0, init, WH, track, frame, uv, split, kf, 0.01, cfg, SOLVER,
                         error_factor=FAC, log=QUIET)
    r3 = core["rungs"]["R3"]
    assert r3["tried"] and r3["taken"] is False and r3["applied"] is False and "point 62" in r3["policy"]
    assert core["best"].name != "R3" and len(r3["params_by_block"]) == 2
    assert r3["start_landmarks_retriangulated"] > 0 and "would_improve" in r3
    assert np.isfinite(r3["continuation"]["error"])
    best = core["best"]
    # point 55: R3 started from the best rung's poses with landmarks triangulated under ITS camera,
    # and its priors are the gauge poses R0–R2 were held to
    assert np.array_equal(seen["R3"]["start"], best.w2c)
    assert np.array_equal(seen["R3"]["prior"], w2c0)
    X3 = seen["R3"]["X_start"]
    X0 = core["X0"]
    common = sorted(set(X3) & set(X0))
    assert common and any(np.linalg.norm(X3[t] - X0[t]) > 1e-9 for t in common)


def test_the_ladder_can_be_restricted_only_by_the_declared_diagnostic_variable(monkeypatch):
    w2c0, init, track, frame, uv, split, kf = _problem(seed=7)
    monkeypatch.setenv(R.RUNGS_ENV, "R0,R1")
    msgs = []
    core = R.refine_core(w2c0, init, WH, track, frame, uv, split, kf, 0.01, CFG, SOLVER,
                         error_factor=FAC, log=msgs.append)
    assert core["ladder"] == ["R0", "R1"] and core["ladder_restricted_by"] == R.RUNGS_ENV
    assert "R2" not in core["rungs"] and any("DIAGNOSTIC" in m for m in msgs)
    monkeypatch.setenv(R.RUNGS_ENV, "R1")
    with pytest.raises(R.RefineError, match="prefix"):
        R.rung_ladder(QUIET)


def test_witness_poses_are_written_round_trip_exact():
    import inspect
    src = inspect.getsource(R.run_refine)
    assert "write_poses_exact" in src and ".10g" not in src
    assert "improvement_error_factor" in src and "TIMING_NAME" in src
