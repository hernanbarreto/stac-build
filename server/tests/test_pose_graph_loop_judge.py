"""The judge of the in-run pose graph (USER 2026-10-07, docs/plan_determinismo.md points 1-2):
EVERY loop closure judges once — the graph is solved n times leaving one closure out, and
that closure's residual is measured at the chain and at the solution that never saw it. The
graph applies only by THE USER'S RULE (metric_lock.decide_change): significant at 95 %,
>= min_judge_closures(0.95) = 5 judges, median improvement >= 2 x the largest bridge σ.
Fewer than 5 closures: declared BEFORE solving, nothing solved. The old every-k-th split by
chunk pair (which changed with the edge count and never reached 5 judges on pccr) is gone."""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "server" / "tests"))
sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))

pytest.importorskip("torch")

from loop_utils import loop_judge as LJ  # noqa: E402
from loop_utils.loop_judge import (build_keyframe_graph, judge_leave_one_out,  # noqa: E402
                                   loop_residuals_m, min_judge_closures)
from loop_utils.lie import se3_exp  # noqa: E402,F401
from synth_metric import fork_graph_cfg  # noqa: E402

SRC = (ROOT / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()


def _yaw(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _circle_walk(n=60, radius=3.0, drift_m_per_step=0.01):
    """True poses on a circle (the walk comes back to its start) and the CHAIN: the same
    walk with an error that accumulates along it (each step 1 cm sideways)."""
    T_true = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        a = 2.0 * np.pi * k / n
        T_true[k, :3, :3] = _yaw(a + np.pi / 2)
        T_true[k, :3, 3] = [radius * np.cos(a), radius * np.sin(a), 0.0]
    T0 = T_true.copy()
    for k in range(1, n):
        step = np.linalg.inv(T_true[k - 1]) @ T_true[k]
        step[:3, 3] += [0.0, drift_m_per_step, 0.0]
        T0[k] = T0[k - 1] @ step
    return T_true, T0


def _closures(T_true, pairs, sigma_m, rng=None, noise_m=0.0):
    out = []
    for b, (i, j) in enumerate(pairs):
        Z = np.linalg.inv(T_true[i]) @ T_true[j]
        if rng is not None and noise_m > 0:
            Z = Z.copy()
            Z[:3, 3] += rng.normal(0.0, noise_m, 3)
        out.append({"i": i, "j": j, "Z": Z, "sigma_m": float(sigma_m), "sigma_deg": 0.5,
                    "bridge": b})
    return out


PAIRS = [(0, 50), (2, 52), (5, 55), (8, 58), (10, 45), (3, 40)]


def _solver(T0, n_calls=None, never_converge=()):
    owner = np.zeros(len(T0), int)
    gcfg = fork_graph_cfg()

    def solve(rest):
        if n_calls is not None:
            n_calls.append(sorted(int(e["bridge"]) for e in rest))
        pg, _ = build_keyframe_graph(T0, owner, {0: 0.02}, {0: 0.2}, {}, gcfg, rest, "cpu")
        pg.solve(log=lambda *a, **k: None)
        left_out = set(range(len(PAIRS))) - {int(e["bridge"]) for e in rest}
        conv = bool(pg.report["converged"]) and not (left_out & set(never_converge))
        return pg.corrections(), conv
    return solve


def test_the_judge_needs_enough_closures_to_tell_a_correction_from_luck():
    assert min_judge_closures(0.95) == 5          # 0.5**5 = 0.031 < 0.05 ; 0.5**4 = 0.0625
    assert min_judge_closures(0.99) == 7
    assert min_judge_closures(0.5) == 1


def test_fewer_closures_than_judges_is_declared_before_solving():
    T_true, T0 = _circle_walk()
    calls = []
    for n in (0, 1, 4):
        res = judge_leave_one_out(_closures(T_true, PAIRS[:n], 0.01), T0, _solver(T0, calls),
                                  error_factor=2.0, confidence=0.95, log=lambda *a: None)
        assert res["declared_before_solving"] and not res["solved"] and not res["improves"]
        assert res["n_closures"] == n and res["min_judges"] == 5 and res["decision"] is None
        assert ("declared before solving" in res["reason"]) if n else ("no verified" in res["reason"])
    assert calls == [], "nothing may be solved when the graph cannot be judged"


def test_every_closure_judges_once_and_a_real_closure_is_applied():
    T_true, T0 = _circle_walk()
    edges = _closures(T_true, PAIRS, 0.01, np.random.default_rng(3), noise_m=0.002)
    calls = []
    res = judge_leave_one_out(edges, T0, _solver(T0, calls), error_factor=2.0,
                              confidence=0.95, log=lambda *a: None)
    # n judges = n closures: each solve leaves exactly ONE closure out, each closure once
    assert len(calls) == len(PAIRS)
    assert [sorted(set(range(6)) - set(c)) for c in calls] == [[q] for q in range(6)]
    assert res["n_judges"] == 6 and res["n_discarded"] == 0 and res["solved"]
    for r in res["per_edge"]:
        assert r["judges"] and r["after_m"] < r["before_m"]
    d = res["decision"]
    # THE USER'S RULE, all three at once, with every margin recorded
    assert res["improves"] and d["significant"] and d["enough_judges"] and d["beyond_error"]
    assert res["error_m"] == pytest.approx(0.01) and d["required_delta"] == pytest.approx(0.02)
    assert d["median_delta"] >= d["required_delta"] and d["ci_low"] > 0
    for k in ("ci_margin", "judges_margin", "error_margin"):
        assert k in d
    # deterministic: the same edges give the same judges and the same verdict, to the bit
    res2 = judge_leave_one_out(edges, T0, _solver(T0), error_factor=2.0, confidence=0.95,
                               log=lambda *a: None)
    assert res2 == res


def test_an_improvement_within_twice_the_largest_bridge_sigma_is_not_applied():
    """Condition (c): the same closures, but ONE bridge measured with a σ so large that
    twice it exceeds what the closures improve — significant, enough judges, NOT applied."""
    T_true, T0 = _circle_walk()
    edges = _closures(T_true, PAIRS, 0.01)
    edges[2]["sigma_m"] = 2.0                       # the LARGEST σ is the error
    res = judge_leave_one_out(edges, T0, _solver(T0), error_factor=2.0, confidence=0.95,
                              log=lambda *a: None)
    d = res["decision"]
    assert res["error_m"] == pytest.approx(2.0)
    assert d["enough_judges"] and not d["beyond_error"] and not res["improves"]
    assert "error" in d["failed"] and d["error_margin"] < 0


def test_a_leave_one_out_solve_that_did_not_converge_cannot_testify():
    T_true, T0 = _circle_walk()
    edges = _closures(T_true, PAIRS, 0.01)
    res = judge_leave_one_out(edges, T0, _solver(T0, never_converge=(0, 1)), error_factor=2.0,
                              confidence=0.95, log=lambda *a: None)
    assert res["n_discarded"] == 2 and res["n_judges"] == 4
    assert [r["judges"] for r in res["per_edge"]] == [False, False, True, True, True, True]
    assert all("discarded" in r for r in res["per_edge"][:2])
    # 4 judges < 5: the rule refuses by condition (b)
    assert not res["improves"] and "judges" in res["decision"]["failed"]


def test_a_closure_without_a_measured_sigma_is_refused():
    T_true, T0 = _circle_walk()
    edges = _closures(T_true, PAIRS, 0.01)
    edges[0]["sigma_m"] = float("nan")
    with pytest.raises(ValueError, match="measured σ"):
        judge_leave_one_out(edges, T0, _solver(T0), error_factor=2.0, confidence=0.95)


def test_loop_residuals_are_the_graphs_own_and_a_correction_closes_them():
    T0 = np.tile(np.eye(4), (4, 1, 1))
    for k in range(4):
        T0[k, :3, 3] = [k * 1.0, 0.0, 0.0]
    Z = np.linalg.inv(T0[0]) @ T0[3]                    # exact closure 0<->3 as the chain says
    e = {"i": 0, "j": 3, "Z": Z.tolist(), "bridge": 0}
    assert loop_residuals_m([e], T0) == [0.0]
    drift = np.eye(4); drift[:3, 3] = [0.0, 0.0, 0.30]  # the chain drifted 30 cm at frame 3
    T0d = T0.copy(); T0d[3] = T0[3] @ drift
    r = loop_residuals_m([e], T0d)[0]
    assert abs(r - 0.30) < 1e-9, r
    X = np.tile(np.eye(4), (4, 1, 1))
    X[3] = T0[3] @ np.linalg.inv(T0d[3])                # the correction that undoes the drift
    assert loop_residuals_m([e], T0d, X)[0] < 1e-9


def test_the_old_split_is_gone_and_the_fork_uses_leave_one_out():
    assert not hasattr(LJ, "split_loop_edges")
    assert "split_loop_edges" not in SRC and "loop_holdout_frac" not in SRC
    assert "judge_leave_one_out(edges_kf, T0, _solve_without" in SRC
    assert "build_keyframe_graph(" in SRC and "def _stac_build_pose_graph" not in SRC
    # the final graph is solved only AFTER the judge passed
    body = SRC[SRC.index("    def _stac_pose_graph(self):"):]
    body = body[:body.index("    def _stac_authority_record(")]
    assert body.index('if not judged["improves"]:') < body.index("pg.solve(log=print)")
    # fewer than min_judges closures: declared before any pose is even read
    assert body.index("if len(edges_kf) < min_judge:") < body.index("T0 = np.zeros((N, 4, 4))")
    # the solver runs on the card ONLY (plan point 12): no CPU fallback
    dev = SRC[SRC.index("    def _stac_solver_device(self):"):]
    dev = dev[:dev.index("        return \"cuda\"")]
    assert "raise RuntimeError" in dev and '"cpu"' not in dev
    assert 'if torch.cuda.is_available() else "cpu"' not in SRC


def test_production_config_has_the_rule_and_no_holdout_share():
    from reconstruction.loops.config import LoopsConfigError, load_loops_config
    raw = yaml.safe_load((ROOT / "server" / "config.yaml").read_text())
    g = load_loops_config(raw).graph
    assert g.heldout_confidence == 0.95 and g.improvement_error_factor == 2.0
    assert not hasattr(g, "loop_holdout_frac") and not hasattr(g, "loop_holdout_min_edges")
    for k, v in (("loop_holdout_frac", 0.25), ("loop_holdout_min_edges", 4)):
        bad = yaml.safe_load((ROOT / "server" / "config.yaml").read_text())
        bad["correction_graph"]["graph"][k] = v
        with pytest.raises(LoopsConfigError, match=k):
            load_loops_config(bad)
