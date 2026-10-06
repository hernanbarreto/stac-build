"""The judge of the in-run pose graph is a held-out share of the loop closures
(USER 2026-10-06): deterministic split by chunk pair, every pair keeps a fitted edge,
residuals measured the graph's way, applied only when the held-out closures improve
beyond their own noise; the solver runs on the card."""

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))

from loop_utils.loop_judge import loop_residuals_m, split_loop_edges  # noqa: E402
from loop_utils.lie import se3_exp  # noqa: E402
from loop_utils.metric_lock import heldout_change  # noqa: E402


def _edges(n, owner_of):
    """n loop edges i<->j with i in [0, n*3): chunk pairs from owner_of."""
    out = []
    for k in range(n):
        i, j = 3 * k, 3 * k + 1
        out.append({"i": i, "j": j, "Z": np.eye(4).tolist(), "bridge": k})
    return out


def test_split_is_deterministic_stratified_and_every_pair_keeps_a_fitted_edge():
    owner = np.zeros(60, int)
    owner[30:] = 1                                      # frames 0-29 chunk 0, 30-59 chunk 1
    edges = _edges(20, owner)                           # i = 0,3,...,57: pairs (0,0),(1,1) mixed
    edges[5]["j"] = 50                                  # one cross-chunk edge (0,1) — alone in its pair
    fit, judge, rep = split_loop_edges(edges, owner, 0.25, 4)
    assert rep["every_kth"] == 4 and rep["n_fit"] + rep["n_judge"] == 20
    assert 3 <= rep["n_judge"] <= 5, rep
    fit2, judge2, _ = split_loop_edges(list(reversed(edges)), owner, 0.25, 4)
    assert [e["bridge"] for e in judge2] == [e["bridge"] for e in judge], "deterministic, order-free"
    # every chunk pair keeps at least one FIT edge
    def pair(e):
        a, b = int(owner[e["i"]]), int(owner[e["j"]])
        return (min(a, b), max(a, b))
    assert {pair(e) for e in edges} == {pair(e) for e in fit}
    assert all(pair(e) in {pair(f) for f in fit} for e in judge)
    assert not ({e["bridge"] for e in fit} & {e["bridge"] for e in judge})


def test_too_few_edges_hold_nothing_out_and_say_so():
    owner = np.zeros(30, int)
    fit, judge, rep = split_loop_edges(_edges(3, owner), owner, 0.25, 4)
    assert len(fit) == 3 and judge == [] and "nothing can be held out" in rep["reason"]
    fit, judge, rep = split_loop_edges(_edges(8, owner), owner, 0.0, 4)
    assert len(fit) == 8 and judge == [] and "loop_holdout_frac is 0" in rep["reason"]


def test_loop_residuals_are_the_graphs_own_and_a_correction_closes_them():
    rng = np.random.default_rng(0)
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
    # the judge: held-out closures that all improve → IMPROVES; untouched → no verdict
    before = list(0.2 + 0.1 * rng.random(8))
    after = [b * 0.5 for b in before]
    assert heldout_change(before, after, 0.95)["improves"] is True
    v = heldout_change(before, before, 0.95)
    assert not v["improves"] and not v["worsens"]


def test_production_config_declares_the_judge_and_the_fork_uses_it():
    from reconstruction.loops.config import load_loops_config
    raw = yaml.safe_load((ROOT / "server" / "config.yaml").read_text())
    g = load_loops_config(raw).graph
    assert 0.0 < g.loop_holdout_frac <= 0.5 and g.loop_holdout_min_edges >= 2
    src = (ROOT / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()
    assert "split_loop_edges(" in src and 'cfg_req(gcfg, "loop_holdout_frac", "graph")' in src
    assert "refused = (not converged) or (not ok_judge)" in src
    assert 'PoseGraph(T0, gcfg, device=_dev)' in src and 'torch.cuda.is_available()' in src
    i_judge = src.index("refused = (not converged) or (not ok_judge)")
    i_fallback = src.index("refused = (not converged) or (not ok_held)")
    assert i_judge < i_fallback
