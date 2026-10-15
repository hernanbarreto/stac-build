"""Phase 7 of the evaluation battery: Pareto frontier and profiles on synthetic tables with a
known answer."""
import json

from evaluation import pareto as P


def _rec(p95, cost, cov=100.0, status="finished", median=None):
    return {"status": status, "summary": {"abs_error_m": {"p95": p95, "median": median if median is not None else p95 / 2}},
            "coverage": {"test_ends_reconstructed_pct": cov}, "cost": {"totals": {"cost_usd": cost, "gpu_hours": cost / 2}},
            "seconds": cost * 60}


def test_pareto_front_and_profiles_on_a_known_table(tmp_path):
    scenes = {
        "s1": {"ref": _rec(0.02, 10.0), "cheap": _rec(0.08, 2.0), "mid": _rec(0.04, 5.0), "bad": _rec(0.09, 6.0),
               "low_cov": _rec(0.01, 1.0, cov=50.0), "broken": {"status": "failed", "reason": "boom"}},
        "s2": {"ref": _rec(0.03, 12.0), "cheap": _rec(0.07, 2.5), "mid": _rec(0.05, 5.5), "bad": _rec(0.10, 6.5),
               "low_cov": _rec(0.012, 1.2, cov=95.0), "broken": {"status": "failed", "reason": "boom"}},
    }
    out = P.run_phase7(tmp_path, scenes, [0.02, 0.05, 0.10, 0.005], 0.9, plots=True)
    rows = {r["variant"]: r for r in out["rows"]}
    assert rows["ref"]["p95_worst_m"] == 0.03 and rows["ref"]["cost_usd_total"] == 22.0
    assert rows["broken"]["status"] == "failed" and rows["broken"]["p95_worst_m"] is None
    # front: low_cov (cheapest, best error) dominates everything; bad is dominated by mid; cheap dominated by low_cov
    assert out["pareto_by_cost_usd"] == ["low_cov"]
    prof = {p["tolerance_m"]: p for p in out["profiles"]}
    assert prof[0.05]["variant"] == "mid" and "ref" in prof[0.05]["candidates"] and "low_cov" not in prof[0.05]["candidates"]
    assert prof[0.10]["variant"] == "cheap"
    assert prof[0.02]["status"] == "none qualifies" and prof[0.02]["variant"] is None     # ref's worst scene is 0.03
    assert prof[0.005]["status"] == "none qualifies"
    for f in ("pareto.json", "profiles.json", "pareto.csv", "pareto.png"):
        assert (tmp_path / f).exists(), f
    assert json.loads((tmp_path / "profiles.json").read_text())["min_test_coverage"] == 0.9


def test_front_keeps_ties_and_handles_missing_costs():
    rows = [{"variant": "a", "p95_worst_m": 0.01, "cost_usd_total": 5.0}, {"variant": "b", "p95_worst_m": 0.05, "cost_usd_total": 1.0},
            {"variant": "c", "p95_worst_m": 0.03, "cost_usd_total": 3.0}, {"variant": "d", "p95_worst_m": 0.03, "cost_usd_total": 3.0},
            {"variant": "e", "p95_worst_m": 0.02, "cost_usd_total": None}]
    assert P.pareto_front(rows) == ["a", "b", "c", "d"]
    assert rows[4]["pareto"] is False
