"""Phase 7 — the cost–error frontier and the profiles. Over every variant of a run: P95 of the
absolute error against the cost (money and minutes), the Pareto frontier (lower error and lower
cost), and per tolerance the cheapest variant whose P95 is under the tolerance in EVERY scene and
whose test-end coverage reaches ``evaluation.profiles.min_test_coverage`` — "none qualifies" when
none does. A computation, not a recommendation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.dimension_error import write_csv, write_json

AGG_COLUMNS = ["variant", "status", "n_scenes", "n_scenes_finished", "p95_worst_m", "p95_mean_m", "median_mean_m",
               "cost_usd_total", "gpu_hours_total", "wall_minutes_total", "test_coverage_min_pct", "pareto"]


def aggregate(scene_results: Dict[str, Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """``scene_results[scene][variant]`` = the variant record → one row per variant."""
    variants = sorted({v for s in scene_results.values() for v in s})
    rows = []
    for v in variants:
        recs = [scene_results[s][v] for s in scene_results if v in scene_results[s]]
        fin = [r for r in recs if r.get("status") == "finished"
               and (r.get("summary") or {}).get("abs_error_m", {}).get("p95") is not None]
        p95 = [r["summary"]["abs_error_m"]["p95"] for r in fin]
        med = [r["summary"]["abs_error_m"]["median"] for r in fin]
        cov = [(r.get("coverage") or {}).get("test_ends_reconstructed_pct") for r in fin]
        cov = [c for c in cov if c is not None]
        cost = [((r.get("cost") or {}).get("totals") or {}).get("cost_usd") for r in recs]
        gpuh = [((r.get("cost") or {}).get("totals") or {}).get("gpu_hours") for r in recs]
        wall = [r.get("seconds") for r in recs]
        rows.append({"variant": v,
                     "status": ("finished" if fin and len(fin) == len(recs) else
                                "partial" if fin else (recs[0].get("status") if recs else "missing")),
                     "n_scenes": len(recs), "n_scenes_finished": len(fin),
                     "p95_worst_m": max(p95) if p95 else None, "p95_mean_m": (sum(p95) / len(p95)) if p95 else None,
                     "median_mean_m": (sum(med) / len(med)) if med else None,
                     "cost_usd_total": sum(c for c in cost if c is not None) if any(c is not None for c in cost) else None,
                     "gpu_hours_total": sum(c for c in gpuh if c is not None) if any(c is not None for c in gpuh) else None,
                     "wall_minutes_total": (sum(w for w in wall if w is not None) / 60.0) if any(w is not None for w in wall) else None,
                     "test_coverage_min_pct": min(cov) if cov else None, "pareto": False,
                     "reasons": [r.get("reason") for r in recs if r.get("status") != "finished" and r.get("reason")]})
    return rows


def pareto_front(rows: List[Dict[str, Any]], error_key: str = "p95_worst_m", cost_key: str = "cost_usd_total"
                 ) -> List[str]:
    """Variants no other variant beats on both axes (lower error AND lower cost, one strictly)."""
    cand = [r for r in rows if r.get(error_key) is not None and r.get(cost_key) is not None]
    front = []
    for r in cand:
        dominated = any((o[error_key] <= r[error_key] and o[cost_key] <= r[cost_key])
                        and (o[error_key] < r[error_key] or o[cost_key] < r[cost_key]) for o in cand if o is not r)
        if not dominated:
            front.append(r["variant"])
    for r in rows:
        r["pareto"] = r["variant"] in front
    return sorted(front)


def profiles(rows: List[Dict[str, Any]], tolerances: Sequence[float], min_coverage: float, n_scenes: int,
             cost_key: str = "cost_usd_total") -> List[Dict[str, Any]]:
    """Per tolerance: the cheapest variant with P95 ≤ tolerance in every scene (its worst P95) and
    the test-end coverage ≥ min_coverage in every scene."""
    out = []
    for tol in tolerances:
        ok = [r for r in rows if r.get("p95_worst_m") is not None and r["p95_worst_m"] <= tol
              and r["n_scenes_finished"] == n_scenes and r.get("test_coverage_min_pct") is not None
              and r["test_coverage_min_pct"] >= 100.0 * min_coverage and r.get(cost_key) is not None]
        if not ok:
            out.append({"tolerance_m": tol, "variant": None, "status": "none qualifies",
                        "min_test_coverage": min_coverage, "candidates": []})
            continue
        best = min(ok, key=lambda r: (r[cost_key], r["p95_worst_m"]))
        out.append({"tolerance_m": tol, "variant": best["variant"], "status": "ok", "min_test_coverage": min_coverage,
                    "cost_usd_total": best[cost_key], "p95_worst_m": best["p95_worst_m"],
                    "candidates": sorted(r["variant"] for r in ok)})
    return out


def run_phase7(run_dir: Path, scene_results: Dict[str, Dict[str, Dict[str, Any]]], tolerances: Sequence[float],
               min_coverage: float, *, plots: bool = True) -> Dict[str, Any]:
    rows = aggregate(scene_results)
    front = pareto_front(rows)
    front_minutes = pareto_front(rows, cost_key="wall_minutes_total")
    prof = profiles(rows, tolerances, min_coverage, len(scene_results))
    doc = {"rows": rows, "pareto_by_cost_usd": front, "pareto_by_wall_minutes": front_minutes,
           "axes": {"error": "worst P95 of the absolute test error over the scenes (m)",
                    "cost": "sum over the scenes of GPU hours x price (USD) / wall minutes"},
           "n_scenes": len(scene_results), "scenes": sorted(scene_results)}
    write_json(run_dir / "pareto.json", doc)
    write_json(run_dir / "profiles.json", {"profiles": prof, "tolerances_m": list(tolerances),
                                           "min_test_coverage": min_coverage, "n_scenes": len(scene_results),
                                           "rule": "cheapest variant with worst-scene P95 <= tolerance and test-end "
                                                   "coverage >= min_test_coverage in every scene; a computation, "
                                                   "not a recommendation"})
    write_csv(run_dir / "pareto.csv", rows, AGG_COLUMNS)
    if plots:
        from evaluation.plots import plot_scatter
        pts = [r for r in rows if r.get("p95_worst_m") is not None and r.get("cost_usd_total") is not None]
        if pts:
            plot_scatter(run_dir / "pareto.png", [r["cost_usd_total"] for r in pts], [r["p95_worst_m"] for r in pts],
                         [r["variant"] for r in pts], "cost (USD, GPU hours x price, all scenes)",
                         "worst P95 of the absolute test error (m)", "error vs cost per variant (diamonds: Pareto front)",
                         [r["pareto"] for r in pts])
    return doc | {"profiles": prof}
