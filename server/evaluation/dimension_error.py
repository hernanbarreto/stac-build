"""Phase 1 — error per reference distance of one product.

Both ends of every distance are rebuilt in 3-D (:mod:`evaluation.endpoints`); the scale factor
is fitted on the GAUGE distances only (weighted least squares) and applied to everything; the
error is measured on the TEST distances only — never on the distance that calibrated.
Per distance: raw and calibrated measurement, true value, absolute and relative error, the
camera distance of each end and the walk from the nearest gauge distance. Per scene: median,
P95 and max of the absolute and relative errors, and the scale applied. Nothing is dropped: a
distance a product cannot rebuild is a row with ``status = not_reconstructed`` and counts in
the summary.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation import geometry as G
from evaluation.config import EvaluationConfig
from evaluation.endpoints import EndResult, reconstruct_end
from evaluation.product import Product

CSV_COLUMNS = ["id", "role", "tool", "value_m", "sigma_m", "status", "reason", "measured_raw",
               "measured_calibrated_m", "abs_error_m", "rel_error", "end_a_distance_m",
               "end_b_distance_m", "end_a_chainage_m", "end_b_chainage_m",
               "walk_from_nearest_gauge_m", "end_a_method", "end_b_method", "end_a_n_obs",
               "end_b_n_obs", "end_a_spread_m", "end_b_spread_m", "nearest_gauge_id", "note"]


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else x


def reconstruct_distances(doc: Dict[str, Any], product: Product, cfg: EvaluationConfig
                          ) -> List[Dict[str, Any]]:
    """Every distance with its two :class:`EndResult`s (product units, uncalibrated)."""
    ch = product.chainage()
    out = []
    for d in doc.get("distances", []):
        a = reconstruct_end(d["ends"][0], product, cfg.endpoints, ch)
        b = reconstruct_end(d["ends"][1], product, cfg.endpoints, ch)
        raw = (float(np.linalg.norm(a.point - b.point))
               if a.point is not None and b.point is not None else None)
        out.append({"ref": d, "a": a, "b": b, "raw": raw})
    return out


def calibrate(recs: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    """(scale, report) from the gauge distances both ends of which were rebuilt."""
    m, t, s, ids = [], [], [], []
    for r in recs:
        if r["ref"]["role"] == "gauge" and r["raw"] is not None and r["raw"] > 0:
            m.append(r["raw"]); t.append(float(r["ref"]["value_m"]))
            s.append(float(r["ref"]["sigma_m"])); ids.append(r["ref"]["id"])
    n_gauge = sum(1 for r in recs if r["ref"]["role"] == "gauge")
    if not m:
        return None, {"scale_factor": None, "n_gauge": n_gauge, "n_gauge_used": 0,
                      "gauge_ids_used": [], "weighting": None,
                      "reason": "no gauge distance could be rebuilt"}
    scale, how = G.gauge_scale(m, t, s)
    resid = [abs(scale * mi - ti) for mi, ti in zip(m, t)]
    return scale, {"scale_factor": scale, "n_gauge": n_gauge, "n_gauge_used": len(m),
                   "gauge_ids_used": ids, "weighting": how,
                   "gauge_fit_residual_rms_m": float(np.sqrt(np.mean(np.square(resid)))),
                   "gauge_fit_residual_max_m": float(max(resid)),
                   "note": "the gauge residuals describe the fit, they are not the error "
                           "(the error is measured on the test distances only)"}


def rows_of(recs: List[Dict[str, Any]], scale: Optional[float]) -> List[Dict[str, Any]]:
    s = scale if scale is not None else float("nan")
    # chainage of every rebuilt gauge distance (mean of its ends), in calibrated metres
    gauges = []
    for r in recs:
        if r["ref"]["role"] == "gauge" and r["raw"] is not None:
            c = np.nanmean([r["a"].chainage_m, r["b"].chainage_m])
            if np.isfinite(c):
                gauges.append((r["ref"]["id"], c * s))
    rows = []
    for r in recs:
        d, a, b = r["ref"], r["a"], r["b"]
        ok = r["raw"] is not None
        rebuilt_scale = ok and scale is not None
        cal = r["raw"] * s if rebuilt_scale else None
        err = abs(cal - float(d["value_m"])) if (cal is not None and d["role"] == "test") else None
        c = np.nanmean([a.chainage_m, b.chainage_m]) * s if ok else float("nan")
        near = min(gauges, key=lambda g: abs(g[1] - c)) if (gauges and np.isfinite(c)) else None
        if not ok:
            status, reason = "not_reconstructed", "; ".join(
                x for x in (f"end A: {a.reason}" if a.point is None else "",
                            f"end B: {b.reason}" if b.point is None else "") if x)
        elif scale is None:
            status, reason = "uncalibrated", "no gauge distance could be rebuilt"
        else:
            status, reason = "ok", ""
        rows.append({
            "id": d["id"], "role": d["role"], "tool": d["tool"], "value_m": float(d["value_m"]),
            "sigma_m": float(d["sigma_m"]), "status": status, "reason": reason,
            "measured_raw": _f(r["raw"]), "measured_calibrated_m": _f(cal),
            "abs_error_m": _f(err), "rel_error": _f(err / float(d["value_m"])) if err is not None else None,
            "end_a_distance_m": _f(a.distance_m * s) if ok else None,
            "end_b_distance_m": _f(b.distance_m * s) if ok else None,
            "end_a_chainage_m": _f(a.chainage_m * s) if ok else None,
            "end_b_chainage_m": _f(b.chainage_m * s) if ok else None,
            "walk_from_nearest_gauge_m": _f(abs(near[1] - c)) if near else None,
            "end_a_method": a.method, "end_b_method": b.method,
            "end_a_n_obs": a.n_used, "end_b_n_obs": b.n_used,
            "end_a_spread_m": _f(a.spread_rms_m * s) if ok else None,
            "end_b_spread_m": _f(b.spread_rms_m * s) if ok else None,
            "nearest_gauge_id": near[0] if near else None, "note": d.get("note", ""),
        })
    return rows


def summarize(rows: List[Dict[str, Any]], scale_report: Dict[str, Any], product: Product
              ) -> Dict[str, Any]:
    test = [r for r in rows if r["role"] == "test"]
    ok = [r for r in test if r["status"] == "ok"]
    abs_e = [r["abs_error_m"] for r in ok]
    rel_e = [r["rel_error"] for r in ok]
    methods: Dict[str, int] = {}
    for r in rows:
        for k in ("end_a_method", "end_b_method"):
            methods[r[k]] = methods.get(r[k], 0) + 1
    return {
        "product": {"name": product.name, "kind": product.kind, "n_keyframes": len(product.frames),
                    "n_points": int(len(product.xyz))},
        "scale": scale_report,
        "n_distances": len(rows), "n_test": len(test), "n_test_evaluated": len(ok),
        "n_test_not_reconstructed": sum(1 for r in test if r["status"] == "not_reconstructed"),
        "n_test_uncalibrated": sum(1 for r in test if r["status"] == "uncalibrated"),
        "end_methods": methods,
        "abs_error_m": {"median": G.percentile(abs_e, 50), "p95": G.percentile(abs_e, 95),
                        "max": (max(abs_e) if abs_e else None), "n": len(abs_e)},
        "rel_error": {"median": G.percentile(rel_e, 50), "p95": G.percentile(rel_e, 95),
                      "max": (max(rel_e) if rel_e else None), "n": len(rel_e)},
        "p95_method": "numpy.percentile, linear interpolation",
        "provenance": "tool_measured",
    }


def evaluate(doc: Dict[str, Any], product: Product, cfg: EvaluationConfig
             ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """(rows, summary, raw records) of phase 1 for one product."""
    recs = reconstruct_distances(doc, product, cfg)
    scale, rep = calibrate(recs)
    rows = rows_of(recs, scale)
    return rows, summarize(rows, rep, product), recs


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return path


def write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_phase1(out_dir: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any],
                 recs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Path]:
    out = {"csv": write_csv(out_dir / "dimensions.csv", rows, CSV_COLUMNS),
           "summary": write_json(out_dir / "dimensions_summary.json", summary)}
    if recs is not None:
        ends = {r["ref"]["id"]: {"a": r["a"].to_dict(), "b": r["b"].to_dict(), "raw": _f(r["raw"])}
                for r in recs}
        out["ends"] = write_json(out_dir / "dimensions_ends.json", ends)
    return out
