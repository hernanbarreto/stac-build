"""Phase 5 — calibrated confidence: the error of the points on the reference planes (their
residual to the fitted plane) and of the test ends (their distance's error), grouped by the
per-point fields the cloud carries — ``confidence`` (quantile bins), ``status``, ``mv_votes``,
``mask_votes`` — with n, median and P95 per group, Spearman's correlation between confidence and
error, and the reliability table (verified against single_witness, measured whichever way it
comes out). A field the cloud does not carry is reported as absent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation import geometry as G
from evaluation.config import ConfidenceConfig, CurvesConfig
from evaluation.curves import NO_DATA, fit_plane, plane_points
from evaluation.dimension_error import write_csv, write_json
from evaluation.product import Product

STATUS_NAMES = {0: "unobserved", 1: "verified", 2: "single_witness", 3: "mask_conflict", 4: "dynamic"}
GROUP_FIELDS = ("confidence", "status", "mv_votes", "mask_votes")
GROUP_COLUMNS = ["source", "field", "group", "n", "median_m", "p95_m", "status"]


def _group_rows(source: str, field: str, labels: Sequence[Any], err: np.ndarray, min_n: int) -> List[Dict[str, Any]]:
    labels = np.asarray(labels)
    rows = []
    for g in sorted(set(labels.tolist()), key=lambda x: (str(type(x)), x)):
        m = labels == g
        n = int(m.sum())
        row = {"source": source, "field": field, "group": g, "n": n}
        if n >= min_n:
            row.update(median_m=float(np.median(err[m])), p95_m=G.percentile(err[m], 95), status="ok")
        else:
            row.update(median_m=None, p95_m=None, status=NO_DATA)
        rows.append(row)
    return rows


def confidence_bins(conf: np.ndarray, bins: int) -> Tuple[np.ndarray, List[str]]:
    """Quantile bins of the confidence (equal counts); labels name the edges."""
    conf = np.asarray(conf, np.float64)
    qs = np.quantile(conf, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(qs, conf, side="right") - 1, 0, bins - 1)
    labels = [f"q{i}:{qs[i]:.3g}-{qs[i + 1]:.3g}" for i in range(bins)]
    return idx, labels


def spearman(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.unique(x[m]).size < 2 or np.unique(y[m]).size < 2:
        return {"rho": None, "p_value": None, "n": int(m.sum()), "status": NO_DATA}
    r = spearmanr(x[m], y[m])
    return {"rho": float(r.statistic), "p_value": float(r.pvalue), "n": int(m.sum()), "status": "ok"}


def plane_point_errors(doc: Dict[str, Any], product: Product, scale: float, ccfg: CurvesConfig
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """(rows, |residual| of each point to its plane, the fields' values of those rows)."""
    rows_all, err_all = [], []
    for plane in doc.get("planes", []):
        sel = plane_points(plane, product, scale, ccfg.plane_fit.max_points, ccfg.plane_fit.seed)
        if sel["rows"].size < 3:
            continue
        P = product.xyz[sel["rows"]] * scale
        c, n = fit_plane(P)
        rows_all.append(sel["rows"]); err_all.append(np.abs((P - c) @ n))
    if not rows_all:
        return np.zeros(0, np.int64), np.zeros(0), {}
    rows = np.concatenate(rows_all); err = np.concatenate(err_all)
    fields = {f: np.asarray(product.data[f][rows]) for f in GROUP_FIELDS if f in (product.data.dtype.names or ())} \
        if hasattr(product, "data") else {}
    return rows, err, fields


def end_point_errors(dim_rows: List[Dict[str, Any]], recs: List[Dict[str, Any]], product: Product, window_px: int
                     ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """The product points at the TEST ends (by provenance, within the lookup window) carry their
    distance's absolute error."""
    if not hasattr(product, "data") or not product.has_provenance():
        return np.zeros(0, np.int64), np.zeros(0), {}
    err_of = {r["id"]: r["abs_error_m"] for r in dim_rows if r["role"] == "test" and r["status"] == "ok"}
    rows_all, err_all = [], []
    for r in recs:
        e = err_of.get(r["ref"]["id"])
        if e is None:
            continue
        for end in (r["a"], r["b"]):
            for ob in end.per_observation:
                if ob.get("method") != "provenance":
                    continue
                f = int(ob["frame"])
                prows, uv = product.points_of_frame(f)
                m = (np.abs(uv[:, 0] - ob["u_und"]) <= window_px) & (np.abs(uv[:, 1] - ob["v_und"]) <= window_px)
                if m.any():
                    rows_all.append(prows[m]); err_all.append(np.full(int(m.sum()), e))
    if not rows_all:
        return np.zeros(0, np.int64), np.zeros(0), {}
    rows = np.concatenate(rows_all); err = np.concatenate(err_all)
    fields = {f: np.asarray(product.data[f][rows]) for f in GROUP_FIELDS if f in (product.data.dtype.names or ())}
    return rows, err, fields


def analyse(source: str, err: np.ndarray, fields: Dict[str, np.ndarray], cfg: ConfidenceConfig
            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    extra: Dict[str, Any] = {"n_points": int(err.size), "fields_present": sorted(fields), 
                             "fields_absent": [f for f in GROUP_FIELDS if f not in fields]}
    if err.size == 0:
        extra["status"] = NO_DATA
        return rows, extra
    if "confidence" in fields:
        idx, labels = confidence_bins(fields["confidence"], cfg.bins)
        rows += _group_rows(source, "confidence", [labels[i] for i in idx], err, cfg.min_group_n)
        extra["spearman_confidence_vs_error"] = spearman(fields["confidence"], err)
    for f in ("status", "mv_votes", "mask_votes"):
        if f in fields:
            vals = fields[f].astype(int)
            lab = [STATUS_NAMES.get(int(v), str(int(v))) for v in vals] if f == "status" else vals.tolist()
            rows += _group_rows(source, f, lab, err, cfg.min_group_n)
            if f != "status":
                extra[f"spearman_{f}_vs_error"] = spearman(vals, err)
    if "status" in fields:
        st = fields["status"].astype(int)
        ver, single = err[st == 1], err[st == 2]
        extra["reliability"] = {
            "verified": {"n": int(ver.size), "median_m": float(np.median(ver)) if ver.size else None,
                         "p95_m": G.percentile(ver, 95)},
            "single_witness": {"n": int(single.size), "median_m": float(np.median(single)) if single.size else None,
                               "p95_m": G.percentile(single, 95)},
            "verified_median_lower": (bool(np.median(ver) < np.median(single)) if ver.size and single.size else None),
            "note": "measured whichever way it comes out; None = a class without points"}
    extra["status"] = "ok"
    return rows, extra


def run_phase5(doc: Dict[str, Any], dim_rows: List[Dict[str, Any]], recs: List[Dict[str, Any]], summary: Dict[str, Any],
               product: Product, ccfg: CurvesConfig, cfg: ConfidenceConfig, window_px: int, out_dir) -> Dict[str, Any]:
    from pathlib import Path
    out_dir = Path(out_dir)
    scale = summary["scale"]["scale_factor"]
    rows: List[Dict[str, Any]] = []
    rep: Dict[str, Any] = {"fields_of_the_cloud": list(product.fields()), "confidence_bins": cfg.bins,
                           "min_group_n": cfg.min_group_n}
    if scale is None:
        rep["status"] = "uncalibrated: no error in metres"
    else:
        _, e_pl, f_pl = plane_point_errors(doc, product, float(scale), ccfg)
        r, x = analyse("planes", e_pl, f_pl, cfg)
        rows += r; rep["planes"] = x
        _, e_en, f_en = end_point_errors(dim_rows, recs, product, window_px)
        r, x = analyse("test_ends", e_en, f_en, cfg)
        rows += r; rep["test_ends"] = x
        rep["status"] = "ok"
    rep["groups"] = rows
    write_csv(out_dir / "confidence_groups.csv", rows, GROUP_COLUMNS)
    rel = [{"source": src, **{f"{cls}_{k}": v for cls in ("verified", "single_witness")
                              for k, v in (rep.get(src, {}).get("reliability", {}).get(cls) or {}).items()},
            "verified_median_lower": rep.get(src, {}).get("reliability", {}).get("verified_median_lower")}
           for src in ("planes", "test_ends") if rep.get(src, {}).get("reliability")]
    write_csv(out_dir / "reliability.csv", rel, ["source", "verified_n", "verified_median_m", "verified_p95_m",
                                                 "single_witness_n", "single_witness_median_m", "single_witness_p95_m",
                                                 "verified_median_lower"])
    write_json(out_dir / "confidence_summary.json", rep)
    return rep
