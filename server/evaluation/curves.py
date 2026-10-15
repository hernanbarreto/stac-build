"""Phase 2 — how the error grows: error against the camera distance and against the walk from
the calibration, per-point noise and resolution on the reference planes, two growth models and
the "confidence distance" per tolerance. Numbers only; nothing here chooses or interprets.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation import geometry as G
from evaluation.config import CurvesConfig, EvaluationConfig
from evaluation.dimension_error import write_csv, write_json
from evaluation.product import Product

NO_DATA = "no data"


# ── binning ──────────────────────────────────────────────────────────────

def bin_labels(edges: Sequence[float]) -> List[str]:
    return [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]


def bin_index(x: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bin of every x (−1 outside [edges[0], edges[-1]]); the last edge is inclusive."""
    e = np.asarray(edges, np.float64)
    idx = np.searchsorted(e, x, side="right") - 1
    idx = np.where(x == e[-1], len(e) - 2, idx)
    idx[(x < e[0]) | (x > e[-1]) | ~np.isfinite(x)] = -1
    return idx


def binned_stats(x: Sequence[float], err_abs: Sequence[float], err_rel: Sequence[float],
                 edges: Sequence[float], min_n: int, ids: Optional[Sequence[str]] = None
                 ) -> List[Dict[str, Any]]:
    """Per bin: n, median / P95 of the absolute and relative errors ('no data' under min_n)."""
    x = np.asarray(x, np.float64)
    ea = np.asarray(err_abs, np.float64)
    er = np.asarray(err_rel, np.float64)
    idx = bin_index(x, edges)
    rows = []
    for b, label in enumerate(bin_labels(edges)):
        m = idx == b
        n = int(m.sum())
        row: Dict[str, Any] = {"bin": label, "lo_m": edges[b], "hi_m": edges[b + 1], "n": n}
        if n >= min_n:
            row.update(abs_median_m=float(np.median(ea[m])), abs_p95_m=G.percentile(ea[m], 95),
                       abs_max_m=float(ea[m].max()), rel_median=float(np.median(er[m])),
                       rel_p95=G.percentile(er[m], 95), status="ok")
        else:
            row.update(abs_median_m=None, abs_p95_m=None, abs_max_m=None, rel_median=None,
                       rel_p95=None, status=NO_DATA)
        if ids is not None:
            row["ids"] = ";".join(str(i) for i, k in zip(ids, m) if k)
        rows.append(row)
    out_of_range = int((idx < 0).sum())
    if out_of_range:
        rows.append({"bin": "outside", "lo_m": None, "hi_m": None, "n": out_of_range, "status": "outside the bins",
                     "abs_median_m": None, "abs_p95_m": None, "abs_max_m": None, "rel_median": None,
                     "rel_p95": None, "ids": ";".join(str(i) for i, k in zip(ids, idx < 0) if k) if ids is not None else ""})
    return rows


def distance_of_row(r: Dict[str, Any], statistic: str) -> float:
    a, b = r.get("end_a_distance_m"), r.get("end_b_distance_m")
    if a is None or b is None:
        return float("nan")
    return float(np.mean([a, b])) if statistic == "mean" else float(max(a, b))


def error_curves(rows: List[Dict[str, Any]], cfg: CurvesConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Error vs camera distance and vs walk from the nearest gauge, over the evaluated TEST rows."""
    ok = [r for r in rows if r["role"] == "test" and r["status"] == "ok"]
    ea = [r["abs_error_m"] for r in ok]
    er = [r["rel_error"] for r in ok]
    ids = [r["id"] for r in ok]
    dist = [distance_of_row(r, cfg.distance_statistic) for r in ok]
    walk = [r["walk_from_nearest_gauge_m"] if r["walk_from_nearest_gauge_m"] is not None else float("nan")
            for r in ok]
    return {"distance": binned_stats(dist, ea, er, cfg.distance_bins_m, cfg.min_bin_n, ids),
            "walk": binned_stats(walk, ea, er, cfg.walk_bins_m, cfg.min_bin_n, ids)}


# ── reference planes: noise and resolution ────────────────────────────────

def _polygon_mask(uv: np.ndarray, poly: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath
    return MplPath(poly).contains_points(uv)


def fit_plane(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(centroid, unit normal) of the least-squares plane (SVD)."""
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2]


def plane_points(plane: Dict[str, Any], product: Product, scale: float, max_points: int, seed: int
                 ) -> Dict[str, Any]:
    """The product points born inside the plane's polygons (by provenance), their camera
    distance from their origin keyframe, in calibrated metres."""
    rows_all, dist_all, frames_all = [], [], []
    per_polygon = []
    for poly in plane["polygons"]:
        f = int(poly["frame"])
        if not product.registered(f) or not product.has_provenance():
            per_polygon.append({"frame": f, "n": 0, "reason": "keyframe not registered or no provenance"})
            continue
        rows, uv = product.points_of_frame(f)
        if rows.size == 0:
            per_polygon.append({"frame": f, "n": 0, "reason": "no product point born in this keyframe"})
            continue
        pts = product.undistort(f, np.asarray(poly["points"], np.float64))
        inside = _polygon_mask(uv, pts)
        sel = rows[inside]
        c = product.pose(f)[:3, 3]
        d = np.linalg.norm(product.xyz[sel] - c, axis=1) * scale
        rows_all.append(sel); dist_all.append(d); frames_all.append(np.full(sel.size, f))
        per_polygon.append({"frame": f, "n": int(sel.size), "n_candidates": int(rows.size)})
    if not rows_all:
        return {"rows": np.zeros(0, np.int64), "dist": np.zeros(0), "frames": np.zeros(0, np.int64),
                "per_polygon": per_polygon, "subsampled": False}
    rows = np.concatenate(rows_all); dist = np.concatenate(dist_all); frames = np.concatenate(frames_all)
    sub = False
    if rows.size > max_points:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(rows.size, max_points, replace=False))
        rows, dist, frames, sub = rows[keep], dist[keep], frames[keep], True
    return {"rows": rows, "dist": dist, "frames": frames, "per_polygon": per_polygon, "subsampled": sub}


def plane_noise(doc: Dict[str, Any], product: Product, scale: float, cfg: CurvesConfig
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """(noise rows per plane and bin, resolution rows per plane and bin, per-plane details).
    Noise = residual of the least-squares plane through the region's points; resolution = mean
    k-NN spacing and density (points / m² of the convex hull of the bin's points on the plane)."""
    from scipy.spatial import ConvexHull, cKDTree
    pf = cfg.plane_fit
    noise_rows, res_rows, details = [], [], {}
    pooled_abs, pooled_dist = [], []
    for plane in doc.get("planes", []):
        sel = plane_points(plane, product, scale, pf.max_points, pf.seed)
        pid = plane["id"]
        det: Dict[str, Any] = {"label": plane.get("label", ""), "n_points": int(sel["rows"].size),
                               "per_polygon": sel["per_polygon"], "subsampled": sel["subsampled"]}
        if sel["rows"].size < 3:
            det["status"] = NO_DATA
            details[pid] = det
            for label in bin_labels(cfg.distance_bins_m):
                noise_rows.append({"plane": pid, "label": det["label"], "bin": label, "n": 0,
                                   "rms_m": None, "median_abs_m": None, "p95_abs_m": None, "status": NO_DATA})
            continue
        P = product.xyz[sel["rows"]] * scale
        c, nrm = fit_plane(P)
        resid = (P - c) @ nrm
        det.update(status="ok", centroid=c.tolist(), normal=nrm.tolist(),
                   rms_m=float(np.sqrt(np.mean(resid ** 2))), p95_abs_m=G.percentile(np.abs(resid), 95))
        # in-plane coordinates for the hull area
        e1 = np.cross(nrm, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(nrm, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
        Q = np.stack([(P - c) @ e1, (P - c) @ e2], axis=1)
        tree = cKDTree(P)
        kk = min(pf.neighbour_k + 1, len(P))
        dd, _ = tree.query(P, k=kk)
        spacing = dd[:, -1] if kk > 1 else np.full(len(P), np.nan)
        idx = bin_index(sel["dist"], cfg.distance_bins_m)
        pooled_abs.append(np.abs(resid)); pooled_dist.append(sel["dist"])
        for b, label in enumerate(bin_labels(cfg.distance_bins_m)):
            m = idx == b
            n = int(m.sum())
            row = {"plane": pid, "label": det["label"], "bin": label, "n": n}
            rrow = {"plane": pid, "label": det["label"], "bin": label, "n": n}
            if n >= cfg.min_bin_n:
                r = resid[m]
                row.update(rms_m=float(np.sqrt(np.mean(r ** 2))), median_abs_m=float(np.median(np.abs(r))),
                           p95_abs_m=G.percentile(np.abs(r), 95), status="ok")
                area = None
                if n >= 3:
                    try:
                        area = float(ConvexHull(Q[m]).volume)      # 2-D hull: 'volume' is the area
                    except Exception:  # noqa: BLE001 — degenerate (collinear) sets have no area
                        area = None
                rrow.update(mean_spacing_mm=float(np.nanmean(spacing[m]) * 1000.0),
                            median_spacing_mm=float(np.nanmedian(spacing[m]) * 1000.0),
                            hull_area_m2=area, density_pts_per_m2=(n / area if area else None),
                            status="ok" if area else "no area")
            else:
                row.update(rms_m=None, median_abs_m=None, p95_abs_m=None, status=NO_DATA)
                rrow.update(mean_spacing_mm=None, median_spacing_mm=None, hull_area_m2=None,
                            density_pts_per_m2=None, status=NO_DATA)
            noise_rows.append(row); res_rows.append(rrow)
        details[pid] = det
    if pooled_abs:
        a = np.concatenate(pooled_abs); d = np.concatenate(pooled_dist)
        idx = bin_index(d, cfg.distance_bins_m)
        for b, label in enumerate(bin_labels(cfg.distance_bins_m)):
            m = idx == b
            n = int(m.sum())
            row = {"plane": "all", "label": "all planes", "bin": label, "n": n}
            if n >= cfg.min_bin_n:
                row.update(rms_m=float(np.sqrt(np.mean(a[m] ** 2))), median_abs_m=float(np.median(a[m])),
                           p95_abs_m=G.percentile(a[m], 95), status="ok")
            else:
                row.update(rms_m=None, median_abs_m=None, p95_abs_m=None, status=NO_DATA)
            noise_rows.append(row)
        details["_pooled"] = {"abs_residual": a, "distance": d}
    return noise_rows, res_rows, details


# ── growth models ─────────────────────────────────────────────────────────

def fit_growth(z: Sequence[float], err: Sequence[float]) -> Dict[str, Any]:
    """a + b·Z and a + c·Z² by least squares: coefficients, residual RMS and AIC of each
    (Gaussian AIC = n·ln(RSS/n) + 2k, k = 2). Numbers only."""
    z = np.asarray(z, np.float64); e = np.asarray(err, np.float64)
    m = np.isfinite(z) & np.isfinite(e)
    z, e = z[m], e[m]
    n = int(z.size)
    out: Dict[str, Any] = {"n": n}
    if n < 3:
        out["status"] = NO_DATA
        out["linear"] = out["quadratic"] = None
        return out
    for name, col in (("linear", z), ("quadratic", z * z)):
        A = np.stack([np.ones(n), col], axis=1)
        coef, *_ = np.linalg.lstsq(A, e, rcond=None)
        rss = float(np.sum((A @ coef - e) ** 2))
        aic = (n * math.log(rss / n) + 2 * 2) if rss > 0 else float("-inf")
        out[name] = {"a": float(coef[0]), ("b" if name == "linear" else "c"): float(coef[1]),
                     "residual_rms": float(math.sqrt(rss / n)), "rss": rss, "aic": aic,
                     "model": "err = a + b*Z" if name == "linear" else "err = a + c*Z^2"}
    out["status"] = "ok"
    return out


def growth_models(rows: List[Dict[str, Any]], plane_details: Dict[str, Any], cfg: CurvesConfig,
                  max_plane_samples: int, seed: int) -> Dict[str, Any]:
    ok = [r for r in rows if r["role"] == "test" and r["status"] == "ok"]
    out = {"distances_abs_error_vs_camera_distance": fit_growth(
        [distance_of_row(r, cfg.distance_statistic) for r in ok], [r["abs_error_m"] for r in ok])}
    out["distances_abs_error_vs_camera_distance"]["x"] = f"camera distance ({cfg.distance_statistic} of the two ends), m"
    pooled = plane_details.get("_pooled")
    if pooled is not None:
        a, d = pooled["abs_residual"], pooled["distance"]
        if a.size > max_plane_samples:
            rng = np.random.default_rng(seed)
            keep = rng.choice(a.size, max_plane_samples, replace=False)
            a, d = a[keep], d[keep]
        out["plane_abs_residual_vs_camera_distance"] = fit_growth(d, a)
        out["plane_abs_residual_vs_camera_distance"]["x"] = "distance of the point from its origin camera, m"
    else:
        out["plane_abs_residual_vs_camera_distance"] = {"n": 0, "status": NO_DATA, "linear": None, "quadratic": None}
    return out


# ── confidence distance ───────────────────────────────────────────────────

def confidence_distance(curve: List[Dict[str, Any]], tolerances: Sequence[float]) -> List[Dict[str, Any]]:
    """Per tolerance: the largest distance up to which EVERY bin from the first has data and a
    P95 under the tolerance. A bin without data stops the walk ('no data' beyond it); a bin
    over the tolerance stops it ('exceeded'). Nothing is extrapolated."""
    out = []
    bins = [b for b in curve if b.get("bin") != "outside"]
    for tol in tolerances:
        reached, status, stopped_at = None, "all bins pass", None
        for b in bins:
            if b["status"] != "ok":
                status, stopped_at = NO_DATA, b["bin"]
                break
            if b["abs_p95_m"] > tol:
                status, stopped_at = "exceeded", b["bin"]
                break
            reached = b["hi_m"]
        out.append({"tolerance_m": tol, "distance_m": reached, "status": status, "stopped_at_bin": stopped_at,
                    "note": ("no bin below the tolerance" if reached is None and status == "exceeded" else
                             "no data in the first bin" if reached is None and status == NO_DATA else "")})
    return out


# ── the phase ─────────────────────────────────────────────────────────────

NOISE_COLUMNS = ["plane", "label", "bin", "n", "rms_m", "median_abs_m", "p95_abs_m", "status"]
RES_COLUMNS = ["plane", "label", "bin", "n", "mean_spacing_mm", "median_spacing_mm", "hull_area_m2",
               "density_pts_per_m2", "status"]
CURVE_COLUMNS = ["bin", "lo_m", "hi_m", "n", "abs_median_m", "abs_p95_m", "abs_max_m", "rel_median", "rel_p95",
                 "status", "ids"]


def run_phase2(doc: Dict[str, Any], rows: List[Dict[str, Any]], summary: Dict[str, Any], product: Product,
               cfg: EvaluationConfig, out_dir: Path, *, plots: bool = True) -> Dict[str, Any]:
    """Phase 2 over one product: CSV per curve, JSON for the models and the confidence
    distance, PNG per curve. Returns what was written (paths) and the tables."""
    c = cfg.curves
    scale = summary["scale"]["scale_factor"]
    curves = error_curves(rows, c)
    if scale is None:
        noise_rows, res_rows, details = [], [], {"_status": "uncalibrated: no plane noise in metres"}
    else:
        noise_rows, res_rows, details = plane_noise(doc, product, float(scale), c)
    growth = growth_models(rows, details, c, c.plane_fit.max_points, c.plane_fit.seed)
    conf = {"by_camera_distance": confidence_distance(curves["distance"], c.tolerances_m),
            "by_walk": confidence_distance(curves["walk"], c.tolerances_m),
            "min_bin_n": c.min_bin_n, "p95_method": "numpy.percentile, linear interpolation"}
    paths = {
        "curve_distance": write_csv(out_dir / "curve_distance.csv", curves["distance"], CURVE_COLUMNS),
        "curve_walk": write_csv(out_dir / "curve_walk.csv", curves["walk"], CURVE_COLUMNS),
        "plane_noise": write_csv(out_dir / "plane_noise.csv", noise_rows, NOISE_COLUMNS),
        "resolution": write_csv(out_dir / "resolution.csv", res_rows, RES_COLUMNS),
        "growth_models": write_json(out_dir / "growth_models.json", growth),
        "confidence_distance": write_json(out_dir / "confidence_distance.json", conf),
        "plane_details": write_json(out_dir / "plane_noise_details.json",
                                    {k: v for k, v in details.items() if not k.startswith("_")}),
    }
    if plots:
        from evaluation import plots as PL
        paths["curve_distance_png"] = PL.plot_curve(out_dir / "curve_distance.png", curves["distance"],
                                                     f"camera distance ({c.distance_statistic} of the two ends) [m]",
                                                     "absolute error of the test distances [m]",
                                                     f"{product.name}: error vs camera distance", c.tolerances_m)
        paths["curve_walk_png"] = PL.plot_curve(out_dir / "curve_walk.png", curves["walk"],
                                                 "walk from the nearest gauge distance [m]",
                                                 "absolute error of the test distances [m]",
                                                 f"{product.name}: error vs walk", c.tolerances_m)
        paths["plane_noise_png"] = PL.plot_noise(out_dir / "plane_noise.png", noise_rows,
                                                  "distance from the origin camera [m]",
                                                  "RMS residual to the fitted plane [m]",
                                                  f"{product.name}: per-point noise on the reference planes")
    return {"paths": paths, "curves": curves, "plane_noise": noise_rows, "resolution": res_rows,
            "growth_models": growth, "confidence_distance": conf}
