"""Per-session confidence calibration (claude_stac.txt §4-F6).

A model's confidence is a number the model chose; what it is worth in THIS
session is measured against a reference that does not come from that model.
Here the reference is the tier-0 plane-sweep depth (photometric evidence
confirmed by other views) — or, before any sweep exists, the landmarks F5
triangulated from the tracks. For each model (the Omega prior, DA3) the
relative depth error ``prior / reference − 1`` is binned by the model's own
confidence and by distance (equal-count bins: quantiles of each axis over the
samples), and every cell carries its sample count, signed median (bias) and
the ``beta_quantile`` quantile of the absolute error.

That quantile is what the plane-sweep reads as β per pixel (the half-width of
the depth search around the prior, capped by ``beta_max``): the interval that
contains the truth with the declared confidence, measured, not chosen.

A cell with fewer than ``min_bin_samples`` samples does not speak for itself:
the lookup falls back to its confidence row (all distances pooled), and a row
that is also starved to the model's global quantile. The report says how many
cells did.

DA3's confidence is the expp1 activation (1 + exp(x)); it is binned as
``conf − 1`` (quantile bins are invariant to the transform — it only makes the
reported edges readable).

Output: ``output/precision/confidence_calibration.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

CALIBRATION_NAME = "confidence_calibration.json"
CALIBRATION_VERSION = 1
PROVENANCE = "tool_measured"
DA3_CONF_TRANSFORM = "expp1-1"
REFERENCES = ("landmarks", "tier0")


class CalibrationError(RuntimeError):
    """The calibration cannot be measured — with the exact reason."""


def _edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-count bin edges; ties collapse (fewer bins, never empty ones)."""
    q = np.quantile(x, np.linspace(0.0, 1.0, int(n_bins) + 1))
    return np.unique(q)


def _bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index in [0, len(edges) − 2]; out-of-range values clamp to the end bins."""
    if len(edges) < 2:
        return np.zeros(np.shape(x), np.int64)
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def calibrate(err_rel: np.ndarray, conf: np.ndarray, dist: np.ndarray, *,
              conf_bins: int, dist_bins: int, quantile: float,
              min_bin_samples: int) -> Dict[str, Any]:
    """One model's error table. ``err_rel`` = prior / reference − 1 per sample."""
    e = np.asarray(err_rel, np.float64).ravel()
    c = np.asarray(conf, np.float64).ravel()
    d = np.asarray(dist, np.float64).ravel()
    ok = np.isfinite(e) & np.isfinite(c) & np.isfinite(d) & (d > 0)
    e, c, d = e[ok], c[ok], d[ok]
    if e.size < int(min_bin_samples):
        raise CalibrationError(f"{e.size} sample(s) — fewer than min_bin_samples "
                               f"({min_bin_samples}) even for the global quantile")
    ce, de = _edges(c, conf_bins), _edges(d, dist_bins)
    ci, di = _bin(c, ce), _bin(d, de)
    a = np.abs(e)
    nc, nd = max(len(ce) - 1, 1), max(len(de) - 1, 1)
    q_cell = np.full((nc, nd), np.nan)
    n_cell = np.zeros((nc, nd), np.int64)
    bias_cell = np.full((nc, nd), np.nan)
    for i in range(nc):
        for j in range(nd):
            m = (ci == i) & (di == j)
            n_cell[i, j] = int(m.sum())
            if n_cell[i, j]:
                q_cell[i, j] = float(np.quantile(a[m], quantile))
                bias_cell[i, j] = float(np.median(e[m]))
    q_row = np.array([float(np.quantile(a[ci == i], quantile)) if np.any(ci == i) else np.nan
                      for i in range(nc)])
    n_row = np.array([int(np.sum(ci == i)) for i in range(nc)])
    return {"n": int(e.size), "quantile": float(quantile),
            "min_bin_samples": int(min_bin_samples),
            "conf_edges": ce.tolist(), "dist_edges_m": de.tolist(),
            "abs_err_quantile": q_cell.tolist(), "bias_median": bias_cell.tolist(),
            "n_samples": n_cell.tolist(),
            "row_abs_err_quantile": q_row.tolist(), "row_n": n_row.tolist(),
            "global_abs_err_quantile": float(np.quantile(a, quantile)),
            "global_bias_median": float(np.median(e)),
            "cells_starved": int(np.sum(n_cell < int(min_bin_samples)))}


def lookup(table: Dict[str, Any], conf: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """The calibrated |error| quantile per sample (any shape), with the starved-cell
    fallback (cell → confidence row → global)."""
    ce = np.asarray(table["conf_edges"], np.float64)
    de = np.asarray(table["dist_edges_m"], np.float64)
    q = np.asarray(table["abs_err_quantile"], np.float64)
    n = np.asarray(table["n_samples"], np.int64)
    row_q = np.asarray(table["row_abs_err_quantile"], np.float64)
    row_n = np.asarray(table["row_n"], np.int64)
    m = int(table["min_bin_samples"])
    g = float(table["global_abs_err_quantile"])
    use = np.where(n >= m, q, np.where(row_n[:, None] >= m, row_q[:, None], g))
    ci = _bin(np.asarray(conf, np.float64), ce)
    di = _bin(np.asarray(dist, np.float64), de)
    return use[ci, di]


def da3_conf(raw_conf: np.ndarray) -> np.ndarray:
    """DA3's expp1 confidence on the calibration axis."""
    return np.asarray(raw_conf, np.float64) - 1.0


def write_calibration(output_dir: Path, reference: str, tables: Dict[str, Dict[str, Any]],
                      epochs: Dict[str, int], params: Dict[str, Any]) -> Path:
    if reference not in REFERENCES:
        raise CalibrationError(f"reference must be one of {REFERENCES}, got {reference!r}")
    doc = {"version": CALIBRATION_VERSION, "provenance": PROVENANCE, **epochs,
           "reference": reference, "da3_conf_transform": DA3_CONF_TRANSFORM,
           "params": params, "models": tables,
           "replaces_in_core_epochs": ["reconstruction.simple.conf_percentile",
                                       "reconstruction.simple.conf_min_norm",
                                       "scale_conf_top_frac"]}
    p = Path(output_dir) / "precision" / CALIBRATION_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1))
    return p


def load_calibration(output_dir: Path, epochs: Optional[Dict[str, int]] = None,
                     reference: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The session's calibration, or None when absent — or measured on another
    geometry/camera epoch, or from another reference, when those are given."""
    p = Path(output_dir) / "precision" / CALIBRATION_NAME
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    if epochs is not None and any(doc.get(k) != v for k, v in epochs.items()):
        return None
    if reference is not None and doc.get("reference") != reference:
        return None
    return doc


def sample_pairs(ref: np.ndarray, prior: np.ndarray, conf: np.ndarray, mask: np.ndarray,
                 n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Up to ``n`` (err_rel, conf, dist) samples of one frame where ``mask`` holds."""
    idx = np.flatnonzero(mask.ravel() & (ref.ravel() > 0) & (prior.ravel() > 0)
                         & np.isfinite(conf.ravel()))
    if idx.size > n:
        idx = rng.choice(idx, size=int(n), replace=False)
    r, p, c = ref.ravel()[idx], prior.ravel()[idx], conf.ravel()[idx]
    return {"err_rel": p / r - 1.0, "conf": c, "dist": r}


def concat(samples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = ("err_rel", "conf", "dist")
    if not samples:
        return {k: np.zeros(0) for k in keys}
    return {k: np.concatenate([s[k] for s in samples]) for k in keys}
