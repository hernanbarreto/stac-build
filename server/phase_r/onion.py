# STAC-Builder — Phase R.3: onion (duplicated-surface) detector.
#
# Per instance, pool its points from all views/windows and analyse the
# distribution ALONG the normal of its OBB. A "cebolla" (onion / doubled
# surface) from window mis-registration shows up as bimodality; the separation
# between the two modes IN METRES is the local registration error — the central
# health metric of Phase R.
#
# Metric: GMM(2) vs GMM(1) by BIC (spec R.3). Separation is |mean_a - mean_b|
# along the axis (local OBB coords are metric).
#
# PROVENANCE: ours (metric). Uses geometry.signed_distances_along_axis (OBB from
# the R3D-ported fitter).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OnionResult:
    bimodal: bool
    separation_m: float          # distance between modes along the OBB normal (m)
    bic_delta: float             # BIC(1) - BIC(2); >0 favours 2 components
    weights: tuple[float, float] # mixture weights of the two modes (0,0 if 1)
    n_points: int


def _obb_normal_axis(aabb: np.ndarray) -> int:
    """Index of the OBB's shortest extent = its surface normal direction."""
    ex = aabb[1] - aabb[0]
    ey = aabb[3] - aabb[2]
    ez = aabb[5] - aabb[4]
    return int(np.argmin([ex, ey, ez]))


def detect_onion(points: np.ndarray, obb_transform: np.ndarray, obb_aabb: np.ndarray,
                 min_points: int = 40, min_separation_m: float = 0.01) -> OnionResult:
    """Fit 1- vs 2-component GMM to the point distribution along the OBB normal.
    Bimodal iff BIC(2) < BIC(1) and the modes are separated by >min_separation_m."""
    from .geometry import signed_distances_along_axis

    n = len(points)
    if n < min_points:
        return OnionResult(False, 0.0, 0.0, (0.0, 0.0), n)

    axis = _obb_normal_axis(obb_aabb)
    x = signed_distances_along_axis(points, obb_transform, axis=axis).astype(np.float64)

    # 1- vs 2-component GMM by BIC, numpy-only (no sklearn dependency — the
    # onion metric must run in any deployment env; this is a 1-D EM).
    fit = _gmm2_em(x)
    if fit is None:
        return OnionResult(False, 0.0, 0.0, (0.0, 0.0), n)
    bic1, bic2, means, weights = fit
    sep = float(abs(means[0] - means[1]))
    # require both modes to carry meaningful mass, not a tiny outlier cluster
    balanced = min(weights) > 0.1
    bimodal = bool(bic2 < bic1 and sep > min_separation_m and balanced)
    return OnionResult(bimodal, sep if bimodal else 0.0, float(bic1 - bic2),
                       (float(weights[0]), float(weights[1])), n)


def onion_heatmap(points: np.ndarray, obb_transform: np.ndarray, obb_aabb: np.ndarray,
                  grid: int = 6, min_cell_points: int = 40) -> dict:
    """Per-instance onion HEATMAP (spec R.3): tile the OBB's two largest faces
    into a grid×grid lattice and run the bimodality test per cell along the OBB
    normal. Cell value = mode separation in metres (0 = clean / insufficient).
    Returns {"grid", "axis_u", "axis_v", "normal_axis", "cells", "counts"}."""
    from .geometry import obb_local_coords

    normal = _obb_normal_axis(obb_aabb)
    axes = [a for a in (0, 1, 2) if a != normal]
    local = obb_local_coords(points, obb_transform)
    lo = np.array([obb_aabb[0], obb_aabb[2], obb_aabb[4]])
    hi = np.array([obb_aabb[1], obb_aabb[3], obb_aabb[5]])
    span = np.maximum(hi - lo, 1e-9)
    u = np.clip(((local[:, axes[0]] - lo[axes[0]]) / span[axes[0]] * grid).astype(int), 0, grid - 1)
    v = np.clip(((local[:, axes[1]] - lo[axes[1]]) / span[axes[1]] * grid).astype(int), 0, grid - 1)
    x = local[:, normal]

    cells = [[0.0] * grid for _ in range(grid)]
    counts = [[0] * grid for _ in range(grid)]
    for i in range(grid):
        rows = u == i
        for j in range(grid):
            sel = rows & (v == j)
            nij = int(sel.sum())
            counts[i][j] = nij
            if nij < min_cell_points:
                continue
            fit = _gmm2_em(x[sel])
            if fit is None:
                continue
            bic1, bic2, means, weights = fit
            if bic2 < bic1 and min(weights) > 0.1:
                cells[i][j] = round(float(abs(means[0] - means[1])), 4)
    return {"grid": grid, "axis_u": axes[0], "axis_v": axes[1],
            "normal_axis": normal, "cells": cells, "counts": counts}


def _gauss_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, 1e-9)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mu) ** 2 / var)


def _bic(loglik: float, k_params: int, n: int) -> float:
    return -2.0 * loglik + k_params * np.log(max(n, 1))


def _gmm2_em(x: np.ndarray, iters: int = 100, tol: float = 1e-6):
    """Fit 1- and 2-component 1-D Gaussian mixtures by EM; return
    (bic1, bic2, means2, weights2) or None. Pure numpy."""
    n = len(x)
    mu0, var0 = float(x.mean()), float(x.var())
    if var0 < 1e-12:
        return None
    ll1 = float(_gauss_logpdf(x, mu0, var0).sum())
    bic1 = _bic(ll1, 2, n)  # mean + var

    # init 2 components at the ±1σ quantiles
    lo, hi = np.percentile(x, [25, 75])
    mu = np.array([lo, hi], float)
    if mu[0] == mu[1]:
        mu = np.array([mu0 - np.sqrt(var0), mu0 + np.sqrt(var0)])
    var = np.array([var0, var0], float)
    w = np.array([0.5, 0.5], float)

    prev_ll = -np.inf
    for _ in range(iters):
        # E-step: responsibilities (log-space for stability)
        logr = np.stack([np.log(w[k] + 1e-12) + _gauss_logpdf(x, mu[k], var[k])
                         for k in range(2)], axis=1)
        m = logr.max(axis=1, keepdims=True)
        logsum = m[:, 0] + np.log(np.exp(logr - m).sum(axis=1))
        ll = float(logsum.sum())
        r = np.exp(logr - logsum[:, None])
        # M-step
        nk = r.sum(axis=0) + 1e-12
        w = nk / n
        mu = (r * x[:, None]).sum(axis=0) / nk
        var = (r * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk
        var = np.maximum(var, 1e-9)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    bic2 = _bic(ll, 5, n)  # 2 means + 2 vars + 1 weight
    return bic1, bic2, mu, w
