"""
Multi-primitive decomposition (user 2026-08-29: "en el tren, donde se pueda
aplicar RANSAC — conos, circunferencias, planos, rectángulos — debe aplicarse,
y Poisson donde no hay manera").

A complex object (a train, a wall with an attached structure) is not ONE
surface: it CONTAINS primitives. This module extracts them iteratively —
fit plane / cylinder / sphere on the remaining points, keep the candidate
with the largest support, remove its members, repeat — until nothing
parametric remains. The leftover is the caller's Poisson residue, so the
final mesh is smooth fitted geometry wherever a model holds and measured
free-form only where there is genuinely no model.

Deterministic, tool_measured; every part records its kind, support and rms.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .escalate import FITTERS, FitContext

logger = logging.getLogger("SurfaceFit")

# primitives only — free-form models never take part in decomposition
PRIMITIVES = ("plane", "cylinder", "sphere")


def extract_primitives(pts: np.ndarray, ctx: FitContext,
                       inlier_dist: float = 0.04,
                       min_points: int = 8000,
                       min_frac_remaining: float = 0.10,
                       max_primitives: int = 6,
                       max_fit_points: int = 200_000,
                       tag: str = "") -> Tuple[List[tuple], np.ndarray]:
    """Iterative largest-support-first primitive extraction.

    Returns (parts, remaining_mask): ``parts`` is a list of
    ``(kind, model, member_mask)`` with ``member_mask`` over the INPUT points;
    ``remaining_mask`` marks the points no primitive claimed (Poisson residue).
    """
    n = len(pts)
    remaining = np.ones(n, dtype=bool)
    parts: List[tuple] = []
    for it in range(int(max_primitives)):
        idx = np.nonzero(remaining)[0]
        if len(idx) < min_points:
            break
        sub = pts[idx]
        if len(sub) > max_fit_points:
            pick = np.random.default_rng(it).choice(
                len(sub), max_fit_points, replace=False)
            fit_pts = sub[pick]
        else:
            fit_pts = sub
        best = None
        for kind in PRIMITIVES:
            fitter = FITTERS.get(kind)
            if fitter is None:
                continue
            try:
                model = fitter(fit_pts, ctx)
            except Exception:  # noqa: BLE001 — a fitter crash just skips it
                model = None
            if model is None:
                continue
            d = np.abs(np.asarray(model.signed_distance(sub)))
            members = d <= inlier_dist
            n_in = int(members.sum())
            if best is None or n_in > best[3]:
                best = (kind, model, members, n_in)
        if best is None:
            break
        kind, model, members, n_in = best
        if n_in < max(min_points, min_frac_remaining * len(idx)):
            break
        mask = np.zeros(n, dtype=bool)
        mask[idx[members]] = True
        parts.append((kind, model, mask))
        remaining &= ~mask
        logger.info("decompose%s: primitive %d = %s, %s pts (%.0f%% of "
                    "remaining); %s pts left",
                    f" {tag}" if tag else "", len(parts), kind,
                    f"{n_in:,}", 100.0 * n_in / max(len(idx), 1),
                    f"{int(remaining.sum()):,}")
    return parts, remaining
