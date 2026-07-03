"""
Stage 4 — fidelity as a deliverable.

Residuals are ALWAYS signed point→surface distances of the **original** metric
cloud (never the consolidated/WLOP one): the smooth surface is the model, the
residual field is the measurement record that backs it.

Produces: summary stats (RMS/p95/max, mm), a construction-spec flatness check
(worst peak-to-valley inside any span×span window, e.g. 5 mm under 2 m), the
Moran white-noise verdict, and localized *findings* — systematic deviation
zones (bulge/"panza", depression, plumb deviation/"desplome") with world
position and magnitude.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .models import Finding, MoranResult, ResidualStats
from .spatial_test import GridField, coarsen, grid_scalar, morans_i


def compute_stats(signed_m: np.ndarray, field: GridField,
                  flatness_tol_mm: float = 5.0,
                  flatness_span_m: float = 2.0,
                  moran: Optional[MoranResult] = None) -> ResidualStats:
    r_mm = np.asarray(signed_m, dtype=np.float64) * 1000.0
    a = np.abs(r_mm)
    worst = flatness_worst(field, flatness_span_m)
    worst_mm = None if worst is None else worst * 1000.0
    return ResidualStats(
        n_points=int(r_mm.size),
        rms_mm=float(np.sqrt(np.mean(r_mm ** 2))) if r_mm.size else 0.0,
        mean_mm=float(r_mm.mean()) if r_mm.size else 0.0,
        std_mm=float(r_mm.std()) if r_mm.size else 0.0,
        p95_mm=float(np.percentile(a, 95)) if r_mm.size else 0.0,
        max_mm=float(a.max()) if r_mm.size else 0.0,
        flatness_worst_mm=worst_mm,
        flatness_tol_mm=float(flatness_tol_mm),
        flatness_span_m=float(flatness_span_m),
        flatness_pass=None if worst_mm is None else bool(worst_mm <= flatness_tol_mm),
        moran=moran,
    )


def flatness_worst(field: GridField, span_m: float) -> Optional[float]:
    """Worst peak-to-valley of cell-mean residuals inside any span×span window
    (the straightedge test: "≤ tol under a `span` rule"), in metres.

    The straightedge spec measures the SYSTEMATIC surface deviation, so the
    field is first coarsened to ~span/10 cells (count-weighted): with sparse
    sampling, fine-cell means are noisy estimates whose max-min would fail a
    perfectly flat surface on sensor noise alone. Empty cells are neutral
    (±inf swapped out) so sparse coverage never fabricates a deviation.
    """
    target_cell = span_m / 10.0
    if target_cell > field.cell * 1.5:
        field = coarsen(field, int(round(target_cell / field.cell)))
    occ = field.occupied
    if occ.sum() < 4:
        return None
    w = max(int(round(span_m / field.cell)), 2)
    hi = np.where(occ, field.mean, -np.inf)
    lo = np.where(occ, field.mean, np.inf)
    mx = ndimage.maximum_filter(hi, size=w, mode="constant", cval=-np.inf)
    mn = ndimage.minimum_filter(lo, size=w, mode="constant", cval=np.inf)
    ptv = mx - mn
    valid = np.isfinite(ptv) & occ
    return float(ptv[valid].max()) if valid.any() else None


def detect_findings(field: GridField, uv_to_world, world_up: np.ndarray,
                    normal: np.ndarray,
                    finding_dev_mm: float = 5.0,
                    finding_min_area_m2: float = 0.05,
                    tilt_mm_per_m: float = 3.0) -> List[Finding]:
    """Localized systematic deviations from the gridded residual field.

    - blobs: connected components of |cell mean| > finding_dev_mm with enough
      area → 'bulge' (positive, along the oriented normal) or 'depression'.
    - tilt ('desplome'): a global linear residual gradient. Only meaningful on
      non-horizontal surfaces (walls/columns), where it is reported along the
      in-plane up axis.
    """
    out: List[Finding] = []
    occ = field.occupied
    if occ.sum() < 4:
        return out
    mean = field.mean
    cell_area = field.cell * field.cell
    thr = finding_dev_mm / 1000.0

    for sign, kind in ((1.0, "bulge"), (-1.0, "depression")):
        mask = occ & np.isfinite(mean) & (sign * mean > thr)
        if not mask.any():
            continue
        labels, nlab = ndimage.label(mask)  # 4-connectivity: distinct zones stay distinct
        for lab in range(1, nlab + 1):
            m = labels == lab
            area = float(m.sum()) * cell_area
            if area < finding_min_area_m2:
                continue
            vals = mean[m]
            uv_c = field.cell_centers_uv(m)
            # weight centroid by |deviation| so it lands on the belly of the zone
            wgt = np.abs(vals)
            wgt = wgt / wgt.sum()
            center_uv = (uv_c * wgt[:, None]).sum(0)
            peak = float(vals[np.argmax(np.abs(vals))])
            out.append(Finding(
                kind=kind,
                center_xyz=uv_to_world(center_uv[None, :])[0],
                area_m2=area,
                mean_dev_mm=float(vals.mean() * 1000.0),
                peak_dev_mm=peak * 1000.0,
                n_points=int(field.count[m].sum()),
            ))

    # tilt / plumb deviation ("desplome"). For a wall the LS plane ABSORBS any
    # linear lean — the residual gradient is ~0 by construction — so plumb is
    # measured on the fitted plane itself: how far its normal is from
    # horizontal, reported as horizontal drift (mm) per metre climbed.
    n_hat = np.asarray(normal, dtype=np.float64)
    up_hat = np.asarray(world_up, dtype=np.float64)
    c = float(n_hat @ up_hat)
    if abs(c) < 0.5:  # wall-ish
        plumb_mm_per_m = abs(c) / max(np.sqrt(1.0 - c * c), 1e-9) * 1000.0
        if plumb_mm_per_m > tilt_mm_per_m:
            # in-plane "up" direction; its horizontal component = lean direction
            w = up_hat - c * n_hat
            w /= max(np.linalg.norm(w), 1e-12)
            lean = w - (w @ up_hat) * up_hat
            ln = np.linalg.norm(lean)
            lean = lean / ln if ln > 1e-12 else -np.sign(c) * n_hat
            uv_c = field.cell_centers_uv()
            center = uv_to_world(uv_c.mean(0, keepdims=True))[0]
            vals = mean[occ]
            out.append(Finding(
                kind="tilt",
                center_xyz=center,
                area_m2=float(occ.sum()) * cell_area,
                mean_dev_mm=float(vals.mean() * 1000.0),
                peak_dev_mm=float(vals[np.argmax(np.abs(vals))] * 1000.0),
                n_points=int(field.count.sum()),
                gradient_mm_per_m=float(plumb_mm_per_m),
                gradient_dir=lean,
            ))

    out.sort(key=lambda f: -abs(f.peak_dev_mm))
    return out


