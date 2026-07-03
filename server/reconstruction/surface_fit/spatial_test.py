"""
White-noise acceptance test for fit residuals (the escalation trigger).

A fitted model is accepted only if its residual field is spatially
unstructured. We grid the signed residuals on the surface's UV frame (cell
mean), then compute Moran's I with queen (8-neighbour) contiguity over the
occupied cells. Gridding first is essential: with millions of points, point-
level neighbours are dominated by sensor noise; cell means expose the
registration/shape structure we care about (double layers, bulges, waviness).

Pure numpy — the z-score uses the standard randomization-assumption variance
(Cliff & Ord), no pysal dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .models import MoranResult

# 8-connected (queen) neighbour offsets
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class GridField:
    """Cell-mean field of a scalar over a UV domain."""
    mean: np.ndarray      # (H,W) float, NaN where empty
    count: np.ndarray     # (H,W) int, points per cell
    u0: float             # world-UV coordinate of cell (0,0)'s lower corner
    v0: float
    cell: float           # cell size (m)

    @property
    def occupied(self) -> np.ndarray:
        return self.count > 0

    def cell_centers_uv(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """(K,2) UV centers of cells selected by ``mask`` (default: occupied)."""
        m = self.occupied if mask is None else mask
        jj, ii = np.nonzero(m)
        return np.column_stack([self.u0 + (ii + 0.5) * self.cell,
                                self.v0 + (jj + 0.5) * self.cell])


def grid_scalar(uv: np.ndarray, values: np.ndarray, cell: float) -> GridField:
    """Bin ``values`` on a UV grid → per-cell mean/count (bincount, O(N))."""
    uv = np.asarray(uv, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    u0, v0 = uv.min(0)
    ii = np.floor((uv[:, 0] - u0) / cell).astype(np.int64)
    jj = np.floor((uv[:, 1] - v0) / cell).astype(np.int64)
    w = int(ii.max()) + 1
    h = int(jj.max()) + 1
    flat = jj * w + ii
    cnt = np.bincount(flat, minlength=h * w)
    tot = np.bincount(flat, weights=values, minlength=h * w)
    count = cnt.reshape(h, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (tot.reshape(h, w) / count).astype(np.float64)
    mean[count == 0] = np.nan
    return GridField(mean=mean, count=count.astype(np.int64),
                     u0=float(u0), v0=float(v0), cell=float(cell))


def morans_i(field: GridField, z_max: float = 4.0,
             i_min: float = 0.05) -> MoranResult:
    """Moran's I over the occupied cells of a gridded residual field.

    structured := (z > z_max) AND (I > i_min). Both gates matter: with
    thousands of cells even I≈0.02 can be "significant" (huge z) while being
    metrically irrelevant, and a tiny segment can have I inflated by chance.
    """
    occ = field.occupied
    n = int(occ.sum())
    if n < 16:  # too few cells for a meaningful spatial test
        return MoranResult(i=0.0, expected=0.0, z=0.0, n_cells=n, structured=False)

    z = np.where(occ, field.mean - field.mean[occ].mean(), 0.0)
    m2 = float((z[occ] ** 2).sum())
    if m2 <= 0.0:
        return MoranResult(i=0.0, expected=-1.0 / (n - 1), z=0.0,
                           n_cells=n, structured=False)

    # Σ_ij w_ij z_i z_j and degrees, via array shifts (binary symmetric weights)
    cross = 0.0
    deg = np.zeros_like(z)
    occ_f = occ.astype(np.float64)
    for dj, di in _OFFSETS:
        zs = _shift(z, dj, di)
        os_ = _shift(occ_f, dj, di)
        cross += float((z * zs * occ_f * os_).sum())
        deg += os_ * occ_f
    s0 = float(deg.sum())                      # Σ_ij w_ij (ordered pairs)
    if s0 <= 0:
        return MoranResult(i=0.0, expected=-1.0 / (n - 1), z=0.0,
                           n_cells=n, structured=False)

    i_val = (n / s0) * (cross / m2)
    e_i = -1.0 / (n - 1)

    # randomization variance (Cliff & Ord). Binary symmetric: S1=2·S0, S2=4·Σdeg².
    s1 = 2.0 * s0
    s2 = 4.0 * float((deg[occ] ** 2).sum())
    zi = z[occ]
    m2n = m2 / n
    b2 = (float((zi ** 4).sum()) / n) / (m2n ** 2)   # kurtosis
    a = n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 * s0)
    b = b2 * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 * s0)
    denom = (n - 1) * (n - 2) * (n - 3) * s0 * s0
    var = (a - b) / denom - e_i * e_i if denom > 0 else 0.0
    zscore = (i_val - e_i) / np.sqrt(var) if var > 1e-15 else 0.0

    structured = bool(zscore > z_max and i_val > i_min)
    return MoranResult(i=float(i_val), expected=float(e_i), z=float(zscore),
                       n_cells=n, structured=structured)


def coarsen(field: GridField, factor: int) -> GridField:
    """Block-reduce a GridField by ``factor`` with count-weighted means."""
    h, w = field.mean.shape
    hh = int(np.ceil(h / factor)) * factor
    ww = int(np.ceil(w / factor)) * factor
    tot = np.zeros((hh, ww))
    cnt = np.zeros((hh, ww), dtype=np.int64)
    tot[:h, :w] = np.where(field.occupied, field.mean * field.count, 0.0)
    cnt[:h, :w] = field.count
    tot = tot.reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
    cnt = cnt.reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = tot / cnt
    mean[cnt == 0] = np.nan
    return GridField(mean=mean, count=cnt, u0=field.u0, v0=field.v0,
                     cell=field.cell * factor)


def assess_field(field: GridField, z_max: float = 4.0, i_min: float = 0.05,
                 structure_min_mm: float = 1.5) -> MoranResult:
    """Full white-noise assessment: Moran's I (statistical) + structure
    amplitude (metric). ``relevant`` — the escalation/report verdict — is True
    only when the field is BOTH statistically structured and its amplitude
    matters against construction tolerances."""
    res = morans_i(field, z_max=z_max, i_min=i_min)
    if res.structured:
        res.amplitude_mm = structure_amplitude_mm(field)
        res.relevant = bool(res.amplitude_mm > structure_min_mm)
    return res


def structure_amplitude_mm(field: GridField, factor: int = 4) -> float:
    """Robust amplitude of the residual field's SPATIAL structure (mm):
    half the p2.5–p97.5 spread of coarsened cell means. Coarsening pools
    points so per-cell sampling noise shrinks and what remains is the
    systematic component — the thing tolerances care about."""
    c = coarsen(field, factor) if factor > 1 else field
    occ = c.occupied
    if occ.sum() < 4:
        return 0.0
    vals = c.mean[occ]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float((hi - lo) / 2.0 * 1000.0)


def _shift(a: np.ndarray, dj: int, di: int) -> np.ndarray:
    """Shift a 2-D array by (dj, di), zero-padding (no wraparound)."""
    out = np.zeros_like(a)
    h, w = a.shape
    js = slice(max(dj, 0), h + min(dj, 0))
    is_ = slice(max(di, 0), w + min(di, 0))
    jd = slice(max(-dj, 0), h + min(-dj, 0))
    id_ = slice(max(-di, 0), w + min(-di, 0))
    out[jd, id_] = a[js, is_]
    return out
