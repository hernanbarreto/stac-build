"""
surface_fit data contract — pure dataclasses + JSON, numpy-only (same spirit
as ``reconstruction/elements.py``: no Open3D types cross module boundaries).

All 3-D quantities are in the floor-aligned display world frame (metres).
Residuals and tolerances are reported in **millimetres** — that is the unit
site engineers reason in — but stored geometry stays metric (m).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from reconstruction.elements import json_safe


# ── residual acceptance / statistics ────────────────────────────────

@dataclass
class MoranResult:
    """Spatial autocorrelation of gridded residuals (white-noise test).

    ``structured`` is the escalation trigger: True when the residual field has
    spatial structure the current model failed to explain (both statistically
    significant z AND practically meaningful I — a huge N makes tiny I values
    "significant" without being actionable).
    """
    i: float                 # Moran's I ∈ [-1, 1]; ~E[I]≈0 for white noise
    expected: float          # E[I] = -1/(n-1)
    z: float                 # z-score under randomization assumption
    n_cells: int             # occupied grid cells used
    structured: bool         # statistical verdict (z AND I gates)
    amplitude_mm: float = 0.0   # robust amplitude of the spatial structure
    relevant: bool = False      # structured AND amplitude > structure_min_mm

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ResidualStats:
    """Point→surface residual summary. Signed distances, millimetres."""
    n_points: int
    rms_mm: float
    mean_mm: float
    std_mm: float
    p95_mm: float             # 95th percentile of |residual|
    max_mm: float             # max |residual|
    # Flatness/spec check: worst residual range inside any (span × span) window,
    # comparable to construction specs like "5 mm under a 2 m straightedge".
    flatness_worst_mm: Optional[float] = None
    flatness_tol_mm: Optional[float] = None
    flatness_span_m: Optional[float] = None
    flatness_pass: Optional[bool] = None
    moran: Optional[MoranResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["moran"] = self.moran.to_dict() if self.moran else None
        return json_safe(d)


@dataclass
class Finding:
    """A localized systematic deviation worth an engineer's attention
    (bulge/"panza", depression, plumb deviation/"desplome", …).

    Not an error of the fit — a *measurement* the smooth model exposes.
    """
    kind: str                 # 'bulge' | 'depression' | 'tilt'
    center_xyz: np.ndarray    # (3,) world, centroid of the deviating zone
    area_m2: float            # in-surface area of the zone (0 for global kinds)
    mean_dev_mm: float        # signed mean deviation of the zone
    peak_dev_mm: float        # signed extreme deviation of the zone
    n_points: int             # measured points inside the zone
    # kind == 'tilt': linear residual gradient along the reported direction.
    gradient_mm_per_m: Optional[float] = None
    gradient_dir: Optional[np.ndarray] = None   # (3,) world unit

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "kind": self.kind,
            "center_xyz": self.center_xyz,
            "area_m2": self.area_m2,
            "mean_dev_mm": self.mean_dev_mm,
            "peak_dev_mm": self.peak_dev_mm,
            "n_points": self.n_points,
            "gradient_mm_per_m": self.gradient_mm_per_m,
            "gradient_dir": self.gradient_dir,
        })


@dataclass
class ResidualReport:
    """Stage-4 deliverable: fidelity of a fitted surface vs the ORIGINAL
    (pre-consolidation) measured cloud."""
    stats: ResidualStats
    findings: List[Finding] = field(default_factory=list)
    # artifact paths (set by the exporter; None when not exported)
    heatmap_png: Optional[str] = None
    deviation_ply: Optional[str] = None
    json_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "stats": self.stats.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "heatmap_png": self.heatmap_png,
            "deviation_ply": self.deviation_ply,
        })


# ── fitted surface ──────────────────────────────────────────────────

@dataclass
class FittedSurface:
    """One segment's fitted smooth surface + its measurement backing.

    ``mesh_vertices``/``mesh_faces`` are already support-trimmed: every face
    is backed by measured points within ``support_radius`` (no extrapolation).
    """
    kind: str                       # 'plane' | 'cylinder' | 'sphere' | 'swept_profile' | 'bspline'
    params: Dict[str, Any]          # model parameters, JSON-safe, kind-specific
    mesh_vertices: np.ndarray       # (V,3) float64, world frame
    mesh_faces: np.ndarray          # (F,3) int
    report: ResidualReport
    instance_id: Optional[int] = None
    label: str = ""
    n_input_points: int = 0
    support_fraction: float = 1.0   # fraction of the model's UV domain with point support
    escalation_path: List[str] = field(default_factory=list)  # models tried, in order
    dof: int = 0                    # degrees of freedom of the accepted model
    # runtime-only (never serialized): the fitted surface-protocol model, so
    # scene-level stages (regularization, edge snapping) can re-derive geometry
    model: Optional[object] = None
    regularized: bool = False       # True when stage 3 adjusted the params

    def to_meta(self) -> Dict[str, Any]:
        """meta.json payload (mesh itself is stored as a file next to it)."""
        return json_safe({
            "method": "surface_fit",
            "kind": self.kind,
            "dof": self.dof,
            "params": self.params,
            "instance_id": self.instance_id,
            "label": self.label,
            "n_input_points": self.n_input_points,
            "n_vertices": int(len(self.mesh_vertices)),
            "n_faces": int(len(self.mesh_faces)),
            "support_fraction": self.support_fraction,
            "escalation_path": self.escalation_path,
            "regularized": self.regularized,
            "residuals": self.report.to_dict(),
        })


# ── shared small helpers ────────────────────────────────────────────

def orient_normal(normal: np.ndarray, world_up: np.ndarray,
                  points: Optional[np.ndarray] = None,
                  centroid_hint: Optional[np.ndarray] = None) -> np.ndarray:
    """Deterministic normal orientation so residual signs are comparable
    across runs: towards +up for horizontal-ish surfaces; otherwise away from
    ``centroid_hint`` (e.g. the whole-scene centroid → walls point into the
    room being surveyed); last resort, positive on the first non-zero axis."""
    n = np.asarray(normal, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)
    if abs(n @ up) > 0.5:
        return n if n @ up > 0 else -n
    if centroid_hint is not None and points is not None and len(points):
        outward = np.asarray(points).mean(0) - np.asarray(centroid_hint)
        if np.linalg.norm(outward) > 1e-9:
            return n if n @ outward > 0 else -n
    for c in n:
        if abs(c) > 1e-9:
            return n if c > 0 else -n
    return n
