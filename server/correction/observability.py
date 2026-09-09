"""Observable-DOF analysis of the evidence, BEFORE solving.

Principle (prompt §5.4): a solve may only move the degrees of freedom the
evidence actually observes. A planar object cannot observe translation inside
its plane nor rotation about its normal; an elongated (cylindrical) object
cannot observe translation along its axis (nor rotation about it — which the
yaw-planar solver never has anyway); a compact asymmetric object, or two
objects with enough baseline, observe the solver's full DOF set (yaw + 3-D
translation, USER 2026-09-06: full 3-D rotation is deliberately out).

What it decides, per displaced visit:
  * a shape class per contributing object (PCA eigenvalue ratios),
  * the observable DOF set → a projection spec the solver enforces,
  * whether depth (k) evidence exists (≥ min_objects_for_depth objects with
    pairwise baseline ≥ min_baseline_m),
  * a rejection with an actionable "mark also …" suggestion when the evidence
    cannot constrain any useful DOF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from correction.config import CorrectionConfig

SHAPE_PLANAR = "planar"
SHAPE_LINEAR = "linear"
SHAPE_COMPACT = "compact"


@dataclass
class ObjectShape:
    iid: int
    label: str
    shape: str
    eig_ratios: List[float]          # [λ2/λ1, λ3/λ1]
    normal: Optional[np.ndarray]     # planar: unit normal
    axis: Optional[np.ndarray]       # linear: unit main axis

    def summary(self) -> dict:
        return {"iid": self.iid, "label": self.label, "shape": self.shape,
                "eig_ratios": [round(float(r), 4) for r in self.eig_ratios]}


@dataclass
class VisitObservability:
    ok: bool
    dof: List[str]                    # observable, e.g. ["yaw","tx","ty","tz"]
    unrestrained: List[str]           # declared free (forced to identity)
    projection: dict                  # spec consumed by solve.project_solution
    depth_allowed: bool
    objects: List[ObjectShape]
    baselines: Dict[str, float]
    suggestion: Optional[str]         # actionable when not ok / partial

    def summary(self) -> dict:
        return {
            "ok": self.ok, "dof": self.dof, "unrestrained": self.unrestrained,
            "depth_allowed": self.depth_allowed,
            "objects": [o.summary() for o in self.objects],
            "baselines": {k: round(v, 3) for k, v in self.baselines.items()},
            "suggestion": self.suggestion,
        }


def classify_object(P: np.ndarray, iid: int, label: str,
                    cfg: CorrectionConfig) -> ObjectShape:
    """PCA shape class of one object's evidence points."""
    c = P.mean(0)
    U = P - c
    # eigenvalues of the covariance via singular values
    sv = np.linalg.svd(U, full_matrices=False)[1]
    ev = (sv ** 2) / max(len(P) - 1, 1)
    ev = np.sort(ev)[::-1]
    r21 = float(ev[1] / ev[0]) if ev[0] > 0 else 0.0
    r31 = float(ev[2] / ev[0]) if ev[0] > 0 else 0.0
    axes = np.linalg.svd(U, full_matrices=False)[2]
    if r31 < cfg.observability.pca_ratio_planar \
            and r21 >= cfg.observability.pca_ratio_cylindrical:
        return ObjectShape(iid, label, SHAPE_PLANAR, [r21, r31],
                           normal=axes[2] / np.linalg.norm(axes[2]),
                           axis=None)
    if r21 < cfg.observability.pca_ratio_cylindrical:
        return ObjectShape(iid, label, SHAPE_LINEAR, [r21, r31], normal=None,
                           axis=axes[0] / np.linalg.norm(axes[0]))
    return ObjectShape(iid, label, SHAPE_COMPACT, [r21, r31], None, None)


def bounded_copies(P_ref: np.ndarray, P_disp: np.ndarray,
                   cfg: CorrectionConfig) -> bool:
    """True when the two copies of an object have the SAME supported extents
    (p5..p95 along the displaced copy's PCA axes, within
    ``bounded_extent_tol``): a bounded, equally-covered object observes the
    full translation through its edges even when its surface is planar or
    elongated (an infinite plane would not — a real desk does)."""
    if len(P_ref) < 3 or len(P_disp) < 3:
        return False
    c = P_disp.mean(0)
    axes = np.linalg.svd(P_disp - c, full_matrices=False)[2]
    ed = (P_disp - c) @ axes.T
    er = (P_ref - P_ref.mean(0)) @ axes.T
    for i in range(3):
        a = float(np.percentile(ed[:, i], 95) - np.percentile(ed[:, i], 5))
        b = float(np.percentile(er[:, i], 95) - np.percentile(er[:, i], 5))
        big = max(a, b)
        if big <= 0:
            continue
        if abs(a - b) / big > cfg.observability.bounded_extent_tol:
            return False
    return True


def analyze_visit(shapes: List[ObjectShape],
                  centroids: Dict[int, np.ndarray],
                  cfg: CorrectionConfig,
                  bounded: Optional[Dict[int, bool]] = None
                  ) -> VisitObservability:
    """Observable DOF of one displaced visit from its objects' shapes and
    pairwise baselines. ``bounded``: per object, whether its two copies
    share the same supported extents (see ``bounded_copies``)."""
    bounded = bounded or {}
    iids = sorted(centroids.keys())
    baselines: Dict[str, float] = {}
    n_baseline_ok = 0
    for i in range(len(iids)):
        for j in range(i + 1, len(iids)):
            d = float(np.linalg.norm(centroids[iids[i]] - centroids[iids[j]]))
            baselines[f"{iids[i]}-{iids[j]}"] = d
            if d >= cfg.evidence.min_baseline_m:
                n_baseline_ok += 1

    n_objects_spread = (len(iids) if n_baseline_ok >= 1 else 1)
    depth_allowed = (len(iids) >= cfg.evidence.min_objects_for_depth
                     and n_baseline_ok >= 1)

    full_dof = ["yaw", "tx", "ty", "tz"]
    # Multiple spread objects, or at least one compact asymmetric object:
    # the solver's full DOF set is observable.
    if n_objects_spread >= 2 or any(s.shape == SHAPE_COMPACT for s in shapes):
        return VisitObservability(
            ok=True, dof=full_dof, unrestrained=[],
            projection={"mode": "full"}, depth_allowed=depth_allowed,
            objects=shapes, baselines=baselines,
            suggestion=(None if depth_allowed else
                        "mark also a second object ≥ "
                        f"{cfg.evidence.min_baseline_m:g} m away to enable a "
                        f"depth (scale) diagnosis"))

    s0 = shapes[0]
    # Single BOUNDED object (planar or elongated) with equal coverage in
    # both copies: its edges observe the full translation; yaw stays free
    # (a rectangle can flip, a column is symmetric).
    if s0.shape in (SHAPE_PLANAR, SHAPE_LINEAR) and bounded.get(s0.iid):
        return VisitObservability(
            ok=True, dof=["tx", "ty", "tz"], unrestrained=["yaw"],
            projection={"mode": "translation"}, depth_allowed=False,
            objects=shapes, baselines=baselines,
            suggestion=("mark also a second object (≥ "
                        f"{cfg.evidence.min_baseline_m:g} m away) to observe "
                        "yaw and enable a depth diagnosis"))

    # Single planar object: only translation along its normal is observed.
    if s0.shape == SHAPE_PLANAR:
        return VisitObservability(
            ok=True, dof=["t_normal"],
            unrestrained=["yaw", "t_inplane(2)"],
            projection={"mode": "normal", "normal": s0.normal.tolist()},
            depth_allowed=False, objects=shapes, baselines=baselines,
            suggestion=("mark also a compact asymmetric object, or a second "
                        "object in the same visit, to observe yaw and "
                        "in-plane translation"))

    # Single elongated object: translation ⊥ its axis (2 DOF); yaw only when
    # the axis is vertical would still spin the object into itself → free.
    if s0.shape == SHAPE_LINEAR:
        return VisitObservability(
            ok=True, dof=["t_perp_axis(2)"],
            unrestrained=["yaw", "t_along_axis"],
            projection={"mode": "perp_axis", "axis": s0.axis.tolist()},
            depth_allowed=False, objects=shapes, baselines=baselines,
            suggestion=("mark also a compact asymmetric object, or a second "
                        "object in the same visit, to observe yaw and the "
                        "along-axis translation"))

    # No usable shape at all.
    return VisitObservability(
        ok=False, dof=[], unrestrained=full_dof,
        projection={"mode": "none"}, depth_allowed=False, objects=shapes,
        baselines=baselines,
        suggestion="mark a compact asymmetric object, or at least two "
                   "objects with sufficient baseline, in this visit")
