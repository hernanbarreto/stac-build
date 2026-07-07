# STAC-Builder — Phase R.6: metric hierarchy (inviolable).
#
# Scale & georeference authority = ChArUco/Umeyama 7-DoF + survey network, ALWAYS.
# Semantic anchors only refine internal consistency (relative poses, seams,
# drift). On conflict anchor-vs-marker, the MARKER wins and the conflict is
# logged with its magnitude. The 25%-most-confident DA3-Streaming scale S is a
# prior; S per window before/after refinement is reported (stability = diagnostic).
#
# PROVENANCE: ours. Umeyama is the standard 7-DoF similarity solution.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """7-DoF similarity (s,R,t) mapping src->dst (Umeyama 1991). src,dst (N,3)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    cov = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / var_s) if with_scale and var_s > 1e-12 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


@dataclass
class ConflictLog:
    conflicts: list[dict] = field(default_factory=list)

    def add(self, kind: str, magnitude: float, detail: dict):
        self.conflicts.append({"kind": kind, "magnitude": magnitude, **detail})


class MetricAuthority:
    """Marker/survey scale is authoritative. Anchors may only refine within a
    tolerance around it; beyond that the marker wins and the conflict is logged."""

    def __init__(self, marker_scale: float, tolerance_frac: float = 0.02):
        self.marker_scale = float(marker_scale)
        self.tolerance = tolerance_frac
        self.log = ConflictLog()

    def resolve_scale(self, anchor_scale: float, window_id: str | int = "global") -> float:
        rel = abs(anchor_scale - self.marker_scale) / max(self.marker_scale, 1e-9)
        if rel > self.tolerance:
            self.log.add("scale_anchor_vs_marker", rel,
                         {"window": window_id, "anchor": anchor_scale,
                          "marker": self.marker_scale, "resolution": "marker_wins"})
            return self.marker_scale  # marker authority
        return anchor_scale


@dataclass
class ScaleReport:
    """S per window before/after refinement — stability is a health diagnostic."""
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)

    def set(self, window_id, s_before: float, s_after: float):
        self.before[window_id] = float(s_before)
        self.after[window_id] = float(s_after)

    def stability(self) -> dict:
        keys = set(self.before) & set(self.after)
        deltas = {k: self.after[k] - self.before[k] for k in keys}
        vals = np.array(list(deltas.values())) if deltas else np.zeros(1)
        return {"per_window_delta": deltas, "max_abs_delta": float(np.abs(vals).max()),
                "std": float(vals.std())}
