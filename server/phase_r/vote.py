# STAC-Builder — Phase R.2: point->instance plurality vote + entropy.
#
# With provisional window poses and DA3 depth, each candidate 3D point is
# projected into every view; it votes for the instance whose (eroded) mask it
# lands in. The per-point vote distribution -> Shannon ENTROPY is the
# misalignment metric (spec R.2). Aggregated per instance and per region it is
# persisted as the alignment-health signal.
#
# PROVENANCE: adapted from R3D's multi-view consensus filter
# (build_scene.py:_multiview_consensus_filter :97) — generalized from a single
# object's >50% consensus to a multi-instance plurality vote with entropy.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def vote_entropy(counts: dict[int, int] | np.ndarray) -> float:
    """Shannon entropy (nats) of a vote distribution. 0 = unanimous."""
    c = np.asarray(list(counts.values()) if isinstance(counts, dict) else counts, float)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log(p)).sum())


@dataclass
class View:
    c2w: np.ndarray                 # (4,4)
    K: np.ndarray                   # (3,3) at mask resolution
    masks: dict[int, np.ndarray]    # instance_id -> bool mask (H,W)
    wh: tuple[int, int]             # (W,H)


def _project(point: np.ndarray, view: View):
    cam = np.linalg.inv(view.c2w) @ np.append(point, 1.0)
    z = cam[2]
    if z <= 1e-6:
        return None
    u = view.K[0, 0] * cam[0] / z + view.K[0, 2]
    v = view.K[1, 1] * cam[1] / z + view.K[1, 2]
    w, h = view.wh
    if 0 <= u < w and 0 <= v < h:
        return int(u), int(v)
    return None


def vote_point(point: np.ndarray, views: list[View]) -> tuple[int | None, dict[int, int], float]:
    """Return (assigned_instance, vote_counts, entropy) for one 3D point."""
    counts: dict[int, int] = {}
    for view in views:
        uv = _project(point, view)
        if uv is None:
            continue
        u, v = uv
        for iid, mask in view.masks.items():
            mh, mw = mask.shape[:2]
            vu = min(int(u * mw / view.wh[0]), mw - 1)
            vv = min(int(v * mh / view.wh[1]), mh - 1)
            if mask[vv, vu]:
                counts[iid] = counts.get(iid, 0) + 1
    if not counts:
        return None, {}, 0.0
    assigned = max(counts, key=counts.get)
    return assigned, counts, vote_entropy(counts)


def vote_points(points: np.ndarray, views: list[View]):
    """Vectorized-ish per-point vote over all points. Returns
    (assignments[N] int (-1 if none), entropies[N])."""
    assignments = np.full(len(points), -1, int)
    entropies = np.zeros(len(points))
    for i, p in enumerate(points):
        a, _c, e = vote_point(p, views)
        assignments[i] = a if a is not None else -1
        entropies[i] = e
    return assignments, entropies


def instance_entropy(points: np.ndarray, instance_id: int, views: list[View]):
    """Mean and max vote entropy over the points assigned to an instance."""
    assigns, ents = vote_points(points, views)
    mask = assigns == instance_id
    if not mask.any():
        return {"mean_entropy": 0.0, "max_entropy": 0.0, "n_points": 0}
    return {"mean_entropy": float(ents[mask].mean()),
            "max_entropy": float(ents[mask].max()),
            "n_points": int(mask.sum())}
