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
    fid: int | None = None          # source frame (for window corrections)


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
    """Per-point vote over all points. Returns
    (assignments[N] int (-1 if none), entropies[N])."""
    assignments, entropies, _shares = vote_points_batch(points, views)
    return assignments, entropies


def vote_points_batch(points: np.ndarray, views: list[View]):
    """Vectorized multi-view plurality vote for all points at once.

    Projects the whole point set into every view in one shot and accumulates a
    dense (N, n_instances) count matrix. Returns:
      assignments[N] int  — plurality instance id (-1 when no view saw the point)
      entropies[N] float  — Shannon entropy of each point's vote distribution
      shares[N] float     — plurality fraction (votes for winner / total votes)
    """
    n = len(points)
    if n == 0 or not views:
        return (np.full(0 if n == 0 else n, -1, int), np.zeros(n), np.zeros(n))
    iids = sorted({iid for v in views for iid in v.masks})
    idx = {iid: k for k, iid in enumerate(iids)}
    counts = np.zeros((n, len(iids)), np.int32)
    ph = np.concatenate([points, np.ones((n, 1))], axis=1)  # (N,4)

    for view in views:
        cam = (np.linalg.inv(view.c2w) @ ph.T).T          # (N,4)
        z = cam[:, 2]
        valid = z > 1e-6
        u = np.where(valid, view.K[0, 0] * cam[:, 0] / np.where(valid, z, 1.0) + view.K[0, 2], -1)
        v = np.where(valid, view.K[1, 1] * cam[:, 1] / np.where(valid, z, 1.0) + view.K[1, 2], -1)
        w, h = view.wh
        inb = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not inb.any():
            continue
        ui, vi = u[inb].astype(int), v[inb].astype(int)
        rows = np.nonzero(inb)[0]
        for iid, mask in view.masks.items():
            mh, mw = mask.shape[:2]
            mu = np.minimum((ui * mw) // w, mw - 1)
            mv = np.minimum((vi * mh) // h, mh - 1)
            hit = mask[mv, mu].astype(bool)
            if hit.any():
                counts[rows[hit], idx[iid]] += 1

    total = counts.sum(axis=1)
    assignments = np.full(n, -1, int)
    entropies = np.zeros(n)
    shares = np.zeros(n)
    voted = total > 0
    if voted.any():
        win = counts[voted].argmax(axis=1)
        assignments[voted] = np.asarray(iids)[win]
        p = counts[voted] / total[voted, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(p > 0, np.log(p), 0.0)
        entropies[voted] = -(p * lp).sum(axis=1)
        shares[voted] = counts[voted].max(axis=1) / total[voted]
    return assignments, entropies, shares


def instance_entropy(points: np.ndarray, instance_id: int, views: list[View]):
    """Mean and max vote entropy over the points assigned to an instance."""
    assigns, ents = vote_points(points, views)
    mask = assigns == instance_id
    if not mask.any():
        return {"mean_entropy": 0.0, "max_entropy": 0.0, "n_points": 0}
    return {"mean_entropy": float(ents[mask].mean()),
            "max_entropy": float(ents[mask].max()),
            "n_points": int(mask.sum())}


def region_entropy(points: np.ndarray, entropies: np.ndarray,
                   cell_m: float = 1.0) -> dict[str, dict]:
    """Aggregate per-point vote entropy over a metric grid (spec R.2: entropy
    per REGION). Returns {"x_y_z": {"mean_entropy", "n_points"}} keyed by the
    cell's integer coordinates at `cell_m` resolution."""
    out: dict[str, dict] = {}
    if len(points) == 0:
        return out
    cells = np.floor(np.asarray(points, float) / max(cell_m, 1e-6)).astype(np.int64)
    order = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))
    cells, ents = cells[order], np.asarray(entropies, float)[order]
    uniq, starts = np.unique(cells, axis=0, return_index=True)
    bounds = np.append(starts, len(cells))
    for i, c in enumerate(uniq):
        seg = ents[bounds[i]:bounds[i + 1]]
        out[f"{c[0]}_{c[1]}_{c[2]}"] = {"mean_entropy": float(seg.mean()),
                                        "n_points": int(len(seg))}
    return out
