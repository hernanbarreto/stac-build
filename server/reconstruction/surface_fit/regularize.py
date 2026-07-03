"""
Stage 3 — global regularization across a scene's fitted PLANES (GlobFit
spirit): parallelism / orthogonality / verticality-horizontality snapping
within a configurable angular tolerance, coplanarity merging within an offset
tolerance, and exact intersections between adjacent planes → clean edges.

Honesty rule (charter): a deviation LARGER than the tolerance is never
forced — it may be a construction finding (desplome, misaligned wall), so it
stays measured, not "fixed". Snapping only absorbs deviations small enough to
be fit noise / construction intent, and every snapped surface's residual
report is recomputed afterwards against the original cloud, so whatever the
snap displaced shows up honestly in the deviation record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .plane import PlaneModel, _plane_basis

logger = logging.getLogger("SurfaceFit")

_EPS = 1e-12


@dataclass
class RegularizeReport:
    n_planes: int = 0
    clusters: List[dict] = field(default_factory=list)
    snapped: List[dict] = field(default_factory=list)      # per-plane action log
    coplanar_groups: List[List[int]] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)        # clean-edge pairs

    def to_dict(self) -> dict:
        return {"n_planes": self.n_planes, "clusters": self.clusters,
                "snapped": self.snapped, "coplanar_groups": self.coplanar_groups,
                "edges": self.edges}


def _canon(n: np.ndarray) -> np.ndarray:
    """Sign-canonical direction (first non-zero component positive)."""
    for c in n:
        if abs(c) > 1e-9:
            return n if c > 0 else -n
    return n


def regularize_planes(models: Sequence[PlaneModel],
                      weights: Optional[Sequence[float]] = None,
                      world_up: Optional[np.ndarray] = None,
                      angle_tol_deg: float = 1.0,
                      coplanar_tol_mm: float = 10.0
                      ) -> Tuple[List[PlaneModel], RegularizeReport]:
    """Snap a set of fitted planes to a consistent global frame.

    1. verticality/horizontality: normals within tol of ⟂/∥ world-up snap to it;
    2. parallelism: remaining normals cluster within tol → shared direction;
    3. orthogonality: cluster directions within tol of 90° to a heavier
       cluster rotate to exact 90°;
    4. coplanarity: same-direction planes with offsets within tol merge to a
       common offset (weighted).
    Returns NEW PlaneModel list (originals untouched) + an action report.
    """
    up = np.array([0.0, 0.0, 1.0]) if world_up is None else \
        np.asarray(world_up, dtype=np.float64)
    tol = np.deg2rad(angle_tol_deg)
    n_pl = len(models)
    rep = RegularizeReport(n_planes=n_pl)
    if n_pl == 0:
        return [], rep
    w = np.ones(n_pl) if weights is None else np.asarray(weights, dtype=np.float64)

    normals = [m.normal.copy() for m in models]

    # ── 1. snap to world-up (horizontal surfaces) or to the horizon (walls) ──
    for i, n in enumerate(normals):
        c = float(n @ up)
        ang_h = np.arccos(np.clip(abs(c), 0, 1))          # angle to up (0 = floor)
        if ang_h <= tol:                                  # floor/ceiling
            normals[i] = up * (1.0 if c >= 0 else -1.0)
            rep.snapped.append({"plane": i, "action": "horizontal",
                                "moved_deg": float(np.rad2deg(ang_h))})
        elif abs(np.pi / 2 - ang_h) <= tol:               # wall → exactly vertical
            nv = n - c * up
            normals[i] = nv / max(np.linalg.norm(nv), _EPS)
            rep.snapped.append({"plane": i, "action": "vertical",
                                "moved_deg": float(np.rad2deg(abs(np.pi / 2 - ang_h)))})

    # ── 2. parallelism: greedy weight-ordered clustering ──
    order = np.argsort(-w)
    cluster_of = [-1] * n_pl
    cluster_dirs: List[np.ndarray] = []
    cluster_w: List[float] = []
    for i in order:
        ni = _canon(normals[i])
        placed = False
        for cid, cd in enumerate(cluster_dirs):
            if np.arccos(np.clip(abs(ni @ cd), 0, 1)) <= tol:
                cluster_of[i] = cid
                # weighted running mean of the cluster direction
                cd2 = _canon(cd * cluster_w[cid] + ni * w[i])
                cluster_dirs[cid] = cd2 / max(np.linalg.norm(cd2), _EPS)
                cluster_w[cid] += w[i]
                placed = True
                break
        if not placed:
            cluster_of[i] = len(cluster_dirs)
            cluster_dirs.append(ni)
            cluster_w.append(float(w[i]))

    # ── 3. orthogonality between clusters (heavier cluster wins) ──
    heavy = np.argsort(-np.asarray(cluster_w))
    for a_i in range(len(heavy)):
        for b_i in range(a_i + 1, len(heavy)):
            a, b = heavy[a_i], heavy[b_i]
            dot = abs(float(cluster_dirs[a] @ cluster_dirs[b]))
            ang = np.arccos(np.clip(dot, 0, 1))
            if abs(np.pi / 2 - ang) <= tol and dot > _EPS:
                nb = cluster_dirs[b] - (cluster_dirs[b] @ cluster_dirs[a]) * cluster_dirs[a]
                cluster_dirs[b] = _canon(nb / max(np.linalg.norm(nb), _EPS))

    for cid, cd in enumerate(cluster_dirs):
        members = [i for i in range(n_pl) if cluster_of[i] == cid]
        rep.clusters.append({"dir": cd.tolist(), "planes": members,
                             "weight": float(cluster_w[cid])})

    # ── rebuild plane models with snapped normals (d re-anchored on origin) ──
    out: List[PlaneModel] = []
    for i, m in enumerate(models):
        cd = cluster_dirs[cluster_of[i]]
        n_new = cd * (1.0 if cd @ m.normal >= 0 else -1.0)
        moved = float(np.rad2deg(np.arccos(np.clip(m.normal @ n_new, -1, 1))))
        d_new = float(-n_new @ m.origin)
        u, v = _plane_basis(n_new, up)
        out.append(PlaneModel(normal=n_new, d=d_new, origin=m.origin.copy(),
                              u=u, v=v, rms=m.rms, inlier_frac=m.inlier_frac,
                              n_points=m.n_points))
        if moved > 1e-6:
            rep.snapped.append({"plane": i, "action": "align", "moved_deg": moved})

    # ── 4. coplanarity within each cluster ──
    ctol = coplanar_tol_mm / 1000.0
    for cid in range(len(cluster_dirs)):
        members = [i for i in range(n_pl) if cluster_of[i] == cid]
        used = set()
        for i_i, i in enumerate(members):
            if i in used:
                continue
            grp = [i]
            for j in members[i_i + 1:]:
                if j in used:
                    continue
                # same-orientation offsets (d sign follows the normal)
                di = out[i].d if out[i].normal @ cluster_dirs[cid] > 0 else -out[i].d
                dj = out[j].d if out[j].normal @ cluster_dirs[cid] > 0 else -out[j].d
                if abs(di - dj) <= ctol:
                    grp.append(j)
            if len(grp) > 1:
                dm = float(np.average(
                    [out[g].d * (1 if out[g].normal @ cluster_dirs[cid] > 0 else -1)
                     for g in grp], weights=[w[g] for g in grp]))
                for g in grp:
                    sign = 1.0 if out[g].normal @ cluster_dirs[cid] > 0 else -1.0
                    out[g].d = float(sign * dm)
                    out[g].origin = out[g].origin - \
                        (out[g].origin @ out[g].normal + out[g].d) * out[g].normal
                used.update(grp)
                rep.coplanar_groups.append(grp)

    n_moved = len({s['plane'] for s in rep.snapped})
    logger.info("regularize: %d planes → %d direction clusters, %d snapped, "
                "%d coplanar groups", n_pl, len(cluster_dirs), n_moved,
                len(rep.coplanar_groups))
    return out, rep


# ── exact intersections → clean edges ───────────────────────────────

def _boundary_vertices(faces: np.ndarray) -> np.ndarray:
    """Vertex ids lying on the mesh's open boundary (edges used by 1 face)."""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return np.unique(uniq[counts == 1])


def snap_edges(fitted_planes: List, snap_dist_m: float = 0.10,
               report: Optional[RegularizeReport] = None) -> int:
    """Pull mesh BOUNDARY vertices of adjacent plane pairs onto the exact
    intersection line, so walls meet floors/other walls with a clean edge
    instead of the support-radius gap or overlap.

    ``fitted_planes``: FittedSurface list whose .model is a PlaneModel and
    whose .mesh_vertices are mutated in place. Only open-boundary vertices
    within ``snap_dist_m`` of the OTHER plane move (each slides inside its own
    plane onto the intersection) — interior geometry never deforms and nothing
    extends further than the snap band, so no extrapolation.
    """
    n_edges = 0
    bnd_cache: Dict[int, np.ndarray] = {}

    def _bnd(idx: int, F) -> np.ndarray:
        if idx not in bnd_cache:
            bnd_cache[idx] = _boundary_vertices(F.mesh_faces)
        return bnd_cache[idx]

    for a_i in range(len(fitted_planes)):
        for b_i in range(a_i + 1, len(fitted_planes)):
            A, B = fitted_planes[a_i], fitted_planes[b_i]
            if A.model is None or B.model is None:
                continue
            if not isinstance(A.model, PlaneModel) or not isinstance(B.model, PlaneModel):
                continue
            nA, nB = A.model.normal, B.model.normal
            if np.linalg.norm(np.cross(nA, nB)) < 0.1:    # near-parallel: no edge
                continue
            moved = 0
            for idx, F, G in ((a_i, A, B), (b_i, B, A)):
                V = F.mesh_vertices
                if len(V) == 0:
                    continue
                bnd = _bnd(idx, F)
                if len(bnd) == 0:
                    continue
                sd = G.model.signed_distance(V[bnd])
                near = np.abs(sd) < snap_dist_m
                if not near.any():
                    continue
                # slide inside F's plane, along the direction that closes the
                # distance to G's plane → lands exactly on the intersection line
                nf, ng = F.model.normal, G.model.normal
                d_in = ng - (ng @ nf) * nf
                ln = np.linalg.norm(d_in)
                if ln < _EPS:
                    continue
                d_in = d_in / ln
                denom = float(ng @ d_in)
                if abs(denom) < 0.1:
                    continue
                ids = bnd[near]
                V[ids] = V[ids] - (sd[near] / denom)[:, None] * d_in
                moved += int(near.sum())
            if moved:
                n_edges += 1
                if report is not None:
                    report.edges.append({
                        "a": A.instance_id, "b": B.instance_id,
                        "moved_vertices": moved})
    logger.info("regularize: %d clean edges snapped", n_edges)
    return n_edges
