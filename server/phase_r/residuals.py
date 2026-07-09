# STAC-Builder — Phase R.4: per-instance Sim(3) residuals + inter-window pose graph.
#
# The same instance seen in windows k and k+1 gives a Sim(3) residual between its
# gravity-aligned OBBs (centre + axes + extents). We optimize a Sim(3) correction
# PER WINDOW (not dense BA) that minimizes the robust (Huber) residuals across all
# shared instances — the 2-pass in-window BA stays untouched (spec R.4).
#
# Sim(3) = (s, R, t), apply(x) = s·R·x + t. Minimal 7-param log = (log s, rotvec, t).
#
# PROVENANCE: ours. OBBs from the R3D-ported fitter; robust optimization is a
# standard Sim(3) pose-graph solve (scipy least_squares + Huber).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# ── Sim(3) algebra ──────────────────────────────────────────────────
@dataclass
class Sim3:
    s: float
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.s * (self.R @ x) + self.t

    def compose(self, o: "Sim3") -> "Sim3":
        return Sim3(self.s * o.s, self.R @ o.R, self.s * (self.R @ o.t) + self.t)

    def inverse(self) -> "Sim3":
        Rinv = self.R.T
        return Sim3(1.0 / self.s, Rinv, -(1.0 / self.s) * (Rinv @ self.t))

    def log(self) -> np.ndarray:
        return np.concatenate([[np.log(self.s)], Rotation.from_matrix(self.R).as_rotvec(), self.t])

    @staticmethod
    def exp(v: np.ndarray) -> "Sim3":
        return Sim3(float(np.exp(v[0])), Rotation.from_rotvec(v[1:4]).as_matrix(), np.asarray(v[4:7], float))

    def magnitude(self, rot_w: float = 1.0, scale_w: float = 1.0) -> float:
        ang = np.linalg.norm(Rotation.from_matrix(self.R).as_rotvec())
        return float(np.linalg.norm(self.t) + rot_w * ang + scale_w * abs(np.log(self.s)))


def sim3_from_obb_pair(T_a: np.ndarray, aabb_a: np.ndarray,
                       T_b: np.ndarray, aabb_b: np.ndarray) -> Sim3:
    """Sim(3) mapping instance OBB pose a -> b (centre + axes + extents).
    Identity when the two windows already agree on the instance."""
    Ra, Rb = T_a[:3, :3], T_b[:3, :3]
    ca, cb = T_a[:3, 3], T_b[:3, 3]
    ext_a = np.array([aabb_a[1] - aabb_a[0], aabb_a[3] - aabb_a[2], aabb_a[5] - aabb_a[4]])
    ext_b = np.array([aabb_b[1] - aabb_b[0], aabb_b[3] - aabb_b[2], aabb_b[5] - aabb_b[4]])
    ext_a = np.where(ext_a < 1e-6, 1e-6, ext_a)
    s = float(np.mean(ext_b / ext_a))
    R = Rb @ Ra.T
    t = cb - s * (R @ ca)
    return Sim3(s, R, t)


# ── R.4 fine primitives (plane / cylinder) for structural classes ───
# The OBB residual is a coarse constraint and biased under partial observation
# (a wall half-seen from window k has a shifted centre/extents vs window k+1).
# For STRUCTURAL classes the spec adds the fine surface_fitting primitive:
#   plane-like  (wall/slab/vault/platform/floor) → plane (n, d)
#   column-like (column)                         → cylinder axis (a, c, r)
# A primitive pair between two windows contributes residuals directly in the
# graph cost — normal/axis misalignment (rad) + offset/axis-distance (m) —
# which are unbiased by how much of the surface each window saw.
@dataclass
class PlanePrim:
    n: np.ndarray   # unit normal
    d: float        # offset: n·x + d = 0


@dataclass
class CylinderPrim:
    a: np.ndarray   # unit axis
    c: np.ndarray   # point on the axis (centroid projection)
    r: float        # radius (m)


def fit_plane_primitive(points: np.ndarray) -> PlanePrim | None:
    """Least-squares plane over the instance's window points (SVD)."""
    pts = np.asarray(points, float)
    if len(pts) < 10:
        return None
    c = pts.mean(0)
    _u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    if sv[1] < 1e-9:           # degenerate (a line, not a surface)
        return None
    n = vt[2] / (np.linalg.norm(vt[2]) + 1e-12)
    return PlanePrim(n, -float(n @ c))


def fit_cylinder_primitive(points: np.ndarray) -> CylinderPrim | None:
    """Cylinder axis via PCA (largest-variance direction), radius as the median
    radial distance. Deterministic, robust enough for column-like instances."""
    pts = np.asarray(points, float)
    if len(pts) < 10:
        return None
    c = pts.mean(0)
    _u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    if sv[0] < 1e-9:
        return None
    a = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    d = pts - c
    radial = d - np.outer(d @ a, a)
    r = float(np.median(np.linalg.norm(radial, axis=1)))
    if not np.isfinite(r) or r < 1e-4:
        return None
    return CylinderPrim(a, c, r)


def transform_plane(p: PlanePrim, M: Sim3) -> PlanePrim:
    """Plane under y = s·R·x + t:  n' = R·n,  d' = s·d − n'·t (n' stays unit)."""
    n2 = M.R @ p.n
    return PlanePrim(n2, float(M.s * p.d - n2 @ M.t))


def transform_cylinder(cy: CylinderPrim, M: Sim3) -> CylinderPrim:
    """Cylinder under a Sim(3): axis rotates, centre maps fully, radius scales."""
    return CylinderPrim(M.R @ cy.a, M.apply(cy.c), float(M.s * cy.r))


def primitive_residual(pk, pl, kind: str) -> np.ndarray:
    """Residual between the SAME instance's primitive seen from two windows,
    both already mapped into the common frame. Zero when the windows agree.
    plane    → [n_k × n_l (3, rad), d_k − d_l (m)]
    cylinder → [a_k × a_l (3, rad), ⊥ axis-line distance (3, m), r_k − r_l (m)]
    Units match the Sim(3) log residuals (rad + m) → directly comparable cost."""
    if kind == "plane":
        n_k, n_l, d_l = pk.n, pl.n, pl.d
        if float(n_k @ n_l) < 0:            # sign-align the normals
            n_l, d_l = -n_l, -d_l
        return np.concatenate([np.cross(n_k, n_l), [pk.d - d_l]])
    # cylinder
    a_k, a_l = pk.a, pl.a
    if float(a_k @ a_l) < 0:
        a_l = -a_l
    a_m = a_k + a_l
    a_m = a_m / (np.linalg.norm(a_m) + 1e-12)
    dc = pl.c - pk.c
    e_perp = dc - (dc @ a_m) * a_m          # displacement ⊥ to the axis (m)
    return np.concatenate([np.cross(a_k, a_l), e_perp, [pl.r - pk.r]])


@dataclass
class PrimitiveEdge:
    win_k: int
    win_l: int
    prim_k: object   # PlanePrim | CylinderPrim in window k's current frame
    prim_l: object   # same instance's primitive in window l's current frame
    kind: str        # "plane" | "cylinder"
    weight: float = 1.0


# ── inter-window pose graph ─────────────────────────────────────────
@dataclass
class WindowEdge:
    win_k: int
    win_l: int
    measured: Sim3   # relative Sim3 from window k to l for a shared instance
    weight: float = 1.0


def optimize_window_graph(n_windows: int, edges: list[WindowEdge],
                          huber_delta: float = 0.05, max_nfev: int = 200,
                          scale_priors: dict[int, float] | None = None,
                          scale_prior_weight: float = 4.0,
                          primitive_edges: list[PrimitiveEdge] | None = None,
                          primitive_weight: float = 1.0,
                          min_window_edges: int = 2):
    """Solve per-window Sim(3) corrections X_w (X_0 = identity fixed) minimizing
    robust residuals r = log( X_k^{-1} · X_l · M_kl ) over all OBB edges PLUS the
    fine primitive residuals (plane/cylinder pairs mapped by X_k / X_l — spec
    R.4 "primitiva fina" for structural classes; unbiased by partial views).

    scale_priors (R.6): expected correction scale per window index (the DA3
    top-confidence scale S per window is kept as a prior, so corrections are
    pulled toward it — 1.0 = "do not rescale"). Soft residual
    sqrt(w)·(log s_w − log prior_w) per prior.
    Returns (corrections: list[Sim3], stats)."""
    from scipy.optimize import least_squares

    prim_edges = primitive_edges or []
    # UNDERDETERMINATION GUARD: a Sim(3) has 7 DoF; one edge is exactly 7
    # residuals — a window supported by a single (possibly bad) edge is solved
    # exactly ONTO that edge's noise. test2 2026-07-09: 5 OBB edges over 11
    # windows produced s=0.055 / |t|=3.15m "corrections" that the R.9 gate had
    # to throw away. Windows with < min_window_edges incident constraints stay
    # LOCKED at identity (no free params); their edges still constrain their
    # neighbours.
    incident = {w: 0 for w in range(n_windows)}
    for e in edges:
        incident[e.win_k] += 1
        incident[e.win_l] += 1
    for pe in prim_edges:
        incident[pe.win_k] += 1
        incident[pe.win_l] += 1
    locked = {w for w in range(1, n_windows) if incident[w] < min_window_edges}
    free = [w for w in range(1, n_windows) if w not in locked]
    idx_of = {w: i for i, w in enumerate(free)}
    priors = {w: s for w, s in (scale_priors or {}).items() if w in idx_of and s > 0}

    def unpack(params) -> dict[int, Sim3]:
        X = {w: Sim3.identity() for w in range(n_windows) if w == 0 or w in locked}
        for w in free:
            X[w] = Sim3.exp(params[7 * idx_of[w]:7 * idx_of[w] + 7])
        return X

    def resid(params):
        X = unpack(params)
        out = []
        for e in edges:
            r = X[e.win_k].inverse().compose(X[e.win_l]).compose(e.measured)
            out.extend(np.sqrt(e.weight) * r.log())
        for pe in prim_edges:
            if pe.kind == "plane":
                pk = transform_plane(pe.prim_k, X[pe.win_k])
                pl = transform_plane(pe.prim_l, X[pe.win_l])
            else:
                pk = transform_cylinder(pe.prim_k, X[pe.win_k])
                pl = transform_cylinder(pe.prim_l, X[pe.win_l])
            out.extend(np.sqrt(pe.weight * primitive_weight)
                       * primitive_residual(pk, pl, pe.kind))
        for w, prior in priors.items():
            out.append(np.sqrt(scale_prior_weight)
                       * (params[7 * idx_of[w]] - np.log(prior)))
        return np.asarray(out) if out else np.zeros(1)

    x0 = np.zeros(7 * len(free))
    if (not edges and not prim_edges) or not free:
        return [Sim3.identity() for _ in range(n_windows)], {"cost0": 0.0, "cost": 0.0,
                                                             "success": True,
                                                             "n_edges": len(edges),
                                                             "n_primitive_edges": len(prim_edges),
                                                             "locked_windows": sorted(locked)}

    cost0 = 0.5 * float(np.sum(resid(x0) ** 2))
    sol = least_squares(resid, x0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev)
    X = unpack(sol.x)
    corrections = [X[w] for w in range(n_windows)]
    return corrections, {"cost0": cost0, "cost": float(sol.cost), "success": bool(sol.success),
                         "n_edges": len(edges), "n_primitive_edges": len(prim_edges),
                         "locked_windows": sorted(locked)}
