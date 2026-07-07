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


# ── inter-window pose graph ─────────────────────────────────────────
@dataclass
class WindowEdge:
    win_k: int
    win_l: int
    measured: Sim3   # relative Sim3 from window k to l for a shared instance
    weight: float = 1.0


def optimize_window_graph(n_windows: int, edges: list[WindowEdge],
                          huber_delta: float = 0.05, max_nfev: int = 200):
    """Solve per-window Sim(3) corrections X_w (X_0 = identity fixed) minimizing
    robust residuals r = log( X_k^{-1} · X_l · M_kl ) over all edges.
    Returns (corrections: list[Sim3], stats)."""
    from scipy.optimize import least_squares

    free = list(range(1, n_windows))  # window 0 fixed as reference
    idx_of = {w: i for i, w in enumerate(free)}

    def unpack(params) -> dict[int, Sim3]:
        X = {0: Sim3.identity()}
        for w in free:
            X[w] = Sim3.exp(params[7 * idx_of[w]:7 * idx_of[w] + 7])
        return X

    def resid(params):
        X = unpack(params)
        out = []
        for e in edges:
            r = X[e.win_k].inverse().compose(X[e.win_l]).compose(e.measured)
            out.extend(np.sqrt(e.weight) * r.log())
        return np.asarray(out) if out else np.zeros(1)

    x0 = np.zeros(7 * len(free))
    if not edges or not free:
        return [Sim3.identity() for _ in range(n_windows)], {"cost0": 0.0, "cost": 0.0, "success": True}

    cost0 = 0.5 * float(np.sum(resid(x0) ** 2))
    sol = least_squares(resid, x0, loss="huber", f_scale=huber_delta, max_nfev=max_nfev)
    X = unpack(sol.x)
    corrections = [X[w] for w in range(n_windows)]
    return corrections, {"cost0": cost0, "cost": float(sol.cost), "success": bool(sol.success),
                         "n_edges": len(edges)}
