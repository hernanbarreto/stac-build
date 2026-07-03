"""
Stage 2.2 — quadrics (cylinder / sphere), the first escalation above a plane.

Cylinder reuses ``reconstruction.geometry.primitives.fit_cylinder_ransac``
(normal-scatter axis + RANSAC circle ⊥ axis); the sphere is a Kåsa algebraic
fit refined by robust least squares. Both are wrapped in the common surface
protocol (signed_distance / to_uv / uv_to_world / params_dict / DOF) with an
UNROLLED, metric UV parametrization — u,v are lengths on the surface, so the
same residual grids, Moran test, findings, support trim and quad meshing used
for planes apply unchanged.

Degenerate-quadric guard: a radius much larger than the patch means the model
is a plane in disguise — the fitters reject it (``max_radius_factor``) so
escalation never "explains" planar data with a giant cylinder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from reconstruction.geometry.primitives import (
    CylinderFit, _orthobasis, estimate_normals, fit_cylinder_ransac)

_EPS = 1e-12
_TWO_PI = 2.0 * np.pi

# radius > this × patch extent ⇒ degenerate (a plane would do) → reject
MAX_RADIUS_FACTOR = 8.0


# ── cylinder ────────────────────────────────────────────────────────

@dataclass
class CylinderModel:
    """Circular cylinder: axis (point+dir), radius, with the surface unrolled
    to UV = (s along axis, r·Δθ along the arc), anchored at θ0 so partial
    cylinders (vaults, curved walls) map to a contiguous rectangle."""
    axis_point: np.ndarray     # (3,) foot of axis
    axis_dir: np.ndarray       # (3,) unit
    radius: float
    theta_ref: np.ndarray      # (3,) unit ⊥ axis, θ=0
    theta0: float              # start of angular extent (rad, from theta_ref)
    theta_span: float          # angular extent covered by data (rad)
    rms: float
    inlier_frac: float
    n_points: int

    DOF = 5  # axis line (4) + radius (1)

    def _frame(self):
        w = self.axis_dir
        u = self.theta_ref
        v = np.cross(w, u)
        return u, v, w

    def _cyl_coords(self, pts: np.ndarray):
        u, v, w = self._frame()
        q = np.asarray(pts, dtype=np.float64) - self.axis_point
        s = q @ w
        x = q @ u
        y = q @ v
        rho = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return s, rho, theta

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        _, rho, _ = self._cyl_coords(pts)
        return rho - self.radius          # + outside, − inside

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        s, _, theta = self._cyl_coords(pts)
        # unwrap against the arc start so the patch is contiguous in v
        dt = np.mod(theta - self.theta0, _TWO_PI)
        return np.column_stack([s, self.radius * dt])

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        u, v, w = self._frame()
        theta = self.theta0 + uv[:, 1] / max(self.radius, _EPS)
        radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        return self.axis_point + uv[:, 0:1] * w + self.radius * radial

    # outward normal at surface points (for future texturing/regularization)
    def normal_at_uv(self, uv: np.ndarray) -> np.ndarray:
        u, v, _ = self._frame()
        theta = self.theta0 + np.asarray(uv)[:, 1] / max(self.radius, _EPS)
        return np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v

    def params_dict(self) -> dict:
        return {
            "axis_point": self.axis_point.tolist(),
            "axis_dir": self.axis_dir.tolist(),
            "radius": float(self.radius),
            "theta_ref": self.theta_ref.tolist(),
            "theta0": float(self.theta0),
            "theta_span": float(self.theta_span),
            "rms_m": float(self.rms),
            "inlier_frac": float(self.inlier_frac),
        }


def fit_cylinder(points: np.ndarray, dist_thresh: float = 0.02,
                 min_inlier_frac: float = 0.5,
                 axis_hint: Optional[np.ndarray] = None) -> Optional[CylinderModel]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 50:
        return None
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    cf: Optional[CylinderFit] = fit_cylinder_ransac(
        pts, dist_thresh=float(dist_thresh), axis_hint=axis_hint,
        min_inlier_frac=float(min_inlier_frac),
        max_radius=MAX_RADIUS_FACTOR * extent)
    if cf is None:
        return None
    if cf.radius > MAX_RADIUS_FACTOR * extent:      # plane in disguise
        return None
    t0, t1 = cf.theta_extent
    model = CylinderModel(axis_point=cf.axis_point, axis_dir=cf.axis_dir,
                          radius=cf.radius, theta_ref=cf.theta_ref,
                          theta0=float(t0), theta_span=float(t1 - t0),
                          rms=cf.rms, inlier_frac=cf.inlier_frac,
                          n_points=cf.n_points)
    model = _refine_cylinder(model, pts, dist_thresh)
    d = model.signed_distance(pts)
    inl = np.abs(d) < dist_thresh
    model.inlier_frac = float(inl.mean())
    model.rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("inf")
    if model.inlier_frac < min_inlier_frac:
        return None
    return model


def _refine_cylinder(model: CylinderModel, pts: np.ndarray,
                     dist_thresh: float) -> CylinderModel:
    """Residual-harmonic refinement of axis / centre / radius.

    The normal-scatter axis carries milliradian error; over a long cylinder
    that displaces the section centre linearly with s and leaves a dipole
    residual r ≈ (e0+e1·s)·(cosθ, sinθ) + Δr — spatial structure a true
    cylinder must not have. One linear LS over all inlier points estimates
    (Δr, e0, e1) with the full N-point redundancy; two iterations converge.
    """
    for _ in range(3):
        r = model.signed_distance(pts)
        keep = np.abs(r) < max(3.0 * dist_thresh, 0.05)   # ignore far outliers
        if keep.sum() < 100:
            break
        s, _, theta = model._cyl_coords(pts[keep])
        s = s - s.mean()
        ct, st = np.cos(theta), np.sin(theta)
        M = np.column_stack([np.ones_like(s), ct, st, s * ct, s * st])
        try:
            coef, *_ = np.linalg.lstsq(M, r[keep], rcond=None)
        except np.linalg.LinAlgError:
            break
        dr, e0x, e0y, e1x, e1y = (float(v) for v in coef)
        u, v, w = model._frame()
        model.radius = model.radius + dr
        model.axis_point = model.axis_point + e0x * u + e0y * v
        new_axis = w + e1x * u + e1y * v
        model.axis_dir = new_axis / np.linalg.norm(new_axis)
        # keep theta_ref ⊥ the new axis
        tr = model.theta_ref - (model.theta_ref @ model.axis_dir) * model.axis_dir
        model.theta_ref = tr / max(np.linalg.norm(tr), _EPS)
        if max(abs(dr), np.hypot(e0x, e0y), np.hypot(e1x, e1y)) < 1e-6:
            break
    return model


# ── sphere ──────────────────────────────────────────────────────────

@dataclass
class SphereModel:
    """Sphere patch (dome, tank cap). UV = metric azimuth/polar chart centred
    on the patch: u = r·(θ−θc)·sin(φc), v = r·(φ−φc) — a local equirectangular
    unroll, accurate for the moderate angular spans of built domes."""
    center: np.ndarray        # (3,)
    radius: float
    pole: np.ndarray          # (3,) unit — chart axis through the patch centre
    theta_ref: np.ndarray     # (3,) unit ⊥ pole, θ=0
    phi_c: float              # polar angle of patch centre (rad, from pole)
    theta_c: float            # azimuth of patch centre (rad)
    rms: float
    inlier_frac: float
    n_points: int

    DOF = 4  # center (3) + radius (1)

    def _frame(self):
        w = self.pole
        u = self.theta_ref
        v = np.cross(w, u)
        return u, v, w

    def _sph_coords(self, pts: np.ndarray):
        u, v, w = self._frame()
        q = np.asarray(pts, dtype=np.float64) - self.center
        rho = np.linalg.norm(q, axis=1)
        z = np.clip((q @ w) / np.maximum(rho, _EPS), -1.0, 1.0)
        phi = np.arccos(z)                            # from pole
        theta = np.arctan2(q @ v, q @ u)
        return rho, phi, theta

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        q = np.asarray(pts, dtype=np.float64) - self.center
        return np.linalg.norm(q, axis=1) - self.radius

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        _, phi, theta = self._sph_coords(pts)
        dtheta = (theta - self.theta_c + np.pi) % _TWO_PI - np.pi
        su = self.radius * dtheta * max(np.sin(self.phi_c), 0.1)
        sv = self.radius * (phi - self.phi_c)
        return np.column_stack([su, sv])

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        u, v, w = self._frame()
        theta = self.theta_c + uv[:, 0] / (self.radius * max(np.sin(self.phi_c), 0.1))
        phi = self.phi_c + uv[:, 1] / self.radius
        sp = np.sin(phi)
        dirs = (sp * np.cos(theta))[:, None] * u + (sp * np.sin(theta))[:, None] * v \
            + np.cos(phi)[:, None] * w
        return self.center + self.radius * dirs

    def params_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "pole": self.pole.tolist(),
            "rms_m": float(self.rms),
            "inlier_frac": float(self.inlier_frac),
        }


def _fit_sphere_kasa(pts: np.ndarray):
    """Algebraic sphere fit: ‖x‖² = 2c·x + (r²−‖c‖²) → linear LS."""
    A = np.column_stack([2.0 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if r2 <= _EPS:
        return None, None
    return c, float(np.sqrt(r2))


def fit_sphere(points: np.ndarray, dist_thresh: float = 0.02,
               min_inlier_frac: float = 0.5) -> Optional[SphereModel]:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 50:
        return None
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))

    c, r = _fit_sphere_kasa(pts)
    if c is None or r > MAX_RADIUS_FACTOR * extent:
        return None
    # trim to inliers and refit once (robustness against clutter)
    for _ in range(2):
        d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        inl = d < dist_thresh
        if inl.sum() < 12:
            return None
        c2, r2 = _fit_sphere_kasa(pts[inl])
        if c2 is None:
            return None
        c, r = c2, r2
        if r > MAX_RADIUS_FACTOR * extent:
            return None
    d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
    inl = d < dist_thresh
    frac = float(inl.mean())
    if frac < min_inlier_frac:
        return None
    rms = float(np.sqrt(np.mean(d[inl] ** 2)))

    # chart axis through the patch centre
    mean_dir = pts[inl].mean(0) - c
    ln = np.linalg.norm(mean_dir)
    pole = mean_dir / ln if ln > _EPS else np.array([0.0, 0.0, 1.0])
    u, v = _orthobasis(pole)
    # centre the chart: phi_c along pole is 0 by construction of `pole` (the
    # patch centroid direction); use a small offset so sin(phi_c) ≠ 0.
    q = pts[inl] - c
    rho = np.linalg.norm(q, axis=1)
    phi = np.arccos(np.clip((q @ pole) / np.maximum(rho, _EPS), -1, 1))
    theta = np.arctan2(q @ np.cross(pole, u), q @ u)
    return SphereModel(center=np.asarray(c), radius=float(r), pole=pole,
                       theta_ref=u, phi_c=float(np.median(phi)),
                       theta_c=float(np.median(theta)), rms=rms,
                       inlier_frac=frac, n_points=n)
