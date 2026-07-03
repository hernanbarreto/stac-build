"""
Stage 2.3 — extruded profile for segments with a dominant axis (vaults,
tunnel linings, long curved walls).

Method (per the project charter):
  1. axis: dominant PCA direction of the segment (snapped to the camera-path
     direction when one is supplied — the trajectory runs along the tunnel);
  2. cross-sections every ``section_spacing_m`` along the axis;
  3. ONE robust average profile: all sections' points pooled in polar form
     (φ, ρ) around the shared cross-section centre and reduced by per-angular-
     bin MEDIAN — thousands of redundant sections beat per-section noise;
  4. a smoothing B-spline ρ(φ) over the medians (knot spacing ≈ the configured
     control-point spacing, so the profile is a geometric low-pass);
  5. the surface = that profile swept straight along the axis.

The profile is measurement all the way down: medians of measured radii,
never invented geometry. Sections that would need extrapolation simply do
not contribute, and the support trim keeps unmeasured spans unmeshed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import interpolate

from reconstruction.geometry.primitives import _angular_extent, _orthobasis
from .models import orient_normal

logger = logging.getLogger("SurfaceFit")

_EPS = 1e-12
_TWO_PI = 2.0 * np.pi
_DENSE = 2048          # dense φ sampling for arc-length tables


@dataclass
class SweptProfileModel:
    """Straight-axis sweep of a smooth polar profile ρ(φ).

    UV = (s along axis, arc length along the profile from its start) — metric
    on the surface, so grids/meshing/support work unchanged.
    """
    origin: np.ndarray        # (3,) point on axis (centroid projection)
    axis_dir: np.ndarray      # (3,) unit (w)
    frame_a: np.ndarray       # (3,) unit ⊥ axis
    frame_b: np.ndarray       # (3,) unit ⊥ axis (w × a)
    center2d: np.ndarray      # (2,) polar centre in the (a,b) cross-plane
    tck: tuple                # B-spline of ρ(φ) (scipy splrep tck)
    phi0: float               # start of angular coverage (rad)
    phi_span: float           # angular coverage (rad); 2π = closed ring
    periodic: bool
    rms: float
    inlier_frac: float
    n_points: int
    dof: int

    # arc-length tables (built in __post_init__)
    _phi_grid: np.ndarray = None
    _arc_grid: np.ndarray = None

    @property
    def DOF(self) -> int:
        return self.dof

    def __post_init__(self):
        phi = np.linspace(self.phi0, self.phi0 + self.phi_span, _DENSE)
        rho = interpolate.splev(np.mod(phi - self.phi0, _TWO_PI) + self.phi0
                                if self.periodic else phi, self.tck)
        drho = interpolate.splev(phi, self.tck, der=1)
        ds = np.sqrt(np.asarray(rho) ** 2 + np.asarray(drho) ** 2)
        arc = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1])
                                               * np.diff(phi))])
        self._phi_grid = phi
        self._arc_grid = arc

    # ── coordinates ──
    def _cross_coords(self, pts: np.ndarray):
        q = np.asarray(pts, dtype=np.float64) - self.origin
        s = q @ self.axis_dir
        x = q @ self.frame_a - self.center2d[0]
        y = q @ self.frame_b - self.center2d[1]
        return s, x, y

    def _phi_of(self, x, y):
        phi = np.arctan2(y, x)
        # map into [phi0, phi0+2π) so partial profiles are contiguous
        return np.mod(phi - self.phi0, _TWO_PI) + self.phi0

    def _rho_spline(self, phi):
        return np.asarray(interpolate.splev(phi, self.tck))

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        _, x, y = self._cross_coords(pts)
        phi = self._phi_of(x, y)
        return np.hypot(x, y) - self._rho_spline(phi)   # radial: + outside

    def to_uv(self, pts: np.ndarray) -> np.ndarray:
        s, x, y = self._cross_coords(pts)
        phi = self._phi_of(x, y)
        v = np.interp(phi, self._phi_grid, self._arc_grid)
        return np.column_stack([s, v])

    def uv_to_world(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        phi = np.interp(uv[:, 1], self._arc_grid, self._phi_grid)
        rho = self._rho_spline(phi)
        x = self.center2d[0] + rho * np.cos(phi)
        y = self.center2d[1] + rho * np.sin(phi)
        return (self.origin + uv[:, 0:1] * self.axis_dir
                + x[:, None] * self.frame_a + y[:, None] * self.frame_b)

    def params_dict(self) -> dict:
        t, c, k = self.tck
        return {
            "origin": self.origin.tolist(),
            "axis_dir": self.axis_dir.tolist(),
            "frame_a": self.frame_a.tolist(),
            "frame_b": self.frame_b.tolist(),
            "center2d": self.center2d.tolist(),
            "profile_knots": np.asarray(t).tolist(),
            "profile_coeffs": np.asarray(c).tolist(),
            "profile_degree": int(k),
            "phi0": float(self.phi0),
            "phi_span": float(self.phi_span),
            "periodic": bool(self.periodic),
            "rms_m": float(self.rms),
            "inlier_frac": float(self.inlier_frac),
        }


def fit_swept_profile(points: np.ndarray,
                      dist_thresh: float = 0.02,
                      min_inlier_frac: float = 0.5,
                      section_spacing_m: float = 0.20,
                      ctrl_point_spacing_m: float = 0.5,
                      min_elongation: float = 1.5,
                      world_up: Optional[np.ndarray] = None,
                      camera_path: Optional[np.ndarray] = None,
                      min_bin_samples: int = 8) -> Optional[SweptProfileModel]:
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 500:
        return None
    up = np.array([0.0, 0.0, 1.0]) if world_up is None else np.asarray(world_up, float)

    # ── axis: dominant PCA direction, snapped to the camera path if it agrees ──
    ctr = pts.mean(0)
    q = pts - ctr
    cov = (q.T @ q) / n
    w_eig, V = np.linalg.eigh(cov)
    axis = V[:, 2] / np.linalg.norm(V[:, 2])
    if camera_path is not None and len(camera_path) >= 2:
        cp = np.asarray(camera_path, dtype=np.float64)
        cdir = cp[-1] - cp[0]
        ln = np.linalg.norm(cdir)
        if ln > _EPS and abs((cdir / ln) @ axis) > 0.71:
            axis = (cdir / ln) * (1.0 if (cdir / ln) @ axis >= 0 else -1.0)

    s = q @ axis
    span_axis = float(s.max() - s.min())
    cross_extent = float(np.sqrt(max(w_eig[0], 0) + max(w_eig[1], 0)) * 4.0)
    if span_axis < min_elongation * max(cross_extent / 2.0, 1e-3):
        logger.info("swept_profile: not elongated enough (axis span %.2fm vs "
                    "cross %.2fm) — skipping", span_axis, cross_extent)
        return None
    if span_axis < 2.0 * section_spacing_m:
        return None

    def _cross_frame(ax):
        """cross-plane frame with b aligned to up (vault crown reads as +b)"""
        aa, bb = _orthobasis(ax)
        up_perp = up - (up @ ax) * ax
        ln = np.linalg.norm(up_perp)
        if ln > 0.3:
            bb = up_perp / ln
            aa = np.cross(bb, ax)
            aa /= np.linalg.norm(aa)
        return aa, bb

    def _build(axis_cur: np.ndarray) -> Optional[SweptProfileModel]:
        """Full profile build for a given axis: frame → polar (chord-centred
        for open profiles) → per-bin medians → smoothing B-spline → model."""
        a, b = _cross_frame(axis_cur)
        x = q @ a
        y = q @ b
        center2d = np.array([np.median(x), np.median(y)])
        xx = x - center2d[0]
        yy = y - center2d[1]
        rho = np.hypot(xx, yy)
        phi = np.arctan2(yy, xx)

        # angular coverage (partial vault vs closed ring)
        phi0, phi1 = _angular_extent(phi)
        span = phi1 - phi0
        periodic = span >= _TWO_PI - 1e-6

        # Open profiles: the area-median centre sits badly for vaults — near
        # the springing lines the curve runs almost RADIAL from it and ρ(φ)
        # degenerates. The midpoint of the two profile ENDPOINTS (chord
        # centre) keeps the curve ~orthogonal to the rays everywhere; blend
        # slightly towards the centroid so the centre never lands exactly on
        # the boundary.
        if not periodic:
            rel = np.mod(phi - phi0, _TWO_PI)
            edge = max(span * 0.02, np.deg2rad(2))
            lo = rel <= edge
            hi = rel >= span - edge
            if lo.sum() >= 8 and hi.sum() >= 8:
                e0 = np.array([np.median(x[lo]), np.median(y[lo])])
                e1 = np.array([np.median(x[hi]), np.median(y[hi])])
                centroid2d = np.array([x.mean(), y.mean()])
                center2d = 0.9 * (0.5 * (e0 + e1)) + 0.1 * centroid2d
                xx = x - center2d[0]
                yy = y - center2d[1]
                rho = np.hypot(xx, yy)
                phi = np.arctan2(yy, xx)
                phi0, phi1 = _angular_extent(phi)
                span = phi1 - phi0
                periodic = span >= _TWO_PI - 1e-6

        # ── robust average profile: per-angular-bin median over ALL sections ──
        r_med = float(np.median(rho))
        if r_med < 1e-3:
            return None
        # bin width ≈ 5 cm of arc, bounded; knots enforce the real low-pass
        nbins = int(np.clip(span * r_med / 0.05, 36, 720))
        rel = np.mod(phi - phi0, _TWO_PI)
        inside = rel <= span + 1e-9
        bin_idx = np.minimum((rel[inside] / span * nbins).astype(np.int64), nbins - 1)
        rho_in = rho[inside]
        order = np.argsort(bin_idx, kind="stable")
        bin_sorted = bin_idx[order]
        rho_sorted = rho_in[order]
        bounds = np.searchsorted(bin_sorted, np.arange(nbins + 1))
        phi_c, rho_c, weight = [], [], []
        for k in range(nbins):
            seg = rho_sorted[bounds[k]:bounds[k + 1]]
            if len(seg) < min_bin_samples:
                continue
            phi_c.append(phi0 + (k + 0.5) * span / nbins)
            rho_c.append(float(np.median(seg)))
            weight.append(np.sqrt(len(seg)))
        if len(rho_c) < 8:
            return None
        phi_c = np.asarray(phi_c)
        rho_c = np.asarray(rho_c)
        weight = np.asarray(weight)

        # smoothing B-spline of the median profile, knots every ctrl spacing
        knot_dphi = ctrl_point_spacing_m / r_med
        interior = np.arange(phi_c[0] + knot_dphi, phi_c[-1] - knot_dphi / 2,
                             knot_dphi)
        try:
            if periodic:
                tck = interpolate.splrep(
                    np.concatenate([phi_c, [phi_c[0] + _TWO_PI]]),
                    np.concatenate([rho_c, [rho_c[0]]]),
                    w=np.concatenate([weight, [weight[0]]]),
                    per=True, s=len(phi_c) * (0.002 ** 2))
            else:
                tck = interpolate.splrep(phi_c, rho_c, w=weight, t=interior, k=3)
        except Exception as e:
            logger.info("swept_profile: spline fit failed (%s)", e)
            return None
        dof = len(np.asarray(tck[1]).ravel()) + 4  # profile coeffs + axis line

        return SweptProfileModel(
            origin=ctr, axis_dir=axis_cur, frame_a=a, frame_b=b,
            center2d=center2d, tck=tck, phi0=float(phi0), phi_span=float(span),
            periodic=periodic, rms=0.0, inlier_frac=0.0, n_points=n, dof=int(dof))

    model = _build(axis)
    if model is None:
        return None

    # ── axis refinement by residual harmonics ──
    # A milliradian of axis error over a long sweep displaces the profile
    # centre linearly with s, leaving a dipole residual r ≈ e(s)·(cosφ, sinφ)
    # with e(s) = e0 + e1·s — real structure a straight sweep must not have
    # (and Moran's I would rightly reject). Estimating (e0, e1) by linear LS
    # over ALL points uses the full redundancy (precision ~ σ/(√N·s_rms), i.e.
    # micro-radians), unlike per-section centroids whose medians drown mm-level
    # drift in the metre-scale cross-section spread. e1 tilts the axis; the
    # rebuild's chord-centring absorbs e0.
    for _ in range(3):
        r = model.signed_distance(pts)
        s_ax, x_c, y_c = model._cross_coords(pts)
        phi = np.arctan2(y_c, x_c)
        M = np.column_stack([np.cos(phi), np.sin(phi),
                             s_ax * np.cos(phi), s_ax * np.sin(phi)])
        try:
            coef, *_ = np.linalg.lstsq(M, r, rcond=None)
        except np.linalg.LinAlgError:
            break
        tilt = float(np.hypot(coef[2], coef[3]))
        if tilt < 1e-6:
            break
        axis = axis + coef[2] * model.frame_a + coef[3] * model.frame_b
        axis /= np.linalg.norm(axis)
        refined = _build(axis)
        if refined is None:
            break
        model = refined

    resid = model.signed_distance(pts)
    inl = np.abs(resid) < max(dist_thresh, 0.02)
    model.inlier_frac = float(inl.mean())
    model.rms = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else float("inf")
    if model.inlier_frac < min_inlier_frac:
        logger.info("swept_profile: inlier fraction %.2f < %.2f — rejected",
                    model.inlier_frac, min_inlier_frac)
        return None
    return model
