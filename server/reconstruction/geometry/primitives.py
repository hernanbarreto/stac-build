"""
Robust geometric primitive fitters — plane, cylinder, cone — plus the
flat-vs-curved discriminator (`surface_kind`) and per-point normal estimation.

All fitters take an (N,3) point array (any frame) and a perpendicular distance
threshold in the same units, and return a dataclass with the parameters, an
inlier boolean mask, and the inlier RMS. RANSAC is hand-rolled (no extra dep);
plane RANSAC + normals delegate to Open3D where it's faster/battle-tested.

Used by `reconstruction.reconstruct.surface` and `.swept`, and replaces the
inline RANSAC-plane loop that used to live in `pipeline.py::_clean_segment_subcloud`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # Open3D is a hard dep of the server env; used for fast plane RANSAC + normals.
    import open3d as o3d
except Exception:  # pragma: no cover - defensive
    o3d = None

_EPS = 1e-12


# ── result types ────────────────────────────────────────────────────

@dataclass
class PlaneFit:
    normal: np.ndarray            # (3,) unit
    d: float                      # plane: n·x + d = 0
    inliers: np.ndarray           # (N,) bool over the input points
    rms: float                    # RMS perpendicular distance of inliers
    n_points: int
    curvature: float = 0.0        # |implied curvature| from quadratic residual fit (1/m); 0 = flat

    @property
    def inlier_frac(self) -> float:
        return float(self.inliers.mean()) if self.n_points else 0.0

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts) @ self.normal + self.d


@dataclass
class CylinderFit:
    axis_point: np.ndarray        # (3,) foot of axis (offset along axis is arbitrary)
    axis_dir: np.ndarray          # (3,) unit
    radius: float
    inliers: np.ndarray
    rms: float
    n_points: int
    theta_ref: np.ndarray         # (3,) unit, ⊥ axis — defines θ=0
    s_extent: Tuple[float, float] # (min,max) along axis from axis_point
    theta_extent: Tuple[float, float]  # (θ0, θ1) radians; θ1-θ0 ∈ [0, 2π]

    @property
    def inlier_frac(self) -> float:
        return float(self.inliers.mean()) if self.n_points else 0.0

    @property
    def is_full_circle(self) -> bool:
        return (self.theta_extent[1] - self.theta_extent[0]) >= 2 * np.pi - 0.3


@dataclass
class ConeFit:
    apex: np.ndarray              # (3,)
    axis_dir: np.ndarray          # (3,) unit, pointing away from apex into the cone
    half_angle: float             # radians
    inliers: np.ndarray
    rms: float
    n_points: int
    theta_ref: np.ndarray
    s_extent: Tuple[float, float] # (min,max) along axis from apex
    theta_extent: Tuple[float, float]

    @property
    def inlier_frac(self) -> float:
        return float(self.inliers.mean()) if self.n_points else 0.0


@dataclass
class SurfaceFit:
    kind: str                     # 'plane' | 'cylinder' | 'cone' | 'freeform'
    fit: Optional[object]         # PlaneFit | CylinderFit | ConeFit | None


# ── small linear-algebra helpers ────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _orthobasis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors u, v completing `axis` to a right-handed basis."""
    axis = _unit(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, ref))
    v = np.cross(axis, u)
    return u, v


def _circle_from_3(p: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Circumcircle of 3 2-D points → (centre(2,), radius) or (None, None)."""
    (x1, y1), (x2, y2), (x3, y3) = p
    a = x2 - x1
    b = y2 - y1
    c = x3 - x1
    d = y3 - y1
    e = a * (x1 + x2) + b * (y1 + y2)
    f = c * (x1 + x3) + d * (y1 + y3)
    g = 2.0 * (a * (y3 - y2) - b * (x3 - x2))
    if abs(g) < _EPS:
        return None, None
    cx = (d * e - b * f) / g
    cy = (a * f - c * e) / g
    r = float(np.hypot(x1 - cx, y1 - cy))
    return np.array([cx, cy]), r


def _fit_circle_ls(p2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic (Kåsa) least-squares circle fit. p2: (M,2). Returns (centre, r)."""
    x = p2[:, 0]
    y = p2[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    r = float(np.sqrt(max(r2, _EPS)))
    return np.array([cx, cy]), r


def _angular_extent(angles: np.ndarray) -> Tuple[float, float]:
    """Smallest arc [θ0, θ1] (θ0 in [-π,π), θ1-θ0 ∈ [0,2π]) covering all `angles`.

    If the largest gap between consecutive sorted angles is tiny → full circle.
    """
    a = np.sort(np.mod(np.asarray(angles), 2 * np.pi))
    if a.size == 0:
        return (0.0, 2 * np.pi)
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    g = int(np.argmax(gaps))
    if gaps[g] < np.deg2rad(8):  # essentially closed
        return (0.0, 2 * np.pi)
    start = a[(g + 1) % a.size]
    span = 2 * np.pi - gaps[g]
    # normalise start to [-π, π)
    start = (start + np.pi) % (2 * np.pi) - np.pi
    return (float(start), float(start + span))


# ── normal estimation ───────────────────────────────────────────────

def estimate_normals(points: np.ndarray, k: int = 18,
                     orient_outward: bool = True) -> np.ndarray:
    """Per-point unit normals via local PCA over k nearest neighbours.

    If `orient_outward`, normals are flipped to point away from the point set's
    centroid (consistent enough for the Gauss-map test; sign-invariant uses
    the scatter matrix anyway).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    k = int(min(max(6, k), n - 1))
    if o3d is not None:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        nrm = np.asarray(pcd.normals)
        if nrm.shape[0] != n:  # pragma: no cover - shouldn't happen
            nrm = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    else:  # pragma: no cover
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k + 1)
        nrm = np.empty((n, 3))
        for i in range(n):
            nb = pts[idx[i]]
            c = nb - nb.mean(0)
            _, V = np.linalg.eigh(c.T @ c)
            nrm[i] = V[:, 0]
    if orient_outward:
        ctr = pts.mean(0)
        flip = np.einsum("ij,ij->i", nrm, pts - ctr) < 0
        nrm[flip] *= -1.0
    # normalise
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    ln[ln < _EPS] = 1.0
    return nrm / ln


def _normal_scatter_eigs(normals: np.ndarray) -> np.ndarray:
    """Ascending eigenvalues of the sign-invariant scatter matrix (1/N)Σ nnᵀ.

    plane ⇒ (≈0, ≈0, ≈1); cylinder ⇒ (≈0, ≈½, ≈½); sphere/freeform ⇒ (⅓,⅓,⅓).
    """
    M = (normals.T @ normals) / max(len(normals), 1)
    return np.sort(np.linalg.eigvalsh(M))


# ── plane ───────────────────────────────────────────────────────────

def fit_plane_ransac(points: np.ndarray, dist_thresh: float = 0.012,
                     iters: int = 300, min_inlier_frac: float = 0.30,
                     measure_curvature: bool = True) -> Optional[PlaneFit]:
    """Robust plane fit. Returns ``None`` if no plane gathers ``min_inlier_frac``.

    `curvature` (when `measure_curvature`): magnitude of the dominant quadratic
    coefficient of (perpendicular residual vs in-plane coords) of the inliers —
    a flat surface has ~0; a curved one has a meaningful value (≈ 1/R).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None

    normal = d = None
    inliers = None
    if o3d is not None:
        try:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            model, idx = pcd.segment_plane(distance_threshold=float(dist_thresh),
                                           ransac_n=3, num_iterations=int(iters))
            a, b, c, dd = model
            normal = _unit([a, b, c])
            d = float(dd) / max(np.linalg.norm([a, b, c]), _EPS)
            inliers = np.zeros(n, dtype=bool)
            inliers[np.asarray(idx, dtype=int)] = True
        except Exception:
            normal = None
    if normal is None:  # numpy RANSAC fallback
        best = (0, None, None, None)
        rng = np.random.default_rng(0)
        for _ in range(iters):
            s = rng.choice(n, 3, replace=False)
            p0, p1, p2 = pts[s]
            nrm = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(nrm)
            if ln < _EPS:
                continue
            nrm = nrm / ln
            dd = -nrm @ p0
            dist = np.abs(pts @ nrm + dd)
            inl = dist < dist_thresh
            if int(inl.sum()) > best[0]:
                best = (int(inl.sum()), nrm, dd, inl)
        if best[1] is None:
            return None
        _, normal, d, inliers = best

    # refine on inliers via PCA
    if inliers.sum() >= 3:
        q = pts[inliers]
        ctr = q.mean(0)
        _, V = np.linalg.eigh((q - ctr).T @ (q - ctr))
        normal = _unit(V[:, 0]) * (1.0 if V[:, 0] @ normal >= 0 else -1.0)
        d = float(-normal @ ctr)
        dist = np.abs(pts @ normal + d)
        inliers = dist < dist_thresh

    if inliers.mean() < min_inlier_frac or inliers.sum() < 3:
        return None
    dist_in = np.abs(pts[inliers] @ normal + d)
    rms = float(np.sqrt(np.mean(dist_in ** 2)))

    curv = 0.0
    if measure_curvature and inliers.sum() >= 12:
        q = pts[inliers]
        ctr = q.mean(0)
        u, v = _orthobasis(normal)
        xx = (q - ctr) @ u
        yy = (q - ctr) @ v
        r = (q - ctr) @ normal  # signed residual
        # quadratic LS:  r = a x² + b y² + c xy + d x + e y + f
        A = np.column_stack([xx * xx, yy * yy, xx * yy, xx, yy, np.ones_like(xx)])
        try:
            coef, *_ = np.linalg.lstsq(A, r, rcond=None)
            curv = float(2.0 * max(abs(coef[0]), abs(coef[1])))
        except Exception:
            curv = 0.0

    return PlaneFit(normal=normal, d=float(d), inliers=inliers, rms=rms,
                    n_points=n, curvature=curv)


# ── cylinder ────────────────────────────────────────────────────────

def fit_cylinder_ransac(points: np.ndarray, dist_thresh: float = 0.02,
                        iters: int = 250, min_inlier_frac: float = 0.40,
                        normals: Optional[np.ndarray] = None,
                        axis_hint: Optional[np.ndarray] = None,
                        max_radius: Optional[float] = None) -> Optional[CylinderFit]:
    """Robust circular-cylinder fit.

    Axis direction = null eigenvector of the per-point normal scatter (snapped
    to ``axis_hint`` if within ~45° of it — e.g. world-up for a vertical curved
    wall). Then RANSAC + algebraic refinement of a circle in the plane ⊥ axis.
    Reports angular extent (a curved wall is a *partial* cylinder).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 8:
        return None
    if normals is None:
        normals = estimate_normals(pts)
    M = (normals.T @ normals) / n
    w, V = np.linalg.eigh(M)         # ascending
    axis = _unit(V[:, 0])
    if axis_hint is not None:
        ah = _unit(axis_hint)
        if abs(axis @ ah) > 0.71:    # within ~45°
            axis = ah * (1.0 if axis @ ah >= 0 else -1.0)

    u, v = _orthobasis(axis)
    p2 = np.column_stack([pts @ u, pts @ v])
    bbox2 = float(np.linalg.norm(p2.max(0) - p2.min(0)))
    rmax = max_radius if max_radius is not None else 50.0 * max(bbox2, 1e-3)

    rng = np.random.default_rng(1)
    best = (0, None, None, None)
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        c2, r = _circle_from_3(p2[s])
        if c2 is None or r is None or r <= _EPS or r > rmax:
            continue
        dist = np.abs(np.linalg.norm(p2 - c2, axis=1) - r)
        inl = dist < dist_thresh
        if int(inl.sum()) > best[0]:
            best = (int(inl.sum()), c2, r, inl)
    if best[1] is None:
        return None
    _, c2, r, inl = best

    # refine: LS circle on inliers, twice
    for _ in range(2):
        if inl.sum() < 3:
            break
        c2, r = _fit_circle_ls(p2[inl])
        if not np.isfinite(r) or r <= _EPS or r > rmax:
            return None
        dist = np.abs(np.linalg.norm(p2 - c2, axis=1) - r)
        inl = dist < dist_thresh

    if inl.mean() < min_inlier_frac or inl.sum() < 3:
        return None
    rms = float(np.sqrt(np.mean((np.abs(np.linalg.norm(p2[inl] - c2, axis=1) - r)) ** 2)))

    axis_point = c2[0] * u + c2[1] * v
    s_along = pts @ axis
    s_extent = (float(s_along[inl].min()), float(s_along[inl].max()))
    ang = np.arctan2(p2[:, 1] - c2[1], p2[:, 0] - c2[0])
    theta_extent = _angular_extent(ang[inl])
    return CylinderFit(axis_point=axis_point, axis_dir=axis, radius=float(r),
                       inliers=inl, rms=rms, n_points=n, theta_ref=u,
                       s_extent=s_extent, theta_extent=theta_extent)


# ── cone ────────────────────────────────────────────────────────────

def fit_cone_ransac(points: np.ndarray, dist_thresh: float = 0.025,
                    min_inlier_frac: float = 0.40,
                    axis_hint: Optional[np.ndarray] = None,
                    normals: Optional[np.ndarray] = None) -> Optional[ConeFit]:
    """Best-effort circular-cone fit via nonlinear least squares (robust loss).

    Init: treat the cluster as a stack of circular slices along a coarse axis,
    fit radius-vs-height → apex (where r→0) and half-angle (atan dr/ds). Then
    refine apex/axis/half-angle minimising |v|·sin(φ−α) with a soft-L1 loss.
    """
    from scipy.optimize import least_squares

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 12:
        return None
    if normals is None:
        normals = estimate_normals(pts)

    # coarse axis: prefer hint, else mean normal direction (cones with small
    # half-angle have normals roughly ⊥ to axis → use the *least* spread dir)
    if axis_hint is not None:
        axis0 = _unit(axis_hint)
    else:
        M = (normals.T @ normals) / n
        _, V = np.linalg.eigh(M)
        axis0 = _unit(V[:, 0])
    # orient axis0 so most points are on the +s side
    s0 = pts @ axis0
    if np.median(s0) < 0:
        axis0 = -axis0
        s0 = -s0

    # slice radii along axis0
    u, v = _orthobasis(axis0)
    p2 = np.column_stack([pts @ u, pts @ v])
    nb = max(4, int(np.sqrt(n) / 2))
    edges = np.quantile(s0, np.linspace(0, 1, nb + 1))
    cs = []
    rs = []
    for i in range(nb):
        m = (s0 >= edges[i]) & (s0 <= edges[i + 1])
        if m.sum() < 3:
            continue
        c2, rr = _fit_circle_ls(p2[m])
        if np.isfinite(rr) and rr > _EPS:
            cs.append(0.5 * (edges[i] + edges[i + 1]))
            rs.append(rr)
    if len(rs) < 3:
        return None
    cs = np.asarray(cs)
    rs = np.asarray(rs)
    # r ≈ slope*(s - s_apex)  → fit r = m s + b ; apex at s = -b/m
    m_slope, b_int = np.polyfit(cs, rs, 1)
    if abs(m_slope) < 1e-4:
        return None  # ~cylinder, not a cone
    s_apex = -b_int / m_slope
    half_angle0 = float(np.arctan(abs(m_slope)))
    # apex 3D (in the axis0 frame); the nonlinear refine + final re-orientation
    # fix the axis sign, so the init sign doesn't matter.
    c_mean = p2.mean(0)
    apex0 = s_apex * axis0 + c_mean[0] * u + c_mean[1] * v

    def residuals(x):
        ap = x[0:3]
        ax = x[3:6]
        ax = ax / (np.linalg.norm(ax) + _EPS)
        al = x[6]
        vv = pts - ap
        vn = np.linalg.norm(vv, axis=1) + _EPS
        phi = np.arccos(np.clip((vv @ ax) / vn, -1.0, 1.0))
        return vn * np.sin(phi - al)

    x0 = np.concatenate([apex0, _unit(axis0), [half_angle0]])
    try:
        sol = least_squares(residuals, x0, loss="soft_l1",
                            f_scale=max(dist_thresh, 1e-3), max_nfev=200)
    except Exception:
        return None
    ap = sol.x[0:3]
    ax = _unit(sol.x[3:6])
    al = float(np.clip(sol.x[6], np.deg2rad(2), np.deg2rad(88)))
    # orient axis into the cone (towards the bulk of points)
    if np.mean((pts - ap) @ ax) < 0:
        ax = -ax
    res = np.abs(residuals(np.concatenate([ap, ax, [al]])))
    inl = res < dist_thresh
    if inl.mean() < min_inlier_frac or inl.sum() < 3:
        return None
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))

    # extents in (s along axis from apex, θ)
    u2, v2 = _orthobasis(ax)
    s_along = (pts - ap) @ ax
    s_extent = (float(s_along[inl].min()), float(s_along[inl].max()))
    q2 = np.column_stack([(pts - ap) @ u2, (pts - ap) @ v2])
    ang = np.arctan2(q2[:, 1], q2[:, 0])
    theta_extent = _angular_extent(ang[inl])
    return ConeFit(apex=ap, axis_dir=ax, half_angle=al, inliers=inl, rms=rms,
                   n_points=n, theta_ref=u2, s_extent=s_extent,
                   theta_extent=theta_extent)


# ── discriminator ───────────────────────────────────────────────────

def surface_kind(points: np.ndarray, dist_thresh: float = 0.012,
                 curved_thresh: Optional[float] = None,
                 axis_hint: Optional[np.ndarray] = None) -> SurfaceFit:
    """Decide whether a sheet-like cluster is a plane, cylinder, cone, or
    free-form, and return the winning fit.

    Strategy: fit a plane; accept it only if the inlier fraction is high AND the
    quadratic-residual curvature is negligible relative to the patch extent. If
    not, use the Gauss-map (normal scatter) as a fast triage:
      - rank-1-ish normals + decent plane → still a plane (very flat cone, etc.)
      - rank-3-ish normals → free-form
      - rank-2-ish → try cylinder, then cone; pick the lower-RMS acceptable fit.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 6:
        return SurfaceFit("freeform", None)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    # "meaningfully curved" = curvature radius < ~12× the patch extent
    if curved_thresh is None:
        curved_thresh = 1.0 / (12.0 * extent)

    pf = fit_plane_ransac(pts, dist_thresh=dist_thresh, measure_curvature=True)
    flat_ok = (pf is not None and pf.inlier_frac >= 0.90
               and pf.curvature <= curved_thresh)
    if flat_ok:
        return SurfaceFit("plane", pf)

    eigs = _normal_scatter_eigs(estimate_normals(pts))
    lo, mid, hi = float(eigs[0]), float(eigs[1]), float(eigs[2])
    rank3 = lo > 0.18                  # normals fill the sphere → bumpy/free
    rank1 = mid < 0.12                 # normals ~collinear → plane-like
    # A rank-1 normal cloud is a plane even when an attached parallel band
    # (skirting / cornice / cladding) makes some points fall outside the on-plane
    # tolerance — those become `ProfileElement`s downstream, not curvature.
    if rank1 and pf is not None and pf.inlier_frac >= 0.50:
        return SurfaceFit("plane", pf)
    if rank3:
        return SurfaceFit("freeform", None)

    # rank-2 region (or borderline) → curved-surface candidates
    cyl = fit_cylinder_ransac(pts, dist_thresh=max(dist_thresh, 1.6 * dist_thresh),
                              axis_hint=axis_hint)
    # a cylinder whose radius dwarfs the patch is just a plane being over-fitted
    if cyl is not None and cyl.radius > 6.0 * extent and pf is not None and pf.inlier_frac >= 0.5:
        return SurfaceFit("plane", pf)
    cone = None
    cyl_ok = cyl is not None and cyl.inlier_frac >= 0.82
    if not cyl_ok:
        cone = fit_cone_ransac(pts, dist_thresh=max(dist_thresh, 2.0 * dist_thresh),
                               axis_hint=axis_hint)
    cone_ok = cone is not None and cone.inlier_frac >= 0.80

    # choose the best acceptable model
    cands = []
    if pf is not None and pf.inlier_frac >= 0.92:
        cands.append(("plane", pf, pf.rms, pf.inlier_frac))
    if cyl_ok:
        cands.append(("cylinder", cyl, cyl.rms, cyl.inlier_frac))
    if cone_ok:
        cands.append(("cone", cone, cone.rms, cone.inlier_frac))
    if not cands:
        # last resort: whichever exists with the most inliers
        if cyl is not None and cyl.inlier_frac >= 0.6:
            return SurfaceFit("cylinder", cyl)
        if pf is not None and pf.inlier_frac >= 0.7:
            return SurfaceFit("plane", pf)
        return SurfaceFit("freeform", None)
    # rank by inlier_frac first, then by lower RMS
    cands.sort(key=lambda c: (-c[3], c[2]))
    return SurfaceFit(cands[0][0], cands[0][1])
