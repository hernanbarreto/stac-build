"""
Swept-solid fit — a 2-D cross-section swept along a 3-D path.

Unifies pipes (circular), HVAC ducts (rectangular / hollow), structural members
(I/H/L/channel), rails, kerbs, cable trays. The robust pipeline:

  1. axis estimate — PCA principal direction of the (elongated) cluster;
  2. slices — bin points along the axis; per slice take the centroid (a path
     vertex) and the 2-D points in the plane ⊥ axis;
  3. cross-section family per slice — least-squares circle vs. min-area
     rectangle vs. free polygon (concave hull); majority vote across slices,
     params = median of the winning family;
  4. path — the slice centroids; if they're collinear → a straight 2-vertex
     path, else a dense polyline (arc segmentation is a later refinement — a
     dense polyline is an acceptable robust representation);
  5. profile — discretised in a moving frame anchored at path[0].

Returns a `SweptFit`; `reconstruct.swept` wraps it into a `SweptElement`.
Genuinely poor fits still return a coarse best-effort `SweptFit` (a rough swept
rectangle beats a hallucinated mesh for a 50 m duct run) — callers check `rms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .primitives import _orthobasis, _unit, _fit_circle_ls, _EPS

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


@dataclass
class SweptFit:
    profile_family: str                  # circle | rect | rect_hollow | free
    profile_params: dict                 # representative (median across slices)
    profile_polygon: np.ndarray          # (M, 2) representative cross-section in the moving (u,v) frame
    profile_frame: np.ndarray            # (3,3) columns [u, v, t0]: profile-x, profile-y, tangent at path[0]
    path: np.ndarray                     # (K, 3) world polyline (slice centroids), K>=2
    arc_segments: List[Tuple[int, int, np.ndarray, np.ndarray]] = field(default_factory=list)
    wall_thickness: float = 0.0
    inliers: Optional[np.ndarray] = None
    rms: float = 0.0
    n_points: int = 0
    elongation: float = 0.0              # λ1 / max(λ2, λ3) of the cluster PCA
    # Per-node section (aligned with ``path``; len == len(path)). The
    # tessellator / IFC export use these to skin a variable cross-section.
    # For uniform-section fits, every entry is a copy of the representative.
    polygons_per_node: List[np.ndarray] = field(default_factory=list)
    params_per_node: List[dict] = field(default_factory=list)

    @property
    def is_straight(self) -> bool:
        return len(self.path) <= 2

    @property
    def path_length(self) -> float:
        p = np.asarray(self.path)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


# ── 2-D cross-section fitting ───────────────────────────────────────

def _min_area_rect(p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Min-area enclosing rectangle of 2-D points via rotating calipers on the
    convex hull. Returns (centre(2,), axes(2,2) unit columns, width, height)."""
    from scipy.spatial import ConvexHull
    if len(p2) < 3:
        c = p2.mean(0)
        return c, np.eye(2), 0.0, 0.0
    try:
        hull = ConvexHull(p2)
        h = p2[hull.vertices]
    except Exception:
        h = p2
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        ln = np.linalg.norm(e)
        if ln < _EPS:
            continue
        ex = e / ln
        ey = np.array([-ex[1], ex[0]])
        proj_x = h @ ex
        proj_y = h @ ey
        w = proj_x.max() - proj_x.min()
        hh = proj_y.max() - proj_y.min()
        area = w * hh
        if best is None or area < best[0]:
            cx = 0.5 * (proj_x.max() + proj_x.min())
            cy = 0.5 * (proj_y.max() + proj_y.min())
            centre = cx * ex + cy * ey
            best = (area, centre, np.column_stack([ex, ey]), float(w), float(hh))
    if best is None:
        c = p2.mean(0)
        return c, np.eye(2), 0.0, 0.0
    _, centre, axes, w, hh = best
    # canonicalise: width >= height, axes right-handed
    if hh > w:
        axes = np.column_stack([axes[:, 1], -axes[:, 0]])
        w, hh = hh, w
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    return centre, axes, w, hh


def _fit_cross_section(p2: np.ndarray, tol: float) -> dict:
    """Classify+fit a single slice's 2-D points. Returns a dict with keys:
    family ('circle'|'rect'|'free'), params, polygon (M,2 in this slice's frame),
    centre (2,), frame (2,2 unit columns), rms, inlier_frac."""
    n = len(p2)
    out = {"family": "free", "params": {}, "centre": p2.mean(0),
           "frame": np.eye(2), "rms": np.inf, "inlier_frac": 0.0, "polygon": None}
    if n < 4:
        out["polygon"] = p2.copy()
        return out

    # — circle —
    try:
        c, r = _fit_circle_ls(p2)
        if np.isfinite(r) and r > _EPS:
            dist = np.abs(np.linalg.norm(p2 - c, axis=1) - r)
            inl = dist < tol
            ifr = float(inl.mean())
            rms = float(np.sqrt(np.mean(dist[inl] ** 2))) if inl.any() else np.inf
            # require the points to actually wrap most of the circle (not an arc)
            ang = np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0])
            span = _ang_span(ang[inl]) if inl.any() else 0.0
            circle_ok = ifr >= 0.85 and span >= np.deg2rad(220)
        else:
            circle_ok, c, r, rms, ifr = False, None, None, np.inf, 0.0
    except Exception:
        circle_ok, c, r, rms, ifr = False, None, None, np.inf, 0.0

    # — rect —
    centre, axes, w, hh = _min_area_rect(p2)
    if w > _EPS and hh > _EPS:
        # the convex hull is biased outward by noise; re-estimate the side
        # lengths from the 1–99th percentile span of the projected coords (and
        # recentre), which is much closer to the true rectangle.
        loc0 = (p2 - centre) @ axes
        xlo, xhi = np.percentile(loc0[:, 0], [1.0, 99.0])
        ylo, yhi = np.percentile(loc0[:, 1], [1.0, 99.0])
        w = float(xhi - xlo)
        hh = float(yhi - ylo)
        centre = centre + (0.5 * (xlo + xhi)) * axes[:, 0] + (0.5 * (ylo + yhi)) * axes[:, 1]
        if hh > w:
            axes = np.column_stack([axes[:, 1], -axes[:, 0]])
            w, hh = hh, w
        local = (p2 - centre) @ axes
        # distance to rectangle boundary [-w/2,w/2]x[-h/2,h/2]
        dx = np.abs(np.abs(local[:, 0]) - w / 2)
        dy = np.abs(np.abs(local[:, 1]) - hh / 2)
        # a point near the boundary is near min(dx,dy) IF it's inside-ish; outside corners use hypot
        inside = (np.abs(local[:, 0]) <= w / 2 + tol) & (np.abs(local[:, 1]) <= hh / 2 + tol)
        d_rect = np.where(inside, np.minimum(dx, dy), np.hypot(np.maximum(np.abs(local[:, 0]) - w / 2, 0),
                                                               np.maximum(np.abs(local[:, 1]) - hh / 2, 0)))
        inl_r = d_rect < tol
        ifr_r = float(inl_r.mean())
        rms_r = float(np.sqrt(np.mean(d_rect[inl_r] ** 2))) if inl_r.any() else np.inf
    else:
        inl_r, ifr_r, rms_r = np.zeros(n, bool), 0.0, np.inf

    # — circle is a viable fit even when its strict gate failed; keep its
    #   numbers around for the "square rect → really a circle" rescue below —
    circle_any = (np.isfinite(rms) and rms < np.inf and c is not None and
                  r is not None and ifr >= 0.65)

    # decide
    if circle_ok and (rms <= rms_r or ifr >= ifr_r + 0.05):
        out.update(family="circle", params={"r": float(r)}, centre=np.asarray(c),
                   frame=np.eye(2), rms=float(rms), inlier_frac=float(ifr))
        th = np.linspace(0, 2 * np.pi, 33)[:-1]
        out["polygon"] = c + r * np.column_stack([np.cos(th), np.sin(th)])
    elif ifr_r >= 0.80:
        # A min-area rectangle whose sides are nearly equal *and* whose interior
        # is full of points lying near a single radius is a circle dressed in its
        # bounding box. Architecture / piping is full of these (round columns,
        # pipes, trunks); rect would be wrong both geometrically and for IFC.
        # Switch to circle if (a) the rect is roughly square, (b) the LS circle
        # is a credible fit (residual + inlier fraction), and (c) its radius is
        # consistent with the rect side. The strict-rect gate (span >= 220°)
        # is intentionally relaxed here — partial wrap is fine, the simpler
        # primitive should win the tie.
        aspect = (max(w, hh) / max(min(w, hh), _EPS))
        circle_rescue = (circle_any and aspect < 1.30 and rms <= 1.3 * rms_r
                         and abs(2 * r - 0.5 * (w + hh)) < 0.20 * max(w, hh))
        if circle_rescue:
            out.update(family="circle", params={"r": float(r)}, centre=np.asarray(c),
                       frame=np.eye(2), rms=float(rms), inlier_frac=float(ifr))
            th = np.linspace(0, 2 * np.pi, 33)[:-1]
            out["polygon"] = c + r * np.column_stack([np.cos(th), np.sin(th)])
        else:
            out.update(family="rect", params={"w": float(w), "h": float(hh)},
                       centre=np.asarray(centre), frame=axes, rms=float(rms_r),
                       inlier_frac=float(ifr_r))
            corners = np.array([[-w / 2, -hh / 2], [w / 2, -hh / 2],
                                [w / 2, hh / 2], [-w / 2, hh / 2]])
            out["polygon"] = centre + corners @ axes.T
    else:
        # free: concave hull of the slice points (fall back to convex if needed)
        out["polygon"] = _hull_2d(p2)
        out.update(family="free", params={}, centre=p2.mean(0), rms=0.0,
                   inlier_frac=1.0)
    return out


def _ang_span(ang: np.ndarray) -> float:
    if ang.size == 0:
        return 0.0
    a = np.sort(np.mod(ang, 2 * np.pi))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(2 * np.pi - gaps.max())


def _hull_2d(p2: np.ndarray) -> np.ndarray:
    """Outline of a free-form cross-section. Uses a *concave* hull so I / L /
    channel / rail sections keep their re-entrant shape (a convex hull would
    collapse them to a rectangle)."""
    try:
        from .outline import concave_hull_2d
        h = concave_hull_2d(np.asarray(p2, dtype=np.float64), ratio=0.12, allow_holes=False)
        if h is not None and len(h) >= 3:
            return h
    except Exception:
        pass
    try:
        from shapely.geometry import MultiPoint
        poly = MultiPoint([tuple(p) for p in p2]).convex_hull
        if poly.geom_type == "Polygon":
            return np.asarray(poly.exterior.coords)[:-1]
    except Exception:
        pass
    return p2.copy()


# ── path collinearity / resampling ─────────────────────────────────

def _line_residual(path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    c = path.mean(0)
    _, _, Vt = np.linalg.svd(path - c, full_matrices=False)
    dirv = Vt[0]
    proj = c + np.outer((path - c) @ dirv, dirv)
    return float(np.max(np.linalg.norm(path - proj, axis=1)))


def _resample_polyline(path: np.ndarray, step: float) -> np.ndarray:
    if len(path) < 2:
        return path
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < _EPS:
        return path[[0, -1]]
    ns = max(2, int(np.ceil(total / max(step, 1e-3))) + 1)
    su = np.linspace(0, total, ns)
    return np.column_stack([np.interp(su, s, path[:, 0]),
                            np.interp(su, s, path[:, 1]),
                            np.interp(su, s, path[:, 2])])


# ── main ────────────────────────────────────────────────────────────

def _median_params(params_list: List[dict]) -> dict:
    """Aggregate per-slice params dicts into a single representative dict by
    median per key. Returns an empty dict if the list is empty or all dicts are."""
    if not params_list:
        return {}
    keys = set()
    for p in params_list:
        keys.update(p.keys())
    out = {}
    for k in keys:
        vals = [float(p[k]) for p in params_list if k in p and np.isfinite(p[k])]
        if vals:
            out[k] = float(np.median(vals))
    return out


def _polygon_from_params(family: str, params: dict) -> Optional[np.ndarray]:
    """Build a parametric polygon centred at the origin for the well-known
    section families. Returns None for ``free`` (caller keeps the slice hull)."""
    if family == "circle" and "r" in params:
        r = float(params["r"])
        if r > _EPS:
            th = np.linspace(0, 2 * np.pi, 33)[:-1]
            return r * np.column_stack([np.cos(th), np.sin(th)])
    if family == "rect" and "w" in params and "h" in params:
        w, h = float(params["w"]), float(params["h"])
        if w > _EPS and h > _EPS:
            return np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                             [w / 2, h / 2], [-w / 2, h / 2]])
    return None


def _refit_slice_to_family(p2: np.ndarray, family: str, tol: float) -> dict:
    """Force-fit a slice's 2-D points to a specific family (used to make every
    slice agree on the dominant family after voting). Returns the same dict
    shape ``_fit_cross_section`` produces."""
    if family == "circle":
        try:
            c, r = _fit_circle_ls(p2)
            if np.isfinite(r) and r > _EPS:
                d = np.abs(np.linalg.norm(p2 - c, axis=1) - r)
                ifr = float((d < tol).mean())
                th = np.linspace(0, 2 * np.pi, 33)[:-1]
                return {"family": "circle", "params": {"r": float(r)},
                        "polygon": c + r * np.column_stack([np.cos(th), np.sin(th)]),
                        "centre": np.asarray(c), "frame": np.eye(2),
                        "rms": float(np.median(d)), "inlier_frac": ifr}
        except Exception:
            pass
    if family == "rect":
        centre, axes, w, hh = _min_area_rect(p2)
        if w > _EPS and hh > _EPS:
            loc0 = (p2 - centre) @ axes
            xlo, xhi = np.percentile(loc0[:, 0], [1.0, 99.0])
            ylo, yhi = np.percentile(loc0[:, 1], [1.0, 99.0])
            w = float(xhi - xlo); hh = float(yhi - ylo)
            centre = centre + (0.5 * (xlo + xhi)) * axes[:, 0] + (0.5 * (ylo + yhi)) * axes[:, 1]
            corners = np.array([[-w / 2, -hh / 2], [w / 2, -hh / 2],
                                [w / 2, hh / 2], [-w / 2, hh / 2]])
            return {"family": "rect", "params": {"w": w, "h": hh},
                    "polygon": centre + corners @ axes.T,
                    "centre": centre, "frame": axes, "rms": 0.0, "inlier_frac": 1.0}
    # free / unknown → concave hull
    return {"family": "free", "params": {}, "polygon": _hull_2d(p2),
            "centre": p2.mean(0), "frame": np.eye(2), "rms": 0.0, "inlier_frac": 1.0}


def _section_polygon_centred(sec: dict) -> np.ndarray:
    """Translate a slice section's polygon so its centre is at the origin (the
    convention SweptElement expects: polygons live in the local profile frame,
    paths carry world positions)."""
    poly = sec.get("polygon")
    if poly is None or len(poly) == 0:
        return np.zeros((1, 2))
    if sec.get("family") != "free":
        return np.asarray(poly) - np.asarray(sec.get("centre", poly.mean(0)))
    return np.asarray(poly) - np.asarray(poly).mean(0)


def fit_swept(points: np.ndarray, dist_thresh: float = 0.02,
              n_slices: Optional[int] = None,
              min_elongation: float = 3.0,
              up_hint: Optional[np.ndarray] = None) -> Optional[SweptFit]:
    """Fit a swept solid to an elongated point cluster (pipe / duct / member /
    rail). Returns ``None`` only if the cluster isn't elongated enough to be a
    swept thing; otherwise always returns a (possibly coarse) ``SweptFit``.

    The fit produces a section *per slice* (``polygons_per_node`` /
    ``params_per_node``), so a tapering trunk / cone / variable-diameter pipe
    can be reconstructed faithfully. The representative ``profile_polygon`` and
    ``profile_params`` are the median across slices — IFC parametric export and
    legacy uniform-section consumers can keep using them unchanged.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 12:
        return None
    ctr = pts.mean(0)
    Q = pts - ctr
    cov = (Q.T @ Q) / n
    evals, evecs = np.linalg.eigh(cov)        # ascending
    evals = np.maximum(evals, 1e-12)
    elong = float(np.sqrt(evals[2] / evals[1]))
    if elong < min_elongation:
        return None
    axis0 = _unit(evecs[:, 2])                 # principal (long) direction

    # — pass 1: slice along axis0, get centroids (path) —
    s = (pts - ctr) @ axis0
    K = n_slices or int(np.clip(np.sqrt(n) / 1.5, 6, 60))
    edges = np.quantile(s, np.linspace(0, 1, K + 1))
    u0, v0 = _orthobasis(axis0)
    path_pts: List[np.ndarray] = []
    bin_indices: List[np.ndarray] = []
    for i in range(K):
        m = (s >= edges[i]) & (s <= edges[i + 1])
        if m.sum() < 4:
            continue
        path_pts.append(pts[m].mean(0))
        bin_indices.append(np.where(m)[0])
    if len(path_pts) < 2:
        return None
    path_pts = np.asarray(path_pts)

    # straight vs polyline — tolerance is the larger of an absolute floor (point
    # cloud noise can wobble centroids by a few mm) and a *relative* fraction of
    # the cluster's lateral spread. We use the second-largest PCA eigenvalue as
    # a proxy for the cross-section "diameter": if the centroid path bends by
    # more than ~25 % of the diameter, the cluster has real curvature regardless
    # of its absolute length. Length-based tolerances alone are wrong — a thin
    # 2 m trunk that bends 5 cm should be curved, while a 50 m pipe that wobbles
    # 5 cm should be straight. The diameter normalisation captures both cases.
    _plen = float(np.sum(np.linalg.norm(np.diff(path_pts, axis=0), axis=1)))
    diam_est = 2.0 * float(np.sqrt(evals[1]))         # 2σ along the second PCA axis ≈ diameter
    residual = _line_residual(path_pts)
    tol_curve = max(2 * dist_thresh, 0.02, 0.25 * diam_est)
    straight = residual < tol_curve
    # — build the world-space path —
    if straight:
        # collapse the centroid scatter onto the LS line through the centroids
        c = path_pts.mean(0)
        _, _, Vt = np.linalg.svd(path_pts - c, full_matrices=False)
        dirv = _unit(Vt[0])
        tt = (path_pts - c) @ dirv
        path = np.array([c + tt.min() * dirv, c + tt.max() * dirv])
    else:
        # resample the curved centroid polyline so consecutive nodes are evenly
        # spaced; spacing scales with K so a long pipe with K=60 isn't oversampled
        path = _resample_polyline(path_pts,
                                  step=max(0.05, 0.5 * (s.max() - s.min()) / max(K, 1)))

    # tangent at each path node — used to choose a local frame for the slice fit
    n_nodes = len(path)
    if n_nodes < 2:
        return None
    tangents = np.zeros_like(path)
    for i in range(n_nodes):
        a = max(i - 1, 0)
        b = min(i + 1, n_nodes - 1)
        tangents[i] = _unit(path[b] - path[a])

    # — per-slice section fit —
    # For each path node, gather the cluster points whose axial projection lands
    # near the node, project them into the local (u_i, v_i) plane perpendicular
    # to the local tangent, and fit a section. Per-node slabs replace the single
    # whole-cluster projection used previously — that one loses the cross-section
    # information when the path is curved (every slice gets smeared into one
    # mancha) and when the section varies along the path.
    plen = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    band = max(plen / max(2 * (n_nodes - 1), 1), 0.05)   # half-thickness of each slab
    slice_secs: List[dict] = []
    slice_frames: List[np.ndarray] = []           # 3x3 [u, v, t] per node
    for i in range(n_nodes):
        t_i = tangents[i]
        u_i, v_i = _orthobasis(t_i)
        if up_hint is not None:
            v_try = _unit(np.asarray(up_hint) - (np.asarray(up_hint) @ t_i) * t_i)
            if np.linalg.norm(v_try) > 0.3:
                v_i = v_try
                u_i = np.cross(v_i, t_i)
        sel = np.abs((pts - path[i]) @ t_i) < band
        if sel.sum() < 8 and i in (0, n_nodes - 1):
            # endpoints can be sparse if the cluster tapers off — widen the band
            sel = np.abs((pts - path[i]) @ t_i) < (2.0 * band)
        if sel.sum() < 4:
            # fall back: project the whole cluster onto this node's plane —
            # better a noisy slice than nothing
            sel = np.ones(n, dtype=bool)
        rel = pts[sel] - path[i]
        p2 = np.column_stack([rel @ u_i, rel @ v_i])
        sec = _fit_cross_section(p2, tol=max(dist_thresh, 1.5 * dist_thresh))
        slice_secs.append(sec)
        slice_frames.append(np.column_stack([u_i, v_i, t_i]))

    # — family vote across slices — the dominant family becomes the element's
    # nominal type; non-conforming slices are *re-fitted* to that family so all
    # per-node polygons share semantics. This is the principled way to give one
    # SweptElement a uniform `profile_family` without losing per-node variation.
    families = [sec["family"] for sec in slice_secs]
    family = max(set(families), key=lambda f: (families.count(f),
                                               -("free" == f)))   # prefer non-"free" ties
    refit_tol = max(dist_thresh, 1.5 * dist_thresh)
    for i, sec in enumerate(slice_secs):
        if sec["family"] != family:
            # rebuild p2 for the slice and refit to the dominant family
            t_i = tangents[i]
            u_i, v_i, _ = slice_frames[i].T
            sel = np.abs((pts - path[i]) @ t_i) < band
            if sel.sum() < 4:
                sel = np.ones(n, dtype=bool)
            rel = pts[sel] - path[i]
            p2 = np.column_stack([rel @ u_i, rel @ v_i])
            slice_secs[i] = _refit_slice_to_family(p2, family, refit_tol)

    # — collect per-node polygons / params (centred polygons, world frames built
    #   from `tangents`/`slice_frames` so the tessellator can place them directly) —
    polygons_per_node = [_section_polygon_centred(sec) for sec in slice_secs]
    params_per_node = [dict(sec.get("params", {})) for sec in slice_secs]

    # representative (median) section
    rep_params = _median_params(params_per_node)
    rep_polygon = _polygon_from_params(family, rep_params)
    if rep_polygon is None:
        # `free` family or missing keys → use the middle slice's polygon
        mid = n_nodes // 2
        rep_polygon = polygons_per_node[mid].copy()

    # overall rms = median of per-slice rms (skip non-finite)
    rms_vals = [float(sec.get("rms", 0.0)) for sec in slice_secs
                if np.isfinite(sec.get("rms", np.inf))]
    rms = float(np.median(rms_vals)) if rms_vals else 0.0

    # shift the path laterally so it goes through each slice's section centre.
    # The section is centred at its own centre in the local frame; the path
    # node must follow so polygon (in u,v at origin) aligns with the cloud.
    for i, sec in enumerate(slice_secs):
        u_i, v_i, _ = slice_frames[i].T
        cu, cv = float(sec["centre"][0]), float(sec["centre"][1])
        path[i] = path[i] + cu * u_i + cv * v_i

    # frame at path[0]
    t0 = tangents[0]
    u0_f, v0_f, _ = slice_frames[0].T
    frame3 = np.column_stack([u0_f, v0_f, t0])

    return SweptFit(profile_family=family, profile_params=rep_params,
                    profile_polygon=rep_polygon, profile_frame=frame3, path=path,
                    wall_thickness=0.0, inliers=None, rms=rms, n_points=n,
                    elongation=elong,
                    polygons_per_node=polygons_per_node,
                    params_per_node=params_per_node)
