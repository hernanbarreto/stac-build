"""
Contour regularization (user concept 2026-08-29: "los segmentos tienen formas
bastante perfectas — la pared es un rectángulo con un semicírculo arriba, la
puerta lo mismo, la escalera un rectángulo, las ruedas círculos — ¿hay manera
de detectar la tendencia para perfeccionarlos?").

The surface is already idealized (plane/cylinder); this module idealizes its
BOUNDARY. Every contour of the audited occupancy grid — the outer outline and
each opening — is tested against a 2-D shape ladder, LOWEST DOF FIRST (the
same doctrine as the surface escalation):

    circle → rectangle → rounded rectangle → arch (rect + circular cap)
    → direction-snapped polygon → raw measured contour

A template is accepted only when it explains the measured contour within
tolerance (p95 boundary deviation gate) — otherwise the next rung is tried,
down to the raw contour. Nothing is invented: the accepted shape's parameters
(width, height, radius…) are reported as tool_measured, and the residual is
recorded.

The mesh is then rebuilt from the idealized region: a fine quad grid clipped
to the shapely polygon, with the boundary vertices PROJECTED exactly onto the
ideal outline — CAD-crisp edges, arcs included.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("SurfaceFit")


# ── template signed-boundary distances (points (N,2) in the shape frame) ────

def _dev_circle(P, cx, cy, r):
    return np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)


def _dev_round_rect(P, cx, cy, hw, hh, r):
    """|SDF| of a rounded rectangle (r=0 → sharp rectangle)."""
    qx = np.abs(P[:, 0] - cx) - (hw - r)
    qy = np.abs(P[:, 1] - cy) - (hh - r)
    ax = np.maximum(qx, 0.0)
    ay = np.maximum(qy, 0.0)
    outside = np.hypot(ax, ay)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return np.abs(outside + inside - r)


def _dev_arch(P, x0, x1, y0, cy, r):
    """Rect sides+bottom up to the springing line cy, circular cap above."""
    cx = 0.5 * (x0 + x1)
    devs = np.full(len(P), np.inf)
    # bottom edge
    m = (P[:, 0] >= x0) & (P[:, 0] <= x1)
    devs = np.minimum(devs, np.where(m, np.abs(P[:, 1] - y0), np.inf))
    # vertical sides (y0..cy)
    for xe in (x0, x1):
        m = (P[:, 1] >= y0) & (P[:, 1] <= cy)
        devs = np.minimum(devs, np.where(m, np.abs(P[:, 0] - xe), np.inf))
    # cap arc (above cy)
    m = P[:, 1] >= cy
    devs = np.minimum(devs, np.where(
        m, np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r), np.inf))
    # corners fallback: distance to the two springing points / bottom corners
    for px, py in ((x0, y0), (x1, y0), (x0, cy), (x1, cy)):
        devs = np.minimum(devs, np.hypot(P[:, 0] - px, P[:, 1] - py))
    return devs


def _fit_circle(P):
    """Kåsa least-squares circle."""
    A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
    b = (P ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def _outline_points(shape: dict, step: float) -> np.ndarray:
    """Sample the ideal outline as a closed polyline (shape frame)."""
    kind = shape["shape"]
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, max(16, int(2 * np.pi * shape["r"] / step)),
                        endpoint=False)
        return np.column_stack([shape["cx"] + shape["r"] * np.cos(t),
                                shape["cy"] + shape["r"] * np.sin(t)])
    if kind in ("rectangle", "rounded_rect"):
        cx, cy, hw, hh = shape["cx"], shape["cy"], shape["hw"], shape["hh"]
        r = shape.get("r", 0.0)
        pts = []
        # walk the rounded-rect boundary
        corners = [(cx + hw - r, cy + hh - r, 0), (cx - hw + r, cy + hh - r, 90),
                   (cx - hw + r, cy - hh + r, 180), (cx + hw - r, cy - hh + r, 270)]
        for ccx, ccy, a0 in corners:
            if r > 1e-6:
                n = max(2, int((np.pi / 2 * r) / step))
                t = np.deg2rad(np.linspace(a0, a0 + 90, n, endpoint=False))
                pts.append(np.column_stack([ccx + r * np.cos(t), ccy + r * np.sin(t)]))
            else:
                a = np.deg2rad(a0 + 45)
                pts.append(np.array([[ccx + np.sign(np.cos(a)) * 0,
                                      ccy + np.sign(np.sin(a)) * 0]])
                           + np.array([[np.sign(np.cos(a)) * 0, 0]]))
                pts[-1] = np.array([[ccx, ccy]])
        # densify straight edges by resampling the corner-to-corner polyline
        poly = np.vstack(pts)
        return _densify(poly, step)
    if kind == "arch":
        x0, x1, y0, cy, r = (shape["x0"], shape["x1"], shape["y0"],
                             shape["cy"], shape["r"])
        cx = 0.5 * (x0 + x1)
        n = max(8, int(np.pi * r / step))
        t = np.linspace(0, np.pi, n)
        cap = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
        poly = np.vstack([[x1, cy], cap[::1] if cap[0, 0] > cap[-1, 0] else cap[::-1],
                          [x0, cy], [x0, y0], [x1, y0]])
        return _densify(poly, step)
    if kind == "polygon":
        return _densify(np.asarray(shape["points"]), step)
    return np.asarray(shape["points"])


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        seg = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(seg / step)))
        for j in range(k):
            out.append(a + (b - a) * (j / k))
    return np.asarray(out)


def fit_contour(P: np.ndarray, tol: float, resolution: float) -> dict:
    """Try the shape ladder on a closed contour (points in the SHAPE frame,
    x horizontal / y up). Returns the accepted shape dict (with 'shape',
    parameters, 'p95_dev_m'); 'raw' when nothing passes."""
    def p95(d):
        return float(np.percentile(d, 95))

    x0, x1 = np.percentile(P[:, 0], [1, 99])
    y0, y1 = np.percentile(P[:, 1], [1, 99])
    w, h = x1 - x0, y1 - y0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    candidates = []   # (shape_dict, p95_dev, dof)
    # circle (DOF 3)
    c = _fit_circle(P)
    if c is not None and abs(w - h) < 0.35 * max(w, h):
        candidates.append(({"shape": "circle", "cx": c[0], "cy": c[1],
                            "r": c[2]}, p95(_dev_circle(P, *c)), 3))
    # rectangle (DOF 4)
    candidates.append(({"shape": "rectangle", "cx": cx, "cy": cy,
                        "hw": w / 2, "hh": h / 2},
                       p95(_dev_round_rect(P, cx, cy, w / 2, h / 2, 0.0)), 4))
    # rounded rectangle (DOF 5)
    best_r, best_d = 0.0, np.inf
    for r in np.linspace(0.0, 0.49 * min(w, h), 12):
        dr = p95(_dev_round_rect(P, cx, cy, w / 2, h / 2, r))
        if dr < best_d:
            best_r, best_d = float(r), dr
    candidates.append(({"shape": "rounded_rect", "cx": cx, "cy": cy,
                        "hw": w / 2, "hh": h / 2, "r": best_r}, best_d, 5))
    # arch (DOF 5): rect + circular cap centred on top
    r_arch = w / 2
    best = None
    for spring in np.linspace(y0 + 0.3 * h, y1 - 0.1 * h, 10):
        da = p95(_dev_arch(P, x0, x1, y0, spring, r_arch))
        if best is None or da < best[1]:
            best = (float(spring), da)
    if best is not None:
        candidates.append(({"shape": "arch", "x0": float(x0), "x1": float(x1),
                            "y0": float(y0), "cy": best[0],
                            "r": float(r_arch)}, best[1], 5))
    # direction-snapped polygon (DOF 2n)
    try:
        import cv2
        scale = 1.0 / max(resolution, 1e-6)
        cnt = ((P - P.min(0)) * scale).astype(np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(cnt, 1.8, True).reshape(-1, 2) / scale + P.min(0)
        if 3 <= len(approx) <= 24:
            snapped = _snap_polygon(approx)
            candidates.append(({"shape": "polygon",
                                "points": snapped.tolist(),
                                "n_vertices": int(len(snapped))},
                               p95(_poly_dev(P, snapped)), 2 * len(snapped)))
    except Exception:  # noqa: BLE001
        pass

    # best PASSING template: smallest deviation (cm-rounded), then lowest DOF
    passing = [(s, d, dof) for s, d, dof in candidates if d <= tol]
    if passing:
        s, d, _dof = min(passing, key=lambda x: (round(x[1], 2), x[2]))
        s = dict(s)
        s["p95_dev_m"] = round(float(d), 4)
        return s
    return {"shape": "raw", "points": P.tolist(), "p95_dev_m": 0.0}


def _snap_polygon(V: np.ndarray, snap_deg: float = 12.0) -> np.ndarray:
    """Snap polygon edges to horizontal/vertical when close, rebuilding
    vertices at consecutive edge intersections."""
    n = len(V)
    dirs = []
    for i in range(n):
        e = V[(i + 1) % n] - V[i]
        ang = np.rad2deg(np.arctan2(e[1], e[0])) % 180.0
        if min(ang, 180 - ang) < snap_deg:
            ang = 0.0
        elif abs(ang - 90) < snap_deg:
            ang = 90.0
        dirs.append(np.deg2rad(ang))
    out = []
    for i in range(n):
        p_prev, a_prev = V[i], dirs[(i - 1) % n]
        p_cur, a_cur = V[i], dirs[i]
        d1 = np.array([np.cos(a_prev), np.sin(a_prev)])
        d2 = np.array([np.cos(a_cur), np.sin(a_cur)])
        # intersect line through prev-edge midpointish with cur edge direction
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-9:
            out.append(p_cur)
            continue
        m_prev = 0.5 * (V[(i - 1) % n] + V[i])
        m_cur = 0.5 * (V[i] + V[(i + 1) % n])
        t = np.linalg.solve(A, m_cur - m_prev)
        out.append(m_prev + t[0] * d1)
    return np.asarray(out)


def _poly_dev(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Distance from points to a closed polygon outline."""
    d = np.full(len(P), np.inf)
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.minimum(d, np.hypot(*(P - proj).T))
    return d


# ── main entry ──────────────────────────────────────────────────────────────

def regularize_mesh(model, keep: np.ndarray, open_grid: Optional[np.ndarray],
                    u0: float, v0: float,
                    resolution: float, tol: float = 0.12,
                    min_hole_area_m2: float = 0.04):
    """Vectorize the audited occupancy into idealized contours and rebuild a
    crisp mesh. The OUTER outline is fitted against the morphologically
    CLOSED support (the boundary's intent, not the ragged ambiguity bites);
    the OPENINGS come from the audit's image-confirmed open cells. Returns
    (verts_uv, faces, contour_reports) or None (caller keeps the grid mesh)."""
    import cv2
    import shapely
    from scipy import ndimage as _ndi
    from shapely.geometry import Polygon

    closed = _ndi.binary_closing(keep, structure=np.ones((5, 5)), iterations=1)
    closed = _ndi.binary_fill_holes(closed) | keep
    grid = (closed.astype(np.uint8) * 255)
    cnts, hier = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if hier is None or not len(cnts):
        return None
    hier = hier[0]

    # shape frame: x = horizontal, y = display-up projected into UV
    up_uv = None
    try:
        p0 = np.asarray(model.uv_to_world(np.array([[u0, v0]])))[0]
        d = (np.asarray(model.to_uv(np.array([p0 + np.array([0, 1.0, 0])])))[0]
             - np.asarray(model.to_uv(np.array([p0])))[0])
        if np.linalg.norm(d) > 0.2:   # degenerate for horizontal surfaces
            up_uv = d / np.linalg.norm(d)
    except Exception:  # noqa: BLE001
        up_uv = None
    if up_uv is None:
        up_uv = np.array([0.0, 1.0])
    R = np.array([[up_uv[1], -up_uv[0]], [up_uv[0], up_uv[1]]])  # up → +y

    def to_uv_pts(cnt):
        c = cnt.reshape(-1, 2).astype(np.float64)
        return np.column_stack([u0 + (c[:, 0] + 0.5) * resolution,
                                v0 + (c[:, 1] + 0.5) * resolution])

    reports: List[dict] = []
    outer_polys = []
    for cnt in cnts:
        if len(cnt) < 8:
            continue
        P = to_uv_pts(cnt) @ R.T
        shape = fit_contour(P, tol, resolution)
        reports.append({"role": "outer", **{k: v for k, v in shape.items()
                                            if k != "points"}})
        ext = _outline_points(shape, resolution / 2.0) @ R
        try:
            poly = Polygon(ext)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.area > 4 * resolution * resolution:
                outer_polys.append(poly)
        except Exception:  # noqa: BLE001
            continue
    if not outer_polys:
        return None
    region = shapely.union_all(outer_polys)

    # openings = the audit's image-confirmed OPEN regions (the arch, the
    # doorway) — carved out of the idealized outline with their own template
    if open_grid is not None and open_grid.any():
        og = _ndi.binary_closing(open_grid, structure=np.ones((3, 3)))
        hcnts, _h = cv2.findContours((og.astype(np.uint8) * 255),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in hcnts:
            if len(cnt) < 8:
                continue
            Ph = to_uv_pts(cnt) @ R.T
            try:
                if Polygon(Ph).area < min_hole_area_m2:
                    continue
            except Exception:  # noqa: BLE001
                continue
            hshape = fit_contour(Ph, tol, resolution)
            reports.append({"role": "opening",
                            **{k: v for k, v in hshape.items()
                               if k != "points"}})
            hole = _outline_points(hshape, resolution / 2.0) @ R
            try:
                hp = Polygon(hole)
                if not hp.is_valid:
                    hp = hp.buffer(0)
                region = region.difference(hp)
            except Exception:  # noqa: BLE001
                continue

    # ── crisp re-mesh: fine grid clipped to the region, boundary vertices
    #    projected exactly onto the ideal outline
    res_f = resolution / 2.0
    minx, miny, maxx, maxy = region.bounds
    nu = max(1, int(np.ceil((maxx - minx) / res_f)))
    nv = max(1, int(np.ceil((maxy - miny) / res_f)))
    if nu * nv > 4_000_000:
        res_f = resolution
        nu = max(1, int(np.ceil((maxx - minx) / res_f)))
        nv = max(1, int(np.ceil((maxy - miny) / res_f)))
    xs = minx + (np.arange(nu) + 0.5) * res_f
    ys = miny + (np.arange(nv) + 0.5) * res_f
    XX, YY = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(region, XX.ravel(), YY.ravel()).reshape(nv, nu)
    if not inside.any():
        return None
    jj, ii = np.nonzero(inside)
    corners = np.stack([jj * (nu + 1) + ii,
                        jj * (nu + 1) + ii + 1,
                        (jj + 1) * (nu + 1) + ii,
                        (jj + 1) * (nu + 1) + ii + 1], axis=1)
    used = np.unique(corners)
    remap = np.full((nv + 1) * (nu + 1), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    cc = remap[corners]
    faces = np.concatenate([cc[:, [0, 1, 3]], cc[:, [0, 3, 2]]], axis=0)
    gj, gi = np.divmod(used, nu + 1)
    verts = np.column_stack([minx + gi * res_f, miny + gj * res_f])

    # boundary vertices → exact projection on the ideal outline
    boundary = region.boundary
    from shapely.geometry import Point
    dists = np.array([boundary.distance(Point(p)) for p in verts])
    edge = dists < res_f * 1.0
    for k in np.nonzero(edge)[0]:
        q = boundary.interpolate(boundary.project(Point(verts[k])))
        verts[k] = (q.x, q.y)
    # drop faces collapsed by the projection
    tri = verts[faces]
    area2 = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                   - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    faces = faces[area2 > res_f * res_f * 1e-4]

    verts_uv = verts @ np.linalg.inv(R).T   # shape frame → UV
    return verts_uv, faces, reports


def ideal_open_outlines(model, open_grid: np.ndarray, u0: float, v0: float,
                        resolution: float, tol: float = 0.12,
                        min_area_m2: float = 0.04) -> List[np.ndarray]:
    """Idealized outlines (UV polylines) of the image-confirmed OPEN regions —
    used to snap cut borders onto clean shapes (user 2026-08-29: cut edges
    looked 'bitten'). Each open region goes through the same 2-D shape ladder;
    'raw' contours are returned as-is (still a snap target, just unidealized)."""
    import cv2
    from scipy import ndimage as _ndi
    from shapely.geometry import Polygon

    if open_grid is None or not open_grid.any():
        return []
    og = _ndi.binary_closing(open_grid, structure=np.ones((3, 3)))
    cnts, _h = cv2.findContours((og.astype(np.uint8) * 255),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # shape frame (display-up in UV), same convention as regularize_mesh
    up_uv = None
    try:
        p0 = np.asarray(model.uv_to_world(np.array([[u0, v0]])))[0]
        d = (np.asarray(model.to_uv(np.array([p0 + np.array([0, 1.0, 0])])))[0]
             - np.asarray(model.to_uv(np.array([p0])))[0])
        if np.linalg.norm(d) > 0.2:
            up_uv = d / np.linalg.norm(d)
    except Exception:  # noqa: BLE001
        up_uv = None
    if up_uv is None:
        up_uv = np.array([0.0, 1.0])
    R = np.array([[up_uv[1], -up_uv[0]], [up_uv[0], up_uv[1]]])

    outlines: List[np.ndarray] = []
    for cnt in cnts:
        if len(cnt) < 8:
            continue
        c = cnt.reshape(-1, 2).astype(np.float64)
        P_uv = np.column_stack([u0 + (c[:, 0] + 0.5) * resolution,
                                v0 + (c[:, 1] + 0.5) * resolution])
        try:
            if Polygon(P_uv).area < min_area_m2:
                continue
        except Exception:  # noqa: BLE001
            continue
        P = P_uv @ R.T
        shape = fit_contour(P, tol, resolution)
        outlines.append(_outline_points(shape, resolution / 2.0) @ R)
    return outlines
