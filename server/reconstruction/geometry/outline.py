"""
Surface outlines & attached profiles.

  concave_hull_2d   — α-shape (concave hull) of a 2-D point set, ordered CCW.
  clip_polygon      — boolean ops on 2-D polygons-with-holes (shapely wrappers).
  residual_bands    — find points that protrude from a dominant plane and form
                      a band running along it (skirting / cornice / pilaster /
                      frame), returning `ProfileElement`-shaped descriptors.

2-D coordinates here are in a surface's parameter space (for a plane: metres
along the in-plane basis u, v).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import shapely
    from shapely.geometry import MultiPoint, Polygon, LineString
    from shapely import concave_hull as _shapely_concave_hull
    _HAS_SHAPELY = True
except Exception:  # pragma: no cover
    _HAS_SHAPELY = False


def _convex_hull_2d(p2: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    if len(p2) < 3:
        return p2.copy()
    h = ConvexHull(p2)
    return p2[h.vertices]


def _ccw(poly: np.ndarray) -> np.ndarray:
    """Ensure counter-clockwise winding."""
    if len(poly) < 3:
        return poly
    a = 0.5 * np.sum(poly[:-1, 0] * poly[1:, 1] - poly[1:, 0] * poly[:-1, 1]) \
        + 0.5 * (poly[-1, 0] * poly[0, 1] - poly[0, 0] * poly[-1, 1])
    return poly if a >= 0 else poly[::-1]


def concave_hull_2d(p2: np.ndarray, ratio: float = 0.25,
                    allow_holes: bool = False) -> np.ndarray:
    """Concave hull (α-shape-ish) of 2-D points → ordered (N,2) CCW polygon.

    `ratio` ∈ (0,1]: smaller = tighter/concaver (shapely's parameterisation).
    Falls back to the convex hull if shapely is unavailable or the result is
    degenerate.
    """
    p2 = np.asarray(p2, dtype=np.float64)
    if len(p2) < 4:
        return _ccw(p2.copy())
    if _HAS_SHAPELY:
        try:
            mp = MultiPoint([tuple(x) for x in p2])
            poly = _shapely_concave_hull(mp, ratio=float(np.clip(ratio, 0.01, 1.0)),
                                         allow_holes=bool(allow_holes))
            if poly is not None and poly.geom_type == "Polygon" and poly.area > 1e-9:
                ring = np.asarray(poly.exterior.coords)[:-1]
                if len(ring) >= 3:
                    return _ccw(ring)
        except Exception:
            pass
    return _ccw(_convex_hull_2d(p2))


def rectangle_outline(umin: float, umax: float, vmin: float, vmax: float) -> np.ndarray:
    return np.array([[umin, vmin], [umax, vmin], [umax, vmax], [umin, vmax]], dtype=np.float64)


def clean_polygon(ring: np.ndarray, tol: float = 0.03, smooth_iters: int = 0) -> np.ndarray:
    """Tidy a closed 2-D polygon ring: Douglas-Peucker simplify (``tol`` in the
    ring's units) → optional Chaikin corner-cutting (``smooth_iters`` rounds).

    Kills the fine zig-zag that a concave hull of a noisy point cloud carries —
    the "escalonado" / ragged-edge artifact — while keeping the real shape. Always
    returns ≥3 vertices, CCW; falls back to the input on any failure.
    """
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) < 4:
        return _ccw(ring.copy()) if len(ring) >= 3 else ring
    out = ring
    if _HAS_SHAPELY and tol > 0:
        try:
            closed = np.vstack([ring, ring[:1]])
            simp = LineString(closed).simplify(float(tol), preserve_topology=False)
            cc = np.asarray(simp.coords, dtype=np.float64)
            if len(cc) >= 4:
                out = cc[:-1] if np.allclose(cc[0], cc[-1]) else cc
        except Exception:
            out = ring
    for _ in range(max(0, int(smooth_iters))):
        if len(out) < 4:
            break
        a = out
        b = np.roll(out, -1, axis=0)
        nxt = np.empty((2 * len(a), 2), dtype=np.float64)
        nxt[0::2] = 0.75 * a + 0.25 * b
        nxt[1::2] = 0.25 * a + 0.75 * b
        out = nxt
    return _ccw(out) if len(out) >= 3 else ring


def _is_polyline_straight(fp: np.ndarray, node_spacing: float,
                          max_bend_deg: float = 20.0) -> bool:
    """A polyline is *straight* iff (a) every interior vertex bends by less than
    ``max_bend_deg`` against its neighbours, AND (b) every interior vertex lies
    within ``node_spacing`` of the endpoint-to-endpoint chord.

    Either condition alone is insufficient:
      - chord-only misses small L corners whose chord deviation is below
        ``node_spacing`` even though there's a genuine 40°+ bend at the kink;
      - angle-only misses gentle banana-shaped polylines whose every bend is
        small but the cumulative arc is far from the chord.

    The combination catches both: an L fails (a) (sharp interior bend) and a
    banana curve fails (b) (chord deviation > node_spacing)."""
    fp = np.asarray(fp, dtype=np.float64)
    if len(fp) <= 2:
        return True
    chord = fp[-1] - fp[0]
    L = float(np.linalg.norm(chord))
    if L < 1e-6:
        return True
    cn = np.array([-chord[1] / L, chord[0] / L])
    dev = float(np.abs((fp - fp[0]) @ cn).max())
    if dev >= max(node_spacing, 0.03):
        return False
    if len(fp) >= 3:
        seg = np.diff(fp, axis=0)
        seg_n = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)
        cos = np.clip((seg_n[:-1] * seg_n[1:]).sum(axis=1), -1.0, 1.0)
        max_bend = float(np.degrees(np.arccos(cos)).max())
        if max_bend > max_bend_deg:
            return False
    return True


def _rdp_simplify(poly: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an OPEN polyline (preserves
    endpoints). Drops vertices whose perpendicular distance to the chord between
    their surviving neighbours is below ``epsilon``."""
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, dtype=bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        ab = p[b] - p[a]
        L = float(np.linalg.norm(ab))
        if L < 1e-9:
            continue
        normal = np.array([-ab[1] / L, ab[0] / L])
        d = np.abs((p[a + 1:b] - p[a]) @ normal)
        if d.size == 0:
            continue
        k = int(np.argmax(d))
        if d[k] > epsilon:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return p[keep]


def _trace_skeleton_endpoint_to_endpoint(skel: np.ndarray) -> Optional[np.ndarray]:
    """Return ``(M, 2)`` pixel coordinates ``(col, row)`` of the longest
    endpoint-to-endpoint path through a boolean 2-D skeleton, or ``None`` if
    the skeleton lacks endpoints. Handles stubs (>2 endpoints) by picking the
    two whose geodesic separation is largest."""
    from collections import deque
    from scipy.ndimage import convolve
    if skel.sum() < 4:
        return None
    nbr = convolve(skel.astype(np.int32),
                   np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32),
                   mode="constant", cval=0) * skel
    endpoints = [tuple(rc) for rc in np.argwhere(nbr == 1)]
    if len(endpoints) < 2:
        return None
    Hsk, Wsk = skel.shape

    def bfs(src):
        dists = {src: 0}; parents = {src: None}; q = deque([src])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < Hsk and 0 <= nc < Wsk and skel[nr, nc] and (nr, nc) not in dists:
                        dists[(nr, nc)] = dists[(r, c)] + 1
                        parents[(nr, nc)] = (r, c)
                        q.append((nr, nc))
        return dists, parents

    if len(endpoints) == 2:
        start, end = endpoints
    else:
        best = None
        for e0 in endpoints:
            d, _ = bfs(e0)
            for e1, dist in d.items():
                if e1 == e0:
                    continue
                if best is None or dist > best[0]:
                    best = (dist, e0, e1)
        if best is None:
            return None
        _, start, end = best
    _, parents = bfs(start)
    if end not in parents:
        return None
    path = [end]
    while path[-1] != start:
        p = parents.get(path[-1])
        if p is None:
            return None
        path.append(p)
    path.reverse()
    return np.array([(c, r) for (r, c) in path], dtype=np.float64)


def _footprint_from_tsdf(tsdf_mesh, up: np.ndarray, node_spacing: float,
                         min_nodes: int = 2) -> Optional[Dict]:
    """Extract the wall footprint directly from the TSDF mesh.

    The pipeline:
      1. Slice the mesh at mid-height ± 10 cm → a 2-D band of vertices.
      2. Rasterise to a 2 cm grid, close gaps morphologically, fill interior.
      3. Compute the medial axis (skimage skeleton).
      4. Trace endpoint-to-endpoint through the skeleton.
      5. Simplify with Ramer-Douglas-Peucker @ ``node_spacing / 2``.

    Why this works where the cloud-based PCA-and-bin path fails: the TSDF is a
    *closed* surface that integrates the cloud over time, so it's free of the
    gaps and density variations that mislead a cloud-only fit. A wall with an
    L-shaped footprint slices to a closed L in the TSDF (even if the cloud has
    a gap at the corner), and the medial axis recovers the 3-vertex polyline
    cleanly. A curved wall slices to a banded ribbon and the medial axis
    follows it without a single-axis PCA flattening it.

    Returns the same dict shape as ``fit_footprint_polyline`` (e1, e2, up,
    base_o, base_s, height, thickness, path3d, is_straight, fit_method)
    so the caller can substitute results transparently. ``None`` if the slice
    is degenerate or the skeleton lacks endpoints."""
    try:
        from skimage.morphology import medial_axis
        from scipy.ndimage import binary_closing, binary_fill_holes
    except Exception:
        return None
    if tsdf_mesh is None:
        return None
    try:
        V = np.asarray(tsdf_mesh.vertices, dtype=np.float64)
    except Exception:
        return None
    if len(V) < 50:
        return None

    up = up / (np.linalg.norm(up) + 1e-12)
    a0 = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a0 - (a0 @ up) * up
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(up, e1)
    o = V.mean(0)
    s = (V - o) @ up
    s_min, s_max = float(s.min()), float(s.max())
    height = s_max - s_min
    if height < 0.2:
        return None

    # Band ±10cm around mid-height — slices a 20 cm tall strip of the wall.
    # Mid-height stays well clear of any base/cap noise; a band rather than a
    # single z-value gives the rasteriser enough vertices to draw a solid blob.
    s_mid = 0.5 * (s_min + s_max)
    band = 0.10
    sel = np.abs(s - s_mid) < band
    if sel.sum() < 50:
        return None
    rel = V[sel] - o
    UV = np.column_stack([rel @ e1, rel @ e2])

    pix = 0.02
    u0 = float(UV[:, 0].min()) - 0.10
    u1 = float(UV[:, 0].max()) + 0.10
    v0 = float(UV[:, 1].min()) - 0.10
    v1 = float(UV[:, 1].max()) + 0.10
    W = int(np.ceil((u1 - u0) / pix)) + 1
    H = int(np.ceil((v1 - v0) / pix)) + 1
    if W * H > 1_500_000:
        pix = max(pix, float(np.sqrt((u1 - u0) * (v1 - v0) / 1_500_000)))
        W = int(np.ceil((u1 - u0) / pix)) + 1
        H = int(np.ceil((v1 - v0) / pix)) + 1
    img = np.zeros((H, W), dtype=bool)
    cc = ((UV[:, 0] - u0) / pix).astype(np.int64)
    rr = ((UV[:, 1] - v0) / pix).astype(np.int64)
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    img[rr[valid], cc[valid]] = True
    img = binary_closing(img, iterations=3)
    img = binary_fill_holes(img).astype(bool)
    n_fill = int(img.sum())
    if n_fill < 20:
        return None
    skel = medial_axis(img)
    pix_path = _trace_skeleton_endpoint_to_endpoint(skel)
    if pix_path is None or len(pix_path) < 2:
        return None
    UV_path = np.column_stack([u0 + (pix_path[:, 0] + 0.5) * pix,
                               v0 + (pix_path[:, 1] + 0.5) * pix])

    # Densidad: epsilon = node_spacing/3 ≈ 4 cm para node_spacing=12 cm. Más
    # nodos en curvas (suavidad visual) sin amplificar ruido del TSDF < ~3 cm.
    # RDP filtra detalle < epsilon, por lo que el ruido fino del integrador
    # volumétrico se elimina mientras la geometría real se preserva.
    rdp_eps = max(node_spacing * 0.33, 0.03)
    fp_uv = _rdp_simplify(UV_path, rdp_eps)
    if len(fp_uv) < 2:
        return None

    spine_len = float(np.linalg.norm(np.diff(UV_path, axis=0), axis=1).sum())
    thickness = (n_fill * pix * pix) / max(spine_len, 0.1)
    thickness = float(np.clip(thickness, 0.02, 0.50))

    is_straight = _is_polyline_straight(fp_uv, node_spacing)
    if is_straight:
        fp_uv = np.array([fp_uv[0], fp_uv[-1]], dtype=np.float64)

    path3d = (o[None, :] + fp_uv[:, [0]] * e1[None, :]
              + fp_uv[:, [1]] * e2[None, :] + s_min * up[None, :])
    return {"e1": e1, "e2": e2, "up": up, "base_o": o, "base_s": s_min,
            "height": float(height), "thickness": thickness,
            "path3d": np.ascontiguousarray(path3d), "is_straight": bool(is_straight),
            "fit_method": "tsdf_medial_axis"}


def fit_footprint_polyline(pts3d: np.ndarray, up: np.ndarray,
                           node_spacing: float = 0.12, min_nodes: int = 4,
                           default_thickness: float = 0.06,
                           tsdf_mesh=None) -> Optional[Dict]:
    """Fit a 2-D *footprint* polyline for any wall — straight, curved, or with
    one or more corners — so it can be represented uniformly as the extrusion
    of its real path.

    **Primary source is the TSDF mesh** when available: we slice it at
    mid-height, rasterise to a 2 cm grid, take the medial axis of the filled
    polygon and simplify with RDP. This works uniformly for straight, curved,
    L, U and zig-zag walls because the TSDF integrates the cloud into a closed
    surface with no gaps.

    When the TSDF is missing we fall back to the legacy cloud path: project to
    the horizontal plane, order by PCA principal axis, bin median, Chaikin.
    That path is known to handle smooth curves correctly but flattens L-shapes
    — acceptable as a fallback since a session without TSDF is unusual.

    Returns ``{e1, e2, up, base_o, base_s, height, thickness, path3d (K,3),
    is_straight, fit_method}`` — ``path3d`` is the polyline lifted to the wall
    *base* (``base_o + base_s·up``); the wall extrudes from there up by
    ``height``.
    """
    if tsdf_mesh is not None:
        out = _footprint_from_tsdf(tsdf_mesh, up, node_spacing, min_nodes=min_nodes)
        if out is not None:
            return out

    # ── Cloud fallback (TSDF unavailable / degenerate) ──
    pts = np.asarray(pts3d, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    if len(pts) < max(min_nodes * 6, 30):
        return None
    a0 = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a0 - (a0 @ up) * up
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(up, e1)
    o = pts.mean(0)
    x = (pts - o) @ e1
    y = (pts - o) @ e2
    s = (pts - o) @ up
    s_min, s_max = float(s.min()), float(s.max())
    height = s_max - s_min
    if height < 0.2:
        return None
    XY = np.column_stack([x, y])
    Q = XY - XY.mean(0)
    _, evec = np.linalg.eigh(Q.T @ Q)
    long_dir = evec[:, 1]
    t = Q @ long_dir
    span_t = float(t.max() - t.min())
    if span_t < 4.0 * node_spacing:
        return None
    nbins = int(np.clip(span_t / node_spacing, min_nodes, 80))
    edges = np.linspace(t.min(), t.max(), nbins + 1)
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
    nodes = [np.median(XY[bin_idx == b], axis=0) for b in range(nbins) if (bin_idx == b).sum() >= 3]
    if len(nodes) < min_nodes:
        return None
    fp = np.asarray(nodes, dtype=np.float64)
    for _ in range(2):
        if len(fp) < 3:
            break
        nxt = [fp[0]]
        for i in range(len(fp) - 1):
            a, b = fp[i], fp[i + 1]
            nxt.append(0.75 * a + 0.25 * b)
            nxt.append(0.25 * a + 0.75 * b)
        nxt.append(fp[-1])
        fp = np.asarray(nxt, dtype=np.float64)
    is_straight = _is_polyline_straight(fp, node_spacing)
    if is_straight:
        fp = np.array([fp[0], fp[-1]], dtype=np.float64)
    path3d = (o[None, :] + fp[:, [0]] * e1[None, :] + fp[:, [1]] * e2[None, :]
              + s_min * up[None, :])
    return {"e1": e1, "e2": e2, "up": up, "base_o": o, "base_s": s_min,
            "height": float(height), "thickness": float(default_thickness),
            "path3d": np.ascontiguousarray(path3d), "is_straight": bool(is_straight),
            "fit_method": "cloud_pca_fallback"}


def to_shapely_polygon(outer: np.ndarray, holes: Optional[List[np.ndarray]] = None):
    if not _HAS_SHAPELY:  # pragma: no cover
        raise RuntimeError("shapely required")
    return Polygon(np.asarray(outer), [np.asarray(h) for h in (holes or [])])


def clip_polygon(outer: np.ndarray, cutters: List[np.ndarray]):
    """Return shapely polygon-with-holes = outer minus each cutter polygon."""
    poly = to_shapely_polygon(outer)
    for c in cutters:
        try:
            poly = poly.difference(to_shapely_polygon(np.asarray(c)))
        except Exception:
            pass
    return poly


# ── attached profiles (skirting / cornice / pilaster / frame) ──────

def residual_bands(points3d: np.ndarray, plane_normal: np.ndarray, plane_d: float,
                   basis_origin: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray,
                   world_up: Optional[np.ndarray] = None,
                   on_plane_tol: float = 0.012, band_max_depth: float = 0.25,
                   min_band_points: int = 60, min_band_length: float = 0.3,
                   max_band_width: float = 0.15) -> List[Dict]:
    """Find points that protrude from a dominant plane and form a band running
    along it. Returns a list of dicts:
        {kind, path (2-D polyline in plane basis), section (2-D profile:
         [depth × thickness] rectangle), depth, thickness, point_mask}.

    `kind` ∈ {skirting, cornice, pilaster, frame, other}: a band whose long axis
    is ~⊥ to in-plane up is a skirting (near v_min) / cornice (near v_max); a
    band whose long axis is ~∥ to in-plane up is a pilaster (or a frame).
    """
    pts = np.asarray(points3d, dtype=np.float64)
    if len(pts) < min_band_points:
        return []
    signed = pts @ plane_normal + plane_d           # perpendicular offset
    # candidates: outside the on-plane band but within band_max_depth (one side; the
    # protruding side — take whichever side has more points)
    near = np.abs(signed) < on_plane_tol
    side_pos = (signed >= on_plane_tol) & (signed < band_max_depth)
    side_neg = (signed <= -on_plane_tol) & (signed > -band_max_depth)
    cand = side_pos if side_pos.sum() >= side_neg.sum() else side_neg
    if cand.sum() < min_band_points:
        return []
    sign = 1.0 if cand is side_pos else -1.0

    # 2-D coords in plane basis
    rel = pts - basis_origin
    uu = rel @ basis_u
    vv = rel @ basis_v
    # in-plane "up" direction in (u,v) coords
    if world_up is not None:
        up3 = np.asarray(world_up, dtype=np.float64)
        up_uv = np.array([up3 @ basis_u, up3 @ basis_v])
        nrm = np.linalg.norm(up_uv)
        up_uv = up_uv / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
    else:
        up_uv = np.array([0.0, 1.0])
    v_lo, v_hi = float(np.min(vv[near] if near.any() else vv)), float(np.max(vv[near] if near.any() else vv))

    # cluster the candidate points in (u,v). A skirting and a vertical frame
    # touch, so DBSCAN on raw points would merge them — first split candidates
    # by local orientation (horizontal-running vs vertical-running in the wall)
    # using a per-point local PCA, then DBSCAN within each orientation group.
    try:
        from sklearn.cluster import DBSCAN
        from scipy.spatial import cKDTree
    except Exception:
        return []
    idx = np.where(cand)[0]
    XY = np.column_stack([uu[idx], vv[idx]])
    if len(XY) < min_band_points:
        return []
    tree = cKDTree(XY)
    kk = int(min(max(8, len(XY) // 40), len(XY) - 1))
    _, nbi = tree.query(XY, k=kk)
    horiz = np.zeros(len(XY), dtype=bool)
    perp_up = np.array([-up_uv[1], up_uv[0]])  # in-plane "horizontal"
    for i in range(len(XY)):
        nb = XY[nbi[i]]
        Q = nb - nb.mean(0)
        ev, evec = np.linalg.eigh(Q.T @ Q)
        ld = evec[:, 1]
        horiz[i] = abs(ld @ perp_up) >= abs(ld @ up_uv)
    eps = max(2.5 * on_plane_tol, 0.05)
    labels = np.full(len(XY), -1)
    next_lab = 0
    for grp_mask in (horiz, ~horiz):
        if grp_mask.sum() < min_band_points:
            continue
        gi = np.where(grp_mask)[0]
        gl = DBSCAN(eps=eps, min_samples=8).fit_predict(XY[gi])
        for lv in sorted(set(gl)):
            if lv == -1:
                continue
            labels[gi[gl == lv]] = next_lab
            next_lab += 1

    bands: List[Dict] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        sel = idx[labels == lab]
        if len(sel) < min_band_points:
            continue
        xy = np.column_stack([uu[sel], vv[sel]])
        c = xy.mean(0)
        # PCA of the cluster in (u,v)
        Q = xy - c
        ev, evec = np.linalg.eigh(Q.T @ Q)
        long_dir = evec[:, 1]
        length = float((Q @ long_dir).max() - (Q @ long_dir).min())
        width = float((Q @ evec[:, 0]).max() - (Q @ evec[:, 0]).min())
        # A real moulding/pilaster/frame is a long, thin, shallow ridge: long
        # (≥ min_band_length), narrow (≤ max_band_width — mouldings are cm-scale,
        # not 0.7 m slabs), with a strong aspect ratio (≥ 8:1) so a stray blob of
        # near-plane points can't masquerade as a profile element.
        if (length < min_band_length or width > max_band_width
                or length < 8.0 * max(width, 1e-3)):
            continue
        depth = float(np.abs(np.median((pts[sel] @ plane_normal + plane_d))))
        if depth < 1.5 * on_plane_tol:           # not a real protrusion, just noise
            continue
        align_up = abs(float(long_dir @ up_uv))
        if align_up < 0.5:                       # band runs ~horizontal in the wall
            mid_v = float(np.median(xy[:, 1]))
            if abs(mid_v - v_lo) <= abs(mid_v - v_hi):
                kind = "skirting"
            else:
                kind = "cornice"
        else:                                    # band runs ~vertical
            kind = "pilaster"
        # path = the long axis segment (2 endpoints in plane basis)
        t = Q @ long_dir
        p0 = c + t.min() * long_dir
        p1 = c + t.max() * long_dir
        path = np.array([p0, p1])
        # section: a rectangle [depth (out of plane) × width (along the band's short axis, in plane)]
        section = rectangle_outline(0.0, sign * max(depth, on_plane_tol),
                                    -width / 2, width / 2)
        mask = np.zeros(len(pts), dtype=bool)
        mask[sel] = True
        bands.append({"kind": kind, "path": path, "section": section,
                      "depth": depth * sign, "thickness": width, "point_mask": mask})
    return bands
