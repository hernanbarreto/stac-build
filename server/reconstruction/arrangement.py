"""
Structural-shell arrangement — plane-arrangement surface reconstruction.
=======================================================================

Rebuilds each planar structural face of a scene (floor / ceiling / planar wall)
so it coincides with the planes of its neighbours where the data says they meet —
the hypothesis-and-selection idea behind PolyFit (Nan & Wonka, ICCV'17) and
Kinetic Shape Reconstruction (Bauchet & Lafarge, TOG'20), adapted to *open*
scans (no watertight constraint): the planar primitives are already detected and
semantically labelled, so each face just gets snapped to its plane-arrangement
neighbours and trimmed where it overshoots one.

For each planar structural face F_i (plane (n_i, d_i), 2-D basis (o,u,v)):

  1. footprint_i / pts_i — the points SAM3 assigned to F_i that actually lie on
     its plane (within `near_plane`; mislabelled bleed from a neighbour surface
     lies *off* the plane and is dropped here) + its TSDF-mesh vertices on the
     plane if available — projected into (u,v); `footprint_i` = a clean concave
     hull of them. (Falls back to F_i's current outline if the data is too sparse.)
  2. region := footprint_i buffered out by `bridge_dist` — the data + a small
     margin (so a concavity in the data wider than ~2·bridge_dist is preserved,
     narrower ones are scan gaps → filled).
  3. for every other planar structural face F_j whose plane isn't ~coplanar, with
     line L_ij = P_i ∩ P_j projected into (u,v), within `adj_dist` of footprint_i:
       • is L_ij an *edge line* of F_i? — i.e. does F_i's data lie essentially all
         on one side of it (a few points across is a thin loose-hull / mislabel
         bleed, which we trim; data spread substantially past it = an interior
         partition the floor/slab runs under, which we leave alone). If so:
       • EXTEND: if the data stops short of L_ij (gap > bridge_dist), grow `region`
         toward L_ij — but only in a band along L_ij of the data's extent there, so
         a free edge never grows more than `bridge_dist`.
       • TRIM: intersect `region` with the half-plane of L_ij on the data side, so
         F_i reaches exactly to L_ij and nothing overshoots it.
       • record an F_i–F_j adjacency.
  4. F_i.outline = clean_polygon( largest connected component of `region` ).

No δ-snap heuristics, no "extend until they touch": the plane arrangement defines
where faces *can* meet; the data decides which lines bound F_i (edge vs interior)
and how far it actually reaches. Works for any configuration — L-rooms, non-
perpendicular walls, sloped roofs, multi-floor, heavy infra; a face may come out
non-convex (a slab runs under partitions). Curved surfaces (plane ∩ cylinder is a
conic, not a line) are handled separately (assembly._snap_curved_wall_corners).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .elements import SurfaceElement, SweptElement
from .geometry.primitives import _unit
from .geometry import concave_hull_2d, clean_polygon

try:
    from shapely.geometry import Polygon, LineString, Point
    from shapely.ops import unary_union, split as _shp_split
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

_WALL_ROLES = {"wall", "retaining_wall", "parapet"}
_FLOORISH = {"floor", "slab", "deck", "platform_edge"}

# ── tunables ────────────────────────────────────────────────────────
_BRIDGE_DIST = 0.10        # m — how far a *free* edge (no neighbour) may grow past the data
_ADJ_DIST = 0.50           # m — max gap between the data and a neighbour's plane-intersection line
                           #     that we'll still bridge across (a wall–floor junction is often this wide)
_ADJ_DIST_MAX = 1.50       # m — for a curved-wall neighbour, the tongue's reach is adaptive (max distance
                           #     from any path point to the data), capped at this so a wildly far curve
                           #     doesn't pull the floor into the void
_PARALLEL_COS = 0.985      # |n_i·n_j| above this ⇒ ~coplanar ⇒ no shared line
_EDGE_TAIL_DIST = 0.20     # m — minority-side points all within this of a line ⇒ it's a thin bleed (trim)
_EDGE_TINY_ABS = 0.40      # m² — (no-points fallback) a line is an edge line if the minority hull area is below
_EDGE_TINY_REL = 0.25     #       this absolute value AND this fraction of the footprint area
_MIN_MINORITY_PTS = 6      # ≤ this many points on a side ⇒ that side is a bleed (trim it)
_HULL_RATIO = 0.30         # concave-hull tightness for the data footprint
_MIN_FOOTPRINT_PTS = 12    # fewer on-plane data points than this ⇒ fall back to the face's current outline
_NEAR_PLANE = 0.06         # m — a data point counts as "on" the face plane within this
_BIG = 1.0e3


# ── small helpers ───────────────────────────────────────────────────

def _is_planar_struct(el) -> bool:
    return isinstance(el, SurfaceElement) and el.is_structure and el.surface_type == "plane" \
        and el.outline is not None and len(el.outline) >= 3


def _uv(face: SurfaceElement, p3) -> np.ndarray:
    rel = np.atleast_2d(np.asarray(p3, dtype=np.float64)) - np.asarray(face.basis_origin, dtype=np.float64)
    return np.column_stack([rel @ np.asarray(face.basis_u, dtype=np.float64),
                            rel @ np.asarray(face.basis_v, dtype=np.float64)])


def _ring_to_poly(ring) -> Optional["Polygon"]:
    try:
        p = Polygon(np.asarray(ring, dtype=np.float64)[:, :2])
        if not p.is_valid:
            p = p.buffer(0)
        if getattr(p, "geom_type", "") == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        if getattr(p, "geom_type", "") != "Polygon" or p.is_empty or p.area <= 1e-7:
            return None
        return p
    except Exception:
        return None


def _largest_poly(g) -> Optional["Polygon"]:
    if g is None:
        return None
    if getattr(g, "geom_type", "") == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)
    if getattr(g, "geom_type", "") in ("GeometryCollection",):
        polys = [p for p in g.geoms if getattr(p, "geom_type", "") == "Polygon" and p.area > 1e-7]
        g = max(polys, key=lambda p: p.area) if polys else None
    if g is None or getattr(g, "geom_type", "") != "Polygon" or g.is_empty or g.area <= 1e-7:
        return None
    if not g.is_valid:
        g = _largest_poly(g.buffer(0))
    return g


def _adj_kind(ra: Optional[str], rb: Optional[str]) -> str:
    s = {ra or "", rb or ""}
    if "ceiling" in s and (s & _WALL_ROLES):
        return "meets_ceiling"
    if (s & _FLOORISH) and (s & _WALL_ROLES):
        return "meets_floor"
    if s <= _WALL_ROLES:
        return "wall_meets_wall"
    return "abuts"


def _half_plane(q0: np.ndarray, d: np.ndarray, n: np.ndarray) -> "Polygon":
    """The half-plane bounded by the line through ``q0`` along ``d``, on the ``n`` side."""
    p = Polygon([q0 - _BIG * d, q0 + _BIG * d, q0 + _BIG * d + _BIG * n, q0 - _BIG * d + _BIG * n])
    return p if p.is_valid else p.buffer(0)


def _band(q0: np.ndarray, d: np.ndarray, n: np.ndarray, half_w: float, t_lo: float, t_hi: float) -> "Polygon":
    """A rectangle straddling the line {q0 + t·d}: ±half_w along ``n``, t ∈ [t_lo, t_hi] along ``d``."""
    a = q0 + t_lo * d
    b = q0 + t_hi * d
    return Polygon([a - half_w * n, b - half_w * n, b + half_w * n, a + half_w * n])


def _data_footprint(face: SurfaceElement, hc_clouds, tsdf_meshes) -> Tuple[Optional["Polygon"], Optional[np.ndarray]]:
    """On-plane data of ``face``: (clean concave hull, the 2-D point set)."""
    n = _unit(np.asarray(face.plane_normal, dtype=np.float64)); d = float(face.plane_d)
    parts: List[np.ndarray] = []
    cl = (hc_clouds or {}).get(int(face.instance_id))
    if cl is not None:
        cl = np.asarray(cl, dtype=np.float64)
        if cl.ndim == 2 and cl.shape[1] >= 3 and len(cl) >= 3:
            cl = cl[:, :3]
            cl = cl[np.abs(cl @ n + d) < _NEAR_PLANE]      # drop bleed from neighbour surfaces (off this plane)
            if len(cl) >= 3:
                parts.append(_uv(face, cl))
    tm = (tsdf_meshes or {}).get(int(face.instance_id))
    if tm is not None:
        try:
            v = np.asarray(tm.vertices, dtype=np.float64)
            v = v[np.abs(v @ n + d) < max(_NEAR_PLANE, 0.10)]
            if len(v) >= 3:
                parts.append(_uv(face, v))
        except Exception:
            pass
    pts = np.vstack(parts) if parts else None
    if pts is not None and len(pts) >= _MIN_FOOTPRINT_PTS:
        try:
            h = concave_hull_2d(pts, ratio=_HULL_RATIO)
            if h is not None and len(h) >= 3:
                h = np.asarray(clean_polygon(h, tol=0.03, smooth_iters=0), dtype=np.float64)[:, :2]
                fp = _ring_to_poly(h)
                if fp is not None:
                    return fp, pts
        except Exception:
            pass
    return _ring_to_poly(np.asarray(face.outline, dtype=np.float64)[:, :2]), pts


def _edge_keep_normal(fp: "Polygon", pts: Optional[np.ndarray],
                      q0: np.ndarray, d: np.ndarray) -> Optional[np.ndarray]:
    """If the line {q0 + t·d} is an *edge line* of the surface — its data lies
    essentially all on one side (a few stray points across = a thin bleed) — return
    the unit normal (⊥ ``d``) pointing to that (kept) side. Otherwise (the surface
    runs substantially past the line, e.g. a slab under a partition) return None."""
    n = np.array([-d[1], d[0]], dtype=np.float64)
    if pts is not None and len(pts) >= 8:
        sd = (np.asarray(pts, dtype=np.float64) - q0) @ n
        n_pos = int(np.count_nonzero(sd > 0)); n_neg = len(sd) - n_pos
        if n_pos == 0 and n_neg == 0:
            return None
        if n_pos >= n_neg:
            keep_n, minority = n, sd[sd < 0]
        else:
            keep_n, minority = -n, sd[sd > 0]
        if len(minority) <= _MIN_MINORITY_PTS:
            return keep_n                                  # a handful of stray points across → trim them
        if float(np.percentile(np.abs(minority), 95)) < _EDGE_TAIL_DIST:
            return keep_n                                  # the minority is a thin tail hugging the line → trim
        return None                                        # real data spread past the line → interior line
    # no points: fall back to a hull-area split
    try:
        ap = float(fp.intersection(_half_plane(q0, d, n)).area)
        am = float(fp.intersection(_half_plane(q0, d, -n)).area)
    except Exception:
        return None
    tot = ap + am
    if tot < 1e-9:
        return None
    if min(ap, am) < min(_EDGE_TINY_ABS, _EDGE_TINY_REL * tot):
        return n if ap >= am else -n
    return None


# ── the arrangement ─────────────────────────────────────────────────

def arrange_structural_faces(elements, hc_clouds=None, tsdf_meshes=None, adjacency=None) -> None:
    """Snap every planar structural face onto its plane-arrangement neighbours and
    trim overshoots (see the module docstring). Mutates ``elements`` in place;
    appends adjacency dicts to ``adjacency``."""
    if not _OK or adjacency is None:
        return
    faces = [e for e in elements if _is_planar_struct(e)]
    if not faces:                       # was `< 2` — but with walls now SweptElements,
                                        # a scene may have just 1 planar face (the floor)
                                        # that still needs the curved-wall path clipping
        return
    planes = [(_unit(np.asarray(f.plane_normal, dtype=np.float64)), float(f.plane_d)) for f in faces]
    data = [_data_footprint(f, hc_clouds, tsdf_meshes) for f in faces]   # (footprint, pts_uv)

    edges: Dict[Tuple[str, frozenset], Dict] = {}
    log: List[str] = []
    for i, f in enumerate(faces):
        fp, pts = data[i]
        if fp is None:
            continue
        ni, di = planes[i]
        bo = np.asarray(f.basis_origin, dtype=np.float64)
        try:
            region = fp.buffer(_BRIDGE_DIST, join_style=1)
            region = _largest_poly(region)
            if region is None:
                continue
        except Exception:
            continue
        n_pts = 0 if pts is None else len(pts)
        n_edge = 0
        adj_js: List[int] = []
        edge_hps: List["Polygon"] = []      # half-planes to trim `region` with, applied all at the end
        for j, g in enumerate(faces):
            if j == i:
                continue
            nj, dj = planes[j]
            if abs(float(ni @ nj)) > _PARALLEL_COS:
                continue
            dl = np.cross(ni, nj)
            nl = float(np.linalg.norm(dl))
            if nl < 1e-6:
                continue
            dl = dl / nl
            A = np.array([ni, nj, dl])
            b = np.array([-di, -dj, float(dl @ (0.5 * (bo + np.asarray(g.basis_origin, dtype=np.float64))))])
            try:
                p0 = np.linalg.solve(A, b)
            except Exception:
                continue
            q0 = _uv(f, p0)[0]
            qd = _uv(f, p0 + dl)[0] - q0
            nq = float(np.linalg.norm(qd))
            if nq < 1e-9:
                continue
            qd = qd / nq
            # how far is this line from F_i's data?
            try:
                gap = float(fp.distance(_line_seg(q0, qd, fp)))   # 0 if the footprint crosses the line
            except Exception:
                gap = None
            if gap is None or gap > _ADJ_DIST:
                continue                                          # too far → not a neighbour here
            adj_js.append(j)
            keep_n = _edge_keep_normal(fp, pts, q0, qd)
            if keep_n is None:
                continue                                          # interior line (a partition the slab runs under) — leave it
            n_edge += 1
            nrm = np.array([-qd[1], qd[0]], dtype=np.float64)
            edge_hps.append(_half_plane(q0, qd, keep_n))
            # EXTEND toward the line if the data falls short — a tongue, bounded to a
            # band along the line of the current region's extent there (so a free edge
            # of the region never grows by more than `bridge_dist`). Overshoot past
            # this (or any) edge line is trimmed by the half-planes applied below.
            if gap > _BRIDGE_DIST:
                try:
                    rv = np.asarray(region.exterior.coords, dtype=np.float64)[:, :2]
                    t = (rv - q0) @ qd
                    tongue = _largest_poly(region.buffer(gap + 0.06, join_style=1).intersection(
                        _band(q0, qd, nrm, gap + 0.12, float(t.min()) - 0.10, float(t.max()) + 0.10)))
                    if tongue is not None:
                        region = _largest_poly(region.union(tongue)) or region
                except Exception:
                    pass
        # Also for a planar wall: each curved-wall whose snapped endpoint lies on this
        # wall's plane bounds it perpendicular to the curve at that endpoint — without
        # this, the wall extends past the corner along its u-axis (its convex hull
        # outline + the corner extension covers the corner but doesn't trim past it).
        if (f.role or "") in _WALL_ROLES:
            for sw in elements:
                if not (isinstance(sw, SweptElement) and sw.meta.get("subtype") == "curved_wall"
                        and sw.path is not None and len(np.asarray(sw.path)) >= 2):
                    continue
                p3 = np.asarray(sw.path, dtype=np.float64)
                for k in (0, -1):
                    ep = p3[k]
                    if abs(float(ni @ ep + di)) > 0.10:           # endpoint not on this wall's plane
                        continue
                    tan3 = (p3[1] - p3[0]) if k == 0 else (p3[-1] - p3[-2])
                    n2 = float(np.linalg.norm(tan3))
                    if n2 < 1e-9:
                        continue
                    tan3 = tan3 / n2
                    tan_ip = tan3 - float(tan3 @ ni) * ni          # project tangent onto the wall's plane
                    n3 = float(np.linalg.norm(tan_ip))
                    # skip nearly-perpendicular intersections (tangent ≈ wall normal): the
                    # in-plane projection is tiny + ill-conditioned and the slightly-tilted
                    # cut would slice a triangle out of the wall. End-face subtract handles
                    # those (it carves the curved wall's actual cross-section out of the wall).
                    if n3 < 0.30:
                        continue
                    tan_ip = tan_ip / n3
                    ep_uv = _uv(f, ep[None, :])[0]
                    tan_uv = _uv(f, (ep + tan_ip)[None, :])[0] - ep_uv
                    n4 = float(np.linalg.norm(tan_uv))
                    if n4 < 1e-9:
                        continue
                    tan_uv = tan_uv / n4
                    perp_uv = np.array([-tan_uv[1], tan_uv[0]], dtype=np.float64)  # the cut line direction
                    keep_n = _edge_keep_normal(fp, pts, ep_uv, perp_uv)
                    if keep_n is None:
                        continue
                    n_edge += 1
                    edge_hps.append(_half_plane(ep_uv, perp_uv, keep_n))

        # TRIM: keep only the data side of every edge line (done last, so any tongue
        # that overshot an edge line gets clipped back to it)
        for hp in edge_hps:
            try:
                clipped = _largest_poly(region.intersection(hp))
                if clipped is not None and clipped.area > 0.10 * region.area:
                    region = clipped
            except Exception:
                pass

        # wall neighbours: a wall (`SweptElement`, straight or curved) bounds the
        # floor along its BASE path and the ceiling along its TOP path (= base +
        # height·up). Extend `region` to the path polyline, then split by it and
        # keep the data side. Same code for floors and ceilings; only which path
        # to project differs.
        adj_curved: List[int] = []
        is_ceiling = (f.role or "") == "ceiling"
        if (f.role or "") in _FLOORISH or is_ceiling:
            for sw in elements:
                if not (isinstance(sw, SweptElement) and sw.meta.get("subtype") == "curved_wall"
                        and sw.path is not None and len(np.asarray(sw.path)) >= 2):
                    continue
                wall_path3 = np.asarray(sw.path, dtype=np.float64)
                if is_ceiling:                                  # use the wall's TOP path
                    H = float(sw.meta.get("wall_height_m", 0.0)
                              or (sw.profile_params or {}).get("h", 0.0) or 0.0)
                    up_axis = None
                    try:
                        pf = np.asarray(sw.profile_frame, dtype=np.float64)
                        if pf.shape == (3, 3):
                            up_axis = pf[:, 1]
                            nrm = float(np.linalg.norm(up_axis))
                            up_axis = up_axis / nrm if nrm > 1e-9 else None
                    except Exception:
                        up_axis = None
                    if H <= 0 or up_axis is None:
                        continue                                # can't lift the path → skip this wall for the ceiling
                    wall_path3 = wall_path3 + H * up_axis[None, :]
                puv = _uv(f, wall_path3)
                if len(puv) > 2:                               # de-dup ~coincident consecutive points (a noisy path)
                    puv = puv[np.r_[True, np.linalg.norm(np.diff(puv, axis=0), axis=1) > 1e-3]]
                if len(puv) < 2:
                    continue
                try:
                    pl = LineString(puv)
                    gap = float(fp.distance(pl))                # nearest part of the curve to the data
                except Exception:
                    continue
                if gap > _ADJ_DIST:
                    continue                                   # the whole curve is too far from this face's data
                adj_curved.append(int(sw.instance_id))
                # EXTEND `region` to follow the *whole* curve, including parts that poke
                # farther from the data than the nearest-distance gap (the gap is the
                # *minimum* over the curve; high-curvature spots can be much further).
                # Reach = max distance from any path point to the region, plus margin;
                # capped at `_ADJ_DIST_MAX` for safety. Only inside a band hugging the
                # curve, so a free edge of `region` never grows by more than `_BRIDGE_DIST`.
                try:
                    max_gap = float(max(region.distance(Point(float(p[0]), float(p[1]))) for p in puv))
                    reach = min(max(_ADJ_DIST, max_gap), _ADJ_DIST_MAX)
                    # cap_style=2 ⇒ flat strip ends, but the strip is still as wide as `reach`
                    # at each endpoint — and that perpendicular width can poke past the path's
                    # endpoint into the corner. Clip the tongue to the perpendicular slab at
                    # each endpoint (with a `_BRIDGE_DIST` margin past the endpoint, so legitimate
                    # bridge geometry between the data and the corner isn't accidentally cut).
                    tongue = region.buffer(reach + 0.06, join_style=1).intersection(
                        pl.buffer(reach + 0.12, cap_style=2))
                    t0 = puv[1] - puv[0]; t0 = t0 / (np.linalg.norm(t0) + 1e-12)
                    t1 = puv[-1] - puv[-2]; t1 = t1 / (np.linalg.norm(t1) + 1e-12)
                    for q0, into in ((puv[0] - _BRIDGE_DIST * t0, t0),
                                     (puv[-1] + _BRIDGE_DIST * t1, -t1)):
                        perp = np.array([-into[1], into[0]], dtype=np.float64)
                        try:
                            tongue = tongue.intersection(_half_plane(q0, perp, into))
                        except Exception:
                            pass
                    tongue = _largest_poly(tongue)
                    if tongue is not None:
                        region = _largest_poly(region.union(tongue)) or region
                except Exception:
                    pass
                # CLIP `region` to the side of the (extended) curve that holds the data: build a big
                # polygon = the extended path + a far arc on the data side, and intersect.
                try:
                    t0 = puv[1] - puv[0]; t0 = t0 / (np.linalg.norm(t0) + 1e-12)
                    t1 = puv[-1] - puv[-2]; t1 = t1 / (np.linalg.norm(t1) + 1e-12)
                    a_far = puv[0] - _BIG * t0
                    b_far = puv[-1] + _BIG * t1
                    mid = puv[len(puv) // 2]
                    tm = puv[min(len(puv) // 2 + 1, len(puv) - 1)] - puv[max(len(puv) // 2 - 1, 0)]
                    tm = tm / (np.linalg.norm(tm) + 1e-12)
                    nm = np.array([-tm[1], tm[0]], dtype=np.float64)
                    cdata = np.asarray(fp.representative_point().coords[0], dtype=np.float64)
                    keep_dir = nm if float((cdata - mid) @ nm) >= 0 else -nm
                    keep_poly = Polygon(np.vstack([[a_far], puv, [b_far],
                                                   [b_far + _BIG * keep_dir], [a_far + _BIG * keep_dir]]))
                    if not keep_poly.is_valid:
                        keep_poly = keep_poly.buffer(0)
                    clipped = _largest_poly(region.intersection(keep_poly))
                    if clipped is not None and clipped.area > 0.10 * region.area:
                        region = clipped
                except Exception:
                    pass

        try:
            ring = np.asarray(region.exterior.coords, dtype=np.float64)[:-1, :2]
            # Per face: a curved-wall neighbour ⇒ tol must be ≪ the path's vertex spacing
            # to preserve the arc's detail in the floor outline; otherwise (a planar wall
            # / ceiling with no curved neighbour) DP-simplify aggressively (~6 cm) to wipe
            # small bumps and "teeth" in the data hull (cloud noise traced by the concave
            # hull) — these show up as visible triangle artifacts at corners. 6 cm preserves
            # any real architectural feature larger than that.
            _tol = 0.001 if adj_curved else 0.06
            r2 = np.asarray(clean_polygon(ring, tol=_tol, smooth_iters=0), dtype=np.float64)[:, :2]
            if len(r2) >= 3 and _ring_to_poly(r2) is not None:
                ring = r2
        except Exception:
            ring = np.asarray(region.exterior.coords, dtype=np.float64)[:-1, :2]
        if len(ring) < 3 or _ring_to_poly(ring) is None:
            continue
        # If the face is a planar wall / ceiling without a curved neighbour and its
        # outline is *near-rectangular* (bbox area / outline area < 1.20 ⇒ no real
        # concavity), snap to the bounding rectangle in (u, v). Wipes both small
        # noise bumps in the data hull AND any "kink" left by a single bulge vertex
        # — the bbox follows the data envelope, so the snap reaches the OUTERMOST
        # data extent (closing gaps with neighbour walls / curved-walls without
        # leaving a triangular protrusion in the middle of the edge).
        if (f.role or "") in (_WALL_ROLES | {"ceiling"}) and not adj_curved:
            try:
                rp = _ring_to_poly(ring)
                if rp is not None:
                    umin, vmin, umax, vmax = rp.bounds
                    bbox_area = max(0.0, (umax - umin) * (vmax - vmin))
                    if bbox_area > 1e-6 and bbox_area / max(rp.area, 1e-6) < 1.20:
                        ring = np.array([[umin, vmin], [umax, vmin],
                                         [umax, vmax], [umin, vmax]], dtype=np.float64)
                        f.meta["snapped_to_bbox"] = True
            except Exception:
                pass
        old_poly = _ring_to_poly(np.asarray(f.outline, dtype=np.float64)[:, :2])
        a_old = float(old_poly.area) if old_poly is not None else 0.0
        a_new = float(_ring_to_poly(ring).area)
        n_v_old = len(np.asarray(f.outline))
        f.outline = ring
        f.meta["arranged"] = {"n_neighbours": len(adj_js), "n_edge_lines": n_edge, "n_curved": len(adj_curved),
                              "n_data_pts": n_pts, "bridge_dist": _BRIDGE_DIST, "adj_dist": _ADJ_DIST}
        log.append(f"#{int(f.instance_id)}{(f.role or '?')[:2]}:{n_v_old}->{len(ring)}v "
                   f"{a_old:.2f}->{a_new:.2f}m2[{n_edge}/{len(adj_js)}edge"
                   f"{('+' + str(len(adj_curved)) + 'curv') if adj_curved else ''} {n_pts}p]")
        for j in adj_js:
            kk = _adj_kind(f.role, faces[j].role)
            edges[(kk, frozenset((int(f.instance_id), int(faces[j].instance_id))))] = \
                {"a": int(f.instance_id), "b": int(faces[j].instance_id), "kind": kk}
        wall_kind = "meets_ceiling" if (f.role or "") == "ceiling" else "meets_floor"
        for cid in adj_curved:
            edges[(wall_kind, frozenset((int(f.instance_id), cid)))] = \
                {"a": int(f.instance_id), "b": cid, "kind": wall_kind}

    for e in edges.values():
        adjacency.append(e)
    try:
        print(f"[Assembly] arrange: {len(faces)} planar faces, {len(edges)} adjacencies "
              f"{sorted((sorted(tuple(k[1])), k[0]) for k in edges)}; " + (" ".join(log) if log else "no face changed"))
    except Exception:
        pass


def _line_seg(q0: np.ndarray, d: np.ndarray, near_poly: "Polygon"):
    """A long segment along the line {q0 + t·d}, sized to the polygon's reach
    (so ``Polygon.distance`` against it gives the perpendicular distance)."""
    from shapely.geometry import LineString
    b = near_poly.bounds
    r = float(np.hypot(b[2] - b[0], b[3] - b[1])) + 1.0 + float(np.linalg.norm(np.asarray(near_poly.centroid.coords[0]) - q0))
    return LineString([q0 - r * d, q0 + r * d])
