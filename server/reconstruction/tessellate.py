"""
Tessellate a `ReconstructedElement` to a mesh / GLB.

Parametric elements (planes, curved surfaces, swept solids, boxes, linear
repeats) get clean triangulations here; `MeshElement`s pass their mesh through,
carrying the per-vertex ``observed`` attribute into the GLB so the viewer can
shade the confidence map. Polygon triangulation uses ``mapbox_earcut`` via
trimesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import trimesh
    from shapely.geometry import Polygon as _SPoly
except Exception:  # pragma: no cover
    trimesh = None
    _SPoly = None

from .elements import (ReconstructedElement, SurfaceElement, SweptElement,
                       LinearRepeatElement, BoxElement, MeshElement, ProfileElement)
from .geometry.primitives import _unit, _orthobasis


# ── small helpers ───────────────────────────────────────────────────

def _shapely_with_holes(outer: np.ndarray, holes: Optional[List[np.ndarray]] = None):
    holes = [np.asarray(h) for h in (holes or []) if h is not None and len(h) >= 3]
    poly = _SPoly(np.asarray(outer)[:, :2], holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _triangulate_2d(outer: np.ndarray, holes: Optional[List[np.ndarray]] = None):
    """(outer, holes) → (verts2d (N,2), faces (M,3))."""
    poly = _shapely_with_holes(outer, holes)
    if poly.is_empty or poly.area < 1e-9:
        return None, None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    v, f = trimesh.creation.triangulate_polygon(poly, engine="earcut")
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def _lift_plane(v2: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return origin[None, :] + v2[:, [0]] * u[None, :] + v2[:, [1]] * v[None, :]


def _double_sided(verts: np.ndarray, faces: np.ndarray):
    """A thin surface, rendered both sides (so it's never invisible from behind)."""
    f2 = np.vstack([faces, faces[:, ::-1] + len(verts)])
    return np.vstack([verts, verts]), f2


def _frames_along_path(path: np.ndarray, u0: np.ndarray, v0: np.ndarray):
    """Rotation-minimising frames (u_i, v_i, t_i) at each path vertex, seeded by
    (u0, v0) at path[0]. Returns (T (K,3), U (K,3), V (K,3))."""
    path = np.asarray(path, dtype=np.float64)
    K = len(path)
    tang = np.zeros_like(path)
    tang[:-1] = np.diff(path, axis=0)
    tang[-1] = tang[-2] if K > 1 else np.array([0, 0, 1.0])
    for i in range(K):
        n = np.linalg.norm(tang[i])
        tang[i] = tang[i] / n if n > 1e-9 else (tang[i - 1] if i else np.array([0, 0, 1.0]))
    U = np.zeros_like(path)
    V = np.zeros_like(path)
    # project (u0, v0) so they're ⊥ to t0
    t0 = tang[0]
    uu = _unit(u0 - (u0 @ t0) * t0)
    if np.linalg.norm(uu) < 1e-6:
        uu, _ = _orthobasis(t0)
    U[0] = uu
    V[0] = np.cross(t0, uu)
    for i in range(1, K):
        # double-reflection RMF
        v1 = path[i] - path[i - 1]
        c1 = v1 @ v1
        if c1 < 1e-18:
            U[i], V[i] = U[i - 1], V[i - 1]
            continue
        rL = U[i - 1] - (2.0 / c1) * (v1 @ U[i - 1]) * v1
        tL = tang[i - 1] - (2.0 / c1) * (v1 @ tang[i - 1]) * v1
        v2 = tang[i] - tL
        c2 = v2 @ v2
        if c2 < 1e-18:
            ui = rL
        else:
            ui = rL - (2.0 / c2) * (v2 @ rL) * v2
        ui = _unit(ui - (ui @ tang[i]) * tang[i])
        U[i] = ui
        V[i] = np.cross(tang[i], ui)
    return tang, U, V


def _resample_closed_polygon(poly: np.ndarray, M: int) -> np.ndarray:
    """Resample a closed polygon to ``M`` evenly arclength-spaced vertices. Used
    to make per-node sections of a variable-section sweep share a common M so
    the side mesh can be stitched without twist."""
    p = np.asarray(poly, dtype=np.float64)
    n = len(p)
    if n == M:
        return p
    if n < 2:
        return np.tile(p[0] if n else np.zeros(2), (M, 1))
    closed = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-9:
        return np.tile(p[0], (M, 1))
    su = np.linspace(0.0, total, M, endpoint=False)
    return np.column_stack([np.interp(su, s, closed[:, 0]),
                            np.interp(su, s, closed[:, 1])])


def _align_polygon_to(prof_a: np.ndarray, prof_b: np.ndarray) -> np.ndarray:
    """Cyclically rotate ``prof_b`` so vertex 0 best matches vertex 0 of
    ``prof_a`` — prevents the side mesh from twisting when consecutive slices
    have valid but rotated polygon orderings (relevant only for ``free``;
    parametric families build polygons in a canonical order so this is a no-op
    for them, just defensive)."""
    M = len(prof_a)
    if M != len(prof_b) or M < 3:
        return prof_b
    a0 = prof_a[0]
    diffs = np.linalg.norm(prof_b - a0[None, :], axis=1)
    k = int(np.argmin(diffs))
    if k == 0:
        return prof_b
    return np.concatenate([prof_b[k:], prof_b[:k]], axis=0)


def _sweep(profile2d: np.ndarray, path: np.ndarray, u0: np.ndarray, v0: np.ndarray,
           cap: bool = True):
    """Sweep a 2-D profile (M,2) along a 3-D path with RMFs. Returns (verts, faces)."""
    prof = np.asarray(profile2d, dtype=np.float64)
    M = len(prof)
    if M < 3 or len(path) < 2:
        return None, None
    _, U, V = _frames_along_path(path, u0, v0)
    K = len(path)
    rings = np.empty((K, M, 3))
    for i in range(K):
        rings[i] = path[i][None, :] + prof[:, [0]] * U[i][None, :] + prof[:, [1]] * V[i][None, :]
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(K - 1):
        a0 = i * M
        a1 = (i + 1) * M
        for j in range(M):
            j2 = (j + 1) % M
            faces.append([a0 + j, a0 + j2, a1 + j2])
            faces.append([a0 + j, a1 + j2, a1 + j])
    faces = np.asarray(faces, dtype=np.int64)
    if cap:
        # triangulate the profile and cap both ends
        v2, f2 = _triangulate_2d(prof)
        if v2 is not None:
            # map cap verts: they're a re-triangulation of the profile polygon;
            # caps as separate fans referencing new vertices
            base = len(verts)
            cap0 = path[0][None, :] + v2[:, [0]] * U[0][None, :] + v2[:, [1]] * V[0][None, :]
            cap1 = path[-1][None, :] + v2[:, [0]] * U[-1][None, :] + v2[:, [1]] * V[-1][None, :]
            verts = np.vstack([verts, cap0, cap1])
            f_cap0 = f2[:, ::-1] + base
            f_cap1 = f2 + base + len(v2)
            faces = np.vstack([faces, f_cap0, f_cap1])
    return verts, faces


def _sweep_variable(per_node_polys: List[np.ndarray], path: np.ndarray,
                    u0: np.ndarray, v0: np.ndarray, cap: bool = True):
    """Skinned sweep — one 2-D profile per path node (variable cross-section).

    All N profiles are resampled to the largest M among them so the side mesh
    quads are well-defined. Consecutive profiles are cyclically aligned so
    vertex 0 of each matches the previous one (avoids the side mesh twisting
    when arbitrary polygon orderings are passed in)."""
    polys = [np.asarray(p, dtype=np.float64) for p in per_node_polys]
    K = len(path)
    if K < 2 or len(polys) != K or any(len(p) < 3 for p in polys):
        return None, None
    M = max(len(p) for p in polys)
    rs = [_resample_closed_polygon(p, M) for p in polys]
    for i in range(1, K):
        rs[i] = _align_polygon_to(rs[i - 1], rs[i])
    _, U, V = _frames_along_path(path, u0, v0)
    rings = np.empty((K, M, 3))
    for i in range(K):
        rings[i] = (path[i][None, :]
                    + rs[i][:, [0]] * U[i][None, :]
                    + rs[i][:, [1]] * V[i][None, :])
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(K - 1):
        a0 = i * M
        a1 = (i + 1) * M
        for j in range(M):
            j2 = (j + 1) % M
            faces.append([a0 + j, a0 + j2, a1 + j2])
            faces.append([a0 + j, a1 + j2, a1 + j])
    faces = np.asarray(faces, dtype=np.int64)
    if cap:
        v2_0, f2_0 = _triangulate_2d(rs[0])
        v2_1, f2_1 = _triangulate_2d(rs[-1])
        if v2_0 is not None and v2_1 is not None:
            base = len(verts)
            cap0 = (path[0][None, :] + v2_0[:, [0]] * U[0][None, :]
                    + v2_0[:, [1]] * V[0][None, :])
            cap1 = (path[-1][None, :] + v2_1[:, [0]] * U[-1][None, :]
                    + v2_1[:, [1]] * V[-1][None, :])
            verts = np.vstack([verts, cap0, cap1])
            f_cap0 = f2_0[:, ::-1] + base
            f_cap1 = f2_1 + base + len(v2_0)
            faces = np.vstack([faces, f_cap0, f_cap1])
    return verts, faces


# ── per-type tessellation ───────────────────────────────────────────

def _tess_surface(el: SurfaceElement):
    if el.surface_type == "plane":
        u, v, origin, n = el.basis_u, el.basis_v, el.basis_origin, el.plane_normal
        holes = [o.polygon for o in el.openings]
        v2, f = _triangulate_2d(el.outline, holes)
        if v2 is None:
            return None
        v3 = _lift_plane(v2, origin, u, v)
        if el.thickness and el.thickness > 1e-4:
            back = v3 + n[None, :] * el.thickness
            # naive slab: front + back faces + a side skirt around the OUTER ring only
            verts = np.vstack([v3, back])
            f_back = f[:, ::-1] + len(v3)
            faces = np.vstack([f, f_back])
            ring = np.asarray(el.outline)[:, :2]
            # find ring vertex indices in v2 (they should be the first len(ring))
            # earcut keeps input vertices first; assume so
            R = len(ring)
            side = []
            for i in range(R):
                a, b = i, (i + 1) % R
                side.append([a, b, b + len(v3)])
                side.append([a, b + len(v3), a + len(v3)])
            faces = np.vstack([faces, np.asarray(side, dtype=np.int64)])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        else:
            verts, faces = _double_sided(v3, f)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # attached profiles → approximate boxes protruding from the plane
        extras = [_tess_profile(pr, origin, u, v, n) for pr in (el.attached or [])]
        extras = [m for m in extras if m is not None]
        if extras:
            mesh = trimesh.util.concatenate([mesh] + extras)
        return mesh

    if el.surface_type in ("cylinder", "cone"):
        return _tess_curved_surface(el)
    return None


def _tess_profile(pr: ProfileElement, origin, u, v, n):
    """Approximate an attached profile as an oriented box along its (2-D) path
    on the parent plane, protruding by the profile's depth."""
    p2 = np.asarray(pr.path, dtype=np.float64)
    sec = np.asarray(pr.section, dtype=np.float64)
    if len(p2) < 2 or len(sec) < 2:
        return None
    a = _lift_plane(p2[:1], origin, u, v)[0]
    b = _lift_plane(p2[-1:], origin, u, v)[0]
    length = float(np.linalg.norm(b - a))
    if length < 1e-3:
        return None
    depth = float(sec[:, 0].ptp()) or 0.02      # out-of-plane
    thick = float(sec[:, 1].ptp()) or 0.04      # across the band, in plane
    run = _unit(b - a)
    # box frame: x=run, y=in-plane ⊥ run, z=plane normal
    yax = _unit(np.cross(n, run))
    T = np.eye(4)
    T[:3, 0] = run
    T[:3, 1] = yax
    T[:3, 2] = n
    sgn = np.sign(float(np.median(np.asarray(pr.section)[:, 0]))) or 1.0
    T[:3, 3] = 0.5 * (a + b) + 0.5 * depth * sgn * n
    box = trimesh.creation.box(extents=[length, thick, depth], transform=T)
    return box


def _tess_curved_surface(el: SurfaceElement):
    """Triangulate a curved (cylinder / cone) surface patch.

    For an OPEN patch we triangulate the (θ, s) outline polygon directly — its
    boundary is densified to the surface's angular/axial pitch, interior Steiner
    points are scattered on a lattice ∩ polygon, a Delaunay triangulation is built
    and triangles whose centroid falls outside the polygon are dropped (so the
    real boundary is followed and openings become holes). This replaces the old
    grid-and-mask, which produced a staircased boundary and dropped-cell holes.
    A FULL wrap (tank shell) still uses a plain grid — there's no open boundary
    there to staircase.
    """
    if el.surface_type == "cylinder":
        radius = float(el.radius)
        ax = _unit(el.axis_dir)
        c0 = el.axis_point
        half_angle = 0.0
    else:  # cone
        radius = None
        ax = _unit(el.axis_dir)
        c0 = el.apex
        half_angle = float(el.half_angle)
    tref = _unit(el.theta_ref - (el.theta_ref @ ax) * ax)
    if np.linalg.norm(tref) < 1e-6:
        tref, _ = _orthobasis(ax)
    bnorm = np.cross(ax, tref)

    def _to3d(uv2: np.ndarray) -> np.ndarray:
        uv2 = np.atleast_2d(np.asarray(uv2, dtype=np.float64))
        th, s = uv2[:, 0], uv2[:, 1]
        rad = (np.full_like(s, radius) if radius is not None
               else np.maximum(1e-4, np.abs(s) * np.tan(half_angle)))
        return (c0[None, :] + s[:, None] * ax[None, :]
                + (rad * np.cos(th))[:, None] * tref[None, :]
                + (rad * np.sin(th))[:, None] * bnorm[None, :])

    # parameter-space outline (θ, s); fall back to the reported extents
    ol = (np.asarray(el.outline, dtype=np.float64)[:, :2]
          if (el.outline is not None and len(el.outline) >= 3) else None)
    if ol is None:
        th0, th1 = el.meta.get("theta_extent", [0.0, 2 * np.pi])
        s0, s1 = el.meta.get("s_extent", [0.0, 1.0])
        ol = np.array([[th0, s0], [th1, s0], [th1, s1], [th0, s1]], dtype=np.float64)
    th0, th1 = float(ol[:, 0].min()), float(ol[:, 0].max())
    s0, s1 = float(ol[:, 1].min()), float(ol[:, 1].max())
    arc = abs(th1 - th0)
    span_s = max(abs(s1 - s0), 1e-4)
    closed = arc >= 2 * np.pi - 0.08
    d_th = float(np.clip(np.pi / 30.0, 1e-3, max(arc / 2.0, 1e-3)))   # ≈6° in θ
    d_s = float(np.clip(0.12, 1e-3, span_s))                         # ≈12 cm in s

    if closed:
        n_th = int(np.clip(arc / d_th, 8, 360))
        n_s = int(np.clip(span_s / d_s, 2, 300))
        ths = np.linspace(th0, th1, n_th, endpoint=False)
        ss = np.linspace(s0, s1, n_s)
        TH, SS = np.meshgrid(ths, ss)
        verts = _to3d(np.column_stack([TH.ravel(), SS.ravel()]))
        faces = []
        for i in range(n_s - 1):
            for j in range(n_th):
                j2 = (j + 1) % n_th
                a = i * n_th + j; b = i * n_th + j2
                cc = (i + 1) * n_th + j; dd = (i + 1) * n_th + j2
                faces.append([a, b, dd]); faces.append([a, dd, cc])
        if not faces:
            return None
        verts, faces = _double_sided(verts, np.asarray(faces, dtype=np.int64))
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # open patch
    poly_uv = _shapely_with_holes(ol, [o.polygon for o in el.openings])
    if poly_uv.is_empty or poly_uv.area < 1e-9:
        return None
    if poly_uv.geom_type == "MultiPolygon":
        poly_uv = max(poly_uv.geoms, key=lambda g: g.area)
    from shapely.geometry import Point
    # densify all rings to the surface pitch (so the boundary doesn't facet)
    rings = [np.asarray(poly_uv.exterior.coords)[:-1]] + \
            [np.asarray(r.coords)[:-1] for r in poly_uv.interiors]
    bnd = []
    for ring in rings:
        K = len(ring)
        for k in range(K):
            p, q = ring[k], ring[(k + 1) % K]
            seg = q - p
            nstep = max(1, int(np.ceil(max(abs(seg[0]) / d_th, abs(seg[1]) / d_s))))
            for t in range(nstep):
                bnd.append(p + (t / nstep) * seg)
    bnd = np.asarray(bnd, dtype=np.float64) if bnd else ol.copy()
    # interior Steiner points: lattice ∩ polygon
    nth = int(np.clip(arc / d_th, 3, 360))
    ns = int(np.clip(span_s / d_s, 2, 300))
    TH, SS = np.meshgrid(np.linspace(th0, th1, nth), np.linspace(s0, s1, ns))
    lattice = np.column_stack([TH.ravel(), SS.ravel()])
    try:
        interior = lattice[[poly_uv.contains(Point(x, y)) for x, y in lattice]]
    except Exception:
        interior = np.zeros((0, 2))
    pts2 = np.vstack([bnd, interior]) if len(interior) else bnd
    pts2 = np.unique(np.round(pts2, 8), axis=0)
    if len(pts2) < 3:
        return None
    try:
        from scipy.spatial import Delaunay
        faces = Delaunay(pts2).simplices
        cent = pts2[faces].mean(axis=1)
        buf = poly_uv.buffer(1e-9)
        faces = faces[[buf.contains(Point(x, y)) for x, y in cent]]
    except Exception:
        return None
    if len(faces) == 0:
        return None
    verts, faces = _double_sided(_to3d(pts2), np.asarray(faces, dtype=np.int64))
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _tess_swept(el: SweptElement):
    # A curved wall renders as a thin double-sided ruled surface — same as a
    # planar wall renders as a zero-thickness sheet — not a chunky swept slab.
    if el.meta.get("subtype") == "curved_wall" and el.path is not None and len(el.path) >= 2:
        path = np.asarray(el.path, dtype=np.float64)
        H = float(el.profile_params.get("h") or el.meta.get("wall_height_m") or 2.5)
        frame = np.asarray(el.profile_frame, dtype=np.float64) if el.profile_frame is not None else np.eye(3)
        up = frame[:, 1]
        nrm = np.linalg.norm(up)
        up = up / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        K = len(path)
        verts = np.vstack([path, path + H * up[None, :]])
        faces = []
        for i in range(K - 1):
            faces.append([i, i + 1, i + 1 + K]); faces.append([i, i + 1 + K, i + K])
        if not faces:
            return None
        verts, faces = _double_sided(verts, np.asarray(faces, dtype=np.int64))
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    prof = np.asarray(el.profile_polygon, dtype=np.float64)
    if prof is None or len(prof) < 3:
        if el.profile_family == "circle" and el.profile_params.get("r"):
            r = float(el.profile_params["r"])
            th = np.linspace(0, 2 * np.pi, 33)[:-1]
            prof = np.column_stack([r * np.cos(th), r * np.sin(th)])
        else:
            return None
    frame = np.asarray(el.profile_frame, dtype=np.float64) if el.profile_frame is not None else np.eye(3)
    u0, v0 = frame[:, 0], frame[:, 1]
    path = np.asarray(el.path, dtype=np.float64)
    # Variable-section path? Use the skinned sweep when the per-node profiles
    # disagree with the representative beyond the tolerance. Falls back to the
    # uniform sweep otherwise (cheaper, identical output).
    per_node = el.profile_polygons_per_node
    if (per_node is not None and len(per_node) == len(path)
            and el.is_section_variable()):
        verts, faces = _sweep_variable(per_node, path, u0, v0, cap=True)
    else:
        verts, faces = _sweep(prof, path, u0, v0, cap=True)
    if verts is None:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _tess_box(el: BoxElement):
    T = np.eye(4)
    T[:3, :3] = np.asarray(el.R, dtype=np.float64)
    T[:3, 3] = np.asarray(el.center, dtype=np.float64)
    he = np.asarray(el.half_extents, dtype=np.float64)
    return trimesh.creation.box(extents=2.0 * np.maximum(he, 1e-4), transform=T)


def _tess_linear_repeat(el: LinearRepeatElement):
    parts = []
    path = np.asarray(el.path, dtype=np.float64)
    if el.member is not None and el.n_members >= 1 and len(path) >= 2:
        run = _unit(path[-1] - path[0])
        base = element_to_trimesh(el.member)
        if base is not None:
            for i in range(int(el.n_members)):
                m = base.copy()
                m.apply_translation(run * (i * el.element_spacing))
                parts.append(m)
    for rail in el.rails:
        rm = element_to_trimesh(rail)
        if rm is not None:
            parts.append(rm)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _tess_mesh(el: MeshElement):
    if el.vertices is None or el.faces is None or len(el.faces) == 0:
        return None
    m = trimesh.Trimesh(vertices=np.asarray(el.vertices, dtype=np.float64),
                        faces=np.asarray(el.faces, dtype=np.int64), process=False)
    if el.observed is not None and len(el.observed) == len(m.vertices):
        m.vertex_attributes["observed"] = np.asarray(el.observed, dtype=np.float32)
    return m


# ── public ──────────────────────────────────────────────────────────

def element_to_trimesh(el: ReconstructedElement):
    """Tessellate an element. Returns a ``trimesh.Trimesh`` (with
    ``vertex_attributes`` set where relevant) or ``None``."""
    if trimesh is None:  # pragma: no cover
        return None
    try:
        if isinstance(el, SurfaceElement):
            m = _tess_surface(el)
        elif isinstance(el, SweptElement):
            m = _tess_swept(el)
        elif isinstance(el, LinearRepeatElement):
            m = _tess_linear_repeat(el)
        elif isinstance(el, BoxElement):
            m = _tess_box(el)
        elif isinstance(el, MeshElement):
            m = _tess_mesh(el)
        else:
            m = None
    except Exception as e:  # pragma: no cover - tessellation is best-effort
        print(f"[Tessellate] {type(el).__name__} #{getattr(el, 'instance_id', '?')}: {e}")
        m = None
    if m is not None and (len(m.vertices) == 0 or len(m.faces) == 0):
        return None
    if m is not None:
        m.metadata["instance_id"] = int(getattr(el, "instance_id", -1))
        m.metadata["label"] = getattr(el, "label", "")
    return m


def write_element_glb(el: ReconstructedElement, out_path: Path,
                      cloud_xyz: Optional[np.ndarray] = None,
                      cloud_rgb: Optional[np.ndarray] = None,
                      seg_color: Optional[str] = None) -> Optional[Path]:
    """Write the element as a GLB with PBR material + IFC metadata + (optional)
    cloud-derived color/texture.

    - ``seg_color`` (hex "#RRGGBB"): segmentation color used as the PBR base
      colour fallback when no cloud colours are available.
    - ``cloud_xyz, cloud_rgb`` (whole cloud): if provided AND the element has
      ``source_indices``, the assigned points' colours are baked onto the mesh —
      a per-vertex colour for `MeshElement`/`BoxElement`/`SweptElement`, or a
      2-D texture map projected on (u,v) for a planar `SurfaceElement` (a wall /
      floor / ceiling). Either way the GLB ends up actually painted.
    - PBR ``metallicFactor`` / ``roughnessFactor`` / ``alphaMode`` come from a
      heuristic parse of the InternVL caption (concrete, wood, metal, glass, …).
    - The mesh's metadata gets an IFC mapping (``ifc_class``,
      ``predefined_type``, ``material_name``) + the element's caption / role /
      confidence / instance_id, so a downstream IFC export can pick it up.
    """
    m = element_to_trimesh(el)
    if m is None:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _apply_visual_and_metadata(el, m, cloud_xyz, cloud_rgb, seg_color)
    except Exception as e:                             # pragma: no cover
        print(f"[Tessellate] visual/metadata #{getattr(el, 'instance_id', '?')}: {e}")
    m.export(str(out_path))
    el.glb_path = out_path
    return out_path


# ── PBR material + cloud-painted colour/texture + IFC metadata ─────

_FLOORISH_ROLES = {"floor", "slab", "deck", "platform_edge"}
_WALL_ROLES_LOC = {"wall", "retaining_wall", "parapet"}

# (keywords found in the caption, IFC material name, metallic, roughness, alpha)
_MATERIAL_PATTERNS = [
    (("concrete", "cement"),                        "Concrete",   0.0, 0.85, 1.0),
    (("brick",),                                    "Brick",      0.0, 0.90, 1.0),
    (("drywall", "gypsum", "plaster"),              "Drywall",    0.0, 0.95, 1.0),
    (("wood", "trunk", "timber", "plywood", "oak"), "Wood",       0.0, 0.65, 1.0),
    (("metal", "steel", "aluminum", "iron",
      "copper", "brass"),                           "Steel",      1.0, 0.30, 1.0),
    (("glass",),                                    "Glass",      0.0, 0.10, 0.30),
    (("tile", "ceramic", "porcelain"),              "Tile",       0.0, 0.40, 1.0),
    (("fabric", "textile", "carpet", "cloth"),      "Fabric",     0.0, 0.95, 1.0),
    (("plastic", "polymer", "pvc"),                 "Plastic",    0.0, 0.55, 1.0),
    (("stone", "marble", "granite"),                "Stone",      0.0, 0.80, 1.0),
    (("plant", "vegetation", "leaf", "leaves",
      "tree"),                                      "Vegetation", 0.0, 0.95, 1.0),
]


def _material_from_caption(caption: str):
    cap = (caption or "").lower()
    for keys, name, met, rough, alpha in _MATERIAL_PATTERNS:
        if any(k in cap for k in keys):
            return {"name": name, "metallic": met, "roughness": rough, "alpha": alpha}
    return {"name": "Generic", "metallic": 0.0, "roughness": 0.70, "alpha": 1.0}


def _ifc_class_from_element(el):
    role = (getattr(el, "role", None) or "").lower()
    gc = (el.geometry_class or "").lower()
    sub = ((el.meta.get("subtype") if hasattr(el, "meta") else "") or "").lower()
    if role in _FLOORISH_ROLES:
        return ("IfcSlab", "FLOOR")
    if role == "ceiling":
        return ("IfcSlab", "ROOF")
    if role in _WALL_ROLES_LOC or sub == "curved_wall":
        return ("IfcWall", "POLYGONAL" if sub == "curved_wall" else "STANDARD")
    if "pipe" in gc or "duct" in gc:
        return ("IfcPipeFitting", "USERDEFINED")
    if gc == "swept":
        return ("IfcBuildingElementProxy", "USERDEFINED")
    if gc == "box":
        return ("IfcBuildingElementProxy", "USERDEFINED")
    if gc in ("volumetric_mesh", "mesh"):
        return ("IfcFurnishingElement", "USERDEFINED")
    return ("IfcBuildingElementProxy", "USERDEFINED")


def _hex_to_rgba(hex_str: Optional[str], alpha: float = 1.0):
    if not hex_str:
        return [0.7, 0.7, 0.7, alpha]
    s = hex_str.lstrip("#")
    try:
        if len(s) == 6:
            return [int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0, alpha]
        if len(s) == 8:
            return [int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
                    int(s[4:6], 16) / 255.0, int(s[6:8], 16) / 255.0]
    except Exception:
        pass
    return [0.7, 0.7, 0.7, alpha]


def _vertex_colors_from_cloud(verts3d, cloud_xyz, cloud_rgb, indices):
    """For each vertex find the nearest assigned cloud point and copy its colour."""
    if cloud_xyz is None or cloud_rgb is None or indices is None or len(indices) == 0 or len(verts3d) == 0:
        return None
    pts = cloud_xyz[indices]
    cls = cloud_rgb[indices]
    if len(pts) < 1:
        return None
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(pts).query(verts3d, k=1, workers=-1)
        col = cls[idx]
    except Exception:
        return None
    rgba = np.zeros((len(verts3d), 4), dtype=np.uint8)
    rgba[:, :3] = (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def _bake_planar_texture(el, cloud_xyz, cloud_rgb, indices, px_per_m=300):
    """Project assigned points to (u,v), bin into a grid, average → RGB image +
    UV bounds. Empty cells filled from nearest populated cell."""
    try:
        from PIL import Image
        from scipy.spatial import cKDTree
    except Exception:
        return None, None
    if cloud_xyz is None or cloud_rgb is None or indices is None or len(indices) == 0:
        return None, None
    if not isinstance(el, SurfaceElement) or el.surface_type != "plane":
        return None, None
    pts = cloud_xyz[indices]; cls = cloud_rgb[indices]
    o = np.asarray(el.basis_origin, dtype=np.float64)
    bu = np.asarray(el.basis_u, dtype=np.float64)
    bv = np.asarray(el.basis_v, dtype=np.float64)
    rel = pts - o
    u = rel @ bu; v = rel @ bv
    ol = np.asarray(el.outline, dtype=np.float64)[:, :2]
    u_min = float(min(u.min(), ol[:, 0].min()))
    u_max = float(max(u.max(), ol[:, 0].max()))
    v_min = float(min(v.min(), ol[:, 1].min()))
    v_max = float(max(v.max(), ol[:, 1].max()))
    if u_max - u_min < 0.05 or v_max - v_min < 0.05:
        return None, None
    W = max(8, min(2048, int(np.ceil((u_max - u_min) * px_per_m))))
    H = max(8, min(2048, int(np.ceil((v_max - v_min) * px_per_m))))
    iu = np.clip(((u - u_min) / (u_max - u_min) * (W - 1)).astype(int), 0, W - 1)
    iv = np.clip(((v - v_min) / (v_max - v_min) * (H - 1)).astype(int), 0, H - 1)
    sum_rgb = np.zeros((H, W, 3), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.int32)
    np.add.at(sum_rgb, (iv, iu), cls)
    np.add.at(cnt, (iv, iu), 1)
    avg = np.zeros((H, W, 3), dtype=np.uint8)
    valid = cnt > 0
    if valid.any():
        avg[valid] = (sum_rgb[valid] / cnt[valid, None] * 255.0).clip(0, 255).astype(np.uint8)
    if not valid.all() and valid.any():
        ys, xs = np.where(valid)
        ey, ex = np.where(~valid)
        if len(ey) > 0:
            try:
                _, idx = cKDTree(np.column_stack([ys, xs])).query(np.column_stack([ey, ex]), k=1, workers=-1)
                avg[ey, ex] = avg[ys[idx], xs[idx]]
            except Exception:
                pass
    img = Image.fromarray(np.flipud(avg), mode="RGB")    # GLTF UV (0,0) is top-left of image; flipped here
    return img, (u_min, v_min, u_max, v_max)


def _apply_visual_and_metadata(el, m, cloud_xyz, cloud_rgb, seg_color):
    """Build PBR material (cloud-painted if possible, else seg_color), set UVs /
    vertex colours / texture as needed, and stuff IFC + caption metadata onto
    the mesh (ends up under GLB ``extras`` via trimesh's exporter)."""
    caption = (getattr(el, "caption", "") or "")
    mat_info = _material_from_caption(caption)
    ifc_cls, ifc_pre = _ifc_class_from_element(el)
    base_rgba = _hex_to_rgba(seg_color, alpha=mat_info["alpha"])

    pbr_kwargs = {
        "name": mat_info["name"],
        "baseColorFactor": base_rgba,
        "metallicFactor": float(mat_info["metallic"]),
        "roughnessFactor": float(mat_info["roughness"]),
    }
    if mat_info["alpha"] < 1.0:
        pbr_kwargs["alphaMode"] = "BLEND"

    indices = (el.source_indices.astype(np.int64) if getattr(el, "source_indices", None) is not None else None)
    verts3d = np.asarray(m.vertices, dtype=np.float64)

    visual_set = False
    # Planar surface → bake a (u,v) texture from the assigned cloud points.
    if (isinstance(el, SurfaceElement) and el.surface_type == "plane"
            and cloud_xyz is not None and cloud_rgb is not None and indices is not None and len(indices) > 50):
        try:
            img, bounds = _bake_planar_texture(el, cloud_xyz, cloud_rgb, indices)
            if img is not None and bounds is not None:
                u_min, v_min, u_max, v_max = bounds
                bu = np.asarray(el.basis_u, dtype=np.float64)
                bv = np.asarray(el.basis_v, dtype=np.float64)
                o = np.asarray(el.basis_origin, dtype=np.float64)
                rel = verts3d - o
                vu = rel @ bu; vv = rel @ bv
                uv = np.column_stack([(vu - u_min) / max(u_max - u_min, 1e-9),
                                      1.0 - (vv - v_min) / max(v_max - v_min, 1e-9)])
                pbr_kwargs["baseColorTexture"] = img
                mat = trimesh.visual.material.PBRMaterial(**pbr_kwargs)
                m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
                visual_set = True
        except Exception as e:                          # pragma: no cover
            print(f"[Tessellate] planar texture #{el.instance_id}: {e}")

    # Mesh / box / swept → per-vertex colours sampled from the cloud.
    if not visual_set and cloud_xyz is not None and cloud_rgb is not None and indices is not None and len(indices) > 0:
        try:
            rgba = _vertex_colors_from_cloud(verts3d, cloud_xyz, cloud_rgb, indices)
            if rgba is not None:
                m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
                visual_set = True
        except Exception as e:                          # pragma: no cover
            print(f"[Tessellate] vertex colors #{el.instance_id}: {e}")

    # Fallback: solid PBR with the segmentation colour.
    if not visual_set:
        try:
            mat = trimesh.visual.material.PBRMaterial(**pbr_kwargs)
            m.visual = trimesh.visual.TextureVisuals(material=mat)
        except Exception:
            pass

    # ── IFC + scene metadata (ends up under GLB extras via trimesh) ──
    extras = {
        "instance_id": int(getattr(el, "instance_id", -1)),
        "label": getattr(el, "label", ""),
        "geometry_class": getattr(el, "geometry_class", ""),
        "role": getattr(el, "role", None),
        "is_structure": bool(getattr(el, "is_structure", False)),
        "caption": caption,
        "caption_fields": dict(getattr(el, "caption_fields", {}) or {}),
        "ifc": {
            "class": ifc_cls, "predefined_type": ifc_pre,
            "material_name": mat_info["name"], "global_id": _ifc_guid(int(getattr(el, "instance_id", 0))),
        },
        "pbr": {
            "metallic": float(mat_info["metallic"]),
            "roughness": float(mat_info["roughness"]),
            "alpha": float(mat_info["alpha"]),
            "base_color_hex": (seg_color or ""),
        },
        "confidence_stats": dict(getattr(el, "confidence_stats", {}) or {}),
        "n_data_pts": int(len(indices)) if indices is not None else 0,
        "meta": {k: v for k, v in (getattr(el, "meta", {}) or {}).items()
                 if isinstance(v, (int, float, str, bool, list, dict)) and k not in ("color",)},
    }
    m.metadata.update({"instance_id": extras["instance_id"], "label": extras["label"], "extras": extras})


def _ifc_guid(seed_int: int) -> str:
    """Stable IfcGuid (22-char compressed UUID) derived from instance_id, so reruns
    of the same scene produce the same GUIDs (idempotent IFC import)."""
    import uuid, base64
    u = uuid.UUID(int=(seed_int & ((1 << 128) - 1)) | (1 << 64))   # bit 64 set so it's never all-zero
    raw = u.bytes
    b = base64.b64encode(raw, altchars=b"_$").decode("ascii").rstrip("=")
    return b[:22]
