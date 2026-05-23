"""
Scene assembly — resolve adjacencies analytically.
==================================================

Takes the per-instance `ReconstructedElement`s for a session and produces a
coherent `Scene`:

  C1  structural shell  — rebuild every planar structural face (floor / ceiling /
                          planar wall) as the *data-supported region of the 2-D
                          arrangement* its plane forms with its neighbours'
                          planes (see `arrangement.py`): each face is cut at its
                          intersection lines with neighbours, and only the cells
                          the cloud / TSDF actually supports are kept — so a face
                          extends to meet a neighbour, is trimmed where it
                          overshoots one, and otherwise keeps the data's shape.
  C1b curved walls (a swept surface ∩ a plane is a conic, not a line, so the
                          arrangement leaves them alone): snap their end nodes
                          onto the planar walls they abut.
  C2  openings          — `occlusion.detect_openings` on every structural planar
                          surface (other elements' meshes as occluders).
  C3  structural clip   — clip `MeshElement`s against the floor / ceiling / wall
                          planes (objects can't poke through the shell).
  C4  object separation — clip interpenetrating object meshes off each other.
  C5  parametric snap   — swept-element ends within ε of a structural plane → snap
                          onto it; box / column bases near the floor → sit on it.
  + adjacency graph; emit `output/shape/scene.json`.

Curved-surface trimming against neighbours (plane∩cylinder = conic) and a true
min-penetration object↔object plane are later refinements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .elements import (ReconstructedElement, SurfaceElement, SweptElement,
                       BoxElement, MeshElement, Scene)
from .geometry.primitives import _unit
from .tessellate import element_to_trimesh
from . import arrangement as _arr

try:
    import trimesh
    _OK = True
except Exception:  # pragma: no cover
    trimesh = None
    _OK = False

_WALL_ROLES = {"wall", "retaining_wall", "parapet"}
_FLOORISH = {"floor", "slab", "deck", "platform_edge"}


def _is_planar_struct(el) -> bool:
    return isinstance(el, SurfaceElement) and el.is_structure and el.surface_type == "plane" \
        and el.outline is not None and len(el.outline) >= 3


def _plane_of(el: SurfaceElement):
    return _unit(el.plane_normal), float(el.plane_d)


# ── C1b: snap curved-wall corners onto the planar walls they abut ───

def _snap_wall_to_wall_corners(elements: List[ReconstructedElement], adjacency: List[Dict],
                                tol: float = 0.6):
    """For each pair of swept walls (`SweptElement`, all walls now — straight or
    curved), if their path endpoints come within ``tol`` of each other, snap both
    to their midpoint so the floor's keep_poly clipping closes the corner exactly.
    Idempotent: a 2nd call on already-snapped endpoints (distance ≈ 0) just snaps
    to the same midpoint (= unchanged); adjacency duplicates dedup'd by the caller."""
    walls = [e for e in elements if isinstance(e, SweptElement)
             and e.meta.get("subtype") == "curved_wall"
             and e.path is not None and len(np.asarray(e.path)) >= 2]
    if len(walls) < 2:
        return
    for i in range(len(walls)):
        a = walls[i]
        pa = np.asarray(a.path, dtype=np.float64).copy()
        for j in range(i + 1, len(walls)):
            b = walls[j]
            pb = np.asarray(b.path, dtype=np.float64).copy()
            best = None
            for ka in (0, -1):
                for kb in (0, -1):
                    d = float(np.linalg.norm(pa[ka] - pb[kb]))
                    if d < tol and (best is None or d < best[0]):
                        best = (d, ka, kb)
            if best is None:
                continue
            _, ka, kb = best
            mid = 0.5 * (pa[ka] + pb[kb])
            pa[ka] = mid; pb[kb] = mid
            a.path = pa; b.path = pb
            adjacency.append({"a": int(a.instance_id), "b": int(b.instance_id), "kind": "wall_meets_wall"})


def _subtract_curved_wall_end_faces(elements: List[ReconstructedElement]):
    """A curved wall's swept solid has, at each path endpoint, a 6-cm-thick "end face"
    quad that lies in the abutting planar wall's plane (after the snap). The two meshes
    then occupy the *same* 3-D region there → visible overlap / z-fighting at the corner
    ("ambas paredes sobresalen"). Subtract that quad from the planar wall's outline so
    the curved wall's end face fills the slot cleanly. If the slot reaches both v-extents
    of the wall (it usually does — the curved wall is full-height), the difference
    *splits* the wall into two pieces; we keep the largest in the original element and
    spawn a new ``SurfaceElement`` for any other piece > 5 % of the original area, so the
    wall on the OTHER side of the curve isn't lost."""
    try:
        from shapely.geometry import Polygon as _SP
    except Exception:  # pragma: no cover
        return
    sweeps = [e for e in elements if isinstance(e, SweptElement)
              and e.meta.get("subtype") == "curved_wall" and e.path is not None
              and getattr(e, "profile_polygon", None) is not None
              and getattr(e, "profile_frame", None) is not None]
    pwalls = [e for e in elements if _is_planar_struct(e) and e.role in _WALL_ROLES]
    if not sweeps or not pwalls:
        return
    next_id_base = 1 + max((int(e.instance_id) for e in elements), default=0)
    for sw in sweeps:
        path = np.asarray(sw.path, dtype=np.float64)
        prof = np.asarray(sw.profile_polygon, dtype=np.float64)
        frame = np.asarray(sw.profile_frame, dtype=np.float64)
        if len(prof) < 3 or frame.shape != (3, 3):
            continue
        for k in (0, -1):
            ep = path[k]
            for w in pwalls:
                n, d = _plane_of(w)
                if abs(float(n @ ep + d)) > 0.05:
                    continue                                           # endpoint not on this wall's plane
                quad3 = ep[None, :] + prof[:, [0]] * frame[0][None, :] + prof[:, [1]] * frame[1][None, :]
                rel = quad3 - np.asarray(w.basis_origin, dtype=np.float64)
                quad_uv = np.column_stack([rel @ np.asarray(w.basis_u, dtype=np.float64),
                                           rel @ np.asarray(w.basis_v, dtype=np.float64)])
                try:
                    qpoly = _SP(quad_uv)
                    if not qpoly.is_valid:
                        qpoly = qpoly.buffer(0)
                    wpoly = _SP(np.asarray(w.outline, dtype=np.float64)[:, :2])
                    if not wpoly.is_valid:
                        wpoly = wpoly.buffer(0)
                    diff = wpoly.difference(qpoly)
                    if diff.is_empty:
                        continue
                    pieces = list(diff.geoms) if getattr(diff, "geom_type", "") == "MultiPolygon" else [diff]
                    pieces = [p for p in pieces
                              if getattr(p, "geom_type", "") == "Polygon" and p.is_valid and not p.is_empty]
                    if not pieces:
                        continue
                    a_thresh = max(0.05 * wpoly.area, 0.05)            # ≥ 5 % of the wall, AND ≥ 5 cm²
                    pieces = sorted([p for p in pieces if p.area > a_thresh], key=lambda p: -p.area)
                    if not pieces:
                        continue
                    w.outline = np.asarray(pieces[0].exterior.coords, dtype=np.float64)[:-1, :2]
                    w.meta.setdefault("end_face_subtracted", []).append(int(sw.instance_id))
                    for extra in pieces[1:]:
                        new = SurfaceElement(
                            instance_id=next_id_base, label=f"{w.label}_split",
                            geometry_class=w.geometry_class, is_structure=True,
                            surface_type="plane", plane_normal=np.asarray(w.plane_normal, dtype=np.float64).copy(),
                            plane_d=float(w.plane_d), basis_origin=np.asarray(w.basis_origin, dtype=np.float64).copy(),
                            basis_u=np.asarray(w.basis_u, dtype=np.float64).copy(),
                            basis_v=np.asarray(w.basis_v, dtype=np.float64).copy(),
                            outline=np.asarray(extra.exterior.coords, dtype=np.float64)[:-1, :2],
                            role=w.role, meta={"split_from": int(w.instance_id)})
                        elements.append(new)
                        next_id_base += 1
                except Exception:
                    pass


def _snap_curved_wall_corners(elements: List[ReconstructedElement], adjacency: List[Dict]):
    """Make a curved wall (`wall_sweep`) actually meet the planar walls it abuts:
    project its end node onto a nearby planar wall's plane, and extend that wall's
    outline to reach the corner. Idempotent — the snap-onto-plane step is only run
    on a curved wall that hasn't already been snapped (`meta["corners_snapped"]`),
    so calling this twice (once before the structural arrangement, once after — the
    arrangement may re-derive the planar wall's outline from the cloud and lose the
    corner extension) is safe and only re-extends; no duplicate adjacency entries."""
    sweeps = [e for e in elements if isinstance(e, SweptElement)
              and e.meta.get("subtype") == "curved_wall" and e.path is not None and len(e.path) >= 2]
    pwalls = [e for e in elements if _is_planar_struct(e) and e.role in _WALL_ROLES]
    if not sweeps or not pwalls:
        return
    for sw in sweeps:
        already_snapped = bool(sw.meta.get("corners_snapped"))
        path = np.asarray(sw.path, dtype=np.float64).copy()
        changed = False
        for k in (0, -1):
            p = path[k]
            best = None
            for w in pwalls:
                n, d = _plane_of(w)
                sd = abs(float(n @ p + d))
                if sd < 0.6 and (best is None or sd < best[0]):
                    best = (sd, w, n, d)
            if best is None:
                continue
            _, w, n, d = best
            if not already_snapped:
                p_on = p - float(n @ p + d) * n           # project onto the planar wall's plane
                path[k] = p_on
                changed = True
                adjacency.append({"a": int(sw.instance_id), "b": int(w.instance_id), "kind": "wall_meets_wall"})
            else:
                p_on = p                                  # already on the plane
            # extend the planar wall's outline to reach the corner (add the corner
            # at the wall's full height and re-hull — never collapse the outline)
            rel = p_on - np.asarray(w.basis_origin)
            u_corner = float(rel @ np.asarray(w.basis_u))
            ol = np.asarray(w.outline, dtype=np.float64)[:, :2]
            umin, umax = float(ol[:, 0].min()), float(ol[:, 0].max())
            if not (umin - 1.6 <= u_corner <= umax + 1.6):
                continue
            if u_corner < umin - 0.02 or u_corner > umax + 0.02:
                from .geometry import concave_hull_2d, clean_polygon
                vmin, vmax = float(ol[:, 1].min()), float(ol[:, 1].max())
                try:
                    # ratio=1.0 ⇒ convex hull (robust, no spurious concavity that would
                    # give the planar-wall ↔ floor edge a weird shape near the corner)
                    hull = concave_hull_2d(np.vstack([ol, [[u_corner, vmin], [u_corner, vmax]]]), ratio=1.0)
                    if hull is not None and len(hull) >= 3:
                        w.outline = clean_polygon(hull, tol=0.02, smooth_iters=0)
                except Exception:
                    pass
        if changed:
            sw.path = path
            sw.meta["corners_snapped"] = True


# ── C3: structural clip of mesh elements (floors / ceilings / walls) ───────

def _slice_keep_above(mesh, plane_origin, plane_normal):
    """Keep the half of ``mesh`` where ``(v - origin)·normal >= 0``. Falls back to
    a crude face-drop (keep faces all of whose vertices are on the keep side) if
    trimesh's plane slice fails or wipes the mesh — a ShapeR mesh is often not
    watertight, and a face-drop still removes the intruding part."""
    n = np.asarray(plane_normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    org = np.asarray(plane_origin, float)
    try:
        m2 = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=n, plane_origin=org, cap=False)
        if m2 is not None and len(m2.vertices) >= 4 and len(m2.faces) >= 4:
            return m2
    except Exception:
        pass
    try:
        V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, np.int64)
        keep_v = ((V - org) @ n) >= -1e-4
        keep_f = keep_v[F].all(axis=1)
        if keep_f.any():
            return trimesh.Trimesh(vertices=V, faces=F[keep_f], process=False)
    except Exception:
        pass
    return mesh


def _clip_meshes_to_structure(elements: List[ReconstructedElement], adjacency: List[Dict]):
    surfs = [e for e in elements if _is_planar_struct(e)]
    if not surfs:
        return
    scene_c = np.mean([np.asarray(e.basis_origin) for e in surfs if e.basis_origin is not None], axis=0)
    # half-spaces to keep an object inside: floor (above), ceiling (below), and
    # every wall (room-interior side). The keep-normal points toward the scene
    # centroid. Floors/ceilings: clip anything poking through. Walls: only clip an
    # object that *straddles* the plane (poking through) — never slice an object
    # that's wholly on one side (that'd be a placement error, not an intersection).
    planes = []   # (origin, keep_normal, instance_id, is_wall)
    for s in surfs:
        n, _d = _plane_of(s)
        kn = n if float(n @ (scene_c - np.asarray(s.basis_origin))) >= 0 else -n
        planes.append((np.asarray(s.basis_origin, float), kn, int(s.instance_id),
                       s.role in _WALL_ROLES))
    for el in elements:
        if not isinstance(el, MeshElement) or el.vertices is None or el.faces is None:
            continue
        m = trimesh.Trimesh(vertices=np.asarray(el.vertices, float), faces=np.asarray(el.faces, np.int64), process=False)
        clipped_by = []
        for org, kn, sid, is_wall in planes:
            sd = (np.asarray(m.vertices, float) - org) @ kn
            penetrates = (sd.min() < -0.02) and ((not is_wall) or (sd.max() > 0.02))
            if penetrates:
                m2 = _slice_keep_above(m, org, kn)
                if len(m2.vertices) >= 4 and len(m2.faces) >= 4:
                    m = m2
                    clipped_by.append(sid)
        if clipped_by:
            # remap observed onto the new vertices (nearest old vertex)
            obs_new = None
            if el.observed is not None and len(el.observed) == len(el.vertices):
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(np.asarray(el.vertices, float))
                    _, idx = tree.query(np.asarray(m.vertices, float), k=1)
                    obs_new = np.asarray(el.observed, bool)[idx]
                except Exception:
                    obs_new = None
            el.vertices = np.asarray(m.vertices, float)
            el.faces = np.asarray(m.faces, np.int64)
            el.observed = obs_new
            el.meta["clipped_by_structure"] = clipped_by
            for sid in clipped_by:
                adjacency.append({"a": int(el.instance_id), "b": sid, "kind": "clipped_by"})


# ── C4: separate interpenetrating object meshes ────────────────────

def _separate_overlapping_objects(elements: List[ReconstructedElement], adjacency: List[Dict]):
    """If two free-form object meshes interpenetrate, clip each to its side of the
    plane through the midpoint of their centroids — so no object pokes into
    another. Only acts on *mutual* penetration (both reach past the midplane), so a
    pair that's merely close isn't sliced."""
    objs = [e for e in elements if isinstance(e, MeshElement)
            and e.vertices is not None and e.faces is not None and len(e.vertices) >= 4]
    if len(objs) < 2:
        return
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            ei, ej = objs[i], objs[j]
            vi = np.asarray(ei.vertices, float); vj = np.asarray(ej.vertices, float)
            if len(vi) < 4 or len(vj) < 4:
                continue
            mni, mxi = vi.min(0), vi.max(0)
            mnj, mxj = vj.min(0), vj.max(0)
            if not (np.all(mxi >= mnj + 0.01) and np.all(mxj >= mni + 0.01)):
                continue                                  # AABBs don't overlap
            ci, cj = vi.mean(0), vj.mean(0)
            d = cj - ci
            nd = float(np.linalg.norm(d))
            if nd < 1e-6:
                continue
            d = d / nd
            mid = 0.5 * (ci + cj)
            if not (float(((vi - mid) @ d).max()) > 0.02 and float(((vj - mid) @ d).min()) < -0.02):
                continue                                  # not mutually penetrating
            for el, kn in ((ei, -d), (ej, d)):
                vv = np.asarray(el.vertices, float)
                m2 = _slice_keep_above(trimesh.Trimesh(vertices=vv, faces=np.asarray(el.faces, np.int64),
                                                       process=False), mid, kn)
                if len(m2.vertices) < 4 or len(m2.faces) < 4:
                    continue
                obs_new = None
                if el.observed is not None and len(el.observed) == len(vv):
                    try:
                        from scipy.spatial import cKDTree
                        _, idx = cKDTree(vv).query(np.asarray(m2.vertices, float), k=1)
                        obs_new = np.asarray(el.observed, bool)[idx]
                    except Exception:
                        obs_new = None
                el.vertices = np.asarray(m2.vertices, float)
                el.faces = np.asarray(m2.faces, np.int64)
                el.observed = obs_new
                other_id = int(ej.instance_id if el is ei else ei.instance_id)
                el.meta.setdefault("separated_from", []).append(other_id)
            adjacency.append({"a": int(ei.instance_id), "b": int(ej.instance_id), "kind": "separated_from"})


# ── C5: parametric snapping ────────────────────────────────────────

def _snap_parametrics(elements: List[ReconstructedElement], adjacency: List[Dict], eps: float = 0.10):
    surfs = [e for e in elements if _is_planar_struct(e)]
    planes = [(_unit(s.plane_normal), float(s.plane_d), int(s.instance_id), s.role) for s in surfs]
    floors = [(n, d, sid) for (n, d, sid, role) in planes if role in _FLOORISH]
    for el in elements:
        if isinstance(el, SweptElement) and el.path is not None and len(el.path) >= 2:
            path = np.asarray(el.path, dtype=np.float64).copy()
            # A curved wall's path runs along its base, horizontally: if that base
            # is near a floor plane, project the WHOLE path onto it so the wall
            # sits flush (projecting only the two ends would tilt it).
            if el.meta.get("subtype") == "curved_wall" and floors:
                n, d, sid = min(floors, key=lambda fl: abs(float(np.median(path @ fl[0] + fl[1]))))
                off = path @ n + d
                if float(np.abs(np.median(off))) < 1.6:        # rest it on the nearest floor — no float
                    el.path = path - off[:, None] * n[None, :]
                    el.meta["snapped_base_to_floor"] = int(sid)
                    adjacency.append({"a": int(el.instance_id), "b": int(sid), "kind": "meets_floor"})
                continue
            changed = False
            for k in (0, -1):
                p = path[k]
                for n, d, sid, role in planes:
                    sd = float(n @ p + d)
                    if abs(sd) < eps:
                        path[k] = p - sd * n           # project onto the plane
                        adjacency.append({"a": int(el.instance_id), "b": sid, "kind": "snapped_to"})
                        changed = True
                        break
            if changed:
                el.path = path
                el.meta["snapped_ends"] = True
        elif isinstance(el, BoxElement) and el.center is not None and floors:
            # if the box's lowest corner along a floor normal is near the floor, sit it on the floor
            R = np.asarray(el.R, float)
            he = np.asarray(el.half_extents, float)
            c = np.asarray(el.center, float)
            for n, d, sid in floors:
                # box extent along n: sum |he_i * (R_i · n)|
                ext = float(np.sum(np.abs(he * (R.T @ n))))
                bottom = float(n @ c + d) - ext
                if abs(bottom) < eps:
                    el.center = c - bottom * n
                    adjacency.append({"a": int(el.instance_id), "b": sid, "kind": "sits_on"})
                    el.meta["snapped_base"] = True
                    break


# ── orchestrator ────────────────────────────────────────────────────

def assemble_scene(elements: List[ReconstructedElement], *, session_id: str,
                   views=None, hc_clouds: Optional[Dict[int, np.ndarray]] = None,
                   tsdf_meshes: Optional[Dict[int, Any]] = None) -> Scene:
    """Run C1/C1b/C2/C3/C4/C5 in order and return the assembled `Scene`. Mutates
    the elements in place. ``hc_clouds`` maps instance_id → its high-confidence
    sub-cloud (world coords) — the data the structural arrangement and the opening
    detector fit; ``tsdf_meshes`` (optional) maps instance_id → its TSDF surface
    mesh — extra support for the structural arrangement when available."""
    elements = list(elements)
    adjacency: List[Dict] = []
    if not _OK:
        return Scene(session_id=session_id, elements=elements, adjacency=adjacency)

    # C1a — wall corner snapping (BEFORE the structural arrangement so the floor's
    #       keep_poly clipping sees the snapped endpoints):
    #         (i)  swept-wall endpoint ↔ planar-wall plane (legacy, no-op once all
    #              walls are SweptElement post-unification);
    #         (ii) swept-wall endpoint ↔ swept-wall endpoint (the unified wall-to-
    #              wall corner — every wall is a swept solid, straight or curved).
    try:
        _snap_curved_wall_corners(elements, adjacency)
        _snap_wall_to_wall_corners(elements, adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] wall corners (pre-arrange): {e}")

    # C1b — structural shell: plane-arrangement reconstruction of every planar
    #       structural face (cut at neighbour-intersection lines / curved-wall paths).
    try:
        _arr.arrange_structural_faces(elements, hc_clouds=hc_clouds, tsdf_meshes=tsdf_meshes, adjacency=adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] structural arrangement: {e}")

    # C1c — re-extend planar walls' outlines to include the curved-wall corners
    #       (legacy: the arrangement above re-derives planar wall outlines from
    #       the cloud, which usually doesn't reach the corner; the snap is a no-op
    #       the 2nd time, the extend re-applies). Post-unification this is a no-op
    #       — there are no planar walls — but harmless.
    try:
        _snap_curved_wall_corners(elements, adjacency)
        _snap_wall_to_wall_corners(elements, adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] wall corners (post-arrange): {e}")


    # (C1d — end-face subtraction was tried and reverted: carving a 6 cm × ~2 m slot
    # out of the planar wall to make room for the curved wall's swept-solid end face
    # tended to remove visible chunks of the wall (a triangle below the slot when the
    # cut tilted, a missing piece on the other side when the slot split the wall).
    # The remaining z-fighting at the corner is a real but small visual artifact —
    # fixing it cleanly needs a mesh-level boolean, deferred.)

    # C2 — openings (use every *other* element's mesh as a potential occluder)
    try:
        from .occlusion import detect_openings
        meshes = {}
        for el in elements:
            try:
                m = element_to_trimesh(el)
                if m is not None and len(m.faces):
                    meshes[int(el.instance_id)] = m
            except Exception:
                pass
        for el in elements:
            # openings = doors / windows in walls (and the odd ceiling hatch). A
            # *floor* is solid — never carve it from sparse cloud coverage (Hernán:
            # "donde no hay info de paredes, completar para que no se vean huecos");
            # pit-mouth detection would be a separate explicit industrial-scan mode.
            if not _is_planar_struct(el) or el.role in _FLOORISH:
                continue
            occl = [m for iid, m in meshes.items() if iid != int(el.instance_id)]
            cloud = (hc_clouds or {}).get(int(el.instance_id))
            try:
                detect_openings(el, cloud, views=views, occluder_meshes=occl, add_to_surface=True)
            except Exception as e:
                print(f"[Assembly] openings for #{el.instance_id}: {e}")
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] opening detection: {e}")

    # C3 — clip object meshes against the structural shell (floor / ceiling / walls)
    try:
        _clip_meshes_to_structure(elements, adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] structural clip: {e}")

    # C4 — separate interpenetrating object meshes from each other
    try:
        _separate_overlapping_objects(elements, adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] object separation: {e}")

    # C5 — parametric snapping (curved-wall base flush to floor, box bases, ...)
    try:
        _snap_parametrics(elements, adjacency)
    except Exception as e:  # pragma: no cover
        print(f"[Assembly] snapping: {e}")

    # final adjacency dedup — multiple stages (arrangement, wall snaps, parametric
    # snaps, mesh clips) can emit the same undirected edge under the same `kind`.
    seen = set(); deduped = []
    for a in adjacency:
        try:
            key = (a.get("kind", ""), frozenset((int(a.get("a", -1)), int(a.get("b", -1)))))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    adjacency[:] = deduped

    return Scene(session_id=session_id, elements=elements, adjacency=adjacency)


def write_scene_json(scene: Scene, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene.to_dict(), indent=2))
    return out_path
