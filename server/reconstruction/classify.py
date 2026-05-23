"""
Geometry-first classification of a segmented instance.
======================================================

Decides which reconstruction primitive to use — `surface_plane`,
`surface_cylinder`, `surface_cone`, `swept`, `linear_repeat`, `box`,
`volumetric_mesh` — and a structural `role`. **Geometry is decisive**: PCA
shape + the flat-vs-curved discriminator + cross-section analysis + normal-vs-up
choose the family; the VLM `caption_fields` only supply hints (a `role`
candidate, an `is_structure` prior, a `profile_family` candidate) and break ties
among geometrically-compatible options.

The expensive surface/swept fits computed here are cached on the result so
`reconstruct.*` doesn't re-fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .geometry import (surface_kind, fit_swept, fit_cylinder_ransac, fit_plane_ransac,
                       fit_footprint_polyline)
from .geometry.primitives import _unit, SurfaceFit


# ── caption → hint mapping ──────────────────────────────────────────

# role keyword → (canonical role, is_structure, geometry hint, profile_family hint)
_CATEGORY_HINTS = [
    # walls / vertical structure
    (r"\b(retaining wall|muro de contenci[oó]n)\b", "retaining_wall", True, "surface", None),
    (r"\b(parapet|parapeto|antepecho)\b", "parapet", True, "surface", None),
    (r"\b(wall|muro|pared|tabique|partition)\b", "wall", True, "surface", None),
    # horizontal structure
    (r"\b(floor|slab on grade|piso|losa de piso|suelo)\b", "floor", True, "surface", None),
    (r"\b(slab|losa)\b", "slab", True, "surface", None),
    (r"\b(deck|bridge deck|tablero)\b", "deck", True, "surface", None),
    (r"\b(ceiling|soffit|techo|cielorraso|cielo raso)\b", "ceiling", True, "surface", None),
    (r"\b(platform|anden|and[eé]n|quay)\b", "platform_edge", True, "surface", None),
    (r"\b(ramp|rampa)\b", "ramp", True, "surface", None),
    (r"\b(stair|stairs|staircase|escalera|escalinata|step)\b", "stair", True, "box", None),
    # vertical members
    (r"\b(column|pillar|columna|pilar|post)\b", "column", True, None, None),
    (r"\b(pier|pile|pila|pilote|bent)\b", "pier", True, None, None),
    # horizontal members
    (r"\b(beam|girder|joist|viga|vigueta|truss chord)\b", "beam", True, "swept", None),
    (r"\b(lintel|dintel)\b", "lintel", True, "swept", None),
    # piping / ducting / trays
    (r"\b(hvac|duct|ducto|conducto de aire|air duct)\b", None, False, "swept", "rect"),
    (r"\b(pipe|pipeline|tube|tubo|tuber[ií]a|ca[nñ]o|conduit)\b", None, False, "swept", "circle"),
    (r"\b(cable tray|bandeja|charola)\b", None, False, "swept", "channel"),
    # railings / fences / tracks
    (r"\b(railing|handrail|guard ?rail|baranda|barandilla|pasamanos)\b", None, False, "linear_repeat", None),
    (r"\b(fence|fencing|cerco|cerca|reja|valla)\b", None, False, "linear_repeat", None),
    (r"\b(rail|track|railway track|v[ií]a (f[eé]rrea|de tren)|riel|durmiente|sleeper|tie)\b", "track", True, "linear_repeat", None),
    (r"\b(catenary|catenaria|overhead line)\b", None, False, "linear_repeat", None),
    # vessels / tanks → curved surface
    (r"\b(tank|silo|vessel|tanque|dep[oó]sito|reservoir|cyclone|hopper|tolva|stack|chimney|chimenea)\b", None, True, "surface_curved", None),
    # pits / trenches
    (r"\b(pit|trench|fosa|c[aá]rcamo|sump|inspection pit|foso)\b", "pit", True, None, None),
    # generic objects
    (r"\b(door|puerta)\b", "door_leaf", False, "surface", None),
    (r"\b(window|ventana)\b", "window_leaf", False, "surface", None),
    (r"\b(table|chair|cabinet|shelf|desk|furniture|mueble|silla|mesa)\b", None, False, None, None),
    (r"\b(equipment|machine|pump|motor|valve|fitting|fixture|equipo|m[aá]quina|bomba|v[aá]lvula)\b", None, False, None, None),
    (r"\b(plant|tree|vegetation|planta|[aá]rbol|foliage)\b", None, False, None, None),
    (r"\b(sign|signage|cartel|se[nñ]al)\b", None, False, None, None),
]

_STRUCTURAL_ROLES = {
    "wall", "curved_wall", "retaining_wall", "parapet", "floor", "slab", "deck",
    "ceiling", "platform_edge", "ramp", "stair", "column", "pier", "beam",
    "lintel", "track", "pit", "pit_wall", "pit_floor", "tank_shell", "tank_roof",
}
# Vertical walls — straight or curved — are represented uniformly as `wall_sweep`
# (a `SweptElement` with a path: 2 points = straight, N points = curved). A
# straight wall is just a curve of radius infinity — same data model, same code
# downstream, no special-case for "planar wall" in the arrangement / IFC export.
_WALL_ROLES_CL = {"wall", "retaining_wall", "parapet", "curved_wall"}


def _caption_hint(caption_fields: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """Parse role / structure / geometry / profile hints out of the caption."""
    txt = ""
    if caption_fields:
        txt = " ".join(str(caption_fields.get(k, "")) for k in
                       ("category", "shape", "material", "detail")).lower()
    # SAM3 labels are often "wall1" / "wall2" / "plant1" / "floor_5" — strip a
    # trailing instance index off each word so "wall1" still matches \bwall\b.
    txt = re.sub(r"([a-z]+)[ _]*\d+\b", r"\1", txt)
    role = None
    is_struct = None
    geom = None
    profile = None
    for pat, r, s, g, pf in _CATEGORY_HINTS:
        if re.search(pat, txt):
            if role is None:
                role, is_struct, geom = r, s, g
            if profile is None:
                profile = pf
            break
    # shape phrases
    sh = (caption_fields or {}).get("shape", "").lower()
    if geom is None:
        if re.search(r"\b(flat|planar)\b", sh):
            geom = "surface"
        elif re.search(r"\b(cylinder|cylindrical|round (column|tube|pipe))\b", sh):
            geom = "swept" if profile != "rect" else "swept"
        elif re.search(r"\b(curved|arc|domed?|barrel|vaulted)\b", sh):
            geom = "surface_curved"
        elif re.search(r"\b(box|rectangular (prism|box)|cuboid)\b", sh):
            geom = "box"
        elif re.search(r"\b(irregular|organic|free.?form|complex)\b", sh):
            geom = "mesh"
    return {"role": role, "is_structure": is_struct, "geometry": geom,
            "profile_family": profile}


# ── PCA shape descriptor ────────────────────────────────────────────

@dataclass
class _Shape:
    evals: np.ndarray          # ascending λ3 ≤ λ2 ≤ λ1 (of the centred cloud)
    evecs: np.ndarray          # columns matching evals
    centroid: np.ndarray
    extent: float              # bbox diagonal

    @property
    def principal(self) -> np.ndarray:    # long axis
        return _unit(self.evecs[:, 2])

    @property
    def normal_dir(self) -> np.ndarray:   # thin axis (≈ surface normal for a sheet)
        return _unit(self.evecs[:, 0])

    @property
    def is_sheet(self) -> bool:
        # thin in one direction, spread in the other two — tolerances loosened
        # for real (noisy) clouds: a scanned wall/floor has cm-scale thickness.
        l3, l2, l1 = self.evals
        return (l3 / max(l2, 1e-12) < 0.10) and (l2 / max(l1, 1e-12) > 0.025)

    @property
    def is_rod(self) -> bool:
        l3, l2, l1 = self.evals
        return (l1 / max(l2, 1e-12) > 8.0) and not self.is_sheet

    @property
    def aspect_long(self) -> float:
        l3, l2, l1 = self.evals
        return float(np.sqrt(l1 / max(l2, 1e-12)))


def _pca_shape(pts: np.ndarray) -> _Shape:
    c = pts.mean(0)
    Q = pts - c
    cov = (Q.T @ Q) / max(len(pts), 1)
    ev, evec = np.linalg.eigh(cov)
    ev = np.maximum(ev, 1e-12)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) or 1.0
    return _Shape(evals=ev, evecs=evec, centroid=c, extent=extent)


# ── result ──────────────────────────────────────────────────────────

@dataclass
class Classification:
    geometry_class: str                    # surface_plane | surface_cylinder | surface_cone | wall_sweep | swept | linear_repeat | box | volumetric_mesh
    role: str = "other"
    is_structure: bool = False
    profile_family: Optional[str] = None   # for swept
    surface_fit: Optional[object] = None   # geometry.SurfaceFit  (when a surface)
    swept_fit: Optional[object] = None     # geometry.SweptFit    (when swept)
    footprint: Optional[Dict[str, Any]] = None  # fit_footprint_polyline result (when wall_sweep)
    shape: Optional[_Shape] = None
    notes: Dict[str, Any] = field(default_factory=dict)


# ── role inference for planar surfaces ──────────────────────────────

def _planar_role(normal: np.ndarray, centroid: np.ndarray, world_up: np.ndarray,
                 scene_up_centroid: Optional[float], hint_role: Optional[str]) -> str:
    if hint_role in _STRUCTURAL_ROLES or hint_role in ("door_leaf", "window_leaf"):
        return hint_role
    up = _unit(world_up)
    vert = abs(float(normal @ up))
    if vert > 0.80:                       # ~horizontal surface
        if scene_up_centroid is not None:
            return "floor" if (centroid @ up) <= scene_up_centroid else "ceiling"
        return "floor"
    if vert < 0.30:                       # ~vertical surface
        return "wall"
    return "ramp"                          # sloped


# ── main ────────────────────────────────────────────────────────────

def classify_instance(xyz: np.ndarray, xyz_hc: Optional[np.ndarray],
                      caption_fields: Optional[Dict[str, str]] = None,
                      world_up: Optional[np.ndarray] = None,
                      obb: Optional[Dict[str, Any]] = None,
                      scene_up_centroid: Optional[float] = None,
                      dist_thresh: float = 0.012,
                      tsdf_mesh=None) -> Classification:
    """Classify one instance. ``xyz`` = the full sub-cloud; ``xyz_hc`` = the
    high-confidence subset (geometry fits run on this); ``world_up`` = unit
    up vector; ``obb`` = the OBB dict from segmentation_result.json (optional);
    ``scene_up_centroid`` = the scene centroid's height along ``world_up`` (used
    to call floor vs ceiling) — optional; ``tsdf_mesh`` = trimesh from the
    TSDF pipeline for this instance (optional) — when present, it's used as a
    second source for the wall footprint fit (fused with the cloud's geodesic
    polyline) for higher robustness on curved / cornered walls.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    pf_cloud = np.asarray(xyz_hc, dtype=np.float64) if (xyz_hc is not None and len(xyz_hc) >= 12) else xyz
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    world_up = _unit(world_up)
    hint = _caption_hint(caption_fields)
    shape = _pca_shape(pf_cloud)
    # real (LiDAR / SLAM-fused) clouds are noisier than the 12 mm default — scale
    # the plane/RANSAC tolerance with the cluster's size so a slightly-noisy wall
    # still reads as a plane (clamped to a sensible band).
    dist_thresh = float(np.clip(0.006 * shape.extent, max(dist_thresh, 0.012), 0.045))

    n = len(pf_cloud)
    if n < 12:
        return Classification("volumetric_mesh", role=hint["role"] or "other",
                              is_structure=bool(hint["is_structure"]), shape=shape,
                              notes={"reason": "too few points"})

    # ── A0. caption clearly says "linear repeat" (railing/fence/track/...) ──
    # take it even if the cross-section blob looks sheet-ish; reconstruct/
    # linear_repeat does the real work and falls back to swept/mesh if needed.
    if hint["geometry"] == "linear_repeat":
        return Classification("linear_repeat", role=hint["role"] or "other",
                              is_structure=bool(hint["is_structure"]) or (hint["role"] in _STRUCTURAL_ROLES),
                              profile_family=hint["profile_family"], shape=shape)

    # ── A1. caption says "swept" (pipe/duct/beam/...) → try a swept fit first ──
    if hint["geometry"] == "swept":
        swf = fit_swept(pf_cloud, dist_thresh=max(dist_thresh, 0.02), up_hint=world_up)
        if swf is not None:
            fam = swf.profile_family
            if fam == "free" and hint["profile_family"] in ("circle", "rect", "channel", "i_section", "l_section", "rail"):
                fam = hint["profile_family"]
            role = hint["role"] if hint["role"] in _STRUCTURAL_ROLES else "other"
            return Classification("swept", role=role,
                                  is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                                  profile_family=fam, swept_fit=swf, shape=shape)
        # fall through — not actually elongated; let the geometry decide

    # ── A. sheet-like → surface (plane / cylinder / cone) ──
    if shape.is_sheet or hint["geometry"] in ("surface", "surface_curved"):
        axis_hint = world_up if hint["geometry"] != "surface_curved" else None
        # for curved-surface hints, also try snapping the cylinder axis to up
        sf = surface_kind(pf_cloud, dist_thresh=dist_thresh,
                          axis_hint=world_up if shape.is_sheet else None)
        if sf.kind == "plane":
            role = _planar_role(sf.fit.normal, shape.centroid, world_up,
                                scene_up_centroid, hint["role"])
            # Wall-like planar surfaces → `wall_sweep` with a 2-point straight path
            # (radius-∞ "curved wall"). Same data model as a curved wall → single
            # downstream code path, clean swept-rectangle mesh, no noisy concave-hull
            # outline. Floors / ceilings / slabs / decks / ramps stay `surface_plane`
            # — they're horizontal surfaces; the structural arrangement cuts them
            # at every wall's path uniformly.
            if role in _WALL_ROLES_CL:
                fp = None
                try:
                    fp = fit_footprint_polyline(pf_cloud, world_up, node_spacing=0.20, tsdf_mesh=tsdf_mesh)
                except Exception:
                    fp = None
                if fp is not None:
                    return Classification("wall_sweep", role=role, is_structure=True,
                                          footprint=fp, shape=shape,
                                          notes={"from": "planar_wall_via_polyline",
                                                 "is_straight": bool(fp.get("is_straight"))})
            return Classification("surface_plane", role=role,
                                  is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                                  surface_fit=sf, shape=shape)
        if sf.kind == "cylinder":
            ax = sf.fit.axis_dir
            vertical_axis = abs(float(ax @ world_up)) > 0.7
            cat = (caption_fields or {}).get("category", "").lower()
            is_tank = ("tank" in cat or "silo" in cat or hint["role"] == "tank_shell")
            # A vertical curved wall that isn't a (by-construction circular) tank/
            # silo: don't force a circular arc — fit the *actual* curve from the
            # footprint and extrude it (`wall_sweep`). Tanks/silos and non-vertical
            # barrel vaults keep the cylinder.
            if vertical_axis and not is_tank:
                fp = None
                try:
                    fp = fit_footprint_polyline(pf_cloud, world_up, node_spacing=0.12, tsdf_mesh=tsdf_mesh)
                except Exception:
                    fp = None
                if fp is not None:
                    role = hint["role"] if hint["role"] in _STRUCTURAL_ROLES else "curved_wall"
                    return Classification("wall_sweep", role=role, is_structure=True,
                                          footprint=fp, shape=shape,
                                          notes={"from": "footprint_polyline",
                                                 "cyl_rms": float(getattr(sf.fit, "rms", 0.0))})
            role = hint["role"]
            if role not in _STRUCTURAL_ROLES:
                role = "tank_shell" if is_tank else ("curved_wall" if vertical_axis else "curved_surface")
            return Classification("surface_cylinder", role=role,
                                  is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                                  surface_fit=sf, shape=shape)
        if sf.kind == "cone":
            role = hint["role"]
            if role not in _STRUCTURAL_ROLES:
                role = "tank_roof" if "tank" in (caption_fields or {}).get("category", "").lower() else "cone_surface"
            return Classification("surface_cone", role=role,
                                  is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                                  surface_fit=sf, shape=shape)
        # `surface_kind` said free-form — but if the caption strongly says this is
        # a planar structural surface (wall/floor/ceiling/slab/...), the geometry
        # is just noisy: re-fit a plane with a generous tolerance and accept it if
        # most points still lie on it. (Semantic prior breaks a noise-driven tie.)
        if hint["geometry"] == "surface" and hint["is_structure"]:
            pf2 = fit_plane_ransac(pf_cloud, dist_thresh=min(2.0 * dist_thresh, 0.08),
                                   min_inlier_frac=0.45, measure_curvature=True)
            if pf2 is not None and pf2.inlier_frac >= 0.55:
                role = _planar_role(pf2.normal, shape.centroid, world_up,
                                    scene_up_centroid, hint["role"])
                # same wall-as-sweep redirect for the noise-fallback path
                if role in _WALL_ROLES_CL:
                    fp = None
                    try:
                        fp = fit_footprint_polyline(pf_cloud, world_up, node_spacing=0.20, tsdf_mesh=tsdf_mesh)
                    except Exception:
                        fp = None
                    if fp is not None:
                        return Classification("wall_sweep", role=role, is_structure=True,
                                              footprint=fp, shape=shape,
                                              notes={"from": "planar_wall_via_polyline_fallback",
                                                     "is_straight": bool(fp.get("is_straight"))})
                return Classification("surface_plane", role=role, is_structure=True,
                                      surface_fit=SurfaceFit("plane", pf2), shape=shape,
                                      notes={"plane_fallback": True, "inlier_frac": pf2.inlier_frac})
        # otherwise: it really isn't a primitive surface — fall through to mesh

    # ── B. rod-like geometry → swept (pipe / beam / column / kerb / ...) ──
    if shape.is_rod:
        swf = fit_swept(pf_cloud, dist_thresh=max(dist_thresh, 0.02), up_hint=world_up)
        if swf is not None:
            fam = swf.profile_family
            if fam == "free" and hint["profile_family"] in ("circle", "rect", "channel", "i_section", "l_section", "rail"):
                fam = hint["profile_family"]
            role = hint["role"] if hint["role"] in _STRUCTURAL_ROLES else "other"
            # a near-vertical rod the caption didn't name is most likely a column
            if role == "other" and hint["role"] is None and abs(float(swf.profile_frame[:, 2] @ world_up)) > 0.8:
                role = "column"
            return Classification("swept", role=role,
                                  is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                                  profile_family=fam, swept_fit=swf, shape=shape)
        # not actually elongated enough → treat as box/blob below

    # ── C. blob → box or volumetric_mesh ──
    # box test: a tight OBB fill ratio + roughly equal-ish extents (not a sheet)
    box_ok = False
    if obb is not None and "half_extents" in obb:
        he = np.asarray(obb["half_extents"], dtype=np.float64)
        vol_obb = 8.0 * float(np.prod(np.maximum(he, 1e-4)))
        try:
            from scipy.spatial import ConvexHull
            vol_hull = float(ConvexHull(pf_cloud).volume) if n >= 5 else 0.0
        except Exception:
            vol_hull = 0.0
        fill = vol_hull / max(vol_obb, 1e-9)
        box_ok = fill > 0.45
    else:
        # fall back: cloud's own OBB via PCA, fill estimate
        he = (pf_cloud @ shape.evecs).ptp(0) / 2.0
        vol_obb = 8.0 * float(np.prod(np.maximum(he, 1e-4)))
        try:
            from scipy.spatial import ConvexHull
            vol_hull = float(ConvexHull(pf_cloud).volume) if n >= 5 else 0.0
        except Exception:
            vol_hull = 0.0
        box_ok = (vol_hull / max(vol_obb, 1e-9)) > 0.50

    if hint["geometry"] == "box" or (box_ok and hint["geometry"] not in ("mesh",)):
        role = hint["role"] if hint["role"] in _STRUCTURAL_ROLES else "other"
        return Classification("box", role=role,
                              is_structure=bool(hint["is_structure"]) or role in _STRUCTURAL_ROLES,
                              shape=shape)

    return Classification("volumetric_mesh", role=hint["role"] or "other",
                          is_structure=bool(hint["is_structure"]) and (hint["role"] in _STRUCTURAL_ROLES),
                          shape=shape)
