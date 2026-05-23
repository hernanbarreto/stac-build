"""
ReconstructedElement contract — typed scene elements.
=====================================================

Reconstruction v2 represents a scene as a list of *typed* elements rather than
a bag of meshes, so the scene-assembly stage can reason about adjacency
analytically (a wall is a plane with a polygon and openings; a pipe is a
cylinder with an axis and extent; only genuinely free-form objects are raw
meshes).

This module is the data contract only — pure dataclasses + JSON
(de)serialisation, numpy-only. Tessellation to GLB lives in
``reconstruction/tessellate.py``; geometry fitting lives in
``reconstruction/reconstruct/*``; adjacency resolution in
``reconstruction/assembly.py``.

Coordinate conventions:
  - All 3-D quantities are in the floor-aligned *display* world frame (the one
    Potree shows; ``floor_transform.npz`` already applied), unless a field name
    says otherwise.
  - "plane basis": a 2-D coordinate system (``basis_origin``, ``basis_u``,
    ``basis_v``) lying in a ``PlanarElement``'s plane. ``world = origin + u*U + v*V``.

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ── helpers ─────────────────────────────────────────────────────────

def json_safe(obj):
    """Recursively replace non-finite floats (NaN / ±inf) with None and numpy
    scalars/arrays with plain python — so the result survives `json.dumps`
    (starlette serialises with allow_nan=False)."""
    import math
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _l(a: Optional[np.ndarray]) -> Optional[list]:
    """np.ndarray → nested python lists (JSON-safe), or None."""
    if a is None:
        return None
    return json_safe(np.asarray(a).tolist())


def _a(x: Any, dtype=np.float64) -> Optional[np.ndarray]:
    """list/None → np.ndarray (or None)."""
    if x is None:
        return None
    return np.asarray(x, dtype=dtype)


# ── attached sub-features (children of a plane) ─────────────────────

@dataclass
class OpeningElement:
    """A real hole in a surface (door / window / passage / pit mouth), in the
    parent surface's 2-D parameter space."""
    polygon: np.ndarray            # (N, 2) ordered contour in parent surface param space
    kind: str = "unknown"          # door | window | passage | pit_mouth | unknown
    confidence: float = 0.5        # how sure the occlusion test was it's a real hole

    def to_dict(self) -> Dict[str, Any]:
        return {"polygon": _l(self.polygon), "kind": self.kind,
                "confidence": float(self.confidence)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OpeningElement":
        return OpeningElement(polygon=_a(d["polygon"]),
                              kind=d.get("kind", "unknown"),
                              confidence=float(d.get("confidence", 0.5)))


@dataclass
class ProfileElement:
    """An extruded profile running along a line on a parent plane — skirting
    board, cornice, pilaster, door/window frame, low step. ``path`` and
    ``section`` are in the parent plane's 2-D basis (section: the cross-section
    swept along the path; its second axis is plane-normal depth)."""
    path: np.ndarray               # (N, 2) polyline in parent plane basis
    section: np.ndarray            # (M, 2) cross-section profile
    kind: str = "other"            # skirting | cornice | pilaster | frame | step | other

    def to_dict(self) -> Dict[str, Any]:
        return {"path": _l(self.path), "section": _l(self.section), "kind": self.kind}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProfileElement":
        return ProfileElement(path=_a(d["path"]), section=_a(d["section"]),
                              kind=d.get("kind", "other"))


# ── base element ────────────────────────────────────────────────────

@dataclass
class ReconstructedElement:
    """Common fields for every reconstructed scene element."""
    instance_id: int
    label: str
    geometry_class: str = "volumetric_object"   # see classify.py taxonomy
    is_structure: bool = False
    caption_fields: Dict[str, str] = field(default_factory=dict)  # {category,shape,material,detail}
    caption: str = ""                            # narrative caption (ShapeR conditioning)
    source_indices: Optional[np.ndarray] = None  # global point indices in cleaned_cloud.ply
    confidence_stats: Dict[str, float] = field(default_factory=dict)  # {observed_fraction, n_points, n_high_conf, ...}
    T_world: Optional[np.ndarray] = None         # (4,4) extra transform applied to local geom, if any
    glb_path: Optional[Path] = None              # tessellated mesh on disk
    meta: Dict[str, Any] = field(default_factory=dict)

    # type tag for (de)serialisation
    kind: str = "element"

    def base_meta(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "instance_id": int(self.instance_id),
            "label": self.label,
            "geometry_class": self.geometry_class,
            "is_structure": bool(self.is_structure),
            "caption_fields": dict(self.caption_fields),
            "caption": self.caption,
            "n_source_points": (int(len(self.source_indices))
                                if self.source_indices is not None else 0),
            "confidence": dict(self.confidence_stats),
            "T_world": _l(self.T_world),
            "glb": (str(self.glb_path) if self.glb_path else None),
            "meta": dict(self.meta),
        }

    def to_meta_dict(self) -> Dict[str, Any]:
        return json_safe(self.base_meta())


@dataclass
class SurfaceElement(ReconstructedElement):
    """An *open* surface — flat (wall/floor/ceiling/slab/deck/platform) OR
    curved (cylindrical: curved wall, tunnel lining, tank/silo shell; conical:
    tank roof, hopper). Carries a 2-D ``outline`` in the surface's natural
    parameter space, plus polygon-with-holes ``openings`` and protruding
    ``attached`` profiles.

    Parameterisation by ``surface_type``:
      - plane:    n·x + d = 0;  param = (u, v) metres along ``basis_u``/``basis_v``
                  from ``basis_origin``.  3-D: origin + u·U + v·V.
      - cylinder: axis through ``axis_point`` dir ``axis_dir``, radius ``radius``;
                  param = (θ rad about axis from ``theta_ref``, s metres along axis).
      - cone:     apex ``apex``, axis dir ``axis_dir``, ``half_angle`` rad;
                  param = (θ rad, s metres along axis from apex).
    For curved surfaces the outline is usually just a rectangle in (θ, s):
    [θ_min,θ_max] × [s_min,s_max] (a partial cylinder/cone) but a free outline is allowed.
    """
    surface_type: str = "plane"                  # plane | cylinder | cone
    # plane params
    plane_normal: Optional[np.ndarray] = None    # (3,) unit
    plane_d: float = 0.0
    basis_origin: Optional[np.ndarray] = None    # (3,) point on plane
    basis_u: Optional[np.ndarray] = None         # (3,) unit, in plane
    basis_v: Optional[np.ndarray] = None         # (3,) unit, in plane (n × u)
    # cylinder / cone params
    axis_point: Optional[np.ndarray] = None      # (3,) on the axis (cylinder); == apex for cone if apex None
    axis_dir: Optional[np.ndarray] = None        # (3,) unit
    radius: float = 0.0                          # cylinder radius
    apex: Optional[np.ndarray] = None            # (3,) cone apex
    half_angle: float = 0.0                      # cone half-angle (rad)
    theta_ref: Optional[np.ndarray] = None       # (3,) unit, in-plane reference for θ=0 (cyl/cone)
    # outline + extras (param space depends on surface_type)
    outline: Optional[np.ndarray] = None         # (N, 2) outer contour in param space
    thickness: float = 0.0                       # 0 = infinitely thin
    role: str = "other"                          # wall|curved_wall|retaining_wall|parapet|floor|slab|deck|ceiling|platform_edge|...
    openings: List[OpeningElement] = field(default_factory=list)
    attached: List[ProfileElement] = field(default_factory=list)
    kind: str = "surface"

    def to_meta_dict(self) -> Dict[str, Any]:
        d = self.base_meta()
        p: Dict[str, Any] = {"type": self.surface_type, "role": self.role,
                             "thickness": float(self.thickness),
                             "outline": _l(self.outline)}
        if self.surface_type == "plane":
            p.update(normal=_l(self.plane_normal), d=float(self.plane_d),
                     basis_origin=_l(self.basis_origin),
                     basis_u=_l(self.basis_u), basis_v=_l(self.basis_v))
        elif self.surface_type == "cylinder":
            p.update(axis_point=_l(self.axis_point), axis_dir=_l(self.axis_dir),
                     radius=float(self.radius), theta_ref=_l(self.theta_ref))
        elif self.surface_type == "cone":
            p.update(apex=_l(self.apex), axis_dir=_l(self.axis_dir),
                     half_angle=float(self.half_angle), theta_ref=_l(self.theta_ref))
        d["parametric"] = p
        d["openings"] = [o.to_dict() for o in self.openings]
        d["attached"] = [pr.to_dict() for pr in self.attached]
        return d


# A flat SurfaceElement is "planar". Kept as a thin constructor for callers that
# just want a plane (most common case).
def make_planar_surface(instance_id: int, label: str, *, plane_normal, plane_d,
                        basis_origin, basis_u, basis_v, outline=None,
                        role="other", **kw) -> "SurfaceElement":
    return SurfaceElement(instance_id=instance_id, label=label,
                          surface_type="plane", plane_normal=_a(plane_normal),
                          plane_d=float(plane_d), basis_origin=_a(basis_origin),
                          basis_u=_a(basis_u), basis_v=_a(basis_v),
                          outline=_a(outline) if outline is not None else None,
                          role=role, **kw)


@dataclass
class SweptElement(ReconstructedElement):
    """A swept solid — a 2-D cross-section swept along a 3-D path. Unifies pipes
    (circle), HVAC ducts (rect / rect_hollow), structural members (i/l/channel),
    rails (rail profile), kerbs, cable trays. ``path`` is a 3-D polyline; arc
    segments (elbows, curved alignments) are recorded in ``arc_segments`` as
    ``[(i, j, centre, axis), ...]`` meaning vertices i..j of ``path`` lie on a
    circular arc with that centre/axis (a dense polyline is also fine on its own).

    The cross-section may **vary** along the path: ``profile_polygons_per_node``
    and ``profile_params_per_node`` are length-N lists (one entry per path vertex)
    of the section at that vertex; the tessellator skins between consecutive
    sections. For uniform-section elements these lists hold N copies of the same
    polygon/params (cheap, lets every downstream consumer use one code path).
    ``profile_polygon`` / ``profile_params`` keep the representative section
    (median/mean across nodes) for callers that only want one number — IFC
    parametric export, summary UI, legacy readers."""
    profile_family: str = "free"                 # circle|rect|rect_hollow|i_section|l_section|channel|rail|free
    profile_params: Dict[str, float] = field(default_factory=dict)  # representative section params
    profile_polygon: Optional[np.ndarray] = None # (M, 2) representative cross-section
    profile_frame: Optional[np.ndarray] = None   # (3,3) [u, v, t] columns: profile-x, profile-y, initial path tangent
    path: Optional[np.ndarray] = None            # (N, 3) world polyline (slice centroids)
    arc_segments: List[List[Any]] = field(default_factory=list)
    wall_thickness: float = 0.0                  # for *_hollow families
    # Variable-section support (always populated by `reconstruct_swept`; lists
    # have ``len(path)`` entries). Set to None only when a caller constructs a
    # SweptElement by hand and doesn't care about variation; consumers should
    # then fall back to ``profile_polygon`` / ``profile_params``.
    profile_polygons_per_node: Optional[List[np.ndarray]] = None
    profile_params_per_node: Optional[List[Dict[str, float]]] = None
    kind: str = "swept"

    def is_section_variable(self) -> bool:
        """True iff the per-node profiles materially differ (any param > 5 %
        away from the representative). Tessellator / IFC use this to pick fast
        vs. skinned path."""
        if not self.profile_params_per_node or not self.profile_params:
            return False
        for params in self.profile_params_per_node:
            for k, v in params.items():
                ref = float(self.profile_params.get(k, v))
                if ref < 1e-9:
                    if abs(v - ref) > 1e-4:
                        return True
                elif abs(v - ref) / max(abs(ref), 1e-9) > 0.05:
                    return True
        return False

    def to_meta_dict(self) -> Dict[str, Any]:
        d = self.base_meta()
        d["parametric"] = {
            "type": "swept",
            "profile_family": self.profile_family,
            "profile_params": dict(self.profile_params),
            "profile_polygon": _l(self.profile_polygon),
            "profile_frame": _l(self.profile_frame),
            "path": _l(self.path),
            "arc_segments": [[int(i), int(j), _l(_a(c)), _l(_a(ax))]
                             for (i, j, c, ax) in self.arc_segments],
            "wall_thickness": float(self.wall_thickness),
            "section_variable": bool(self.is_section_variable()),
            "profile_polygons_per_node": (
                [_l(p) for p in self.profile_polygons_per_node]
                if self.profile_polygons_per_node is not None else None),
            "profile_params_per_node": (
                [dict(p) for p in self.profile_params_per_node]
                if self.profile_params_per_node is not None else None),
        }
        return d


@dataclass
class LinearRepeatElement(ReconstructedElement):
    """A linear repetitive run — railing, fence, railway track (rails + sleepers),
    pipe rack — as continuous ``rails`` (swept profiles) alongside a regularly
    spaced ``member`` repeated along ``path``."""
    path: Optional[np.ndarray] = None            # (N, 3) world polyline (run direction)
    element_spacing: float = 0.0                 # spacing of the repeated member along the path
    member: Optional["ReconstructedElement"] = None   # one instance of the repeated member (Box/Swept/...), at path[0]
    rails: List["SweptElement"] = field(default_factory=list)  # continuous longitudinal members
    n_members: int = 0
    kind: str = "linear_repeat"

    def to_meta_dict(self) -> Dict[str, Any]:
        d = self.base_meta()
        d["parametric"] = {
            "type": "linear_repeat",
            "path": _l(self.path),
            "element_spacing": float(self.element_spacing),
            "n_members": int(self.n_members),
            "member": (self.member.to_meta_dict() if self.member is not None else None),
            "rails": [r.to_meta_dict() for r in self.rails],
        }
        return d


@dataclass
class BoxElement(ReconstructedElement):
    """An oriented bounding box — boxy furniture, rectangular beam/column."""
    center: Optional[np.ndarray] = None          # (3,)
    R: Optional[np.ndarray] = None               # (3,3) columns = box axes
    half_extents: Optional[np.ndarray] = None    # (3,)
    kind: str = "box"

    def to_meta_dict(self) -> Dict[str, Any]:
        d = self.base_meta()
        d["parametric"] = {
            "type": "box",
            "center": _l(self.center),
            "R": _l(self.R),
            "half_extents": _l(self.half_extents),
        }
        return d


@dataclass
class MeshElement(ReconstructedElement):
    """A free-form mesh (ShapeR output, possibly shrink-wrapped + visual-hull
    carved). Carries a per-vertex ``observed`` mask (True = backed by real
    points/TSDF, False = ShapeR prediction) — the confidence map — plus
    silhouette-vs-mask IoU per camera and a coarse quality flag."""
    vertices: Optional[np.ndarray] = None        # (N, 3) world
    faces: Optional[np.ndarray] = None           # (M, 3) int
    observed: Optional[np.ndarray] = None        # (N,) bool
    iou_per_camera: Dict[str, float] = field(default_factory=dict)
    quality_flag: str = "ok"                     # ok | low_iou | tsdf_divergence | sparse
    kind: str = "mesh"

    def to_meta_dict(self) -> Dict[str, Any]:
        d = self.base_meta()
        obs = None if self.observed is None else np.asarray(self.observed)
        d["mesh"] = {
            "n_vertices": (int(len(self.vertices)) if self.vertices is not None else 0),
            "n_faces": (int(len(self.faces)) if self.faces is not None else 0),
            "observed_fraction": (float(obs.mean()) if obs is not None and obs.size else None),
            "observed_attr": "observed",       # name of the per-vertex GLB attribute
            "iou_per_camera": dict(self.iou_per_camera),
            "quality_flag": self.quality_flag,
        }
        return d


# ── scene container ────────────────────────────────────────────────

@dataclass
class Scene:
    """The assembled scene: all elements + an adjacency graph (edges between
    elements that share an interface — wall∩floor, object-clipped-by-wall, ...)."""
    session_id: str
    elements: List[ReconstructedElement] = field(default_factory=list)
    adjacency: List[Dict[str, Any]] = field(default_factory=list)  # [{a, b, kind}, ...] by instance_id

    def to_dict(self) -> Dict[str, Any]:
        return json_safe({
            "session_id": self.session_id,
            "elements": [e.to_meta_dict() for e in self.elements],
            "adjacency": list(self.adjacency),
        })


__all__ = [
    "OpeningElement", "ProfileElement", "ReconstructedElement",
    "SurfaceElement", "make_planar_surface", "SweptElement",
    "LinearRepeatElement", "BoxElement", "MeshElement", "Scene",
]
