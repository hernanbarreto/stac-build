"""
Reconstruction v2 — semantic-geometric, neighbor-aware reconstruction.
=====================================================================

Downstream of point-cloud generation + SAM3 segmentation, this package
turns per-instance point sub-clouds into a coherent *scene* of typed
elements (parametric planes / cylinders / boxes where possible, ShapeR
meshes for free-form volumes), then resolves adjacencies analytically
(structural shell, occlusion reasoning, clipping).

See plan: reconstruction v2 — semantic-geometric, neighbor-aware.

Authors: Hernán Barreto — Ingerop IN3
"""

from .subcloud import high_confidence_subcloud, load_ply_xyz_conf
from .classify import classify_instance, Classification
from .reconstruct import reconstruct_element, reconstruct_mesh_from_glb
from .tessellate import element_to_trimesh, write_element_glb
from .visual_hull import CameraView, silhouette_iou, carve_to_visual_hull, project_points
from .occlusion import detect_openings
from .assembly import assemble_scene, write_scene_json
from .elements import Scene

__all__ = [
    "high_confidence_subcloud", "load_ply_xyz_conf",
    "classify_instance", "Classification",
    "reconstruct_element", "reconstruct_mesh_from_glb",
    "element_to_trimesh", "write_element_glb",
    "CameraView", "silhouette_iou", "carve_to_visual_hull", "project_points",
    "detect_openings", "assemble_scene", "write_scene_json", "Scene",
]
