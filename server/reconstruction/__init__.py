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

# The package's exports are imported LAZILY (PEP 562, 2026-11-01): importing any submodule —
# `reconstruction.loops.config`, read by nearly every stage process — used to import classify →
# geometry.primitives → open3d first: 14 s of CPU per stage process on this pod (python -X importtime),
# ~15 stage processes per reconstruction with the card idle. `from reconstruction import X` and
# `reconstruction.X` resolve the same objects, on first use.
_EXPORTS = {
    "high_confidence_subcloud": ".subcloud", "load_ply_xyz_conf": ".subcloud",
    "classify_instance": ".classify", "Classification": ".classify",
    "reconstruct_element": ".reconstruct", "reconstruct_mesh_from_glb": ".reconstruct",
    "element_to_trimesh": ".tessellate", "write_element_glb": ".tessellate",
    "CameraView": ".visual_hull", "silhouette_iou": ".visual_hull",
    "carve_to_visual_hull": ".visual_hull", "project_points": ".visual_hull",
    "detect_openings": ".occlusion",
    "assemble_scene": ".assembly", "write_scene_json": ".assembly",
    "Scene": ".elements",
}


def __getattr__(name):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "high_confidence_subcloud", "load_ply_xyz_conf",
    "classify_instance", "Classification",
    "reconstruct_element", "reconstruct_mesh_from_glb",
    "element_to_trimesh", "write_element_glb",
    "CameraView", "silhouette_iou", "carve_to_visual_hull", "project_points",
    "detect_openings", "assemble_scene", "write_scene_json", "Scene",
]
