"""
reconstruction.reconstruct — turn a classified instance + its sub-clouds into a
typed `ReconstructedElement` (a `SurfaceElement`, `SweptElement`,
`LinearRepeatElement`, `BoxElement`, or — for genuinely free-form blobs — a
`MeshElement` produced by the ShapeR path).

`reconstruct_element(...)` is the single dispatch entry point.
"""

from .surface import reconstruct_surface
from .box import reconstruct_box
from .swept import reconstruct_swept
from .wall_sweep import reconstruct_wall_sweep
from .linear_repeat import reconstruct_linear_repeat
from .mesh import (reconstruct_mesh_from_glb, postprocess_shaper_mesh,
                   make_mesh_element)

__all__ = [
    "reconstruct_surface", "reconstruct_box", "reconstruct_swept",
    "reconstruct_wall_sweep", "reconstruct_linear_repeat", "reconstruct_element",
    "reconstruct_mesh_from_glb", "postprocess_shaper_mesh", "make_mesh_element",
]


def reconstruct_element(cls, *, instance_id, label, xyz, xyz_hc=None, world_up=None,
                        caption="", caption_fields=None, source_indices=None,
                        obb=None, dist_thresh=0.012):
    """Dispatch on ``cls.geometry_class``. Returns a `ReconstructedElement` or
    ``None`` (caller should then run the ShapeR mesh path for ``volumetric_mesh``).
    """
    gc = cls.geometry_class
    common = dict(instance_id=instance_id, label=label, xyz=xyz, xyz_hc=xyz_hc,
                  world_up=world_up, caption=caption, caption_fields=caption_fields,
                  source_indices=source_indices, dist_thresh=dist_thresh)
    if gc in ("surface_plane", "surface_cylinder", "surface_cone"):
        return reconstruct_surface(cls, **common)
    if gc == "wall_sweep":
        return reconstruct_wall_sweep(cls, **common)
    if gc == "swept":
        return reconstruct_swept(cls, **common)
    if gc == "linear_repeat":
        return reconstruct_linear_repeat(cls, **common)
    if gc == "box":
        return reconstruct_box(cls, obb=obb, **common)
    return None  # volumetric_mesh → ShapeR path, handled by the caller
