"""
reconstruction.geometry — robust geometric fitters.

  primitives.py  plane / cylinder / cone RANSAC, the flat-vs-curved
                 discriminator (`surface_kind`), per-point normal estimation.
  swept.py       swept-solid fit: axis → moving-frame cross-sections →
                 section family classification → path fit (line / polyline-with-arcs).
  outline.py     concave hull (alpha-shape), polygon-with-holes utilities,
                 near-surface residual banding (skirting / cornice / pilaster).
"""

from .primitives import (
    PlaneFit, CylinderFit, ConeFit, SurfaceFit,
    fit_plane_ransac, fit_cylinder_ransac, fit_cone_ransac,
    estimate_normals, surface_kind,
)
from .swept import SweptFit, fit_swept
from .outline import (concave_hull_2d, rectangle_outline, residual_bands,
                      clip_polygon, clean_polygon, fit_footprint_polyline)
from .section import classify_section

__all__ = [
    "PlaneFit", "CylinderFit", "ConeFit", "SurfaceFit",
    "fit_plane_ransac", "fit_cylinder_ransac", "fit_cone_ransac",
    "estimate_normals", "surface_kind",
    "SweptFit", "fit_swept",
    "concave_hull_2d", "rectangle_outline", "residual_bands", "clip_polygon",
    "clean_polygon", "fit_footprint_polyline",
    "classify_section",
]
