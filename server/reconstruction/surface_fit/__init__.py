"""surface_fit — what the pipeline still uses of the former per-object surface fitting (USER 2026-10-17:
"el resto vuela todo"; the plane-segment / Manhattan / contour / escalation meshing is GONE, the scene mesh is
GenRecon — workers/genrecon_worker.py):
  consolidate.scene_consolidate   the cloud stage's consolidation (precision/corrected_cloud, correction/apply)
  plane.fit_plane / quadric.fit_cylinder   the fusion's primitives
  hole_audit._Evidence / _evidence  the per-keyframe z-buffer + mask votes (mask_filter, render_reproj, shape_proposer)
  support.support_grid              the shape proposer
  fit_torch                         the torch RANSAC the plane fit and the support grid run on
"""
from .models import FittedSurface, ResidualReport, ResidualStats, Finding  # noqa: F401
