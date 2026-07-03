"""
surface_fit — measurement-backed smooth-surface reconstruction.
===============================================================

Replaces the generative ShapeR path for architectural classes with a strictly
measurement-driven pipeline: fit the lowest-DOF smooth model that explains the
segment (plane → quadric → swept profile → B-spline), keep the point→surface
residuals as a first-class deliverable (stats + heatmap + findings), and never
output geometry without point support.

Guiding rule (project charter): *the geometry is never smoothed; a smooth model
with controlled degrees of freedom is fitted, and the residuals are preserved
as measurement*. No learned priors, no hallucinated geometry.

Modules:
  models        — dataclasses (FittedSurface, ResidualReport, Finding, …)
  plane         — stage 2.1: RANSAC + total-LS plane with in-plane UV basis
  quadric       — stage 2.2: cylinder / sphere when plane residuals are structured
  profile_sweep — stage 2.3: extruded profile along a dominant axis (vaults)
  bspline       — stage 2.4: free-form B-spline low-pass (control-point spacing)
  escalate      — automatic model escalation driven by spatial residual tests
  spatial_test  — Moran's I on gridded residuals (white-noise acceptance test)
  residuals     — stage 4: stats vs the ORIGINAL cloud + localized findings
  heatmap       — stage 4: residual-colored PLY + PNG deviation map
  support       — no-extrapolation trimming: mesh only where points exist
  consolidate   — stage 1: WLOP (CGAL satellite) / robust-MLS layer collapse
  fine_register — stage 0: plane-constrained inter-chunk registration (post-BA)
  regularize    — stage 3: GlobFit-style plane snapping + clean edges
  scene         — stage 5: hybrid per-session orchestration (fitted vs TSDF)
  runner        — per-segment orchestration + config plumbing
"""

from .models import FittedSurface, ResidualReport, ResidualStats, Finding
from .runner import fit_segment, surface_fit_config_keys, build_surface_fit_kwargs

__all__ = [
    "FittedSurface", "ResidualReport", "ResidualStats", "Finding",
    "fit_segment", "surface_fit_config_keys", "build_surface_fit_kwargs",
]
