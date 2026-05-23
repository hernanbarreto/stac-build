"""Reconstruct a `SweptElement` (pipe / HVAC duct / beam / rail / kerb / tray)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..elements import SweptElement
from ..geometry import fit_swept, classify_section
from ..geometry.primitives import _unit


def reconstruct_swept(cls, *, instance_id: int, label: str, xyz: np.ndarray,
                      xyz_hc: Optional[np.ndarray] = None,
                      world_up: Optional[np.ndarray] = None, caption: str = "",
                      caption_fields: Optional[Dict[str, str]] = None,
                      source_indices=None, dist_thresh: float = 0.012):
    swf = cls.swept_fit
    if swf is None:
        pts = np.asarray(xyz_hc if (xyz_hc is not None and len(xyz_hc) >= 12) else xyz, dtype=np.float64)
        swf = fit_swept(pts, dist_thresh=max(dist_thresh, 0.02),
                        up_hint=world_up if world_up is not None else None)
    if swf is None:
        return None  # not elongated enough → caller falls to ShapeR mesh

    # classify the cross-section properly (circle/rect/i_section/l_section/channel/rail);
    # the swept fit's own family ('circle'/'rect'/'free') is just a coarse first pass.
    poly = np.asarray(swf.profile_polygon, dtype=np.float64) if swf.profile_polygon is not None else None
    if poly is not None and len(poly) >= 3:
        fam, params = classify_section(poly, caption_hint=cls.profile_family)
    else:
        fam, params = (swf.profile_family or "free"), dict(swf.profile_params)
    # the swept fit's own circle/rect params are usually cleaner for those two
    if fam == swf.profile_family and swf.profile_params:
        params = dict(swf.profile_params)

    # Per-node sections: the swept fit produces one polygon + params per path
    # node. They're already aligned with `swf.path`, so they go through as-is.
    # We only re-derive when `classify_section` upgraded the family (e.g.
    # circle → i_section): in that case the per-node parametric polygons no
    # longer fit the upgraded family, so we drop the variable section and let
    # the tessellator use the single representative polygon. This keeps the
    # data model consistent without inventing fake per-node params.
    polys_per_node = getattr(swf, "polygons_per_node", None) or None
    params_per_node = getattr(swf, "params_per_node", None) or None
    if polys_per_node is not None and fam != (swf.profile_family or "free"):
        polys_per_node = None
        params_per_node = None
    if polys_per_node is not None:
        polys_per_node = [np.asarray(p, dtype=np.float64) for p in polys_per_node]
        params_per_node = [dict(p) for p in params_per_node]

    el = SweptElement(instance_id=instance_id, label=label, geometry_class="swept",
                      is_structure=cls.is_structure, caption=caption,
                      caption_fields=dict(caption_fields or {}),
                      source_indices=source_indices,
                      profile_family=fam, profile_params=dict(params),
                      profile_polygon=np.asarray(swf.profile_polygon, dtype=np.float64),
                      profile_frame=np.asarray(swf.profile_frame, dtype=np.float64),
                      path=np.asarray(swf.path, dtype=np.float64),
                      arc_segments=list(swf.arc_segments),
                      wall_thickness=float(swf.wall_thickness),
                      profile_polygons_per_node=polys_per_node,
                      profile_params_per_node=params_per_node)
    el.meta["role"] = cls.role
    el.meta["elongation"] = float(swf.elongation)
    el.confidence_stats = {
        "n_points": int(len(xyz)),
        "n_high_conf": int(len(xyz_hc)) if xyz_hc is not None else int(len(xyz)),
        "observed_fraction": (float(np.mean(swf.inliers)) if getattr(swf, "inliers", None) is not None else None),
        "fit_rms": float(swf.rms),
        "path_length_m": float(swf.path_length),
        "is_straight": bool(swf.is_straight),
    }
    return el
