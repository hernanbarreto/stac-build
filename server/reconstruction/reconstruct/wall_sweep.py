"""Reconstruct any wall — straight OR curved — as a `SweptElement`: a thin, tall
rectangle swept along the wall's fitted 2-D footprint polyline. A straight wall
is just a 2-point path (radius-∞), a curved wall is N points; same code path,
same data model, no special cases downstream. Unlike `surface_cylinder` which
forces a circular arc and visibly deforms walls whose curvature isn't a clean
arc, the polyline represents the *actual* shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..elements import SweptElement
from ..geometry.primitives import _unit


def reconstruct_wall_sweep(cls, *, instance_id: int, label: str, xyz: np.ndarray,
                           xyz_hc: Optional[np.ndarray] = None,
                           world_up: Optional[np.ndarray] = None, caption: str = "",
                           caption_fields: Optional[Dict[str, str]] = None,
                           source_indices=None, dist_thresh: float = 0.012):
    fp = getattr(cls, "footprint", None)
    if not fp or fp.get("path3d") is None:
        return None
    path = np.asarray(fp["path3d"], dtype=np.float64)
    if len(path) < 2:
        return None
    up = _unit(np.asarray(fp.get("up", world_up if world_up is not None else (0.0, 0.0, 1.0)),
                          dtype=np.float64))
    height = float(fp.get("height", 2.5))
    half_t = max(float(fp.get("thickness", 0.15)), 0.02) / 2.0

    # Profile in the moving frame (u, v): u = wall-normal (thickness), v = up
    # (height). The path is at the wall *base*, so the profile spans v ∈ [0, H].
    prof = np.array([[-half_t, 0.0], [half_t, 0.0], [half_t, height], [-half_t, height]],
                    dtype=np.float64)
    # initial frame at path[0]
    t0 = path[1] - path[0]
    n = np.linalg.norm(t0)
    t0 = t0 / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    v0 = up - (up @ t0) * t0
    n = np.linalg.norm(v0)
    v0 = v0 / n if n > 1e-9 else _unit(np.cross(t0, np.array([0.0, 0.0, 1.0])))
    u0 = _unit(np.cross(v0, t0))
    frame = np.column_stack([u0, v0, t0])

    path_len = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    el = SweptElement(instance_id=instance_id, label=label, geometry_class="swept",
                      is_structure=bool(cls.is_structure), caption=caption,
                      caption_fields=dict(caption_fields or {}),
                      source_indices=source_indices,
                      profile_family="rect", profile_params={"w": float(2 * half_t), "h": height},
                      profile_polygon=prof, profile_frame=frame, path=path,
                      arc_segments=[], wall_thickness=0.0)
    el.meta["role"] = cls.role
    el.meta["subtype"] = "curved_wall"                # historical name; covers straight + curved walls
    el.meta["is_curved"] = not bool(fp.get("is_straight", False))
    el.meta["wall_thickness_m"] = float(2 * half_t)
    el.meta["wall_height_m"] = height
    el.confidence_stats = {
        "n_points": int(len(xyz)),
        "n_high_conf": int(len(xyz_hc)) if xyz_hc is not None else int(len(xyz)),
        "path_length_m": path_len,
        "fit_rms": float((cls.notes or {}).get("cyl_rms", 0.0)),
    }
    return el
