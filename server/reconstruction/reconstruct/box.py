"""Reconstruct a `BoxElement` (oriented bounding box) from a classified instance."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..elements import BoxElement
from ..geometry.primitives import _unit

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


def _obb_from_points(pts: np.ndarray):
    """Robust oriented bounding box → (center(3,), R(3,3 cols=axes), half_extents(3,))."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) >= 5 and o3d is not None:
        try:
            obb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).get_oriented_bounding_box()
            return np.asarray(obb.center), np.asarray(obb.R), np.asarray(obb.extent) / 2.0
        except Exception:
            pass
    # PCA fallback
    c = pts.mean(0)
    Q = pts - c
    _, evec = np.linalg.eigh(Q.T @ Q)
    R = evec[:, ::-1]  # columns: largest→smallest variance
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    loc = Q @ R
    lo, hi = loc.min(0), loc.max(0)
    center = c + R @ ((lo + hi) / 2.0)
    return center, R, (hi - lo) / 2.0


def reconstruct_box(cls, *, instance_id: int, label: str, xyz: np.ndarray,
                    xyz_hc: Optional[np.ndarray] = None,
                    world_up: Optional[np.ndarray] = None, caption: str = "",
                    caption_fields: Optional[Dict[str, str]] = None,
                    source_indices=None, obb: Optional[Dict[str, Any]] = None,
                    dist_thresh: float = 0.012) -> BoxElement:
    pts_fit = np.asarray(xyz_hc if (xyz_hc is not None and len(xyz_hc) >= 8) else xyz, dtype=np.float64)
    # prefer a re-fit OBB; fall back to the segmentation OBB only if the fit is degenerate
    center, R, he = _obb_from_points(pts_fit)
    if (he < 1e-3).any() and obb is not None and "half_extents" in obb:
        center = np.asarray(obb["center"], dtype=np.float64)
        R = np.asarray(obb.get("rotation", np.eye(3)), dtype=np.float64)
        he = np.asarray(obb["half_extents"], dtype=np.float64)

    el = BoxElement(instance_id=instance_id, label=label, geometry_class="box",
                    is_structure=cls.is_structure, caption=caption,
                    caption_fields=dict(caption_fields or {}),
                    source_indices=source_indices, center=center, R=R,
                    half_extents=he)
    el.meta["role"] = cls.role
    # observed fraction: points inside the (slightly grown) box
    loc = (np.asarray(xyz, dtype=np.float64) - center) @ R
    inside = np.all(np.abs(loc) <= (he + 2.0 * dist_thresh), axis=1)
    el.confidence_stats = {"n_points": int(len(xyz)),
                           "n_high_conf": int(len(pts_fit)),
                           "observed_fraction": float(inside.mean()) if len(xyz) else 0.0}
    return el
