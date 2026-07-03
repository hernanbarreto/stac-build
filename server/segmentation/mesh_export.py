"""
MeshFlow input exporter — one segment point-cloud PLY per SAM3 instance.
========================================================================

Replaces the retired ShapeR PKL exporter. MeshFlow conditions on the segment
geometry directly (surface/point samples for its RoPE geometry condition), so
the multi-view PKL machinery (camera poses, Fisheye624 renders, T5 captions)
is gone: the export is just cleaned_cloud[globalIndices] → PLY.

Routing (config `meshflow:` + `surface_fit.fitted_roles`):
  - ARCHITECTURAL classes (walls, floors, vaults…) never come here — they go
    through the metric surface_fit pipeline. Instances whose label matches an
    architectural role are skipped with an explicit reason.
  - Segments larger than ``max_extent_m`` are skipped too: MeshFlow's ~4096-
    vertex budget is for individual OBJECTS, not scene-scale geometry; those
    stay on the TSDF path.

⚠ NON-METRIC OUTPUT: everything MeshFlow generates is a GENERATIVE visual
asset. The meta.json written here (and completed by run_meshflow_batch.py)
carries ``"metric": false`` and the GLB is named ``<folder>_visual.glb`` so
no downstream consumer can mistake it for measurement.

Output layout (same contract the UI/endpoints already speak):
    output/shape/<safe_label>_<id>/<safe_label>_<id>.ply          (input)
    output/shape/<safe_label>_<id>/<safe_label>_<id>_visual.glb   (generated)
    output/shape/<safe_label>_<id>/meta.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from segmentation.session_io import _safe_label

logger = logging.getLogger("MeshExport")

# Fallback architectural set when config is unavailable — keep in sync with
# reconstruction.surface_fit.scene.DEFAULT_FITTED_ROLES
_DEFAULT_ARCH_ROLES = ("wall", "floor", "ceiling", "slab", "vault", "tunnel",
                       "column", "beam", "deck", "platform", "ramp")


def _is_architectural(label: str, arch_roles: Sequence[str]) -> bool:
    ll = (label or "").lower()
    return any(r in ll for r in arch_roles)


def export_segment_plys(output_dir: Path,
                        segments_result: dict,
                        obj_ids: Optional[Sequence[int]] = None,
                        max_extent_m: float = 6.0,
                        arch_roles: Optional[Sequence[str]] = None,
                        progress_cb: Optional[Callable] = None,
                        ) -> Tuple[List[Path], List[Dict]]:
    """Write one PLY per eligible instance. Returns (exported, skipped).

    ``skipped`` entries carry {"instance_id", "label", "reason"} so the
    endpoint can tell the UI exactly why an instance has no visual mesh.
    """
    import open3d as o3d

    output_dir = Path(output_dir)
    if arch_roles is None:
        try:
            from config import get_param
            arch_roles = tuple(get_param("surface_fit.fitted_roles",
                                         list(_DEFAULT_ARCH_ROLES)))
        except Exception:
            arch_roles = _DEFAULT_ARCH_ROLES

    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        raise FileNotFoundError(f"missing {cloud_path} — run CloudComPy first")
    pts = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)
    logger.info("export: cloud %s pts, %d instances requested",
                f"{len(pts):,}", len(obj_ids) if obj_ids else -1)

    wanted = set(int(i) for i in obj_ids) if obj_ids else None
    exported: List[Path] = []
    skipped: List[Dict] = []

    for inst in segments_result.get("instances", []):
        iid = inst.get("instance_id", inst.get("id"))
        label = inst.get("label", f"object_{iid}")
        if wanted is not None and int(iid) not in wanted:
            continue

        if _is_architectural(label, arch_roles):
            skipped.append({"instance_id": iid, "label": label,
                            "reason": "architectural class → surface_fit (metric path)"})
            logger.info("export: %s_%s SKIPPED — architectural → surface_fit", label, iid)
            continue

        idx = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(pts))]
        if len(idx) < 100:
            skipped.append({"instance_id": iid, "label": label,
                            "reason": f"too few points ({len(idx)})"})
            continue
        seg = pts[idx]
        extent = float(np.linalg.norm(seg.max(0) - seg.min(0)))
        if extent > max_extent_m:
            skipped.append({"instance_id": iid, "label": label,
                            "reason": f"extent {extent:.1f} m > {max_extent_m:.1f} m "
                                      "(object budget) → TSDF path"})
            logger.warning("export: %s_%s SKIPPED — %.1f m exceeds max_extent_m=%.1f "
                           "(MeshFlow is per-object; stays on TSDF)", label, iid,
                           extent, max_extent_m)
            continue

        folder = _safe_label(label, int(iid))
        obj_dir = output_dir / "shape" / folder
        obj_dir.mkdir(parents=True, exist_ok=True)
        ply_path = obj_dir / f"{folder}.ply"
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seg))
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=True)

        meta = {
            "method": "meshflow",
            "metric": False,               # ⚠ generative visual asset, NOT measurement
            "generative": True,
            "instance_id": int(iid),
            "label": label,
            "n_points": int(len(seg)),
            "extent_m": extent,
            "input_ply": ply_path.name,
        }
        (obj_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        exported.append(ply_path)
        logger.info("export: %s → %s (%s pts, %.2f m)", folder, ply_path.name,
                    f"{len(seg):,}", extent)
        if progress_cb:
            progress_cb(instance_id=int(iid), phase="ply_ready")

    logger.info("export: %d PLYs written, %d skipped", len(exported), len(skipped))
    return exported, skipped
