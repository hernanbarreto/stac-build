"""
MeshFlow input exporter — one segment point-cloud PLY per SAM3 instance.
========================================================================

Replaces the retired ShapeR PKL exporter. MeshFlow conditions on the segment
geometry directly (surface/point samples for its RoPE geometry condition), so
the multi-view PKL machinery (camera poses, Fisheye624 renders, T5 captions)
is gone: the export is just cleaned_cloud[globalIndices] → PLY.

Routing (config `meshflow:`):
  - By default EVERY class can be generated (user decision 2026-07-03): the
    assets are visual/non-metric regardless of category and are never compared
    against BIM — surface_fit owns the metric surfaces. Set
    ``exclude_architectural: true`` to re-enable the old category routing.
  - Segments larger than ``max_extent_m`` are skipped: MeshFlow's ~4096-vertex
    budget degrades on scene-scale geometry; those stay on the TSDF path.

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


def _best_view_ref_image(output_dir: Path, frames_dir: Optional[Path],
                         obj_id: int, out_path: Path,
                         pad_frac: float = 0.18) -> Optional[dict]:
    """Pick the frame where this object's mask covers the most pixels
    (traceability: seg_masks.npz keys are 'f<real_frame>_o<obj_id>'), crop the
    frame image to the mask bbox (+padding) and save it as the MeshFlow
    reference image. Returns {"frame", "area_px", "path"} or None."""
    import cv2
    masks_path = Path(output_dir) / "seg_masks.npz"
    if not masks_path.exists() or frames_dir is None:
        return None
    frames_dir = Path(frames_dir)
    # frame files by their numeric id (e.g. "001188.jpg" → 1188)
    import re
    frame_files = {}
    for f in frames_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            m = re.search(r"(\d+)", f.stem)
            if m:
                frame_files[int(m.group(1))] = f
    if not frame_files:
        return None

    npz = np.load(masks_path, allow_pickle=True)
    best = None   # (area, frame, mask)
    suffix = f"_o{obj_id}"
    for key in npz.files:
        if not (key.startswith("f") and key.endswith(suffix)):
            continue
        try:
            frame = int(key[1:-len(suffix)])
        except ValueError:
            continue
        if frame not in frame_files:
            continue
        mask = npz[key]
        area = int(np.count_nonzero(mask))
        if area > 0 and (best is None or area > best[0]):
            best = (area, frame, mask)
    if best is None:
        return None
    area, frame, mask = best
    img = cv2.imread(str(frame_files[frame]))
    if img is None:
        return None
    mh, mw = mask.shape[:2]
    ih, iw = img.shape[:2]
    ys, xs = np.nonzero(mask)
    # mask → image scale (masks may be stored at a lower resolution)
    sx, sy = iw / mw, ih / mh
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = int(max(0, x0 - pw)); x1 = int(min(iw, x1 + pw))
    y0 = int(max(0, y0 - ph)); y1 = int(min(ih, y1 + ph))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"frame": frame, "area_px": area, "path": str(out_path)}


def export_segment_plys(output_dir: Path,
                        segments_result: dict,
                        obj_ids: Optional[Sequence[int]] = None,
                        max_extent_m: float = 15.0,
                        arch_roles: Optional[Sequence[str]] = None,
                        exclude_architectural: bool = False,
                        frames_dir: Optional[Path] = None,
                        require_ref_image: bool = True,
                        progress_cb: Optional[Callable] = None,
                        ) -> Tuple[List[Path], List[Dict]]:
    """Write one PLY (+ mandatory reference image) per eligible instance.
    Returns (exported, skipped).

    Reference image: MeshFlow's visual conditioning is OPTIONAL upstream but
    MANDATORY here (project decision) — geometry-only generations came out
    wrong on real scan segments. The best view is chosen by mask pixel count
    (full traceability via seg_masks.npz); instances without a usable view
    are SKIPPED with an explicit reason rather than generated blind.

    ``skipped`` entries carry {"instance_id", "label", "reason"} so the
    endpoint can tell the UI exactly why an instance has no visual mesh.
    """
    import open3d as o3d

    output_dir = Path(output_dir)
    if arch_roles is None:
        # surface_fit.fitted_roles no longer exists (name-based routing removed
        # 2026-08-29); this dormant gate (exclude_architectural, OFF since
        # 2026-07-03) keeps its local fallback list only.
        arch_roles = _DEFAULT_ARCH_ROLES

    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        raise FileNotFoundError(f"missing {cloud_path} — run CloudComPy first")
    pts = np.asarray(o3d.io.read_point_cloud(str(cloud_path)).points)

    # seg_masks.npz keys use the SAM3 OBJ id space ('id' in segmentation.json),
    # while segmentation_result instances carry DBSCAN-renumbered instance_ids.
    # Looking masks up by instance_id fetched ANOTHER object's view (ladder
    # inst 1 vs floor obj 1 on test3). Map via segmentation.json: instance_id
    # → obj id, with a label-match fallback.
    mask_obj_by_iid: Dict[int, int] = {}
    mask_obj_by_label: Dict[str, int] = {}
    seg_json = output_dir / "segmentation.json"
    if seg_json.exists():
        try:
            for e in (json.loads(seg_json.read_text()).get("instances") or []):
                if e.get("id") is not None:
                    if e.get("instance_id") is not None:
                        mask_obj_by_iid[int(e["instance_id"])] = int(e["id"])
                    if e.get("label"):
                        mask_obj_by_label.setdefault(str(e["label"]), int(e["id"]))
        except Exception as e:
            logger.warning("export: could not map obj ids from segmentation.json: %s", e)
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

        # Category routing is OFF by default (user decision 2026-07-03):
        # generative assets may be produced for ANY class — they are still
        # non-metric and never compared against BIM (surface_fit owns that).
        if exclude_architectural and _is_architectural(label, arch_roles):
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

        # MANDATORY reference image (best view by mask area) — see docstring.
        # NPZ masks are keyed by the SAM3 obj id, NOT the result instance_id.
        mask_oid = mask_obj_by_iid.get(int(iid),
                                       mask_obj_by_label.get(label, int(iid)))
        ref = _best_view_ref_image(output_dir, frames_dir, mask_oid,
                                   obj_dir / f"{folder}_ref.jpg")
        if ref is None and require_ref_image:
            skipped.append({"instance_id": iid, "label": label,
                            "reason": "no reference view found (image "
                                      "conditioning is mandatory)"})
            logger.warning("export: %s_%s SKIPPED — no usable mask/frame for "
                           "the mandatory reference image", label, iid)
            continue
        if ref:
            logger.info("export: %s ref image ← frame %d (%d mask px)",
                        folder, ref["frame"], ref["area_px"])

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
            "ref_image": (Path(ref["path"]).name if ref else None),
            "ref_frame": (ref["frame"] if ref else None),
        }
        (obj_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        exported.append(ply_path)
        logger.info("export: %s → %s (%s pts, %.2f m)", folder, ply_path.name,
                    f"{len(seg):,}", extent)
        if progress_cb:
            progress_cb(instance_id=int(iid), phase="ply_ready")

    logger.info("export: %d PLYs written, %d skipped", len(exported), len(skipped))
    return exported, skipped
