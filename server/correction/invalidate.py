"""Downstream consistency after a swap: what regenerates, what is stamped
stale (consumer → action matrix, docs/correction_plan.md §7).

Regenerated here (cheap, in place):
  * instance store geometry: ``instance_points``/``instance_obb`` recomputed
    from the corrected cloud (display frame), 3-D finding anchors
    re-transformed via their origin ``frame_id``; ``user_volumes`` are USER
    geometry and are never moved; the epoch lands in ``scene_meta``.

Invalidated only (costly — the user decides, USER 2026-08-29): per-object
meshes (tsdf/poisson/surface_fit/perfect/shape), BIM registration/comparison,
coverage. ``derived_artifacts_status`` enumerates them with their stamped
epoch so the UI can badge them stale and offer Regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from correction.epoch import current_epoch

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)


def update_instance_store(output_dir, R_kf: np.ndarray, t_kf: np.ndarray,
                          k_kf: np.ndarray, frames: List[int],
                          log=print, b_kf: Optional[np.ndarray] = None) -> dict:
    """In-place geometry refresh of scene_r.db from the CORRECTED session
    files. A full store rebuild would delete findings/volumes/notes — this
    updates instead. Returns a summary; raises with an actionable message on
    failure (the correction stays applied; re-run this step after fixing)."""
    output_dir = Path(output_dir)
    db = output_dir / "scene_r.db"
    if not db.exists():
        return {"store": "absent"}
    res_path = output_dir / "segmentation_result.json"
    if not res_path.exists():
        return {"store": "no segmentation_result.json"}

    from correction.session import read_ply, read_poses
    from phase_r.instance_store import InstanceStore
    from segmentation.pipeline import _compute_obb

    _, data = read_ply(output_dir / "cleaned_cloud.ply")
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    ft = output_dir / "floor_transform.npz"
    if ft.exists():
        d = np.load(ft)
        s_, R_, t_ = float(d["s"]), d["R"], d["t"]
    else:
        s_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
    xyz_display = (s_ * (xyz @ R_.T)) + t_

    result = json.loads(res_path.read_text())
    kf_of_frame = {int(f): k for k, f in enumerate(frames)}

    store = InstanceStore(db)
    try:
        n_inst = 0
        live_ids = []
        known = {int(r.get("instance_id")) for r in store.list_instances()}
        for inst in result.get("instances", []):
            iid = int(inst.get("instance_id", inst.get("id", -1)))
            if iid < 0:
                continue
            live_ids.append(iid)
            # an instance the store never heard of (a brush segment, a split)
            # used to be silently skipped: the geometry refreshed around a hole
            if iid not in known:
                store.upsert_instance(iid, str(inst.get("label") or "segment"),
                                      source="sam3_concepts", status="proposed",
                                      label_origin="vlm_proposed")
                known.add(iid)
            g = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            g = g[(g >= 0) & (g < len(xyz_display))]
            if not len(g):
                continue
            pts = xyz_display[g].astype(np.float32)
            fids = data["frame_global"][g].astype(np.int32)
            store.set_points(iid, pts, frame_ids=fids)
            # the OBB needs a volume to fit; the points stand on their own
            if len(g) < 4:
                n_inst += 1
                continue
            obb = _compute_obb(xyz_display[g])
            # same OBB→store mapping as segmentation.pipeline._write_instance_store
            c = np.asarray(obb["center"], float)
            h = np.asarray(obb["half_extents"], float)
            Rd = np.asarray(obb.get("rotation", np.eye(3)), float)
            T = np.eye(4)
            T[:3, :3] = Rd
            T[:3, 3] = c
            aabb = np.array([-h[0], h[0], -h[1], h[1], -h[2], h[2]])
            store.set_obb(iid, T, aabb, c, n_points=len(pts),
                          obb_origin="tool_measured")
            n_inst += 1

        # 3-D findings: re-anchor per origin keyframe (they carry frame_id;
        # the anchor lives in the same world frame as the cloud). Full warp:
        # depth k along the ray from the PRE-correction camera centre, then
        # the keyframe's rigid transform. The pre-correction centre is
        # recovered from the corrected pose: c0 = R^T (c_corr - t).
        poses_corr = read_poses(output_dir / "camera_poses.txt")
        n_findings = 0
        n_findings_lost = 0
        for f in store.list_findings():
            p3 = f.get("point3d")
            fid = f.get("frame_id")
            if p3 is None:
                continue
            kf = kf_of_frame.get(int(fid)) if fid is not None else None
            if kf is None:
                n_findings_lost += 1
                continue
            p = np.asarray(p3, dtype=np.float64).reshape(3)
            kv = float(k_kf[kf])
            bv = float(b_kf[kf]) if b_kf is not None else 0.0
            if kv != 1.0 or bv != 0.0:
                cam_corr = poses_corr[kf][:3, 3]
                cam0 = R_kf[kf].T @ (cam_corr - t_kf[kf])
                if bv != 0.0:
                    axis0 = R_kf[kf].T @ poses_corr[kf][:3, 2]
                    z = float((p - cam0) @ axis0)
                    zc = z if abs(z) > 1e-9 else 1e-9
                    p = cam0 + (p - cam0) * ((kv * z + bv) / zc)
                else:
                    p = cam0 + (p - cam0) * kv
            p = R_kf[kf] @ p + t_kf[kf]
            store.conn.execute(
                "UPDATE findings SET point3d = ? WHERE finding_id = ?",
                (np.ascontiguousarray(p, dtype=np.float32).tobytes(),
                 int(f["finding_id"])))
            n_findings += 1
        store.conn.commit()
        # what the segmentation no longer has stops existing here too — the
        # store is the canonical object list and a ghost in it is a lie the
        # chat, the findings and the reports all repeat
        recon = store.reconcile(live_ids)
        store.set_meta("geometry_epoch", str(current_epoch(output_dir)))
        summary = {"store": "updated", "instances": n_inst,
                   "removed": recon["removed"],
                   "findings_orphaned": recon["findings_orphaned"],
                   "findings_transformed": n_findings,
                   "findings_unresolvable": n_findings_lost}
        if recon["removed"]:
            log(f"  instance store: {recon['removed']} instance(s) dropped — "
                f"{recon.get('removed_ids')}")
        log(f"  instance store updated in place: {summary}")
        return summary
    finally:
        store.close()


# Derived artifacts that are INVALIDATED, not regenerated. Each row:
# (kind, glob relative to output_dir, meta filename inside each hit or None
#  = the hit itself is the meta json).
_DERIVED = [
    ("tsdf_mesh", "tsdf/*/", "*.meta.json"),
    ("scene_mesh", "tsdf/", "scene.meta.json"),
    ("surface_fit", "surface_fit/*/", "meta.json"),
    ("surface_fit_report", "surface_fit/", "scene_report.json"),
    ("bim_comparison", "sabana/", "sabana_meta.json"),
    ("pgsr_render", "pgsr_render/", "report.json"),
]


def derived_artifacts_status(output_dir) -> List[dict]:
    """Every derived artifact with its stamped epoch and staleness — drives
    the UI badges. An artifact without a stamp predates the epoch system and
    counts as epoch 0."""
    output_dir = Path(output_dir)
    cur = current_epoch(output_dir)
    rows: List[dict] = []
    seen = set()
    for kind, pattern, meta_pat in _DERIVED:
        for base in sorted(output_dir.glob(pattern)):
            metas = sorted(base.glob(meta_pat)) if base.is_dir() else []
            for meta_path in metas:
                if meta_path in seen:
                    continue
                seen.add(meta_path)
                try:
                    meta = json.loads(meta_path.read_text())
                except (ValueError, OSError):
                    meta = {}
                art_epoch = int(meta.get("geometry_epoch", 0) or 0)
                rows.append({
                    "kind": kind,
                    "name": (base.name if base.name not in ("tsdf",
                                                            "surface_fit",
                                                            "sabana")
                             else meta_path.stem),
                    "path": str(meta_path.relative_to(output_dir)),
                    "geometry_epoch": art_epoch,
                    "stale": art_epoch != cur,
                })
    return rows
