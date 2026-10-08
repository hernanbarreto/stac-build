"""Downstream consistency after a swap: what regenerates, what is stamped
stale (consumer → action matrix, docs/correction_plan.md §7).

Regenerated here (cheap, in place):
  * instance store geometry: ``instance_points``/``instance_obb`` recomputed
    from the corrected cloud (display frame), 3-D finding anchors
    re-transformed via their origin ``frame_id``; ``user_volumes`` are USER
    geometry and are never moved; the epoch lands in ``scene_meta``.

Rebuilt from scratch (docs/plan_determinismo.md point 142, 2026-10-08):
  * after the certification ``rebuild_store_fresh`` writes scene_r.db into a
    NEW file from segmentation_result.json with a fixed insertion order
    (``segmentation.pipeline.rebuild_instance_store``, validated bit-identical
    to the matcher-built store) and re-inserts the user's and the chat's
    annotations in a fixed order — the in-place updates leave the SQLite
    pages in whatever shape every earlier writer left them.

Invalidated only (costly — the user decides, USER 2026-08-29): per-object
meshes (tsdf/poisson/surface_fit/perfect/shape), BIM registration/comparison,
coverage. ``derived_artifacts_status`` enumerates them with their stamped
epoch so the UI can badge them stale and offer Regenerate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
    # `globalIndices` index the cloud the masks were PROJECTED on (its size is the
    # result's `total_points`). A transform epoch warps that same cloud point by point,
    # so the indices still hold; a NEW cloud (another epoch's own reconstruction) has
    # other points in another order, and the same indices land on arbitrary points of
    # the whole scene. pccr 2026-09-30: every select between new-cloud epochs refit
    # every OBB from such indices and the store filled with walk-sized cubes. A
    # segmentation that does not index the live cloud is left as it is (points, OBBs,
    # instances) and declared stale — the masks have to be re-projected on this cloud.
    n_seg = result.get("total_points")
    indices_hold = n_seg is None or int(n_seg) == len(xyz)
    if not indices_hold:
        log(f"  ⚠ the segmentation indexes a cloud of {int(n_seg):,} points and the live "
            f"cloud has {len(xyz):,} — its points and OBBs are NOT refit from another "
            f"cloud's indices; re-project the masks on this cloud to update them")

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
            if not indices_hold:
                continue
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
        store.set_meta("segmentation_indexes_live_cloud", "true" if indices_hold else "false")
        summary = {"store": "updated" if indices_hold else "stale_segmentation",
                   "instances": n_inst,
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


# scene_meta rows the store's WRITERS own — regenerated with the store, never carried over
# (the ledger mirror is history that no longer lives in the store at all, point 136)
_MACHINE_META = ("built_from", "geometry_epoch", "segmentation_indexes_live_cloud",
                 "corrections_ledger")


def _read_annotations(db: Path) -> Dict[str, Any]:
    """Everything in scene_r.db that is NOT derived from segmentation_result.json — the
    user's and the chat's annotations: findings, user volumes, scene_meta (notes, scene
    description, dossiers, loop classes ...), instance classifications. Read in a FIXED order
    (by id / key) so re-inserting them is deterministic."""
    import sqlite3
    con = sqlite3.connect(str(db))
    try:
        con.row_factory = sqlite3.Row

        def _rows(table: str, order: str) -> List[dict]:
            try:
                return [dict(r) for r in con.execute(f"SELECT * FROM {table} ORDER BY {order}")]
            except sqlite3.OperationalError:
                return []                                     # a store without that table
        return {"findings": _rows("findings", "finding_id"),
                "user_volumes": _rows("user_volumes", "volume_id"),
                "scene_meta": [r for r in _rows("scene_meta", "key") if r["key"] not in _MACHINE_META],
                "instance_classification": _rows("instance_classification", "instance_id")}
    finally:
        con.close()


def _write_annotations(db: Path, ann: Dict[str, Any], live_ids: set) -> Dict[str, int]:
    """Re-insert the annotations into the FRESH store, in the order they were read; a
    finding or a classification of an instance the segmentation no longer has is dropped
    (and counted) — the store never lists a ghost."""
    import sqlite3
    con = sqlite3.connect(str(db))
    out = {"findings": 0, "user_volumes": 0, "scene_meta": 0, "instance_classification": 0,
           "findings_orphaned": 0, "classifications_orphaned": 0}
    try:
        for r in ann["findings"]:
            iid = r.get("instance_id")
            if iid is not None and int(iid) not in live_ids:
                out["findings_orphaned"] += 1
                continue
            cols = [k for k in r.keys()]
            con.execute(f"INSERT INTO findings ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [r[k] for k in cols])
            out["findings"] += 1
        for r in ann["user_volumes"]:
            cols = [k for k in r.keys()]
            con.execute(f"INSERT INTO user_volumes ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [r[k] for k in cols])
            out["user_volumes"] += 1
        for r in ann["instance_classification"]:
            if int(r["instance_id"]) not in live_ids:
                out["classifications_orphaned"] += 1
                continue
            cols = [k for k in r.keys()]
            con.execute(f"INSERT OR REPLACE INTO instance_classification ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})", [r[k] for k in cols])
            out["instance_classification"] += 1
        for r in ann["scene_meta"]:
            con.execute("INSERT OR REPLACE INTO scene_meta (key, value) VALUES (?, ?)",
                        (r["key"], r["value"]))
            out["scene_meta"] += 1
        con.commit()
    finally:
        con.close()
    return out


def rebuild_store_fresh(output_dir, log: Callable[[str], Any] = print) -> dict:
    """scene_r.db REBUILT into a new file from segmentation_result.json (point 142).

    The in-place updates (:func:`update_instance_store`, SAM3's writer, the chat) leave the
    SQLite file's pages in the shape of every write that ever touched it, so two certifications
    of the same inputs never gave the same bytes. The geometric tables are rebuilt from the
    result with the matcher's own writer (``segmentation.pipeline.rebuild_instance_store``,
    fixed insertion order), then the annotations — findings (already re-anchored in place by
    the epoch that moved them), user volumes, chat notes / scene description / dossiers / loop
    classes in scene_meta, classifications — are re-inserted in a fixed order. A session with
    nothing to annotate rebuilds to the same bytes every time.

    DECLARED: the plan asks the annotations to live in a SEPARATE database; its readers
    (phase5_qa, segmentation.object_analysis, loops.semantic_classes — other packages) open
    scene_r.db, so until they read a second file the annotations are carried over into the
    fresh store in a fixed order instead. A failure of the rebuild RAISES (the old store is
    kept intact beside the attempt, nothing half-written)."""
    output_dir = Path(output_dir)
    db = output_dir / "scene_r.db"
    res_path = output_dir / "segmentation_result.json"
    if not res_path.exists():
        return {"store": "no segmentation_result.json"}
    result = json.loads(res_path.read_text())
    live_ids = {int(i.get("instance_id", i.get("id", -1))) for i in (result.get("instances") or [])
                if i.get("globalIndices")}
    if not live_ids:
        return {"store": "no instance with points"}
    ann = _read_annotations(db) if db.exists() else {"findings": [], "user_volumes": [],
                                                     "scene_meta": [], "instance_classification": []}
    # the rebuild writes output/scene_r.db; the current store steps aside first and comes
    # back untouched if the rebuild cannot deliver
    backup = output_dir / "scene_r.db.before_rebuild"
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(backup) + suffix)
        if p.exists():
            p.unlink()
    had_store = db.exists()
    if had_store:
        db.rename(backup)
        for suffix in ("-wal", "-shm"):
            p = Path(str(db) + suffix)
            if p.exists():
                p.unlink()
    try:
        from segmentation.pipeline import rebuild_instance_store
        ok = bool(rebuild_instance_store(output_dir))
        if not ok or not db.exists():
            raise RuntimeError("segmentation.pipeline.rebuild_instance_store wrote no store "
                               "(it logs why) — the certification's object store is incomplete")
        carried = _write_annotations(db, ann, live_ids)
        from phase_r.instance_store import InstanceStore
        st = InstanceStore(db)
        try:
            st.set_meta("geometry_epoch", str(current_epoch(output_dir)))
            st.set_meta("segmentation_indexes_live_cloud", "true")
        finally:
            st.close()
        # WAL mode leaves the journal beside the file: checkpoint it so the .db holds everything
        import sqlite3
        con = sqlite3.connect(str(db))
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()
    except BaseException:
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(db) + suffix)
            if p.exists():
                p.unlink()
        if had_store:
            backup.rename(db)
        raise
    if had_store:
        backup.unlink()
    summary = {"store": "rebuilt", "instances": len(live_ids), **carried}
    log(f"  instance store rebuilt from segmentation_result.json (fixed order): {summary}")
    return summary


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
