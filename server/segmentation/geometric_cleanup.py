# STAC-Builder — the SECOND moment of the geometric filter: what the
# correction could not save comes out of the cloud.
#
# USER 2026-09-15: "sé que ese punto debe estar, no es ruido, es un huérfano mal
# ubicado, si no lo puedo corregir, lamentablemente lo voy a tener que sacar".
#
# THE CYCLE HAS TWO MOMENTS, AND THIS IS THE SECOND.
#
# The first (segmentation/mask_filter, at match time) MARKS every point that
# lands off its own mask in a view that sees it unoccluded, and writes the marks
# to `out_of_place.npy`. It removes nothing, on purpose: the certification may
# still move that point to where it belongs. A drift orphan falls badly before
# the correction and correctly after it, and deleting it first would destroy the
# very evidence the closure is measured from.
#
# This is the moment after. It re-measures the SAME points on the CORRECTED
# geometry and asks the only question left: is it STILL in the wrong place?
#   · what now lands inside was a drift orphan and is cured — nothing happens
#     to it, and the mark is dropped;
#   · what still lands outside has no correction left to wait for, and comes
#     out of the cloud.
#
# Why removal is legitimate here and nowhere else. USER: "sale de la nube, es
# como el filtro SOR, la nube es la verdad para el TSDF o el PGSR, acá estamos
# aplicando filtrado geométrico". The doctrine protects the cloud from a MODEL
# replacing it, not from the images correcting it — CloudCompy already filters
# it with SOR, on far weaker evidence than a mask.
#
# No threshold decides anything here. A point is removed when two independent
# measurements, before and after the correction, both say it lands off its own
# mask in a frame that sees it. The precision asked of it is the MASK's own,
# border error included ("no le vamos a pedir más precisión que la de la propia
# máscara" — the audit's `dilate_px` is that border, and nothing else).
#
# Removal is physical but NOT unrecoverable: every removed row is written to
# `corrections/geometric_cleanup_<epoch>.npz` with its old index, so
# `undo_cleanup` puts the cloud back exactly as it was.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

MARK_FILE = "out_of_place.npy"
REPORT = "geometric_cleanup.json"
UNDO_DIR = "corrections"

# every cloud that shares the point order of cleaned_cloud.ply and must follow
# it row for row
CLOUD_FILES = ("cleaned_cloud.ply", "cleaned_cloud_raw.ply", "corrected_cloud.ply")


def _cfg(cfg: Optional[dict]) -> dict:
    if cfg:
        return cfg
    from config import cfg as server_cfg
    return (server_cfg.get("segmentation") or {})


def _load_marks(output_dir: Path) -> Optional[np.ndarray]:
    p = output_dir / MARK_FILE
    if not p.is_file():
        return None
    try:
        return np.asarray(np.load(p), bool)
    except Exception as e:  # noqa: BLE001 — declared, never silent
        print(f"[cleanup] {MARK_FILE} unreadable ({e}) — no second pass")
        return None


def _instance_id(inst: dict) -> int:
    return int(inst.get("instance_id", inst.get("id", -1)))


def measure(output_dir, session_dir=None, cfg: Optional[dict] = None,
            log: Callable[[str], None] = print) -> Tuple[Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         dict]:
    """Re-measure the marked points on the geometry as it stands NOW.

    Returns ``(still_out, measured, report)`` — two per-point boolean arrays
    over the current cloud and the report. ``still_out`` is what the correction
    failed to save; ``measured`` is what could be judged at all (an instance
    with no mask evidence judges nothing, and its marks survive untouched for a
    later pass). Both are ``None`` when there was nothing to measure.
    """
    from correction.session import load_session
    from segmentation.mask_filter import MaskAudit
    from segmentation.pipeline import _mask_frame_lookup

    output_dir = Path(output_dir)
    session_dir = Path(session_dir) if session_dir else output_dir.parent
    rep: dict = {"version": 1, "provenance": "tool_measured", "applied": False,
                 "marked": 0, "cured": 0, "removable": 0, "removed": 0,
                 "unmeasured": 0, "orphaned": 0, "reason": None,
                 "per_instance": []}

    marks = _load_marks(output_dir)
    if marks is None:
        rep["reason"] = (f"no {MARK_FILE} — the first moment of the cycle never "
                         f"ran on this session")
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep
    rep["marked"] = int(marks.sum())
    if not marks.any():
        rep["reason"] = "the audit marked nothing out of place — nothing to clean"
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep

    res_path = output_dir / "segmentation_result.json"
    if not res_path.is_file():
        rep["reason"] = "no segmentation_result.json — no instances to re-measure"
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep

    try:
        session = load_session(output_dir)
    except Exception as e:  # noqa: BLE001
        rep["reason"] = f"the cloud could not be loaded ({e})"
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep
    if len(marks) != session.n_points:
        rep["reason"] = (f"the marks are stale: {len(marks):,} flags against "
                         f"{session.n_points:,} points — the cloud was rebuilt "
                         f"after the audit, so nothing is removed")
        log(f"[cleanup] ⚠ {rep['reason']}")
        return None, None, rep

    masks_npz = output_dir / "seg_masks.npz"
    if not masks_npz.is_file():
        rep["reason"] = "no seg_masks.npz — no ground truth to measure against"
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep

    seg_cfg = _cfg(cfg)
    mf_cfg = seg_cfg.get("mask_filter") or {}
    z = np.load(masks_npz, allow_pickle=True)
    c2m = _mask_frame_lookup(output_dir, z["frames"].tolist(),
                             {int(f) for f in np.unique(session.fg)})
    audit = MaskAudit(output_dir, session_dir, mf_cfg, cloud_to_mask=c2m, log=log)
    if not audit.ok:
        rep["reason"] = ("no mask/camera evidence — nothing measured, nothing "
                         "removed")
        log(f"[cleanup] {rep['reason']}")
        return None, None, rep

    instances = json.loads(res_path.read_text()).get("instances") or []
    still = np.zeros(session.n_points, bool)
    measured = np.zeros(session.n_points, bool)
    owned = np.zeros(session.n_points, bool)
    per_instance: List[dict] = []
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < session.n_points)]
        owned[gi] = True
        if len(gi) == 0 or not marks[gi].any():
            continue
        rec = audit.audit(_instance_id(inst), session.xyz[gi],
                          session.fg[gi].astype(np.int64))
        out = rec.get("out_of_place")
        if out is None or len(out) != len(gi):
            continue          # no verdict for this instance: its marks stand
        now = np.asarray(out, bool)
        seen = rec.get("judged")
        if seen is None or len(seen) != len(gi):
            continue          # the masks did not look: its marks stand
        # A marked point is only DECIDED where the masks looked at it again.
        # One that no frame saw this time was not found clean, it was not
        # asked — its mark survives for a later pass.
        was = marks[gi]
        decided = was & np.asarray(seen, bool)
        measured[gi[decided]] = True
        still[gi[decided & now]] = True
        per_instance.append({
            "instance_id": _instance_id(inst), "label": inst.get("label"),
            "verdict": rec.get("verdict"),
            "marked": int(was.sum()),
            "cured": int((decided & ~now).sum()),
            "still_out": int((decided & now).sum()),
            "unmeasured": int((was & ~decided).sum())})

    rep["removable"] = int(still.sum())
    rep["cured"] = int((measured & ~still).sum())
    # Two different silences, and the report must not mix them. A marked point
    # that now belongs to NO instance has already taken the softer landing the
    # user allowed ("o quedará como unsegmented, y después vemos"): there is no
    # mask that claims it, so there is nothing to judge it against and it stays.
    # A marked point that IS in an instance but that no mask frame saw this time
    # was simply not asked, and is worth re-asking later.
    rep["orphaned"] = int((marks & ~owned).sum())
    rep["unmeasured"] = int((marks & owned & ~measured).sum())
    rep["per_instance"] = sorted(per_instance, key=lambda r: -r["still_out"])
    log(f"[cleanup] the audit had marked {rep['marked']:,} point(s) out of "
        f"place; the correction cured {rep['cured']:,}, {rep['removable']:,} "
        f"still land off their own mask, {rep['unmeasured']:,} were not looked "
        f"at again and {rep['orphaned']:,} now belong to no instance")
    return still, measured, rep


def geometric_cleanup(output_dir, session_dir=None, cfg: Optional[dict] = None,
                      apply: bool = True, log: Callable[[str], None] = print) -> dict:
    """Close the cycle: measure, then remove what the correction did not save.

    ``apply=False`` measures and reports without touching a byte.
    """
    output_dir = Path(output_dir)
    still, measured, rep = measure(output_dir, session_dir, cfg, log=log)
    if still is None:
        _write_report(output_dir, rep)
        return rep
    marks = _load_marks(output_dir)

    if not apply:
        rep["reason"] = "measured only — nothing was removed"
        _write_report(output_dir, rep)
        return rep
    if not still.any():
        # the correction saved everything: drop the cured marks and stop
        np.save(output_dir / MARK_FILE, marks & ~measured)
        rep["reason"] = "the correction saved every marked point — nothing removed"
        rep["applied"] = True
        log(f"[cleanup] {rep['reason']}")
        _write_report(output_dir, rep)
        return rep

    ok, detail = _remove(output_dir, still, marks & ~measured, log=log)
    rep.update(detail)
    rep["applied"] = bool(ok)
    if not ok:
        rep["reason"] = detail.get("reason") or "removal aborted — cloud untouched"
        log(f"[cleanup] ⚠ {rep['reason']}")
    _write_report(output_dir, rep)
    return rep


def _write_report(output_dir: Path, rep: dict) -> None:
    from atomic_io import atomic_write_json
    try:    # every derived artifact carries the geometry epoch it was measured on
        from correction.epoch import stamp
        stamp(rep, output_dir)
    except Exception:  # noqa: BLE001 — a session with no epoch machinery
        pass
    try:
        atomic_write_json(Path(output_dir) / REPORT, rep, indent=1, default=float)
    except Exception as e:  # noqa: BLE001
        print(f"[cleanup] report not written ({e})")


def _refit_obbs(output_dir: Path, touched: List[dict], n_new: int,
                log: Callable[[str], None] = print) -> int:
    """Recompute the box of every instance that lost points.

    Not a courtesy: the flyers the cleanup just removed are exactly what was
    inflating those boxes, and an inflated box is not cosmetic — the floor
    levelling picks its candidate by height and the duplicate measurements read
    the extent. Leaving the old box would keep the error the removal cured.
    """
    if not touched:
        return 0
    try:
        import open3d as o3d
        from segmentation.pipeline import _compute_obb
        xyz = np.asarray(o3d.io.read_point_cloud(
            str(output_dir / "cleaned_cloud.ply")).points)
        if len(xyz) != n_new:
            log(f"[cleanup] boxes left alone: the cloud reads {len(xyz):,} "
                f"points against {n_new:,} kept")
            return 0
        # the display frame the matcher and the viewer both use
        tp = output_dir / "floor_transform.npz"
        if tp.is_file():
            d = np.load(tp)
            s, R, t = float(d["s"]), np.asarray(d["R"]), np.asarray(d["t"])
            if not (np.allclose(R, np.eye(3)) and np.allclose(t, np.zeros(3))):
                xyz = s * (xyz @ R.T) + t
    except Exception as e:  # noqa: BLE001 — declared; a stale box is not a corrupt one
        log(f"[cleanup] boxes not refitted ({e}) — they keep the extent the "
            f"removed points gave them")
        return 0
    done = 0
    for inst in touched:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        if len(gi) == 0:
            continue
        try:
            inst["obb"] = _compute_obb(xyz[gi])
            done += 1
        except Exception as e:  # noqa: BLE001
            log(f"[cleanup] box of instance {_instance_id(inst)} kept ({e})")
    log(f"[cleanup] {done} box(es) refitted on the cleaned points")
    return done


def _remove(output_dir: Path, drop: np.ndarray, keep_marks: np.ndarray,
            log: Callable[[str], None] = print) -> Tuple[bool, dict]:
    """Physically remove ``drop`` from every cloud and carry every index with it.

    The undo snapshot is written BEFORE the first cloud is touched, so a failure
    midway is always recoverable.
    """
    from segmentation.erase import _rewrite_ply_keep, _write_classification
    from correction.session import read_ply
    from atomic_io import atomic_write_json

    n = len(drop)
    keep = ~drop
    detail: dict = {"points_before": int(n), "points_after": int(keep.sum()),
                    "removed": int(drop.sum()), "clouds": [], "reason": None}

    present = [p for p in (output_dir / f for f in CLOUD_FILES) if p.is_file()]
    if not present:
        detail["reason"] = "no cloud file to clean"
        return False, detail
    # One pass over each cloud: validate the point count and keep ONLY the rows
    # that are about to leave. The clouds run to tens of millions of points, so
    # reading them twice (once to check, once to snapshot) is half a gigabyte of
    # pointless I/O per file.
    doomed: dict = {}
    for p in present:
        try:
            _h, d = read_ply(p)
        except Exception as e:  # noqa: BLE001
            detail["reason"] = f"{p.name} unreadable ({e}) — nothing touched"
            return False, detail
        if len(d) != n:
            detail["reason"] = (f"{p.name} has {len(d):,} points against "
                                f"{n:,} marks — the clouds are out of step, "
                                f"nothing touched")
            return False, detail
        doomed[p.name] = d[drop].copy()
        del d

    res_path = output_dir / "segmentation_result.json"
    doc = json.loads(res_path.read_text())
    instances = doc.get("instances") or []

    # the undo snapshot, before anything moves. It carries the OWNER of every
    # removed point as well as its row: putting the geometry back without
    # putting it back in its instance would restore a cloud and lose a segment.
    try:
        from correction.epoch import current_epoch
        epoch = int(current_epoch(output_dir))
    except Exception:  # noqa: BLE001 — the snapshot is worth more than its name
        epoch = -1
    removed_index = np.flatnonzero(drop).astype(np.int64)
    owner = np.full(n, -1, np.int64)
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < n)]
        owner[gi] = _instance_id(inst)
    undo_dir = output_dir / UNDO_DIR
    undo_dir.mkdir(parents=True, exist_ok=True)
    undo_path = undo_dir / f"geometric_cleanup_{epoch}.npz"
    try:
        snap = {"removed_index": removed_index, "removed_owner": owner[drop],
                "n_points": np.int64(n), "epoch": np.int64(epoch)}
        for name, rows in doomed.items():
            snap[f"rows__{name}"] = rows
        np.savez_compressed(undo_path, **snap)
        detail["undo"] = str(undo_path.relative_to(output_dir))
    except Exception as e:  # noqa: BLE001
        detail["reason"] = f"undo snapshot failed ({e}) — nothing removed"
        return False, detail

    for p in present:
        if not _rewrite_ply_keep(p, keep):
            detail["reason"] = (f"{p.name} could not be rewritten — the clouds "
                                f"are now out of step; restore with "
                                f"`undo_cleanup` ({undo_path.name})")
            return False, detail
        detail["clouds"].append(p.name)

    # every stored index follows the cloud
    remap = np.full(n, -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    touched: List[dict] = []
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < n)]
        ng = remap[gi]
        ng = ng[ng >= 0]
        if len(ng) != len(gi):
            touched.append(inst)
        inst["globalIndices"] = ng.tolist()
        inst["total_points"] = int(len(ng))
    _refit_obbs(output_dir, touched, int(keep.sum()), log=log)
    doc["total_points"] = int(keep.sum())
    doc["segmented_points"] = sum(int(i.get("total_points") or 0) for i in instances)
    doc["coverage"] = round(doc["segmented_points"] / max(1, doc["total_points"]), 4)
    atomic_write_json(res_path, doc)

    # classification.npy is what the octree bakes its colours from
    try:
        _write_classification(output_dir, instances, int(keep.sum()))
    except Exception as e:  # noqa: BLE001
        log(f"[cleanup] classification rebuild failed ({e}) — the octree will "
            f"be rebuilt from stale colours until the next match")

    np.save(output_dir / MARK_FILE, keep_marks[keep])

    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        log(f"[cleanup] instance store rebuild failed ({e})")

    try:
        from potree_converter import convert_ply_to_potree
        if convert_ply_to_potree(output_dir.parent, force=True,
                                 ply_override=output_dir / "cleaned_cloud.ply"):
            detail["potree_rebuilt"] = True
        else:
            log("[cleanup] ⚠ the octree was NOT rebuilt — the viewer keeps "
                "showing the removed points until the next build")
    except Exception as e:  # noqa: BLE001
        log(f"[cleanup] octree rebuild failed ({e}) — the viewer is stale")

    log(f"[cleanup] removed {int(drop.sum()):,} point(s) from the cloud "
        f"({n:,} → {int(keep.sum()):,}); {len(detail['clouds'])} cloud file(s) "
        f"and every instance index followed")
    return True, detail


def undo_cleanup(output_dir, undo_path=None, log: Callable[[str], None] = print) -> dict:
    """Put back exactly what the last cleanup removed.

    The rows return to their original positions, so every index that was valid
    before the cleanup is valid again. Instance indices are re-expanded through
    the inverse of the remap.
    """
    from correction.session import read_ply, write_ply
    from segmentation.erase import _write_classification
    from atomic_io import atomic_write_json

    output_dir = Path(output_dir)
    if undo_path is None:
        cands = sorted((output_dir / UNDO_DIR).glob("geometric_cleanup_*.npz"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {"restored": 0, "reason": "no cleanup snapshot to undo"}
        undo_path = cands[-1]
    undo_path = Path(undo_path)
    z = np.load(undo_path, allow_pickle=False)
    removed = np.asarray(z["removed_index"], np.int64)
    n_before = int(z["n_points"])
    n_now = n_before - len(removed)

    back = np.zeros(n_before, bool)
    back[removed] = True
    keep = ~back
    restored = []
    for name in CLOUD_FILES:
        key = f"rows__{name}"
        p = output_dir / name
        if key not in z.files or not p.is_file():
            continue
        header, data = read_ply(p)
        if len(data) != n_now:
            log(f"[cleanup] {name} has {len(data):,} points, expected "
                f"{n_now:,} — it changed since the cleanup, left alone")
            continue
        full = np.empty(n_before, dtype=data.dtype)
        full[keep] = data
        full[back] = z[key].astype(data.dtype, copy=False)
        write_ply(p, header, full)
        restored.append(name)

    remap_back = np.flatnonzero(keep)          # new index → old index
    owner = np.asarray(z["removed_owner"], np.int64) if "removed_owner" in z.files \
        else np.full(len(removed), -1, np.int64)
    res_path = output_dir / "segmentation_result.json"
    if res_path.is_file() and restored:
        doc = json.loads(res_path.read_text())
        instances = doc.get("instances") or []
        touched = []
        for inst in instances:
            iid = int(inst.get("instance_id", inst.get("id", -1)))
            gi = np.asarray(inst.get("globalIndices") or [], np.int64)
            gi = gi[(gi >= 0) & (gi < n_now)]
            # the survivors go back to the indices they had, and the points the
            # cleanup took from THIS instance rejoin it
            back_here = removed[owner == iid]
            full = np.union1d(remap_back[gi], back_here)
            inst["globalIndices"] = full.tolist()
            inst["total_points"] = int(len(full))
            if len(back_here):
                touched.append(inst)
        doc["total_points"] = int(n_before)
        doc["segmented_points"] = sum(int(i.get("total_points") or 0) for i in instances)
        doc["coverage"] = round(doc["segmented_points"] / max(1, n_before), 4)
        _refit_obbs(output_dir, touched, n_before, log=log)
        atomic_write_json(res_path, doc)
        try:
            _write_classification(output_dir, instances, n_before)
        except Exception as e:  # noqa: BLE001
            log(f"[cleanup] classification rebuild failed ({e})")
        try:
            from segmentation.pipeline import rebuild_instance_store
            rebuild_instance_store(output_dir)
        except Exception as e:  # noqa: BLE001
            log(f"[cleanup] instance store rebuild failed ({e})")
        try:
            from potree_converter import convert_ply_to_potree
            convert_ply_to_potree(output_dir.parent, force=True,
                                  ply_override=output_dir / "cleaned_cloud.ply")
        except Exception as e:  # noqa: BLE001
            log(f"[cleanup] octree rebuild failed ({e}) — the viewer still "
                f"shows the cleaned cloud until the next build")

    mark_p = output_dir / MARK_FILE
    if mark_p.is_file():
        try:
            m = np.asarray(np.load(mark_p), bool)
            if len(m) == n_now:
                full_m = np.zeros(n_before, bool)
                full_m[keep] = m
                full_m[back] = True       # they were removed for being out of place
                np.save(mark_p, full_m)
        except Exception:  # noqa: BLE001
            pass

    log(f"[cleanup] restored {len(removed):,} point(s) into "
        f"{', '.join(restored) or 'no cloud'} ({n_now:,} → {n_before:,})")
    return {"restored": int(len(removed)), "clouds": restored,
            "points_after": int(n_before), "snapshot": str(undo_path)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="second moment of the geometric cleanup cycle")
    ap.add_argument("--session", required=True)
    ap.add_argument("--measure-only", action="store_true",
                    help="measure and report without touching the cloud")
    ap.add_argument("--undo", action="store_true",
                    help="restore the last cleanup's removed points")
    args = ap.parse_args(argv)
    out = Path(args.session) / "output"
    if args.undo:
        print(json.dumps(undo_cleanup(out), indent=1))
        return 0
    rep = geometric_cleanup(out, Path(args.session), apply=not args.measure_only)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_instance"},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
