"""ONE OWNER PER SURFACE applied to a scan that is ALREADY segmented (USER 2026-11-18: "hay que
arreglarlo de raíz"; to "do I change your rule of 11-11 to allow this merge?": "1. Si").

The projection now merges the objects that share a surface point by point
(segmentation/interleave.py, ``segmentation.interleave_merge``). This is the same merge for the
scans projected before, WITHOUT running the projection again:

  1. ``segmentation_result.json`` — the survivors take the absorbed objects' points and a box
     refitted on all of them; the absorbed leave ``instances`` and are recorded in ``absorbed`` as
     ``same_surface`` (into which object, the share that decided, their label and points); the
     report goes in ``decisions.same_surface``;
  2. ``segmentation.republish.republish_membership`` — the ONE writer of the per-point siblings
     (``classification.npy``, ``instance_ids.npy``, ``class_map.json``, the census);
  3. the instance store — the absorbed objects leave it, the survivors' points and boxes are
     written again;
  4. the octree — rebuilt by the caller (its stamp carries the classification's hash);
  5. the open viewers — told to load the octree again (apply_min_points.notify_viewers).

No point moves and no point changes from segmented to unsegmented: only WHICH object owns it.
The files of steps 1-3 are copied to ``<output>/_before_same_surface/`` first (``--restore``).

    python -m segmentation.apply_same_surface <output_dir> [--dry] [--restore]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

BACKUP_DIR = "_before_same_surface"
RESULT = "segmentation_result.json"
STORE = "scene_r.db"
BROADCAST = "seg_broadcast.json"


def _files(out: Path):
    from segmentation.republish import CLASS_MAP, CLASSIFICATION, INSTANCE_IDS
    return [out / RESULT, out / CLASSIFICATION, out / INSTANCE_IDS, out / CLASS_MAP, out / STORE, out / BROADCAST]


def _display_cloud(out: Path, result: dict) -> np.ndarray:
    """The cloud the result indexes, in the display frame (float64)."""
    from exporter import cloud as C
    ply = out / str(result.get("cloud_source") or "cleaned_cloud.ply")
    data = C.open_ply(ply if ply.exists() else out / "cleaned_cloud.ply")
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    A, t = C.floor_transform(out)
    return xyz if (np.array_equal(A, np.eye(3)) and not t.any()) else xyz @ A.T + t


def _config() -> dict:
    import yaml
    return yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text()) or {}


def merge(output_dir, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """The merge computed on the scan's result, in memory: (result, report, the survivors that grew)."""
    from segmentation import interleave
    out = Path(output_dir)
    cfg = _config()
    il = interleave.config(cfg)
    res = json.loads((out / RESULT).read_text())
    instances = [i for i in res.get("instances") or [] if i.get("globalIndices")]
    n_points = int(res.get("total_points") or 0)
    if n_points <= 0:
        raise RuntimeError(f"{out / RESULT}: total_points is missing — the cloud's size is not known")
    xyz = _display_cloud(out, res)
    if len(xyz) != n_points:
        raise RuntimeError(f"{out}: the cloud holds {len(xyz):,} points and the result indexes {n_points:,}")
    before = {int(i.get("instance_id", i.get("id"))): len(i["globalIndices"]) for i in instances}
    record: Dict[int, dict] = {}
    report = interleave.merge_instances(instances, xyz, n_points, share=il["share"], neighbours=il["neighbours"],
                                        record=record, log=log)
    grown = [i for i in instances if len(i["globalIndices"]) != before[int(i.get("instance_id", i.get("id")))]]
    return {"result": res, "instances": instances, "record": record, "report": report, "grown": grown, "xyz": xyz,
            "cfg": cfg, "n_points": n_points}


def apply(output_dir, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Apply the merge to the scan in ``output_dir`` (steps 1-3; the caller rebuilds the octree)."""
    from atomic_io import atomic_write_json
    from phase_r.instance_store import InstanceStore, store_on_local_disk
    from segmentation.pipeline import _compute_obb, _write_instance_rows
    from segmentation.republish import republish_membership
    out = Path(output_dir)
    bak = out / BACKUP_DIR
    if bak.exists():
        raise RuntimeError(f"{bak} exists: the merge was already applied here (or a run was interrupted) — "
                           f"restore it or remove it first")
    m = merge(out, log)
    report = m["report"]
    if not report["absorbed"]:
        log(f"[same-surface] {out}: no two objects share a surface — nothing to do")
        return {**report, "changed": False}
    # the survivors' boxes FIRST (they are fitted on the card: a failure here must leave nothing written)
    res, xyz = m["result"], m["xyz"]
    for inst in m["grown"]:
        inst["obb"] = _compute_obb(xyz[np.asarray(inst["globalIndices"], dtype=np.int64)], cfg=m["cfg"])
    bak.mkdir()
    for f in _files(out):
        if f.exists():
            shutil.copy2(f, bak / f.name)
    log(f"[same-surface] {out}: backup of {len(list(bak.iterdir()))} file(s) in {bak.name}/")

    # ── 1. the result ─────────────────────────────────────────────────────────────────────────
    res["instances"] = m["instances"]
    absorbed = res.setdefault("absorbed", {})
    for loser, how in m["record"].items():
        absorbed[str(loser)] = how
    res.setdefault("decisions", {})["same_surface"] = report
    atomic_write_json(out / RESULT, res)
    keep_ids = [int(i.get("instance_id", i.get("id"))) for i in res["instances"]]

    # ── 2. the per-point siblings, by their one writer ────────────────────────────────────────
    rep = republish_membership(out, n_points=m["n_points"], source_dir=out, log=log)

    # ── 3. the instance store: the absorbed leave, the survivors' points and boxes are rewritten
    removed = None
    if (out / STORE).exists():
        with store_on_local_disk(out / STORE, keep_existing=True) as local:
            store = InstanceStore(local)
            removed = store.reconcile(keep_ids)
            _write_instance_rows(store, m["grown"], xyz)          # (closes the store)
        log(f"[same-surface] store: {removed['removed']} object(s) removed, {len(m['grown'])} rewritten")
    out_report = {**report, "changed": True, "kept": len(keep_ids), "segmented_points": rep["classified_points"],
                  "store_removed": (removed or {}).get("removed"), "survivors_rewritten": len(m["grown"]),
                  "backup": str(bak)}
    log(f"[same-surface] {out}: {report['objects_before']} → {report['objects_after']} objects")
    return out_report


def restore(output_dir, log: Callable[[str], None] = print) -> int:
    """Put back the files a previous ``apply`` saved; the octree is then rebuilt by the caller."""
    out = Path(output_dir)
    bak = out / BACKUP_DIR
    if not bak.is_dir():
        raise RuntimeError(f"{bak} does not exist: nothing to restore")
    n = 0
    for f in sorted(bak.iterdir()):
        shutil.copy2(f, out / f.name)
        n += 1
    for stale in (out / (STORE + "-wal"), out / (STORE + "-shm")):
        if stale.exists():
            stale.unlink()
    shutil.rmtree(bak)
    log(f"[same-surface] {out}: {n} file(s) restored")
    return n


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    if "--restore" in sys.argv:
        restore(args[0])
    elif "--dry" in sys.argv:
        r = merge(args[0])
        labels: Dict[str, int] = {}
        for how in r["record"].values():
            key = f"{how['label']} → {how['into_label']}"
            labels[key] = labels.get(key, 0) + 1
        print(json.dumps({**r["report"], "absorbed_by_label": dict(sorted(labels.items(), key=lambda kv: -kv[1])[:25])},
                         indent=1, ensure_ascii=False))
    else:
        print(json.dumps(apply(args[0]), indent=1))
