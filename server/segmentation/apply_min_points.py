"""The minimum object size applied to a scan that is ALREADY segmented (USER 2026-11-13: "en las
listas de segmentaciones aparecen muchos elementos con 1 o muy pocos puntos segmentados" / "Quién
dijo que lo estructural nunca se exime").

From 2026-10-15 to 2026-11-13 the projection kept every STRUCTURAL instance under
``min_instance_points`` (its own reading of ``structural_always``): doors, beams and pipes of one
point sat in the lists of every session. The projection no longer does (segmentation/pipeline.py).
This is the same drop for the scans projected in between, WITHOUT running the projection again:

  1. ``segmentation_result.json`` — every instance under the minimum leaves ``instances`` and is
     recorded in ``absorbed`` as ``too_small`` with its points, the minimum and its margin: the
     record the projection itself writes, the one the list endpoint hides by;
  2. ``segmentation.republish.republish_membership`` — the ONE writer of the per-point siblings:
     the census (segmented points, coverage), ``classification.npy``, ``instance_ids.npy``,
     ``class_map.json`` (and ``seg_broadcast.json`` when the scan has one). The points of a dropped
     instance become unsegmented, as they are after a projection;
  3. the instance store — ``InstanceStore.reconcile``: the chat and the reports stop naming them;
  4. the octree — rebuilt by the caller (``potree_converter.convert_ply_to_potree``: its stamp
     carries the classification's hash, so the published one no longer matches).

Nothing else changes: no point moves, no instance above the minimum is touched, the masks and the
fusion map stay as provenance. The files of steps 1-3 are copied to ``<output>/_before_min_points/``
first (``restore`` puts them back).

    python -m segmentation.apply_min_points <output_dir> [--dry] [--restore]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

BACKUP_DIR = "_before_min_points"
RESULT = "segmentation_result.json"
STORE = "scene_r.db"
BROADCAST = "seg_broadcast.json"


def _min_points() -> int:
    """``segmentation.min_instance_points`` of config.yaml (wherever the section nests it); a
    missing key fails naming it."""
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    full = yaml.safe_load(cfg_path.read_text()) or {}

    def find(node: Any):
        if isinstance(node, dict):
            if "min_instance_points" in node:
                return node["min_instance_points"]
            for v in node.values():
                got = find(v)
                if got is not None:
                    return got
        return None
    v = find(full)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise RuntimeError("config.yaml: min_instance_points is missing or not a non-negative integer")
    return int(v)


def _files(out: Path) -> List[Path]:
    from segmentation.republish import CLASS_MAP, CLASSIFICATION, INSTANCE_IDS
    return [out / RESULT, out / CLASSIFICATION, out / INSTANCE_IDS, out / CLASS_MAP, out / STORE,
            out / BROADCAST]


def plan(output_dir, min_points: int | None = None) -> Dict[str, Any]:
    """What applying the minimum would drop — nothing is written."""
    out = Path(output_dir)
    mn = _min_points() if min_points is None else int(min_points)
    res = json.loads((out / RESULT).read_text())
    insts = res.get("instances") or []
    tiny = [i for i in insts if len(i.get("globalIndices") or ()) < mn]
    labels: Dict[str, int] = {}
    for t in tiny:
        labels[str(t.get("label"))] = labels.get(str(t.get("label")), 0) + 1
    return {"output_dir": str(out), "min_points": mn, "instances": len(insts), "drop": len(tiny),
            "drop_points": sum(len(t.get("globalIndices") or ()) for t in tiny),
            "drop_by_label": dict(sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "total_points": int(res.get("total_points") or 0), "_result": res, "_tiny": tiny}


def apply(output_dir, min_points: int | None = None,
          log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Apply the minimum to the scan in ``output_dir`` (steps 1-3 of the module docstring; the
    caller rebuilds the octree). Returns the report; ``changed`` False when nothing was under it."""
    from atomic_io import atomic_write_json
    from phase_r.instance_store import InstanceStore, store_on_local_disk
    from segmentation.republish import republish_membership
    out = Path(output_dir)
    p = plan(out, min_points)
    res, tiny, mn = p.pop("_result"), p.pop("_tiny"), p["min_points"]
    if not tiny:
        log(f"[min-points] {out}: no instance under {mn} points — nothing to do")
        return {**p, "changed": False}
    n_points = p["total_points"]
    if n_points <= 0:
        raise RuntimeError(f"{out / RESULT}: total_points is missing — the cloud's size is not known")

    # ── the backup: what steps 1-3 write, as it is now ────────────────────────────────────────
    bak = out / BACKUP_DIR
    if bak.exists():
        raise RuntimeError(f"{bak} exists: the minimum was already applied here (or a run was "
                           f"interrupted) — restore it or remove it first")
    bak.mkdir()
    for f in _files(out):
        if f.exists():
            shutil.copy2(f, bak / f.name)
    log(f"[min-points] {out}: backup of {len(list(bak.iterdir()))} file(s) in {bak.name}/")

    # ── 1. the result: out of `instances`, into `absorbed` as too_small ───────────────────────
    absorbed = res.setdefault("absorbed", {})
    drop = set()
    for t in tiny:
        iid = int(t.get("instance_id", t.get("id")))
        n = len(t.get("globalIndices") or ())
        absorbed[str(iid)] = {"into": None, "into_label": None, "reason": "too_small",
                              "label": t.get("label"), "points": n, "min_points": mn,
                              "margin": n - mn}
        drop.add(id(t))
    res["instances"] = [i for i in res["instances"] if id(i) not in drop]
    for i in res["instances"]:
        i.pop("kept_structural_under_min_points", None)
    atomic_write_json(out / RESULT, res)                       # no indent: the C encoder
    keep_ids = [int(i.get("instance_id", i.get("id"))) for i in res["instances"]]
    del res

    # ── 2. the per-point siblings, by their one writer ────────────────────────────────────────
    rep = republish_membership(out, n_points=n_points, source_dir=out, log=log)

    # ── 3. the instance store ─────────────────────────────────────────────────────────────────
    removed = None
    if (out / STORE).exists():
        with store_on_local_disk(out / STORE, keep_existing=True) as local:
            store = InstanceStore(local)
            try:
                removed = store.reconcile(keep_ids)
            finally:
                store.close()
        log(f"[min-points] store: {removed['removed']} instance(s) removed, {removed['kept']} kept")
    report = {**p, "changed": True, "kept": len(keep_ids), "segmented_points": rep["classified_points"],
              "store_removed": (removed or {}).get("removed"), "backup": str(bak)}
    log(f"[min-points] {out}: {p['drop']} instance(s) under {mn} points dropped "
        f"({p['drop_points']:,} points), {len(keep_ids)} kept")
    return report


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
    log(f"[min-points] {out}: {n} file(s) restored")
    return n


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    if "--restore" in sys.argv:
        restore(args[0])
    elif "--dry" in sys.argv:
        r = plan(args[0])
        r.pop("_result"), r.pop("_tiny")
        print(json.dumps(r, indent=1))
    else:
        print(json.dumps(apply(args[0]), indent=1))
