"""Geometry epoch — the version number of the session's geometry.

What it persists: ``output/geometry_epoch.json``
``{"epoch": N, "created_at": ..., "correction_id": ..., "parent_epoch": N-1}``.
A session with no file is epoch 0 (the original reconstruction).

What it decides: nothing geometric. It gives every derived artifact a way to
say WHICH geometry it was computed from (``stamp``) and every consumer a way to
refuse or flag geometry from another epoch (``check``). Kept dependency-free
(json + pathlib only) so any server module can import it without cycles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

EPOCH_FILE = "geometry_epoch.json"
LEDGER_FILE = "corrections.jsonl"


def read_epoch(output_dir) -> Dict[str, Any]:
    """The session's current epoch record (epoch 0 when no file exists).
    A corrupt file is an error — it means the last swap was interrupted."""
    p = Path(output_dir) / EPOCH_FILE
    if not p.exists():
        return {"epoch": 0, "created_at": None, "correction_id": None,
                "parent_epoch": None}
    try:
        data = json.loads(p.read_text())
        int(data["epoch"])
        return data
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"corrupt {EPOCH_FILE} at {p}: {e} — the last correction swap "
            f"may have been interrupted; inspect output/_epoch_*/ and the "
            f"ledger before touching the session") from e


def current_epoch(output_dir) -> int:
    return int(read_epoch(output_dir)["epoch"])


def make_epoch_record(epoch: int, correction_id: str,
                      parent_epoch: int) -> Dict[str, Any]:
    return {"epoch": int(epoch),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "correction_id": correction_id,
            "parent_epoch": int(parent_epoch)}


def corrections_summary(output_dir) -> Dict[str, Any]:
    """{approved: N, overridden: [correction_id…]} from the ledger (approved
    runs only; used by stamp())."""
    p = Path(output_dir) / LEDGER_FILE
    approved = 0
    overridden: List[str] = []
    if p.exists():
        runs: Dict[str, Dict[str, Any]] = {}
        verdicts: Dict[str, str] = {}
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") == "run":
                runs[entry["correction_id"]] = entry
            elif entry.get("type") == "verdict":
                verdicts[entry["correction_id"]] = entry["verdict"]
        for cid, run in runs.items():
            if verdicts.get(cid) == "approved":
                approved += 1
                if run.get("overrides"):
                    overridden.append(cid)
    return {"approved": approved, "overridden": overridden}


def stamp(meta: Dict[str, Any], output_dir) -> Dict[str, Any]:
    """Add the provenance trio to a derived artifact's metadata (in place and
    returned): geometry_epoch, human_directed_corrections,
    corrections_overridden. Every measurement/export that leaves the module
    must carry these — a supervision report never hides human intervention."""
    summary = corrections_summary(output_dir)
    meta["geometry_epoch"] = current_epoch(output_dir)
    meta["human_directed_corrections"] = summary["approved"]
    meta["corrections_overridden"] = summary["overridden"]
    return meta


def stamp_nearest(meta: Dict[str, Any], start_dir) -> Dict[str, Any]:
    """``stamp`` for writers that only know their own artifact directory
    (e.g. output/surface_fit/<name>/): walks up the tree to the session's
    output dir (recognised by geometry_epoch.json, cleaned_cloud.ply or
    camera_frames.txt) and stamps against it. A writer outside any session
    tree raises — a deliverable must always declare its geometry epoch."""
    d = Path(start_dir).resolve()
    for _ in range(8):
        if (d / EPOCH_FILE).exists() or (d / "cleaned_cloud.ply").exists() \
                or (d / "camera_frames.txt").exists():
            return stamp(meta, d)
        if d.parent == d:
            break
        d = d.parent
    raise RuntimeError(
        f"stamp_nearest: no session output dir found above {start_dir} — "
        f"cannot stamp the geometry epoch on this artifact")


def check(artifact_meta: Optional[Dict[str, Any]], output_dir) -> Dict[str, Any]:
    """Compare an artifact's stamped epoch with the session's current one.
    Returns {stale: bool, artifact_epoch, current_epoch}. An artifact with no
    stamp predates the epoch system and counts as epoch 0."""
    cur = current_epoch(output_dir)
    art = 0
    if artifact_meta:
        try:
            art = int(artifact_meta.get("geometry_epoch", 0))
        except (TypeError, ValueError):
            art = 0
    return {"stale": art != cur, "artifact_epoch": art, "current_epoch": cur}
