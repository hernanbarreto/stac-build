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
from typing import Any, Dict, List, Optional, Tuple

EPOCH_FILE = "geometry_epoch.json"
# a stored epoch lives in output/_epoch_<N>/ (correction.apply re-exports this)
EPOCH_DIR_PREFIX = "_epoch_"
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


def epoch_lineage(output_dir, epoch: int,
                  live: Optional[int] = None) -> List[int]:
    """The ancestry of ``epoch``, root first — [0, …, epoch].

    Each state carries its own ``geometry_epoch.json`` (the live one in the
    session, a stored one inside ``_epoch_<N>/``), and that record names the
    ``parent_epoch`` the correction ran on top of. A session that never
    branched gives the plain [0, 1, …, N]; one where a correction ran on top of
    a re-selected older epoch gives the real line.
    """
    output_dir = Path(output_dir)
    live = current_epoch(output_dir) if live is None else int(live)
    chain: List[int] = []
    e: Optional[int] = int(epoch)
    seen = set()
    while e is not None and e not in seen:
        seen.add(e)
        chain.append(e)
        if e == 0:
            break
        rec_p = ((output_dir / EPOCH_FILE) if e == live
                 else (output_dir / f"{EPOCH_DIR_PREFIX}{e}" / EPOCH_FILE))
        parent: Optional[int] = e - 1          # a session written before the
        if rec_p.is_file():                    # parent was recorded is linear
            try:
                data = json.loads(rec_p.read_text())
                if data.get("parent_epoch") is not None:
                    parent = int(data["parent_epoch"])
            except (OSError, ValueError, TypeError):
                pass
        e = parent
    return list(reversed(chain))


def epoch_path(output_dir, frm: int, to: int) -> List[Tuple[int, bool]]:
    """The edges to travel from epoch ``frm`` to epoch ``to``, in order.

    Each edge is ``(epoch, inverse)``: the transform stored as
    ``corrections/epoch_<epoch>.npz``, applied inverted while climbing from
    ``frm`` to the common ancestor and forward while descending to ``to``.
    On a session that never branched this is exactly the old arithmetic
    (cur…to+1 inverted, or cur+1…to forward).
    """
    live = current_epoch(output_dir)
    a = epoch_lineage(output_dir, frm, live)
    b = epoch_lineage(output_dir, to, live)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    up = [(e, True) for e in reversed(a[i:])]     # undo frm → ancestor
    down = [(e, False) for e in b[i:]]            # apply ancestor → to
    return up + down


def corrections_summary(output_dir) -> Dict[str, Any]:
    """{applied: N, overridden: [correction_id…]} from the ledger.

    It used to count only the runs whose verdict was "approved". There is no
    approving any more (USER 2026-09-16: every epoch stays on disk and is
    selected, never approved or undone), so what a derived artifact needs to
    know is how many corrections REACHED the session — every run that was
    applied, which is every run the ledger holds with an epoch of its own.
    """
    p = Path(output_dir) / LEDGER_FILE
    applied = 0
    overridden: List[str] = []
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") != "run":
                continue
            if entry.get("verdict") == "rejected":
                continue          # never touched a byte of the session
            applied += 1
            if entry.get("overrides"):
                overridden.append(entry["correction_id"])
    return {"applied": applied, "approved": applied, "overridden": overridden}


def stamp(meta: Dict[str, Any], output_dir) -> Dict[str, Any]:
    """Add the provenance trio to a derived artifact's metadata (in place and
    returned): geometry_epoch, human_directed_corrections,
    corrections_overridden. Every measurement/export that leaves the module
    must carry these — a supervision report never hides human intervention."""
    summary = corrections_summary(output_dir)
    meta["geometry_epoch"] = current_epoch(output_dir)
    meta["human_directed_corrections"] = summary["applied"]
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
