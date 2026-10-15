"""Manhattan marks per object (USER 2026-10-15: "debe indicarlo el usuario, por defecto marcado como
manhattan").

Every object is Manhattan by DEFAULT: its model (reconstruction.surface_fit.manhattan) is snapped to
the scene's orthogonal frame — floor exactly horizontal, walls exactly vertical and exactly along
one of the two horizontal axes, coplanar walls one plane — the straight geometry a BIM wants. The
user unmarks what is not (a sloped ramp, a curved wall): only the EXCEPTIONS are stored, in
``output/manhattan.json``, never in the SAM3 store nor in the projection result (a toggle must not
invalidate the projection stamp or rewrite a 1 GB result). The mark is the user's
(``human_validated``); the deviation of every snapped model from the measurement is recorded next
to the model, never hidden."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

FILE = "manhattan.json"
VERSION = 1


def _path(output_dir) -> Path:
    return Path(output_dir) / FILE


def load(output_dir) -> dict:
    """``{"version": 1, "off": [instance ids NOT Manhattan]}`` — an absent file means every
    object is Manhattan (the default)."""
    p = _path(output_dir)
    if not p.exists():
        return {"version": VERSION, "off": []}
    doc = json.loads(p.read_text())
    off = sorted({int(i) for i in (doc.get("off") or [])})
    return {"version": int(doc.get("version", VERSION)), "off": off}


def off_set(output_dir) -> set:
    return set(load(output_dir)["off"])


def is_manhattan(output_dir, instance_id: int) -> bool:
    return int(instance_id) not in off_set(output_dir)


def flags_for(output_dir, instance_ids: Iterable[int]) -> Dict[int, bool]:
    off = off_set(output_dir)
    return {int(i): int(i) not in off for i in instance_ids}


def set_flag(output_dir, instance_id: int, manhattan: bool) -> dict:
    """Mark / unmark one object; returns the document written."""
    from atomic_io import atomic_write_json
    doc = load(output_dir)
    off = set(doc["off"])
    if bool(manhattan):
        off.discard(int(instance_id))
    else:
        off.add(int(instance_id))
    doc = {"version": VERSION, "off": sorted(off), "provenance": "human_validated"}
    atomic_write_json(_path(output_dir), doc, indent=1, sort_keys=True)
    return doc


def annotate(output_dir, instances: List[dict]) -> List[dict]:
    """Add ``manhattan`` to every instance dict of a listing (keyed by instance_id, else id)."""
    off = off_set(output_dir)
    for inst in instances:
        iid = inst.get("instance_id", inst.get("id"))
        try:
            inst["manhattan"] = int(iid) not in off
        except (TypeError, ValueError):
            inst["manhattan"] = True
    return instances
