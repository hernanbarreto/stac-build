"""The fusion of masklets into objects — kept BESIDE the raw SAM3 store, never inside it.

USER 2026-09-20: *"la etapa de fusion de instancias es sumamente importante
porque justamente evita estas cosas, que dos instancias con nombre diferente
que son lo mismo sean tratadas como objetos independientes ... cuando se hace
todo el analisis de la certificacion, y la correccion, se hace sobre las
instancias fusionadas y no sobre 'las partes'"*.

Until 2026-10-08 the fusion REWROTE the parent: it OR-ed each part's masks into
its survivor's oid inside `seg_masks.npz`, dropped the parts from
`segmentation.json`, archived the previous pair under `_sam3_raw/gen_NNN` and
counted rounds. The projection then read the rewritten store on its next pass,
so a second projection of the same session gave another result (the survivor's
OR-ed masks are larger and lose the smaller-area contests), the rounds and the
generation names counted how often the session had been projected, and
`fusion_map.json` carried the wall clock (docs/plan_determinismo.md points 100,
101, 118).

NOW (point 100): the raw store — `segmentation.json` + `seg_masks.npz`, what SAM3
wrote (plus what the interactive manager adds to it) — is IMMUTABLE for the
fusion. The fusion is a separate artifact, `fusion_map.json`, tied to the sha256
of the two raw files, written as a pure function of (raw store, projection
result): no rounds, no archive, no clock. Every reader that measures OBJECTS
applies it over the raw masks through :func:`fused_parent` (the instance list
with the parts folded into their survivors) and :class:`FusedMasks` (the mask
store as the fused objects see it — a survivor's mask in a frame is the OR of
its own and its parts'). The projection itself never reads the map: it derives
the merges from the raw masks each time, which is what makes it a pure function
of its inputs (point 99).

THE SURVIVOR KEEPS BOTH OF ITS IDS. A fused object is the surviving masklet,
absorbing its parts' masks while keeping its own `id` (the npz obj id) and
`instance_id`; nothing is renumbered, so `instance_id == id + 1` holds exactly
where it held before. The map is REMOVE-ONLY in id-space: a mask propagated
after the matching appears in no record, so the fused view keeps it untouched.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

PARENT = "segmentation.json"
MASKS = "seg_masks.npz"
MAP = "fusion_map.json"
#: the archive directory the fusion used to write generations into (point 100: no longer
#: written; the name is kept so an old session's leftover can be recognised and ignored)
ARCHIVE = "_sam3_raw"
MAP_VERSION = 2

# the merges that mean "these masks are ONE object". `too_small`, `unmatched`
# and `fused_or_unmatched` carry `into: None` — they are not part of anything
# and their masks stay in the parent, where the user can still re-propagate
# them.
FUSING = ("fragment", "space_dedupe", "overlap_dedupe")

_MASK_KEY_RE = re.compile(r"^f(\d+)_o(\d+)$")


class FusionStale(RuntimeError):
    """`fusion_map.json` was written for another raw store (another sha256 of
    `segmentation.json` / `seg_masks.npz`): it is not applied — re-project."""


def _iid(entry: dict) -> Optional[int]:
    v = entry.get("instance_id", entry.get("id"))
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def plan_fusion(parent: dict, absorbed: dict) -> Dict[int, dict]:
    """{survivor_iid: {"oid", "parts": [...]}} — what to fold into what.

    Raises when a record names a target the parent does not have: a fusion is
    all or nothing, and a half-applied one loses masks with no way back.
    """
    live = {}
    for e in parent.get("instances") or []:
        i = _iid(e)
        if i is not None:
            live[i] = e
    out: Dict[int, dict] = {}
    for k, rec in (absorbed or {}).items():
        try:
            part = int(k)
        except (TypeError, ValueError):
            continue
        into = rec.get("into")
        if into is None or str(rec.get("reason")) not in FUSING:
            continue
        if part not in live:
            continue                      # not a masklet of this parent (already gone)
        into = int(into)
        if into not in live:
            raise ValueError(
                f"the record folds masklet {part} into {into}, which the "
                f"parent does not have — the fusion is refused rather than "
                f"applied half way")
        if into == part:
            continue
        e = live[part]
        out.setdefault(into, {"oid": int(live[into].get("id", into - 1)),
                              "parts": []})
        out[into]["parts"].append({
            "id": int(e.get("id", part - 1)), "instance_id": part,
            "label": e.get("label"), "reason": str(rec.get("reason"))})
    for v in out.values():
        v["parts"].sort(key=lambda p: p["instance_id"])
    return dict(sorted(out.items()))


# ── the raw store's identity ─────────────────────────────────────────────

def parent_identity(output_dir) -> Dict[str, Optional[str]]:
    """sha256 of the two raw files the map is tied to (None for an absent one)."""
    from repro import sha256_file
    out = Path(output_dir)
    return {"segmentation_json_sha256": (sha256_file(out / PARENT) if (out / PARENT).exists()
                                         else None),
            "seg_masks_sha256": (sha256_file(out / MASKS) if (out / MASKS).exists() else None)}


def _canonical_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ── writing the map ──────────────────────────────────────────────────────

def apply_fusion(output_dir, result: dict, *,
                 log: Callable[[str], None] = print) -> dict:
    """Write the matcher's verdict as `fusion_map.json` beside the raw store.
    Returns `result` unchanged. The parent files are never touched.

    The map is a pure function of (raw store, `result["absorbed"]`): the same
    inputs write the same bytes, and a file that already holds them is left
    alone (its mtime included). With nothing to fold there is no map — a
    leftover from another store is removed, never left to be applied.
    """
    output_dir = Path(output_dir)
    parent_p = output_dir / PARENT
    map_p = output_dir / MAP
    if not parent_p.exists():
        return result
    parent = json.loads(parent_p.read_text())
    plan = plan_fusion(parent, result.get("absorbed") or {})
    if not plan:
        if map_p.exists():
            map_p.unlink()
            log("  fusion: nothing to fold — the previous fusion_map.json removed")
        return result

    part_ids = {p["instance_id"] for v in plan.values() for p in v["parts"]}
    by_reason: Dict[str, int] = {}
    for v in plan.values():
        for p in v["parts"]:
            by_reason[p["reason"]] = by_reason.get(p["reason"], 0) + 1
    n_parent = len(parent.get("instances") or [])
    doc = {
        "version": MAP_VERSION,
        "parent": parent_identity(output_dir),
        "map": {str(k): {"oid": v["oid"], "parts": v["parts"]} for k, v in plan.items()},
        "applied": len(part_ids),
        "by_reason": dict(sorted(by_reason.items())),
        "masks_total": int(parent.get("masks_total") or n_parent),
        "objects": n_parent - len(part_ids),
        "provenance": "tool_measured",
    }
    try:                                   # point 118: the reconstruction's identity, not a clock
        from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
        doc[RECONSTRUCTION_ID_KEY] = reconstruction_id_or_none(output_dir)
    except Exception:  # noqa: BLE001 — a session with no epoch machinery
        pass
    data = _canonical_bytes(doc)
    if map_p.exists() and map_p.read_bytes() == data:
        log(f"  fusion: fusion_map.json already holds this verdict ({len(part_ids)} part(s)) — unchanged")
        return result
    _atomic_bytes(map_p, data)
    log(f"  fusion: {n_parent} masklet(s) -> {doc['objects']} object(s), {len(part_ids)} "
        f"part(s) folded ({by_reason}) — fusion_map.json, the raw store untouched")
    return result


# ── reading the map ──────────────────────────────────────────────────────

def load_map(output_dir) -> Optional[dict]:
    """The session's fusion map, VERIFIED against the raw store it was written
    for — None when there is none, :class:`FusionStale` when the raw files
    changed since (a re-projection rewrites it)."""
    out = Path(output_dir)
    p = out / MAP
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict) or "map" not in doc or int(doc.get("version") or 0) < MAP_VERSION:
        raise FusionStale(f"{p} was written by the pre-2026-10-08 fusion (it rewrote the parent) "
                          f"— re-project the masks to write the map beside the raw store")
    now = parent_identity(out)
    saved = doc.get("parent") or {}
    diffs = [k for k in now if saved.get(k) != now[k]]
    if diffs:
        raise FusionStale(f"{p} is tied to another raw store ({', '.join(diffs)} changed) — "
                          f"re-project the masks to refresh it")
    return doc


def fused_entries(parent: dict, plan: Dict[int, dict]) -> List[dict]:
    """The parent's instance list with ``plan`` applied: parts removed, each
    survivor carrying its ``parts`` (pure; the parent is not modified)."""
    part_ids = {p["instance_id"] for v in plan.values() for p in v["parts"]}
    out: List[dict] = []
    for e in parent.get("instances") or []:
        i = _iid(e)
        if i in part_ids:
            continue
        e = dict(e)
        if i in plan:
            e["parts"] = list(e.get("parts") or []) + [dict(p) for p in plan[i]["parts"]]
        out.append(e)
    return out


def fused_parent(output_dir) -> dict:
    """`segmentation.json` as the fused OBJECTS: the map applied over the raw
    instance list. Without a map, the raw parent itself. The document carries
    ``fusion`` = {"applied", "map"} so a reader can say what it measured."""
    out = Path(output_dir)
    parent = json.loads((out / PARENT).read_text())
    doc = load_map(out)
    if doc is None:
        parent["fusion"] = {"applied": 0, "map": None}
        return parent
    plan = {int(k): {"oid": int(v["oid"]), "parts": list(v["parts"])}
            for k, v in doc["map"].items()}
    parent = dict(parent)
    parent["instances"] = fused_entries(parent, plan)
    parent["fusion"] = {"applied": int(doc.get("applied") or 0), "map": MAP}
    return parent


class FusedMasks:
    """The mask store as the fused objects see it, over the raw `seg_masks.npz`:
    a retired part's keys disappear, and a survivor's mask in a frame is the OR
    of its own mask and its parts' (masks of different shapes are refused: they
    cannot be one object's). ``files`` / ``__getitem__`` / ``__contains__`` mimic
    the ``np.load`` handle every reader already holds, so the swap is one line.
    """

    def __init__(self, output_dir, masks=None):
        self.dir = Path(output_dir)
        self._own = masks is None
        self.raw = np.load(self.dir / MASKS) if masks is None else masks
        doc = load_map(self.dir)
        self.root_of: Dict[int, int] = {}            # part oid -> survivor oid
        self.parts_of: Dict[int, List[int]] = {}     # survivor oid -> part oids
        if doc is not None:
            for v in doc["map"].values():
                for p in v["parts"]:
                    self.root_of[int(p["id"])] = int(v["oid"])
                    self.parts_of.setdefault(int(v["oid"]), []).append(int(p["id"]))
        self._frames: Dict[int, Dict[int, List[str]]] = {}   # fused oid -> frame -> raw keys
        self.meta: List[str] = []
        for k in self.raw.files:
            m = _MASK_KEY_RE.match(k)
            if not m:
                self.meta.append(k)
                continue
            f, o = int(m.group(1)), int(m.group(2))
            self._frames.setdefault(self.root_of.get(o, o), {}).setdefault(f, []).append(k)
        self.files: List[str] = sorted(self.meta) + [
            f"f{f}_o{o}" for o in sorted(self._frames) for f in sorted(self._frames[o])]
        self._set = set(self.files)

    def __contains__(self, key) -> bool:
        return str(key) in self._set

    def __iter__(self) -> Iterator[str]:
        return iter(self.files)

    def frames_for(self, oid: int) -> List[Tuple[int, str]]:
        """[(frame, key)] of a FUSED object, sorted by frame."""
        return [(f, f"f{f}_o{int(oid)}") for f in sorted(self._frames.get(int(oid), {}))]

    def __getitem__(self, key):
        key = str(key)
        m = _MASK_KEY_RE.match(key)
        if not m:
            if key == "obj_ids" and "obj_ids" in self.raw.files:
                ids = [int(v) for v in np.asarray(self.raw["obj_ids"]).ravel()]
                return np.asarray(sorted(set(ids) - set(self.root_of)), np.int32)
            return self.raw[key]
        f, o = int(m.group(1)), int(m.group(2))
        keys = self._frames.get(o, {}).get(f)
        if not keys:
            raise KeyError(f"{key} is not a mask of the fused store")
        acc = None
        for k in sorted(keys):
            a = np.asarray(self.raw[k])
            if acc is None:
                acc = a
            elif a.shape != acc.shape:
                raise ValueError(f"mask {k} is {a.shape} and {sorted(keys)[0]} is {acc.shape} — "
                                 f"they cannot be one object's mask")
            else:
                acc = np.maximum(acc, a)
        return acc

    def close(self) -> None:
        if self._own:
            try:
                self.raw.close()
            except Exception:  # noqa: BLE001
                pass


def _atomic_json(path: Path, doc: dict) -> None:
    _atomic_bytes(path, _canonical_bytes(doc))


def high_water(output_dir) -> Tuple[int, int]:
    """(id, instance_id) ever allocated in this session — the parent's own
    (the raw store keeps every masklet, fused or not, so nothing retires)."""
    p = Path(output_dir) / PARENT
    if not p.exists():
        return 0, 0
    try:
        doc = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return 0, 0
    ins = doc.get("instances") or []
    hi_id = max([int(doc.get("id_high_water") or 0)]
                + [int(e.get("id", 0)) for e in ins]
                + [int(p_["id"]) for e in ins for p_ in (e.get("parts") or [])])
    hi_iid = max([int(doc.get("instance_id_high_water") or 0)]
                 + [i for i in (_iid(e) for e in ins) if i is not None]
                 + [int(p_["instance_id"]) for e in ins
                    for p_ in (e.get("parts") or [])])
    return hi_id, hi_iid


def oids_of(entry: dict) -> List[int]:
    """Every npz obj id a FUSED entry owns: its own, plus the parts it absorbed.

    `reconstruction/loops` splits an instance only when it has more than one
    mask id, and after the fusion a survivor's parts ARE those ids.
    """
    out = [int(entry.get("id", 0))]
    out += [int(p["id"]) for p in (entry.get("parts") or [])]
    return sorted(set(out))
