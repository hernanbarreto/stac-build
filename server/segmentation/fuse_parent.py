"""The fusion becomes real in the parent.

USER 2026-09-20: *"la etapa de fusion de instancias es sumamente importante
porque justamente evita estas cosas, que dos instancias con nombre diferente
que son lo mismo sean tratadas como objetos independientes, entonces lo ideal
es que incluso se modificara con la fusion el padre de todos, el
segmentation.json, y el npz, porque es la fusion real, de lo que estamos
viendo, y ahi, cuando se hace todo el analisis de la certificacion, y la
correccion, se hace sobre las instancias fusionadas y no sobre 'las partes'"*.

Until now the matcher decided which masks are one object and wrote that only
into `segmentation_result.json`. The PARENT — `segmentation.json` and
`seg_masks.npz`, which every downstream measurement reads — kept the parts. So
the correction measured pieces: on pccr `wooden_desk#230` (69,608 pts, kf
199-215) read as ONE visit and died at the "2+ visits" gate, along with 149 of
270 masklets, while its second copy sat 0.64 m away as a DIFFERENT masklet,
`#226` (kf 0-9), which the matcher had already absorbed into it as a fragment.
Fused, the same object gives visits kf 0-12 and kf 199-215, shares 34.1 % /
65.9 %, and a closure of 68.9 cm with 3.0 cm of disagreement against a 9.5 cm
bar — determined, over 17.17 m of walk, the longest lever arm in the session.

THE SURVIVOR KEEPS BOTH OF ITS IDS. A fused object is not a new object with a
new number: it is the surviving masklet, absorbing its parts' masks and their
provenance while keeping its own `id` (the npz obj id) and `instance_id`.
Nothing is renumbered anywhere, so `instance_id == id + 1` still holds exactly
where it held before, and the fusion is a pure DELETION in id-space: every id
that remains means what it meant yesterday, and every id that disappears takes
its npz keys with it in the same pass.

The rewrite is REMOVE-ONLY and driven by the record, never by the survivor
list. A mask propagated after the last matching appears in neither, so it is
kept untouched — building the parent from the survivors instead would silently
delete the user's work. That is the single most important invariant here.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PARENT = "segmentation.json"
MASKS = "seg_masks.npz"
MAP = "fusion_map.json"
ARCHIVE = "_sam3_raw"

# the merges that mean "these masks are ONE object". `too_small`, `unmatched`
# and `fused_or_unmatched` carry `into: None` — they are not part of anything
# and their masks stay in the parent, where the user can still re-propagate
# them.
FUSING = ("fragment", "space_dedupe", "overlap_dedupe")


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
            continue                      # already fused — this is idempotence
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
    return out


def _archive(output_dir: Path, log: Callable[[str], None]) -> str:
    """A copy of the PRE-fusion pair. Generations, not one snapshot: a session
    keeps producing raw masklets after round 1 and `gen_000` alone would not
    hold their masks once a later round fuses them."""
    base = output_dir / ARCHIVE
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    while (base / f"gen_{n:03d}").exists():
        n += 1
    gen = base / f"gen_{n:03d}"
    gen.mkdir()
    for f in (PARENT, MASKS):
        if (output_dir / f).exists():
            shutil.copy2(output_dir / f, gen / f)
    rel = f"{ARCHIVE}/gen_{n:03d}"
    log(f"  fusion: raw SAM3 output archived in {rel}")
    return rel


def _rewrite_masks(output_dir: Path, plan: Dict[int, dict],
                   log: Callable[[str], None]) -> Tuple[int, int]:
    """Re-key the mask store: every part's masks OR-ed into its survivor's oid.

    Written MEMBER BY MEMBER. The pccr store is 6.3 MB on disk and 1.87 GB
    uncompressed, so `dict(np.load(...))` + `savez_compressed` would ask for
    ~1.9 GB in one go; here the peak is a handful of 832x464 masks.
    """
    src = output_dir / MASKS
    if not src.exists():
        return 0, 0
    z = np.load(src)
    keys = set(z.files)

    root_of: Dict[int, int] = {}          # part oid -> survivor oid
    for surv in plan.values():
        for p in surv["parts"]:
            root_of[int(p["id"])] = int(surv["oid"])

    # which oids feed each survivor, and which frames each of them has
    groups: Dict[int, List[int]] = {}
    for part_oid, surv_oid in root_of.items():
        groups.setdefault(surv_oid, []).append(part_oid)
    by_oid: Dict[int, Dict[int, str]] = {}
    for k in keys:
        if not k.startswith("f") or "_o" not in k:
            continue
        try:
            f_s, o_s = k[1:].split("_o", 1)
            f, o = int(f_s), int(o_s)
        except ValueError:
            continue
        by_oid.setdefault(o, {})[f] = k

    space = z[  # copied VERBATIM: losing it makes every later reader fall back
        "mask_frame_space"] if "mask_frame_space" in keys else None
    res = z["scaled_res"] if "scaled_res" in keys else None
    frames = z["frames"] if "frames" in keys else None
    obj_ids = (set(int(v) for v in z["obj_ids"]) if "obj_ids" in keys
               else set(by_oid))
    obj_ids -= set(root_of)

    tmp = Path(tempfile.mkstemp(dir=str(output_dir), suffix=".tmp.npz")[1])
    written = merged = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        def put(name: str, arr: np.ndarray):
            with zf.open(name + ".npy", "w") as fh:
                np.lib.format.write_array(fh, np.asarray(arr),
                                          allow_pickle=False)

        for o, frames_of in by_oid.items():
            if o in root_of:
                continue                          # emitted with its survivor
            parts = groups.get(o, [])
            if not parts:
                for f, k in frames_of.items():
                    put(k, z[k]); written += 1
                continue
            union: Dict[int, List[str]] = {f: [k] for f, k in frames_of.items()}
            for po in parts:
                for f, k in by_oid.get(po, {}).items():
                    union.setdefault(f, []).append(k)
            for f, ks in union.items():
                if len(ks) == 1:
                    put(f"f{f}_o{o}", z[ks[0]])
                else:
                    a = z[ks[0]]
                    for k in ks[1:]:
                        b = z[k]
                        if b.shape != a.shape:
                            raise ValueError(
                                f"mask {k} is {b.shape} and {ks[0]} is "
                                f"{a.shape} — they cannot be one object's mask")
                        a = np.maximum(a, b)
                    put(f"f{f}_o{o}", a)
                    merged += len(ks) - 1
                written += 1
        if obj_ids:
            put("obj_ids", np.asarray(sorted(obj_ids), np.int32))
        if frames is not None:
            put("frames", frames)
        if res is not None:
            put("scaled_res", res)
        if space is not None:
            put("mask_frame_space", space)
    z.close()
    os.replace(tmp, src)
    from segmentation import mask_space
    mask_space.invalidate(output_dir)
    log(f"  fusion: mask store re-keyed — {written} mask(s), "
        f"{merged} OR-ed into a survivor, {len(root_of)} obj id(s) retired")
    return written, merged


def apply_fusion(output_dir, result: dict, *,
                 log: Callable[[str], None] = print) -> dict:
    """Fold the matcher's verdict into the parent. Returns `result` unchanged.

    Write order is part of the contract: archive → masks → parent, and the
    caller writes `segmentation_result.json` LAST. The pipeline decides whether
    the mask→cloud mapping is up to date by comparing mtimes, so the result
    must end up the newest of the three. Masks before the parent because a
    crash between them leaves entries whose oids lost their keys, which the
    next pass repairs — the reverse order would drop the parts' pixels forever.
    """
    output_dir = Path(output_dir)
    parent_p = output_dir / PARENT
    if not parent_p.exists():
        return result
    parent = json.loads(parent_p.read_text())
    plan = plan_fusion(parent, result.get("absorbed") or {})
    if not plan:
        # NOT an optimisation: rewriting the parent bumps its mtime and the
        # pipeline reads that as "the mask→cloud mapping is pending again"
        return result

    raw_ids = [_iid(e) for e in (parent.get("instances") or [])]
    raw_oids = [int(e.get("id", 0)) for e in (parent.get("instances") or [])]
    masks_total = int(parent.get("masks_total")
                      or len(parent.get("instances") or []))

    archive = _archive(output_dir, log)
    _rewrite_masks(output_dir, plan, log)

    part_ids = {p["instance_id"] for v in plan.values() for p in v["parts"]}
    rounds = list((parent.get("fusion") or {}).get("rounds_log") or [])
    round_no = len(rounds) + 1
    kept: List[dict] = []
    for e in parent.get("instances") or []:
        i = _iid(e)
        if i in part_ids:
            continue                                   # folded into its object
        e = dict(e)
        if i in plan:
            # append-only: a part absorbed in an earlier round keeps its round
            e["parts"] = list(e.get("parts") or []) + plan[i]["parts"]
            e["fused_round"] = round_no
        kept.append(e)

    doc = dict(parent)
    doc["instances"] = kept
    doc["masks_total"] = masks_total
    doc["id_high_water"] = max([int(parent.get("id_high_water") or 0)]
                               + [o for o in raw_oids])
    doc["instance_id_high_water"] = max(
        [int(parent.get("instance_id_high_water") or 0)]
        + [i for i in raw_ids if i is not None])
    doc["fusion"] = {"rounds": round_no, "archive": archive,
                     "applied": len(part_ids),
                     "rounds_log": rounds + [{"round": round_no,
                                              "archive": archive,
                                              "applied": len(part_ids)}]}
    try:
        from correction.epoch import current_epoch
        doc["geometry_epoch"] = int(current_epoch(output_dir))
    except Exception:  # noqa: BLE001 — a session with no epoch record is fine
        pass
    _atomic_json(parent_p, doc)

    by_reason: Dict[str, int] = {}
    for v in plan.values():
        for p in v["parts"]:
            by_reason[p["reason"]] = by_reason.get(p["reason"], 0) + 1
    ledger = {}
    if (output_dir / MAP).exists():
        try:
            ledger = json.loads((output_dir / MAP).read_text())
        except Exception:  # noqa: BLE001
            ledger = {}
    ledger.setdefault("rounds", []).append({
        "round": round_no, "at": datetime.now().isoformat(timespec="seconds"),
        "archive": archive, "applied": len(part_ids), "by_reason": by_reason,
        "map": {str(k): {"oid": v["oid"], "parts": v["parts"]}
                for k, v in sorted(plan.items())}})
    ledger["masks_total"] = masks_total
    ledger["objects"] = len(kept)
    ledger["geometry_epoch"] = doc.get("geometry_epoch")
    ledger["provenance"] = "tool_measured"
    _atomic_json(output_dir / MAP, ledger)

    log(f"  fusion round {round_no}: {len(parent.get('instances') or [])} "
        f"masklet(s) -> {len(kept)} object(s), {len(part_ids)} part(s) "
        f"absorbed ({by_reason})")
    return result


def _atomic_json(path: Path, doc: dict) -> None:
    tmp = Path(tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")[1])
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, path)


def high_water(output_dir) -> Tuple[int, int]:
    """(id, instance_id) ever allocated in this session, RETIRED ONES INCLUDED.

    A new masklet must not reuse the oid of a mask that was folded into an
    object: the archive and the `absorbed` record still speak about it.
    """
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
    """Every npz obj id an entry owns: its own, plus the parts it absorbed.

    `reconstruction/loops` splits an instance only when it has more than one
    mask id, and after the fusion a survivor's parts ARE those ids.
    """
    out = [int(entry.get("id", 0))]
    out += [int(p["id"]) for p in (entry.get("parts") or [])]
    return sorted(set(out))
