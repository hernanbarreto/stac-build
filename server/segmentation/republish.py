"""Everything that hangs off a point cloud, republished in one place.

A cloud never travels alone. Beside `cleaned_cloud.ply` live arrays that are
indexed BY ROW — `classification.npy` (the per-point instance id the Potree
converter bakes into the octree) and `out_of_place.npy` (the geometric-cleanup
marks) — and a census that has to agree with it, `segmentation_result.json`
(`total_points`, `segmented_points`, `coverage`).

Nothing derives these on the fly: they are files, and whoever republishes the
cloud has to republish them too. A correction epoch did not (pccr 2026-09-20:
after an epoch deleted 338,030 points, `classification.npy` still described
the previous cloud — 22,770,025 values against 22,431,995 rows. The converter
compares the two lengths, writes NOTHING and logs one line, so the epoch's
octree came out with every class byte at zero; `geometric_cleanup` refused to
run at all, saying the marks were stale).

So there is one function, and it is the only writer of these files. It takes
the directory the cloud lives in — a staged `_tx_epoch_<N>/` or a live
`output/`, same code — and leaves every sibling in step with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

CLASSIFICATION = "classification.npy"
CLASS_MAP = "class_map.json"
OUT_OF_PLACE = "out_of_place.npy"
RESULT = "segmentation_result.json"
BROADCAST = "seg_broadcast.json"


def _encode(ids: List[int]) -> dict:
    """instance_id → the byte that carries it in the octree.

    The class travels as ONE BYTE: `classification.npy` is uint8 and the LAS
    field the Potree converter writes it into is a byte too. Instance ids are
    not bounded by that — they are a 1-based counter over SAM3's masklets, so
    a session with 270 masklets has ids up to 270 (pccr 2026-09-20: eight live
    instances above 254 — `exposed_ceiling_beams`, four `black_electrical_
    cable`, three `exposed_ceiling_wiring` — all saturated to 255 by a
    `min(id, 255)`, 45,091 points that the viewer showed as one block and one
    toggle switched all eight at once).

    So the byte is the instance id WHILE THE IDS FIT, which is every ordinary
    session and keeps the array byte-identical to what every reader already
    expects; and a compact 1..N index when they do not, with `class_map.json`
    beside it saying which is which. Above 255 OBJECTS there is no byte left
    and it says so instead of painting them the same colour.
    """
    ids = sorted({int(i) for i in ids if int(i) > 0})
    if not ids:
        return {"encoding": "identity", "class_of": {}, "instance_of": {}}
    if ids[-1] <= 255:
        code = {i: i for i in ids}
        enc = "identity"
    else:
        if len(ids) > 255:
            raise ValueError(
                f"{len(ids)} instances cannot be carried in one class byte "
                f"(255 codes, 0 reserved for unsegmented) — the octree would "
                f"paint different objects the same colour")
        code = {iid: n for n, iid in enumerate(ids, start=1)}
        enc = "compact"
    return {"encoding": enc,
            "class_of": {str(k): int(v) for k, v in code.items()},
            "instance_of": {str(v): int(k) for k, v in code.items()}}


def class_of(dirpath) -> dict:
    """instance_id → class byte. An absent map means identity: every session
    written before the map existed carries the id in the byte."""
    p = Path(dirpath) / CLASS_MAP
    if not p.exists():
        return {}
    try:
        return {int(k): int(v) for k, v in
                (json.loads(p.read_text()).get("class_of") or {}).items()}
    except Exception:
        return {}


def instance_of(dirpath) -> dict:
    """class byte → instance_id. Identity when there is no map."""
    p = Path(dirpath) / CLASS_MAP
    if not p.exists():
        return {}
    try:
        return {int(k): int(v) for k, v in
                (json.loads(p.read_text()).get("instance_of") or {}).items()}
    except Exception:
        return {}


def class_byte(dirpath, instance_id: int) -> int:
    """The byte that carries this instance, whatever the encoding."""
    return int(class_of(dirpath).get(int(instance_id), int(instance_id)))


def as_instance_ids(dirpath, cls: np.ndarray) -> np.ndarray:
    """A class-byte array read back as INSTANCE ids (int32, 0 = unsegmented).
    Identity when the session has no map, so it is always safe to call."""
    m = instance_of(dirpath)
    out = np.asarray(cls).astype(np.int32)
    if not m:
        return out
    lut = np.zeros(256, np.int32)
    for b, iid in m.items():
        if 0 <= b < 256:
            lut[b] = iid
    return lut[np.clip(out, 0, 255)]


def write_classification(dirpath: Path, instances: List[dict],
                         n_points: int) -> np.ndarray:
    """Rebuild `classification.npy`: per-point INSTANCE id, 0 = unsegmented.

    It is THE colour source the Potree converter bakes into the octree.
    Without it an edit updated the indices and the rebuilt octree kept
    painting the removed points with their old segment colour (user
    2026-08-30).

    The value is the INSTANCE id, never the mask obj id: brush-created
    segments carry `id = instance_id − 1`, and writing THAT painted their
    points with the PREVIOUS segment's class — the "reassigned points answer
    to another segment's toggle" bug (user 2026-08-31).
    """
    dirpath = Path(dirpath)
    ids = [int(i.get("instance_id", i.get("id", 0))) for i in instances]
    m = _encode(ids)
    code = {int(k): int(v) for k, v in m["class_of"].items()}
    classification = np.zeros(int(n_points), dtype=np.uint8)
    for inst in instances:
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < int(n_points))]
        iid = int(inst.get("instance_id", inst.get("id", 0)))
        classification[gi] = code.get(iid, 0)
    np.save(dirpath / CLASSIFICATION, classification)
    (dirpath / CLASS_MAP).write_text(json.dumps(m, indent=1))
    return classification


def republish_membership(dirpath, *, n_points: int,
                         keep: Optional[np.ndarray] = None,
                         source_dir=None,
                         broadcast: bool = True,
                         log: Callable[[str], None] = print) -> dict:
    """Leave every per-point sibling of the cloud in `dirpath` in step with it.

    `n_points`  rows of the cloud that now lives in `dirpath`.
    `keep`      the boolean mask that took `source_dir`'s cloud to this one,
                when rows were deleted; None when nothing was removed.
    `source_dir` where the PREVIOUS generation of the per-point arrays lives
                (the live `output/` while `dirpath` is a transaction). Arrays
                that are MEASUREMENTS — `out_of_place.npy` — are carried
                through `keep`, never recomputed: the marks are evidence.

    Returns the paths it actually wrote, relative to `dirpath`, so a caller
    staging a transaction registers exactly what exists and nothing else.
    """
    dirpath = Path(dirpath)
    source_dir = Path(source_dir) if source_dir is not None else None
    n_points = int(n_points)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        if int(keep.sum()) != n_points:
            raise ValueError(
                f"the keep mask keeps {int(keep.sum()):,} rows and the cloud "
                f"has {n_points:,} — republishing would write arrays that do "
                f"not describe it")
    written: List[str] = []
    out = {"files": written, "instances": 0, "classified_points": 0,
           "max_instance_id": 0, "provenance": "tool_measured"}

    # ── the census: the result file has to agree with the cloud beside it ──
    res_path = dirpath / RESULT
    result = None
    if res_path.exists():
        result = json.loads(res_path.read_text())
        instances = result.get("instances") or []
        segmented = sum(int(i.get("total_points") or 0) for i in instances)
        result["total_points"] = n_points
        result["segmented_points"] = segmented
        result["coverage"] = (segmented / n_points) if n_points else 0.0
        res_path.write_text(json.dumps(result))
        written.append(RESULT)
        out["instances"] = len(instances)
        out["classified_points"] = segmented
        out["max_instance_id"] = max(
            (int(i.get("instance_id", i.get("id", 0))) for i in instances),
            default=0)
        log(f"  republish: {len(instances)} instance(s), {segmented:,} of "
            f"{n_points:,} points segmented ({result['coverage']:.2%})")

    # ── classification.npy — rebuilt from the census when there is one, and
    #    otherwise carried through `keep`: membership is all that is left, and
    #    a per-point array longer than its cloud is never left behind ────────
    if result is not None:
        write_classification(dirpath, result.get("instances") or [], n_points)
        written.extend((CLASSIFICATION, CLASS_MAP))
    elif source_dir is not None and (source_dir / CLASSIFICATION).exists():
        src = np.load(source_dir / CLASSIFICATION)
        if len(src) == (len(keep) if keep is not None else n_points):
            np.save(dirpath / CLASSIFICATION,
                    src[keep] if keep is not None else src)
            written.append(CLASSIFICATION)
        else:
            log(f"  republish: {CLASSIFICATION} in the source has {len(src):,} "
                f"values and cannot be carried — not staged")

    # ── out_of_place.npy — a MEASUREMENT, carried, never recomputed ────────
    if source_dir is not None and (source_dir / OUT_OF_PLACE).exists():
        src = np.load(source_dir / OUT_OF_PLACE)
        if len(src) == (len(keep) if keep is not None else n_points):
            np.save(dirpath / OUT_OF_PLACE,
                    src[keep] if keep is not None else src)
            written.append(OUT_OF_PLACE)
            log(f"  republish: {OUT_OF_PLACE} carried "
                f"({int(np.asarray(src).sum()):,} marks)")
        else:
            log(f"  republish: {OUT_OF_PLACE} in the source has {len(src):,} "
                f"flags against {len(keep) if keep is not None else n_points:,} "
                f"rows — stale already, not carried")

    # ── seg_broadcast.json — the SAM3 stage's snapshot. It carries
    #    globalIndices, so against a filtered cloud it indexes the wrong
    #    points (`workers/instance_cleaner_worker` reads it that way). It is
    #    only kept in step when the session already has one: an epoch never
    #    invents a stage artifact that was not there ─────────────────────────
    if broadcast and result is not None and source_dir is not None \
            and (source_dir / BROADCAST).exists():
        (dirpath / BROADCAST).write_text(json.dumps(result))
        written.append(BROADCAST)

    return out
