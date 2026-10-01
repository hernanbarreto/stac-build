"""The reconstruction's PRODUCT: which cloud the precision core published and whether
it is the live epoch. One answer for the pipeline manager's stage probe, the cloud
stage, the map worker and the viewer — they used to key on ``fuse_report.json`` alone
(F7 witness fusion); since 2026-09-29 the default product is the corrected Omega cloud
(``corrected_cloud.json``, written by precision/corrected_cloud.publish for the bend
and the corrected chain alike) and the fusion is selectable
(``reconstruction.precision.cloud.source``).

LIVE means: the live epoch IS the product's epoch, or DESCENDS from it through
transform epochs only (the certification warps the published cloud per keyframe —
depth per chunk, floor, mask filter — and publishes the result as a new epoch; it is
still the product's cloud, moved). A new-cloud epoch published after the product's
without a report of its own is NOT the product. The ancestry is read from the epoch
records (``parent_epoch``) and each epoch's kind from its record or, once the stored
directories are gone (``certify.single_final_epoch``), from the ledger — never from a
file that the single final epoch deletes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

# newest product first: a session that ran both keeps the last one published live
PRODUCT_REPORTS = ("corrected_cloud.json", "fuse_report.json")


def product_report(output_dir) -> Optional[dict]:
    """The report of the published product (with ``product_file``), the one whose
    ``epoch_to`` is the live epoch when both exist; None when the core published no
    cloud."""
    out = Path(output_dir)
    live = None
    ge = out / "geometry_epoch.json"
    if ge.exists():
        try:
            live = json.loads(ge.read_text()).get("epoch")
        except (OSError, ValueError):
            live = None
    found = []
    for name in PRODUCT_REPORTS:
        p = out / name
        if not p.exists():
            continue
        try:
            rep = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        rep = dict(rep, product_file=name)
        if live is not None and rep.get("epoch_to") == live:
            return rep
        found.append(rep)
    return found[0] if found else None


def transform_descent(output_dir, product_epoch: int, live: int) -> Tuple[Optional[list], str]:
    """The transform epochs between the product's epoch and the live one, in order
    — ``([], "")`` when they are the same epoch, ``(None, why)`` when the live epoch
    does not descend from the product's through transform epochs only."""
    from correction.epoch import EPOCH_KIND_TRANSFORM, epoch_kind, epoch_lineage
    out = Path(output_dir)
    product_epoch, live = int(product_epoch), int(live)
    if product_epoch == live:
        return [], ""
    lineage = epoch_lineage(out, live)
    if product_epoch not in lineage:
        return None, (f"the live epoch {live} does not descend from the published epoch "
                      f"{product_epoch} (its ancestry is {lineage})")
    after = lineage[lineage.index(product_epoch) + 1:]
    for e in after:
        if epoch_kind(out, e) != EPOCH_KIND_TRANSFORM:
            return None, (f"epoch {e} is a NEW CLOUD published after the product's epoch "
                          f"{product_epoch}, with no report of its own — the live epoch "
                          f"{live} is not the product warped")
    return after, ""


def product_is_live(output_dir) -> Tuple[bool, str]:
    """(live?, reason): the published product is the epoch the session shows, or
    the live epoch is that product warped by transform epochs (the certification)."""
    out = Path(output_dir)
    rep = product_report(out)
    if rep is None:
        return False, "no published cloud (corrected_cloud.json / fuse_report.json) yet"
    ge = out / "geometry_epoch.json"
    try:
        live = json.loads(ge.read_text()).get("epoch")
    except (OSError, ValueError):
        return False, "geometry_epoch.json unreadable"
    ep = rep.get("epoch_to")
    name = rep["product_file"].split(".")[0]
    if ep is None or live is None:
        return False, f"the live epoch {live} is not the published one {ep} ({rep['product_file']})"
    through, why = transform_descent(out, ep, live)
    if through is None:
        return False, why
    # the record is not the cloud (pccr 2026-09-30: the report and the octree survived, the
    # PLY was deleted — a probe that trusted the record called the stage done)
    if not (out / "cleaned_cloud.ply").exists():
        return False, f"{rep['product_file']} names epoch {ep} but cleaned_cloud.ply is not on disk"
    if through:
        return True, (f"{name} (epoch {ep}) on disk, warped by transform epoch(s) "
                      f"{through} to the live epoch {live}")
    return True, f"{name} (epoch {ep}) on disk"
