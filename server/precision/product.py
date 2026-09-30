"""The reconstruction's PRODUCT: which cloud the precision core published and whether
it is the live epoch. One answer for the pipeline manager's stage probe, the cloud
stage, the map worker and the viewer — they used to key on ``fuse_report.json`` alone
(F7 witness fusion); since 2026-09-29 the default product is the corrected Omega cloud
(``corrected_cloud.json``, precision/corrected_cloud.py) and the fusion is selectable
(``reconstruction.precision.cloud.source``)."""
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


def product_is_live(output_dir) -> Tuple[bool, str]:
    """(live?, reason): the published product names the epoch the session shows."""
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
    if ep != live:
        return False, f"the live epoch {live} is not the published one {ep} ({rep['product_file']})"
    # the record is not the cloud (pccr 2026-09-30: the report and the octree survived, the
    # PLY was deleted — a probe that trusted the record called the stage done)
    if not (out / "cleaned_cloud.ply").exists():
        return False, f"{rep['product_file']} names epoch {ep} but cleaned_cloud.ply is not on disk"
    return True, f"{rep['product_file'].split('.')[0]} (epoch {ep}) on disk"
