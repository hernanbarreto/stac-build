"""Which masks are OBJECTS and which are provenance (USER 2026-09-17).

``segmentation.json`` lists every mask SAM3 produced; ``segmentation_result
.json`` lists the objects the matching pass produced from them. On pccr that
is 216 against 77 — the other 139 were resolved into another instance (same
space, or same label and contiguous: 84 of them were one floor), dropped for
being too small, or matched no cloud point at all.

The list endpoint takes its entries from the MASK file on purpose: a mask the
user just propagated has to appear before the expensive matching runs again.
The cost was that a fused mask stayed there forever with no points, no OBB and
nothing to select — "aparecen muchisimos en cero ... deben ser los que despues
se fusionaron, pero quedaron en cero y siguen apareciendo en la lista".

So the matching now writes down what it did with every mask, and this module
answers the one question the endpoint has: is this mask an object, or is it
already part of one?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

REASONS = ("space_dedupe", "overlap_dedupe", "fragment", "too_small",
           "unmatched", "fused_or_unmatched")


def instance_id_of(entry: dict) -> Optional[int]:
    """``instance_id`` is the only id consistent across the two files — the
    mask file numbers from 0 and the result file from 1."""
    iid = entry.get("instance_id", entry.get("id"))
    if iid is None:
        return None
    try:
        return int(iid)
    except (TypeError, ValueError):
        return None


def resolve_absorbed(result_data: Optional[dict], raw_instances: Sequence[dict],
                     result_is_newer: bool) -> Dict[int, Dict[str, Any]]:
    """The masks that are NOT objects, by instance_id.

    Three cases, in order:

    * the result carries the record the matching wrote → use it, it says which
      object absorbed each mask and why;
    * a mask that is NEITHER a survivor NOR in the record, while the result is
      newer than the mask file → it did not survive the last pass either. The
      record is not a census: an instance dropped AFTER the matching (a
      correction epoch's ``min_points``, a delete) leaves exactly this gap, and
      without filling it the mask comes back to the list with no points — the
      zero-point entries the record was written to remove. WHICH of fusion or
      no-match it was is not known and is not claimed
      (``fused_or_unmatched``);
    * the mask file is newer → masks were propagated after the last matching
      and a missing one may simply not have been matched yet. Nothing is
      hidden: that is exactly the case the mask-file-as-list-source exists for.
    """
    if not result_data:
        return {}
    recorded = result_data.get("absorbed") or {}
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in recorded.items():
        try:
            out[int(k)] = dict(v)
        except (TypeError, ValueError):
            continue
    survivors = {iid for iid in (instance_id_of(i)
                                 for i in (result_data.get("instances") or []))
                 if iid is not None}
    if not survivors or not result_is_newer:
        return out
    for inst in raw_instances:
        iid = instance_id_of(inst)
        if iid is None or iid in survivors or iid in out:
            continue
        out[iid] = {"into": None, "into_label": None,
                    "reason": "fused_or_unmatched",
                    "detail": "result predates the provenance record; "
                              "re-run the matching to learn which"}
    return out


def split_list(raw_instances: Sequence[dict], enriched_by_id: Dict[int, dict],
               absorbed: Dict[int, Dict[str, Any]]):
    """(listed, hidden): the objects, and the masks that are part of one.

    A mask with NO record passes through untouched even when it carries no
    points — that is the just-propagated case, and hiding it would bring back
    the bug this list source was built to fix.
    """
    listed: List[dict] = []
    hidden: List[dict] = []
    for inst in raw_instances:
        iid = instance_id_of(inst)
        if iid is not None and iid in absorbed:
            hidden.append({**inst, **absorbed[iid]})
        elif iid is not None and iid in enriched_by_id:
            listed.append({**inst, **enriched_by_id[iid]})
        else:
            listed.append(dict(inst))
    return listed, hidden


def by_reason(hidden: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for h in hidden:
        r = str(h.get("reason") or "unknown")
        counts[r] = counts.get(r, 0) + 1
    return counts
