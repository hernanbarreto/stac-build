"""USER 2026-09-17, looking at the segmentation list: "aparecen muchisimos en
cero, aparecen todos, deben ser los que despues se fusionaron, pero quedaron en
cero y siguen apareciendo en la lista".

They were right about the cause. The list is built from ``segmentation.json``
(every mask SAM3 produced) and enriched from ``segmentation_result.json`` (the
objects the matching produced) — on pccr, 216 against 77. The 139 masks the
matching resolved into another object had nothing to enrich them with, so they
rendered as zero-point rows.

Taking the list from the result file instead is NOT the fix: a mask the user
just propagated has to appear before the expensive matching runs again, which
is the bug the mask-file source was introduced to fix. What was missing is the
record of what the matching DID with each mask. These tests pin both halves:
a recorded mask is not an object, and an unrecorded one is never hidden.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# loaded by path: segmentation/__init__ pulls the whole SAM3 stack (cv2, torch)
# and this module depends on nothing but dicts
_spec = importlib.util.spec_from_file_location(
    "_mask_fates", Path(__file__).resolve().parents[1] / "segmentation" / "mask_fates.py")
mask_fates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mask_fates)
by_reason, resolve_absorbed, split_list = (mask_fates.by_reason,
                                           mask_fates.resolve_absorbed,
                                           mask_fates.split_list)

# the mask file numbers from 0 and the result file from 1 — only instance_id
# is consistent across the two, which is why everything here keys on it
MASKS = [{"id": i, "instance_id": i + 1, "label": f"l{i + 1}"} for i in range(5)]


def _result(survivors, absorbed=None, **extra):
    r = {"instances": [{"id": s, "instance_id": s, "label": f"l{s}",
                        "total_points": 100 * s, "globalIndices": [s]} for s in survivors]}
    if absorbed is not None:
        r["absorbed"] = absorbed
    r.update(extra)
    return r


def _enriched(result):
    return {int(i["instance_id"]): i for i in result["instances"]}


# ── the record decides ───────────────────────────────────────────────────

def test_a_recorded_mask_is_not_an_object():
    res = _result([1], absorbed={"2": {"into": 1, "into_label": "l1", "reason": "fragment"},
                                 "3": {"into": 1, "into_label": "l1", "reason": "space_dedupe"}})
    ab = resolve_absorbed(res, MASKS, result_is_newer=True)
    listed, hidden = split_list(MASKS, _enriched(res), ab)
    assert [m["instance_id"] for m in listed] == [1, 4, 5]
    assert [h["instance_id"] for h in hidden] == [2, 3]
    assert by_reason(hidden) == {"fragment": 1, "space_dedupe": 1}


def test_the_hidden_row_carries_where_it_went():
    """Hiding without saying where would make the fusion silent — the row must
    be able to answer "part of what?"."""
    res = _result([1], absorbed={"2": {"into": 1, "into_label": "l1", "reason": "fragment"}})
    _listed, hidden = split_list(MASKS, _enriched(res), resolve_absorbed(res, MASKS, True))
    assert hidden[0]["into"] == 1 and hidden[0]["into_label"] == "l1"
    assert hidden[0]["label"] == "l2"           # its own label survives the merge of dicts


def test_a_mask_with_no_record_stays_in_the_list_even_with_no_points():
    """The just-propagated case: it has no points yet because the matching has
    not run over it. Hiding it is the original bug."""
    res = _result([1], absorbed={"2": {"into": 1, "reason": "fragment"}})
    listed, _hidden = split_list(MASKS, _enriched(res), resolve_absorbed(res, MASKS, True))
    fresh = [m for m in listed if m["instance_id"] == 5][0]
    assert "total_points" not in fresh and "globalIndices" not in fresh


def test_a_survivor_is_never_hidden_even_if_the_record_names_it():
    """A contradictory record must not make an object disappear from the list."""
    res = _result([1, 2], absorbed={"2": {"into": 1, "reason": "fragment"}})
    listed, hidden = split_list(MASKS, _enriched(res), resolve_absorbed(res, MASKS, True))
    # split_list honours the record, so this is the guard that must hold upstream
    assert [h["instance_id"] for h in hidden] == [2]
    assert 2 not in [m["instance_id"] for m in listed]
    # _mask_fates never writes a survivor into the record — pinned in its own test


# ── without a record: only what is provable ──────────────────────────────

def test_a_result_newer_than_the_masks_proves_the_absent_ones_did_not_survive():
    res = _result([1, 4])                       # no "absorbed" key: an old result
    ab = resolve_absorbed(res, MASKS, result_is_newer=True)
    assert sorted(ab) == [2, 3, 5]
    assert all(v["reason"] == "fused_or_unmatched" for v in ab.values())
    assert all(v["into"] is None for v in ab.values()), \
        "which object absorbed it is NOT known here and must not be claimed"


def test_an_older_result_hides_nothing():
    """The mask file being newer means masks were propagated after the last
    matching, so a missing one may simply not have been matched yet."""
    res = _result([1, 4])
    assert resolve_absorbed(res, MASKS, result_is_newer=False) == {}


def test_no_result_at_all_hides_nothing():
    assert resolve_absorbed(None, MASKS, result_is_newer=True) == {}
    assert resolve_absorbed({}, MASKS, result_is_newer=True) == {}


def test_an_empty_result_hides_nothing():
    """A matching that produced zero objects is a failure, not evidence that
    every mask was fused."""
    assert resolve_absorbed(_result([]), MASKS, result_is_newer=True) == {}


def test_the_record_wins_over_the_timestamp_fallback():
    res = _result([1, 4], absorbed={"2": {"into": 1, "reason": "space_dedupe"}})
    ab = resolve_absorbed(res, MASKS, result_is_newer=True)
    assert sorted(ab) == [2], "a real record is never widened by the fallback"


def test_a_malformed_record_entry_is_skipped_not_fatal():
    res = _result([1], absorbed={"two": {"into": 1, "reason": "fragment"},
                                 "3": {"into": 1, "reason": "fragment"}})
    assert sorted(resolve_absorbed(res, MASKS, result_is_newer=True)) == [3]


# ── the producer: every mask gets a fate, survivors get none ─────────────

def _mask_fates_fn():
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "pipeline.py").read_text()
    body = src[src.index("def _mask_fates("):src.index("def _match_masks_to_cloud(")]
    ns = {}
    exec(compile(body, "pipeline.py", "exec"), ns)
    return ns["_mask_fates"]


def test_every_mask_that_is_not_an_instance_gets_a_fate():
    fates = _mask_fates_fn()({"instances": MASKS}, [{"instance_id": 1, "label": "l1"}],
                             {2: {"into": 1, "reason": "space_dedupe"}})
    assert sorted(fates) == [2, 3, 4, 5]
    assert 1 not in fates, "a survivor is an object, not provenance"
    assert fates[3]["reason"] == "unmatched"    # nothing recorded it → nothing matched it


def test_an_absorption_chain_resolves_to_the_instance_that_survived():
    """Fragment consolidation absorbs into a keeper that space-dedupe may then
    absorb in turn. Pointing the row at the middle link would name a ghost."""
    fates = _mask_fates_fn()({"instances": MASKS}, [{"instance_id": 1, "label": "floor"}],
                             {2: {"into": 1, "reason": "space_dedupe"},
                              3: {"into": 2, "into_label": "stale", "reason": "fragment"}})
    assert fates[3]["into"] == 1
    assert fates[3]["into_label"] == "floor", "the label must follow the resolved target"


def test_a_cycle_in_the_record_cannot_hang_the_resolution():
    fates = _mask_fates_fn()({"instances": MASKS}, [{"instance_id": 1, "label": "l1"}],
                             {2: {"into": 3, "reason": "fragment"},
                              3: {"into": 2, "reason": "fragment"}})
    assert fates[2]["into"] is None and fates[3]["into"] is None


# ── the record has to survive the save ───────────────────────────────────

def _merge_absorbed_fn():
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "pipeline.py").read_text()
    body = src[src.index("def _merge_absorbed("):src.index("def _match_and_save_result(")]
    ns = {}
    exec(compile(body, "pipeline.py", "exec"), ns)
    return ns["_merge_absorbed"]


def test_the_result_writer_carries_the_absorbed_key():
    """pccr 2026-09-17: the matcher printed "138 mask(s) ... are not separate
    objects" and segmentation_result.json came out with no record at all.
    _match_and_save_result rebuilds the dict key by key, so a key the matcher
    returns reaches disk only if it is named there. This pins the name."""
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "pipeline.py").read_text()
    writer = src[src.index("        merged_result = {"):src.index("atomic_write_json(result_path, merged_result)")]
    assert '"absorbed"' in writer,         "merged_result must carry 'absorbed' or the record never reaches the file"


def test_the_newer_record_wins_and_a_survivor_leaves_the_record():
    merge = _merge_absorbed_fn()
    prev = {"2": {"into": 1, "reason": "fragment"}, "9": {"into": 1, "reason": "space_dedupe"}}
    new = {"3": {"into": 1, "reason": "fragment"}, "2": {"into": 5, "reason": "space_dedupe"}}
    out = merge(prev, new, [{"instance_id": 1}, {"instance_id": 9}])
    assert sorted(out) == ["2", "3"]
    assert out["2"]["into"] == 5                      # the new pass overrides the old
    assert "9" not in out, "a mask that is an instance again is absorbed by nothing"


def test_merging_nothing_is_empty_and_a_bad_key_is_skipped():
    merge = _merge_absorbed_fn()
    assert merge(None, None, [{"instance_id": 1}]) == {}
    assert merge({"dos": {"into": 1}}, None, []) == {}
