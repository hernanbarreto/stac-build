"""USER 2026-09-20: "la etapa de fusion de instancias es sumamente importante
porque justamente evita estas cosas, que dos instancias con nombre diferente
que son lo mismo sean tratadas como objetos independientes, entonces lo ideal
es que incluso se modificara con la fusion el padre de todos, el
segmentation.json, y el npz … cuando se hace todo el analisis de la
certificacion, y la correccion, se hace sobre las instancias fusionadas y no
sobre 'las partes'".

The motive, measured on pccr 2026-08-31: `wooden_desk#230` (69,608 pts, kf
199-215) reads as ONE visit as a masklet and dies at the "2+ visits" gate with
149 of the 270; its second copy is the masklet `#226` (kf 0-9), 0.64 m away,
which the matcher had already absorbed into it as a fragment. Fused, the same
object gives visits kf 0-12 and kf 199-215, shares 34.1 %/65.9 %, and a closure
of 68.9 cm with 3.0 cm of disagreement against a 9.5 cm bar.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.fuse_parent import (MAP, MASKS, PARENT, apply_fusion,
                                      high_water, oids_of, plan_fusion)


# ── a session on disk: the parent, its masks, and the matcher's record ───

def _session(tmp_path, masklets, masks, absorbed):
    """masklets: [(iid, label)] · masks: {iid: [frame, ...]}"""
    (tmp_path / PARENT).write_text(json.dumps({
        "version": "3.0", "prompt": "p", "prompts": ["p"],
        "resolution": {"scaled": [8, 4], "original": [8, 4]},
        "mask_file": MASKS,
        "instances": [{"id": i - 1, "instance_id": i, "label": lab,
                       "color": "#FFFFFF"} for i, lab in masklets]}))
    store = {}
    for iid, frames in masks.items():
        for f in frames:
            a = np.zeros((8, 4), np.uint8)
            a[iid % 8, 0] = 1                 # a pixel only this oid has
            store[f"f{f}_o{iid - 1}"] = a
    store["obj_ids"] = np.asarray([i - 1 for i, _ in masklets], np.int32)
    store["frames"] = np.asarray(sorted({f for v in masks.values()
                                         for f in v}), np.int32)
    store["scaled_res"] = np.asarray([8, 4], np.int32)
    store["mask_frame_space"] = np.asarray("keyframe_position")
    np.savez_compressed(tmp_path / MASKS, **store)
    # line i = the video frame of keyframe i; the visits are numbered on it
    (tmp_path / "camera_frames.txt").write_text(
        "\n".join(str(k) for k in range(216)) + "\n")
    return {"instances": [{"instance_id": i, "id": i, "label": lab,
                           "globalIndices": [], "total_points": 0}
                          for i, lab in masklets
                          if str(i) not in absorbed],
            "absorbed": absorbed}


def _parent(tmp_path):
    return json.loads((tmp_path / PARENT).read_text())


def _keys(tmp_path):
    return set(np.load(tmp_path / MASKS).files)


# ── the plan ─────────────────────────────────────────────────────────────

def test_the_plan_is_remove_only_and_spares_what_is_part_of_nothing(tmp_path):
    res = _session(tmp_path,
                   [(1, "desk"), (2, "desk"), (3, "light"), (4, "door"),
                    (5, "fresh")],
                   {1: [0], 2: [9], 3: [0], 4: [0], 5: [9]},
                   {"2": {"into": 1, "reason": "fragment", "label": "desk"},
                    "3": {"into": None, "reason": "too_small"},
                    "4": {"into": None, "reason": "unmatched"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    doc = _parent(tmp_path)
    got = {e["instance_id"]: e for e in doc["instances"]}
    assert sorted(got) == [1, 3, 4, 5], \
        "only the absorbed PART leaves; into:None and the unrecorded stay"
    assert [p["instance_id"] for p in got[1]["parts"]] == [2]
    assert "parts" not in got[3] and "parts" not in got[5]


def test_a_mask_propagated_after_the_matching_is_never_deleted(tmp_path):
    """The rewrite is driven by the RECORD, never by the survivor list.
    Building the parent from the survivors would silently delete the user's
    fresh work — the single most important invariant of this change."""
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    doc = json.loads((tmp_path / PARENT).read_text())
    doc["instances"].append({"id": 8, "instance_id": 9, "label": "fresh",
                             "color": "#000000"})
    (tmp_path / PARENT).write_text(json.dumps(doc))
    apply_fusion(tmp_path, res, log=lambda m: None)
    assert 9 in {e["instance_id"] for e in _parent(tmp_path)["instances"]}


def test_a_record_naming_a_target_the_parent_lacks_is_refused(tmp_path):
    parent = {"instances": [{"id": 0, "instance_id": 1, "label": "a"}]}
    with pytest.raises(ValueError, match="refused"):
        plan_fusion(parent, {"1": {"into": 77, "reason": "fragment"}})


# ── the mask store ───────────────────────────────────────────────────────

def test_the_parts_masks_are_ored_into_the_survivor_and_their_keys_retire(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0, 1], 2: [1, 9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    before = _keys(tmp_path)
    assert "f1_o1" in before
    apply_fusion(tmp_path, res, log=lambda m: None)
    z = np.load(tmp_path / MASKS)
    ks = set(z.files)
    assert not any(k.endswith("_o1") for k in ks), "the part's oid retires"
    assert {"f0_o0", "f1_o0", "f9_o0"} <= ks, "the survivor gains its frames"
    # frame 1 had both: the union carries the part's own pixel
    assert z["f1_o0"][1 % 8, 0] == 1 and z["f1_o0"][2 % 8, 0] == 1
    assert list(z["obj_ids"]) == [0]
    assert str(z["mask_frame_space"]) == "keyframe_position"
    assert list(z["scaled_res"]) == [8, 4]


def test_masks_of_different_shapes_are_refused_not_ored(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [0]},
                   {"2": {"into": 1, "reason": "fragment"}})
    store = dict(np.load(tmp_path / MASKS))
    store["f0_o1"] = np.zeros((4, 4), np.uint8)
    np.savez_compressed(tmp_path / MASKS, **store)
    with pytest.raises(ValueError, match="cannot be one object"):
        apply_fusion(tmp_path, res, log=lambda m: None)


# ── idempotence and the mtime contract ───────────────────────────────────

def test_a_second_pass_writes_nothing_at_all(tmp_path):
    """Rewriting the parent bumps its mtime, and the pipeline reads that as
    'the mask->cloud mapping is pending again' — a re-run of every stage."""
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    st = [(p.stat().st_mtime_ns, p.stat().st_size)
          for p in (tmp_path / PARENT, tmp_path / MASKS)]
    apply_fusion(tmp_path, res, log=lambda m: None)
    assert st == [(p.stat().st_mtime_ns, p.stat().st_size)
                  for p in (tmp_path / PARENT, tmp_path / MASKS)]
    assert len(json.loads((tmp_path / MAP).read_text())["rounds"]) == 1


def test_an_empty_record_leaves_the_session_untouched(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "b")], {1: [0], 2: [9]}, {})
    st = (tmp_path / PARENT).stat().st_mtime_ns
    apply_fusion(tmp_path, res, log=lambda m: None)
    assert (tmp_path / PARENT).stat().st_mtime_ns == st
    assert not (tmp_path / MAP).exists()


# ── the ids: nothing is renumbered, nothing is reused ───────────────────

def test_the_survivor_keeps_both_of_its_ids(tmp_path):
    """The fused object is not a new object with a new number. This is what
    keeps `instance_id == id + 1` true where it was true, and every consumer
    of the oid convention working untouched."""
    res = _session(tmp_path, [(1, "a"), (2, "a"), (7, "b")],
                   {1: [0], 2: [9], 7: [3]},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    got = {e["instance_id"]: e["id"] for e in _parent(tmp_path)["instances"]}
    assert got == {1: 0, 7: 6}, "ids unchanged, and non-contiguous is fine"


def test_a_retired_id_is_never_handed_out_again(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    assert high_water(tmp_path) == (1, 2), \
        "the absorbed part's ids stay reserved — the archive and the record " \
        "still speak about them"


def test_the_split_gate_still_sees_more_than_one_mask(tmp_path):
    """An instance may be separated only on SAM3's own evidence that it is
    several objects — which after the fusion lives in `parts`."""
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    surv = _parent(tmp_path)["instances"][0]
    assert oids_of(surv) == [0, 1]


# ── the archive, and back ────────────────────────────────────────────────

def test_the_raw_sam3_output_is_archived_and_restores(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "a")], {1: [0], 2: [9]},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    assert len(_parent(tmp_path)["instances"]) == 1
    from segmentation.unfuse import generations, unfuse
    assert [g.name for g in generations(tmp_path)] == ["gen_000"]
    unfuse(tmp_path, log=lambda m: None)
    assert len(_parent(tmp_path)["instances"]) == 2
    assert "f9_o1" in _keys(tmp_path)


def test_the_ledger_records_the_round_and_its_reasons(tmp_path):
    res = _session(tmp_path, [(1, "a"), (2, "a"), (3, "a")],
                   {1: [0], 2: [9], 3: [5]},
                   {"2": {"into": 1, "reason": "fragment"},
                    "3": {"into": 1, "reason": "overlap_dedupe"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    led = json.loads((tmp_path / MAP).read_text())
    r = led["rounds"][0]
    assert r["applied"] == 2
    assert r["by_reason"] == {"fragment": 1, "overlap_dedupe": 1}
    assert [p["instance_id"] for p in r["map"]["1"]["parts"]] == [2, 3]
    assert led["masks_total"] == 3 and led["objects"] == 1


# ── the motive, pinned ───────────────────────────────────────────────────

def test_fusion_turns_one_visit_into_the_closure_the_correction_needs(tmp_path):
    """The desk, in miniature: two masklets of ONE object, one seen at the
    start of the walk and one at the end. As parts each reads a single visit
    and dies at the gate; fused, the object has two."""
    from correction import visit_drift as vd
    res = _session(tmp_path, [(1, "desk"), (2, "desk")],
                   {1: list(range(0, 10)), 2: list(range(199, 216))},
                   {"2": {"into": 1, "reason": "fragment"}})
    raw = vd.masklet_visits(tmp_path, log=lambda m: None)
    assert sorted(m.n_visits for m in raw) == [1, 1], "as parts: one visit each"
    apply_fusion(tmp_path, res, log=lambda m: None)
    fused = vd.masklet_visits(tmp_path, log=lambda m: None)
    assert len(fused) == 1
    assert fused[0].n_visits == 2, "fused: the revisit becomes measurable"
    assert fused[0].visits == [(0, 9), (199, 215)]


def test_contiguous_parts_collapse_to_one_visit_and_that_is_the_cost(tmp_path):
    """The other side of the trade, stated so nobody rediscovers it as a bug:
    parts that tile one surface WITHOUT a gap fuse into a single continuous
    visit. The pccr floor is 81 such parts."""
    from correction import visit_drift as vd
    res = _session(tmp_path, [(1, "floor"), (2, "floor")],
                   {1: list(range(0, 100)), 2: list(range(100, 200))},
                   {"2": {"into": 1, "reason": "fragment"}})
    apply_fusion(tmp_path, res, log=lambda m: None)
    fused = vd.masklet_visits(tmp_path, log=lambda m: None)
    assert fused[0].n_visits == 1
