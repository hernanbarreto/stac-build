"""USER 2026-10-15: the VLM/segmentation merges ONLY what belongs to the support surfaces
(floor, ground, platform, base …); every other element keeps its own prompt and instance
(maximum detail); the structural envelope (walls, windows, doors, columns, beams, ceilings …)
is always segmented and never forgotten. CPU only; the VLM is a fake answering through the
real parsers."""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from segmentation.autoprompt import consolidate_prompts as CP  # noqa: E402
from segmentation.autoprompt import scene_understanding as SU  # noqa: E402
from segmentation.autoprompt.rounds import select_fresh  # noqa: E402
from segmentation.pipeline import _cross_label_merge_allowed  # noqa: E402

RAW = yaml.safe_load((SERVER / "config.yaml").read_text())
POLICY = CP.merge_policy(RAW["autoprompt"])


def _quiet(_m):
    pass


# ── the policy is configured, every key required ───────────────────────────────────────

def test_the_policy_is_read_from_the_configuration_and_every_key_is_required():
    assert POLICY.support is not None and "floor" in POLICY.support and "platform" in POLICY.support
    assert POLICY.same_head_only is True and POLICY.structural_always is True
    assert POLICY.to_dict()["fold_only_into_support"] is True
    for key in CP.POLICY_KEYS:
        broken = copy.deepcopy(RAW["autoprompt"])
        del broken[key]
        with pytest.raises(KeyError, match=f"autoprompt.{key}"):
            CP.merge_policy(broken)
    assert CP.merge_policy({}) is CP.LEGACY_POLICY and CP.LEGACY_POLICY.support is None
    off = copy.deepcopy(RAW["autoprompt"]); off["fold_only_into_support"] = False
    assert CP.merge_policy(off).support is None


def test_support_and_head_predicates():
    assert CP.is_support("white tiled floor", POLICY.support) and CP.is_support("train platforms", POLICY.support)
    assert CP.is_support("concrete slab", POLICY.support) and not CP.is_support("wall", POLICY.support)
    assert not CP.is_support("desk", POLICY.support) and CP.is_support("anything", None)
    assert CP.same_head("black desk", "desks") and not CP.same_head("desk", "table")
    for name in ("railing", "handrail", "duct", "pipe", "platform", "deck", "ramp", "window", "door"):
        assert CP.is_structural(name), name
    assert not CP.is_structural("door handle") and not CP.is_structural("pipe wrench")


# ── classify_phrases (the production merge) under the policy ───────────────────────────

class _ShortVLM:
    """Answers the per-phrase questions the way the scene is built: the tile is a piece of the
    floor, the drawer a piece of the desk, the rock a unit of the ground, 'black desk' the same
    kind as 'desk'; everything else is whole and of no other kind."""

    def __init__(self):
        self.questions: list[tuple[str, str, list[str]]] = []

    def chat(self, messages, max_tokens=None, consumer=None):
        text = getattr(messages[-1], "text", "") or ""
        listing = [ln[2:] for ln in text.splitlines() if ln.startswith("- ")]
        q = text.split("\n\n")[-1]
        asked = q.split('"')[1]
        self.questions.append((asked, q, listing))
        if "WHOLE object on its own" in q:
            ans = "piece" if asked in ("tile", "drawer") else "whole"
        elif "piece, unit or material of which" in q:
            ans = {"tile": "floor", "drawer": "desk"}.get(asked, "none")
        elif "units or pieces that a" in q:
            built = q.split('"')[3]
            ans = "yes" if (asked, built) == ("rock", "ground") else "no"
        elif "SAME KIND of object as another" in q:
            ans = "desk" if asked == "black desk" and "desk" in listing else "none"
        elif "Answer yes or no" in q:
            ans = "yes"
        else:
            ans = "none"
        return SimpleNamespace(content=ans, finish_reason="stop", usage={"completion_tokens": 2})


PHRASES = ["floor", "ground", "wall", "door", "tile", "desk", "drawer", "black desk", "table", "rock", "cable"]


def test_classify_folds_only_into_support_and_keeps_every_other_piece():
    vlm = _ShortVLM()
    calls: list = []
    part_of: dict = {}
    alias = CP.classify_phrases(vlm, "hall", PHRASES, log=_quiet, calls=calls, part_of=part_of, policy=POLICY)
    assert alias == {"tile": "floor", "rock": "ground", "black desk": "desk"}
    assert part_of == {"tile": "floor", "rock": "ground"}
    by = {c["phrase"]: c for c in calls}
    assert by["drawer"]["decision"] == "detail_kept" and by["drawer"]["piece_of_kept"] == "desk"
    assert "not a support surface" in by["drawer"]["reason"]
    assert by["table"]["decision"] == "own" and by["desk"]["decision"] == "own"
    # the unit-of question is asked against SUPPORT surfaces only: never "is a rock a unit of a wall"
    built_of = [(a, q.split('"')[3]) for a, q, _ in vlm.questions if "units or pieces that a" in q]
    assert built_of and all(b in ("floor", "ground") for _, b in built_of), built_of
    # the same-kind question is asked only among the phrases of one head noun: 'table' and
    # 'cable' have none ('black desk' was merged by its name before any question)
    same_kind = [(a, lst) for a, q, lst in vlm.questions if "SAME KIND of object as another" in q]
    assert same_kind == [("desk", ["black desk"])], same_kind     # only its own head noun is listed
    assert by["black desk"]["by"] == "name_suffix"
    for name in ("table", "cable", "wall", "door"):
        assert by[name]["same_kind_not_asked"] == "no other phrase with this head noun", name
    survivors = [p for p in PHRASES if p not in alias]
    assert survivors == ["floor", "ground", "wall", "door", "desk", "drawer", "table", "cable"]


def test_the_legacy_policy_still_folds_into_any_whole():
    vlm = _ShortVLM()
    part_of: dict = {}
    alias = CP.classify_phrases(vlm, "hall", PHRASES, log=_quiet, part_of=part_of, policy=CP.LEGACY_POLICY)
    assert alias["drawer"] == "desk" and part_of["drawer"] == "desk"


# ── the JSON merge pass and the consolidation under the policy ─────────────────────────

def test_merge_synonyms_keeps_the_pieces_of_a_non_support_whole():
    from tests.test_whole_objects import _MergeVLM, _head_noun
    phrases = ["floor", "tile", "wall", "drywall", "door", "door frame", "car", "wheel", "white wall", "cable"]
    vlm = _MergeVLM({"groups": [
        {"name": "floor", "parts": ["tile"]},
        {"name": "wall", "same_as": ["white wall"], "parts": ["drywall"]},
        {"name": "door", "parts": ["door frame"]},
        {"name": "car", "parts": ["wheel"]},
    ]})
    calls: list = []
    part_of: dict = {}
    alias = CP.merge_synonyms(vlm, "office", phrases, _head_noun, max_phrases_per_call=400,
                              context_len=24576, log=_quiet, calls=calls, part_of=part_of, policy=POLICY)
    assert alias == {"tile": "floor", "white wall": "wall"} and part_of == {"tile": "floor"}
    assert sorted(calls[0]["detail_kept"]) == ["door frame", "drywall", "wheel"]


def test_consolidate_keeps_the_pieces_of_a_non_support_whole():
    from tests.test_consolidate_prompts import _Client
    answer = {"objects": [{"name": "desk", "aliases": [], "parts": ["drawer"]},
                          {"name": "floor", "aliases": [], "parts": ["tile"]},
                          {"name": "table", "aliases": ["desk"], "parts": []}]}
    res = CP.consolidate(_Client(answer), "office", ["desk", "drawer", "floor", "tile", "table"], log=_quiet,
                         policy=POLICY)
    assert res.objects == ["desk", "drawer", "floor", "table"]
    assert res.kept_detail == ["drawer"] and res.parts == {"floor": ["tile"]}


# ── the understanding prompt: structure in a field of its own, maximum detail ──────────

def test_the_prompt_asks_for_the_structure_apart_and_the_parser_adds_it_to_the_objects():
    p = SU._PROMPT
    assert '"structure": [' in p and p.index('"structure"') < p.index('"objects"')
    assert "5. MAXIMUM DETAIL" in p and "its own entry, however small" in p
    assert "railing, duct and pipe" in p
    doc = {"scene_type": "hall", "summary": "s", "structure": ["wall", "Door", "wall", {"category": "window"}],
           "objects": [{"category": "desk"}, {"category": "wall"}]}

    class C:
        def chat(self, msgs, max_tokens=0, consumer=""):
            return SimpleNamespace(content=json.dumps(doc), finish_reason="stop", usage={})
    from PIL import Image
    fu = SU.understand_frame(C(), Image.new("RGB", (8, 8)), 1)
    assert fu.objects == ["desk", "wall", "door", "window"] and fu.structure == ["wall", "door", "window"]
    assert SU.GROUPING_VERSION == "support_folds_v4"


# ── the rounds: one keyframe naming a door is a door ───────────────────────────────────

def test_a_round_keeps_a_structural_name_named_by_one_keyframe():
    counts = Counter({"object": 3, "lamp": 2, "door": 1, "chair": 1, "railing": 1})
    fresh, single, generic = select_fresh(counts, have={"lamp"}, policy=POLICY)
    assert fresh == ["door", "railing"] and single == ["chair"] and generic == ["object"]
    fresh2, single2, _ = select_fresh(counts, have={"lamp"}, policy=CP.LEGACY_POLICY)
    assert fresh2 == [] and sorted(single2) == ["chair", "door", "railing"]


# ── the projection: different labels merge only into a support surface ────────────────

def test_space_dedupe_merges_different_labels_only_into_a_support_surface():
    assert _cross_label_merge_allowed("white_tiled_floor", "tile", POLICY)
    assert _cross_label_merge_allowed("desk", "black_desk", POLICY)            # one head noun
    assert not _cross_label_merge_allowed("desk", "monitor", POLICY)
    assert not _cross_label_merge_allowed("wall", "cabinet", POLICY)           # the wall is not support
    assert _cross_label_merge_allowed("desk", "monitor", CP.LEGACY_POLICY)


# ── end to end through the auto-prompter: the census shows the structure per keyframe ──

FLOOR = {"category": "floor", "description": "grey concrete",
         "shape": {"form": "flat horizontal surface", "material": "concrete", "detail": "wide"}}
TILE = {"category": "tile", "description": "white ceramic"}
DESK = {"category": "desk", "description": "black metal"}
DRAWER = {"category": "drawer", "description": "small"}
CABLE = {"category": "cable", "description": "black rubber"}


class _SceneVLM:
    def __init__(self):
        self.n = 0

    def chat(self, messages, max_tokens=None, consumer=None):
        if consumer == "phase1.merge_synonyms":
            text = getattr(messages[-1], "text", "") or ""
            q = text.split("\n\n")[-1]
            asked = q.split('"')[1]
            if "WHOLE object on its own" in q:
                ans = "piece" if asked in ("tile", "drawer") else "whole"
            elif "piece, unit or material of which" in q:
                ans = {"tile": "floor", "drawer": "desk"}.get(asked, "none")
            elif "Answer yes or no" in q:
                ans = "yes" if "units or pieces" not in q else "no"
            else:
                ans = "none"
            return SimpleNamespace(content=ans, finish_reason="stop", usage={"completion_tokens": 3})
        self.n += 1
        # keyframe 2 of the sampling names NO structure at all (not even the floor)
        structure = [] if self.n == 2 else ["wall"]
        objects = ([TILE, DESK, DRAWER, CABLE] if self.n == 2
                   else [FLOOR, TILE, DESK, DRAWER, CABLE, {"category": "wall"}])
        return SimpleNamespace(content=json.dumps({"scene_type": "hall", "summary": "s", "structure": structure,
                                                   "objects": objects}),
                               finish_reason="stop", usage={"completion_tokens": 50, "prompt_tokens": 800})


def test_the_autoprompter_folds_the_tile_keeps_the_drawer_and_prompts_the_structure_first(tmp_path, monkeypatch):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    from segmentation.census import build_census
    from tests.test_whole_objects import _session
    raw = _session(tmp_path)
    vlm = _SceneVLM()
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: vlm)
    res = AutoPrompter(tmp_path, tmp_path / "output", config=raw).run()
    prompts = res.prompt.split(";")
    assert prompts[:2] == ["floor", "wall"], "the envelope (the floor is structural too) is prompted first"
    assert set(prompts) == {"wall", "floor", "desk", "drawer", "cable"}
    doc = json.loads(Path(res.vlm_analysis_path).read_text())
    rec = json.loads((tmp_path / "output" / "autoprompt_concepts.json").read_text())
    assert rec["parts_folded"] == {"tile": "floor"} and rec["parts_kept_detail"] == ["drawer"]
    assert rec["merge_policy"]["fold_only_into_support"] is True and rec["structural_first"] == ["floor", "wall"]
    census = build_census(tmp_path / "output", prompt=res.prompt, vlm_doc=doc, gap_kf=1,
                          frames_dir=tmp_path / "frames", log=_quiet)
    assert census["accounting"]["closed"] is True
    assert census["vlm"]["n_keyframes_without_structure"] == 1
    assert len(census["vlm"]["keyframes_without_structure"]) == 1
    looked = {e["frame"]: e for e in census["vlm"]["looked_at"]}
    assert sum(1 for e in looked.values() if e["n_structural"] == 0) == 1
