"""The second VLM pass fuses the names that mean the same thing (USER 2026-09-29),
per head-noun family, before the SAM3 bound; different colour/kind never merges
unless the VLM says so, and phrases of different heads are never asked together."""
from types import SimpleNamespace
import json

import pathlib

import yaml

from segmentation.autoprompt import consolidate_prompts as CP
from segmentation.autoprompt.consolidate_prompts import merge_synonyms
from segmentation.autoprompt.scene_understanding import _head_noun

# the served context of the configured engine: the merge answer's bound is derived from it
CONTEXT = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml")
                         .read_text())["semantic"]["backends"]["qwen_local"]["max_model_len"]


PHRASES = {"white wall", "white painted wall", "beige wall", "black office desk",
           "black desk", "white workbench", "glass door"}


class FakeVLM:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def chat(self, msgs, max_tokens=0, consumer=""):
        text = getattr(msgs[-1], "text", None) or str(msgs[-1])
        listed = [ln[2:] for ln in str(text).splitlines() if ln.startswith("- ") and ln[2:] in PHRASES]
        self.asked.append(listed)
        for key, ans in self.answers.items():
            if key in listed:
                return SimpleNamespace(content=json.dumps(ans))
        return SimpleNamespace(content=json.dumps({"groups": []}))


def test_merges_only_what_the_vlm_calls_the_same_within_a_family():
    phrases = ["white wall", "white painted wall", "beige wall", "black office desk",
               "black desk", "white workbench", "glass door"]
    vlm = FakeVLM({"white wall": {"groups": [{"name": "white wall", "same_as": ["white painted wall"]}]},
                   "black office desk": {"groups": [{"name": "black desk", "same_as": ["black office desk"]}]}})
    alias = merge_synonyms(vlm, "office", phrases, _head_noun, context_len=CONTEXT,
                           log=lambda m: None)
    assert alias == {"white painted wall": "white wall", "black office desk": "black desk"}
    # one call per family with 2+ phrases; singletons never asked; families never mixed
    heads = [{_head_noun(p) for p in a} for a in vlm.asked]
    assert all(len(h) == 1 for h in heads) and len(vlm.asked) == 2


def test_a_broken_answer_fails_the_merge_instead_of_keeping_the_family():
    """docs/plan_determinismo.md point 88 (DECIDIDO 2026-10-08): a fusion call that answers
    nothing usable fails the stage — 'kept as is' let one token decide whether 134 names
    were merged. An answer that FINISHED but does not parse whole keeps its complete groups
    (salvaged, declared); a CUT one is never rescued in part (2026-10-09, below)."""
    import pytest
    from segmentation.autoprompt.consolidate_prompts import MergeError

    class Bad:
        def chat(self, *a, **k):
            return SimpleNamespace(content="no json", finish_reason="stop")
    with pytest.raises(MergeError, match="no usable group"):
        merge_synonyms(Bad(), "x", ["red door", "red doors door"], _head_noun,
                       context_len=CONTEXT, log=lambda m: None)

    class Broken:
        def chat(self, *a, **k):
            return SimpleNamespace(
                content='{"groups": [{"name": "red door", "same_as": ["red doors door"]}, {"name": "red',
                finish_reason="stop")
    calls = []
    alias = merge_synonyms(Broken(), "x", ["red door", "red doors door"], _head_noun,
                           context_len=CONTEXT, log=lambda m: None, calls=calls)
    assert alias == {"red doors door": "red door"}
    assert calls[0]["truncated"] is False and calls[0]["salvaged"] == 1 and calls[0]["n_merged"] == 1


# ── 2026-10-09: the merge answer is never cut ─────────────────────────────────────────

class ByList:
    """Answers per LISTED set of phrases (cut when the set is in ``cut``); records each call's
    list and max_tokens."""

    def __init__(self, answers, cut=()):
        self.answers, self.cut = answers, [frozenset(c) for c in cut]
        self.asked, self.max_tokens = [], []

    def chat(self, msgs, max_tokens=0, consumer=""):
        text = getattr(msgs[-1], "text", None) or str(msgs[-1])
        listed = [ln[2:] for ln in str(text).splitlines() if ln.startswith("- ")]
        self.asked.append(listed)
        self.max_tokens.append(max_tokens)
        if frozenset(listed) in self.cut:
            return SimpleNamespace(content='{"groups": [{"name": "' + listed[0] + '", "same_as": []}, {"na',
                                   finish_reason="length", usage={"completion_tokens": max_tokens})
        ans = self.answers.get(frozenset(listed), {"groups": []})
        return SimpleNamespace(content=json.dumps(ans), finish_reason="stop",
                               usage={"completion_tokens": 10})


def test_the_answer_may_use_all_the_room_the_served_context_leaves():
    phrases = ["floor", "white floor"]
    vlm = ByList({frozenset(phrases): {"groups": [{"name": "floor", "same_as": ["white floor"]}]}})
    calls = []
    alias = merge_synonyms(vlm, "room", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=CONTEXT, log=lambda m: None, calls=calls)
    assert alias == {"white floor": "floor"}
    room, bound = CP.merge_answer_bound(CONTEXT, CP._MERGE_SYSTEM, CP._merge_prompt("room", phrases))
    # 2026-10-09: the one-line-per-phrase answer is bounded by the list itself when that is
    # smaller than the room the context leaves (a looping answer stops after every phrase)
    want = min(room, CP._line_answer_bound(phrases))
    assert vlm.max_tokens == [want] and room == CONTEXT - bound
    # the prompt is counted from ABOVE: the bytes of the rendered chat, never below its tokens
    rendered = (f"<|im_start|>system\n{CP._MERGE_SYSTEM}<|im_end|>\n<|im_start|>user\n"
                f"{CP._merge_prompt('room', phrases)}<|im_end|>\n<|im_start|>assistant\n")
    assert bound == len(rendered.encode("utf-8")) >= len(rendered)
    assert calls[0]["context_len"] == CONTEXT and calls[0]["max_tokens"] == want
    assert calls[0]["prompt_bound_tokens"] == bound and calls[0]["split"] is None
    # the bound is no configuration key any more
    raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert "merge_max_tokens" not in raw["autoprompt"]
    assert "merge_max_tokens" in CP.REMOVED_MERGE_KEYS


def test_a_cut_answer_is_split_in_sorted_halves_and_a_final_pass_folds_across_them():
    """pccr 2026-10-09: the 85-name answer was cut and 0 parts were folded. A cut answer is
    never rescued in part: the sorted halves are merged on their own, then the survivors of
    both — the 'tile' of one half folds into the 'floor' of the other."""
    phrases = ["floor", "white floor", "tile", "ceramic tile"]
    # sorted: ceramic tile, floor | tile, white floor
    vlm = ByList({
        frozenset({"ceramic tile", "floor"}): {"groups": [{"name": "floor", "parts": ["ceramic tile"]}]},
        frozenset({"tile", "white floor"}): {"groups": []},
        frozenset({"floor", "tile", "white floor"}): {"groups": [
            {"name": "floor", "same_as": ["white floor"], "parts": ["tile"]}]},
    }, cut=[phrases])
    calls, part_of, said = [], {}, []
    alias = merge_synonyms(vlm, "room", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=CONTEXT, log=said.append, calls=calls, part_of=part_of)
    assert vlm.asked == [phrases, ["ceramic tile", "floor"], ["tile", "white floor"],
                         ["floor", "tile", "white floor"]], "sorted halves, then the survivors"
    assert alias == {"ceramic tile": "floor", "white floor": "floor", "tile": "floor"}
    assert part_of == {"ceramic tile": "floor", "tile": "floor"}
    top = calls[0]
    assert top["truncated"] is True and top["salvaged"] is None and top["n_merged"] == 0, \
        "nothing is rescued from the cut answer"
    assert top["split_into"] == {"halves": [2, 2], "survivors": 3}
    assert [c["split"] for c in calls] == [None, "1", "2", "x"]
    assert any("CUT" in m and "nothing is rescued" in m for m in said)
    assert any("DECLARED split" in m for m in said)


def test_a_carrier_of_one_half_may_be_merged_in_the_final_pass():
    phrases = ["chair", "office chair", "seat", "swivel seat"]
    # sorted: chair, office chair | seat, swivel seat
    vlm = ByList({
        frozenset({"chair", "office chair"}): {"groups": [{"name": "chair", "same_as": ["office chair"]}]},
        frozenset({"seat", "swivel seat"}): {"groups": [{"name": "seat", "same_as": ["swivel seat"]}]},
        frozenset({"chair", "seat"}): {"groups": [{"name": "chair", "same_as": ["seat"]}]},
    }, cut=[phrases])
    alias = merge_synonyms(vlm, "office", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=CONTEXT, log=lambda m: None)
    assert alias == {"office chair": "chair", "seat": "chair", "swivel seat": "chair"}


def test_a_cut_list_whose_halves_merge_nothing_fails_instead_of_a_partial_merge():
    import pytest
    from segmentation.autoprompt.consolidate_prompts import MergeError
    phrases = ["red door", "red doors door"]
    vlm = ByList({}, cut=[phrases])
    calls = []
    with pytest.raises(MergeError, match="never a partial merge"):
        merge_synonyms(vlm, "x", phrases, _head_noun, context_len=CONTEXT,
                       log=lambda m: None, calls=calls)
    assert len(vlm.asked) == 1 and calls[0]["split_into"] == {"halves": [1, 1], "survivors": 2}


def test_a_list_whose_prompt_fills_the_context_is_split_without_asking():
    phrases = ["floor", "white floor", "tile", "ceramic tile"]
    _mt, full = CP.merge_answer_bound(0, CP._MERGE_SYSTEM, CP._merge_prompt("room", phrases))
    vlm = ByList({
        frozenset({"ceramic tile", "floor"}): {"groups": [{"name": "floor", "parts": ["ceramic tile"]}]},
        frozenset({"floor", "tile", "white floor"}): {"groups": [
            {"name": "floor", "same_as": ["white floor"], "parts": ["tile"]}]},
    })
    calls = []
    alias = merge_synonyms(vlm, "room", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=full, log=lambda m: None, calls=calls)
    assert phrases not in vlm.asked, "a list whose prompt fills the context is not asked"
    assert calls[0]["asked"] is False and calls[0]["no_room"] is True and calls[0]["max_tokens"] <= 0
    assert alias == {"ceramic tile": "floor", "white floor": "floor", "tile": "floor"}
    assert all(m > 0 for m in vlm.max_tokens)


def test_the_served_context_is_read_from_the_engine_then_the_configuration():
    import pytest
    from segmentation.autoprompt.consolidate_prompts import MergeError, served_context_len
    ident = {"argv": ["vllm", "serve", "w", "--max-model-len", "12345", "--seed", "0"]}
    assert served_context_len(ident, {}, "qwen_local")[0] == 12345
    raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    n, src = served_context_len(None, raw, "qwen_local")
    assert n == CONTEXT and "max_model_len" in src
    with pytest.raises(MergeError, match="max_model_len"):
        served_context_len({"sha256": "x"}, {"semantic": {"backends": {}}}, "qwen_local")


def test_a_client_read_timeout_is_a_cut_answer_split_never_a_failed_stage():
    """2026-10-09: the merge call is non-streaming, so a client read bound cuts the answer
    before its token bound (600 s ≈ 15 k tokens on pccr's engine). A read timeout is treated
    like finish_reason 'length' — split, declared; a refused connection still fails."""
    import pytest
    import requests
    from segmentation.autoprompt.consolidate_prompts import MergeError

    class TimesOut(ByList):
        def chat(self, msgs, max_tokens=0, consumer=""):
            text = getattr(msgs[-1], "text", None) or str(msgs[-1])
            listed = [ln[2:] for ln in str(text).splitlines() if ln.startswith("- ")]
            if frozenset(listed) in self.cut:
                self.asked.append(listed)
                self.max_tokens.append(max_tokens)
                raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=600)")
            return super().chat(msgs, max_tokens=max_tokens, consumer=consumer)

    phrases = ["floor", "white floor", "tile", "ceramic tile"]
    vlm = TimesOut({
        frozenset({"ceramic tile", "floor"}): {"groups": [{"name": "floor", "parts": ["ceramic tile"]}]},
        frozenset({"floor", "tile", "white floor"}): {"groups": [
            {"name": "floor", "same_as": ["white floor"], "parts": ["tile"]}]},
    }, cut=[phrases])
    calls, said = [], []
    alias = merge_synonyms(vlm, "room", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=CONTEXT, log=said.append, calls=calls)
    assert alias == {"ceramic tile": "floor", "white floor": "floor", "tile": "floor"}
    top = calls[0]
    assert top["read_timeout"] is True and top["truncated"] is True and top["n_merged"] == 0
    assert top["split_into"] == {"halves": [2, 2], "survivors": 3}
    assert any("read timeout" in m and "CUT" in m for m in said)
    # body-read timeout wrapped by requests in a ConnectionError: also a cut
    from urllib3.exceptions import ReadTimeoutError
    assert CP._is_read_timeout(requests.exceptions.ConnectionError(
        ReadTimeoutError(None, "/v1/chat/completions", "Read timed out.")))

    class Refused:
        def chat(self, *a, **k):
            raise requests.exceptions.ConnectionError("Connection refused")
    with pytest.raises(MergeError, match="the VLM call failed"):
        merge_synonyms(Refused(), "x", ["red door", "red doors door"], _head_noun,
                       context_len=CONTEXT, log=lambda m: None)
    assert not CP._is_read_timeout(requests.exceptions.ConnectTimeout("connect timed out"))


def test_the_merge_client_has_no_read_timeout_and_the_config_is_untouched():
    """The merge's own client waits for the engine (its max_tokens is the bound); the
    ordinary client keeps service.request_timeout_s, and the semantic config — digested into
    the engine identity — is not changed."""
    import semantic.client as sc
    from semantic.semantic_config import load_semantic_config
    before = load_semantic_config()
    sc._clients.pop("qwen_local", None)
    sc._clients.pop("qwen_local#no_read_timeout", None)
    try:
        plain = sc.get_semantic_client(backend="qwen_local", consumer="t")
        merge = sc.get_semantic_client(backend="qwen_local", consumer="m", no_read_timeout=True)
        assert merge is not plain
        assert plain.backend._timeout == before["service"]["request_timeout_s"]
        assert merge.backend._timeout is None
        assert sc.get_semantic_client(backend="qwen_local", consumer="t") is plain
        assert load_semantic_config() == before
    finally:
        sc._clients.pop("qwen_local", None)
        sc._clients.pop("qwen_local#no_read_timeout", None)


def test_production_turns_it_on_with_one_pass_per_keyframe():
    import yaml, pathlib
    raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert raw["autoprompt"]["merge_synonyms"] is True
    v = raw["autoprompt"]["vlm_sampling"]
    assert v["all_keyframes"] is True and v["tile_rows"] == 1 and v["tile_cols"] == 1


def test_the_whole_list_goes_in_one_call_when_it_fits_and_kinds_merge():
    phrases = ["floor", "white tiled floor", "floor tiles", "desk", "black metal desk", "chair"]
    PHRASES.update(phrases)
    vlm = FakeVLM({"floor": {"groups": [{"name": "floor", "same_as": ["white tiled floor", "floor tiles"]},
                                        {"name": "desk", "same_as": ["black metal desk"]}]}})
    alias = merge_synonyms(vlm, "office", phrases, _head_noun, max_phrases_per_call=400,
                           context_len=CONTEXT, log=lambda m: None)
    assert len(vlm.asked) == 1 and set(vlm.asked[0]) == set(phrases)
    assert alias == {"white tiled floor": "floor", "floor tiles": "floor", "black metal desk": "desk"}
