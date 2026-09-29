"""The second VLM pass fuses the names that mean the same thing (USER 2026-09-29),
per head-noun family, before the SAM3 bound; different colour/kind never merges
unless the VLM says so, and phrases of different heads are never asked together."""
from types import SimpleNamespace
import json

from segmentation.autoprompt.consolidate_prompts import merge_synonyms
from segmentation.autoprompt.scene_understanding import _head_noun


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
    alias = merge_synonyms(vlm, "office", phrases, _head_noun, log=lambda m: None)
    assert alias == {"white painted wall": "white wall", "black office desk": "black desk"}
    # one call per family with 2+ phrases; singletons never asked; families never mixed
    heads = [{_head_noun(p) for p in a} for a in vlm.asked]
    assert all(len(h) == 1 for h in heads) and len(vlm.asked) == 2


def test_a_broken_answer_keeps_the_family():
    class Bad:
        def chat(self, *a, **k):
            return SimpleNamespace(content="no json")
    assert merge_synonyms(Bad(), "x", ["red door", "red doors door"], _head_noun, log=lambda m: None) == {}


def test_production_turns_it_on_with_one_pass_per_keyframe():
    import yaml, pathlib
    raw = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert raw["autoprompt"]["merge_synonyms"] is True
    v = raw["autoprompt"]["vlm_sampling"]
    assert v["all_keyframes"] is True and v["tile_rows"] == 1 and v["tile_cols"] == 1
