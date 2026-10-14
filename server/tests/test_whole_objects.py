"""Objects as large as possible (USER 2026-10-09).

*"los objetos deben ser el más grande posible, si tengo mil baldosas que forman el piso el
objeto es piso no baldosa. Si tengo un desk no lo voy a segmentar en sus patas la tabla etc,
es un objeto único"*. pccr's Autosegment that night prompted SAM3 with 'tile' (the floor's
tiles), 'panel', 'frame', 'rod', 'tape', 'sheet', 'drywall' …: the SAM3 session for 'tile'
grew from 8 to 77.7 GB — every floor tile became an object — and the card ran out at
category 42 of 45. The rule lives in BOTH VLM passes:

  · the understanding prompt asks for ONE WHOLE OBJECT per entry, never its parts, the
    units its surface is made of, or its material, named by its most specific common name;
  · the merge pass folds a phrase naming a part / component / building unit / material of
    another LISTED object INTO that object — the whole's name survives, the part is
    recorded as merged (with its reason) in the census, and the census still closes.

CPU only; the VLM is a fake that answers through the real parsers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from PIL import Image

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from segmentation.autoprompt import consolidate_prompts as CP  # noqa: E402
from segmentation.autoprompt import scene_understanding as SU  # noqa: E402
from segmentation.autoprompt.scene_understanding import _head_noun  # noqa: E402


def _quiet(_m):
    pass


# the served context of the configured engine (the merge answer's bound is derived from it)
CONTEXT = yaml.safe_load((SERVER / "config.yaml").read_text())["semantic"]["backends"][
    "qwen_local"]["max_model_len"]


# ── the rule is IN the prompts that are sent ──────────────────────────────────────────

def test_the_understanding_prompt_asks_for_whole_objects_first():
    p = SU._PROMPT.lower()
    assert "1. whole objects, never their pieces" in p, "the rule is stated, and stated first"
    assert p.index("1. whole objects") < p.index("2. atomic") < p.index("3. one name") \
        < p.index("4. the envelope")
    assert "the largest physical unit a person would call one thing" in p
    assert "never its parts or components" in p
    assert "never the units a surface or structure is made of" in p and "tiles" in p
    assert "never a material or finish" in p and "drywall" in p
    assert "a floor made of a thousand tiles is one entry, 'floor'" in p
    assert "a cable lying on the floor) is its own entry" in p, \
        "a separate object touching another is not a part"
    # the name of ONE object, never its class (USER 2026-10-08, the segmentation rounds)
    assert "most specific common name" in p and "never 'furniture'" in p
    assert "common name of one whole object" in p
    # what stays: the schema, the ShapeR entry, only what is visible
    assert '"scene_type"' in p and '"objects"' in p and '"category"' in p and '"shape"' in p
    assert "'shape' entry is for a 3d shape generator" in p
    assert "only on what is visible" in p
    assert SU._PROMPT.startswith("Study this image and return JSON:\n")


def test_the_merge_prompt_folds_a_part_into_its_whole():
    text = CP._merge_prompt("room", ["floor", "tile"])
    low = text.lower()
    assert "2. part into whole" in low
    assert "is folded into that whole object" in low
    assert "the tiles of a floor or ceiling" in low
    assert "is not a part" in low and "a cable lying on the floor" in low
    assert "structure is never a part" in low
    # ONE LINE PER PHRASE since 2026-10-09 (the JSON-groups answer looped on pccr's 85 names)
    assert "exactly one line per phrase" in low and "<phrase> -> <target> | <same or part or keep>" in text
    assert "so does a part and its whole" not in low, "the old 'keep the part' rule is gone"
    assert "- floor\n- tile" in text
    assert "whole objects" in CP._MERGE_SYSTEM.lower()


def test_the_grouping_rule_is_versioned():
    """A vocabulary derived under the old rule is not reused (the stamp's 'grouping')."""
    assert SU.GROUPING_VERSION == "whole_objects_v3"


# ── the merge parser ─────────────────────────────────────────────────────────────────

class _MergeVLM:
    """One merge answer per call; records the phrases each call listed."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked: list[list[str]] = []

    def chat(self, msgs, max_tokens=0, consumer=""):
        text = getattr(msgs[-1], "text", None) or str(msgs[-1])
        self.asked.append([ln[2:] for ln in str(text).splitlines() if ln.startswith("- ")])
        ans = self.answers[min(len(self.asked) - 1, len(self.answers) - 1)]
        return SimpleNamespace(content=json.dumps(ans), finish_reason="stop",
                               usage={"completion_tokens": 10})


def test_a_part_is_folded_into_its_whole_and_a_separate_object_stays():
    phrases = ["floor", "tile", "ceiling", "ceiling tile", "wall", "drywall", "door",
               "door frame", "car", "wheel", "cable", "white wall"]
    vlm = _MergeVLM({"groups": [
        {"name": "floor", "same_as": [], "parts": ["tile"]},
        {"name": "ceiling", "parts": ["ceiling tile"]},
        {"name": "wall", "same_as": ["white wall"], "parts": ["drywall", "door"]},
        {"name": "door", "parts": ["door frame"]},
        {"name": "car", "parts": ["wheel", "an invented hubcap"]},
    ]})
    calls: list = []
    part_of: dict = {}
    alias = CP.merge_synonyms(vlm, "office", phrases, _head_noun, max_phrases_per_call=400,
                              context_len=CONTEXT, log=_quiet, calls=calls, part_of=part_of)
    assert len(vlm.asked) == 1 and vlm.asked[0] == phrases, "the whole list, one call"
    assert alias == {"tile": "floor", "ceiling tile": "ceiling", "white wall": "wall",
                     "drywall": "wall", "door frame": "door", "wheel": "car"}
    assert part_of == {"tile": "floor", "ceiling tile": "ceiling", "drywall": "wall",
                       "door frame": "door", "wheel": "car"}, "a synonym is not a fold"
    # the door filed as a PART of the wall survives (structure is never a part), recorded
    assert "door" not in alias and calls[0]["structural_kept"] == ["door"]
    assert calls[0]["n_merged"] == 6 and calls[0]["n_parts"] == 5
    # the cable lying on the floor was never claimed: it stays its own prompt
    survivors = [p for p in phrases if p not in alias]
    assert survivors == ["floor", "ceiling", "wall", "door", "car", "cable"]


def test_a_synonym_of_a_part_is_a_part_whatever_the_answer_order():
    phrases = ["floor", "tile", "ceramic tile", "cable"]
    for groups in (
            [{"name": "tile", "same_as": ["ceramic tile"]},
             {"name": "floor", "parts": ["tile"]}],
            # the group of an already-folded name: its members follow it to the whole
            [{"name": "floor", "parts": ["tile"]},
             {"name": "tile", "same_as": ["ceramic tile"]}]):
        part_of: dict = {}
        alias = CP.merge_synonyms(_MergeVLM({"groups": groups}), "room", phrases, _head_noun,
                                  max_phrases_per_call=400, context_len=CONTEXT, log=_quiet,
                                  part_of=part_of)
        assert alias == {"tile": "floor", "ceramic tile": "floor"}, groups
        assert part_of == {"tile": "floor", "ceramic tile": "floor"}, groups


def test_a_part_of_a_part_reaches_the_whole_and_a_cycle_is_refused():
    phrases = ["car", "wheel", "tyre", "cable"]
    part_of: dict = {}
    alias = CP.merge_synonyms(_MergeVLM({"groups": [
        {"name": "wheel", "parts": ["tyre"]},
        {"name": "car", "parts": ["wheel"]},
        {"name": "wheel", "parts": ["car"]},          # would close a cycle: refused
    ]}), "street", phrases, _head_noun, max_phrases_per_call=400, context_len=CONTEXT,
        log=_quiet, part_of=part_of)
    assert alias == {"tyre": "car", "wheel": "car"} and part_of == alias


def test_the_merge_answer_without_parts_still_parses_as_before():
    """An answer of the 2026-09-29 schema (no 'parts') merges synonyms only."""
    part_of: dict = {}
    alias = CP.merge_synonyms(_MergeVLM({"groups": [{"name": "wall", "same_as": ["white wall"]}]}),
                              "room", ["wall", "white wall"], _head_noun,
                              max_phrases_per_call=400, context_len=CONTEXT, log=_quiet,
                              part_of=part_of)
    assert alias == {"white wall": "wall"} and part_of == {}


# ── through the real auto-prompter: the census records the fold and closes ──────────

FLOOR = {"category": "floor", "description": "grey concrete",
         "shape": {"form": "flat horizontal surface", "material": "concrete", "detail": "wide"}}
TILE = {"category": "tile", "description": "white square ceramic",
        "shape": {"form": "thin square plate", "material": "ceramic", "detail": "glazed"}}
CABLE = {"category": "cable", "description": "black rubber"}


class _SceneVLM:
    """Understanding calls name the floor AND its tiles (the frames disagree on the
    plural); the merge call folds the tile into the floor."""

    def __init__(self):
        self.n_understand = 0
        self.merge_listing: list[str] = []

    def chat(self, messages, max_tokens=None, consumer=None):
        if consumer == "phase1.merge_synonyms":
            text = getattr(messages[-1], "text", "") or ""
            self.merge_listing = [ln[2:] for ln in text.splitlines() if ln.startswith("- ")]
            # short questions per phrase since 2026-10-09 (classify_phrases): whole or piece,
            # then of which — answered the way Qwen3-VL 8B writes ('tile: floor')
            q = text.split("\n\n")[-1]
            asked = q.split('"')[1]
            if "WHOLE object on its own" in q:
                ans = "piece" if asked == "tile" else "whole"
            elif "Answer yes or no" in q:
                ans = "yes"
            elif "piece, unit or material of which" in q:
                ans = "tile: floor"
            else:
                ans = "none"
            return SimpleNamespace(content=ans, finish_reason="stop",
                                   usage={"completion_tokens": 3})
        self.n_understand += 1
        tile = dict(TILE, category="tiles") if self.n_understand == 2 else TILE
        return SimpleNamespace(content=json.dumps({
            "scene_type": "hall", "summary": "s", "objects": [FLOOR, tile, CABLE]}),
            finish_reason="stop", usage={"completion_tokens": 50, "prompt_tokens": 800})


def _session(tmp_path, n=6):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    files = []
    for i in range(n):
        fn = f"{i * 7:06d}.jpg"
        Image.new("RGB", (64, 36), (i, i, i)).save(frames / fn)
        files.append(fn)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    raw["autoprompt"]["consolidate_prompts"] = False
    raw["autoprompt"]["merge_synonyms"] = True
    raw["autoprompt"]["vlm_sampling"].update(all_keyframes=False, spacing_kf=3, tile_rows=1,
                                             tile_cols=1, max_calls=300)
    raw["reconstruction"]["simple"]["enabled"] = True
    return raw


def test_the_autoprompter_folds_the_tile_into_the_floor_and_the_census_closes(
        tmp_path, monkeypatch):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    from segmentation.census import build_census
    raw = _session(tmp_path)
    vlm = _SceneVLM()
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: vlm)
    res = AutoPrompter(tmp_path, tmp_path / "output", config=raw).run()

    # SAM3 receives the whole object, never the part
    prompts = res.prompt.split(";")
    assert prompts == ["floor", "cable"]
    assert "tile" in vlm.merge_listing and "floor" in vlm.merge_listing
    doc = json.loads(Path(res.vlm_analysis_path).read_text())
    assert doc["scene_understanding"]["grouping"] == "whole_objects_v3"
    assert "grouping" in doc["stamp"]["config"], "the rule's version is a stamped section"

    # the census: both spellings of the part are MERGED into the floor, with the reason
    fates = {c["concept"]: c for c in doc["census"]["concepts"]}
    assert set(fates) == {"floor", "tile", "tiles", "cable"}
    for part in ("tile", "tiles"):
        c = fates[part]
        assert c["fate"] == "merged" and c["prompt"] == "floor" and c["part_of"] == "floor"
        assert "part, component, building unit or material of 'floor'" in c["reason"]
        assert "2026-10-09" in c["reason"]
    assert fates["floor"]["fate"] == fates["cable"]["fate"] == "prompt"
    assert "part_of" not in fates["floor"] and "part_of" not in fates["cable"]
    mc = doc["census"]["merge_calls"]
    # one record per phrase (classify_phrases, 2026-10-09): the tile folded into the floor
    assert [(c["phrase"], c.get("into")) for c in mc if c["decision"] == "part"] == [("tile", "floor")]

    # the part is never retried as its whole, and never describes it
    for fb in doc["fallback_prompts"].get("floor", []):
        assert "tile" not in fb and "ceramic" not in fb, fb
    assert doc["fallback_prompts"]["floor"] == ["grey concrete floor"]
    assert doc["shape_descriptions"]["floor"]["shape"] == "flat horizontal surface"
    assert doc["shape_descriptions"]["floor"]["material"] == "concrete"

    # the session's vocabulary record names the fold
    rec = json.loads((tmp_path / "output" / "autoprompt_concepts.json").read_text())
    assert rec["synonyms"] == {"tile": "floor"} and rec["parts_folded"] == {"tile": "floor"}
    assert rec["prompts"] == ["floor", "cable"]

    # the accounting closes: every proposed concept is a prompt, merged, or not prompted
    census = build_census(tmp_path / "output", prompt=res.prompt, vlm_doc=doc, gap_kf=1,
                          frames_dir=tmp_path / "frames", log=_quiet)
    acc = census["accounting"]
    assert acc["closed"] is True
    assert acc["by_fate"] == {"prompt": 2, "merged": 2, "not_prompted": 0, "unaccounted": 0}
    floor_row = next(r for r in census["prompts"] if r["prompt"] == "floor")
    assert set(floor_row["raw_concepts"]) == {"floor", "tile", "tiles"}


def test_a_leftover_merge_max_tokens_fails_the_load(tmp_path):
    """The merge answer's bound is no key since 2026-10-09 (the served context minus the
    prompt): a configuration still carrying it fails naming it, before any engine."""
    import pytest
    from segmentation.autoprompt.session_builder import AutoPrompter
    raw = _session(tmp_path)
    assert "merge_max_tokens" not in raw["autoprompt"]
    raw["autoprompt"]["merge_max_tokens"] = 8192
    with pytest.raises(KeyError, match="merge_max_tokens.*REMOVED"):
        AutoPrompter(tmp_path, tmp_path / "output", config=raw)
