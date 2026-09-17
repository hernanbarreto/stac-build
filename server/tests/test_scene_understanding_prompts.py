"""The prompts SAM3 receives are the segments it will produce (USER 2026-09-15).

In the SIMPLE pipeline the VLM's whole job is scene understanding: the phrases
it writes go STRAIGHT to SAM3 as separate text prompts, one segment each. So a
compound phrase asks the segmenter for a mask spanning two different things
("son objetos compuestos, eso no está bien"), and a second name for the same
thing asks for a duplicate of it.

The cases below are verbatim from the pccr run of 2026-09-15 that exposed both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt.scene_understanding import (  # noqa: E402
    _PROMPT, _head_noun, aggregate, FrameUnderstanding)


# ── the head noun is the OBJECT, not what hangs off it ───────────────────

def test_a_trailing_prepositional_phrase_does_not_steal_the_head():
    """pccr: every one of these keyed on the attached object, so the concepts
    never merged and each phrasing became its own SAM3 prompt."""
    assert _head_noun("black computer desk with monitor") == "desk"
    assert _head_noun("workstation desk with computer") == "desk"
    assert _head_noun("white tiled floor with dark grout") == "floor"
    assert _head_noun("metal server rack with glass doors") == "rack"
    assert _head_noun("black server rack with open shelves") == "rack"


def test_three_phrasings_of_one_floor_collapse_to_one_prompt():
    """The floor arrived under three names from three keyframes and all three
    survived into the prompt list — three SAM3 sessions over one surface."""
    frames = [
        FrameUnderstanding(1, "server room", "", ["white tiled floor"]),
        FrameUnderstanding(2, "server room", "", ["white tiled floor"]),
        FrameUnderstanding(3, "server room", "", ["white tiled floor with dark grout"]),
    ]
    objs = aggregate(frames).objects
    assert len([o for o in objs if "floor" in o]) == 1


def test_an_honest_phrase_keeps_its_own_head():
    """The cut must not eat a phrase that never had a preposition."""
    assert _head_noun("concrete support columns") == "column"
    assert _head_noun("overhead fluorescent light fixture") == "fixture"
    assert _head_noun("white painted wall") == "wall"
    assert _head_noun("stairs") == "stair"


def test_and_is_not_a_cut_point_because_it_joins_adjectives():
    """'white and gray checkered floor tiles' is ONE object described by two
    colours. Cutting at 'and' would key it on 'white'."""
    assert _head_noun("white and gray checkered floor tiles") == "tile"


def test_a_leading_preposition_is_not_a_cut_point():
    """Nothing sensible starts with one, and cutting at index 0 would leave the
    phrase empty."""
    assert _head_noun("on ramp") == "ramp"


# ── the rules have to be IN the prompt that is actually sent ─────────────

def test_the_prompt_forbids_compounds():
    """`detector.py` carries the same rules and is NEVER reached in the SIMPLE
    pipeline — editing it alone changed nothing on pccr. This asserts the rule
    is in the prompt scene_understanding actually sends."""
    p = _PROMPT.lower()
    assert "atomic" in p
    assert "desk with computer" in p, "the compound rule lost its example"
    assert "'with'" in p or '"with"' in p


def test_the_prompt_asks_for_one_name_per_object():
    p = _PROMPT.lower()
    assert "one name per object" in p
    assert "same physical thing twice" in p


def test_the_frames_the_vlm_sees_are_a_COVER_not_a_count():
    """What the VLM never sees it cannot name, and in the SIMPLE pipeline these
    phrases ARE the SAM3 prompts — so a frame left out is an object left out of
    the segmentation. pccr showed 8 frames picked by linspace across a
    216-keyframe walk and the main door was in none of them. The frames are now
    chosen by COVERAGE of the scene, so the count comes out of the place: 67
    for pccr, and a small room would need a handful."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    src = (root / "segmentation" / "autoprompt" / "session_builder.py").read_text()
    assert "from .coverage_sample import cover_keyframes" in src
    assert "if self.understand_cover:" in src
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    assert cfg["autoprompt"]["understand_cover"] is True
    assert 0.0 < float(cfg["autoprompt"]["understand_cover_overlap"]) <= 1.0


def test_the_prompt_demands_the_structural_envelope():
    """USER: 'puertas, ventanas, paredes, pisos, techos, columnas, eso es
    estructural, debe estar'. Qwen had skipped the main door entirely."""
    p = _PROMPT.lower()
    assert "envelope is not background" in p
    for part in ("floor", "ceiling", "wall", "door", "window", "column",
                 "beam", "stair"):
        assert part in p, f"the envelope rule stopped naming {part}"
    assert "shut and flush" in p, \
        "the rule no longer says a closed door is still an object"
