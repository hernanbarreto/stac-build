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


def test_only_the_SAME_NAME_folds_and_every_fold_is_recorded():
    """2026-09-15 this collapsed every phrase sharing a HEAD NOUN into one, and
    that silently deleted objects: on pccr 2026-09-29 'cardboard box' vanished
    into 'red fire alarm box' (head 'box') and was never written anywhere. USER
    2026-09-29: "debe segmentar todo, absolutamente preciso y completo" — only
    true synonyms fold now (case, plural, article, punctuation), each fold is
    recorded for the census, and two DIFFERENT names are grouped by the
    consolidation pass and settled on the cloud (segmentation.dedupe_overlap)."""
    frames = [
        FrameUnderstanding(1, "server room", "", ["white tiled floor", "cardboard box"]),
        FrameUnderstanding(2, "server room", "", ["white tiled floors", "red fire alarm box"]),
        FrameUnderstanding(3, "server room", "", ["white tiled floor with dark grout"]),
    ]
    und = aggregate(frames)
    assert "cardboard box" in und.objects and "red fire alarm box" in und.objects
    assert "white tiled floor with dark grout" in und.objects
    assert len([o for o in und.objects if o.startswith("white tiled floor")]) == 2
    assert und.merged == {"white tiled floors": "white tiled floor"}
    assert set(und.objects) | set(und.merged) == {o for f in frames for o in f.objects}, \
        "a phrase disappeared without a record"


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
    assert "'with'" in p or '"with"' in p
    assert "two things are two entries" in p


def test_the_prompts_carry_no_example_from_any_scene():
    """USER 2026-09-30: "no es un prompt genérico, es un prompt hecho para pccr" — the
    prompts state RULES; object names from one scene bias the VLM on every other one."""
    from segmentation.autoprompt.consolidate_prompts import _merge_prompt, _prompt
    texts = [_PROMPT.lower(), _merge_prompt("room", ["a"]).lower(), _prompt("room", ["a"]).lower()]
    for w in ("desk", "table", "server rack", "fire extinguisher", "backpack", "office chair",
              "white tiled floor", "doorway in"):
        assert not any(w in t for t in texts), w


def test_the_prompt_asks_for_one_name_per_object():
    p = _PROMPT.lower()
    assert "one name per object" in p
    assert "same physical thing twice" in p


def test_the_frames_the_vlm_sees_span_the_WHOLE_walk_not_a_count():
    """What the VLM never sees it cannot name, and in the SIMPLE pipeline these
    phrases ARE the SAM3 prompts — so a frame left out is an object left out of
    the segmentation. pccr showed 8 frames picked by linspace across the walk
    and the main door was in none of them. The coverage cover (2026-09-16)
    cannot measure at the INTAKE, where the VLM runs since 2026-09-28 — there is
    no cloud of this run — so pccr 2026-09-29 fell back to the same 8. The
    frames are now spread uniformly along the walk at a declared density
    (autoprompt.vlm_sampling; tests/test_vlm_sampling.py pins the rule); the
    cover stays selectable for a session that has a cloud."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    src = (root / "segmentation" / "autoprompt" / "session_builder.py").read_text()
    assert "plan_vlm_frames(" in src and "walk_chainage(" in src
    assert "if self.understand_cover:" in src
    assert "understand_sample" not in src, "the fixed-count fallback came back"
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    assert cfg["autoprompt"]["understand_cover"] is False
    assert "understand_sample" not in cfg["autoprompt"]
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
