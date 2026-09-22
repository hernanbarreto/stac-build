"""The second VLM pass: one object, one prompt (USER 2026-09-15).

*"podríamos tener dos o tres pasadas de vllm para fusionar palabras y prompts"*.

Scene understanding runs frame by frame and no frame sees the others' answers,
so the union carries one object under several names plus entries that are only
PARTS of another. Each phrase becomes one SAM3 session and one segment: on pccr
84 phrases became 149 fragmented instances, and the certification then matched
fragments of one label as revisits, measuring a 336 cm closure over a 3.2 m
walk and ending in a −1205 % regression.

What is asserted here is the SAFETY of the pass, because its failure mode is
silent: a dropped object is an object that never gets segmented and nobody
notices. Nothing may be lost that the model did not explicitly account for, and
the structural envelope may never be dropped at all.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt.consolidate_prompts import (  # noqa: E402
    consolidate, is_structural)

PCCR = ["white tiled floor", "checkered floor tiles", "white tile floor grout",
        "black server rack", "server rack fan", "server rack hard drive",
        "dark wooden door", "doorway", "red fire extinguisher"]


class _Reply:
    def __init__(self, content):
        self.content = content


class _Client:
    """Answers with whatever JSON the test hands it, once per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        r = self.replies[min(self.calls - 1, len(self.replies) - 1)]
        self.prompt = messages[-1].content if hasattr(messages[-1], "content") else ""
        return _Reply(r if isinstance(r, str) else json.dumps(r))


def _quiet(_m):
    pass


# ── what the pass is for ─────────────────────────────────────────────────

def test_synonyms_collapse_and_parts_are_folded_in():
    answer = {"objects": [
        {"name": "white tiled floor",
         "aliases": ["checkered floor tiles", "white tile floor grout"], "parts": []},
        {"name": "black server rack", "aliases": [],
         "parts": ["server rack fan", "server rack hard drive"]},
        {"name": "dark wooden door", "aliases": ["doorway"], "parts": []},
        {"name": "red fire extinguisher", "aliases": [], "parts": []},
    ]}
    res = consolidate(_Client(answer), "server room", PCCR, log=_quiet)
    assert res.objects == ["white tiled floor", "black server rack",
                           "dark wooden door", "red fire extinguisher"]
    assert res.merged["white tiled floor"] == ["checkered floor tiles",
                                               "white tile floor grout"]
    assert res.parts["black server rack"] == ["server rack fan",
                                              "server rack hard drive"]


def test_it_stops_when_a_pass_changes_nothing():
    """Convergence, the same shape as the greedy correction loop — so there is
    no 'right' number of passes to pick."""
    answer = {"objects": [{"name": p, "aliases": [], "parts": []} for p in PCCR]}
    c = _Client(answer)
    res = consolidate(c, "server room", PCCR, max_passes=3, log=_quiet)
    assert res.objects == PCCR
    assert c.calls == 1, "a pass that changed nothing must not run again"


def test_it_keeps_going_while_the_list_still_shrinks():
    first = {"objects": [{"name": "white tiled floor",
                          "aliases": ["checkered floor tiles"], "parts": []}]
             + [{"name": p, "aliases": [], "parts": []}
                for p in PCCR if p not in ("white tiled floor", "checkered floor tiles")]}
    second = {"objects": [{"name": "white tiled floor",
                           "aliases": ["white tile floor grout"], "parts": []}]
              + [{"name": p, "aliases": [], "parts": []}
                 for p in PCCR if p not in ("white tiled floor", "checkered floor tiles",
                                            "white tile floor grout")]}
    c = _Client(first, second, second)
    res = consolidate(c, "server room", PCCR, max_passes=3, log=_quiet)
    assert "checkered floor tiles" not in res.objects
    assert "white tile floor grout" not in res.objects
    assert c.calls >= 2


# ── the safety, which is the whole point ─────────────────────────────────

def test_a_phrase_the_model_never_mentions_survives():
    """The silent failure: an object dropped here is an object that never gets
    segmented, and nothing downstream can tell it was ever proposed."""
    answer = {"objects": [{"name": "white tiled floor", "aliases": [], "parts": []}]}
    res = consolidate(_Client(answer), "server room", PCCR, log=_quiet)
    for p in PCCR:
        assert p in res.objects, f"{p} was lost without the model ever naming it"
    assert set(res.kept_unmentioned) == set(PCCR) - {"white tiled floor"}


def test_the_envelope_is_never_dropped_as_somebody_s_part():
    """USER: 'puertas, ventanas, paredes, pisos, techos, columnas, eso es
    estructural, debe estar'. A door filed as a part of a wall would delete it,
    so the rule is enforced in code, not trusted to the prompt."""
    answer = {"objects": [
        {"name": "black server rack", "aliases": [],
         "parts": ["dark wooden door", "doorway", "white tiled floor"]},
    ]}
    res = consolidate(_Client(answer), "server room", PCCR, log=_quiet)
    for part in ("dark wooden door", "doorway", "white tiled floor"):
        assert part in res.objects
        assert part in res.kept_structural
    assert res.parts.get("black server rack") is None


def test_a_non_structural_part_really_is_dropped():
    """The rescue must not defeat the pass: a rack fan is not structure."""
    answer = {"objects": [{"name": "black server rack", "aliases": [],
                           "parts": ["server rack fan"]}]}
    res = consolidate(_Client(answer), "server room",
                      ["black server rack", "server rack fan"], log=_quiet)
    assert res.objects == ["black server rack"]
    assert res.parts["black server rack"] == ["server rack fan"]


def test_an_invented_phrase_is_ignored():
    """The model may only regroup the words it was given; a name it made up
    would silently replace a real prompt."""
    answer = {"objects": [{"name": "the floor of the room", "aliases": [], "parts": []}]}
    res = consolidate(_Client(answer), "server room", PCCR, log=_quiet)
    assert "the floor of the room" not in res.objects
    assert res.objects == PCCR


def test_unparseable_output_leaves_the_list_alone():
    res = consolidate(_Client("I think the floor and the walls are the same."),
                      "server room", PCCR, log=_quiet)
    assert res.objects == PCCR
    assert res.passes == 0


def test_a_failing_client_leaves_the_list_alone():
    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("semantic service down")

    res = consolidate(_Boom(), "server room", PCCR, log=_quiet)
    assert res.objects == PCCR
    assert res.passes == 0


def test_the_same_phrase_claimed_twice_is_only_counted_once():
    answer = {"objects": [
        {"name": "white tiled floor", "aliases": ["checkered floor tiles"], "parts": []},
        {"name": "black server rack", "aliases": ["checkered floor tiles"], "parts": []},
    ]}
    res = consolidate(_Client(answer), "server room", PCCR, log=_quiet)
    assert res.objects.count("checkered floor tiles") == 0
    assert res.merged["white tiled floor"] == ["checkered floor tiles"]
    assert "checkered floor tiles" not in res.merged.get("black server rack", [])


# ── the envelope test itself ─────────────────────────────────────────────

def test_is_structural_covers_what_the_user_named():
    for p in ("white tiled floor", "exposed ceiling", "white wall",
              "dark wooden door", "doorway", "glass window", "metal support column",
              "exposed ceiling beams", "concrete stairs", "wall opening",
              "doorway to server room", "dark door at end"):
        assert is_structural(p), p
    for p in ("black server rack", "red fire extinguisher", "server rack fan",
              "black office chair", "computer monitor"):
        assert not is_structural(p), p


def test_the_ironmongery_of_a_door_is_not_the_door():
    """Measured on pccr 2026-09-15: the first version tested for a structural
    WORD anywhere in the phrase, so 'metal door handle' and 'door lock
    mechanism' were rescued from 'parts' and survived as two prompts of their
    own. The test is the HEAD noun — what the phrase actually names."""
    for p in ("metal door handle", "door lock mechanism", "metal window frame",
              "glass door server cabinet", "stacked drywall sheets"):
        assert not is_structural(p), p


def test_the_pass_groups_but_never_removes_a_prompt():
    """USER 2026-09-22: *"tampoco segmento una puerta, de entrada una locura"*.

    The pass still runs — its grouping is what labels the scene and what the
    report shows — but it may not decide, from the WORDS alone and before
    anything has been segmented, that two names are one object. On pccr
    2026-09-22 it folded `doorway` into `glass door` and `desk` into `white
    workbench`, and four of its five merges differed by COLOUR in their own
    names (`white server cabinet` into `black server rack`, `red painted wall
    section` into `white wall`). Neither object was ever segmented. Identity is
    settled downstream, on the geometry.
    """
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "autoprompt" / "session_builder.py").read_text()
    assert "from .consolidate_prompts import consolidate" in src
    assert "consolidation = consolidate(" in src, "the pass must still run"
    assert "phrases = consolidation.objects" not in src, \
        "the consolidated list may NOT replace the prompts — it groups them"
    i = src.index("consolidation = consolidate(")
    j = src.index("_grouped = list(consolidation.objects)", i)
    assert j > i, "the grouping must be kept for labelling and for the report"


def test_the_vocabulary_is_recorded_and_reusable():
    """USER 2026-09-22: *"no debe cambiar en silencio, debe ser lo mas
    determinista posible, debe ser reproducible"*. The engine cannot promise the
    same tokens twice, so the guarantee is the artifact."""
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "autoprompt" / "session_builder.py").read_text()
    assert "autoprompt_concepts.json" in src, "the session must record its vocabulary"
    assert "reuse_vocabulary" in src, "and be able to reuse it"
    i = src.index("if _reused:")
    j = src.index("if self.consolidate_prompts", i)
    assert i < j, "a reused vocabulary must short-circuit the consolidation"
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert cfg["autoprompt"]["reuse_vocabulary"] is True
    assert cfg["semantic"]["generation"]["temperature"] == 0.0
    assert cfg["semantic"]["generation"].get("seed") is not None, \
        "the sampler must be pinned too"


def test_the_config_carries_the_switch_and_the_bound():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert cfg["autoprompt"]["consolidate_prompts"] is True
    assert int(cfg["autoprompt"]["consolidate_passes"]) >= 1
