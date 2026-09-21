"""The second VLM pass: one object, one prompt (USER 2026-09-15) — and, since
2026-09-21, only the merges the code cannot be wrong about.

*"podríamos tener dos o tres pasadas de vllm para fusionar palabras y prompts"*
(2026-09-15). Scene understanding runs frame by frame and no frame sees the
others' answers, so the union carries one object under several names plus
entries that are only PARTS of another. Each phrase becomes one SAM3 session
and one segment: on pccr 84 phrases became 149 fragmented instances, and the
certification then matched fragments of one label as revisits, measuring a
336 cm closure over a 3.2 m walk and ending in a −1205 % regression.

*"el detalle de vlm sobre la escena debe ser altamente detallado, como no va a
haber desk, incluso hay muchos de ellos… ojo con eso no arruines lo que
funcionaba"* (2026-09-21). The 15:30 run of that day folded 'desk', 'white
table' and 'computer workstation' into 'white workbench' as ALIASES and the
rack's internals into 'black server rack'. Those objects ARE the evidence the
correction measures closures on: 61 raw concepts gave 5 closures and recovered
10.2 % of depth drift, 34 concepts gave 2 and recovered 2.2 %.

So what is asserted here is, in order:
  · the pass still merges what is literally one name written twice;
  · it never again merges two different objects, and never removes a part;
  · nothing is lost that the model did not explicitly account for, the
    structural envelope may never be dropped at all, and everything the pass
    proposed and could not apply is RECORDED with its reason.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt.consolidate_prompts import (  # noqa: E402
    MERGE_RULES, consolidate, is_structural, load_merge_rule, same_name)

PCCR = ["white tiled floor", "checkered floor tiles", "white tile floor grout",
        "black server rack", "server rack fan", "server rack hard drive",
        "dark wooden door", "doorway", "red fire extinguisher"]

# The four phrases of the 2026-09-21 15:30 run that the pass turned into one.
DESKS = ["white workbench", "white table", "desk", "computer workstation",
         "black metal table legs"]

# One name written twice — the merge that must still happen — plus the third
# phrasing that no longer joins it. 'exposed ceiling with ducts' heads on
# 'ceiling' once its trailing prepositional phrase is cut, so since 2026-09-21
# it survives as its own prompt: the declared COST of reading a head noun by
# ONE reading, and the cheap side of the trade (see `_head_stem`).
DUCTS = ["exposed ceiling ducts", "exposed ceiling ductwork",
         "exposed ceiling with ducts", "exposed ceiling grid"]

# Two literal restatements in one list — the collective suffix and a plural.
# What the pass may still apply is exactly this shape and nothing wider.
RESTATED = ["exposed ceiling ducts", "exposed ceiling ductwork",
            "server rack", "server racks", "exposed ceiling grid"]

# The nine internals of pccr's rack, the ones the user praised seeing.
RACK = ["black server rack", "server rack label", "server rack door handle",
        "server rack cable bundle", "server rack hard drive bay",
        "server rack network switch", "server rack power supply unit",
        "server rack mounting bracket", "server rack cable management",
        "server rack shelf"]


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


def _answer(groups: dict) -> dict:
    """{name: (aliases, parts)} → the JSON the model would have returned."""
    return {"objects": [{"name": n, "aliases": list(a), "parts": list(p)}
                        for n, (a, p) in groups.items()]}


def _refusal(res, phrase: str) -> dict:
    hits = [r for r in res.refused if r["phrase"] == phrase]
    assert len(hits) == 1, f"{phrase} is not recorded exactly once: {res.refused}"
    return hits[0]


# ── what the pass is still for ───────────────────────────────────────────

def test_one_name_written_twice_is_still_merged_and_the_third_way_survives():
    """'exposed ceiling ductwork' IS 'exposed ceiling ducts': one head noun up
    to the declared collective suffix, and the content words are the same set.
    That merge is the whole reason the pass still applies anything at all.

    'exposed ceiling with ducts' no longer joins them, and that is the cost the
    2026-09-21 hardening bought ON PURPOSE. `_head_stem` reads a phrase by ONE
    reading — cut at its trailing preposition — so this one heads on 'ceiling',
    not on 'duct'. The loose second reading that used to catch it is the very
    one that merged 'wall' into 'yellow arrow sign on wall'; one extra prompt
    for one object is the cheap side of that trade. Nothing is lost in silence:
    the phrase stays a prompt and the proposal is recorded against it.
    """
    res = consolidate(_Client(_answer({
        "exposed ceiling ducts": (["exposed ceiling ductwork",
                                   "exposed ceiling with ducts"], []),
        "exposed ceiling grid": ([], []),
    })), "server room", DUCTS, rule="literal", log=_quiet)
    assert res.objects == ["exposed ceiling ducts", "exposed ceiling with ducts",
                           "exposed ceiling grid"]
    assert res.merged["exposed ceiling ducts"] == ["exposed ceiling ductwork"]
    r = _refusal(res, "exposed ceiling with ducts")
    assert r["into"] == "exposed ceiling ducts" and r["as"] == "alias"
    assert "different head nouns (ceiling vs duct)" in r["reason"]


def test_a_qualified_name_is_not_the_bare_one_restated():
    """The modifier test, on its own, and the property it acquired on
    2026-09-21: the content words must be the SAME SET, never a subset.

    It used to accept a subset, on the reading that 'white tiled floor' is
    'tiled floor' said with more detail. USER, the same day: *"el detalle de
    vlm sobre la escena debe ser altamente detallado, como no va a haber desk,
    incluso hay muchos de ellos, debe ser altamente detallado ojo con eso no
    arruines lo que funcionaba"*. When several of a thing exist the bare word
    names none of them in particular, so a subset restates a TYPE and never an
    IDENTITY — and folding the qualified name into the bare one costs exactly
    the objects the correction measures its closures on (61 raw concepts gave
    5 closures and 10.2 % of depth drift, 34 gave 2 and 2.2 %).

    A contradiction was always refused and still is; the same set written twice
    is still one name, and that is all that is left to merge.
    """
    ok, why = same_name("tiled floor", "white tiled floor")
    assert not ok and "a qualified name is not the bare one restated" in why
    ok, why = same_name("white table", "black table")
    assert not ok and "the words differ (white vs black)" in why
    ok, why = same_name("tiled floors", "tiled floor")
    assert ok and "same name restated" in why


def test_it_stops_when_a_pass_changes_nothing():
    """Convergence, the same shape as the greedy correction loop — so there is
    no 'right' number of passes to pick."""
    answer = {"objects": [{"name": p, "aliases": [], "parts": []} for p in PCCR]}
    c = _Client(answer)
    res = consolidate(c, "server room", PCCR, max_passes=3, rule="literal",
                      log=_quiet)
    assert res.objects == PCCR
    assert c.calls == 1, "a pass that changed nothing must not run again"


def test_it_keeps_going_while_the_list_still_shrinks():
    """Two literal restatements the model only notices one at a time: the pass
    runs again while the list still CHANGES and stops when one changes nothing.
    Both pairs are merges the hardened rule still applies — 'ductwork' is the
    declared collective suffix of 'ducts', 'server racks' is a plural — so the
    convergence is exercised without asking for a merge that no longer happens.
    """
    first = _answer({"exposed ceiling ducts": (["exposed ceiling ductwork"], []),
                     "server rack": ([], []),
                     "server racks": ([], []),
                     "exposed ceiling grid": ([], [])})
    second = _answer({"server rack": (["server racks"], []),
                      "exposed ceiling ducts": ([], []),
                      "exposed ceiling grid": ([], [])})
    c = _Client(first, second, second)
    res = consolidate(c, "server room", RESTATED, max_passes=3, rule="literal",
                      log=_quiet)
    assert res.objects == ["server rack", "exposed ceiling ducts",
                           "exposed ceiling grid"]
    assert res.merged == {"exposed ceiling ducts": ["exposed ceiling ductwork"],
                          "server rack": ["server racks"]}
    assert res.refused == []
    assert c.calls >= 2


# ── 2026-09-21: what it may no longer do ─────────────────────────────────

def test_the_desks_of_the_room_are_not_one_workbench():
    """USER: "como no va a haber desk, incluso hay muchos de ellos". A desk, a
    white table and a computer workstation are three objects, and every one of
    them is evidence the correction can measure a closure on."""
    res = consolidate(_Client(_answer({
        "white workbench": (["white table", "desk", "computer workstation"],
                            ["black metal table legs"]),
    })), "server room", DESKS, rule="literal", log=_quiet)
    for phrase in DESKS:
        assert phrase in res.objects, f"{phrase} was merged away"
    assert res.merged == {}
    for phrase in ("white table", "desk", "computer workstation"):
        r = _refusal(res, phrase)
        assert r["into"] == "white workbench" and r["as"] == "alias"
        assert "different head nouns" in r["reason"]


def test_the_wall_is_not_a_sign_hanging_on_it():
    """DEFECT 1 of the three measured on 2026-09-21, and the worst of them.

    `_head_stem` used to read a phrase TWICE — cut at its trailing preposition
    ('sign') AND uncut ('wall') — so 'yellow arrow sign on wall' shared a head
    with 'wall', whose single content word is trivially a subset, and
    `same_name('wall', 'yellow arrow sign on wall')` returned MERGE. The wall
    then left the list as an alias of the sign. Aliases do not travel through
    the parts door, which is where `is_structural` rescues the envelope, so
    the deletion happened with NO refusal recorded and nothing on disk said
    the room had lost a wall — against the user's own sentence: "puertas
    ventanas paredes pisos techos columnas, eso es estructural, debe estar".

    Any 'X <preposition> Y' phrase shared a head with anything headed Y, so
    the defect was general; the wall is simply the case that cost the most.
    """
    ok, why = same_name("wall", "yellow arrow sign on wall")
    assert not ok and "different head nouns (wall vs sign)" in why
    listing = ["wall", "yellow arrow sign on wall", "white tiled floor"]
    res = consolidate(_Client(_answer({
        "yellow arrow sign on wall": (["wall"], []),
        "white tiled floor": ([], []),
    })), "server room", listing, rule="literal", log=_quiet)
    assert "wall" in res.objects, "the WALL was deleted as an alias of the sign"
    assert res.merged == {}
    r = _refusal(res, "wall")
    assert r["into"] == "yellow arrow sign on wall" and r["as"] == "alias"
    # the same shape, on the other structural words of the user's sentence
    for phrase, into in (("ceiling", "metal pipe on ceiling"),
                         ("floor", "stacked boxes on floor"),
                         ("wall", "metal ladder leaning against wall")):
        assert not same_name(phrase, into)[0], f"{phrase} merged into {into}"


def test_the_table_legs_are_not_the_table():
    """DEFECT 2, the same loose reading pointing the other way: 'white table
    with metal legs' cut at 'with' heads on 'table', uncut it heads on 'legs',
    so it shared a head with 'metal table legs' — whose words {metal, table,
    leg} were a subset of {white, table, metal, leg} — and the LEGS were merged
    away. A part with its own mass in the cloud is an object ("segment
    EVERYTHING"), and one that disappears takes its evidence with it.

    With the head taken by one reading the two phrases head on 'leg' and
    'table' and never meet.
    """
    ok, why = same_name("metal table legs", "white table with metal legs")
    assert not ok and "different head nouns (leg vs table)" in why
    listing = ["metal table legs", "white table with metal legs"]
    res = consolidate(_Client(_answer({
        "white table with metal legs": (["metal table legs"], []),
    })), "server room", listing, rule="literal", log=_quiet)
    assert res.objects == listing
    assert res.merged == {}
    assert _refusal(res, "metal table legs")["as"] == "alias"


def test_a_wooden_desk_is_not_the_desk():
    """DEFECT 3, the modifier one: the content-word test accepted a SUBSET, so
    the bare word swallowed every qualified name sharing its head — 'desk' /
    'wooden desk', 'chair' / 'black office chair', 'floor' / 'dark tiled
    floor', 'door' / 'server rack door'. USER 2026-09-21: "como no va a haber
    desk, incluso hay muchos de ellos". The room holds SEVERAL desks; the bare
    word is not the qualified one restated, and each merge cost one object and
    the closure the correction could have measured on it.
    """
    ok, why = same_name("desk", "wooden desk")
    assert not ok and "a qualified name is not the bare one restated" in why
    for bare, qualified in (("chair", "black office chair"),
                            ("floor", "dark tiled floor"),
                            ("door", "server rack door")):
        assert not same_name(bare, qualified)[0], f"{bare} merged {qualified}"
    listing = ["wooden desk", "desk"]
    res = consolidate(_Client(_answer({"wooden desk": (["desk"], [])})),
                      "server room", listing, rule="literal", log=_quiet)
    assert res.objects == listing
    assert res.merged == {}
    assert _refusal(res, "desk")["into"] == "wooden desk"


def test_the_rack_internals_survive_as_objects():
    """A part with its own mass in the cloud is an object — CLAUDE.md's
    doctrine is "segment EVERYTHING". The relationship stays as provenance;
    only the removal is gone."""
    res = consolidate(_Client(_answer({
        "black server rack": ([], RACK[1:]),
    })), "server room", RACK, rule="literal", log=_quiet)
    for phrase in RACK:
        assert phrase in res.objects, f"{phrase} was folded into the rack"
    assert res.parts["black server rack"] == RACK[1:]
    assert res.parts_kept_as_objects is True
    for phrase in RACK[1:]:
        r = _refusal(res, phrase)
        assert r["as"] == "part" and r["into"] == "black server rack"
        assert "a part is an object too" in r["reason"]


def test_a_part_is_recorded_once_however_many_passes_name_it():
    """An applied alias leaves the list and is never proposed again; a PART
    now survives, so every pass names it again. The record counts it once.

    Three passes are needed to show it, so the list has to keep shrinking on
    something the hardened rule still applies: the collective suffix in pass 1
    and the plural in pass 2. The label is named as a part by all three.
    """
    first = _answer({"exposed ceiling ducts": (["exposed ceiling ductwork"], []),
                     "server rack": ([], ["server rack label"]),
                     "server racks": ([], []),
                     "server rack label": ([], [])})
    later = _answer({"server rack": (["server racks"], ["server rack label"]),
                     "exposed ceiling ducts": ([], []),
                     "server rack label": ([], [])})
    listing = ["exposed ceiling ducts", "exposed ceiling ductwork",
               "server rack", "server racks", "server rack label"]
    res = consolidate(_Client(first, later), "server room", listing,
                      max_passes=3, rule="literal", log=_quiet)
    assert res.parts["server rack"] == ["server rack label"]
    assert len(_refusal(res, "server rack label")) == 4
    assert res.objects == ["server rack", "server rack label",
                           "exposed ceiling ducts"]


def test_the_floor_is_not_its_tiles_and_a_label_is_not_its_rack():
    for phrase, into in (("checkered floor tiles", "white tiled floor"),
                         ("white tile floor grout", "white tiled floor"),
                         ("server rack label", "black server rack"),
                         ("desk", "white workbench")):
        ok, why = same_name(phrase, into)
        assert not ok, f"{phrase} must not merge into {into} ({why})"


def test_a_refused_proposal_is_recorded_with_its_reason_everywhere():
    """The refusals are the record of what the pass WANTED to do: in the
    result, in the per-pass history, in the dict that lands in
    output/autoprompt_concepts.json, and in `stopped_reason`, which is the
    field of the summary that travels to segmentation.json."""
    res = consolidate(_Client(_answer({
        "white workbench": (["desk"], []),
    })), "server room", ["white workbench", "desk"], rule="literal", log=_quiet)
    assert res.objects == ["white workbench", "desk"]
    r = _refusal(res, "desk")
    assert set(r) == {"phrase", "into", "as", "reason"}
    d = res.to_dict()
    assert d["refused"] == res.refused and d["n_refused"] == 1
    assert d["merge_rule"] == "literal" and d["parts_kept_as_objects"] is True
    assert res.history[0]["n_refused"] == 1
    assert "desk" in res.stopped_reason and "NOT applied" in res.stopped_reason
    # a refusal is not a failure: the warning stays for the things that are
    assert res.warning() is None


def test_the_model_rule_restores_the_behaviour_before_2026_09_21():
    """Kept selectable for evaluation, and it is NOT the default: the same
    answer that keeps five concepts under 'literal' keeps one under 'model'."""
    answer = _answer({"white workbench": (["white table", "desk",
                                           "computer workstation"],
                                          ["black metal table legs"])})
    res = consolidate(_Client(answer), "server room", DESKS, rule="model",
                      log=_quiet)
    assert res.objects == ["white workbench"]
    assert res.merged["white workbench"] == ["white table", "desk",
                                             "computer workstation"]
    assert res.parts["white workbench"] == ["black metal table legs"]
    assert res.refused == [] and res.parts_kept_as_objects is False


def test_the_default_rule_is_the_conservative_one_from_config():
    res = consolidate(_Client(_answer({"white workbench": (["desk"], [])})),
                      "server room", ["white workbench", "desk"], log=_quiet)
    assert res.merge_rule == "literal"
    assert res.objects == ["white workbench", "desk"]


# ── the safety, which is the whole point ─────────────────────────────────

def test_a_phrase_the_model_never_mentions_survives():
    """The silent failure: an object dropped here is an object that never gets
    segmented, and nothing downstream can tell it was ever proposed."""
    answer = {"objects": [{"name": "white tiled floor", "aliases": [], "parts": []}]}
    res = consolidate(_Client(answer), "server room", PCCR, rule="literal",
                      log=_quiet)
    for p in PCCR:
        assert p in res.objects, f"{p} was lost without the model ever naming it"
    assert set(res.kept_unmentioned) == set(PCCR) - {"white tiled floor"}


def test_the_envelope_is_never_dropped_as_somebody_s_part():
    """USER: 'puertas, ventanas, paredes, pisos, techos, columnas, eso es
    estructural, debe estar'. A door filed as a part of a wall would delete it,
    so the rule is enforced in code, not trusted to the prompt — under BOTH
    rules, because 'model' is the one that still folds parts."""
    answer = {"objects": [
        {"name": "black server rack", "aliases": [],
         "parts": ["dark wooden door", "doorway", "white tiled floor"]},
    ]}
    for rule in MERGE_RULES:
        res = consolidate(_Client(answer), "server room", PCCR, rule=rule,
                          log=_quiet)
        for part in ("dark wooden door", "doorway", "white tiled floor"):
            assert part in res.objects
            assert part in res.kept_structural
            assert _refusal(res, part)["reason"].startswith("structure is never")
        assert res.parts.get("black server rack") is None


def test_an_invented_phrase_is_ignored():
    """The model may only regroup the words it was given; a name it made up
    would silently replace a real prompt."""
    answer = {"objects": [{"name": "the floor of the room", "aliases": [], "parts": []}]}
    res = consolidate(_Client(answer), "server room", PCCR, rule="literal",
                      log=_quiet)
    assert "the floor of the room" not in res.objects
    assert res.objects == PCCR


def test_unparseable_output_leaves_the_list_alone():
    res = consolidate(_Client("I think the floor and the walls are the same."),
                      "server room", PCCR, rule="literal", log=_quiet)
    assert res.objects == PCCR
    assert res.passes == 0


def test_a_failing_client_leaves_the_list_alone():
    class _Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("semantic service down")

    res = consolidate(_Boom(), "server room", PCCR, rule="literal", log=_quiet)
    assert res.objects == PCCR
    assert res.passes == 0


def test_the_same_phrase_claimed_twice_is_only_counted_once():
    """Two objects claim the same alias. The first claim spends it — and it has
    to be a claim the hardened rule actually APPLIES, or the phrase never
    leaves the list and there is nothing to claim twice. The second claim finds
    it already spent and must not put it back in the list or in a second
    survivor's `merged`; it is not a refusal either, because by then there is
    nothing left to refuse.
    """
    listing = ["exposed ceiling ducts", "exposed ceiling ductwork",
               "black server rack"]
    answer = {"objects": [
        {"name": "exposed ceiling ducts",
         "aliases": ["exposed ceiling ductwork"], "parts": []},
        {"name": "black server rack",
         "aliases": ["exposed ceiling ductwork"], "parts": []},
    ]}
    res = consolidate(_Client(answer), "server room", listing, rule="literal",
                      log=_quiet)
    assert res.objects.count("exposed ceiling ductwork") == 0
    assert res.objects == ["exposed ceiling ducts", "black server rack"]
    assert res.merged["exposed ceiling ducts"] == ["exposed ceiling ductwork"]
    assert "exposed ceiling ductwork" not in res.merged.get("black server rack", [])
    assert res.refused == []


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


# ── the real list of 2026-09-21 15:30, end to end ────────────────────────

RAW_15_30 = [
    "white tiled floor", "white wall", "red fire extinguisher",
    "black server rack", "exposed ceiling ducts", "yellow caution tape",
    "glass door", "fluorescent light fixture", "computer monitor",
    "exposed ceiling grid", "white server cabinet", "exposed ceiling beams",
    "metal support column", "white workbench", "white table",
    "checkered floor tiles", "desk", "black office chair",
    "exposed ceiling ductwork", "black door frame", "exposed ceiling with ducts",
    "red fire alarm box", "metal ladder", "black backpack",
    "green indicator light", "doorway", "stacked white panels",
    "plastic sheeting", "red fire alarm", "server rack door handle",
    "black cable tray", "electronic equipment", "server rack cable bundle",
    "metallic conduit pipes", "ventilation grille", "dark baseboard",
    "tall rolled-up insulation material", "red painted wall section",
    "red safety barrier", "stacked metal framing", "metallic electrical conduit",
    "white panel standing", "white machine with vents", "wooden plank",
    "computer server tower", "dark gray tiled floor patch",
    "server rack network switch", "server rack power supply unit",
    "server rack hard drive bay", "server rack cable management",
    "small blue screen", "white tile floor grout", "server rack label",
    "black electrical cable", "yellow directional sign", "black electrical wires",
    "silver refrigerator", "black metal table legs", "dark grout lines",
    "computer workstation",
]

# exactly what the VLM proposed that afternoon, read back from
# projects/pccr/scans/2026-08-31/src_default/output/autoprompt_concepts.json
# (history[0].merged and history[0].parts — 18 aliases and 8 parts). The
# fixture is checked against that file below whenever the session is on disk.
PROPOSED_15_30 = {
    "white tiled floor": (["checkered floor tiles", "dark gray tiled floor patch",
                           "white tile floor grout", "dark grout lines"], []),
    "white wall": (["red painted wall section"], []),
    "black server rack": (["white server cabinet"],
                          ["server rack door handle", "server rack network switch",
                           "server rack power supply unit", "server rack hard drive bay",
                           "server rack cable management", "server rack label",
                           "server rack cable bundle"]),
    "exposed ceiling ducts": (["exposed ceiling ductwork",
                               "exposed ceiling with ducts"], []),
    "glass door": (["doorway"], []),
    "computer monitor": (["small blue screen"], []),
    "white workbench": (["white table", "desk", "computer workstation"],
                        ["black metal table legs"]),
    "red fire alarm box": (["red fire alarm"], []),
    "stacked white panels": (["white panel standing"], []),
    "electronic equipment": (["computer server tower"], []),
    "metallic conduit pipes": (["metallic electrical conduit"], []),
    "black electrical cable": (["black electrical wires"], []),
}


_RECORD_15_30 = (Path(__file__).resolve().parents[1] / "projects" / "pccr"
                 / "scans" / "2026-08-31" / "src_default" / "output"
                 / "autoprompt_concepts.json")


@pytest.mark.skipif(not _RECORD_15_30.exists(),
                    reason="the pccr session is not on this machine")
def test_the_fixture_is_the_run_that_is_on_disk():
    """The replay below is only worth its number while the fixture IS the
    record: 60 raw concepts in the order the VLM wrote them, and the 12
    groupings of the first pass. Measured 2026-09-21 — both match exactly."""
    rec = json.loads(_RECORD_15_30.read_text())
    assert rec["raw"]["objects"] == RAW_15_30
    first = rec["consolidation"]["history"][0]
    proposed: dict[str, tuple[list, list]] = {}
    for name, aliases in (first["merged"] or {}).items():
        proposed.setdefault(name, ([], []))[0].extend(aliases)
    for name, parts in (first["parts"] or {}).items():
        proposed.setdefault(name, ([], []))[1].extend(parts)
    assert {k: (list(a), list(p)) for k, (a, p) in PROPOSED_15_30.items()} == \
        {k: (list(a), list(p)) for k, (a, p) in proposed.items()}
    # what that run shipped, and what this test exists to move
    assert len(rec["consolidated"]["objects"]) == 34


def test_the_run_of_2026_09_21_keeps_59_of_its_60_concepts():
    """The measured case, replayed. That afternoon the VLM proposed 18 aliases
    and 8 parts over the 60 raw concepts and the pass applied all 26, shipping
    34 prompts to SAM3.

    Replayed through the hardened rule the answer is **60 → 59**, measured, not
    chosen: exactly ONE proposal is a literal restatement — 'exposed ceiling
    ductwork' → 'exposed ceiling ducts' — and the other 25 (17 aliases, 8
    parts) are recorded as refusals with both concepts surviving. Even the
    third phrasing of the same ducts stays: 'exposed ceiling with ducts' heads
    on 'ceiling' once its trailing prepositional phrase is cut.

    59 prompts instead of 34 is the detail the user asked for and 25 more
    objects the correction can measure a closure on — the 61/32/34 table of
    the module header is what that difference is worth.
    """
    groups = dict(PROPOSED_15_30)
    claimed = {p for a, pa in groups.values() for p in list(a) + list(pa)}
    for phrase in RAW_15_30:
        if phrase not in groups and phrase not in claimed:
            groups[phrase] = ([], [])
    res = consolidate(_Client(_answer(groups)), "server room", RAW_15_30,
                      rule="literal", log=_quiet)
    assert len(RAW_15_30) == 60
    assert len(res.objects) == 59
    assert res.merged == {"exposed ceiling ducts": ["exposed ceiling ductwork"]}
    assert [p for p in RAW_15_30 if p not in res.objects] == \
        ["exposed ceiling ductwork"]
    assert len(res.refused) == 25
    assert sum(1 for r in res.refused if r["as"] == "alias") == 17
    assert sum(1 for r in res.refused if r["as"] == "part") == 8
    for phrase in ("desk", "white table", "computer workstation", "white workbench",
                   "checkered floor tiles", "server rack label", "doorway",
                   "white server cabinet", "black metal table legs",
                   "exposed ceiling with ducts", "dark grout lines",
                   "white tile floor grout", "red painted wall section"):
        assert phrase in res.objects, phrase


# ── the config key ───────────────────────────────────────────────────────

def test_a_missing_merge_rule_fails_at_load_naming_itself():
    with pytest.raises(KeyError) as e:
        load_merge_rule({"autoprompt": {"consolidate_passes": 3}})
    assert "autoprompt.consolidate_merge_rule" in str(e.value)
    with pytest.raises(KeyError):
        load_merge_rule({})
    with pytest.raises(ValueError) as e:
        load_merge_rule({"autoprompt": {"consolidate_merge_rule": "aggressive"}})
    assert "autoprompt.consolidate_merge_rule" in str(e.value)


def test_no_decision_literal_hides_in_the_module():
    """The rule is read from config.yaml, never defaulted in code."""
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "autoprompt"
           / "consolidate_prompts.py").read_text()
    assert 'rule = load_merge_rule() if rule is None else rule' in src


def test_the_pass_is_wired_into_the_simple_pipeline():
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "autoprompt" / "session_builder.py").read_text()
    assert "from .consolidate_prompts import consolidate" in src
    i = src.index("consolidation = consolidate(")
    assert src.index('prompt = ";".join(phrases)') > i, \
        "the prompt must be built from the CONSOLIDATED list"


def test_the_config_carries_the_switch_the_bound_and_the_rule():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert cfg["autoprompt"]["consolidate_prompts"] is True
    assert int(cfg["autoprompt"]["consolidate_passes"]) >= 1
    assert cfg["autoprompt"]["consolidate_merge_rule"] == "literal"
    assert load_merge_rule() == "literal"
