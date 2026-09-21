# STAC-Builder — Auto-prompter: CONSOLIDATION of the concept list (Phase 1).
#
# USER 2026-09-15: *"podríamos tener dos o tres pasadas de vLLM para fusionar
# palabras y prompts"*.
#
# ── WHY THE PASS EXISTS (2026-09-15, unchanged) ────────────────────────────
# Scene understanding runs per frame, and in the SIMPLE pipeline the phrases it
# writes go straight to SAM3 as one text prompt — and therefore one segment —
# each. The per-frame prompt already asks for atomic objects and one name per
# object, and it obeys: each frame ANSWERS WELL ON ITS OWN. What no per-frame
# instruction can fix is the disagreement BETWEEN frames, because no frame ever
# sees the others' answers. Over pccr's 93 keyframes the union came to 84
# phrases, most of the growth being the same thing under different words and
# things too small to be objects at scan resolution:
#
#     floor  5   white tiled floor / checkered floor tiles / dark grey tiled
#                floor patch / dark grey tiled floor section / white tile floor grout
#     rack  14   server rack motherboard / hard drive / fan / cable management /
#                power supply / ventilation grille / mounting bracket / label …
#
# `aggregate` already merges phrasings that share a head noun; these do not
# ('floor' / 'tiles' / 'patch' / 'section' / 'grout'). That is a LANGUAGE
# judgement over the assembled list, so it gets its own pass over the WHOLE
# list — which is the one thing the per-frame pass can never have. On pccr
# those 84 phrases became 149 fragmented instances; the certification then
# matched FRAGMENTS OF ONE LABEL against each other as revisits of one object
# and measured a 336 cm closure over a 3.2 m walk, ending in a −1205 %
# regression. That is what justified merging aggressively.
#
# ── WHAT 2026-09-21 MEASURED (why it may no longer merge on judgement) ─────
# USER: *"el detalle de vlm sobre la escena debe ser altamente detallado, como
# no va a haber desk, incluso hay muchos de ellos, debe ser altamente detallado
# ojo con eso no arruines lo que funcionaba"*.
#
# FIRST, the failure this pass was protecting against is now handled where it
# actually broke: since 2026-09-21 a pair whose supports do not overlap on the
# surface they share is refused as an identity by
# reconstruction/loops/spatial_gate.py (`surface_overlap`) and
# reconstruction/certify/loops_posthoc.py (`instance_edges` / `copy_scale_rows`),
# so fragments of one label are no longer read as revisits of one object.
#
# SECOND, the merge itself became the problem, because these objects ARE the
# evidence the correction runs on. Three runs of the SAME scan:
#
#   categories masklets objects >=1000pt 2+visits after walk after 25% closures
#     61 (raw)    358      178     123       24         23         5        5
#     32          255      111      75       18         16         3        3
#     34          269      134      79       14         12         2        2
#
#   61 → r per chunk 0.9533-1.0507 = 10.2 % drift, cameras up to 38.2 cm
#   32 → 0.9638-1.0393 =  7.8 %, up to 30.0 cm
#   34 → 0.9913-1.0107 =  2.2 %, up to  9.2 cm
#
# Fewer and larger objects means fewer objects with two separated visits, which
# means fewer closures, which means a weaker correction. The run the user
# validated ("la corrección es realmente muy buena… el piso está increíble") is
# the one whose consolidation pass CRASHED and left the raw 61-concept list
# untouched. The 15:30 run of the same day folded 'desk', 'white table' and
# 'computer workstation' into 'white workbench' as ALIASES — different objects,
# and the room holds several of them — and the rack's nine internals (label,
# door handle, cable bundle, hard drive bay, network switch, power supply unit,
# mounting bracket, cable management, shelf) into 'black server rack'.
#
# ── WHAT THE PASS MAY DO NOW (`autoprompt.consolidate_merge_rule`) ─────────
# "literal" (default):
#   · the pass still RUNS and still RECORDS every grouping it proposes;
#   · it may only APPLY an alias it cannot be wrong about — a LITERAL
#     restatement of the same name: the two phrases share a head noun (or head
#     nouns that are morphological variants of one stem: plural/singular,
#     'ductwork' vs 'ducts') AND the content words of one are a subset of the
#     other's, so no modifier of one contradicts the other. 'exposed ceiling
#     ductwork' and 'exposed ceiling with ducts' are one name written twice;
#     'desk', 'white table' and 'computer workstation' are not 'white
#     workbench';
#   · it NEVER removes a PART. CLAUDE.md's doctrine is "segment EVERYTHING",
#     and a part with its own mass in the cloud is an object — folding them is
#     what erased the rack internals. The relationship is still detected and
#     still recorded, only the removal is gone;
#   · everything proposed and not applied is recorded as a REFUSAL with its
#     reason — in the result, in output/autoprompt_concepts.json (through
#     `Consolidation.to_dict`) and in the one-line summary that reaches
#     segmentation.json (through `stopped_reason`).
# "model": the pre-2026-09-21 behaviour — apply whatever the VLM says, with the
#   structural rescue as the only veto. Kept selectable for evaluation.
#
# The morphology is deliberately tiny (plural/singular plus the declared
# collective suffix). When in doubt the pass REFUSES and both concepts survive:
# a wrong merge destroys an object and the evidence it carried, a missed merge
# only leaves a duplicate — and `segmentation.merge_duplicates` (mutual overlap
# on the cloud, not on the words) is where a real duplicate collapses.
#
# PROVENANCE: vlm_proposed, like everything else the auto-prompter names. It
# never measures — it only decides which words name the same thing.
#
# SAFETY, in order of importance:
#   · nothing is ever LOST silently — a phrase the model does not mention
#     survives untouched;
#   · the structural envelope is never dropped as somebody's part (USER: "puertas
#     ventanas paredes pisos techos columnas, eso es estructural, debe estar");
#   · a failed or unparseable pass returns the list unchanged — and it is
#     the one thing here that can silently change the session vocabulary,
#     so it is RECORDED (history, reason, numbers) and declared as a
#     warning the session carries on disk, not only in a log line.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_SYSTEM = (
    "You are consolidating the object list of a 3D scan. The list was written "
    "frame by frame by an earlier pass, so the same thing is often named "
    "several ways and some entries are parts of others. Return STRICT JSON only."
)


def _prompt(scene_type: str, phrases: list[str]) -> str:
    listing = "\n".join(f"- {p}" for p in phrases)
    return (
        f"These phrases were collected across many frames of ONE {scene_type or 'place'}. "
        f"Every phrase became one segmentation prompt, so two names for one thing "
        f"produce two copies of it, and a phrase naming a PART produces a segment "
        f"nobody can separate from its parent.\n\n"
        f"{listing}\n\n"
        f"Group them. Return JSON:\n"
        '{"objects": [{"name": "<the single best phrase FROM THE LIST for this object>",\n'
        '              "aliases": ["<other phrases from the list naming the SAME physical thing>"],\n'
        '              "parts":   ["<phrases from the list that are PARTS of this object>"]}]}\n\n'
        "Rules:\n"
        "1. SAME THING means the SAME NAME WRITTEN TWICE, not two things that "
        "resemble each other: 'exposed ceiling ductwork' and 'exposed ceiling "
        "with ducts' are one name. A 'desk', a 'white table' and a 'computer "
        "workstation' are THREE objects and the room holds several of each — "
        "never put them together. If you cannot restate one phrase with the "
        "other's words, they are different objects.\n"
        "2. A PART is a piece of a bigger object (a rack's fan, hard drive, "
        "label, cable management; a door's handle). Name them in 'parts': they "
        "are kept and segmented like any other object, the relationship is "
        "recorded as provenance. Never put a part in 'aliases'.\n"
        "3. DIFFERENT objects stay separate even when they touch or look alike. "
        "A wall and a cabinet against it are two objects. When unsure, keep "
        "them separate — a wrong merge destroys an object, a missed merge only "
        "leaves a duplicate.\n"
        "4. Structure is never a part of anything: every floor, ceiling, wall, "
        "door, window, opening, column, beam and stair in the list must appear "
        "as an object's 'name' or as an alias of one, never under 'parts'.\n"
        "5. Use the phrases EXACTLY as written above. Every phrase must appear "
        "exactly once, as a name, an alias or a part.\n"
        "Output ONLY the JSON."
    )


# The envelope, by head word. Rule 4 asks the model for this; the code enforces
# it, because a dropped door is the failure the user called fundamental.
_STRUCTURAL = ("floor", "floors", "ceiling", "ceilings", "wall", "walls",
               "door", "doors", "doorway", "doorways", "window", "windows",
               "opening", "openings", "column", "columns", "pillar", "pillars",
               "beam", "beams", "stair", "stairs", "staircase", "vault",
               "ground", "slab", "roof")

# What the code is allowed to APPLY. "literal" is the 2026-09-21 rule, "model"
# the behaviour before it. Read from config.yaml, never defaulted here.
MERGE_RULES = ("literal", "model")
_CONFIG_KEY = "autoprompt.consolidate_merge_rule"


def is_structural(phrase: str) -> bool:
    """True when the phrase names the ENVELOPE ITSELF, not something fixed to it.

    The test is the HEAD noun, never "a structural word appears somewhere":
    'metal door handle' and 'door lock mechanism' both contain 'door' and both
    are ironmongery — pccr 2026-09-15 rescued them from 'parts' and they
    survived as two prompts of their own. 'dark wooden door' heads on 'door'
    and is the door; 'glass door server cabinet' heads on 'cabinet' and is a
    cabinet.
    """
    from .scene_understanding import _head_noun
    return _head_noun(str(phrase).lower()) in _STRUCTURAL


# ── the literal-restatement test ─────────────────────────────────────────
#
# Words that carry no object identity: dropping them is what lets 'exposed
# ceiling with ducts' read as 'exposed ceiling ducts'. The preposition list is
# scene_understanding's own, so both passes cut a phrase at the same places.
def _function_words() -> frozenset[str]:
    from .scene_understanding import _PREPOSITIONS
    return frozenset(_PREPOSITIONS) | {"a", "an", "the", "and", "or", "some",
                                       "this", "that", "its", "their"}


# Collective/derivational suffixes that name the SAME thing as their stem —
# 'ductwork' is ducts, 'pipework' is pipes. Kept to the one form the data
# actually shows, with a 4-character floor so 'network' does not become 'net'.
_COLLECTIVE_SUFFIXES = ("work",)


def _words(phrase: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(phrase).lower())


def _stem(word: str) -> str:
    """Plural-stripped (the same rules `_head_noun` uses) plus the declared
    collective suffix. Deterministic, no lexicon, no model."""
    w = str(word).strip().lower()
    if not w:
        return w
    for suf in ("sses", "xes", "ches", "shes"):
        if w.endswith(suf):
            w = w[:-2]
            break
    else:
        if w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
    for suf in _COLLECTIVE_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            break
    return w


def _content_stems(phrase: str) -> frozenset[str]:
    fn = _function_words()
    return frozenset(_stem(w) for w in _words(phrase) if w not in fn) - {""}


def _head_stem(phrase: str) -> str:
    """The head noun, by ONE reading: the phrase with its trailing
    prepositional phrase cut (``scene_understanding._head_noun``, whose whole
    purpose is that 'desk with monitor' does not key on 'monitor').

    It used to keep the UNCUT last word as a second reading as well, so that
    'exposed ceiling with ducts' could match 'exposed ceiling ducts'. That
    second reading is what let any 'X <prep> Y' phrase share a head with
    anything headed Y, and it was measured deleting the things this module
    exists to protect:

        same_name('wall', 'yellow arrow sign on wall')              -> merge
        same_name('metal table legs', 'white table with metal legs') -> merge

    The first removes the WALL — "puertas ventanas paredes pisos techos
    columnas, eso es estructural, debe estar" — and the second the legs, both
    with no refusal recorded, because neither was travelling through the parts
    door where the rescues live. The cost of dropping the loose reading is
    that 'exposed ceiling with ducts' survives next to 'exposed ceiling
    ducts': one extra prompt for one object. That is the cheap side of this
    trade and the side the user asked for."""
    from .scene_understanding import _head_noun
    ws = _words(phrase)
    if not ws:
        return ""
    return _stem(_head_noun(" ".join(ws)))


def same_name(a: str, b: str) -> tuple[bool, str]:
    """Do these two phrases name the same thing LITERALLY? → (verdict, reason).

    Two tests, both deterministic and both on the words alone:
      1. the phrases share their head noun, up to plural/singular and the
         declared collective suffix ('ductwork' ≡ 'ducts');
      2. their content words are the SAME SET, so neither says anything the
         other does not. 'exposed ceiling ductwork' restates 'exposed ceiling
         ducts'; 'desk' does NOT restate 'wooden desk'.

    Test 2 used to accept a SUBSET, on the reading that one phrase is the other
    said with more detail. A subset is a restatement of a TYPE, never of an
    IDENTITY, and the scene this was built for is full of the difference:
    'desk' vs 'wooden desk', 'chair' vs 'black office chair', 'door' vs
    'server rack door'. USER 2026-09-21: "como no va a haber desk, incluso hay
    muchos de ellos". When several of a thing exist, the bare word is not the
    same object as the qualified one, and merging them costs exactly the
    evidence the correction runs on.

    The reason is written for the refusal record: it says WHICH test failed and
    with which words, because that record is what the user reads when a concept
    he expected to survive did not.
    """
    ha, hb = _head_stem(a), _head_stem(b)
    if not ha or not hb or ha != hb:
        return False, (f"different head nouns ({ha or '?'} vs {hb or '?'}) — "
                       f"not one name restated")
    ca, cb = _content_stems(a), _content_stems(b)
    if ca != cb:
        extra_a = sorted(ca - cb)
        extra_b = sorted(cb - ca)
        return False, (f"head '{ha}' is shared but the words differ "
                       f"({'+'.join(extra_a) or '—'} vs "
                       f"{'+'.join(extra_b) or '—'}) — a qualified name is not "
                       f"the bare one restated")
    return True, f"the same name restated (head '{ha}')"


def _read_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(
            f"{_CONFIG_KEY} could not be read: {cfg_path} does not exist")
    import yaml
    return yaml.safe_load(cfg_path.read_text()) or {}


def load_merge_rule(config: Optional[dict] = None) -> str:
    """`autoprompt.consolidate_merge_rule`, MANDATORY.

    What this pass is allowed to APPLY decides the session's vocabulary, and
    the session's vocabulary decides how many objects the correction can
    measure a closure on (the 61/32/34 table in the header). A key that
    decides that much is never defaulted in code: a missing one fails at load
    naming itself, like every key of `correction:`.
    """
    cfg = _read_config() if config is None else (config or {})
    ap = cfg.get("autoprompt")
    if not isinstance(ap, dict) or "consolidate_merge_rule" not in ap:
        raise KeyError(
            f"'{_CONFIG_KEY}' is missing from config.yaml — it decides what the "
            f"consolidation pass may APPLY ({' | '.join(MERGE_RULES)}) and the "
            f"session's whole object vocabulary with it, so it is not defaulted")
    rule = str(ap["consolidate_merge_rule"]).strip().lower()
    if rule not in MERGE_RULES:
        raise ValueError(
            f"'{_CONFIG_KEY}' is {ap['consolidate_merge_rule']!r}; "
            f"expected one of {' | '.join(MERGE_RULES)}")
    return rule


@dataclass
class Consolidation:
    objects: list[str]                       # the surviving prompts, in order
    merged: dict[str, list[str]] = field(default_factory=dict)   # name → aliases APPLIED
    parts: dict[str, list[str]] = field(default_factory=dict)    # name → parts detected
    kept_unmentioned: list[str] = field(default_factory=list)
    kept_structural: list[str] = field(default_factory=list)
    passes: int = 0
    # ── what actually happened, because it DECIDES the vocabulary ────────
    # Two runs of the SAME scan, same 61 object types out of the VLM:
    #   04:57  "the pass returned nothing parseable — list unchanged"
    #          61 → 61 concept(s) in 0 pass(es)   → 61 categories to SAM3
    #   11:34  pass 1: 61 → 33 ; pass 2: 33 → 32  → 32 categories to SAM3
    # The VLM was stable; the whole difference was here, and nothing on disk
    # said so. The detail of a session's segmentation is decided in this
    # dataclass, so it carries the evidence: the list it was given, every
    # pass and its outcome, and whether any pass failed to parse.
    input_objects: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    parse_failed: bool = False
    max_passes_reached: bool = False
    stopped_reason: str = ""
    # ── what the pass PROPOSED and the code did not apply ────────────────
    # The 2026-09-21 record: {"phrase", "into", "as", "reason"}. Both concepts
    # survive, and the reader can see the pass wanted otherwise and why.
    refused: list[dict] = field(default_factory=list)
    merge_rule: str = "literal"
    # In "literal" mode `parts` is a RELATIONSHIP, not a removal: every phrase
    # in it is still in `objects`. The flag travels so a reader of the JSON
    # never has to infer it from the mode.
    parts_kept_as_objects: bool = True

    def warning(self) -> Optional[str]:
        """The one sentence the session must be able to say about its own
        vocabulary, or None when the list converged normally."""
        if self.passes == 0:
            return (f"the consolidation never ran — SAM3 is segmenting with "
                    f"the RAW per-frame list of {len(self.objects)} concept(s) "
                    f"({self.stopped_reason})")
        if self.parse_failed:
            return (f"the consolidation stopped after {self.passes} pass(es) "
                    f"with {len(self.objects)} concept(s), which is not a "
                    f"converged list ({self.stopped_reason})")
        if self.max_passes_reached:
            return (f"the consolidation used all {self.passes} pass(es) and "
                    f"the list was still changing at the last one — "
                    f"{len(self.objects)} concept(s)")
        return None

    def to_dict(self) -> dict:
        return {"origin": "vlm_proposed", "objects": self.objects,
                "merged": self.merged, "parts": self.parts,
                "kept_unmentioned": self.kept_unmentioned,
                "kept_structural": self.kept_structural, "passes": self.passes,
                "input_objects": self.input_objects,
                "n_input": len(self.input_objects),
                "n_objects": len(self.objects),
                "history": self.history,
                "parse_failed": self.parse_failed,
                "max_passes_reached": self.max_passes_reached,
                "stopped_reason": self.stopped_reason,
                "merge_rule": self.merge_rule,
                "parts_kept_as_objects": self.parts_kept_as_objects,
                "refused": self.refused,
                "n_refused": len(self.refused),
                "warning": self.warning()}


def _parse(txt: str) -> Optional[dict]:
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return d if isinstance(d, dict) else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _one_pass(client, scene_type: str, phrases: list[str], max_tokens: int,
              rule: str, log: Callable[[str], None]
              ) -> tuple[Optional[Consolidation], dict]:
    """(result, note) — the note is kept whether the pass worked or not.

    A pass that does not parse used to write one log line and vanish. It is a
    REFUSAL and it records its reason and its numbers like every other refusal
    in this codebase: how many characters came back, why the JSON did not
    survive, and what the model said it stopped for (`finish_reason` 'length'
    is a truncated answer, not a confused one — the list grows with the scene
    and `max_tokens` does not).
    """
    from semantic.types import system, user

    resp = client.chat([system(_SYSTEM), user(_prompt(scene_type, phrases))],
                       max_tokens=max_tokens, consumer="phase1.consolidate")
    txt = resp.content or ""
    note = {"before": len(phrases), "chars": len(txt),
            "finish_reason": getattr(resp, "finish_reason", None)}
    d = _parse(txt)
    if d is None or not isinstance(d.get("objects"), list):
        if not txt:
            why = "the model returned an empty response"
        elif d is None:
            why = f"no JSON object in the {len(txt)} character(s) returned"
        else:
            why = f"the JSON of {len(txt)} character(s) carries no 'objects' list"
        if note["finish_reason"] == "length":
            why += f" (finish_reason=length: the answer was cut at max_tokens={max_tokens})"
        note.update(status="unparsed", reason=why, after=len(phrases))
        log(f"[consolidate] the pass returned nothing parseable — "
            f"list unchanged ({why})")
        return None, note

    known = {_norm(p): p for p in phrases}
    out = Consolidation(objects=[], passes=1, merge_rule=rule,
                        parts_kept_as_objects=(rule == "literal"))
    seen: set[str] = set()

    def _keep(phrase: str, into: str, role: str, reason: str) -> None:
        """The concept SURVIVES and the proposal is recorded against it."""
        out.objects.append(phrase)
        out.refused.append({"phrase": phrase, "into": into, "as": role,
                            "reason": reason})
        seen.add(_norm(phrase))

    for entry in d["objects"]:
        if not isinstance(entry, dict):
            continue
        name = known.get(_norm(entry.get("name", "")))
        if name is None or _norm(name) in seen:
            continue        # invented a phrase, or named the same object twice
        seen.add(_norm(name))
        out.objects.append(name)
        for key, bucket in (("aliases", out.merged), ("parts", out.parts)):
            for raw in (entry.get(key) or []):
                other = known.get(_norm(raw))
                if other is None or _norm(other) in seen:
                    continue
                # Rule 4, enforced rather than trusted: the envelope is never
                # dropped as somebody's part. An alias is fine (it is the same
                # surface under another word); a PART would delete it.
                if key == "parts" and is_structural(other):
                    out.objects.append(other)
                    out.kept_structural.append(other)
                    out.refused.append(
                        {"phrase": other, "into": name, "as": "part",
                         "reason": "structure is never a part of anything "
                                   "(floor/ceiling/wall/door/window/column/"
                                   "beam/stair)"})
                    seen.add(_norm(other))
                    continue
                if rule == "literal":
                    if key == "parts":
                        # A part with its own mass in the cloud is an object —
                        # "segment EVERYTHING". The relationship is kept as
                        # provenance, the concept is kept as a prompt.
                        bucket.setdefault(name, []).append(other)
                        _keep(other, name, "part",
                              "a part is an object too — it keeps its own "
                              "prompt and the relationship is recorded")
                        continue
                    ok, why = same_name(other, name)
                    if not ok:
                        _keep(other, name, "alias", why)
                        continue
                bucket.setdefault(name, []).append(other)
                seen.add(_norm(other))

    # anything the model forgot survives — losing an object in silence is the
    # one outcome this pass must never produce
    for p in phrases:
        if _norm(p) not in seen:
            out.objects.append(p)
            out.kept_unmentioned.append(p)
    note.update(status="applied", reason="", after=len(out.objects),
                merged={k: list(v) for k, v in out.merged.items()},
                parts={k: list(v) for k, v in out.parts.items()},
                kept_structural=list(out.kept_structural),
                kept_unmentioned=list(out.kept_unmentioned),
                refused=[dict(r) for r in out.refused],
                n_refused=len(out.refused))
    return out, note


def consolidate(client, scene_type: str, phrases: list[str],
                max_passes: int = 3, max_tokens: int = 4096,
                rule: Optional[str] = None,
                log: Callable[[str], None] = print) -> Consolidation:
    """Merge the words that name one thing; keep everything else.

    Runs the pass again while it still CHANGES the list and stops when a pass
    changes nothing — convergence, the same shape as the greedy correction
    loop, so there is no "right" number of passes to pick. ``max_passes`` is a
    bound against a model that oscillates, not a decision: reaching it is
    declared in the result.

    ``rule`` defaults to `autoprompt.consolidate_merge_rule` in config.yaml,
    which must exist (see `load_merge_rule`).
    """
    rule = load_merge_rule() if rule is None else rule
    if rule not in MERGE_RULES:
        raise ValueError(f"'{_CONFIG_KEY}' is {rule!r}; "
                         f"expected one of {' | '.join(MERGE_RULES)}")
    current = list(phrases)
    merged: dict[str, list[str]] = {}
    parts: dict[str, list[str]] = {}
    unmentioned: list[str] = []
    structural: list[str] = []
    refused: list[dict] = []
    seen_refusals: set[tuple[str, str, str]] = set()
    history: list[dict] = []
    parse_failed = False
    stopped = ""
    done = 0
    bound = max(1, int(max_passes))
    for _ in range(bound):
        try:
            res, note = _one_pass(client, scene_type, current, max_tokens, rule, log)
        except Exception as e:  # noqa: BLE001 — declared, never fatal
            stopped = f"the pass raised {type(e).__name__}: {e}"
            history.append({"pass": len(history) + 1, "status": "failed",
                            "reason": stopped, "before": len(current),
                            "after": len(current)})
            log(f"[consolidate] pass failed ({e}) — keeping the list as it is")
            break
        note["pass"] = len(history) + 1
        history.append(note)
        if res is None:
            # THE defect this record exists for: an unparseable pass changed
            # the session vocabulary (61 concepts instead of 32) and said so
            # only in a log line nothing keeps.
            parse_failed = True
            stopped = note.get("reason", "the pass returned nothing parseable")
            break
        done += 1
        # Deduped on the way in: an applied alias leaves the list and is never
        # proposed again, but a PART survives as its own prompt, so every pass
        # names it again and the record would count it once per pass.
        for bucket, seen_in in ((res.merged, merged), (res.parts, parts)):
            for name, phrases_in in bucket.items():
                kept = seen_in.setdefault(name, [])
                kept.extend(w for w in phrases_in if w not in kept)
        unmentioned.extend(res.kept_unmentioned)
        structural.extend(res.kept_structural)
        for r in res.refused:
            key = (r["phrase"], r["into"], r["as"])
            if key not in seen_refusals:
                seen_refusals.add(key)
                refused.append(r)
        # Convergence is asked of the SET, not of the list order. Under the
        # literal rule most passes apply nothing at all, and a model that
        # merely reorders what it was given must not buy another VLM call.
        if {_norm(o) for o in res.objects} == {_norm(o) for o in current}:
            note["status"] = "converged"
            stopped = "converged: the pass changed nothing"
            break
        log(f"[consolidate] pass {done}: {len(current)} → {len(res.objects)} concept(s)")
        current = res.objects

    # the bound is a BOUND, not a decision — but reaching it means the list was
    # still moving when we stopped, and that has to be declared, not inferred
    max_reached = (len(history) >= bound and not parse_failed
                   and bool(history) and history[-1].get("status") == "applied")
    if max_reached:
        stopped = (f"the {bound}-pass bound was reached with the list still "
                   f"changing")
    if refused:
        # The refusals have to reach segmentation.json, and `stopped_reason` is
        # the free-text field of the summary that travels there.
        head = "; ".join(f"'{r['phrase']}' ← {r['as']} of '{r['into']}' ({r['reason']})"
                         for r in refused[:3])
        clause = (f"{len(refused)} proposed merge(s) recorded and NOT applied "
                  f"under rule '{rule}' — both concepts survive: {head}"
                  + (" …" if len(refused) > 3 else ""))
        stopped = f"{stopped} — {clause}" if stopped else clause

    out = Consolidation(objects=current, merged=merged, parts=parts,
                        kept_unmentioned=sorted(set(unmentioned)),
                        kept_structural=sorted(set(structural)), passes=done,
                        input_objects=list(phrases), history=history,
                        parse_failed=parse_failed,
                        max_passes_reached=max_reached, stopped_reason=stopped,
                        refused=refused, merge_rule=rule,
                        parts_kept_as_objects=(rule == "literal"))
    n_al = sum(len(v) for v in merged.values())
    n_pt = sum(len(v) for v in parts.values())
    log(f"[consolidate] {len(phrases)} → {len(current)} concept(s) in {done} pass(es) "
        f"[rule '{rule}']: {n_al} alias(es) merged, {n_pt} part(s) "
        + ("recorded and kept as objects" if out.parts_kept_as_objects
           else "folded into their object")
        + (f", {len(out.refused)} proposal(s) refused" if out.refused else "")
        + (f", {len(out.kept_structural)} structural entr(y/ies) rescued from 'parts'"
           if out.kept_structural else "")
        + (f", {len(out.kept_unmentioned)} kept because the pass never mentioned them"
           if out.kept_unmentioned else ""))
    warn = out.warning()
    if warn:
        log(f"[consolidate] ⚠️ {warn}")
    return out
