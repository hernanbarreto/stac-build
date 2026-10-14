# STAC-Builder — Auto-prompter: CONSOLIDATION of the concept list (Phase 1).
#
# USER 2026-09-15: *"podríamos tener dos o tres pasadas de vLLM para fusionar
# palabras y prompts"*.
#
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
# ('floor' / 'tiles' / 'patch' / 'section' / 'grout'), and no amount of string
# work will decide that a 'hard drive' inside a rack is not a segmentable
# object. That is a LANGUAGE judgement over the assembled list, so it gets its
# own pass over the WHOLE list — which is the one thing the per-frame pass can
# never have.
#
# The damage it prevents is not cosmetic. On pccr those 84 phrases became 149
# fragmented instances; the certification then matched fragments of one label
# against each other as revisits of one object and measured a 336 cm closure
# over a 3.2 m walk, ending in a −1205 % regression.
#
# PROVENANCE: vlm_proposed, like everything else the auto-prompter names. It
# never measures — it only decides which words name the same thing.
#
# SAFETY, in order of importance:
#   · nothing is ever LOST silently — a phrase the model does not mention
#     survives untouched;
#   · the structural envelope is never dropped as somebody's part (USER: "puertas
#     ventanas paredes pisos techos columnas, eso es estructural, debe estar");
#   · a failed or unparseable pass returns the list unchanged.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
        "1. SAME THING means the same physical object or surface, however "
        "differently worded. Pieces of one continuous surface are the SAME "
        "surface, not different objects, even when different frames saw different "
        "patches of it.\n"
        "2. A PART is something you could not point at separately from two "
        "metres away with a handheld camera. Parts go in 'parts', never in "
        "'objects'.\n"
        "3. DIFFERENT objects stay separate even when they touch or look alike. "
        "When unsure, keep "
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


#: the structural SURFACES — what a mass of small units (tiles, bricks, stones) is built into
_SURFACES = ("floor", "floors", "ceiling", "ceilings", "wall", "walls", "ground", "slab",
             "roof", "vault")


def is_surface(phrase: str) -> bool:
    """True when the phrase's head noun is a structural SURFACE (floor, ground, wall …)."""
    from .scene_understanding import _head_noun
    return _head_noun(str(phrase).lower()) in _SURFACES


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


@dataclass
class Consolidation:
    objects: list[str]                       # the surviving prompts, in order
    merged: dict[str, list[str]] = field(default_factory=dict)   # name → aliases
    parts: dict[str, list[str]] = field(default_factory=dict)    # name → parts
    kept_unmentioned: list[str] = field(default_factory=list)
    kept_structural: list[str] = field(default_factory=list)
    passes: int = 0

    def to_dict(self) -> dict:
        return {"origin": "vlm_proposed", "objects": self.objects,
                "merged": self.merged, "parts": self.parts,
                "kept_unmentioned": self.kept_unmentioned,
                "kept_structural": self.kept_structural, "passes": self.passes}


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
              log: Callable[[str], None]) -> Optional[Consolidation]:
    from semantic.types import system, user

    resp = client.chat([system(_SYSTEM), user(_prompt(scene_type, phrases))],
                       max_tokens=max_tokens, consumer="phase1.consolidate")
    d = _parse(resp.content or "")
    if d is None or not isinstance(d.get("objects"), list):
        log("[consolidate] the pass returned nothing parseable — list unchanged")
        return None

    known = {_norm(p): p for p in phrases}
    out = Consolidation(objects=[], passes=1)
    seen: set[str] = set()

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
                    seen.add(_norm(other))
                    continue
                bucket.setdefault(name, []).append(other)
                seen.add(_norm(other))

    # anything the model forgot survives — losing an object in silence is the
    # one outcome this pass must never produce
    for p in phrases:
        if _norm(p) not in seen:
            out.objects.append(p)
            out.kept_unmentioned.append(p)
    return out


_MERGE_SYSTEM = (
    "You are merging the object list of a 3D scan into WHOLE objects. The list was "
    "written frame by frame, so one object often appears under several wordings, and "
    "some entries name only a part, a building unit or the material of another listed "
    "object. Return STRICT JSON only."
)


def _merge_prompt(scene_type: str, phrases: list[str]) -> str:
    listing = "\n".join(f"- {p}" for p in phrases)
    return (
        f"These phrases name things seen in ONE {scene_type or 'place'}:\n\n{listing}\n\n"
        # GENERIC (USER 2026-09-30): rules only, no example from any scene and no forced
        # identity between kinds — two kinds that are one object in a scene are fused
        # AFTER SAM3 by their shared points (segmentation.dedupe_overlap, label-agnostic)
        "Merge the list into WHOLE objects, in two ways:\n"
        "1. SAME KIND: merge the phrases that name the SAME KIND of object. Colour, "
        "material, finish, size, state and position do NOT make a different kind (USER "
        "2026-09-30: 'piso es piso').\n"
        # USER 2026-10-09: "los objetos deben ser el más grande posible, si tengo mil
        # baldosas que forman el piso el objeto es piso no baldosa. Si tengo un desk no lo
        # voy a segmentar en sus patas la tabla etc, es un objeto único". Until that night
        # this said "so does a part and its whole [stay separate]", and pccr's 'tile' went
        # to SAM3 as its own prompt: every floor tile became an object, and the session
        # grew to 77.7 GB of the card. The part's NAME is folded into the whole, never the
        # whole into the part; a part whose whole was never named stays (nothing invented).
        "2. PART INTO WHOLE: a phrase that names a PART or COMPONENT of another listed "
        "object (legs, top, drawers, wheels, handles, the frame of a door or window), a "
        "UNIT a listed surface or structure is made of (the tiles of a floor or ceiling, "
        "the bricks, boards or panels of a wall), or the MATERIAL or finish of a listed "
        "object (the drywall or plaster of a wall, tape or paint used as a covering) is "
        "folded INTO that whole object: the whole object is the 'name', the part goes in "
        "its 'parts'. A phrase naming a SEPARATE object that only touches or lies on "
        "another (a cable lying on the floor) is NOT a part. Structure is never a part: a "
        "floor, ceiling, wall, door, window, opening, column, beam or stair is always its "
        "own object.\n"
        "Different kinds stay separate. When unsure, keep them separate. The target of a "
        "merge is the plainest category word: the WHOLE object.\n"
        # ONE LINE PER PHRASE (2026-10-09, pccr): the JSON-groups answer looped on 85 names
        # (185 groups, the tail repeating) and a cut answer lost the folds. A line per input
        # phrase bounds the answer by the list itself, and a repeat is harmless: the FIRST
        # line of each phrase decides (merge_lines).
        "Answer with EXACTLY ONE LINE PER PHRASE, in the order of the list, and nothing else:\n"
        "<phrase> -> <target> | <same or part or keep>\n"
        "where <target> is a phrase FROM THE LIST written EXACTLY as in the list: the phrase "
        "itself with 'keep' when it stays its own object, the phrase of the same kind with "
        "'same', or the whole object it is a part, component, unit or material of with 'part'."
    )


_SHORT_SYSTEM = (
    "You answer short questions about the object list of a 3D scan. Answer with exactly the "
    "words asked for and nothing else."
)


def _listing_question(scene_type: str, phrases: list[str], question: str) -> str:
    listing = "\n".join(f"- {p}" for p in phrases)
    return f"These phrases name things seen in ONE {scene_type or 'place'}:\n\n{listing}\n\n{question}"


def _whole_or_piece_question(phrase: str) -> str:
    return (
        f'Is a "{phrase}" a WHOLE object on its own, or only a PIECE of something bigger: a part '
        "or component of it (the leg or top of a desk, the frame of a door), a unit it is made "
        "of (a tile of a floor or ceiling, a brick or board of a wall), or its material or "
        "covering (the drywall of a wall, tape or paint on something)? A separate object that "
        "only touches or lies on another (a cable on the floor) is WHOLE. Answer with one word: "
        "whole or piece."
    )


def _piece_of_question(phrase: str) -> str:
    return (
        f'A "{phrase}" is a piece, unit or material of which ONE thing in the list? Answer with '
        "that phrase written exactly as in the list, or none."
    )


def _same_kind_question(phrase: str) -> str:
    return (
        f'Does "{phrase}" name the SAME KIND of object as another phrase in the list (colour, '
        "material, finish, size, state and position do not make a different kind)? Answer with "
        "that other phrase written exactly as in the list, or none."
    )


def _confirm_question(kind: str, phrase: str, other: str) -> str:
    """The yes/no check every proposed link must pass (zaragoza 2026-10-09: asked which listed
    phrase is the same kind, the model named one almost every time — 'ceiling' → 'wall',
    'cable' → 'pipe', 'ladder' → 'stair' — instead of answering none)."""
    if kind == "part":
        return (f'Is a "{phrase}" a part, a building unit or the material OF a "{other}" — '
                f'something a "{other}" is made of or carries, not a separate object of its '
                f"own? Answer yes or no.")
    return (f'Are "{phrase}" and "{other}" two names for the SAME KIND of object? Two different '
            "kinds that are related, attached, next to each other or look alike are NOT the same "
            "kind. Answer yes or no.")


def _unit_of_structure_question(phrase: str, structure: str) -> str:
    return (f'Is a "{phrase}" one of the many units or pieces that a "{structure}" itself is '
            f'built of (as the tiles that make up a tiled floor), rather than a separate object '
            f'placed on, hung from or attached to it? Answer yes or no.')


def _confirmed(answer: str) -> bool:
    return answer.strip().strip("\"'`.").lower().startswith("yes")


def _listed_in(answer: str, known: dict[str, str], exclude: str) -> str | None:
    """The listed phrase an answer names: the answer itself or any ':'-separated fragment of it
    (the model writes 'tile: floor', 'same: duct'), else the LONGEST listed phrase it contains as
    whole words; never ``exclude`` (the phrase asked about)."""
    a = answer.strip().strip("\"'`.").strip()
    for frag in [a] + [f.strip().strip("\"'`.").strip() for f in a.split(":")]:
        hit = known.get(_norm(frag))
        if hit is not None and hit != exclude:
            return hit
    words = f" {_norm(a)} "
    best = None
    for k, p in known.items():
        if p != exclude and k and f" {k} " in words and (best is None or len(k) > len(_norm(best))):
            best = p
    return best


#: last words that name no kind of thing — never the general name a phrase is merged into
_NO_KIND = frozenset({"object", "thing", "item", "stuff", "something", "element", "entity"})


def _fold_word(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def suffix_merges(phrases: list[str]) -> dict[str, str]:
    """{phrase: the most general listed phrase it ENDS WITH} — decided by the names alone
    (USER 2026-10-10, observatorio: 'tunnel wall' and 'brick wall' next to 'wall', 'red box'
    next to 'box', 'industrial equipment' next to 'equipment' were each prompted on their own).
    A phrase whose words end with another listed phrase's words (plurals folded) is that thing
    qualified: the same object, merged into the SHORTEST such phrase — the object is the
    largest whole (USER 2026-10-09). A name that ends with something else ('fire hydrant
    cabinet' next to 'fire hydrant') is untouched."""
    words = {p: [_fold_word(w) for w in _norm(p).split()] for p in phrases}
    out: dict[str, str] = {}
    for p in phrases:
        wp = words[p]
        best = None
        for q in phrases:
            wq = words[q]
            if q == p or not wq or len(wq) >= len(wp) or wq[-1] in _NO_KIND:
                continue
            if wp[-len(wq):] == wq and (best is None or len(wq) < len(words[best])):
                best = q
        if best is not None:
            out[p] = best
    return out


def classify_phrases(client, scene_type: str, phrases: list[str], *,
                     log: Callable[[str], None] = print, calls: list | None = None,
                     part_of: dict | None = None, kv: dict | None = None,
                     judges: tuple | None = None, record: dict | None = None) -> dict[str, str]:
    """The merge pass as SHORT QUESTIONS PER PHRASE, in two steps a small model answers (pccr
    2026-10-09: asked 'own | same | part' in one question, Qwen3-VL 8B answered 'tile: own' and
    folded nothing). 1. whole or piece? 2. a piece: of which listed thing? — folded into it; a
    whole: the same kind as which listed phrase, or none? — merged. Answers are read leniently
    (a listed phrase named anywhere in the answer). Same contract as before: {alias: carrier},
    ``part_of`` receives the folds, ``calls`` one record per phrase, structure never a part,
    a link closing a cycle skipped, an answer naming nothing listed keeps the phrase.

    SEVERAL PHRASES AT ONCE (USER 2026-10-09: "Mandemos la mayor cantidad que podamos"): a
    phrase's question chain depends on its OWN answers only (the list and the phrase), so the
    chains go to the engine concurrently (semantic.parallel.run_ordered, as many as ``kv`` —
    the engine's KV cache — holds without preempting one: each chain bounded from above by the
    UTF-8 bytes of its longest question + the answer's bound) and are DECIDED in the list's
    order exactly as before (the alias links, the cycle test, the folds, ``calls``). ``kv``
    None: one phrase at a time, declared. A call that fails raises :class:`MergeError` — the
    first failing phrase in the list's order (point 88). ``judges`` (semantic.parallel.
    self_check_judges' result): a sample of that many phrases has its FIRST question re-asked
    ALONE after the pass and the answer compared byte for byte (declared, never a failure).
    ``record`` (a dict the caller owns) receives ``concurrency`` and ``self_check``."""
    from semantic.parallel import (SELF_CHECK_SUFFIX, AnswerCapture, Job, concurrency_plan,
                                   cost_of, not_checked, plan_line, run_ordered,
                                   sample_positions, self_check)
    from semantic.types import system, user
    known = {_norm(p): p for p in phrases}
    longest = max((len(p.encode("utf-8")) for p in phrases), default=0)
    bound = len("same: ".encode("utf-8")) + longest + 1
    alias_of: dict[str, str] = {}
    kind_of: dict[str, str] = {}
    unusable: list[str] = []
    # by the names alone, before any question: a phrase that ends with another listed phrase
    # is that thing qualified — merged, and not asked about
    pre = suffix_merges(phrases)
    for p_, q_ in pre.items():
        alias_of[p_], kind_of[p_] = q_, "same"
        if calls is not None:
            calls.append({"phrase": p_, "decision": "same", "into": q_, "by": "name_suffix"})
    if pre:
        log(f"[merge] {len(pre)} phrase(s) end with another listed name — merged by the name: "
            f"{dict(sorted(pre.items()))}")
    ask = [p_ for p_ in phrases if p_ not in pre]

    def _messages(question: str, without: str | None = None) -> list:
        # ``without``: the phrase asked about is left out of the listing when the answer must
        # name ANOTHER phrase (pccr 2026-10-09: listed, 'tile' was answered 'tile')
        shown = [q for q in phrases if q != without] if without else phrases
        return [system(_SHORT_SYSTEM), user(_listing_question(scene_type, shown, question))]

    def _ask(chat_client, p: str, question: str, without: str | None = None) -> str:
        try:
            resp = chat_client.chat(_messages(question, without), max_tokens=bound,
                                    consumer="phase1.merge_synonyms")
        except Exception as e:  # noqa: BLE001 — point 88: a merge call that fails fails the stage
            raise MergeError(f"[merge] '{p}': the VLM call failed ({e}) — the merge pass cannot be "
                             f"completed (point 88)") from e
        lines = (resp.content or "").strip().splitlines()
        return lines[0].strip() if lines else ""

    longest_phrase = max(phrases, key=lambda q: len(q.encode("utf-8")), default="")

    def _questions(p: str) -> list[str]:
        """Every question ``p``'s chain may ask, the first one first (a confirmation written
        with the longest phrase stands for any of them, for the footprint bound)."""
        return (([] if is_structural(p) else [_whole_or_piece_question(p), _piece_of_question(p),
                                              _confirm_question("part", p, longest_phrase),
                                              _unit_of_structure_question(p, longest_phrase)])
                + [_same_kind_question(p), _confirm_question("same", p, longest_phrase)])

    def _carrier(p: str) -> str:
        seen = {p}
        while p in alias_of and alias_of[p] not in seen:
            p = alias_of[p]
            seen.add(p)
        return p

    def _chain(p: str):
        """The questions of ``p`` (a worker thread when several chains are in flight): its
        record so far, the link it proposes and its first answer's text."""
        cap = AnswerCapture(client)
        rec: dict = {"phrase": p}
        kind, other = None, None
        if not is_structural(p):
            first = _ask(cap, p, _whole_or_piece_question(p))
            rec["whole_or_piece"] = first
            if "piece" in first.lower():
                ans = _ask(cap, p, _piece_of_question(p), without=p)
                rec["piece_of"] = ans
                kind, other = "part", _listed_in(ans, known, p)
                if other is not None:
                    yes = _ask(cap, p, _confirm_question("part", p, other))
                    rec["part_confirmed"] = yes
                    if not _confirmed(yes):
                        kind, other = None, None
                if other is None:
                    # USER 2026-10-09 ("todo es piso, no queremos un segmento por baldosa"): a
                    # piece whose whole the model did not name (pccr: 'tile' → "none") is asked,
                    # yes or no, whether it is a UNIT a listed whole is built of — the STRUCTURE
                    # first, then every other listed object (observatorio: 'railway sleeper' of
                    # the 'railroad track', USER "sí, ampliálo"), each in the list's order; the
                    # first yes folds it there
                    tried = {}
                    wholes = ([q for q in phrases if q != p and is_structural(q)]
                              + [q for q in phrases if q != p and not is_structural(q)])
                    for s_ in wholes:
                        yes = _ask(cap, p, _unit_of_structure_question(p, s_))
                        tried[s_] = yes
                        if _confirmed(yes):
                            kind, other = "part", s_
                            break
                    rec["unit_of_structure"] = tried
            else:
                # a WHOLE may still be one of the many units a SURFACE is built of (bufferStop
                # 2026-10-10: 'rock' answered whole, prompted alone, 307 masklets — one per stone
                # of the gravel ground; USER: "si tengo mil baldosas que forman el piso el objeto
                # es piso"): asked, yes or no, against every listed surface; the first yes folds it
                tried = {}
                for s_ in [q for q in phrases if q != p and is_surface(q)]:
                    yes = _ask(cap, p, _unit_of_structure_question(p, s_))
                    tried[s_] = yes
                    if _confirmed(yes):
                        kind, other = "part", s_
                        break
                if tried:
                    rec["unit_of_surface"] = tried
        if other is None:
            ans = _ask(cap, p, _same_kind_question(p), without=p)
            rec["same_as"] = ans
            other = _listed_in(ans, known, p)
            kind = "same" if other is not None else None
            if other is not None:
                yes = _ask(cap, p, _confirm_question("same", p, other))
                rec["same_confirmed"] = yes
                if not _confirmed(yes):
                    kind, other = None, None
        return rec, kind, other, (cap.contents[0] if cap.contents else None)

    first_answers: dict[int, str | None] = {}

    def _decide(i: int, out) -> None:
        """The chain of phrases[i], decided in the list's order (the alias links so far)."""
        rec, kind, other, text = out
        p = ask[i]
        first_answers[i] = text
        if other is None:
            rec["decision"] = "own"
            if kind is None and rec.get("piece_of") and _norm(rec["piece_of"]) != "none":
                unusable.append(p)
        elif is_structural(p) and not is_structural(other):
            # structure is the WHOLE: a wall is never renamed into its drywall (pccr 2026-10-09)
            rec["decision"] = "structural_kept"
        elif _carrier(other) == p:
            rec["decision"] = "cycle_skipped"
        else:
            alias_of[p], kind_of[p] = other, kind
            rec["decision"] = kind
            rec["into"] = other
        if calls is not None:
            calls.append(rec)

    # each chain's upper bound: its longest question's chat-template bytes + the answer's bound
    foot = [max(_chat_prompt_bytes(_SHORT_SYSTEM, _listing_question(scene_type, phrases, q))
                for q in _questions(p)) + bound for p in ask]
    conc = concurrency_plan(kv, foot, n_requests=len(ask),
                            footprint_source=f"the UTF-8 bytes of each phrase's longest question "
                                             f"(chat template included) + its answer bound {bound}")
    stats = run_ordered((Job(fn=(lambda p=p: _chain(p)), cost=cost_of(conc, f))
                         for p, f in zip(ask, foot)),
                        capacity=conc["capacity_blocks"], threads=conc["threads"],
                        on_result=_decide)
    conc["observed"] = stats
    log(f"[merge] {plan_line(conc, stats)}")
    # the byte-identity self-check: a sample of first questions re-asked ALONE
    n_judges, judges_rec = judges if judges else (None, {"error": "no sample size handed in"})
    if int(stats.get("max_in_flight") or 0) <= 1:
        check = not_checked("every chain ran alone — nothing to compare", judges_rec)
    elif n_judges is None:
        log(f"[merge] [concurrency] ⚠ DECLARED: the self-check is not run — "
            f"{judges_rec.get('error')}")
        check = not_checked(str(judges_rec.get("error")), judges_rec)
    else:
        def _alone(i: int):
            cap = AnswerCapture(client, consumer_suffix=SELF_CHECK_SUFFIX)
            # the first question exactly as the chain asked it: a structural phrase's is the
            # same-kind question, asked with the phrase left out of the listing
            p0 = ask[i]
            cap.chat(_messages(_questions(p0)[0], p0 if is_structural(p0) else None),
                     max_tokens=bound,
                     consumer="phase1.merge_synonyms")
            return cap.last

        check = self_check(sample_positions(range(len(ask)), n_judges), first_answers,
                           _alone, describe=lambda i: {"phrase": ask[i]}, judges=judges_rec,
                           what="merge questions", log=lambda m: log(f"[merge] {m}"))
    if record is not None:
        record.update(concurrency=conc, self_check=check)
    final: dict[str, str] = {}
    folds: dict[str, str] = {}
    for a in alias_of:
        x, is_part, seen = a, False, {a}
        while x in alias_of and alias_of[x] not in seen:
            is_part = is_part or kind_of[x] == "part"
            x = alias_of[x]
            seen.add(x)
        final[a] = x
        if is_part:
            folds[a] = x
    if part_of is not None:
        part_of.update(folds)
    log(f"[merge] {len(phrases)} phrase(s), short questions each (answers ≤ {bound} tokens): "
        f"{len(final) - len(folds)} merged as the same thing, "
        f"{len(folds)} folded into their whole object → {len(phrases) - len(final)} concept(s)"
        + (f"; folds: {dict(sorted(folds.items()))}" if folds else "")
        + (f"; same: {dict(sorted((a, c) for a, c in final.items() if a not in folds))}"
           if len(final) > len(folds) else "")
        + (f"; {len(unusable)} piece(s) whose whole named nothing listed, kept: {unusable}"
           if unusable else ""))
    return final


def _line_answer_bound(phrases: list[str]) -> int:
    """An UPPER BOUND on the tokens of a COMPLETE one-line-per-phrase answer: the UTF-8 bytes of
    every line written with the LONGEST phrase as its target and the longest tag (a byte-level
    BPE never has more tokens than bytes, _chat_prompt_bytes). The slack between each phrase's
    real target and the longest one is what a short preamble of the model uses."""
    if not phrases:
        return 0
    longest = max(len(p.encode("utf-8")) for p in phrases)
    tail = len(" -> ".encode("utf-8")) + longest + len(" | part\n".encode("utf-8"))
    return sum(len(p.encode("utf-8")) + tail for p in phrases)


def merge_lines(txt: str, fam: list[str]) -> tuple[list | None, dict]:
    """The one-line-per-phrase answer as the ``groups`` :func:`merge_synonyms` applies, and its
    record: the FIRST line naming each phrase of ``fam`` decides it (a repeat — a looping
    answer — is counted and ignored), a line for a phrase not in ``fam`` or with a target not in
    ``fam`` is counted and ignored, a phrase no line assigned stays its own object (counted).
    ``(None, record)`` when no line of the answer assigns any phrase (not this format)."""
    known = {_norm(p): p for p in fam}
    decided: dict[str, tuple[str, str]] = {}
    rec = {"lines": 0, "repeats": 0, "unknown": 0}
    for raw in (txt or "").splitlines():
        line = raw.strip().strip("-*• ").strip()
        if "->" not in line:
            continue
        rec["lines"] += 1
        left, right = line.split("->", 1)
        tag = "keep"
        if "|" in right:
            right, tag = right.rsplit("|", 1)
        src, dst = known.get(_norm(left)), known.get(_norm(right))
        if src is None or dst is None:
            rec["unknown"] += 1
            continue
        if src in decided:
            rec["repeats"] += 1
            continue
        tag = _norm(tag)
        decided[src] = (dst, "part" if tag.startswith("part") else "same")
    rec["assigned"] = len(decided)
    rec["unassigned"] = sorted(p for p in fam if p not in decided)
    if not decided:
        return None, rec
    groups: dict[str, dict] = {}
    for src in fam:
        if src not in decided:
            continue
        dst, kind = decided[src]
        if dst == src:
            continue
        g = groups.setdefault(dst, {"name": dst, "same_as": [], "parts": []})
        g["parts" if kind == "part" else "same_as"].append(src)
    return list(groups.values()), rec


class MergeError(RuntimeError):
    """A merge call failed or answered nothing usable (docs/plan_determinismo.md point 88,
    DECIDIDO: a fusion call that fails fails the stage — never 'kept as is', where one
    token decided whether 134 names were merged or not)."""


# Keys of ``autoprompt`` that decided the merge before 2026-10-09 and must not linger in a
# configuration that no longer reads them (a leftover would look like a decision). The
# auto-prompter's loader (session_builder.AutoPrompter) fails naming the key.
REMOVED_MERGE_KEYS = {
    "merge_max_tokens": "the merge answer's bound is derived from the served context "
                        "(semantic.backends.<backend>.max_model_len, the engine's "
                        "--max-model-len) minus the prompt (consolidate_prompts."
                        "merge_answer_bound); an answer still cut is split, never rescued",
}


def served_context_len(service: dict | None, config: dict | None, backend: str) -> tuple[int, str]:
    """The context, in tokens, of the engine that answers the merge call, and where it was
    read: the ``--max-model-len`` of the verified engine's own launch line (the identity
    ``semantic.serve`` records, ``argv``) when there is one; otherwise the job
    configuration's ``semantic.backends.<backend>.max_model_len`` — the value a launch of that
    backend serves (``semantic.serve.build_argv``). Neither → :class:`MergeError` naming the
    key: the answer's bound is never a literal."""
    argv = (service or {}).get("argv") if isinstance(service, dict) else None
    if isinstance(argv, list) and "--max-model-len" in argv:
        i = argv.index("--max-model-len")
        if i + 1 < len(argv):
            try:
                return int(argv[i + 1]), "the engine's launch line (--max-model-len)"
            except (TypeError, ValueError):
                pass
    b = (((config or {}).get("semantic") or {}).get("backends") or {}).get(backend) or {}
    v = b.get("max_model_len")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise MergeError(f"the served context cannot be read: no engine launch line carries "
                         f"--max-model-len and 'semantic.backends.{backend}.max_model_len' is "
                         f"{v!r} — the merge answer's bound is derived from it, never a literal")
    return v, f"semantic.backends.{backend}.max_model_len"


def _chat_prompt_bytes(system_text: str, user_text: str) -> int:
    """An UPPER BOUND on the prompt tokens of a text-only [system, user] call: the UTF-8 bytes
    of the text the served Qwen3-VL chat template renders for it (the weights'
    chat_template.json, no tools, string contents, generation prompt on). The tokenizer is a
    byte-level BPE: every token — a special one like <|im_start|> included — stands for at
    least one byte of that text, so the byte count is never below the token count (a character
    count would be: one non-ASCII character is several bytes and can be several tokens)."""
    rendered = (f"<|im_start|>system\n{system_text}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
    return len(rendered.encode("utf-8"))


def merge_answer_bound(context_len: int, system_text: str, user_text: str) -> tuple[int, int]:
    """``(max_tokens, prompt_bound)`` of one merge call: the answer may use ALL the room the
    served context leaves after the prompt (counted from above, :func:`_chat_prompt_bytes`) —
    an answer cut at its bound is lost information, and the folds are its last part (pccr
    2026-10-09: cut at 8 192, 0 folds). ``max_tokens`` < 1 means the prompt alone fills the
    context: the list is split without asking."""
    prompt_bound = _chat_prompt_bytes(system_text, user_text)
    return int(context_len) - prompt_bound, prompt_bound


def _is_read_timeout(e: BaseException) -> bool:
    """Whether a failed call is the CLIENT giving up on an answer the engine was still
    writing: ``requests``' ReadTimeout (no header before the bound — a non-streaming call
    sends none until the answer is complete), or urllib3's ReadTimeoutError anywhere in the
    exception's chain or arguments (requests wraps a body-read timeout in a ConnectionError).
    A connect timeout or a refused connection is NOT one: the engine never got the call."""
    try:
        from requests.exceptions import ReadTimeout
    except Exception:  # noqa: BLE001 — no requests, no such timeout
        ReadTimeout = ()  # type: ignore[assignment]
    try:
        from urllib3.exceptions import ReadTimeoutError
    except Exception:  # noqa: BLE001
        ReadTimeoutError = ()  # type: ignore[assignment]
    seen: set[int] = set()
    todo: list = [e]
    while todo:
        x = todo.pop()
        if not isinstance(x, BaseException) or id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, ReadTimeout) or isinstance(x, ReadTimeoutError):
            return True
        todo.extend([x.__cause__, x.__context__, *x.args])
    return False


def _groups_or_salvage(txt: str) -> tuple[list | None, int | None]:
    """The ``groups`` list of the merge answer, whole — or the complete group objects an
    answer that FINISHED but does not parse whole still holds (point 88: rescued, as the
    detector does). A CUT answer never comes here since 2026-10-09: it is split
    (:func:`merge_synonyms`). ``(groups, n_salvaged)``; ``n_salvaged`` is None when the answer
    parsed whole; ``(None, None)`` when nothing is usable."""
    d = _parse(txt)
    if d is not None and isinstance(d.get("groups"), list):
        return d["groups"], None
    if not txt:
        return None, None
    from .detector import _salvage_objects
    k = txt.find('"groups"')
    entries = [g for g in _salvage_objects(txt[k:] if k >= 0 else txt)
               if isinstance(g, dict) and g.get("name")]
    if not entries:
        return None, None
    return entries, len(entries)


def merge_synonyms(client, scene_type: str, phrases: list[str], head_of: Callable[[str], str],
                   max_phrases_per_call: int = 0, *, context_len: int,
                   log: Callable[[str], None] = print,
                   calls: list | None = None,
                   part_of: dict | None = None) -> dict[str, str]:
    """USER 2026-09-29: *"una segunda pasada de VLM sobre los prompts para fundir los
    que significan lo mismo"*. The whole list in ONE call while it fits
    ``max_phrases_per_call``; past it, one VLM call per HEAD-NOUN family with 2+ phrases
    (all the '… wall's together, all the '… door's together) and two phrases with
    different heads are never merged. Returns {alias: name}; every phrase not in it
    stays a prompt. Two merges, and only these: the SAME KIND (the 2026-09-22 lesson:
    a different kind is a different object — 'white workbench' ← 'desk' destroyed a
    desk), and since 2026-10-09 a PART INTO ITS WHOLE (USER: *"los objetos deben ser el
    más grande posible, si tengo mil baldosas que forman el piso el objeto es piso no
    baldosa"*): a part, component, building unit or material of another LISTED object
    is folded into that object, the whole's name survives. ``part_of`` (a dict the
    caller owns) receives {part: the whole it was folded into} — the subset of the
    returned map that is a fold, not a synonym (a synonym of a part, or a part of a
    synonym, is a part: the relation is followed to the final carrier). Declared
    limits: a part whose whole was never named stays a prompt (the pass may only regroup
    the phrases it was given); in the per-head-noun mode a part with another head than
    its whole ('tile' / 'floor') is never asked together with it. Structure is never a
    part (rule of the prompt, ENFORCED here as in :func:`consolidate`): a floor, wall,
    door … filed under 'parts' stays a prompt and is recorded in its call's
    ``structural_kept``.

    THE ANSWER IS NEVER CUT (2026-10-09 — pccr: the 85-name call was cut at its fixed
    8 192-token bound, 185 groups were rescued from a repeating answer and the folds, its
    last part, never came: 0 parts folded, 41 prompts still 'tile', 'panel', 'frame' …).
    Each call's ``max_tokens`` is ALL the room ``context_len`` (the served context,
    :func:`served_context_len`) leaves after the prompt counted from above
    (:func:`merge_answer_bound`). An answer STILL cut (finish_reason 'length') is never
    rescued in part — nor is one the client stopped waiting for (an HTTP read timeout, the
    same lost tail; the caller hands a client without one, session_builder): its list is
    split DETERMINISTICALLY in two halves of its sorted order,
    each half is merged on its own, and a final pass goes over the names that survive in
    both halves — so a part in one half still folds into its whole in the other. Every
    split call is bounded by a list strictly smaller than the one it stands for (a half; the
    survivors when the halves merged something), so the recursion is bounded by the list
    size; when the halves merged NOTHING the final pass would be the very call that was cut
    — :class:`MergeError`, the stage fails naming it (point 88: never a partial merge).

    Point 88 (DECIDIDO): a call that fails, or answers nothing usable, RAISES
    :class:`MergeError` — the stage fails instead of silently merging nothing; an answer
    that FINISHED but does not parse whole keeps the complete groups it holds (salvaged,
    declared). ``calls`` (a list the caller owns) receives one record per call for the
    census: family, size, the split it belongs to (``split``: None for a whole list, '1' /
    '2' for the halves, 'x' for the final pass, nested as '1.x' …), the served context and
    the answer's bound (``context_len``, ``prompt_bound_tokens``, ``max_tokens``),
    finish_reason, token usage, whether it was cut (``truncated``; ``read_timeout`` when
    the client stopped waiting) and how it was split
    (``split_into``), groups salvaged, merges applied (``n_merged``, of them ``n_parts``
    folds), structural phrases kept."""
    from semantic.types import system, user
    if isinstance(context_len, bool) or not isinstance(context_len, int) or context_len < 1:
        raise MergeError(f"the served context {context_len!r} is not a token count — the merge "
                         f"answer's bound is derived from it (served_context_len)")
    # the whole list in ONE call when it fits the bound (a kind is often named with
    # different head nouns: 'floor tiles' / 'floor'); per head-noun family otherwise
    if len(phrases) <= int(max_phrases_per_call):
        families = {"*": list(phrases)}
    else:
        families = {}
        for p in phrases:
            families.setdefault(head_of(p), []).append(p)
    alias_of: dict[str, str] = {}
    kind_of: dict[str, str] = {}          # alias → "same" | "part" (its own link)
    counts = {"calls": 0, "split": 0}

    def _carrier(p: str) -> str:
        """The phrase ``p`` is merged into, followed to the end (``p`` itself when none)."""
        seen = {p}
        while p in alias_of and alias_of[p] not in seen:
            p = alias_of[p]
            seen.add(p)
        return p

    def _apply(groups: list, fam: list[str], rec: dict, label: str) -> None:
        known = {_norm(p): p for p in fam}
        n_before = len(alias_of)
        n_parts = 0
        kept_structural: list[str] = []
        carriers_here: set[str] = set()   # names THIS answer merged something into
        for g in groups:
            if not isinstance(g, dict):
                continue
            name = known.get(_norm(g.get("name", "")))
            if name is None:
                continue
            # A group whose name was itself merged away earlier in the answer still
            # counts: its members follow the name to its final carrier (2026-10-09 — one
            # group 'tile' same_as ['ceramic tile'] after another 'floor' parts ['tile']
            # used to be skipped whole, and 'ceramic tile' went to SAM3 as its own prompt).
            for key, kind in (("same_as", "same"), ("parts", "part")):
                raw_list = g.get(key) or []
                if not isinstance(raw_list, list):
                    continue
                for raw in raw_list:
                    other = known.get(_norm(raw))
                    if other is None or other == name or other in alias_of:
                        continue
                    # a carrier of THIS answer is never re-filed as somebody's SYNONYM
                    # (first answer wins); a PART is folded with every name already merged
                    # into it. A carrier of an EARLIER call (a half of a cut list) is one
                    # survivor among the others of the final pass and may be merged.
                    if kind == "same" and other in carriers_here:
                        continue
                    if _carrier(name) == other:
                        continue                  # would close a cycle
                    # the envelope is never somebody's part — enforced, not trusted
                    # (USER 2026-09-15: "puertas ventanas paredes pisos techos columnas,
                    # eso es estructural, debe estar")
                    if kind == "part" and is_structural(other):
                        if other not in kept_structural:
                            kept_structural.append(other)
                        continue
                    alias_of[other] = name
                    kind_of[other] = kind
                    carriers_here.add(name)
                    n_parts += kind == "part"
        rec["n_merged"] = len(alias_of) - n_before
        rec["n_parts"] = n_parts
        rec["structural_kept"] = kept_structural
        if kept_structural:
            log(f"[merge] {label}: {len(kept_structural)} structural phrase(s) filed as a "
                f"part and KEPT as their own prompt (structure is never a part): "
                f"{kept_structural}")

    def _ask(head: str, fam: list[str], split: str) -> None:
        label = f"'{head}'" + (f" [split {split}]" if split else "")
        user_text = _merge_prompt(scene_type, fam)
        room, prompt_bound = merge_answer_bound(context_len, _MERGE_SYSTEM, user_text)
        # a complete one-line-per-phrase answer never needs more than _line_answer_bound: a
        # loop stops there, after every phrase could have been assigned (nothing is lost)
        line_bound = _line_answer_bound(fam)
        max_tokens = min(room, line_bound) if room >= 1 else room
        rec: dict = {"family": head, "n_phrases": len(fam), "split": split or None,
                     "context_len": context_len, "prompt_bound_tokens": prompt_bound,
                     "max_tokens": max_tokens, "line_answer_bound": line_bound}
        if calls is not None:
            calls.append(rec)
        if max_tokens < 1:
            # the prompt alone fills the served context: nothing to ask, the list is split
            rec.update(asked=False, no_room=True, finish_reason=None, truncated=False,
                       completion_tokens=None, salvaged=None, parsed=False, n_merged=0,
                       n_parts=0, structural_kept=[])
            log(f"[merge] {label} ({len(fam)} phrases): the prompt alone (≤ {prompt_bound} "
                f"tokens) fills the served context ({context_len}) — not asked, split")
            _split(head, fam, split, rec, label)
            return
        counts["calls"] += 1
        try:
            resp = client.chat([system(_MERGE_SYSTEM), user(user_text)],
                               max_tokens=max_tokens, consumer="phase1.merge_synonyms")
        except Exception as e:  # noqa: BLE001 — declared, and the stage fails with it
            rec.update(error=f"{type(e).__name__}: {e}")
            if _is_read_timeout(e):
                # the client stopped waiting before the engine finished: the answer was cut
                # by the HTTP read bound instead of its token bound — the same lost tail as a
                # finish_reason 'length', so the same treatment (the merge's own client has
                # no read bound, session_builder; this covers a client that still has one)
                rec.update(asked=True, finish_reason=None, truncated=True, read_timeout=True,
                           completion_tokens=None, salvaged=None, parsed=False, n_merged=0,
                           n_parts=0, structural_kept=[])
                log(f"[merge] {label} ({len(fam)} phrases): the HTTP call hit the client's "
                    f"read timeout before the engine finished (max_tokens {max_tokens}) — "
                    f"treated as a CUT answer, nothing is rescued from it")
                _split(head, fam, split, rec, label)
                return
            raise MergeError(f"[merge] {label} ({len(fam)} phrases): the VLM call failed "
                             f"({e}) — the merge pass cannot be completed (point 88)") from e
        usage = getattr(resp, "usage", None) or {}
        finish = getattr(resp, "finish_reason", None)
        rec.update(asked=True, finish_reason=finish, truncated=(finish == "length"),
                   completion_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None)
        # the one-line-per-phrase answer (the format asked since 2026-10-09): its FIRST line per
        # phrase decides, finished or stopped at its bound — a phrase no line assigned stays its
        # own object, declared; never a split, never a failure for a repeat
        line_groups, line_rec = merge_lines(resp.content or "", fam)
        if line_groups is not None:
            rec.update(format="lines", salvaged=None, parsed=True, **line_rec)
            if finish == "length" or line_rec["unassigned"]:
                log(f"[merge] {label} ({len(fam)} phrases): {line_rec['assigned']} assigned "
                    f"(finish_reason {finish!r}, {line_rec['repeats']} repeated line(s) ignored)"
                    + (f"; {len(line_rec['unassigned'])} never assigned stay their own object: "
                       f"{line_rec['unassigned']}" if line_rec["unassigned"] else ""))
            _apply(line_groups, fam, rec, label)
            return
        if finish == "length":
            # NEVER a partial merge: what a cut answer holds is not what the model would
            # have said about the rest of the list (the folds come last)
            rec.update(salvaged=None, parsed=False, n_merged=0, n_parts=0, structural_kept=[])
            log(f"[merge] {label} ({len(fam)} phrases): the answer was CUT at its bound "
                f"(finish_reason 'length', max_tokens {max_tokens} = served context "
                f"{context_len} − prompt ≤ {prompt_bound}) — nothing is rescued from it")
            _split(head, fam, split, rec, label)
            return
        groups, n_salvaged = _groups_or_salvage(resp.content or "")
        rec.update(salvaged=n_salvaged, parsed=groups is not None)
        if groups is None:
            raise MergeError(f"[merge] {label} ({len(fam)} phrases): the answer holds no "
                             f"usable group (finish_reason {finish!r}) — the merge pass cannot "
                             f"be completed (point 88)")
        if n_salvaged is not None:
            log(f"[merge] {label} ({len(fam)} phrases): the answer (finish_reason {finish!r}) "
                f"did not parse whole — {n_salvaged} complete group(s) rescued from it")
        _apply(groups, fam, rec, label)

    def _split(head: str, fam: list[str], split: str, rec: dict, label: str) -> None:
        counts["split"] += 1
        order = sorted(fam)
        half = len(order) // 2
        halves = (("1", order[:half]), ("2", order[half:]))
        rec["split_into"] = {"halves": [len(h) for _t, h in halves]}
        log(f"[merge] {label}: DECLARED split of the {len(fam)} phrase(s) in two halves of "
            f"their sorted order ({half} + {len(order) - half}), each merged on its own, then a "
            f"final pass over the names that survive in both")
        for tag, part in halves:
            if len(part) >= 2:           # a single phrase has nothing to merge with
                _ask(head, part, f"{split}.{tag}" if split else tag)
        survivors = [p for p in order if p not in alias_of]
        rec["split_into"]["survivors"] = len(survivors)
        if len(survivors) >= len(fam):
            raise MergeError(
                f"[merge] {label} ({len(fam)} phrases): the answer was cut and its halves "
                f"merged nothing — the final pass over the survivors "
                f"would be that same call; the merge pass cannot be completed (point 88: never "
                f"a partial merge)")
        if len(survivors) >= 2:
            _ask(head, survivors, f"{split}.x" if split else "x")

    for head, fam in families.items():
        if len(fam) < 2:
            continue
        _ask(head, fam, "")
    # a name that was itself merged away points to its final carrier; a fold anywhere
    # along the way makes the whole chain a fold (a synonym of a part is a part, a part
    # of a synonym is a part of the carrier)
    final: dict[str, str] = {}
    folds: dict[str, str] = {}
    for a in alias_of:
        x, is_part, seen = a, False, {a}
        while x in alias_of and alias_of[x] not in seen:
            is_part = is_part or kind_of[x] == "part"
            x = alias_of[x]
            seen.add(x)
        final[a] = x
        if is_part:
            folds[a] = x
    alias_of = final
    if part_of is not None:
        part_of.update(folds)
    log(f"[merge] {len(phrases)} phrase(s) in {len(families)} head-noun famil(y/ies), "
        f"{counts['calls']} VLM call(s)"
        + (f" ({counts['split']} cut list(s) split in halves + a final pass)"
           if counts["split"] else "")
        + f": {len(alias_of) - len(folds)} merged as the same thing, "
        f"{len(folds)} folded into their whole object (part / component / unit / material) "
        f"→ {len(phrases) - len(alias_of)} concept(s)"
        + (f"; folds: {dict(sorted(folds.items()))}" if folds else ""))
    return alias_of


def consolidate(client, scene_type: str, phrases: list[str],
                max_passes: int = 3, max_tokens: int = 4096,
                log: Callable[[str], None] = print) -> Consolidation:
    """Merge the words that name one thing and drop what is only a part.

    Runs the pass again while it still CHANGES the list and stops when a pass
    changes nothing — convergence, the same shape as the greedy correction
    loop, so there is no "right" number of passes to pick. ``max_passes`` is a
    bound against a model that oscillates, not a decision: reaching it is
    declared in the result.
    """
    current = list(phrases)
    merged: dict[str, list[str]] = {}
    parts: dict[str, list[str]] = {}
    unmentioned: list[str] = []
    structural: list[str] = []
    done = 0
    for _ in range(max(1, int(max_passes))):
        try:
            res = _one_pass(client, scene_type, current, max_tokens, log)
        except Exception as e:  # noqa: BLE001 — declared, never fatal
            log(f"[consolidate] pass failed ({e}) — keeping the list as it is")
            break
        if res is None:
            break
        done += 1
        for name, al in res.merged.items():
            merged.setdefault(name, []).extend(al)
        for name, pt in res.parts.items():
            parts.setdefault(name, []).extend(pt)
        unmentioned.extend(res.kept_unmentioned)
        structural.extend(res.kept_structural)
        if res.objects == current:
            break                       # converged: the pass changed nothing
        log(f"[consolidate] pass {done}: {len(current)} → {len(res.objects)} concept(s)")
        current = res.objects

    out = Consolidation(objects=current, merged=merged, parts=parts,
                        kept_unmentioned=sorted(set(unmentioned)),
                        kept_structural=sorted(set(structural)), passes=done)
    n_al = sum(len(v) for v in merged.values())
    n_pt = sum(len(v) for v in parts.values())
    log(f"[consolidate] {len(phrases)} → {len(current)} concept(s) in {done} pass(es): "
        f"{n_al} alias(es) merged, {n_pt} part(s) folded into their object"
        + (f", {len(out.kept_structural)} structural entr(y/ies) rescued from 'parts'"
           if out.kept_structural else "")
        + (f", {len(out.kept_unmentioned)} kept because the pass never mentioned them"
           if out.kept_unmentioned else ""))
    return out
