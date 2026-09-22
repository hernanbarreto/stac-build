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
        "differently worded: 'white tiled floor', 'checkered floor tiles' and "
        "'tiled floor with dark grout' are one floor. Pieces of one continuous "
        "surface are the SAME surface, not different objects — a floor is one "
        "floor even when different frames saw different patches of it.\n"
        "2. A PART is something you could not point at separately from two "
        "metres away with a handheld camera: a rack's fan, hard drive, label, "
        "cable management or mounting bracket are parts of the rack. A door "
        "handle is part of the door. Parts go in 'parts', never in 'objects'.\n"
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
