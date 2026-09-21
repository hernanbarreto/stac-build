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
              log: Callable[[str], None]) -> tuple[Optional[Consolidation], dict]:
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
    note.update(status="applied", reason="", after=len(out.objects),
                merged={k: list(v) for k, v in out.merged.items()},
                parts={k: list(v) for k, v in out.parts.items()},
                kept_structural=list(out.kept_structural),
                kept_unmentioned=list(out.kept_unmentioned))
    return out, note


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
    history: list[dict] = []
    parse_failed = False
    stopped = ""
    done = 0
    bound = max(1, int(max_passes))
    for _ in range(bound):
        try:
            res, note = _one_pass(client, scene_type, current, max_tokens, log)
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
        for name, al in res.merged.items():
            merged.setdefault(name, []).extend(al)
        for name, pt in res.parts.items():
            parts.setdefault(name, []).extend(pt)
        unmentioned.extend(res.kept_unmentioned)
        structural.extend(res.kept_structural)
        if res.objects == current:
            note["status"] = "converged"
            stopped = "converged: the pass changed nothing"
            break                       # converged: the pass changed nothing
        log(f"[consolidate] pass {done}: {len(current)} → {len(res.objects)} concept(s)")
        current = res.objects

    # the bound is a BOUND, not a decision — but reaching it means the list was
    # still moving when we stopped, and that has to be declared, not inferred
    max_reached = (len(history) >= bound and not parse_failed
                   and bool(history) and history[-1].get("status") == "applied")
    if max_reached:
        stopped = (f"the {bound}-pass bound was reached with the list still "
                   f"changing")

    out = Consolidation(objects=current, merged=merged, parts=parts,
                        kept_unmentioned=sorted(set(unmentioned)),
                        kept_structural=sorted(set(structural)), passes=done,
                        input_objects=list(phrases), history=history,
                        parse_failed=parse_failed,
                        max_passes_reached=max_reached, stopped_reason=stopped)
    n_al = sum(len(v) for v in merged.values())
    n_pt = sum(len(v) for v in parts.values())
    log(f"[consolidate] {len(phrases)} → {len(current)} concept(s) in {done} pass(es): "
        f"{n_al} alias(es) merged, {n_pt} part(s) folded into their object"
        + (f", {len(out.kept_structural)} structural entr(y/ies) rescued from 'parts'"
           if out.kept_structural else "")
        + (f", {len(out.kept_unmentioned)} kept because the pass never mentioned them"
           if out.kept_unmentioned else ""))
    warn = out.warning()
    if warn:
        log(f"[consolidate] ⚠️ {warn}")
    return out
