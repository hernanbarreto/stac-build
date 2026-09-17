# STAC-Builder — Auto-prompter: scene understanding (Phase 1).
#
# BEFORE localizing anything, the system COMPREHENDS what it is looking at — the
# kind of space and every object/surface present — WITHOUT assuming a domain.
# That understanding then DRIVES open-vocabulary segmentation (detect everything
# the scene actually contains, not a fixed class list). The obra vocabulary is
# only a downstream canonicalization/routing overlay, never a filter.
#
# Persisted to output/scene_understanding.json; also consumed by Phase 2
# (classification), Phase 4 (capture QC) and Phase 6 (reports).
#
# PROVENANCE: ours. Everything here is vlm_proposed.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from PIL import Image

_SYSTEM = (
    "You are a scene-understanding module for a 3D scanning pipeline. Look "
    "carefully and report WHAT you are actually seeing — do NOT assume any "
    "domain or invent things not visible. Return STRICT JSON only."
)

_PROMPT = (
    'Study this image and return JSON:\n'
    '{"scene_type": "<short phrase: what kind of place/space is this>",\n'
    ' "summary": "<1-2 sentences: the setting and what is happening>",\n'
    ' "objects": ["<RICH noun phrase for every distinct object or surface '
    'visible>", ...]}\n'
    "Each object must be a RICH, descriptive noun phrase (2-5 words) that a "
    "text-promptable segmenter can match visually: include material, color or "
    "context when it helps discriminate — e.g. 'concrete support column', "
    "'overhead fluorescent light fixture', 'yellow metro train car' — never a "
    "bare word like 'column' when a richer phrase describes it better. One "
    "phrase PER OBJECT TYPE (all wheels are one entry). Base everything ONLY "
    "on what is visible.\n"
    # Every phrase here becomes ONE SAM3 text prompt and therefore ONE segment,
    # so a compound phrase asks the segmenter for a mask spanning two different
    # things and a second name for the same thing asks for a duplicate of it.
    # USER 2026-09-15: "son objetos compuestos, eso no está bien".
    "THREE RULES ABOUT THE LIST, and they matter more than richness:\n"
    "1. ATOMIC objects only. Never a compound and never a phrase joining two "
    "things with 'with', 'and', 'containing' or 'on': write 'desk' and "
    "'monitor' as two entries, never 'desk with computer'; 'server rack' and "
    "'glass door', never 'server rack with glass doors'. The modifiers may "
    "describe the object itself (material, colour, shape) but never a "
    "DIFFERENT object attached to it.\n"
    "2. ONE name per object. Do not name the same physical thing twice under "
    "different words: if you already wrote 'white tiled floor', do not also "
    "write 'checkered floor tiles' or 'tiled floor with dark grout'. Pick the "
    "one phrase that best describes it and use only that.\n"
    # USER 2026-09-15: "ojo que Qwen no segmentó una puerta, la puerta
    # principal por ejemplo, es fundamental; puertas, ventanas, paredes, pisos,
    # techos, columnas, etc., eso es estructural, debe estar".
    "3. THE ENVELOPE IS NOT BACKGROUND. The structure that encloses the space "
    "is the most important thing in the image, not scenery to skip: every "
    "floor, ceiling, wall, DOOR, window, opening, column, beam and stair you "
    "can see must appear in the list. A door is an object even when it is "
    "shut and flush with its wall.\n"
    "Output ONLY the JSON."
)

# A phrase becomes a concept by its head noun, and a trailing prepositional
# phrase moves that head onto the WRONG word: 'desk with monitor' keys on
# 'monitor', 'floor with dark grout' on 'grout'. Rule 1 above forbids those
# phrases, but a VLM disobeys sometimes and the cost is a duplicate segment —
# pccr 2026-09-15 kept three names for one floor for exactly this reason. Cut
# the phrase at the preposition and the real head comes back.
# 'and' is deliberately NOT here: it usually joins ADJECTIVES ('white and gray
# checkered floor tiles'), and cutting there would key the phrase on 'white'.
_PREPOSITIONS = ("with", "without", "containing", "holding", "in", "on", "of",
                 "for", "under", "over", "behind", "near", "beside", "atop",
                 "against", "inside", "beneath", "above", "below", "to", "at",
                 "from", "along", "around", "leaning", "attached")


@dataclass
class FrameUnderstanding:
    frame_id: int
    scene_type: str
    summary: str
    objects: list[str] = field(default_factory=list)


@dataclass
class SceneUnderstanding:
    scene_type: str                       # consensus across keyframes
    summary: str                          # representative summary
    objects: list[str]                    # deduped union of understood objects
    per_frame: list[FrameUnderstanding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "summary": self.summary,
            "objects": self.objects,
            "origin": "vlm_proposed",
            "per_frame": [
                {"frame_id": f.frame_id, "scene_type": f.scene_type,
                 "summary": f.summary, "objects": f.objects}
                for f in self.per_frame
            ],
        }


def _parse(txt: str) -> dict | None:
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm_obj(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def understand_frame(client, image: Image.Image, frame_id: int, max_tokens: int = 512) -> FrameUnderstanding | None:
    from semantic.types import system, user
    resp = client.chat([system(_SYSTEM), user(_PROMPT, images=[image])],
                        max_tokens=max_tokens, consumer="phase1.understand")
    d = _parse(resp.content or "")
    if d is None:
        return None
    objs = [_norm_obj(o) for o in d.get("objects", []) if str(o).strip()]
    return FrameUnderstanding(
        frame_id=frame_id,
        scene_type=str(d.get("scene_type", "")).strip(),
        summary=str(d.get("summary", "")).strip(),
        objects=objs,
    )


def _head_noun(phrase: str) -> str:
    """Concept key of a noun phrase = its (plural-stripped) head noun, e.g.
    'concrete support columns' -> 'column'. Used ONLY to merge near-duplicate
    phrasings of the same concept across keyframes.

    A trailing prepositional phrase is cut first: the head of 'desk with
    monitor' is 'desk', not 'monitor', and keying it on the attached object is
    what let three phrasings of pccr's floor survive as three prompts.
    """
    words = phrase.split()
    if not words:
        return phrase
    for i, w in enumerate(words):
        if i > 0 and w in _PREPOSITIONS:
            words = words[:i]
            break
    if not words:
        return phrase
    w = words[-1]
    for suf in ("sses", "xes", "ches", "shes"):
        if w.endswith(suf):
            return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def aggregate(frames: list[FrameUnderstanding], min_object_freq: int = 1) -> SceneUnderstanding:
    """Consensus scene_type + one RICH phrase per concept across keyframes.
    Different frames phrase the same concept differently ('metro train car' /
    'subway train car'); grouping by head noun keeps exactly one — the most
    frequently used phrasing (ties → the richest) — so SAM3 gets one text
    prompt per distinct object type, never N variants of the same thing."""
    frames = [f for f in frames if f is not None]
    if not frames:
        return SceneUnderstanding("unknown", "", [], [])
    type_counts = Counter(f.scene_type for f in frames if f.scene_type)
    scene_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"
    # representative summary = a frame whose type == consensus, longest summary
    cand = [f for f in frames if f.scene_type == scene_type] or frames
    summary = max((f.summary for f in cand), key=len, default="")
    obj_counts = Counter(o for f in frames for o in set(f.objects))
    by_head: dict[str, list[str]] = {}
    for o, c in obj_counts.items():
        if c >= min_object_freq:
            by_head.setdefault(_head_noun(o), []).append(o)
    objects = []
    for head, variants in by_head.items():
        best = max(variants, key=lambda v: (obj_counts[v], len(v)))
        objects.append(best)
    objects.sort(key=lambda o: -obj_counts[o])
    return SceneUnderstanding(scene_type=scene_type, summary=summary,
                              objects=objects, per_frame=frames)
