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
    ' "objects": [{"category": "<the COMMON NAME of the kind of object>", '
    '"description": "<2-5 words that describe THIS one visually: colour, material, '
    'shape>", '
    '"shape": {"form": "<the dominant geometric form, e.g. flat vertical surface, '
    'rectangular box, vertical cylinder, thin elongated bar, irregular organic volume>", '
    '"material": "<the main material(s)>", '
    '"detail": "<ONE short sentence, at most 20 words: the most distinctive visible '
    'features>"}}, ...]}\n'
    # USER 2026-10-01: "el VLM, para preparar los prompts, pasa SAM3 y las descripciones
    # para ShapeR" — the SAME call that names the SAM3 prompts describes each kind for
    # ShapeR (a 3D shape generator conditioned on a narrative caption: category, form,
    # material, detail — segmentation/object_captioner.py composes it). The entry is
    # parsed defensively: a kind without it simply has no description, nothing is
    # invented in its place.
    "The 'shape' entry is for a 3D shape generator: 'form' names the dominant geometric "
    "form of this kind of object, 'material' its main material(s), 'detail' one short "
    "sentence with its most distinctive visible features. Geometric, concrete, only "
    "what is visible.\n"
    # USER 2026-09-30: the CATEGORY is the SAM3 prompt ("prompts sin tanto detalle");
    # the DESCRIPTION is kept as the fallback when a bare category finds nothing
    # ("column" found no masklet on pccr where "metal support column" found one in 62
    # keyframes) — "a ese se lo puede probar con los que se originó".

    # USER 2026-09-30: "prompts sin tanto detalle a SAM3 … piso es piso, no piso gris, piso
    # verde con blanco … desk de metal color negro es desk, punto" — a rich phrase per
    # variant made SAM3 segment the same object 20 times under 20 names.
    # GENERIC (USER 2026-09-30: "no es un prompt genérico, es un prompt hecho para pccr"):
    # rules only, no example object from any particular scene
    "Each 'category' is the plain CATEGORY NAME of an object type, 1-2 words, the way a "
    "person would name it, at the level of generality of: furniture, tool, vehicle, "
    "pipe, column, beam, wall, floor, ceiling, door, window, stair, cable, equipment, "
    "container, sign, light. "
    # USER 2026-09-30: "agregale furniture, tools, vehicle, structural, pipes, column, etc"
    # — the GRANULARITY, stated with generic classes that fit any scene
    "NEVER colour, material, finish, size, state or position words "
    "in the category; colour and material go in 'description' only. One entry PER KIND "
    "of object (every instance of a kind is one entry). "
    "Base everything ONLY on what is visible.\n"
    # Every phrase here becomes ONE SAM3 text prompt and therefore ONE segment,
    # so a compound phrase asks the segmenter for a mask spanning two different
    # things and a second name for the same thing asks for a duplicate of it.
    # USER 2026-09-15: "son objetos compuestos, eso no está bien".
    "THREE RULES ABOUT THE LIST, and they matter more than richness:\n"
    "1. ATOMIC objects only. Never a compound and never a phrase joining two "
    "things with 'with', 'and', 'containing' or 'on': two things are two entries. "
    "The description may describe the object itself (material, colour, shape) but "
    "never a DIFFERENT object attached to it.\n"
    "2. ONE name per object. Do not name the same physical thing twice under "
    "different words. Pick the one name that fits it best and use only that.\n"
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
    tile: str | None = None               # None = the full frame; "r<i>c<j>" = a crop of it
    descriptions: dict[str, str] = field(default_factory=dict)   # category → its visual description
    # category → the ShapeR fields {category, shape, material, detail} the call gave
    # that kind (USER 2026-10-01); absent when the answer carried none
    shapes: dict[str, dict] = field(default_factory=dict)


@dataclass
class SceneUnderstanding:
    scene_type: str                       # consensus across keyframes
    summary: str                          # representative summary
    objects: list[str]                    # one phrase per NAME (see `aggregate`)
    per_frame: list[FrameUnderstanding] = field(default_factory=list)
    # every phrase folded into another because it is the SAME NAME
    # (`same_name_key`): {phrase: the phrase that carries it}
    merged: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "scene_type": self.scene_type,
            "summary": self.summary,
            "objects": self.objects,
            "merged_same_name": self.merged,
            "grouping": GROUPING_VERSION,
            "origin": "vlm_proposed",
            "per_frame": [
                {"frame_id": f.frame_id, "tile": f.tile, "scene_type": f.scene_type,
                 "summary": f.summary, "objects": f.objects, "descriptions": f.descriptions,
                 "shapes": f.shapes}
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


def understand_frame(client, image: Image.Image, frame_id: int, max_tokens: int = 512,
                     tile: str | None = None) -> FrameUnderstanding | None:
    """One VLM call on one image — a keyframe, or (``tile``) a crop of it shown
    at the frame's size. The prompt is the same for both: a crop is still an
    image of the scene. None when the answer does not parse."""
    from semantic.types import system, user
    from segmentation.object_captioner import fields_from_shape_entry
    resp = client.chat([system(_SYSTEM), user(_PROMPT, images=[image])],
                        max_tokens=max_tokens, consumer="phase1.understand")
    d = _parse(resp.content or "")
    if d is None:
        return None
    objs, descs, shapes = [], {}, {}
    for o in d.get("objects", []) or []:
        if isinstance(o, dict):                   # {"category", "description", "shape"}
            cat = _norm_obj(o.get("category") or o.get("name") or "")
            if cat:
                objs.append(cat)
                de = _norm_obj(o.get("description") or "")
                if de and de != cat:
                    descs.setdefault(cat, de)
                sh = fields_from_shape_entry(o.get("shape"), cat)
                if sh is not None:
                    shapes.setdefault(cat, sh)
        elif str(o).strip():                      # a bare string (older answers)
            objs.append(_norm_obj(o))
    return FrameUnderstanding(
        frame_id=frame_id,
        scene_type=str(d.get("scene_type", "")).strip(),
        summary=str(d.get("summary", "")).strip(),
        objects=objs,
        tile=tile,
        descriptions=descs,
        shapes=shapes,
    )


def _head_noun(phrase: str) -> str:
    """The (plural-stripped) head noun of a noun phrase, e.g. 'concrete support
    columns' -> 'column'. EVIDENCE only since 2026-09-29: the census reports it
    and the consolidation's structural rescue reads it, but two phrases sharing
    a head are no longer merged (`aggregate` folds true synonyms only —
    'cardboard box' and 'red fire alarm box' share a head and are two objects).

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
    return _singular(words[-1])


# Words that change nothing about WHICH thing a phrase names when they lead it.
_LEADING = ("a", "an", "the", "some", "several", "multiple", "many", "two", "three",
            "four", "five", "six", "seven", "eight", "nine", "ten")
# Bumped whenever the rule that folds one phrase into another changes: a
# session vocabulary derived under another rule is not reused (session_builder).
GROUPING_VERSION = "category_names_v2"   # 2026-09-30: plain category names + kind merge


def _singular(w: str) -> str:
    for suf in ("sses", "xes", "ches", "shes"):
        if w.endswith(suf):
            return w[:-2]
    return w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def same_name_key(phrase: str) -> str:
    """Two phrases are the SAME NAME — true synonyms, nothing else — when they
    differ only in case, spacing, punctuation/hyphenation, a leading article or
    count word, or plural endings: 'Fire extinguishers' = 'fire extinguisher',
    'two computer monitors' = 'computer monitor', 'plastic-wrapped appliance' =
    'plastic wrapped appliance'. A different word is a different name: 'red fire
    extinguisher' and 'fire extinguisher' both stay."""
    words = re.sub(r"[^\w\s]", " ", str(phrase).lower().replace("-", " ")).split()
    while words and (words[0] in _LEADING or words[0].isdigit()):
        words = words[1:]
    return " ".join(_singular(w) for w in words) or _norm_obj(phrase)


def aggregate(frames: list[FrameUnderstanding]) -> SceneUnderstanding:
    """Consensus scene_type + the union of every phrase the calls proposed, ONE
    per NAME (``same_name_key``).

    USER 2026-09-29: *"debe segmentar todo, absolutamente preciso y completo"*.
    This used to keep ONE phrase per HEAD NOUN, and that silently deleted
    objects: on pccr 'cardboard box' was folded into 'red fire alarm box'
    (head 'box'), the white / black / gray server racks into one, 'metal pipe'
    into 'exposed ceiling pipes' — and none of the dropped phrases was ever
    written anywhere. Only true synonyms are folded now, and every fold is kept
    in ``merged`` (the census reads it). Deciding that two DIFFERENT names are
    one object is not a lexical question: the consolidation pass groups them
    for labelling and the cloud settles identity (``segmentation.dedupe_overlap``).

    The scene type / summary come from the FULL frames — a crop of a floor is
    not a statement about what kind of place this is — unless there are none.
    (The old ``min_object_freq`` knob is gone: a frequency floor drops a phrase
    seen once, and seen once is enough to NAME a thing.)"""
    frames = [f for f in frames if f is not None]
    if not frames:
        return SceneUnderstanding("unknown", "", [], [])
    full = [f for f in frames if f.tile is None] or frames
    type_counts = Counter(f.scene_type for f in full if f.scene_type)
    scene_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"
    # representative summary = a frame whose type == consensus, longest summary
    cand = [f for f in full if f.scene_type == scene_type] or full
    summary = max((f.summary for f in cand), key=len, default="")
    obj_counts = Counter(o for f in frames for o in set(f.objects))
    first_seen: dict[str, int] = {}
    for i, f in enumerate(frames):
        for o in f.objects:
            first_seen.setdefault(o, i)
    by_key: dict[str, list[str]] = {}
    for o in obj_counts:
        by_key.setdefault(same_name_key(o), []).append(o)
    objects: list[str] = []
    merged: dict[str, str] = {}
    for key, variants in by_key.items():
        # the most used spelling carries the name (ties → the plainest — the
        # variants differ only by plural / article / punctuation, so the shortest
        # is the bare singular — then alphabetical: deterministic)
        best = sorted(variants, key=lambda v: (-obj_counts[v], len(v), v))[0]
        objects.append(best)
        for v in variants:
            if v != best:
                merged[v] = best
    objects.sort(key=lambda o: (-sum(obj_counts[v] for v in by_key[same_name_key(o)]),
                                first_seen[o], o))
    return SceneUnderstanding(scene_type=scene_type, summary=summary,
                              objects=objects, per_frame=frames, merged=merged)
