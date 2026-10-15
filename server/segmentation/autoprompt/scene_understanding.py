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
    # USER 2026-10-15 ("debe segmentar siempre ventanas puertas paredes etc, todo lo estructural …
    # no debe olvidarse lo estructural nunca"): the envelope is asked for in a field OF ITS OWN,
    # before the objects, so a plain wall or a shut door is never left out of a long list; every
    # name here is also an object (the parser adds what the objects list forgot)
    ' "structure": ["<EVERY structural element visible, each by its plain name: floor, ceiling, '
    'wall, door, window, opening, column, beam, stair, railing, duct, pipe …>", ...],\n'
    ' "objects": [{"category": "<the COMMON NAME of ONE WHOLE object>", '
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
    # THE NAME OF ONE OBJECT, NOT OF ITS CLASS (USER 2026-10-08, the segmentation rounds:
    # "the most specific common name of ONE object (desk, chair — never 'furniture')"). The
    # 2026-09-30 list ("agregale furniture, tools, vehicle, structural, pipes, column, etc")
    # used to be the level of generality of the NAME; it stays as the SCOPE — every kind of
    # thing is to be named — and each thing is named by its own common name.
    "Each 'category' is the plain COMMON NAME of ONE WHOLE OBJECT, 1-2 words, the way a "
    "person would name it: its most specific common name, never the class it belongs to "
    "(a chair is 'chair', never 'furniture'; never 'equipment', 'device' or 'object' when "
    "the thing has a name of its own). Name every kind of thing you see: furniture, "
    "tools, vehicles, pipes, columns, beams, walls, floors, ceilings, doors, windows, "
    "stairs, cables, equipment, containers, signs, lights, each by its own name. "
    "NEVER colour, material, finish, size, state or position words "
    "in the category; colour and material go in 'description' only. One entry PER KIND "
    "of object (every instance of a kind is one entry). "
    "Base everything ONLY on what is visible.\n"
    # Every phrase here becomes ONE SAM3 text prompt and therefore ONE segment,
    # so a compound phrase asks the segmenter for a mask spanning two different
    # things and a second name for the same thing asks for a duplicate of it.
    # USER 2026-09-15: "son objetos compuestos, eso no está bien".
    "FOUR RULES ABOUT THE LIST, and they matter more than richness:\n"
    # USER 2026-10-09: "los objetos deben ser el más grande posible, si tengo mil baldosas
    # que forman el piso el objeto es piso no baldosa. Si tengo un desk no lo voy a
    # segmentar en sus patas la tabla etc, es un objeto único". pccr the same night: the
    # list carried 'tile' (the floor's tiles), 'panel', 'frame', 'rod', 'tape', 'sheet',
    # 'drywall', and SAM3's session for 'tile' grew from 8 to 77.7 GB — every floor tile
    # became an object — until the card ran out at category 42 of 45; zaragoza's carried
    # 'wheel' (of the train). A part is not segmentable apart from its whole and a unit of
    # a surface multiplies one object into hundreds, so the rule is stated FIRST. The
    # examples are generic (USER 2026-09-30), never one scene's objects.
    "1. WHOLE OBJECTS, NEVER THEIR PIECES. Each entry is ONE WHOLE OBJECT: the largest "
    "physical unit a person would call one thing (a floor, a wall, a ceiling, a door, a "
    "chair, a car). NEVER its parts or components (legs, top, drawers, wheels, handles, "
    "the frame of a door or window). NEVER the units a surface or structure is made of "
    "(the tiles, bricks, boards or panels of a floor, wall or ceiling). NEVER a material "
    "or finish (drywall, plaster, concrete, paint, tape used as a covering). A part, a "
    "unit or a covering belongs to the object it is part of: name that whole object "
    "instead, once. A floor made of a thousand tiles is ONE entry, 'floor'. A separate "
    "object that only touches another (a cable lying on the floor) is its own entry.\n"
    # 'atomic' means ONE thing per entry (no compound phrase), never the smallest piece:
    # rule 1 above says which unit the thing is.
    "2. ATOMIC entries, one thing each. Never a compound and never a phrase joining two "
    "things with 'with', 'and', 'containing' or 'on': two things are two entries. "
    "The description may describe the object itself (material, colour, shape) but "
    "never a DIFFERENT object attached to it.\n"
    "3. ONE name per object. Do not name the same physical thing twice under "
    "different words. Pick the one name that fits it best and use only that.\n"
    # USER 2026-09-15: "ojo que Qwen no segmentó una puerta, la puerta
    # principal por ejemplo, es fundamental; puertas, ventanas, paredes, pisos,
    # techos, columnas, etc., eso es estructural, debe estar".
    "4. THE ENVELOPE IS NOT BACKGROUND. The structure that encloses the space "
    "is the most important thing in the image, not scenery to skip: every "
    "floor, ceiling, wall, DOOR, window, opening, column, beam and stair you "
    "can see must appear in the list. A door is an object even when it is "
    "shut and flush with its wall. List EVERY structural element you can see in "
    "'structure' as well — each wall, floor, ceiling, door, window, opening, column, "
    "beam, stair, railing, duct and pipe, even when plain, partly visible or at the "
    "edge of the image — and give each one its own 'objects' entry too.\n"
    # USER 2026-10-15: "debe segmentar a máximo detalle" — every distinct thing is its own
    # entry; a whole object is never hidden inside a bigger one it stands on or hangs from
    "5. MAXIMUM DETAIL. Every distinct thing that is an object of its own gets its own "
    "entry, however small, and nothing is folded into the floor, the wall or the furniture "
    "it stands on, hangs from or is fixed to: a screen standing on a bench, a sign on a "
    "wall, a cabinet against a wall, a lamp hanging from the ceiling are each their own "
    "entry. Only the parts, units and coverings of rule 1 are not entries.\n"
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
    # the structural elements the answer listed in its 'structure' field (USER 2026-10-15),
    # normalised; every one is also in ``objects``
    structure: list[str] = field(default_factory=list)
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
    # the scene-type vote (point 89): every answer counted, most voted first, so the
    # margin of the winner over the runner-up is on record
    scene_type_votes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        votes = list(self.scene_type_votes.items())
        return {
            "scene_type": self.scene_type,
            "summary": self.summary,
            "objects": self.objects,
            # sorted by key (point 87): the bytes of this file never follow a hash seed
            "merged_same_name": dict(sorted(self.merged.items())),
            "scene_type_votes": dict(votes),
            "scene_type_margin": ((votes[0][1] - votes[1][1]) if len(votes) > 1
                                  else (votes[0][1] if votes else 0)),
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


_SALVAGE_FIELD = {"scene_type": re.compile(r'"scene_type"\s*:\s*"((?:[^"\\]|\\.)*)"'),
                  "summary": re.compile(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"')}


def parse_or_salvage(txt: str) -> tuple[dict | None, int | None]:
    """The answer's JSON, or what a CUT answer still holds (docs/plan_determinismo.md point
    88 — DECIDIDO: the complete JSON objects of a truncated answer are rescued, as the
    detector already does, instead of the whole frame's objects being thrown away because
    one token hit max_tokens). Returns ``(doc, n_salvaged)``: ``n_salvaged`` is None when the
    answer parsed whole, the number of complete ``objects`` entries recovered otherwise —
    ``scene_type`` / ``summary`` come from their own fields when they were written before the
    cut. ``(None, None)`` when nothing at all is usable."""
    d = _parse(txt)
    if d is not None:
        return d, None
    if not txt:
        return None, None
    from .detector import _salvage_objects
    k = txt.find('"objects"')
    tail = txt[k:] if k >= 0 else txt
    entries = [o for o in _salvage_objects(tail) if isinstance(o, dict)
               and (o.get("category") or o.get("name"))]
    if not entries:
        return None, None
    doc: dict = {"objects": entries}
    for key, rx in _SALVAGE_FIELD.items():
        m = rx.search(txt[:k] if k >= 0 else txt)
        if m:
            try:
                doc[key] = json.loads('"' + m.group(1) + '"')
            except ValueError:
                doc[key] = m.group(1)
    return doc, len(entries)


def _norm_obj(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def understand_frame(client, image: Image.Image, frame_id: int, max_tokens: int = 512,
                     tile: str | None = None, extra: str | None = None,
                     prompt: str | None = None,
                     record: dict | None = None) -> FrameUnderstanding | None:
    """One VLM call on one image — a keyframe, or (``tile``) a crop of it shown
    at the frame's size. The prompt is the same for both: a crop is still an
    image of the scene; ``extra`` (the second pass over the unsegmented region,
    segmentation/second_pass.py) adds one instruction in front of it, the rules
    stay. ``prompt`` replaces the shipped understanding prompt — the session's own
    (USER 2026-10-05: edited and saved from the Autosegment window,
    segmentation/autoprompt/autosegment.py). None when nothing of the answer is
    usable (a cut answer keeps its complete objects — :func:`parse_or_salvage`).

    ``record`` (a dict the caller owns) is filled with the call's facts for the census
    (points 88 / 94): the sha1 of every image sent (as encoded by the client — the bytes
    the engine saw), ``finish_reason``, the token usage, whether the answer was cut
    (``truncated``), how many objects were salvaged from a cut answer, ``parsed``."""
    from semantic.types import system, user
    from segmentation.object_captioner import fields_from_shape_entry
    base = prompt if (prompt and prompt.strip()) else _PROMPT
    prompt = (f"{extra}\n" + base) if extra else base
    msg = user(prompt, images=[image])
    if record is not None:
        record["image_sha1"] = [r.sha1 for r in msg.images]
    resp = client.chat([system(_SYSTEM), msg], max_tokens=max_tokens,
                       consumer="phase1.understand")
    d, n_salvaged = parse_or_salvage(resp.content or "")
    if record is not None:
        usage = getattr(resp, "usage", None) or {}
        finish = getattr(resp, "finish_reason", None)
        record.update(finish_reason=finish,
                      completion_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
                      prompt_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                      truncated=(finish == "length"), salvaged=n_salvaged,
                      parsed=d is not None)
    if d is None:
        return None
    objs, descs, shapes = [], {}, {}
    # the envelope named apart (USER 2026-10-15): every entry of 'structure' is an object too —
    # added when the objects list forgot it, in the answer's order
    structure = []
    raw_struct = d.get("structure")
    if isinstance(raw_struct, list):
        for s_ in raw_struct:
            name = _norm_obj(str(s_.get("category") or s_.get("name") or "") if isinstance(s_, dict) else str(s_))
            if name and name not in structure:
                structure.append(name)
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
    have = {same_name_key(o) for o in objs}
    for name in structure:
        if same_name_key(name) not in have:
            objs.append(name)
            have.add(same_name_key(name))
    return FrameUnderstanding(
        frame_id=frame_id,
        scene_type=str(d.get("scene_type", "")).strip(),
        summary=str(d.get("summary", "")).strip(),
        objects=objs,
        tile=tile,
        descriptions=descs,
        shapes=shapes,
        structure=structure,
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
# 2026-09-30 v2: plain category names + kind merge. 2026-10-09 v3 (USER: "los objetos deben
# ser el más grande posible"): one WHOLE object per name, and the merge pass folds a part /
# component / building unit / material INTO the listed whole it belongs to.
GROUPING_VERSION = "support_folds_v4"     # USER 2026-10-15: folds only into support, structure always


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
    # most voted first; a tie goes to the type seen first (Counter.most_common is a stable
    # sort over insertion order) — recorded with its margin (point 89)
    votes = type_counts.most_common()
    scene_type = votes[0][0] if votes else "unknown"
    # representative summary = a frame whose type == consensus, longest summary
    cand = [f for f in full if f.scene_type == scene_type] or full
    summary = max((f.summary for f in cand), key=len, default="")
    # ORDERED dedupe per frame (point 87): a str set iterates in the order of the process's
    # hash seed, which then wrote by_key / merged / the file's bytes in another order on
    # every run (verified with 4 seeds)
    obj_counts = Counter(o for f in frames for o in dict.fromkeys(f.objects))
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
                              objects=objects, per_frame=frames, merged=merged,
                              scene_type_votes=dict(votes))
