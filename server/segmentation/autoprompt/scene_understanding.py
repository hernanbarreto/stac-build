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
    "on what is visible. Output ONLY the JSON."
)


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
    phrasings of the same concept across keyframes."""
    words = phrase.split()
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
