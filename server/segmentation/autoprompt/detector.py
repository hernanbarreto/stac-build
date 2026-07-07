# STAC-Builder — Auto-prompter: Qwen3-VL grounded open-vocabulary detector.
#
# Phase 1 step 1: keyframe -> Qwen3-VL grounded detection -> boxes + labels +
# confidence. Strict-JSON output, parsed with tolerance (one correction retry on
# broken JSON). All boxes are normalized to [0,1] xyxy (top-left origin) so the
# rest of the pipeline is resolution-agnostic and matches SAM3's normalized
# prompt convention.
#
# PROVENANCE: ours. The VLM only PROPOSES boxes/labels (vlm_proposed); no metric
# ever comes from these — geometry/tools measure (inviolable rule).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .vocabulary import Vocabulary


@dataclass
class Detection:
    label: str                     # canonical vocabulary label
    box: tuple[float, float, float, float]  # normalized xyxy in [0,1]
    confidence: float
    raw_label: str = ""            # what the VLM actually said (audit)
    frame_id: int | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "box": [round(c, 5) for c in self.box],
            "confidence": round(self.confidence, 4),
            "raw_label": self.raw_label,
            "frame_id": self.frame_id,
            "origin": "vlm_proposed",
        }


_SYSTEM = (
    "You are an open-vocabulary object detector for railway/tunnel/station "
    "construction inspection. You LOCATE and LABEL objects. You never measure. "
    "Return STRICT JSON only — no prose, no markdown."
)


def _build_user_prompt(vocab: Vocabulary) -> str:
    lines = [f'  - "{c.label}": {c.grounding_phrase}' for c in vocab.classes]
    classlist = "\n".join(lines)
    return (
        "Detect EVERY visible instance of these classes (there can be several of "
        "the same class — e.g. multiple columns):\n"
        f"{classlist}\n\n"
        "Rules:\n"
        "1. Use ONLY the class ids on the left. Ignore anything not in the list.\n"
        "2. One JSON object per instance (do NOT merge two columns into one box).\n"
        '3. Each object: {"label": <class id>, "box": [x1,y1,x2,y2], '
        '"confidence": <0..1>}\n'
        "4. box is NORMALIZED to the image size, values in [0,1], top-left "
        "origin, x1<x2, y1<y2.\n"
        "5. confidence is your visual certainty in [0,1].\n"
        "6. Output ONLY a JSON array. If nothing is visible, output [].\n"
    )


def _extract_json_array(text: str) -> list | None:
    if not text:
        return None
    # strip code fences
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # sometimes wrapped as {"objects": [...]}
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _normalize_box(
    raw: Any, img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except Exception:
        return None
    coords = [x1, y1, x2, y2]
    mx = max(abs(c) for c in coords)
    # Decide the source space: [0,1] normalized, 0-1000 normalized, or pixels.
    if mx <= 1.5:
        pass  # already [0,1]
    elif mx <= 1000.0 and mx > max(img_w, img_h):
        x1, x2 = x1 / 1000.0, x2 / 1000.0
        y1, y2 = y1 / 1000.0, y2 / 1000.0
    else:  # pixel coordinates in original image space
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
    # order + clip
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box = (
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    )
    if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
        return None
    return box


class GroundedDetector:
    """Qwen3-VL open-vocabulary grounded detector over the construction vocab."""

    def __init__(self, client, vocab: Vocabulary, max_tokens: int = 1536):
        self.client = client
        self.vocab = vocab
        self.max_tokens = max_tokens
        self._user_prompt = _build_user_prompt(vocab)

    def detect(self, image: Image.Image, frame_id: int | None = None) -> list[Detection]:
        from semantic.types import system, user  # local import: consumer-agnostic

        img_w, img_h = image.size
        messages = [system(_SYSTEM), user(self._user_prompt, images=[image])]
        resp = self.client.chat(
            messages, max_tokens=self.max_tokens, consumer="phase1.detect"
        )
        arr = _extract_json_array(resp.content or "")
        if arr is None:
            # one correction retry (spec: reintento con corrección si el JSON viene roto)
            arr = self._retry_fix(messages, resp.content or "")
        if arr is None:
            return []
        return self._parse(arr, img_w, img_h, frame_id)

    def _retry_fix(self, messages, bad_text: str) -> list | None:
        from semantic.types import assistant, user

        convo = list(messages)
        convo.append(assistant(text=bad_text))
        convo.append(
            user(
                "That was not valid JSON. Reply with ONLY a JSON array of "
                '{"label","box","confidence"} objects, nothing else.'
            )
        )
        resp = self.client.chat(convo, max_tokens=self.max_tokens, consumer="phase1.detect.retry")
        return _extract_json_array(resp.content or "")

    def _parse(self, arr: list, img_w: int, img_h: int, frame_id) -> list[Detection]:
        out: list[Detection] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            raw_label = str(item.get("label", ""))
            label = self.vocab.resolve_label(raw_label)
            if label is None:
                continue
            box = _normalize_box(item.get("box"), img_w, img_h)
            if box is None:
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = min(max(conf, 0.0), 1.0)
            out.append(Detection(label=label, box=box, confidence=conf,
                                  raw_label=raw_label, frame_id=frame_id))
        return out
