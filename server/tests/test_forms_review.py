"""FORMS P5 (USER 2026-10-16): the second VLM round — the fitted form drawn over the worst views,
a structured verdict, applied only when it changes the form. The client is faked; the overlays
are rendered on synthetic photos."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.catalog import FORMS, parse_answer  # noqa: E402
from reconstruction.forms.review import (  # noqa: E402
    apply_review, ask_review, overlay_image, overlays_for, review_prompt, review_schema, verdict_record)
from reconstruction.forms.silhouette import Silhouettes  # noqa: E402

CAT = list(FORMS)


def _ans(form="cylinder", **kw):
    d = {"form": form, "orientation": "vertical", "axis_aligned": True, "hollow": False, "outline": "unknown",
         "profile": "unknown", "parts": [], "proportions": "", "alternatives": [], "confidence": "high",
         "what_would_help": "", "notes": ""}
    d.update(kw)
    return parse_answer(d, CAT)


class _Qwen:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None, consumer=None, extra_body=None):
        self.calls.append({"n_images": len(messages[-1].images), "prompt": messages[-1].text,
                           "structured": extra_body is not None})
        return SimpleNamespace(content=json.dumps(self.answer) if isinstance(self.answer, dict) else self.answer,
                               finish_reason="stop")


def test_the_prompt_and_schema_carry_the_current_form_and_the_measured_facts():
    p = review_prompt(_ans("composite", parts=[{"name": "leg", "form": "cylinder", "count": 4,
                                                  "orientation": "vertical", "relation": "equal"}]),
                      "table", 3, 0.87, 4.2)
    assert "composite" in p and "4 x cylinder (leg)" in p and "87 %" in p and "4 pixel" in p
    assert "Never give a dimension" in p and "table" in p
    s = review_schema(["cylinder", "box", "freeform"])
    assert s["properties"]["problem"]["enum"][0] == "ok" and s["properties"]["form"]["enum"] == ["cylinder", "box", "freeform"]
    assert "matches" in s["required"]


def test_an_overlay_draws_the_form_and_the_mask_outline(tmp_path):
    fp = tmp_path / "000010.jpg"
    Image.new("RGB", (160, 120), (90, 90, 90)).save(fp)
    mask = np.zeros((120, 160), bool)
    mask[30:90, 60:100] = True
    pts = np.column_stack([np.linspace(62, 98, 40), np.full(40, 60.0)])     # a row of form samples
    img = overlay_image(fp, mask, pts)
    assert img.size[0] <= 160 and img.size[1] <= 120 and img.size[0] > 30
    a = np.asarray(img)
    red = (a[..., 0] > 200) & (a[..., 1] < 100)
    green = (a[..., 1] > 200) & (a[..., 0] < 100)
    assert red.sum() > 40 and green.sum() > 100


def test_the_review_is_structured_and_applied_only_when_the_form_changes():
    cur = _ans("cylinder")
    # it matches → nothing changes
    vlm = _Qwen({"matches": True, "problem": "ok", "form": "cylinder", "orientation": "vertical",
                 "axis_aligned": True, "hollow": False, "outline": "unknown", "profile": "unknown",
                 "parts": [], "confidence": "high", "notes": ""})
    got = ask_review(vlm, "p", [Image.new("RGB", (8, 8))], CAT, 512)
    assert got["error"] is None and got["verdict"]["matches"] and vlm.calls[0]["structured"]
    assert apply_review(cur, got["verdict"]) is None
    assert verdict_record(got["verdict"])["answer"]["form"] == "cylinder"
    # a size complaint with the same form: the optimiser already converged on the masks → nothing
    vlm = _Qwen({"matches": False, "problem": "too_small", "form": "cylinder", "orientation": "vertical",
                 "axis_aligned": True, "hollow": False, "outline": "unknown", "profile": "unknown",
                 "parts": [], "confidence": "medium", "notes": ""})
    got = ask_review(vlm, "p", [], CAT, 512)
    assert apply_review(cur, got["verdict"]) is None
    # a wrong form → the corrected answer
    vlm = _Qwen({"matches": False, "problem": "wrong_form", "form": "profile", "orientation": "vertical",
                 "axis_aligned": True, "hollow": None, "outline": "unknown", "profile": "I",
                 "parts": [], "confidence": "high", "notes": "an H column"})
    got = ask_review(vlm, "p", [], CAT, 512)
    new = apply_review(cur, got["verdict"])
    assert new is not None and new.form == "profile" and new.profile == "I"
    # missing parts → a composite
    vlm = _Qwen({"matches": False, "problem": "missing_parts", "form": "composite", "orientation": "unknown",
                 "axis_aligned": True, "hollow": None, "outline": "unknown", "profile": "unknown",
                 "parts": [{"name": "post", "form": "cylinder", "count": 2, "orientation": "vertical",
                            "relation": "equal"}, {"name": "rail", "form": "cylinder", "count": 1,
                                                   "orientation": "horizontal", "relation": "on top"}],
                 "confidence": "medium", "notes": ""})
    got = ask_review(vlm, "p", [], CAT, 512)
    new = apply_review(cur, got["verdict"])
    assert new.form == "composite" and [p.count for p in new.parts] == [2, 1]
    # garbage → declared, nothing applied
    vlm = _Qwen("not json at all")
    got = ask_review(vlm, "p", [], CAT, 512)
    assert got["error"] == "the answer is not JSON" and got["verdict"] is None
    assert apply_review(cur, got["verdict"]) is None
    # an unknown problem word is kept as 'other', a bad form answer is declared
    vlm = _Qwen({"matches": False, "problem": "weird", "form": "banana", "orientation": "vertical",
                 "axis_aligned": True, "hollow": None, "outline": "unknown", "profile": "unknown",
                 "parts": [], "confidence": "low", "notes": ""})
    got = ask_review(vlm, "p", [], CAT, 512)
    assert got["verdict"]["problem"] == "other" and got["verdict"]["answer"] is None
    assert "AnswerError" in got["verdict"]["answer_error"] and apply_review(cur, got["verdict"]) is None


def test_overlays_for_the_worst_views_use_the_views_projection(tmp_path):
    from test_forms_silhouette import _scene, _ans as _sans, FRAME, DEV, SIGMA  # noqa: E402
    from reconstruction.forms.templates import build_template
    truth, cloud, views = _scene()
    frames = tmp_path / "frames"
    frames.mkdir()
    for v in views:
        Image.new("RGB", (960, 720), (60, 60, 60)).save(frames / f"{v.cloud_frame:06d}.jpg")
    sil = Silhouettes(views, SIGMA, 1.345, DEV)
    t = build_template(_sans("cylinder"), cloud, FRAME, DEV)
    ov = overlays_for(sil, t, [0, 3], frames, 2048)
    assert [o["fidx"] for o in ov] == [0, 3] and all(o["image"].size[0] > 20 for o in ov)
    a = np.asarray(ov[0]["image"])
    assert ((a[..., 0] > 200) & (a[..., 1] < 100)).sum() > 50      # the form's red dots are there
