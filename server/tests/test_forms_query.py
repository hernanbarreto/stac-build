"""FORMS P1 (USER 2026-10-16): the VLM decides the FORM — a closed catalogue, a structured
answer per object from its isolated mask views, recorded with its stamp. No GPU, no vLLM: the
semantic client is faked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.catalog import (  # noqa: E402
    FORMS, AnswerError, FormsConfigError, answer_from_dict, answer_schema, family_of,
    load_forms_config, parse_answer)
from reconstruction.forms.query import (  # noqa: E402
    build_prompt, form_of_record, load_record, query_session_forms, record_path)
from segmentation import mask_space  # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
CONFIG_YAML = SERVER / "config.yaml"
KF_VIDEO = [10, 40, 70, 100]


def _cfg():
    return yaml.safe_load(CONFIG_YAML.read_text())


# ── the configuration ──────────────────────────────────────────────────────────

def test_production_config_declares_the_block():
    c = load_forms_config(_cfg())
    assert c.enabled is True and set(c.catalog) <= set(FORMS) and "freeform" in c.catalog
    assert c.views >= 1 and c.max_rounds >= 1 and c.device in ("cuda", "cpu")


@pytest.mark.parametrize("drop", ["enabled", "catalog", "views", "max_rounds", "device",
                                  "block_iterations", "seed"])
def test_a_missing_key_fails_naming_it(drop):
    raw = _cfg()
    del raw["forms"][drop]
    with pytest.raises(FormsConfigError, match=f"forms.{drop}"):
        load_forms_config(raw)


def test_a_form_outside_the_catalogue_cannot_be_configured():
    raw = _cfg()
    raw["forms"]["catalog"] = ["cylinder", "banana"]
    with pytest.raises(FormsConfigError, match="banana"):
        load_forms_config(raw)
    with pytest.raises(ValueError):
        family_of("banana")
    assert family_of("box") == "section" and family_of("freeform") == "freeform"


# ── the answer ─────────────────────────────────────────────────────────────────

def test_the_schema_enums_are_the_catalogue():
    s = answer_schema(["cylinder", "panel", "composite", "freeform"])
    assert s["properties"]["form"]["enum"] == ["cylinder", "panel", "composite", "freeform"]
    assert "parts" in s["required"] and "confidence" in s["required"]
    assert s["properties"]["parts"]["items"]["properties"]["form"]["enum"] == ["panel", "cylinder", "freeform"]


def test_parse_validates_against_the_catalogue():
    cat = list(FORMS)
    a = parse_answer({"form": "cylinder", "orientation": "vertical", "axis_aligned": True,
                      "hollow": True, "outline": "unknown", "profile": "unknown", "parts": [],
                      "proportions": "tall", "alternatives": ["profile", "cylinder", "nope"],
                      "confidence": "high", "what_would_help": "", "notes": ""}, cat)
    assert a.form == "cylinder" and a.family == "section" and a.hollow is True
    assert a.alternatives == ["profile"]                 # itself and the unknown are dropped
    d = a.to_dict()
    assert d["provenance"] == "vlm_proposed" and answer_from_dict(d).to_dict() == d
    with pytest.raises(AnswerError, match="form"):
        parse_answer({"form": "banana"}, cat)
    with pytest.raises(AnswerError, match="no parts"):
        parse_answer({"form": "composite", "parts": []}, cat)
    with pytest.raises(AnswerError, match="hollow"):
        parse_answer({"form": "box", "hollow": "yes"}, cat)
    with pytest.raises(AnswerError):
        parse_answer("cylinder", cat)
    c = parse_answer({"form": "composite", "parts": [
        {"name": "top", "form": "panel", "count": 1, "orientation": "horizontal", "relation": ""},
        {"name": "leg", "form": "cylinder", "count": 4, "orientation": "vertical",
         "relation": "four equal legs under the top"}]}, cat)
    assert [p.count for p in c.parts] == [1, 4] and c.confidence == "medium"


def test_the_prompt_names_the_catalogue_and_never_asks_for_metres():
    p = build_prompt(["cylinder", "freeform"], "column", 3, "Column, vertical cylinder, concrete.")
    assert '"cylinder" = ' in p and '"freeform" = ' in p and '"panel"' not in p
    assert "never give a dimension" in p and "Column, vertical cylinder" in p and "3 view(s)" in p


# ── the session ────────────────────────────────────────────────────────────────

def _mask(n: int) -> np.ndarray:
    m = np.zeros((9, 16), bool)
    m[2:2 + min(n, 7), 3:3 + min(n, 12)] = True
    return m


def _session(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, v in enumerate(KF_VIDEO):
        Image.new("RGB", (64, 36), (40 * i, 10, 10)).save(frames / f"{v:06d}.jpg")
    out = tmp_path / "output"
    out.mkdir()
    (out / "camera_frames.txt").write_text("\n".join(str(v) for v in KF_VIDEO) + "\n")
    masks = {"f0_o0": _mask(4), "f1_o0": _mask(9), "f2_o0": _mask(1), "f3_o0": _mask(6),
             "f1_o1": _mask(2),
             "obj_ids": np.array([0, 1, 2], np.int32), "frames": np.array([0, 1, 2, 3], np.int32),
             "scaled_res": np.array([9, 16], np.int32),
             mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME)}
    np.savez_compressed(out / "seg_masks.npz", **masks)
    (out / "segmentation.json").write_text(json.dumps({
        "prompt": "column;door;ghost", "mask_file": "seg_masks.npz",
        "instances": [{"id": 0, "label": "column", "instance_id": 1},
                      {"id": 1, "label": "door", "instance_id": 2},
                      {"id": 2, "label": "ghost", "instance_id": 3}]}))
    (out / "segmentation_result.json").write_text(json.dumps({
        "instances": [{"id": 0, "instance_id": 1, "label": "column", "globalIndices": [0, 1, 2],
                       "shape_caption": {"caption": "Column, vertical cylinder, concrete."}},
                      {"id": 1, "instance_id": 2, "label": "door", "globalIndices": [3]},
                      {"id": 2, "instance_id": 3, "label": "ghost", "globalIndices": [4]}],
        "total_points": 5}))
    return out


_CYL = {"form": "cylinder", "orientation": "vertical", "axis_aligned": True, "hollow": False,
        "outline": "unknown", "profile": "unknown", "parts": [], "proportions": "tall and thin",
        "alternatives": ["profile"], "confidence": "high", "what_would_help": "", "notes": ""}
_PANEL = {**_CYL, "form": "panel", "outline": "rectangle", "alternatives": []}


class _Qwen:
    def __init__(self, answers):
        self.answers = dict(answers)
        self.calls = []
        self.backend = SimpleNamespace(model="Qwen3-VL-8B", name="qwen_local")

    def chat(self, messages, temperature=None, max_tokens=None, consumer=None, extra_body=None):
        text = messages[-1].text
        self.calls.append({"n_images": len(messages[-1].images), "structured": extra_body is not None,
                           "prompt": text})
        label = text.split("The element is tagged as: ")[1].split(".")[0]
        a = self.answers[label]
        return SimpleNamespace(content=(json.dumps(a) if isinstance(a, dict) else a),
                               finish_reason="stop")


def test_one_structured_call_per_object_recorded_with_its_stamp(tmp_path):
    out = _session(tmp_path)
    fcfg = load_forms_config(_cfg())
    vlm = _Qwen({"column": _CYL, "door": _PANEL})
    st = query_session_forms(out, tmp_path, fcfg, client=vlm, log=lambda m: None)
    assert st["asked"] == 2 and st["declared"] == 1 and st["kept"] == 0 and st["n_instances"] == 3
    assert len(vlm.calls) == 2 and all(c["structured"] for c in vlm.calls)
    assert vlm.calls[0]["n_images"] == 4 and vlm.calls[1]["n_images"] == 1   # the column: 4 masks
    assert "Column, vertical cylinder" in vlm.calls[0]["prompt"]
    rec = load_record(out, "column_1")
    assert rec["form"]["form"] == "cylinder" and rec["form"]["provenance"] == "vlm_proposed"
    assert rec["stamp"]["n_views"] == 4 and len(rec["stamp"]["images_sha1"]) == 4
    assert rec["stamp"]["model"] == "Qwen3-VL-8B" and rec["rounds"][0]["kind"] == "query"
    assert [v["position"] for v in rec["views"]] == [1, 3, 0, 2]      # largest masks first
    assert form_of_record(rec).form == "cylinder"
    ghost = load_record(out, "ghost_3")
    assert ghost["form"] is None and "no usable mask view" in ghost["error"]
    assert form_of_record(ghost) is None
    # the same stamp: kept, no second call; refresh asks again
    st2 = query_session_forms(out, tmp_path, fcfg, client=vlm, log=lambda m: None)
    assert st2["kept"] == 2 and st2["asked"] == 0 and len(vlm.calls) == 2
    st3 = query_session_forms(out, tmp_path, fcfg, client=vlm, refresh=True, instance_ids=[1],
                              log=lambda m: None)
    assert st3["asked"] == 1 and len(vlm.calls) == 3


def test_an_unusable_answer_is_declared_never_a_form(tmp_path):
    out = _session(tmp_path)
    fcfg = load_forms_config(_cfg())
    vlm = _Qwen({"column": "I think it is a column, a cylinder.", "door": {"form": "composite", "parts": []}})
    st = query_session_forms(out, tmp_path, fcfg, instance_ids=[1, 2], client=vlm, log=lambda m: None)
    assert st["asked"] == 0 and st["declared"] == 2
    col = load_record(out, "column_1")
    assert col["form"] is None and col["error"] == "the answer is not JSON" and col["rounds"][0]["raw"]
    door = load_record(out, "door_2")
    assert door["form"] is None and "AnswerError" in door["error"]
    assert record_path(out, "door_2").exists()
