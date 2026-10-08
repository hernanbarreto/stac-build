"""Wave 2, package P2 — the VLM analysis (docs/plan_determinismo.md points 82, 86, 87, 88,
89, 94, 97, 159): ordered dedupe and sorted bytes, cut answers salvaged and recorded,
the stamp that reuses vlm_analysis.json whole, the prompt record, the decisions' margins,
no clock in autosegment.json and one source for an edited SAM3 list. CPU only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from segmentation.autoprompt import scene_understanding as SU  # noqa: E402
from segmentation.autoprompt.scene_understanding import FrameUnderstanding, aggregate  # noqa: E402


# ── point 87: bytes never follow a hash seed ──────────────────────────────────────────

def test_aggregate_dedupes_in_order_and_writes_merged_sorted():
    f1 = FrameUnderstanding(0, "room", "s", ["zebra wall", "apple", "zebra wall", "Apples"])
    f2 = FrameUnderstanding(1, "room", "s", ["apple", "door"])
    su = aggregate([f1, f2])
    assert su.objects[0] == "apple", "most proposals first"
    assert su.merged == {"Apples": "apple"}
    d = su.to_dict()
    assert list(d["merged_same_name"]) == sorted(d["merged_same_name"])
    assert d["scene_type_votes"] == {"room": 2} and d["scene_type_margin"] == 2
    src = Path(SU.__file__).read_text()
    assert "set(f.objects)" not in src and "dict.fromkeys(f.objects)" in src


# ── point 88: a cut answer keeps its complete objects, every call is recorded ─────────

CUT = ('{"scene_type": "server room", "summary": "racks", "objects": ['
       '{"category": "rack", "description": "black", "shape": {"form": "box", "material": "steel", "detail": "d"}}, '
       '{"category": "floor", "description": "grey"}, {"category": "ca')


def test_a_cut_understanding_answer_is_salvaged_and_the_call_recorded():
    doc, n = SU.parse_or_salvage(CUT)
    assert n == 2 and [o["category"] for o in doc["objects"]] == ["rack", "floor"]
    assert doc["scene_type"] == "server room" and doc["summary"] == "racks"
    assert SU.parse_or_salvage('{"a": 1}') == ({"a": 1}, None)
    assert SU.parse_or_salvage("nothing here") == (None, None)

    class _Cut:
        def chat(self, msgs, max_tokens=0, consumer=""):
            return SimpleNamespace(content=CUT, finish_reason="length",
                                   usage={"completion_tokens": 2048, "prompt_tokens": 900})

    rec = {}
    fu = SU.understand_frame(_Cut(), Image.new("RGB", (8, 8)), 7, record=rec)
    assert fu.objects == ["rack", "floor"] and fu.scene_type == "server room"
    assert rec["truncated"] is True and rec["salvaged"] == 2 and rec["parsed"] is True
    assert rec["finish_reason"] == "length" and rec["completion_tokens"] == 2048
    assert len(rec["image_sha1"]) == 1 and len(rec["image_sha1"][0]) == 40, "the sha1 of the image sent"


# ── the stamp, the reuse, the records (82 / 89 / 94 / 97) ───────────────────────────

class _VLM:
    def __init__(self, scene="server room"):
        self.calls = 0
        self.scene = scene

    def chat(self, messages, max_tokens=None, consumer=None):
        self.calls += 1
        return SimpleNamespace(content=json.dumps({
            "scene_type": self.scene, "summary": "s",
            "objects": [{"category": "desk", "description": "black metal desk",
                         "shape": {"form": "rectangular box", "material": "steel",
                                   "detail": "two drawers"}},
                        {"category": "floor", "description": "grey tiles"}]}),
            finish_reason="stop", usage={"completion_tokens": 50, "prompt_tokens": 800})


def _session(tmp_path, n=6):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    files = []
    for i in range(n):
        fn = f"{i * 7:06d}.jpg"
        Image.new("RGB", (64, 36), (i, i, i)).save(frames / fn)
        files.append(fn)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    raw = yaml.safe_load((SERVER / "config.yaml").read_text())
    raw["autoprompt"]["consolidate_prompts"] = False
    raw["autoprompt"]["merge_synonyms"] = False
    raw["autoprompt"]["vlm_sampling"].update(all_keyframes=False, spacing_kf=3, tile_rows=1,
                                             tile_cols=1, max_calls=300)
    raw["reconstruction"]["simple"]["enabled"] = True
    return raw


def _run(tmp_path, raw, monkeypatch, vlm, service=None, out="output"):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: vlm)
    return AutoPrompter(tmp_path, tmp_path / out, config=raw).run(service=service)


def test_vlm_analysis_is_stamped_recorded_and_reused_whole_under_the_same_stamp(tmp_path, monkeypatch):
    raw = _session(tmp_path)
    service = {"sha256": "engine-a", "served_model_name": "qwen_local", "weights": {"revision": "r"}}
    vlm = _VLM()
    res = _run(tmp_path, raw, monkeypatch, vlm, service=service)
    doc = json.loads(Path(res.vlm_analysis_path).read_text())
    # 6 keyframes one every 3 (both ends in) = 3 VLM frames, one call each
    assert res.reused is False and vlm.calls == 3 == doc["census"]["n_calls"]
    # the stamp (repro.stamp): the keyframes' bytes, the code, the named sections
    st = doc["stamp"]
    assert set(st) == {"stamp_version", "inputs", "code", "config", "sha256"}
    assert {"frames/000000.jpg", "frames/000007.jpg", "frames/selected_frames.json"} <= set(st["inputs"])
    assert any(k.endswith("scene_understanding.py") for k in st["code"])
    assert {"understand_prompt", "vlm_sampling", "max_sam3_prompts", "merge", "service",
            "grouping", "sam3_fallback_max"} <= set(st["config"])
    # point 97: the effective prompt, its sha and its origin
    up = doc["understand_prompt"]
    assert up["source"] == "shipped" and up["text"] == SU._PROMPT
    assert up["sha256"] == hashlib.sha256(SU._PROMPT.encode()).hexdigest()
    # point 155 / 82: the engine that answered
    assert doc["service"] == service
    # point 94: per call the image sha1 and the client's encoder; 88: finish_reason
    c = doc["census"]["calls"][0]
    assert len(c["image_sha1"][0]) == 40 and c["finish_reason"] == "stop" and c["truncated"] is False
    assert doc["census"]["image_encoding"]["format"] == "JPEG"
    assert doc["census"]["image_encoding"]["pillow"]["version"]
    # point 89: the margins of every integer cut-off
    dec = doc["census"]["decisions"]
    assert dec["prompt_bound"]["bound_reached"] is False and dec["prompt_bound"]["margin_proposals"] is None
    assert dec["prompt_bound"]["last_admitted"] == {"name": "floor", "n_proposals": 3}, \
        "the last name admitted is the list's last (desk;floor), with its proposals"
    assert dec["scene_type"] == {"winner": "server room", "votes": {"server room": 3}, "margin": 3}
    assert dec["merge_strategy"] is None, "the merge pass is off in this configuration"
    # point 86: no clock in the file; the times are beside it
    assert "generated" in doc["shape_descriptions"]["desk"]
    assert not doc["shape_descriptions"]["desk"]["generated"].startswith("20")
    timing = json.loads((tmp_path / "output" / "vlm_analysis.timing.json").read_text())
    assert "understanding_s" in timing and timing["n_calls"] == 3
    rec = json.loads((tmp_path / "output" / "autoprompt_concepts.json").read_text())
    assert rec["stamp"] == st["sha256"] and rec["prompts"] == ["desk", "floor"]

    # the SAME inputs → the file is reused WHOLE, the VLM is not called
    vlm2 = _VLM()
    res2 = _run(tmp_path, raw, monkeypatch, vlm2, service=service)
    assert res2.reused is True and vlm2.calls == 0 and res2.prompt == "desk;floor"
    assert json.loads(Path(res2.vlm_analysis_path).read_text()) == doc, "not rewritten"

    # another engine identity → derived again (and the log says why)
    vlm3 = _VLM()
    res3 = _run(tmp_path, raw, monkeypatch, vlm3, service=dict(service, sha256="engine-b"))
    assert res3.reused is False and vlm3.calls == 3

    # another prompt (the session's own) → another stamp, derived again, origin 'session'
    from segmentation.autoprompt import autosegment as AS
    AS.save_vlm_prompt(tmp_path / "output", "Name every object. JSON.")
    vlm4 = _VLM()
    res4 = _run(tmp_path, raw, monkeypatch, vlm4, service=service)
    doc4 = json.loads(Path(res4.vlm_analysis_path).read_text())
    assert vlm4.calls == 3 and doc4["understand_prompt"]["source"] == "session"
    assert doc4["stamp"]["config"]["understand_prompt"] != st["config"]["understand_prompt"]

    # a changed keyframe byte → another stamp
    Image.new("RGB", (64, 36), (9, 9, 9)).save(tmp_path / "frames" / "000000.jpg")
    vlm5 = _VLM()
    assert _run(tmp_path, raw, monkeypatch, vlm5, service=service).reused is False and vlm5.calls == 3

    # reuse_vocabulary false → always derived
    raw2 = json.loads(json.dumps(raw))
    raw2["autoprompt"]["reuse_vocabulary"] = False
    vlm6 = _VLM()
    assert _run(tmp_path, raw2, monkeypatch, vlm6, service=service).reused is False and vlm6.calls == 3


def test_a_failed_merge_call_fails_the_stage(tmp_path, monkeypatch):
    raw = _session(tmp_path)
    raw["autoprompt"]["merge_synonyms"] = True
    from segmentation.autoprompt.consolidate_prompts import MergeError

    class _Boom(_VLM):
        def chat(self, messages, max_tokens=None, consumer=None):
            if consumer == "phase1.merge_synonyms":
                raise ConnectionError("engine gone")
            return super().chat(messages, max_tokens, consumer)

    with pytest.raises(MergeError, match="engine gone"):
        _run(tmp_path, raw, monkeypatch, _Boom())
    assert not (tmp_path / "output" / "vlm_analysis.json").exists(), "nothing half-written"


def test_the_bound_records_the_margin_at_the_cut(tmp_path, monkeypatch):
    raw = _session(tmp_path)
    raw["autoprompt"]["max_sam3_prompts"] = 1

    class _Three(_VLM):
        def chat(self, messages, max_tokens=None, consumer=None):
            self.calls += 1
            return SimpleNamespace(content=json.dumps({
                "scene_type": "room", "summary": "s",
                "objects": [{"category": "desk"}, {"category": "floor"}]
                + ([{"category": "chair"}] if self.calls == 1 else [])}),
                finish_reason="stop", usage={})

    res = _run(tmp_path, raw, monkeypatch, _Three())
    doc = json.loads(Path(res.vlm_analysis_path).read_text())
    pb = doc["census"]["decisions"]["prompt_bound"]
    assert pb["bound_reached"] and pb["not_prompted"] == ["floor", "chair"]
    assert pb["last_admitted"] == {"name": "desk", "n_proposals": 3}
    assert pb["first_excluded"] == {"name": "floor", "n_proposals": 3} and pb["margin_proposals"] == 0, \
        "a tie at the cut (decided by first-seen order) is on record as margin 0"


# ── autosegment (86 / 159) ───────────────────────────────────────────────────────────

def test_an_edited_sam3_list_has_one_source_and_no_clock(tmp_path):
    from segmentation.autoprompt import autosegment as AS
    out = tmp_path / "output"
    out.mkdir()
    (out / AS.CONCEPTS_FILE).write_text(json.dumps({"prompts": ["desk", "floor"], "stamp": "x"}))
    (out / AS.VLM_ANALYSIS_FILE).write_text(json.dumps({"prompt": "desk;floor", "stamp": {"sha256": "x"}}))
    got = AS.set_sam3_prompts(out, ["desk", "door"])
    assert got == ["desk", "door"]
    vlm = json.loads((out / AS.VLM_ANALYSIS_FILE).read_text())
    rec = json.loads((out / AS.CONCEPTS_FILE).read_text())
    assert vlm["prompt"] == "desk;door" and rec["prompts"] == ["desk", "door"]
    assert vlm["prompt_source"] == rec["prompt_source"] == "human_validated"
    assert vlm["prompt_sha256"] == rec["prompt_sha256"] == hashlib.sha256(b"desk;door").hexdigest()
    assert "prompt_edited_at" not in vlm and "edited_at" not in json.dumps(rec)
    assert json.loads((out / AS.AUTOSEGMENT_TIMING_FILE).read_text())["sam3_prompts_edited_at"]
    assert vlm["stamp"] == {"sha256": "x"}, "the rest of the analysis stays"
