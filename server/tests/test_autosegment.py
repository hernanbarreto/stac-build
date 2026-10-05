"""Autosegment (USER 2026-10-05): a per-scan "segment when done" check in "Reconstruir"
(off by default — the run ends at the published cloud) and the Instances panel's
Autosegment window: the VLM prompt and the SAM3 prompts editable and SAVED IN THE
SESSION, a checkbox per remaining stage, the chosen stages forced to run on the cloud
on disk."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_manager import (SEGMENTATION_CHAIN, PipelineStage, StageId,  # noqa: E402
                              build_pipeline_stages, select_stages)
from segmentation.autoprompt import autosegment as AS  # noqa: E402


def _full():
    return [PipelineStage(id=s, enabled=True) for s in
            (StageId.RECONSTRUCTION, StageId.CLOUDCOMPY, StageId.VLM, StageId.SAM3, StageId.CERTIFY)]


def test_the_chain_is_vlm_sam3_certify_and_the_check_off_ends_at_the_cloud():
    assert SEGMENTATION_CHAIN == (StageId.VLM, StageId.SAM3, StageId.CERTIFY)
    off = select_stages(_full(), segment=False)
    assert [s.id for s in off if s.enabled] == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY]
    on = select_stages(_full(), segment=True)
    assert [s.id for s in on if s.enabled] == [s.id for s in _full()]


def test_select_stages_never_mutates_the_shared_list():
    base = _full()
    select_stages(base, segment=False)
    select_stages(base, only={StageId.SAM3})
    assert all(s.enabled for s in base)


def test_only_keeps_exactly_the_chosen_stages_and_respects_a_disabled_one():
    base = _full()
    base[4] = PipelineStage(id=StageId.CERTIFY, enabled=False)       # config switched it off
    out = select_stages(base, only={StageId.VLM, StageId.CERTIFY})
    assert [s.id for s in out if s.enabled] == [StageId.VLM]
    assert [s.id for s in out] == [s.id for s in base]               # order kept


def test_production_stage_list_accepts_the_selection():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    stages = build_pipeline_stages(backend=str(raw["reconstruction"]["backend"]))
    off = [s.id for s in select_stages(stages, segment=False) if s.enabled]
    assert off == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY]
    auto = [s.id for s in select_stages(stages, only={StageId.SAM3, StageId.CERTIFY}) if s.enabled]
    assert auto == [StageId.SAM3, StageId.CERTIFY]


# ── the prompts ─────────────────────────────────────────────────────────

def test_vlm_prompt_default_then_saved_then_restored(tmp_path):
    out = tmp_path / "output"
    p, over = AS.vlm_prompt_for(out)
    assert p == AS.default_vlm_prompt() and not over
    assert AS.save_vlm_prompt(out, "Name every object you see. Return JSON.") is True
    p, over = AS.vlm_prompt_for(out)
    assert over and p.startswith("Name every object")
    doc = json.loads((out / AS.AUTOSEGMENT_FILE).read_text())
    assert doc["provenance"] == "human_validated" and doc["vlm_prompt_saved_at"]
    # the default itself, or nothing, removes the override
    assert AS.save_vlm_prompt(out, AS.default_vlm_prompt()) is False
    assert AS.vlm_prompt_for(out) == (AS.default_vlm_prompt(), False)
    AS.save_vlm_prompt(out, "x")
    assert AS.save_vlm_prompt(out, "   ") is False
    assert "vlm_prompt" not in json.loads((out / AS.AUTOSEGMENT_FILE).read_text())


def test_the_autoprompter_reads_the_sessions_prompt_and_understand_frame_uses_it(tmp_path, monkeypatch):
    import inspect
    from segmentation.autoprompt import scene_understanding as SU
    from segmentation.autoprompt import session_builder as SB
    assert "prompt" in inspect.signature(SU.understand_frame).parameters
    src = inspect.getsource(SB.AutoPrompter.__init__)
    assert "vlm_prompt_for(self.output_dir)" in src
    assert "prompt=self.understand_prompt" in inspect.getsource(SB)
    # understand_frame sends the session's prompt, the default otherwise
    sent = {}

    class _Resp:
        content = json.dumps({"scene_type": "room", "summary": "a room", "objects": []})

    class _Client:
        def chat(self, msgs, max_tokens=0, consumer=""):
            sent["prompt"] = msgs[1].content if hasattr(msgs[1], "content") else msgs[1]
            return _Resp()
    from PIL import Image
    img = Image.new("RGB", (8, 8))
    SU.understand_frame(_Client(), img, 0, prompt="MY PROMPT")
    assert "MY PROMPT" in str(sent["prompt"]) and SU._PROMPT[:30] not in str(sent["prompt"])
    SU.understand_frame(_Client(), img, 0, prompt=None)
    assert SU._PROMPT[:30] in str(sent["prompt"])


def test_sam3_prompts_read_and_written_where_the_sam3_stage_reads(tmp_path):
    out = tmp_path / "output"
    assert AS.sam3_prompts(out) == []
    got = AS.set_sam3_prompts(out, ["floor", " wall ", "floor", "", "door;"])
    assert got == ["floor", "wall", "door"]
    doc = json.loads((out / AS.VLM_ANALYSIS_FILE).read_text())
    assert doc["prompt"] == "floor;wall;door" and doc["frame_map"] == {}
    assert doc["prompt_source"] == "human_validated"
    # the rest of an existing analysis survives; unchanged prompts touch nothing
    doc["shape_descriptions"] = {"floor": {"form": "flat"}}
    (out / AS.VLM_ANALYSIS_FILE).write_text(json.dumps(doc))
    m0 = (out / AS.VLM_ANALYSIS_FILE).stat().st_mtime_ns
    assert AS.set_sam3_prompts(out, ["floor", "wall", "door"]) == ["floor", "wall", "door"]
    assert (out / AS.VLM_ANALYSIS_FILE).stat().st_mtime_ns == m0
    AS.set_sam3_prompts(out, ["floor"])
    doc = json.loads((out / AS.VLM_ANALYSIS_FILE).read_text())
    assert doc["prompt"] == "floor" and doc["shape_descriptions"] == {"floor": {"form": "flat"}}
    st = AS.state(out)
    assert st["sam3_prompts"] == ["floor"] and st["has_vlm_analysis"] and not st["has_cloud"]


def test_forced_stages_skip_no_resume_probe():
    import inspect
    import pipeline_manager as PM
    src = inspect.getsource(PM.PipelineManager._run_pipeline)
    assert "upstream_ran = replace or recon_requested or force" in src
    assert "force" in inspect.signature(PM.PipelineManager.start_pipeline).parameters


def test_the_websocket_command_carries_segment_and_autosegment():
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    assert 'cmd.get("segment")' in src and 'cmd.get("autosegment")' in src
    assert src.count("stages=_stages_for(") == 2 and src.count("force=_force,") == 2
    assert '@app.get("/api/autosegment/{session_id}")' in src
    assert '@app.post("/api/autosegment/{session_id}")' in src
