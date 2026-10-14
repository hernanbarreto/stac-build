"""Wave 2, package P2 — the close (docs/plan_determinismo.md points 94, 112 / 151, 150, 159
and the VLM stage's reuse without a launch, 2026-10-08): the client's image encoder in every
stamp an image enters and in every per-call record; the SAM3 stage on the JOB's frozen
configuration; the product epoch made live BEFORE SAM3 and ONE projection; a census that
carries the sha256 of its inputs and nothing that varies between identical runs; the VLM
worker reusing the analysis without launching, hashing or stopping a vLLM when the engine's
identity — computed without a launch — closes the stamp. CPU only, no vLLM, no SAM3."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

CONFIG = yaml.safe_load((SERVER / "config.yaml").read_text())


class _Pipe:
    def __init__(self):
        self.logs = []

    def send_log(self, msg, level="info"):
        self.logs.append(str(msg))

    def send_progress(self, *a, **k):
        pass

    def check_cancel(self):
        return False


# ── point 94: the client's image encoder ──────────────────────────────────────────────

def test_the_client_encoder_record_is_the_intakes_pillow_record_and_is_strict(monkeypatch):
    from intake import stamps
    from segmentation.object_captioner import client_image_encoding_record
    from segmentation.autoprompt.session_builder import image_encoder_summary, image_encoding_record
    rec = client_image_encoding_record()
    assert rec["format"] == "JPEG" and rec["quality"] == 90, "what semantic.types._encode_image does"
    assert rec["pillow"] == stamps.pillow_record(), "ONE record of the library, the intake's"
    assert rec["pillow"]["version"] and rec["pillow"]["jpg_codec"]
    assert image_encoding_record() == rec, "the VLM stage reads the same definition"
    s = image_encoder_summary(rec)
    assert s == {"format": "JPEG", "quality": 90, "pillow": rec["pillow"]["version"],
                 "libjpeg_turbo": rec["pillow"]["libjpeg_turbo"]}
    # a Pillow that cannot be identified fails the stamp — never a blank in it
    monkeypatch.setattr(stamps, "pillow_record",
                        lambda: (_ for _ in ()).throw(stamps.IntakeStampError("no JPEG codec")))
    with pytest.raises(stamps.IntakeStampError, match="no JPEG codec"):
        client_image_encoding_record()


def test_the_clients_pillow_is_pinned_where_the_da3_env_is_pinned():
    """The client (the VLM worker, env da3) re-encodes every image with Pillow's libjpeg-turbo:
    the version installed here is the one environment.yml pins (point 94)."""
    import PIL
    text = (SERVER.parent / "environment.yml").read_text()
    m = re.search(r"pillow==([0-9][0-9A-Za-z.]*)", text)
    assert m, "environment.yml pins no pillow version"
    assert m.group(1) == PIL.__version__, "the pin is not the da3 env's Pillow"


def test_the_caption_stamp_carries_the_encoder(monkeypatch):
    from segmentation import object_captioner as oc
    from semantic.types import _encode_image
    refs = [_encode_image(Image.new("RGB", (8, 8), (1, 2, 3)))]
    enc_a = {"format": "JPEG", "quality": 90, "pillow": {"version": "12.0.0", "jpg_codec": "6.2",
                                                         "libjpeg_turbo": "3.1.2"}}
    enc_b = json.loads(json.dumps(enc_a))
    enc_b["pillow"]["libjpeg_turbo"] = "3.0.3"
    sa = oc.caption_stamp(refs, "p", {"sha256": "e"}, {"max_tokens": 256}, enc_a)
    sb = oc.caption_stamp(refs, "p", {"sha256": "e"}, {"max_tokens": 256}, enc_b)
    assert sa != sb, "another libjpeg-turbo build is another stamp: the caption is made again"
    assert sa == oc.caption_stamp(refs, "p", {"sha256": "e"}, {"max_tokens": 256}, dict(enc_a))
    # without the record handed in, the stamp reads the client's own
    monkeypatch.setattr(oc, "client_image_encoding_record", lambda: enc_a)
    assert oc.caption_stamp(refs, "p", {"sha256": "e"}, {"max_tokens": 256}) == sa


class _VLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=None, consumer=None):
        self.calls += 1
        return SimpleNamespace(content=json.dumps({
            "scene_type": "server room", "summary": "s",
            "objects": [{"category": "desk", "description": "black metal desk"},
                        {"category": "floor", "description": "grey tiles"}]}),
            finish_reason="stop", usage={"completion_tokens": 50, "prompt_tokens": 800})


def _vlm_session(tmp_path, n=4):
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    files = []
    for i in range(n):
        fn = f"{i * 7:06d}.jpg"
        Image.new("RGB", (64, 36), (i, i, i)).save(frames / fn)
        files.append(fn)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    raw = json.loads(json.dumps(CONFIG))
    raw["autoprompt"]["consolidate_prompts"] = False
    raw["autoprompt"]["merge_synonyms"] = False
    raw["autoprompt"]["vlm_sampling"].update(all_keyframes=False, spacing_kf=3, tile_rows=1,
                                             tile_cols=1, max_calls=300)
    raw["reconstruction"]["simple"]["enabled"] = True
    return raw


def _run_autoprompter(tmp_path, raw, monkeypatch, vlm, service=None):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: vlm)
    return AutoPrompter(tmp_path, tmp_path / "output", config=raw).run(service=service)


def test_the_vlm_stamp_and_every_call_record_carry_the_encoder(tmp_path, monkeypatch):
    from segmentation.autoprompt import session_builder as sb
    raw = _vlm_session(tmp_path)
    enc = {"format": "JPEG", "quality": 90, "pillow": {"version": "12.0.0", "jpg_codec": "6.2",
                                                       "libjpeg_turbo": "3.1.2"}}
    monkeypatch.setattr(sb, "image_encoding_record", lambda: dict(enc))
    service = {"sha256": "engine-a"}
    res = _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service=service)
    doc = json.loads(Path(res.vlm_analysis_path).read_text())
    assert "image_encoding" in doc["stamp"]["config"], "the encoder is a stamp section"
    import repro
    assert doc["stamp"]["config"]["image_encoding"] == repro.sha256_json(enc)
    assert doc["census"]["image_encoding"] == enc
    for c in doc["census"]["calls"]:
        assert c["image_encoding"] == {"format": "JPEG", "quality": 90, "pillow": "12.0.0",
                                       "libjpeg_turbo": "3.1.2"}, "beside the image's sha1"
        assert len(c["image_sha1"][0]) == 40
    # the same encoder: reused whole; another libjpeg-turbo build: derived again, declared
    assert _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service=service).reused is True
    other = json.loads(json.dumps(enc))
    other["pillow"]["libjpeg_turbo"] = "3.0.3"
    monkeypatch.setattr(sb, "image_encoding_record", lambda: dict(other))
    vlm = _VLM()
    assert _run_autoprompter(tmp_path, raw, monkeypatch, vlm, service=service).reused is False
    assert vlm.calls > 0


# ── points 112 / 151: the SAM3 wrapper on the job's configuration ─────────────────────

def test_the_wrapper_reads_the_jobs_block_and_keeps_the_no_argument_call(monkeypatch, tmp_path):
    import torch
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(w, "_sam3_wrapper", None)
    job = json.loads(json.dumps(CONFIG))
    job["models"]["segmentation"]["keyframe_interval"] = 1234
    assert w.segmentation_config(job)["keyframe_interval"] == 1234
    assert w.segmentation_config()["keyframe_interval"] == \
        CONFIG["models"]["segmentation"]["keyframe_interval"], "None: the server's"
    # the thresholds check of the job's block, not the server's
    bad = json.loads(json.dumps(CONFIG))
    bad["models"]["segmentation"]["sam3_thresholds"] = {}
    with pytest.raises(w.SAM3ConfigError, match="sam3_thresholds"):
        w.check_sam3_thresholds(config=bad)
    job["models"]["segmentation"]["sam3_thresholds"][
        CONFIG["models"]["segmentation"]["version"]]["score_threshold_detection"] = 0.33
    assert w.check_sam3_thresholds(config=job)["score_threshold_detection"] == 0.33, \
        "validated from the JOB's block, not the server's"
    # the singleton: built with the job's configuration, the no-argument call returns it
    a = w.get_sam3_wrapper(config=job)
    assert a._config is job and a.config_source.startswith("job")
    assert w.get_sam3_wrapper() is a and w.get_sam3_wrapper(config=job) is a
    # another block while nothing is loaded: rebuilt; once loaded: refused
    job2 = json.loads(json.dumps(job))
    job2["models"]["segmentation"]["keyframe_interval"] = 99
    b = w.get_sam3_wrapper(config=job2)
    assert b is not a and b._config is job2
    b.is_loaded = True
    with pytest.raises(w.SAM3ConfigError, match="never runs on two configurations"):
        w.get_sam3_wrapper(config=job)
    assert w.get_sam3_wrapper(config=job2) is b
    # a server-config wrapper
    monkeypatch.setattr(w, "_sam3_wrapper", None)
    s = w.get_sam3_wrapper()
    assert s._config is None and s.config_source == "server config.yaml"


def test_load_model_builds_from_the_jobs_thresholds_not_the_servers(monkeypatch, tmp_path):
    import torch
    import segmentation.sam3_wrapper as w
    built = {}
    pkg = types.ModuleType("sam3")
    pkg.__path__ = []
    mb = types.ModuleType("sam3.model_builder")

    def _multiplex(**kw):
        built["kwargs"] = kw
        built["predictor"] = SimpleNamespace(model=SimpleNamespace(
            score_threshold_detection=0.4, new_det_thresh=0.65, hotstart_delay=15,
            hotstart_unmatch_thresh=8, hotstart_dup_thresh=8,
            masklet_confirmation_consecutive_det_thresh=3))
        return built["predictor"]

    mb.build_sam3_multiplex_video_predictor = _multiplex
    monkeypatch.setitem(sys.modules, "sam3", pkg)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", mb)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ckpt = tmp_path / "sam3.1_multiplex.pt"
    ckpt.write_bytes(b"weights")
    job = json.loads(json.dumps(CONFIG))
    scfg = job["models"]["segmentation"]
    scfg["version"] = "sam3.1"
    scfg["checkpoint_path"] = str(ckpt)
    scfg["sam3_thresholds"]["sam3.1"]["score_threshold_detection"] = 0.31      # not the server's
    # the server's config (w.cfg) is left as it is: it must not be what the model gets
    wrapper = w.SAM3Wrapper(config=job)
    wrapper.load_model()
    assert built["predictor"].model.score_threshold_detection == 0.31
    assert wrapper.applied_thresholds["applied"]["score_threshold_detection"] == 0.31
    assert wrapper.model_record["config_source"].startswith("job")


# ── point 150 / 159: the SAM3 worker — product epoch first, one projection, the census ──

def _worker_session(tmp_path, prompts, with_cloud=True):
    (tmp_path / "frames").mkdir(parents=True)
    out = tmp_path / "output"
    out.mkdir()
    (out / "vlm_analysis.json").write_text(json.dumps(
        {"prompt": ";".join(prompts), "frame_map": {}, "prompt_source": "vlm_proposed",
         "understand_prompt": {"sha256": "p" * 64, "source": "shipped"},
         "stamp": {"sha256": "s" * 64}, "service": {"sha256": "e" * 64},
         "census": {"concepts": [], "calls": [], "sampling": None}}))
    if with_cloud:
        (out / "cleaned_cloud.ply").write_bytes(b"ply")
    cfg = json.loads(json.dumps(CONFIG))
    cfg["reconstruction"]["simple"]["exclusive_gpu"] = False
    return out, cfg


def test_the_worker_selects_the_product_epoch_before_sam3_and_projects_once(tmp_path, monkeypatch):
    import correction.chain as chain
    import segmentation_pipeline
    from workers import sam3_worker
    out, cfg = _worker_session(tmp_path, ["desk"])
    events = []
    monkeypatch.setattr(chain, "select_product_epoch",
                        lambda output_dir, log=print: (events.append("select"),
                                                       {"base_epoch": 0, "live_before": 3,
                                                        "selected": True})[1])

    def run(frames_dir, output_dir, prompt, frame_map, boxes_map, on_progress, prompt_status,
            fallback_prompts=None, config=None, defer_cloud_mapping=False):
        events.append(("sam3", config is cfg, defer_cloud_mapping))
        prompt_status["desk"] = {"status": "ran", "n_objects": 1}
        return {"instances": [{"id": 0}]}

    monkeypatch.setattr(segmentation_pipeline, "run_segmentation", run)
    monkeypatch.setattr(segmentation_pipeline, "map_segmentation_to_cloud",
                        lambda output_dir: (events.append("project"),
                                            {"instances": [{"id": 0}], "coverage": 0.5})[1])
    pipe = _Pipe()
    sam3_worker._sam3_work(pipe, str(tmp_path), cfg)
    assert events == ["select", ("sam3", True, True), "project"], \
        "the product epoch first, SAM3 on the job's dict with the in-run projection deferred, " \
        "then the ONE projection"
    assert any("product epoch 0" in m and "made live again" in m for m in pipe.logs)
    # the census carries the identity of its inputs (point 159)
    doc = json.loads((out / "segmentation_census.json").read_text())
    assert doc["version"] == 2
    assert doc["inputs"] == {"vlm_prompt_sha256": "p" * 64, "vlm_prompt_source": "shipped",
                             "vlm_analysis_stamp_sha256": "s" * 64, "service_sha256": "e" * 64,
                             "sam3_prompts_sha256": hashlib.sha256(b"desk").hexdigest(),
                             "sam3_prompts_source": "vlm_proposed", "n_sam3_prompts": 1}
    # no cloud: nothing to select, nothing to project
    events.clear()
    out2, cfg2 = _worker_session(tmp_path / "two", ["desk"], with_cloud=False)
    sam3_worker._sam3_work(_Pipe(), str(tmp_path / "two"), cfg2)
    assert events == [("sam3", False, True)]


def test_a_product_epoch_no_longer_on_disk_fails_before_sam3_runs(tmp_path, monkeypatch):
    import correction.chain as chain
    import segmentation_pipeline
    from workers import sam3_worker
    out, cfg = _worker_session(tmp_path, ["desk"])
    monkeypatch.setattr(chain, "select_product_epoch",
                        lambda output_dir, log=print: (_ for _ in ()).throw(chain.ChainError(
                            "the reconstruction's product epoch 1 cannot be made live")))
    ran = []
    monkeypatch.setattr(segmentation_pipeline, "run_segmentation",
                        lambda *a, **k: ran.append(1) or {"instances": []})
    with pytest.raises(chain.ChainError, match="cannot be made live"):
        sam3_worker._sam3_work(_Pipe(), str(tmp_path), cfg)
    assert not ran, "SAM3 never ran"
    assert not (out / "segmentation_census.json").exists()


def test_the_census_bytes_do_not_follow_the_moment(tmp_path):
    """Two identical runs give byte-identical segmentation_census.json (point 159): the
    exclusive-GPU check's memory reading of the moment and the per-prompt temporary
    directory names are not in it."""
    from segmentation import census
    from segmentation.autoprompt import autosegment as AS

    def _one(root, used_mib, tmp_suffix):
        out = root / "output"
        out.mkdir(parents=True)
        (root / "frames").mkdir()
        status = {"desk": {"status": "ran", "n_objects": 0, "seconds": 12.5 + used_mib},
                  "door": {"status": "failed",
                           "reason": f"SAM3RunError: propagate_in_video failed for prompt 'door' "
                                     f"in /tmp/sam3_batch_{tmp_suffix} (frames 0-9)"}}
        rec = {"version": "sam3.1", "applied": {},
               "model": {"version": "sam3.1", "checkpoint": {"sha256": "c" * 64}},
               "exclusive_gpu": {"cards": [{"uuid": "GPU-1", "name": "A100", "used_mib": used_mib,
                                            "total_mib": 81920, "driver_version": "595.91.07"}],
                                 "compute_processes": 0}}
        census.build_census(out, prompt="desk;door", vlm_doc={"prompt": "desk;door"}, gap_kf=1,
                            frames_dir=root / "frames",
                            seg_error=f"the run stopped in /tmp/sam3_batch_{tmp_suffix}",
                            prompt_status=status, sam3_thresholds=rec, log=lambda m: None)
        return (out / census.CENSUS_NAME).read_bytes()

    a = _one(tmp_path / "a", 0, "k3j9x2q1")
    b = _one(tmp_path / "b", 4, "p7m2z8w5")
    assert a == b, "identical runs, identical bytes"
    doc = json.loads(a)
    assert "used_mib" not in json.dumps(doc) and "sam3_batch_k3j9" not in json.dumps(doc)
    assert doc["sam3"]["thresholds"]["exclusive_gpu"]["cards"][0] == {
        "uuid": "GPU-1", "name": "A100", "total_mib": 81920, "driver_version": "595.91.07"}
    rows = {r["prompt"]: r for r in doc["prompts"]}
    assert rows["door"]["sam3_status"]["reason"] == \
        "SAM3RunError: propagate_in_video failed for prompt 'door' in <sam3_batch_dir> (frames 0-9)"
    assert "seconds" not in rows["desk"]["sam3_status"]
    assert doc["sam3"]["error"] == "the run stopped in <sam3_batch_dir>"
    # the SAM3 list's sha is the one an edited list carries (autosegment.set_sam3_prompts)
    out = tmp_path / "c" / "output"
    out.mkdir(parents=True)
    AS.set_sam3_prompts(out, ["desk", "door"])
    edited = json.loads((out / AS.VLM_ANALYSIS_FILE).read_text())
    assert doc["inputs"]["sam3_prompts_sha256"] == edited["prompt_sha256"]
    assert census.census_inputs("desk;door", edited)["sam3_prompts_source"] == "human_validated"
    assert census.census_inputs("desk", None)["vlm_analysis_stamp_sha256"] is None


# ── the VLM worker: reuse without a launch ────────────────────────────────────────────

def _frozen_session(tmp_path):
    from intake.run_config import freeze_run_config
    raw = _vlm_session(tmp_path)
    freeze_run_config(tmp_path, raw, log=lambda m: None)
    return raw


def test_the_vlm_worker_reuses_the_analysis_without_launching_an_engine(tmp_path, monkeypatch):
    import semantic.serve as serve
    import semantic.service as svc
    from workers import vlm_worker
    raw = _frozen_session(tmp_path)
    ident_a = {"sha256": "engine-a", "served_model_name": "qwen_local", "card": {"key": "A100"}}
    # an analysis made by engine A
    _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service=ident_a)
    asked = []
    monkeypatch.setattr(serve, "expected_identity",
                        lambda backend, config_path=None, log=print, **kw: (
                            asked.append((backend, str(config_path))), dict(ident_a))[1])
    launched = []

    @contextlib.contextmanager
    def job_engine(config, **kw):
        launched.append(kw)
        yield {"sha256": "engine-a"}

    monkeypatch.setattr(svc, "job_engine", job_engine)
    pipe = _Pipe()
    cm = vlm_worker._ensure_semantic_service(pipe, dict(raw, _session_dir=str(tmp_path)))
    with cm as identity:
        assert identity == ident_a
    assert asked == [("qwen_local", str(tmp_path / "output" / "run_config.yaml"))], \
        "the identity computed from the FROZEN configuration, once"
    assert not launched, "no vLLM launched"
    assert any("DECLARED" in m and "NOT launched" in m for m in pipe.logs)
    # and the whole stage, end to end: the analysis is the answer, no engine, no VLM call
    import semantic.client as sc
    monkeypatch.setattr(sc, "get_semantic_client",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("a VLM call")))
    vlm_worker._vlm_work(_Pipe(), str(tmp_path), raw)
    assert not launched

    # another engine would answer: the stamp differs → launched, and its live identity must be
    # the expected one
    ident_b = dict(ident_a, sha256="engine-b")
    monkeypatch.setattr(serve, "expected_identity",
                        lambda backend, config_path=None, log=print, **kw: dict(ident_b))
    pipe = _Pipe()
    cm = vlm_worker._ensure_semantic_service(pipe, dict(raw, _session_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="not the one this job describes"):
        with cm:
            pass
    assert len(launched) == 1 and launched[0]["run_config_path"] == \
        tmp_path / "output" / "run_config.yaml"

    @contextlib.contextmanager
    def job_engine_b(config, **kw):
        launched.append(kw)
        yield dict(ident_b)

    monkeypatch.setattr(svc, "job_engine", job_engine_b)
    cm = vlm_worker._ensure_semantic_service(_Pipe(), dict(raw, _session_dir=str(tmp_path)))
    with cm as identity:
        assert identity["sha256"] == "engine-b"
    assert len(launched) == 2


def test_the_identity_is_not_computed_when_the_analysis_differs_beyond_the_engine(tmp_path, monkeypatch):
    """The weights' sha256 costs minutes: it is paid only when the engine's identity is the
    last thing standing between the session and a reuse — a session without an analysis, or
    with one made from other keyframes / another prompt, launches (the launch verifies the
    weights once, as always)."""
    import semantic.serve as serve
    import semantic.service as svc
    from workers import vlm_worker
    raw = _frozen_session(tmp_path)
    monkeypatch.setattr(serve, "expected_identity",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("weights hashed")))
    launched = []

    @contextlib.contextmanager
    def job_engine(config, **kw):
        launched.append(kw)
        yield {"sha256": "engine-a"}

    monkeypatch.setattr(svc, "job_engine", job_engine)
    pipe = _Pipe()
    with vlm_worker._ensure_semantic_service(pipe, dict(raw, _session_dir=str(tmp_path))) as ident:
        assert ident["sha256"] == "engine-a"
    assert len(launched) == 1 and any("derived again" in m for m in pipe.logs)
    # an analysis of other keyframes: the same
    _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service={"sha256": "engine-a"})
    Image.new("RGB", (64, 36), (9, 9, 9)).save(tmp_path / "frames" / "000000.jpg")
    with vlm_worker._ensure_semantic_service(_Pipe(), dict(raw, _session_dir=str(tmp_path))):
        pass
    assert len(launched) == 2
    # the AutoPrompter's own answer on what is pending
    from segmentation.autoprompt.session_builder import AutoPrompter
    ap = AutoPrompter(tmp_path, tmp_path / "output", config=raw)
    pending, why = ap.reuse_awaits_engine_identity()
    assert pending is False and "beyond the engine" in why and "frames/000000.jpg" in why
    Image.new("RGB", (64, 36), (0, 0, 0)).save(tmp_path / "frames" / "000000.jpg")
    pending, why = AutoPrompter(tmp_path, tmp_path / "output", config=raw).reuse_awaits_engine_identity()
    assert pending is True
    doc, why = AutoPrompter(tmp_path, tmp_path / "output", config=raw).analysis_reusable({"sha256": "engine-a"})
    assert doc is not None and why == "stamp identical"
    assert AutoPrompter(tmp_path, tmp_path / "output", config=raw).analysis_reusable({"sha256": "x"})[0] is None


def test_expected_identity_runs_the_launcher_in_identity_only_mode(monkeypatch, tmp_path):
    import semantic.serve as serve
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n")
    calls = []
    doc = {"identity": {"sha256": "abc", "served_model_name": "qwen_local", "card": {"key": "A"}},
           "process": {"pid": 1}}

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="[semantic.serve] weights of 'm' @ r verified (14 file(s))\n"
                                      + serve.IDENTITY_ONLY_MARK + json.dumps(doc) + "\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    said = []
    ident = serve.expected_identity("qwen_local", tmp_path / "run_config.yaml", log=said.append,
                                    python=py)
    assert ident == doc["identity"]
    argv, kw = calls[0]
    assert argv == [str(py), "-m", "semantic.serve", "--backend", "qwen_local", "--identity-only",
                    "--config", str(tmp_path / "run_config.yaml")]
    assert kw["env"]["VLLM_BATCH_INVARIANT"] == "1" and kw["env"]["HF_HUB_OFFLINE"] == "1"
    assert any("weights of 'm'" in m for m in said) and any("no vLLM launched" in m for m in said)
    # the launcher's refusal is the error, with its words
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="ServeError: not the weights every validated run served"))
    with pytest.raises(serve.ServeError, match="not the weights"):
        serve.expected_identity("qwen_local", None, log=said.append, python=py)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=0, stdout="nothing\n", stderr=""))
    with pytest.raises(serve.ServeError, match="no identity line"):
        serve.expected_identity("qwen_local", None, log=said.append, python=py)
    # the semantic env's python is found under CONDA_ROOT, or the call fails naming it
    monkeypatch.setenv("CONDA_ROOT", str(tmp_path / "nowhere"))
    with pytest.raises(serve.ServeError, match="semantic env's python"):
        serve.semantic_env_python()


def test_identity_only_prints_the_identity_and_writes_nothing(monkeypatch, capsys, tmp_path):
    import semantic.serve as serve
    from semantic.semantic_config import load_semantic_config
    cfg = load_semantic_config()
    monkeypatch.setattr(serve, "semantic_config_from", lambda p: cfg)
    monkeypatch.setattr(serve, "weights_identity",
                        lambda b, log=print: {"model_id": b["model_id"], "revision": "r", "path": "w",
                                              "sha256": {}})
    monkeypatch.setattr(serve, "service_environment", lambda base=None: {"HF_HOME": "/x"})
    monkeypatch.setattr(serve.repro, "card_identity",
                        lambda i=0: {"key": "A | 1 MiB | sm_8.0", "name": "A", "capability": "8.0",
                                     "memory_total_mib": 1, "uuid": "GPU-x"})
    monkeypatch.setattr(serve, "write_identity",
                        lambda doc, path=None: (_ for _ in ()).throw(AssertionError("written")))
    monkeypatch.setattr(serve.os, "execvpe",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("exec'd")))
    monkeypatch.setattr(sys, "argv", ["semantic.serve", "--backend", "qwen_local", "--identity-only"])
    serve.main()
    out = capsys.readouterr().out.splitlines()
    marked = [ln for ln in out if ln.startswith(serve.IDENTITY_ONLY_MARK)]
    assert len(marked) == 1
    doc = json.loads(marked[0][len(serve.IDENTITY_ONLY_MARK):])
    ident = doc["identity"]
    assert ident["served_model_name"] == "qwen_local" and ident["sha256"]
    assert ident["batch_invariance"]["form"] == "batch_invariant_matmul"
    # the same record a launch writes: its sha256 is of its own parts
    body = dict(ident)
    body.pop("sha256")
    assert ident["sha256"] == serve.repro.sha256_json(body)


# ── the skeptics' corrections (close of wave 2, 2026-10-08 night) ─────────────────────────

def _loop_store(monkeypatch):
    meta = {}

    class _Store:
        def __init__(self, p): pass
        def get_meta(self, k): return meta.get(k)
        def set_meta(self, k, v): meta[k] = v

    monkeypatch.setattr("phase_r.instance_store.InstanceStore", _Store)
    return meta


def test_loop_classes_take_the_frozen_configuration_and_launch_only_for_an_uncached_instance(
        tmp_path, monkeypatch):
    """Point 95: with ``config`` None on a session that holds a frozen run_config.yaml the
    engine is launched from THAT file and verified against ITS block (one configuration —
    never the frozen file for the launch and the server's config.yaml for the check); the
    identity is computed without a launch, every verdict cached launches nothing, an
    uncached instance launches, and an engine that comes up as another identity FAILS."""
    import semantic.serve as serve
    import semantic.service as svc
    from reconstruction.loops import semantic_classes as sc
    from intake.run_config import freeze_run_config
    raw = json.loads(json.dumps(CONFIG))
    raw["semantic"]["service"]["startup_timeout_s"] = 4321          # not the server's
    freeze_run_config(tmp_path, raw, log=lambda m: None)
    out = tmp_path / "output"
    meta = _loop_store(monkeypatch)
    crop = np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(sc, "_mask_crop", lambda *a, **k: [crop])
    cfg = SimpleNamespace(enabled=True, default_class="structural", crops_per_instance=1,
                          max_tokens=64)
    expected = {"sha256": "engine-a"}
    asked, launched = [], []
    monkeypatch.setattr(serve, "expected_identity",
                        lambda backend, config_path=None, log=print, **kw: (
                            asked.append(config_path), dict(expected))[1])
    live = {"sha256": "engine-a"}

    @contextlib.contextmanager
    def job_engine(config, **kw):
        launched.append((config, kw))
        yield dict(live)

    monkeypatch.setattr(svc, "job_engine", job_engine)
    import semantic.client as scl
    monkeypatch.setattr(scl, "get_semantic_client", lambda **kw: SimpleNamespace())
    import segmentation.shape_proposer as sp
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: ({"class": "movable", "confidence": 0.9}, "{}"))
    insts = [{"instance_id": 1, "label": "desk"}]
    said = []
    res = sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]}, log=said.append)
    assert res[1]["class"] == "movable" and len(meta) == 1
    assert asked == [tmp_path / "output" / "run_config.yaml"], \
        "the identity of the engine the FROZEN file would launch"
    (config, kw), = launched
    assert config["semantic"]["service"]["startup_timeout_s"] == 4321 and \
        config["semantic"] == raw["semantic"], "launched from the frozen configuration's own block"
    assert kw["run_config_path"] == tmp_path / "output" / "run_config.yaml"
    assert any("frozen output/run_config.yaml" in m for m in said)
    # every verdict cached: the identity is computed, no engine comes up
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked")))
    said.clear()
    res = sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]}, log=said.append)
    assert res[1]["class"] == "movable" and len(launched) == 1 and len(asked) == 2
    assert any("DECLARED" in m and "no vLLM is launched" in m for m in said)
    # an uncached instance: the engine comes up — as another identity than the one the keys
    # were built with: a failure, nothing asked, nothing recorded
    live["sha256"] = "engine-z"
    with pytest.raises(sc.LoopClassError, match="not the one this job describes"):
        sc.classify_instances(out, tmp_path, [{"instance_id": 2, "label": "door"}], cfg, {2: 0},
                              {2: [0]}, log=said.append)
    assert len(launched) == 2 and len(meta) == 1
    # an identity that cannot be computed fails naming the launcher's words
    monkeypatch.setattr(serve, "expected_identity",
                        lambda *a, **k: (_ for _ in ()).throw(serve.ServeError("not the weights")))
    with pytest.raises(sc.LoopClassError, match="not the weights"):
        sc.classify_instances(out, tmp_path, [{"instance_id": 2, "label": "door"}], cfg, {2: 0},
                              {2: [0]}, log=said.append)
    # a session nobody froze: the server's configuration, said
    (tmp_path / "output" / "run_config.yaml").unlink()
    monkeypatch.setattr(serve, "expected_identity",
                        lambda backend, config_path=None, log=print, **kw: (
                            asked.append(config_path), dict(expected))[1])
    said.clear()
    sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]}, log=said.append)
    assert asked[-1] is None and any("server's config.yaml" in m for m in said)


def test_reuse_awaits_engine_identity_is_a_structural_test_of_the_stamp(tmp_path, monkeypatch):
    """The decision is taken on repro.check_stamp's list over the saved stamp and the one this
    job computes with the saved engine section in place — never by parsing the wording of
    the reuse log line (a wording change there would have turned every reuse into a launch)."""
    from segmentation.autoprompt.session_builder import AutoPrompter
    raw = _frozen_session(tmp_path)
    _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service={"sha256": "engine-a"})
    monkeypatch.setattr(AutoPrompter, "_reusable_analysis",
                        lambda self, stamp: (_ for _ in ()).throw(AssertionError("the log line")))
    ap = AutoPrompter(tmp_path, tmp_path / "output", config=raw)
    assert ap.reuse_awaits_engine_identity() == (True, "every stamped input but the engine's "
                                                        "identity matches")
    Image.new("RGB", (64, 36), (9, 9, 9)).save(tmp_path / "frames" / "000000.jpg")
    pending, why = AutoPrompter(tmp_path, tmp_path / "output", config=raw).reuse_awaits_engine_identity()
    assert pending is False and why.startswith("its stamp differs beyond the engine: ") \
        and "input 'frames/000000.jpg' changed" in why
    # an analysis made without any engine identity: the engine is still the one thing to verify
    Image.new("RGB", (64, 36), (0, 0, 0)).save(tmp_path / "frames" / "000000.jpg")
    monkeypatch.undo()
    _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service=None)
    assert AutoPrompter(tmp_path, tmp_path / "output", config=raw).reuse_awaits_engine_identity()[0] is True
    # no analysis at all: nothing awaits an identity
    (tmp_path / "output" / "vlm_analysis.json").unlink()
    pending, why = AutoPrompter(tmp_path, tmp_path / "output", config=raw).reuse_awaits_engine_identity()
    assert pending is False and why == "no vlm_analysis.json in the session"


def test_the_vlm_worker_says_nothing_is_auto_prompted_on_the_no_launch_path(tmp_path, monkeypatch):
    import semantic.serve as serve
    import semantic.service as svc
    from workers import vlm_worker
    raw = _frozen_session(tmp_path)
    ident = {"sha256": "engine-a"}
    _run_autoprompter(tmp_path, raw, monkeypatch, _VLM(), service=ident)
    monkeypatch.setattr(serve, "expected_identity",
                        lambda backend, config_path=None, log=print, **kw: dict(ident))
    monkeypatch.setattr(svc, "job_engine",
                        lambda config, **kw: (_ for _ in ()).throw(AssertionError("launched")))
    import semantic.client as sc
    monkeypatch.setattr(sc, "get_semantic_client",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("a VLM call")))
    pipe = _Pipe()
    vlm_worker._vlm_work(pipe, str(tmp_path), raw)
    assert not any("Starting Qwen3-VL auto-prompter" in m for m in pipe.logs)
    assert any("nothing is auto-prompted (no engine)" in m for m in pipe.logs)


def test_a_server_wrapper_adopts_the_jobs_equal_block(monkeypatch):
    """get_sam3_wrapper(config=job) on a wrapper first built by the no-argument call with the
    SAME models.segmentation block: the job's dict is adopted (its keys, the record's
    config_source), nothing the model was built with changes."""
    import torch
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(w, "_sam3_wrapper", None)
    s = w.get_sam3_wrapper()
    assert s._config is None and s.config_source == w.SAM3Wrapper.SERVER_CONFIG_SOURCE
    s.model_record = {"version": "sam3.1", "config_source": s.config_source}
    job = json.loads(json.dumps(CONFIG))
    assert w.get_sam3_wrapper(config=job) is s
    assert s._config is job and s.config_source == w.SAM3Wrapper.JOB_CONFIG_SOURCE
    assert s.model_record["config_source"] == w.SAM3Wrapper.JOB_CONFIG_SOURCE
    assert w.get_sam3_wrapper() is s and s._config is job, "a later no-argument call keeps it"


def test_the_census_scrubs_the_interactive_session_directory_too():
    from segmentation.census import without_temp_paths
    assert without_temp_paths("failed in /tmp/sam3_kf_a1b2c3/frames and /tmp/sam3_batch_x9/f") == \
        "failed in <sam3_kf_dir>/frames and <sam3_batch_dir>/f"
    assert without_temp_paths(None) is None
    assert without_temp_paths("no path here") == "no path here"


def test_the_caption_cli_reads_the_sessions_frozen_configuration(tmp_path, monkeypatch, capsys):
    from intake.run_config import freeze_run_config
    from segmentation import object_captioner as oc
    raw = json.loads(json.dumps(CONFIG))
    raw["segmentation"]["object_captions"]["views"] = 7                                  # not the server's
    freeze_run_config(tmp_path, raw, log=lambda m: None)
    got = {}

    def fake(output_dir, session_dir, *, views, refresh=False, log=print, config=None, **kw):
        got.update(views=views, config=config)
        return {"generated": 0}

    monkeypatch.setattr(oc, "caption_session_objects", fake)
    assert oc._main(["--session", str(tmp_path)]) == 0
    assert got["views"] == 7 and got["config"]["segmentation"]["object_captions"]["views"] == 7, \
        "the frozen job's configuration, its views"
    assert "frozen output/run_config.yaml" in capsys.readouterr().out
    assert oc._main(["--session", str(tmp_path), "--views", "2"]) == 0 and got["views"] == 2
    # a session nobody froze: the server's, said
    (tmp_path / "output" / "run_config.yaml").unlink()
    assert oc._main(["--session", str(tmp_path)]) == 0
    assert got["views"] == CONFIG["segmentation"]["object_captions"]["views"]
    assert "server's config.yaml" in capsys.readouterr().out
