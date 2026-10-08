"""Wave 2, package P2 — the SAM3 stage and the cloud stage (docs/plan_determinismo.md points
90, 91, 92, 95, 96, 164, 165): no object cap, deterministic numerics after the build and the
model recorded, no partial prompt results, the cloud stage projects no stale masks, the
loop-class cache keyed by content. CPU only, the vendor stubbed."""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

CONFIG = yaml.safe_load((SERVER / "config.yaml").read_text())


# ── points 90 / 91 / 165: the model built ────────────────────────────────────────────

def test_production_has_no_object_cap_and_the_vendor_sizes_by_objects_present():
    assert CONFIG["models"]["segmentation"]["max_num_objects"] == -1
    base = SERVER.parent / "vendor" / "sam31" / "sam3" / "model"
    if not base.is_dir():
        pytest.skip("vendor/sam31 not on this checkout")
    tracking = (base / "sam3_multiplex_tracking.py").read_text()
    assert "batched_buffer_size = max(int(total_objects), 1)" in tracking
    assert "self.postprocess_batch_size * self.max_num_objects" not in tracking
    mbase = (base / "sam3_multiplex_base.py").read_text()
    assert 'getattr(self, "compile_model", False)' in mbase


@pytest.fixture
def stubbed_sam3(monkeypatch, tmp_path):
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
    scfg = dict(CONFIG["models"]["segmentation"], checkpoint_path=str(ckpt))
    monkeypatch.setattr(w, "cfg", {"models": {"segmentation": scfg}})
    return w, built, ckpt


def test_the_load_passes_no_cap_runs_deterministic_and_records_the_model(stubbed_sam3):
    import torch
    import repro
    w, built, ckpt = stubbed_sam3
    wrapper = w.SAM3Wrapper()
    wrapper.load_model()
    assert built["kwargs"]["max_num_objects"] == -1, "the vendor's no limit"
    rec = wrapper.model_record
    assert rec["version"] == "sam3.1" and rec["device"] == "cuda"
    assert rec["checkpoint"]["sha256"] == repro.sha256_bytes(b"weights")
    assert rec["numerics"]["deterministic_algorithms"] is True
    assert rec["numerics"]["cudnn_allow_tf32"] is False and rec["numerics"]["matmul_allow_tf32"] is False
    assert rec["numerics"]["seed"] == w.SAM3_DETERMINISTIC_SEED
    assert torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32
    assert rec["thresholds"]["version"] == "sam3.1"


def test_an_unknown_version_fails_naming_it(stubbed_sam3):
    w, built, ckpt = stubbed_sam3
    with pytest.raises(w.SAM3ConfigError, match="version"):
        w.sam3_build_version({"version": "sam4"})
    assert w.sam3_build_version({"version": "sam3"}) == "sam3", "no CUDA condition any more"


# ── point 92: never a partial result ─────────────────────────────────────────────────

class _Predictor:
    """add_prompt fine; the propagation yields one frame then dies."""

    def __init__(self, fail_with):
        self.fail_with = fail_with

    def handle_request(self, request):
        if request["type"] == "start_session":
            return {"session_id": "s"}
        if request["type"] == "add_prompt":
            return {"out_obj_ids": [1], "out_binary_masks": np.zeros((1, 2, 2), bool)}
        return {}

    def handle_stream_request(self, request):
        yield {"frame_index": 0, "outputs": {"out_binary_masks": np.zeros((1, 2, 2), bool),
                                             "out_obj_ids": np.array([1])}}
        raise self.fail_with


def test_a_propagation_error_fails_the_prompt_instead_of_returning_partial_frames(monkeypatch):
    import torch
    import segmentation.sam3_wrapper as w
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    wrapper = w.SAM3Wrapper()
    wrapper.is_loaded = True
    wrapper.predictor = _Predictor(RuntimeError("tracker exploded"))
    with pytest.raises(w.SAM3RunError, match="propagate_in_video failed for prompt 'desk'"):
        wrapper.process_batch("/tmp/batch", "desk", {0: 0, 1: 1})
    assert wrapper._batch_session is None, "the session is dropped, not reused"
    wrapper.predictor = _Predictor(RuntimeError("CUDA out of memory"))
    with pytest.raises(w.SAM3OutOfMemory, match="out of memory"):
        wrapper.process_batch("/tmp/batch", "desk", {0: 0, 1: 1})

    class _NoPrompt(_Predictor):
        def handle_request(self, request):
            if request["type"] == "add_prompt":
                raise ValueError("prompt refused")
            return super().handle_request(request)

    wrapper.predictor = _NoPrompt(None)
    with pytest.raises(w.SAM3RunError, match="add_prompt failed"):
        wrapper.process_batch("/tmp/batch", "desk", {0: 0})


class _Pipe:
    def __init__(self):
        self.logs = []

    def send_log(self, msg, level="info"):
        self.logs.append(msg)

    def send_progress(self, *a, **k):
        pass

    def check_cancel(self):
        return False


def _worker_session(tmp_path, prompts):
    (tmp_path / "frames").mkdir()
    out = tmp_path / "output"
    out.mkdir()
    (out / "vlm_analysis.json").write_text(json.dumps(
        {"prompt": ";".join(prompts), "frame_map": {},
         "census": {"concepts": [], "calls": [], "sampling": None}}))
    cfg = json.loads(json.dumps(CONFIG))
    cfg["reconstruction"]["simple"]["exclusive_gpu"] = False
    return out, cfg


def test_the_sam3_worker_fails_the_stage_on_any_prompt_not_run_to_the_end(tmp_path, monkeypatch):
    import segmentation_pipeline
    from workers import sam3_worker
    out, cfg = _worker_session(tmp_path, ["desk", "floor", "door"])

    def partial(frames_dir, output_dir, prompt, frame_map, boxes_map, on_progress,
                prompt_status, fallback_prompts=None):
        prompt_status["desk"] = {"status": "ran", "n_objects": 1}
        prompt_status["floor"] = {"status": "ran", "n_objects": 1,
                                  "note": "ran after one CUDA out-of-memory recovery"}
        prompt_status["door"] = {"status": "failed", "reason": "SAM3RunError: propagation died"}
        return {"instances": [{"id": 0}]}

    monkeypatch.setattr(segmentation_pipeline, "run_segmentation", partial)
    with pytest.raises(RuntimeError, match="no partial segmentation") as e:
        sam3_worker._sam3_work(_Pipe(), str(tmp_path), cfg)
    assert "'door': failed" in str(e.value) and "'floor': ran only after a recovery" in str(e.value)
    assert (out / "segmentation_census.json").exists(), "the census still describes the run"


def test_the_sam3_worker_fails_the_stage_when_the_projection_fails(tmp_path, monkeypatch):
    import segmentation_pipeline
    from workers import sam3_worker
    out, cfg = _worker_session(tmp_path, ["desk"])
    (out / "cleaned_cloud.ply").write_bytes(b"ply")

    def ok(frames_dir, output_dir, prompt, frame_map, boxes_map, on_progress,
           prompt_status, fallback_prompts=None):
        prompt_status["desk"] = {"status": "ran", "n_objects": 1}
        return {"instances": [{"id": 0}]}

    monkeypatch.setattr(segmentation_pipeline, "run_segmentation", ok)
    monkeypatch.setattr(segmentation_pipeline, "map_segmentation_to_cloud",
                        lambda output_dir: {"error": "classification overflow", "instances": []})
    with pytest.raises(RuntimeError, match="mask→cloud projection failed: classification overflow"):
        sam3_worker._sam3_work(_Pipe(), str(tmp_path), cfg)


def test_the_sam3_worker_crops_meshes_only_from_a_scene_mesh_stamped_for_this_reconstruction(
        tmp_path, monkeypatch):
    import segmentation_pipeline
    from workers import sam3_worker
    out, cfg = _worker_session(tmp_path, ["desk"])
    (out / "cleaned_cloud.ply").write_bytes(b"ply")

    def ok(frames_dir, output_dir, prompt, frame_map, boxes_map, on_progress,
           prompt_status, fallback_prompts=None):
        prompt_status["desk"] = {"status": "ran", "n_objects": 1}
        return {"instances": [{"id": 0}]}

    monkeypatch.setattr(segmentation_pipeline, "run_segmentation", ok)
    monkeypatch.setattr(segmentation_pipeline, "map_segmentation_to_cloud",
                        lambda output_dir: {"instances": [{"id": 0}], "coverage": 0.5})
    cropped = []
    import segmentation.tsdf_export as te
    monkeypatch.setattr(te, "crop_scene_mesh_to_instances", lambda **kw: cropped.append(1) or [])
    scene = out / "tsdf" / "scene"
    scene.mkdir(parents=True)
    (scene / "scene.glb").write_bytes(b"glb")
    (out / "segmentation_result.json").write_text(json.dumps({"instances": []}))
    pipe = _Pipe()
    sam3_worker._sam3_work(pipe, str(tmp_path), cfg)
    assert not cropped and any("carries no scene.glb.stamp.json" in m for m in pipe.logs), \
        "an unstamped scene mesh is never cropped (declared)"
    # a stamp of another reconstruction: refused, the stage fails
    import correction.epoch as ep
    monkeypatch.setattr(ep, "reconstruction_id", lambda output_dir: "this-recon")
    (scene / sam3_worker.SCENE_MESH_STAMP).write_text(json.dumps({ep.RECONSTRUCTION_ID_KEY: "other"}))
    with pytest.raises(RuntimeError, match="another reconstruction"):
        sam3_worker._sam3_work(_Pipe(), str(tmp_path), cfg)
    (scene / sam3_worker.SCENE_MESH_STAMP).write_text(json.dumps({ep.RECONSTRUCTION_ID_KEY: "this-recon"}))
    sam3_worker._sam3_work(_Pipe(), str(tmp_path), cfg)
    assert cropped == [1]


# ── point 96: the cloud stage projects no stale masks ────────────────────────────────

def test_the_cloud_worker_projects_nothing_and_removes_stale_projection_products():
    src = (SERVER / "workers" / "cloudcompy_worker.py").read_text()
    assert "map_segmentation_to_cloud" not in src and "_result_is_stale" not in src
    assert "run_second_pass" not in src
    assert "PROJECTION_PRODUCTS" in src and '"segmentation_result.json"' in src
    assert "stop_semantic_service_verified(pipe" in src and "require_exclusive_gpu" in src
    assert "Potree build failed (non-critical)" not in src


# ── point 95: the loop classes cached by content, never a default for a dead service ──

def test_loop_classes_are_keyed_by_content_and_a_dead_service_fails(tmp_path, monkeypatch):
    from reconstruction.loops import semantic_classes as sc
    out = tmp_path / "output"
    out.mkdir()
    cfg = SimpleNamespace(enabled=True, default_class="structural", crops_per_instance=1,
                          max_tokens=64)
    # a store that only holds meta
    meta = {}

    class _Store:
        def __init__(self, p): pass
        def get_meta(self, k): return meta.get(k)
        def set_meta(self, k, v): meta[k] = v

    monkeypatch.setattr("phase_r.instance_store.InstanceStore", _Store)
    crop = np.zeros((4, 4, 3), np.uint8)
    monkeypatch.setattr(sc, "_mask_crop", lambda *a, **k: [crop])
    monkeypatch.setattr(sc, "_model_identity", lambda: {"sha256": "engine-a"})
    import semantic.service as svc
    monkeypatch.setattr(svc, "ensure_service", lambda *a, **k: False)
    insts = [{"instance_id": 1, "label": "desk"}]
    with pytest.raises(sc.LoopClassError, match="did not come up"):
        sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]})
    assert not meta, "no default recorded for a service that was down"
    # the service up: the verdict is cached under the CONTENT key, not the instance id
    monkeypatch.setattr(svc, "ensure_service", lambda *a, **k: True)
    import semantic.client as scl
    client = SimpleNamespace(health=lambda: {"ok": True})
    monkeypatch.setattr(scl, "get_semantic_client", lambda **kw: client)
    import segmentation.shape_proposer as sp
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: ({"class": "movable", "confidence": 0.9}, "{}"))
    res = sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]})
    assert res[1]["class"] == "movable" and res[1]["provenance"] == "vlm_proposed"
    key = next(iter(meta))
    assert key.startswith(sc.CACHE_PREFIX) and "loop_class_1" not in meta
    # the same crops under another instance id hit the cache; another model does not
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    res2 = sc.classify_instances(out, tmp_path, [{"instance_id": 7, "label": "desk"}], cfg, {7: 0}, {7: [0]})
    assert res2[7]["class"] == "movable"
    monkeypatch.setattr(sc, "_model_identity", lambda: {"sha256": "engine-b"})
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: ({"class": "structural", "confidence": 0.8}, "{}"))
    assert sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]})[1]["class"] == "structural"
    # an unusable answer is a failure, never a default
    monkeypatch.setattr(sc, "_model_identity", lambda: {"sha256": "engine-c"})
    monkeypatch.setattr(sp, "_chat_json", lambda *a, **k: (None, "garbage"))
    with pytest.raises(sc.LoopClassError, match="nothing usable"):
        sc.classify_instances(out, tmp_path, insts, cfg, {1: 0}, {1: [0]})
