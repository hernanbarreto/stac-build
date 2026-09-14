"""USER 2026-09-13: ONE command ("Reconstruir") delivers the finished cloud —
the semantic chain and the certification are stages of the reconstruction
pipeline, nothing is manual. These tests pin the stage list the production
config builds, the certification stage's resume probe, and the SALAD
candidate thresholds' path from config.yaml into the fork's Omega config."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"


def _raw():
    return yaml.safe_load(CONFIG_YAML.read_text())


def test_production_pipeline_runs_the_whole_chain():
    from pipeline_manager import build_pipeline_stages, StageId
    raw = _raw()
    assert raw["pipeline"]["auto_segment"] is True, "the automatic segmentation must be ON"
    assert raw["certify"]["auto_after_segmentation"] is True, "the automatic certification must be ON"
    stages = build_pipeline_stages(backend=str(raw["reconstruction"]["backend"]))
    enabled = [s.id for s in stages if s.enabled]
    # reconstruction → VLM → SAM3 → cleaned cloud → certification, in this order
    want = [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY, StageId.VLM, StageId.SAM3, StageId.CERTIFY]
    assert enabled[:len(want)] == want, enabled
    if raw["pipeline"]["auto_tsdf"] is False:
        assert StageId.TSDF not in enabled and StageId.PGSR not in enabled


def test_production_config_loads_through_every_typed_loader():
    """config.yaml must load through the typed loaders the pipeline stages use
    (a missing key fails at load naming it — pccr 2026-09-13: the graph's
    gate_mode / certify.gates.mode / nonstructural_sigma_factor were required
    by the code and absent from the file). The production modes: every gate
    advisory — measured, declared, the correction applied (USER 2026-09-13)."""
    from reconstruction.loops.config import load_loops_config
    from correction.config import load_correction_config
    raw = _raw()
    cfg = load_loops_config(raw)
    assert cfg.certify.gates.mode == "advisory"
    assert cfg.graph.gate_mode == "advisory"
    assert cfg.loops.semantic.nonstructural_sigma_factor >= 1.0
    assert cfg.certify.auto_after_segmentation is True
    ccfg = load_correction_config(raw)
    assert ccfg.gates.mode == "advisory"
    assert ccfg.apply.potree_rebuild is True, "the certified epoch must carry its own octree"
    # the fork receives the same modes
    from reconstruction.loops.config import fork_model_graph, fork_model_loops
    assert fork_model_graph(cfg)["gate_mode"] == "advisory"
    assert fork_model_loops(cfg, "/x")["nonstructural_sigma_factor"] == cfg.loops.semantic.nonstructural_sigma_factor


def test_certify_stage_registered_and_cascaded():
    from pipeline_manager import (DEFAULT_STAGE_ORDER, STAGE_REGISTRY, PipelineManager, StageId)
    assert StageId.CERTIFY in DEFAULT_STAGE_ORDER
    assert DEFAULT_STAGE_ORDER.index(StageId.CERTIFY) > DEFAULT_STAGE_ORDER.index(StageId.CLOUDCOMPY)
    # the cloud must exist BEFORE the semantic stages: SAM3 projects its masks
    # onto it the moment they exist, and a SAM3 failure must not cost the cloud
    assert DEFAULT_STAGE_ORDER.index(StageId.CLOUDCOMPY) < DEFAULT_STAGE_ORDER.index(StageId.VLM)
    assert DEFAULT_STAGE_ORDER.index(StageId.CLOUDCOMPY) < DEFAULT_STAGE_ORDER.index(StageId.SAM3)
    assert STAGE_REGISTRY[StageId.CERTIFY]["module"] == "workers.certify_worker"
    import importlib
    mod = importlib.import_module(STAGE_REGISTRY[StageId.CERTIFY]["module"])
    assert callable(getattr(mod, "run"))
    # every upstream stage invalidates the acta; the acta invalidates the meshes
    for up in (StageId.RECONSTRUCTION, StageId.VLM, StageId.SAM3, StageId.CLOUDCOMPY):
        assert StageId.CERTIFY in PipelineManager.CASCADE_INVALIDATION[up], up
    assert StageId.TSDF in PipelineManager.CASCADE_INVALIDATION[StageId.CERTIFY]
    assert "certify_acta.json" in PipelineManager.STAGE_OUTPUT_FILES[StageId.CERTIFY]


def test_certify_probe_follows_the_cloud_and_the_segmentation(tmp_path):
    from pipeline_manager import PipelineManager, StageId
    out = tmp_path / "output"
    out.mkdir()
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.CERTIFY)
    assert not done and "no certification acta" in why
    (out / "cleaned_cloud.ply").write_bytes(b"ply")
    (out / "segmentation_result.json").write_text("{}")
    time.sleep(0.05)
    (out / "certify_acta.json").write_text("{}")
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.CERTIFY)
    assert done, why
    # a newer segmentation (the manager re-ran) makes the acta stale
    time.sleep(0.05)
    (out / "segmentation_result.json").write_text("{}")
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.CERTIFY)
    assert not done and "segmentation_result.json" in why
    # the acta's own epochs rewrite the cloud AFTER the acta: not stale
    os.utime(out / "certify_acta.json")
    time.sleep(0.05)
    (out / "cleaned_cloud.ply").write_bytes(b"ply2")
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.CERTIFY)
    assert not done and "cleaned_cloud.ply" in why
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 1}))
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.CERTIFY)
    assert done, why


def test_salad_thresholds_come_from_config_and_reach_the_fork():
    from reconstruction.loops.config import load_loops_config, fork_loop_salad, LoopsConfigError
    raw = _raw()
    cfg = load_loops_config(raw)
    s = cfg.loops.salad
    assert 0 < s.similarity_threshold < 1 and s.top_k >= 1 and s.min_gap_keyframes >= 1
    d = fork_loop_salad(cfg)
    assert d["similarity_threshold"] == pytest.approx(raw["loops"]["salad"]["similarity_threshold"])
    assert d["min_gap"] == raw["loops"]["salad"]["min_gap_keyframes"]
    assert d["use_nms"] == (raw["loops"]["salad"]["nms_threshold"] > 0)
    # a missing key aborts naming it (no silent default)
    bad = json.loads(json.dumps(raw))
    del bad["loops"]["salad"]["similarity_threshold"]
    with pytest.raises(LoopsConfigError, match="loops.salad.similarity_threshold"):
        load_loops_config(bad)
    assert d["top_k"] == raw["loops"]["salad"]["top_k"]
    assert d["image_size"] == list(raw["loops"]["salad"]["image_size"])


def test_no_hard_coded_salad_thresholds_in_the_omega_path():
    """The vendor's 0.85 / NMS 25 must not be written by the Omega path any more."""
    src = (Path(__file__).resolve().parents[1] / "workers" / "map_worker.py").read_text()
    omega = src[src.index("def _build_vggtomega_config"):src.index("def _emit_omega_depth")]
    assert "similarity_threshold" not in omega
    chunked = src[src.index("def _apply_chunked_metric"):src.index("def _ensure_anchors")]
    assert "fork_loop_salad" in chunked
