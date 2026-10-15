"""Phase 6 of the evaluation battery: the manifest is complete and stable; two synthetic runs
with a fixed seed are identical bit for bit; a changed run is measured."""
import json

import numpy as np

from evaluation import determinism as DET
from evaluation import variants as V
from evaluation.product import StacProduct
from tests import synth_evaluation as S
from tests.test_evaluation_variants import _raw


def test_seed_keys_and_regime_cover_every_seed_of_the_configuration():
    raw = _raw()
    seeds = DET.seed_keys(raw)
    assert "reconstruction.precision.bend.seed" in seeds and "semantic.generation.temperature" in seeds
    assert all(k.endswith("seed") or k.endswith("temperature") for k in seeds)
    assert len([k for k in seeds if k.endswith("seed")]) >= 14
    reg = DET.deterministic_regime(raw)
    assert reg["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and reg["env"]["PYTHONHASHSEED"] == "0"
    assert reg["torch"]["use_deterministic_algorithms"] is True and reg["vlm"]["temperature"] == 0.0
    decl = DET.declared_non_deterministic({"reference": "chain", "colmap_global": "colmap"})
    assert any(d["variant"] == "colmap_global" for d in decl) and all(d["operation"] for d in decl)


def test_manifest_is_complete_and_stable(tmp_path):
    scene = S.make_session(tmp_path / "s")
    (scene.session_dir / "source_video.mp4").write_bytes(b"video bytes")
    cfg = S.eval_config()
    raw = _raw()
    specs = V.load_variants(raw["evaluation"])
    eff = {"reference": raw, "no_f5": V.apply_overrides(raw, specs["no_f5"].overrides)}
    scenes = [V.SceneSpec("s", scene.session_dir)]
    m1 = DET.run_manifest("r1", scenes, specs, cfg, raw, eff, gpu=False, argv=["run"], repeat=1)
    m2 = DET.run_manifest("r1", scenes, specs, cfg, raw, eff, gpu=False, argv=["run"], repeat=1)
    for key in ("scene_inputs", "variant_specs", "regime", "declared_non_deterministic", "checkpoints", "environment",
                "evaluation_config", "base_config_sha256", "effective_config_sha256", "effective_configs"):
        assert key in m1, key
    assert m1["scene_inputs"]["s"]["video"]["sha256"] == DET.sha256_file(scene.session_dir / "source_video.mp4")
    assert m1["checkpoints"]["da3"].get("revision") or m1["checkpoints"]["da3"].get("error")
    assert m1["checkpoints"]["vlm"] and m1["checkpoints"]["vggsfm_tracker"]["sha256"]
    assert m1["effective_config_sha256"]["no_f5"] != m1["effective_config_sha256"]["reference"]
    assert m1["environment"].get("python") or m1["environment"].get("error")
    assert DET.stable_part(m1) == DET.stable_part(m2)            # stable across calls
    json.dumps(m1, default=str)                                  # serialisable


def test_two_runs_with_the_same_seed_are_bit_identical_and_a_changed_one_is_measured(tmp_path):
    a = S.make_session(tmp_path / "a", noise_m=0.004, seed=7)
    b = S.make_session(tmp_path / "b", noise_m=0.004, seed=7)
    pa = StacProduct(a.session_dir, "a", log=lambda *x: None)
    pb = StacProduct(b.session_dir, "b", log=lambda *x: None)
    cmp = DET.compare_products(pa, pb)
    assert cmp["bit_identical_cloud"] and cmp["position_diff_m"]["max"] == 0.0 and cmp["n_only_a"] == 0
    assert cmp["cloud_sha256_a"] == cmp["cloud_sha256_b"] and cmp["poses_max_diff"] == 0.0
    c = S.make_session(tmp_path / "c", noise_m=0.004, seed=8, v_max=200)
    pc = StacProduct(c.session_dir, "c", log=lambda *x: None)
    cmp2 = DET.compare_products(pa, pc)
    assert not cmp2["bit_identical_cloud"] and cmp2["position_diff_m"]["max"] > 0 and cmp2["n_only_a"] > 0
    assert cmp2["n_paired"] == len(pc.xyz)
    # metrics of two records
    ra = {"status": "finished", "summary": {"abs_error_m": {"p95": 0.01, "median": 0.005}}, "coverage": {"n_models": 1}}
    rb = {"status": "finished", "summary": {"abs_error_m": {"p95": 0.012, "median": 0.005}}, "coverage": {"n_models": 1}}
    d = DET.compare_records(ra, rb)
    assert d["n_differences"] == 1 and d["differences"]["summary.abs_error_m.p95"]["diff"] == -0.002
    rep = DET.repeatability(tmp_path, "v", "v__rep2", {**ra, "session_dir": str(a.session_dir)},
                            {**rb, "session_dir": str(b.session_dir)})
    assert rep["points"]["bit_identical_cloud"] and (tmp_path / "repeatability_v.json").exists()


def test_run_scene_with_repeat_writes_the_repeatability_report(tmp_path):
    from tests.test_evaluation_variants import FakeRunner, _scene_with_refs
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "all"})
    specs = [V.VariantSpec("reference", "chain"), V.VariantSpec("skipped", "not_executed", reason="x")]
    res = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run", cfg, FakeRunner(), _raw(),
                      log=lambda *a: None, plots=False, repeat=2)
    assert set(res["results"]) == {"reference", "reference__rep2", "skipped"}
    rep = json.loads((tmp_path / "run" / "synth" / "repeatability_reference.json").read_text())
    assert rep["points"]["bit_identical_cloud"] and rep["metrics"]["n_differences"] == 0
