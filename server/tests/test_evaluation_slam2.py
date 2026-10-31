"""claude_stac_slam2.txt, phase 4 — the VGGT-SLAM 2.0 variants of the battery (evaluation.slam2):
the registry, the resolution from the scene's calibration, the skip rule, the bit-for-bit
comparator, the table, the viewer publication and a scene without reference distances — with
simulated stages, no GPU."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation import slam2 as S2
from evaluation import variants as V
from tests import synth_evaluation as S
from tests.test_evaluation_variants import FakeRunner, _raw

SERVER_DIR = Path(__file__).resolve().parent.parent


def _slam2():
    return S2.slam2_config(_raw()["evaluation"])


def test_the_four_variants_are_registered_with_their_overrides_and_apply_to_the_configuration():
    raw = _raw()
    specs = V.load_variants(raw["evaluation"])
    for name in ("slam2_focal_common", "slam2_attention_gate", "slam2_attention_gate_salad", "slam2_all"):
        assert specs[name].kind == "chain" and specs[name].overrides["certify.keep_unaligned_chunks"] is True
    assert specs["reference"].overrides == {"certify.keep_unaligned_chunks": True}
    fc = specs["slam2_focal_common"].overrides[S2.SEAMS_KEY]
    assert fc == {"focal_model": "common_K", "common_K_source": "session_median", "judge_samples": 20000}
    g = specs["slam2_attention_gate"].overrides[S2.GATE_KEY]
    assert g["mode"] == "gate" and g["layer"] is None and g["min_score"] is None and g["salad_similarity_threshold"] is None
    assert specs["slam2_attention_gate_salad"].overrides[S2.GATE_KEY]["salad_similarity_threshold"] == 0.60
    assert S2.SEAMS_KEY in specs["slam2_all"].overrides and S2.GATE_KEY in specs["slam2_all"].overrides
    # the overrides land where the loaders read them (a resolved gate passes the typed loader)
    from reconstruction.loops.config import load_loops_config
    eff = V.apply_overrides(raw, specs["slam2_focal_common"].overrides)
    mg = load_loops_config(eff)
    assert mg.seams.focal_model == "common_K" and eff["certify"]["keep_unaligned_chunks"] is True
    resolved = copy.deepcopy(specs["slam2_all"].overrides)
    resolved[S2.GATE_KEY] = dict(resolved[S2.GATE_KEY], layer=4, min_score=0.5)
    mg2 = load_loops_config(V.apply_overrides(raw, resolved))
    assert mg2.loops.attention_gate.mode == "gate" and mg2.loops.attention_gate.layer == 4
    from reconstruction.loops.config import fork_loop_salad
    assert fork_loop_salad(mg2)["similarity_threshold"] == 0.60
    assert S2.gate_params_from_overrides(list(specs.values()))["salad_grid"] == [0.55, 0.60, 0.65, 0.70]


def _ref_run(tmp_path, *, calibration=None, seam_rule=None):
    sd = tmp_path / "ref_session"
    run = sd / "output" / "maplong_run"
    run.mkdir(parents=True)
    if calibration is not None:
        (run / "attention_calibration").mkdir()
        (run / "attention_calibration" / "attention_calibration.json").write_text(json.dumps({"choice": calibration}))
    if seam_rule is not None:
        (run / "seam_radial").mkdir()
        (run / "seam_radial" / "seam_radial.json").write_text(json.dumps(
            {"session_rule": {"common_K_session_median": seam_rule}}))
    return sd


def test_gate_variants_take_layer_and_threshold_from_the_calibration_or_are_not_run(tmp_path):
    specs = V.load_variants(_raw()["evaluation"])
    scene = V.SceneSpec("pccr", tmp_path / "scene_session")
    (scene.session_dir / "output" / "maplong_run").mkdir(parents=True)
    # nothing calibrated anywhere: not run, by name
    out = S2.resolve_variant(specs["slam2_attention_gate"], scene_session=scene.session_dir, results={}, slam2=_slam2())
    assert out.kind == "not_executed" and "attention_calibration.json" in out.reason and "does not exist" in out.reason
    # the reference of this run carries a choice: filled from it
    ref = _ref_run(tmp_path, calibration={"chosen": True, "layer": 7, "threshold": 0.4321, "precision": 0.96, "recall": 0.8,
                                          "n_pos": 9, "n_neg": 40})
    results = {"reference": {"status": "finished", "session_dir": str(ref)}}
    out = S2.resolve_variant(specs["slam2_attention_gate_salad"], scene_session=scene.session_dir, results=results, slam2=_slam2())
    assert out.kind == "chain" and out.overrides[S2.GATE_KEY]["layer"] == 7 and out.overrides[S2.GATE_KEY]["min_score"] == 0.4321
    assert out.overrides[S2.GATE_KEY]["salad_similarity_threshold"] == 0.60 and out.resolution["calibration"]["available"]
    # a calibration that chose nothing: not run with its reason
    ref2 = _ref_run(tmp_path / "b", calibration={"chosen": False, "reason": "insufficient labels: 2 positive(s)"})
    out2 = S2.resolve_variant(specs["slam2_all"], scene_session=scene.session_dir,
                              results={"reference": {"status": "finished", "session_dir": str(ref2)}}, slam2=_slam2())
    assert out2.kind == "not_executed" and "insufficient labels" in out2.reason
    # other variants pass through untouched
    assert S2.resolve_variant(specs["no_f5"], scene_session=scene.session_dir, results={}, slam2=_slam2()) is specs["no_f5"]


def test_the_skip_rule_records_focal_common_as_not_run_when_the_offline_instrument_says_identity(tmp_path):
    specs = V.load_variants(_raw()["evaluation"])
    scene = V.SceneSpec("pccr", tmp_path / "scene_session")
    (scene.session_dir / "output" / "maplong_run").mkdir(parents=True)
    ref = _ref_run(tmp_path, seam_rule={"improves_by_rule": False, "reason": "does not clear the error",
                                        "median_before_m": 0.021, "median_after_m": 0.0205})
    results = {"reference": {"status": "finished", "session_dir": str(ref)}}
    out = S2.resolve_variant(specs["slam2_focal_common"], scene_session=scene.session_dir, results=results, slam2=_slam2())
    assert out.kind == "not_executed" and "skip_if_offline_identity" in out.reason and "would be the reference" in out.reason
    # the rule off: the variant runs whatever the instrument says
    off = dict(_slam2(), skip_if_offline_identity=False)
    assert S2.resolve_variant(specs["slam2_focal_common"], scene_session=scene.session_dir, results=results, slam2=off).kind == "chain"
    # the instrument says it improves: runs
    ref2 = _ref_run(tmp_path / "b", seam_rule={"improves_by_rule": True, "reason": "ok"})
    out2 = S2.resolve_variant(specs["slam2_focal_common"], scene_session=scene.session_dir,
                              results={"reference": {"status": "finished", "session_dir": str(ref2)}}, slam2=_slam2())
    assert out2.kind == "chain" and out2.resolution["offline_common_K"]["improves_by_rule"] is True
    # no instrument anywhere: runs, declared
    logs = []
    out3 = S2.resolve_variant(specs["slam2_focal_common"], scene_session=scene.session_dir, results={}, slam2=_slam2(),
                              log=logs.append)
    assert out3.kind == "chain" and any("no offline common-K verdict" in m for m in logs)


def test_the_bitwise_comparator_reports_equality_or_the_first_difference_by_name(tmp_path):
    a = S.make_session(tmp_path / "a")
    b = S.make_session(tmp_path / "b")
    for sd in (a.session_dir, b.session_dir):
        run = sd / "output" / "maplong_run"
        run.mkdir()
        (run / "camera_poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
        (run / "metric_lock.json").write_text(json.dumps({"fork_stamp": sd.name, "chunks": {"0": {"s": 1.5}}}))
    rep = S2.bitwise_compare(a.session_dir, b.session_dir, log=lambda *a_: None)
    assert rep["products"]["published_cloud"]["equal"] and rep["products"]["camera_poses.txt"]["equal"]
    assert rep["products"]["metric_lock.json"]["equal"]                       # stamps ignored
    assert rep["equal_everywhere_compared"] and rep["first_difference"] is None
    assert "_tmp_results_unaligned" in rep["not_compared"] and "kept no chunk npys" in rep["not_compared"]["_tmp_results_unaligned"]
    # one point moved: the cloud differs, named with the pairing counts
    from evaluation.cloud_io import write_ply
    from evaluation.product import read_ply_memmap
    data = np.array(read_ply_memmap(b.session_dir / "output" / "cleaned_cloud.ply"))
    data["x"][0] += 0.01
    write_ply(b.session_dir / "output" / "cleaned_cloud.ply", data)
    rep2 = S2.bitwise_compare(a.session_dir, b.session_dir, log=lambda *a_: None)
    assert rep2["products"]["published_cloud"]["equal"] is False
    assert rep2["first_difference"].startswith("cleaned_cloud.ply:") and "paired by provenance" in rep2["first_difference"]
    # a json record that differs beyond its stamp
    (b.session_dir / "output" / "maplong_run" / "metric_lock.json").write_text(
        json.dumps({"fork_stamp": "x", "chunks": {"0": {"s": 1.6}}}))
    rep3 = S2.bitwise_compare(a.session_dir, b.session_dir, log=lambda *a_: None)
    assert rep3["products"]["metric_lock.json"]["first_difference"] == "metric_lock.json.chunks.0.s: 1.5 != 1.6"
    # chunk npys on both sides: compared array by array
    for sd, val in ((a.session_dir, 1.0), (b.session_dir, 1.0)):
        d = sd / "output" / "maplong_run" / "_tmp_results_unaligned"
        d.mkdir()
        np.save(d / "chunk_0.npy", {"world_points": np.ones((2, 3, 3), np.float32) * val, "_stac_fork_stamp": sd.name})
    assert S2.bitwise_compare(a.session_dir, b.session_dir, log=lambda *a_: None)["products"]["_tmp_results_unaligned"]["equal"]
    np.save(b.session_dir / "output" / "maplong_run" / "_tmp_results_unaligned" / "chunk_0.npy",
            {"world_points": np.ones((2, 3, 3), np.float32) * 2.0, "_stac_fork_stamp": "y"})
    rep4 = S2.bitwise_compare(a.session_dir, b.session_dir, log=lambda *a_: None)
    assert "chunk_0.npy[world_points]: arrays differ" in rep4["products"]["_tmp_results_unaligned"]["first_difference"]


def test_the_table_carries_every_variant_with_its_difference_against_the_reference(tmp_path):
    def rec(status, metrics=None, seconds=100.0):
        return {"status": status, "reason": "" if status == "finished" else "x", "seconds": seconds,
                "summary": {"abs_error_m": {}}, "coverage": {"keyframes_registered_pct": 100.0},
                "cost": {"gpu_hours": 0.5, "cost_usd": 0.8}, "slam2": {"metrics": metrics or {}}}
    base = {"salad_candidates": 40, "gate_candidates": None, "gate_passed": None, "bridges": 9, "bridges_accepted": 9,
            "closures_cycle_consistent": 8, "graph_judges": 8, "graph_verdict": "IDENTITY", "fork_heldout_pairs": 1200,
            "fork_heldout_after_m": 0.031, "f5_heldout_px": 0.82, "seam_held_median_m_mean": 0.0215,
            "seams": [{"seam": 0, "held_norm_median_m": 0.0215, "lateral_median_m": 0.01, "axial_abs_median_m": 0.015,
                       "radius_bins": [{"lo": 0, "hi": 0.3, "n": 500, "lateral_median_m": 0.008, "axial_abs_median_m": 0.014}],
                       "intrinsics_offset_px": {"fx": 1.4}}]}
    gated = dict(base, gate_candidates=40, gate_passed=12, bridges=12, fork_heldout_after_m=0.030, f5_heldout_px=0.80,
                 seam_held_median_m_mean=0.0210)
    results = {"reference": rec("finished", base), "slam2_attention_gate": rec("finished", gated, 90.0),
               "slam2_focal_common": {"status": "not_executed", "reason": "skip rule", "seconds": 0.1},
               "slam2_all": rec("failed")}
    outs = S2.write_table(tmp_path, "pccr", results, "reference", bitwise={"equal_everywhere_compared": True}, png=True)
    rows = {r["variant"]: r for r in json.loads(Path(outs["json"]).read_text())["rows"]}
    assert set(rows) == set(results)
    g = rows["slam2_attention_gate"]
    assert g["gate_passed"] == 12 and g["d_fork_heldout_after_m_vs_reference"] == pytest.approx(-0.001)
    assert g["d_f5_heldout_px_vs_reference"] == pytest.approx(-0.02) and g["d_seconds_vs_reference"] == -10.0
    assert g["d_seam_held_median_m_mean_vs_reference"] == pytest.approx(-0.0005)
    assert rows["reference"]["d_fork_heldout_after_m_vs_reference"] == 0.0
    assert rows["slam2_focal_common"]["status"] == "not_executed" and rows["slam2_focal_common"]["d_f5_heldout_px_vs_reference"] is None
    csv_rows = Path(outs["csv"]).read_text().splitlines()
    assert csv_rows[0].startswith("scene,variant,status,reason,salad_candidates") and len(csv_rows) == 5
    assert Path(outs["png"]).is_file()
    doc = json.loads(Path(outs["json"]).read_text())
    assert doc["bitwise_comparator"] == {"equal_everywhere_compared": True} and "no verdict" in doc["note"]


def test_variant_metrics_read_the_records_or_leave_none(tmp_path):
    sd = tmp_path / "s"
    run = sd / "output" / "maplong_run"
    run.mkdir(parents=True)
    (run / "loop_closures.txt").write_text("# stamp\n140, 8, 0.7\n100, 30, 0.66\n# 0: a.jpg\n")
    (run / "attention_gate.json").write_text(json.dumps({"n_candidates": 2, "n_passed": 1, "layer": 4, "min_score": 0.5}))
    (run / "loop_edges.json").write_text(json.dumps({"edges": [{"stage": "verification", "status": "accepted"},
                                                               {"stage": "verification", "status": "rejected"},
                                                               {"stage": "spatial_gate", "status": "rejected"}]}))
    (run / "pose_graph.json").write_text(json.dumps({"verdict": "IDENTITY", "reason": "r", "cycle_consistency": {"n_kept": 1},
                                                     "leave_one_out_closures": {"n_judges": 1},
                                                     "gates": {"holdout_surface_pairs": {"n_pairs": 10, "before_m": 0.03, "after_m": 0.029}}}))
    (run / "focal_model.json").write_text(json.dumps({"applied": True, "reason": "ok", "seams_rule": {"median_before_m": 0.02, "median_after_m": 0.01}}))
    m = S2.variant_metrics(sd)
    assert m["salad_candidates"] == 2 and m["gate_passed"] == 1 and m["bridges"] == 2 and m["bridges_accepted"] == 1
    assert m["closures_cycle_consistent"] == 1 and m["graph_verdict"] == "IDENTITY" and m["fork_heldout_after_m"] == 0.029
    assert m["focal_model_applied"] is True and m["focal_seams_after_m"] == 0.01
    assert "seams" not in m and m.get("f5_heldout_px") is None
    assert S2.variant_metrics(None) == {}


def test_publication_makes_the_variant_a_scan_of_the_eval_project(tmp_path):
    sess = tmp_path / "run" / "pccr" / "slam2_all" / "session"
    (sess / "output").mkdir(parents=True)
    projects = tmp_path / "projects"
    rec = S2.publish_variant_scan(projects, "pccr", "slam2_all", sess, "__eval", log=lambda *a: None)
    link = projects / "pccr__eval" / "scans" / "slam2_all" / "src_default"
    assert link.is_symlink() and link.resolve() == sess.resolve() and rec["scan_key"] == "slam2_all/default"
    pj = json.loads((projects / "pccr__eval" / "project.json").read_text())
    assert pj["slug"] == "pccr__eval" and pj["evaluation"]["scene"] == "pccr"
    rec2 = S2.publish_variant_scan(projects, "pccr", "slam2_all", sess, "__eval", log=lambda *a: None)   # idempotent
    assert rec2 == rec and link.is_symlink()
    from project_paths import ProjectPaths
    pp = ProjectPaths(str(projects), "pccr__eval")
    assert pp.list_scan_days() == ["slam2_all"] and pp.list_sources("slam2_all") == ["default"]
    assert pp.source_output("slam2_all", "default").exists()


def test_a_scene_without_reference_distances_runs_with_the_hooks_and_declares_the_error_phases(tmp_path):
    scene = S.make_session(tmp_path / "scene", omega_records=True)            # no reference_measures.json
    cfg = S.eval_config(**{"variants_keep": "all"})
    raw = _raw()
    specs = [V.VariantSpec("reference", "chain", overrides={"certify.keep_unaligned_chunks": True}),
             V.VariantSpec("slam2_focal_common", "chain",
                           overrides={"certify.keep_unaligned_chunks": True,
                                      S2.SEAMS_KEY: {"focal_model": "common_K", "common_K_source": "session_median", "judge_samples": 20000}}),
             V.VariantSpec("slam2_attention_gate", "chain",
                           overrides={S2.GATE_KEY: {"mode": "gate", "layer": None, "min_score": None, "top_fraction": 0.25,
                                                    "salad_similarity_threshold": None,
                                                    "calibration": {"select": "max_auc_lower_ci", "target_precision": 0.95,
                                                                    "min_labels": 5, "n_negatives": 50, "n_boot": 100,
                                                                    "confidence": 0.95, "salad_grid": [0.6]}}})]
    slam2 = dict(_slam2(), publish_to_viewer=True)
    hooks = S2.Slam2Hooks(slam2, raw, specs, projects_dir=tmp_path / "projects", gpu=False, log=lambda *a: None)
    # without references the executor used to refuse the scene; with references_optional it runs
    with pytest.raises(Exception):
        V.run_scene(V.SceneSpec("synth", scene.session_dir), specs[:1], tmp_path / "run0", cfg, FakeRunner(), raw,
                    log=lambda *a: None, plots=False)
    res = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run", cfg, FakeRunner(), raw,
                      log=lambda *a: None, plots=False, references_optional=True, hooks=hooks)
    r = res["results"]
    assert r["reference"]["status"] == "finished"
    assert r["reference"]["summary"]["n_test_evaluated"] == 0 and "no reference distances" in r["reference"]["summary"]["status"]
    assert r["reference"]["coverage"]["keyframes_registered_pct"] == 100.0
    s2 = r["reference"]["slam2"]
    assert s2["seam_radial"]["ran"] is False and "kept no unaligned chunk npys" in s2["seam_radial"]["reason"]
    assert s2["attention_calibrate"]["ran"] is False and "no card" in s2["attention_calibrate"]["reason"]
    assert s2["bitwise_vs_existing"]["compared"] is False and "existing_runs" in s2["bitwise_vs_existing"]["reason"]
    assert s2["viewer"]["project"] == "synth__eval" and (tmp_path / "projects" / "synth__eval" / "scans" / "reference" / "src_default").is_symlink()
    assert isinstance(s2["metrics"], dict)
    # the focal variant ran (no offline verdict → declared, runs); the gate variant: no calibration → not run
    assert r["slam2_focal_common"]["status"] == "finished"
    assert r["slam2_attention_gate"]["status"] == "not_executed" and "attention_calibration.json" in r["slam2_attention_gate"]["reason"]
    assert (tmp_path / "run" / "synth" / "slam2_comparison.csv").is_file() and (tmp_path / "run" / "synth" / "slam2_comparison.json").is_file()
    rows = {x["variant"]: x for x in json.loads((tmp_path / "run" / "synth" / "slam2_comparison.json").read_text())["rows"]}
    assert rows["slam2_attention_gate"]["status"] == "not_executed"


def test_the_manifest_records_branch_and_commit_of_both_repos():
    from evaluation.cli import repo_git_state
    g = repo_git_state()
    for key in ("stac_build", "fork_vggt_long"):
        assert "commit" in g[key] and "branch" in g[key] and len(g[key]["commit"]) >= 7, g[key]
