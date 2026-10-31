"""Phase 8 of the evaluation battery: the CLI on synthetic scenes with simulated stages, and the
module entry point."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluation import cli as CLI
from evaluation import colmap_io as CIO
from evaluation.product import StacProduct
from tests import synth_evaluation as S
from tests.test_evaluation_variants import FakeRunner, _scene_with_refs

SERVER_DIR = Path(__file__).resolve().parent.parent


def test_run_command_builds_a_complete_run(tmp_path, capsys):
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "cloud",
                           "runs_dir": str(tmp_path / "runs"), "slam2.projects_dir": str(tmp_path / "projects")})
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference,no_f6_bend,scale_mapanything",
                   "--run-id", "t1", "--no-gpu-check", "--no-plots"], runner=FakeRunner(), cfg=cfg, log=lambda *a: None)
    assert rc == 0
    run = tmp_path / "runs" / "t1"
    man = json.loads((run / "manifest.json").read_text())
    assert man["status"] == "finished" and man["variants"] == ["reference", "no_f6_bend", "scale_mapanything"]
    assert man["results"]["scene"]["scale_mapanything"]["status"] == "not_executed"
    scene_dir = run / "scene"
    for f in ("comparison.csv", "reference/dimensions.csv", "reference/curve_distance.csv", "reference/coverage.json",
              "reference/cost.json", "reference/confidence_groups.csv", "reference/variant.json", "reference/log.txt",
              "no_f6_bend/dimensions_summary.json", "scale_mapanything/variant.json"):
        assert (scene_dir / f).exists(), f
    for f in ("pareto.json", "profiles.json", "pareto.csv", "run.log"):
        assert (run / f).exists(), f
    # keep policy 'cloud': the copy holds the products, not the frames
    sess = scene_dir / "reference" / "session"
    assert (sess / "output" / "cleaned_cloud.ply").exists() and not (sess / "frames").exists()
    prof = json.loads((run / "profiles.json").read_text())["profiles"]
    assert prof[0]["variant"] in ("reference", "no_f6_bend")
    assert str(run) in capsys.readouterr().out
    assert CLI.main(["list", "--out", str(tmp_path / "runs")], cfg=cfg) == 0
    assert "t1" in capsys.readouterr().out


def test_run_refuses_unknown_scene_or_variant_and_handles_a_scene_without_references(tmp_path, capsys):
    cfg = S.eval_config(**{"runs_dir": str(tmp_path / "runs"), "slam2.projects_dir": str(tmp_path / "projects")})
    assert CLI.main(["run", "--scenes", "nope", "--variants", "reference", "--no-gpu-check"], runner=FakeRunner(), cfg=cfg) == 2
    scene = S.make_session(tmp_path / "noref")
    assert CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "bogus", "--no-gpu-check"],
                    runner=FakeRunner(), cfg=cfg) == 2
    # references optional (evaluation.slam2, USER 2026-10-31: no cota exists): the scene RUNS, its error
    # phases declared not measured, the manifest carries both repos' branch and commit
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference", "--run-id", "t2", "--no-gpu-check",
                   "--no-plots"], runner=FakeRunner(), cfg=cfg, log=lambda *a: None)
    assert rc == 0
    man = json.loads((tmp_path / "runs" / "t2" / "manifest.json").read_text())
    assert man["status"] == "finished" and man["git"]["stac_build"]["commit"] and man["git"]["fork_vggt_long"]["branch"]
    rec = json.loads((tmp_path / "runs" / "t2" / "noref" / "reference" / "variant.json").read_text())
    assert rec["status"] == "finished" and "no reference distances" in rec["summary"]["status"]
    assert (tmp_path / "runs" / "t2" / "noref" / "slam2_comparison.csv").exists()
    assert (tmp_path / "projects" / "noref__eval" / "scans" / "reference" / "src_default").is_symlink()
    # references required: the old refusal, recorded
    cfg2 = S.eval_config(**{"runs_dir": str(tmp_path / "runs"), "slam2.references_optional": False,
                            "slam2.projects_dir": str(tmp_path / "projects")})
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference", "--run-id", "t3", "--no-gpu-check",
                   "--no-plots"], runner=FakeRunner(), cfg=cfg2, log=lambda *a: None)
    assert rc == 1
    err = json.loads((tmp_path / "runs" / "t3" / "noref" / "scene_error.json").read_text())
    assert "cannot be evaluated" in err["error"]
    assert json.loads((tmp_path / "runs" / "t3" / "manifest.json").read_text())["status"] == "failed"


def test_gpu_gate_turns_gpu_variants_into_not_executed_with_the_reason(tmp_path, monkeypatch):
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"runs_dir": str(tmp_path / "runs"), "curves.min_bin_n": 1,
                           "slam2.projects_dir": str(tmp_path / "projects")})
    monkeypatch.setattr(CLI, "gpu_gate", lambda specs, log: "the card is not free: vLLM holds 38 GB")
    monkeypatch.setattr(CLI, "card_key_or_none", lambda log: None)
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference", "--run-id", "t3", "--no-plots"],
                  runner=FakeRunner(), cfg=cfg, log=lambda *a: None)
    assert rc == 0
    rec = json.loads((tmp_path / "runs" / "t3" / "scene" / "reference" / "variant.json").read_text())
    assert rec["status"] == "not_executed" and "not free" in rec["reason"]


def test_import_command_adds_a_variant_to_a_run(tmp_path):
    scene, doc = _scene_with_refs(tmp_path)
    prod = StacProduct(scene.session_dir, "s", log=lambda *a: None)
    cam = CIO.ColmapCamera(1, "PINHOLE", S.W, S.H, np.array(scene.cam.params[:4], np.float64))
    images = {i + 1: CIO.ColmapImage(i + 1, f"{f:06d}.jpg", 1, CIO.rotmat_to_qvec(np.linalg.inv(scene.c2w[i])[:3, :3]),
                                     np.linalg.inv(scene.c2w[i])[:3, 3]) for i, f in enumerate(scene.frames)}
    sparse = CIO.write_model_text(tmp_path / "ext" / "0", {1: cam}, images, prod.xyz[::1000])
    from evaluation.cloud_io import write_ply
    data = np.zeros(len(prod.xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    data["x"], data["y"], data["z"] = prod.xyz[:, 0], prod.xyz[:, 1], prod.xyz[:, 2]
    write_ply(tmp_path / "ext" / "fused.ply", data)
    cfg = S.eval_config(**{"runs_dir": str(tmp_path / "runs"), "curves.min_bin_n": 1,
                           "slam2.projects_dir": str(tmp_path / "projects")})
    rc = CLI.main(["import-colmap", "--scene", str(scene.session_dir), "--name", "metashape", "--sparse", str(sparse),
                   "--dense", str(tmp_path / "ext" / "fused.ply"), "--run", "imp", "--no-plots"], cfg=cfg, log=lambda *a: None)
    assert rc == 0
    man = json.loads((tmp_path / "runs" / "imp" / "manifest.json").read_text())
    assert man["results"]["scene"]["metashape"]["status"] == "finished" and "metashape" in man["variants"]
    assert (tmp_path / "runs" / "imp" / "scene" / "metashape" / "dimensions.csv").exists()


def test_module_entry_point_prints_help():
    r = subprocess.run([sys.executable, "-m", "server.evaluation", "--help"], cwd=SERVER_DIR.parent,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "import-colmap" in r.stdout
    r = subprocess.run([sys.executable, "-m", "evaluation", "list", "--out", "/nonexistent"], cwd=SERVER_DIR,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0


def test_stages_off_of_the_run_joins_every_chain_variant_and_is_recorded(tmp_path):
    """--stages-off (claude_stac_slam2.txt phase 5): the ids are switched off in every chain variant of
    the run on top of the variant's own, reach the chain runner, and are recorded in the manifest and in
    each variant's record; an unknown id is refused by name."""
    scene, _doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "cloud",
                           "runs_dir": str(tmp_path / "runs"), "slam2.projects_dir": str(tmp_path / "projects")})
    runner = FakeRunner()
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference,no_drift_correction,scale_mapanything",
                   "--run-id", "t4", "--no-gpu-check", "--no-plots", "--stages-off", "genrecon,certify"],
                  runner=runner, cfg=cfg, log=lambda *a: None)
    assert rc == 0
    run = tmp_path / "runs" / "t4"
    man = json.loads((run / "manifest.json").read_text())
    assert man["stages_off_cli"] == ["certify", "genrecon"]
    by_name = {c[0]: c for c in runner.calls}
    assert by_name["reference"][2] == ["certify", "genrecon"]
    assert by_name["no_drift_correction"][2] == ["certify", "genrecon"]          # its own 'certify' joined, once
    ref = json.loads((run / "scene" / "reference" / "variant.json").read_text())
    assert ref["variant"]["stages_off"] == ["certify", "genrecon"]
    ne = json.loads((run / "scene" / "scale_mapanything" / "variant.json").read_text())
    assert ne["variant"]["stages_off"] == [] and ne["status"] == "not_executed"
    rc = CLI.main(["run", "--scenes", str(scene.session_dir), "--variants", "reference", "--run-id", "t5",
                   "--no-gpu-check", "--no-plots", "--stages-off", "certify,meshing"], runner=FakeRunner(), cfg=cfg,
                  log=lambda *a: None)
    assert rc == 2
    assert not (tmp_path / "runs" / "t5").exists()
