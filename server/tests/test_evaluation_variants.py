"""Phase 3 of the evaluation battery: the variant executor (overrides, identity of a default
variant, isolated failures), endpoint methods on a dense product, coverage, COLMAP model I/O and
the gauge-only known dimensions — all with simulated stages, no GPU."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation import colmap_io as CIO
from evaluation import coverage as COV
from evaluation import dimension_error as D
from evaluation import variants as V
from evaluation.product import StacProduct
from tests import synth_evaluation as S

SERVER_DIR = Path(__file__).resolve().parent.parent


def _raw():
    return yaml.safe_load((SERVER_DIR / "config.yaml").read_text())


def test_overrides_set_leaves_and_refuse_unknown_sections():
    raw = _raw()
    same = V.apply_overrides(raw, {})
    assert same == raw and same is not raw
    over = V.apply_overrides(raw, {"reconstruction.precision.bend.enabled": False,
                                   "reconstruction.precision.gauge.instruments": ["known_dims"]})
    assert over["reconstruction"]["precision"]["bend"]["enabled"] is False
    assert over["reconstruction"]["precision"]["gauge"]["instruments"] == ["known_dims"]
    assert raw["reconstruction"]["precision"]["bend"]["enabled"] is True          # the defaults never move
    with pytest.raises(V.VariantError, match="does not exist"):
        V.apply_overrides(raw, {"reconstruction.nope.key": 1})
    # a variant whose overrides restate the defaults is the reference's configuration exactly
    defaults = V.apply_overrides(raw, {"reconstruction.precision.bend.enabled": True,
                                       "reconstruction.precision.mono_detail.enabled": True,
                                       "reconstruction.precision.refine.enabled": True})
    assert defaults == raw


def test_variant_list_loads_and_is_validated():
    raw = _raw()
    specs = V.load_variants(raw["evaluation"])
    # the reference keeps the unaligned chunk npys for the offline seam instrument (claude_stac_slam2.txt
    # phase 4): a retention flag, not a product change
    assert specs["reference"].kind == "chain" and specs["reference"].overrides == {"certify.keep_unaligned_chunks": True}
    assert specs["no_drift_correction"].stages_off == ["certify"]
    assert specs["scale_mapanything"].kind == "not_executed" and specs["scale_mapanything"].reason
    assert specs["scale_gauge_only"].base == "reference"
    bad = copy.deepcopy(raw["evaluation"]); bad["variants"]["x"] = {"kind": "chain_continuation"}
    with pytest.raises(Exception, match="needs 'base'"):
        V.load_variants(bad)


class FakeRunner:
    """A simulated chain: writes a synthetic product into the session copy; a scale bias when the
    bend is off (so variants differ), a failure for the variant named so."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def run(self, session_dir, config, stages_off, *, replace, force, log, on_stage=None):
        self.calls.append((Path(session_dir).parent.name, dict(config["reconstruction"]["precision"]["bend"]),
                           list(stages_off), replace, force))
        name = Path(session_dir).parent.name
        if name in self.fail:
            raise RuntimeError("simulated stage crash")
        bend_off = not config["reconstruction"]["precision"]["bend"]["enabled"]
        if on_stage:
            on_stage("reconstruction", "start")
        S.make_session(Path(session_dir), scale=0.9 if bend_off else 1.0)
        return {"ok": True, "seconds_by_stage": {"reconstruction": 0.1}}


def _scene_with_refs(tmp_path):
    scene = S.make_session(tmp_path / "scene", omega_records=True)
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60); P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    P3 = S.point_on_plane_at_pixel(scene, 1, 100, 150); P4 = S.point_on_plane_at_pixel(scene, 1, 220, 200)
    planes = [{"id": "p_0001", "label": "floor", "note": "",
               "polygons": [{"frame": 10, "points": [[20, 20], [300, 20], [300, 220], [20, 220]]}]}]
    doc = S.references_doc(scene, [(P1, P2, "gauge", "d_0001"), (P3, P4, "test", "d_0002")], planes=planes)
    from evaluation.references import save_references
    save_references(scene.session_dir / "output", "reference_measures.json", doc)
    return scene, doc


def test_executor_runs_variants_isolates_failures_and_compares(tmp_path):
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "all"})
    raw = _raw()
    specs = [V.VariantSpec("reference", "chain"),
             V.VariantSpec("no_f6_bend", "chain", overrides={"reconstruction.precision.bend.enabled": False}),
             V.VariantSpec("same_as_reference", "chain", overrides={"reconstruction.precision.bend.enabled": True}),
             V.VariantSpec("broken", "chain"),
             V.VariantSpec("skipped", "not_executed", reason="no such instrument"),
             V.VariantSpec("no_cert", "chain", stages_off=["certify"])]
    runner = FakeRunner(fail={"broken"})
    res = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run", cfg, runner, raw,
                      log=lambda *a: None, plots=False)
    r = res["results"]
    assert r["reference"]["status"] == "finished" and r["reference"]["summary"]["abs_error_m"]["max"] < 1e-5
    same = {k: v for k, v in r["same_as_reference"]["summary"].items() if k != "product"}
    assert same == {k: v for k, v in r["reference"]["summary"].items() if k != "product"}   # identical numbers
    assert r["no_f6_bend"]["summary"]["scale"]["scale_factor"] == pytest.approx(1 / 0.9, abs=1e-6)
    assert r["broken"]["status"] == "failed" and "simulated stage crash" in r["broken"]["reason"] and "traceback" in r["broken"]
    assert r["skipped"]["status"] == "not_executed" and r["skipped"]["reason"] == "no such instrument"
    assert r["no_cert"]["status"] == "finished"
    assert [c[2] for c in runner.calls if c[0] == "no_cert"] == [["certify"]]
    assert all(c[3] is True and c[4] is False for c in runner.calls)              # chain variants: replace, not force
    rows = {row["variant"]: row for row in res["comparison"]}
    assert set(rows) == {s.name for s in specs}
    assert rows["broken"]["status"] == "failed" and rows["skipped"]["abs_p95_m"] is None
    assert rows["reference"]["d_abs_p95_vs_reference_m"] == 0.0
    assert (tmp_path / "run" / "synth" / "comparison.csv").exists()
    for name in ("reference", "broken", "skipped"):
        rec = json.loads((tmp_path / "run" / "synth" / name / "variant.json").read_text())
        assert rec["status"] in ("finished", "failed", "not_executed")
    assert (tmp_path / "run" / "synth" / "reference" / "coverage.json").exists()
    cov = r["reference"]["coverage"]
    assert cov["keyframes_registered_pct"] == 100.0 and cov["test_ends_reconstructed_pct"] == 100.0
    assert cov["planes"][0]["surface_with_points_pct"] == pytest.approx(100.0)


def test_continuation_writes_known_dimensions_from_omegas_record_and_runs_forced(tmp_path):
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "all"})
    specs = [V.VariantSpec("reference", "chain"),
             V.VariantSpec("gauge_only", "chain_continuation", base="reference",
                           overrides={"reconstruction.precision.gauge.instruments": ["known_dims"]}),
             V.VariantSpec("orphan", "chain_continuation", base="missing")]

    class Runner(FakeRunner):
        def run(self, session_dir, config, stages_off, *, replace, force, log, on_stage=None):
            # the continuation must find the inputs AND the base's products in the copy
            if force:
                assert (Path(session_dir) / "output" / "known_dimensions.json").exists()
                assert (Path(session_dir) / "output" / "cleaned_cloud.ply").exists()
                return {"ok": True, "seconds_by_stage": {}}
            S.make_session(Path(session_dir), omega_records=True)
            return {"ok": True, "seconds_by_stage": {}}

    runner = Runner()
    res = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run", cfg, runner, _raw(),
                      log=lambda *a: None, plots=False)
    r = res["results"]
    assert r["gauge_only"]["status"] == "finished"
    kd = json.loads(Path(r["gauge_only"]["known_dimensions"]).read_text())
    assert len(kd["dimensions"]) == 1 and kd["dimensions"][0]["role"] == "gauge"
    assert kd["dimensions"][0]["measured_m"] == pytest.approx(kd["dimensions"][0]["true_m"], abs=2e-2)
    assert r["orphan"]["status"] == "not_executed" and "did not finish" in r["orphan"]["reason"]


def test_coverage_counts_missing_frames_and_distances(tmp_path):
    scene = S.make_session(tmp_path, v_max=120)
    prod = StacProduct(scene.session_dir, "cov", log=lambda *a: None)
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60); P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    P3 = S.point_on_plane_at_pixel(scene, 0, 100, 200)       # below row 120: no points
    doc = S.references_doc(scene, [(P1, P2, "gauge", "d_0001"), (P1, P3, "test", "d_0002"), (P2, P3, "test", "d_0003")],
                           planes=[{"id": "p_0001", "label": "floor", "note": "",
                                    "polygons": [{"frame": 10, "points": [[0, 0], [320, 0], [320, 240], [0, 240]]},
                                                 {"frame": 999, "points": [[0, 0], [10, 0], [10, 10]]}]}],
                           frames_of={"d_0002": [10], "d_0003": [10]})
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    rows, summary, recs = D.evaluate(doc, prod, cfg)
    cov = COV.coverage(prod, [10, 20, 30, 40, 50], doc, recs, cell_px=8)
    assert cov["keyframes_total"] == 5 and cov["keyframes_registered"] == 4 and cov["keyframes_unregistered"] == [50]
    assert cov["test_ends_total"] == 4 and cov["test_ends_reconstructed"] == 2 and cov["test_ends_reconstructed_pct"] == 50.0
    assert cov["distances_not_reconstructed"] == ["d_0002", "d_0003"]
    pl = cov["planes"][0]
    assert pl["per_polygon"][1]["reason"] == "keyframe not registered"
    assert pl["surface_with_points_pct"] == pytest.approx(100.0 * 120 / 240, abs=2.0)


def test_colmap_model_io_round_trip_and_dense_product(tmp_path):
    scene = S.make_session(tmp_path / "s", lens=True)
    prod = StacProduct(scene.session_dir, "s", log=lambda *a: None)
    k = scene.cam.params
    cam = CIO.ColmapCamera(1, "OPENCV", S.W, S.H, np.array(k, np.float64))
    images = {}
    for i, f in enumerate(scene.frames):
        w2c = np.linalg.inv(scene.c2w[i])
        images[i + 1] = CIO.ColmapImage(i + 1, f"{f:06d}.jpg", 1, CIO.rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3])
    pts = prod.xyz[::500]
    for writer in (CIO.write_model_text, CIO.write_model_binary):
        mdir = writer(tmp_path / writer.__name__, {1: cam}, images, pts)
        m = CIO.read_model(mdir)
        assert m.n_images == 4 and m.n_points3d == len(pts) and m.cameras[1].model == "OPENCV"
        im = m.images[1]
        assert np.allclose(im.c2w(), scene.c2w[0], atol=1e-9)
        # the lens round trip of the camera model: distorted pixel → undistorted → the chain's own undistortion
        uv_d = np.array([[30.0, 40.0], [300.0, 200.0]])
        und = m.cameras[1].undistort_pixels(uv_d)
        assert np.allclose(und, prod.undistort(10, uv_d), atol=1e-6)
        dense = CIO.model_product(m, prod.xyz, "colmap")
        X = S.point_on_plane_at_pixel(scene, 0, 100, 100)
        doc = S.references_doc(scene, [(S.point_on_plane_at_pixel(scene, 0, 40, 60), S.point_on_plane_at_pixel(scene, 0, 280, 70), "gauge", "g"),
                                       (X, S.point_on_plane_at_pixel(scene, 1, 220, 200), "test", "t")])
        rows, summary, _ = D.evaluate(doc, dense, S.eval_config())
        assert summary["abs_error_m"]["max"] < 1e-4 and rows[1]["end_a_method"] == "triangulation"
    assert CIO.list_models(tmp_path / "write_model_text") == [tmp_path / "write_model_text"]
    assert CIO.frame_of_name("000051.jpg") == 51 and CIO.frame_of_name("IMG_1.jpg") is None


def test_import_of_an_external_colmap_model_is_evaluated_as_a_variant(tmp_path):
    scene, doc = _scene_with_refs(tmp_path)
    prod = StacProduct(scene.session_dir, "s", log=lambda *a: None)
    cam = CIO.ColmapCamera(1, "PINHOLE", S.W, S.H, np.array(scene.cam.params[:4], np.float64))
    images = {}
    for i, f in enumerate(scene.frames):
        w2c = np.linalg.inv(scene.c2w[i])
        images[i + 1] = CIO.ColmapImage(i + 1, f"{f:06d}.jpg", 1, CIO.rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3])
    sparse = tmp_path / "ext" / "sparse" / "0"
    CIO.write_model_binary(sparse, {1: cam}, images, prod.xyz[::1000])
    from evaluation.cloud_io import write_ply
    dense = tmp_path / "ext" / "fused.ply"
    data = np.zeros(len(prod.xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    data["x"], data["y"], data["z"] = prod.xyz[:, 0], prod.xyz[:, 1], prod.xyz[:, 2]
    write_ply(dense, data)
    cfg = S.eval_config(**{"curves.min_bin_n": 1})
    rec = V.import_colmap_variant(V.SceneSpec("synth", scene.session_dir), "realityscan", tmp_path / "ext" / "sparse",
                                  dense, tmp_path / "run", cfg, log=lambda *a: None, plots=False)
    assert rec["status"] == "finished", rec.get("reason")
    assert rec["summary"]["abs_error_m"]["max"] < 1e-4 and rec["coverage"]["n_models"] == 1
    assert (tmp_path / "run" / "synth" / "realityscan" / "dimensions.csv").exists()
    assert (tmp_path / "run" / "synth" / "comparison.csv").exists()


def test_colmap_baseline_flow_with_a_fake_binary(tmp_path):
    """The whole h-flow (features + matching once per workspace, incremental AND global mapping on
    the same database, dense per model, the manifest with every command) against a fake colmap
    that records its argv and writes the outputs the real one would."""
    from evaluation import colmap_baseline as CB
    scene, doc = _scene_with_refs(tmp_path)
    prod = StacProduct(scene.session_dir, "s", log=lambda *a: None)
    cam = CIO.ColmapCamera(1, "OPENCV", S.W, S.H, np.array(scene.cam.params, np.float64))
    images = {}
    for i, f in enumerate(scene.frames):
        w2c = np.linalg.inv(scene.c2w[i])
        images[i + 1] = CIO.ColmapImage(i + 1, f"{f:06d}.jpg", 1, CIO.rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3])
    model_src = CIO.write_model_binary(tmp_path / "model_src", {1: cam}, images, prod.xyz[::1000])
    from evaluation.cloud_io import write_ply
    data = np.zeros(len(prod.xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    data["x"], data["y"], data["z"] = prod.xyz[:, 0], prod.xyz[:, 1], prod.xyz[:, 2]
    write_ply(tmp_path / "fused_src.ply", data)
    fake = tmp_path / "colmap"
    fake.write_text(f"""#!/bin/bash
echo "$@" >> {tmp_path}/argv.log
cmd=$1; shift
case $cmd in
  -h) echo "I1014 COLMAP 4.2.0 (fake)";;
  feature_extractor|sequential_matcher|image_undistorter|patch_match_stereo) ;;
  mapper|global_mapper) out=""; while [ $# -gt 0 ]; do [ "$1" = "--output_path" ] && out=$2; shift; done
      mkdir -p "$out/0" "$out/1"; cp {model_src}/* "$out/0/"; cp {model_src}/* "$out/1/";;
  stereo_fusion) out=""; while [ $# -gt 0 ]; do [ "$1" = "--output_path" ] && out=$2; shift; done; cp {tmp_path}/fused_src.ply "$out";;
esac
""")
    fake.chmod(0o755)
    cfg = S.eval_config(**{"curves.min_bin_n": 1, "colmap.binary": str(fake), "colmap.min_model_images": 3})
    specs = [V.VariantSpec("h1", "colmap", colmap={"features": "SIFT", "matcher": "SIFT_BRUTEFORCE", "mapper": "incremental"}),
             V.VariantSpec("h2", "colmap", colmap={"features": "SIFT", "matcher": "SIFT_BRUTEFORCE", "mapper": "global"}),
             V.VariantSpec("h3", "colmap", colmap={"features": "ALIKED_N16ROT", "matcher": "ALIKED_LIGHTGLUE", "mapper": "global"})]
    res = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run", cfg, FakeRunner(), _raw(),
                      log=lambda *a: None, plots=False)
    r = res["results"]
    for n in ("h1", "h2", "h3"):
        assert r[n]["status"] == "finished", r[n].get("reason")
        assert r[n]["coverage"]["n_models"] == 2 and r[n]["summary"]["abs_error_m"]["max"] < 1e-4
    argv = (tmp_path / "argv.log").read_text().splitlines()
    assert sum(1 for a in argv if a.startswith("feature_extractor")) == 2          # SIFT once, ALIKED once
    assert sum(1 for a in argv if a.startswith("sequential_matcher")) == 2
    assert sum(1 for a in argv if a.startswith("mapper")) == 1 and sum(1 for a in argv if a.startswith("global_mapper")) == 2
    assert sum(1 for a in argv if a.startswith("patch_match_stereo")) == 6         # 2 models × 3 variants
    sift = [a for a in argv if a.startswith("feature_extractor")][0]
    assert "--ImageReader.single_camera 1" in sift and "--FeatureExtraction.type SIFT" in sift and "--ImageReader.camera_model OPENCV" in sift
    aliked = [a for a in argv if "ALIKED_LIGHTGLUE" in a][0]
    assert "aliked_n16rot" in aliked and "--SequentialMatching.loop_detection 1" in aliked
    man = json.loads((tmp_path / "run" / "synth" / "h2" / "colmap_manifest.json").read_text())
    assert man["config"]["camera_model"] == "OPENCV" and man["mapping"]["mapper"] == "global" and len(man["dense"]) == 2
    assert all("argv" in c for c in man["matching"]["commands"])
    # the workspace (images, database, matching) is shared by h1 and h2
    assert (tmp_path / "run" / "synth" / "colmap" / "sift__sift_bruteforce" / "images" / "000010.jpg").exists()
