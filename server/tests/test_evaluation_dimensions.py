"""Phase 1 of the evaluation battery: error per reference distance on a synthetic scene."""
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation import dimension_error as D
from evaluation import geometry as G
from evaluation.endpoints import reconstruct_end
from evaluation.product import DenseProduct, StacProduct
from tests import synth_evaluation as S


def _pairs(scene):
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60)
    P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    P3 = S.point_on_plane_at_pixel(scene, 1, 100, 150)
    P4 = S.point_on_plane_at_pixel(scene, 1, 220, 200)
    P5 = S.point_on_plane_at_pixel(scene, 2, 60, 30)
    return [(P1, P2, "gauge", "d_0001"), (P3, P4, "test", "d_0002"), (P1, P5, "test", "d_0003"),
            (P2, P4, "gauge", "d_0004")]


def test_exact_scene_gives_zero_error(tmp_path):
    scene = S.make_session(tmp_path)
    doc = S.references_doc(scene, _pairs(scene))
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    prod = StacProduct(scene.session_dir, "synthetic", log=lambda *a: None)
    rows, summary, recs = D.evaluate(doc, prod, cfg)
    assert summary["scale"]["scale_factor"] == pytest.approx(1.0, abs=1e-6)
    assert summary["n_test"] == 2 and summary["n_test_evaluated"] == 2
    assert summary["abs_error_m"]["max"] < 1e-5 and summary["rel_error"]["p95"] < 1e-5
    by = {r["id"]: r for r in rows}
    assert by["d_0001"]["abs_error_m"] is None            # a gauge never evaluates itself
    assert by["d_0002"]["end_a_method"] == "provenance" and by["d_0002"]["end_a_n_obs"] >= 1
    assert by["d_0002"]["end_a_distance_m"] == pytest.approx(S.Z0, abs=1e-5)
    assert by["d_0002"]["walk_from_nearest_gauge_m"] is not None and by["d_0002"]["nearest_gauge_id"] in ("d_0001", "d_0004")
    # a point seen from several keyframes: its median over observations has ~0 spread
    assert by["d_0002"]["end_a_n_obs"] >= 2 and by["d_0002"]["end_a_spread_m"] < 1e-5
    paths = D.write_phase1(tmp_path / "run", rows, summary, recs)
    assert paths["csv"].read_text().startswith("id,role,tool")
    s = json.loads(paths["summary"].read_text())
    assert s["abs_error_m"]["n"] == 2 and s["provenance"] == "tool_measured"


def test_known_scale_bias_is_recovered_by_the_gauge(tmp_path):
    scene = S.make_session(tmp_path, scale=0.8)
    doc = S.references_doc(scene, _pairs(scene))
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    prod = StacProduct(scene.session_dir, "biased", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, cfg)
    assert summary["scale"]["scale_factor"] == pytest.approx(1.25, abs=1e-6)
    assert summary["scale"]["n_gauge_used"] == 2 and summary["scale"]["weighting"].startswith("1/sigma^2")
    by = {r["id"]: r for r in rows}
    assert by["d_0002"]["measured_raw"] == pytest.approx(0.8 * by["d_0002"]["value_m"], abs=1e-5)
    assert by["d_0002"]["measured_calibrated_m"] == pytest.approx(by["d_0002"]["value_m"], abs=1e-5)
    assert summary["abs_error_m"]["max"] < 1e-5
    # camera distances and walk are reported in calibrated metres
    assert by["d_0002"]["end_a_distance_m"] == pytest.approx(S.Z0, abs=1e-5)


def test_lens_anchors_are_undistorted_before_the_lookup(tmp_path):
    scene = S.make_session(tmp_path, lens=True)
    doc = S.references_doc(scene, _pairs(scene))
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    prod = StacProduct(scene.session_dir, "lens", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, cfg)
    assert summary["abs_error_m"]["max"] < 1e-4, summary["abs_error_m"]
    assert summary["scale"]["scale_factor"] == pytest.approx(1.0, abs=1e-5)


def test_an_end_without_product_points_is_not_reconstructed_and_counted(tmp_path):
    scene = S.make_session(tmp_path, v_max=200)       # no cloud below row 200
    pairs = _pairs(scene)
    P_low = S.point_on_plane_at_pixel(scene, 0, 150, 230)     # anchored at row 230: no points
    pairs.append((pairs[0][0], P_low, "test", "d_0005"))
    doc = S.references_doc(scene, pairs, frames_of={"d_0005": [10]})
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    prod = StacProduct(scene.session_dir, "holes", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, cfg)
    by = {r["id"]: r for r in rows}
    assert by["d_0005"]["status"] == "not_reconstructed" and "end B" in by["d_0005"]["reason"]
    assert by["d_0005"]["abs_error_m"] is None
    assert summary["n_test"] == 3 and summary["n_test_not_reconstructed"] == 1 and summary["n_test_evaluated"] == 2
    assert len(rows) == 5                                      # nothing dropped
    assert summary["end_methods"]["none"] == 1 and summary["end_methods"]["provenance"] >= 8


def test_two_observations_without_product_depth_fall_back_to_triangulation_declared(tmp_path):
    scene = S.make_session(tmp_path, v_max=100)
    P_a = S.point_on_plane_at_pixel(scene, 0, 40, 150); P_b = S.point_on_plane_at_pixel(scene, 0, 200, 160)
    doc = S.references_doc(scene, [(P_a, P_b, "gauge", "d_0001")])
    prod = StacProduct(scene.session_dir, "tri", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, S.eval_config())
    assert rows[0]["status"] == "ok" and rows[0]["end_a_method"] == "triangulation"
    assert summary["end_methods"] == {"triangulation": 2}


def test_no_gauge_rebuilt_leaves_rows_uncalibrated(tmp_path):
    scene = S.make_session(tmp_path, v_max=100)
    P_a = S.point_on_plane_at_pixel(scene, 0, 40, 150); P_b = S.point_on_plane_at_pixel(scene, 0, 200, 160)
    P_c = S.point_on_plane_at_pixel(scene, 0, 40, 20); P_d = S.point_on_plane_at_pixel(scene, 0, 200, 30)
    doc = S.references_doc(scene, [(P_a, P_b, "gauge", "d_0001"), (P_c, P_d, "test", "d_0002")],
                           frames_of={"d_0001": [10]})       # one observation: nothing to triangulate
    prod = StacProduct(scene.session_dir, "nogauge", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, S.eval_config())
    assert summary["scale"]["scale_factor"] is None and summary["n_test_uncalibrated"] == 1
    assert rows[1]["status"] == "uncalibrated" and rows[1]["measured_raw"] > 0


def test_dense_product_triangulates_two_observations_and_zbuffers_one(tmp_path):
    scene = S.make_session(tmp_path)
    prod_s = StacProduct(scene.session_dir, "s", log=lambda *a: None)
    dense = DenseProduct(frames=scene.frames, c2w=prod_s.c2w, xyz=prod_s.xyz, K_of=lambda f: scene.K, name="dense")
    cfg = S.eval_config()
    X = S.point_on_plane_at_pixel(scene, 0, 100, 100)
    ch = dense.chainage()
    two = {"observations": S.anchors_of(scene, X, frames=[10, 20], round_px=False)}
    r2 = reconstruct_end(two, dense, cfg.endpoints, ch)
    assert r2.method == "triangulation" and np.linalg.norm(r2.point - X) < 1e-6 and r2.n_used == 2
    assert r2.distance_m == pytest.approx(S.Z0, abs=1e-6)
    one = {"observations": S.anchors_of(scene, X, frames=[10])}
    r1 = reconstruct_end(one, dense, cfg.endpoints, ch)
    assert r1.method == "zbuffer" and np.linalg.norm(r1.point - X) < 1e-5
    # two observations of the SAME keyframe cannot triangulate: the z-buffer answers, declared
    same = {"observations": [one["observations"][0], one["observations"][0]]}
    rs = reconstruct_end(same, dense, cfg.endpoints, ch)
    assert rs.method == "zbuffer" and "parallax" in rs.reason
    none = {"observations": [{"frame": 999, "u": 1, "v": 1}]}
    assert reconstruct_end(none, dense, cfg.endpoints, ch).method == "none"


def test_geometry_helpers():
    c2w = np.eye(4); K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]], float)
    X = G.unproject(np.array([[160.0, 120.0], [460.0, 120.0]]), np.array([2.0, 2.0]), K, c2w)
    assert np.allclose(X, [[0, 0, 2], [2, 0, 2]])
    uv, z = G.project(X, K, c2w)
    assert np.allclose(uv, [[160, 120], [460, 120]]) and np.allclose(z, 2)
    s, how = G.gauge_scale([2.0, 4.0], [1.0, 2.0], [0.0, 0.0])
    assert s == pytest.approx(0.5) and how.startswith("equal")
    s, how = G.gauge_scale([2.0, 4.0], [1.0, 2.2], [0.001, 10.0])
    assert s == pytest.approx(0.5, abs=1e-3)           # the tight gauge dominates
    assert np.allclose(G.chainage(S.true_poses(3)), [0, np.hypot(0.5, 0.05), 2 * np.hypot(0.5, 0.05)])
