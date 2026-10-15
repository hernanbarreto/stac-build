"""Phase 2 of the evaluation battery: error curves, plane noise, resolution, growth models,
confidence distance — on synthetic data with known answers."""
import json

import numpy as np
import pytest

from evaluation import curves as C
from evaluation import dimension_error as D
from evaluation.product import StacProduct
from tests import synth_evaluation as S


def test_binned_stats_and_no_data_bins():
    x = [1, 1.5, 3, 4, 4.5, 7, 35]
    ea = [0.01, 0.02, 0.03, 0.05, 0.04, 0.10, 0.5]
    rows = C.binned_stats(x, ea, [e / 2 for e in ea], [0, 2, 5, 10, 20, 30], min_n=2, ids=list("abcdefg"))
    by = {r["bin"]: r for r in rows}
    assert by["0-2"]["n"] == 2 and by["0-2"]["abs_median_m"] == pytest.approx(0.015) and by["0-2"]["ids"] == "a;b"
    assert by["2-5"]["n"] == 3 and by["2-5"]["abs_max_m"] == 0.05 and by["2-5"]["rel_median"] == pytest.approx(0.02)
    assert by["5-10"]["n"] == 1 and by["5-10"]["status"] == C.NO_DATA and by["5-10"]["abs_p95_m"] is None
    assert by["10-20"]["n"] == 0 and by["20-30"]["n"] == 0
    assert by["outside"]["n"] == 1 and by["outside"]["ids"] == "g"
    # the last edge is inclusive, the first bin takes x == lo
    idx = C.bin_index(np.array([0.0, 30.0, 30.1, -0.1]), [0, 2, 5, 10, 20, 30])
    assert idx.tolist() == [0, 4, -1, -1]


def test_growth_models_recover_known_coefficients():
    rng = np.random.default_rng(0)
    z = np.linspace(0.5, 25, 60)
    lin = 0.01 + 0.004 * z + rng.normal(0, 1e-5, z.size)
    quad = 0.01 + 0.0005 * z * z + rng.normal(0, 1e-5, z.size)
    g = C.fit_growth(z, lin)
    assert g["status"] == "ok" and g["linear"]["a"] == pytest.approx(0.01, abs=1e-4)
    assert g["linear"]["b"] == pytest.approx(0.004, abs=1e-5)
    assert g["linear"]["aic"] < g["quadratic"]["aic"]            # reported, not chosen
    g = C.fit_growth(z, quad)
    assert g["quadratic"]["a"] == pytest.approx(0.01, abs=1e-4) and g["quadratic"]["c"] == pytest.approx(0.0005, abs=1e-6)
    assert g["quadratic"]["aic"] < g["linear"]["aic"]
    assert C.fit_growth([1, 2], [0.1, 0.2])["status"] == C.NO_DATA


def test_confidence_distance_stops_at_no_data_or_exceeded():
    def bin_(lo, hi, p95, n=5, status="ok"):
        return {"bin": f"{lo}-{hi}", "lo_m": lo, "hi_m": hi, "n": n, "abs_p95_m": p95, "status": status}
    curve = [bin_(0, 2, 0.01), bin_(2, 5, 0.03), bin_(5, 10, 0.08), bin_(10, 20, None, 1, C.NO_DATA), bin_(20, 30, 0.2)]
    out = {r["tolerance_m"]: r for r in C.confidence_distance(curve, [0.02, 0.05, 0.10, 0.005])}
    assert out[0.02] == {"tolerance_m": 0.02, "distance_m": 2, "status": "exceeded", "stopped_at_bin": "2-5", "note": ""}
    assert out[0.05]["distance_m"] == 5 and out[0.05]["status"] == "exceeded"
    assert out[0.10]["distance_m"] == 10 and out[0.10]["status"] == C.NO_DATA and out[0.10]["stopped_at_bin"] == "10-20"
    assert out[0.005]["distance_m"] is None and out[0.005]["note"] == "no bin below the tolerance"
    all_ok = [bin_(0, 2, 0.01), bin_(2, 5, 0.01)]
    assert C.confidence_distance(all_ok, [0.02])[0] == {"tolerance_m": 0.02, "distance_m": 5, "status": "all bins pass",
                                                       "stopped_at_bin": None, "note": ""}


def _plane_doc(scene, noise_doc_pairs):
    poly1 = [{"frame": 10, "points": [[20, 20], [300, 20], [300, 220], [20, 220]]}]
    poly2 = [{"frame": 10, "points": [[20, 20], [300, 20], [300, 220], [20, 220]]},
             {"frame": 20, "points": [[40, 40], [280, 40], [280, 200]]}]
    planes = [{"id": "p_0001", "label": "floor", "note": "", "polygons": poly1},
              {"id": "p_0002", "label": "floor", "note": "two keyframes", "polygons": poly2}]
    return S.references_doc(scene, noise_doc_pairs, planes=planes)


def _pairs(scene):
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60); P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    P3 = S.point_on_plane_at_pixel(scene, 1, 100, 150); P4 = S.point_on_plane_at_pixel(scene, 1, 220, 200)
    P5 = S.point_on_plane_at_pixel(scene, 2, 60, 30)
    return [(P1, P2, "gauge", "d_0001"), (P3, P4, "test", "d_0002"), (P1, P5, "test", "d_0003")]


def test_plane_noise_resolution_and_phase2_outputs(tmp_path):
    scene = S.make_session(tmp_path / "clean")
    doc = _plane_doc(scene, _pairs(scene))
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1})
    prod = StacProduct(scene.session_dir, "clean", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, cfg)
    res = C.run_phase2(doc, rows, summary, prod, cfg, tmp_path / "run")
    noise = {(r["plane"], r["bin"]): r for r in res["plane_noise"]}
    # a perfect plane: zero residual in the bin that holds it (Z0 = 4 m from every camera → bin 2-5)
    assert noise[("p_0001", "2-5")]["n"] > 1000 and noise[("p_0001", "2-5")]["rms_m"] < 1e-5
    assert noise[("p_0001", "0-2")]["status"] == C.NO_DATA
    assert noise[("p_0002", "2-5")]["n"] > noise[("p_0001", "2-5")]["n"]      # two keyframes pooled
    assert noise[("all", "2-5")]["n"] == noise[("p_0001", "2-5")]["n"] + noise[("p_0002", "2-5")]["n"]
    r = {(x["plane"], x["bin"]): x for x in res["resolution"]}[("p_0001", "2-5")]
    footprint_mm = S.Z0 / S.FX * 1000.0                       # one pixel at 4 m = 13.3 mm
    assert r["mean_spacing_mm"] == pytest.approx(footprint_mm, rel=0.02)
    assert r["density_pts_per_m2"] == pytest.approx(1.0 / (S.Z0 / S.FX) ** 2, rel=0.05)
    r2 = {(x["plane"], x["bin"]): x for x in res["resolution"]}[("p_0002", "2-5")]
    assert r2["mean_spacing_mm"] < footprint_mm                # two interleaved grids are denser
    assert res["growth_models"]["plane_abs_residual_vs_camera_distance"]["n"] > 0
    for k in ("curve_distance", "curve_walk", "plane_noise", "resolution", "growth_models", "confidence_distance",
              "curve_distance_png", "curve_walk_png", "plane_noise_png"):
        assert res["paths"][k].exists(), k
    assert res["paths"]["curve_distance"].read_text().startswith("bin,lo_m,hi_m,n")
    conf = json.loads(res["paths"]["confidence_distance"].read_text())
    assert conf["by_camera_distance"][0]["tolerance_m"] == cfg.curves.tolerances_m[0]
    # the curve holds the two test distances at ~4 m
    byb = {b["bin"]: b for b in res["curves"]["distance"]}
    assert byb["2-5"]["n"] == 2 and byb["2-5"]["abs_p95_m"] < 1e-5
    # with 5 mm of Gaussian noise along the normal, the RMS reads 5 mm
    noisy = S.make_session(tmp_path / "noisy", noise_m=0.005)
    doc2 = _plane_doc(noisy, _pairs(noisy))
    prod2 = StacProduct(noisy.session_dir, "noisy", log=lambda *a: None)
    rows2, summary2, _ = D.evaluate(doc2, prod2, cfg)
    res2 = C.run_phase2(doc2, rows2, summary2, prod2, cfg, tmp_path / "run2", plots=False)
    n2 = {(r["plane"], r["bin"]): r for r in res2["plane_noise"]}[("p_0001", "2-5")]
    assert n2["rms_m"] == pytest.approx(0.005, rel=0.05)


def test_uncalibrated_product_writes_no_plane_noise(tmp_path):
    scene = S.make_session(tmp_path, v_max=100)
    P_a = S.point_on_plane_at_pixel(scene, 0, 40, 150); P_b = S.point_on_plane_at_pixel(scene, 0, 200, 160)
    doc = _plane_doc(scene, [(P_a, P_b, "gauge", "d_0001")])
    doc["distances"][0]["ends"] = [{"observations": [doc["distances"][0]["ends"][0]["observations"][0]]},
                                   {"observations": [doc["distances"][0]["ends"][1]["observations"][0]]}]
    prod = StacProduct(scene.session_dir, "u", log=lambda *a: None)
    rows, summary, _ = D.evaluate(doc, prod, S.eval_config())
    res = C.run_phase2(doc, rows, summary, prod, S.eval_config(), tmp_path / "run", plots=False)
    assert res["plane_noise"] == [] and summary["scale"]["scale_factor"] is None
