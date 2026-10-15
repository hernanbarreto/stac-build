"""Phase 5 of the evaluation battery: errors grouped by confidence / status / votes, Spearman,
the reliability table — a scene whose confidence correlates with the error and one where it
does not."""
import numpy as np
import pytest

from evaluation import confidence as CF
from evaluation import dimension_error as D
from evaluation.cloud_io import read_ply_memmap, write_ply
from evaluation.product import StacProduct
from tests import synth_evaluation as S


def _scene(tmp_path, correlated: bool):
    scene = S.make_session(tmp_path, witness_fields=True, seed=1)
    ply = scene.session_dir / "output" / "cleaned_cloud.ply"
    data = np.array(read_ply_memmap(ply))
    rng = np.random.default_rng(2)
    # noise along z whose size depends (or not) on the confidence; 'verified' points get less noise
    noise = rng.normal(0, 0.02, len(data)).astype(np.float32)
    conf = data["confidence"]
    if correlated:
        data["z"] += noise * (1.0 - conf)                 # high confidence → small error
    else:
        data["z"] += noise
    ver = data["status"] == 1
    data["z"][ver] *= 1.0
    data["z"][~ver] += rng.normal(0, 0.01, int((~ver).sum())).astype(np.float32) if correlated else 0.0
    write_ply(ply, data)
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60); P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    P3 = S.point_on_plane_at_pixel(scene, 1, 100, 150); P4 = S.point_on_plane_at_pixel(scene, 1, 220, 200)
    planes = [{"id": "p_0001", "label": "floor", "note": "",
               "polygons": [{"frame": 10, "points": [[10, 10], [310, 10], [310, 230], [10, 230]]}]}]
    doc = S.references_doc(scene, [(P1, P2, "gauge", "d_0001"), (P3, P4, "test", "d_0002")], planes=planes)
    return scene, doc


@pytest.mark.parametrize("correlated", [True, False])
def test_confidence_groups_spearman_and_reliability(tmp_path, correlated):
    scene, doc = _scene(tmp_path, correlated)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 2, "curves.min_bin_n": 1})
    prod = StacProduct(scene.session_dir, "c", log=lambda *a: None)
    rows, summary, recs = D.evaluate(doc, prod, cfg)
    rep = CF.run_phase5(doc, rows, recs, summary, prod, cfg.curves, cfg.confidence, cfg.endpoints.depth_window_px,
                        tmp_path / "run")
    assert rep["status"] == "ok" and set(rep["planes"]["fields_present"]) == set(CF.GROUP_FIELDS)
    sp = rep["planes"]["spearman_confidence_vs_error"]
    assert sp["status"] == "ok" and sp["n"] > 1000
    groups = {(g["field"], g["group"]): g for g in rep["groups"] if g["source"] == "planes"}
    conf_groups = [g for g in rep["groups"] if g["source"] == "planes" and g["field"] == "confidence"]
    assert len(conf_groups) == cfg.confidence.bins and all(g["status"] == "ok" for g in conf_groups)
    rel = rep["planes"]["reliability"]
    if correlated:
        assert sp["rho"] < -0.3                              # higher confidence, lower error
        assert conf_groups[0]["median_m"] > conf_groups[-1]["median_m"]
        assert rel["verified_median_lower"] is True
    else:
        assert abs(sp["rho"]) < 0.05
        assert rel["verified_median_lower"] in (True, False)   # measured, not assumed
    assert ("status", "verified") in groups and ("mv_votes", 0) in groups
    assert (tmp_path / "run" / "confidence_groups.csv").exists() and (tmp_path / "run" / "reliability.csv").exists()
    ends = rep["test_ends"]
    assert ends["n_points"] > 0 and ends["status"] == "ok"


def test_absent_fields_are_reported_not_invented(tmp_path):
    scene = S.make_session(tmp_path)        # no witness fields in this cloud
    P1 = S.point_on_plane_at_pixel(scene, 0, 40, 60); P2 = S.point_on_plane_at_pixel(scene, 0, 280, 70)
    doc = S.references_doc(scene, [(P1, P2, "gauge", "d_0001"), (P1, P2, "test", "d_0002")],
                           planes=[{"id": "p_0001", "label": "floor", "note": "",
                                    "polygons": [{"frame": 10, "points": [[10, 10], [310, 10], [310, 230]]}]}])
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1})
    prod = StacProduct(scene.session_dir, "c", log=lambda *a: None)
    rows, summary, recs = D.evaluate(doc, prod, cfg)
    rep = CF.run_phase5(doc, rows, recs, summary, prod, cfg.curves, cfg.confidence, 1, tmp_path / "run")
    assert rep["planes"]["fields_present"] == ["confidence"]
    assert set(rep["planes"]["fields_absent"]) == {"status", "mv_votes", "mask_votes"}
    assert "reliability" not in rep["planes"]
