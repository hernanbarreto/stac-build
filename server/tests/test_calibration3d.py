"""Calibration measurements taken on the 3D model (USER 2026-11-13: "para calibrar debe hacerse
tomando mediciones sobre el modelo 3d no sobre las imagenes … se debe poder seleccionar que una
medicion es de calibracion y deben quedar almacenadas")."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import calibration3d as C
import calibration_api


def test_the_models_length_is_computed_from_the_points_never_taken_from_the_client():
    doc = C.empty_document()
    m = C.add(doc, {"a": [0, 0, 0], "b": [3, 4, 0], "value_m": 5.5}, geometry_epoch=4)
    assert m["id"] == "c1" and m["measured_m"] == 5.0 and m["role"] == "gauge"
    assert m["geometry_epoch"] == 4 and m["sigma_m"] == 0.0 and m["tool"] == "tape"
    with pytest.raises(C.CalibrationError, match="unknown field"):
        C.add(doc, {"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1.0, "measured_m": 9.0}, 4)
    for bad, why in (({"a": [0, 0, 0], "b": [0, 0, 0], "value_m": 1.0}, "the same"),
                     ({"a": [0, 0], "b": [1, 0, 0], "value_m": 1.0}, "three coordinates"),
                     ({"a": [0, 0, 0], "b": [1, 0, float("nan")], "value_m": 1.0}, "finite"),
                     ({"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 0}, "> 0"),
                     ({"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1, "sigma_m": -1}, ">= 0"),
                     ({"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1, "role": "x"}, "role"),
                     ({"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1, "tool": "eye"}, "tool"),
                     ({"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1, "tool": "other_scan"},
                      "reference_scan")):
        with pytest.raises(C.CalibrationError, match=why):
            C.add(doc, bad, 4)
    assert len(doc["measurements"]) == 1


def test_the_factor_is_the_batterys_estimator_and_stale_measurements_are_not_used():
    doc = C.empty_document()
    C.add(doc, {"a": [0, 0, 0], "b": [2, 0, 0], "value_m": 2.2}, 4)                 # ratio 1.10
    C.add(doc, {"a": [0, 0, 0], "b": [0, 4, 0], "value_m": 4.2}, 4)                 # ratio 1.05
    C.add(doc, {"a": [0, 0, 0], "b": [0, 0, 1], "value_m": 1.07, "role": "test"}, 4)
    C.add(doc, {"a": [0, 0, 0], "b": [9, 0, 0], "value_m": 1.0}, 3)                 # another epoch
    s = C.summary(doc, geometry_epoch=4)
    want = (2 * 2.2 + 4 * 4.2) / (2 * 2 + 4 * 4)        # Σ m·t / Σ m², equal weights
    assert s["scale_factor"] == pytest.approx(want, abs=1e-12)
    assert (s["n"], s["n_gauge"], s["n_test"], s["n_stale"]) == (4, 2, 1, 1)
    assert s["weighting"].startswith("equal")
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["c4"]["stale"] and rows["c4"]["calibrated_m"] is None
    assert rows["c3"]["residual_m"] == pytest.approx(want * 1.0 - 1.07, abs=1e-12)
    assert s["test_abs_error_m"]["n"] == 1
    assert s["test_abs_error_m"]["max"] == pytest.approx(abs(want - 1.07), abs=1e-12)
    assert s["gauge_ratio_min"] == pytest.approx(1.05) and s["gauge_ratio_max"] == pytest.approx(1.10)
    # sigmas weigh the fit: the precise one wins
    C.update(doc, "c1", {"sigma_m": 0.001})
    C.update(doc, "c2", {"sigma_m": 0.1})
    assert abs(C.summary(doc, 4)["scale_factor"] - 1.10) < 1e-3
    # no gauge on this epoch: nothing calibrates, said
    s5 = C.summary(doc, geometry_epoch=5)
    assert s5["scale_factor"] is None and s5["n_stale"] == 4 and "nothing calibrates" in s5["reason"]


def test_what_was_measured_never_changes_only_what_the_user_states():
    doc = C.empty_document()
    C.add(doc, {"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1.0}, 2)
    m = C.update(doc, "c1", {"value_m": 0.98, "role": "test", "tool": "other_scan",
                             "reference_scan": "2026-08-24/default", "note": "puerta"})
    assert (m["value_m"], m["role"], m["tool"], m["reference_scan"]) == (
        0.98, "test", "other_scan", "2026-08-24/default")
    assert m["a"] == [0.0, 0.0, 0.0] and m["measured_m"] == 1.0 and m["geometry_epoch"] == 2
    with pytest.raises(C.CalibrationError, match="not editable"):
        C.update(doc, "c1", {"a": [5, 5, 5]})
    with pytest.raises(KeyError):
        C.update(doc, "c9", {"note": "x"})
    C.remove(doc, "c1")
    assert doc["measurements"] == [] and C.next_id(doc) == "c1"
    with pytest.raises(KeyError):
        C.remove(doc, "c1")


def test_the_document_is_stored_and_a_tampered_one_is_refused(tmp_path):
    assert C.load(tmp_path) == C.empty_document()
    doc = C.empty_document()
    C.add(doc, {"a": [0, 0, 0], "b": [0, 0, 2], "value_m": 2.1, "note": "columna"}, 7)
    p = C.save(tmp_path, doc)
    assert p.name == "calibration_3d.json" and C.load(tmp_path) == doc
    bad = json.loads(p.read_text())
    bad["measurements"][0]["measured_m"] = 3.0            # not the length between its points
    p.write_text(json.dumps(bad))
    with pytest.raises(C.CalibrationError, match="measured_m is not the length"):
        C.load(tmp_path)


def test_the_routes(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    seen = []
    calibration_api.configure(
        resolve_ctx=lambda sid, scan=None: (seen.append((sid, scan)),
                                            SimpleNamespace(session_dir=tmp_path, output_dir=out))[1],
        audit_log=lambda *a, **k: None)
    monkeypatch.setattr(calibration_api, "_epoch", lambda o: 4)
    app = FastAPI()
    app.include_router(calibration_api.router)
    c = TestClient(app)
    r = c.get("/api/calibration/pccr", params={"scan_key": "2026-08-31/default"}).json()
    assert r["document"]["measurements"] == [] and r["summary"]["scale_factor"] is None
    r = c.post("/api/calibration/pccr/measurements",
               json={"scan_key": "2026-08-31/default", "a": [0, 0, 0], "b": [0.9, 0, 0],
                     "value_m": 0.87, "tool": "other_scan", "reference_scan": "2026-08-24/default"})
    assert r.status_code == 200 and r.json()["measurement"]["id"] == "c1"
    assert r.json()["summary"]["scale_factor"] == pytest.approx(0.87 / 0.9)
    assert c.post("/api/calibration/pccr/measurements", json={"a": [0, 0, 0], "b": [0, 0, 0],
                                                              "value_m": 1}).status_code == 400
    r = c.patch("/api/calibration/pccr/measurements/c1", json={"role": "test"})
    assert r.status_code == 200 and r.json()["summary"]["scale_factor"] is None
    assert c.patch("/api/calibration/pccr/measurements/zz", json={"role": "test"}).status_code == 404
    assert c.delete("/api/calibration/pccr/measurements/c1").json()["document"]["measurements"] == []
    assert c.delete("/api/calibration/pccr/measurements/c1").status_code == 404
    assert ("pccr", "2026-08-31/default") in seen and (out / "calibration_3d.json").exists()
