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


def test_a_plane_marked_on_the_model_is_fitted_and_stored(tmp_path):
    doc = C.empty_document()
    floor = C.add_plane(doc, {"points": [[0, 0, 0], [4, 0.02, 0], [4, 0, 3], [0, -0.02, 3]],
                              "label": "floor"}, geometry_epoch=4)
    assert floor["id"] == "p1" and floor["geometry_epoch"] == 4
    assert floor["inclination_deg"] < 1.0 and floor["normal"][1] > 0.99           # a floor: level, up
    assert floor["area_m2"] == pytest.approx(12.0, rel=1e-3)
    assert 0 < floor["rms_m"] < 0.02 and floor["max_m"] >= floor["rms_m"]
    wall = C.add_plane(doc, {"points": [[0, 0, 5], [3, 0, 5], [3, 2.5, 5], [0, 2.5, 5]]}, 4)
    assert wall["id"] == "p2" and wall["inclination_deg"] == pytest.approx(90.0)
    assert wall["rms_m"] == pytest.approx(0.0, abs=1e-12) and wall["area_m2"] == pytest.approx(7.5)
    tilted = C.fit_plane([[0, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 1]])
    assert tilted["inclination_deg"] == pytest.approx(45.0)
    for bad, why in (({"points": [[0, 0, 0], [1, 0, 0]]}, "three or more"),
                     ({"points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}, "one line"),
                     ({"points": [[0, 0, 0], [1, 0, 0], [0, 1, "x"]]}, "finite"),
                     ({"points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "normal": [0, 1, 0]}, "unknown")):
        with pytest.raises(C.CalibrationError, match=why):
            C.add_plane(doc, bad, 4)
    # the label and the note are the user's; the points and what they measure are not
    assert C.update_plane(doc, "p1", {"label": "piso", "note": "sala"})["points"] == floor["points"]
    with pytest.raises(C.CalibrationError, match="not editable"):
        C.update_plane(doc, "p1", {"points": [[0, 0, 0], [1, 0, 0], [0, 0, 1]]})
    # lengths and planes share the ids' space, the summary lists the planes with their staleness
    assert C.add(doc, {"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1.0}, 4)["id"] == "c1"
    s = C.summary(doc, geometry_epoch=5)
    assert [(p["id"], p["stale"], p["n_points"]) for p in s["planes"]] == [("p1", True, 4), ("p2", True, 4)]
    C.save(tmp_path, doc)
    assert C.load(tmp_path) == doc
    bad = json.loads((tmp_path / "calibration_3d.json").read_text())
    bad["planes"][0]["inclination_deg"] = 30.0
    (tmp_path / "calibration_3d.json").write_text(json.dumps(bad))
    with pytest.raises(C.CalibrationError, match="not what its points measure"):
        C.load(tmp_path)
    C.remove_plane(doc, "p2")
    assert [p["id"] for p in doc["planes"]] == ["p1"]
    # a document stored before the planes existed is still read
    old = {"schema_version": 1, "provenance": "human_validated", "frame": "display", "measurements": []}
    assert C.validate_document(old) == [] and C.summary(old, 4)["planes"] == []


def test_the_plane_routes(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    calibration_api.configure(resolve_ctx=lambda sid, scan=None: SimpleNamespace(session_dir=tmp_path, output_dir=out),
                              audit_log=lambda *a, **k: None)
    monkeypatch.setattr(calibration_api, "_epoch", lambda o: 4)
    app = FastAPI()
    app.include_router(calibration_api.router)
    c = TestClient(app)
    r = c.post("/api/calibration/pccr/planes", json={"points": [[0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2]],
                                                     "label": "floor"})
    assert r.status_code == 200 and r.json()["plane"]["id"] == "p1"
    assert r.json()["summary"]["planes"][0]["inclination_deg"] == pytest.approx(0.0, abs=1e-9)
    assert c.post("/api/calibration/pccr/planes", json={"points": [[0, 0, 0]]}).status_code == 400
    assert c.patch("/api/calibration/pccr/planes/p1", json={"note": "x"}).json()["plane"]["note"] == "x"
    assert c.patch("/api/calibration/pccr/planes/p9", json={"note": "x"}).status_code == 404
    assert c.delete("/api/calibration/pccr/planes/p1").json()["document"]["planes"] == []
    assert c.delete("/api/calibration/pccr/planes/p1").status_code == 404


def test_the_state_of_the_scan_and_the_verification():
    doc = C.empty_document()
    assert C.summary(doc, 4)["state"] == "uncalibrated" and C.summary(doc, 4)["verification"] is None
    C.add(doc, {"a": [0, 0, 0], "b": [2, 0, 0], "value_m": 2.2}, 4)                   # factor 1.10
    s = C.summary(doc, 4)
    assert s["state"] == "calibrated" and s["verification"] is None
    assert s["rows"][0]["elevation_deg"] == pytest.approx(0.0)                        # horizontal
    C.add(doc, {"a": [0, 0, 0], "b": [0, 1, 0], "value_m": 1.13, "role": "test", "sigma_m": 0.05}, 4)
    C.add(doc, {"a": [0, 0, 0], "b": [0, 0, 3], "value_m": 3.0, "role": "test", "sigma_m": 0.01}, 4)
    C.add(doc, {"a": [0, 0, 0], "b": [1, 1, 0], "value_m": 1.6, "role": "test"}, 4)
    s = C.summary(doc, 4)
    rows = {r["id"]: r for r in s["rows"]}
    assert s["state"] == "verified"
    assert rows["c2"]["elevation_deg"] == pytest.approx(90.0)                         # vertical
    assert rows["c4"]["elevation_deg"] == pytest.approx(45.0)
    assert rows["c2"]["residual_m"] == pytest.approx(1.10 - 1.13) and rows["c2"]["within_sigma"] is True
    assert rows["c3"]["residual_m"] == pytest.approx(3.30 - 3.0) and rows["c3"]["within_sigma"] is False
    assert rows["c4"]["within_sigma"] is None                                         # no sigma stated
    v = s["verification"]
    assert v["n"] == 3 and v["worst_id"] == "c3" and v["worst_difference_m"] == pytest.approx(0.30)
    assert v["max_abs_m"] == pytest.approx(0.30) and v["max_rel"] == pytest.approx(0.10)
    assert v["n_with_sigma"] == 2 and v["all_within_sigma"] is False
    # on another geometry nothing calibrates any more: the scan is uncalibrated again
    assert C.summary(doc, 5)["state"] == "uncalibrated"


def test_the_type_of_calibration_global_or_per_axis():
    doc = C.empty_document()
    C.add(doc, {"a": [0, 0, 0], "b": [2, 0.1, 0], "value_m": 2.2}, 4)                  # along X
    s = C.summary(doc, 4)
    assert (s["mode"], s["mode_used"]) == ("global", "global") and s["rows"][0]["axis"] == "X"
    assert s["axes"]["ready"] is False and s["axes"]["missing"] == ["Y", "Z"]
    # per axis is REFUSED with one or two measures: the calibration is global, the others verify
    with pytest.raises(C.CalibrationError, match="missing: Y, Z"):
        C.set_mode(doc, "axes", 4)
    C.add(doc, {"a": [0, 0, 0], "b": [0, 3, 0], "value_m": 3.6}, 4)                    # along Y
    with pytest.raises(C.CalibrationError, match="missing: Z"):
        C.set_mode(doc, "axes", 4)
    assert doc.get("mode", "global") == "global"
    C.add(doc, {"a": [0, 0, 0], "b": [0, 0, 5], "value_m": 4.5}, 4)                    # along Z
    C.add(doc, {"a": [0, 0, 0], "b": [1, 1, 1], "value_m": 1.9, "role": "test"}, 4)
    s = C.summary(doc, 4)
    assert s["axes"]["ready"] and s["axes"]["covered"] == ["X", "Y", "Z"]
    # X: real² = (sx·2)² + (sy·0.1)² … solved jointly; Y and Z alone give 1.2 and 0.9
    assert s["axes"]["factors"]["y"] == pytest.approx(1.2) and s["axes"]["factors"]["z"] == pytest.approx(0.9)
    assert s["axes"]["factors"]["x"] == pytest.approx(((2.2 ** 2 - (1.2 * 0.1) ** 2) ** 0.5) / 2)
    assert s["mode_used"] == "global"                       # ready, but global until it is chosen
    assert C.set_mode(doc, "axes", 4) == "axes"
    s = C.summary(doc, 4)
    assert s["mode_used"] == "axes"
    rows = {r["id"]: r for r in s["rows"]}
    for k in ("c1", "c2", "c3"):                            # the three gauges are met exactly
        assert rows[k]["residual_m"] == pytest.approx(0.0, abs=1e-9)
    f = s["axes"]["factors"]
    assert rows["c4"]["calibrated_m"] == pytest.approx((f["x"] ** 2 + f["y"] ** 2 + f["z"] ** 2) ** 0.5)
    assert s["verification"]["n"] == 1 and s["state"] == "verified"
    # one of the three goes: per axis is no longer possible — global is what is used, and said
    C.remove(doc, "c3")
    s = C.summary(doc, 4)
    assert (s["mode"], s["mode_used"]) == ("axes", "global") and "missing: Z" in s["axes"]["reason"]
    with pytest.raises(C.CalibrationError, match="mode must be"):
        C.set_mode(doc, "xyz", 4)
    assert C.validate_document({**doc, "mode": "xyz"}) != []


def test_the_mode_route(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    calibration_api.configure(resolve_ctx=lambda sid, scan=None: SimpleNamespace(session_dir=tmp_path, output_dir=out),
                              audit_log=lambda *a, **k: None)
    monkeypatch.setattr(calibration_api, "_epoch", lambda o: 4)
    app = FastAPI()
    app.include_router(calibration_api.router)
    c = TestClient(app)
    c.post("/api/calibration/pccr/measurements", json={"a": [0, 0, 0], "b": [1, 0, 0], "value_m": 1.1})
    r = c.put("/api/calibration/pccr/mode", json={"mode": "axes"})
    assert r.status_code == 400 and "missing: Y, Z" in r.json()["detail"]["error"]
    c.post("/api/calibration/pccr/measurements", json={"a": [0, 0, 0], "b": [0, 1, 0], "value_m": 1.0})
    c.post("/api/calibration/pccr/measurements", json={"a": [0, 0, 0], "b": [0, 0, 1], "value_m": 0.9})
    r = c.put("/api/calibration/pccr/mode", json={"mode": "axes"})
    assert r.status_code == 200 and r.json()["summary"]["mode_used"] == "axes"
    assert c.put("/api/calibration/pccr/mode", json={"mode": "global"}).json()["summary"]["mode_used"] == "global"
