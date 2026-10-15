"""Phase 0 of the evaluation battery: reference measures (schema, round trip, roles, API)."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evaluation import references as R
from evaluation.config import EvaluationConfigError, load_evaluation_config

SERVER_DIR = Path(__file__).resolve().parent.parent


def _doc():
    d = R.empty_document()
    d["image_size"] = [464, 832]
    d["distances"] = [
        R.new_distance("d_0001", 2.345, 0.005, "gauge", "tape",
                       [R.observation(51, 100.5, 200.0, "000051.jpg"), R.observation(120, 90.0, 210.0)],
                       [R.observation(51, 300.0, 205.0, "000051.jpg")], note="desk edge"),
        R.new_distance("d_0002", 1.10, 0.002, "test", "laser",
                       [R.observation(200, 10.0, 20.0)], [R.observation(200, 400.0, 20.0)]),
    ]
    d["planes"] = [R.new_plane("p_0001", "floor", [{"frame": 51, "points": [[0, 700], [460, 700], [460, 830], [0, 830]]}])]
    return d


def test_schema_accepts_a_valid_document():
    assert R.validate_document(_doc()) == []
    assert R.counts(_doc()) == {"distances": 2, "gauge": 1, "test": 1, "planes": 1}


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["distances"][0].pop("role"), "role"),
    (lambda d: d["distances"][0].update(tool="ruler"), "tool"),
    (lambda d: d["distances"][0]["ends"].pop(), "ends"),
    (lambda d: d["distances"][0]["ends"][0].update(observations=[]), "observations"),
    (lambda d: d["distances"][0].update(value_m=0), "value_m"),
    (lambda d: d["distances"][0].update(sigma_m=-1), "sigma_m"),
    (lambda d: d["planes"][0]["polygons"][0].update(points=[[0, 0], [1, 1]]), "points"),
    (lambda d: d["distances"][1].update(id="d_0001"), "already used"),
    (lambda d: d["distances"][0]["ends"][0]["observations"][0].update(u=9999), "outside"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.update(extra=1), "extra"),
])
def test_schema_rejects_invalid_documents(mutate, fragment):
    d = _doc()
    mutate(d)
    problems = R.validate_document(d)
    assert problems and any(fragment in p for p in problems), problems


def test_round_trip_read_write(tmp_path):
    out = tmp_path / "output"
    p = R.save_references(out, "reference_measures.json", _doc())
    assert p == out / "reference_measures.json"
    back = R.load_references(out, "reference_measures.json")
    assert back == _doc()
    # written sorted, no clock, no absolute path
    text = p.read_text()
    assert str(tmp_path) not in text
    assert json.loads(text) == back


def test_missing_file_is_an_empty_document_and_a_broken_one_raises(tmp_path):
    out = tmp_path / "output"
    assert R.load_references(out, "reference_measures.json") == R.empty_document()
    out.mkdir()
    (out / "reference_measures.json").write_text("{not json")
    with pytest.raises(R.ReferencesError, match="unreadable"):
        R.load_references(out, "reference_measures.json")
    (out / "reference_measures.json").write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(R.ReferencesError, match="invalid"):
        R.load_references(out, "reference_measures.json")


def test_evaluation_refuses_a_document_without_a_gauge_and_a_test(tmp_path):
    out = tmp_path / "output"
    d = _doc()
    d["distances"] = [d["distances"][0]]          # gauge only
    R.save_references(out, "reference_measures.json", d)
    with pytest.raises(R.ReferencesError, match="no test distance"):
        R.load_references(out, "reference_measures.json", require_roles=True)
    d["distances"][0]["role"] = "test"             # test only
    R.save_references(out, "reference_measures.json", d)
    with pytest.raises(R.ReferencesError, match="no gauge distance"):
        R.load_references(out, "reference_measures.json", require_roles=True)
    assert R.role_warnings(R.empty_document()) == ["no gauge distance: nothing calibrates the scale",
                                                  "no test distance: nothing measures the error"]
    R.check_roles(_doc())                          # both → fine


def test_config_section_loads_and_a_missing_key_is_named():
    raw = yaml.safe_load((SERVER_DIR / "config.yaml").read_text())
    cfg = load_evaluation_config(raw)
    assert cfg.references.file == "reference_measures.json"
    assert cfg.curves.distance_bins_m[0] == 0.0 and cfg.curves.min_bin_n >= 1
    assert "default" in cfg.cost.gpu_price_per_hour
    broken = copy.deepcopy(raw)
    del broken["evaluation"]["curves"]["tolerances_m"]
    with pytest.raises(EvaluationConfigError, match="evaluation.curves.tolerances_m"):
        load_evaluation_config(broken)
    broken = copy.deepcopy(raw)
    broken["evaluation"]["curves"]["distance_bins_m"] = [5, 2]
    with pytest.raises(EvaluationConfigError, match="increasing"):
        load_evaluation_config(broken)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from evaluation import api as A
    raw = yaml.safe_load((SERVER_DIR / "config.yaml").read_text())
    raw["evaluation"]["runs_dir"] = str(tmp_path / "runs")
    cfg = load_evaluation_config(raw)
    scan = tmp_path / "projects" / "s1" / "scans" / "2026-01-01" / "src_default"
    (scan / "output").mkdir(parents=True)
    audits = []

    def resolve(session_id, scan_key=None):
        d = scan if session_id == "s1" else tmp_path / "projects" / session_id / "scans" / "x" / "src_default"
        return SimpleNamespace(session_dir=d, output_dir=d / "output", frames_dir=d / "frames")

    A.configure(resolve_ctx=resolve, audit_log=lambda *a, **k: audits.append((a, k)), cfg=cfg)
    app = FastAPI()
    app.include_router(A.router)
    c = TestClient(app)
    c.scan = scan
    c.audits = audits
    c.runs = tmp_path / "runs"
    return c


def test_api_references_get_post(client):
    r = client.get("/api/evaluation/references/s1")
    assert r.status_code == 200 and r.json()["document"] == R.empty_document()
    assert r.json()["warnings"]
    r = client.post("/api/evaluation/references/s1", json={"scan_key": None, "document": _doc()})
    assert r.status_code == 200, r.text
    assert r.json()["counts"] == {"distances": 2, "gauge": 1, "test": 1, "planes": 1}
    assert r.json()["warnings"] == [] and client.audits
    assert json.loads((client.scan / "output" / "reference_measures.json").read_text()) == _doc()
    assert client.get("/api/evaluation/references/s1").json()["document"] == _doc()
    bad = _doc(); bad["distances"][0]["role"] = "calib"
    r = client.post("/api/evaluation/references/s1", json={"document": bad})
    assert r.status_code == 400 and "role" in json.dumps(r.json()["detail"]["problems"])
    assert client.post("/api/evaluation/references/nope", json={"document": _doc()}).status_code == 404


def test_api_runs_listing_and_files(client):
    run = client.runs / "20261014_120000"
    (run / "pccr" / "reference").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"created": "2026-10-14T12:00:00", "scenes": ["pccr"],
                                                   "variants": ["reference"], "status": "finished"}))
    (run / "pccr" / "reference" / "dimensions.csv").write_text("id,error_m\nd_0002,0.01\n")
    (run / "pccr" / "reference" / "curve_distance.png").write_bytes(b"\x89PNG\r\n")
    (run / "pareto.json").write_text("{}")
    r = client.get("/api/evaluation/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert runs[0]["run_id"] == "20261014_120000" and runs[0]["scenes"] == ["pccr"] and runs[0]["n_files"] == 4
    r = client.get("/api/evaluation/runs/20261014_120000").json()
    by_path = {f["path"]: f for f in r["files"]}
    assert by_path["pccr/reference/dimensions.csv"] == {"path": "pccr/reference/dimensions.csv", "kind": "csv",
                                                        "scene": "pccr", "variant": "reference", "phase": "error",
                                                        "size": len("id,error_m\nd_0002,0.01\n")}
    assert by_path["pccr/reference/curve_distance.png"]["phase"] == "curves"
    assert by_path["pareto.json"]["scene"] is None and by_path["pareto.json"]["phase"] == "frontier"
    r = client.get("/api/evaluation/runs/20261014_120000/file", params={"path": "pccr/reference/dimensions.csv"})
    assert r.status_code == 200 and r.text.startswith("id,error_m") and r.headers["content-type"].startswith("text/csv")
    assert client.get("/api/evaluation/runs/20261014_120000/file", params={"path": "../manifest.json"}).status_code == 404
    assert client.get("/api/evaluation/runs/nope").status_code == 404
