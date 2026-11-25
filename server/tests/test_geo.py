"""WHERE A WORK STANDS ON THE GLOBE (USER 2026-11-24 — geo.py / geo_api.py): one georeference per project, what
it holds, how it is checked, and its routes."""
from __future__ import annotations

import json
import math

import pytest

import geo as G


def _project(root, pid="grupo_electrogeno", days=("2026-11-24",)):
    for day in days:
        (root / pid / "scans" / day / "src_default" / "output").mkdir(parents=True)
    (root / pid / "project.json").write_text(json.dumps({"name": pid}))
    return root / pid


def test_a_georeference_is_checked_and_says_where_it_came_from(tmp_path):
    g = G.validate({"lat": 41.6488, "lon": -0.8891, "height_m": 199.5, "heading_deg": -30, "source": "manual"},
                   set_by="hernan", scan="2026-11-24/default")
    assert (g["lat"], g["lon"], g["height_m"], g["heading_deg"]) == (41.6488, -0.8891, 199.5, 330.0)
    assert g["provenance"] == "human_stated" and g["accuracy_m"] is None and g["set_by"] == "hernan"
    assert G.validate({"lat": 0, "lon": 0, "height_m": 0, "source": "video", "accuracy_m": 8},
                      set_by="x", scan=None)["provenance"] == "device_measured"
    for bad in ({"lat": 91, "lon": 0, "height_m": 0, "source": "manual"},
                {"lat": 0, "lon": 181, "height_m": 0, "source": "manual"},
                {"lat": 0, "lon": 0, "height_m": float("nan"), "source": "manual"},
                {"lat": 0, "lon": 0, "height_m": 0, "source": "guessed"},
                {"lat": "north", "lon": 0, "height_m": 0, "source": "manual"},
                {"lat": 0, "lon": 0, "height_m": 0, "heading_deg": 400, "source": "manual"}):
        with pytest.raises(G.GeoError):
            G.validate(bad, set_by="x", scan=None)
    d = _project(tmp_path)
    assert G.load(d) is None
    G.save(d, g)
    assert G.load(d) == g and [p.name for p in d.iterdir() if p.name.startswith(".")] == []   # nothing half-written
    assert G.remove(d) and G.load(d) is None and not G.remove(d)


def test_the_plan_turns_by_the_heading():
    assert G.plan_to_enu(3.0, 4.0, 0.0) == (3.0, 4.0)                     # +Y north, +X east
    e, n = G.plan_to_enu(0.0, 1.0, 90.0)                                  # the plan's up points east
    assert math.isclose(e, 1.0) and math.isclose(n, 0.0, abs_tol=1e-12)
    e, n = G.plan_to_enu(1.0, 0.0, 90.0)                                  # … and its +X south
    assert math.isclose(e, 0.0, abs_tol=1e-12) and math.isclose(n, -1.0)


def test_the_routes_list_every_project_and_setting_one_needs_a_user(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import geo_api

    _project(tmp_path, "grupo_electrogeno")
    _project(tmp_path, "pccr", days=("2026-08-24", "2026-08-31"))
    audit = []
    geo_api.configure(resolve_ctx=lambda *a: None, projects_dir=lambda: tmp_path,
                      audit_log=lambda *a, **k: audit.append(a[0]))
    app = FastAPI()
    app.include_router(geo_api.router)
    client = TestClient(app)
    body = {"lat": 41.6488, "lon": -0.8891, "height_m": 199.5, "heading_deg": 12.5, "source": "manual"}
    assert client.put("/api/geo/pccr", json=body).status_code == 401              # nobody signed in
    app.dependency_overrides[geo_api._user] = lambda: {"username": "hernan", "role": "admin"}
    r = client.put("/api/geo/pccr", json=body).json()
    assert r["georef"]["scan"] == "2026-08-24/default" and r["georef"]["set_by"] == "hernan"
    assert client.put("/api/geo/pccr", json={**body, "lat": 95}).status_code == 400
    assert client.put("/api/geo/nowhere", json=body).status_code == 404
    assert client.put("/api/geo/..", json=body).status_code in (404, 405)
    rows = {p["session_id"]: p for p in client.get("/api/geo").json()["projects"]}
    assert set(rows) == {"grupo_electrogeno", "pccr"} and rows["grupo_electrogeno"]["georef"] is None
    assert rows["pccr"]["georef"]["heading_deg"] == 12.5 and rows["pccr"]["scans"] == ["2026-08-24/default",
                                                                                       "2026-08-31/default"]
    assert client.get("/api/geo/pccr").json()["georef"]["lat"] == 41.6488
    assert client.delete("/api/geo/pccr").json()["removed"] is True
    assert client.get("/api/geo/pccr").json()["georef"] is None
    assert audit == ["geo.set", "geo.remove"]
