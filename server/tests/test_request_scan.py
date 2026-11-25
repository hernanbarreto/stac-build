"""EACH DEVICE WORKS ON ITS OWN SCAN (USER 2026-11-24: "debe mantener el contexto y la conversación para cada
sesión de usuario que está trabajando, OJO con eso!!!") — request_scan.py: a request that says which scan it is
on resolves that one, never the project's active scan another device rewrites."""
from __future__ import annotations

import asyncio
import json

import config
import request_scan
from project_paths import resolve_session


def _projects(tmp_path, monkeypatch):
    """pccr with two scans, the 31st active (another device opened it last); zaragoza with one."""
    root = tmp_path / "projects"
    for pid, days in (("pccr", ("2026-08-24", "2026-08-31")), ("zaragoza", ("2026-06-03",))):
        for day in days:
            (root / pid / "scans" / day / "src_default" / "output").mkdir(parents=True)
        (root / pid / "project.json").write_text(json.dumps({"name": pid, "active_scan": f"{days[-1]}/default"}))
    monkeypatch.setattr(config, "PROJECTS_DIR", root)
    return str(tmp_path)


def _day(ctx) -> str:
    return ctx.source_dir.parent.name


def test_a_request_resolves_the_scan_it_is_on_never_the_one_another_device_made_active(tmp_path, monkeypatch):
    server = _projects(tmp_path, monkeypatch)
    assert _day(resolve_session(server, "pccr")) == "2026-08-31"                 # no request scan: as before
    token = request_scan.set_scan("pccr", "2026-08-24/default")
    try:
        assert _day(resolve_session(server, "pccr")) == "2026-08-24"
        # a route that names its scan keeps it
        assert _day(resolve_session(server, "pccr", scan_key="2026-08-31/default")) == "2026-08-31"
        # another project is not this request's
        assert _day(resolve_session(server, "zaragoza")) == "2026-06-03"
    finally:
        request_scan.reset(token)
    # a scan that is not there (deleted meanwhile) is not taken: the project's own
    token = request_scan.set_scan("pccr", "2026-09-01/default")
    try:
        assert _day(resolve_session(server, "pccr")) == "2026-08-31"
    finally:
        request_scan.reset(token)


def test_what_a_request_says_is_read_safely():
    assert request_scan.parse("pccr|2026-08-24%2Fdefault") == ("pccr", "2026-08-24/default")
    assert request_scan.parse("grupo_electrogeno|2026-11-24/default") == ("grupo_electrogeno", "2026-11-24/default")
    for bad in (None, "", "pccr", "pccr|", "|2026-08-24/default", "pccr|2026-08-24", "pccr|../../etc",
                "pccr|2026-08-24/default/x", "../x|2026-08-24/default", "pccr|..%2Fdefault"):
        assert request_scan.parse(bad) is None, bad


def test_the_header_or_the_query_says_it_and_the_routes_threads_see_it():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.testclient import TestClient

    pool = request_scan.CopyContextExecutor(max_workers=2)

    async def where(request):
        in_thread = await asyncio.get_running_loop().run_in_executor(pool, request_scan.scan_for, "pccr")
        return JSONResponse({"here": request_scan.scan_for("pccr"), "thread": in_thread,
                             "other": request_scan.scan_for("zaragoza")})

    app = Starlette(routes=[Route("/where", where)])
    app.add_middleware(request_scan.RequestScanMiddleware)
    with TestClient(app) as client:
        a = client.get("/where", headers={"X-STAC-Scan": "pccr|2026-08-24%2Fdefault"}).json()
        b = client.get("/where", headers={"X-STAC-Scan": "pccr|2026-08-31%2Fdefault"}).json()
        img = client.get("/where", params={"stac_scan": "pccr|2026-08-24/default"}).json()
        none = client.get("/where").json()
        both = client.get("/where", params={"stac_scan": "pccr|2026-08-31/default"},
                          headers={"X-STAC-Scan": "pccr|2026-08-24%2Fdefault"}).json()
    pool.shutdown()
    assert a == {"here": "2026-08-24/default", "thread": "2026-08-24/default", "other": None}
    assert b["here"] == b["thread"] == "2026-08-31/default"                    # two devices, two scans
    assert img["here"] == "2026-08-24/default"                                 # an <img> says it in its query
    assert none == {"here": None, "thread": None, "other": None}               # no word of it: as before
    assert both["here"] == "2026-08-31/default"                                # the URL's own scan first
