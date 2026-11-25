"""The BIM file route (main.py /api/sessions/{id}/bim/{file}): an IFC rewritten in place under its own name reaches the
viewer — never cached as fresh (no-cache), the file's ETag, 304 when the browser holds that same file, the new file when
it changed (USER 2026-11-25: "Generale el ifc con el color dominante … Generala para las sesiones que tengan ifc
generado y avisame asi las veo"). ``import main`` loads the FastAPI app (no server, no GPU); the lifespan never runs."""
import os
import types

import pytest


@pytest.fixture(scope="module")
def main_mod():
    import main
    return main


@pytest.fixture
def client(main_mod, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    ifcs = tmp_path / "ifcs"
    ifcs.mkdir()
    ctx = types.SimpleNamespace(ifcs_dir=ifcs, project_dir=tmp_path, bim_comparison_dir=tmp_path / "bim_comparison")
    monkeypatch.setattr(main_mod, "_ctx", lambda session_id, scan_key=None: ctx)
    return TestClient(main_mod.app), ifcs


def test_an_ifc_rewritten_in_place_reaches_the_viewer(client):
    c, ifcs = client
    f = ifcs / "scan_elements.ifc"
    f.write_bytes(b"ISO-10303-21; first")
    r = c.get("/api/sessions/s1/bim/scan_elements.ifc")
    assert r.status_code == 200 and r.content == b"ISO-10303-21; first"
    assert r.headers["cache-control"] == "no-cache"
    etag = r.headers["etag"]
    r = c.get("/api/sessions/s1/bim/scan_elements.ifc", headers={"If-None-Match": etag})
    assert r.status_code == 304 and r.content == b"" and r.headers["etag"] == etag      # the same file: not sent again
    f.write_bytes(b"ISO-10303-21; second, with colours")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))                          # a later date, as a rewrite gives
    r = c.get("/api/sessions/s1/bim/scan_elements.ifc", headers={"If-None-Match": etag})
    assert r.status_code == 200 and r.content == b"ISO-10303-21; second, with colours" and r.headers["etag"] != etag


def test_a_missing_bim_file_is_404(client):
    c, _ = client
    assert c.get("/api/sessions/s1/bim/none.ifc").status_code == 404
