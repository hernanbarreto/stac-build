"""The viewer streams a tiled mesh (USER 2026-11-02: "hay que implementar el tema de los tiles … para que pueda ser
manejable una malla grande"): /api/segmentation/tsdf/list names the tileset of a mesh that has one, and
/api/segmentation/tsdf/tiles/... serves its files — nothing outside the mesh's tiles/ folder. ``import main`` loads
the FastAPI app (no server, no GPU); the endpoints run through Starlette's TestClient without the lifespan."""
import json
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def main_mod():
    import main
    return main


@pytest.fixture
def client(main_mod, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(main_mod, "_mesh_watch_register", lambda session_id, output_dir: None)
    return TestClient(main_mod.app)              # no `with`: the lifespan (vLLM) never runs


def _session(tmp_path, monkeypatch, main_mod, tiled: bool) -> Path:
    out = tmp_path / "output"
    d = out / "tsdf" / "genrecon_photo"
    d.mkdir(parents=True)
    (d / "genrecon_photo.glb").write_bytes(b"glTF")
    (d / "genrecon_photo.meta.json").write_text(json.dumps({"method": "genrecon"}))
    if tiled:
        (d / "tiles").mkdir()
        (d / "tiles" / "tileset.json").write_text(json.dumps({"asset": {"version": "1.1"}}))
        (d / "tiles" / "tile_3.glb").write_bytes(b"glTF-tile")
    (tmp_path / "secret.json").write_text("{}")
    ctx = types.SimpleNamespace(output_dir=out, session_dir=tmp_path, frames_dir=tmp_path / "frames")
    monkeypatch.setattr(main_mod, "_ctx", lambda session_id, scan_key=None: ctx)
    return d


def test_a_mesh_without_tiles_lists_its_glb_only(tmp_path, monkeypatch, main_mod, client):
    _session(tmp_path, monkeypatch, main_mod, tiled=False)
    shapes = client.get("/api/segmentation/tsdf/list/s1").json()["shapes"]
    assert [s["folder"] for s in shapes] == ["genrecon_photo"]
    assert "tileset_url" not in shapes[0]
    assert shapes[0]["glb_url"].endswith("/genrecon_photo/genrecon_photo.glb")


def test_a_tiled_mesh_lists_its_tileset(tmp_path, monkeypatch, main_mod, client):
    _session(tmp_path, monkeypatch, main_mod, tiled=True)
    shapes = client.get("/api/segmentation/tsdf/list/s1").json()["shapes"]
    assert shapes[0]["tileset_url"] == "/api/segmentation/tsdf/tiles/s1/genrecon_photo/tileset.json"


def test_the_tiles_route_serves_the_tileset_and_its_tiles(tmp_path, monkeypatch, main_mod, client):
    _session(tmp_path, monkeypatch, main_mod, tiled=True)
    r = client.get("/api/segmentation/tsdf/tiles/s1/genrecon_photo/tileset.json")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json() == {"asset": {"version": "1.1"}}
    r = client.get("/api/segmentation/tsdf/tiles/s1/genrecon_photo/tile_3.glb")
    assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary" and r.content == b"glTF-tile"
    assert client.get("/api/segmentation/tsdf/tiles/s1/genrecon_photo/tile_9.glb").status_code == 404


def test_the_tiles_route_never_leaves_the_tiles_folder(tmp_path, monkeypatch, main_mod, client):
    _session(tmp_path, monkeypatch, main_mod, tiled=True)
    # the mesh's own GLB sits beside tiles/, not in it
    assert client.get("/api/segmentation/tsdf/tiles/s1/genrecon_photo/genrecon_photo.glb").status_code == 404
    assert client.get("/api/segmentation/tsdf/tiles/s1/genrecon_photo/..secret.json").status_code == 400
    assert client.get("/api/segmentation/tsdf/tiles/s1/..genrecon_photo/tileset.json").status_code == 400
