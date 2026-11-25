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


def test_the_place_a_video_recorded_is_read_from_its_container(tmp_path):
    assert G.parse_iso6709("+41.6488-000.8891/") == {"lat": 41.6488, "lon": -0.8891}
    assert G.parse_iso6709("+40.4168-003.7038+650.000/") == {"lat": 40.4168, "lon": -3.7038, "altitude_m": 650.0}
    for bad in ("", "north", "+91.0+000.0/", "+40.0+190.0/"):
        assert G.parse_iso6709(bad) is None, bad
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg on this machine")
    with_gps, without = tmp_path / "with.mp4", tmp_path / "without.mp4"
    for path, extra in ((with_gps, ["-metadata", "location=+41.6488-000.8891/"]), (without, [])):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2", *extra,
                        str(path)], check=True)
    place = G.video_location(with_gps)
    assert (place["lat"], place["lon"], place["tag"]) == (41.6488, -0.8891, "location")
    assert G.video_location(without) is None


def _ply(path, rows):
    """A binary PLY with the published cloud's fields: (x, y, z, r, g, b)."""
    import numpy as np
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])
    a = np.zeros(len(rows), dt)
    for k, (x, y, z, r, g, b) in enumerate(rows):
        a[k] = (x, y, z, r, g, b, 1.0, 0, 0, 0)
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(rows)}\n"
            + "".join(f"property {t} {n}\n" for n, t in (("x", "float"), ("y", "float"), ("z", "float"),
                                                       ("red", "uchar"), ("green", "uchar"), ("blue", "uchar"),
                                                       ("confidence", "float"), ("frame_global", "int"),
                                                       ("pixel_row", "short"), ("pixel_col", "short")))
            + "end_header\n").encode("ascii")
    path.write_bytes(head + a.tobytes())


def test_the_plan_shows_each_cell_from_above_in_the_georeferences_frame(tmp_path):
    import io

    import numpy as np
    from PIL import Image
    out = tmp_path / "output"
    out.mkdir()
    # display frame, y up: the plan is X = x, Y = -z. A low red point and a high blue one in the same cell (x 0,
    # z 0); a green one 1 m to the plan's +Y (z = -1); a white one 1 m to +X
    _ply(out / "cleaned_cloud.ply", [(0.02, 0.0, -0.02, 255, 0, 0), (0.02, 2.0, -0.02, 0, 0, 255),
                                     (0.02, 0.0, -1.02, 0, 255, 0), (1.02, 0.0, -0.02, 255, 255, 255)])
    r = G.plan_raster(out, min_cell_m=0.05)
    img = np.asarray(Image.open(io.BytesIO(r["png"])))
    assert r["cell_m"] == 0.05 and (r["width"], r["height"]) == img.shape[1::-1]
    def at(x, y):                                   # the cell of plan point (x, y), as the cloud stores it (float32)
        xf, yf = float(np.float32(x)), -float(np.float32(-y))
        return img[int((r["y_max"] - yf) // r["cell_m"]), int((xf - r["x_min"]) // r["cell_m"])]
    assert tuple(at(0.02, 0.02)) == (0, 0, 255, 255)           # the higher point's colour
    assert tuple(at(0.02, 1.02)) == (0, 255, 0, 255)           # +Y is up in the image (a smaller row)
    assert tuple(at(1.02, 0.02)) == (255, 255, 255, 255)
    assert img[..., 3].sum() == 3 * 255                         # nothing else drawn: the imagery shows through
    # the floor adjustment of the scan is applied (here a quarter turn about the vertical)
    np.savez(out / "floor_transform.npz", R=np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]]), s=np.float64(1.0),
             t=np.zeros(3))
    turned = G.plan_raster(out, min_cell_m=0.05)
    assert (turned["width"], turned["height"]) != (r["width"], r["height"]) or turned["x_min"] != r["x_min"]


def test_the_plan_and_the_videos_place_through_the_routes(tmp_path):
    import types

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import geo_api

    d = _project(tmp_path, "grupo_electrogeno")
    src = d / "scans" / "2026-11-24" / "src_default"
    _ply(src / "output" / "cleaned_cloud.ply", [(0.0, 0.0, 0.0, 10, 20, 30), (3.0, 1.0, -4.0, 40, 50, 60)])
    ctx = types.SimpleNamespace(output_dir=src / "output", source_dir=src, session_dir=src)
    geo_api.configure(resolve_ctx=lambda *a: ctx, projects_dir=lambda: tmp_path)
    app = FastAPI()
    app.include_router(geo_api.router)
    client = TestClient(app)
    plan = client.get("/api/geo/grupo_electrogeno/plan").json()
    assert plan["scan"] == "2026-11-24/default" and plan["image"].startswith("data:image/png;base64,")
    assert plan["points"] == 2 and plan["x_min"] == 0.0 and plan["y_max"] == 4.0
    assert client.get("/api/geo/grupo_electrogeno/plan").json() == plan           # the same, from memory
    video = client.get("/api/geo/grupo_electrogeno/video").json()
    assert video == {"ok": True, "scan": "2026-11-24/default", "video": None, "location": None}


def _glb(data: bytes):
    import json as _json
    import struct
    magic, version, length = struct.unpack("<4sII", data[:12])
    jl, jt = struct.unpack("<I4s", data[12:20])
    doc = _json.loads(data[20:20 + jl])
    bl, bt = struct.unpack("<I4s", data[20 + jl:28 + jl])
    return (magic, version, length, jt, bt), doc, data[28 + jl:28 + jl + bl]


def test_the_cloud_goes_on_the_globe_as_one_gltf_of_points_with_its_colours(tmp_path):
    import numpy as np
    out = tmp_path / "output"
    out.mkdir()
    _ply(out / "cleaned_cloud.ply", [(0.0, 0.0, 0.0, 255, 0, 0), (2.0, 3.0, -4.0, 0, 255, 0), (1.0, 1.0, -1.0, 0, 0, 255)])
    r = G.cloud_points_glb(out)
    (magic, version, length, jt, bt), doc, blob = _glb(r["glb"])
    assert (magic, version, length, jt, bt) == (b"glTF", 2, len(r["glb"]), b"JSON", b"BIN\0")
    prim = doc["meshes"][0]["primitives"][0]
    assert prim["mode"] == 0 and r["points"] == r["points_used"] == 3                       # POINTS
    pos = np.frombuffer(blob[:36], "<f4").reshape(3, 3)
    col = np.frombuffer(blob[36:48], np.uint8).reshape(3, 4)
    assert pos.tolist() == [[0, 0, 0], [2, 3, -4], [1, 1, -1]]                               # glTF's own axes, Y up
    assert col[:, :3].tolist() == [[255, 0, 0], [0, 255, 0], [0, 0, 255]] and (col[:, 3] == 255).all()
    # the 3D Tiles box is in the Z-up frame the viewer turns glTF into: X = x, Y = -z, Z = y
    assert r["box"][:3] == [1.0, 2.0, 1.5] and [r["box"][i] for i in (3, 7, 11)] == [1.0, 2.0, 1.5]
    assert G.floor_adjustment(out) is None
    np.savez(out / "floor_transform.npz", R=np.eye(3), s=np.float64(2.0), t=np.array([0.0, 1.0, 0.0]))
    assert G.floor_adjustment(out) == {"R": np.eye(3).tolist(), "s": 2.0, "t": [0.0, 1.0, 0.0]}
    _, _, blob2 = _glb(G.cloud_points_glb(out)["glb"])
    assert np.frombuffer(blob2[:36], "<f4").reshape(3, 3)[1].tolist() == [4, 7, -8]           # the adjustment applied


def test_what_the_globe_shows_of_a_work_and_its_cloud_through_the_routes(tmp_path):
    import types

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import geo_api

    d = _project(tmp_path, "zaragoza", days=("2026-06-03",))
    src = d / "scans" / "2026-06-03" / "src_default"
    _ply(src / "output" / "cleaned_cloud.ply", [(0.0, 0.0, 0.0, 1, 2, 3), (1.0, 2.0, -3.0, 4, 5, 6)])
    ctx = types.SimpleNamespace(output_dir=src / "output", source_dir=src, session_dir=src)
    geo_api.configure(resolve_ctx=lambda *a: ctx, projects_dir=lambda: tmp_path)
    app = FastAPI()
    app.include_router(geo_api.router)
    client = TestClient(app)
    c = client.get("/api/geo/zaragoza/content").json()
    assert c["mesh_tileset"] is None and c["floor"] is None and c["georef"] is None and c["ifcs"] == []
    (d / "ifcs").mkdir()
    (d / "ifcs" / "zaragoza elements.ifc").write_text("ISO-10303-21;")
    assert client.get("/api/geo/zaragoza/content").json()["ifcs"] == [
        {"name": "zaragoza elements.ifc", "url": "/api/sessions/zaragoza/bim/zaragoza%20elements.ifc"}]
    assert client.get("/api/geo").json()["projects"][0]["ifcs"] == ["zaragoza elements.ifc"]
    assert c["cloud_tileset"] == "/api/geo/zaragoza/cloud/tileset.json?scan=2026-06-03%2Fdefault"
    (src / "output" / "tsdf" / "genrecon_photo" / "tiles").mkdir(parents=True)
    (src / "output" / "tsdf" / "genrecon_photo" / "tiles" / "tileset.json").write_text("{}")
    c = client.get("/api/geo/zaragoza/content").json()
    assert c["mesh_folder"] == "genrecon_photo" and c["mesh_tileset"] == (
        "/api/segmentation/tsdf/tiles/zaragoza/genrecon_photo/tileset.json?stac_scan=zaragoza%7C2026-06-03%2Fdefault")
    ts = client.get("/api/geo/zaragoza/cloud/tileset.json").json()
    assert ts["asset"]["version"] == "1.1" and ts["root"]["content"]["uri"] == "cloud.glb?scan=2026-06-03%2Fdefault"
    glb = client.get("/api/geo/zaragoza/cloud/cloud.glb")
    assert glb.headers["content-type"] == "model/gltf-binary" and glb.content[:4] == b"glTF"
