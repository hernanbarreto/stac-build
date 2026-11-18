"""One export of a scan (USER 2026-11-17: "podriamos exportar la nube de puntos compatible con
varios programas, la malla glb con la textura, y el IFC … y que la persona lo pueda descargar en
su maquina"): the chosen items in one ZIP with a README that says the epoch and the calibration —
the geometry as it is on disk; one ZIP per scan."""
from __future__ import annotations

import io
import json
import types
import zipfile

import numpy as np
import pytest

from exporter import cloud as C
from exporter import package as P

from test_ifc_from_mesh import _scan as _mesh_scan

pytest.importorskip("ifcopenshell")
pytest.importorskip("laspy")
pytest.importorskip("pye57")


@pytest.fixture
def scan(tmp_path, monkeypatch):
    # a wall and a chair: their mesh (a binary glTF), the cloud on it, its per-point ids, the store
    out, _glb, _wall, _chair = _mesh_scan(tmp_path)
    (out / "tsdf" / "genrecon_photo" / "genrecon_photo.meta.json").write_text(json.dumps(
        {"glb_file": "genrecon_photo.glb", "n_faces": 1152, "textured": True, "provenance": "geometry: vlm_generated"}))
    data = C.open_ply(out / "cleaned_cloud.ply")
    rec = np.asarray(data).copy()
    del data
    rng = np.random.default_rng(1)
    rec["confidence"] = rng.uniform(1.0, 9.0, len(rec))
    rec["frame_global"], rec["pixel_col"] = rng.integers(0, 9000, len(rec)), rng.integers(0, 1280, len(rec))
    head = (out / "cleaned_cloud.ply").read_bytes().split(b"end_header\n")[0] + b"end_header\n"
    (out / "cleaned_cloud.ply").write_bytes(head + rec.tobytes())
    (out / "potree").mkdir()
    (out / "potree" / "metadata.json").write_text(json.dumps({"attributes": [{"name": "confidence", "min": [0.0], "max": [10.0]}]}))
    ctx = types.SimpleNamespace(output_dir=out, project_dir=tmp_path / "project")
    monkeypatch.setattr(P, "context", lambda session, scan_key: ctx)
    import reconstruction.grid_knn as G
    values = {("export",): {"scratch_dir": str(tmp_path / "scratch")},
              ("export", "ifc"): {"keep": 1.0, "neighbours": 8, "max_edge_voxels": 0},
              ("genrecon",): {"cube_size_m": 2.5}}
    monkeypatch.setattr(G, "config_section", lambda path, keys: {k: values[tuple(path)][k] for k in keys})
    return ctx, rec


def test_the_plan_says_what_the_scan_has(scan):
    plan = P.plan("s", "2026-01-01/default")
    assert plan["points"] == 26000 and all(plan["items"][k]["available"] for k in P.ITEMS)
    assert plan["items"]["ifc"]["elements"] == 2 and plan["items"]["csv"]["objects"] == 2
    assert plan["items"]["glb"]["meshes"] == ["genrecon_photo"] and plan["previous"] is None
    assert plan["state"]["geometry_epoch"] == 0 and plan["state"]["calibration"]["state"] == "uncalibrated"


def test_everything_in_one_zip_with_its_readme(scan, tmp_path):
    ctx, rec = scan
    seen = []
    report = P.run("s", "2026-01-01/default", list(P.ITEMS), lang="es", progress=lambda p, m: seen.append(p), now=0.0)
    path = ctx.project_dir / "exports" / "s_2026-01-01_stac_export.zip"
    assert report["path"] == str(path) and path.exists() and report["points"] == report["points_published"] == 26000
    assert seen == sorted(seen) and seen[-1] == 100
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
        assert names == sorted(f"s_2026-01-01/{n}" for n in (
            "LEEME.txt", "s_2026-01-01_cloud.e57", "s_2026-01-01_cloud.las", "s_2026-01-01_cloud.ply",
            "s_2026-01-01_mesh_genrecon_photo.glb", "s_2026-01-01_elements.ifc", "s_2026-01-01_objects.csv"))
        text = z.read("s_2026-01-01/LEEME.txt").decode()
        assert "SIN CALIBRAR" in text and "época 0" in text and "eje Z hacia arriba" in text
        assert "26,000 puntos de los 26,000" in text and "cortada por la segmentación, 2 elementos" in text and "vlm_generated" in text
        assert "Se conserva el 100 % de los triángulos" in text and report["ifc"]["faces_no_object"] == 0
        (tmp_path / "back.ply").write_bytes(z.read("s_2026-01-01/s_2026-01-01_cloud.ply"))
        back = C.open_ply(tmp_path / "back.ply")
        assert np.array_equal(back["z"], rec["y"]) and np.array_equal(back["pixel_col"], rec["pixel_col"])
        rows = z.read("s_2026-01-01/s_2026-01-01_objects.csv").decode().strip().split("\n")
        assert rows[0].startswith("instance_id,label,points,center_x_m") and len(rows) == 1 + 2
        # the mesh is the file on disk, stored as it is
        info = z.getinfo("s_2026-01-01/s_2026-01-01_mesh_genrecon_photo.glb")
        mesh_file = ctx.output_dir / "tsdf" / "genrecon_photo" / "genrecon_photo.glb"
        assert info.compress_type == zipfile.ZIP_STORED and z.read(info) == mesh_file.read_bytes()
        ifc_text = z.read("s_2026-01-01/s_2026-01-01_elements.ifc")
        assert ifc_text.startswith(b"ISO-10303-21") and ifc_text.count(b"IFCTRIANGULATEDFACESET") == 2
        assert b"wall:wall:1" in ifc_text and b"IFCBUILDINGELEMENTPROXY" in ifc_text
    # nothing is left on the scratch disk
    assert not any((tmp_path / "scratch").rglob("*.*"))


def test_a_new_export_replaces_the_scans_previous_one(scan):
    ctx, rec = scan
    P.run("s", "2026-01-01/default", ["csv", "ply"], confidence=0.5, lang="en")
    keep = int(((rec["confidence"].astype(np.float64)) / 10.0 >= 0.5).sum())
    path = ctx.project_dir / "exports" / "s_2026-01-01_stac_export.zip"
    with zipfile.ZipFile(path) as z:
        assert sorted(n.split("/")[1] for n in z.namelist()) == ["README.txt", "s_2026-01-01_cloud.ply", "s_2026-01-01_objects.csv"]
        text = z.read("s_2026-01-01/README.txt").decode()
        assert f"{keep:,} points of the 26,000 published (confidence filter 0.50" in text and "UNCALIBRATED" in text
    assert [p.name for p in path.parent.iterdir()] == [path.name]
    assert P.plan("s", "2026-01-01/default")["previous"]["file"] == path.name


def test_what_the_scan_does_not_have_is_refused(scan):
    ctx, _rec = scan
    (ctx.output_dir / "scene_r.db").unlink()
    plan = P.plan("s", "2026-01-01/default")
    assert not plan["items"]["ifc"]["available"] and plan["items"]["ifc"]["why"] == "the scan is not segmented"
    with pytest.raises(P.ExportError, match="ifc: the scan is not segmented"):
        P.run("s", "2026-01-01/default", ["ply", "ifc"])
    with pytest.raises(P.ExportError, match="nothing to export"):
        P.run("s", "2026-01-01/default", ["dwg"])
    assert not (ctx.project_dir / "exports").exists() or not list((ctx.project_dir / "exports").iterdir())
