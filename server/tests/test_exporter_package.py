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

from test_exporter_cloud import _scan
from test_ifc_from_scan import _store

pytest.importorskip("ifcopenshell")
pytest.importorskip("laspy")
pytest.importorskip("pye57")


@pytest.fixture
def scan(tmp_path, monkeypatch):
    out, rec = _scan(tmp_path, n=3000, octree_range=(0.0, 10.0))
    _store(tmp_path).rename(out / "scene_r.db")
    mesh = out / "tsdf" / "genrecon_photo"
    mesh.mkdir(parents=True)
    (mesh / "genrecon_photo.glb").write_bytes(b"glTF" + bytes(64))
    (mesh / "genrecon_photo.meta.json").write_text(json.dumps(
        {"glb_file": "genrecon_photo.glb", "n_faces": 1234, "textured": True, "provenance": "geometry: vlm_generated"}))
    ctx = types.SimpleNamespace(output_dir=out, project_dir=tmp_path / "project")
    monkeypatch.setattr(P, "context", lambda session, scan_key: ctx)
    import reconstruction.grid_knn as G
    monkeypatch.setattr(G, "config_section", lambda path, keys: {"scratch_dir": str(tmp_path / "scratch")})
    return ctx, rec


def test_the_plan_says_what_the_scan_has(scan):
    plan = P.plan("s", "2026-01-01/default")
    assert plan["points"] == 3000 and all(plan["items"][k]["available"] for k in P.ITEMS)
    assert plan["items"]["ifc"]["elements"] == 3 and plan["items"]["csv"]["objects"] == 4
    assert plan["items"]["glb"]["meshes"] == ["genrecon_photo"] and plan["previous"] is None
    assert plan["state"]["geometry_epoch"] == 0 and plan["state"]["calibration"]["state"] == "uncalibrated"


def test_everything_in_one_zip_with_its_readme(scan, tmp_path):
    ctx, rec = scan
    seen = []
    report = P.run("s", "2026-01-01/default", list(P.ITEMS), lang="es", progress=lambda p, m: seen.append(p), now=0.0)
    path = ctx.project_dir / "exports" / "s_2026-01-01_stac_export.zip"
    assert report["path"] == str(path) and path.exists() and report["points"] == report["points_published"] == 3000
    assert seen == sorted(seen) and seen[-1] == 100
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
        assert names == sorted(f"s_2026-01-01/{n}" for n in (
            "LEEME.txt", "s_2026-01-01_cloud.e57", "s_2026-01-01_cloud.las", "s_2026-01-01_cloud.ply",
            "s_2026-01-01_mesh_genrecon_photo.glb", "s_2026-01-01_elements.ifc", "s_2026-01-01_objects.csv"))
        text = z.read("s_2026-01-01/LEEME.txt").decode()
        assert "SIN CALIBRAR" in text and "época 0" in text and "eje Z hacia arriba" in text
        assert "3,000 puntos de los 3,000" in text and "3 elementos constructivos" in text and "vlm_generated" in text
        (tmp_path / "back.ply").write_bytes(z.read("s_2026-01-01/s_2026-01-01_cloud.ply"))
        back = C.open_ply(tmp_path / "back.ply")
        assert np.array_equal(back["z"], rec["y"]) and np.array_equal(back["pixel_col"], rec["pixel_col"])
        rows = z.read("s_2026-01-01/s_2026-01-01_objects.csv").decode().strip().split("\n")
        assert rows[0].startswith("instance_id,label,points,center_x_m") and len(rows) == 1 + 4
        # the mesh is the file on disk, stored as it is
        info = z.getinfo("s_2026-01-01/s_2026-01-01_mesh_genrecon_photo.glb")
        assert info.compress_type == zipfile.ZIP_STORED and z.read(info) == b"glTF" + bytes(64)
        assert z.read("s_2026-01-01/s_2026-01-01_elements.ifc").startswith(b"ISO-10303-21")
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
        assert f"{keep:,} points of the 3,000 published (confidence filter 0.50" in text and "UNCALIBRATED" in text
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
