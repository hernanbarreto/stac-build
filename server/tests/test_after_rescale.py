"""bim/after_rescale.py — after a calibration's rescale, the scan's stale products renewed (USER 2026-11-28: "cual es el
paso correcto para el genrecon el ifc y la comparacion" / "perfecto, dale"): which IFCs are the scan's (from their
header alone) and with which share of the mesh they were made, the stored comparison removed only when it is the
scan's, and an IFC from outside never touched."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bim import after_rescale as AR  # noqa: E402
from bim.comparison import SHEET_META, SHEET_OCTREE, stored_sheet  # noqa: E402


def _ifc(path: Path, file_name: str, described: str = "") -> Path:
    path.write_text("ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');\n"
                    f"FILE_NAME('{file_name}','',(''),(''),'IfcOpenShell','IfcOpenShell','');\nENDSEC;\nDATA;\n"
                    f"#10=IFCPROJECT('x',$,'p','{described}',$,$,$,(#3),#9);\nENDSEC;\nEND-ISO-10303-21;\n")
    return path


def _sheet(bim: Path, scan: str) -> None:
    (bim / SHEET_OCTREE).mkdir(parents=True)
    (bim / SHEET_OCTREE / "metadata.json").write_text("{}")
    (bim / "sabana_cloud.ply").write_bytes(b"ply")
    (bim / SHEET_META).write_text(json.dumps({"scan": scan}))


def test_the_scans_ifcs_are_told_by_their_header_with_their_share_of_the_mesh(tmp_path):
    whole = _ifc(tmp_path / "a_elements.ifc", "grupo_electrogeno_2026-11-24_default_from_mesh.ifc",
                 "… Triangles kept: 7,284,398 of 7,284,398; …")
    light = _ifc(tmp_path / "b.ifc", "pccr_2026-08-31_default_from_mesh.ifc", "Triangles kept: 1,328,421 of 13,284,219")
    boxes = _ifc(tmp_path / "c.ifc", "pccr_2026-08-31_default_from_scan.ifc")
    design = _ifc(tmp_path / "d.ifc", "Proyecto_Sala_Tecnica_R2.ifc")
    assert AR.stac_ifc(whole) == {"session": "grupo_electrogeno", "scan": "2026-11-24/default", "kind": "mesh", "keep": 1.0}
    assert AR.stac_ifc(light)["keep"] == 0.1
    assert AR.stac_ifc(boxes) == {"session": "pccr", "scan": "2026-08-31/default", "kind": "scan", "keep": 1.0}
    assert AR.stac_ifc(design) is None                                    # an IFC from outside: never written again


def test_the_comparison_is_removed_only_when_it_measured_this_scan(tmp_path):
    ifcs, bim = tmp_path / "ifcs", tmp_path / "bim_comparison"
    ifcs.mkdir()
    _sheet(bim, "2026-08-24/default")
    r = AR.renew("pccr", "2026-08-31/default", ifcs, bim, log=lambda *_: None)
    assert r == {"ifcs": [], "errors": [], "comparison_removed": False} and stored_sheet(bim) is not None
    r = AR.renew("pccr", "2026-08-24/default", ifcs, bim, log=lambda *_: None)
    assert r["comparison_removed"] and stored_sheet(bim) is None
    assert not (bim / SHEET_OCTREE).exists() and not (bim / "sabana_cloud.ply").exists()


def test_an_ifc_of_another_scan_or_from_outside_is_left_as_it_is(tmp_path):
    ifcs, bim = tmp_path / "ifcs", tmp_path / "bim_comparison"
    ifcs.mkdir()
    other = _ifc(ifcs / "o.ifc", "pccr_2026-08-24_default_from_mesh.ifc")
    design = _ifc(ifcs / "d.ifc", "Proyecto.ifc")
    before = (other.read_bytes(), design.read_bytes())
    r = AR.renew("pccr", "2026-08-31/default", ifcs, bim, log=lambda *_: None)
    assert r["ifcs"] == [] and r["errors"] == []
    assert (other.read_bytes(), design.read_bytes()) == before
