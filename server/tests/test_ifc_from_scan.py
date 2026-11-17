"""An IFC built from a scan's own segmentation (USER 2026-11-17: "create un IFC vos a partir de
PCCR3108, para comparar contra ese mismo, porque es simple"): one element per segmented building
element, its solid the box the store holds, named so the comparison finds its segment, the same
bytes every time."""
from __future__ import annotations

import numpy as np
import pytest

from bim import ifc_from_scan
from phase_r.geometry import fit_gravity_aligned_obb
from phase_r.instance_store import InstanceStore

ifcopenshell = pytest.importorskip("ifcopenshell")


def _store(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "scene_r.db"
    store = InstanceStore(path)
    yaw = np.deg2rad(30.0)
    turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    for iid, label, size, centre, R in ((1, "floor", [6.0, 0.02, 4.0], [0, 0, 0], np.eye(3)),
                                        (2, "wall", [4.0, 2.6, 0.1], [1, 1.3, 2], turn),
                                        (3, "chair", [0.5, 0.9, 0.5], [0, 0.45, 0], np.eye(3)),
                                        (7, "door", [0.9, 2.0, 0.05], [2, 1.0, -1], turn)):
        pts = (rng.uniform(-0.5, 0.5, size=(600, 3)) * size) @ R.T + centre
        store.upsert_instance(iid, label, source="test")
        store.set_points(iid, pts)
        obb = fit_gravity_aligned_obb(np.asarray(pts, np.float32).astype(np.float64))
        store.set_obb(iid, obb[0], obb[1], obb[2], n_points=len(pts))
    store.close()
    return path


def test_only_building_elements_each_as_its_stored_box(tmp_path):
    import ifcopenshell.geom
    path = _store(tmp_path)
    elements = ifc_from_scan.elements_of(path)
    assert [(e["id"], e["label"], e["ifc"][0]) for e in elements] == [
        (1, "floor", "IfcSlab"), (2, "wall", "IfcWall"), (7, "door", "IfcDoor")]          # no chair
    model = ifc_from_scan.build(elements, session="s", scan_key="2026-01-01/default", geometry_epoch="0")
    out = tmp_path / "m.ifc"
    model.write(str(out))
    f = ifcopenshell.open(str(out))
    assert f.schema == "IFC4" and f.by_type("IfcSIUnit")[0].Name == "METRE"
    products = {int(p.Tag): p for p in f.by_type("IfcElement")}
    assert sorted(products) == [1, 2, 7]
    assert products[2].Name == "wall:wall:2" and products[1].PredefinedType == "FLOOR"
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    by_id = {e["id"]: e for e in elements}
    for iid, product in products.items():
        shape = ifcopenshell.geom.create_shape(settings, product)      # (kept alive: its buffers are the geometry's)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        # the store's frame (Y up) written Z-up: (x, y, z) → (x, −z, y)
        corners = ifc_from_scan.box_corners(by_id[iid]["transform"], by_id[iid]["aabb"]) @ ifc_from_scan.TO_IFC.T
        far = np.abs(verts[:, None] - corners[None]).max(2).min(1).max()
        missing = np.abs(corners[:, None] - verts[None]).max(2).min(1).max()
        assert max(far, missing) < 1e-5, (iid, far, missing)
    # the element's key for the comparison is the instance id
    from bim.comparison import _build_name_index
    assert sorted(_build_name_index(f)) == ["1", "2", "7"]


def test_the_same_store_gives_the_same_bytes(tmp_path):
    path = _store(tmp_path)
    elements = ifc_from_scan.elements_of(path)
    files = []
    for name in ("a.ifc", "b.ifc"):
        ifc_from_scan.build(elements, session="s", scan_key="k/default", geometry_epoch="3").write(str(tmp_path / name))
        files.append((tmp_path / name).read_bytes())
    assert files[0] == files[1] and files[0].startswith(b"ISO-10303-21")
    # another scan: other identities, the same geometry
    ifc_from_scan.build(elements, session="s", scan_key="other/default", geometry_epoch="3").write(str(tmp_path / "c.ifc"))
    assert (tmp_path / "c.ifc").read_bytes() != files[0]
