"""AN IFC BUILT FROM A SCAN — the scan's own segmented building elements as a BIM model.

USER 2026-11-17: "create un IFC vos a partir de PCCR3108, para comparar contra ese mismo, porque
es simple". The BIM comparison needs a model to compare a scan against; with none at hand, the
simplest one that is certain to be IN PLACE is the scan's own: every object of the segmentation
whose label is a building element becomes one IFC element, its solid THE BOX THE STORE HOLDS for
it (the OBB fitted to its points — ``tool_measured``, never refit here). Nothing is invented: no
thickness, no snapping, no element the segmentation does not have.

  - which labels, and as which IFC class: :data:`IFC_CLASS` (a routing table of the construction
    vocabulary — the elements a building model holds; furniture, equipment and site material are
    not in a BIM of the room and are left out);
  - the frame: the store's display frame (Y up, the floor at y = 0, metres) written Z-up as IFC
    asks — (x, y, z) → (x, −z, y) — which is what the viewer's loader turns back, so the model
    lands on the cloud;
  - an element's Name is ``<label>:<label>:<instance id>``: its last part is the key the
    comparison matches a segment with (``bim.comparison._build_name_index``);
  - deterministic: the same store gives the same file byte for byte (GlobalIds derived from the
    scan and the instance id, no clock in the header).

    python -m bim.ifc_from_scan <session> <scan key> [--out file.ifc]
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# label of the segmentation → (IFC class, PredefinedType or None)
IFC_CLASS: Dict[str, Tuple[str, str | None]] = {
    "floor": ("IfcSlab", "FLOOR"),
    "tile": ("IfcCovering", "FLOORING"),
    "wall": ("IfcWall", None),
    "drywall": ("IfcWall", None),
    "ceiling": ("IfcCovering", "CEILING"),
    "ceiling_grid": ("IfcCovering", "CEILING"),
    "door": ("IfcDoor", None),
    "window": ("IfcWindow", None),
    "beam": ("IfcBeam", None),
    "column": ("IfcColumn", None),
    "duct": ("IfcDuctSegment", None),
    "pipe": ("IfcPipeSegment", None),
}
# display frame (Y up) → IFC (Z up): (x, y, z) → (x, −z, y)
TO_IFC = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_NAMESPACE = uuid.UUID("6f1d0c2e-5b7a-4c56-9d0e-57ac0b111d17")


def _lo_hi(aabb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The store keeps a box's extents per axis: (x min, x max, y min, y max, z min, z max)."""
    a = np.asarray(aabb, np.float64)
    return a[0::2], a[1::2]


def box_corners(transform: np.ndarray, aabb: np.ndarray) -> np.ndarray:
    """The 8 corners of a stored box in the store's frame (float64)."""
    lo, hi = _lo_hi(aabb)
    local = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    T = np.asarray(transform, np.float64)
    return local @ T[:3, :3].T + T[:3, 3]


def elements_of(store_path: Path | str) -> List[Dict[str, Any]]:
    """The building elements of a scan's instance store, in instance order: id, label, IFC class
    and the stored box (transform, aabb, n_points)."""
    import sqlite3
    con = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT i.instance_id, i.label, o.obb_transform, o.obb_aabb, o.n_points, o.obb_origin "
            "FROM instances i JOIN instance_obb o ON o.instance_id = i.instance_id AND o.window_id = 'global' "
            "ORDER BY i.instance_id").fetchall()
    finally:
        con.close()
    out = []
    for iid, label, t_blob, a_blob, n_points, origin in rows:
        if label not in IFC_CLASS:
            continue
        T = np.frombuffer(t_blob, np.float32).reshape(4, 4).astype(np.float64)
        aabb = np.frombuffer(a_blob, np.float32).reshape(6).astype(np.float64)
        if not (np.isfinite(T).all() and np.isfinite(aabb).all()) or np.any(aabb[1::2] - aabb[0::2] <= 0):
            raise ValueError(f"instance {iid} ({label}): its stored box is not a volume ({aabb.tolist()})")
        out.append({"id": int(iid), "label": str(label), "ifc": IFC_CLASS[label], "transform": T, "aabb": aabb,
                    "n_points": int(n_points), "origin": str(origin)})
    return out


def _guid(*parts: Any) -> str:
    import ifcopenshell.guid
    return ifcopenshell.guid.compress(uuid.uuid5(_NAMESPACE, "/".join(str(p) for p in parts)).hex)


def build(elements: List[Dict[str, Any]], *, session: str, scan_key: str, geometry_epoch: str):
    """The IFC4 model of ``elements`` (an ``ifcopenshell.file``)."""
    import ifcopenshell
    f = ifcopenshell.file(schema="IFC4")
    f.header.file_name.name = f"{session}_{scan_key.replace('/', '_')}_from_scan.ifc"
    f.header.file_name.time_stamp = ""                         # no clock in the artifact
    key = (session, scan_key)
    origin = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None)
    context = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, origin, None)
    body = f.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, context, None,
                                                        "MODEL_VIEW", None)
    units = f.createIfcUnitAssignment([
        f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"),
        f.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE"),
        f.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE"),
        f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")])
    described = (f"STAC Build — built from the scan {session} {scan_key}, geometry epoch {geometry_epoch}: "
                 f"each element is the stored box of one segmented object (tool_measured)")
    project = f.createIfcProject(_guid(*key, "project"), None, f"{session} {scan_key} (from scan)", described,
                                 None, None, None, [context], units)
    place = lambda rel=None: f.createIfcLocalPlacement(rel, origin)   # noqa: E731
    site = f.createIfcSite(_guid(*key, "site"), None, "Site", None, None, place(), None, None, "ELEMENT",
                           None, None, None, None, None)
    building = f.createIfcBuilding(_guid(*key, "building"), None, session, None, None,
                                   place(site.ObjectPlacement), None, None, "ELEMENT", None, None, None)
    storey = f.createIfcBuildingStorey(_guid(*key, "storey"), None, "Scanned level", None, None,
                                       place(building.ObjectPlacement), None, None, "ELEMENT", 0.0)
    f.createIfcRelAggregates(_guid(*key, "agg", "project"), None, None, None, project, [site])
    f.createIfcRelAggregates(_guid(*key, "agg", "site"), None, None, None, site, [building])
    f.createIfcRelAggregates(_guid(*key, "agg", "building"), None, None, None, building, [storey])

    products = []
    for e in elements:
        T, aabb = e["transform"], e["aabb"]
        lo, hi = _lo_hi(aabb)
        size = hi - lo
        R = T[:3, :3]
        # the box as an extrusion: its base rectangle (local x × local z) at the local y minimum,
        # extruded along the local y (the store's boxes are gravity-aligned: local y is up)
        base = R @ np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) + T[:3, 3]
        # (float(): python floats print their shortest exact form — the same bytes every time)
        at = f.createIfcCartesianPoint([float(v) for v in TO_IFC @ base])
        up = f.createIfcDirection([float(v) for v in TO_IFC @ R[:, 1]])
        ref = f.createIfcDirection([float(v) for v in TO_IFC @ R[:, 0]])
        solid_at = f.createIfcAxis2Placement3D(at, up, ref)
        profile = f.createIfcRectangleProfileDef("AREA", None, None, float(size[0]), float(size[2]))
        solid = f.createIfcExtrudedAreaSolid(profile, solid_at, f.createIfcDirection((0.0, 0.0, 1.0)), float(size[1]))
        shape = f.createIfcProductDefinitionShape(None, None, [
            f.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])])
        cls, predefined = e["ifc"]
        name = f"{e['label']}:{e['label']}:{e['id']}"
        product = f.create_entity(cls, GlobalId=_guid(*key, "element", e["id"]), Name=name,
                                  ObjectType=e["label"], ObjectPlacement=place(storey.ObjectPlacement),
                                  Representation=shape, Tag=str(e["id"]))
        if predefined:
            product.PredefinedType = predefined
        props = [
            f.createIfcPropertySingleValue("InstanceId", None, f.createIfcInteger(e["id"]), None),
            f.createIfcPropertySingleValue("Label", None, f.createIfcLabel(e["label"]), None),
            f.createIfcPropertySingleValue("Points", None, f.createIfcInteger(e["n_points"]), None),
            f.createIfcPropertySingleValue("Scan", None, f.createIfcLabel(f"{session} {scan_key}"), None),
            f.createIfcPropertySingleValue("GeometryEpoch", None, f.createIfcLabel(str(geometry_epoch)), None),
            f.createIfcPropertySingleValue("Provenance", None, f.createIfcLabel(e["origin"]), None),
        ]
        pset = f.createIfcPropertySet(_guid(*key, "pset", e["id"]), None, "STAC_FromScan", None, props)
        f.createIfcRelDefinesByProperties(_guid(*key, "relpset", e["id"]), None, None, None, [product], pset)
        products.append(product)
    if products:
        f.createIfcRelContainedInSpatialStructure(_guid(*key, "contained"), None, None, None, products, storey)
    return f


def write(session: str, scan_key: str, out: Path | str | None = None) -> Dict[str, Any]:
    """Build the IFC of a scan and write it (default: the project's IFC directory). Returns what
    was written: the path, the elements by label and by IFC class."""
    import sqlite3
    from project_paths import resolve_session
    server_dir = Path(__file__).resolve().parent.parent
    ctx = resolve_session(str(server_dir), session, scan_key)
    store = Path(ctx.output_dir) / "scene_r.db"
    if not store.exists():
        raise FileNotFoundError(f"{session} {scan_key}: no instance store ({store}) — the scan is not segmented")
    con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM scene_meta WHERE key = 'geometry_epoch'").fetchone()
    finally:
        con.close()
    elements = elements_of(store)
    if not elements:
        raise ValueError(f"{session} {scan_key}: no object of the segmentation is a building element "
                         f"({sorted(IFC_CLASS)})")
    model = build(elements, session=session, scan_key=scan_key, geometry_epoch=row[0] if row else "?")
    out = Path(out) if out is not None else Path(ctx.ifcs_dir) / f"{session}_{scan_key.replace('/', '_')}_from_scan.ifc"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    model.write(str(tmp))
    tmp.replace(out)
    by_label: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for e in elements:
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
        by_class[e["ifc"][0]] = by_class.get(e["ifc"][0], 0) + 1
    return {"path": str(out), "elements": len(elements), "by_label": by_label, "by_class": by_class,
            "points": sum(e["n_points"] for e in elements)}


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    out = argv[argv.index("--out") + 1] if "--out" in argv else None
    report = write(argv[0], argv[1], out)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
