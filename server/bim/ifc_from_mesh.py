"""AN IFC FROM THE GENRECON MESH, CUT BY THE SEGMENTATION — a scan's as-built model.

USER 2026-11-17, on the first IFC (each object as the box the store holds — "muy grande gordo"):
"por que no exportas el genrecon que esta perfecto, para eso esta" / "asi debe ser ifc desde la
malla genrecon cortada desde la segmentacion, exacto". The boxes were measured too: a wall's box
25 cm thick (median) around points 2 cm thick, the floors' boxes 1.5 m.

  1. the mesh: the scan's published GenRecon GLB, its positions and triangles as they are on disk
     (it is in the scan's display frame — the baker's Y-up);
  2. SIMPLIFICATION of the whole mesh as the one surface it is (``keep``: the share of its
     triangles kept, quadric decimation): an IFC is text and the whole mesh does not open in the
     programs that read it. ``keep = 1`` writes every triangle. What a value costs is MEASURED:
     the distance from the full surface to the simplified one (:func:`deviation`);
  3. the cut, AFTER: a triangle belongs to the object most of the cloud points lying on it
     belong to (:func:`cut` — the cloud and its per-point instance ids are the segmentation's
     own); one with no point farther than one GenRecon voxel from the cloud belongs to none;
  4. one IFC element per object — its part of the mesh as an IfcTriangulatedFaceSet, its class
     from its label (:data:`bim.ifc_from_scan.IFC_CLASS`; any other object is an
     IfcBuildingElementProxy), named ``<label>:<label>:<instance id>`` (the key the comparison
     matches by); the triangles of unsegmented points are ONE proxy, "unsegmented", so the model
     has no holes the mesh does not have;

Coordinates are written Z-up, (x, y, z) → (x, −z, y), rounded to 0.1 mm (the mesh's cells are
4.9 mm): it keeps the file a third smaller than the float32's full digits.
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ifc_from_scan import IFC_CLASS, TO_IFC, _guid

PROXY = ("IfcBuildingElementProxy", None)
UNSEGMENTED = 0
_DECIMALS = 4                      # 0.1 mm
_MIN_FACES = 4                     # a surface simplified under a tetrahedron's faces is none


class MeshError(ValueError):
    pass


def read_glb(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices N×3 float32, triangles M×3 int64) of a binary glTF: every primitive's positions
    and indices, as stored (the nodes of a published GenRecon mesh carry no transform — checked)."""
    with open(path, "rb") as f:
        magic, _version, _total = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise MeshError(f"{path}: not a binary glTF")
        clen, _ctype = struct.unpack("<II", f.read(8))
        doc = json.loads(f.read(clen))
        f.read(8)
        base = f.tell()
        if doc.get("extensionsRequired"):
            raise MeshError(f"{path}: requires {doc['extensionsRequired']} (plain accessors are read)")
        for n in doc.get("nodes", []):
            if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
                raise MeshError(f"{path}: node '{n.get('name')}' carries a transform (a baked mesh is read)")
        acc, views = doc["accessors"], doc["bufferViews"]

        def array(index: int, dtype: str, comps: int) -> np.ndarray:
            a, v = acc[index], views[acc[index]["bufferView"]]
            size = np.dtype(dtype).itemsize * comps
            if v.get("byteStride") not in (None, size):
                raise MeshError(f"{path}: an interleaved buffer view")
            f.seek(base + v.get("byteOffset", 0) + a.get("byteOffset", 0))
            return np.frombuffer(f.read(a["count"] * size), dtype).reshape(-1, comps)

        types = {5123: "<u2", 5125: "<u4"}
        verts, faces, offset = [], [], 0
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                if prim.get("mode", 4) != 4:
                    raise MeshError(f"{path}: a primitive that is not triangles")
                v = array(prim["attributes"]["POSITION"], "<f4", 3)
                idx = acc[prim["indices"]]
                i = array(prim["indices"], types[idx["componentType"]], 1).reshape(-1, 3)
                verts.append(v)
                faces.append(i.astype(np.int64) + offset)
                offset += len(v)
    return np.vstack(verts), np.vstack(faces)


def cut(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, point_ids: np.ndarray,
        max_distance: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The object of each triangle: THE OBJECT MOST OF THE POINTS LYING ON IT BELONG TO.

    Every cloud point lies on its nearest triangle (exact, over the surface); a triangle is the
    object of the majority of its points (a tie: the smaller id). A triangle no point lies on
    takes the object of the cloud point nearest its centre — and belongs to no object (−1) when
    that point is farther than ``max_distance``.

    Why not the nearest point alone (measured on pccr 3108, 2026-11-18): on one surface the points
    of several objects are interleaved, and by the nearest point 20 % of the neighbouring
    triangles fell in different objects — an element came out as tens of thousands of crumbs."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))
    face_of = np.empty(len(points), np.int64)
    for k in range(0, len(points), 4_000_000):
        hit = scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(points[k:k + 4_000_000], np.float32)))
        face_of[k:k + 4_000_000] = hit["primitive_ids"].numpy()
    ids = np.asarray(point_ids, np.int64)
    # (face, id) pairs counted; per face the most frequent id, the smaller on a tie
    order = np.lexsort((ids, face_of))
    fo, io = face_of[order], ids[order]
    start = np.flatnonzero(np.r_[True, (fo[1:] != fo[:-1]) | (io[1:] != io[:-1])])
    count = np.diff(np.r_[start, len(fo)])
    gf, gi = fo[start], io[start]
    best = np.lexsort((gi, -count, gf))
    first = np.r_[True, gf[best][1:] != gf[best][:-1]]
    owner = np.full(len(faces), -2, np.int64)
    owner[gf[best][first]] = gi[best][first]
    empty = np.flatnonzero(owner == -2)
    if len(empty):
        d, k = cKDTree(points).query(vertices[faces[empty]].mean(axis=1), k=1, workers=-1)
        owner[empty] = np.where(d <= max_distance, ids[k], -1)
    return owner, {"faces_without_points": int(len(empty)), "faces_no_object": int((owner == -1).sum())}


def submesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The triangles ``faces`` with their own vertices, WELDED BY POSITION. A textured mesh
    repeats a vertex wherever the texture's charts meet (one copy per chart, the same position):
    by index, an object's surface is tens of thousands of loose pieces (measured on pccr 3108:
    45,793 for one wall, 44,329 of them under 20 triangles) and a simplification deletes them.
    Vertices at the same float32 position are one; a triangle that collapses is dropped."""
    used, inverse = np.unique(faces, return_inverse=True)
    v = np.ascontiguousarray(vertices[used])
    _, first, weld = np.unique(v.view([("", v.dtype)] * 3).ravel(), return_index=True, return_inverse=True)
    f = weld[inverse].reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v[first], f


def simplify(v: np.ndarray, f: np.ndarray, keep: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric decimation to ``keep`` of the triangles (the element as it is when keep = 1 or it
    is too small to simplify)."""
    target = max(_MIN_FACES, int(round(len(f) * keep)))
    if keep >= 1.0 or len(f) <= target:
        return v, f
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)), o3d.utility.Vector3iVector(f.astype(np.int32)))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    m.remove_unreferenced_vertices()
    out_v, out_f = np.asarray(m.vertices), np.asarray(m.triangles).astype(np.int64)
    return (out_v, out_f) if len(out_f) else (v, f)


def deviation(v: np.ndarray, sv: np.ndarray, sf: np.ndarray) -> np.ndarray:
    """The distance from each original vertex ``v`` to the simplified surface (sv, sf), metres."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(sv.astype(np.float32)), o3d.core.Tensor(sf.astype(np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(v.astype(np.float32))).numpy()


def elements_of(output_dir: Path | str, mesh_path: Path | str, *, keep: float, max_distance: float,
                measure: bool = False, log: Callable[[str], Any] = print) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The scan's elements, each with its part of the mesh (display frame), and the report.

    THE ORDER MATTERS: the whole mesh is simplified FIRST, as the one surface it is, and cut
    AFTER. Simplifying each object's part on its own was measured (pccr 3108): the parts' ragged
    borders and loose crumbs are deleted by the decimation — 12 mm of deviation at the 95th
    percentile keeping 25 % of the triangles, metres at the worst."""
    from exporter import cloud as C
    out = Path(output_dir)
    if not (0.0 < keep <= 1.0):
        raise MeshError(f"keep must be in (0, 1], got {keep!r}")
    A, t = C.floor_transform(out)
    if not (np.array_equal(A, np.eye(3)) and not t.any()):
        raise MeshError(f"{out}: the scan has a floor transform — the mesh's frame is not the display frame "
                        f"(an IFC from the mesh is built on a scan whose orientation is baked)")
    raw_v, raw_f = read_glb(mesh_path)
    full_v, full_f = submesh(raw_v, raw_f)                       # welded: one surface
    del raw_v, raw_f
    log(f"mesh: {len(full_v):,} vertices, {len(full_f):,} triangles (welded)")
    vertices, faces = simplify(full_v, full_f, keep)
    log(f"simplified to {len(faces):,} triangles (keep {keep})")
    report: Dict[str, Any] = {"mesh_vertices": int(len(full_v)), "mesh_faces": int(len(full_f)), "keep": float(keep),
                              "faces": int(len(faces)), "max_distance": float(max_distance)}
    if measure and len(faces) != len(full_f):
        d = deviation(full_v, vertices, faces)
        report["deviation_m"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                                 "p99": float(np.percentile(d, 99)), "max": float(d.max())}
    del full_v, full_f
    cloud = C.open_ply(out / "cleaned_cloud.ply")
    ids = np.load(out / "instance_ids.npy")
    if len(ids) != len(cloud):
        raise MeshError(f"{out}: instance_ids.npy holds {len(ids):,} ids for a cloud of {len(cloud):,} points")
    points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    owner, cut_report = cut(vertices, faces, points, ids, max_distance)
    report.update(cut_report)
    log(f"cut: {cut_report}")
    con = sqlite3.connect(f"file:{out / 'scene_r.db'}?mode=ro", uri=True)
    try:
        labels = dict(con.execute("SELECT instance_id, label FROM instances"))
        n_points = dict(con.execute("SELECT instance_id, n_points FROM instance_obb WHERE window_id = 'global'"))
    finally:
        con.close()
    order = np.argsort(owner, kind="stable")
    owners, starts = np.unique(owner[order], return_index=True)
    bounds = list(starts) + [len(order)]
    elements = []
    for k, iid in enumerate(owners):
        iid = int(iid)
        if iid < 0:
            continue
        if iid != UNSEGMENTED and iid not in labels:
            raise MeshError(f"the cloud holds instance {iid}, which the store does not")
        v, f = submesh(vertices, faces[order[bounds[k]:bounds[k + 1]]])
        if not len(f):
            continue
        label = "unsegmented" if iid == UNSEGMENTED else str(labels[iid])
        elements.append({"id": iid, "label": label, "ifc": IFC_CLASS.get(label, PROXY), "vertices": v, "faces": f,
                         "n_points": int(n_points.get(iid, 0))})
    with_faces = {e["id"] for e in elements}
    report.update(elements=len(elements), faces_written=int(sum(len(e["faces"]) for e in elements)),
                  faces_unsegmented=int((owner == UNSEGMENTED).sum()),
                  objects_without_faces=sorted(int(i) for i in set(labels) - with_faces))
    return elements, report


def build(elements: List[Dict[str, Any]], *, session: str, scan_key: str, geometry_epoch: str, described: str):
    """The IFC4 model of ``elements`` (each a triangulated face set)."""
    import ifcopenshell
    f = ifcopenshell.file(schema="IFC4")
    f.header.file_name.name = f"{session}_{scan_key.replace('/', '_')}_from_mesh.ifc"
    f.header.file_name.time_stamp = ""                         # no clock in the artifact
    key = (session, scan_key, "mesh")
    origin = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None)
    context = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, origin, None)
    body = f.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, context, None,
                                                        "MODEL_VIEW", None)
    units = f.createIfcUnitAssignment([
        f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"), f.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE"),
        f.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE"), f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")])
    project = f.createIfcProject(_guid(*key, "project"), None, f"{session} {scan_key} (from the mesh)", described,
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
        xyz = np.round(np.asarray(e["vertices"], np.float64) @ TO_IFC.T, _DECIMALS) + 0.0     # (+ 0.0: no "-0.")
        coords = f.createIfcCartesianPointList3D(xyz.tolist())
        mesh = f.createIfcTriangulatedFaceSet(coords, None, None, (np.asarray(e["faces"]) + 1).tolist(), None)
        shape = f.createIfcProductDefinitionShape(None, None, [
            f.createIfcShapeRepresentation(body, "Body", "Tessellation", [mesh])])
        cls, predefined = e["ifc"]
        product = f.create_entity(cls, GlobalId=_guid(*key, "element", e["id"]), Name=f"{e['label']}:{e['label']}:{e['id']}",
                                  ObjectType=e["label"], ObjectPlacement=place(storey.ObjectPlacement),
                                  Representation=shape, Tag=str(e["id"]))
        if predefined:
            product.PredefinedType = predefined
        props = [
            f.createIfcPropertySingleValue("InstanceId", None, f.createIfcInteger(e["id"]), None),
            f.createIfcPropertySingleValue("Label", None, f.createIfcLabel(e["label"]), None),
            f.createIfcPropertySingleValue("Points", None, f.createIfcInteger(e["n_points"]), None),
            f.createIfcPropertySingleValue("Triangles", None, f.createIfcInteger(int(len(e["faces"]))), None),
            f.createIfcPropertySingleValue("Scan", None, f.createIfcLabel(f"{session} {scan_key}"), None),
            f.createIfcPropertySingleValue("GeometryEpoch", None, f.createIfcLabel(str(geometry_epoch)), None),
            f.createIfcPropertySingleValue("Provenance", None, f.createIfcLabel(
                "shape: vlm_generated (GenRecon mesh); object: the segmentation; label: vlm_proposed"), None),
        ]
        pset = f.createIfcPropertySet(_guid(*key, "pset", e["id"]), None, "STAC_FromMesh", None, props)
        f.createIfcRelDefinesByProperties(_guid(*key, "relpset", e["id"]), None, None, None, [product], pset)
        products.append(product)
    if products:
        f.createIfcRelContainedInSpatialStructure(_guid(*key, "contained"), None, None, None, products, storey)
    return f


def voxel_of(config_cube_m: Optional[float] = None) -> float:
    """One GenRecon voxel (the cube / 32): how far from the cloud a triangle can be and still be
    the cloud's (config.yaml genrecon.cube_size_m — the user's)."""
    if config_cube_m is None:
        from reconstruction.grid_knn import config_section
        config_cube_m = float(config_section(("genrecon",), ("cube_size_m",))["cube_size_m"])
    return float(config_cube_m) / 32.0


def mesh_of(output_dir: Path | str) -> Path:
    """The scan's published GenRecon mesh (the binary glTF its meta names)."""
    folder = Path(output_dir) / "tsdf" / "genrecon_photo"
    meta = folder / "genrecon_photo.meta.json"
    name = json.loads(meta.read_text()).get("glb_file") if meta.exists() else None
    glb = folder / str(name or "genrecon_photo.glb")
    if not glb.exists():
        raise MeshError(f"{output_dir}: no published GenRecon mesh ({glb})")
    return glb


def write(session: str, scan_key: str, *, keep: float, out: Path | str | None = None, measure: bool = True,
          log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Build the scan's IFC from its GenRecon mesh and write it (default: the project's IFC
    directory). Returns the report: elements, triangles, the simplification's measured deviation."""
    from project_paths import resolve_session
    server_dir = Path(__file__).resolve().parent.parent
    ctx = resolve_session(str(server_dir), session, scan_key)
    output_dir = Path(ctx.output_dir)
    con = sqlite3.connect(f"file:{output_dir / 'scene_r.db'}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM scene_meta WHERE key = 'geometry_epoch'").fetchone()
    finally:
        con.close()
    epoch = str(row[0]) if row else "?"
    elements, report = elements_of(output_dir, mesh_of(output_dir), keep=keep, max_distance=voxel_of(),
                                   measure=measure, log=log)
    dev = report.get("deviation_m")
    described = (f"STAC Build — built from the GenRecon mesh of the scan {session} {scan_key}, geometry epoch {epoch}, "
                 f"cut by its segmentation: each element is its object's part of the mesh. Triangles kept: "
                 f"{report['faces']:,} of {report['mesh_faces']:,}"
                 + (f"; distance from the full mesh: median {dev['median'] * 1000:.2f} mm, 95 % under {dev['p95'] * 1000:.2f} mm"
                    if dev else ""))
    model = build(elements, session=session, scan_key=scan_key, geometry_epoch=epoch, described=described)
    out = Path(out) if out is not None else Path(ctx.ifcs_dir) / f"{session}_{scan_key.replace('/', '_')}_from_mesh.ifc"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    model.write(str(tmp))
    tmp.replace(out)
    by_class: Dict[str, int] = {}
    for e in elements:
        by_class[e["ifc"][0]] = by_class.get(e["ifc"][0], 0) + 1
    report.update(path=str(out), bytes=out.stat().st_size, by_class=by_class)
    return report


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bim.ifc_from_mesh")
    ap.add_argument("session")
    ap.add_argument("scan")
    ap.add_argument("--keep", type=float, required=True, help="the share of the mesh's triangles kept, (0, 1]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(write(a.session, a.scan, keep=a.keep, out=a.out), default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
