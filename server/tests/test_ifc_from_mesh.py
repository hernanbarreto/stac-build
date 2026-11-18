"""An IFC from the GenRecon mesh cut by the segmentation (USER 2026-11-17: "asi debe ser ifc desde
la malla genrecon cortada desde la segmentacion, exacto"): the mesh welded into the surface it
is, simplified as a whole, each triangle given to the object most of the points lying on it
belong to, one IFC element per object with its real shape."""
from __future__ import annotations

import numpy as np
import pytest

from bim import ifc_from_mesh as M
from phase_r.instance_store import InstanceStore

trimesh = pytest.importorskip("trimesh")
ifcopenshell = pytest.importorskip("ifcopenshell")
pytest.importorskip("open3d")

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                  ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])


def _scan(tmp_path, noise=0.0):
    """A wall (object 1) and a chair (object 5) as two subdivided boxes in one GLB whose vertices
    are NOT shared between triangles (as a textured mesh's charts); the cloud = points on them."""
    rng = np.random.default_rng(3)
    wall = trimesh.creation.box(extents=[4.0, 2.6, 0.1]).subdivide().subdivide().subdivide()
    wall.apply_translation([0, 1.3, 2])
    chair = trimesh.creation.box(extents=[0.5, 0.9, 0.5]).subdivide().subdivide()
    chair.apply_translation([0, 0.45, 0])
    out = tmp_path / "output"
    (out / "tsdf" / "genrecon_photo").mkdir(parents=True)
    both = trimesh.util.concatenate([wall, chair])
    soup = trimesh.Trimesh(vertices=both.vertices[both.faces].reshape(-1, 3),
                           faces=np.arange(len(both.faces) * 3).reshape(-1, 3), process=False)
    glb = out / "tsdf" / "genrecon_photo" / "genrecon_photo.glb"
    glb.write_bytes(trimesh.exchange.gltf.export_glb(soup))
    pts, ids = [], []
    for mesh, iid, n in ((wall, 1, 20000), (chair, 5, 6000)):
        p, _ = trimesh.sample.sample_surface(mesh, n, seed=iid)
        pts.append(p + rng.normal(0, noise, p.shape) if noise else p)
        ids.append(np.full(n, iid))
    xyz, ids = np.vstack(pts), np.concatenate(ids).astype(np.uint32)
    rec = np.zeros(len(xyz), DTYPE)
    rec["x"], rec["y"], rec["z"] = xyz.T
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rec)}"]
    head += [f"property {'float' if DTYPE[k].kind == 'f' else {'u1': 'uchar', 'i4': 'int', 'i2': 'short'}[DTYPE[k].str[1:]]} {k}" for k in DTYPE.names]
    (out / "cleaned_cloud.ply").write_bytes(("\n".join(head + ["end_header"]) + "\n").encode() + rec.tobytes())
    np.save(out / "instance_ids.npy", ids)
    store = InstanceStore(out / "scene_r.db")
    for iid, label in ((1, "wall"), (5, "chair")):
        store.upsert_instance(iid, label, source="test")
        store.set_obb(iid, np.eye(4), np.array([-1, 1, -1, 1, -1, 1], np.float32), np.zeros(3), n_points=int((ids == iid).sum()))
    store.close()
    return out, glb, wall, chair


def test_the_mesh_is_read_and_welded_into_its_surface(tmp_path):
    _out, glb, wall, chair = _scan(tmp_path)
    v, f = M.read_glb(glb)
    assert len(f) == len(wall.faces) + len(chair.faces) and len(v) == 3 * len(f)       # a triangle soup on disk
    wv, wf = M.submesh(v, f)
    assert len(wf) == len(f) and len(wv) == len(wall.vertices) + len(chair.vertices)   # one vertex per position
    area = lambda vv, ff: np.linalg.norm(np.cross(vv[ff[:, 1]] - vv[ff[:, 0]], vv[ff[:, 2]] - vv[ff[:, 0]]), axis=1).sum() / 2
    assert area(wv, wf) == pytest.approx(wall.area + chair.area, rel=1e-5)


def test_each_object_gets_its_part_of_the_mesh(tmp_path):
    out, glb, wall, chair = _scan(tmp_path)
    elements, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, log=lambda _m: None)
    assert [(e["id"], e["label"], e["ifc"][0]) for e in elements] == [(1, "wall", "IfcWall"), (5, "chair", "IfcBuildingElementProxy")]
    assert [len(e["faces"]) for e in elements] == [len(wall.faces), len(chair.faces)]
    assert report["faces_no_object"] == 0 and report["objects_without_faces"] == [] and report["faces_written"] == report["mesh_faces"]
    model = M.build(elements, session="s", scan_key="k/default", geometry_epoch="0", described="test")
    path = tmp_path / "m.ifc"
    model.write(str(path))
    f = ifcopenshell.open(str(path))
    products = {int(p.Tag): p for p in f.by_type("IfcElement")}
    assert products[1].Name == "wall:wall:1" and products[5].is_a("IfcBuildingElementProxy")
    mesh = products[1].Representation.Representations[0].Items[0]
    assert mesh.is_a("IfcTriangulatedFaceSet") and len(mesh.CoordIndex) == len(wall.faces)
    xyz = np.array(mesh.Coordinates.CoordList)
    # Z up, (x, y, z) → (x, −z, y): the wall stands from z = 0 to 2.6 at y = −2
    assert xyz[:, 2].min() == pytest.approx(0.0, abs=1e-4) and xyz[:, 2].max() == pytest.approx(2.6, abs=1e-4)
    assert xyz[:, 1].min() == pytest.approx(-2.05, abs=1e-4) and np.array(mesh.CoordIndex).min() == 1
    from bim.comparison import _build_name_index
    assert sorted(_build_name_index(f)) == ["1", "5"]
    # the same scan gives the same bytes
    M.build(elements, session="s", scan_key="k/default", geometry_epoch="0", described="test").write(str(tmp_path / "n.ifc"))
    assert path.read_bytes() == (tmp_path / "n.ifc").read_bytes()


def test_interleaved_points_do_not_crumble_the_cut(tmp_path):
    """On one surface the points of two objects can be interleaved: a triangle goes to the object
    of MOST of the points lying on it, not of the single nearest one."""
    out, glb, wall, _chair = _scan(tmp_path)
    ids = np.load(out / "instance_ids.npy")
    rng = np.random.default_rng(0)
    on_wall = np.flatnonzero(ids == 1)
    ids[rng.choice(on_wall, len(on_wall) // 5, replace=False)] = 5          # a fifth of the wall's points say "chair"
    np.save(out / "instance_ids.npy", ids)
    elements, _report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, log=lambda _m: None)
    wall_faces = len(next(e for e in elements if e["id"] == 1)["faces"])
    # … and by a neighbourhood of nearest points the wall is whole
    wide, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, neighbours=32, log=lambda _m: None)
    assert len(next(e for e in wide if e["id"] == 1)["faces"]) == len(wall.faces) and report["neighbours"] == 32
    # (the nearest point alone would give it about 80 %: the share of its points that still say "wall")
    assert wall_faces >= 0.9 * len(wall.faces)


def test_simplifying_keeps_the_surface_and_what_is_far_from_the_cloud_is_no_object(tmp_path):
    out, glb, wall, chair = _scan(tmp_path)
    elements, report = M.elements_of(out, glb, keep=0.25, max_distance=0.05, measure=True, log=lambda _m: None)
    assert report["faces"] <= 0.26 * report["mesh_faces"] and report["deviation_m"]["max"] < 1e-3      # flat faces: no loss
    assert {e["id"] for e in elements} == {1, 5}
    # the cloud without the chair's points: the chair's triangles are farther than max_distance from it
    ids = np.load(out / "instance_ids.npy")
    from exporter import cloud as C
    data = C.open_ply(out / "cleaned_cloud.ply")
    keep = ids == 1
    head = (out / "cleaned_cloud.ply").read_bytes().split(b"end_header\n")[0].replace(
        f"element vertex {len(ids)}".encode(), f"element vertex {int(keep.sum())}".encode()) + b"end_header\n"
    rec = np.asarray(data)[keep].tobytes()
    del data
    (out / "cleaned_cloud.ply").write_bytes(head + rec)
    np.save(out / "instance_ids.npy", ids[keep])
    elements, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, log=lambda _m: None)
    assert [e["id"] for e in elements] == [1] and report["faces_no_object"] == len(chair.faces)
    assert report["objects_without_faces"] == [5]
