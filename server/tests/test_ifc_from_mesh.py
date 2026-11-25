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


def test_large_triangles_are_split_in_their_plane():
    v = np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0], [0, 0, 0.1]], np.float32)
    f = np.array([[0, 1, 2], [0, 1, 3]])                       # a 4 x 3 m triangle and a 4 m x 10 cm sliver
    area = lambda vv, ff: np.linalg.norm(np.cross(vv[ff[:, 1]] - vv[ff[:, 0]], vv[ff[:, 2]] - vv[ff[:, 0]]), axis=1).sum() / 2
    nv, nf = M.split_large(v, f, 0.5)
    assert np.linalg.norm(nv[nf] - nv[nf[:, [1, 2, 0]]], axis=2).max() <= 0.5 and len(nf) > 60
    assert area(nv, nf) == pytest.approx(area(v, f), rel=1e-6)                  # the same surface
    assert M.split_large(v, f, 10.0)[1] is f                                    # nothing over the size: untouched


def test_every_object_appears_even_one_the_majority_leaves_out(tmp_path):
    """USER 2026-11-18: "deben aparecer todos los elementos". A small object on a surface another
    one dominates loses every triangle to the majority of the neighbours — it takes the triangles
    its own points lie on."""
    out, glb, wall, _chair = _scan(tmp_path)
    ids = np.load(out / "instance_ids.npy")
    from exporter import cloud as C
    data = C.open_ply(out / "cleaned_cloud.ply")
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    del data
    # a "label" stuck on the wall: one point in ten of a 30 cm patch
    patch = np.flatnonzero((ids == 1) & (np.abs(xyz[:, 0] - 1.0) < 0.15) & (np.abs(xyz[:, 1] - 1.3) < 0.15))
    ids[patch[::10]] = 9
    np.save(out / "instance_ids.npy", ids)
    store = InstanceStore(out / "scene_r.db")
    store.upsert_instance(9, "label", source="test")
    store.set_obb(9, np.eye(4), np.array([-1, 1, -1, 1, -1, 1], np.float32), np.zeros(3), n_points=int((ids == 9).sum()))
    store.close()
    elements, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, neighbours=32, log=lambda _m: None)
    assert sorted(e["id"] for e in elements) == [1, 5, 9] and report["objects_without_faces"] == []
    assert report["objects_left_out_by_the_majority"] == 1 and report["objects_rescued"] == 1
    label = next(e for e in elements if e["id"] == 9)
    assert 0 < len(label["faces"]) < 0.1 * len(wall.faces)


def test_a_floor_adjustment_reaches_the_ifc(tmp_path):
    """USER 2026-11-19 (bufferStop's export was refused: "the scan has a floor transform"): "dale, si,
    hacelo siempre" / "cuando se aplica ajuste de piso debe alcanzar a las mallas tambien, total es
    una rotacion". The mesh and the cloud on disk are in one frame; the floor adjustment the viewer
    applies to both is applied to every element — the same elements, moved by it."""
    out, glb, _wall, _chair = _scan(tmp_path)
    plain, plain_report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, log=lambda m: None)
    assert plain_report["floor_adjusted"] is False
    c, s_ = np.cos(np.radians(7.0)), np.sin(np.radians(7.0))
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
    t = np.array([0.0, 0.25, -0.1])
    np.savez(out / "floor_transform.npz", s=np.float64(1.0), R=R, t=t)
    moved, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, log=lambda m: None)
    assert report["floor_adjusted"] is True and [e["id"] for e in moved] == [e["id"] for e in plain]
    for a, b in zip(plain, moved):
        assert np.array_equal(a["faces"], b["faces"])
        assert np.allclose(b["vertices"], np.asarray(a["vertices"], np.float64) @ R.T + t, atol=1e-12)
    # … and the IFC is written from them (it used to be refused)
    text = M.build(moved, session="s", scan_key="2026-01-01/default", geometry_epoch="0", described="t").to_string()
    assert text.count("IFCTRIANGULATEDFACESET") == 2



def _textured_scan(tmp_path):
    """_scan's wall and chair, each with its own atlas image (a red wall, a green chair) and texture coordinates that
    follow the position (so a corner's coordinates can be checked from where it is)."""
    from PIL import Image
    out, glb, wall, chair = _scan(tmp_path)

    def textured(mesh, rgb):
        soup = trimesh.Trimesh(vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
                               faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3), process=False)
        vv = soup.vertices
        uv = np.column_stack([(vv[:, 0] - vv[:, 0].min()) / np.ptp(vv[:, 0]), (vv[:, 1] - vv[:, 1].min()) / np.ptp(vv[:, 1])])
        soup.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new("RGB", (8, 8), rgb))
        return soup
    glb.write_bytes(trimesh.exchange.gltf.export_glb(trimesh.Scene([textured(wall, (200, 30, 30)),
                                                                     textured(chair, (30, 180, 60))])))
    return out, glb


def test_the_genrecon_texture_reaches_the_ifc_with_each_elements_dominant_colour(tmp_path):
    """USER 2026-11-24: "es posible generar los IFC con la textura de genrecon?" / "proba mejor con la textura real
    primero si no sale vamos por el color dominante"."""
    import ifcopenshell.validate
    out, glb = _textured_scan(tmp_path)
    src = M.read_glb_textured(glb)
    assert src["uv"] is not None and len(src["images"]) == 2
    # glTF's coordinates of each position (they follow the position: one per position)
    want = {tuple(np.round(src["vertices"][src["faces"][k, c]], 4)): src["uv"][k, c] for k in range(len(src["faces"])) for c in range(3)}
    elements, report = M.elements_of(out, glb, keep=1.0, max_distance=0.05, max_edge=0.3)   # the large triangles split
    assert report["texture"]["materials_used"] == 2 and report["faces_after_split"] > report["faces"]
    assert all("uv" in e and len(e["uv"]) == len(e["faces"]) for e in elements)
    M.element_colours(elements, report["texture"])
    colour = {e["label"]: e["colour"] for e in elements}
    assert np.allclose(colour["wall"], (200 / 255, 30 / 255, 30 / 255), atol=1e-6)            # the dominant colour
    assert np.allclose(colour["chair"], (30 / 255, 180 / 255, 60 / 255), atol=1e-6)
    path = tmp_path / "scan_elements.ifc"
    urls = M.write_textures(elements, report["texture"], tmp_path / "scan_elements_textures")
    M.build(elements, session="s", scan_key="2026-01-01/default", geometry_epoch="0", described="test",
            textures=urls).write(str(path))
    assert sorted(urls.values()) == sorted(f"scan_elements_textures/{p.name}" for p in (tmp_path / "scan_elements_textures").iterdir())
    f = ifcopenshell.open(str(path))
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, logger)
    assert logger.statements == []                                                            # a valid IFC4 file
    maps = f.by_type("IfcIndexedTriangleTextureMap")
    assert len(maps) >= 2 and {t.URLReference for t in f.by_type("IfcImageTexture")} == set(urls.values())
    checked = 0
    for tm in maps:
        fs = tm.MappedTo
        xyz = np.asarray(fs.Coordinates.CoordList)
        disp = np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])                           # IFC's Z up back to the display frame
        st = np.asarray(tm.TexCoords.TexCoordsList)
        assert len(tm.TexCoordIndex) == len(fs.CoordIndex)
        for tri, ti in zip(fs.CoordIndex, tm.TexCoordIndex):
            for c in range(3):
                p = disp[tri[c] - 1]
                s, t = st[ti[c] - 1]
                key = tuple(np.round(p, 4))
                if key in want:                                                               # a corner of the mesh as stored
                    u, v = want[key]
                    assert abs(s - u) < 2e-5 and abs(t - (1.0 - v)) < 2e-5                    # IFC's t is up: 1 − glTF's v
                    checked += 1
    assert checked > 100
    # the split triangles' new corners lie in their parent: their coordinates in the parent's span
    assert all(0.0 - 1e-6 <= x <= 1.0 + 1e-6 for tm in maps for row in tm.TexCoords.TexCoordsList for x in row)
    # each face set's style: its texture over its element's colour
    for item in f.by_type("IfcStyledItem"):
        layers = item.Styles[0].Styles
        assert {l.is_a() for l in layers} == {"IfcSurfaceStyleShading", "IfcSurfaceStyleWithTextures"}


def test_a_simplified_mesh_carries_no_texture(tmp_path):
    out, glb = _textured_scan(tmp_path)
    elements, report = M.elements_of(out, glb, keep=0.5, max_distance=0.05)
    assert report["texture"] is None and not any("uv" in e for e in elements)


def test_a_simplified_mesh_takes_each_objects_colour_from_its_cloud_points(tmp_path):
    """The IFC the viewer loads is the light one (USER 2026-11-18): its triangles carry no texture, its elements get
    their objects' dominant colour from their own points."""
    out, glb = _textured_scan(tmp_path)
    from exporter import cloud as C
    data = C.open_ply(out / "cleaned_cloud.ply")
    ids = np.load(out / "instance_ids.npy")
    raw = (out / "cleaned_cloud.ply").read_bytes()
    head, _ = raw.split(b"end_header\n", 1)
    rec = np.array(data)
    rec["red"], rec["green"], rec["blue"] = np.where(ids == 1, 210, 20), np.where(ids == 1, 40, 170), 30
    rec["red"][::50] = 255                                                          # a few points of another colour
    (out / "cleaned_cloud.ply").write_bytes(head + b"end_header\n" + rec.tobytes())
    elements, report = M.elements_of(out, glb, keep=0.5, max_distance=0.05)
    assert report["texture"] is None
    assert M.colour_elements(elements, None, out) == "cloud"
    colour = {e["label"]: e["colour"] for e in elements}
    assert np.allclose(colour["wall"], (210 / 255, 40 / 255, 30 / 255), atol=1e-6)
    assert np.allclose(colour["chair"], (20 / 255, 170 / 255, 30 / 255), atol=1e-6)
    path = tmp_path / "light.ifc"
    M.build(elements, session="s", scan_key="2026-01-01/default", geometry_epoch="0", described="test").write(str(path))
    f = ifcopenshell.open(str(path))
    assert not f.by_type("IfcIndexedTriangleTextureMap") and len(f.by_type("IfcSurfaceStyleShading")) == len(elements)
