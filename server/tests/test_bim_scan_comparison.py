"""The BIM comparison on a scan of today (bim/scan_data.py, bim/comparison.py).

USER 2026-11-17: "create un IFC vos a partir de PCCR3108, para comparar contra ese mismo, porque es
simple". The comparison was March's: segments labelled by hand with their BIM element's number,
clouds of a million points, the nearest of 8 triangle centroids as "the distance". What is held
here: the objects of a scan are matched by INSTANCE ID to an IFC that STAC wrote from it (and by
label to any other, as before), the distance to the mesh is EXACT, the same comparison gives the
same numbers twice, and its answer carries no point.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bim import comparison as C  # noqa: E402
from bim.scan_data import Scan, ScanError, element_keys, first_ifc, stac_source  # noqa: E402

ifcopenshell = pytest.importorskip("ifcopenshell")


# ── the distance ───────────────────────────────────────────────────────────────────────────────
def brute(points, verts, faces):
    """Every point against every triangle: the distance by definition."""
    tri = verts[faces]
    best = np.full(len(points), np.inf)
    for a, b, c in tri:
        n = len(points)
        best = np.minimum(best, C._point_to_tri_batch(points, np.tile(a, (n, 1)), np.tile(b, (n, 1)), np.tile(c, (n, 1))))
    return best


def nearest_8_centroids(points, verts, faces):
    """What was "the distance" until 2026-11-18."""
    from scipy.spatial import cKDTree
    tri = verts[faces]
    _, idx = cKDTree(tri.mean(axis=1)).query(points, k=min(8, len(faces)))
    idx = idx.reshape(len(points), -1)
    best = np.full(len(points), np.inf)
    for j in range(idx.shape[1]):
        t = tri[idx[:, j]]
        best = np.minimum(best, C._point_to_tri_batch(points, t[:, 0], t[:, 1], t[:, 2]))
    return best


def box(lo, hi):
    """A solid as a BIM tool writes it: 8 corners, 12 triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 5, 1], [0, 4, 5],
                  [1, 6, 2], [1, 5, 6], [2, 7, 3], [2, 6, 7], [3, 4, 0], [3, 7, 4]], dtype=np.int32)
    return v, f


def grid(n, size, y=0.0):
    """A horizontal square of n × n cells, two triangles each (a fine, even mesh)."""
    xs = np.linspace(0, size, n + 1)
    gx, gz = np.meshgrid(xs, xs, indexing="ij")
    v = np.stack([gx.ravel(), np.full(gx.size, y), gz.ravel()], axis=1)
    i = np.arange(n * (n + 1)).reshape(n, n + 1)[:, :-1].ravel()
    f = np.concatenate([np.stack([i, i + 1, i + n + 1], axis=1), np.stack([i + 1, i + n + 2, i + n + 1], axis=1)])
    return v, f.astype(np.int32)


def test_the_distance_is_exact_on_a_triangle_soup():
    rng = np.random.default_rng(1)
    verts = rng.uniform(-2, 2, (300, 3))
    faces = rng.integers(0, 300, (500, 3)).astype(np.int32)
    points = rng.uniform(-3, 3, (4000, 3))
    assert np.abs(C.compute_c2m_distances(points, verts, faces, quiet=True, device="numpy") - brute(points, verts, faces)).max() < 1e-12


def cases():
    """The meshes the distance is held on: a soup, a pipe, a fine mesh with huge triangles in it, a
    shell 4 cm off its points (pccr's floor: tens of triangles tried per point), degenerate triangles."""
    rng = np.random.default_rng(21)
    soup_v = rng.uniform(-2, 2, (300, 3))
    yield "soup", rng.uniform(-3, 3, (3000, 3)), soup_v, rng.integers(0, 300, (500, 3)).astype(np.int32)
    pv, pf = pipe()
    ang = rng.uniform(0, 2 * np.pi, 3000)
    off = 0.1 + rng.uniform(0.0, 0.03, 3000)
    yield "pipe", np.stack([rng.uniform(-0.2, 6.2, 3000), off * np.cos(ang), off * np.sin(ang)], axis=1), pv, pf
    fine_v, fine_f = grid(40, 2.0)
    big_v, big_f = box((-20.0, -3.0, -20.0), (20.0, -2.9, 20.0))
    yield ("mixed", rng.uniform(-6, 6, (3000, 3)), np.concatenate([fine_v, big_v]),
           np.concatenate([fine_f, big_f + len(fine_v)]).astype(np.int32))
    up_v, up_f = grid(60, 3.0, y=0.04)
    down_v, down_f = grid(60, 3.0, y=-0.04)
    yield ("shell", np.stack([rng.uniform(0, 3, 6000), rng.normal(0, 0.004, 6000), rng.uniform(0, 3, 6000)], axis=1),
           np.concatenate([up_v, down_v]), np.concatenate([up_f, down_f + len(up_v)]).astype(np.int32))
    deg_v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0.5, 0.5, 0.5]], dtype=np.float64)
    yield "degenerate", rng.uniform(-1, 3, (2000, 3)), deg_v, np.array([[0, 1, 2], [0, 0, 0], [0, 1, 3], [4, 4, 1]], dtype=np.int32)


def test_the_torch_path_gives_the_record_distances():
    """bim/c2m_torch.py on the CPU against numpy — the path of record: 1e-9 is the rule, and the
    same triangles tried in the same order give the same bits."""
    for name, points, verts, faces in cases():
        record = C.compute_c2m_distances(points, verts, faces, quiet=True, device="numpy")
        twin = C.compute_c2m_distances(points, verts, faces, quiet=True, device="cpu")
        assert np.abs(twin - record).max() < 1e-9, name
        assert np.array_equal(twin, record), name
        assert np.abs(record - brute(points, verts, faces)).max() < 1e-12, name


def test_the_card_gives_the_record_distances():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no card visible")
    for name, points, verts, faces in cases():
        record = C.compute_c2m_distances(points, verts, faces, quiet=True, device="numpy")
        card = C.compute_c2m_distances(points, verts, faces, quiet=True, device="cuda")
        assert np.abs(card - record).max() < 1e-9, name
        assert np.array_equal(card, record), name


def test_the_device_is_a_declared_one():
    from bim.c2m_torch import resolve_device
    with pytest.raises(ValueError, match="bim.deviation.device must be one of"):
        resolve_device("gpu")
    assert C.device_of() == "cpu"                                             # (conftest: the tests run the torch path on the CPU)
    before = C.use_device(None)
    try:
        assert C.device_of() in ("numpy", "cpu", "cuda")                      # the configuration's
    except RuntimeError as e:                                                 # cuda declared, no card visible here: said
        assert "bim.deviation.device is 'cuda' and torch sees no card" in str(e)
    finally:
        C.use_device("cpu")


def pipe(sides=16, radius=0.1, length=6.0):
    """A pipe as a BIM tool writes it: long triangles along it, a fan of small ones on each end."""
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.stack([np.zeros(sides), radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    v = np.concatenate([ring, ring + [length, 0, 0], [[0, 0, 0]], [[length, 0, 0]]])
    i = np.arange(sides)
    j = (i + 1) % sides
    f = np.concatenate([np.stack([i, j, j + sides], axis=1), np.stack([i, j + sides, i + sides], axis=1),      # the side
                        np.stack([np.full(sides, 2 * sides), j, i], axis=1),                                   # the ends
                        np.stack([np.full(sides, 2 * sides + 1), i + sides, j + sides], axis=1)])
    return v, f.astype(np.int32)


def test_the_distance_is_exact_on_a_solid_of_long_triangles():
    """A pipe 6 m long: beside it near an end, the 8 nearest CENTROIDS are all of the end's small
    triangles — the long one the point lies against has its centroid metres away. The nearest 8
    centroids gave a distance that was not the distance."""
    verts, faces = pipe()
    rng = np.random.default_rng(2)
    ang = rng.uniform(0, 2 * np.pi, 4000)
    off = 0.1 + rng.uniform(0.0, 0.03, 4000)                                  # up to 3 cm off its side
    points = np.stack([rng.uniform(-0.2, 6.2, 4000), off * np.cos(ang), off * np.sin(ang)], axis=1)
    exact = brute(points, verts, faces)
    assert np.abs(C.compute_c2m_distances(points, verts, faces, quiet=True) - exact).max() < 1e-12
    old = nearest_8_centroids(points, verts, faces)
    wrong = (old - exact) > 1e-6
    assert wrong.mean() > 0.02 and (old - exact).max() > 0.02                 # centimetres too far, near both ends
    assert (old - exact).min() > -1e-12                                       # (it could only say farther than the truth)
    # a wall of 12 triangles too
    verts, faces = box((0.0, 0.0, 0.0), (10.0, 3.0, 0.2))
    points = rng.uniform(-1, 11, (3000, 3)) * np.array([1.0, 0.4, 0.1])
    assert np.abs(C.compute_c2m_distances(points, verts, faces, quiet=True) - brute(points, verts, faces)).max() < 1e-12


def test_the_distance_is_exact_when_huge_triangles_sit_in_a_fine_mesh():
    fine_v, fine_f = grid(40, 2.0)
    big_v, big_f = box((-20.0, -3.0, -20.0), (20.0, -2.9, 20.0))
    verts = np.concatenate([fine_v, big_v])
    faces = np.concatenate([fine_f, big_f + len(fine_v)]).astype(np.int32)
    rng = np.random.default_rng(3)
    points = np.concatenate([np.stack([rng.uniform(0, 2, 2000), rng.normal(0, 0.01, 2000), rng.uniform(0, 2, 2000)], axis=1),
                             rng.uniform(-6, 6, (1500, 3))])
    got = C.compute_c2m_distances(points, verts, faces, quiet=True)
    assert np.abs(got - brute(points, verts, faces)).max() < 1e-12
    # in passes of a few points, and with a wider first look: the same distances
    for chunk, first in ((97, 1), (1000, 3)):
        old = C.C2M_CHUNK, C.C2M_FIRST_K
        C.C2M_CHUNK, C.C2M_FIRST_K = chunk, first
        try:
            assert np.array_equal(C.compute_c2m_distances(points, verts, faces, quiet=True), got)
        finally:
            C.C2M_CHUNK, C.C2M_FIRST_K = old


def test_no_points_and_no_mesh():
    verts, faces = grid(2, 1.0)
    assert len(C.compute_c2m_distances(np.empty((0, 3)), verts, faces, quiet=True)) == 0
    with pytest.raises(ValueError, match="no triangle"):
        C.compute_c2m_distances(np.zeros((3, 3)), verts, np.empty((0, 3), dtype=np.int32), quiet=True)


# ── the coverage ───────────────────────────────────────────────────────────────────────────────
def test_the_coverage_is_the_same_twice_and_is_the_covered_area():
    verts, faces = grid(20, 10.0)                                             # 100 m²
    rng = np.random.default_rng(4)
    half = np.stack([rng.uniform(0, 5, 40000), np.zeros(40000), rng.uniform(0, 10, 40000)], axis=1)   # points on x < 5
    a = C.compute_coverage_pct(verts, faces, half, 0.15, seed=7, quiet=True)
    assert a == C.compute_coverage_pct(verts, faces, half, 0.15, seed=7, quiet=True)
    assert 49.0 <= a <= 53.5                                                  # (half the plate, + 15 cm of reach)
    assert C.compute_coverage_pct(verts, faces, half[:0], 0.15, quiet=True) == 0.0
    # an element of many small triangles is sampled once per triangle, 50,000 at most, weighted by area
    fine_v, fine_f = grid(200, 10.0)                                          # 80,000 triangles
    b = C.compute_coverage_pct(fine_v, fine_f, half, 0.15, seed=7, quiet=True)
    assert abs(b - a) < 1.5


# ── a scan of today, and one of March ──────────────────────────────────────────────────────────
def write_ply(path: Path, xyz: np.ndarray) -> None:
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path.write_bytes(header.encode("ascii") + rec.tobytes())


def write_store(path: Path, instances: dict) -> None:
    con = sqlite3.connect(path)
    con.execute("create table instances (instance_id integer primary key, label text)")
    con.execute("create table instance_obb (instance_id integer, window_id text, n_points integer)")
    for iid, (label, n) in instances.items():
        con.execute("insert into instances values (?, ?)", (iid, label))
        con.execute("insert into instance_obb values (?, 'global', ?)", (iid, n))
    con.commit()
    con.close()


FLOOR, WALL, RACK = 3, 7, 12                                                  # instance ids


def make_project(root: Path, session: str = "site", offset: float = 0.020, stac_ifc: bool = True, store: bool = True) -> Path:
    """A project with one scan — a floor 4 × 4 m, a wall, a rack (two objects labelled 'rack' in the
    March variant) — and its IFC. The scan's points lie ``offset`` off the floor's surface."""
    from bim.ifc_from_mesh import build
    scan_dir = root / "projects" / session / "scans" / "2026-01-02" / "src_default"
    out = scan_dir / "output"
    out.mkdir(parents=True)
    (root / "projects" / session / "ifcs").mkdir()
    (root / "projects" / session / "project.json").write_text(json.dumps({"name": session}))
    rng = np.random.default_rng(11)
    floor_v, floor_f = grid(8, 4.0)                                           # y = 0, display frame (Y up)
    wall_v, wall_f = box((0.0, 0.0, -0.2), (4.0, 2.5, 0.0))
    rack_v, rack_f = box((1.0, 0.0, 1.0), (1.6, 2.0, 2.0))
    floor_pts = np.stack([rng.uniform(0, 4, 6000), np.full(6000, offset), rng.uniform(0, 4, 6000)], axis=1)
    wall_pts = np.stack([rng.uniform(0, 4, 3000), rng.uniform(0, 2.5, 3000), np.zeros(3000)], axis=1)       # on its face
    rack_pts = np.stack([np.full(2000, 1.6), rng.uniform(0, 2, 2000), rng.uniform(1, 2, 2000)], axis=1)     # one side seen
    loose = rng.uniform(5, 6, (500, 3))                                       # unsegmented
    xyz = np.concatenate([loose, floor_pts, wall_pts, rack_pts]).astype(np.float32)
    ids = np.concatenate([np.zeros(500), np.full(6000, FLOOR), np.full(3000, WALL), np.full(2000, RACK)]).astype(np.uint32)
    perm = rng.permutation(len(xyz))                                          # (points of an object are not contiguous)
    xyz, ids = xyz[perm], ids[perm]
    write_ply(out / "cleaned_cloud.ply", xyz)
    labels = {FLOOR: "floor", WALL: "wall", RACK: "rack"}
    if store:
        np.save(out / "instance_ids.npy", ids)
        write_store(out / "scene_r.db", {i: (labels[i], int((ids == i).sum())) for i in labels})
    else:                                                                     # March: the result file, labels = BIM numbers
        numbers = {FLOOR: "292127", WALL: "292200", RACK: "292300"}
        instances = [{"id": i, "label": numbers[i], "globalIndices": np.flatnonzero(ids == i).tolist()} for i in (FLOOR, WALL)]
        rack = np.flatnonzero(ids == RACK)                                    # the rack as TWO objects with one label
        instances += [{"id": RACK, "label": numbers[RACK], "globalIndices": rack[:900].tolist()},
                      {"id": RACK + 1, "label": numbers[RACK], "globalIndices": rack[900:].tolist()}]
        (out / "segmentation_result.json").write_text(json.dumps({"total_points": len(xyz), "instances": instances}))
    elements = [
        {"id": FLOOR, "label": "floor", "vertices": floor_v, "faces": floor_f, "ifc": ("IfcSlab", "FLOOR"), "n_points": 6000},
        {"id": WALL, "label": "wall", "vertices": wall_v, "faces": wall_f, "ifc": ("IfcWall", None), "n_points": 3000},
        {"id": RACK, "label": "rack", "vertices": rack_v, "faces": rack_f, "ifc": ("IfcBuildingElementProxy", None), "n_points": 2000},
        {"id": 99, "label": "beam", "vertices": box((0, 2.6, 0), (4, 2.9, 0.3))[0], "faces": box((0, 2.6, 0), (4, 2.9, 0.3))[1],
         "ifc": ("IfcBeam", None), "n_points": 0},                            # in the model, not in the scan
    ]
    ifc = build(elements, session=session, scan_key="2026-01-02/default", geometry_epoch="1", described="test")
    if not stac_ifc:                                                          # an IFC made elsewhere: other name, BIM numbers
        ifc.header.file_name.name = "architect_model.ifc"
        numbers = {str(FLOOR): "292127", str(WALL): "292200", str(RACK): "292300", "99": "292999"}
        for product in [p for p in ifc.by_type("IfcProduct") if p.Representation]:
            product.Name = f"Family:Type:{numbers[product.Tag]}"
    ifc.write(str(root / "projects" / session / "ifcs" / "model.ifc"))
    return scan_dir


def matches_of(scan_dir: Path):
    scan = Scan(scan_dir / "output")
    ifc = ifcopenshell.open(str(first_ifc(C.project_dirs(scan_dir)[0])))
    return scan, scan.match(element_keys(ifc), by_instance_id=stac_source(ifc) is not None), stac_source(ifc)


def test_a_scan_is_matched_by_instance_id_to_the_ifc_stac_wrote_from_it(tmp_path):
    scan_dir = make_project(tmp_path)
    scan, matches, source = matches_of(scan_dir)
    assert source == ("site", "2026-01-02/default")
    assert scan.source.startswith("the instance store") and {i: v["label"] for i, v in scan.instances.items()} == {3: "floor", 7: "wall", 12: "rack"}
    assert [(m["segment_label"], m["segment_ids"], m["element_key"], m["ifc_type"]) for m in matches] == [
        ("floor", [3], "3", "IfcSlab"), ("wall", [7], "7", "IfcWall"), ("rack", [12], "12", "IfcBuildingElementProxy")]
    idx = scan.indices(FLOOR)
    assert len(idx) == 6000 and np.all(np.diff(idx) > 0) and np.allclose(scan.xyz[idx, 1], 0.020, atol=1e-6)
    assert len(scan.indices(404)) == 0


def test_an_ifc_made_elsewhere_is_matched_by_the_label_typed_on_the_segment(tmp_path):
    scan_dir = make_project(tmp_path, stac_ifc=False, store=False)
    scan, matches, source = matches_of(scan_dir)
    assert source is None and scan.source == "segmentation_result.json"
    assert [(m["segment_label"], m["segment_ids"], m["element_key"]) for m in matches] == [
        ("292127", [3], "292127"), ("292200", [7], "292200"), ("292300", [12, 13], "292300")]
    assert len(scan.indices_of_match(matches[2])) == 2000                     # two objects, one element's points
    assert len(scan.indices_of_match({"segment_label": "292200", "element_key": "292200"})) == 3000   # a match written by hand


def test_a_scan_with_no_segmentation_is_said(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(ScanError, match="no segmentation"):
        Scan(tmp_path / "output")


def test_the_comparison_of_a_scan_with_its_own_ifc(tmp_path):
    scan_dir = make_project(tmp_path, offset=0.020)
    _, matches, _ = matches_of(scan_dir)
    result = C.run_comparison(str(scan_dir), matches, tolerance=0.05)
    assert result["ok"] and result["summary"] == {
        "total_elements": 4, "evaluated": 3, "unmatched": 1, "errors": 0, "registration": "own_scan", "ifc": "model.ifc"}
    assert np.array_equal(np.array(result["transform"]), np.eye(4))           # the same frame by construction
    by = {r["element_key"]: r for r in result["results"]}
    floor, wall, rack, beam = by["3"], by["7"], by["12"], by["99"]
    # the floor's points lie 20 mm over its slab; the wall's and the rack's on their faces
    assert floor["status"] == "evaluated" and floor["total_points"] == 6000
    assert abs(floor["stats"]["mean_mm"] - 20.0) < 0.01 and abs(floor["stats"]["max_mm"] - 20.0) < 0.01
    assert floor["stats"]["pass_rate"] == 100.0 and floor["coverage_pct"] > 99.0 and abs(floor["bim_surface_m2"] - 16.0) < 1e-6
    assert wall["stats"]["max_mm"] < 0.01 and rack["stats"]["max_mm"] < 0.01
    assert 20.0 < rack["coverage_pct"] < 45.0                                 # one of its sides was seen
    assert beam == {"element_key": "99", "label": "beam:beam:99", "ifc_type": "IfcBeam", "status": "unmatched", "n_faces": 12}
    # the answer carries numbers, not points
    assert all("sabana_positions" not in r and "sabana_colors" not in r for r in result["results"])
    assert len(json.dumps(result)) < 20_000
    # the sheet is on disk: every evaluated point, coloured (green: within tolerance)
    out = C.project_dirs(scan_dir)[1]
    assert {p.name for p in out.iterdir()} >= {"sabana.npz", "sabana_cloud.ply", "sabana_meta.json"}
    from exporter.cloud import open_ply
    sheet = open_ply(out / "sabana_cloud.ply")
    assert len(sheet) == 11000 and sheet["green"].min() == 255
    meta = json.loads((out / "sabana_meta.json").read_text())
    assert meta["total_points"] == 11000 and meta["scan"] == "2026-01-02/default" and "geometry_epoch" in meta
    assert {e["element_key"]: e["status"] for e in meta["elements"]} == {"3": "evaluated", "7": "evaluated", "12": "evaluated", "99": "unmatched"}
    # the same comparison again: the same numbers
    again = C.run_comparison(str(scan_dir), matches, tolerance=0.05)
    assert again["results"] == result["results"]


def test_a_deviation_beyond_the_tolerance_is_seen(tmp_path):
    scan_dir = make_project(tmp_path, offset=0.080)                           # the floor 80 mm off its slab
    _, matches, _ = matches_of(scan_dir)
    by = {r["element_key"]: r for r in C.run_comparison(str(scan_dir), matches, tolerance=0.05)["results"]}
    assert abs(by["3"]["stats"]["mean_mm"] - 80.0) < 0.01 and by["3"]["stats"]["pass_rate"] == 0.0
    assert by["7"]["stats"]["pass_rate"] == 100.0


def test_the_comparison_with_an_ifc_made_elsewhere_uses_the_alignment_it_is_given(tmp_path):
    """March's way still runs: labels typed by hand, the manual alignment (the scan's floor transform)."""
    scan_dir = make_project(tmp_path, stac_ifc=False, store=False)
    _, matches, _ = matches_of(scan_dir)
    result = C.run_comparison(str(scan_dir), matches, tolerance=0.05, skip_registration=True)
    assert result["ok"] and result["summary"]["registration"] == "manual" and result["summary"]["evaluated"] == 3
    by = {r["element_key"]: r for r in result["results"]}
    assert abs(by["292127"]["stats"]["mean_mm"] - 20.0) < 0.01 and by["292300"]["total_points"] == 2000
    assert by["292999"]["status"] == "unmatched"


def test_a_triangulated_element_is_read_from_the_file_itself(tmp_path):
    """An IFC whose elements are meshes is read as written (the kernel built a face per triangle:
    minutes on pccr's 1.4 M) — and gives the kernel's surface."""
    import ifcopenshell.geom as kernel
    scan_dir = make_project(tmp_path)
    ifc = ifcopenshell.open(str(first_ifc(C.project_dirs(scan_dir)[0])))
    direct = C.extract_all_ifc_geometry("", ifc)
    assert set(direct) == {"3", "7", "12", "99"}
    verts, faces, ifc_type, name = direct["3"]
    assert ifc_type == "IfcSlab" and name == "floor:floor:3" and len(faces) == 128
    assert np.allclose(verts[:, 1], 0.0) and verts[:, 0].min() == 0.0 and verts[:, 2].max() == 4.0   # the display frame (Y up)
    settings = kernel.settings()
    settings.set("use-world-coords", True)
    for product in [p for p in ifc.by_type("IfcProduct") if p.Representation]:
        shape = kernel.create_shape(settings, product)                        # (kept alive: the arrays live in it)
        g = shape.geometry
        kv = np.array(g.verts, dtype=np.float64).reshape(-1, 3)[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
        kf = np.array(g.faces, dtype=np.int32).reshape(-1, 3)
        v, f, _, _ = direct[product.Name.split(":")[-1]]
        rng = np.random.default_rng(5)
        probes = rng.uniform(v.min(axis=0) - 0.5, v.max(axis=0) + 0.5, (500, 3))
        assert np.abs(C.compute_c2m_distances(probes, v, f, quiet=True) - C.compute_c2m_distances(probes, kv, kf, quiet=True)).max() < 1e-6


# ── the comparison in its own process ──────────────────────────────────────────────────────────
def run_job(*args):
    import subprocess
    server = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-m", "bim.compare_job", "--device", "cpu", *args], cwd=server, capture_output=True,
                          text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"})


def test_the_job_says_its_progress_and_its_result(tmp_path):
    """What the server launches and reads (main.py /api/bim/compare): one line a step, the summary last."""
    scan_dir = make_project(tmp_path)
    _, matches, _ = matches_of(scan_dir)
    (tmp_path / "matches.json").write_text(json.dumps(matches))
    run = run_job("--session-dir", str(scan_dir), "--matches", str(tmp_path / "matches.json"), "--tolerance-mm", "50")
    assert run.returncode == 0, run.stdout + run.stderr
    lines = run.stdout.splitlines()
    progress = [int(l.split()[1]) for l in lines if l.startswith("PROGRESS ")]
    assert progress and progress == sorted(progress) and progress[-1] == 100
    result = json.loads(next(l for l in lines if l.startswith("RESULT "))[7:])
    assert result["summary"]["evaluated"] == 3 and result["summary"]["registration"] == "own_scan" and result["tolerance_mm"] == 50.0
    assert result["summary"]["octree"] is True and (C.project_dirs(scan_dir)[1] / "sabana_potree" / "metadata.json").exists()
    assert "results" not in result                                            # the numbers are in sabana_meta.json
    meta = json.loads((C.project_dirs(scan_dir)[1] / "sabana_meta.json").read_text())
    assert {e["element_key"] for e in meta["elements"] if e["status"] == "evaluated"} == {"3", "7", "12"}


def test_the_job_says_why_it_failed(tmp_path):
    scan_dir = make_project(tmp_path)
    (tmp_path / "none.json").write_text("[]")
    run = run_job("--session-dir", str(scan_dir), "--matches", str(tmp_path / "none.json"))
    assert run.returncode == 1 and any(l.startswith("ERROR ValueError: no matches") for l in run.stdout.splitlines())
    # a scan with no IFC
    _, matches, _ = matches_of(scan_dir)
    (tmp_path / "matches.json").write_text(json.dumps(matches))
    first_ifc(C.project_dirs(scan_dir)[0]).unlink()
    run = run_job("--session-dir", str(scan_dir), "--matches", str(tmp_path / "matches.json"))
    assert run.returncode == 1 and any("No IFC file found" in l for l in run.stdout.splitlines() if l.startswith("ERROR "))


def test_the_job_on_the_card_with_no_card_says_the_key(tmp_path):
    scan_dir = make_project(tmp_path)
    _, matches, _ = matches_of(scan_dir)
    (tmp_path / "matches.json").write_text(json.dumps(matches))
    import subprocess
    run = subprocess.run([sys.executable, "-m", "bim.compare_job", "--device", "cuda", "--session-dir", str(scan_dir),
                          "--matches", str(tmp_path / "matches.json")], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert run.returncode == 1
    assert any(l.startswith("ERROR RuntimeError: bim.deviation.device is 'cuda' and torch sees no card") for l in run.stdout.splitlines())
