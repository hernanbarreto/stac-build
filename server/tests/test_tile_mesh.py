"""tools/glb/tile_mesh.mjs (USER 2026-11-02: "hay que implementar el tema de los tiles … para que pueda ser manejable
una malla grande"): a textured mesh of four primitives → a 3D Tiles tree whose leaves are the primitives at full
detail, whose parents never claim less error than their children, whose boxes enclose what they hold, every tile a
glTF with KTX2 textures and meshopt geometry. A tile never holds an EMPTY primitive (pccr 2026-08-24, 2026-11-12: a
speck the simplifier left with no triangle stopped the writer): the speck keeps the geometry of the tile below, and
a source primitive with no triangle is left out, declared. Needs node with tools/glb's packages and the ktx tool
($KTX_BIN or the repo's vendor/KTX-Software); skipped, saying which, when either is missing."""
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
GLB_TOOLS = REPO / "tools" / "glb"
KTX = Path(os.environ.get("KTX_BIN", REPO / "vendor" / "KTX-Software-4.4.2-Linux-x86_64" / "bin" / "ktx"))
NODE = shutil.which("node") or "/workspace/miniforge3/envs/nodejs/bin/node"


def _need_tools():
    if not Path(NODE).exists():
        pytest.skip(f"node not found ({NODE})")
    if not (GLB_TOOLS / "node_modules" / "@gltf-transform" / "core").exists():
        pytest.skip(f"tools/glb's packages are not installed ({GLB_TOOLS / 'node_modules'})")
    if not KTX.exists():
        pytest.skip(f"the ktx tool is missing ({KTX}; set KTX_BIN)")


def _grid(nx, ny, x0, z0, size=1.0):
    """A textured, gently curved patch on the floor plane: (nx+1)(ny+1) vertices, 2·nx·ny triangles, UVs over [0,1]²."""
    u, v = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
    pos = np.stack([x0 + size * u, 0.05 * np.sin(6 * u) * np.cos(5 * v), z0 + size * v], -1).reshape(-1, 3)
    idx = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            idx += [[a, a + nx + 1, a + 1], [a + 1, a + nx + 1, a + nx + 2]]
    return pos, np.array(idx), np.stack([u, v], -1).reshape(-1, 2)


def _speck(centre, r=0.004):
    """A closed octahedron 1.4 cm across: the kind of piece (8 triangles) the simplifier leaves with NO triangle once
    its budget is one."""
    pos = np.asarray(centre, float) + r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
    idx = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    uv = np.array([[0.9, 0.5], [0.1, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.9], [0.5, 0.1]])
    return pos, idx, uv


def _write_glb(path: Path, parts):
    """One textured primitive per (positions, triangles, UVs), each with its own 64 px atlases → the triangle count."""
    import trimesh
    from PIL import Image
    rng = np.random.default_rng(0)
    scene = trimesh.Scene()
    for k, (pos, idx, uv) in enumerate(parts):
        img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        mr = Image.fromarray(np.full((64, 64, 3), [0, 200, 255], np.uint8))
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicRoughnessTexture=mr,
                                                  metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        m = trimesh.Trimesh(pos, idx, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
        scene.add_geometry(m, node_name=f"chunk_{k}", geom_name=f"chunk_{k}")
    path.write_bytes(scene.export(file_type="glb"))
    return sum(len(idx) for _, idx, _ in parts)


def _source_glb(path: Path):
    return _write_glb(path, [_grid(24, 24, x0, z0) for x0, z0 in [(0, 0), (1, 0), (0, 1), (1, 1)]])


def _add_empty_primitive(path: Path):
    """A second primitive on the first mesh, with its vertices and its material and NO triangle (an index accessor
    of count 0)."""
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[12:16])[0]
    j, rest = json.loads(raw[20:20 + n]), raw[20 + n:]
    p0 = j["meshes"][0]["primitives"][0]
    acc = {k: v for k, v in j["accessors"][p0["indices"]].items() if k not in ("min", "max")}
    j["accessors"].append(dict(acc, count=0))
    j["meshes"][0]["primitives"].append(dict(p0, indices=len(j["accessors"]) - 1))
    body = json.dumps(j).encode()
    body += b" " * (-len(body) % 4)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20 + len(body) + len(rest))
                     + struct.pack("<I4s", len(body), b"JSON") + body + rest)


def _tile(tmp_path: Path):
    """The tiler on tmp_path/mesh.glb → (the finished process, the tiles folder)."""
    out = tmp_path / "tiles"
    r = subprocess.run([NODE, "tile_mesh.mjs", str(tmp_path / "mesh.glb"), str(out), "--jobs", "2"],
                       cwd=GLB_TOOLS, env=dict(os.environ, KTX_BIN=str(KTX)), capture_output=True, text=True,
                       timeout=600)
    return r, out


def _gltf_json(path: Path) -> dict:
    with open(path, "rb") as fh:
        fh.read(12)
        n, _ = struct.unpack("<II", fh.read(8))
        return json.loads(fh.read(n))


def _index_counts(path: Path):
    """The number of indices of every primitive of a tile."""
    j = _gltf_json(path)
    return [j["accessors"][p["indices"]]["count"] for p in j["meshes"][0]["primitives"]]


def _tile_id(tile) -> int:
    return int(tile["content"]["uri"][5:-4])


def _walk(tile, depth=0):
    yield tile, depth
    for c in tile.get("children", []):
        yield from _walk(c, depth + 1)


def test_a_mesh_becomes_a_tileset(tmp_path):
    _need_tools()
    n_faces = _source_glb(tmp_path / "mesh.glb")
    r, out = _tile(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert not (tmp_path / "tiles.part").exists()               # built aside, renamed whole
    ts = json.loads((out / "tileset.json").read_text())
    assert ts["asset"]["gltfUpAxis"] == "Z"                     # no axis turn: the mesh's own frame
    tiles = list(_walk(ts["root"]))
    leaves = [t for t, _ in tiles if not t.get("children")]
    assert len(leaves) == 4 and all(t["geometricError"] == 0 for t in leaves)
    report = json.loads((out / "tiles_report.json").read_text())
    by_id = {t["id"]: t for t in report["per_tile"]}
    assert report["triangles"] == n_faces
    assert sum(by_id[_tile_id(t)]["triangles"] for t in leaves) == n_faces   # leaves = full detail
    assert report["primitives_kept_from_below"] == [] and report["source_primitives_without_triangles"] == 0
    for t, _ in tiles:
        assert (out / t["content"]["uri"]).is_file()
        assert t["refine"] == "REPLACE"
        for c in t.get("children", []):
            assert t["geometricError"] >= c["geometricError"]
            # a parent's box encloses its child's
            pc, ph = np.array(t["boundingVolume"]["box"][:3]), np.array(t["boundingVolume"]["box"])[[3, 7, 11]]
            cc, ch = np.array(c["boundingVolume"]["box"][:3]), np.array(c["boundingVolume"]["box"])[[3, 7, 11]]
            assert np.all(cc - ch >= pc - ph - 1e-5) and np.all(cc + ch <= pc + ph + 1e-5)
    assert ts["root"]["geometricError"] > 0 and ts["geometricError"] == ts["root"]["geometricError"]
    # the root's box encloses the source mesh (x, z over [0, 2], the patches' ±5 cm relief)
    c, h = np.array(ts["root"]["boundingVolume"]["box"][:3]), np.array(ts["root"]["boundingVolume"]["box"])[[3, 7, 11]]
    assert np.allclose(c - h, [0, -0.05, 0], atol=0.01) and np.allclose(c + h, [2, 0.05, 2], atol=0.01)
    for t, _ in tiles:
        j = _gltf_json(out / t["content"]["uri"])
        assert {"KHR_texture_basisu", "EXT_meshopt_compression"} <= set(j["extensionsRequired"])
        assert all(im["mimeType"] == "image/ktx2" for im in j["images"])
        assert len(j["meshes"][0]["primitives"]) == len(j["materials"])


def test_a_speck_the_simplifier_erases_stays_in_every_tile(tmp_path):
    """pccr 2026-08-24 (2026-11-12): a piece of 8 triangles in 1.4 cm came back from the simplifier with NO triangle
    at height 3 (a budget of one); the tiler handed the writer an empty primitive and the meshopt encoder stopped the
    run ("Assertion failed", 223 of 251 tiles written). A tile holds every primitive below it: the speck keeps the
    geometry of the tile below."""
    _need_tools()
    patches = [_grid(12, 12, x0, z0) for x0 in range(4) for z0 in range(2)]
    _write_glb(tmp_path / "mesh.glb", patches[:7] + [_speck((3.5, 0.0, 1.5))])
    r, out = _tile(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    ts = json.loads((out / "tileset.json").read_text())
    report = json.loads((out / "tiles_report.json").read_text())
    by_id = {t["id"]: t for t in report["per_tile"]}
    assert report["primitives"] == 8 and report["depth"] == 3          # the speck's budget at the root: one triangle
    counts = {}
    for t, _ in _walk(ts["root"]):
        counts[_tile_id(t)] = _index_counts(out / t["content"]["uri"])
        assert len(counts[_tile_id(t)]) == by_id[_tile_id(t)]["primitives"]   # every primitive below it …
        assert min(counts[_tile_id(t)]) >= 3                                  # … and none of them empty
    root = _tile_id(ts["root"])
    assert by_id[root]["kept_from_below"] == 1, "the simplifier no longer erases the speck: this case tests nothing"
    assert len(report["primitives_kept_from_below"]) == 1
    assert "kept the geometry of the tile below" in r.stdout
    # the speck (the smallest primitive of every tile that holds it) as the tile below the root holds it
    assert min(counts[root]) == min(min(counts[_tile_id(c)]) for c in ts["root"]["children"])
    for t, _ in _walk(ts["root"]):
        for c in t.get("children", []):
            assert t["geometricError"] >= c["geometricError"]


def test_a_source_primitive_with_no_triangle_is_left_out(tmp_path):
    _need_tools()
    n_faces = _write_glb(tmp_path / "mesh.glb", [_grid(12, 12, 0, 0), _grid(12, 12, 1, 0)])
    _add_empty_primitive(tmp_path / "mesh.glb")
    r, out = _tile(tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "1 primitive(s) of the source have no triangle" in r.stdout
    report = json.loads((out / "tiles_report.json").read_text())
    assert report["source_primitives_without_triangles"] == 1
    assert report["primitives"] == 2 and report["triangles"] == n_faces and report["tiles"] == 3
    for t in report["per_tile"]:
        assert min(_index_counts(out / f"tile_{t['id']}.glb")) >= 3
