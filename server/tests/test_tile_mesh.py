"""tools/glb/tile_mesh.mjs (USER 2026-11-02: "hay que implementar el tema de los tiles … para que pueda ser manejable
una malla grande"): a textured mesh of four primitives → a 3D Tiles tree whose leaves are the primitives at full
detail, whose parents never claim less error than their children, whose boxes enclose what they hold, every tile a
glTF with KTX2 textures and meshopt geometry. Needs node with tools/glb's packages and the ktx tool ($KTX_BIN or the
repo's vendor/KTX-Software); skipped, saying which, when either is missing."""
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


def _source_glb(path: Path):
    import trimesh
    from PIL import Image
    meshes = []
    rng = np.random.default_rng(0)
    for k, (x0, z0) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        pos, idx, uv = _grid(24, 24, x0, z0)
        img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        mr = Image.fromarray(np.full((64, 64, 3), [0, 200, 255], np.uint8))
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicRoughnessTexture=mr,
                                                  metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        m = trimesh.Trimesh(pos, idx, visual=trimesh.visual.TextureVisuals(uv=uv, material=mat), process=False)
        meshes.append((f"chunk_{k}", m))
    scene = trimesh.Scene()
    for name, m in meshes:
        scene.add_geometry(m, node_name=name, geom_name=name)
    path.write_bytes(scene.export(file_type="glb"))
    return sum(len(m.faces) for _, m in meshes)


def _gltf_json(path: Path) -> dict:
    with open(path, "rb") as fh:
        fh.read(12)
        n, _ = struct.unpack("<II", fh.read(8))
        return json.loads(fh.read(n))


def _walk(tile, depth=0):
    yield tile, depth
    for c in tile.get("children", []):
        yield from _walk(c, depth + 1)


def test_a_mesh_becomes_a_tileset(tmp_path):
    _need_tools()
    n_faces = _source_glb(tmp_path / "mesh.glb")
    out = tmp_path / "tiles"
    env = dict(os.environ, KTX_BIN=str(KTX))
    r = subprocess.run([NODE, "tile_mesh.mjs", str(tmp_path / "mesh.glb"), str(out), "--jobs", "2"],
                       cwd=GLB_TOOLS, env=env, capture_output=True, text=True, timeout=600)
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
    assert sum(by_id[int(t["content"]["uri"][5:-4])]["triangles"] for t in leaves) == n_faces   # leaves = full detail
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
