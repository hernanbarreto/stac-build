# STAC-Builder — server-side GLB → USDZ for AR Quick Look (iOS native viewer).
#
# Runs under the ISOLATED /workspace/usdtools venv (usd-core + trimesh +
# pymeshlab + pillow), NOT the server env. Invoked by ar_api's
# /api/ar/usdz/{session} endpoint:
#
#   /workspace/usdtools/bin/python glb_to_usdz.py --glb scene.glb.orig \
#       --out ar_scene.usdz [--floor-transform '<16 floats json>'] \
#       [--target-tris 700000]
#
# Pipeline: trimesh reads the textured GLB (use scene.glb.orig — PNG/JPEG
# textures; Quick Look does not decode the WebP of the compressed GLB) →
# optional upright floor transform baked into the vertices → per-submesh
# UV-PRESERVING quadric decimation (pymeshlab *_with_texture; plain quadric
# would destroy the texrecon atlas mapping) → hand-authored USD
# (UsdGeomMesh + UsdPreviewSurface/UsdUVTexture per submesh, upAxis Y,
# metersPerUnit 1 — our units ARE meters, so Quick Look projects true 1:1) →
# UsdUtils.CreateNewARKitUsdzPackage (Pixar's own ARKit-compliant packager).
#
# Prints a single JSON line with stats on success; non-zero exit on failure.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_glb(path: Path):
    import trimesh
    scene = trimesh.load(str(path), force="scene", file_type="glb")
    out = []
    for name, geom in scene.geometry.items():
        if not hasattr(geom, "triangles") or len(geom.faces) == 0:
            continue
        # bake the scene-graph transform of this node into the vertices
        T = np.eye(4)
        for node in scene.graph.nodes_geometry:
            tf, gname = scene.graph[node]
            if gname == name:
                T = tf
                break
        v = geom.vertices
        if not np.allclose(T, np.eye(4)):
            v = (T[:3, :3] @ v.T).T + T[:3, 3]
        uv = None
        tex = None
        if geom.visual is not None and hasattr(geom.visual, "uv"):
            uv = geom.visual.uv
            mat = getattr(geom.visual, "material", None)
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                tex = img
        out.append({"name": name, "v": np.asarray(v, np.float64),
                    "f": np.asarray(geom.faces, np.int64),
                    "uv": None if uv is None else np.asarray(uv, np.float64),
                    "tex": tex})
    if not out:
        raise RuntimeError("no triangle geometry in the GLB")
    return out


def decimate_with_texture(sub, target_faces: int, tmp: Path, idx: int):
    """UV-preserving quadric decimation through pymeshlab, round-tripped via
    OBJ (the filter needs wedge texcoords + a registered texture)."""
    import pymeshlab
    if len(sub["f"]) <= target_faces or sub["uv"] is None:
        return sub
    obj_in = tmp / f"in_{idx}.obj"
    tex_name = f"tex_{idx}.png"
    if sub["tex"] is not None:
        sub["tex"].convert("RGB").save(tmp / tex_name)
    with open(obj_in, "w") as f:
        f.write(f"mtllib in_{idx}.mtl\n")
        for p in sub["v"]:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for t in sub["uv"]:
            f.write(f"vt {t[0]} {t[1]}\n")
        f.write("usemtl m0\n")
        for a, b, c in sub["f"] + 1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    with open(tmp / f"in_{idx}.mtl", "w") as f:
        f.write(f"newmtl m0\nKd 1 1 1\nmap_Kd {tex_name}\n")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj_in))
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=int(target_faces), preserveboundary=True,
        planarquadric=True)
    obj_out = tmp / f"out_{idx}.obj"
    ms.save_current_mesh(str(obj_out))
    import trimesh
    dec = trimesh.load(str(obj_out), force="mesh")
    uv = dec.visual.uv if hasattr(dec.visual, "uv") else None
    return {"name": sub["name"], "v": np.asarray(dec.vertices, np.float64),
            "f": np.asarray(dec.faces, np.int64),
            "uv": None if uv is None else np.asarray(uv, np.float64),
            "tex": sub["tex"]}


def author_usd(subs, usd_path: Path, tex_dir: Path):
    import hashlib
    from pxr import Usd, UsdGeom, UsdShade, Sdf
    tex_by_hash: dict = {}   # texrecon atlas pages repeat across submeshes —
                             # dedupe by content or the package balloons
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    for i, s in enumerate(subs):
        mpath = f"/Scene/mesh_{i}"
        mesh = UsdGeom.Mesh.Define(stage, mpath)
        mesh.CreatePointsAttr(s["v"].astype(np.float32).tolist())
        mesh.CreateFaceVertexIndicesAttr(s["f"].reshape(-1).astype(int).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(s["f"]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        if s["uv"] is not None:
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex)
            pv.Set(s["uv"].astype(np.float32).tolist())
        if s["tex"] is not None:
            img = s["tex"].convert("RGB")
            h = hashlib.sha1(img.tobytes()).hexdigest()[:16]
            if h in tex_by_hash:
                tex_file = tex_by_hash[h]
            else:
                tex_file = tex_dir / f"tex_{len(tex_by_hash)}.png"
                img.save(tex_file)
                tex_by_hash[h] = tex_file
            mat = UsdShade.Material.Define(stage, f"{mpath}_mat")
            sh = UsdShade.Shader.Define(stage, f"{mpath}_mat/pbr")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            st_reader = UsdShade.Shader.Define(stage, f"{mpath}_mat/stReader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            tx = UsdShade.Shader.Define(stage, f"{mpath}_mat/tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                f"./{tex_file.name}")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f) \
                .ConnectToSource(tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
            mat.CreateSurfaceOutput().ConnectToSource(
                sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    stage.Save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-transform", default=None,
                    help="row-major 4x4 as JSON list (bakes upright into verts)")
    ap.add_argument("--target-tris", type=int, default=700_000)
    args = ap.parse_args()

    subs = load_glb(Path(args.glb))
    n_in = sum(len(s["f"]) for s in subs)

    if args.floor_transform:
        M = np.array(json.loads(args.floor_transform), np.float64).reshape(4, 4)
        for s in subs:
            s["v"] = (M[:3, :3] @ s["v"].T).T + M[:3, 3]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if n_in > args.target_tris:
            budget = args.target_tris
            subs = [decimate_with_texture(
                s, max(int(budget * len(s["f"]) / n_in), 1000), tmp, i)
                for i, s in enumerate(subs)]
        n_out = sum(len(s["f"]) for s in subs)

        usd_path = tmp / "scene.usd"
        author_usd(subs, usd_path, tmp)

        from pxr import UsdUtils
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = UsdUtils.CreateNewARKitUsdzPackage(str(usd_path), str(out))
        if not ok or not out.exists():
            raise RuntimeError("CreateNewARKitUsdzPackage failed")

    print(json.dumps({"tris_in": int(n_in), "tris_out": int(n_out),
                      "submeshes": len(subs),
                      "bytes": out.stat().st_size}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
