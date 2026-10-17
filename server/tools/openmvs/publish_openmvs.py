"""OpenMVS's textured GLB → the session's mesh folders (output/tsdf/<name>/), written atomically for the viewer's
watcher. The scene was handed over in the RAW frame, so the mesh is already where the viewer draws meshes
(under floor_transform.npz). OpenMVS's glTF exporter rotates the scene as if it were Z-up (x, y, z → x, z, −y);
the frame is CHECKED against the refined mesh's own PLY bounds and undone when it was applied, recorded in the meta.
    python publish_openmvs.py <glb> <session src_default> <name> <variant>"""
import sys, json, os, time, numpy as np, trimesh
from pathlib import Path
glb = Path(sys.argv[1]); SES = Path(sys.argv[2]); name = sys.argv[3]; variant = sys.argv[4] if len(sys.argv) > 4 else ""
sc = trimesh.load(str(glb), force="scene")
frame_note = "as exported"
ply = glb.parent / "refined.ply"
if ply.exists():
    import open3d as o3d
    V = np.asarray(o3d.io.read_triangle_mesh(str(ply)).vertices); ref = np.array([V.min(0), V.max(0)])
    M = np.eye(4); M[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]          # Rx(+90°): undoes x, z, −y
    rot = sc.copy(); rot.apply_transform(M)
    e_id, e_rot = np.abs(sc.bounds - ref).max(), np.abs(rot.bounds - ref).max()
    if e_rot < e_id:
        sc = rot; frame_note = f"OpenMVS glTF export rotated the scene (Z-up assumed); undone with Rx(+90°), bounds match refined.ply within {e_rot*100:.1f} cm"
    else:
        frame_note = f"frame as exported (bounds match refined.ply within {e_id*100:.1f} cm)"
d = SES / "output" / "tsdf" / name; d.mkdir(parents=True, exist_ok=True)
out = d / f"{name}.glb"; tmp = d / f"{name}.glb.part"
sc.export(str(tmp), file_type="glb"); os.replace(tmp, out)
n_tex = sum(1 for g in sc.geometry.values() if any(getattr(getattr(g.visual, "material", None), a, None) is not None for a in ("baseColorTexture", "image")))
meta = {"method": "openmvs", "kind": "scene", "instance_id": None, "label": name, "glb_file": out.name, "textured": n_tex > 0,
        "provenance": "tool_measured: OpenMVS (cdcseacave/openMVS, AGPL-3.0) — " + {
            "full": "its own dense MVS cloud from the keyframes and the certified poses",
            "colmap": "the pure classical chain: COLMAP 4.2 SfM (no prior) aligned by one Sim(3) to the certified trajectory, then its own dense MVS cloud",
        }.get(variant, "the session's certified cloud with per-point visibility") + " → ReconstructMesh → RefineMesh → TextureMesh; nothing generated",
        "variant": variant, "n_faces": int(sum(len(g.faces) for g in sc.geometry.values())), "n_primitives": len(sc.geometry), "textured_primitives": n_tex,
        "bounds_published": sc.bounds.tolist(), "frame": "raw (the COLMAP model was written in the session's raw frame); " + frame_note,
        "published_at": time.strftime("%Y-%m-%d %H:%M")}
(d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"published {out} ({out.stat().st_size/1e6:.0f} MB, {meta['n_faces']:,} faces, {n_tex} textured primitives); {frame_note}; bounds {np.round(sc.bounds, 2).tolist()}")
