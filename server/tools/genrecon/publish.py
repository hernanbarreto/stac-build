"""GenRecon's scene.glb → the session's mesh folders (output/tsdf/<name>/). The baker writes glTF Y-up in
the frame it was given (the session's DISPLAY frame via the adapter); the viewer draws tsdf meshes in the
RAW frame under the floor transform, so a non-identity display transform is undone here (M_raw_to_genrecon
without the Y-up → Z-up turn, which the baker already reverts)."""
import sys, json, shutil, time, numpy as np, trimesh
from pathlib import Path
glb = Path(sys.argv[1]); SES = Path(sys.argv[2]); name = sys.argv[3]; work = Path(sys.argv[4]) if len(sys.argv) > 4 else None
note = sys.argv[5] if len(sys.argv) > 5 else None
sc = trimesh.load(str(glb), force="scene")
b0 = sc.bounds.copy(); frame_note = "the baker's glTF Y-up = the session's raw frame (identity display transform)"
if work is not None and (work / "M_raw_to_genrecon.npy").exists():
    M = np.load(work / "M_raw_to_genrecon.npy"); R_up = np.load(work / "R_up.npy")
    D = np.eye(4); D[:3, :3] = R_up.T @ M[:3, :3]; D[:3, 3] = R_up.T @ M[:3, 3]      # raw → display (Y-up)
    if not np.allclose(D, np.eye(4), atol=1e-6):
        sc.apply_transform(np.linalg.inv(D))                                          # display → raw
        frame_note = "display transform undone: the mesh is in the session's RAW frame (the viewer applies floor_transform.npz)"
d = SES / "output" / "tsdf" / name; d.mkdir(parents=True, exist_ok=True)
out = d / f"{name}.glb"
# ATOMIC: the viewer's folder watcher lists the mesh the moment the folder appears and fetched a GLB still being
# written ("Too much data for declared Content-Length", 2026-10-17) — write beside it, then rename
tmp = d / f"{name}.glb.part"
if frame_note.startswith("the baker"):
    shutil.copy2(glb, tmp)
else:
    sc.export(str(tmp), file_type="glb")
import os
os.replace(tmp, out)
meta = {"method": "genrecon", "kind": "scene", "instance_id": None, "label": name, "glb_file": out.name, "textured": True,
        "provenance": "vlm_generated: GenRecon (Trellis.2 fine-tuned, kasothaphie/GenRecon eaf1468) conditioned on the "
                      "session's keyframes, certified poses and published cloud — generative: presence and shape, not a measurement",
        "variant": note,
        "n_faces": int(sum(len(g.faces) for g in sc.geometry.values())), "n_primitives": len(sc.geometry),
        "bounds_baked": b0.tolist(), "bounds_published": sc.bounds.tolist(), "frame": frame_note, "published_at": time.strftime("%Y-%m-%d %H:%M")}
(d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"published {out} ({out.stat().st_size/1e6:.0f} MB, {meta['n_faces']:,} faces); {frame_note}; bounds {np.round(sc.bounds, 2).tolist()}")
