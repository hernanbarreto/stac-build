"""GenRecon's scene.glb (our Z-up adapter frame) → the session's raw Y-up frame, published as a
scene mesh folder the viewer lists (output/tsdf/genrecon_scene/)."""
import sys, json, shutil, time, numpy as np, trimesh
from pathlib import Path
glb = Path(sys.argv[1]); SES = Path(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else "genrecon_scene"
R_up = np.load(Path(sys.argv[4])) if len(sys.argv) > 4 else np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
T = np.eye(4); T[:3, :3] = R_up.T                       # Z-up (adapter) → the session's Y-up raw frame
sc = trimesh.load(str(glb), force="scene")
n_faces = sum(len(g.faces) for g in sc.geometry.values()); n_geom = len(sc.geometry)
b0 = sc.bounds.copy()
sc.apply_transform(T)
b1 = sc.bounds
d = SES / "output" / "tsdf" / name; d.mkdir(parents=True, exist_ok=True)
out = d / f"{name}.glb"
sc.export(str(out))
meta = {"method": "genrecon", "kind": "scene", "instance_id": None, "label": name, "glb_file": out.name, "textured": True,
        "provenance": "vlm_generated: GenRecon (Trellis.2 fine-tuned, kasothaphie/GenRecon eaf1468) conditioned on the "
                      "session's keyframes, certified poses (epoch 4) and published cloud — generative: presence/shape, "
                      "not a measurement", "n_faces": int(n_faces), "n_primitives": int(n_geom),
        "bounds_adapter_zup": b0.tolist(), "bounds_session_yup": b1.tolist(), "published_at": time.strftime("%Y-%m-%d %H:%M")}
(d / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"published {out} ({out.stat().st_size/1e6:.0f} MB, {n_faces:,} faces, {n_geom} primitives); bounds Y-up min {np.round(b1[0],2).tolist()} max {np.round(b1[1],2).tolist()}")
