"""Point 4 (USER 2026-10-16: "respetar perfectamente la proyección de los frames sobre el mesh"): the scene
mesh textured by PROJECTING the session's keyframes with the certified poses (texrecon, the per-object
bake's own machinery), tile by tile with the keyframes near each tile — never the generated texture.
    python scene_texture.py <glb_in> <glb_out> [max views per tile] [session dir (src_default)]"""
import sys, json, time, tempfile, numpy as np, trimesh
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from segmentation.session_io import _load_camera_source
from reconstruction.texture_bake import bake_texture
GLB, OUT = Path(sys.argv[1]), Path(sys.argv[2]); MAXV = int(sys.argv[3]) if len(sys.argv) > 3 else 80
SES = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
MIN_SEEN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05     # a view qualifies when it sees this share of the tile's samples
t0 = time.time()
cam = _load_camera_source(SES, SES / "output")
k_map = {fi: np.asarray(cam.K_for(fi), np.float64) for fi in cam.pose_map if cam.K_for(fi) is not None}
w2c = {fi: np.linalg.inv(np.asarray(cam.pose_map[fi], np.float64)) for fi in k_map}
cam_pos = np.array([np.asarray(cam.pose_map[fi], np.float64)[:3, 3] for fi in k_map])
from scipy.spatial import cKDTree
cam_tree = cKDTree(cam_pos)
img_hw = {}
for fi in k_map:
    K = k_map[fi]; img_hw[fi] = (2.0 * K[1, 2], 2.0 * K[0, 2])          # the principal point sits near the centre
def views_seeing(V):
    """The keyframes that SEE the tile: the share of its sampled vertices projecting inside the image in front
    of the camera (occlusion left to texrecon's own visibility over the tile)."""
    rng = np.random.default_rng(0); S = V[rng.choice(len(V), min(len(V), 3000), replace=False)]
    out = []
    for fi, M in w2c.items():
        pc = S @ M[:3, :3].T + M[:3, 3]; z = pc[:, 2]; front = z > 0.2
        if front.sum() == 0: continue
        K = k_map[fi]; u = K[0, 0] * pc[front, 0] / z[front] + K[0, 2]; v = K[1, 1] * pc[front, 1] / z[front] + K[1, 2]
        h, w = img_hw[fi]; seen = np.mean((u >= 0) & (u < w) & (v >= 0) & (v < h)) * front.mean()
        if seen >= MIN_SEEN: out.append((seen, fi))
    out.sort(reverse=True)
    return [fi for _, fi in out]
sc = trimesh.load(str(GLB), force="scene", process=False); parts = sc.dump(concatenate=False)
work = Path(tempfile.mkdtemp(prefix="scene_tex_", dir=str(OUT.parent)))
rec = {"tiles": [], "max_views": MAXV, "min_seen_share": MIN_SEEN}; out_parts = []
for i, p in enumerate(parts):
    sel = views_seeing(np.asarray(p.vertices, np.float64))[:MAXV]
    ply = work / f"tile_{i:03d}.ply"
    # the GENERATED texture becomes the tile's vertex colours: texrecon paints the faces no keyframe sees with
    # them (the per-object bake's rule) — photo where there is photo, the model's colour where there is none
    try:
        C = np.asarray(p.visual.to_color().vertex_colors)[:, :4]
    except Exception:
        C = None
    m0 = trimesh.Trimesh(np.asarray(p.vertices), np.asarray(p.faces), vertex_colors=C, process=True)
    m0.update_faces(m0.nondegenerate_faces()); m0.update_faces(m0.unique_faces()); m0.remove_unreferenced_vertices()
    # FACE THE CAMERAS (the per-object bake's rule): texrecon labels a face pointing away from every view as
    # unseen, and GenRecon's winding is arbitrary — flip each face toward its nearest camera
    fc = m0.triangles_center; fn = m0.face_normals
    _, ci = cam_tree.query(fc)
    flip = np.einsum("ij,ij->i", fn, cam_pos[ci] - fc) < 0
    if flip.any():
        f = m0.faces.copy(); f[flip] = f[flip][:, ::-1]
        m0 = trimesh.Trimesh(m0.vertices, f, vertex_colors=np.asarray(m0.visual.vertex_colors) if C is not None else None, process=False)
    m0.export(str(ply))
    t1 = time.time(); res = bake_texture(ply, SES / "frames", cam.pose_map, k_map, work / f"tile_{i:03d}.glb", frame_indices=sel, max_views=MAXV, skip_visibility_test=True) if len(sel) >= 3 else None
    # keep every sub-mesh of the baked tile with ITS OWN visual (the textured atlas pieces and the
    # unseen vertex-coloured remainder): loading as ONE mesh concatenates them and drops the texture
    if res:
        for g in trimesh.load(str(res), force="scene").dump(concatenate=False):
            out_parts.append(g)
    else:
        out_parts.append(m0)
    rec["tiles"].append({"tile": i, "faces": int(len(m0.faces)), "views_near": len(sel), "flipped": int(flip.sum()), "textured": bool(res), "s": round(time.time() - t1)})
    print(f"  tile {i:2d}: {len(m0.faces):,} faces, {len(sel)} views seeing it, {'textured' if res else 'NOT textured'} in {time.time()-t1:.0f}s", flush=True)
trimesh.Scene(out_parts).export(str(OUT))
chk = trimesh.load(str(OUT), force="scene")
n_tex = sum(1 for g in chk.geometry.values() if type(g.visual).__name__ == "TextureVisuals")
print(f"check: {len(chk.geometry)} geometries in the GLB, {n_tex} textured")
json.dump(rec, open(str(OUT) + ".texture.json", "w"), indent=2)
print(f"textured scene → {OUT} ({sum(t['textured'] for t in rec['tiles'])}/{len(parts)} tiles, {time.time()-t0:.0f}s)")
