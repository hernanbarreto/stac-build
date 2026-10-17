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
# OUR VISIBILITY (USER 2026-10-17: zones textured from views that do not see them): the cloud's z-buffer per
# keyframe (the measured truth, min-filtered) decides which views see a face; texrecon gets the labeling and
# only cuts patches and levels seams
import torch
from segmentation.perfect_object import _read_ply_fields
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_f = _read_ply_fields(SES / "output" / "cleaned_cloud.ply"); _xyz = np.column_stack([_f["x"], _f["y"], _f["z"]]).astype(np.float32)
Pc = torch.from_numpy(_xyz).to(dev); del _f, _xyz
fids = sorted(k_map); FAR = 1.0e6
K0 = k_map[fids[0]]; Wd, Hd = int(round(2 * K0[0, 2])), int(round(2 * K0[1, 2]))
w2c_t = {fi: torch.from_numpy(np.linalg.inv(np.asarray(cam.pose_map[fi], np.float64)).astype(np.float32)).to(dev) for fi in fids}
zbuf = {}
for fi in fids:
    Mk = w2c_t[fi]; pc = Pc @ Mk[:3, :3].T + Mk[:3, 3]; z = pc[:, 2]; fr = z > 0.2
    K = k_map[fi]; u = (K[0, 0] * pc[fr, 0] / z[fr] + K[0, 2]).long(); v = (K[1, 1] * pc[fr, 1] / z[fr] + K[1, 2]).long()
    inb = (u >= 0) & (u < Wd) & (v >= 0) & (v < Hd)
    zb = torch.full((Hd * Wd,), FAR, device=dev); zb.scatter_reduce_(0, (v[inb] * Wd + u[inb]), z[fr][inb], reduce="amin")
    zb = -torch.nn.functional.max_pool2d(-zb.view(1, 1, Hd, Wd), kernel_size=5, stride=1, padding=2)[0, 0]
    zbuf[fi] = zb
del Pc
print(f"z-buffers of the cloud for {len(fids)} keyframes ({Hd}x{Wd})", flush=True)

def labels_for(mesh, sel_sorted):
    """Per face: the view of sorted(sel) that sees it best (largest projected area among the views where its
    three vertices project inside the image in front of the camera, the face points at the camera, and its
    centre is not behind the cloud's surface), as texrecon's label (1-based; 0 = no view)."""
    V = torch.from_numpy(np.asarray(mesh.vertices, np.float32)).to(dev); F = torch.from_numpy(np.asarray(mesh.faces, np.int64)).to(dev)
    N = torch.from_numpy(np.asarray(mesh.face_normals, np.float32)).to(dev); C = V[F].mean(1)
    best_q = torch.zeros(len(F), device=dev); best_l = torch.zeros(len(F), dtype=torch.int64, device=dev)
    for j, fi in enumerate(sel_sorted):
        Mk = w2c_t[fi]; K = k_map[fi]
        P3 = V[F] @ Mk[:3, :3].T + Mk[:3, 3]                                 # (nF, 3, 3)
        z3 = P3[..., 2]; front = (z3 > 0.2).all(1); zs = torch.where(z3 > 0.2, z3, torch.ones_like(z3))
        u3 = K[0, 0] * P3[..., 0] / zs + K[0, 2]; v3 = K[1, 1] * P3[..., 1] / zs + K[1, 2]
        inb = front & (u3 >= 1).all(1) & (u3 <= Wd - 2).all(1) & (v3 >= 1).all(1) & (v3 <= Hd - 2).all(1)
        cam_p = torch.from_numpy(np.asarray(cam.pose_map[fi], np.float64)[:3, 3].astype(np.float32)).to(dev)
        facing = ((cam_p - C) * N).sum(1) > 0
        pcc = C @ Mk[:3, :3].T + Mk[:3, 3]; zc = pcc[:, 2]; zcs = torch.where(zc > 0.2, zc, torch.ones_like(zc))
        uc = (K[0, 0] * pcc[:, 0] / zcs + K[0, 2]).long().clamp(0, Wd - 1); vc = (K[1, 1] * pcc[:, 1] / zcs + K[1, 2]).long().clamp(0, Hd - 1)
        zb = zbuf[fi][vc, uc]
        visible = zc <= zb + (0.03 + 0.02 * zc)
        area = 0.5 * torch.abs((u3[:, 1] - u3[:, 0]) * (v3[:, 2] - v3[:, 0]) - (u3[:, 2] - u3[:, 0]) * (v3[:, 1] - v3[:, 0]))
        q = torch.where(inb & facing & visible, area, torch.zeros_like(area))
        upd = q > best_q
        best_q = torch.where(upd, q, best_q); best_l = torch.where(upd, torch.full_like(best_l, j + 1), best_l)
    return best_l.cpu().numpy()
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
    t1 = time.time(); res = None
    if len(sel) >= 3:
        sel_sorted = sorted(sel)
        lab = labels_for(m0, sel_sorted)
        n_seen = int((lab > 0).sum())
        res = bake_texture(ply, SES / "frames", cam.pose_map, k_map, work / f"tile_{i:03d}.glb", frame_indices=sel_sorted, max_views=max(MAXV, len(sel_sorted)),
                           labeling=lab) if n_seen else None
    else:
        n_seen = 0
    # keep every sub-mesh of the baked tile with ITS OWN visual (the textured atlas pieces and the
    # unseen vertex-coloured remainder): loading as ONE mesh concatenates them and drops the texture
    if res:
        for g in trimesh.load(str(res), force="scene").dump(concatenate=False):
            out_parts.append(g)
    else:
        out_parts.append(m0)
    rec["tiles"].append({"tile": i, "faces": int(len(m0.faces)), "views_near": len(sel), "flipped": int(flip.sum()), "faces_seen": n_seen, "textured": bool(res), "s": round(time.time() - t1)})
    print(f"  tile {i:2d}: {len(m0.faces):,} faces, {len(sel)} views seeing it, {n_seen:,} faces with a view, {'textured' if res else 'NOT textured'} in {time.time()-t1:.0f}s", flush=True)
import os
trimesh.Scene(out_parts).export(str(OUT) + ".part", file_type="glb"); os.replace(str(OUT) + ".part", str(OUT))   # atomic for the viewer's watcher
chk = trimesh.load(str(OUT), force="scene")
n_tex = sum(1 for g in chk.geometry.values() if type(g.visual).__name__ == "TextureVisuals")
print(f"check: {len(chk.geometry)} geometries in the GLB, {n_tex} textured")
json.dump(rec, open(str(OUT) + ".texture.json", "w"), indent=2)
print(f"textured scene → {OUT} ({sum(t['textured'] for t in rec['tiles'])}/{len(parts)} tiles, {time.time()-t0:.0f}s)")
