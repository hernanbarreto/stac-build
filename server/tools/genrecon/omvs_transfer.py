"""OpenMVS's TextureMesh on GenRecon's mesh, GenRecon's material carried over (USER 2026-10-17: "¿podremos usar algún paso
del openmvs de texturizado?" + "mantengamos ese tono plástico o metálico que brilla"): TextureMesh re-charts the mesh
into its own atlas (photo texture, seam levelling), so every texel of the NEW atlas is mapped back to the OLD triangle
it lies on (faces matched by centroid, barycentric coordinates in 3-D) and GenRecon's metallic-roughness texture is
sampled there; texels TextureMesh left empty (its sentinel colour) take GenRecon's generated base colour, tone-matched.
    python omvs_transfer.py <session src_default> <openmvs glb> <genrecon glb> <reference ply> <mesh name out>"""
import sys, os, json, time, numpy as np, torch, trimesh
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from PIL import Image
import nvdiffrast.torch as dr
import torch.nn.functional as Fn
from scipy.spatial import cKDTree

SES, GLB_O, GLB_G, PLY_REF, NAME = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
SENTINEL = np.array([255, 0, 255]); t0 = time.time(); dev = torch.device("cuda"); glctx = dr.RasterizeCudaContext()

# OpenMVS's glTF export rotates a Y-up scene (x, y, z → x, z, −y): undo it when the reference PLY says so
sc = trimesh.load(str(GLB_O), force="scene", process=False)
import open3d as o3d
R = np.asarray(o3d.io.read_triangle_mesh(str(PLY_REF)).vertices); ref = np.array([R.min(0), R.max(0)])
M = np.eye(4); M[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
rot = sc.copy(); rot.apply_transform(M)
if np.abs(rot.bounds - ref).max() < np.abs(sc.bounds - ref).max(): sc = rot; print("glTF rotation undone (Rx +90°)")
prims = list(sc.geometry.items())
print(f"OpenMVS texture: {len(prims)} primitives, {sum(len(g.faces) for _, g in prims):,} faces, bounds {np.round(sc.bounds, 2).tolist()}")

# GenRecon: parts with UV atlases, base colour + metallic-roughness textures
gs = trimesh.load(str(GLB_G), force="scene", process=False); gparts = list(gs.geometry.values())
gV = np.concatenate([g.vertices for g in gparts]).astype(np.float64); off = 0; gF, gPart, gUV = [], [], []
for pi, g in enumerate(gparts):
    gF.append(g.faces + off); gPart.append(np.full(len(g.faces), pi)); gUV.append(np.asarray(g.visual.uv, np.float32)); off += len(g.vertices)
gF = np.concatenate(gF); gPart = np.concatenate(gPart); gUV = np.concatenate(gUV)
cent = gV[gF].mean(1); tree = cKDTree(cent)
g_base = [torch.from_numpy(np.asarray(g.visual.material.baseColorTexture.convert("RGB")).astype(np.float32)).to(dev) for g in gparts]
g_mr = [torch.from_numpy(np.asarray(g.visual.material.metallicRoughnessTexture.convert("RGB")).astype(np.float32)).to(dev) for g in gparts]
gV_t = torch.from_numpy(gV.astype(np.float32)).to(dev); gF_t = torch.from_numpy(gF.astype(np.int64)).to(dev); gUV_t = torch.from_numpy(gUV).to(dev); gPart_t = torch.from_numpy(gPart).to(dev)
print(f"GenRecon: {len(gparts)} parts, {len(gF):,} faces ({time.time()-t0:.0f} s)")

def sample(tex, uv):
    """bilinear sample of a (T, T, 3) image-row texture at trimesh UVs (v = 0 at the bottom)."""
    grid = torch.stack([uv[:, 0] * 2 - 1, (1 - uv[:, 1]) * 2 - 1], 1)[None, None]
    return Fn.grid_sample(tex.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T

stats = []
for pi, (pname, g) in enumerate(prims):
    tex = g.visual.material.baseColorTexture.convert("RGB"); Tw, Th = tex.size
    base = torch.from_numpy(np.asarray(tex).astype(np.float32)).to(dev)
    # new face → old face by centroid
    c_new = g.vertices[g.faces].mean(1); d, old = tree.query(c_new, k=1, workers=8)
    matched = d < 1e-3
    uv = torch.from_numpy(np.asarray(g.visual.uv, np.float32)).to(dev); Vp = torch.from_numpy(g.vertices.astype(np.float32)).to(dev); Fp = torch.from_numpy(g.faces.astype(np.int32)).to(dev)
    pos = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], 1)[None].contiguous()
    rast, _ = dr.rasterize(glctx, pos, Fp, resolution=[Th, Tw])
    Pw, _ = dr.interpolate(Vp[None].contiguous(), rast, Fp); Pw = torch.flip(Pw[0], [0]); tri = torch.flip(rast[0, :, :, 3].long() - 1, [0]); cov = tri >= 0
    P = Pw[cov]; fnew = tri[cov]; fold = torch.from_numpy(old).to(dev)[fnew]; ok = torch.from_numpy(matched).to(dev)[fnew]
    # barycentric coordinates of the texel's point in the OLD triangle (least squares in 3-D)
    tv = gV_t[gF_t[fold]]                                         # (n, 3, 3)
    e1, e2 = tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]; w = P - tv[:, 0]
    d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1); dw1, dw2 = (w * e1).sum(1), (w * e2).sum(1)
    det = (d11 * d22 - d12 * d12).clamp(min=1e-12); l1 = (d22 * dw1 - d12 * dw2) / det; l2 = (d11 * dw2 - d12 * dw1) / det
    l1, l2 = l1.clamp(0, 1), l2.clamp(0, 1); l0 = (1 - l1 - l2).clamp(0, 1)
    uv_old = l0[:, None] * gUV_t[gF_t[fold, 0]] + l1[:, None] * gUV_t[gF_t[fold, 1]] + l2[:, None] * gUV_t[gF_t[fold, 2]]
    part = gPart_t[fold]
    mr = torch.zeros((len(P), 3), device=dev); gen = torch.zeros((len(P), 3), device=dev)
    for q in range(len(gparts)):
        sel = (part == q) & ok
        if sel.any(): mr[sel] = sample(g_mr[q], uv_old[sel]); gen[sel] = sample(g_base[q], uv_old[sel])
    photo = base[cov]; empty = (photo - torch.tensor(SENTINEL, device=dev, dtype=torch.float32)).abs().max(1).values < 2
    seen = ~empty & ok
    if seen.sum() > 1000:
        mp, sp = photo[seen].mean(0), photo[seen].std(0); mg, sg = gen[seen].mean(0), gen[seen].std(0)
        gen = ((gen - mg) / sg.clamp(min=1) * sp + mp).clamp(0, 255)
    newbase = base.clone(); newbase[cov] = torch.where(empty[:, None], gen, photo)
    newmr = torch.full_like(base, 255.0); newmr[cov] = mr
    mat = g.visual.material
    mat.baseColorTexture = Image.fromarray(newbase.clamp(0, 255).byte().cpu().numpy())
    mat.metallicRoughnessTexture = Image.fromarray(newmr.clamp(0, 255).byte().cpu().numpy())
    mat.metallicFactor = 1.0; mat.roughnessFactor = 1.0; mat.baseColorFactor = [255, 255, 255, 255]; mat.doubleSided = True
    stats.append({"primitive": pname, "faces": int(len(g.faces)), "matched_faces_frac": float(matched.mean()), "texels": int(cov.sum()), "photographed_frac": float((~empty).float().mean())})
    print(f"  primitive {pi}: {len(g.faces):,} faces ({matched.mean()*100:.1f} % matched to GenRecon's), atlas {Tw}x{Th}, {int(cov.sum()):,} texels, photographed {(~empty).float().mean()*100:.1f} %  ({time.time()-t0:.0f} s)")

d = SES / "output" / "tsdf" / NAME; d.mkdir(parents=True, exist_ok=True)
out = d / f"{NAME}.glb"; tmp = d / f"{NAME}.glb.part"; sc.export(str(tmp), file_type="glb"); os.replace(tmp, out)
meta = {"label": NAME, "glb_file": out.name, "textured": True, "method": "genrecon", "kind": "scene", "instance_id": None,
        "texture": "OpenMVS TextureMesh (view selection by graph cut, global + local seam levelling) on GenRecon's mesh from the certified poses; "
                   "GenRecon's metallic-roughness texture mapped into the new atlas by face correspondence + barycentric coordinates; "
                   "texels no keyframe sees take GenRecon's generated base colour, tone-matched",
        "provenance": "geometry + material: vlm_generated (GenRecon); base colour: tool_measured (OpenMVS TextureMesh, AGPL-3.0 tool)",
        "source_glb": str(GLB_G), "openmvs_glb": str(GLB_O), "primitives": stats, "bounds_published": sc.bounds.tolist(), "published_at": time.strftime("%Y-%m-%d %H:%M")}
(d / f"{NAME}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"published {out} ({out.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f} s")
