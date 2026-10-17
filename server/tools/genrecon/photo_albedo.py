"""The REAL texture on GenRecon's mesh, its material kept (USER 2026-10-17: "tenemos que generar la textura real sobre la
malla de genrecon, y mantengamos ese tono plástico o metálico que brilla"): every texel of GenRecon's OWN UV atlas is
re-coloured from the keyframes that SEE its surface point — the mesh's own depth per keyframe (nvdiffrast) decides the
occlusion, the view that faces the surface most closely weighs most (cos⁴/z², the best three blended) — and the
metallic-roughness texture, the factors and the UV layout stay exactly as generated, so the look he likes is untouched.
Texels no keyframe sees keep the generated colour, tone-matched (per part, per channel) to the photographed ones.
    python photo_albedo.py <session src_default> <glb in> <mesh name out> [top-k 3]
Record: output/tsdf/<name>/<name>.meta.json (provenance: geometry + material vlm_generated, base colour tool_measured)."""
import sys, os, json, time, numpy as np, torch, trimesh
sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from PIL import Image
import nvdiffrast.torch as dr
import torch.nn.functional as Fn
from segmentation.session_io import _load_camera_source

SES = Path(sys.argv[1]); GLB = Path(sys.argv[2]); NAME = sys.argv[3]
TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 3
TOL_M, TOL_REL, COS_MIN, PAD_PX = 0.03, 0.01, 0.2, 3
t0 = time.time(); dev = torch.device("cuda"); glctx = dr.RasterizeCudaContext()
NEAR, FARP = 0.05, 100.0

# ── the session: keyframes, certified poses, K ───────────────────────────────────────────────────────────────────
cam = _load_camera_source(SES, SES / "output")
fids = sorted(fi for fi in cam.pose_map if cam.K_for(fi) is not None and (SES / "frames" / f"{int(fi):06d}.jpg").exists())
K = np.array([np.asarray(cam.K_for(fi), np.float64) for fi in fids]); c2w = np.array([np.asarray(cam.pose_map[fi], np.float64) for fi in fids])
w2c = np.linalg.inv(c2w); cam_pos = c2w[:, :3, 3]
img0 = Image.open(SES / "frames" / f"{int(fids[0]):06d}.jpg"); W, H = img0.size
W8, H8 = (W + 7) // 8 * 8, (H + 7) // 8 * 8          # the rasterizer renders a multiple of 8; the padding rows/cols are cropped
imgs = np.stack([np.asarray(Image.open(SES / "frames" / f"{int(fi):06d}.jpg").convert("RGB")) for fi in fids])   # (n, H, W, 3) uint8
imgs_t = torch.from_numpy(imgs).to(dev)
print(f"{len(fids)} keyframes {W}x{H}, K[0] fx {K[0,0,0]:.1f} fy {K[0,1,1]:.1f} cx {K[0,0,2]:.1f} cy {K[0,1,2]:.1f}")

# ── the mesh: GenRecon's parts, each with its UV atlas, base colour and metallic-roughness textures ───────────────
sc = trimesh.load(str(GLB), force="scene", process=False)
# the scene graph carries the raw↔display transform of the session (identity on pccr, not elsewhere): every part's
# vertices and normals are taken in the WORLD frame; the textures are written back into the same geometry objects
node_T = {}
for node in sc.graph.nodes_geometry:
    T, gname = sc.graph[node]; node_T[gname] = np.asarray(T, np.float64)
parts = []
for n, g in sc.geometry.items():
    T = node_T.get(n, np.eye(4)); Vw = (g.vertices @ T[:3, :3].T + T[:3, 3]).astype(np.float64)
    parts.append((n, g, Vw))
V_all = np.concatenate([Vw for _, _, Vw in parts]).astype(np.float32); off = 0; F_all = []
for _, g, _ in parts:
    F_all.append(g.faces + off); off += len(g.vertices)
F_all = np.concatenate(F_all).astype(np.int32)
Vt = torch.from_numpy(V_all).to(dev); Ft = torch.from_numpy(F_all).to(dev)
print(f"{len(parts)} parts, {len(V_all):,} vertices, {len(F_all):,} faces, bounds {np.round(V_all.min(0),2).tolist()}..{np.round(V_all.max(0),2).tolist()}")

def clip_coords(V, Kk, w2ck, flip_y):
    """camera-space → clip-space for nvdiffrast, OpenGL-style depth; the image-y direction is decided by a self-check."""
    Pc = V @ torch.from_numpy(w2ck[:3, :3].T.astype(np.float32)).to(dev) + torch.from_numpy(w2ck[:3, 3].astype(np.float32)).to(dev)
    X, Y, Z = Pc[:, 0], Pc[:, 1], Pc[:, 2]
    fx, fy, cx, cy = float(Kk[0, 0]), float(Kk[1, 1]), float(Kk[0, 2]), float(Kk[1, 2])
    xn = (2 * fx * X + (2 * cx - W8) * Z) / W8; yn = (2 * fy * Y + (2 * cy - H8) * Z) / H8
    if flip_y: yn = -yn
    zc = (Z * (FARP + NEAR) - 2 * FARP * NEAR) / (FARP - NEAR)
    return torch.stack([xn, yn, zc, Z], 1)[None].contiguous(), Z.contiguous()

def render_depth(k, flip_y):
    pos, Z = clip_coords(Vt, K[k], w2c[k], flip_y)
    rast, _ = dr.rasterize(glctx, pos, Ft, resolution=[H8, W8])
    z, _ = dr.interpolate(Z[None, :, None].contiguous(), rast, Ft)
    z = z[0, :, :, 0]; z = torch.where(rast[0, :, :, 3] > 0, z, torch.full_like(z, 1e6))
    return z            # (H8, W8) rows in nvdiffrast's order (y = −1 first); the self-check below fixes the convention

# self-check of the image-y convention against the certified cloud's own z-buffer on one keyframe
from segmentation.perfect_object import _read_ply_fields
_f = _read_ply_fields(SES / "output" / "cleaned_cloud.ply"); cloud = torch.from_numpy(np.column_stack([_f["x"], _f["y"], _f["z"]]).astype(np.float32)).to(dev); del _f
def cloud_zbuf(k):
    Pc = cloud @ torch.from_numpy(w2c[k][:3, :3].T.astype(np.float32)).to(dev) + torch.from_numpy(w2c[k][:3, 3].astype(np.float32)).to(dev)
    z = Pc[:, 2]; fr = z > 0.2; u = (K[k][0, 0] * Pc[fr, 0] / z[fr] + K[k][0, 2]).long(); v = (K[k][1, 1] * Pc[fr, 1] / z[fr] + K[k][1, 2]).long()
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H); zb = torch.full((H * W,), 1e6, device=dev)
    zb.scatter_reduce_(0, v[inb] * W + u[inb], z[fr][inb], reduce="amin"); return zb.view(H, W)
k0 = len(fids) // 2; zc0 = cloud_zbuf(k0); best = None
for flip_y in (False, True):
    for flip_rows in (False, True):
        zm = render_depth(k0, flip_y); zm = (torch.flip(zm, [0]) if flip_rows else zm)[:H, :W]
        both = (zc0 < 1e5) & (zm < 1e5); err = (zm[both] - zc0[both]).abs().median().item() if both.any() else 9.9
        if best is None or err < best[0]: best = (err, flip_y, flip_rows)
ERR0, FLIP_Y, FLIP_ROWS = best
print(f"image convention: flip_y={FLIP_Y} flip_rows={FLIP_ROWS} (mesh depth vs cloud z-buffer median |Δ| {ERR0*100:.1f} cm on keyframe {fids[k0]})")
if ERR0 > 0.25:        # a wrong frame disagrees by metres (bufferStop before the world-frame fix: 332 cm); a right one by centimetres
    raise SystemExit(f"bake failed: the mesh and the keyframes' poses are not in the same frame (median |Δ| {ERR0*100:.0f} cm)")
del cloud

# the mesh's own depth per keyframe (min-filtered 3×3: no leak at silhouettes), in IMAGE row order
depth = torch.empty((len(fids), H, W), device=dev)
for k in range(len(fids)):
    z = render_depth(k, FLIP_Y); z = (torch.flip(z, [0]) if FLIP_ROWS else z)[:H, :W]
    depth[k] = -Fn.max_pool2d(-z[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
print(f"mesh depth rendered for {len(fids)} keyframes ({time.time()-t0:.0f} s)")

# ── per part: rasterize the UV atlas, colour every covered texel from the views that see it ──────────────────────
w2c_t = torch.from_numpy(w2c.astype(np.float32)).to(dev); K_t = torch.from_numpy(K.astype(np.float32)).to(dev); cam_t = torch.from_numpy(cam_pos.astype(np.float32)).to(dev)
stats = []; new_textures = {}; UV_FLIP = None
for pi, (pname, g, Vw) in enumerate(parts):
    tex = g.visual.material.baseColorTexture; T = tex.size[0]; assert tex.size == (T, T) and T % 8 == 0, tex.size
    base = torch.from_numpy(np.asarray(tex.convert("RGB")).astype(np.float32)).to(dev)             # (T, T, 3) image rows
    uv = torch.from_numpy(np.asarray(g.visual.uv, np.float32)).to(dev); Vp = torch.from_numpy(Vw.astype(np.float32)).to(dev)
    Fp = torch.from_numpy(g.faces.astype(np.int32)).to(dev)
    pos = torch.cat([uv * 2 - 1, torch.zeros_like(uv[:, :1]), torch.ones_like(uv[:, :1])], 1)[None].contiguous()
    rast, _ = dr.rasterize(glctx, pos, Fp, resolution=[T, T])
    if UV_FLIP is None:    # self-check: interpolate the UVs themselves; texel (r, c) must read (c/T, 1 − r/T) in image row order or its flip
        uvi, _ = dr.interpolate(uv[None].contiguous(), rast, Fp); uvi = uvi[0]; cov = rast[0, :, :, 3] > 0
        rr, cc = torch.meshgrid(torch.arange(T, device=dev), torch.arange(T, device=dev), indexing="ij")
        e_top = ((uvi[..., 1] - (1 - (rr + 0.5) / T)).abs()[cov]).mean().item(); e_bot = ((uvi[..., 1] - (rr + 0.5) / T).abs()[cov]).mean().item()
        UV_FLIP = e_top > e_bot            # True: raster row 0 is v = 0 (bottom) → flip rows to image order
        print(f"atlas convention: flip rows={UV_FLIP} (|Δv| top-order {e_top:.4f} vs bottom-order {e_bot:.4f})")
    Pw, _ = dr.interpolate(Vp[None].contiguous(), rast, Fp); Pw = Pw[0]; tri = rast[0, :, :, 3].long() - 1; cov = tri >= 0
    if UV_FLIP: Pw, tri, cov = torch.flip(Pw, [0]), torch.flip(tri, [0]), torch.flip(cov, [0])
    fw = Vw[g.faces]; fn_np = np.cross(fw[:, 1] - fw[:, 0], fw[:, 2] - fw[:, 0]); fn_np /= np.linalg.norm(fn_np, axis=1, keepdims=True).clip(1e-12)
    fn = torch.from_numpy(fn_np.astype(np.float32)).to(dev)                                           # world-frame face normals
    P = Pw[cov]; N = fn[tri[cov]]; n_tex = int(cov.sum())
    # candidate views: every keyframe whose frustum holds a point of the part's box (its 3×3×3 lattice) or whose
    # camera stands inside it — no distance bound: a hall's far wall is photographed from 20 m away
    lo, hi = Vp.min(0).values, Vp.max(0).values
    lat = torch.stack(torch.meshgrid(*[torch.linspace(float(lo[i]), float(hi[i]), 3, device=dev) for i in range(3)], indexing="ij"), -1).view(-1, 3)
    Lc = torch.einsum("bij,nj->bni", w2c_t[:, :3, :3], lat) + w2c_t[:, :3, 3][:, None]; Zl = Lc[..., 2]; zl = torch.where(Zl > 0.1, Zl, torch.ones_like(Zl))
    ul = K_t[:, 0, 0][:, None] * Lc[..., 0] / zl + K_t[:, 0, 2][:, None]; vl = K_t[:, 1, 1][:, None] * Lc[..., 1] / zl + K_t[:, 1, 2][:, None]
    in_frustum = ((Zl > 0.1) & (ul >= 0) & (ul < W) & (vl >= 0) & (vl < H)).any(1)
    inside = ((cam_t >= lo) & (cam_t <= hi)).all(1)
    views = torch.nonzero(in_frustum | inside).flatten().tolist()
    wtop = torch.zeros((TOPK, n_tex), device=dev); ctop = torch.zeros((TOPK, n_tex, 3), device=dev)
    for v0 in range(0, len(views), 16):
        vb = views[v0:v0 + 16]; B = len(vb)
        Rw = w2c_t[vb, :3, :3]; tw = w2c_t[vb, :3, 3]
        Pc = torch.einsum("bij,nj->bni", Rw, P) + tw[:, None]; Z = Pc[..., 2]; zs = torch.where(Z > 0.1, Z, torch.ones_like(Z))
        u = K_t[vb, 0, 0][:, None] * Pc[..., 0] / zs + K_t[vb, 0, 2][:, None]; vv = K_t[vb, 1, 1][:, None] * Pc[..., 1] / zs + K_t[vb, 1, 2][:, None]
        inb = (Z > 0.1) & (u >= 0) & (u < W - 1) & (vv >= 0) & (vv < H - 1)
        ui, vi = u.clamp(0, W - 1).long(), vv.clamp(0, H - 1).long()
        zb = torch.stack([depth[k][vi[b], ui[b]] for b, k in enumerate(vb)])
        vis = inb & (Z <= zb + TOL_M + TOL_REL * Z)
        dirs = cam_t[vb][:, None] - P[None]; dirs = dirs / dirs.norm(dim=2, keepdim=True).clamp(min=1e-6)
        cosang = (dirs * N[None]).sum(2).abs(); ok = vis & (cosang > COS_MIN)
        w = torch.where(ok, cosang ** 4 / (Z.clamp(min=0.3) ** 2), torch.zeros_like(Z))
        grid = torch.stack([u / (W - 1) * 2 - 1, vv / (H - 1) * 2 - 1], 2)[:, None]             # (B, 1, n, 2)
        col = Fn.grid_sample(imgs_t[vb].permute(0, 3, 1, 2).float(), grid, mode="bilinear", align_corners=True)[:, :, 0].permute(0, 2, 1)  # (B, n, 3)
        wall = torch.cat([wtop, w], 0); call = torch.cat([ctop, col], 0)
        wtop, idx = wall.topk(TOPK, dim=0); ctop = torch.gather(call, 0, idx[..., None].expand(-1, -1, 3))
    wsum = wtop.sum(0); seen = wsum > 0
    photo = (ctop * wtop[..., None]).sum(0) / wsum.clamp(min=1e-9)[:, None]
    # unseen texels keep the generated colour, tone-matched per channel to the photographed ones (affine on mean/std)
    gen = base[cov]
    if seen.sum() > 1000:
        mp, sp = photo[seen].mean(0), photo[seen].std(0); mg, sg = gen[seen].mean(0), gen[seen].std(0)
        gen_m = ((gen - mg) / sg.clamp(min=1) * sp + mp).clamp(0, 255)
    else:
        gen_m = gen
    out = torch.where(seen[:, None], photo, gen_m)
    newtex = base.clone(); newtex[cov] = out
    # a few texels of padding past the triangle edges so mip-mapping does not bleed the old colour at seams
    m = cov.float()[None, None]; img = newtex.permute(2, 0, 1)[None]
    for _ in range(PAD_PX):
        grown = Fn.max_pool2d(m, 3, 1, 1); summed = Fn.avg_pool2d(img * m, 3, 1, 1) * 9; cnt = Fn.avg_pool2d(m, 3, 1, 1) * 9
        fill = summed / cnt.clamp(min=1); img = torch.where((grown > 0) & (m == 0), fill, img); m = grown
    newtex = img[0].permute(1, 2, 0)
    new_textures[pname] = Image.fromarray(newtex.clamp(0, 255).byte().cpu().numpy())
    stats.append({"part": pname, "faces": int(len(g.faces)), "texels": n_tex, "seen_frac": float(seen.float().mean()), "views": len(views)})
    print(f"  part {pi:2d}: {len(g.faces):>8,} faces, {n_tex:>9,} texels, {len(views):3d} candidate views, photographed {seen.float().mean()*100:5.1f} %  ({time.time()-t0:.0f} s)")

# ── publish: the same mesh, the same UVs, the same metallic-roughness, only the base colour replaced ─────────────
for pname, g, _ in parts:
    g.visual.material.baseColorTexture = new_textures[pname]
d = SES / "output" / "tsdf" / NAME; d.mkdir(parents=True, exist_ok=True)
out = d / f"{NAME}.glb"; tmp = d / f"{NAME}.glb.part"; sc.export(str(tmp), file_type="glb"); os.replace(tmp, out)
src_meta = json.loads((GLB.parent / (GLB.stem + ".meta.json")).read_text()) if (GLB.parent / (GLB.stem + ".meta.json")).exists() else {}
seen_all = sum(s["seen_frac"] * s["texels"] for s in stats) / max(1, sum(s["texels"] for s in stats))
meta = dict(src_meta); meta.update({"label": NAME, "glb_file": out.name, "textured": True, "method": src_meta.get("method", "genrecon"), "kind": "scene",
    "texture": "photo_albedo: base colour re-projected from the keyframes with the certified poses into GenRecon's own UV atlas "
               f"(mesh depth occlusion, cos⁴/z² weights, best {TOPK} blended, {PAD_PX} px padding); unseen texels keep the generated colour tone-matched; "
               "metallic-roughness texture, factors and UVs as generated",
    "provenance": "geometry + material: vlm_generated (GenRecon, presence from the certified cloud); base colour: tool_measured (keyframe projection)",
    "source_glb": str(GLB), "photographed_texel_frac": seen_all, "parts": stats, "image_convention": {"flip_y": FLIP_Y, "flip_rows": FLIP_ROWS, "atlas_flip": bool(UV_FLIP), "check_cm": ERR0 * 100},
    "published_at": time.strftime("%Y-%m-%d %H:%M")})
(d / f"{NAME}.meta.json").write_text(json.dumps(meta, indent=2))
print(f"published {out} ({out.stat().st_size/1e6:.0f} MB): {seen_all*100:.1f} % of the covered texels photographed; {time.time()-t0:.0f} s")
