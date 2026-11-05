"""GenRecon's per-chunk bake, several chunks at once (station 2026-11-02: 100 stacked 3.7 m cubes, the bake one
chunk after another in ONE process — its UV parameterisation is single-threaded CPU and the decoder's surface
comes in tens of thousands of small clusters per chunk: 1-2.5 min a chunk, ~3-4 h). The chunks are independent once
the global stage has run: chunked_to_glb.py --dump_geometry_plys writes every chunk's pre-bake mesh
(chunk_NNN_geom.ply); this worker bakes the chunks it is given EXACTLY as inference/chunked_glb.py's loop does
(o_voxel.postprocess.to_glb on that mesh, decimation min(simplify_threshold, faces), remesh off, the attribute
volume cropped to the chunk ± STAC_BAKE_CROP_MARGIN voxels as the patched loop) and writes chunk_NNN.glb into the
chunk cache, which chunked_to_glb.py then resumes from. A chunk already in the cache, or claimed by another worker
(a .lock), is skipped; a GLB is written beside and renamed.

  python bake_chunks_parallel.py <to_glb_inputs.pt> <dump dir> <chunks dir> <k> <n> [simplify] [texture]
  (worker k of n takes the chunks whose position from the END is ≡ k mod n)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

inputs, dump, cache, k, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
simplify = int(sys.argv[6]) if len(sys.argv) > 6 else 150_000
texsize = int(sys.argv[7]) if len(sys.argv) > 7 else 2048
margin = int(os.environ.get("STAC_BAKE_CROP_MARGIN", "2"))
sys.path.insert(0, "/workspace/stac-build/vendor/GenRecon")
import o_voxel.postprocess as op  # noqa: E402

d = torch.load(inputs, map_location="cpu", weights_only=False)
dev = "cuda"
attr_volume = d["attr_volume"].to(dev)
coords = d["coords"].to(dev)
aabb_t = torch.as_tensor(np.asarray(d["aabb"], np.float32), device=dev)
vs = float(d["voxel_size"])
geoms = sorted(dump.glob("chunk_*_geom.ply"), reverse=True)            # from the END


def _ply_faces(path):
    """The face count from a PLY header (no body read)."""
    with open(path, "rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element face"):
                return int(line.split()[-1])
            if line == "end_header":
                return 0
    return 0


# THE SINGLE GLB'S LIMIT (zaragoza 2026-11-05: 431 cubes of 2.85 m undecimated would write ~6.5 GB of geometry): a GLB
# counts its length in 32 bits and the tiler reads it into one Node buffer (4 GiB), so "no decimation" (simplify ≥ 1e9)
# becomes a TOTAL face budget shared in proportion — every chunk keeps the same fraction of its own detail — and the
# chunk atlases (base colour + metallic-roughness each) a total byte budget. Bytes per face and per texel MEASURED on
# zaragoza's raw GLB of 2026-11-05 (17.8 M faces, 268 textures of 2048²: 1364 MB → ~34.5 B/face, ~0.67 B/texel).
FACE_BUDGET = int(os.environ.get("STAC_FACE_BUDGET", "64000000"))
ATLAS_BYTES_BUDGET = float(os.environ.get("STAC_ATLAS_BYTES_BUDGET", "6e8"))
counts = {g: _ply_faces(g) for g in geoms}
total_faces = sum(counts.values())
share = 1.0
if simplify >= 1_000_000_000 and total_faces > FACE_BUDGET:
    share = FACE_BUDGET / total_faces
GLB_LIMIT = 3.5e9                     # bytes: the 4 GiB GLB / Node buffer, with a margin
est = share * total_faces * 34.5 + len(geoms) * 2 * texsize * texsize * 0.67
if est > GLB_LIMIT:                   # only then are the chunk atlases reduced (as always otherwise: texsize)
    side = 2 ** int(np.floor(np.log2(max(64.0, np.sqrt(ATLAS_BYTES_BUDGET / (2 * max(1, len(geoms)) * 0.67))))))
    if side < texsize:
        texsize = int(side)
if k == 0:
    print(f"[bake-worker] {len(geoms)} chunks, {total_faces:,} faces before decimation; each keeps "
          f"{share * 100:.1f} % of its own (total budget {FACE_BUDGET:,}); atlases {texsize}² "
          f"({len(geoms)} x 2 within {ATLAS_BYTES_BUDGET / 1e9:.1f} GB)", flush=True)
mine = [g for i, g in enumerate(geoms) if i % n == k]
cache.mkdir(parents=True, exist_ok=True)
for g in mine:
    idx = int(g.stem.split("_")[1])
    out = cache / f"chunk_{idx:03d}.glb"
    lock = cache / f"chunk_{idx:03d}.lock"
    if out.exists():
        continue
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        continue
    try:
        m = trimesh.load(str(g), process=False)
        chunk_v = torch.as_tensor(np.asarray(m.vertices, np.float32), device=dev).contiguous()
        chunk_f = torch.as_tensor(np.asarray(m.faces, np.int32), device=dev).contiguous()
        lo = torch.floor((chunk_v.min(dim=0).values - aabb_t[0]) / vs) - margin
        hi = torch.ceil((chunk_v.max(dim=0).values - aabb_t[0]) / vs) + margin
        keep = ((coords[:, -3:].float() >= lo) & (coords[:, -3:].float() <= hi)).all(dim=1)
        if not bool(keep.any()):              # no voxel near this chunk: the whole volume, as the original loop
            keep = torch.ones_like(keep)
        t1 = time.perf_counter()
        glb = op.to_glb(vertices=chunk_v, faces=chunk_f, attr_volume=attr_volume[keep], coords=coords[keep],
                        attr_layout=d["attr_layout"], aabb=aabb_t, voxel_size=vs,
                        decimation_target=max(1000, min(simplify, int(np.ceil(int(chunk_f.shape[0]) * share)))), texture_size=texsize,
                        remesh=False, verbose=False)
        part = cache / f"chunk_{idx:03d}.part.glb"
        (glb if isinstance(glb, trimesh.Scene) else trimesh.Scene([glb])).export(str(part))
        os.replace(part, out)
        print(f"[bake-worker {k}/{n}] chunk {idx:03d}: {int(chunk_f.shape[0]):,} faces baked in "
              f"{time.perf_counter() - t1:.1f}s", flush=True)
    finally:
        lock.unlink(missing_ok=True)
        torch.cuda.empty_cache()
print(f"[bake-worker {k}/{n}] done", flush=True)
