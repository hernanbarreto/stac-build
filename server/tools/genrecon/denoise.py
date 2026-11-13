"""The mesh SMOOTH and its edges kept, on the card, before the bake (USER 2026-11-12: "pero no se ve nada liso" /
"no tiene la calidad de genrecon lo que veo" / "la malla debe ser correcta … lisa … con detalle").

Bilateral normal filtering (Zheng, Fu, Au & Tai 2011, "Bilateral Normal Filtering for Mesh Denoising"): every face's
normal becomes the average of the normals of the faces sharing a vertex with it, weighted by their area, by how near
their centres are (a Gaussian of the mean distance between neighbouring centres) and by how alike their normals are
(a Gaussian of SIGMA_R in the normals' difference) — the cloud's noise and the decoder's voxel quilt average out on a
face, while across a crease the other face's normals weigh nothing and the crease stays; repeated NORMAL_ITERS times.
Then the vertices move to fit the filtered normals (each toward the planes of its faces, VERTEX_ITERS times). GenRecon's
own Taubin smoothing locks every vertex whose faces bend more than 25 degrees, which on a noisy surface is most of it:
the noise stayed. Deterministic (torch's deterministic algorithms), float64. GenRecon's frame.

Two callers:
- the GENERATED mesh (USER 2026-11-12 night, after four side-by-side views of bufferStop with and without it — "No noto
  la diferencia": "Lo dejamos el filtro igual porque podrían haber mallas que si lo requieran. Implementamos el filtro y
  hacemos las mallas de todas las sesiones con el filtro"): GenRecon's mesh step loads this file by its path and calls
  filter_mesh at its Step 1.5 — after the global remesh, before the chunks' simplification and bake
  (vendor inference/chunked_glb.py, chunked_to_glb.py --edge_filter);
- the shape from the cloud (run_cloud.sh, this file's command line): the decoded mesh before the bake.

MASSIVE MESHES (USER 2026-11-12: "arreglá genrecon y hacelo escalable que pueda manejar nubes masivas"): the pairs of
neighbouring faces are never held all at once — they are made for BLOCK_FACES source faces at a time, from the faces
around each vertex, at every pass. Every face's sums run over its own neighbours in the same order whatever the
blocks, so the result does not depend on BLOCK_FACES (tests/test_cloud_shape.py holds it bit for bit).

Usage: denoise.py <to_glb_inputs.pt in> <out .pt> <normal iters> <vertex iters> <sigma_r> <block faces>"""
from __future__ import annotations

import math
import sys
import time

import torch


def _centres(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    cen = V[F[:, 0]] + V[F[:, 1]]
    cen += V[F[:, 2]]
    cen /= 3
    return cen


def _face_geometry(V: torch.Tensor, F: torch.Tensor):
    """(unit normal, centre, area) of every face."""
    a = V[F[:, 0]]
    e1 = V[F[:, 1]] - a
    e2 = V[F[:, 2]] - a
    del a
    cr = torch.cross(e1, e2, dim=1)
    del e1, e2
    length = cr.norm(dim=1)
    return cr / length[:, None].clamp_min(1e-20), _centres(V, F), 0.5 * length


def _incidence(F: torch.Tensor, n_vertices: int):
    """The faces around every vertex: those of vertex v are inc[start[v] : start[v] + deg[v]], in the faces' order."""
    corners = F.reshape(-1)
    inc = torch.argsort(corners, stable=True) // 3
    deg = torch.bincount(corners, minlength=n_vertices)
    return inc, torch.cumsum(deg, 0) - deg, deg


def _block_pairs(F: torch.Tensor, inc: torch.Tensor, start: torch.Tensor, deg: torch.Tensor, a: int, b: int):
    """(src, dst) of the source faces a … b - 1: every ordered pair of DIFFERENT faces sharing at least one vertex,
    sorted by src then dst — each pair once, however many vertices the two faces share."""
    dev, nf = F.device, len(F)
    v = F[a:b].reshape(-1)
    d = deg[v]                                         # each corner pairs with every face around its vertex
    src = torch.arange(a, b, device=dev).repeat_interleave(3).repeat_interleave(d)
    k = torch.arange(len(src), device=dev) - torch.repeat_interleave(torch.cumsum(d, 0) - d, d)
    dst = inc[torch.repeat_interleave(start[v], d) + k]
    keep = src != dst
    key = torch.unique(src[keep] * nf + dst[keep])
    return key // nf, key % nf


def median_p99(x: torch.Tensor):
    """(median, 99th percentile) of a 1-D float64 tensor, the values torch.median and torch.quantile(0.99) give, from ONE
    sort: torch.quantile refuses an input over 16,777,216 elements, and a scene's mesh has more faces than that."""
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    xs = torch.sort(x).values
    pos = 0.99 * (n - 1)
    lo = math.floor(pos)
    w = torch.tensor(pos - lo, dtype=xs.dtype, device=xs.device)
    return float(xs[(n - 1) // 2]), float(torch.lerp(xs[lo], xs[math.ceil(pos)], w))


def bilateral_denoise(V: torch.Tensor, F: torch.Tensor, normal_iters: int, vertex_iters: int, sigma_r: float,
                      block_faces: int | None = None):
    """The vertices moved to the bilaterally filtered normals; (V, report). The neighbour pairs of BLOCK_FACES source
    faces are in memory at a time (None: of all the faces); the result does not depend on it."""
    if V.dtype != torch.float64:
        raise TypeError(f"the filter runs in float64 (got {V.dtype})")
    F = F.long()
    nv, nf = len(V), len(F)
    if nf >= 1 << 31:
        raise ValueError(f"{nf:,} faces: a pair of faces no longer fits one 64-bit key")
    step = max(1, nf if block_faces is None else int(block_faces))
    blocks = [(a, min(a + step, nf)) for a in range(0, nf, step)]
    inc, start, deg = _incidence(F, nv)
    n, cen, area = _face_geometry(V, F)
    # the centres' sigma = the mean distance between neighbouring centres: each face's own sum first, then the faces' —
    # the same sum whatever the blocks
    dsum = torch.zeros(nf, dtype=V.dtype, device=V.device)
    n_pairs = 0
    for a, b in blocks:
        src, dst = _block_pairs(F, inc, start, deg, a, b)
        dsum[a:b].index_add_(0, src - a, (cen[src] - cen[dst]).norm(dim=1))
        n_pairs += len(src)
    sigma_s = float(dsum.sum()) / n_pairs if n_pairs else 1.0
    del dsum
    n0 = n.clone()
    for _ in range(normal_iters):
        acc = area[:, None] * n
        for a, b in blocks:
            src, dst = _block_pairs(F, inc, start, deg, a, b)
            dc = (cen[src] - cen[dst]).norm(dim=1)
            ws = area[dst] * torch.exp(-0.5 * (dc / max(sigma_s, 1e-12)) ** 2)
            wr = torch.exp(-0.5 * ((n[src] - n[dst]).norm(dim=1) / sigma_r) ** 2)
            acc[a:b].index_add_(0, src - a, (ws * wr)[:, None] * n[dst])
            del src, dst, dc, ws, wr
        n = acc / acc.norm(dim=1, keepdim=True).clamp_min(1e-20)
        del acc
    del inc, start, cen, area
    turned = torch.rad2deg(torch.acos((n * n0).sum(1).clamp(-1, 1)))
    del n0
    turned_median, turned_p99 = median_p99(turned)
    del turned
    cnt = deg.double().clamp_min(1)[:, None]
    V = V.clone()
    for _ in range(vertex_iters):
        cen = _centres(V, F)
        move = torch.zeros_like(V)
        for j in range(3):
            vi = F[:, j]
            move.index_add_(0, vi, n * (n * (cen - V[vi])).sum(1, keepdim=True))
        V = V + move / cnt
    return V, {"faces": nf, "neighbour_pairs": n_pairs, "blocks": len(blocks), "sigma_s_m": sigma_s, "sigma_r": sigma_r,
               "normal_iters": normal_iters, "vertex_iters": vertex_iters,
               "normals_turned_deg_median": turned_median, "normals_turned_deg_p99": turned_p99}


def filter_mesh(vertices: torch.Tensor, faces: torch.Tensor, normal_iters: int, vertex_iters: int, sigma_r: float,
                block_faces: int | None):
    """A mesh's own tensors in (any float and integer type, on their device), the filtered vertices out in the
    vertices' own type, the report and the TWO lines that say what the filter did — the mesh and the filter's values,
    then what it changed, its time and the card (each short: the stage log cuts a line at 300 characters). float64
    under torch's deterministic algorithms (set here, put back after). A vertex that is not finite RAISES. The card's
    numbers are the process's: what it held when the filter began and its peak so far (the caller's own statistics
    are left as they are)."""
    t = time.perf_counter()
    on_card = vertices.device.type == "cuda"
    held = torch.cuda.memory_allocated(vertices.device) if on_card else 0
    was = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        V = vertices.double()
        V2, rep = bilateral_denoise(V, faces, normal_iters, vertex_iters, sigma_r, block_faces)
        if not bool(torch.isfinite(V2).all()):
            raise RuntimeError("the edge filter returned a vertex that is not finite")
        moved = (V2 - V).norm(dim=1)
        del V
        med, p99 = median_p99(moved)
        rep["vertex_moved_mm_median"], rep["vertex_moved_mm_p99"] = med * 1000, p99 * 1000
        rep["vertex_moved_mm_max"] = float(moved.max()) * 1000 if len(moved) else 0.0
        del moved
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)
    out = V2.to(vertices.dtype).contiguous()
    del V2
    rep["seconds"] = time.perf_counter() - t
    lines = [f"{len(vertices):,} vertices, {rep['faces']:,} faces, {rep['neighbour_pairs']:,} neighbour pairs in "
             f"{rep['blocks']} block(s); bilateral normal filtering {normal_iters} x sigma_r {sigma_r} (centres sigma "
             f"{rep['sigma_s_m'] * 1000:.2f} mm), {vertex_iters} vertex updates",
             f"normals turned median {rep['normals_turned_deg_median']:.2f} deg (p99 "
             f"{rep['normals_turned_deg_p99']:.2f}), vertices moved median {rep['vertex_moved_mm_median']:.3f} mm (p99 "
             f"{rep['vertex_moved_mm_p99']:.3f}, max {rep['vertex_moved_mm_max']:.2f} mm); {rep['seconds']:.1f} s"]
    if on_card:
        rep["vram_held_at_entry_gb"] = held / 2 ** 30
        rep["vram_peak_gb"] = torch.cuda.max_memory_allocated(vertices.device) / 2 ** 30
        lines[1] += (f"; the card held {rep['vram_held_at_entry_gb']:.1f} GB when the filter began, VRAM peak so far "
                     f"{rep['vram_peak_gb']:.1f} GB")
        torch.cuda.empty_cache()
    return out, rep, lines


def main() -> None:
    inp, out = sys.argv[1], sys.argv[2]
    ni, vi, sr, block = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    dev = torch.device("cuda")
    d = torch.load(inp, map_location="cpu", weights_only=False)
    V2, rep, lines = filter_mesh(d["vertices"].to(dev), d["faces"].to(dev), ni, vi, sr, block)
    d = dict(d)
    d["vertices"] = V2.float().cpu()
    d["stac_denoise"] = rep
    torch.save(d, out)
    for line in lines:
        print(f"[denoise] {line}", flush=True)


if __name__ == "__main__":
    main()
