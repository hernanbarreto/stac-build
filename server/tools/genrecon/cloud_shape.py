"""GenRecon's shape FROM THE CLOUD (USER 2026-11-12: "la geometría debe ser correcta, exactamente igual a la nube lisa
y correcta con detalle" / "el piso es una lámina así debe ser igual que las paredes" / "todo debe tener un único malla" /
"la única que acepto es genrecon pero correcta").

Measured 2026-11-12 (CLAUDE.md §7): GenRecon generates its surfaces from the photos and takes from the cloud only WHERE
there is something — so it leaves out surfaces the cloud has (bufferStop's front, pccr 3108's walls), inflates or
closes openings depending on that presence, and lays half its surface 3–11 cm off the cloud; GenRecon as released does
the same. Here the cloud's SHAPE itself enters GenRecon: written in GenRecon's own cell representation (o_voxel's
flexible dual grid, 512 cells per chunk) and encoded by GenRecon's own shape encoder (TRELLIS.2
shape_enc_next_dc_f16c32, the encoder of the decoder the pipeline uses); the decoder, the texture flow and the bake then
run as always. GenRecon's encoder -> decoder returns its own mesh within 1 cm everywhere (half within 3 mm, bufferStop).

Per chunk (its unit cube, the chunk transforms of the run itself):
  1. every point carries the plane of its 3x3x3 neighbourhood of cells (moments on the card): the normal and the mean;
  2. every cell corner within BAND cells of a point's plane (a BOX along its normal) and near it across (a Gaussian of
     SIGMA cells) gathers those planes: their weight, the structure tensor of their normals and their means weighted by
     that tensor — the noise averaged, two close layers of a duplicated surface merged, and the same planes for every
     corner of a column, so the mean does not follow the corner up and down through the cloud's noise (an isotropic
     Gaussian did: bufferStop's floor, +-2 cm of noise, came out in 2-3 layers in 37 % of its 1 cm columns; the box,
     15 %, against 94 % in the closed shell published before);
  3. a corner is supported where its weight reaches HALF the largest weight of one face within BAND cells (the half
     maximum of a step under a symmetric kernel: a surface ends at its last points, not BAND cells past them; one face's
     weight, so that the two faces adding up at a crease do not raise the bar for either) and at least one point's at
     1.2 sigma — nothing is made where the cloud has no points: the openings stay open, to their rim;
  4. ONE SHEET per surface — an edge between two supported corners is crossed where its face's plane passes between
     them: the dominant direction of their tensors, through the mean of the points of that face (each mean weighted by
     how much its normal lies along that direction, so the other face of a crease does not pull it); a plane through a
     corner itself is crossed on the corner's upper edge. The floor and the walls a single lamina, never a closed shell;
  5. o_voxel's convention (flexible_dual_grid_to_mesh): voxel (i,j,k) flags its x-edge (i,j+1,k+1)->(i+1,j+1,k+1), its
     y-edge (i+1,j,k+1)->(i+1,j+1,k+1), its z-edge (i+1,j+1,k)->(i+1,j+1,k+1); the 4 voxels around a crossed edge are
     active; a voxel's dual vertex by DUAL CONTOURING — the point minimising the squared distances to the planes of the
     crossings on its edges, regularised towards their mass point with o_voxel's own QEF weight (0.1 per crossing);
  6. GenRecon's shape encoder: (vertices - 0.5, intersected - 0.5) -> the chunk's latent.
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence

import torch

SHAPE_ENCODER = ("microsoft/TRELLIS.2-4B", "ckpts/shape_enc_next_dc_f16c32_fp16")


def _eigh(C: torch.Tensor):
    """Batched symmetric 3x3 eigen-decomposition in slices (cusolver's batched limits): (values ascending, vectors)."""
    if len(C) == 0:
        return C.new_zeros((0, 3)), C.new_zeros((0, 3, 3))
    parts = [torch.linalg.eigh(C[i:i + 4096]) for i in range(0, len(C), 4096)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _full(t6: torch.Tensor) -> torch.Tensor:
    """The symmetric 3x3 matrices of (xx, yy, zz, xy, xz, yz)."""
    return torch.stack([torch.stack([t6[:, 0], t6[:, 3], t6[:, 4]], 1), torch.stack([t6[:, 3], t6[:, 1], t6[:, 5]], 1),
                        torch.stack([t6[:, 4], t6[:, 5], t6[:, 2]], 1)], 1)


def _lambda_max(t6: torch.Tensor) -> torch.Tensor:
    """The largest eigenvalue of symmetric 3x3 matrices (xx, yy, zz, xy, xz, yz), closed form (Smith 1961)."""
    q = (t6[:, 0] + t6[:, 1] + t6[:, 2]) / 3
    p1 = t6[:, 3] ** 2 + t6[:, 4] ** 2 + t6[:, 5] ** 2
    p = torch.sqrt(((t6[:, 0] - q) ** 2 + (t6[:, 1] - q) ** 2 + (t6[:, 2] - q) ** 2 + 2 * p1) / 6)
    ps = p.clamp_min(1e-300)
    b0, b1, b2 = (t6[:, 0] - q) / ps, (t6[:, 1] - q) / ps, (t6[:, 2] - q) / ps
    b3, b4, b5 = t6[:, 3] / ps, t6[:, 4] / ps, t6[:, 5] / ps
    det = b0 * (b1 * b2 - b5 * b5) - b3 * (b3 * b2 - b5 * b4) + b4 * (b3 * b5 - b1 * b4)
    phi = torch.acos((det / 2).clamp(-1, 1)) / 3
    return torch.where(p > 0, q + 2 * p * torch.cos(phi), q)


def plane_frames(P: torch.Tensor, cell: float):
    """Every point's local plane from the points of its 3x3x3 neighbourhood of cells of side ``cell``: (unit normal,
    mean) — the covariance from per-cell moments, on P's device. (Planes picked per point among smaller blocks gave
    sharper creases on clean synthetic data but more layers on bufferStop's noisy floor: 58 % single sheet against 85 %.)"""
    dev = P.device
    k = torch.floor(P / cell).long()
    k = k - (k.min(0).values - 1)
    D = int(k.max()) + 3
    key = (k[:, 0] * D + k[:, 1]) * D + k[:, 2]
    uk, cell_of = torch.unique(key, return_inverse=True)
    S1 = torch.zeros((len(uk), 3), device=dev, dtype=torch.float64).index_add_(0, cell_of, P)
    S2 = torch.zeros((len(uk), 3, 3), device=dev, dtype=torch.float64).index_add_(0, cell_of, P[:, :, None] * P[:, None, :])
    N = torch.bincount(cell_of, minlength=len(uk)).double()
    cs = torch.stack([uk // (D * D), (uk // D) % D, uk % D], 1)
    T1, T2, TN = torch.zeros_like(S1), torch.zeros_like(S2), torch.zeros_like(N)
    r = torch.arange(-1, 2, device=dev)
    for o in torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3):
        nk = cs + o
        nkey = (nk[:, 0] * D + nk[:, 1]) * D + nk[:, 2]
        pos = torch.searchsorted(uk, nkey).clamp(max=len(uk) - 1)
        hit = uk[pos] == nkey
        T1[hit] += S1[pos[hit]]
        T2[hit] += S2[pos[hit]]
        TN[hit] += N[pos[hit]]
    mean = T1 / TN[:, None]
    C = T2 / TN[:, None, None] - mean[:, :, None] * mean[:, None, :]
    nrm = _eigh(C)[1][:, :, 0]
    return nrm[cell_of], mean[cell_of]


def chunk_cells(Pc: torch.Tensor, Nc: torch.Tensor, Mc: torch.Tensor, R: int = 512, sigma: float = 1.0, band: int = 2):
    """The flexible dual grid of the points ``Pc`` (the chunk's unit-cube coordinates) with their plane normals ``Nc``
    and plane means ``Mc``: (coords (n, 3) int32, dual vertices (n, 3) in [0, 1], intersected (n, 3) bool)."""
    dev = Pc.device
    g = (Pc + 0.5) * R
    m_all = (Mc + 0.5) * R
    base = torch.floor(g).long()
    r = torch.arange(-band, band + 2, device=dev)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    DD = R + 2 * band + 6
    B = 1 << 14

    def batch(s):
        corners = (base[s:s + B][:, None, :] + offs[None]).reshape(-1, 3)
        p = g[s:s + B].repeat_interleave(len(offs), 0)
        mp = m_all[s:s + B].repeat_interleave(len(offs), 0)
        n = Nc[s:s + B].repeat_interleave(len(offs), 0)
        d = corners.double() - p
        dn = ((corners.double() - mp) * n).sum(1)                 # from the point's plane, along its normal
        dt = d - (d * n).sum(1, keepdim=True) * n                  # from the point, across
        w = torch.exp(-0.5 * (dt.norm(dim=1) / sigma) ** 2) * (dn.abs() <= band)
        ok = (corners >= 0).all(1) & (corners <= R).all(1) & (w > 1e-3)
        key = (corners[ok, 0] * DD + corners[ok, 1]) * DD + corners[ok, 2]
        return key, w[ok], mp[ok], n[ok]

    def t6(n):
        return torch.stack([n[:, 0] * n[:, 0], n[:, 1] * n[:, 1], n[:, 2] * n[:, 2],
                            n[:, 0] * n[:, 1], n[:, 0] * n[:, 2], n[:, 1] * n[:, 2]], 1)

    def quad(t, n):                                     # n^T T n for T stored as xx yy zz xy xz yz
        return (t[..., 0] * n[..., 0] ** 2 + t[..., 1] * n[..., 1] ** 2 + t[..., 2] * n[..., 2] ** 2
                + 2 * (t[..., 3] * n[..., 0] * n[..., 1] + t[..., 4] * n[..., 0] * n[..., 2] + t[..., 5] * n[..., 1] * n[..., 2]))

    uk = torch.unique(torch.cat([batch(s)[0] for s in range(0, len(Pc), B)]))
    W = torch.zeros(len(uk), device=dev, dtype=torch.float64)
    TT = torch.zeros((len(uk), 6), device=dev, dtype=torch.float64)          # xx yy zz xy xz yz
    for s in range(0, len(Pc), B):
        key, w, mp, n = batch(s)
        pos = torch.searchsorted(uk, key)
        W.index_add_(0, pos, w)
        TT.index_add_(0, pos, w[:, None] * t6(n))
    F = W * _lambda_max(TT / W[:, None])                # the weight of the corner's strongest face
    cu = torch.stack([uk // (DD * DD), (uk // DD) % DD, uk % DD], 1)
    fmax = F.clone()
    rb = torch.arange(-band, band + 1, device=dev)
    for o in torch.stack(torch.meshgrid(rb, rb, rb, indexing="ij"), -1).reshape(-1, 3):
        c = cu + o
        kk = (c[:, 0] * DD + c[:, 1]) * DD + c[:, 2]
        pos = torch.searchsorted(uk, kk).clamp(max=len(uk) - 1)
        hit = uk[pos] == kk
        fmax[hit] = torch.maximum(fmax[hit], F[pos[hit]])
    sup = (W > math.exp(-0.5 * 1.2 ** 2)) & (W >= 0.5 * fmax)
    uk, Wn, TT = uk[sup], W[sup], TT[sup] / W[sup][:, None]
    cc = torch.stack([uk // (DD * DD), (uk // DD) % DD, uk % DD], 1)
    Q = torch.zeros((len(uk), 3, 6), device=dev, dtype=torch.float64)   # sum w m_k n n^T: the means per face
    for s in range(0, len(Pc), B):
        key, w, mp, n = batch(s)
        pos = torch.searchsorted(uk, key).clamp(max=len(uk) - 1)
        hit = uk[pos] == key
        Q.index_add_(0, pos[hit], (w[hit][:, None] * mp[hit])[:, :, None] * t6(n[hit])[:, None, :])
    Q /= Wn[:, None, None]

    def lookup(c):
        kk = (c[:, 0] * DD + c[:, 1]) * DD + c[:, 2]
        pos = torch.searchsorted(uk, kk).clamp(max=len(uk) - 1)
        return uk[pos] == kk, pos

    owners, Vs, Ps, Ns, Ds = [], [], [], [], []
    for a in range(3):
        e = torch.zeros(3, dtype=torch.long, device=dev)
        e[a] = 1
        h2, pos2 = lookup(cc + e)
        i0 = torch.nonzero(h2).squeeze(1)
        i1 = pos2[i0]
        T2 = TT[i0] + TT[i1]
        nrm = _eigh(_full(T2))[1][:, :, 2]
        mm = quad(Q[i0] + Q[i1], nrm[:, None, :]) / quad(T2, nrm)[:, None]
        s0 = ((cc[i0].double() - mm) * nrm).sum(1)
        s1 = ((cc[i1].double() - mm) * nrm).sum(1)
        cr = (s0 >= 0) != (s1 >= 0)                     # the face's plane passes between the two corners: one crossing
        c0 = cc[i0[cr]]
        t = (s0[cr] / (s0[cr] - s1[cr])).clamp(0, 1)
        pt = c0.double() + t[:, None] * e.double()
        owners.append((a, c0 - (1 - e)))
        o1, o2 = [i for i in range(3) if i != a]
        for d1 in (0, 1):
            for d2 in (0, 1):
                v = c0.clone()
                v[:, o1] -= d1
                v[:, o2] -= d2
                Vs.append(v)
                Ps.append(pt)
                Ns.append(nrm[cr])
                Ds.append((nrm[cr] * mm[cr]).sum(1))
    V, Pt, Nn, Dn = torch.cat(Vs), torch.cat(Ps), torch.cat(Ns), torch.cat(Ds)
    ok = (V >= 0).all(1) & (V < R).all(1)
    V, Pt, Nn, Dn = V[ok], Pt[ok], Nn[ok], Dn[ok]
    uv, vinv = torch.unique((V[:, 0] * R + V[:, 1]) * R + V[:, 2], return_inverse=True)
    nv = len(uv)
    coords = torch.stack([uv // (R * R), (uv // R) % R, uv % R], 1)
    rel = Pt - coords[vinv].double()
    off = Dn - (Nn * coords[vinv].double()).sum(1)      # the crossing's plane, from the voxel's corner
    cnt = torch.bincount(vinv, minlength=nv).double()
    mass = torch.zeros((nv, 3), device=dev, dtype=torch.float64).index_add_(0, vinv, rel) / cnt[:, None]
    A = torch.zeros((nv, 3, 3), device=dev, dtype=torch.float64).index_add_(0, vinv, Nn[:, :, None] * Nn[:, None, :])
    b = torch.zeros((nv, 3), device=dev, dtype=torch.float64).index_add_(0, vinv, Nn * off[:, None])
    lam = 0.1 * cnt                                     # o_voxel's QEF regularization weight per crossing
    A = A + lam[:, None, None] * torch.eye(3, device=dev, dtype=torch.float64)
    b = b + lam[:, None] * mass
    dual = torch.cat([torch.linalg.solve(A[i:i + 65536], b[i:i + 65536]) for i in range(0, nv, 65536)]).clamp(0, 1)
    inter = torch.zeros((nv, 3), dtype=torch.bool, device=dev)
    for a, own in owners:
        okk = (own >= 0).all(1) & (own < R).all(1)
        k2 = (own[okk, 0] * R + own[okk, 1]) * R + own[okk, 2]
        pos = torch.searchsorted(uv, k2).clamp(max=nv - 1)
        hit = uv[pos] == k2
        inter[pos[hit], a] = True
    return coords.int(), dual.float(), inter


def load_encoder(device):
    """GenRecon's shape encoder (TRELLIS.2's, the pair of the shape decoder the pipeline loads)."""
    from huggingface_hub import hf_hub_download
    from genrecon import models
    repo, name = SHAPE_ENCODER
    path = hf_hub_download(repo, f"{name}.json")
    hf_hub_download(repo, f"{name}.safetensors")
    return models.from_pretrained(path[:-len(".json")]).to(device).eval()


@torch.no_grad()
def encode_chunks(P: torch.Tensor, m_o2c: Sequence[torch.Tensor], encoder, R: int = 512, sigma: float = 1.0,
                  band: int = 2, log: Callable[[str], None] = print) -> List:
    """The cloud ``P`` (the original frame, on the encoder's device) encoded per chunk (``m_o2c``: the chunks'
    original-to-unit-cube transforms, the run's own order): a list of SparseTensors (the shape latents)."""
    from genrecon.modules import sparse as sp
    dev = P.device
    M0 = m_o2c[0].to(dev, torch.float64)
    cell = float(1.0 / R / M0[0, 0])                    # one cell in the original frame's units
    N, Mn = plane_frames(P, cell)
    out = []
    for i, M in enumerate(m_o2c):
        M = M.to(dev, torch.float64)
        Pc = P @ M[:3, :3].T + M[:3, 3]
        sel = (Pc.abs() <= 0.5 + (band + 1) / R).all(1)
        if int(sel.sum()) == 0:
            raise RuntimeError(f"chunk {i} holds no cloud point — the cloud's chunks always hold one")
        Nc = N[sel] @ M[:3, :3].T
        Nc = Nc / Nc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        coords, dual, inter = chunk_cells(Pc[sel], Nc, Mn[sel] @ M[:3, :3].T + M[:3, 3], R, sigma, band)
        bc = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device=dev), coords], 1)
        z = encoder(sp.SparseTensor(dual, bc), sp.SparseTensor(inter.float(), bc))
        out.append(z)
        log(f"[cloud_shape] chunk {i}: {int(sel.sum()):,} points -> {len(coords):,} cells, {int(inter.sum()):,} crossed "
            f"edges -> latent {z.feats.shape[0]:,} x {z.feats.shape[1]}")
    return out
