"""GenRecon's shape FROM THE CLOUD (USER 2026-11-12: "la geometría debe ser correcta, exactamente igual a la nube lisa
y correcta con detalle" / "el piso es una lámina así debe ser igual que las paredes" / "todo debe tener un único malla" /
"debe ser mejor incluso que la nube" / "la única que acepto es genrecon pero correcta").

Measured 2026-11-12 (CLAUDE.md §7): GenRecon generates its surfaces from the photos and takes from the cloud only WHERE
there is something — so it leaves out surfaces the cloud has, inflates or closes openings and lays half its surface 3–11
cm off the cloud. Here the cloud's SHAPE itself enters GenRecon: written in GenRecon's own cell representation (o_voxel's
flexible dual grid, 512 cells per chunk) and encoded by GenRecon's own shape encoder (TRELLIS.2
shape_enc_next_dc_f16c32); the decoder, the texture flow and the bake then run as always. GenRecon's encoder -> decoder
keeps what it is given: an exact box comes back with its faces within 0.007 cell and its edges within 0.16 cell
(measured 2026-11-12) — so the sheet, the edges and the flat faces are decided HERE.

Per chunk (its unit cube, the chunk transforms of the run itself):
  1. every point carries two planes from the cells around it (moments on the card): the plane of its 3x3x3 cells (the
     noise averaged — the FIELD) and the plane of ONE FACE (the most planar 2x2x2 block of each cell, and per point the
     one of its 27 cells' planes passing nearest to it — a crease's top keeps the top's plane, its side the side's);
  2. the field normal is turned toward the camera that saw the point (its birth keyframe; the adapter's
     cam_genrecon.npy / cam_of_point.npy): the side of the surface where the air is;
  3. every cell corner within BAND cells of a point's field plane (a BOX along its normal) and near it across (a
     Gaussian of SIGMA cells) gathers those planes. The corner is SUPPORTED where its strongest face (field normals)
     reaches half the largest such weight within BAND cells, or where one of its faces (face normals: one on a face, two
     at a crease) reaches half the largest weight OF THAT FACE around — the surface ends at the cloud's last points (2.6
     cells past them without it), and a crease's sparser face keeps its corners beside the denser one (it lost them,
     and the crease came out cut by half a cell); and at least one point's weight at 1.2 sigma;
  4. the corner's ORIENTATION is its dominant field normal turned to the side its cameras saw (their majority); its
     VALUE the mean signed distance to the planes around it, each read on that side and weighted by how squarely its
     camera saw it: the same planes for every corner of a column, so along any line the value changes sign once per
     surface — ONE SHEET (the edges' own planes, used before, crossed 15 % of bufferStop's floor columns twice);
  5. o_voxel's convention (flexible_dual_grid_to_mesh): voxel (i,j,k) flags its x-edge (i,j+1,k+1)->(i+1,j+1,k+1), its
     y-edge (i+1,j,k+1)->(i+1,j+1,k+1), its z-edge (i+1,j+1,k)->(i+1,j+1,k+1). An edge is crossed where its corners'
     values, both read on the FIRST corner's side, change sign (a corner read on the opposite side changes sign exactly):
     a thin plate seen from both faces — its corners' majorities flip — is crossed once at its middle, whole (17 % of a
     one-cell plate was lost when each corner kept its own side), and no wall stands where the side flips. A voxel's dual
     vertex by DUAL CONTOURING on the plane of ONE FACE per crossing (the faces at the edge carrying at least half the
     weight of the strongest, the one most transverse to the edge, its normal re-weighted three times by
     (n . n_face)^2, through the mean of its points): on the crease where two faces meet (0.09 cell from it on clean
     data, 0.17 with noise of 0.13 cell), on the face elsewhere; regularised towards the crossings' mass point with
     o_voxel's QEF weight (0.1 per crossing);
  6. GenRecon's shape encoder: (vertices - 0.5, intersected - 0.5) -> the chunk's latent.
"""
from __future__ import annotations

import math
from typing import Callable, List, Sequence

import torch

SHAPE_ENCODER = ("microsoft/TRELLIS.2-4B", "ckpts/shape_enc_next_dc_f16c32_fp16")
_I4 = torch.tensor([(i, j, k, l) for i in range(3) for j in range(i, 3) for k in range(j, 3) for l in range(k, 3)])
_J4 = torch.tensor([[[[[tuple(x) for x in _I4.tolist()].index(tuple(sorted((i, j, k, l)))) for l in range(3)]
                      for k in range(3)] for j in range(3)] for i in range(3)])   # (i,j,k,l) -> its component in _I4


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


def _t6(n: torch.Tensor) -> torch.Tensor:
    return torch.stack([n[:, 0] * n[:, 0], n[:, 1] * n[:, 1], n[:, 2] * n[:, 2],
                        n[:, 0] * n[:, 1], n[:, 0] * n[:, 2], n[:, 1] * n[:, 2]], 1)


def _quad(t, n):
    """n^T T n for T stored as (xx, yy, zz, xy, xz, yz)."""
    return (t[..., 0] * n[..., 0] ** 2 + t[..., 1] * n[..., 1] ** 2 + t[..., 2] * n[..., 2] ** 2
            + 2 * (t[..., 3] * n[..., 0] * n[..., 1] + t[..., 4] * n[..., 0] * n[..., 2] + t[..., 5] * n[..., 1] * n[..., 2]))


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
    """Every point's two planes from the cells of side ``cell`` around it (moments on P's device): (field normal, field
    mean) — the plane of its 3x3x3 cells — and (face normal, face mean) — the most planar of the eight 2x2x2 blocks of
    cells holding each cell (the smallest ratio of the two lowest eigenvalues; fewer than 3 points or no plane falls
    back to the 3x3x3), per point the one of its 27 cells' planes passing nearest to it. Normals unoriented."""
    dev = P.device
    k = torch.floor(P / cell).long()
    k = k - (k.min(0).values - 2)
    D = int(k.max()) + 4

    def enc(c):
        return (c[..., 0] * D + c[..., 1]) * D + c[..., 2]

    def dec(u):
        return torch.stack([u // (D * D), (u // D) % D, u % D], -1)

    def find(table, keys):
        pos = torch.searchsorted(table, keys).clamp(max=len(table) - 1)
        return table[pos] == keys, pos

    uk, cell_of = torch.unique(enc(k), return_inverse=True)
    S1 = torch.zeros((len(uk), 3), device=dev, dtype=torch.float64).index_add_(0, cell_of, P)
    S2 = torch.zeros((len(uk), 3, 3), device=dev, dtype=torch.float64).index_add_(0, cell_of, P[:, :, None] * P[:, None, :])
    N = torch.bincount(cell_of, minlength=len(uk)).double()
    cs = dec(uk)

    def block_sums(anchor_cells, offsets):
        T1 = torch.zeros((len(anchor_cells), 3), device=dev, dtype=torch.float64)
        T2 = torch.zeros((len(anchor_cells), 3, 3), device=dev, dtype=torch.float64)
        TN = torch.zeros(len(anchor_cells), device=dev, dtype=torch.float64)
        for o in offsets:
            hit, pos = find(uk, enc(anchor_cells + o))
            T1[hit] += S1[pos[hit]]
            T2[hit] += S2[pos[hit]]
            TN[hit] += N[pos[hit]]
        mean = T1 / TN.clamp_min(1)[:, None]
        return mean, T2 / TN.clamp_min(1)[:, None, None] - mean[:, :, None] * mean[:, None, :], TN

    def grid(lo, hi):
        r = torch.arange(lo, hi + 1, device=dev)
        return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)

    o2, o3 = grid(0, 1), grid(-1, 1)
    m27, C27, _ = block_sums(cs, o3)
    n27 = _eigh(C27)[1][:, :, 0]
    anchors = torch.unique(enc(cs[:, None, :] - o2[None]).reshape(-1))  # every 2x2x2 block holding an occupied cell
    bm, bC, bN = block_sums(dec(anchors), o2)
    ev, evec = _eigh(bC)
    score = ev[:, 0].clamp_min(0) / ev[:, 1]
    score[(bN < 3) | ~(ev[:, 1] > 0) | ~torch.isfinite(score)] = float("inf")
    _, apos = find(anchors, enc(cs[:, None, :] - o2[None]))             # (cells, 8): all present by construction
    best = apos.gather(1, score[apos].argmin(1, keepdim=True)).squeeze(1)
    ok = torch.isfinite(score[best])
    cn = torch.where(ok[:, None], evec[best][:, :, 0], n27)
    cm = torch.where(ok[:, None], bm[best], m27)
    best_d = torch.full((len(P),), float("inf"), device=dev, dtype=torch.float64)
    sn = torch.zeros_like(P)
    sm = torch.zeros_like(P)
    for o in o3:
        hit, pos = find(uk, enc(k + o))
        d = ((P - cm[pos]) * cn[pos]).sum(1).abs()
        d[~hit] = float("inf")
        better = d < best_d
        best_d = torch.where(better, d, best_d)
        sn[better] = cn[pos[better]]
        sm[better] = cm[pos[better]]
    return n27[cell_of], m27[cell_of], sn, sm


def orient(N: torch.Tensor, P: torch.Tensor, cam: torch.Tensor):
    """The normals turned toward the camera that saw each point (``cam``: its centre, per point), and how squarely
    it saw it (|n . view|, 0 grazing .. 1 square)."""
    v = cam - P
    c = (N * v).sum(1) / v.norm(dim=1).clamp_min(1e-12)
    s = torch.where(c < 0, -1.0, 1.0).to(N.dtype)
    return N * s[:, None], c.abs()


def chunk_cells(Pc: torch.Tensor, Nf: torch.Tensor, Mf: torch.Tensor, Cf: torch.Tensor, Ns: torch.Tensor,
                Ms: torch.Tensor, R: int = 512, sigma: float = 1.0, band: int = 2):
    """The flexible dual grid of the points ``Pc`` (the chunk's unit-cube coordinates): ``Nf`` / ``Mf`` their field
    planes (normals ORIENTED toward their cameras), ``Cf`` how squarely they were seen, ``Ns`` / ``Ms`` their face
    planes: (coords (n, 3) int32, dual vertices (n, 3) in [0, 1], intersected (n, 3) bool)."""
    dev = Pc.device
    g = (Pc + 0.5) * R
    mf_all = (Mf + 0.5) * R
    ms_all = (Ms + 0.5) * R
    base = torch.floor(g).long()
    r = torch.arange(-band, band + 2, device=dev)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    DD = R + 2 * band + 6
    B = 1 << 14

    def batch(s):
        corners = (base[s:s + B][:, None, :] + offs[None]).reshape(-1, 3)
        rep = lambda t: t[s:s + B].repeat_interleave(len(offs), 0)
        p, mf, n = rep(g), rep(mf_all), rep(Nf)
        d = corners.double() - p
        dn = ((corners.double() - mf) * n).sum(1)                 # from the point's plane, along its normal
        dt = d - (d * n).sum(1, keepdim=True) * n                  # from the point, across
        w = torch.exp(-0.5 * (dt.norm(dim=1) / sigma) ** 2) * (dn.abs() <= band)
        ok = (corners >= 0).all(1) & (corners <= R).all(1) & (w > 1e-3)
        key = (corners[ok, 0] * DD + corners[ok, 1]) * DD + corners[ok, 2]
        return key, w[ok], ok, rep

    uk = torch.unique(torch.cat([batch(s)[0] for s in range(0, len(Pc), B)]))
    W = torch.zeros(len(uk), device=dev, dtype=torch.float64)
    TT = torch.zeros((len(uk), 6), device=dev, dtype=torch.float64)
    NO = torch.zeros((len(uk), 3), device=dev, dtype=torch.float64)     # sum w c n (toward each point's camera)
    WC = torch.zeros(len(uk), device=dev, dtype=torch.float64)          # sum w c
    TS = torch.zeros((len(uk), 6), device=dev, dtype=torch.float64)     # the FACE normals' tensor
    for s in range(0, len(Pc), B):
        key, w, ok, rep = batch(s)
        n, cf = rep(Nf)[ok], rep(Cf)[ok]
        pos = torch.searchsorted(uk, key)
        W.index_add_(0, pos, w)
        TT.index_add_(0, pos, w[:, None] * _t6(n))
        TS.index_add_(0, pos, w[:, None] * _t6(rep(Ns)[ok]))
        wc = w * cf
        NO.index_add_(0, pos, wc[:, None] * n)
        WC.index_add_(0, pos, wc)
    # each FACE at the corner (the tensor's directions carrying at least half the weight of the strongest: one on a
    # face, two at a crease) against the largest weight OF THAT SAME FACE within BAND cells (the neighbours' tensors
    # read along it): the corner is supported where one of its faces reaches half its own maximum — a denser face
    # meeting it at a crease does not raise its bar (the strongest face of any direction did: the sparser face lost
    # its corners beside the crease and the crease came out cut)
    ev, evec = _eigh(_full(TS))                       # the faces by their own planes' normals (pure at a crease)
    present = ev >= 0.5 * ev[:, 2:3]
    F = W * _lambda_max(TT / W[:, None])                # ... and the strongest face of the smoothed normals
    cu = torch.stack([uk // (DD * DD), (uk // DD) % DD, uk % DD], 1)
    fmax = ev.clone()
    gmax = F.clone()
    rb = torch.arange(-band, band + 1, device=dev)
    for o in torch.stack(torch.meshgrid(rb, rb, rb, indexing="ij"), -1).reshape(-1, 3):
        c = cu + o
        kk = (c[:, 0] * DD + c[:, 1]) * DD + c[:, 2]
        pos = torch.searchsorted(uk, kk).clamp(max=len(uk) - 1)
        hit = uk[pos] == kk
        gmax[hit] = torch.maximum(gmax[hit], F[pos[hit]])
        for j in range(3):
            fmax[hit, j] = torch.maximum(fmax[hit, j], _quad(TS[pos[hit]], evec[hit][:, :, j]))
    # supported where the strongest face (smoothed normals) reaches half its maximum around — the rule of the rough
    # and sparse surfaces, gravel, vegetation — OR where one of the faces (pure normals) reaches half ITS maximum:
    # a crease's sparser face keeps its corners beside the denser one
    sup = (W > math.exp(-0.5 * 1.2 ** 2)) & ((F >= 0.5 * gmax) | (present & (ev >= 0.5 * fmax)).any(1))
    del ev, evec, present, F, gmax
    uk, Wn, cc = uk[sup], W[sup], cu[sup]
    TS = TS[sup] / Wn[:, None]
    # the corner's ORIENTATION: its dominant face normal (the field normals' tensor) turned to the side its cameras saw
    # (the majority of the oriented normals) — where a face was seen from both sides (a sheet of metal) the majority
    # may flip from corner to corner; each edge then reads its two corners on the same side (below)
    u = _eigh(_full(TT[sup] / Wn[:, None]))[1][:, :, 2]
    sig = torch.where((u * NO[sup]).sum(1) < 0, -1.0, 1.0).to(u.dtype)
    del W, TT, NO, WC, fmax, cu
    on = u * sig[:, None]                              # the corner's orientation
    # the corner's value: the mean signed distance to the planes around it, each read on the corner's side
    NS = torch.zeros((len(uk), 3), device=dev, dtype=torch.float64)
    DS = torch.zeros(len(uk), device=dev, dtype=torch.float64)
    WS = torch.zeros(len(uk), device=dev, dtype=torch.float64)
    for s in range(0, len(Pc), B):
        key, w, ok, rep = batch(s)
        pos = torch.searchsorted(uk, key).clamp(max=len(uk) - 1)
        hit = uk[pos] == key
        n, mf, cf = rep(Nf)[ok][hit], rep(mf_all)[ok][hit], rep(Cf)[ok][hit]
        pos, wc = pos[hit], w[hit] * cf
        sg = torch.where((n * on[pos]).sum(1) < 0, -1.0, 1.0).to(n.dtype)
        ns = n * sg[:, None]
        NS.index_add_(0, pos, wc[:, None] * ns)
        DS.index_add_(0, pos, wc * (ns * mf).sum(1))
        WS.index_add_(0, pos, wc)
    fval = ((NS * cc.double()).sum(1) - DS) / WS.clamp_min(1e-300)       # the mean signed distance, in cells
    del NS, DS, WS
    QS = torch.zeros((len(uk), 3, 6), device=dev, dtype=torch.float64)  # the face planes' means and 4th moments
    M4 = torch.zeros((len(uk), 15), device=dev, dtype=torch.float64)
    I4 = _I4.to(dev)
    for s in range(0, len(Pc), B):
        key, w, ok, rep = batch(s)
        pos = torch.searchsorted(uk, key).clamp(max=len(uk) - 1)
        hit = uk[pos] == key
        n, ms = rep(Ns)[ok][hit], rep(ms_all)[ok][hit]
        pos, w = pos[hit], w[hit]
        t = _t6(n)
        QS.index_add_(0, pos, (w[:, None] * ms)[:, :, None] * t[:, None, :])
        M4.index_add_(0, pos, w[:, None] * (n[:, I4[:, 0]] * n[:, I4[:, 1]] * n[:, I4[:, 2]] * n[:, I4[:, 3]]))
    QS /= Wn[:, None, None]
    M4 /= Wn[:, None]
    J4 = _J4.to(dev)

    def lookup(c):
        kk = (c[:, 0] * DD + c[:, 1]) * DD + c[:, 2]
        pos = torch.searchsorted(uk, kk).clamp(max=len(uk) - 1)
        return uk[pos] == kk, pos

    owners, Vs, Ps, Ns_, Ds = [], [], [], [], []
    for a in range(3):
        e = torch.zeros(3, dtype=torch.long, device=dev)
        e[a] = 1
        h2, pos2 = lookup(cc + e)
        i0 = torch.nonzero(h2).squeeze(1)
        i1 = pos2[i0]
        f0, f1 = fval[i0], fval[i1]
        # both corners read on the SAME side: where their orientations are opposite (the two faces of a thin plate, a
        # flip between a plate's two halves) the second is read on the first's side — its value changes sign exactly
        # (every plane around it turned): the plate's middle is crossed once, and no wall stands where the side flips
        f1 = torch.where((on[i0] * on[i1]).sum(1) < 0, -f1, f1)
        cr = (f0 >= 0) != (f1 >= 0)                     # the values change sign between the two corners
        i0, i1, f0, f1 = i0[cr], i1[cr], f0[cr], f1[cr]
        c0 = cc[i0]
        t = (f0 / (f0 - f1)).clamp(0, 1)
        pt = c0.double() + t[:, None] * e.double()
        T2 = TS[i0] + TS[i1]
        ev, evec = _eigh(_full(T2))
        present = ev >= 0.5 * ev[:, 2:3]                # the faces at the edge: at least half the strongest's weight
        k = torch.where(present, evec[:, a, :].abs(), torch.full_like(ev, -1.0)).argmax(1)
        nrm = evec.gather(2, k[:, None, None].expand(-1, 3, 1)).squeeze(2)
        M4e = M4[i0] + M4[i1]
        for _ in range(3):
            v2 = _eigh(torch.einsum("eijkl,ek,el->eij", M4e[:, J4], nrm, nrm))[1][:, :, 2]
            nrm = v2 * torch.where((v2 * nrm).sum(1, keepdim=True) < 0, -1.0, 1.0)
        mm = _quad(QS[i0] + QS[i1], nrm[:, None, :]) / _quad(T2, nrm).clamp_min(1e-300)[:, None]
        owners.append((a, c0 - (1 - e)))
        o1, o2 = [i for i in range(3) if i != a]
        for d1 in (0, 1):
            for d2 in (0, 1):
                v = c0.clone()
                v[:, o1] -= d1
                v[:, o2] -= d2
                Vs.append(v)
                Ps.append(pt)
                Ns_.append(nrm)
                Ds.append((nrm * mm).sum(1))
    V, Pt, Nn, Dn = torch.cat(Vs), torch.cat(Ps), torch.cat(Ns_), torch.cat(Ds)
    ok = (V >= 0).all(1) & (V < R).all(1)
    V, Pt, Nn, Dn = V[ok], Pt[ok], Nn[ok], Dn[ok]
    uv, vinv = torch.unique((V[:, 0] * R + V[:, 1]) * R + V[:, 2], return_inverse=True)
    nv = len(uv)
    coords = torch.stack([uv // (R * R), (uv // R) % R, uv % R], 1)
    rel = Pt - coords[vinv].double()
    off = Dn - (Nn * coords[vinv].double()).sum(1)      # the face's plane, from the voxel's corner
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


# the context each chunk is encoded with beyond its cube, in fine cells (a multiple of the latent's 16): GenRecon's
# pipeline AVERAGES the latents of overlapping chunks (joint_decode._merge_chunk_slats — its own flow makes them
# near-equal); encoded with no context the latents of a chunk's border differ from its neighbour's (p99 50 % on
# bufferStop) and the average decodes as doubled sheets (the floor 95.0 -> 87.3 % one lamina); with 64 cells they agree
# to the encoder's fp16 noise (p99 0.1 %, the same with 384), measured 2026-11-12
MARGIN = 64


@torch.no_grad()
def encode_chunks(P: torch.Tensor, cam: torch.Tensor, m_o2c: Sequence[torch.Tensor], encoder, R: int = 512,
                  sigma: float = 1.0, band: int = 2, log: Callable[[str], None] = print) -> List:
    """The cloud ``P`` (the original frame, on the encoder's device; ``cam``: the centre of the camera that saw each
    point, same frame) encoded per chunk (``m_o2c``: the chunks' original-to-unit-cube transforms, the run's own order):
    a list of SparseTensors (the shape latents). Each chunk's cells are built and encoded with MARGIN cells of the cloud
    around it, and its latent cropped back to the chunk — the same latent where chunks overlap."""
    from genrecon.modules import sparse as sp
    dev = P.device
    M0 = m_o2c[0].to(dev, torch.float64)
    cell = float(1.0 / R / M0[:3, 0].norm())            # one cell in the original frame's units
    Nf, Mf, Ns, Ms = plane_frames(P, cell)
    Nf, Cf = orient(Nf, P, cam)
    Rp, lo = R + 2 * MARGIN, MARGIN // 16
    out = []
    for i, M in enumerate(m_o2c):
        M = M.to(dev, torch.float64)
        Pc = P @ M[:3, :3].T + M[:3, 3]
        sel = (Pc.abs() <= 0.5 + (MARGIN + band + 1) / R).all(1)
        if int(((Pc[sel].abs() <= 0.5).all(1)).sum()) == 0:
            raise RuntimeError(f"chunk {i} holds no cloud point — the cloud's chunks always hold one")
        Rm = M[:3, :3] / M[:3, 0].norm()                # the chunk's rotation (its scale apart)

        def pad(X):                                     # the chunk's unit cube -> the padded lattice's
            return ((X + 0.5) * R + MARGIN) / Rp - 0.5
        coords, dual, inter = chunk_cells(pad(Pc[sel]), Nf[sel] @ Rm.T, pad(Mf[sel] @ M[:3, :3].T + M[:3, 3]),
                                          Cf[sel], Ns[sel] @ Rm.T, pad(Ms[sel] @ M[:3, :3].T + M[:3, 3]), Rp, sigma,
                                          band)
        bc = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device=dev), coords], 1)
        z = encoder(sp.SparseTensor(dual, bc), sp.SparseTensor(inter.float(), bc))
        c = z.coords[:, 1:] - lo
        keep = ((c >= 0) & (c < R // 16)).all(1)
        if int(keep.sum()) == 0:
            raise RuntimeError(f"chunk {i}: no latent voxel inside the chunk")
        zc = torch.cat([z.coords[keep, :1], c[keep]], 1).to(torch.int32)
        z = sp.SparseTensor(feats=z.feats[keep], coords=zc)
        out.append(z)
        log(f"[cloud_shape] chunk {i}: {int(sel.sum()):,} points (with {MARGIN} cells around) -> {len(coords):,} cells, "
            f"{int(inter.sum()):,} crossed edges -> latent {z.feats.shape[0]:,} x {z.feats.shape[1]}")
    return out
