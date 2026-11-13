"""The loose pieces out of GenRecon's decoded mesh, on the card, before the bake (USER 2026-11-12: "quitando los
pedazos sueltos, esos no deben estar, tenes mi ok" / "sin voladores").

A piece is a set of faces joined by their (welded) edges. KEPT: the largest piece, every piece reaching the floor (its
lowest point within FLOOR of levels.json's z_floor) and — repeated TO CONVERGENCE — every piece whose GAP to the kept
surface is no more than ITS OWN SIZE (the diagonal of its bounding box): a part of the same structure split by a
crack, a crease or a stretch nobody saw. Every other piece floats loose — farther from everything than it is large —
and goes. No length decides: the piece's own size is the bar.

Why not a fixed distance (2026-11-12, bufferStop, measured): the rule "within 2 cm of a kept piece" dropped the back
wall and its vegetation whenever their junction with the rest was open by more than 2 cm — 13.7 m² at refinement
t 0.5, 12.8 m² at t 0.6, 16.1 m² at t 0, of which 13.6 / 16.1 m² lie within 3 cm of the cloud (pieces of 7.5 m² at
58 cm and 4.7 m² at 7 cm). Under this rule, on the card, the meshes at t 0.5 / 0.6 / 0.7 lose 0.10 / 0.14 / 0.17 m² of
crumbs (324 / 188 / 138 pieces of 41.1 / 38.5 / 36.8 m²; the largest dropped 0.019 / 0.048 / 0.077 m²). (Its first
version, stopped after ten rounds, had dropped 28 m².)

The gap is measured between the pieces' points (their vertices and a sampling of their faces at the mesh's own
density), nearest neighbour on the card, on a ladder of radii doubling from the mesh's face size: a piece no larger
than the radius is settled there; beyond RES face sizes both sides are gathered in cells of radius / RES, so the gap
of a piece is known to a few percent of its size whatever the mesh holds. GenRecon's frame (z up).
Usage: drop_loose.py <to_glb_inputs.pt> <work dir (levels.json)> <out .pt> [floor m]"""
from __future__ import annotations

import json
import sys

import torch

RES = 64          # the ladder's resolution: cells of radius / RES (numerical, not a gate: the bar is the piece's size)


def _o3(t):
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    return o3c.Tensor.from_dlpack(tdl.to_dlpack(t.float().contiguous()))


def _nearest_within(ref: torch.Tensor, q: torch.Tensor, r: float) -> torch.Tensor:
    """For every q: the distance to its nearest ref point within r, inf beyond (open3d's CUDA hybrid search)."""
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    out = torch.full((len(q),), float("inf"), dtype=torch.float64, device=q.device)
    if not len(ref) or not len(q):
        return out
    nns = o3c.nns.NearestNeighborSearch(_o3(ref))
    nns.hybrid_index(r)
    for s in range(0, len(q), 1 << 21):
        _, d2, cnt = nns.hybrid_search(_o3(q[s:s + (1 << 21)]), r, 1)
        d = tdl.from_dlpack(d2.to_dlpack()).reshape(-1).double().clamp_min(0).sqrt()   # open3d: squared distances
        ok = tdl.from_dlpack(cnt.to_dlpack()).reshape(-1) > 0
        out[s:s + (1 << 21)][ok] = d[ok]
    return out


def _cells(P: torch.Tensor, key: torch.Tensor, h: float):
    """One point per (key, cell of side h): the mean of the points of P in it -> (points, their keys)."""
    rows = torch.cat([key[:, None], torch.floor(P / h).long()], 1)
    u, inv = torch.unique(rows, dim=0, return_inverse=True)
    C = torch.zeros((len(u), 3), dtype=P.dtype, device=P.device).index_add_(0, inv, P)
    n = torch.zeros(len(u), dtype=P.dtype, device=P.device).index_add_(0, inv, torch.ones(len(P), dtype=P.dtype, device=P.device))
    return C / n[:, None], u[:, 0]


def _samples(a, b, c, area, step):
    """(points over the faces at one per step², at least one a face; the face of each)."""
    g = torch.Generator(device=a.device)
    g.manual_seed(0)
    n = torch.floor(area / step ** 2 + torch.rand(area.shape, generator=g, device=a.device, dtype=a.dtype)).long() + 1
    fi = torch.repeat_interleave(torch.arange(len(a), device=a.device), n)
    u = torch.rand(len(fi), generator=g, device=a.device, dtype=a.dtype)
    v = torch.rand(len(fi), generator=g, device=a.device, dtype=a.dtype)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a[fi] + u[:, None] * (b[fi] - a[fi]) + v[:, None] * (c[fi] - a[fi]), fi


def loose_faces(V: torch.Tensor, F: torch.Tensor, z_floor: float, floor: float):
    """(keep mask over the faces, report), under torch's deterministic algorithms (set here, put back after): the
    pieces' areas and the cells' mean points are sums on the card, and left to the card's own order their last digit
    changes from one call to the next (measured on the test's 13 pieces: the largest dropped piece's area took 10
    values in 40 calls, none under this mode)."""
    was = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _loose_faces(V, F, z_floor, floor)
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)


def _loose_faces(V: torch.Tensor, F: torch.Tensor, z_floor: float, floor: float):
    dev = V.device
    _, inv = torch.unique(torch.round(V / 1e-5).long(), dim=0, return_inverse=True)
    Fw = inv[F]
    E = torch.cat([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    lab = torch.arange(int(inv.max()) + 1, device=dev)
    while True:
        m = torch.minimum(lab[E[:, 0]], lab[E[:, 1]])
        new = lab.clone().scatter_reduce_(0, E[:, 0], m, "amin").scatter_reduce_(0, E[:, 1], m, "amin")
        new = new[new]
        if torch.equal(new, lab):
            break
        lab = new
    pieces, pof = torch.unique(lab[Fw[:, 0]], return_inverse=True)
    n = len(pieces)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * torch.cross(b - a, c - a, dim=1).norm(dim=1)
    parea = torch.zeros(n, device=dev, dtype=area.dtype).index_add_(0, pof, area)
    idx = pof[:, None].expand(-1, 3)
    lo = torch.full((n, 3), float("inf"), device=dev, dtype=V.dtype).scatter_reduce_(
        0, idx, torch.minimum(torch.minimum(a, b), c), "amin")
    hi = torch.full((n, 3), float("-inf"), device=dev, dtype=V.dtype).scatter_reduce_(
        0, idx, torch.maximum(torch.maximum(a, b), c), "amax")
    size = (hi - lo).norm(dim=1)                              # a piece's own size: the diagonal of its bounding box
    keep = lo[:, 2] <= z_floor + floor
    keep[int(parea.argmax())] = True
    # the pieces' points: the vertices their faces use and a sampling of the faces at the mesh's own density
    step = float(area.median().sqrt())
    S, fs = _samples(a, b, c, area, step)
    used = torch.zeros(len(V), dtype=torch.bool, device=dev)
    used[F.reshape(-1)] = True
    pv = torch.zeros(len(V), dtype=torch.long, device=dev)
    pv[F.reshape(-1)] = pof.repeat_interleave(3)
    P = torch.cat([V[used], S])
    pp = torch.cat([pv[used], pof[fs]])
    zero = torch.zeros(1, dtype=torch.long, device=dev)
    fresh, rounds = keep.clone(), 0
    while fresh.any() and not keep.all():
        rounds += 1
        ref = P[fresh[pp]]                                    # the pieces kept in the last round: the others are done
        open_, add, r = ~keep, torch.zeros_like(keep), step
        while open_.any():
            q = open_[pp]
            Q, kq, Rf = P[q], pp[q], ref
            if r / RES > step:
                Q, kq = _cells(Q, kq, r / RES)
                Rf = _cells(Rf, zero.expand(len(Rf)), r / RES)[0]
            g = torch.full((n,), float("inf"), device=dev, dtype=torch.float64).scatter_reduce_(
                0, kq, _nearest_within(Rf, Q, r), "amin")
            hit = open_ & (g <= size)
            add |= hit
            open_ = open_ & ~hit & (size > r)                 # a piece no larger than the radius is settled here
            r *= 2
        keep |= add
        fresh = add
    kf = keep[pof]
    rep = {"pieces": n, "area_m2": float(area.sum()), "rounds": rounds,
           "kept_pieces": int(keep.sum()), "kept_m2": float(area[kf].sum()),
           "dropped_pieces": int((~keep).sum()), "dropped_m2": float(area[~kf].sum()),
           "largest_dropped_m2": 0.0, "largest_dropped_size_m": 0.0}
    if not keep.all():
        k = int(torch.where(keep, torch.full_like(parea, -1.0), parea).argmax())
        rep["largest_dropped_m2"], rep["largest_dropped_size_m"] = float(parea[k]), float(size[k])
    return kf, rep


def main() -> None:
    inp, work, out = sys.argv[1], sys.argv[2], sys.argv[3]
    floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
    dev = torch.device("cuda")
    d = torch.load(inp, map_location="cpu", weights_only=False)
    V = d["vertices"].to(dev).double()
    F = d["faces"].to(dev).long()
    zf = float(json.load(open(f"{work}/levels.json"))["z_floor"])
    keep, rep = loose_faces(V, F, zf, floor)
    F2 = F[keep]
    used = torch.zeros(len(V), dtype=torch.bool, device=dev)
    used[F2.reshape(-1)] = True
    remap = torch.cumsum(used.long(), 0) - 1
    d = dict(d)
    d["vertices"] = V[used].float().cpu()
    d["faces"] = remap[F2].to(torch.int32).cpu()
    d["stac_drop_loose"] = rep
    torch.save(d, out)
    print(f"[drop_loose] {rep['pieces']:,} pieces, {rep['area_m2']:.1f} m2: kept {rep['kept_pieces']:,} "
          f"({rep['kept_m2']:.2f} m2 — the largest, those reaching the floor within {floor * 100:.0f} cm of z {zf:+.3f} and, "
          f"in {rep['rounds']} rounds to convergence, those no farther from a kept one than their own size); dropped "
          f"{rep['dropped_pieces']:,} floating ({rep['dropped_m2']:.2f} m2; the largest {rep['largest_dropped_m2']:.3f} m2, "
          f"{rep['largest_dropped_size_m']:.2f} m across)", flush=True)


if __name__ == "__main__":
    main()
