"""The loose pieces out of GenRecon's decoded mesh, on the card, before the bake (USER 2026-11-12: "quitando los
pedazos sueltos, esos no deben estar, tenes mi ok" / "sin voladores").

A piece is a set of faces joined by their (welded) edges. KEPT: the largest piece, every piece reaching the floor (its
lowest point within FLOOR of levels.json's z_floor) and — repeated TO CONVERGENCE — every piece with a face within
ATTACH of a kept piece (a part of the same structure split by a crack or a crease: open sheets split into many pieces;
on bufferStop 2026-11-12 the first version of this rule, stopped after ten rounds, dropped 28 m² of real surface).
Every other piece floats loose — the far background the cloud saw in a few frames — and goes. GenRecon's frame (z up).
Usage: drop_loose.py <to_glb_inputs.pt> <work dir (levels.json)> <out .pt> [attach m] [floor m]"""
from __future__ import annotations

import json
import sys

import torch


def _o3(t):
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    return o3c.Tensor.from_dlpack(tdl.to_dlpack(t.float().contiguous()))


def _within(ref: torch.Tensor, q: torch.Tensor, r: float) -> torch.Tensor:
    """For every q: a ref point within r (open3d's CUDA hybrid search)."""
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    nns = o3c.nns.NearestNeighborSearch(_o3(ref))
    nns.hybrid_index(r)
    out = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    for s in range(0, len(q), 1 << 21):
        _, _, cnt = nns.hybrid_search(_o3(q[s:s + (1 << 21)]), r, 1)
        out[s:s + (1 << 21)] = tdl.from_dlpack(cnt.to_dlpack()).reshape(-1) > 0
    return out


def _samples(a, b, c, area, step):
    g = torch.Generator(device=a.device)
    g.manual_seed(0)
    n = torch.floor(area / step ** 2 + torch.rand(area.shape, generator=g, device=a.device, dtype=a.dtype)).long() + 1
    fi = torch.repeat_interleave(torch.arange(len(a), device=a.device), n)
    u = torch.rand(len(fi), generator=g, device=a.device, dtype=a.dtype)
    v = torch.rand(len(fi), generator=g, device=a.device, dtype=a.dtype)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return a[fi] + u[:, None] * (b[fi] - a[fi]) + v[:, None] * (c[fi] - a[fi])


def loose_faces(V: torch.Tensor, F: torch.Tensor, z_floor: float, attach: float, floor: float):
    """(keep mask over the faces, report)."""
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
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * torch.cross(b - a, c - a, dim=1).norm(dim=1)
    parea = torch.zeros(len(pieces), device=dev, dtype=area.dtype).index_add_(0, pof, area)
    zmin = torch.full((len(pieces),), float("inf"), device=dev, dtype=V.dtype).scatter_reduce_(
        0, pof, torch.minimum(torch.minimum(a[:, 2], b[:, 2]), c[:, 2]), "amin")
    keep = zmin <= z_floor + floor
    keep[int(parea.argmax())] = True
    cen = (a + b + c) / 3
    rounds = 0
    while True:
        rounds += 1
        kf = keep[pof]
        cand = ~kf
        if not cand.any():
            break
        S = _samples(a[kf], b[kf], c[kf], area[kf], attach / 2)
        near = _within(S, cen[cand], attach)
        add = torch.zeros(len(pieces), dtype=torch.bool, device=dev)
        add[pof[cand][near]] = True
        add &= ~keep
        if not add.any():
            break
        keep |= add
    kf = keep[pof]
    rep = {"pieces": int(len(pieces)), "area_m2": float(area.sum()), "rounds": rounds,
           "kept_pieces": int(keep.sum()), "kept_m2": float(area[kf].sum()),
           "dropped_pieces": int((~keep).sum()), "dropped_m2": float(area[~kf].sum())}
    return kf, rep


def main() -> None:
    inp, work, out = sys.argv[1], sys.argv[2], sys.argv[3]
    attach = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    floor = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
    dev = torch.device("cuda")
    d = torch.load(inp, map_location="cpu", weights_only=False)
    V = d["vertices"].to(dev).double()
    F = d["faces"].to(dev).long()
    zf = float(json.load(open(f"{work}/levels.json"))["z_floor"])
    keep, rep = loose_faces(V, F, zf, attach, floor)
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
          f"({rep['kept_m2']:.1f} m2 — the largest, those reaching the floor within {floor * 100:.0f} cm of z {zf:+.3f} and, "
          f"in {rep['rounds']} rounds to convergence, those within {attach * 100:.0f} cm of a kept one); dropped "
          f"{rep['dropped_pieces']:,} floating ({rep['dropped_m2']:.1f} m2)", flush=True)


if __name__ == "__main__":
    main()
