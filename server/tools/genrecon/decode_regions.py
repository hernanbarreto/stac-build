"""GenRecon's joint decode BY REGIONS: the single pass's mesh, with the card's memory bound by the region.

USER 2026-11-13: "Debemos hacer entrar pccr2408 hay que ver la forma la técnica para que sea escalable a nubes
masivas". The vendor decodes the latents of the whole scene in ONE pass of the sparse decoders, so the card's memory
grows with the scene (measured without solid ground: 4.2 / 4.4 GB for bufferStop's 7,014 latent voxels, 17.8 / 19.0 GB
for pccr 3108's 42,689 — about 0.4 GB per 1000 voxels); pccr 2408 holds 220,739 at 2.5 m cubes and an 80 GB card
does not take it.

Both decoders are LOCAL: 3x3x3 submanifold convolutions, per-voxel norms and linears, and a parent -> children
subdivision between levels — no attention, no pooling, nothing global. What the decoder writes on a fine voxel is
therefore decided by the latents within a FINITE number of latent voxels of its level-0 ancestor
(`dependency_radius`). So:

  * the latent lattice is cut into DISJOINT boxes (`partition`), each small enough that the box plus its margin
    holds at most `cap` latent voxels;
  * each box is decoded together with a margin of `margin(...)` latent voxels around it, larger than that radius:
    inside the box the decoder computes what the single pass computes;
  * every fine voxel is taken ONCE, from the box that owns its level-0 ancestor: its vertex, the quads anchored on
    it and its texture voxel. A quad's vertices are looked up by voxel, so a quad that straddles two boxes joins the
    two boxes' own vertices and the mesh has no seam;
  * the parts are put in the order the single pass gives its rows (`hier_key`), so the result is the same arrays —
    vertices, faces, texture voxels — whatever the cut.

In exact arithmetic the result is identical to the single pass. On the card the convolutions run in half precision
and a region is another batch: the last bits may differ, which is measured, not assumed (tests hold the logic exact
on an integer decoder; the real decoders are compared on a real session).

Pure torch, no vendor import: the vendor's joint_decode.py loads this file by path and hands it two callables per
stage (decode + extract one region); tests/test_genrecon_decode_regions.py drives it on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch

CAP_KEY = "genrecon.joint_decode_max_voxels"


def _ceil_half(v: int) -> int:
    return (v + 1) // 2


def _radius_from(blocks: Sequence[int], level: int, r: int) -> int:
    """Latent voxels around a level-0 voxel whose latents decide the features LEAVING the stack of `level` within `r`
    voxels (of that level) of the level-0 voxel's descendants.

    A stack of B 3x3x3 blocks widens the reach by B voxels of its level. The up block before it (level-1 -> level)
    runs one 3x3x3 convolution on the children — reach r + B + 1 children, whose parents lie within
    ceil((r + B + 1) / 2) — after one 3x3x3 convolution on the parents — one more parent; the children that EXIST
    are the parents' own subdivision, read from the same parents."""
    for lev in range(level, 0, -1):
        r = _ceil_half(r + int(blocks[lev]) + 1) + 1
    return r + int(blocks[0])


def dependency_radius(num_blocks: Sequence[int]) -> int:
    """Latent voxels beyond which a latent no longer reaches the decoder's output on a level-0 voxel's descendants."""
    blocks = [int(b) for b in num_blocks]
    if len(blocks) < 1 or min(blocks) < 0:
        raise ValueError(f"num_blocks must be a non-empty list of non-negative integers, got {list(num_blocks)}")
    return _radius_from(blocks, len(blocks) - 1, 0)


def guided_radius(num_blocks: Sequence[int], guide_blocks: Sequence[int]) -> int:
    """The same for a decoder whose subdivisions are GIVEN by another decoder (the texture decoder, guided by the
    shape decoder's): the children that exist are decided by the guide's features on the parents, so the guide's
    own radius at each level enters."""
    blocks = [int(b) for b in num_blocks]
    guide = [int(b) for b in guide_blocks]
    if len(blocks) != len(guide):
        raise ValueError(f"the guided decoder has {len(blocks)} levels and its guide {len(guide)}: the subdivisions "
                         "of one cannot drive the other")

    def leaving(level: int, r: int) -> int:
        if level == 0:
            return r + blocks[0]
        half = _ceil_half(r + blocks[level] + 1)
        return max(leaving(level - 1, half + 1), _radius_from(guide, level - 1, half))

    return leaving(len(blocks) - 1, 0)


def margin(shape_blocks: Sequence[int], tex_blocks: Sequence[int]) -> int:
    """Latent voxels decoded around a region's own box.

    The shape decoder's radius PLUS ONE: a quad anchored on an owned fine voxel reads the split weight of its three
    neighbours one fine voxel further, whose ancestor may be one latent voxel outside the box. The texture decoder
    needs its guided radius."""
    return max(dependency_radius(shape_blocks) + 1, guided_radius(tex_blocks, shape_blocks))


@dataclass(frozen=True)
class Region:
    lo: Tuple[int, int, int]   # the box it OWNS, [lo, hi) in latent voxels of the joint lattice
    hi: Tuple[int, int, int]
    owned: int                 # latent voxels inside the box
    inflated: int              # latent voxels its decode receives (the box and the margin around it)


def _box(xyz: torch.Tensor, lo, hi) -> torch.Tensor:
    lo_t = torch.as_tensor(lo, dtype=xyz.dtype, device=xyz.device)
    hi_t = torch.as_tensor(hi, dtype=xyz.dtype, device=xyz.device)
    return ((xyz >= lo_t) & (xyz < hi_t)).all(dim=1)


def owned_mask(xyz: torch.Tensor, region: Region) -> torch.Tensor:
    return _box(xyz, region.lo, region.hi)


def inflated_mask(xyz: torch.Tensor, region: Region, margin_voxels: int) -> torch.Tensor:
    return _box(xyz, [v - margin_voxels for v in region.lo], [v + margin_voxels for v in region.hi])


def partition(xyz: torch.Tensor, margin_voxels: int, cap: int) -> List[Region]:
    """Disjoint boxes covering every latent voxel, each decoded (box + margin) on at most `cap` latent voxels.

    A box that fits is a region. One that does not is cut across its longest axis (the first of equal ones) at the
    median of its voxels, each half shrunk to its own voxels, low half first. A pure function of the voxels, the
    margin and the cap — the same cut on every run."""
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"the latent voxels must be a non-empty [N, 3] tensor, got {tuple(xyz.shape)}")
    if int(cap) < 1 or int(margin_voxels) < 0:
        raise ValueError(f"{CAP_KEY} must be at least 1 to cut the decode (got {cap}); margin {margin_voxels}")
    pts_all = xyz.detach().to("cpu", torch.int64)
    regions: List[Region] = []

    def cut(pts: torch.Tensor) -> None:
        lo = pts.min(dim=0).values
        hi = pts.max(dim=0).values + 1
        received = int(_box(pts_all, lo - margin_voxels, hi + margin_voxels).sum())
        if received <= cap:
            regions.append(Region(tuple(int(v) for v in lo), tuple(int(v) for v in hi), int(pts.shape[0]), received))
            return
        extent = [int(v) for v in (hi - lo)]
        axis = extent.index(max(extent))
        if extent[axis] < 2:
            raise ValueError(
                f"{CAP_KEY} = {cap} is under what ONE latent voxel's decode receives here ({received} voxels within "
                f"{margin_voxels} of it): the cut cannot go below a voxel — raise the key")
        along = pts[:, axis]
        mid = int(torch.sort(along).values[along.shape[0] // 2])
        if mid == int(lo[axis]):
            mid += 1
        low = along < mid
        cut(pts[low])
        cut(pts[~low])

    cut(pts_all)
    return regions


def check_key_room(lattice: Sequence[int], levels: int) -> None:
    cells = int(lattice[0]) * int(lattice[1]) * int(lattice[2]) * 8 ** int(levels)
    if cells >= 2 ** 63:
        raise ValueError(f"the joint lattice {tuple(int(v) for v in lattice)} at {levels} subdivision levels does "
                         "not fit a 64-bit row key: the decode by regions cannot order this scene")


def hier_key(fine_xyz: torch.Tensor, levels: int, lattice: Sequence[int]) -> torch.Tensor:
    """The place a fine voxel takes among the single pass's rows.

    The single pass orders the latent voxels by (x, y, z) on the joint lattice and every subdivision writes a
    parent's children in the parent's row order, by slot (x the fastest bit, then y, then z). So a fine voxel's row
    is ordered by its level-0 ancestor first and then, level by level, by its slot under each ancestor."""
    f = fine_xyz.to(torch.int64)
    a = f >> levels
    key = (a[:, 0] * int(lattice[1]) + a[:, 1]) * int(lattice[2]) + a[:, 2]
    for k in range(levels - 1, -1, -1):
        o = (f >> k) & 1
        key = key * 8 + (o[:, 0] + 2 * o[:, 1] + 4 * o[:, 2])
    return key


def _report(regions: Sequence[Region], total: int, margin_voxels: int, cap: int) -> str:
    owned = [r.owned for r in regions]
    received = [r.inflated for r in regions]
    return (f"[STAC] joint decode by regions: {total:,} latent voxels over {CAP_KEY} = {cap:,} -> {len(regions)} "
            f"regions with a margin of {margin_voxels} latent voxels; a region owns {min(owned):,}..{max(owned):,} "
            f"and decodes {min(received):,}..{max(received):,} ({sum(received) / total:.2f} x the scene in all)")


def describe(regions: Sequence[Region], total: int, margin_voxels: int, cap: int) -> str:
    return _report(regions, total, margin_voxels, cap)


def decode_shape(xyz: torch.Tensor, regions: Sequence[Region], margin_voxels: int, levels: int,
                 lattice: Sequence[int], run_region: Callable, log: Callable = print):
    """The scene's mesh from the regions' decodes: (vertices [V, 3], faces [F, 3] int32, dropped faces).

    `run_region(n, rows)` decodes the latent rows `rows` (a mask over `xyz`, the joint order kept) and extracts
    their mesh: (fine voxels [m, 3], vertices [m, 3] — one per fine voxel, in its row —, faces [f, 3] over those
    rows). Kept from each region: the vertices of the fine voxels it owns and the faces whose ANCHOR it owns (the
    anchor of a dual quad is the voxel the quad's edge belongs to — the per-axis minimum of its corners, for both
    triangles of either split)."""
    check_key_room(lattice, levels)
    device = xyz.device
    v_key, v_pos, f_key, f_corners = [], [], [], []
    for n, region in enumerate(regions):
        fine, verts, faces = run_region(n, inflated_mask(xyz, region, margin_voxels))
        if verts.shape[0] != fine.shape[0]:
            raise RuntimeError(f"region {n}: the extraction returned {verts.shape[0]:,} vertices for "
                               f"{fine.shape[0]:,} fine voxels — one vertex per voxel is what the merge relies on")
        key = hier_key(fine, levels, lattice)
        own = owned_mask(fine.to(torch.int64) >> levels, region)
        tri = faces.long()
        anchor = fine[tri].min(dim=1).values
        tri_own = owned_mask(anchor.to(torch.int64) >> levels, region)
        v_key.append(key[own].cpu())
        v_pos.append(verts[own].cpu())
        f_key.append(hier_key(anchor[tri_own], levels, lattice).cpu())
        f_corners.append(key[tri[tri_own]].cpu())
        log(f"[STAC] joint decode (shape) region {n + 1}/{len(regions)}: box {list(region.lo)}..{list(region.hi)}, "
            f"{region.inflated:,} latent voxels decoded -> {fine.shape[0]:,} fine voxels, {int(own.sum()):,} owned; "
            f"{faces.shape[0]:,} faces, {int(tri_own.sum()):,} owned")
        del fine, verts, faces, key, own, tri, anchor, tri_own
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keys = torch.cat(v_key).to(device)
    order = torch.argsort(keys)
    keys = keys[order]
    if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
        raise RuntimeError("a fine voxel came out of two regions: the regions' boxes are not disjoint")
    vertices = torch.cat(v_pos).to(device)[order]
    del order, v_key, v_pos
    if keys.numel() >= 2 ** 31:
        raise RuntimeError(f"{keys.numel():,} vertices do not fit the mesh's 32-bit face indices")

    corners = torch.cat(f_corners).to(device)
    anchors = torch.cat(f_key).to(device)
    del f_corners, f_key
    if keys.numel() == 0:
        return vertices, torch.zeros((0, 3), dtype=torch.int32, device=device), int(corners.shape[0])
    index = torch.searchsorted(keys, corners.reshape(-1)).clamp_(max=keys.numel() - 1).reshape(-1, 3)
    held = (keys[index] == corners).all(dim=1)
    del corners
    dropped = int((~held).sum())
    if dropped:
        # A corner's voxel exists in the region that anchors the face and NOT in the region that owns that voxel:
        # the two decodes disagree on a subdivision bit (half precision, another batch). The owner decides what
        # exists; the face has no fourth corner and is not made. Counted and declared, never silent.
        index = index[held]
        anchors = anchors[held]
    order = torch.argsort(anchors, stable=True)
    faces = index[order].to(torch.int32)
    return vertices, faces, dropped


def decode_tex(xyz: torch.Tensor, regions: Sequence[Region], margin_voxels: int, levels: int,
               lattice: Sequence[int], run_region: Callable, log: Callable = print):
    """The scene's texture voxels from the regions' decodes: (fine voxels [M, 3] int32, features [M, C]), in the
    single pass's row order. `run_region(n, rows)` -> (fine voxels [m, 3], features [m, C])."""
    check_key_room(lattice, levels)
    device = xyz.device
    t_key, t_xyz, t_feat = [], [], []
    for n, region in enumerate(regions):
        fine, feats = run_region(n, inflated_mask(xyz, region, margin_voxels))
        if feats.shape[0] != fine.shape[0]:
            raise RuntimeError(f"region {n}: {feats.shape[0]:,} texture rows for {fine.shape[0]:,} fine voxels")
        own = owned_mask(fine.to(torch.int64) >> levels, region)
        t_key.append(hier_key(fine[own], levels, lattice).cpu())
        t_xyz.append(fine[own].to(torch.int32).cpu())
        t_feat.append(feats[own].cpu())
        log(f"[STAC] joint decode (texture) region {n + 1}/{len(regions)}: {region.inflated:,} latent voxels decoded "
            f"-> {fine.shape[0]:,} fine voxels, {int(own.sum()):,} owned")
        del fine, feats, own
        if device.type == "cuda":
            torch.cuda.empty_cache()
    keys = torch.cat(t_key).to(device)
    order = torch.argsort(keys)
    keys = keys[order]
    if keys.numel() > 1 and not bool((keys[1:] > keys[:-1]).all()):
        raise RuntimeError("a texture voxel came out of two regions: the regions' boxes are not disjoint")
    del keys, t_key
    coords = torch.cat(t_xyz).to(device)[order]
    feats = torch.cat(t_feat).to(device)[order]
    return coords, feats
