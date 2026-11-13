"""GenRecon's joint decode by regions gives the single pass's arrays, whatever the cut.

The real decoders run on the card only (flex_gemm). Here the same STRUCTURE — a per-voxel input layer, stacks of
3x3x3 submanifold blocks, the up block (subdivision -> children by slot, one convolution on the parents, one on the
children, the skip) and the texture decoder guided by the shape decoder's subdivisions — runs in INTEGER arithmetic,
so "equal" means equal: any difference between the single pass and the regions is a fault of the logic, never of a
rounding. The dual-grid extraction mirrors o_voxel's flexible_dual_grid_to_mesh (train=False).
"""
import sys
from pathlib import Path

import pytest
import torch

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "genrecon"
sys.path.insert(0, str(_TOOLS))
import decode_regions as DR  # noqa: E402

P = 1009   # every feature lives in [0, P)
C = 8      # channels at every level: an up block hands each child C from its convolution and C / 8 from the skip
SLOT = torch.tensor([[s % 2, (s // 2) % 2, (s // 4) % 2] for s in range(8)])
NEAR = torch.tensor([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
QUAD = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                     [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]])


def _rows(xyz, query):
    """The row of each query voxel in xyz, -1 where there is none."""
    def key(c):
        c = c + 2
        return (c[:, 0] * (1 << 20) + c[:, 1]) * (1 << 20) + c[:, 2]
    keys = key(xyz)
    order = torch.argsort(keys)
    q = key(query)
    pos = torch.searchsorted(keys[order], q).clamp(max=keys.numel() - 1)
    return torch.where(keys[order][pos] == q, order[pos], torch.full_like(pos, -1))


def _conv(near, feats, w):
    out = torch.zeros((feats.shape[0], w.shape[2]), dtype=torch.int64)
    for k in range(27):
        ok = near[k] >= 0
        out[ok] += feats[near[k][ok]] @ w[k]
    return out % P


def _near(xyz):
    return [_rows(xyz, xyz + NEAR[k]) for k in range(27)]


class Stub:
    """A local sparse decoder with the vendor's structure, in integers."""

    def __init__(self, num_blocks, out_channels, seed, full=False):
        g = torch.Generator().manual_seed(seed)

        def w(*shape):
            return torch.randint(1, 9, shape, generator=g)
        self.num_blocks, self.full = list(num_blocks), full
        self.w_in = w(4, C)
        self.stacks = [[w(27, C, C) for _ in range(b)] for b in num_blocks]
        self.ups = [(w(C, 8), w(27, C, 8 * C), w(27, C, C)) for _ in num_blocks[:-1]]
        self.w_out = w(C, out_channels)

    def __call__(self, xyz, feats, guide=None):
        h = (feats @ self.w_in) % P
        subs = []
        for lev, stack in enumerate(self.stacks):
            near = _near(xyz) if stack or lev < len(self.ups) else None
            for wk in stack:
                h = (h + (_conv(near, h, wk) * 5 + 3) % P) % P
            if lev == len(self.ups):
                break
            w_sub, w1, w2 = self.ups[lev]
            if guide is not None:
                sub = guide[lev]
            elif self.full:
                sub = torch.ones((h.shape[0], 8), dtype=torch.bool)
            else:
                sub = ((h @ w_sub) % P) > P // 3
            subs.append(sub)
            wide = _conv(near, (h * 7 + 1) % P, w1)                 # on the parents, 8 children's worth of channels
            parent, slot = sub.nonzero(as_tuple=True)               # parents in row order, slots ascending
            child_h = wide.reshape(-1, 8, C)[parent, slot]
            child_x = h.reshape(-1, 8, C // 8)[parent, slot].repeat_interleave(8, dim=1)
            xyz = xyz[parent] * 2 + SLOT[slot]
            h = (_conv(_near(xyz), (child_h * 3 + 2) % P, w2) + child_x) % P
        return xyz, (h @ self.w_out) % P, subs


def _heads(out):
    return (out[:, 0:3] % 16).float() / 16.0, (out[:, 3:6] % 3) == 0, (out[:, 6] % 97 + 1)


def _extract(fine, offs, inter, lerp, origin=-0.5, cell=1.0 / 64):
    verts = (fine.float() + offs) * cell + origin
    corner = (fine.reshape(-1, 1, 1, 3) + QUAD)[inter]              # (M, 4, 3) by (voxel row, axis)
    idx = _rows(fine, corner.reshape(-1, 3)).reshape(-1, 4)
    idx = idx[(idx >= 0).all(dim=1)]
    ws = lerp[idx]
    first = (ws[:, 0] * ws[:, 2]) > (ws[:, 1] * ws[:, 3])
    faces = torch.where(first[:, None], idx[:, [0, 1, 2, 0, 2, 3]], idx[:, [0, 1, 3, 3, 1, 2]]).reshape(-1, 3)
    return verts, faces


LATTICE = (30, 24, 5)
SHAPE_BLOCKS, TEX_BLOCKS = [1, 2, 1, 0], [2, 1, 1, 0]
LEVELS = len(SHAPE_BLOCKS) - 1


def _scene(seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.zeros(LATTICE, dtype=torch.bool)
    grid[:, :, 1] = torch.rand(LATTICE[:2], generator=g) > 0.15    # a floor with holes
    grid[LATTICE[0] // 2, :, 1:] = True                            # a wall
    grid |= torch.rand(LATTICE, generator=g) > 0.97                # loose voxels
    xyz = grid.nonzero()                                           # ascending (x, y, z): the joint slat's order
    return xyz, torch.randint(0, P, (xyz.shape[0], 4), generator=g), torch.randint(0, P, (xyz.shape[0], 4), generator=g)


@pytest.fixture(scope="module")
def scene():
    xyz, shape_lat, tex_lat = _scene()
    shape_dec, tex_dec = Stub(SHAPE_BLOCKS, 7, seed=1), Stub(TEX_BLOCKS, 6, seed=2)
    fine, out, subs = shape_dec(xyz, shape_lat)
    verts, faces = _extract(fine, *_heads(out))
    tex_fine, tex_out, _ = tex_dec(xyz, tex_lat, guide=subs)
    assert torch.equal(tex_fine, fine)
    return dict(xyz=xyz, shape_lat=shape_lat, tex_lat=tex_lat, shape_dec=shape_dec, tex_dec=tex_dec, fine=fine,
                verts=verts, faces=faces, tex=tex_out, margin=DR.margin(SHAPE_BLOCKS, TEX_BLOCKS))


def _by_regions(s, regions, margin, spoil=None):
    stash = {}

    def shape_region(n, rows):
        fine, out, subs = s["shape_dec"](s["xyz"][rows], s["shape_lat"][rows])
        stash[n] = subs
        verts, faces = _extract(fine, *_heads(out))
        if spoil is not None:
            fine, verts, faces = spoil(n, fine, verts, faces)
        return fine, verts, faces

    def tex_region(n, rows):
        fine, out, _ = s["tex_dec"](s["xyz"][rows], s["tex_lat"][rows], guide=stash[n])
        return fine, out

    quiet = lambda line: None  # noqa: E731
    verts, faces, dropped = DR.decode_shape(s["xyz"], regions, margin, LEVELS, LATTICE, shape_region, log=quiet)
    tex_xyz, tex = DR.decode_tex(s["xyz"], regions, margin, LEVELS, LATTICE, tex_region, log=quiet)
    return verts, faces, dropped, tex_xyz, tex


def test_the_radius_of_the_real_decoders_and_the_margin():
    real = [4, 16, 8, 4, 0]     # TRELLIS.2's shape and texture decoders (ckpts/*_dec_next_dc_f16c32_fp16.json)
    assert DR.dependency_radius(real) == 18
    assert DR.guided_radius(real, real) == 18
    assert DR.margin(real, real) == 19
    with pytest.raises(ValueError, match="levels"):
        DR.guided_radius([1, 0], [1, 2, 0])


@pytest.mark.parametrize("blocks", [[1, 2, 1, 0], [0, 1, 0], [2, 0, 3], [3], [4, 16, 8, 4, 0]])
def test_the_radius_is_exactly_how_far_a_latent_reaches(blocks):
    """One latent changed on a dense line: the output changes on the descendants of the latent voxels within the
    radius — up to it, and not one voxel further."""
    radius, levels = DR.dependency_radius(blocks), len(blocks) - 1
    n = 2 * radius + 7
    xyz = torch.zeros((n, 3), dtype=torch.int64)
    xyz[:, 0] = torch.arange(n)
    feats = torch.randint(0, P, (n, 4), generator=torch.Generator().manual_seed(3))
    other = feats.clone()
    other[n // 2] = (other[n // 2] + 1) % P
    dec = Stub(blocks, 7, seed=4, full=True)
    fine, a, _ = dec(xyz, feats)
    _, b, _ = dec(xyz, other)
    far = ((fine[:, 0] >> levels) - n // 2).abs()[(a != b).any(dim=1)]
    assert int(far.max()) == radius


def test_the_cut_is_disjoint_covers_every_voxel_and_holds_the_cap(scene):
    xyz, m = scene["xyz"], scene["margin"]
    cap = xyz.shape[0] // 2
    regions = DR.partition(xyz, m, cap)
    assert len(regions) > 1
    owners = torch.stack([DR.owned_mask(xyz, r) for r in regions]).sum(dim=0)
    assert bool((owners == 1).all())
    for r in regions:
        assert r.inflated <= cap and r.inflated == int(DR.inflated_mask(xyz, r, m).sum())
        assert r.owned == int(DR.owned_mask(xyz, r).sum())
    shuffled = xyz[torch.randperm(xyz.shape[0], generator=torch.Generator().manual_seed(5))]
    assert DR.partition(shuffled, m, cap) == regions
    whole = DR.partition(xyz, m, xyz.shape[0])
    assert len(whole) == 1 and whole[0].owned == whole[0].inflated == xyz.shape[0]
    assert "regions" in DR.describe(regions, xyz.shape[0], m, cap)


def test_a_cap_under_one_voxels_neighbourhood_fails_naming_the_key(scene):
    with pytest.raises(ValueError, match="genrecon.joint_decode_max_voxels"):
        DR.partition(scene["xyz"], scene["margin"], 20)


@pytest.mark.parametrize("fraction", [1.0, 0.75, 0.5, 0.36])
def test_the_regions_give_the_single_pass(scene, fraction):
    xyz, m = scene["xyz"], scene["margin"]
    regions = DR.partition(xyz, m, int(xyz.shape[0] * fraction))
    assert (len(regions) > 1) == (fraction < 1.0)
    verts, faces, dropped, tex_xyz, tex = _by_regions(scene, regions, m)
    assert dropped == 0
    assert torch.equal(verts, scene["verts"])
    assert torch.equal(faces.long(), scene["faces"])
    assert torch.equal(tex_xyz.long(), scene["fine"])
    assert torch.equal(tex, scene["tex"])


def test_without_the_margin_the_regions_do_not_give_the_single_pass(scene):
    """The equality above is the margin's doing: the same cut decoded on the boxes alone differs."""
    regions = DR.partition(scene["xyz"], scene["margin"], scene["xyz"].shape[0] // 2)
    verts, faces, _, _, _ = _by_regions(scene, regions, 0)
    assert not (verts.shape == scene["verts"].shape and torch.equal(verts, scene["verts"])
                and faces.shape[0] == scene["faces"].shape[0] and torch.equal(faces.long(), scene["faces"]))


def test_the_owner_decides_what_exists_and_a_face_without_its_corner_is_dropped_and_counted(scene):
    """On the card two regions may disagree on a subdivision bit. Here region `owner` loses one fine voxel that
    faces anchored in ANOTHER region use: those faces are dropped and counted, everything else is the single pass
    without that voxel."""
    xyz, m, fine, faces = scene["xyz"], scene["margin"], scene["fine"], scene["faces"]
    regions = DR.partition(xyz, m, xyz.shape[0] // 2)
    region_of = torch.stack([DR.owned_mask(fine >> LEVELS, r) for r in regions]).long().argmax(dim=0)
    anchor = fine[faces].min(dim=1).values
    anchor_region = region_of[_rows(fine, anchor)]
    foreign = (region_of[faces] != anchor_region[:, None])
    assert bool(foreign.any()), "the scene has no face across two regions: the test would prove nothing"
    f, c = foreign.nonzero()[0].tolist()
    lost = int(faces[f, c])
    owner = int(region_of[lost])
    lost_xyz = fine[lost]

    def spoil(n, r_fine, r_verts, r_faces):
        if n != owner:
            return r_fine, r_verts, r_faces
        keep = ~(r_fine == lost_xyz).all(dim=1)
        new_row = torch.cumsum(keep.long(), dim=0) - 1
        r_faces = r_faces[keep[r_faces].all(dim=1)]
        return r_fine[keep], r_verts[keep], new_row[r_faces]

    verts, got, dropped, _, _ = _by_regions(scene, regions, m, spoil=spoil)
    keep = torch.ones(fine.shape[0], dtype=torch.bool)
    keep[lost] = False
    uses = (faces == lost).any(dim=1)
    assert dropped == int((uses & (anchor_region != owner)).sum()) and dropped > 0
    assert torch.equal(verts, scene["verts"][keep])
    assert torch.equal(got.long(), (torch.cumsum(keep.long(), dim=0) - 1)[faces[~uses]])
