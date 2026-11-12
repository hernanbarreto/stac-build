"""tools/genrecon/cloud_shape.py and tools/genrecon/drop_loose.py: GenRecon's shape from the cloud — one sheet per
surface even through noise, the faces of a crease kept, nothing where the cloud has no points — and the loose floating
pieces out (USER 2026-11-12)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("cloud_shape / drop_loose run on the card", allow_module_level=True)
pytest.importorskip("open3d")

_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "genrecon"
sys.path.insert(0, str(_TOOLS))
import cloud_shape as CS  # noqa: E402
import drop_loose as DL  # noqa: E402

DEV = torch.device("cuda")
R = 64                                                      # a small lattice: 1 / 64 of the unit cube per cell


def _cells(P):
    P = torch.as_tensor(P, device=DEV, dtype=torch.float64)
    N, M = CS.plane_frames(P, 1.0 / R)
    return CS.chunk_cells(P, N, M, R=R)


def _plane(z=0.013, hole=None, step=0.004, noise=0.0005, seed=0):
    g = np.arange(-0.4, 0.4, step)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    if hole is not None:
        P = P[~((np.abs(P[:, 0]) < hole) & (np.abs(P[:, 1]) < hole))]
    return P + np.random.default_rng(seed).normal(0, noise, P.shape)


def test_a_floor_is_one_sheet_at_its_height():
    coords, dual, inter = _cells(_plane())
    z = ((coords[:, 2].double() + dual[:, 2].double()) / R - 0.5)
    crossed = inter[:, 2]
    assert crossed.sum() > 0 and not inter[:, 0].any() and not inter[:, 1].any()
    cols = coords[crossed][:, :2]
    assert len(torch.unique(cols, dim=0)) == len(cols)      # one crossing per column: one sheet, never a shell
    assert (z[crossed] - 0.013).abs().max() < 1.0 / R       # at the cloud's height, within a cell


def test_a_noisy_floor_is_still_one_sheet():
    # noise of half a cell across the floor (bufferStop's floor: +-2 cm on 7.2 mm cells came out in 2-3 layers in 37 %
    # of its columns with an isotropic Gaussian; the box along the normal gives every corner of a column the same planes)
    coords, dual, inter = _cells(_plane(noise=0.5 / R))
    crossed = inter[:, 2]
    _, n = torch.unique(coords[crossed][:, :2], dim=0, return_counts=True)
    assert float((n > 1).double().mean()) < 0.005


def test_a_thin_plate_seen_from_both_sides_is_one_sheet():
    a = _plane(z=0.013, noise=0.0)
    b = a.copy()
    b[:, 2] += 0.5 / R                                       # its two faces half a cell apart
    coords, dual, inter = _cells(np.concatenate([a, b]))
    _, n = torch.unique(coords[:, :2], dim=0, return_counts=True)
    assert int((n > 1).sum()) == 0


def test_an_opening_stays_open_to_its_rim():
    hole = 0.15
    coords, dual, inter = _cells(_plane(hole=hole))
    cen = (coords.double() + dual.double()) / R - 0.5
    inside = torch.maximum(cen[:, 0].abs(), cen[:, 1].abs()) < hole - 0.5 / R
    assert not inside.any()                                  # the surface ends at the last points (half a cell), not
    #                                                          BAND cells into the opening (measured 2.6 cells without the half maximum)


def test_a_crease_keeps_its_two_faces_and_grows_no_fin():
    xc, zc = 0.1037, 0.0071                                  # the crease off the lattice
    g = np.arange(-0.3, 0.3, 0.004)
    A, B = np.meshgrid(g, g, indexing="ij")
    top = np.stack([A.ravel(), B.ravel(), np.full(A.size, zc)], 1)
    top = top[top[:, 0] <= xc]
    side = np.stack([np.full(A.size, xc), A.ravel(), B.ravel() * 0.5 - 0.15 + zc], 1)
    side = side[side[:, 2] <= zc]
    coords, dual, inter = _cells(np.concatenate([top, side]))
    cen = (coords.double() + dual.double()) / R - 0.5
    mid = cen[:, 1].abs() < 0.2
    dx, dz = cen[mid, 0] - xc, cen[mid, 2] - zc
    on_faces = torch.where((dx <= 0) & (dz <= 0), torch.minimum(dx.abs(), dz.abs()),
                           torch.where(dx > 0, torch.where(dz > 0, torch.sqrt(dx ** 2 + dz ** 2), dx), dz))
    assert float(on_faces.max()) < 0.5 / R                   # every vertex on one of the two faces (the corner cut by
    #                                                          under a cell: 0.72 measured), none off them
    fin = ((dx > 0.25 / R) & (dz > -1.0 / R)) | ((dz > 0.25 / R) & (dx > -1.0 / R))
    assert int(fin.sum()) == 0                               # nothing past the crease


def test_a_pipe_of_three_cells_keeps_its_round_shape():
    r = 3.0 / R
    th = np.arange(0, 2 * np.pi, 0.004 / r)
    y = np.arange(-0.3, 0.3, 0.004)
    T, Y = np.meshgrid(th, y, indexing="ij")
    P = np.stack([0.0123 + r * np.cos(T.ravel()), Y.ravel(), 0.0071 + r * np.sin(T.ravel())], 1)
    P = P + np.random.default_rng(0).normal(0, 0.13 / R, P.shape)
    coords, dual, inter = _cells(P)
    cen = (coords.double() + dual.double()) / R - 0.5
    q = cen[(cen[:, 1] - 0.0).abs() < 0.25]
    rr = torch.sqrt((q[:, 0] - 0.0123) ** 2 + (q[:, 2] - 0.0071) ** 2)
    assert len(q) > 100 and float((rr - r).abs().max()) < 0.5 / R


def _square(x0, y0, z, s, n=8):
    g = np.linspace(0, s, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = np.stack([x0 + X.ravel(), y0 + Y.ravel(), np.full(X.size, z)], 1)
    F = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            F += [[a, a + n, a + n + 1], [a, a + n + 1, a + 1]]
    return V, np.array(F)


def test_floating_pieces_go_and_a_chain_of_close_ones_stays():
    parts = [_square(-1, -1, 0.0, 2.0)]                       # the floor: the largest
    parts.append(_square(0.2, 0.2, 0.6, 0.3))                 # floating 60 cm up, far from all: goes
    for k in range(14):                                       # a chain up a wall, each piece 1 cm from the last
        parts.append(_square(1.01 + 0.31 * k, 0.0, 0.01 + 0.0, 0.3))
    Vs, Fs, nv = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + nv)
        nv += len(V)
    V = torch.as_tensor(np.concatenate(Vs), device=DEV, dtype=torch.float64)
    F = torch.as_tensor(np.concatenate(Fs), device=DEV)
    V[V[:, 0] > 1.0, 2] += 0.2                                # the chain floats 20 cm up, joined only piece to piece
    keep, rep = DL.loose_faces(V, F, z_floor=0.0, attach=0.02, floor=0.05)
    per = [len(F) for _, F in parts]
    starts = np.cumsum([0] + per)
    kept = [bool(keep[starts[i]:starts[i + 1]].all()) for i in range(len(parts))]
    assert kept[0] and not kept[1]
    assert rep["dropped_pieces"] >= 1


def test_the_runner_the_worker_and_the_vendor_carry_the_shape_from_the_cloud():
    sh = (_TOOLS / "run_cloud.sh").read_text()
    assert "--shape_from_cloud --shape_refine_t ${GENRECON_SHAPE_REFINE_T:?" in sh
    assert "drop_loose.py $O/to_glb_inputs.pt $W $O/to_glb_inputs_clean.pt" in sh
    assert 'BAKE_FLAGS="--skip_remesh --smooth_iters ${GENRECON_SMOOTH_ITERS:?' in sh
    assert sh.count("--inputs $INPUTS") == 2 and "bake_chunks_parallel.py $INPUTS" in sh
    patch = (_TOOLS / "genrecon_stac.patch").read_text()
    assert "shape_latents_override" in patch and "t_start" in patch and "--shape_from_cloud" in patch
    sys.path.insert(0, str(_TOOLS.parents[1]))
    from workers import genrecon_worker as GW
    assert GW._number({"shape_refine_t": 0.0}, "shape_refine_t", float, 0.0, 1.0) == "0.0"
    assert GW._number({"smooth_iters": 30}, "smooth_iters", int, 0, 1000) == "30"
    with pytest.raises(RuntimeError, match="genrecon.shape_refine_t is missing"):
        GW._number({}, "shape_refine_t", float, 0.0, 1.0)
    with pytest.raises(RuntimeError, match="must be a int"):
        GW._number({"smooth_iters": 2.5}, "smooth_iters", int, 0, 1000)
