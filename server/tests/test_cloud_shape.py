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


def _cells(P, cam):
    P = torch.as_tensor(P, device=DEV, dtype=torch.float64)
    cam = torch.as_tensor(cam, device=DEV, dtype=torch.float64)
    Nf, Mf, Ns, Ms = CS.plane_frames(P, 1.0 / R)
    Nf, Cf = CS.orient(Nf, P, cam)
    return CS.chunk_cells(P, Nf, Mf, Cf, Ns, Ms, R=R)


def _plane(z=0.013, hole=None, step=0.004, noise=0.0005, seed=0):
    g = np.arange(-0.4, 0.4, step)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    if hole is not None:
        P = P[~((np.abs(P[:, 0]) < hole) & (np.abs(P[:, 1]) < hole))]
    return P + np.random.default_rng(seed).normal(0, 1, P.shape) * np.array([noise * 0.3, noise * 0.3, noise])


def _above(P, h=1.0):
    return P + np.array([0.0, 0.0, h])


def _multi(coords, inter):
    _, n = torch.unique(coords[inter[:, 2]][:, :2], dim=0, return_counts=True)
    return float((n > 1).double().mean())


def test_a_floor_is_one_sheet_at_its_height():
    P = _plane()
    coords, dual, inter = _cells(P, _above(P))
    z = ((coords[:, 2].double() + dual[:, 2].double()) / R - 0.5)
    crossed = inter[:, 2]
    assert crossed.sum() > 0 and not inter[:, 0].any() and not inter[:, 1].any()
    assert _multi(coords, inter) == 0.0                      # one crossing per column: one sheet, never a shell
    assert (z[crossed] - 0.013).abs().max() < 1.0 / R       # at the cloud's height, within a cell


def test_a_noisy_floor_is_still_one_sheet():
    # noise of a cell across the floor: every corner of a column reads the same planes (the box along the normal) and
    # one value per corner (the mean signed distance, toward the cameras): one sign change per column
    P = _plane(noise=1.0 / R)
    coords, dual, inter = _cells(P, _above(P))
    assert _multi(coords, inter) == 0.0


def test_a_thin_plate_seen_from_both_sides_is_one_whole_sheet():
    for gap in (0.25, 0.5, 1.0):                             # its faces a quarter, half, a whole cell apart
        A = _plane(z=0.013, noise=0.1 / R)
        Bp = _plane(z=0.013 - gap / R, noise=0.1 / R, seed=1)
        coords, dual, inter = _cells(np.concatenate([A, Bp]), np.concatenate([_above(A), _above(Bp, -1.0)]))
        cols = torch.unique(coords[inter[:, 2]][:, :2], dim=0)
        assert _multi(coords, inter) == 0.0
        assert len(cols) / (0.8 * R) ** 2 > 0.97            # whole: no holes (an unoriented read lost 17 % at 1 cell)
        z = (coords[:, 2].double() + dual[:, 2].double()) / R - 0.5
        assert float((z - (0.013 - gap / R / 2)).abs().quantile(0.99)) < 0.25 / R   # flat, at its middle


def test_an_opening_stays_open_to_its_rim():
    hole = 0.15
    P = _plane(hole=hole)
    coords, dual, inter = _cells(P, _above(P))
    cen = (coords.double() + dual.double()) / R - 0.5
    inside = torch.maximum(cen[:, 0].abs(), cen[:, 1].abs()) < hole - 0.5 / R
    assert not inside.any()                                  # the surface ends at the last points (half a cell)


def test_a_crease_is_sharp_and_grows_no_fin():
    xc, zc = 0.1037, 0.0071                                  # the crease off the lattice
    g = np.arange(-0.3, 0.3, 0.004)
    A, B = np.meshgrid(g, g, indexing="ij")
    top = np.stack([A.ravel(), B.ravel(), np.full(A.size, zc)], 1)
    top = top[top[:, 0] <= xc]
    side = np.stack([np.full(A.size, xc), A.ravel(), B.ravel() * 0.5 - 0.15 + zc], 1)
    side = side[side[:, 2] <= zc]                            # twice as dense as the top: its corners must not cut it
    cam = np.concatenate([top + np.array([0.3, 0, 1.0]), side + np.array([1.0, 0, 0.3])])
    coords, dual, inter = _cells(np.concatenate([top, side]), cam)
    cen = (coords.double() + dual.double()) / R - 0.5
    mid = cen[:, 1].abs() < 0.2
    dx, dz = cen[:, 0] - xc, cen[:, 2] - zc
    line = torch.sqrt(dx ** 2 + dz ** 2)
    nearest = torch.stack([line[mid & (coords[:, 1] == y)].min() for y in torch.unique(coords[mid][:, 1])])
    assert float(nearest.median()) < 0.15 / R                # a vertex ON the crease (0.09 cell; 0.45 before)
    dx, dz = dx[mid], dz[mid]
    on_faces = torch.where((dx <= 0) & (dz <= 0), torch.minimum(dx.abs(), dz.abs()),
                           torch.where(dx > 0, torch.where(dz > 0, torch.sqrt(dx ** 2 + dz ** 2), dx), dz))
    assert float(on_faces.max()) < 0.1 / R                   # every vertex on one of the two faces
    fin = ((dx > 0.25 / R) & (dz > -1.0 / R)) | ((dz > 0.25 / R) & (dx > -1.0 / R))
    assert int(fin.sum()) == 0                               # nothing past the crease


def test_a_pipe_of_three_cells_keeps_its_round_shape():
    r = 3.0 / R
    th = np.arange(0, 2 * np.pi, 0.004 / r)
    y = np.arange(-0.3, 0.3, 0.004)
    T, Y = np.meshgrid(th, y, indexing="ij")
    nrm = np.stack([np.cos(T.ravel()), np.zeros(T.size), np.sin(T.ravel())], 1)
    P = np.array([0.0123, 0.0, 0.0071]) + r * nrm + np.stack([np.zeros(T.size), Y.ravel(), np.zeros(T.size)], 1)
    P = P + np.random.default_rng(0).normal(0, 0.13 / R, P.shape)
    coords, dual, inter = _cells(P, P + nrm)                 # seen from outside all around
    cen = (coords.double() + dual.double()) / R - 0.5
    q = cen[cen[:, 1].abs() < 0.25]
    rr = torch.sqrt((q[:, 0] - 0.0123) ** 2 + (q[:, 2] - 0.0071) ** 2)
    assert len(q) > 100 and float((rr - r).abs().max()) < 0.5 / R


def _noisy_box(subdivisions=5):
    """The unit box of 12 x 4^subdivisions faces: its vertices, the same pushed 1 cm (sigma) along their normals, its
    faces."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_box(1, 1, 1).subdivide_midpoint(number_of_iterations=subdivisions)
    m.compute_vertex_normals()
    V0, F = np.asarray(m.vertices).copy(), np.asarray(m.triangles).copy()
    return V0, V0 + np.asarray(m.vertex_normals) * np.random.default_rng(0).normal(0, 0.01, (len(V0), 1)), F


def test_the_smoothing_flattens_the_faces_and_keeps_the_edges():
    import denoise as DN
    V0, Vn, F = _noisy_box()
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)                 # as denoise.py runs (its main sets it)
    try:
        V2, rep = DN.bilateral_denoise(torch.as_tensor(Vn, device=DEV), torch.as_tensor(F, device=DEV), 20, 10, 0.35)
        V3, _ = DN.bilateral_denoise(torch.as_tensor(Vn, device=DEV), torch.as_tensor(F, device=DEV), 20, 10, 0.35)
    finally:
        torch.use_deterministic_algorithms(was)
    V2, V3 = V2.cpu().numpy(), V3.cpu().numpy()

    def off_box(V):
        q = np.abs(V - 0.5) - 0.5
        return np.median(np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)))
    on = np.isclose(V0, 0) | np.isclose(V0, 1)
    edge = on.sum(1) == 2

    def off_edge(V):
        return np.median(np.linalg.norm(np.where(on[edge], V[edge] - V0[edge], 0.0), axis=1))
    assert off_box(V2) < off_box(Vn) / 4                     # the faces flat (5.5x measured)
    assert off_edge(V2) < off_edge(Vn) / 2                   # the edges sharper, not rounded (2.2x measured)
    assert np.array_equal(V2, V3)                            # the same every time


def test_the_filter_does_not_depend_on_its_blocks():
    """USER 2026-11-12 ("hacelo escalable que pueda manejar nubes masivas"): the neighbour pairs are made for BLOCK_FACES
    faces at a time — the same vertices BIT FOR BIT and the same report whatever the block (bufferStop's remeshed
    7,211,584 faces, measured: one block 13.3 GB on the card, 2,000,000 faces a block 4.7 GB, not one coordinate apart)."""
    import denoise as DN
    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)                 # as filter_mesh runs it
    try:
        for subdivisions, iters, blocks in ((5, (20, 10), (97, 1000, 4096, 12288 + 5)), (2, (3, 2), (1, 7))):
            _, Vn, F = _noisy_box(subdivisions)
            V, Ft = torch.as_tensor(Vn, device=DEV), torch.as_tensor(F, device=DEV)
            whole, rep = DN.bilateral_denoise(V, Ft, *iters, 0.35)
            assert rep["blocks"] == 1 and rep["faces"] == len(F) == 12 * 4 ** subdivisions
            assert rep["neighbour_pairs"] > 3 * len(F)       # more than the three faces across its edges: by vertex
            for block in blocks:
                got, rep_b = DN.bilateral_denoise(V, Ft, *iters, 0.35, block)
                assert torch.equal(got, whole), block
                assert rep_b.pop("blocks") == -(-len(F) // block)
                assert rep_b == {k: v for k, v in rep.items() if k != "blocks"}, block
    finally:
        torch.use_deterministic_algorithms(was)


def test_the_median_and_the_p99_are_torchs_own():
    # torch.quantile refuses over 16,777,216 values (a scene's mesh has more faces): one sort gives both, the same values
    import denoise as DN
    g = torch.Generator().manual_seed(0)
    for n in (1, 2, 3, 100, 101, 1000, 4097):
        x = torch.rand(n, generator=g, dtype=torch.float64).to(DEV)
        assert DN.median_p99(x) == (float(torch.median(x)), float(torch.quantile(x, 0.99))), n
    assert DN.median_p99(torch.zeros(0, dtype=torch.float64, device=DEV)) == (0.0, 0.0)


def test_the_mesh_filter_keeps_the_meshs_types_and_says_what_it_did():
    """filter_mesh is what GenRecon's mesh step calls on its global mesh (float32 vertices, int32 faces, on the card)."""
    import denoise as DN
    _, Vn, F = _noisy_box(3)
    V = torch.as_tensor(Vn, device=DEV, dtype=torch.float32)
    Ft = torch.as_tensor(F, device=DEV, dtype=torch.int32)
    was = torch.are_deterministic_algorithms_enabled()
    out, rep, lines = DN.filter_mesh(V, Ft, 20, 10, 0.35, 100)
    assert torch.are_deterministic_algorithms_enabled() == was            # put back
    assert out.dtype == torch.float32 and out.device == V.device and out.shape == V.shape and out.is_contiguous()
    torch.use_deterministic_algorithms(True)
    try:
        ref, _ = DN.bilateral_denoise(V.double(), Ft, 20, 10, 0.35)
    finally:
        torch.use_deterministic_algorithms(was)
    assert torch.equal(out, ref.float())                                  # float64 inside, the mesh's own type out
    assert rep["faces"] == len(F) == 768 and rep["blocks"] == 8
    assert rep["vertex_moved_mm_max"] >= rep["vertex_moved_mm_p99"] >= rep["vertex_moved_mm_median"] > 0
    assert len(lines) == 2 and "768 faces" in lines[0] and "in 8 block(s)" in lines[0]
    assert "normals turned median" in lines[1] and "vertices moved median" in lines[1] and "VRAM peak so far" in lines[1]
    # the stage log cuts a forwarded line at 300 characters: a scene's numbers are some 30 characters longer than these
    assert all(len("[STAC] edge filter: " + line) < 240 for line in lines)
    bad = V.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="not finite"):
        DN.filter_mesh(bad, Ft, 20, 10, 0.35, 100)
    assert torch.are_deterministic_algorithms_enabled() == was            # put back after a failure too


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


def test_a_piece_is_loose_when_it_floats_farther_than_its_own_size(monkeypatch):
    """bufferStop 2026-11-12: "within 2 cm of a kept piece" dropped the back wall (7.5 m² at 58 cm, 13.7 m² in all,
    13.6 of them on the cloud). The bar is the piece's own size: the wall stays, the crumbs go."""
    parts = [_square(-1.5, -1.5, 0.0, 3.0, n=60)]             # 0 the floor: the largest
    parts.append(_square(2.0, 0.2, 0.6, 0.3))                 # 1 a 30 cm piece 78 cm from everything: goes
    parts.append(_square(-0.75, -0.75, 0.58, 1.5, n=30))      # 2 a 1.5 m sheet 58 cm over the floor (a wall whose foot
    #                                                             nobody saw): stays
    for k in range(6):                                        # 3-8 a chain of 5 cm crumbs off that sheet, each 3 cm
        parts.append(_square(0.78 + 0.08 * k, 0.0, 0.58, 0.05))   # from the last: stays, to convergence
    parts.append(_square(-0.9, -0.9, 0.10, 0.02))             # 9 a 2 cm crumb 10 cm over the floor: goes
    parts.append(_square(3.0, -3.0, 0.02, 0.05))              # 10 a small piece far away ON the floor: stays
    parts.append(_square(-1.25, 4.3, 0.5, 2.5, n=40))         # 11 a 2.5 m sheet 2.84 m away (less than its size): stays
    parts.append(_square(6.0, -1.0, 1.0, 2.0, n=40))          # 12 a 2 m sheet 4.6 m away (more than its size): goes
    Vs, Fs, nv = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + nv)
        nv += len(V)
    V = torch.as_tensor(np.concatenate(Vs), device=DEV, dtype=torch.float64)
    F = torch.as_tensor(np.concatenate(Fs), device=DEV)
    coarse, cells = [], DL._cells
    monkeypatch.setattr(DL, "_cells", lambda P, key, h: (coarse.append(h), cells(P, key, h))[1])
    keep, rep = DL.loose_faces(V, F, z_floor=0.0, floor=0.05)
    starts = np.cumsum([0] + [len(F) for _, F in parts])
    kept = [bool(keep[starts[i]:starts[i + 1]].all()) for i in range(len(parts))]
    gone = [bool((~keep[starts[i]:starts[i + 1]]).all()) for i in range(len(parts))]
    assert [i for i in range(len(parts)) if gone[i]] == [1, 9, 12]
    assert [i for i in range(len(parts)) if kept[i]] == [0, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    assert rep["pieces"] == 13 and rep["kept_pieces"] == 10 and rep["dropped_pieces"] == 3
    assert rep["rounds"] >= 7                                 # the sheet, then the chain one crumb a round
    assert abs(rep["largest_dropped_m2"] - 4.0) < 1e-6 and abs(rep["largest_dropped_size_m"] - 2.0 * 2 ** 0.5) < 1e-6
    assert coarse                                             # the far sheets were measured on the ladder's cells
    keep2, rep2 = DL.loose_faces(V, F, z_floor=0.0, floor=0.05)
    assert torch.equal(keep, keep2) and rep == rep2           # the same every time


def test_the_runner_the_worker_and_the_vendor_carry_the_shape_from_the_cloud():
    sh = (_TOOLS / "run_cloud.sh").read_text()
    assert "--shape_from_cloud --shape_refine_t ${GENRECON_SHAPE_REFINE_T:?" in sh
    assert "drop_loose.py $O/to_glb_inputs.pt $W $O/to_glb_inputs_clean.pt" in sh
    assert "denoise.py $O/to_glb_inputs_clean.pt $O/to_glb_inputs_smooth.pt" in sh and "INPUTS=$O/to_glb_inputs_smooth.pt" in sh
    assert 'BAKE_FLAGS="--skip_remesh --smooth_iters ${GENRECON_SMOOTH_ITERS:?' in sh
    assert sh.count("--inputs $INPUTS") == 2 and "bake_chunks_parallel.py $INPUTS" in sh
    patch = (_TOOLS / "genrecon_stac.patch").read_text()
    assert "shape_latents_override" in patch and "t_start" in patch and "--shape_from_cloud" in patch
    sys.path.insert(0, str(_TOOLS.parents[1]))
    from workers import genrecon_worker as GW
    assert GW._number({"shape_refine_t": 0.0}, "shape_refine_t", float, 0.0, 1.0) == "0.0"
    assert GW._number({"smooth_iters": 30}, "smooth_iters", int, 0, 1000) == "30"
    assert "cam_genrecon.npy" in patch and "cam_of_point.npy" in patch
    with pytest.raises(RuntimeError, match="genrecon.shape_refine_t is missing"):
        GW._number({}, "shape_refine_t", float, 0.0, 1.0)
    with pytest.raises(RuntimeError, match="must be a int"):
        GW._number({"smooth_iters": 2.5}, "smooth_iters", int, 0, 1000)


def test_the_edge_filter_is_wired_into_both_mesh_steps():
    """USER 2026-11-12 (night, the generated mesh): "Lo dejamos el filtro igual porque podrían haber mallas que si lo
    requieran. Implementamos el filtro y hacemos las mallas de todas las sesiones con el filtro" — genrecon.edge_filter:
    the generated shape filtered inside the vendor's mesh step after its global remesh (both passes), the shape from
    the cloud on its decoded mesh; the four values read before the reconstruction."""
    sh = (_TOOLS / "run_cloud.sh").read_text()
    assert "EDGE=${GENRECON_EDGE_FILTER:?genrecon.edge_filter (the worker sets it)}" in sh
    for key in ("NORMAL_ITERS", "VERTEX_ITERS", "SIGMA_R", "BLOCK_FACES"):
        assert f"${{GENRECON_DENOISE_{key}:?genrecon.denoise_{key.lower()} (the worker sets it)}}" in sh
        assert sh.index(f"GENRECON_DENOISE_{key}:?") < sh.index("resume.py check")     # before anything is run
    assert 'elif [ "$EDGE" = "1" ]; then' in sh and 'BAKE_FLAGS="--edge_filter $EDGE_ARGS"' in sh
    assert "$O/to_glb_inputs_smooth.pt $EDGE_ARGS)" in sh
    passes = [line for line in sh.splitlines() if "chunked_to_glb.py --inputs $INPUTS" in line]
    assert len(passes) == 2 and all("$BAKE_FLAGS)" in line for line in passes)          # the dump and the merge alike
    patch = (_TOOLS / "genrecon_stac.patch").read_text()
    for said in ("--edge_filter", "_stac_edge_filter", "_edge_filter.filter_mesh(", "[STAC] edge filter:",
                 "edge_filter=args.edge_filter"):
        assert said in patch, said
    vendor = _TOOLS.parents[2] / "vendor" / "GenRecon"
    if (vendor / "inference" / "chunked_glb.py").is_file():                              # the clone itself, when it is here
        src = (vendor / "inference" / "chunked_glb.py").read_text()
        assert (src.index("gv, gf = mesh.read()") < src.index("_edge_filter.filter_mesh(")
                < src.index("Step 2: Spatial partition"))                               # after the remesh, before the chunks
        assert '"server" / "tools" / "genrecon" / "denoise.py"' in src
        assert "edge_filter=args.edge_filter" in (vendor / "chunked_to_glb.py").read_text()
