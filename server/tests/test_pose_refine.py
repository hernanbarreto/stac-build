"""
E-full pose_refine tests: synthetic corridor observed by a line of cameras;
per-frame SE3 noise creates the measured defect (covisible frames place the
same surface at different poses = layering). The pose graph must recover the
noise (up to gauge), the gate must hold on already-consistent data, and the
end-to-end run() must move PLY points + poses together.
"""
import numpy as np
import pytest

from reconstruction.pose_refine import (se3_exp, se3_log, robust_rigid,
                                        rasterize_frame, gather_matches,
                                        grazing_mask, normal_grid,
                                        solve_pose_graph, edge_errors, run)

H, W = 96, 128
FX = FY = 90.0
CX, CY = W / 2.0, H / 2.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


def _corridor_points(rng, n=60000, span=20.0):
    """Floor y=0 + wall x=2 along z (grazing), PLUS frontal facades crossing
    the walk every 4 m (the realistic near-frontal content pixel matching
    feeds on — a pure-grazing corridor is pathological by construction)."""
    floor = np.column_stack([rng.uniform(-2, 2, n // 3),
                             np.zeros(n // 3),
                             rng.uniform(0, span, n // 3)])
    wall = np.column_stack([np.full(n // 3, 2.0),
                            rng.uniform(0.0, 2.5, n // 3),
                            rng.uniform(0, span, n // 3)])
    slabs = []
    n_slab = (n // 3) // 5
    for zc in np.arange(4.0, span + 1e-6, 4.0):
        # facades GROW with distance (like real scenes): far ones peek around
        # the near ones instead of being fully occluded
        hx = 0.45 * zc
        hy = 0.5 + 0.35 * zc
        slabs.append(np.column_stack([rng.uniform(-hx, hx * 1.1, n_slab),
                                      rng.uniform(0.0, hy, n_slab),
                                      np.full(n_slab, zc)]))
    return np.vstack([floor, wall] + slabs)


def _camera_line(n_frames, span=16.0):
    """c2w poses walking along +z at y=1, looking forward."""
    poses = []
    for k in range(n_frames):
        M = np.eye(4)
        M[:3, 3] = [0.0, 1.0, k * span / max(n_frames - 1, 1)]
        # camera looks along +z with y down-ish: use axes x->x, y->-y, z->+z
        M[:3, :3] = np.diag([1.0, -1.0, 1.0])
        poses.append(M)
    return poses


def _observe(structure, c2w):
    """Pixels + world points the camera actually sees (in front, in bounds)."""
    w2c = np.linalg.inv(c2w)
    X = structure @ w2c[:3, :3].T + w2c[:3, 3]
    z = X[:, 2]
    m = z > 0.5
    u = np.round(X[m, 0] / z[m] * FX + CX).astype(int)
    v = np.round(X[m, 1] / z[m] * FY + CY).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return structure[m][inb], v[inb], u[inb], z[m][inb]


def _noisy_frames(seed=7, n_frames=24, noise_t=0.06, noise_r=0.4):
    """Each frame observes the TRUE structure, then its whole point set (and
    pose) is displaced by a per-frame rigid noise N_k — the omega defect."""
    rng = np.random.default_rng(seed)
    structure = _corridor_points(rng)
    poses_true = _camera_line(n_frames)
    frames = []
    for k, M in enumerate(poses_true):
        P, rows, cols, z = _observe(structure, M)
        if k == 0:
            N = np.eye(4)                          # gauge frame stays put
        else:
            w = rng.normal(0, np.radians(noise_r), 3)
            t = rng.normal(0, noise_t, 3)
            N = se3_exp(np.concatenate([w, t]))
        Pn = P @ N[:3, :3].T + N[:3, 3]
        frames.append({"P": Pn, "rows": rows, "cols": cols,
                       "pose": N @ M, "noise": N})
    return frames


def _build_edges(frames, pair_window=6, samples=1500, seed=3):
    rng = np.random.default_rng(seed)
    rasters, worlds, w2cs, samp, trusts, norms = [], [], [], [], [], []
    for f in frames:
        w2c = np.linalg.inv(f["pose"])
        Xc = f["P"] @ w2c[:3, :3].T + w2c[:3, 3]
        r, Wg = rasterize_frame(f["P"], Xc[:, 2], f["rows"], f["cols"], H, W)
        rasters.append(r)
        trusts.append(grazing_mask(r))
        norms.append(normal_grid(Wg, r > 1e-6))
        worlds.append(Wg)
        w2cs.append(w2c)
        take = min(samples, len(f["P"]))
        samp.append(f["P"][rng.choice(len(f["P"]), take, replace=False)])
    edges = []
    for i in range(len(frames)):
        for j in range(i + 1, min(i + pair_window + 1, len(frames))):
            e = gather_matches(samp[i], w2cs[j], K, rasters[j], worlds[j],
                               norms[j], min_matches=80, trust_j=trusts[j],
                               seed=seed + i * 100 + j)
            if e is not None:
                edges.append((i, j) + e)         # (i, j, P, Q, N, W)
    return edges


def test_gather_matches_depth_weights():
    """Far matches enter beyond near_ref but with (near_ref/z)^2 weight; a
    max_depth of 8 excludes them entirely (the F1 chimney failure mode)."""
    frames = _noisy_frames(seed=21, noise_t=0.0, noise_r=0.0, n_frames=4)
    f0, f1 = frames[0], frames[1]
    w2c = np.linalg.inv(f1["pose"])
    Xc = f1["P"] @ w2c[:3, :3].T + w2c[:3, 3]
    r, Wg = rasterize_frame(f1["P"], Xc[:, 2], f1["rows"], f1["cols"], H, W)
    tr = grazing_mask(r)
    nr = normal_grid(Wg, r > 1e-6)
    rng = np.random.default_rng(4)
    samp = f0["P"][rng.choice(len(f0["P"]), 6000, replace=False)]
    wide = gather_matches(samp, w2c, K, r, Wg, nr, min_matches=50,
                          max_depth=30.0, near_ref=8.0, trust_j=tr,
                          max_matches=100000)
    near = gather_matches(samp, w2c, K, r, Wg, nr, min_matches=50,
                          max_depth=8.0, near_ref=8.0, trust_j=tr,
                          max_matches=100000)
    assert wide is not None and near is not None
    P, Q, N, Wt = wide
    zj = (Q @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    far = zj > 8.5
    assert far.any(), "wide gather must include far matches"
    assert len(P) > len(near[0])
    # weights: 1.0 in the near field, (8/z)^2 beyond
    assert np.allclose(Wt[zj <= 8.0], 1.0)
    exp = (8.0 / zj[far]) ** 2
    assert np.allclose(Wt[far], exp, rtol=1e-6)


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        xi = np.concatenate([rng.normal(0, 0.2, 3), rng.normal(0, 0.5, 3)])
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)


def test_robust_rigid_recovers_known_transform_with_outliers():
    rng = np.random.default_rng(2)
    src = rng.uniform(-3, 3, (2000, 3))
    T = se3_exp(np.array([0.01, -0.02, 0.015, 0.08, -0.05, 0.11]))
    dst = src @ T[:3, :3].T + T[:3, 3]
    dst[:200] += rng.uniform(-2, 2, (200, 3))       # 10% outliers
    fit, med, n = robust_rigid(src, dst)
    assert np.allclose(fit, T, atol=2e-3), fit - T


def test_edges_expose_the_layering():
    """With per-frame noise the measured edge errors are ~noise-sized; on
    clean frames they are ~zero."""
    noisy = _build_edges(_noisy_frames(noise_t=0.06))
    clean = _build_edges(_noisy_frames(noise_t=0.0, noise_r=0.0))
    n_err = edge_errors(noisy, [np.eye(4)] * 24)
    c_err = edge_errors(clean, [np.eye(4)] * 24)
    assert np.median(n_err) > 0.03                  # layering visible
    assert np.median(c_err) < 0.005                 # clean is clean


def test_solver_exact_on_perfect_correspondences():
    """The joint point-to-plane solve, isolated from association: with exact
    same-physical-point matches it must invert the injected noise to mm."""
    rng = np.random.default_rng(9)

    def sample_surface(n):
        kind = rng.integers(0, 3, n)
        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        m = kind == 0
        pts[m] = np.column_stack([rng.uniform(-2, 2, m.sum()),
                                  np.zeros(m.sum()),
                                  rng.uniform(0, 16, m.sum())])
        nrm[m] = [0, 1, 0]
        m = kind == 1
        pts[m] = np.column_stack([np.full(m.sum(), 2.0),
                                  rng.uniform(0, 2.5, m.sum()),
                                  rng.uniform(0, 16, m.sum())])
        nrm[m] = [1, 0, 0]
        m = kind == 2
        zc = rng.choice([4.0, 8.0, 12.0, 16.0], m.sum())
        pts[m] = np.column_stack([rng.uniform(-1.5, 1.8, m.sum()),
                                  rng.uniform(0, 2.2, m.sum()), zc])
        nrm[m] = [0, 0, 1]
        return pts, nrm

    n_fr = 24
    noises = [np.eye(4)] + [se3_exp(np.concatenate(
        [rng.normal(0, np.radians(0.4), 3), rng.normal(0, 0.06, 3)]))
        for _ in range(n_fr - 1)]
    edges = []
    for i in range(n_fr):
        for j in range(i + 1, min(i + 7, n_fr)):
            s, nr = sample_surface(400)
            P = s @ noises[i][:3, :3].T + noises[i][:3, 3]
            Q = s @ noises[j][:3, :3].T + noises[j][:3, 3]
            Nn = nr @ noises[j][:3, :3].T
            edges.append((i, j, P, Q, Nn))
    # near-zero priors: this test isolates the SOLVER math (production priors
    # deliberately trade a little exactness for clean-data stability)
    Cs = solve_pose_graph(n_fr, edges, odo_weight=0.05, leash_weight=0.005)
    post = edge_errors(edges, Cs)
    assert np.median(post) < 0.003, np.median(post)
    res = [np.linalg.norm((Cs[k] @ noises[k])[:3, 3]) for k in range(n_fr)]
    assert np.median(res) < 0.02, np.median(res)


def test_pose_graph_improves_raster_associated_noise():
    """Through the real raster association (pixel quantization, grazing
    filter, normal estimation) a single measure+solve round must already
    improve the held-out disagreement by >=20% — the outer ICP loop in run()
    then compounds it."""
    frames = _noisy_frames(noise_t=0.06, noise_r=0.4)
    edges = _build_edges(frames)
    assert len(edges) > 60
    rng = np.random.default_rng(5)
    order = rng.permutation(len(edges))
    hold = [edges[t] for t in order[:len(edges) // 5]]
    fit = [edges[t] for t in order[len(edges) // 5:]]
    Cs = solve_pose_graph(24, fit, odo_weight=5.0, leash_weight=0.1)
    h0 = np.median(edge_errors(hold, [np.eye(4)] * 24))
    h1 = np.median(edge_errors(hold, Cs))
    assert h1 < h0 * 0.8, (h0, h1)


def test_pose_graph_near_identity_on_clean_data():
    """Clean frames: whatever the solver does must stay within association
    noise — no invented corrections beyond ~2 cm with production weights."""
    frames = _noisy_frames(noise_t=0.0, noise_r=0.0)
    edges = _build_edges(frames)
    Cs = solve_pose_graph(24, edges, odo_weight=30.0, leash_weight=0.5)
    shifts = [np.linalg.norm(C[:3, 3]) for C in Cs]
    assert max(shifts) < 0.02, max(shifts)


class TestRunEndToEnd:
    def _write_scene(self, out, frames):
        from plyfile import PlyData, PlyElement
        allP, allF, allR, allC = [], [], [], []
        for k, f in enumerate(frames):
            allP.append(f["P"])
            allF.append(np.full(len(f["P"]), k * 10, np.int64))  # real nums
            allR.append(f["rows"].astype(np.int16))
            allC.append(f["cols"].astype(np.int16))
        P = np.vstack(allP)
        v = np.zeros(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        v["x"], v["y"], v["z"] = P[:, 0], P[:, 1], P[:, 2]
        PlyData([PlyElement.describe(v, "vertex")], text=False).write(
            str(out / "chunk_000.ply"))
        np.savez(out / "chunk_000_origins.npz",
                 frame_global=np.concatenate(allF),
                 pixel_row=np.concatenate(allR),
                 pixel_col=np.concatenate(allC),
                 confidence=np.ones(len(P), np.float32),
                 scaled_resolution=np.array([H, W], np.int32))
        (out / "camera_frames.txt").write_text(
            " ".join(str(k * 10) for k in range(len(frames))))
        (out / "camera_poses.txt").write_text("\n".join(
            " ".join(f"{x:.9g}" for x in f["pose"].reshape(-1))
            for f in frames) + "\n")
        (out / "intrinsic.txt").write_text("\n".join(
            f"{FX} {FY} {CX} {CY}" for _ in frames) + "\n")

    def test_run_refines_noisy_scene(self, tmp_path):
        import json
        frames = _noisy_frames(seed=11, noise_t=0.06, noise_r=0.4)
        self._write_scene(tmp_path, frames)
        moved = run(tmp_path, pair_window=6, samples_per_frame=1500,
                    odo_weight=2.0, leash_weight=0.1, min_gain=0.10,
                    outer_iters=3)
        assert moved > 0
        rep = json.loads((tmp_path / "pose_refine_report.json").read_text())
        assert rep["accepted"]
        assert rep["holdout_after_m"] < rep["holdout_before_m"] * 0.65
        assert (tmp_path / "camera_poses.txt.preposerefine").exists()
        # PLY points moved coherently: re-measuring edges on the written
        # artifacts must show the layering collapsed
        from plyfile import PlyData
        v = np.array(PlyData.read(str(tmp_path / "chunk_000.ply"))["vertex"].data)
        z = np.load(tmp_path / "chunk_000_origins.npz")
        P = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        poses = [np.array([float(x) for x in ln.split()]).reshape(4, 4)
                 for ln in (tmp_path / "camera_poses.txt").read_text().splitlines()]
        new_frames = []
        for k in range(len(frames)):
            m = z["frame_global"] == k * 10
            new_frames.append({"P": P[m], "rows": z["pixel_row"][m].astype(int),
                               "cols": z["pixel_col"][m].astype(int),
                               "pose": poses[k]})
        errs = edge_errors(_build_edges(new_frames),
                           [np.eye(4)] * len(new_frames))
        before = edge_errors(_build_edges(frames), [np.eye(4)] * len(frames))
        assert np.median(errs) < np.median(before) * 0.65

    def test_run_gate_holds_on_clean_scene(self, tmp_path):
        import json
        frames = _noisy_frames(seed=12, noise_t=0.0, noise_r=0.0)
        self._write_scene(tmp_path, frames)
        poses_before = (tmp_path / "camera_poses.txt").read_text()
        moved = run(tmp_path, pair_window=6, samples_per_frame=1500,
                    odo_weight=5.0, leash_weight=0.1, min_gain=0.10)
        rep = json.loads((tmp_path / "pose_refine_report.json").read_text())
        if not rep["accepted"]:
            assert moved == 0
            assert (tmp_path / "camera_poses.txt").read_text() == poses_before
        else:
            # if the gate passed on clean data the corrections must be tiny
            assert rep["correction_m"]["max"] < 0.01
