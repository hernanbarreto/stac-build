"""
Stage-0 fine-registration tests: two synthetic chunks of the same room (floor
+ two walls) with a KNOWN rigid bias between them. The plane-constrained
correction must recover the bias and bring the residual plane separation
under the TSDF-truncation acceptance.
"""
import numpy as np

from reconstruction.surface_fit.fine_register import (extract_chunk_planes,
                                                      ground_patches,
                                                      match_planes,
                                                      register_chunks,
                                                      solve_joint)

SIGMA = 0.002


def make_room_chunk(n_per_surf=40_000, sigma=SIGMA, seed=0, span=6.0):
    """Floor (z=0) + wall (y=0) + wall (x=0), each span×span metres."""
    rng = np.random.default_rng(seed)
    f = np.column_stack([rng.uniform(0, span, n_per_surf),
                         rng.uniform(0, span, n_per_surf),
                         rng.normal(0, sigma, n_per_surf)])
    w1 = np.column_stack([rng.uniform(0, span, n_per_surf),
                          rng.normal(0, sigma, n_per_surf),
                          rng.uniform(0, 3, n_per_surf)])
    w2 = np.column_stack([rng.normal(0, sigma, n_per_surf),
                          rng.uniform(0, span, n_per_surf),
                          rng.uniform(0, 3, n_per_surf)])
    return np.vstack([f, w1, w2])


def rigid(rx=0.0, ry=0.0, rz=0.0, t=(0, 0, 0)):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
    T[:3, 3] = t
    return T


class TestPlaneExtraction:
    def test_three_room_planes_found(self):
        planes = extract_chunk_planes(make_room_chunk(), max_planes=6)
        assert len(planes) >= 3
        normals = np.abs(np.array([p.normal for p in planes[:3]]))
        # the three dominant planes are the room's axis-aligned surfaces
        assert sorted(np.argmax(normals, axis=1).tolist()) == [0, 1, 2]

    def test_matching_finds_shared_surfaces(self):
        pa = extract_chunk_planes(make_room_chunk(seed=1), seed=1)
        pb = extract_chunk_planes(make_room_chunk(seed=2), seed=2)
        assert len(match_planes(pa, pb)) >= 3


class TestRegistration:
    def test_known_bias_recovered(self):
        """Chunk B biased by 6 mm translation + 0.05° yaw: registration must
        cancel it to sub-mm and pass the δ acceptance."""
        A = make_room_chunk(seed=3)
        bias = rigid(rz=np.deg2rad(0.05), t=(0.004, -0.003, 0.006))
        B = make_room_chunk(seed=4) @ bias[:3, :3].T + bias[:3, 3]
        corr, report = register_chunks({0: A, 1: B}, accept_sep_m=0.024)
        assert report.accepted
        # applying the correction to B must undo the bias: corr ≈ bias⁻¹
        residual = corr[1] @ bias
        assert np.linalg.norm(residual[:3, 3]) < 0.0015          # < 1.5 mm
        assert abs(np.trace(residual[:3, :3]) - 3.0) < 1e-5      # ~no rotation
        assert max(report.sep_after_m.values()) < 0.005

    def test_oversized_bias_refused(self):
        """A 6 cm shift with a 4 cm correction cap is a BA failure, not
        residual bias — the stage must refuse the correction AND flag
        non-acceptance instead of faking alignment."""
        A = make_room_chunk(seed=5)
        bias = rigid(t=(0.06, 0.0, 0.0))
        B = make_room_chunk(seed=6) @ bias[:3, :3].T + bias[:3, 3]
        corr, report = register_chunks({0: A, 1: B}, accept_sep_m=0.024,
                                       max_correction_m=0.04)
        assert np.allclose(corr[1], np.eye(4))
        assert not report.accepted

    def test_anchor_never_moves(self):
        A = make_room_chunk(seed=7)                       # largest → anchor
        B = make_room_chunk(n_per_surf=20_000, seed=8)
        corr, _ = register_chunks({0: A, 1: B})
        assert np.allclose(corr[0], np.eye(4))


class TestJointSolve:
    def test_many_chunks_inconsistent_drift(self):
        """Replica del modo de falla de test2: 6 chunks con offsets crecientes
        (deriva acumulada) — el solver CONJUNTO debe dejar todas las
        separaciones bajo δ, cosa que el secuencial no lograba."""
        chunks = {}
        for c in range(6):
            drift = rigid(rz=np.deg2rad(0.08 * c), t=(0.02 * c, -0.015 * c, 0.01 * c))
            pts = make_room_chunk(n_per_surf=25_000, seed=20 + c)
            chunks[c] = pts @ drift[:3, :3].T + drift[:3, 3]
        corr, report = register_chunks(chunks, accept_sep_m=0.024,
                                       max_correction_m=0.2)
        assert report.accepted, f"after: {report.sep_after_m}"
        assert max(report.sep_after_m.values()) < 0.010   # < 1 cm everywhere


def make_yard_chunk(x0=0.0, x1=6.0, ground_y=0.0, n_per_surf=30_000, seed=0,
                    cross_wall_x=None):
    """Y-UP outdoor-style chunk (the pipeline's frame after orient): ground at
    y=ground_y spanning x∈[x0,x1], long wall at z=0. Optionally a cross wall
    at x=cross_wall_x (the only surface that observes x-translation)."""
    rng = np.random.default_rng(seed)
    g = np.column_stack([rng.uniform(x0, x1, n_per_surf),
                         rng.normal(ground_y, SIGMA, n_per_surf),
                         rng.uniform(0, 6, n_per_surf)])
    w = np.column_stack([rng.uniform(x0, x1, n_per_surf),
                         rng.uniform(ground_y, ground_y + 3, n_per_surf),
                         rng.normal(0, SIGMA, n_per_surf)])
    parts = [g, w]
    if cross_wall_x is not None:
        parts.append(np.column_stack([
            rng.normal(cross_wall_x, SIGMA, n_per_surf),
            rng.uniform(ground_y, ground_y + 3, n_per_surf),
            rng.uniform(0, 3, n_per_surf)]))
    return np.vstack(parts)


class TestGroundDatum:
    def test_ground_patch_selection_excludes_roof(self):
        """Among horizontal planes, only those at the session ground level
        qualify — a flat roof must never pose as ground."""
        rng = np.random.default_rng(40)
        n = 30_000
        ground = np.column_stack([rng.uniform(0, 6, n),
                                  rng.normal(0, SIGMA, n),
                                  rng.uniform(0, 6, n)])
        roof = np.column_stack([rng.uniform(0, 6, n),
                                rng.normal(4.5, SIGMA, n),
                                rng.uniform(0, 6, n)])
        wall = np.column_stack([rng.uniform(0, 6, n),
                                rng.uniform(0, 4.5, n),
                                rng.normal(0, SIGMA, n)])
        planes = {0: extract_chunk_planes(np.vstack([ground, roof, wall]), seed=1),
                  1: extract_chunk_planes(np.vstack([ground + [8, 0, 0],
                                                     roof + [8, 0, 0]]), seed=2)}
        levels = ground_patches(planes)
        assert len(levels) == 1                                # roof is NOT a level
        assert set(levels[0]) == {0, 1}
        for pts in levels[0].values():
            assert abs(float(np.median(pts[:, 1]))) < 0.05     # the LOW plane

    def test_multi_level_stairs_preserved(self):
        """Not every scan shares one ground plane (stairs, split levels).
        Candidates must cluster into one datum PER LEVEL: a 1.5 m upper floor
        forms its own group and the real step is never flattened."""
        A = make_yard_chunk(0, 6, ground_y=0.0, seed=45)
        B = make_yard_chunk(9, 15, ground_y=0.0, seed=46)
        C = make_yard_chunk(18, 24, ground_y=1.5, seed=47)     # upper level
        D = make_yard_chunk(27, 33, ground_y=1.5, seed=48)
        planes = {k: extract_chunk_planes(p, seed=10 + k)
                  for k, p in enumerate([A, B, C, D])}
        levels = ground_patches(planes)
        assert len(levels) == 2
        units_by_level = [set(g) for g in levels]
        assert {0, 1} in units_by_level and {2, 3} in units_by_level

        # tie a +3 cm drift on B (level 0) and on D (level 1): each collapses
        # against ITS level; the 1.5 m step must survive intact
        planes = {0: extract_chunk_planes(A, seed=10),
                  1: extract_chunk_planes(B + [0, 0.03, 0], seed=11),
                  2: extract_chunk_planes(C, seed=12),
                  3: extract_chunk_planes(D + [0, 0.03, 0], seed=13)}
        corr = solve_joint(planes, anchor=0, ground_pts=ground_patches(planes))
        assert abs(corr[1][1, 3] + 0.03) < 0.006          # B onto anchored level 0
        # level 1 has no absolute anchor: C and D must MEET (relative gap
        # collapses) without either drifting away from the 1.5 m step
        gap = corr[3][1, 3] - corr[2][1, 3]
        assert abs(gap + 0.03) < 0.006
        assert abs(corr[2][1, 3]) < 0.025 and abs(corr[3][1, 3]) < 0.025

    def test_vertical_drift_between_disjoint_chunks(self):
        """test4 failure mode: two chunks that never overlap (no matched
        planes) but each sees the ground. Without the datum nothing couples
        them and a 3 cm vertical bias survives; the shared ground plane must
        collapse it."""
        A = make_yard_chunk(0, 6, ground_y=0.0, seed=41)
        B = make_yard_chunk(9, 15, ground_y=0.03, seed=42)   # 3 cm too high
        planes = {0: extract_chunk_planes(A, seed=3),
                  1: extract_chunk_planes(B, seed=4)}
        assert not match_planes(planes[0], planes[1])         # truly disjoint

        free = solve_joint(planes, anchor=0)                  # no datum
        assert abs(free[1][1, 3]) < 0.005                     # nothing to pull

        levels = ground_patches(planes)
        assert len(levels) == 1 and set(levels[0]) == {0, 1}  # 3 cm ≪ level gap
        tied = solve_joint(planes, anchor=0, ground_pts=levels)
        assert abs(tied[1][1, 3] + 0.03) < 0.005              # bias collapsed


class TestAnnealedSolve:
    def test_10cm_drift_recovered(self):
        """test4 regression: a ~10-16 cm inter-chunk drift was SEEN but not
        corrected (fixed 2 cm Huber treated it as an outlier; 12 cm match
        radius hid part of it). The annealed solve with the graduated capture
        radius must recover it to mm."""
        A = make_room_chunk(seed=50)
        bias = rigid(rz=np.deg2rad(0.3), t=(0.10, -0.02, 0.16))
        B = make_room_chunk(seed=51) @ bias[:3, :3].T + bias[:3, 3]
        corr, report = register_chunks({0: A, 1: B}, accept_sep_m=0.024,
                                       max_correction_m=0.3)
        assert report.accepted, f"after: {report.sep_after_m}"
        residual = corr[1] @ bias
        assert np.linalg.norm(residual[:3, 3]) < 0.005
        assert max(report.sep_after_m.values()) < 0.008


class TestGeometricLoop:
    def test_nonadjacent_overlap_closes_the_chain(self):
        """Walk-around-a-building topology: chunks 0-1-2 chained by surfaces
        BLIND to x-translation (ground + long z-wall); chunk 0 ALSO re-observes
        chunk 2's cross wall (the loop closure). An 8 cm x-drift on chunk 2 is
        invisible to the chain — only the non-adjacent 0↔2 constraint can
        close it."""
        bias = rigid(t=(0.08, 0.0, 0.0))

        def chunks(with_loop):
            A = make_yard_chunk(0, 6.5, seed=60,
                                cross_wall_x=16.0 if with_loop else None)
            B = make_yard_chunk(5.5, 11.5, seed=61)
            C = make_yard_chunk(10.5, 16, seed=62, cross_wall_x=16.0)
            return {0: A, 1: B, 2: C @ bias[:3, :3].T + bias[:3, 3]}

        corr, _ = register_chunks(chunks(with_loop=False), max_correction_m=0.3)
        assert abs(corr[2][0, 3]) < 0.02          # chain alone cannot see it

        corr, report = register_chunks(chunks(with_loop=True),
                                       max_correction_m=0.3)
        assert abs(corr[2][0, 3] + 0.08) < 0.01   # loop pair closed it
        assert report.sep_after_m.get("0-2", 1.0) < 0.02


class TestOnionMatching:
    def test_double_sheet_ground_never_cross_matches(self):
        """test4 failure mode: the intra-chunk depth 'onion' reconstructs the
        same ground as two parallel sheets ~9 cm apart. One-to-one matching
        must pair sheet-1↔sheet-1 and sheet-2↔sheet-2 (the 2 cm inter-chunk
        drift) and never the ~9 cm cross pairs — so the solve sees drift, not
        contradictions, and the metric reports drift, not the onion."""
        rng = np.random.default_rng(70)
        n = 25_000

        def sheeted_chunk(dy):
            g1 = np.column_stack([rng.uniform(0, 8, n),
                                  rng.normal(dy, SIGMA, n),
                                  rng.uniform(0, 8, n)])
            g2 = np.column_stack([rng.uniform(0, 8, n),
                                  rng.normal(dy + 0.09, SIGMA, n),   # onion sheet
                                  rng.uniform(0, 8, n)])
            wall = np.column_stack([rng.uniform(0, 8, n),
                                    rng.uniform(dy, dy + 3, n),
                                    rng.normal(0, SIGMA, n)])
            return np.vstack([g1, g2, wall])

        A, B = sheeted_chunk(0.0), sheeted_chunk(0.02)     # 2 cm true drift
        pa = extract_chunk_planes(A, seed=71)
        pb = extract_chunk_planes(B, seed=72)
        matches = match_planes(pa, pb)
        seps = sorted(s for _, _, s in matches)
        assert all(s < 0.035 for s in seps), seps          # no ~9 cm cross pair
        # each plane used at most once (one-to-one)
        assert len({i for i, _, _ in matches}) == len(matches)
        assert len({j for _, j, _ in matches}) == len(matches)

        corr, report = register_chunks({0: A, 1: B}, accept_sep_m=0.024)
        assert report.accepted, f"after: {report.sep_after_m}"
        assert abs(corr[1][1, 3] + 0.02) < 0.005           # drift recovered


class TestPiecewiseRun:
    def test_intra_chunk_drift_corrected(self, tmp_path):
        """Dos chunks donde el 2º DERIVA INTERNAMENTE (transformación que
        crece con el frame): una corrección rígida por chunk no puede alinear
        ambas puntas — la partición en piezas + interpolación por frame sí."""
        import numpy as np
        from plyfile import PlyData, PlyElement
        from reconstruction.surface_fit.fine_register import run as run_finereg

        rng = np.random.default_rng(31)
        out = tmp_path

        def write_chunk(cid, pts, frames):
            v = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            PlyData([PlyElement.describe(v, "vertex")], text=False).write(
                str(out / f"chunk_{cid:03d}.ply"))
            np.savez(out / f"chunk_{cid:03d}_origins.npz",
                     frame_global=frames.astype(np.int64))

        base = make_room_chunk(n_per_surf=20_000, seed=32)
        f0 = rng.integers(0, 10, len(base))
        write_chunk(0, base, f0)

        # chunk 1: same room re-observed, frames 10..29, drifting linearly
        # from 0 at frame 10 to 5 cm + 0.4° at frame 29
        pts1 = make_room_chunk(n_per_surf=20_000, seed=33)
        f1 = rng.integers(10, 30, len(pts1))
        drifted = pts1.copy()
        for fi in np.unique(f1):
            aa = (fi - 10) / 19.0
            T = rigid(rz=np.deg2rad(0.4 * aa), t=(0.05 * aa, -0.03 * aa, 0.02 * aa))
            m = f1 == fi
            drifted[m] = pts1[m] @ T[:3, :3].T + T[:3, 3]
        write_chunk(1, drifted, f1)

        frames_all = list(range(30))
        (out / "camera_frames.txt").write_text(" ".join(str(f) for f in frames_all))
        eye = " ".join(f"{x:.9g}" for x in np.eye(4).reshape(-1))
        (out / "camera_poses.txt").write_text("\n".join([eye] * 30) + "\n")

        n = run_finereg(out, accept_sep_m=0.024, max_correction_m=0.2,
                        pieces_per_chunk=3)
        assert n >= 1
        import json
        rep = json.loads((out / "fine_register_report.json").read_text())
        assert rep["accepted"], f"sep_after: {rep['sep_after_m']}"
        assert max(rep["sep_after_m"].values()) < 0.015
        assert (out / "camera_poses.txt.prefinereg").exists()
