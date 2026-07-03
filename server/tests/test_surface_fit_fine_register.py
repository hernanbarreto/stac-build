"""
Stage-0 fine-registration tests: two synthetic chunks of the same room (floor
+ two walls) with a KNOWN rigid bias between them. The plane-constrained
correction must recover the bias and bring the residual plane separation
under the TSDF-truncation acceptance.
"""
import numpy as np

from reconstruction.surface_fit.fine_register import (extract_chunk_planes,
                                                      match_planes,
                                                      register_chunks)

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
