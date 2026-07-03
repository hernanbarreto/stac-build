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
