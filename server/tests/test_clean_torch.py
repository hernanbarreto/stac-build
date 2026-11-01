"""The per-object cleaning's faces and voxel pass on the device (segmentation/clean_torch.py) against the
numpy path of record (segmentation.pipeline._faces_and_voxels_numpy): the same RANSAC faces (same seeded
hypotheses, same inliers), the same face id per point, the same 5 cm voxel records; and the bounded
neighbour queries (segmentation/knn_torch.py) against scipy's cKDTree."""
import os

import numpy as np
import pytest
from scipy.spatial import cKDTree

torch = pytest.importorskip("torch")
from segmentation import clean_torch as CT  # noqa: E402
from segmentation import knn_torch as KT  # noqa: E402
from segmentation import pipeline as SP  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")
PARAMS = {"ransac_tolerance": 0.01, "min_face_points": 100, "max_faces": 8, "planarity_threshold": 0.3}


def _get_param(path, default):
    return {"x.ransac_tolerance": PARAMS["ransac_tolerance"], "x.min_face_points": PARAMS["min_face_points"],
            "x.max_faces": PARAMS["max_faces"]}.get(path, default)


def _box_object(seed=0, n_face=6000, noise=0.003):
    """Three faces of a box (top and two sides), a pipe standing on the top (a curved residue), strays."""
    rng = np.random.default_rng(seed)
    top = np.column_stack([rng.uniform(0, 1.2, n_face), np.full(n_face, 0.8), rng.uniform(0, 0.6, n_face)])
    side_a = np.column_stack([np.zeros(n_face), rng.uniform(0, 0.8, n_face), rng.uniform(0, 0.6, n_face)])
    side_b = np.column_stack([rng.uniform(0, 1.2, n_face), rng.uniform(0, 0.8, n_face), np.zeros(n_face)])
    th = rng.uniform(0, 2 * np.pi, 1500)
    pipe = np.column_stack([0.6 + 0.08 * np.cos(th), rng.uniform(0.8, 1.4, 1500), 0.3 + 0.08 * np.sin(th)])
    strays = rng.uniform(-0.5, 1.7, (40, 3))
    P = np.concatenate([top, side_a, side_b, pipe, strays])
    P = P + rng.normal(0, noise, P.shape)
    return P


def _both(P):
    n = len(P)
    idx = np.arange(n)
    quiet = lambda *a, **k: None  # noqa: E731
    import builtins
    saved = builtins.print
    builtins.print = quiet                      # the reference prints per face
    try:
        a = SP._faces_and_voxels_numpy(P, P, idx, 7, _get_param, "x", PARAMS["planarity_threshold"], n, 0)
    finally:
        builtins.print = saved
    b = CT.faces_and_voxels_t(P, idx, 7, PARAMS, n, 0, DEV, log=quiet)
    return a, b


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_faces_face_ids_and_voxels_equal_the_numpy_recipe(seed):
    P = _box_object(seed)
    a, b = _both(P)
    assert np.array_equal(a[0], b[0])
    fa, fb = a[2], b[2]                                      # face_normals_summary [(normal, n), ...]
    assert len(fa) == len(fb) >= 3
    for (na, ca), (nb, cb) in zip(fa, fb):
        assert ca == cb
        assert abs(float(np.dot(na, nb))) > 1.0 - 1e-9
    for (na, da), (nb, db) in zip(a[3], b[3]):
        sgn = 1.0 if float(np.dot(na, nb)) > 0 else -1.0
        assert np.isclose(da, sgn * db, atol=1e-9)
    assert np.array_equal(a[4], b[4])                        # the face id per point
    va, vb = np.asarray(a[1], np.float64), np.asarray(b[1], np.float64)
    assert va.shape == vb.shape and len(va) > 50
    assert np.allclose(va[:, :3], vb[:, :3], atol=1e-9, rtol=0)
    assert np.allclose(np.abs(va[:, 3:]), np.abs(vb[:, 3:]), atol=1e-9, rtol=0)


def test_bounded_knn_and_radius_pairs_equal_the_kd_tree():
    rng = np.random.default_rng(3)
    P = np.column_stack([rng.uniform(-2, 2, 40000), rng.normal(0, 0.01, 40000), rng.uniform(-1, 1, 40000)])
    Q = np.column_stack([rng.uniform(-2, 2, 30000), rng.normal(0, 0.03, 30000), rng.uniform(-1, 1, 30000)])
    r, k = 0.03, 3
    d_ref, i_ref = cKDTree(P).query(Q, k=k, distance_upper_bound=r)
    D, I = KT.knn_within_radius_numpy(P, Q, k, r, DEV)
    miss, miss_ref = ~np.isfinite(D), ~np.isfinite(d_ref)
    assert np.array_equal(miss, miss_ref) and miss.sum() < miss.size
    assert np.allclose(D[~miss], d_ref[~miss_ref], atol=1e-12, rtol=0)
    assert np.array_equal(I[~miss], i_ref[~miss_ref])
    cnt = np.zeros(len(Q), np.int64)
    Pt, Qt = torch.as_tensor(P, device=DEV), torch.as_tensor(Q, device=DEV)
    for q, p, d2 in KT.radius_pairs(Pt, Qt, r):
        cnt += np.bincount(q.cpu().numpy(), minlength=len(Q))
    assert np.array_equal(cnt, cKDTree(P).query_ball_point(Q, r, return_length=True))
