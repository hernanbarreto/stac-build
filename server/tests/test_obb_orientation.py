"""OBB yaw by the object's own VERTICAL RANSAC planes (USER 2026-09-30: "muchas veces
queda cruzado … coplanar con el plano dominante, el menor vacío posible, la cara
inferior paralela a y = 0")."""
import numpy as np

from segmentation.pipeline import _compute_obb


def _yaw(obb):
    R = np.asarray(obb["rotation"])
    return np.degrees(np.arctan2(R[2, 0], R[0, 0])) % 90.0


def _table(yaw_deg, rng):
    """A 1.6 x 0.8 m table top (the face with MOST points, horizontal) on a 0.7 m high
    front panel and two side panels, rotated by yaw_deg about +Y."""
    top = np.stack([rng.uniform(-0.8, 0.8, 6000), np.full(6000, 0.75), rng.uniform(-0.4, 0.4, 6000)], 1)
    front = np.stack([rng.uniform(-0.8, 0.8, 2500), rng.uniform(0, 0.75, 2500), np.full(2500, 0.4)], 1)
    left = np.stack([np.full(1200, -0.8), rng.uniform(0, 0.75, 1200), rng.uniform(-0.4, 0.4, 1200)], 1)
    right = left * [-1, 1, 1]
    P = np.concatenate([top, front, left, right])
    P = P + rng.normal(0, 0.004, P.shape)
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    return P @ R.T + [3.0, 0.0, -2.0]


def test_a_table_is_aligned_to_its_own_sides_not_to_the_world_axes():
    rng = np.random.default_rng(0)
    for yaw in (0.0, 17.0, 30.0, 63.0):
        obb = _compute_obb(_table(yaw, rng))
        err = min(abs(_yaw(obb) - yaw % 90.0), 90.0 - abs(_yaw(obb) - yaw % 90.0))
        assert err < 1.5, (yaw, _yaw(obb))
        he = sorted(obb["half_extents"][::2])
        assert abs(he[1] - 0.8) < 0.05 and abs(he[0] - 0.4) < 0.05      # no empty corners


def test_the_bottom_face_stays_parallel_to_the_floor():
    rng = np.random.default_rng(1)
    R = np.asarray(_compute_obb(_table(30.0, rng))["rotation"])
    assert np.allclose(R[1], [0, 1, 0]) and np.allclose(R[:, 1], [0, 1, 0])


def test_a_face_normals_hint_that_is_horizontal_does_not_decide():
    """The old rule took the face with MOST points; a table top (normal +Y) then left the
    box on the world axes — crossed."""
    rng = np.random.default_rng(2)
    obb = _compute_obb(_table(30.0, rng), face_normals=[(np.array([0.0, 1.0, 0.0]), 6000)])
    err = min(abs(_yaw(obb) - 30.0), 90.0 - abs(_yaw(obb) - 30.0))
    assert err < 1.5, _yaw(obb)
