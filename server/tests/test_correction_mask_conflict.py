"""The mask filter's fourth rule (USER 2026-09-29: "no debes probar los puntos contra
su propia máscara, son contra el resto de las máscaras con las vistas de ellos"): a
point of object X that, seen unoccluded from a keyframe of ANOTHER visit, lands inside
object Y's mask and never inside X's own, leaves. Masklets fused into one object never
conflict with each other; the point's own visit never votes."""
from __future__ import annotations

import numpy as np

from correction import visit_drift as VD


class _Vis:
    """A duck-typed Visibility: two keyframes, pinhole 100 px, masks at trace resolution."""
    Ht, Wt = 100, 100
    Hm, Wm = 100, 100
    min_depth = 0.05
    tol = 0.05

    def __init__(self, poses, K, masks, zbufs):
        self.poses, self.K, self._masks, self._z = poses, K, masks, zbufs

    def masks_of(self, oid, visit):
        a, b = visit
        return [(kf, m) for (o, kf), m in self._masks.items() if o == oid and a <= kf <= b]

    def zbuf(self, kf):
        return self._z[kf]


def _cam(pos):
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def _scene():
    """X: a small object at the origin (kf 0-1 see it); Y: an object 1 m away seen from
    kf 5-6, whose camera also sees X's region. One flyer of X floats in front of Y."""
    K = {kf: (100.0, 100.0, 50.0, 50.0) for kf in (0, 1, 5, 6)}
    poses = {0: _cam([0, 0, -3]), 1: _cam([0.1, 0, -3]), 5: _cam([0, 0, -3]), 6: _cam([0.1, 0, -3])}
    rng = np.random.default_rng(0)
    x_pts = rng.uniform(-0.05, 0.05, (10, 3))                  # X around the origin, z ≈ 0
    flyer = np.array([[0.8, 0.0, 0.0]])                        # X's flyer, sitting where Y is
    y_pts = np.array([[0.8, 0.0, 0.0]]) + rng.uniform(-0.05, 0.05, (10, 3))
    xyz = np.vstack([x_pts, flyer, y_pts]).astype(np.float64)
    ks = np.array([0] * 5 + [1] * 5 + [1] + [5] * 5 + [6] * 5)   # birth keyframes
    # masks: X's mask in kf 0-1 around the projection of the origin (u,v ≈ 50,50);
    # Y's mask in kf 5-6 around Y (u ≈ 50 + 100*0.8/3 ≈ 77)
    def disk(u0, v0, rad):
        v, u = np.mgrid[0:100, 0:100]
        return ((u - u0) ** 2 + (v - v0) ** 2 <= rad ** 2).astype(np.uint8)
    masks = {(0, 0): disk(50, 50, 4), (0, 1): disk(47, 50, 4), (1, 5): disk(77, 50, 4), (1, 6): disk(74, 50, 4)}
    zbufs = {kf: np.full((100, 100), 10.0) for kf in (0, 1, 5, 6)}   # nothing in front of anything
    return xyz, ks, masks, poses, K, zbufs


def _m(oid, label, visits):
    kfs = np.array([k for a, b in visits for k in range(a, b + 1)])
    return VD.Masklet(oid=oid, instance_id=oid + 1, label=label, keyframes=kfs, visits=visits)


def _masklets():
    return [_m(0, "x", [(0, 1)]), _m(1, "y", [(5, 6)])]


def test_a_point_inside_another_objects_mask_from_another_visit_leaves():
    xyz, ks, masks, poses, K, zbufs = _scene()
    vis = _Vis(poses, K, masks, zbufs)
    pts = {0: np.arange(0, 11), 1: np.arange(11, 21)}
    kill, rep = VD.cloud_filter_masklets(pts, _masklets(), ks, xyz, vis, min_points=1, min_visit_share=0.0,
                                         max_frames_per_visit=8, dilate_px=0, log=lambda m: None,
                                         group_points=None, group_roots={0: 1, 1: 2})
    assert kill[10] and not kill[:10].any() and not kill[11:].any(), kill
    assert any(d["reason"].startswith("inside another object") and d["points"] == 1 for d in rep.detail)


def test_masklets_of_one_fused_object_never_conflict():
    xyz, ks, masks, poses, K, zbufs = _scene()
    vis = _Vis(poses, K, masks, zbufs)
    pts = {0: np.arange(0, 11), 1: np.arange(11, 21)}
    kill, _ = VD.cloud_filter_masklets(pts, _masklets(), ks, xyz, vis, min_points=1, min_visit_share=0.0,
                                       max_frames_per_visit=8, dilate_px=0, log=lambda m: None,
                                       group_points=None, group_roots={0: 7, 1: 7})
    assert not kill.any()


def test_the_points_own_visit_never_votes():
    """Y's keyframes inside X's own visit range say nothing about X's points."""
    xyz, ks, masks, poses, K, zbufs = _scene()
    masks = {(0, 0): masks[(0, 0)], (0, 1): masks[(0, 1)], (1, 0): masks[(1, 5)], (1, 1): masks[(1, 6)]}
    ks = ks.copy()
    ks[11:16], ks[16:21] = 0, 1                     # Y's points born in the same visit as X's
    vis = _Vis(poses, K, masks, zbufs)
    ml = [_m(0, "x", [(0, 1)]), _m(1, "y", [(0, 1)])]
    pts = {0: np.arange(0, 11), 1: np.arange(11, 21)}
    kill, _ = VD.cloud_filter_masklets(pts, ml, ks, xyz, vis, min_points=1, min_visit_share=0.0,
                                       max_frames_per_visit=8, dilate_px=0, log=lambda m: None,
                                       group_points=None, group_roots={0: 1, 1: 2})
    assert not kill.any()
