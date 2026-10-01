"""The mask filter's fourth rule (USER 2026-09-29: "no debes probar los puntos contra
su propia máscara, son contra el resto de las máscaras con las vistas de ellos"),
REWRITTEN 2026-10-01 for edge definition (USER: "lo más importante es que los objetos
deben tener mucha definición, corte en los filos, las aristas"; rule 4 decides by the
MAJORITY of the views that saw the point):

  a point of object X leaves when, among the keyframes where X has a mask (its own
  visit included, its birth keyframe excluded) that SAW it — in frame, at a parallax of
  at least min_tri_deg from its birth ray, not occluded by that keyframe's measured
  depth (relative tolerance) — at least min_votes voted, it lies ON another object's
  surface inside that object's UNDILATED mask in at least min_inside_frac of them, and
  inside its own DILATED mask in less than min_inside_frac of them.

The views here are a TABLE (pixel and depth of every point in every keyframe), so each
clause is tested on its own; the projection through the session camera is tested on a
rendered scene in test_certify_mask_filter_camera.py.

Changed from the first version of this file (2026-09-29), on purpose:
  * the flyer now has to lie ON the other object's surface (its depth = the depth that
    keyframe measured there) — the old scene had "nothing in front of anything" (inf
    z-buffers) and accused the flyer from one view;
  * "the point's own visit never votes" became "the point's BIRTH keyframe never votes":
    the own visit's other keyframes vote (parallax inside one visit counts) — that is
    what makes "never inside its own" meaningful for single-visit objects.
"""
from __future__ import annotations

import numpy as np

from correction import visit_drift as VD

H = W = 10
TOL_REL = 0.05          # the declared loops.witness.occlusion_tol_rel, as a test value
MIN_TRI = 2.0           # the declared precision.refine.min_tri_deg, as a test value


class _TableVis:
    """A duck-typed Visibility whose projections are a table: ``pix[kf][i]`` = (row,
    col) of point i in keyframe kf (None = out of frame) and ``z[kf][i]`` its depth."""
    min_depth = 0.05
    tol = 0.15              # rule 1's absolute occlusion tolerance (unchanged rule)

    def __init__(self, xyz, centres, pix, z, zbufs, masks):
        self.xyz = np.asarray(xyz, np.float64)
        self.poses = np.tile(np.eye(4), (len(centres), 1, 1))
        self.poses[:, :3, 3] = np.asarray(centres, np.float64)
        self._pix, self._z, self._zb, self._masks = pix, z, zbufs, masks
        self._id = {tuple(np.round(p, 9)): i for i, p in enumerate(self.xyz)}

    def masks_of(self, oid, visit):
        a, b = visit
        return [(kf, m) for (o, kf), m in sorted(self._masks.items())
                if o == oid and a <= kf <= b]

    def oids_at(self, kf):
        return sorted({o for (o, k) in self._masks if k == kf})

    def zbuf(self, kf):
        return self._zb[kf]

    def project(self, P, kf):
        ids = [self._id[tuple(np.round(p, 9))] for p in np.asarray(P, np.float64)]
        ok = np.array([self._pix[kf][i] is not None for i in ids], bool)
        r = np.array([self._pix[kf][i][0] for i, o in zip(ids, ok) if o], np.int64)
        c = np.array([self._pix[kf][i][1] for i, o in zip(ids, ok) if o], np.int64)
        z = np.array([self._z[kf][i] for i, o in zip(ids, ok) if o], np.float64)
        return ok, r, c, z


def _box(r0, r1, c0, c1):
    m = np.zeros((H, W), np.uint8)
    m[r0:r1 + 1, c0:c1 + 1] = 1
    return m


def _m(oid, label, visits):
    kfs = np.array([k for a, b in visits for k in range(a, b + 1)])
    return VD.Masklet(oid=oid, instance_id=oid + 1, label=label, keyframes=kfs, visits=visits)


def _filter(vis, pts, ks, masklets, roots, dilate_px=0, min_votes=2, frac=0.5):
    return VD.cloud_filter_masklets(pts, masklets, np.asarray(ks), vis.xyz, vis, min_points=1,
                                    min_visit_share=0.0, max_frames_per_visit=8,
                                    dilate_px=dilate_px, occlusion_tol_rel=TOL_REL,
                                    min_votes=min_votes, min_inside_frac=frac,
                                    min_tri_deg=MIN_TRI, log=lambda m: None,
                                    group_points=None, group_roots=roots)


# ── the scene: point 0 belongs to X (oid 0), points 1-2 to Y (oid 1) ──────
# Cameras 1 m apart, the points ~3 m away (parallax ≈ 18° between neighbours);
# point 0 sits inside Y's 3-D extent (only overlapping objects can conflict).
# X's mask: columns 0-3; Y's mask: columns 6-9; the measured depth (z-buffer) is
# 3.0 everywhere (Y's surface), so a point at depth 3.0 in Y's columns lies ON Y.

CENTRES = [[-2.0, 0, 0], [-1.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]
XYZ = [[0.55, 0.0, 3.0], [0.5, 0.0, 3.0], [0.6, 0.0, 3.0]]


def _masks(kfs=range(5)):
    m = {}
    for k in kfs:
        m[(0, k)] = _box(0, 9, 0, 3)
        m[(1, k)] = _box(0, 9, 6, 9)
    return m


def _zb(value=3.0):
    return {k: np.full((H, W), value) for k in range(5)}


def _table(col_of_point0_per_kf, z0=3.0):
    """Point 0 at column ``col[kf]`` (None = out of frame) and depth z0 in every view;
    Y's points always in Y's columns at Y's depth."""
    pix, z = {}, {}
    for k in range(5):
        c0 = col_of_point0_per_kf[k]
        pix[k] = [None if c0 is None else (5, c0), (5, 7), (5, 8)]
        z[k] = [z0, 3.0, 3.0]
    return pix, z


def _run(cols, z0=3.0, zb=3.0, masks=None, roots=None, ks=(0, 4, 4), visits_x=((0, 4),),
         dilate_px=0, min_votes=2, frac=0.5):
    pix, z = _table(cols, z0)
    vis = _TableVis(XYZ, CENTRES, pix, z, _zb(zb), masks or _masks())
    pts = {0: np.array([0]), 1: np.array([1, 2])}
    ml = [_m(0, "x", list(visits_x)), _m(1, "y", [(0, 4)])]
    return _filter(vis, pts, ks, ml, roots if roots is not None else {0: 1, 1: 2},
                   dilate_px=dilate_px, min_votes=min_votes, frac=frac)


def test_a_point_on_another_objects_surface_in_most_views_leaves():
    # birth kf 0; kf 1-4 vote: Y's columns (on Y's surface) in three, X's in one
    kill, rep = _run([1, 7, 8, 7, 2])
    assert kill[0] and not kill[1:].any(), kill
    assert any(d["reason"].startswith("on another object's surface") and d["points"] == 1
               for d in rep.detail)


def test_one_accusing_view_is_not_a_majority():
    kill, _ = _run([1, 7, 2, 2, 3])          # Y in one of four voting views, X in three
    assert not kill.any()


def test_a_tie_keeps_the_point():
    kill, _ = _run([1, 7, 8, 2, 3])          # two on Y, two inside its own mask
    assert not kill.any()


def test_fewer_votes_than_min_votes_judge_nothing():
    kill, _ = _run([1, 7, None, None, None], min_votes=2)   # one voting view only
    assert not kill.any()


def test_masklets_of_one_fused_object_never_conflict():
    kill, _ = _run([1, 7, 8, 7, 2], roots={0: 7, 1: 7})
    assert not kill.any()


def test_the_birth_keyframe_never_votes():
    """Born in kf 0, where it sits in Y's columns (a mask leak): kf 0 says nothing;
    the other views, inside its own mask, keep it."""
    masks = _masks()
    masks[(0, 0)] = _box(0, 9, 0, 9)          # X's leaky mask in its birth keyframe
    kill, _ = _run([7, 1, 2, 3, 1], masks=masks)
    assert not kill.any()
    # ...and the leak does not vouch for it either: kf 1-2 put it on Y, kf 3 inside
    # X, kf 4 on neither (column 5). Without the birth keyframe: 2 of 4 on Y, 1 of 4
    # inside X → it leaves. Were kf 0 to vote, its leaky mask would add an 'own'
    # (2 of 5) and dilute the accusation (2 of 5) → it would stay.
    kill, _ = _run([7, 7, 8, 2, 5], masks=masks)
    assert kill[0] and not kill[1:].any(), kill


def test_the_points_own_visit_votes_for_it():
    """(c) A single-visit object used to be judged by ONE view of another visit and
    could never be 'inside its own' there (rule 1 only marks 'inside' from OTHER
    visits). Now the keyframes of its own visit vote: inside its own mask in kf 1-3,
    on Y only in kf 4 → it stays."""
    kill, _ = _run([1, 2, 3, 2, 8], visits_x=((0, 4),))
    assert not kill.any()
    pix, z = _table([1, 2, 3, 2, 8])
    vis = _TableVis(XYZ, CENTRES, pix, z, _zb(), _masks())
    own_views = {k: m for (o, k), m in _masks().items() if o == 0}
    nv, no, nt = VD.other_mask_votes(vis, vis.xyz[:1], np.array([0]), own_views,
                                     lambda kf: _box(0, 9, 6, 9), 0, TOL_REL, MIN_TRI)
    assert (nv[0], no[0], nt[0]) == (4, 3, 1)


def test_views_where_the_object_has_no_mask_do_not_vote():
    """X has masks in kf 0-2 only: kf 3 and 4 put the point on Y but cannot say
    whether it is inside X's own silhouette there, so they do not vote."""
    masks = {k_: v for k_, v in _masks().items() if not (k_[0] == 0 and k_[1] > 2)}
    kill, _ = _run([1, 2, None, 7, 8], masks=masks, visits_x=((0, 2),))
    assert not kill.any()


def test_the_foreign_mask_is_never_dilated():
    """(a) The rim tolerance belongs to the point's OWN silhouette. Column 5 is one
    pixel outside Y's mask (6-9) and outside X's (0-3, dilated by 1 → 0-4): with Y's
    mask dilated it would be accused in every view; undilated, nothing accuses it."""
    kill, _ = _run([1, 5, 5, 5, 5], dilate_px=1)
    assert not kill.any()
    kill, _ = _run([1, 6, 6, 6, 6], dilate_px=1)   # one column further: inside Y
    assert kill[0] and not kill[1:].any()


def test_the_own_mask_is_dilated():
    """...and the point's own rim tolerance still holds: column 4 is inside X's mask
    dilated by one pixel."""
    masks = _masks()
    for k in range(5):
        masks[(1, k)] = _box(0, 9, 4, 9)          # Y's mask covers column 4 too
    kill, _ = _run([1, 4, 4, 4, 4], masks=masks, dilate_px=1)
    assert not kill.any()


def test_a_point_behind_another_object_is_occluded_not_on_it():
    """(b) Inside Y's mask but 10 % deeper than Y's measured surface: occluded, no
    vote — the background behind a foreground rim (a contact crease)."""
    kill, _ = _run([1, 7, 8, 7, 8], z0=3.3, zb=3.0)
    assert not kill.any()


def test_a_point_in_front_of_another_object_is_not_on_it():
    """(b) Inside Y's mask but 10 % NEARER than what that keyframe measured: the camera
    saw Y through it — it is not on Y's surface, so rule 4 does not accuse it."""
    kill, _ = _run([1, 7, 8, 7, 8], z0=2.7, zb=3.0)
    assert not kill.any()


def test_a_point_within_the_relative_tolerance_is_on_the_surface():
    kill, _ = _run([1, 7, 8, 7, 8], z0=3.0 * (1 + 0.6 * TOL_REL), zb=3.0)
    assert kill[0]


def test_a_view_along_the_birth_ray_does_not_vote():
    """Views without parallax on the birth ray see every depth of that line at one
    pixel. Cameras 1-4 sit on the line from camera 0 through the point."""
    xyz = np.asarray(XYZ, np.float64)
    c0 = np.array([-2.0, 0.0, 0.0])
    ray = xyz[0] - c0
    centres = [c0.tolist()] + [(c0 + f * ray).tolist() for f in (0.2, 0.4, 1.5, 2.0)]
    pix, z = _table([1, 7, 8, 7, 8])
    vis = _TableVis(XYZ, centres, pix, z, _zb(), _masks())
    pts = {0: np.array([0]), 1: np.array([1, 2])}
    ml = [_m(0, "x", [(0, 4)]), _m(1, "y", [(0, 4)])]
    kill, _ = _filter(vis, pts, (0, 4, 4), ml, {0: 1, 1: 2})
    assert not kill.any()


def test_ballot_counts():
    """other_mask_votes itself: (n_votes, n_own, n_other) per point."""
    pix, z = _table([1, 7, 8, 2, None])
    vis = _TableVis(XYZ, CENTRES, pix, z, _zb(), _masks())
    own_views = {k: m for (o, k), m in _masks().items() if o == 0}
    nv, no, nt = VD.other_mask_votes(vis, vis.xyz[:1], np.array([0]), own_views,
                                     lambda kf: _box(0, 9, 6, 9), 0, TOL_REL, MIN_TRI)
    assert (nv[0], no[0], nt[0]) == (3, 1, 2)
