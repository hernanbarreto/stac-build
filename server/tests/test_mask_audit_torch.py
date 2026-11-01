"""The mask audit's per-object frame loop on the evidence's device (segmentation.mask_filter.MaskAudit
.audit → shape_proposer._visible_in_frame_t) against the numpy twin (_visible_in_frame + _inst_zbuf)
on the synthetic session of test_hole_audit_torch: the same visible / inside verdict per point and
frame, the same per-visit tallies, the same per-point marks."""
import numpy as np
import pytest

from tests.test_hole_audit_torch import _session, FRAMES, MH, MW   # noqa: E402

torch = pytest.importorskip("torch")

CFG = {"max_frames_per_visit": 0, "depth_tol_m": 0.15, "self_tol_m": 0.10, "visit_gap_kf": 1, "dilate_px": 2,
       "min_inside_frac": 0.5, "min_votes": 2, "enabled": True}


def test_the_audit_on_the_device_equals_the_numpy_twin(tmp_path):
    from reconstruction.surface_fit import hole_audit
    from segmentation.mask_filter import MaskAudit
    from segmentation.shape_proposer import _visible_in_frame
    hole_audit.use_device("cpu")
    out, xyz, plate = _session(tmp_path)
    audit = MaskAudit(out, out.parent, CFG, cloud_to_mask={int(f): k for k, f in enumerate(FRAMES)})
    assert audit.ok
    audit.obj_of = {1: 0}                 # the matcher's instance → mask object id (no segmentation.json here)
    # the plate (oid 0 → instance id 1): its points, born across the three frames
    n_wall = len(xyz) - len(plate)
    pts = xyz[n_wall:]
    fg = np.asarray(FRAMES)[np.arange(n_wall, len(xyz)) % 3]
    rec = audit.audit(1, pts, fg)
    assert rec["reason"] is None and rec["on_mask"] > 0
    # the numpy twin over the same frames: the same tallies and marks
    ev = audit.ev
    oid = audit.obj_of[1]
    mh, mw = audit.res
    seen = np.zeros(len(pts), np.int32)
    off = np.zeros(len(pts), np.int32)
    on_total = off_total = 0
    visits = audit._visits([f for f, _k in ev.frames_for(oid)])
    vis_of = audit._assign_visits(fg, visits)
    for vi, vframes in enumerate(visits):
        sel = np.flatnonzero(vis_of == vi)
        if not len(sel):
            continue
        p = pts[sel]
        zbuf = {}
        for mf in vframes:
            m = audit._mask_of(mf, oid)
            if m is None or not m.any():
                continue
            pr = _visible_in_frame(ev, audit._cloud_frame(mf), mh, mw, p, inst_pts=p, inst_zbuf_cache=zbuf,
                                   depth_tol=CFG["depth_tol_m"], self_tol=CFG["self_tol_m"])
            if pr is None:
                continue
            mu, mv, vis = pr
            if not vis.any():
                continue
            inside = np.zeros(len(p), bool)
            inside[vis] = m[mv[vis], mu[vis]]
            on_total += int(inside.sum()); off_total += int(vis.sum()) - int(inside.sum())
            seen[sel[vis]] += 1
            off[sel[vis & ~inside]] += 1
    assert rec["on_mask"] == on_total and rec["off_mask"] == off_total
    assert np.array_equal(rec["judged_views"], seen) and np.array_equal(rec["off_views"], off)
    assert np.array_equal(rec["out_of_place"], (seen > 0) & (off > 0))
