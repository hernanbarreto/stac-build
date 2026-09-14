"""Are two clusters of one instance the SAME object displaced by drift, or two
different objects? The images decide, not a distance threshold.

The spatial gate rules on the separation between the two copies against a drift
budget δ(L). That budget is an estimate, and on a session whose real drift
exceeds it the gate splits genuine duplicates apart — pccr 2026-09-14 measured
a revisit demanding 2.59 m of translation while the budget at those keyframes
was 0.90 m, and copies 1.24 m apart were declared "not drift".

The discriminator the images give is a reprojection: take cluster A's points,
project them into the keyframes where cluster B was OBSERVED, and compare the
footprint with the instance's mask there.

  * the SAME object seen twice lands offset from the mask by a CONSISTENT rigid
    shift across every frame — that shift IS the drift, measured in pixels;
  * two DIFFERENT objects land nowhere near the mask, and whatever shift each
    frame suggests disagrees with the next.

GRIDS. Three of them are in play and mixing them fails silently, with a low IoU
that reads as "different objects" — the wrong answer, delivered confidently.
K lives on the TRACE grid (the cloud's pixel_row/pixel_col extent, 688x384 on
pccr), the masks on SAM3's own grid (832x464) and the RGB frames on a third.
Everything here is measured in the MASK grid, reached through
``_visible_in_frame``, which scales per axis (832/688 and 464/384 are not the
same number) off the measured extent rather than any file's metadata.

Nothing here edits geometry. It measures and declares — tool_measured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def _footprint(ev, fidx: int, pts: np.ndarray, mh: int, mw: int,
               dilate_px: int, occlusion: bool) -> Optional[np.ndarray]:
    """Binary mask-grid footprint of ``pts`` in frame ``fidx``.

    ``occlusion`` decides whether the scene's Z-buffer may cull the points, and
    the choice is the whole experiment:

      * TRUE for the self-check. There we ask what the camera really saw, so
        geometry hidden behind other surfaces must not count.
      * FALSE for the cross-check. A drift-displaced copy lands INSIDE or
        BEHIND the geometry of the frame it is being tested against — that is
        what being displaced means — so the Z-buffer culls exactly the case we
        are trying to detect and the pair comes back "distinct" for the one
        reason that proves they are the same. The first run of this test said
        "neither copy reaches a frame where the other was observed" about a
        ceiling beam for precisely that.
    """
    if occlusion:
        from segmentation.shape_proposer import _visible_in_frame
        r = _visible_in_frame(ev, fidx, mh, mw, pts)
        if r is None:
            return None
        mu, mv, keep = r
    else:
        from segmentation.shape_proposer import _project_frame
        pr = _project_frame(ev, fidx, pts)
        if pr is None:
            return None
        u, v, _z, front = pr
        # K lives on the TRACE grid; the masks on SAM3's. Per axis, off the
        # measured extent — 832/688 and 464/384 are not the same number.
        mu = (u * mw / ev.kw).astype(np.int64)
        mv = (v * mh / ev.kh).astype(np.int64)
        keep = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
    if keep.sum() < 8:
        return None
    fp = np.zeros((mh, mw), dtype=bool)
    fp[mv[keep], mu[keep]] = True
    if dilate_px > 0:
        # the projection is a sparse point set, the mask is a filled region:
        # close the gaps so the overlap is measured between AREAS, not samples
        from scipy import ndimage as ndi
        fp = ndi.binary_dilation(fp, iterations=int(dilate_px))
    return fp


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _agree(fp: np.ndarray, mask: np.ndarray) -> float:
    """How well a projected footprint and a mask agree: max(precision, recall).

    Neither one alone survives the range of sizes in play, and IoU survives
    neither. A ceiling duct crossing the room projects its whole run into a
    frame where SAM3 masked one slice: precision is low, recall high, IoU near
    zero on geometry that is perfectly placed (measured: 0.00 on 1.7 M points).
    The far copy of a chair is 634 points against a 1,500-pixel mask: it cannot
    cover the mask however well placed, so recall is low — 0.00 in the run of
    2026-09-14, which declared the pair `unusable` and let the split through —
    while precision is high.

    One containing the other is agreement in both directions, so the measure is
    the larger of the two: the projection covers the mask, OR it falls inside
    it.
    """
    inter = float(np.logical_and(fp, mask).sum())
    if inter == 0.0:
        return 0.0
    d_mask, d_fp = float(mask.sum()), float(fp.sum())
    recall = inter / d_mask if d_mask else 0.0
    precision = inter / d_fp if d_fp else 0.0
    return max(recall, precision)


def _shift(fp: np.ndarray, dv: int, du: int) -> np.ndarray:
    out = np.zeros_like(fp)
    h, w = fp.shape
    v0, v1 = max(0, dv), min(h, h + dv)
    u0, u1 = max(0, du), min(w, w + du)
    if v0 >= v1 or u0 >= u1:
        return out
    out[v0:v1, u0:u1] = fp[v0 - dv:v1 - dv, u0 - du:u1 - du]
    return out


def _centroid(m: np.ndarray):
    vs, us = np.nonzero(m)
    if not len(vs):
        return None
    return float(vs.mean()), float(us.mean())


def compare_in_frames(ev, mask_of_frame, pts: np.ndarray, frames: Sequence[int],
                      mh: int, mw: int, dilate_px: int, max_frames: int,
                      occlusion: bool) -> List[dict]:
    """Per frame: how the footprint of ``pts`` sits against the mask — recall
    and IoU as projected, the rigid shift that aligns the two centroids, and
    recall and IoU after it."""
    # Pick the frames that CARRY evidence, largest mask first, and only then
    # take max_frames of them. Truncating the raw list first threw the whole
    # measurement away whenever the first few frames happened to have no mask:
    # pccr's 1.7 M-point duct has a mask in 140 of its 167 frames and scored
    # 0.00 because none of the first 8 was among them — while every frame that
    # does carry one agrees 0.99.
    usable = []
    for fidx in frames:
        m = mask_of_frame(fidx)
        if m is not None and m.any():
            usable.append((int(m.sum()), int(fidx), m))
    usable.sort(reverse=True)
    out = []
    for _area, fidx, m in usable[:max_frames]:
        fp = _footprint(ev, fidx, pts, mh, mw, dilate_px, occlusion)
        if fp is None:
            continue
        c_fp, c_m = _centroid(fp), _centroid(m)
        if c_fp is None or c_m is None:
            continue
        dv, du = int(round(c_m[0] - c_fp[0])), int(round(c_m[1] - c_fp[1]))
        fp_s = _shift(fp, dv, du)
        out.append({"frame": int(fidx),
                    "recall": _agree(fp, m), "iou": _iou(fp, m),
                    "recall_aligned": _agree(fp_s, m), "iou_aligned": _iou(fp_s, m),
                    "shift_v": dv, "shift_u": du,
                    "shift_px": float(np.hypot(dv, du))})
    return out


def copy_evidence(output_dir, session_dir, instance_id: int, oid: Optional[int],
                  pts_a: np.ndarray, pts_b: np.ndarray,
                  frames_a: Sequence[int], frames_b: Sequence[int],
                  oid_b: Optional[int] = None,
                  cloud_to_mask: Optional[Dict[int, int]] = None,
                  dilate_px: int = 3, max_frames: int = 8,
                  min_self_recall: float = 0.40, min_cross_recall: float = 0.40,
                  min_agreeing_frac: float = 0.60) -> dict:
    """Do the two clusters of one instance show the SAME object?

    Returns the measurements and a verdict ∈ {same_object, distinct, unusable}.
    ``unusable`` is not a failure to hide: it means the projection does not even
    reach the instance's own mask in its own frames, so the geometry, the poses
    or the grid scaling are off and NO verdict here can be trusted. It is
    reported, never quietly turned into ``distinct``.
    """
    from reconstruction.surface_fit.hole_audit import _Evidence
    ev = _Evidence(Path(output_dir), Path(session_dir))
    if not ev.ok or oid is None:
        return {"verdict": "unusable", "reason": "no mask/camera evidence for this session"}

    # Two clusters of ONE instance share an oid. Two instances that a previous
    # split tore apart do not, and each copy has to be measured against ITS OWN
    # mask — checking the child against the parent's mask reports 0.00 recall
    # and buries a perfectly good pair under "unusable".
    oid_a = int(oid)
    oid_b = int(oid_b) if oid_b is not None else oid_a
    c2m = cloud_to_mask or {}

    def mask_for(o: int):
        def _m(fidx: int):
            key = f"f{c2m.get(int(fidx), int(fidx))}_o{int(o)}"
            return (ev.masks[key] > 0) if key in ev.masks.files else None
        return _m

    mask_a, mask_b = mask_for(oid_a), mask_for(oid_b)

    probe = None
    for f, mf in [(f, mask_a) for f in frames_a] + [(f, mask_b) for f in frames_b]:
        m = mf(f)
        if m is not None:
            probe = m
            break
    if probe is None:
        return {"verdict": "unusable",
                "reason": f"no mask of object {oid_a}/{oid_b} in the copies' frames"}
    mh, mw = probe.shape[:2]

    # ── sanity: each cluster must land on its own mask in its own frames ──
    # Occlusion OFF here too. The self-check exists to prove the grids and the
    # poses, and its own mask IS by definition where the cluster was seen — but
    # a copy displaced by drift sits inside geometry that does not belong to it,
    # so the scene Z-buffer culls it entirely. In the run of 2026-09-14 the far
    # copy of the chair projected to 732-858 px agreeing 1.00 with its mask
    # without occlusion, and to NOTHING with it: the pair was declared
    # `unusable` and the split went through. Culling the evidence of drift is
    # exactly what this check must not do.
    self_a = compare_in_frames(ev, mask_a, pts_a, frames_a, mh, mw, dilate_px, max_frames, False)
    self_b = compare_in_frames(ev, mask_b, pts_b, frames_b, mh, mw, dilate_px, max_frames, False)
    sa = float(np.median([r["recall"] for r in self_a])) if self_a else 0.0
    sb = float(np.median([r["recall"] for r in self_b])) if self_b else 0.0
    out = {"instance_id": int(instance_id), "obj_id": oid_a, "obj_id_b": oid_b,
           "self_recall_a": sa, "self_recall_b": sb,
           "self_a": self_a, "self_b": self_b, "provenance": "tool_measured"}
    if min(sa, sb) < min_self_recall:
        out.update(verdict="unusable",
                   reason=(f"a cluster does not reproject onto its own mask "
                           f"(median recall {sa:.2f} / {sb:.2f} < {min_self_recall}) — "
                           f"poses, geometry or grid scaling are off, no verdict"))
        return out

    # ── the question: does A land where B's mask is, and vice versa? ────
    # occlusion OFF: a displaced copy sits behind the geometry of the frame it
    # is tested against, and culling it would answer "distinct" for the very
    # reason that proves the opposite.
    # A is tested against B's mask in B's frames, and B against A's in A's.
    cross = (compare_in_frames(ev, mask_b, pts_a, frames_b, mh, mw, dilate_px, max_frames, False)
             + compare_in_frames(ev, mask_a, pts_b, frames_a, mh, mw, dilate_px, max_frames, False))
    out["cross"] = cross
    if not cross:
        out.update(verdict="unusable",
                   reason="neither copy projects into the other's frames at all")
        return out

    aligned = float(np.median([r["recall_aligned"] for r in cross]))
    raw = float(np.median([r["recall"] for r in cross]))
    shifts = np.array([[r["shift_v"], r["shift_u"]] for r in cross], dtype=np.float64)
    dispersion = float(np.hypot(*shifts.std(axis=0))) if len(shifts) > 1 else 0.0
    # What separates one displaced object from two is not how tightly the shift
    # vectors cluster — measured on pccr, the scatter relative to the shift is
    # 1.05 for the chair that IS a duplicate and 0.81 for the duct that is not,
    # so dispersion ranks them backwards. It is whether MOST frames agree once
    # aligned. The chair reads 0.38/0.75/0.56/0.69/0.78/0.79 — every frame
    # moderately high; the duct reads 0.06/0.04/0.04/0.11/0.86/0.91 — bimodal,
    # two accidental hits among near-zeros, with dv steady while du swings from
    # -129 to +213. There is no single rigid shift there, and the spread of the
    # agreements says so where the spread of the shifts does not.
    agreeing = sum(1 for r in cross if r["recall_aligned"] >= min_cross_recall)
    frac = agreeing / float(len(cross))
    out.update(cross_recall=raw, cross_recall_aligned=aligned,
               shift_px_median=float(np.median([r["shift_px"] for r in cross])),
               shift_dispersion_px=dispersion, agreeing_frames=agreeing,
               agreeing_frac=frac, n_cross_frames=len(cross))

    if aligned >= min_cross_recall and frac >= min_agreeing_frac:
        out.update(verdict="same_object",
                   reason=(f"one copy lands on the other's mask under a single rigid shift of "
                           f"{out['shift_px_median']:.0f} px in {agreeing} of {len(cross)} "
                           f"frame(s) (agreement {raw:.2f} → {aligned:.2f}) — the same object, "
                           f"displaced"))
    else:
        out.update(verdict="distinct",
                   reason=(f"no single shift aligns the copies: {agreeing} of {len(cross)} "
                           f"frame(s) agree (median {raw:.2f} → {aligned:.2f})"))
    return out
