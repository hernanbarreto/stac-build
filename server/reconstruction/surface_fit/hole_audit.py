"""
Stage-1 evidence-based hole audit (user concept 2026-08-29).

A hole in a fitted surface's point support is one of two very different
things: a REAL opening (door/window — the scan frames show it) or a
reconstruction gap (occlusion, low confidence, textureless surface — the
frames show continuous surface there). The cloud alone cannot tell them
apart; the SCAN FRAMES can. This module makes the surface see them:

  every unsupported cell of the surface's UV grid is projected into the
  SAM3 mask keyframes (poses + intrinsics from the session) and voted:
    - covered by the instance's own mask in most views → the surface IS
      there in the images → the cell is FILLED (provenance image_supported);
    - uncovered → the opening is real → the cell stays OPEN, and the
      opening border follows the mask evidence at cell resolution;
    - too few views / mixed evidence → AMBIGUOUS → stays open (we never
      invent; conservative default).

Nothing is generated from priors: every filled triangle is backed by
measured mask pixels from the frames that built the cloud, and the audit
report says exactly which cells were filled on image evidence.

Frame convention: RAW cloud frame (the fit frame) — poses and the cleaned
cloud share it; the viewer's floor transform is applied downstream.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("SurfaceFit")

# per-process cache: trace grid (K's pixel grid) per output dir
_TRACE_GRID: Dict[str, Tuple[int, int]] = {}


def _k_grid(output_dir: Path):
    """(W, H, cloud_xyz): the intrinsics' pixel grid = the cloud traceability
    grid (session_io builds K at the reconstruction grid, not the RGB grid —
    verified on test3: cx,cy sit at the centre of the 384x688 trace grid),
    plus the cloud points for the per-frame Z-buffers."""
    key = str(output_dir)
    if key in _TRACE_GRID:
        return _TRACE_GRID[key]
    try:
        from segmentation.pipeline import _load_ply_origins
        origins = _load_ply_origins(Path(output_dir) / "cleaned_cloud.ply")
        if origins is None:
            return None
        xyz, _fg, pr, pc = origins
        out = (int(pc.max()) + 1, int(pr.max()) + 1, np.asarray(xyz))  # (W, H, pts)
        _TRACE_GRID[key] = out
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("hole_audit: could not derive K grid: %s", e)
        return None


class _Evidence:
    """Masks + cameras for one session, shared across instances."""

    def __init__(self, output_dir: Path, session_dir: Path):
        self.ok = False
        self._others_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        p = Path(output_dir) / "seg_masks.npz"
        if not p.exists():
            return
        try:
            self.masks = np.load(p, allow_pickle=True)
            from segmentation.session_io import _load_camera_source
            self.cam = _load_camera_source(Path(session_dir), Path(output_dir))
            grid = _k_grid(Path(output_dir))
            if self.cam is None or grid is None:
                return
            self.kw, self.kh, self._cloud = grid
            self._zbuf_cache: Dict[int, Optional[np.ndarray]] = {}
            self.ok = True
        except Exception as e:  # noqa: BLE001
            logger.warning("hole_audit: evidence load failed: %s", e)

    def _zbuf(self, fidx: int, mh: int, mw: int) -> Optional[np.ndarray]:
        """Per-frame Z-buffer from the FULL cloud at mask resolution: what
        measured geometry is closest along each pixel ray. THE depth check
        the 2-D masks lack (without it, a phantom plane cell beyond the wall
        edge projected onto some other surface and voted 'occluded')."""
        if fidx in self._zbuf_cache:
            return self._zbuf_cache[fidx]
        c2w = self.cam.pose_map.get(fidx)
        K = self.cam.K_for(fidx)
        if c2w is None or K is None:
            self._zbuf_cache[fidx] = None
            return None
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        p = (M[:3, :3] @ self._cloud.T).T + M[:3, 3]
        z = p[:, 2]
        ok = z > 0.05
        u = K[0, 0] * p[ok, 0] / z[ok] + K[0, 2]
        v = K[1, 1] * p[ok, 1] / z[ok] + K[1, 2]
        mu = (u * mw / self.kw).astype(np.int64)
        mv = (v * mh / self.kh).astype(np.int64)
        inb = (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        zb = np.full((mh, mw), np.inf, dtype=np.float64)
        np.minimum.at(zb, (mv[inb], mu[inb]), z[ok][inb])
        # window-min: the cloud is sparse at mask resolution, so single pixels
        # along a ray are often unmeasured (inf) while their NEIGHBOURS are —
        # wall3's arch kept filling because the background seen through it had
        # no point on the exact pixel (user 2026-08-29). A 5-px minimum filter
        # gives every pixel the closest measured depth nearby.
        from scipy import ndimage as _ndi
        zb = _ndi.minimum_filter(zb, size=5, mode="nearest")
        self._zbuf_cache[fidx] = zb
        return zb

    def frames_for(self, oid: int):
        out = []
        for k in self.masks.files:
            m = re.match(rf"^f(\d+)_o{oid}$", k)
            if m:
                out.append((int(m.group(1)), k))
        return sorted(out)

    def _others_mask(self, fidx: int, oid: int) -> Optional[np.ndarray]:
        """Union of every OTHER instance's mask in a frame — the occluders."""
        key = (fidx, oid)
        if key in self._others_cache:
            return self._others_cache[key]
        acc = None
        for k in self.masks.files:
            m = re.match(rf"^f{fidx}_o(\d+)$", k)
            if not m or int(m.group(1)) == oid:
                continue
            a = self.masks[k] > 0
            acc = a if acc is None else (acc | a)
        self._others_cache[key] = acc
        return acc

    def vote(self, world_pts: np.ndarray, oid: int, depth_tol: float = 0.15):
        """Per-point evidence tallies over the instance's mask keyframes,
        DEPTH-VERIFIED against the cloud Z-buffer:
        - occluded: measured geometry sits ≥depth_tol IN FRONT of the point
          along the ray — the frame cannot testify (real occlusion; also kills
          phantom votes where a plane cell beyond the wall edge merely
          projected onto some other surface);
        - covered: visible at its own depth AND the instance's OWN mask covers
          the pixel — a direct, depth-consistent witness;
        - neither (empty): visible, and the pixel is not this instance — the
          frame sees PAST the surface (or a coplanar in-fill object like a
          door leaf occupies it) → witness the surface is NOT there.
        Returns (covered, occluded, valid)."""
        n = len(world_pts)
        covered = np.zeros(n, dtype=np.int32)
        occluded = np.zeros(n, dtype=np.int32)
        valid = np.zeros(n, dtype=np.int32)
        for fidx, key in self.frames_for(oid):
            c2w = self.cam.pose_map.get(fidx)
            K = self.cam.K_for(fidx)
            if c2w is None or K is None:
                continue
            m = self.masks[key]
            mh, mw = m.shape
            zb = self._zbuf(fidx, mh, mw)
            c2w4 = np.eye(4)
            c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
            M = np.linalg.inv(c2w4)
            pcam = (M[:3, :3] @ world_pts.T).T + M[:3, 3]
            z = pcam[:, 2]
            front = z > 0.05
            u = np.full(n, -1.0)
            v = np.full(n, -1.0)
            u[front] = K[0, 0] * pcam[front, 0] / z[front] + K[0, 2]
            v[front] = K[1, 1] * pcam[front, 1] / z[front] + K[1, 2]
            mu = (u * mw / self.kw).astype(np.int64)
            mv = (v * mh / self.kh).astype(np.int64)
            inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
            valid[inb] += 1
            occ = np.zeros(n, dtype=bool)
            behind = np.zeros(n, dtype=bool)
            if zb is not None:
                zpix = np.full(n, np.inf)
                zpix[inb] = zb[mv[inb], mu[inb]]
                occ = zpix < (z - depth_tol)
                # measured geometry BEHIND the surface through this cell: the
                # camera sees PAST it (wall3's arch: through the arch you see
                # the attached cone — same instance, so the 2-D mask covered
                # it and the arch got filled; depth says OPEN)
                behind = np.isfinite(zpix) & (zpix > (z + depth_tol))
            occluded[occ] += 1
            hit = np.zeros(n, dtype=bool)
            hit[inb] = m[mv[inb], mu[inb]] > 0
            covered[hit & ~occ & ~behind] += 1
        return covered, occluded, valid

    def see_through(self, world_pts: np.ndarray, depth_tol: float = 0.15):
        """Per-point (valid, at_depth, behind) tallies over ALL posed keyframes
        — no mask involved, pure depth evidence. 'behind' counts frames whose
        cloud Z-buffer puts the nearest measured geometry ≥depth_tol BEHIND
        the point (the camera sees PAST it); 'at_depth' counts frames that
        measure geometry AT the point's depth (±depth_tol) — testimony the
        surface is real. Rays with no finite Z-buffer say nothing (not valid)."""
        try:
            sr = self.masks["scaled_res"]
            mh, mw = int(sr[0]), int(sr[1])
        except Exception:  # noqa: BLE001
            mh, mw = 640, 360
        n = len(world_pts)
        valid = np.zeros(n, dtype=np.int32)
        at_depth = np.zeros(n, dtype=np.int32)
        behind = np.zeros(n, dtype=np.int32)
        for fidx in sorted(self.cam.pose_map.keys()):
            c2w = self.cam.pose_map.get(fidx)
            K = self.cam.K_for(fidx)
            if c2w is None or K is None:
                continue
            zb = self._zbuf(fidx, mh, mw)
            if zb is None:
                continue
            c2w4 = np.eye(4)
            c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
            M = np.linalg.inv(c2w4)
            pcam = (M[:3, :3] @ world_pts.T).T + M[:3, 3]
            z = pcam[:, 2]
            front = z > 0.05
            u = np.full(n, -1.0)
            v = np.full(n, -1.0)
            u[front] = K[0, 0] * pcam[front, 0] / z[front] + K[0, 2]
            v[front] = K[1, 1] * pcam[front, 1] / z[front] + K[1, 2]
            mu = (u * mw / self.kw).astype(np.int64)
            mv = (v * mh / self.kh).astype(np.int64)
            inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
            zpix = np.full(n, np.inf)
            zpix[inb] = zb[mv[inb], mu[inb]]
            measured = np.isfinite(zpix)
            valid[measured] += 1
            at_depth[measured & (np.abs(zpix - z) <= depth_tol)] += 1
            behind[measured & (zpix > z + depth_tol)] += 1
        return valid, at_depth, behind

    def calibrate_oid(self, world_pts: np.ndarray, instance_id: int) -> Optional[int]:
        """Find the npz object id for this instance by projecting its OWN
        on-surface points: the right mask covers them (~100%). Convention is
        oid = instance_id - 1, but this self-check survives drift."""
        sample = world_pts[:: max(1, len(world_pts) // 2000)]
        candidates = [int(instance_id) - 1]
        try:
            candidates += [int(o) for o in self.masks["obj_ids"].tolist()
                           if int(o) != instance_id - 1]
        except Exception:  # noqa: BLE001
            pass
        for oid in candidates:
            if not self.frames_for(oid):
                continue
            cov, _occ, val = self.vote(sample, oid)
            seen = val > 0
            if seen.sum() >= 50 and (cov[seen] / val[seen]).mean() > 0.5:
                if oid != instance_id - 1:
                    logger.warning("hole_audit: instance %d matched mask o%d "
                                   "(non-standard mapping)", instance_id, oid)
                return oid
        return None


_EVIDENCE_CACHE: Dict[str, _Evidence] = {}


def _evidence(output_dir: Path, session_dir: Path) -> _Evidence:
    key = str(output_dir)
    ev = _EVIDENCE_CACHE.get(key)
    if ev is None:
        ev = _Evidence(output_dir, session_dir)
        _EVIDENCE_CACHE[key] = ev
    return ev


def audit_and_fill(model, uv_support: np.ndarray,
                   output_dir: Path, session_dir: Path, instance_id: int,
                   resolution: float = 0.05, support_radius: float = 0.08,
                   fill_ratio: float = 0.6, open_ratio: float = 0.3,
                   occluded_ratio: float = 0.5,
                   min_votes: int = 2, snap_max_m: float = 0.2,
                   max_fill_dist_cells: Optional[int] = None,
                   interp_max_cells: int = 30):
    """Audit the holes of a fitted surface against the scan-frame masks and
    rebuild its mesh with image-supported fills.

    Returns (verts_world, faces, report) or None when the audit cannot run
    (no masks / no camera / no holes). ``report`` counts filled / open /
    ambiguous cells and the evidence used."""
    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok or len(uv_support) < 100:
        return None

    uv = np.asarray(uv_support, dtype=np.float64)
    from .support import support_grid

    # ── supported quads (same grid convention as support.trimmed_quad_mesh)
    occ_cell = max(min(resolution, support_radius / 2.0), 1e-4)
    occ, ou0, ov0 = support_grid(uv, occ_cell, support_radius)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    nu = max(int(np.ceil((u1 - u0) / resolution)), 1)
    nv = max(int(np.ceil((v1 - v0) / resolution)), 1)
    uc = u0 + (np.arange(nu) + 0.5) * resolution
    vc = v0 + (np.arange(nv) + 0.5) * resolution
    ci = np.clip(np.floor((uc - ou0) / occ_cell).astype(np.int64), 0, occ.shape[1] - 1)
    cj = np.clip(np.floor((vc - ov0) / occ_cell).astype(np.int64), 0, occ.shape[0] - 1)
    keep = occ[np.ix_(cj, ci)]                        # (nv, nu) point-supported

    # ── audit EVERY unsupported cell of the UV domain. Real openings often
    # touch the domain border (a doorway reaches the floor edge), so a
    # border-connectivity filter would exempt exactly the interesting regions;
    # the mask vote is the arbiter everywhere. No extrapolation risk: the
    # domain itself is bounded by the measured points.
    hole_mask = ~keep
    n_holes = int(hole_mask.sum())
    if n_holes == 0:
        return None

    # ── mask calibration on the surface's own supported points
    sup_world = model.uv_to_world(uv[:: max(1, len(uv) // 4000)])
    oid = ev.calibrate_oid(np.asarray(sup_world, dtype=np.float64), instance_id)
    if oid is None:
        logger.info("hole_audit: instance %d — no matching mask, audit skipped",
                    instance_id)
        return None

    # ── vote every hole cell against the frames (three-way evidence)
    hj, hi = np.nonzero(hole_mask)
    centers_uv = np.column_stack([u0 + (hi + 0.5) * resolution,
                                  v0 + (hj + 0.5) * resolution])
    centers_world = np.asarray(model.uv_to_world(centers_uv), dtype=np.float64)
    cov, occ, val = ev.vote(centers_world, oid)
    vals = np.maximum(val, 1)
    r_cov = np.where(val > 0, cov / vals, 0.0)
    r_occ = np.where(val > 0, occ / vals, 0.0)
    r_empty = np.where(val > 0, 1.0 - r_cov - r_occ, 0.0)
    seen = val >= min_votes
    # direct witnesses: the object's own mask covers the cell
    fill_img = seen & (r_cov >= fill_ratio)
    # mixed: some views see the surface, the rest see an occluder — still a
    # direct witness exists, so image_supported
    fill_mix = seen & ~fill_img & (cov >= 1) & ((r_cov + r_occ) >= fill_ratio)
    # OCCLUSION REASONING (user 2026-08-29, "importantísimo"): no direct
    # witness, but measured geometry stands IN FRONT (Z-buffer verified) in
    # most views — the frames CANNOT testify either way, and on a fitted
    # surface the parsimonious reading is continuity behind the occluder
    # (e.g. the floor behind the ladder). RESTRICTED to INTERIOR holes:
    # a phantom plane extension past the wall edge is border-connected by
    # construction and gets no inferred continuity.
    from scipy import ndimage as _ndi
    # interior = enclosed by point support in all four grid directions — the
    # ladder's shadow on the floor qualifies (floor measured all around it);
    # a phantom extension past the wall edge has no support beyond it and
    # fails. (Strict border-connectivity lost the ladder case: unsupported
    # cells percolate to the domain border on real floors.)
    left = np.maximum.accumulate(keep, axis=1)
    right = np.maximum.accumulate(keep[:, ::-1], axis=1)[:, ::-1]
    top = np.maximum.accumulate(keep, axis=0)
    bot = np.maximum.accumulate(keep[::-1, :], axis=0)[::-1, :]
    interior_grid = left & right & top & bot & ~keep
    interior = interior_grid[hj, hi]
    fill_occl = (seen & ~fill_img & ~fill_mix & interior
                 & (r_occ >= occluded_ratio))
    # curved models (bspline/swept) extrapolate wildly far from their data —
    # cap every fill to cells NEAR the supported region (spikes on the
    # ceiling, user 2026-08-29)
    if max_fill_dist_cells is not None:
        near = _ndi.binary_dilation(keep, iterations=int(max_fill_dist_cells))
        near_cells = near[hj, hi]
        fill_img &= near_cells
        fill_mix &= near_cells
        fill_occl &= near_cells
    fill = fill_img | fill_mix | fill_occl
    # confirmed open: most views see PAST the surface (no mask at all there)
    open_ = seen & ~fill & (r_empty >= 1.0 - open_ratio)
    ambiguous = ~fill & ~open_
    # tiny ENCLOSED gaps with no image verdict: interpolate across the fitted
    # surface — the same bounded bridging Poisson does implicitly (user
    # 2026-08-29: Poisson had fewer holes than ransac). Provenance
    # 'interpolated'; confirmed openings are never touched.
    fill_interp = np.zeros_like(fill)
    if interp_max_cells > 0 and ambiguous.any():
        cand = np.zeros_like(keep)
        cand[hj[ambiguous], hi[ambiguous]] = True
        cand &= interior_grid
        lab2, n2 = _ndi.label(cand)
        if n2:
            sizes = np.bincount(lab2.ravel())
            small_ids = np.nonzero(sizes <= int(interp_max_cells))[0]
            small_ids = small_ids[small_ids > 0]
            if len(small_ids):
                ig = np.isin(lab2, small_ids) & cand
                fill_interp = ig[hj, hi] & ambiguous
                if max_fill_dist_cells is not None:
                    fill_interp &= near_cells
                fill |= fill_interp
                ambiguous = ~fill & ~open_

    keep2 = keep.copy()
    keep2[hj[fill], hi[fill]] = True

    # ── rebuild the quad mesh from keep2 (same corner assembly as support.py)
    jj, ii = np.nonzero(keep2)
    corners = np.stack([jj * (nu + 1) + ii,
                        jj * (nu + 1) + ii + 1,
                        (jj + 1) * (nu + 1) + ii,
                        (jj + 1) * (nu + 1) + ii + 1], axis=1)
    used = np.unique(corners)
    remap = np.full((nv + 1) * (nu + 1), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    c = remap[corners]
    faces = np.concatenate([c[:, [0, 1, 3]], c[:, [0, 3, 2]]], axis=0)
    gj, gi = np.divmod(used, nu + 1)
    verts_uv = np.column_stack([u0 + gi * resolution, v0 + gj * resolution])

    # border snap ONLY near measured points: the overhang ring snaps onto the
    # cloud border (as in support.py); image-filled interiors sit far from any
    # point and MUST stay on the grid — snapping them would collapse the fill.
    from scipy.spatial import cKDTree
    d, nearest = cKDTree(uv).query(verts_uv, k=1)
    band = (d > max(0.6 * resolution, 1e-6)) & (d < snap_max_m)
    if band.any():
        verts_uv = verts_uv.copy()
        verts_uv[band] = uv[nearest[band]]
        tri = verts_uv[faces]
        area2 = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                       - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        faces = faces[area2 > resolution * resolution * 1e-4]

    verts_world = np.asarray(model.uv_to_world(verts_uv), dtype=np.float64)
    report = {
        "hole_cells": n_holes,
        "filled_cells": int(fill.sum()),
        "filled_image_supported": int((fill_img | fill_mix).sum()),
        "filled_occlusion_inferred": int(fill_occl.sum()),
        "filled_interpolated": int(fill_interp.sum()),
        "open_cells": int(open_.sum()),
        "ambiguous_cells": int(ambiguous.sum()),
        "filled_area_m2": round(float(fill.sum()) * resolution * resolution, 3),
        "filled_occluded_area_m2": round(float(fill_occl.sum()) * resolution * resolution, 3),
        "open_area_m2": round(float(open_.sum()) * resolution * resolution, 3),
        "mask_oid": int(oid),
        "frames_used": [f for f, _ in ev.frames_for(oid)],
        "fill_ratio_gate": fill_ratio,
        "open_ratio_gate": open_ratio,
        "occluded_ratio_gate": occluded_ratio,
        "_grid": (keep2,
                  _cells_grid(keep.shape, hj[open_], hi[open_]),
                  float(u0), float(v0), float(resolution)),
        "provenance": "image_supported fills = the instance's own mask covers "
                      "the cell (direct witness); occlusion_inferred fills = "
                      "another object blocks the view in most frames, surface "
                      "continuity inferred behind it; open cells = the frames "
                      "see past the surface (image-confirmed opening); "
                      "ambiguous cells stay open — nothing is invented",
    }
    return verts_world, faces, report


def silhouette_report(verts_world: np.ndarray,
                      output_dir: Path, session_dir: Path,
                      instance_id: int) -> Optional[dict]:
    """Stage-2 metric: how well does the FINAL surface match the images?
    The mesh's footprint is projected into each mask keyframe and compared
    with the instance's own mask: precision (mesh pixels the mask confirms),
    recall (mask pixels the mesh reaches) and IoU, per frame + mean. A low
    score flags 'the curve/shape does not match what the frames show' —
    tool_measured, drives future refinement, never edits geometry itself."""
    from scipy import ndimage as _ndi

    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok or len(verts_world) < 50:
        return None
    sample = np.asarray(verts_world, dtype=np.float64)
    sample = sample[:: max(1, len(sample) // 20000)]
    oid = ev.calibrate_oid(sample, instance_id)
    if oid is None:
        return None
    frames = []
    for fidx, key in ev.frames_for(oid):
        c2w = ev.cam.pose_map.get(fidx)
        K = ev.cam.K_for(fidx)
        if c2w is None or K is None:
            continue
        m = ev.masks[key] > 0
        mh, mw = m.shape
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        pcam = (M[:3, :3] @ sample.T).T + M[:3, 3]
        z = pcam[:, 2]
        front = z > 0.05
        if front.sum() < 50:
            continue
        u = K[0, 0] * pcam[front, 0] / z[front] + K[0, 2]
        v = K[1, 1] * pcam[front, 1] / z[front] + K[1, 2]
        mu = (u * mw / ev.kw).astype(np.int64)
        mv = (v * mh / ev.kh).astype(np.int64)
        inb = (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        if inb.sum() < 50:
            continue
        foot = np.zeros((mh, mw), dtype=bool)
        foot[mv[inb], mu[inb]] = True
        foot = _ndi.binary_dilation(foot, iterations=2)   # close vertex-grid gaps
        inter = float((foot & m).sum())
        union = float((foot | m).sum())
        frames.append({
            "frame": fidx,
            "precision": round(inter / max(foot.sum(), 1), 3),
            "recall": round(inter / max(m.sum(), 1), 3),
            "iou": round(inter / max(union, 1.0), 3),
        })
    if not frames:
        return None
    return {
        "mean_iou": round(float(np.mean([f["iou"] for f in frames])), 3),
        "mean_precision": round(float(np.mean([f["precision"] for f in frames])), 3),
        "mean_recall": round(float(np.mean([f["recall"] for f in frames])), 3),
        "frames": frames,
        "source": "tool_measured",
    }


def _cells_grid(shape, jj, ii) -> np.ndarray:
    g = np.zeros(shape, dtype=bool)
    g[jj, ii] = True
    return g


def save_report(out_dir: Path, report: dict) -> None:
    try:
        (Path(out_dir) / "hole_audit.json").write_text(json.dumps(report, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning("hole_audit: report write failed: %s", e)


def see_through_filter(mesh_verts: np.ndarray, mesh_faces: np.ndarray,
                       output_dir: Path, session_dir: Path,
                       behind_ratio: float = 0.5, min_votes: int = 2,
                       depth_tol: float = 0.15):
    """PHANTOM-SURFACE cut (user 2026-08-29: a decomposition plane covered
    wall3's access arch). A RANSAC primitive collects real points but its MESH
    can bridge across open space between them; the witness is depth: a face
    whose center the cameras consistently see PAST — nearest measured geometry
    ≥depth_tol BEHIND it in ≥behind_ratio of the depth-valid views, and NO
    view measuring geometry AT its depth — is not a real surface.

    Returns (keep_face_mask, report) or None when evidence is unavailable.
    Conservative by design: one at-depth witness keeps the face."""
    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok:
        return None
    verts = np.asarray(mesh_verts, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int64)
    if not len(faces):
        return None
    centers = verts[faces].mean(axis=1)
    valid, at_depth, behind = ev.see_through(centers, depth_tol=depth_tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = behind / np.maximum(valid, 1)
    refuted = (valid >= int(min_votes)) & (at_depth == 0) & \
        (frac >= float(behind_ratio))
    report = {
        "faces": int(len(faces)),
        "refuted": int(refuted.sum()),
        "min_votes": int(min_votes),
        "behind_ratio": float(behind_ratio),
        "depth_tol_m": float(depth_tol),
        "provenance": "tool_measured",
    }
    return ~refuted, report
