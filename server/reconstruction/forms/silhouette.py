"""The corrector: the form against the SAM3 masks of EVERY keyframe that sees the object.

USER 2026-10-16: "no solo juez sino corrector … debe iterar hasta encontrar la forma con el mejor
consenso". Analysis by synthesis: the template's surface samples are projected into every view
(the camera the cloud was built with, the lens included), occlusion-verified against the cloud's
z-buffer, and compared with the mask two ways — PRECISION (a visible sample outside the mask: the
form is too big there) and RECALL (a mask boundary pixel far from every visible sample: the form
is too small or a part is missing) — both in METRES at the object's depth, Huber at the IMAGE's own
resolution (one pixel at that depth) while the cloud term is Huber at the session's σ — each
evidence weighed by its own instrument, the two added. Mask boundary pixels
that are OCCLUSION edges (something closer right outside the mask) testify to nothing and are
left out. The optimiser runs in blocks; after each the per-view consensus is measured and the fit
stops when a block no longer improves it beyond the views' own noise (the paired bootstrap of
``metric_lock.heldout_change``). The bar of "coincides" is MEASURED too: the same consensus of
the object's own cloud points against the masks — the instrument's agreement with the images —
and the form is in doubt only when it is clearly worse than that.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .fit import REL_TOL, fit_cloud, huber, subsample
from .templates import Template, _t

Z_FRONT = 0.05   # m: a point closer than this to the camera is behind / at its centre (no pixel)


@dataclass
class View:
    """One keyframe that masks the object: its camera, mask, z-buffer and usable boundary."""
    fidx: int                       # the mask store's frame index
    cloud_frame: int                # the frame the poses are keyed by
    R: torch.Tensor                 # (3, 3) world → camera
    t: torch.Tensor                 # (3,)
    proj: Callable[[torch.Tensor, torch.Tensor], tuple]   # pcam (N,3), front → (px, py) mask-grid pixels
    mask: torch.Tensor              # (H, W) bool
    zbuf: torch.Tensor              # (H, W) float32: the closest measured depth per pixel (inf: none)
    boundary: torch.Tensor          # (B, 2) float64: (col, row) of the usable mask boundary pixels
    z_boundary: torch.Tensor        # (B,) float64: the object's depth at those pixels
    f_px: float                     # focal length on the mask grid (px) — pixels ↔ metres at a depth
    area: int                       # mask pixels
    ztol: Optional[torch.Tensor] = None   # (H, W) float32: the z-buffer's own local spread (see z_spread)

    @property
    def hw(self):
        return int(self.mask.shape[0]), int(self.mask.shape[1])


# ── building views ─────────────────────────────────────────────────────────────

def boundary_pixels(mask: torch.Tensor, zbuf: torch.Tensor, sigma: float, f_px: float,
                    n_max: int, seed: int) -> tuple:
    """(B, 2) (col, row) of the mask's boundary pixels that are NOT occlusion edges, and their
    depth. An occlusion edge: measured geometry right outside the mask (a 3-px ring) is closer
    than the object at the pixel by more than the tolerance (σ or the pixel's footprint) — the
    mask ends because something covers the object, not because the object ends."""
    from precision.f6_torch import erode3, min_filter
    m = mask.to(torch.bool)
    ring = m & ~erode3(m)
    if not bool(ring.any()):
        return torch.zeros((0, 2), dtype=torch.float64, device=mask.device), torch.zeros(0, dtype=torch.float64, device=mask.device)
    zb = zbuf.to(torch.float64)
    inf = torch.full_like(zb, float("inf"))
    z_out = min_filter(torch.where(m, inf, zb), 3)          # the closest measured depth just outside
    # the object's depth at a mask pixel: its own z-buffer; a pixel the cloud does not cover (a
    # hole, the rim the cloud fell short of) takes the median depth of the mask's measured pixels
    measured = m & torch.isfinite(zb)
    med = (torch.median(zb[measured]) if bool(measured.any())
           else torch.tensor(float("nan"), dtype=torch.float64, device=mask.device))
    z_in = torch.where(torch.isfinite(zb), zb, med)
    tol = torch.maximum(torch.full_like(z_in, float(sigma)),
                        torch.where(torch.isfinite(z_in), z_in / float(f_px), torch.zeros_like(z_in)))
    occluded_edge = torch.isfinite(z_out) & torch.isfinite(z_in) & (z_out < z_in - tol)
    usable = ring & ~occluded_edge
    rows, cols = torch.nonzero(usable, as_tuple=True)
    n = int(rows.numel())
    if n == 0:
        return torch.zeros((0, 2), dtype=torch.float64, device=mask.device), torch.zeros(0, dtype=torch.float64, device=mask.device)
    if n > int(n_max):
        # a deterministic stride through the raster order (every k-th), never a random draw
        step = n / float(n_max)
        pick = torch.floor(torch.arange(int(n_max), dtype=torch.float64, device=mask.device) * step).to(torch.int64)
        rows, cols = rows[pick], cols[pick]
    zb_at = z_in[rows, cols]
    return torch.stack([cols.to(torch.float64), rows.to(torch.float64)], 1), zb_at


def z_spread(zbuf: torch.Tensor, size: int = 5) -> torch.Tensor:
    """The z-buffer's own uncertainty per pixel: it holds the CLOSEST measured depth within a
    ``size``-px window (the hole audit's construction), so a surface sample at its true depth
    can sit behind it by the depth variation across that window (a cylinder's flank, a wall seen
    obliquely) plus the noise. The spread = the window's maximum of the buffer minus the buffer;
    0 where nothing is measured (a hole never occludes)."""
    from precision.f6_torch import max_filter
    zb = zbuf.to(torch.float64)
    fin = torch.isfinite(zb)
    zz = torch.where(fin, zb, torch.zeros_like(zb))
    mx = max_filter(torch.where(fin, zb, torch.full_like(zb, -1.0)), int(size))
    sp = torch.where(fin & (mx >= 0), mx - zz, torch.zeros_like(zz))
    return torch.clamp(sp, min=0.0).to(torch.float32)


def pinhole_view(fidx: int, c2w: np.ndarray, K: np.ndarray, mask: np.ndarray, zbuf: np.ndarray,
                 sigma: float, boundary_px: int, seed: int, dev) -> View:
    """A view with a plain pinhole camera on the mask grid (the synthetic tests; the production
    views come from :func:`views_from_evidence`)."""
    c2w4 = np.eye(4)
    c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
    M = np.linalg.inv(c2w4)
    R, t = _t(M[:3, :3], dev), _t(M[:3, 3], dev)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    def proj(pcam: torch.Tensor, front: torch.Tensor):
        z = torch.where(front, pcam[:, 2], torch.ones_like(pcam[:, 2]))
        return fx * pcam[:, 0] / z + cx, fy * pcam[:, 1] / z + cy

    m = torch.as_tensor(np.asarray(mask, bool), device=dev)
    zb = torch.as_tensor(np.asarray(zbuf, np.float32), device=dev)
    b, zb_at = boundary_pixels(m, zb, sigma, fx, boundary_px, seed)
    return View(fidx=int(fidx), cloud_frame=int(fidx), R=R, t=t, proj=proj, mask=m, zbuf=zb,
                boundary=b, z_boundary=zb_at, f_px=fx, area=int(m.sum()), ztol=z_spread(zb))


def views_from_evidence(ev, oids: Sequence[int], sigma: float, max_views: int, boundary_px: int,
                        seed: int, log: Callable[[str], None] = lambda m: None) -> List[View]:
    """The object's views from the hole-audit evidence (the camera the cloud was built with, the
    lens, the record→mask grid maps, the full cloud's z-buffers on the device): every keyframe
    where any of the object's masklets has a mask, the ``max_views`` largest masks first."""
    import re
    dev = ev.device
    mh, mw = ev.mask_hw
    by_pos: Dict[int, List[str]] = {}
    for oid in oids:
        for fidx, key in ev.frames_for(int(oid)):
            by_pos.setdefault(int(fidx), []).append(key)
    entries = []
    for fidx in sorted(by_pos):
        m = None
        for k in sorted(by_pos[fidx]):
            a = torch.as_tensor(np.asarray(ev.masks[k]) > 0, device=dev)
            m = a if m is None else (m | a)
        if m is None or tuple(m.shape) != (mh, mw):
            continue
        area = int(m.sum())
        if area > 0:
            entries.append((area, fidx, m))
    entries.sort(key=lambda e: (-e[0], e[1]))
    rg, mg = ev.record_grid, ev.mask_grid
    views: List[View] = []
    for area, fidx, m in entries[:int(max_views)]:
        cf = ev.cloud_frame(fidx)
        c2w = ev.pose_at(fidx)
        K = ev.K_at(fidx)
        if cf is None or c2w is None or K is None:
            continue
        zb = ev._zbuf_t(int(cf), mh, mw)
        if zb is None:
            continue
        R, t = ev._cam_t(c2w)
        f_px = float(K[0, 0]) * float(rg.scale_x) / float(mg.scale_x)

        def proj(pcam: torch.Tensor, front: torch.Tensor, _cf=int(cf)):
            u, v = ev._record_px_t(_cf, pcam, front)          # record-grid pixels, the lens included
            xn = (u - rg.pad_left + 0.5) * rg.scale_x - 0.5 + rg.crop_x
            yn = (v - rg.pad_top + 0.5) * rg.scale_y - 0.5 + rg.crop_y
            gx = (xn - mg.crop_x + 0.5) / mg.scale_x - 0.5 + mg.pad_left
            gy = (yn - mg.crop_y + 0.5) / mg.scale_y - 0.5 + mg.pad_top
            return gx, gy

        b, zb_at = boundary_pixels(m, zb, sigma, f_px, boundary_px, seed)
        views.append(View(fidx=int(fidx), cloud_frame=int(cf), R=R.to(torch.float64), t=t.to(torch.float64),
                          proj=proj, mask=m, zbuf=zb.to(torch.float32), boundary=b, z_boundary=zb_at,
                          f_px=f_px, area=area, ztol=z_spread(zb)))
    log(f"[forms] {len(views)} view(s) of {len(entries)} with a mask (largest first, "
        f"{sum(int(v.boundary.shape[0]) for v in views)} boundary px)")
    return views


# ── the silhouette term ────────────────────────────────────────────────────────

class Silhouettes:
    def __init__(self, views: List[View], sigma: float, huber_k: float, dev):
        self.views = list(views)
        self.sigma = max(float(sigma), 1e-6)
        self.k = float(huber_k)
        self.dev = dev

    def _project(self, view: View, X: torch.Tensor):
        """World points → (px, py, z, visible, inside): visible = in front, in bounds and not
        behind measured geometry by more than the tolerance (σ or the pixel's footprint)."""
        H, W = view.hw
        pcam = X @ view.R.T + view.t
        z = pcam[:, 2]
        front = z > Z_FRONT
        px, py = view.proj(pcam, front)
        inb = front & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        with torch.no_grad():
            ci = torch.clamp(torch.round(px), 0, W - 1).to(torch.int64)
            ri = torch.clamp(torch.round(py), 0, H - 1).to(torch.int64)
            zb = view.zbuf[ri, ci].to(torch.float64)
            # occluded = measured geometry clearly IN FRONT of the sample: beyond the session's σ,
            # the pixel's footprint and the z-buffer's own local spread (z_spread)
            tol = torch.full_like(z, self.sigma) + z / float(view.f_px)
            if view.ztol is not None:
                tol = tol + view.ztol[ri, ci].to(torch.float64)
            visible = inb & ((z <= zb + tol) | ~torch.isfinite(zb))
            inside = view.mask[ri, ci]
        return px, py, z, visible, inside

    def _terms(self, view: View, X: torch.Tensor):
        """(precision distances of the visible samples, recall distances of the boundary pixels),
        both in metres at the object's depth; None when the view cannot judge."""
        px, py, z, visible, inside = self._project(view, X)
        if int(visible.sum()) == 0 or int(view.boundary.shape[0]) == 0:
            return None, None
        P = torch.stack([px[visible], py[visible]], 1)           # (S, 2)
        D = torch.cdist(P, view.boundary)                        # (S, B) pixels
        d_s = D.min(dim=1).values
        prec_px = torch.where(inside[visible], torch.zeros_like(d_s), d_s)
        foot_s = z[visible] / float(view.f_px)                    # one pixel at the sample's depth
        prec_m = prec_px * foot_s
        rec_px = D.min(dim=0).values                              # (B,)
        foot_b = view.z_boundary / float(view.f_px)
        rec_m = rec_px * foot_b
        return (prec_m, foot_s), (rec_m, foot_b)

    def loss(self, template: Template, n_samples: int) -> torch.Tensor:
        """Mean over the views of (precision + recall) Huber at σ — differentiable in the template."""
        X = template.samples(int(n_samples))
        acc, n = None, 0
        for v in self.views:
            prec, rec = self._terms(v, X)
            if prec is None:
                continue
            # the IMAGE's own resolution is the scale of its evidence: one pixel at the depth of
            # the element (USER: "la nube deforma mucho" — where the masks and the points disagree
            # beyond their noises, the images decide the extents, the points the position)
            (prec_m, foot_s), (rec_m, foot_b) = prec, rec
            term = huber(prec_m / foot_s.detach(), self.k).mean() + huber(rec_m / foot_b, self.k).mean()
            acc = term if acc is None else acc + term
            n += 1
        if acc is None:
            return torch.zeros((), dtype=torch.float64, device=self.dev)
        return acc / n

    @torch.no_grad()
    def chamfer(self, X: torch.Tensor) -> np.ndarray:
        """Per view: the mean precision + recall distance in metres (NaN: the view cannot judge)."""
        out = np.full(len(self.views), np.nan)
        for i, v in enumerate(self.views):
            prec, rec = self._terms(v, X)
            if prec is None:
                continue
            out[i] = 0.5 * (float(prec[0].mean()) + float(rec[0].mean()))
        return out

    @torch.no_grad()
    def iou(self, X: torch.Tensor) -> np.ndarray:
        """Per view: IoU of the visible samples' footprint (each dilated by one pixel) with the
        mask — an APPROXIMATE indicator for the report (a sample footprint, not a rasterised
        surface), never a loss or a verdict."""
        from precision.f6_torch import dilate
        out = np.full(len(self.views), np.nan)
        for i, v in enumerate(self.views):
            H, W = v.hw
            px, py, z, visible, inside = self._project(v, X)
            if int(visible.sum()) == 0:
                continue
            img = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
            ci = torch.clamp(torch.round(px[visible]), 0, W - 1).to(torch.int64)
            ri = torch.clamp(torch.round(py[visible]), 0, H - 1).to(torch.int64)
            img[ri, ci] = True
            img = dilate(img, 1)
            inter = int((img & v.mask).sum())
            union = int((img | v.mask).sum())
            out[i] = inter / union if union else float("nan")
        return out


# ── the judges (measured, never a chosen number) ──────────────────────────────

def _heldout_change(before: np.ndarray, after: np.ndarray, confidence: float, seed: int) -> dict:
    import sys
    from pathlib import Path
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from loop_utils.metric_lock import heldout_change
    ok = np.isfinite(before) & np.isfinite(after)
    return heldout_change(before[ok], after[ok], confidence=float(confidence), seed=int(seed))


def correct(template: Template, P: np.ndarray, sigma: float, sil: Silhouettes, *, lr: float,
            block_iterations: int, max_blocks: int, huber_k: float, max_points: int,
            surface_samples: int, seed: int, dev, confidence: float,
            log: Callable[[str], None] = lambda m: None) -> dict:
    """Optimise the template against the points AND the masks until the per-view consensus
    stops improving beyond its own noise; then judge it against the cloud's own consensus.

    Returns the record: blocks, the consensus per block (median over views, metres), the stop,
    the bar (the cloud's chamfer per view), the verdict ``coincides`` / ``doubt`` / ``undecided``
    with the paired bootstrap that decided it, and the per-view chamfer + IoU at the end."""
    t0 = time.time()
    Pc = _t(subsample(P, surface_samples, seed), dev)
    bar = sil.chamfer(Pc)                                    # the instrument's own agreement
    history: List[np.ndarray] = []
    medians: List[float] = []
    changes: List[dict] = []
    losses: List[float] = []

    def judge(block: int, loss: float) -> bool:
        """Stop when the block neither lowered the joint loss beyond the optimiser's own
        tolerance nor improved the per-view consensus beyond the views' own noise (the paired
        bootstrap over the views: with hundreds of views it is sharp, with a handful the loss
        tolerance carries)."""
        with torch.no_grad():
            X = template.samples(int(surface_samples))
        ch = sil.chamfer(X)
        history.append(ch)
        losses.append(float(loss))
        medians.append(float(np.nanmedian(ch)) if np.isfinite(ch).any() else float("nan"))
        if len(history) < 2:
            return False
        c = _heldout_change(history[-2], history[-1], confidence, seed)
        prev, last = losses[-2], losses[-1]
        loss_moves = (prev - last) > REL_TOL * max(abs(prev), 1e-12)
        changes.append({"block": block, "loss_moves": bool(loss_moves), **c})
        return not (c["improves"] or loss_moves)

    with torch.no_grad():
        start = sil.chamfer(template.samples(int(surface_samples)))
    rec = fit_cloud(template, P, sigma, lr=lr, block_iterations=block_iterations, max_blocks=max_blocks,
                    huber_k=huber_k, max_points=max_points, seed=seed, dev=dev,
                    extra_loss=lambda: sil.loss(template, surface_samples), judge=judge,
                    measure_extents=False, log=log)
    with torch.no_grad():
        X = template.samples(int(surface_samples))
        final = sil.chamfer(X)
        ious = sil.iou(template.samples(int(surface_samples) * 8))
    vs_bar = _heldout_change(bar, final, confidence, seed)
    if vs_bar["worsens"]:
        verdict = "doubt"
    elif vs_bar["improves"] or vs_bar["n"] >= 2:
        verdict = "coincides"
    else:
        verdict = "undecided"
    out = {"blocks": rec["blocks"], "iterations": rec["iterations"], "stop": rec["stop"],
           "n_views": len(sil.views), "n_views_judging": int(np.isfinite(final).sum()),
           "consensus_start_m": float(np.nanmedian(start)) if np.isfinite(start).any() else None,
           "consensus_m": medians, "consensus_final_m": float(np.nanmedian(final)) if np.isfinite(final).any() else None,
           "bar_m": float(np.nanmedian(bar)) if np.isfinite(bar).any() else None,
           "bar_is": "the object's own cloud points against the same masks (the instrument's agreement)",
           "changes": changes, "verdict": verdict, "vs_bar": vs_bar,
           "iou_median": float(np.nanmedian(ious)) if np.isfinite(ious).any() else None,
           "per_view": [{"fidx": v.fidx, "chamfer_m": (None if not np.isfinite(final[i]) else float(final[i])),
                         "bar_m": (None if not np.isfinite(bar[i]) else float(bar[i])),
                         "iou": (None if not np.isfinite(ious[i]) else float(ious[i])),
                         "area": v.area} for i, v in enumerate(sil.views)],
           "seconds": round(time.time() - t0, 2)}
    log(f"[forms] corrector: {out['n_views_judging']}/{len(sil.views)} view(s), consensus "
        f"{(out['consensus_start_m'] or 0) * 100:.2f} → {(out['consensus_final_m'] or 0) * 100:.2f} cm "
        f"(bar {(out['bar_m'] or 0) * 100:.2f} cm), {rec['blocks']} block(s), IoU "
        f"{(out['iou_median'] or 0):.2f} → {verdict}")
    return out


def compare_forms(per_view_a: np.ndarray, per_view_b: np.ndarray, confidence: float, seed: int) -> dict:
    """Is form A's consensus clearly better than form B's over the same views? The paired
    bootstrap of the per-view chamfers: ``a_better`` / ``b_better`` when the whole interval
    sits on one side, else ``tie`` (the views cannot tell — the VLM's first choice stands)."""
    c = _heldout_change(np.asarray(per_view_b, np.float64), np.asarray(per_view_a, np.float64),
                        confidence, seed)
    c["a_better"], c["b_better"] = bool(c["improves"]), bool(c["worsens"])
    c["tie"] = not (c["a_better"] or c["b_better"])
    return c


def per_view_chamfer(record: dict) -> np.ndarray:
    return np.array([(np.nan if p.get("chamfer_m") is None else float(p["chamfer_m"]))
                     for p in record.get("per_view", [])], np.float64)


def worst_views(record: dict, n: int) -> List[int]:
    """The ``n`` views (fidx) that disagree most with the form — the review's overlays."""
    pv = [p for p in record.get("per_view", []) if p.get("chamfer_m") is not None]
    pv.sort(key=lambda p: -(p["chamfer_m"] - (p["bar_m"] or 0.0)))
    return [int(p["fidx"]) for p in pv[:int(n)]]
