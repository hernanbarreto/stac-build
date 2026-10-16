"""The batched silhouette terms (views stacked in chunks, float32): precision, recall, chamfer.

One projection and one distance matrix per chunk of views — the per-view Python loop over 300
views cost a second per iteration on pccr. The maths is the per-view one of ``silhouette``:
precision over the VISIBLE samples outside the mask, recall over every mask boundary pixel
against every sample in front of the camera, both in metres at the depth, the image's pixel
footprint the scale.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .fit import huber

Z_FRONT = 0.05
_D_EPS = 1e-8    # px²: the distances' floor — torch.cdist's backward is NaN at an exact zero
#                  (a sample landing on a boundary pixel), and one NaN sinks the whole fit


def safe_cdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Euclidean distances (c, S, B) between (c, S, 2) and (c, B, 2) with a finite gradient
    everywhere: the squared distance by expansion, floored, then the square root."""
    aa = (A * A).sum(-1, keepdim=True)                 # (c, S, 1)
    bb = (B * B).sum(-1).unsqueeze(1)                  # (c, 1, B)
    ab = torch.bmm(A, B.transpose(1, 2))                # (c, S, B)
    d2 = torch.clamp(aa + bb - 2.0 * ab, min=_D_EPS)
    return torch.sqrt(d2)


class ViewBatch:
    """A chunk of views as stacked tensors on the device."""

    def __init__(self, views: list, dev, lens: Optional[tuple], grids: Optional[tuple]):
        self.views = list(views)
        self.n = len(views)
        self.dev = dev
        self.lens = lens                      # (k1, k2, p1, p2, sx, sy) or None
        self.grids = grids                    # (rg, mg) or None: record → native → mask maps
        f32 = torch.float32
        self.R = torch.stack([v.R.to(f32) for v in views])                 # (c, 3, 3)
        self.t = torch.stack([v.t.to(f32) for v in views])                 # (c, 3)
        self.fx = torch.tensor([v.fx for v in views], dtype=f32, device=dev)
        self.fy = torch.tensor([v.fy for v in views], dtype=f32, device=dev)
        self.cx = torch.tensor([v.cx for v in views], dtype=f32, device=dev)
        self.cy = torch.tensor([v.cy for v in views], dtype=f32, device=dev)
        self.f_px = torch.tensor([v.f_px for v in views], dtype=f32, device=dev)
        self.H, self.W = views[0].hw
        self.mask = torch.stack([v.mask for v in views])                   # (c, H, W) bool
        self.zbuf = torch.stack([v.zbuf.to(f32) for v in views])           # (c, H, W)
        self.ztol = torch.stack([(v.ztol if v.ztol is not None else torch.zeros_like(v.zbuf)).to(f32)
                                 for v in views])
        bmax = max(int(v.boundary.shape[0]) for v in views)
        self.bmax = bmax
        B = torch.zeros((self.n, max(bmax, 1), 2), dtype=f32, device=dev)
        Zb = torch.zeros((self.n, max(bmax, 1)), dtype=f32, device=dev)
        valid = torch.zeros((self.n, max(bmax, 1)), dtype=torch.bool, device=dev)
        for i, v in enumerate(views):
            k = int(v.boundary.shape[0])
            if k:
                B[i, :k] = v.boundary.to(f32)
                Zb[i, :k] = v.z_boundary.to(f32)
                valid[i, :k] = True
        self.B, self.Zb, self.bvalid = B, Zb, valid
        self.sigma_t = None

    def project(self, X: torch.Tensor):
        """World samples (S, 3) float32 → px, py (c, S), z (c, S), front, inb, visible, inside."""
        pcam = torch.einsum("cij,sj->csi", self.R, X) + self.t[:, None, :]   # (c, S, 3)
        z = pcam[..., 2]
        front = z > Z_FRONT
        zs = torch.where(front, z, torch.ones_like(z))
        x = pcam[..., 0] / zs
        y = pcam[..., 1] / zs
        if self.lens is not None:
            k1, k2, p1, p2, sx, sy = self.lens
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2
            xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            u = (self.fx[:, None] * xd + self.cx[:, None]) * sx
            v = (self.fy[:, None] * yd + self.cy[:, None]) * sy
        else:
            u = self.fx[:, None] * x + self.cx[:, None]
            v = self.fy[:, None] * y + self.cy[:, None]
        if self.grids is not None:
            rg, mg = self.grids
            xn = (u - rg.pad_left + 0.5) * rg.scale_x - 0.5 + rg.crop_x
            yn = (v - rg.pad_top + 0.5) * rg.scale_y - 0.5 + rg.crop_y
            px = (xn - mg.crop_x + 0.5) / mg.scale_x - 0.5 + mg.pad_left
            py = (yn - mg.crop_y + 0.5) / mg.scale_y - 0.5 + mg.pad_top
        else:
            px, py = u, v
        inb = front & (px >= 0) & (px <= self.W - 1) & (py >= 0) & (py <= self.H - 1)
        with torch.no_grad():
            ci = torch.clamp(torch.round(px), 0, self.W - 1).to(torch.int64)
            ri = torch.clamp(torch.round(py), 0, self.H - 1).to(torch.int64)
            idx = torch.arange(self.n, device=self.dev)[:, None].expand_as(ci)
            zb = self.zbuf[idx, ri, ci]
            tol = self.sigma_t + z / self.f_px[:, None] + self.ztol[idx, ri, ci]
            visible = inb & ((z <= zb + tol) | ~torch.isfinite(zb))
            inside = self.mask[idx, ri, ci]
        return px, py, z, front & inb, visible, inside

    def terms(self, X: torch.Tensor):
        """Per view: (precision Huber mean, recall Huber mean, chamfer metres, judges) — the
        Huber means differentiable; ``judges`` marks the views that can judge."""
        px, py, z, front, visible, inside = self.project(X)
        P = torch.stack([px, py], -1)                                      # (c, S, 2)
        D = safe_cdist(P, self.B)                                          # (c, S, Bmax)
        inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device)
        D = torch.where(self.bvalid[:, None, :], D, inf)
        # precision: visible samples outside the mask → their distance to the boundary. The
        # footprint is taken at a depth never under Z_FRONT: a sample behind the camera has a
        # negative or zero z, and a NaN in the branch torch.where discards still poisons the
        # gradient (the work surface's plate went NaN at the second step, 2026-10-16)
        d_s = D.min(dim=2).values                                           # (c, S)
        foot_s = torch.clamp(torch.abs(z), min=Z_FRONT) / self.f_px[:, None]
        prec_m = torch.where(visible & ~inside, d_s, torch.zeros_like(d_s)) * foot_s
        n_vis = visible.sum(dim=1)
        prec_h = (torch.where(visible, huber(prec_m / foot_s.detach(), self.k), torch.zeros_like(prec_m)).sum(dim=1)
                  / torch.clamp(n_vis, min=1))
        # recall: every boundary pixel against every sample in front
        Df = torch.where(front[:, :, None], D, inf)
        rec_px = Df.min(dim=1).values                                       # (c, Bmax)
        foot_b = self.Zb / self.f_px[:, None]
        rec_m = torch.where(self.bvalid, rec_px, torch.zeros_like(rec_px)) * foot_b
        n_b = self.bvalid.sum(dim=1)
        rec_h = (torch.where(self.bvalid, huber(rec_m / torch.clamp(foot_b, min=1e-9), self.k), torch.zeros_like(rec_m)).sum(dim=1)
                 / torch.clamp(n_b, min=1))
        judges = (n_vis > 0) & (front.sum(dim=1) > 0) & (n_b > 0)
        with torch.no_grad():
            prec_mean = torch.where(visible, prec_m, torch.zeros_like(prec_m)).sum(1) / torch.clamp(n_vis, min=1)
            rec_mean = torch.where(self.bvalid, rec_m, torch.zeros_like(rec_m)).sum(1) / torch.clamp(n_b, min=1)
            chamfer = 0.5 * (prec_mean + rec_mean)
        return prec_h, rec_h, chamfer, judges
