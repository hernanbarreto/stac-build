"""The certification's mask filter on the card (USER 2026-11-01 00:30 CDMX, watching it run on one CPU
core: "no es posible hacer esa etapa en GPU?, todo debe hacerse en GPU").

The hot paths of ``correction.visit_drift``'s step 12 — a keyframe's Z-BUFFER (the points born in it,
projected, the nearest depth per mask pixel, 5-px minimum filter), RULE 1's cross-visit tally and RULE 4's
ballot over the views of an object (project, parallax, occlusion, own / other mask lookups) — on torch
tensors on ONE device, with the numpy path kept as the reference:

* the projection is SessionProjection's, in float64 and in its order of operations (numpy's w2c inverse,
  K, the undistorted pixel, the OPENCV lens of cv2.projectPoints, the native → mask-grid map, half-to-even
  rounding = torch.round), so a point lands on the same pixel;
* the z-buffer is a stable sort by (pixel, depth) and the first entry of each pixel — no scatter, no
  atomics (precision.f6_torch.splat's recipe; two runs are bit-identical); the minimum filter is
  precision.f6_torch.min_filter (scipy's mode 'nearest');
* the dilation of a mask is a max-pool over the (2·px + 1)² window with zero padding (cv2.dilate with a
  ones kernel: the border dilates inward only);
* the votes are gathers and sums of 0/1 at UNIQUE indices (deterministic index_put).

Selected by ``correction.visit_drift.device``: numpy (the path of record, what ran until today) | cpu (the
torch path on the CPU — the tests) | cuda. tests/test_visit_drift_torch.py holds the two paths equal vote
for vote. The record's signed edge distances run on the device too (``edge_distance_at``: the exact nearest
boundary pixel, cv2's DIST_MASK_PRECISE result at the query pixels). What stays on the CPU, declared: the
per-view mask unions and counts (``_union`` / ``_objects_over``, from Visibility's decompressed-mask
cache) and the numpy copy of each z-buffer the other rules read.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEVICES = ("numpy", "cpu", "cuda")


def _torch():
    import torch
    return torch


def resolve_device(name: str) -> str:
    """'numpy' | 'cpu' | 'cuda', validated; a cuda with no visible card FAILS naming the key — nothing
    runs somewhere else (the F6 rule, 2026-10-15)."""
    name = str(name)
    if name not in DEVICES:
        raise ValueError(f"correction.visit_drift.device must be one of {list(DEVICES)}, got {name!r}")
    if name == "cuda" and not _torch().cuda.is_available():
        raise RuntimeError("correction.visit_drift.device is 'cuda' and torch sees no card — the mask "
                           "filter does not fall back to the CPU by itself; declare cpu or numpy")
    return name


class TorchProjection:
    """``SessionProjection.to_mask`` on tensors: (ok, rows, cols, z) of world points in a keyframe on
    the mask grid — ``ok`` per point (in front and in frame), the pixel and the depth of the ``ok``
    points only, in point order."""

    def __init__(self, camera, device):
        self.device = device
        K = np.asarray(camera.K, np.float64)
        self.fx, self.fy, self.cx, self.cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        d = np.asarray(camera.cam.dist(), np.float64).reshape(-1)
        if len(d) not in (4, 5):
            raise ValueError(f"the torch projection knows the OPENCV lens (k1 k2 p1 p2 [k3]); the camera "
                             f"carries {len(d)} coefficients")
        k = np.zeros(5)
        k[:len(d)] = d
        self.k1, self.k2, self.p1, self.p2, self.k3 = (float(v) for v in k)
        self.lens = bool(np.any(d))
        g = camera.mask_grid
        self.grid = (float(g.crop_x), float(g.crop_y), float(g.scale_x), float(g.scale_y),
                     float(g.pad_left), float(g.pad_top), int(g.w), int(g.h))

    def to_mask(self, P, pose, min_depth: float):
        torch = _torch()
        c2w = np.eye(4)
        c2w[:3, :4] = np.asarray(pose, np.float64)[:3, :4]
        M = torch.as_tensor(np.linalg.inv(c2w), dtype=torch.float64, device=self.device)   # numpy's inverse
        q = P @ M[:3, :3].T + M[:3, 3]
        z = q[:, 2]
        front = z > float(min_depth)
        zs = torch.where(front, z, torch.ones_like(z))
        # the undistorted native pixel first (SessionProjection.continuous), then the lens as
        # precision.camera.distort_points: normalise back with K, cv2.projectPoints' polynomial, K again
        u = self.fx * (q[:, 0] / zs) + self.cx
        v = self.fy * (q[:, 1] / zs) + self.cy
        if self.lens:
            x = (u - self.cx) / self.fx
            y = (v - self.cy) / self.fy
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            cdist = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            a1 = 2.0 * x * y
            xd = x * cdist + self.p1 * a1 + self.p2 * (r2 + 2.0 * x * x)
            yd = y * cdist + self.p1 * (r2 + 2.0 * y * y) + self.p2 * a1
            u = self.fx * xd + self.cx
            v = self.fy * yd + self.cy
        crop_x, crop_y, sx, sy, pad_l, pad_t, W, H = self.grid
        gx = (u - crop_x + 0.5) / sx - 0.5 + pad_l
        gy = (v - crop_y + 0.5) / sy - 0.5 + pad_t
        c = torch.round(gx)
        r = torch.round(gy)
        inb = (front & torch.isfinite(c) & torch.isfinite(r)
               & (c >= 0) & (c < W) & (r >= 0) & (r < H))
        idx = torch.nonzero(inb, as_tuple=False).reshape(-1)
        return inb, r[idx].to(torch.int64), c[idx].to(torch.int64), z[idx]


def zbuffer(rows, cols, z, hw: Tuple[int, int], min_filter_px: int):
    """The nearest depth per mask pixel (+inf where nothing landed), then the minimum filter: a stable
    sort by (pixel, depth) and the first entry of each pixel — no atomics, bit-identical twice."""
    torch = _torch()
    H, W = int(hw[0]), int(hw[1])
    zb = torch.full((H * W,), float("inf"), dtype=torch.float64, device=z.device)
    if z.numel():
        lin = rows * W + cols
        o1 = torch.argsort(z, stable=True)
        o2 = torch.argsort(lin[o1], stable=True)
        order = o1[o2]
        lin_s, z_s = lin[order], z[order]
        first = torch.ones(lin_s.shape[0], dtype=torch.bool, device=z.device)
        first[1:] = lin_s[1:] != lin_s[:-1]
        zb[lin_s[first]] = z_s[first]
    zb = zb.reshape(H, W)
    if int(min_filter_px) > 1:
        from precision.f6_torch import min_filter
        zb = min_filter(zb, int(min_filter_px))
    return zb


def dilate(mask, px: int):
    """``cv2.dilate`` with a (2·px + 1)² ones kernel on a {0,1} mask: a max-pool with zero padding."""
    torch = _torch()
    if int(px) <= 0:
        return mask
    import torch.nn.functional as F
    m = mask.to(torch.float32)[None, None]
    return F.max_pool2d(m, 2 * int(px) + 1, stride=1, padding=int(px))[0, 0] > 0.5


class TorchVisibility:
    """The torch side of ``correction.visit_drift.Visibility``: the cloud, the consumed observations and
    the camera centres on the device, the z-buffers cached there per keyframe."""

    def __init__(self, vis, device):
        torch = _torch()
        self.vis = vis
        self.device = torch.device(str(device))
        self.proj = TorchProjection(vis.camera, self.device)
        self.xyz = torch.as_tensor(np.asarray(vis.xyz, np.float64), dtype=torch.float64, device=self.device)
        self.extra = (None if getattr(vis, "extra_xyz", None) is None else
                      torch.as_tensor(np.asarray(vis.extra_xyz, np.float64), dtype=torch.float64,
                                      device=self.device))
        self.centres = torch.as_tensor(np.asarray(vis.poses, np.float64)[:, :3, 3], dtype=torch.float64,
                                       device=self.device)
        self._zb: Dict[int, object] = {}

    def project(self, P, kf: int):
        return self.proj.to_mask(P, self.vis.poses[int(kf)], self.vis.min_depth)

    def zbuf(self, kf: int):
        """``Visibility.zbuf`` on the device: the keyframe's OWN points (+ its consumed observations),
        the nearest depth per pixel, 5-px minimum filter. Cached."""
        kf = int(kf)
        if kf in self._zb:
            return self._zb[kf]
        torch = _torch()
        v = self.vis
        own = torch.as_tensor(np.asarray(v._order[v._start[kf]:v._end[kf]], np.int64), device=self.device)
        _ok, r, c, z = self.project(self.xyz[own], kf)
        if self.extra is not None:
            ex = np.asarray(v._eorder[v._estart[kf]:v._eend[kf]], np.int64)
            if len(ex):
                _o2, r2, c2, z2 = self.project(self.extra[torch.as_tensor(ex, device=self.device)], kf)
                r, c, z = torch.cat([r, r2]), torch.cat([c, c2]), torch.cat([z, z2])
        zb = zbuffer(r, c, z, (v.Hm, v.Wm), 5)
        self._zb[kf] = zb
        return zb

    def mask(self, arr: np.ndarray, px: int):
        """A mask of the npz as a bool tensor on the device, dilated by ``px`` there."""
        torch = _torch()
        m = torch.as_tensor(np.ascontiguousarray(np.asarray(arr) > 0), device=self.device)
        return dilate(m, int(px)) if int(px) > 0 else m


def edge_distance_at(mask, rows, cols, chunk: int = 2048):
    """``correction.visit_drift._edge_distance(mask)[rows, cols]`` on the device: the signed Euclidean
    distance (px, + inside) of the query pixels to the edge of ``mask`` — exact, as cv2's
    DIST_MASK_PRECISE: the nearest outside pixel of an inside query is 4-adjacent to the mask (walking
    one step toward the query from any farther outside pixel strictly shortens the distance), so the
    search runs over the boundary pixels only; the root correctly rounded to float32 like cv2's. A mask all set / all clear
    reads ±hypot(H, W), farther than any pixel can be."""
    torch = _torch()
    from precision.f6_torch import shift
    H, W = mask.shape
    far = float(np.hypot(H, W))
    n = int(rows.shape[0])
    n_in = int(mask.sum())
    if n_in == 0 or n_in == H * W:
        return torch.full((n,), far if n_in else -far, dtype=torch.float32, device=mask.device)
    m = mask
    nb_in = shift(m, 1, 0, False) | shift(m, -1, 0, False) | shift(m, 0, 1, False) | shift(m, 0, -1, False)
    b_out = torch.nonzero(~m & nb_in, as_tuple=False)                         # outside, touching the mask
    nm = ~m
    nb_out = shift(nm, 1, 0, False) | shift(nm, -1, 0, False) | shift(nm, 0, 1, False) | shift(nm, 0, -1, False)
    b_in = torch.nonzero(m & nb_out, as_tuple=False)                          # inside, touching the outside
    q = torch.stack([rows, cols], 1).to(torch.int32)
    inside = m[rows, cols]
    out = torch.empty(n, dtype=torch.float32, device=mask.device)

    def nearest(qs, bnd):
        best = torch.empty(qs.shape[0], dtype=torch.int32, device=mask.device)
        b = bnd.to(torch.int32)
        for i in range(0, qs.shape[0], chunk):
            qq = qs[i:i + chunk]
            d2 = (qq[:, None, 0] - b[None, :, 0]) ** 2 + (qq[:, None, 1] - b[None, :, 1]) ** 2
            best[i:i + chunk] = d2.min(1).values
        # the correctly rounded float32 of √n (cv2's sqrtf): torch's float32 sqrt on the CPU is 1 ulp off
        # on ~13 % of the values (measured), the float64 root cast down is exact for these integers
        return torch.sqrt(best.to(torch.float64)).to(torch.float32)

    if bool(inside.any()):
        out[inside] = nearest(q[inside], b_out)
    if bool((~inside).any()):
        out[~inside] = -nearest(q[~inside], b_in)
    return out.clamp(-far, far)


def ballot_record_torch(tv: TorchVisibility, xyz: np.ndarray, pts: np.ndarray, ks: np.ndarray,
                        rule1_views, own_views: Dict[int, np.ndarray], rival_masks, dilate_px: int,
                        occlusion_tol_rel: float, min_tri_deg: float) -> List[Tuple[np.ndarray, ...]]:
    """``correction.visit_drift._ballot_record``'s blocks (pt, kf, rule, d_own, d_other, inside, other)
    for the deleted points, judged and measured on the device."""
    torch = _torch()
    dev = tv.device
    P = np.asarray(xyz, np.float64)[np.asarray(pts, np.int64)]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
    ks_t = torch.as_tensor(np.asarray(ks, np.int64), device=dev)
    tol = float(tv.vis.tol)
    blocks: List[Tuple[np.ndarray, ...]] = []
    for (a, b, kf, mm) in rule1_views:
        w = torch.nonzero((ks_t < int(a)) | (ks_t > int(b)), as_tuple=False).reshape(-1)
        if not w.numel():
            continue
        ok, r, c, z = tv.project(Pt[w], int(kf))
        if not r.numel():
            continue
        zpix = tv.zbuf(kf)[r, c]
        q = torch.nonzero(~(zpix < (z - tol)), as_tuple=False).reshape(-1)
        if not q.numel():
            continue
        dm = tv.mask(mm, int(dilate_px))
        rq, cq = r[q], c[q]
        nq = int(q.shape[0])
        blocks.append((w[ok][q].cpu().numpy().astype(np.int64), np.full(nq, int(kf), np.int32),
                       np.full(nq, 1, np.uint8), edge_distance_at(dm, rq, cq).cpu().numpy().astype(np.float32),
                       np.full(nq, np.nan, np.float32), dm[rq, cq].cpu().numpy(), np.zeros(nq, bool)))
    if own_views and rival_masks is not None:
        rec4: list = []
        other_mask_votes_torch(tv, P, ks, own_views, rival_masks, dilate_px, occlusion_tol_rel, min_tri_deg,
                               record=rec4)
        for blk in rec4:
            m_ = len(blk["pt"])
            blocks.append((np.asarray(blk["pt"], np.int64), np.asarray(blk["kf"], np.int32),
                           np.full(m_, 4, np.uint8), np.asarray(blk["d_own"], np.float32),
                           np.asarray(blk["d_other"], np.float32), np.asarray(blk["own"], bool),
                           np.asarray(blk["other"], bool)))
    return blocks


def rule1_votes_torch(tv: TorchVisibility, P: np.ndarray, ks: np.ndarray,
                      views: Sequence[Tuple[int, int, int, np.ndarray]], dilate_px: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """``correction.visit_drift.rule1_votes`` on the device: per point, the other visits' frames that
    saw it (nothing measured in front, ``vis.tol``) and those with it inside the dilated mask."""
    torch = _torch()
    dev = tv.device
    n = len(P)
    Pt = torch.as_tensor(np.asarray(P, np.float64), dtype=torch.float64, device=dev)
    ks_t = torch.as_tensor(np.asarray(ks, np.int64), device=dev)
    n_saw = torch.zeros(n, dtype=torch.int32, device=dev)
    n_inside = torch.zeros(n, dtype=torch.int32, device=dev)
    tol = float(tv.vis.tol)
    for (a, b, kf, mm) in views:
        w = torch.nonzero((ks_t < int(a)) | (ks_t > int(b)), as_tuple=False).reshape(-1)
        if not w.numel():
            continue
        ok, r, c, z = tv.project(Pt[w], int(kf))
        if not r.numel():
            continue
        zpix = tv.zbuf(kf)[r, c]
        visible = ~(zpix < (z - tol))
        hit = tv.mask(mm, int(dilate_px))[r, c]
        wo = w[ok]
        n_saw[wo] += visible.to(torch.int32)
        n_inside[wo] += (visible & hit).to(torch.int32)
    return n_saw.cpu().numpy(), n_inside.cpu().numpy()


def other_mask_votes_torch(tv: TorchVisibility, P: np.ndarray, birth: np.ndarray,
                           own_views: Dict[int, np.ndarray],
                           rival_masks: Callable[[int], Optional[np.ndarray]],
                           dilate_px: int, occlusion_tol_rel: float, min_tri_deg: float,
                           record: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``correction.visit_drift.other_mask_votes`` (RULE 4's ballot) on the device — the same votes."""
    torch = _torch()
    dev = tv.device
    n = len(P)
    Pt = torch.as_tensor(np.asarray(P, np.float64), dtype=torch.float64, device=dev)
    birth_t = torch.as_tensor(np.asarray(birth, np.int64), device=dev)
    n_votes = torch.zeros(n, dtype=torch.int32, device=dev)
    n_own = torch.zeros(n, dtype=torch.int32, device=dev)
    n_other = torch.zeros(n, dtype=torch.int32, device=dev)
    cos_max = float(np.cos(np.radians(float(min_tri_deg))))
    tol = float(occlusion_tol_rel)
    for kf in sorted(own_views):
        w = torch.nonzero(birth_t != int(kf), as_tuple=False).reshape(-1)   # the birth keyframe never votes
        if not w.numel():
            continue
        ok, r, c, z = tv.project(Pt[w], int(kf))
        if not r.numel():
            continue
        wo = w[ok]
        a = tv.centres[birth_t[wo]] - Pt[wo]
        b = tv.centres[int(kf)][None, :] - Pt[wo]
        cos = (a * b).sum(1) / torch.clamp(torch.linalg.norm(a, dim=1) * torch.linalg.norm(b, dim=1),
                                           min=1e-9)
        zb = tv.zbuf(kf)[r, c]
        measured = torch.isfinite(zb)
        occluded = measured & (z > zb * (1.0 + tol))
        on_surface = measured & (torch.abs(z - zb) <= tol * zb)
        vote = (torch.abs(cos) <= cos_max) & ~occluded
        own_t = tv.mask(own_views[kf], int(dilate_px))
        own = own_t[r, c]
        fm = rival_masks(int(kf))
        fm_t = tv.mask(fm, 0) if fm is not None else None
        other = ((~own) & on_surface & fm_t[r, c]) if fm_t is not None \
            else torch.zeros(wo.shape[0], dtype=torch.bool, device=dev)
        n_votes[wo] += vote.to(torch.int32)
        n_own[wo] += (vote & own).to(torch.int32)
        n_other[wo] += (vote & other).to(torch.int32)
        if record is not None and bool(vote.any()):
            # the ballot's record (points 106 / 133): the signed pixel distances of every counting vote
            # to the edge of the own dilated mask and of the other objects' mask, on the device
            v = torch.nonzero(vote, as_tuple=False).reshape(-1)
            rv, cv = r[v], c[v]
            d_own = edge_distance_at(own_t, rv, cv).cpu().numpy()
            d_oth = (edge_distance_at(fm_t, rv, cv).cpu().numpy() if fm_t is not None
                     else np.full(int(v.shape[0]), np.nan, np.float32))
            record.append({"pt": wo[v].cpu().numpy(), "kf": np.full(int(v.shape[0]), int(kf), np.int32),
                           "d_own": d_own.astype(np.float32), "d_other": d_oth.astype(np.float32),
                           "own": own[v].cpu().numpy(), "other": other[v].cpu().numpy()})
    return n_votes.cpu().numpy(), n_own.cpu().numpy(), n_other.cpu().numpy()
