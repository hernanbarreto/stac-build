"""F6's map arithmetic on the card — strict torch, one device (USER 2026-10-08).

USER 2026-10-08: *"todos los pasos que son CPU y pueden ser GPU estricto deben ser gpu estricto …
la premisa es que cada paso sea lo más veloz posible, manteniendo la calidad y el determinismo"*.
MEASURED on zaragoza 2026-06-03 (183 keyframes at 1520x848, 15 tiles each): F6's numpy/scipy phases
took 741 s, PointDiT 3371 s tile by tile (log server_20261007_210725). Here every map operation of
mono_detail (the per-tile affine fit, the blend, the lowpass, the band, the sides) and of the
edge-keeping vote (splat, projection, medians, snapping) runs on torch tensors on ONE device chosen by
the stage: the card (:func:`require_cuda`, the pipeline) or the CPU (the synthetic tests,
:func:`use_device`) — the same code path on both, never a silent fallback: a stage on CUDA that finds
torch's deterministic mode off FAILS (:func:`_strict`).

What keeps it deterministic on the card, under ``torch.use_deterministic_algorithms(True)`` STRICT
(repro.deterministic_torch: an op without a deterministic kernel RAISES):
- no scatter / index_put at duplicate indices anywhere: the z-buffer of a splat is a stable SORT by
  (pixel, depth) and the first entry of each pixel (:func:`splat`); dense maps are built with
  ``torch.where``, never written at indices;
- medians and percentiles are sort-based (:func:`quantile`, :func:`nanmedian0`), never
  ``torch.median`` (its CUDA kernel is in torch's non-deterministic list);
- the Gaussian lowpass is two matrix products with explicit boundary matrices (:func:`gauss_matrix`;
  cuBLAS with the pinned workspace, TF32 off) — scipy's ``gaussian_filter(mode='nearest')`` exactly in
  exact arithmetic;
- dilations, erosions and min/max filters are ``max_pool2d`` on padded maps; gathers only.
The bits are NOT those of the numpy recipe (other summation orders): the same input gives the same
bytes on the same card, driver and torch — the identity the stage records next to its result
(:func:`record`)."""
from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MAD_TO_SIGMA = 1.0 / 0.6744897501960817   # σ of a normal from its MAD (1.4826)
GAUSS_TRUNCATE = 4.0                       # scipy.ndimage.gaussian_filter's default: kernel radius int(4σ + 0.5)

# per-pixel provenance of the refined depth (mono_detail re-exports them)
SRC_OMEGA = 0            # Omega bent (no tile covered it, or a band pixel on a surface)
SRC_DETAIL = 1           # lowpass(Omega) + PointDiT detail
SRC_BAND_FRONT = 2       # a mixed pixel resolved to the FRONT surface
SRC_BAND_BACK = 3        # … to the BACK surface
SRC_UNRESOLVED = 4       # mixed, PointDiT intermediate too → excluded from the measurement tier

_DEVICE = "cpu"


class F6TorchError(RuntimeError):
    pass


# ── the device ──────────────────────────────────────────────────────────────

def use_device(dev: str) -> str:
    """The ONE device of every function here ('cpu' in the synthetic tests; the stage sets the card
    through :func:`require_cuda`)."""
    global _DEVICE
    _DEVICE = str(dev)
    return _DEVICE


def device() -> str:
    return _DEVICE


def require_cuda(seed: int) -> dict:
    """The stage entry: a CUDA card or FAIL (no CPU fallback — other kernels, other bits), torch's
    deterministic mode STRICT for the whole process (repro.enable_deterministic_torch, seeded), the
    device set to the card. Returns :func:`record`."""
    if not torch.cuda.is_available():
        raise F6TorchError("F6 runs its maps on a CUDA card only and none is visible (no CPU fallback)")
    from repro import enable_deterministic_torch
    enable_deterministic_torch(int(seed))
    use_device("cuda")
    _strict()
    return record()


def record() -> dict:
    """What the bits of every map here depend on: device, card, torch and its numerics flags."""
    cuda = _DEVICE.startswith("cuda") and torch.cuda.is_available()
    return {"device": _DEVICE, "torch": torch.__version__,
            "card": torch.cuda.get_device_name(0) if cuda else None,
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32)}


def _strict() -> None:
    if _DEVICE.startswith("cuda") and not (torch.are_deterministic_algorithms_enabled()
                                           and not torch.is_deterministic_algorithms_warn_only_enabled()):
        raise F6TorchError("F6 on CUDA needs torch's deterministic mode STRICT (repro.enable_deterministic_torch)"
                           " — it is off or warn-only")


def _t(a, dtype=None) -> torch.Tensor:
    """numpy (or tensor) → a tensor on the device; float maps as float64 unless asked otherwise."""
    if isinstance(a, torch.Tensor):
        t = a.to(_DEVICE)
        return t if dtype is None else t.to(dtype)
    a = np.ascontiguousarray(a)
    if dtype is None:
        if a.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(a.dtype, np.floating):
            dtype = torch.float64
        elif np.issubdtype(a.dtype, np.integer):
            dtype = torch.int64
    return torch.as_tensor(a, device=_DEVICE).to(dtype)


def _np(t: torch.Tensor, dtype=None) -> np.ndarray:
    a = t.detach().cpu().numpy()
    return a if dtype is None else a.astype(dtype)


# ── order statistics (sort-based: deterministic on every device) ────────────

def quantile(v: torch.Tensor, qs: Sequence[float]) -> torch.Tensor:
    """numpy's 'linear' percentile of the FINITE entries of ``v`` (1-D), by sort — no size limit,
    NaN when nothing is finite."""
    v = v.reshape(-1).to(torch.float64)
    v = v[torch.isfinite(v)]
    q = torch.as_tensor([float(x) for x in qs], dtype=torch.float64, device=v.device)
    n = v.numel()
    if n == 0:
        return torch.full_like(q, float("nan"))
    s = torch.sort(v).values
    pos = q * (n - 1)
    lo = torch.floor(pos).long().clamp(0, n - 1)
    hi = torch.clamp(lo + 1, max=n - 1)
    frac = pos - lo.to(torch.float64)
    return s[lo] * (1.0 - frac) + s[hi] * frac


def median(v: torch.Tensor) -> torch.Tensor:
    return quantile(v, [0.5])[0]


def nanmedian0(C: torch.Tensor) -> torch.Tensor:
    """np.nanmedian(C, 0) for C [V, ...]: the mean of the two middle FINITE values of each column, NaN
    where none is finite. Sort-based (NaN sorts last in torch)."""
    V = C.shape[0]
    if V == 0:
        return torch.full(C.shape[1:], float("nan"), dtype=torch.float64, device=C.device)
    C = C.to(torch.float64)
    s = torch.sort(C, dim=0).values
    k = torch.isfinite(C).sum(0)
    k1 = torch.clamp(k, min=1)
    lo = ((k1 - 1) // 2).unsqueeze(0)
    hi = (k1 // 2).unsqueeze(0)
    vlo = torch.gather(s, 0, lo)[0]
    vhi = torch.gather(s, 0, hi)[0]
    med = 0.5 * (vlo + vhi)
    return torch.where(k == 0, torch.full_like(med, float("nan")), med)


def margin_quantiles(x: torch.Tensor) -> Optional[dict]:
    """The distribution of a per-pixel MARGIN to a bar (positive = on the accepted side), as the
    reports record it (docs/plan_determinismo.md point 53): n and the 5 / 25 / 50 / 75 / 95 %
    quantiles. None when nothing was judged."""
    v = torch.as_tensor(x).reshape(-1).to(torch.float64)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return None
    q = quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    return {"n": int(v.numel()), "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p95": float(q[4]), "share_beyond": float((v < 0).to(torch.float64).mean().item())}


# ── neighbourhood filters (max_pool2d on padded maps) ────────────────────────

def _pool_max(x: torch.Tensor, size: int, pad_value: float, mode: str = "constant") -> torch.Tensor:
    """max over the (size x size) window of a 2-D map; outside the image the map reads ``pad_value``
    ('constant') or its own border ('replicate' = scipy's mode 'nearest')."""
    r = size // 2
    x4 = x.to(torch.float64)[None, None]
    if mode == "replicate":
        x4 = F.pad(x4, (r, r, r, r), mode="replicate")
    else:
        x4 = F.pad(x4, (r, r, r, r), mode="constant", value=float(pad_value))
    return F.max_pool2d(x4, size, stride=1)[0, 0]


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    """scipy binary_dilation with a (2r+1)^2 square structure, border_value 0."""
    return _pool_max(mask.to(torch.float64), 2 * int(r) + 1, 0.0) > 0.5


def erode3(mask: torch.Tensor) -> torch.Tensor:
    """scipy binary_erosion with the 3x3 structure, border_value 0 (the border row is never interior)."""
    inv = 1.0 - mask.to(torch.float64)
    return _pool_max(inv, 3, 1.0) < 0.5


def max_filter(x: torch.Tensor, size: int) -> torch.Tensor:
    """scipy maximum_filter(size, mode='nearest')."""
    return _pool_max(x, size, 0.0, mode="replicate")


def min_filter(x: torch.Tensor, size: int) -> torch.Tensor:
    """scipy minimum_filter(size, mode='nearest')."""
    return -_pool_max(-x.to(torch.float64), size, 0.0, mode="replicate")


def shift(a: torch.Tensor, dy: int, dx: int, fill) -> torch.Tensor:
    """out[r, c] = a[r + dy, c + dx]; ``fill`` where that falls outside the image (|dy|, |dx| ≤ 1)."""
    H, W = a.shape
    pad = F.pad(a.to(torch.float64)[None, None], (1, 1, 1, 1), mode="constant", value=float(fill))[0, 0]
    out = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    return out.to(a.dtype) if a.dtype != torch.bool else out > 0.5


# ── the Gaussian lowpass as matrix products ─────────────────────────────────

@functools.lru_cache(maxsize=32)
def _gauss_matrix_np(n: int, sigma: float) -> np.ndarray:
    """G [n, n]: scipy.ndimage.gaussian_filter1d(sigma, mode='nearest', truncate 4.0) as a matrix —
    out = G @ x. The boundary taps that fall outside are folded onto the edge sample (mode 'nearest')."""
    radius = int(GAUSS_TRUNCATE * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= k.sum()
    G = np.zeros((n, n), np.float64)
    ii = np.repeat(np.arange(n), len(x))
    jj = np.clip(np.tile(np.arange(-radius, radius + 1), n) + ii, 0, n - 1)
    np.add.at(G, (ii, jj), np.tile(k, n))        # sequential on the CPU: deterministic
    return G


@functools.lru_cache(maxsize=32)
def _gauss_matrix(n: int, sigma: float, dev: str) -> torch.Tensor:
    return torch.as_tensor(_gauss_matrix_np(int(n), float(sigma)), device=dev)


def gauss_matrix(n: int, sigma: float) -> torch.Tensor:
    return _gauss_matrix(int(n), float(sigma), _DEVICE)


def gaussian_filter(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """scipy gaussian_filter(z, sigma, mode='nearest') on a 2-D map: Gh @ z @ Gw^T."""
    H, W = z.shape
    Gh, Gw = gauss_matrix(H, sigma), gauss_matrix(W, sigma)
    return Gh @ z.to(torch.float64) @ Gw.T


def masked_lowpass(z: torch.Tensor, mask: torch.Tensor, sigma_px: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(lowpass, ok): Gaussian lowpass of ``z`` over the ``mask`` pixels only (normalised by the mask's
    own blur), and where it is defined (at least half the kernel's weight on measured pixels)."""
    m = mask.to(torch.float64)
    num = gaussian_filter(torch.where(mask, z.to(torch.float64), torch.zeros_like(m)), sigma_px)
    den = gaussian_filter(m, sigma_px)
    ok = den > 0.5
    lp = torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)), torch.zeros_like(den))
    return lp, ok


# ── resize ──────────────────────────────────────────────────────────────────

def resize(img: torch.Tensor, h: int, w: int, nearest: bool = False) -> torch.Tensor:
    """A 2-D map or an H x W x C image to (h, w): bilinear with half-pixel centres (cv2 INTER_LINEAR's
    convention) or nearest."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    x = img.to(torch.float64)
    x4 = x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]
    if nearest:
        y = F.interpolate(x4, size=(int(h), int(w)), mode="nearest")
    else:
        y = F.interpolate(x4, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    y = y[0, 0] if x.ndim == 2 else y[0].permute(1, 2, 0)
    return y > 0.5 if img.dtype == torch.bool else y.to(img.dtype)


# ── the per-tile affine fit (Phase 3) ───────────────────────────────────────

def irls_affine(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, huber_k: float, iterations: int) -> Tuple[float, float]:
    """y ≈ s·x + b by Huber IRLS from the weighted least-squares start (weights ``w`` ≥ 0 = the
    session's calibrated confidence), the construction of epoch 7's bend fit. Two unknowns: the
    weighted normal equations in closed form about the weighted mean (no lstsq kernel), every sum a
    plain reduction."""
    x = x.to(torch.float64); y = y.to(torch.float64)
    sw = torch.sqrt(torch.clamp(w.to(torch.float64), min=0.0))
    ww = sw.clone()
    s = torch.zeros((), dtype=torch.float64, device=x.device)
    b = torch.zeros((), dtype=torch.float64, device=x.device)
    has = sw > 0
    for _ in range(int(iterations)):
        w2 = ww * ww
        S = w2.sum()
        if float(S) <= 0:
            break
        mx = (w2 * x).sum() / S
        my = (w2 * y).sum() / S
        dx = x - mx
        sxx = (w2 * dx * dx).sum()
        sxy = (w2 * dx * (y - my)).sum()
        s = torch.where(sxx > 0, sxy / torch.where(sxx > 0, sxx, torch.ones_like(sxx)), torch.zeros_like(sxx))
        b = my - s * mx
        e = y - (s * x + b)
        sig = MAD_TO_SIGMA * median(torch.abs(e)[has]) + 1e-12
        ww = sw * torch.sqrt(torch.clamp(float(huber_k) * sig / torch.clamp(torch.abs(e), min=1e-12), max=1.0))
    return float(s), float(b)


def align_tile(z_cal: torch.Tensor, z_mono: torch.Tensor, support: torch.Tensor, weight: torch.Tensor,
               fit_space: str, huber_k: float, iterations: int) -> Tuple[torch.Tensor, float, float, float]:
    """(z_al, s, b, residual): PointDiT's tile mapped onto the calibrated depth. ``fit_space`` 'depth':
    z_cal ≈ s·z_mono + b; 'inverse': 1/z_cal ≈ s·z_mono + b. z_al (float32) is 0 where the map is not
    positive; residual = MAD (as σ) of the relative residual on the support."""
    x = z_mono.to(torch.float64)[support]
    yz = z_cal.to(torch.float64)[support]
    w = weight.to(torch.float64)[support]
    zm = z_mono.to(torch.float64)
    if fit_space == "inverse":
        s, b = irls_affine(x, 1.0 / yz, w, huber_k, iterations)
        den = s * zm + b
        z_al = torch.where(den > 1e-9, 1.0 / torch.where(den > 1e-9, den, torch.ones_like(den)), torch.zeros_like(den))
    elif fit_space == "depth":
        s, b = irls_affine(x, yz, w, huber_k, iterations)
        z_al = s * zm + b
    else:
        raise F6TorchError(f"fit_space must be 'depth' or 'inverse', got {fit_space!r}")
    z_al = torch.where(z_al > 0, z_al, torch.zeros_like(z_al)).to(torch.float32)
    r = (yz - z_al.to(torch.float64)[support]) / yz
    residual = float(MAD_TO_SIGMA * median(torch.abs(r - median(r)))) if r.numel() else float("nan")
    return z_al, s, b, residual


# ── discontinuities, band, sides (Phases 4 and 6) ───────────────────────────

def relative_jump(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per pixel the largest 1-pixel relative depth jump |Δz| / min(z) to a valid 4-neighbour."""
    H, W = z.shape
    zz = z.to(torch.float64)
    out = torch.zeros((H, W), dtype=torch.float64, device=z.device)
    for dy, dx in ((0, 1), (1, 0)):
        a = zz[:H - dy, :W - dx]; b = zz[dy:, dx:]
        va = valid[:H - dy, :W - dx] & valid[dy:, dx:]
        j = torch.where(va, torch.abs(a - b) / torch.clamp(torch.minimum(a, b), min=1e-9), torch.zeros_like(a))
        pad = torch.zeros((H, W), dtype=torch.float64, device=z.device)
        pad[:H - dy, :W - dx] = j
        out = torch.maximum(out, pad)
        pad2 = torch.zeros((H, W), dtype=torch.float64, device=z.device)
        pad2[dy:, dx:] = j
        out = torch.maximum(out, pad2)
    return out


def discontinuity_band(z: torch.Tensor, valid: torch.Tensor, tau: float, band_px: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(discontinuity, band): the pixels where a 1-pixel jump of ``z`` exceeds ``tau`` (relative), and
    that set dilated by ``band_px`` (≥ 1) restricted to the valid pixels."""
    disc = valid & (relative_jump(z, valid) > float(tau))
    band = dilate(disc, max(1, int(band_px))) & valid
    return disc, band


def side_depths(z_cal: torch.Tensor, usable: torch.Tensor, reach_px: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(z_front, z_back): the nearest and the farthest calibrated depth among the ``usable`` (valid,
    non-band) pixels within ``reach_px`` of each pixel (inf / −inf where there is none)."""
    size = 2 * int(reach_px) + 1
    z = z_cal.to(torch.float64)
    big = torch.where(usable, z, torch.full_like(z, float("inf")))
    small = torch.where(usable, z, torch.full_like(z, float("-inf")))
    return min_filter(big, size), max_filter(small, size)


def refine_frame(z_cal: torch.Tensor, valid_cal: torch.Tensor, z_al: torch.Tensor, al_valid: torch.Tensor,
                 tau: float, band_px: int, sigma_px: float, reach_px: int, align_err: float = 0.0,
                 detail_scope: str = "edges", detail_zone_px: int = 12) -> dict:
    """One keyframe's refined depth from its calibrated (bent) map and PointDiT's aligned map — the
    arithmetic of mono_detail.refine_frame (see its docstring for the rules and the user's 'edges'
    scope) on tensors. Returns dict(depth float32, source uint8, band, detail float32, counts, margins)."""
    H, W = z_cal.shape
    zc = z_cal.to(torch.float64)
    za = z_al.to(torch.float64)
    zero = torch.zeros((H, W), dtype=torch.float64, device=zc.device)
    both = valid_cal & al_valid & (za > 0) & (zc > 0)
    disc, band = discontinuity_band(za, both, tau, band_px)
    out = torch.where(valid_cal, zc, zero)
    src = torch.zeros((H, W), dtype=torch.int64, device=zc.device)
    detail = zero.clone()
    # Phase 4: outside the band, Omega's coarse + PointDiT's fine
    surf = both & ~band
    lp_c, ok_c = masked_lowpass(zc, surf, sigma_px)
    lp_a, ok_a = masked_lowpass(za, surf, sigma_px)
    m = surf & ok_c & ok_a
    d = za - lp_a
    z_new = lp_c + d
    m = m & (z_new > 0)
    margins: Dict[str, object] = {}
    if detail_scope == "edges":
        r = max(1, int(detail_zone_px))
        zone = dilate(disc, r)
        zc64 = torch.clamp(zc, min=1e-9)
        rel_change = torch.abs(z_new - zc) / zc64
        within = rel_change <= float(tau)
        judged = m & zone
        margins["detail_within_tau"] = margin_quantiles(
            (float(tau) - rel_change[judged]) / max(float(tau), 1e-12)) if bool(judged.any()) else None
        m = m & zone & within
    elif detail_scope != "surfaces":
        raise F6TorchError(f"detail_scope must be 'edges' or 'surfaces', got {detail_scope!r}")
    out = torch.where(m, z_new, out)
    detail = torch.where(m, d, detail)
    src = torch.where(m, torch.full_like(src, SRC_DETAIL), src)
    # Phase 6: the band — mixed pixels of the calibrated depth go to a side, never in between
    usable = valid_cal & (zc > 0) & ~band
    zf, zb = side_depths(zc, usable, int(band_px) + int(reach_px))
    distinct = band & torch.isfinite(zf) & torch.isfinite(zb) & (zb > zf * (1.0 + tau))
    mixed = distinct & valid_cal & (zc > zf * (1.0 + tau)) & (zc < zb * (1.0 - tau))
    margin = max(float(tau), float(align_err))
    near_f = torch.abs(za - zf) <= margin * zf
    near_b = torch.abs(za - zb) <= margin * zb
    judged = mixed & al_valid & (za > 0)
    to_front = judged & near_f & ~(near_b & (torch.abs(za - zb) < torch.abs(za - zf)))
    to_back = judged & near_b & ~to_front
    unresolved = judged & ~to_front & ~to_back
    out = torch.where(to_front, zf, out)
    out = torch.where(to_back, zb, out)
    out = torch.where(unresolved, zero, out)
    src = torch.where(to_front, torch.full_like(src, SRC_BAND_FRONT), src)
    src = torch.where(to_back, torch.full_like(src, SRC_BAND_BACK), src)
    src = torch.where(unresolved, torch.full_like(src, SRC_UNRESOLVED), src)
    # point 53: the margins of the band tests, in units of their bar (positive = on the side taken)
    dv = distinct & valid_cal
    if bool(dv.any()):
        one = torch.ones_like(zc)
        inside = torch.minimum((zc - zf * (1.0 + tau)) / torch.where(dv, zf * tau, one),
                               (zb * (1.0 - tau) - zc) / torch.where(dv, zb * tau, one))
        margins["mixed_inside_gap"] = margin_quantiles(inside[dv])
    if bool(judged.any()):
        one = torch.ones_like(zc)
        side = torch.minimum(torch.abs(za - zf) / torch.where(judged, margin * zf, one),
                             torch.abs(za - zb) / torch.where(judged, margin * zb, one))
        margins["band_side_within_margin"] = margin_quantiles(1.0 - side[judged])
    counts = {"valid": int(valid_cal.sum()), "covered": int(both.sum()), "detail": int(m.sum()),
              "band": int(band.sum()), "mixed": int(mixed.sum()), "band_front": int(to_front.sum()),
              "band_back": int(to_back.sum()), "mixed_unresolved": int(unresolved.sum()),
              "mixed_unjudged": int((mixed & ~judged).sum())}
    return {"depth": out.to(torch.float32), "source": src.to(torch.uint8), "band": band,
            "detail": detail.to(torch.float32), "counts": counts, "margins": margins}


# ── projections, splats and the vote (corrected_cloud's pure pieces) ────────

@functools.lru_cache(maxsize=8)
def _rays_cached(H: int, W: int, fx: float, fy: float, cx: float, cy: float, dev: str) -> torch.Tensor:
    u = torch.arange(W, dtype=torch.float64, device=dev)
    v = torch.arange(H, dtype=torch.float64, device=dev)
    x = ((u - cx) / fx)[None, :].expand(H, W)
    y = ((v - cy) / fy)[:, None].expand(H, W)
    return torch.stack([x, y, torch.ones((H, W), dtype=torch.float64, device=dev)], -1).contiguous()


def camera_rays(H: int, W: int, K: np.ndarray) -> torch.Tensor:
    """[H, W, 3] camera-frame rays through every pixel centre, (u − cx)/fx, (v − cy)/fy, 1."""
    K = np.asarray(K, np.float64)
    return _rays_cached(int(H), int(W), float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), _DEVICE)


def project(X: torch.Tensor, K: np.ndarray, H: int, W: int):
    """Camera-frame points [..., 3] → (u, v) long pixel indices (round half to even, like np.rint),
    their depth and the in-image / in-front mask."""
    zd = X[..., 2]
    ok = zd > 0
    zs = torch.where(ok, zd, torch.ones_like(zd))
    u = torch.round(float(K[0, 0]) * X[..., 0] / zs + float(K[0, 2])).long()
    v = torch.round(float(K[1, 1]) * X[..., 1] / zs + float(K[1, 2])).long()
    ok = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, zd, ok


def splat(depth, enter, K: np.ndarray, c2w_src: np.ndarray, w2c_dst: np.ndarray, shape) -> torch.Tensor:
    """The surface one keyframe sees, carried into another keyframe's image: per pixel the nearest
    depth (z-buffer), +inf where nothing lands. The z-buffer is a stable SORT by (pixel, depth) with a
    sentinel entry per pixel, and the first entry of each pixel — no scatter, no atomics."""
    H, W = int(shape[0]), int(shape[1])
    d = _t(depth); e = _t(enter, torch.bool)
    K = np.asarray(K, np.float64)
    rays = camera_rays(d.shape[0], d.shape[1], K)
    Rs = _t(c2w_src[:3, :3]); ts = _t(c2w_src[:3, 3])
    Rd = _t(w2c_dst[:3, :3]); td = _t(w2c_dst[:3, 3])
    idx = e.reshape(-1).nonzero()[:, 0]
    z = d.reshape(-1)[idx]
    r = rays.reshape(-1, 3)[idx]
    Xw = (r * z[:, None]) @ Rs.T + ts
    Xd = Xw @ Rd.T + td
    u, v, zd, ok = project(Xd, K, H, W)
    p = (v * W + u)[ok]
    zp = zd[ok]
    n_pix = H * W
    p_all = torch.cat([torch.arange(n_pix, dtype=torch.int64, device=d.device), p])
    z_all = torch.cat([torch.full((n_pix,), float("inf"), dtype=torch.float64, device=d.device), zp])
    o1 = torch.sort(z_all, stable=True).indices
    p1 = p_all[o1]; z1 = z_all[o1]
    o2 = torch.sort(p1, stable=True).indices
    p2 = p1[o2]; z2 = z1[o2]
    first = torch.ones_like(p2, dtype=torch.bool)
    first[1:] = p2[1:] != p2[:-1]
    pos = first.nonzero()[:, 0]                      # exactly H*W entries, in pixel order (the sentinels)
    return z2[pos].reshape(H, W)


def window_extremes(depth: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(min, max) of the VALID depths of each pixel's 3x3 window (inf / 0 where it holds none)."""
    z = depth.to(torch.float64)
    lo = torch.where(valid, z, torch.full_like(z, float("inf")))
    hi = torch.where(valid, z, torch.zeros_like(z))
    mn = -_pool_max(-lo, 3, float("-inf"))
    mx = _pool_max(hi, 3, 0.0)
    return mn, mx


def mixed_pixels(depth: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    mn, mx = window_extremes(depth, valid)
    z = depth.to(torch.float64)
    return valid & (z > mn * (1.0 + tau)) & (z < mx * (1.0 - tau))


def depth_steps(depth: torch.Tensor, valid: torch.Tensor, tau: float) -> torch.Tensor:
    mn, mx = window_extremes(depth, valid)
    return valid & (mx * (1.0 - tau) > mn * (1.0 + tau))


_NB8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))   # 4-neighbours first


def snap_mixed(depth: torch.Tensor, valid: torch.Tensor, labels: torch.Tensor, tau: float):
    """corrected_cloud.snap_mixed on tensors (see its docstring): returns (depth, mixed, snapped)."""
    H, W = depth.shape
    z = depth.to(torch.float64)
    mixed = mixed_pixels(z, valid, tau)
    snapped = torch.zeros((H, W), dtype=torch.bool, device=z.device)
    if not bool(mixed.any()):
        return depth.clone(), mixed, snapped
    mn, mx = window_extremes(z, valid)
    lab = labels.to(torch.int64)
    good = valid & ~mixed
    nan = torch.full((H, W), float("nan"), dtype=torch.float64, device=z.device)
    Z = []
    for dy, dx in _NB8:
        ok = shift(good, dy, dx, 0.0) & (shift(lab, dy, dx, -1) == lab) & (lab >= 0)
        Z.append(torch.where(ok, shift(z, dy, dx, float("nan")), nan))
    Z = torch.stack(Z)                                         # [8, H, W]
    has = torch.isfinite(Z)
    Z0 = torch.where(has, Z, torch.zeros_like(Z))
    near = has & (Z0 - mn[None] <= mx[None] - Z0)
    far = has & ~near
    to_near = near.any(0) & ~far.any(0)
    to_far = far.any(0) & ~near.any(0)
    decided = (to_near | to_far) & mixed
    if not bool(decided.any()):
        return depth.clone(), mixed, snapped
    pick = torch.where(to_near[None], near, far) & decided[None]
    four = pick[:4].any(0)                                     # a 4-neighbour is nearer than a diagonal
    is4 = (torch.arange(len(_NB8), device=z.device) < 4)[:, None, None]
    sel = torch.where(four[None], pick & is4, pick)
    z_new = nanmedian0(torch.where(sel, Z, nan[None].expand_as(Z)))
    out = torch.where(decided, z_new, z).to(depth.dtype)
    return out, mixed, decided


def two_sided_vote(i: int, order: List[int], depth: Dict[int, np.ndarray], judge: Dict[int, np.ndarray],
                   cand, K: np.ndarray, c2w: Dict[int, np.ndarray], w2c: Dict[int, np.ndarray],
                   neighbors, tau: float) -> dict:
    """corrected_cloud.two_sided_vote (see its docstring) on DENSE maps: every entry is an [H, W]
    tensor over the candidate pixels (``cand``), ``splats`` [V, H, W] (NaN where nothing landed;
    [0, H, W] without a neighbour). The compact (rr, cc) view is corrected_cloud's wrapper."""
    f = order[i]
    H, W = depth[f].shape
    K = np.asarray(K, np.float64)
    z = _t(depth[f])
    candt = _t(cand, torch.bool)
    rays = camera_rays(H, W, K)
    Rf = _t(c2w[f][:3, :3]); C = _t(c2w[f][:3, 3])
    ray = rays @ Rf.T                                          # world directions
    one = torch.ones_like(z); zero = torch.zeros_like(z)
    nan = torch.full_like(z, float("nan"))
    agree = torch.zeros((H, W), dtype=torch.int32, device=z.device)
    contra = torch.zeros_like(agree); contra_fwd = torch.zeros_like(agree)
    cands, splats = [z], []
    Xw = C + z[..., None] * ray
    for d in neighbors:
        j = i + int(d)
        if not 0 <= j < len(order):
            continue
        g = order[j]
        T = _t(w2c[g])
        a = T[2, :3] @ C + T[2, 3]
        b = ray @ T[2, :3]
        Xg = Xw @ T[:3, :3].T + T[:3, 3]
        u, v, zg, ok = project(Xg, K, H, W)
        ok = ok & candt
        idx = (v.clamp(0, H - 1) * W + u.clamp(0, W - 1)).reshape(-1)
        dg = torch.where(ok, _t(depth[g]).reshape(-1)[idx].reshape(H, W), zero)
        jg = _t(judge[g], torch.bool).reshape(-1)[idx].reshape(H, W)
        ok = ok & jg & (dg > 0)
        e = torch.where(ok, (zg - dg) / torch.where(ok, dg, one), zero)
        ag = ok & (torch.abs(e) <= tau) & (torch.abs(b) > 0)
        fwd = ok & (e < -tau)
        s = splat(depth[g], judge[g], K, c2w[g], w2c[f], (H, W))
        landed = torch.isfinite(s)
        s0 = torch.where(landed, s, zero)
        front = landed & (z - s0 > tau * s0)
        agree = agree + ag.to(torch.int32)
        contra_fwd = contra_fwd + fwd.to(torch.int32)
        contra = contra + (fwd | front).to(torch.int32)
        cands.append(torch.where(ag, (dg - a) / torch.where(torch.abs(b) > 0, b, one), nan))
        splats.append(torch.where(landed, s, nan))
    zmed = nanmedian0(torch.stack(cands))
    sp = torch.stack(splats) if splats else torch.zeros((0, H, W), dtype=torch.float64, device=z.device)
    return {"z": z, "agree": agree, "contra": contra, "contra_fwd": contra_fwd, "zmed": zmed, "splats": sp}


def agreeing_median(C: torch.Tensor, tau: float, min_views: int):
    """Per column of C [V, ...] (NaN = that view put nothing there): the views within tau (relative)
    of the column's median and, where at least ``min_views`` of them agree, their median.
    Returns (ok, depth (NaN where not ok), n_agree)."""
    if C.ndim < 2 or C.shape[0] == 0 or C[0].numel() == 0:
        shape = C.shape[1:] if C.ndim >= 2 else (0,)
        return (torch.zeros(shape, dtype=torch.bool, device=C.device),
                torch.full(shape, float("nan"), dtype=torch.float64, device=C.device),
                torch.zeros(shape, dtype=torch.int32, device=C.device))
    C = C.to(torch.float64)
    nan = torch.full_like(C, float("nan"))
    C = torch.where(torch.isfinite(C), C, nan)
    med = nanmedian0(C)
    agree = torch.abs(C - med[None]) <= tau * med[None]
    n_ag = agree.sum(0).to(torch.int32)
    ok = (n_ag >= int(min_views)) & torch.isfinite(med)
    zz = nanmedian0(torch.where(agree, C, nan))
    zz = torch.where(ok, zz, torch.full_like(zz, float("nan")))
    return ok, zz, n_ag


def measured_tau(depth: Dict[int, np.ndarray], enter: Dict[int, np.ndarray], K: np.ndarray,
                 c2w: Dict[int, np.ndarray], order: List[int], neighbors, quantile_pct: float,
                 stride: int = 6) -> float:
    """The session's own neighbour disagreement (|z_i→j − d_j| / d_j over pixels both keep), its
    ``quantile_pct`` percentile — corrected_cloud.measured_tau on the card."""
    w2c = {f: np.linalg.inv(c2w[f]) for f in order}
    samp = []
    for i in range(0, len(order), int(stride)):
        f = order[i]
        for d in neighbors:
            j = i + int(d)
            if not 0 <= j < len(order):
                continue
            g = order[j]
            s = splat(depth[f], enter[f], K, c2w[f], w2c[g], depth[g].shape)
            dg = _t(depth[g])
            m = torch.isfinite(s) & _t(enter[g], torch.bool)
            if bool(m.any()):
                samp.append((torch.abs(s[m] - dg[m]) / dg[m])[::7])
    if not samp:
        return float("nan")
    return float(quantile(torch.cat(samp), [float(quantile_pct) / 100.0])[0])


def interior(passed: torch.Tensor) -> torch.Tensor:
    """Floor-passing pixels whose whole 3x3 window passed too."""
    return erode3(passed)
