"""Mono detail — PointDiT refines the BENT Omega depth inside f6_bend (claude_stac.txt, 2026-10-04).

USER 2026-10-04: *"pointdit tiene mucha mejor nitidez en el mapa de profundidad monocular que da3 y que
omega, lo quiero usar para perfeccionar esos mapas, no como voto, sino como perfección"* and *"la métrica
es la de da3"*. So, per keyframe, between the bend (Omega × k on the DA3 gauge, precision/depth_on_f5.py
step 2) and the edge-keeping vote (step 4):

  Phase 2  TILES at native resolution: ``tile_px`` windows on the camera grid, ``tile_overlap_frac`` of
           overlap, the last tile flush with the border; ``context_scale`` 1.0 feeds the native tile,
           2.0 a window twice as large downscaled to the tile (more context, coarser detail). Feather
           weights (linear ramps over the overlaps, full weight at the frame border) sum to 1.
  Phase 3  AFFINE per tile: z_cal ≈ s·z_mono + b (``fit_space`` depth) or 1/z_cal ≈ s·z_mono + b
           (inverse), Huber IRLS weighted by the session's calibrated confidence, on support pixels
           outside the discontinuity band (two passes: the band needs the aligned map). A tile with
           less than ``min_support_frac`` of support, or a residual above the ``tile_residual_quantile``
           percentile of the session's own tiles, is REJECTED: its pixels keep Omega. Accepted tiles
           are feathered after alignment → z_al, PointDiT's depth in metres.
  Phase 4  DETAIL = z_al − lowpass(z_al); the lowpass sigma derives from the effective resolution of
           the calibrated depth (Omega's patch on the camera grid × ``lowpass_patch_fraction``).
           DISCONTINUITY = a 1-pixel relative jump of z_al above the session's own agreement tolerance
           (the vote's τ); the BAND is that map dilated by the ratio of PointDiT's effective pixel to
           the native one (``context_scale``) — never narrower than one pixel.
  Phase 6  BAND pixels: front and back = the nearest and farthest calibrated depth of the non-band
           pixels within ``side_reach_px``. A pixel whose calibrated depth is INTERMEDIATE (a mixed
           pixel) goes to the side PointDiT's aligned depth is nearer to — front or back, never in
           between; when PointDiT is itself intermediate beyond the session's margin (the vote's τ, or
           the frame's own alignment error if larger) the pixel is ``mixed_unresolved``: marked,
           excluded from the measurement tier (it never becomes a point), never silently kept.
           A band pixel already ON a surface keeps Omega's depth.
  Result   outside the band: z = lowpass(z_cal) + detail — Omega in the coarse, PointDiT in the fine;
           metric and gauge untouched (the affine map per tile removes PointDiT's own scale).
           The edge-keeping vote then runs on the refined maps exactly as in epoch 8.

Per-pixel provenance (``SRC_*``) reaches the cloud's ``source`` column; the per-tile fits, the band
accounting and the timing reach ``depth_on_f5.json`` (``mono_detail``). With ``mono_detail.enabled``
false nothing here runs and the output is epoch 8's bit for bit.

USER 2026-10-08 (speed with determinism): every map operation here runs on torch tensors on the
device of :mod:`precision.f6_torch` (the card in the pipeline, strict deterministic mode; the CPU in
the synthetic tests), and PointDiT runs ALL the tiles of a keyframe as one batch
(``PointDiTRunner.depth_batch``, ``mono_detail.tile_batch``). The numpy-facing functions below keep
their signatures (the tests and loger_pdit use them) and convert at the boundary.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from precision import f6_torch as FT
from precision.f6_torch import (SRC_BAND_BACK, SRC_BAND_FRONT, SRC_DETAIL, SRC_OMEGA,  # noqa: F401 — re-exported
                                SRC_UNRESOLVED)

LOG_TAG = "[mono-detail]"
PATCH = 16
MAD_TO_SIGMA = FT.MAD_TO_SIGMA              # σ of a normal from its MAD (1.4826)
INLIER_K = 3                              # a declared SUMMARY: support within k·residual (never a gate)

SRC_NAMES = ("omega_bent", "mono_detail", "band_front", "band_back", "mixed_unresolved")


class MonoDetailError(RuntimeError):
    pass


# ── Phase 2: tiles ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    h: int               # native window height
    w: int               # native window width
    run_h: int           # the size PointDiT runs at (multiples of 16)
    run_w: int


def _positions(n: int, size: int, stride: int) -> List[int]:
    if size >= n:
        return [0]
    pos = list(range(0, n - size, stride))
    if not pos or pos[-1] != n - size:
        pos.append(n - size)
    return pos


def _down16(v: int) -> int:
    return max(PATCH, (int(v) // PATCH) * PATCH)


def plan_tiles(H: int, W: int, tile_px: int, overlap_frac: float, context_scale: float) -> List[Tile]:
    """The windows that cover an H x W frame. The native window side is ``tile_px × context_scale``
    (clipped to the frame, a multiple of 16); PointDiT runs it at ``window / context_scale``
    (a multiple of 16 too). ``overlap_frac`` of each window is shared with the next."""
    if tile_px < PATCH or not (0.0 <= overlap_frac < 1.0) or context_scale < 1.0:
        raise MonoDetailError(f"tile_px >= 16, 0 <= overlap < 1, context_scale >= 1 required "
                              f"(got {tile_px}, {overlap_frac}, {context_scale})")
    win = int(round(tile_px * context_scale))
    wh = _down16(min(win, H)); ww = _down16(min(win, W))
    sh = max(1, int(round(wh * (1.0 - overlap_frac)))); sw = max(1, int(round(ww * (1.0 - overlap_frac))))
    tiles = []
    for y0 in _positions(H, wh, sh):
        for x0 in _positions(W, ww, sw):
            rh = _down16(int(round(wh / context_scale))); rw = _down16(int(round(ww / context_scale)))
            tiles.append(Tile(y0=y0, x0=x0, h=wh, w=ww, run_h=rh, run_w=rw))
    return tiles


def feather_weights(H: int, W: int, tiles: Sequence[Tile]) -> List[np.ndarray]:
    """Per tile its weight over the FRAME (H x W): 1 inside, a linear ramp to 0 over the overlap with a
    neighbour, full weight at a frame border. Normalised so the weights sum to 1 on every pixel."""
    ws = []
    for t in tiles:
        wy = np.ones(t.h); wx = np.ones(t.w)
        # the ramp length = the overlap with the neighbours on that side (another tile's window reaches in)
        for arr, a0, size, n in ((wy, t.y0, t.h, H), (wx, t.x0, t.w, W)):
            ov_lo = max((o.y0 + o.h if arr is wy else o.x0 + o.w) - a0 for o in tiles
                        if ((o.y0 < t.y0) if arr is wy else (o.x0 < t.x0))) if any(
                ((o.y0 < t.y0) if arr is wy else (o.x0 < t.x0)) for o in tiles) else 0
            ov_hi = max((a0 + size) - (o.y0 if arr is wy else o.x0) for o in tiles
                        if ((o.y0 > t.y0) if arr is wy else (o.x0 > t.x0))) if any(
                ((o.y0 > t.y0) if arr is wy else (o.x0 > t.x0)) for o in tiles) else 0
            ov_lo = int(np.clip(ov_lo, 0, size)); ov_hi = int(np.clip(ov_hi, 0, size))
            if ov_lo > 0:
                arr[:ov_lo] = np.minimum(arr[:ov_lo], (np.arange(ov_lo) + 1) / (ov_lo + 1))
            if ov_hi > 0:
                arr[size - ov_hi:] = np.minimum(arr[size - ov_hi:], (np.arange(ov_hi, 0, -1)) / (ov_hi + 1))
        full = np.zeros((H, W))
        full[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] = wy[:, None] * wx[None, :]
        ws.append(full)
    total = np.sum(ws, axis=0)
    if not np.all(total > 0):
        raise MonoDetailError("the tiles do not cover the frame")
    return [w / total for w in ws]


def resize_to(img: np.ndarray, h: int, w: int, nearest: bool = False) -> np.ndarray:
    """Bilinear (or nearest) resize of a 2-D map / H x W x C image to (h, w)."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    out = FT._np(FT.resize(FT._t(img), int(h), int(w), nearest=nearest))
    return out.astype(img.dtype)


# ── Phase 3: affine per tile ─────────────────────────────────────────────────

@dataclass
class TileFit:
    tile: Tile
    s: float = float("nan")
    b: float = float("nan")
    residual: float = float("nan")     # MAD of the relative residual (z_cal − z_al) / z_cal on the support
    inlier_ratio: float = 0.0          # share of the support within 3·residual (declared summary)
    support: int = 0
    support_frac: float = 0.0
    accepted: bool = False
    reason: str = ""


def irls_affine(x: np.ndarray, y: np.ndarray, w: np.ndarray, huber_k: float, iterations: int) -> Tuple[float, float]:
    """y ≈ s·x + b by Huber IRLS from the weighted least-squares start (weights ``w`` ≥ 0 = the
    session's calibrated confidence), the construction of epoch 7's bend fit."""
    return FT.irls_affine(FT._t(np.asarray(x, np.float64)), FT._t(np.asarray(y, np.float64)),
                          FT._t(np.asarray(w, np.float64)), huber_k, iterations)


def align_tile(z_cal: np.ndarray, z_mono: np.ndarray, support: np.ndarray, weight: np.ndarray,
               fit_space: str, huber_k: float, iterations: int) -> Tuple[np.ndarray, float, float, float]:
    """(z_al, s, b, residual): PointDiT's tile mapped onto the calibrated depth. ``fit_space`` 'depth':
    z_cal ≈ s·z_mono + b; 'inverse': 1/z_cal ≈ s·z_mono + b. z_al is 0 where the map is not positive."""
    try:
        z_al, s, b, res = FT.align_tile(FT._t(z_cal), FT._t(z_mono), FT._t(support, FT.torch.bool),
                                        FT._t(weight), fit_space, huber_k, iterations)
    except FT.F6TorchError as e:
        raise MonoDetailError(str(e)) from e
    return FT._np(z_al, np.float32), s, b, res


# ── Phase 4: discontinuities, band, lowpass, detail ──────────────────────────

def relative_jump(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per pixel the largest 1-pixel relative depth jump |Δz| / min(z) to a valid 4-neighbour."""
    return FT._np(FT.relative_jump(FT._t(z), FT._t(valid, FT.torch.bool)))


def discontinuity_band(z: np.ndarray, valid: np.ndarray, tau: float, band_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """(discontinuity, band): the pixels where a 1-pixel jump of ``z`` exceeds ``tau`` (relative), and
    that set dilated by ``band_px`` (≥ 1) restricted to the valid pixels."""
    disc, band = FT.discontinuity_band(FT._t(z), FT._t(valid, FT.torch.bool), tau, band_px)
    return FT._np(disc), FT._np(band)


def masked_lowpass(z: np.ndarray, mask: np.ndarray, sigma_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """(lowpass, ok): Gaussian lowpass of ``z`` over the ``mask`` pixels only (normalised by the mask's
    own blur), and where it is defined (enough mask weight under the kernel)."""
    lp, ok = FT.masked_lowpass(FT._t(z), FT._t(mask, FT.torch.bool), sigma_px)
    return FT._np(lp), FT._np(ok)


def lowpass_sigma_px(omega_patch_px: int, native_px_per_omega_px: float, patch_fraction: float) -> float:
    """The lowpass scale derived from the calibrated depth's effective resolution: Omega's patch on the
    camera grid, times ``patch_fraction``."""
    return max(0.5, float(omega_patch_px) * float(native_px_per_omega_px) * float(patch_fraction))


def band_half_width_px(context_scale: float) -> int:
    """PointDiT's effective pixel over the native one, never under one pixel."""
    return max(1, int(math.ceil(float(context_scale))))


# ── Phase 6 + the refined map ────────────────────────────────────────────────

def side_depths(z_cal: np.ndarray, usable: np.ndarray, reach_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """(z_front, z_back): the nearest and the farthest calibrated depth among the ``usable`` (valid,
    non-band) pixels within ``reach_px`` of each pixel (inf / −inf where there is none)."""
    zf, zb = FT.side_depths(FT._t(z_cal), FT._t(usable, FT.torch.bool), reach_px)
    return FT._np(zf), FT._np(zb)


@dataclass
class FrameRefinement:
    depth: np.ndarray                  # the refined map (0 = no depth: invalid or mixed_unresolved)
    source: np.ndarray                 # SRC_* per pixel (uint8)
    band: np.ndarray                   # the discontinuity band
    detail: np.ndarray                 # the detail term applied (0 elsewhere)
    counts: Dict[str, int] = field(default_factory=dict)
    margins: Dict[str, object] = field(default_factory=dict)   # per-pixel margins to the bars (point 53)


def margin_quantiles(x: np.ndarray) -> Optional[dict]:
    """The distribution of a per-pixel MARGIN to a bar (positive = on the accepted side), as the
    report records it (docs/plan_determinismo.md point 53: the bars stay — the validated recipe — and
    every margin to them is written down): n and the 5 / 25 / 50 / 75 / 95 % quantiles. None when
    nothing was judged."""
    return FT.margin_quantiles(FT._t(np.asarray(x, np.float64).ravel()))


def _refinement(fr: dict) -> FrameRefinement:
    return FrameRefinement(depth=FT._np(fr["depth"], np.float32), source=FT._np(fr["source"], np.uint8),
                           band=FT._np(fr["band"]), detail=FT._np(fr["detail"], np.float32),
                           counts=fr["counts"], margins=fr["margins"])


def refine_frame(z_cal: np.ndarray, valid_cal: np.ndarray, z_al: np.ndarray, al_valid: np.ndarray,
                 tau: float, band_px: int, sigma_px: float, reach_px: int, align_err: float = 0.0,
                 detail_scope: str = "edges", detail_zone_px: int = 12) -> FrameRefinement:
    """One keyframe's refined depth from its calibrated (bent) map and PointDiT's aligned map.
    ``tau``: the session's agreement tolerance (relative); ``align_err``: the frame's own alignment
    error (relative MAD) — the band margin is the larger of the two.

    ``detail_scope`` (USER 2026-10-04, after the first end-to-end run: PointDiT's detail over every
    surface bent walls and shed points off a rack, while ducts and cables Omega smeared came out —
    "lo que habíamos dicho que iba a hacer era mejorar los filos"):
      'edges'     the detail is applied ONLY within ``detail_zone_px`` of a discontinuity of the
                  aligned map (edges, ducts, cables, thin structures: every pixel of a thin object
                  lies near its own two discontinuities) and only where it stays within the
                  session's own tolerance (|detail| ≤ τ·z); every other pixel keeps Omega untouched.
      'surfaces'  the detail everywhere PointDiT covers (the first run's recipe, kept for the A/B)."""
    try:
        fr = FT.refine_frame(FT._t(z_cal), FT._t(valid_cal, FT.torch.bool), FT._t(z_al), FT._t(al_valid, FT.torch.bool),
                             tau, band_px, sigma_px, reach_px, align_err, detail_scope, detail_zone_px)
    except FT.F6TorchError as e:
        raise MonoDetailError(str(e)) from e
    return _refinement(fr)


# ── the stage: every keyframe, two passes (fit all tiles, then the session's residual bar) ─

@dataclass
class StageReport:
    tiles: int = 0
    accepted: int = 0
    rejected_support: int = 0
    rejected_residual: int = 0
    residual_bar: float = float("nan")
    per_frame: Dict[int, dict] = field(default_factory=dict)
    totals: Dict[str, int] = field(default_factory=dict)
    seconds_pointdit: float = 0.0
    seconds_total: float = 0.0
    params: Dict[str, object] = field(default_factory=dict)
    unresolved: Dict[int, tuple] = field(default_factory=dict)   # frame → (rows, cols, Omega's depth there)


def _batches(n: int, size: int) -> List[Tuple[int, int]]:
    """The tiles of a keyframe in batches of ``size`` (0 = all of them at once), in tile order."""
    if size <= 0 or size >= n:
        return [(0, n)]
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def run_stage(frames: Sequence[int], dep: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
              weight: Dict[int, np.ndarray], image_of: Callable[[int], np.ndarray], runner, mcfg,
              tau: float, native_px_per_omega_px: float, huber_k: float, log: Callable = print,
              progress: Optional[Callable[[float, str], None]] = None,
              overlay_dir=None) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], StageReport]:
    """Refine every keyframe's bent depth IN PLACE semantics: returns (refined depth maps, source maps,
    report). ``image_of(f)`` gives the RGB frame on the camera grid; ``weight[f]`` the calibrated
    confidence weight in 0..1 (0 under the floor); ``tau`` the session's agreement tolerance.
    PointDiT runs the tiles of a keyframe as ONE batch (``runner.depth_batch``; ``mcfg.tile_batch``
    0 = all of them, otherwise batches of that many, a bound of the card — the batch composition is
    part of the run's config, so it is the same on every run)."""
    import torch
    t_start = time.time()
    first = dep[frames[0]]
    H, W = first.shape
    tiles = plan_tiles(H, W, int(mcfg.tile_px), float(mcfg.tile_overlap_frac), float(mcfg.context_scale))
    weights = feather_weights(H, W, tiles)
    weights_t = [FT._t(w) for w in weights]
    band_px = band_half_width_px(float(mcfg.context_scale))
    sigma_px = lowpass_sigma_px(PATCH, native_px_per_omega_px, float(mcfg.lowpass_patch_fraction))
    tile_batch = int(getattr(mcfg, "tile_batch", 0))
    rep = StageReport(params={"detail_scope": str(mcfg.detail_scope), "detail_zone_px": int(mcfg.detail_zone_px),
                              "tiles_per_frame": len(tiles), "tile": [tiles[0].h, tiles[0].w],
                              "run_size": [tiles[0].run_h, tiles[0].run_w], "band_px": band_px,
                              "lowpass_sigma_px": round(sigma_px, 2), "tau": tau, "fit_space": mcfg.fit_space,
                              "context_scale": float(mcfg.context_scale), "tile_batch": tile_batch,
                              "maps_numerics": FT.record()})
    log(f"{LOG_TAG} {len(frames)} keyframes × {len(tiles)} tile(s) of {tiles[0].w}x{tiles[0].h} px run at "
        f"{tiles[0].run_w}x{tiles[0].run_h} (context {mcfg.context_scale:g}); band ±{band_px} px, lowpass σ "
        f"{sigma_px:.1f} px, τ {tau * 100:.2f} %; tiles per PointDiT batch {tile_batch or len(tiles)}; maps on "
        f"{FT.device()}")
    # pass 1: PointDiT on every tile (the keyframe's tiles as one batch), the affine fit, the aligned
    # tile kept for the blend
    fits: Dict[int, List[TileFit]] = {}
    aligned: Dict[int, List[Optional[np.ndarray]]] = {}
    t_pd = 0.0
    for n, f in enumerate(frames):
        img = image_of(f)
        if img.shape[0] != H or img.shape[1] != W:
            raise MonoDetailError(f"frame {f} is {img.shape[1]}x{img.shape[0]}, the depth grid {W}x{H}")
        fits[f] = []; aligned[f] = []
        zc = FT._t(dep[f]); vc = FT._t(valid[f], torch.bool) & (zc > 0); wc = FT._t(weight[f])
        # every tile of a frame runs at the same size (plan_tiles); PointDiT sees them as one batch
        mono: List[Tuple[np.ndarray, np.ndarray]] = []
        for a, b in _batches(len(tiles), tile_batch):
            crops = [img[t.y0:t.y0 + t.h, t.x0:t.x0 + t.w] for t in tiles[a:b]]
            sizes = {(t.run_h, t.run_w) for t in tiles[a:b]}
            if len(sizes) != 1:
                raise MonoDetailError(f"the tiles of frame {f} run at different sizes {sorted(sizes)} — one batch, one size")
            t0 = time.time()
            zb, vb = runner.depth_batch(crops, size=next(iter(sizes)))
            t_pd += time.time() - t0
            mono.extend((np.asarray(zb[k]), np.asarray(vb[k])) for k in range(len(crops)))
        for t, (z_mono_run, v_run) in zip(tiles, mono):
            sl = (slice(t.y0, t.y0 + t.h), slice(t.x0, t.x0 + t.w))
            z_mono = FT._t(z_mono_run)
            v_mono = FT._t(v_run, torch.bool)
            if (t.run_h, t.run_w) != (t.h, t.w):
                z_mono = FT.resize(z_mono, t.h, t.w)
                v_mono = FT.resize(v_mono, t.h, t.w, nearest=True)
            tf = TileFit(tile=t)
            sup = vc[sl] & v_mono & (wc[sl] > 0)
            tf.support = int(sup.sum()); tf.support_frac = tf.support / float(t.h * t.w)
            if tf.support_frac < float(mcfg.min_support_frac) or tf.support < 3:
                tf.reason = "support"; fits[f].append(tf); aligned[f].append(None); continue
            try:
                z_al, s, b, res = FT.align_tile(zc[sl], z_mono, sup, wc[sl], mcfg.fit_space, huber_k, int(mcfg.irls_iterations))
                # second pass: the band of the aligned map is no support for the fit
                _, band = FT.discontinuity_band(z_al, sup & (z_al > 0), tau, band_px)
                sup2 = sup & ~band
                if int(sup2.sum()) >= 3:
                    z_al, s, b, res = FT.align_tile(zc[sl], z_mono, sup2, wc[sl], mcfg.fit_space, huber_k, int(mcfg.irls_iterations))
                    supj = sup2
                else:
                    supj = sup
            except FT.F6TorchError as e:
                raise MonoDetailError(str(e)) from e
            tf.s, tf.b, tf.residual = s, b, res
            r = torch.abs((zc[sl][supj] - z_al.to(torch.float64)[supj]) / zc[sl][supj])
            tf.inlier_ratio = float((r <= INLIER_K * max(res, 1e-9)).to(torch.float64).mean()) if r.numel() else 0.0
            fits[f].append(tf); aligned[f].append(FT._np(z_al, np.float32))
        if progress is not None and (n % 10 == 0 or n == len(frames) - 1):
            progress(50 * (n + 1) / len(frames), f"PointDiT {n + 1}/{len(frames)} keyframes")
    rep.seconds_pointdit = round(t_pd, 1)
    # the session's own residual bar
    all_res = np.array([tf.residual for fl in fits.values() for tf in fl if np.isfinite(tf.residual)])
    if len(all_res) == 0:
        raise MonoDetailError("no tile had enough support for an affine fit — nothing can be aligned")
    bar = float(np.percentile(all_res, float(mcfg.tile_residual_quantile)))
    rep.residual_bar = bar
    # point 53: the bars this stage judges with — the validated recipe's, written down as such,
    # with every tile's and pixel's margin to them below
    rep.params["bars"] = {"tile_residual_bar_rel": bar,
                          "tile_residual_quantile": float(mcfg.tile_residual_quantile),
                          "tile_residual_bar_is": "the session's own percentile of its tile residuals "
                                                  "(self-referential: about 1 - q of the tiles fall above it)",
                          "min_support_frac": float(mcfg.min_support_frac), "tau": float(tau),
                          "tau_is": "bend.tau_quantile percentile of the session's neighbour disagreement"}
    # pass 2: accept / reject, blend, refine
    out_dep: Dict[int, np.ndarray] = {}
    out_src: Dict[int, np.ndarray] = {}
    totals: Dict[str, int] = {}
    tile_margins: List[float] = []
    support_margins: List[float] = []
    for n, f in enumerate(frames):
        zc = FT._t(dep[f]); vc = FT._t(valid[f], torch.bool) & (zc > 0)
        num = torch.zeros((H, W), dtype=torch.float64, device=zc.device); den = torch.zeros_like(num)
        per = {"tiles": []}
        for tf, z_al, wmap in zip(fits[f], aligned[f], weights_t):
            rep.tiles += 1
            if z_al is None:
                rep.rejected_support += 1
            elif tf.residual > bar:
                tf.reason = "residual"; rep.rejected_residual += 1
            else:
                tf.accepted = True; rep.accepted += 1
                t = tf.tile; sl = (slice(t.y0, t.y0 + t.h), slice(t.x0, t.x0 + t.w))
                za = FT._t(z_al)
                ok = za > 0
                zero = torch.zeros_like(za)
                num[sl] = num[sl] + torch.where(ok, za, zero) * wmap[sl]
                den[sl] = den[sl] + torch.where(ok, wmap[sl], zero)
            # margins to the two tile bars (positive = accepted side), in the bar's own unit
            m_res = (bar - tf.residual) if np.isfinite(tf.residual) else float("nan")
            m_sup = tf.support_frac - float(mcfg.min_support_frac)
            if np.isfinite(m_res):
                tile_margins.append(m_res)
            support_margins.append(m_sup)
            per["tiles"].append({"y0": tf.tile.y0, "x0": tf.tile.x0, "s": tf.s, "b": tf.b, "residual": tf.residual,
                                 "inlier_ratio": tf.inlier_ratio, "support_frac": round(tf.support_frac, 4),
                                 "accepted": tf.accepted, "reason": tf.reason,
                                 "margin_to_residual_bar": m_res, "margin_to_support_bar": m_sup})
        al_valid = den > 1e-6
        z_al_f = torch.where(al_valid, num / torch.clamp(den, min=1e-12), torch.zeros_like(num)).to(torch.float32)
        acc = [tf.residual for tf in fits[f] if tf.accepted]
        align_err = float(np.median(acc)) if acc else 0.0
        try:
            frt = FT.refine_frame(zc, vc, z_al_f, al_valid, tau, band_px, sigma_px, int(mcfg.side_reach_px), align_err,
                                  detail_scope=str(mcfg.detail_scope), detail_zone_px=int(mcfg.detail_zone_px))
        except FT.F6TorchError as e:
            raise MonoDetailError(str(e)) from e
        fr = _refinement(frt)
        out_dep[f] = fr.depth; out_src[f] = fr.source
        ur, uc = np.nonzero(fr.source == SRC_UNRESOLVED)
        if len(ur):
            rep.unresolved[int(f)] = (ur, uc, dep[f][ur, uc].astype(np.float32))
        per.update(fr.counts); per["align_err"] = align_err; per["margins"] = fr.margins
        rep.per_frame[int(f)] = per
        for k, v in fr.counts.items():
            totals[k] = totals.get(k, 0) + v
        if overlay_dir is not None:
            write_overlay(overlay_dir, f, image_of(f), dep[f], FT._np(z_al_f, np.float32), fr)
        if progress is not None and (n % 10 == 0 or n == len(frames) - 1):
            progress(50 + 50 * (n + 1) / len(frames), f"mono detail {n + 1}/{len(frames)} keyframes")
    rep.totals = totals
    rep.params["bars"]["tile_margins"] = {"residual": margin_quantiles(np.array(tile_margins)),
                                          "support": margin_quantiles(np.array(support_margins))}
    rep.seconds_total = round(time.time() - t_start, 1)
    nv = max(totals.get("valid", 1), 1)
    log(f"{LOG_TAG} tiles {rep.accepted}/{rep.tiles} accepted (support {rep.rejected_support}, residual "
        f"{rep.rejected_residual} over the p{mcfg.tile_residual_quantile:g} bar {bar * 100:.2f} %); detail on "
        f"{totals.get('detail', 0) / nv * 100:.1f} % of the valid pixels; band {totals.get('band', 0) / nv * 100:.2f} %, "
        f"mixed {totals.get('mixed', 0):,} → front {totals.get('band_front', 0):,}, back {totals.get('band_back', 0):,}, "
        f"unresolved {totals.get('mixed_unresolved', 0):,}, unjudged {totals.get('mixed_unjudged', 0):,}; PointDiT "
        f"{rep.seconds_pointdit:.0f} s, total {rep.seconds_total:.0f} s")
    return out_dep, out_src, rep


def write_overlay(out_dir, f: int, img: np.ndarray, z_cal: np.ndarray, z_al: np.ndarray, fr: FrameRefinement) -> None:
    """Per keyframe one PNG: RGB | bent depth | aligned PointDiT | detail | band + status (front green,
    back blue, unresolved red, detail grey)."""
    from pathlib import Path
    from PIL import Image
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v = z_cal > 0
    lo, hi = (np.percentile(z_cal[v], [2, 98]) if v.any() else (0.0, 1.0))

    def col(d, a, b):
        x = np.clip((d - a) / max(b - a, 1e-9), 0, 1)
        k = 3 / 2   # a 'jet'-like ramp for the eye only
        r = np.clip(k - np.abs(4 * x - 3), 0, 1); g = np.clip(k - np.abs(4 * x - 2), 0, 1); bb = np.clip(k - np.abs(4 * x - 1), 0, 1)
        o = (np.stack([r, g, bb], -1) * 255).astype(np.uint8); o[~np.isfinite(d) | (d == 0)] = 0
        return o

    st = np.zeros(img.shape, np.uint8)
    st[fr.source == SRC_DETAIL] = (110, 110, 110); st[fr.band] = (60, 60, 0)
    st[fr.source == SRC_BAND_FRONT] = (0, 200, 0); st[fr.source == SRC_BAND_BACK] = (0, 90, 255)
    st[fr.source == SRC_UNRESOLVED] = (255, 0, 0)
    dmax = max(float(np.percentile(np.abs(fr.detail[fr.detail != 0]), 98)) if (fr.detail != 0).any() else 1 / 100, 1 / 1000)
    panel = np.concatenate([np.asarray(img, np.uint8), col(z_cal, lo, hi), col(z_al, lo, hi),
                            col(fr.detail + dmax, 0, 2 * dmax), st], axis=1)
    Image.fromarray(panel).save(out_dir / f"mono_{f:06d}.png")
