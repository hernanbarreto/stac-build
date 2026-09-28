"""Intake I1 — keyframes by PARALLAX, witness frames, coverage warnings
(claude_stac.txt §4-F1).

The measurement, for every usable frame ``f`` of the chain (I0's usable
frames, in order), against the window's ANCHOR ``a``:

1. LK tracks (cv2, a ``grid_side`` × ``grid_side`` seed grid on the anchor, at
   ``process_scale``) matched from the anchor to ``f`` directly — each track
   starts from its last matched position (the prediction) but is always
   measured against the anchor's own patch, so its error does not grow with
   the frames in between (a per-step sum did: the keyframe count then depended
   on the frame rate). The target frame is first brought to the anchor's mean
   and contrast (a light change is not motion). A track counts at ``f`` only
   when its forward–backward disagreement is ≤ ``fb_max_px`` (native px); a
   track that fails once is dead for the window (re-acquiring it from a stale
   position produced gross mismatches).
2. The REFERENCE: the pure rotation of the SESSION CAMERA, H = K·R·K⁻¹ with K
   MEASURED once for the session (intake/focal.py: DA3 over frames spread over
   the video) and only R fitted — the homography a camera that did not
   translate induces. Least squares over the tracks, started from the rotation
   nearest the RANSAC homography (``ransac_px``), warm-started from the
   previous frame, solved TO CONVERGENCE (``reference_tol``; a fit the
   ``reference_max_eval`` bound stops is not a measurement) with a
   deterministic solver. A general homography is NOT the reference: it also
   explains the expansion of a forward walk, which then read as no baseline;
   neither is a K fitted per frame: it bends to absorb translation (pccr
   2026-08-24: a free K settled at fx≈1e-4 px, skew≈−15, and the reading
   depended on where the solver stopped — 746 / 743 / 742 keyframes from the
   same frames).
3. PARALLAX = the ``parallax_quantile`` (0.9) of the tracks' symmetric transfer
   error w.r.t. that rotation (native px): the image motion no rotation
   explains, read on the tracks that show the most of it. DECLARED DEVIATION
   from the spec's median: on a 2.5 m sideways walk the median never exceeded
   4.3 px (the near structure — the parallax — leaves the view first and the
   far background dominates the median) while the 0.9 quantile reached 12 px
   at 0.45 m; a 42° pan reads ≤ 1.6 px at 0.9. A pure rotation never earns a
   keyframe.

Keyframes: once the reading reaches ``parallax_quantum_px`` the window is
followed until it leaves the band ``quantum × (1 ± keyframe_band_frac)``; the
keyframe is the SHARPEST frame (I0 ``sharp_rank``) whose reading lies in that
band, and it becomes the next anchor. When the tracks are lost
(< ``min_tracks``) before the quantum, a window that measured a baseline
(≥ ``witness_min_parallax_px``) closes on its sharpest frame near its largest
reading (``track_loss``); one that did not is a coverage break — no keyframe,
the next anchor is the frame where the tracks were lost, and the run is warned.

Witness frames: every usable frame whose parallax from the last CHOSEN witness
reaches ``witness_min_parallax_px`` (or whose tracks from it are lost), plus
every keyframe — no cap.

Coverage warnings (``<session>/intake/coverage_warnings.json``): runs of at
least ``warn_min_run_frames`` frames that are ``static`` (median displacement
from the anchor < ``warn_static_disp_px``), ``pure_rotation`` (the view moved
≥ ``warn_rotation_min_disp_px`` with parallax < ``witness_min_parallax_px``),
``tracking_lost``, or ``exposure`` (rejected by I0).

Outputs: ``frames/selected_frames.json`` (the v2 contract: ``version`` "2.0",
``method``, ``total_frames``, ``selected_count``, ``selected_files``),
``frames/witness_frames.json`` and ``<session>/intake/coverage_warnings.json``.
Every length is in NATIVE px; every parameter is a ``ParallaxConfig`` field
read from ``intake.parallax`` in config.yaml.

CLI: ``python -m intake.parallax --session <dir>`` (I0's
``frames/quality_features.json`` and ``intake/focal_probe.json`` must exist).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from intake.config import ParallaxConfig, load_intake_config
from intake.quality import (QUALITY_VERSION, Cancelled, QualityError, _write_json_atomic,
                            check_cancelled, list_frames, read_gray, read_session_epochs)

SELECTED_FRAMES_NAME = "selected_frames.json"
WITNESS_FRAMES_NAME = "witness_frames.json"
COVERAGE_WARNINGS_NAME = "coverage_warnings.json"
INTAKE_SUBDIR = "intake"
SELECTED_CONTRACT_VERSION = "2.0"       # the v2 contract of frames/selected_frames.json
PARALLAX_VERSION = 5                    # 3: anchor-based parallax (2026-09-27)
                                        # 4: rotation reference (2026-09-28)
                                        # 5: cut back to the §4-F1 spec (2026-09-28): quantile of
                                        #    the residual w.r.t. the rotation, no rigidity /
                                        #    refinement / lens / twin / held-object machinery
PROVENANCE = "tool_measured"
METHOD = "parallax_lk"
WARNING_KINDS = ("static", "pure_rotation", "tracking_lost", "exposure")
LOST_REASONS = ("too_few_tracks", "rotation_fit_failed")
WINDOW_CLOSERS = ("quantum", "track_loss", "coverage_break", "end")
WITNESS_REASONS = ("first_usable_frame", "parallax", "track_loss", "keyframe")
LOG_TAG = "[intake.parallax]"

ScaleXY = Tuple[float, float]


class ParallaxError(RuntimeError):
    """A structural impossibility of I1 (no usable frame, a quality report
    measured on another frame inventory, an unreadable frame, frames of mixed
    sizes) — always with the exact reason."""


@dataclass(frozen=True)
class FrameMeasure:
    """One usable frame measured from its anchor, NATIVE px. NaN where nothing
    was measured (``lost``)."""
    frame: int
    anchor: int
    n_tracks: int
    disp_px: float          # median displacement of the tracks from the anchor
    parallax_px: float      # parallax_quantile of the residuals w.r.t. the rotation
    fb_px: float            # median forward–backward disagreement of the counted tracks
    lost: bool
    lost_reason: Optional[str]


def _cv2():
    import cv2
    return cv2


def _finite_or_none(x: Any) -> Any:
    """JSON-safe values: NaN / ±inf become None, numpy scalars Python ones."""
    if isinstance(x, dict):
        return {k: _finite_or_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite_or_none(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def measure_dict(m: FrameMeasure) -> Dict[str, Any]:
    return _finite_or_none(asdict(m))


# ── geometry helpers ─────────────────────────────────────────────────────

def to_native(pts_small: np.ndarray, scale: ScaleXY) -> np.ndarray:
    """Process-scale pixel coordinates → native ones (pixel-centre convention:
    pixel ``i`` of an image resized by ``s`` sits at native ``(i + 0.5)/s − 0.5``)."""
    sx, sy = float(scale[0]), float(scale[1])
    p = np.asarray(pts_small, dtype=np.float64).reshape(-1, 2)
    return np.c_[(p[:, 0] + 0.5) / sx - 0.5, (p[:, 1] + 0.5) / sy - 0.5]


def downscale(gray: np.ndarray, process_scale: float) -> Tuple[np.ndarray, ScaleXY]:
    """The gray frame at the tracking scale (INTER_AREA) and the EXACT per-axis
    scale applied (w_small / w, h_small / h)."""
    if gray.ndim != 2:
        raise ParallaxError(f"expected a 2-D gray frame, got shape {gray.shape}")
    h, w = gray.shape
    if process_scale == 1.0:
        return gray, (1.0, 1.0)
    ws = max(1, int(round(w * process_scale)))
    hs = max(1, int(round(h * process_scale)))
    small = _cv2().resize(gray, (ws, hs), interpolation=_cv2().INTER_AREA)
    return small, (ws / float(w), hs / float(h))


def seed_grid(shape_hw: Tuple[int, int], grid_side: int, margin: int = 0) -> np.ndarray:
    """(grid_side², 2) float32 seeds at the centres of a grid over the image
    inset by ``margin`` px (process-scale pixel coordinates, row-major)."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    g, m = int(grid_side), int(margin)
    if g < 1 or m < 0 or h - 2 * m < 1 or w - 2 * m < 1:
        raise ParallaxError(f"seed grid needs a positive grid side and image, got "
                            f"grid_side={g}, shape={shape_hw}, margin={m}")
    xs = m + (np.arange(g, dtype=np.float64) + 0.5) * ((w - 2 * m) / float(g)) - 0.5
    ys = m + (np.arange(g, dtype=np.float64) + 0.5) * ((h - 2 * m) / float(g)) - 0.5
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)


def photometric_match(gray: np.ndarray, ref: Tuple[float, float]) -> np.ndarray:
    """``gray`` mapped by the global gain/offset that gives it the reference
    (mean, std) — LK assumes brightness constancy and an exposure change is a
    global gain. A frame with no contrast is returned unchanged."""
    g = np.asarray(gray, dtype=np.float64)
    mean, std = float(g.mean()), float(g.std())
    if not std > 0.0:
        return gray
    return np.clip(np.rint((g - mean) * (ref[1] / std) + ref[0]), 0, 255).astype(np.uint8)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


def symmetric_transfer_error(H: np.ndarray, H_inv: np.ndarray, a: np.ndarray,
                             b: np.ndarray) -> np.ndarray:
    """Per-track (|H·a − b| + |H⁻¹·b − a|) / 2 (same units as a, b)."""
    fwd = np.hypot(*(_apply_h(H, a) - b).T)
    bwd = np.hypot(*(_apply_h(H_inv, b) - a).T)
    return (fwd + bwd) / 2.0


def _inv3(M: np.ndarray) -> Optional[np.ndarray]:
    """Closed-form inverse of a 3×3 (adjugate / determinant); None when singular
    or not finite. numpy's LAPACK inverse spins up OpenBLAS's threads on every
    call — measured 1.7 ms per 3×3 on this machine, a third of the stage's time."""
    a, b, c, d, e, f, g, h, i = (float(v) for v in np.asarray(M).ravel())
    A, B, C = e * i - f * h, -(d * i - f * g), d * h - e * g
    det = a * A + b * B + c * C
    if not (math.isfinite(det) and det != 0.0):
        return None
    return np.array([[A, -(b * i - c * h), b * f - c * e],
                     [B, a * i - c * g, -(a * f - c * d)],
                     [C, -(a * h - b * g), a * e - b * d]]) / det


def _rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    R, _ = _cv2().Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    det = float(np.linalg.det(M))
    M = M / np.cbrt(det) if det != 0.0 else M
    U, _s, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def rotation_homography(w: np.ndarray, Kn: np.ndarray) -> np.ndarray:
    """H = K·R(w)·K⁻¹: the homography a pure ROTATION w (rotation vector) of the camera
    ``Kn`` (its intrinsics in the normalised frame) induces."""
    return Kn @ _rotvec_to_matrix(w) @ _inv3(Kn)


def _transfer_jacobian(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Jacobian of the transfer M·p w.r.t. the nine entries of M, row-major
    (n, 2, 9) — analytic."""
    ph = np.c_[p, np.ones(len(p))]
    q = ph @ M.T
    w = q[:, 2:3]
    J = np.zeros((len(p), 2, 9))
    J[:, 0, 0:3] = ph / w
    J[:, 1, 3:6] = ph / w
    J[:, 0, 6:9] = -ph * (q[:, 0:1] / (w * w))
    J[:, 1, 6:9] = -ph * (q[:, 1:2] / (w * w))
    return J


def image_normaliser(native_wh: Tuple[int, int]) -> np.ndarray:
    """The similarity taking native px to the image-centred frame scaled by the
    half-diagonal — FIXED per session, so a rotation fit can be warm-started
    from the previous frame's parameters (conditioning only)."""
    w, h = float(native_wh[0]), float(native_wh[1])
    s = 1.0 / (math.hypot(w, h) / 2.0)
    return np.array([[s, 0.0, -s * (w - 1.0) / 2.0], [0.0, s, -s * (h - 1.0) / 2.0],
                     [0.0, 0.0, 1.0]])


def fit_rotation(a: np.ndarray, b: np.ndarray, T: np.ndarray, cfg: ParallaxConfig,
                 Kn: np.ndarray, x0: Optional[np.ndarray] = None
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Least-squares fit of the pure-rotation homography H = K·R·K⁻¹ of the session's
    MEASURED camera (``Kn`` = T·K, intrinsics in the normalised frame ``T``; intake/
    focal.py) to the tracks a → b (native px): only R, symmetric transfer residuals,
    analytic Jacobian, solved TO CONVERGENCE by scipy's trust-region reflective method
    with the exact solver — deterministic (scipy 1.15's MINPACK Levenberg–Marquardt
    returned different solutions for identical inputs) and well-posed (a free K had no
    unique minimum: it settled at fx≈1e-4 px, skew≈−15 on pccr).

    Converged = step, cost change and gradient all under ``reference_tol``; a fit the
    ``reference_max_eval`` bound stops first is NOT a measurement (None: the frame is
    lost, reason rotation_fit_failed). ``x0``: the rotation vector to start from (the
    previous frame's); None → the rotation nearest the RANSAC homography. Returns
    (rotation vector, H native, cost) or None."""
    from scipy.optimize import least_squares
    cv2 = _cv2()
    an = _apply_h(T, np.asarray(a, dtype=np.float64))
    bn = _apply_h(T, np.asarray(b, dtype=np.float64))
    n = len(an)
    if n < 4:
        return None
    Kn = np.asarray(Kn, dtype=np.float64)
    Ki = _inv3(Kn)
    if x0 is None or not np.all(np.isfinite(x0)):
        H0, _m = cv2.findHomography(an, bn, cv2.RANSAC, float(cfg.ransac_px) * float(T[0, 0]))
        if H0 is None or not np.all(np.isfinite(H0)):
            H0 = np.eye(3)
        w0, _ = cv2.Rodrigues(_nearest_rotation(Ki @ H0 @ Kn))
        x0 = w0.ravel()
    x0 = np.array(x0, dtype=np.float64)

    def resid(x):
        R = _rotvec_to_matrix(x)
        H, G = Kn @ R @ Ki, Kn @ R.T @ Ki
        return np.r_[(_apply_h(H, an) - bn).ravel(), (_apply_h(G, bn) - an).ravel()]

    def jac(x):
        R, dR = cv2.Rodrigues(np.asarray(x, dtype=np.float64).reshape(3, 1))
        dR = dR.reshape(3, 3, 3)                        # dR[k] = ∂R/∂w_k
        H, G = Kn @ R @ Ki, Kn @ R.T @ Ki
        dH = np.stack([(Kn @ dR[k] @ Ki).ravel() for k in range(3)], 1)
        dG = np.stack([(Kn @ dR[k].T @ Ki).ravel() for k in range(3)], 1)
        return np.r_[_transfer_jacobian(H, an).reshape(-1, 9) @ dH,
                     _transfer_jacobian(G, bn).reshape(-1, 9) @ dG]

    tol = float(cfg.reference_tol)
    try:
        sol = least_squares(resid, x0, jac=jac, method="trf", tr_solver="exact",
                            x_scale=np.ones(3), xtol=tol, ftol=tol, gtol=tol,
                            max_nfev=int(cfg.reference_max_eval))
    except (ValueError, np.linalg.LinAlgError):
        return None
    if sol.status <= 0 or not np.all(np.isfinite(sol.x)):
        return None                              # the bound, not convergence, stopped it
    return sol.x, _inv3(T) @ rotation_homography(sol.x, Kn) @ T, float(sol.cost)


# ── frames ───────────────────────────────────────────────────────────────

class _Frames:
    """Process-scale gray frames by frame number, read on demand with a small
    LRU (the passes are sequential; a re-anchor re-reads at most one band)."""

    CACHE = 64

    def __init__(self, paths: Dict[int, Path], native_wh: Tuple[int, int], cfg: ParallaxConfig):
        self.paths = paths
        self.native_wh = native_wh
        self.cfg = cfg
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.scale: Optional[ScaleXY] = None
        self.n_reads = 0

    def small(self, f: int) -> np.ndarray:
        if f in self.cache:
            self.cache.move_to_end(f)
            return self.cache[f]
        try:
            g = read_gray(self.paths[f])
        except QualityError as e:
            raise ParallaxError(str(e)) from e
        if (g.shape[1], g.shape[0]) != self.native_wh:
            raise ParallaxError(f"frame {self.paths[f].name} is {g.shape[1]}x{g.shape[0]}, the "
                                f"session's frames are {self.native_wh[0]}x{self.native_wh[1]}")
        s, scale = downscale(g, self.cfg.process_scale)
        self.scale = scale
        self.n_reads += 1
        self.cache[f] = s
        if len(self.cache) > self.CACHE:
            self.cache.popitem(last=False)
        return s


class _Window:
    """Tracks seeded on one anchor, each matched to the anchor at every frame."""

    def __init__(self, anchor: int, frames: _Frames, cfg: ParallaxConfig, Kn: np.ndarray):
        self.anchor = anchor
        self.frames = frames
        self.cfg = cfg
        self.Kn = Kn                              # the session camera, normalised frame
        self.img = frames.small(anchor)
        g = self.img.astype(np.float64)
        self.ref = (float(g.mean()), float(g.std()))
        self.seeds = seed_grid(self.img.shape, cfg.grid_side, margin=int(cfg.lk_win) // 2)
        self.pred = self.seeds.copy()
        self.alive = np.ones(len(self.seeds), dtype=bool)
        self.x: Optional[np.ndarray] = None       # warm start of the rotation fit

    def match(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(anchor px, frame px, fb) — native px, the tracks counted at ``f``."""
        cv2 = _cv2()
        cfg = self.cfg
        tgt = photometric_match(self.frames.small(f), self.ref)
        win = (int(cfg.lk_win), int(cfg.lk_win))
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        idx = np.flatnonzero(self.alive)
        if len(idx) == 0:
            empty = np.zeros((0, 2))
            return empty, empty.copy(), np.zeros(0)
        p0 = np.ascontiguousarray(self.seeds[idx].reshape(-1, 1, 2))
        guess = np.ascontiguousarray(self.pred[idx].reshape(-1, 1, 2).astype(np.float32))
        fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(self.img, tgt, p0, guess, winSize=win,
                                                maxLevel=int(cfg.lk_levels), flags=flags)
        bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(tgt, self.img, fwd, p0.copy(), winSize=win,
                                                maxLevel=int(cfg.lk_levels), flags=flags)
        sx, sy = self.frames.scale
        a = self.seeds[idx].astype(np.float64)
        b = fwd.reshape(-1, 2).astype(np.float64)
        back = bwd.reshape(-1, 2).astype(np.float64)
        h, w = self.img.shape
        ok = (st_f.ravel() == 1) & (st_b.ravel() == 1)
        ok &= np.all(np.isfinite(b), axis=1) & np.all(np.isfinite(back), axis=1)
        ok &= (b[:, 0] >= 0) & (b[:, 0] <= w - 1) & (b[:, 1] >= 0) & (b[:, 1] <= h - 1)
        with np.errstate(invalid="ignore"):
            fb = np.hypot((back[:, 0] - a[:, 0]) / sx, (back[:, 1] - a[:, 1]) / sy)
        ok &= np.isfinite(fb) & (fb <= cfg.fb_max_px)
        # a track that fails once is DEAD for the window: re-acquiring it from a stale
        # position is what produced gross mismatches (a point that left the view
        # "found" on similar texture elsewhere, forward-backward consistent, 580 px off)
        self.alive[idx[~ok]] = False
        self.pred[idx[ok]] = b[ok].astype(np.float32)
        return to_native(a[ok], (sx, sy)), to_native(b[ok], (sx, sy)), fb[ok]

    def measure(self, f: int, T: np.ndarray) -> FrameMeasure:
        a, b, fb = self.match(f)
        n = len(a)
        nan = float("nan")
        if n < self.cfg.min_tracks:
            return FrameMeasure(f, self.anchor, n, nan, nan, nan, True, "too_few_tracks")
        disp = float(np.median(np.hypot(*(b - a).T)))
        # warm-started from the previous frame of the window (speed only: the camera is
        # fixed and the fit converges to the same minimum from any nearby start)
        fit = fit_rotation(a, b, T, self.cfg, self.Kn, self.x)
        if fit is None:
            return FrameMeasure(f, self.anchor, n, disp, nan, float(np.median(fb)), True,
                                "rotation_fit_failed")
        self.x, H, _cost = fit
        H_inv = _inv3(H)
        if H_inv is None:
            return FrameMeasure(f, self.anchor, n, disp, nan, float(np.median(fb)), True,
                                "rotation_fit_failed")
        sym = symmetric_transfer_error(H, H_inv, a, b)
        sym = sym[np.isfinite(sym)]
        if len(sym) < self.cfg.min_tracks:
            return FrameMeasure(f, self.anchor, n, disp, nan, float(np.median(fb)), True,
                                "rotation_fit_failed")
        return FrameMeasure(f, self.anchor, n, disp,
                            float(np.quantile(sym, self.cfg.parallax_quantile)),
                            float(np.median(fb)), False, None)


class _Progress:
    def __init__(self, log: Callable, what: str, total: int, heartbeat_s: float,
                 cancelled: Cancelled):
        self.log, self.what, self.total = log, what, total
        self.heartbeat_s, self.cancelled = heartbeat_s, cancelled
        self.t0 = self.last = time.monotonic()
        self.n = 0

    def tick(self, f: int) -> None:
        self.n += 1
        check_cancelled(self.cancelled, f"intake I1 ({self.what})")
        now = time.monotonic()
        if now - self.last >= self.heartbeat_s:
            self.last = now
            rate = self.n / max(now - self.t0, 1e-9)
            self.log(f"{LOG_TAG} {self.what}: frame {f} ({self.n} measurements, "
                     f"{rate:.1f}/s)")


# ── selection ────────────────────────────────────────────────────────────

def _sharpest(cands: Sequence[FrameMeasure], by_frame: Dict[int, Dict[str, Any]]) -> FrameMeasure:
    """Highest I0 sharp_rank; ties → the earliest frame."""
    return max(cands, key=lambda m: (float(by_frame[m.frame]["sharp_rank"]), -m.frame))


def select_keyframes(chain: List[int], frames: _Frames, by_frame: Dict[int, Dict[str, Any]],
                     cfg: ParallaxConfig, T: np.ndarray, Kn: np.ndarray, prog: _Progress
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, FrameMeasure]]:
    """(keyframes, windows, {frame: its last measurement})."""
    q = float(cfg.parallax_quantum_px)
    lo, hi = q * (1.0 - cfg.keyframe_band_frac), q * (1.0 + cfg.keyframe_band_frac)
    pos = {f: i for i, f in enumerate(chain)}
    keyframes = [{"frame": chain[0], "file": by_frame[chain[0]]["file"], "anchor": None,
                  "parallax_px": 0.0, "closed_by": "first_usable_frame"}]
    windows: List[Dict[str, Any]] = []
    records: Dict[int, FrameMeasure] = {}
    anchor, anchor_is_kf = chain[0], True
    i = 1
    while i < len(chain):
        win = _Window(anchor, frames, cfg, Kn)
        seen: List[FrameMeasure] = []
        closed_by, chosen, restart = "end", None, len(chain)
        j = i
        while j < len(chain):
            f = chain[j]
            prog.tick(f)
            m = win.measure(f, T)
            records[f] = m
            if m.lost:
                base = [s for s in seen if s.parallax_px >= cfg.witness_min_parallax_px]
                if base:
                    top = max(s.parallax_px for s in base)
                    chosen = _sharpest([s for s in base
                                        if s.parallax_px >= top * (1.0 - cfg.keyframe_band_frac)],
                                       by_frame)
                    closed_by, restart = "track_loss", pos[chosen.frame] + 1
                else:
                    closed_by, restart = "coverage_break", j + 1
                break
            seen.append(m)
            if m.parallax_px > hi:
                band = [s for s in seen if lo <= s.parallax_px <= hi]
                if not band:
                    band = [next(s for s in seen if s.parallax_px >= lo)]
                chosen = _sharpest(band, by_frame)
                closed_by, restart = "quantum", pos[chosen.frame] + 1
                break
            j += 1
        if closed_by == "end":
            band = [s for s in seen if lo <= s.parallax_px <= hi]
            if band:
                chosen = _sharpest(band, by_frame)
                closed_by = "quantum"
        windows.append({"anchor": anchor, "anchor_is_keyframe": anchor_is_kf,
                        "closed_by": closed_by, "n_frames": len(seen) + (closed_by in
                                                                         ("track_loss",
                                                                          "coverage_break")),
                        "keyframe": chosen.frame if chosen is not None else None,
                        "max_parallax_px": (max(s.parallax_px for s in seen) if seen else None)})
        if chosen is not None:
            keyframes.append({"frame": chosen.frame, "file": by_frame[chosen.frame]["file"],
                              "anchor": anchor, "parallax_px": chosen.parallax_px,
                              "closed_by": closed_by})
            anchor, anchor_is_kf = chosen.frame, True
        elif closed_by == "coverage_break":
            anchor, anchor_is_kf = chain[j], False
        if closed_by == "end":
            break
        i = restart
    return keyframes, windows, records


def select_witnesses(chain: List[int], kf_frames: Sequence[int], frames: _Frames,
                     by_frame: Dict[int, Dict[str, Any]], cfg: ParallaxConfig, T: np.ndarray,
                     Kn: np.ndarray, prog: _Progress) -> List[Dict[str, Any]]:
    """Every usable frame whose parallax from the last CHOSEN witness reaches
    ``witness_min_parallax_px`` (or whose tracks from it are lost), plus every
    keyframe."""
    kf = set(int(k) for k in kf_frames)
    out = [{"frame": chain[0], "file": by_frame[chain[0]]["file"],
            "reason": "first_usable_frame", "parallax_px": 0.0}]
    win = _Window(chain[0], frames, cfg, Kn)
    for f in chain[1:]:
        prog.tick(f)
        m = win.measure(f, T)
        reason = ("keyframe" if f in kf else "track_loss" if m.lost else
                  "parallax" if m.parallax_px >= cfg.witness_min_parallax_px else None)
        if reason is None:
            continue
        out.append({"frame": f, "file": by_frame[f]["file"], "reason": reason,
                    "parallax_px": None if m.lost else m.parallax_px})
        win = _Window(f, frames, cfg, Kn)
    return out


# ── warnings ─────────────────────────────────────────────────────────────

def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """[start, end) index ranges of consecutive True values."""
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def frame_flags(m: FrameMeasure, cfg: ParallaxConfig) -> Dict[str, bool]:
    if m.lost:
        return {"static": False, "pure_rotation": False, "tracking_lost": True}
    return {"static": m.disp_px < cfg.warn_static_disp_px,
            "pure_rotation": (m.disp_px >= cfg.warn_rotation_min_disp_px
                              and m.parallax_px < cfg.witness_min_parallax_px),
            "tracking_lost": False}


def coverage_warnings(chain: List[int], records: Dict[int, FrameMeasure],
                      rows: List[Dict[str, Any]], cfg: ParallaxConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    measured = [f for f in chain if f in records]
    flags = {f: frame_flags(records[f], cfg) for f in measured}
    detail = {
        "static": f"median displacement from the anchor < {cfg.warn_static_disp_px:g} px: the "
                  f"camera is still — no new viewpoint",
        "pure_rotation": f"the view moved ≥ {cfg.warn_rotation_min_disp_px:g} px with parallax "
                         f"< {cfg.witness_min_parallax_px:g} px: pure rotation, no baseline, "
                         f"no keyframe",
        "tracking_lost": f"fewer than {cfg.min_tracks} tracks matched to the anchor",
    }
    for kind in ("static", "pure_rotation", "tracking_lost"):
        for s, e in _runs([flags[f][kind] for f in measured]):
            if e - s >= cfg.warn_min_run_frames:
                out.append({"kind": kind, "frame_start": measured[s], "frame_end": measured[e - 1],
                            "n_frames": e - s, "detail": detail[kind]})
    bad = [not r["usable"] for r in rows]
    for s, e in _runs(bad):
        if e - s >= cfg.warn_min_run_frames:
            reasons = sorted({str(rows[k].get("reject_reason")) for k in range(s, e)})
            out.append({"kind": "exposure", "frame_start": int(rows[s]["frame"]),
                        "frame_end": int(rows[e - 1]["frame"]), "n_frames": e - s,
                        "detail": f"rejected by I0 ({', '.join(reasons)}): no frame to select"})
    return sorted(out, key=lambda w: (w["frame_start"], w["kind"]))


# ── run ──────────────────────────────────────────────────────────────────

def _frame_table(quality: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(quality, dict) or quality.get("version") != QUALITY_VERSION:
        raise ParallaxError(f"the quality report must be an I0 report of version "
                            f"{QUALITY_VERSION}, got version {quality.get('version')!r}"
                            if isinstance(quality, dict) else
                            "the quality report must be the dict returned by intake.quality")
    frames = quality.get("frames")
    if not frames:
        raise ParallaxError("the quality report lists no frame — nothing to select from")
    rows = sorted(frames, key=lambda f: int(f["frame"]))
    for r in rows:
        for k in ("frame", "file", "sharp_rank", "usable"):
            if k not in r:
                raise ParallaxError(f"quality report frame record lacks '{k}': {r}")
    return rows


def _check_inventory(paths: List[Path], rows: List[Dict[str, Any]], frames_dir: Path) -> None:
    on_disk = [int(p.stem) for p in paths]
    reported = [int(r["frame"]) for r in rows]
    if on_disk != reported:
        only_disk = sorted(set(on_disk) - set(reported))[:5]
        only_rep = sorted(set(reported) - set(on_disk))[:5]
        raise ParallaxError(
            f"the quality report was measured on another frame inventory than {frames_dir}: "
            f"{len(reported)} frame(s) reported vs {len(on_disk)} on disk (on disk only: "
            f"{only_disk}…, reported only: {only_rep}…) — re-run intake I0")


def run_parallax(frames_dir: os.PathLike, quality: Dict[str, Any], cfg: ParallaxConfig,
                 K: np.ndarray, log: Callable = print, *, heartbeat_s: float,
                 geometry_epoch: int = 0,
                 camera_epoch: int = 0, cancelled: Cancelled = None) -> Dict[str, Any]:
    """Keyframes, witnesses and coverage warnings over the usable frames of
    ``frames_dir``. ``quality`` is the I0 report of the SAME frame inventory; ``K`` the
    session camera's intrinsics in native px (intake/focal.py)."""
    if heartbeat_s <= 0:
        raise ParallaxError(f"heartbeat_s must be positive, got {heartbeat_s}")
    frames_dir = Path(frames_dir)
    try:
        paths = list_frames(frames_dir)
    except QualityError as e:
        raise ParallaxError(str(e)) from e
    rows = _frame_table(quality)
    _check_inventory(paths, rows, frames_dir)
    by_frame = {int(r["frame"]): r for r in rows}
    chain = [int(r["frame"]) for r in rows if r["usable"]]
    if not chain:
        rejected = quality.get("rejected", {})
        raise ParallaxError(
            f"no usable frame in {frames_dir}: every one of the {len(rows)} frame(s) was "
            f"rejected by exposure ({', '.join(f'{k}={v}' for k, v in rejected.items())})")
    native_wh = (int(quality["native_w"]), int(quality["native_h"]))
    T = image_normaliser(native_wh)
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ParallaxError(f"the session K must be a finite 3x3 with positive focals, got {K}")
    Kn = T @ K
    log(f"{LOG_TAG} {len(chain)}/{len(rows)} usable frame(s); LK grid {cfg.grid_side}x"
        f"{cfg.grid_side} at scale {cfg.process_scale:g} matched to the anchor; parallax = "
        f"{cfg.parallax_quantile:g}-quantile of the residuals w.r.t. the rotation of the "
        f"session camera (fx {K[0, 0]:.1f} px, measured); quantum "
        f"{cfg.parallax_quantum_px:g} px (band ±{cfg.keyframe_band_frac:g}), witness "
        f"{cfg.witness_min_parallax_px:g} px (native px)")
    frames = _Frames({int(p.stem): p for p in paths}, native_wh, cfg)

    # BLAS single-threaded: the passes are sequential, and no reading may depend on how
    # many threads split a reduction (identical inputs → bit-identical keyframes)
    from threadpoolctl import threadpool_limits
    _blas = threadpool_limits(limits=1)
    t0 = time.monotonic()
    prog = _Progress(log, "keyframes", len(chain), heartbeat_s, cancelled)
    keyframes, windows, records = select_keyframes(chain, frames, by_frame, cfg, T, Kn, prog)
    dt = time.monotonic() - t0
    log(f"{LOG_TAG} keyframes: {len(keyframes)} in {dt:.1f} s "
        f"({prog.n} measurements, {prog.n / max(dt, 1e-9):.1f}/s)")
    t1 = time.monotonic()
    prog_w = _Progress(log, "witnesses", len(chain), heartbeat_s, cancelled)
    witnesses = select_witnesses(chain, [k["frame"] for k in keyframes], frames, by_frame,
                                 cfg, T, Kn, prog_w)
    _blas.restore_original_limits()
    dt = time.monotonic() - t1
    log(f"{LOG_TAG} witnesses: {len(witnesses)} in {dt:.1f} s "
        f"({prog_w.n} measurements, {prog_w.n / max(dt, 1e-9):.1f}/s)")

    warnings = coverage_warnings(chain, records, rows, cfg)
    frame_recs = []
    for f in chain[1:]:
        if f in records:
            d = measure_dict(records[f])
            d.update(frame_flags(records[f], cfg))
            frame_recs.append(d)
    n_lost = sum(1 for r in frame_recs if r["lost"])
    log(f"{LOG_TAG} {len(keyframes)} keyframe(s), {len(witnesses)} witness(es), "
        f"{len(warnings)} coverage warning(s), {n_lost} frame(s) lost")
    for w in warnings:
        log(f"{LOG_TAG} ⚠ {w['kind']} frames {w['frame_start']}–{w['frame_end']}: {w['detail']}")
    return {
        "keyframes": keyframes,
        "witnesses": witnesses,
        "warnings": warnings,
        "windows": windows,
        "frames": frame_recs,
        "n_frames": len(rows),
        "n_usable_frames": len(chain),
        "n_lost": n_lost,
        "n_keyframes": len(keyframes),
        "n_witness": len(witnesses),
        "n_frame_reads": frames.n_reads,
        "version": PARALLAX_VERSION,
        "provenance": PROVENANCE,
        "geometry_epoch": int(geometry_epoch),
        "camera_epoch": int(camera_epoch),
        "method": METHOD,
        "native_w": native_wh[0],
        "native_h": native_wh[1],
        "params": asdict(cfg),
        "K": K.tolist(),
        "inputs": {
            "frames_dir": str(frames_dir),
            "n_frames": len(paths),
            "first": paths[0].name,
            "last": paths[-1].name,
            "bytes_total": int(sum(p.stat().st_size for p in paths)),
            "quality_n_frames": int(quality.get("n_frames", len(rows))),
            "quality_n_usable": int(quality.get("n_usable", len(chain))),
        },
    }


# ── documents ────────────────────────────────────────────────────────────

def _stamps(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "provenance": PROVENANCE,
        "geometry_epoch": int(result["geometry_epoch"]),
        "camera_epoch": int(result["camera_epoch"]),
        "params": result["params"],
        "inputs": result["inputs"],
    }


def selected_frames_document(result: Dict[str, Any], cfg: ParallaxConfig) -> Dict[str, Any]:
    """frames/selected_frames.json: the v2 contract plus this stage's records."""
    files = sorted((k["file"] for k in result["keyframes"]), key=lambda f: int(Path(f).stem))
    total = int(result["n_frames"])
    return _finite_or_none({
        "version": SELECTED_CONTRACT_VERSION,
        "method": f"{METHOD}_{cfg.parallax_quantum_px:g}",
        "total_frames": total,
        "selected_count": len(files),
        "selected_files": files,
        "reduction": (1.0 - len(files) / float(total)) if total else 0.0,
        **_stamps(result),
        "intake_version": PARALLAX_VERSION,
        "parallax_quantum_px": float(cfg.parallax_quantum_px),
        "keyframe_band_frac": float(cfg.keyframe_band_frac),
        "keyframes": result["keyframes"],
        "windows": result["windows"],
        "n_witness": int(result["n_witness"]),
        "n_usable_frames": int(result["n_usable_frames"]),
        "n_lost": int(result["n_lost"]),
    })


def witness_frames_document(result: Dict[str, Any], cfg: ParallaxConfig) -> Dict[str, Any]:
    files = sorted((w["file"] for w in result["witnesses"]), key=lambda f: int(Path(f).stem))
    return _finite_or_none({
        "version": PARALLAX_VERSION,
        **_stamps(result),
        "method": METHOD,
        "witness_min_parallax_px": float(cfg.witness_min_parallax_px),
        "frames": result["witnesses"],
        "total_frames": int(result["n_frames"]),
        "selected_count": len(files),
        "selected_files": files,
    })


def coverage_warnings_document(result: Dict[str, Any]) -> Dict[str, Any]:
    return _finite_or_none({
        "version": PARALLAX_VERSION,
        **_stamps(result),
        "method": METHOD,
        "warnings": result["warnings"],
        "warning_kinds": list(WARNING_KINDS),
        "n_lost": int(result["n_lost"]),
        "frames": result["frames"],
    })


def write_selection(session_dir: os.PathLike, result: Dict[str, Any], cfg: ParallaxConfig
                    ) -> Tuple[Path, Path, Path]:
    """Write ``frames/selected_frames.json``, ``frames/witness_frames.json`` and
    ``<session>/intake/coverage_warnings.json`` (each atomic). Returns the three
    paths in that order."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        raise ParallaxError(f"{frames_dir} is not a directory — the selection belongs next "
                            f"to the frames it selects")
    p_kf = _write_json_atomic(frames_dir / SELECTED_FRAMES_NAME,
                              selected_frames_document(result, cfg))
    p_w = _write_json_atomic(frames_dir / WITNESS_FRAMES_NAME,
                             witness_frames_document(result, cfg))
    p_warn = _write_json_atomic(session_dir / INTAKE_SUBDIR / COVERAGE_WARNINGS_NAME,
                                coverage_warnings_document(result))
    return p_kf, p_w, p_warn


def load_selection(frames_dir: os.PathLike) -> Dict[str, Any]:
    """The ``selected_frames.json`` this module wrote; ParallaxError when it is
    absent or was written by another selector."""
    p = Path(frames_dir) / SELECTED_FRAMES_NAME
    if not p.exists():
        raise ParallaxError(f"{p} does not exist — run intake I1 "
                            f"(python -m intake.parallax --session <dir>) first")
    with open(p) as f:
        doc = json.load(f)
    if not str(doc.get("method", "")).startswith(METHOD):
        raise ParallaxError(f"{p} was written by selector {doc.get('method')!r}, not by "
                            f"{METHOD} — re-run intake I1 to select by parallax")
    return doc


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    from intake.quality import load_quality
    ap = argparse.ArgumentParser(
        prog="python -m intake.parallax",
        description="I1 — keyframes by parallax, witness frames, coverage warnings.")
    ap.add_argument("--session", required=True,
                    help="session directory (frames in <session>/frames; I0 must have run)")
    args = ap.parse_args(argv)
    session_dir = Path(args.session)
    icfg = load_intake_config()
    frames_dir = session_dir / "frames"
    quality = load_quality(frames_dir)
    from intake.focal import default_probe
    K = default_probe()(session_dir, quality, icfg.parallax, print, None)
    result = run_parallax(frames_dir, quality, icfg.parallax, K, log=print,
                          heartbeat_s=icfg.runtime.heartbeat_s,
                          **read_session_epochs(session_dir))
    p_kf, p_w, p_warn = write_selection(session_dir, result, icfg.parallax)
    print(f"{LOG_TAG} keyframes={result['n_keyframes']} witnesses={result['n_witness']} "
          f"warnings={len(result['warnings'])} → {p_kf}, {p_w}, {p_warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
