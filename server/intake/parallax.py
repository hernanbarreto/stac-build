"""I1 — keyframes by MEASURED parallax, witness frames, coverage warnings
(claude_stac.txt §4-F1; the anchor-based measurement of 2026-09-27, with every
window matched to the anchor at every frame).

What a keyframe selector must know is how much BASELINE separates two views:
how much the scene's geometry moved in the image, not how much the image
changed. A pure rotation or a light change turns every pixel over and shows
no new geometry. This module measures parallax against an ANCHOR — the last
keyframe — so the reading is the parallax of the baseline anchor → t itself:
it grows with the baseline under translation, stays at the tracker's own
error under a rotation, and does not depend on how densely the walk was
sampled.

THE MEASURE, per usable frame t of the window opened at the anchor
(:class:`AnchorChain`):
  1. Windows matched to the ANCHOR. ``grid_side`` × ``grid_side`` seeds on
     the anchor's gray (downscaled by ``process_scale``, the grid inset by
     half an LK window so every window lies in the anchor). At every frame
     each seed's ``lk_win`` window is refined DIRECTLY against the anchor's
     own window — an inverse-compositional homography per window, on windows
     normalised to zero mean and unit variance (a gain or offset that
     differs between the two windows cancels: an exposure ramp, the rolling
     bands of a flickering lamp), iterated to ``refine_eps_px`` or
     ``refine_max_iter`` — started from a PREDICTION: one pyramidal LK link
     t−1 → t for the windows matched at t−1 (locally zero-mean frames,
     forward–backward ≤ ``fb_max_px``), their carried neighbours' own window
     shapes for the others (their own surface), the link's frame-to-frame
     homography otherwise. The refined window is checked by LK on the
     rectified tile (status, forward–backward ≤ ``fb_max_px``) and must lie
     whole in frame t. A track's position at t is its window's centre: one
     match's error whatever the number of links behind it (a chained LK
     track drifted 0.24 → 2.9 px median, 0.8 → 6.7 px p90 over 51 links on a
     forward walk and followed occlusion boundaries on a hall walk), and a
     track's validity is decided at t alone — not by the product of every
     earlier link's check, which made the survivors a function of the frame
     rate. A track leaves for good when its prediction leaves the view; it is
     never re-seeded inside a window.
  2. On the trusted matched tracks (anchor positions → positions at t,
     NATIVE px): a RANSAC homography H (bound ``ransac_px``) and a RANSAC
     fundamental matrix F (same bound; the library's own RANSAC confidence /
     iteration defaults for both). H is the window's REFERENCE PLANE and is
     FOLLOWED: from the second frame on, RANSAC runs on the previous frame's
     H-inliers and the inliers are then read over every track.
  3. APPEARANCE: a window whose normalised residual at convergence exceeds
     ``rotation_floor_factor`` × the ``parallax_quantile`` of the warp twin's
     residuals (NOISE REFERENCE below) is not the anchor's patch — an
     occlusion edge, a texture aliasing as the view moves — and leaves the
     frame (the homography is refitted without it). It passed the
     forward–backward check; on the synthetic hall such windows were 7 % of
     the tracks and doubled the reading.
  4. The RIGID set — the tracks consistent with a static scene seen by a
     moving camera. By default the F-inliers (verdict ``epipolar``). That
     test is blind when H holds more tracks than the off-homography
     F-inliers: F is then fixed by that minority alone, and an object moving
     along a line during a pan or in front of a still camera satisfies it
     exactly as near structure would. The majority then testifies:
       * ``still``    — the identity keeps at least the ``parallax_quantile``
                        share of what H keeps within ``ransac_px``;
       * ``rotation`` — H carries a rotation (a complex eigenvalue pair, K-free:
                        the bootstrap upper bound of the pair discriminant over
                        ``rigidity_bootstrap`` resamples stays below zero at
                        ``rigidity_confidence``) AND the pure rotation of a
                        zero-skew, square-pixel pinhole whose focal length AND
                        principal point are fitted (the image centre is only
                        the start: 2 % off-centre failed a centre-pinned fit)
                        explains the majority at its OWN NOISE FLOOR — the
                        q-quantile of its symmetric transfer error over the
                        H-inliers ≤ ``rotation_floor_factor`` × the frame's
                        measured floor — AND the homography a plane induces
                        under a pure TRANSLATION (I + u·vᵀ, any K) does not
                        explain them at that floor. When both models explain
                        the majority the verdict stays ``epipolar`` (ambiguity
                        keeps the parallax).
     In both no-translation verdicts the rigid set is the H-inliers and a
     trusted track that moved more than ``witness_min_parallax_px`` relative
     to H is MARKED dynamic for the rest of the window (released by the first
     ``epipolar`` verdict).
  5. parallax_t = the ``parallax_quantile`` (q) quantile of the symmetric
     transfer error of the rigid tracks w.r.t. the homography that best
     explains them ALL (least squares, no inlier choice): the image motion no
     homography explains, read where the scene exhibits it most. The reading
     w.r.t. the reference plane and the quantile over ALL matched tracks
     (``parallax_all_px``) are recorded as evidence.

KEYFRAMES. The first usable frame is a keyframe and the first anchor. When
parallax_t reaches ``parallax_quantum_px`` the window is FOLLOWED until the
reading leaves the symmetric band [(1 − ``keyframe_band_frac``), (1 +
``keyframe_band_frac``)] × quantum (or the tracks are lost, or the sequence
ends); the keyframe is the frame with the highest ``sharp_rank`` (I0) whose
reading lies in that band — the plan's "best sharp_rank within the window",
with a baseline that does not depend on how many frames sampled the window
(the one-sided band [0.75 q, reading] drifted toward 0.875 q at 60 fps: 63 /
57 / 50 keyframes at 60 / 30 / 15 fps). Where the sampling steps over the
whole band, the frame whose reading is closest to the quantum. The keyframe
becomes the new anchor and every frame after it — already seen from the old
anchor — is re-measured from it (precision over runtime). Each keyframe
records its anchor, its reading and reading / quantum.
TRACK LOSS (fewer than ``min_tracks`` rigid tracks before the quantum): a
keyframe is made ONLY after a view change with a measured baseline — a frame
displaced ≥ ``warn_rotation_min_disp_px`` from the anchor whose reading is ≥
max(``witness_min_parallax_px``, ``rotation_floor_factor`` × its floor):
then the sharpest frame within the band of the largest such reading (reason
``track_loss``). Without one — a pan past the field of view, a texture the
tracker cannot hold, noise — the loss is a COVERAGE BREAK: no keyframe; the
last measured frame becomes a PENDING anchor (not a keyframe) the
measurement continues from (a rotation that turns past the view earned a
keyframe every ~50° at 0.7 px of parallax, a still camera on a weak texture
one per frame). When nothing was measured from the anchor at all (the first
link fails) the lost frame itself becomes the pending anchor (reason
``tracking_lost`` for the window). A window from a pending anchor closes like
any other; a keyframe made from one records ``anchor_is_keyframe`` false. A
window still open at the end yields no keyframe, and its record states the
largest reading and the last frame's rigid / matched counts.

WITNESSES. The same measure, an independent pass: the anchor is the last
CHOSEN witness — keyframes are witnesses too and reset it — and a frame
becomes a witness when its parallax from that anchor reaches
``witness_min_parallax_px`` (``dedup``). A loss makes no witness (nothing was
measured): the measurement continues from a pending anchor. keyframes ⊂
witnesses. A witness window that starts on an anchor the keyframe pass
measured from reads that pass's measurements (the tracker is deterministic).

NOISE REFERENCE (the warnings, the rotation verdict, the appearance test),
measured on every frame:
  * the warp TWIN — a zero-parallax video that follows the real one: frame t
    of it is the anchor warped by t's homography onto t's pixel grid,
    tracked EXACTLY like the real frames (its own prediction link, the same
    direct refinement over the windows the real frame was measured on), with
    the frame's own measured disagreement as noise: every twin window carries
    the residual window, at convergence, of another real track of the
    reference plane (fixed seed per anchor and frame). Its reading, with the
    same statistic (``twin_px``), is the tracker's error for THIS warp of
    THIS texture at THIS frame's noise;
  * the consistency match's forward–backward error / √2, q-quantile over the
    matched tracks (``fb_px``);
  * floor_t = hypot(twin_t, fb_t) (``fb_px`` alone when the twin could not
    be measured, and ``floor_source`` says so).

COVERAGE WARNINGS (advisory — nothing here changes a selection): runs of at
least ``warn_min_run_frames`` consecutive measured frames (chain order) that
are STATIC (median displacement from the anchor < max(``warn_static_disp_px``,
``rotation_floor_factor`` × floor)), PURE_ROTATION (displacement ≥
``warn_rotation_min_disp_px`` and parallax ≤ ``rotation_floor_factor`` ×
floor), DYNAMIC_CONTENT (more than (1 − ``parallax_quantile``) of the
matched tracks moved on their own beyond max(``witness_min_parallax_px``,
``rotation_floor_factor`` × floor) and were left out of the rigid set — or,
under a verdict that says the camera did not translate, stopped showing the
anchor's patch: with no translation nothing is occluded by parallax),
TRACKING_LOST (lost frames, and frames measured from an anchor whose tracks
were lost within ``warn_min_run_frames`` frames — the tracker cannot hold an
anchor there), runs of frames unusable by EXPOSURE (I0), and — written by the
exclusion audit after I2 — EXCLUDED_PARALLAX. Each names the frames it
covers and what was measured. Every frame record keeps its window's closure,
and every lost measurement of a frame is kept (``losses``) even when the frame
was measured again from a new anchor.

EXCLUSION AUDIT (:func:`audit_exclusions`, run by ``intake.run`` after I2):
every keyframe's rigid tracks are kept in ``intake/keyframe_tracks.npz``;
once I2 has written its exclusion masks, each keyframe's reading is taken
again without the tracks whose anchor position lies in the anchor's mask or
whose keyframe position lies in the keyframe's, and a keyframe whose baseline
then falls under the band (for a track-loss keyframe, under its measured
bar) is warned ``excluded_parallax`` with both readings.

DECLARED LIMITS — measured on the synthetic scenes of tests/synth_precision.py
and written in the report, never hidden:
  * A homography explains a purely PLANAR scene under translation as well as
    it explains a rotation: facing a single textured wall and walking
    sideways the parallax reads its floor and the run is flagged as a pure
    rotation. The intake cannot tell them apart from images; the walk (I3)
    can.
  * Off-plane structure on less than (1 − ``parallax_quantile``) of the rigid
    tracks does not move the statistic, and structure on about (1 − q) of
    them makes it switch between that structure's parallax and the
    background's. A window that straddles a depth edge follows the side with
    the stronger texture (the near side's motion at a far seed): on the hall
    (a box 2.5 m ahead on ≈ 9 % of the tracks) the first keyframe read 14.5 px
    where the ground truth over every static seed is 6.7 px — over the tracks
    whose windows do not straddle the box the reading follows the ground
    truth within 3 % (1.92 vs 1.91 px, 4.95 vs 5.07 px).
  * An object moving ALONG the epipolar lines of the camera's own motion is
    F-consistent and reads as structure: a person crossing sideways during a
    sideways walk gave 10 keyframes for the clean walk's 5 (textured slabs of
    10–40 % of the view crossing: 13–19). I1 cannot tell it from the images;
    the exclusion audit reports every such window once I2 has masked the
    object (excluded_parallax).
  * An object held in view while the camera walks or pans (a person the
    camera follows) is structure at its own depth to the geometry: at 10–20 %
    of the view a walk gave 7–8 keyframes for 5; at 30–40 % the majority
    becomes the object (still / dynamic_content warnings, 2 keyframes for 5),
    and one track-loss keyframe was made on a 2°/frame pan with a 30 % slab.
  * The ``still`` / ``rotation`` verdicts rest on the majority and a zero-
    skew, square-pixel pinhole: lens distortion, rolling shutter or
    electronic stabilisation fail the pinhole-rotation fit and the verdict
    stays ``epipolar``. The homography is fitted in DISTORTED pixels (no
    camera model exists at intake): a pure rotation through a k1 = −0.12
    lens reads 4–6 px after 40–60° of pan (no keyframe).
  * A window is matched only while it lies whole in frame t. On a forward
    walk the largest parallax lives at the periphery, which leaves the view
    first: the reading follows the ground truth over whole windows (corridor
    at production scale: 14 keyframes for 14; 17 over every seed), less where
    the periphery carries most of it (a 12 × 16 m room: 5 for 6, 8 over every
    seed; the fixture room walking at its back wall: 1 for 1, 4 over every
    seed, and the last 20 frames — the wall filling the view — flagged a
    pure rotation).
  * The twin's noise is the frame's own residuals, not the real noise of a
    second exposure: it reads 0.6–0.7× the error a still camera shows on a
    well-textured wall (compression makes real noise more damaging than its
    residual spread says) — ``rotation_floor_factor`` covers that — and a
    third of it on a wall whose contrast is at the sensor noise (±4 grey levels
    under σ 2): there noise readings pass the witness bar (one keyframe, a
    witness nearly every frame on a still camera).
  * A keyframe's baseline lies in the band whatever the frame rate, but the
    sharpest frame of the band follows the content's sharpness, and where one
    frame step is wider than the whole band the closest frame is taken: the
    same walk gave 32 / 32 / 30 keyframes at 60 / 30 / 15 fps (total measured
    baseline 376 / 372 / 372 px).
  * Occluded tracks are not checked geometrically (there is no depth here);
    the appearance test and the consistency match are what remove them.

Artifacts (frames/ and <session>/intake/ survive a reconstruction replace; each
carries version, provenance, geometry_epoch, camera_epoch, the effective
parameters and the inputs' identity):
  * ``frames/selected_frames.json`` — the v2 contract every consumer reads
    (``version`` "2.0", ``method`` "parallax_lk_<quantum>", ``total_frames``,
    ``selected_count``, ``selected_files`` sorted by frame number) plus the
    keyframe records and windows;
  * ``frames/witness_frames.json`` — the witness records (+ ``selected_files``
    so a v2 reader can consume it too);
  * ``<session>/intake/coverage_warnings.json`` — the warnings and every
    frame's measurement;
  * ``<session>/intake/keyframe_tracks.npz`` — every keyframe's rigid tracks
    (the exclusion audit's input); ``<session>/intake/exclusion_audit.json``
    — the audit (written after I2).

No decision literal lives here (``tests/test_intake_config.py`` and
``tests/test_precision_config.py`` scan this package): every bound is a
``ParallaxConfig`` field read from ``intake.parallax`` in config.yaml.

CLI: ``python -m intake.parallax --session <dir>`` (I0's
``quality_features.json`` must exist: run ``python -m intake.quality`` first).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from intake.config import GRAY_MAX, ParallaxConfig, load_intake_config
from intake.quality import (QUALITY_VERSION, Cancelled, QualityError, _write_json_atomic,
                            check_cancelled, list_frames, read_gray, read_session_epochs)

SELECTED_FRAMES_NAME = "selected_frames.json"
WITNESS_FRAMES_NAME = "witness_frames.json"
COVERAGE_WARNINGS_NAME = "coverage_warnings.json"
KEYFRAME_TRACKS_NAME = "keyframe_tracks.npz"
EXCLUSION_AUDIT_NAME = "exclusion_audit.json"
MASK_EXCLUDED = 255                     # intake I2's exclusion PNGs: 255 = excluded
INTAKE_SUBDIR = "intake"
SELECTED_CONTRACT_VERSION = "2.0"       # the v2 contract of frames/selected_frames.json
PARALLAX_VERSION = 3                    # 3: anchor-based parallax + rigid set (2026-09-27)
PROVENANCE = "tool_measured"
METHOD = "parallax_lk"
WARNING_KINDS = ("pure_rotation", "static", "dynamic_content", "tracking_lost", "exposure",
                 "excluded_parallax")
LOST_REASONS = ("too_few_tracks", "homography_failed", "homography_singular",
                "residuals_undefined", "too_few_rigid_tracks")
VERDICTS = ("epipolar", "still", "rotation")
TWIN_REASONS = ("twin_too_few_tracks", "twin_no_residuals", "twin_homography_failed",
                "twin_homography_singular", "twin_residuals_undefined")
FLOOR_SOURCES = ("twin+fb", "fb")
KEYFRAME_REASONS = ("first_usable_frame", "quantum", "track_loss")
WITNESS_REASONS = ("first_usable_frame", "keyframe", "dedup")
WINDOW_CLOSERS = ("quantum", "track_loss", "coverage_break", "tracking_lost", "end")
LOSS_CLOSERS = ("track_loss", "coverage_break", "tracking_lost")
LOG_TAG = "[intake.parallax]"

ScaleXY = Tuple[float, float]


class ParallaxError(RuntimeError):
    """A structural impossibility of I1 (no usable frame, a quality report
    measured on another frame inventory, an unreadable frame, frames of mixed
    sizes) — always with the exact reason."""


@dataclass(frozen=True)
class FrameMeasure:
    """One usable frame measured from its anchor: every length in NATIVE px.
    Floats are NaN where nothing was measured (``lost``)."""
    frame: int                          # video frame number
    anchor: int                         # the anchor it was measured from
    lost: bool                          # the chain could not measure this frame
    reason: Optional[str] = None        # why lost (LOST_REASONS), None otherwise
    n_links: int = 0                    # links chained from the anchor
    n_seeds: int = 0                    # tracks seeded on the anchor
    n_surviving: int = 0                # tracks matched to the anchor at this frame
    n_h_inliers: int = 0                # RANSAC homography inliers among them
    n_f_inliers: int = 0                # RANSAC fundamental-matrix inliers among them
    n_rigid: int = 0                    # the rigid set the parallax is read on
    verdict: Optional[str] = None       # VERDICTS — how the rigid set was chosen
    parallax_px: float = float("nan")   # q-quantile symmetric transfer error of the rigid tracks
                                        # w.r.t. their least-squares homography
    parallax_all_px: float = float("nan")   # q-quantile over every surviving track w.r.t. the
                                            # reference plane (evidence)
    residual_median_px: float = float("nan")  # median of the parallax_px residuals (evidence)
    disp_px: float = float("nan")       # median displacement from the anchor, surviving tracks
    dynamic_share: float = float("nan")  # 1 − n_rigid / n_surviving
    dynamic_moving_share: float = float("nan")  # tracks out of the rigid set deviating from the
                                                # static model beyond the moving bar / n_surviving
    twin_px: float = float("nan")       # the q-quantile transfer error of the warp twin
    twin_n_tracks: int = 0
    twin_reason: Optional[str] = None   # TWIN_REASONS when twin_px is NaN
    fb_px: float = float("nan")         # q-quantile of the accumulated fb error / √2
    floor_px: float = float("nan")      # hypot(twin, fb) — the measured noise reference
    floor_source: Optional[str] = None  # FLOOR_SOURCES
    rigidity: Dict[str, Any] = field(default_factory=dict)   # the tests' evidence


# ── helpers ──────────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2


def _finite_or_none(x: Any) -> Any:
    """JSON-safe values: NaN / ±inf become None (strict readers reject NaN),
    numpy scalars become Python scalars, containers are walked."""
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


def _scale_xy(scale: Union[float, ScaleXY]) -> ScaleXY:
    if isinstance(scale, (tuple, list, np.ndarray)):
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = sy = float(scale)
    if not (sx > 0.0 and sy > 0.0):
        raise ParallaxError(f"process scale must be positive, got {(sx, sy)}")
    return sx, sy


def to_native(pts_small: np.ndarray, scale: Union[float, ScaleXY]) -> np.ndarray:
    """Process-scale pixel coordinates → native pixel coordinates under the
    pixel-centre convention: pixel centre ``i`` of an image resized by ``s``
    sits at native ``(i + 0.5) / s − 0.5`` (the resize maps the image span
    [−0.5, w − 0.5] onto [−0.5, w_small − 0.5])."""
    sx, sy = _scale_xy(scale)
    p = np.asarray(pts_small, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(p)
    out[:, 0] = (p[:, 0] + 0.5) / sx - 0.5
    out[:, 1] = (p[:, 1] + 0.5) / sy - 0.5
    return out


def small_to_native_matrix(scale: Union[float, ScaleXY]) -> np.ndarray:
    """The 3×3 affine ``A`` with native = A · small (homogeneous), the matrix
    form of :func:`to_native`."""
    sx, sy = _scale_xy(scale)
    return np.array([[1.0 / sx, 0.0, 0.5 / sx - 0.5],
                     [0.0, 1.0 / sy, 0.5 / sy - 0.5],
                     [0.0, 0.0, 1.0]])


def homography_to_process(H_native: np.ndarray, scale: Union[float, ScaleXY]) -> np.ndarray:
    """A native-px homography expressed on the process-scale grid:
    H_small = A⁻¹ · H · A (``A`` from :func:`small_to_native_matrix`)."""
    A = small_to_native_matrix(scale)
    return np.linalg.inv(A) @ np.asarray(H_native, dtype=np.float64) @ A


def downscale(gray: np.ndarray, process_scale: float) -> Tuple[np.ndarray, ScaleXY]:
    """The gray frame at the tracking scale (INTER_AREA) and the EXACT
    per-axis scale that was applied (w_small / w, h_small / h — the requested
    scale rounded to whole pixels), so every result can be undone to native
    px without the rounding error. ``process_scale`` 1.0 returns the frame."""
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
    """(grid_side², 2) float32 seed points at the centres of a grid_side ×
    grid_side partition of the image inset by ``margin`` px on every side
    (the tracker insets by half its window, so every seed's window lies in
    the image), in the image's own pixel coordinates (row-major over the
    grid)."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    g = int(grid_side)
    m = int(margin)
    if g < 1 or h - 2 * m < 1 or w - 2 * m < 1 or m < 0:
        raise ParallaxError(f"seed grid needs a positive grid side and image, got "
                            f"grid_side={g}, shape={shape_hw}, margin={m}")
    xs = m + (np.arange(g, dtype=np.float64) + 0.5) * ((w - 2 * m) / float(g)) - 0.5
    ys = m + (np.arange(g, dtype=np.float64) + 0.5) * ((h - 2 * m) / float(g)) - 0.5
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)


# ── tracking ─────────────────────────────────────────────────────────────

def track_step(gray_a_small: np.ndarray, gray_b_small: np.ndarray, seeds_xy: np.ndarray,
               cfg: ParallaxConfig, scale_xy: Optional[Union[float, ScaleXY]] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One LINK: pyramidal LK from ``gray_a_small`` to ``gray_b_small`` at
    ``seeds_xy`` and back. Returns ``(pts_a, pts_b, ok, fb_px)``: the start
    and tracked positions (N, 2) in process-scale px, the OK mask (both
    directions succeeded, the target lies inside the image, and the
    forward–backward disagreement is ≤ ``cfg.fb_max_px`` NATIVE px) and the
    forward–backward disagreement of every track in native px (NaN where a
    direction failed). ``scale_xy`` is the exact scale ``downscale`` applied
    (default ``cfg.process_scale``)."""
    cv2 = _cv2()
    if gray_a_small.shape != gray_b_small.shape:
        raise ParallaxError(f"the two frames of a link differ in shape: {gray_a_small.shape} "
                            f"vs {gray_b_small.shape}")
    sx, sy = _scale_xy(cfg.process_scale if scale_xy is None else scale_xy)
    seeds = np.ascontiguousarray(np.asarray(seeds_xy, dtype=np.float32).reshape(-1, 1, 2))
    n = len(seeds)
    if n == 0:
        empty = np.zeros((0, 2))
        return empty, empty.copy(), np.zeros(0, dtype=bool), np.zeros(0)
    win = (int(cfg.lk_win), int(cfg.lk_win))
    fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(gray_a_small, gray_b_small, seeds, None,
                                            winSize=win, maxLevel=int(cfg.lk_levels))
    if fwd is None:
        fwd = np.full_like(seeds, np.nan)
        st_f = np.zeros((n, 1), dtype=np.uint8)
    bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(gray_b_small, gray_a_small,
                                            np.ascontiguousarray(fwd.astype(np.float32)), None,
                                            winSize=win, maxLevel=int(cfg.lk_levels))
    if bwd is None:
        bwd = np.full_like(seeds, np.nan)
        st_b = np.zeros((n, 1), dtype=np.uint8)
    pts_a = seeds.reshape(-1, 2).astype(np.float64)
    pts_b = fwd.reshape(-1, 2).astype(np.float64)
    back = bwd.reshape(-1, 2).astype(np.float64)
    ok = (st_f.reshape(-1) == 1) & (st_b.reshape(-1) == 1)
    ok &= np.all(np.isfinite(pts_b), axis=1) & np.all(np.isfinite(back), axis=1)
    h, w = gray_a_small.shape
    ok &= ((pts_b[:, 0] >= 0.0) & (pts_b[:, 0] <= w - 1.0)
           & (pts_b[:, 1] >= 0.0) & (pts_b[:, 1] <= h - 1.0))
    with np.errstate(invalid="ignore"):
        fb = np.hypot((back[:, 0] - pts_a[:, 0]) / sx, (back[:, 1] - pts_a[:, 1]) / sy)
    fb_px = np.where(ok, fb, np.nan)
    ok &= np.isfinite(fb) & (fb <= cfg.fb_max_px)
    return pts_a, pts_b, ok, fb_px


def gray_stats(gray: np.ndarray) -> Tuple[float, float]:
    """(mean, standard deviation) of a gray frame."""
    g = np.asarray(gray, dtype=np.float64)
    return float(g.mean()), float(g.std())


def photometric_match(gray: np.ndarray, ref: Tuple[float, float]) -> np.ndarray:
    """``gray`` mapped by the global gain/offset that gives it the reference
    mean and standard deviation (uint8, rounded and clipped to the gray
    range). LK assumes brightness constancy along each link and a chained
    track accumulates any per-link bias: measured on the synthetic room, a
    still camera under a 0.6 → 1.4 exposure ramp drifted 3.3 px over 29
    links and read 5.6 px of parallax. An exposure change is a global gain;
    mapping every frame of a window to its anchor's statistics removes it
    before tracking. A frame with no contrast is returned unchanged (its
    tracks die on their own)."""
    mean, std = gray_stats(gray)
    if not std > 0.0:
        return gray
    out = (np.asarray(gray, dtype=np.float64) - mean) * (ref[1] / std) + ref[0]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def twin_image(gray_a_small: np.ndarray, H_small: np.ndarray) -> np.ndarray:
    """The anchor warped by a homography (process-scale grid): the image
    frame t would be if the scene had no parallax at all. Bilinear, border
    reflected (tracks whose window reaches past the border are what the
    forward–backward check removes)."""
    cv2 = _cv2()
    h, w = gray_a_small.shape
    return cv2.warpPerspective(gray_a_small, np.asarray(H_small, dtype=np.float64), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def zero_mean_local(gray: np.ndarray, win: int) -> np.ndarray:
    """``gray`` minus its own mean over a ``win`` × ``win`` box (the LK
    window), re-centred on mid-grey (uint8, rounded and clipped). A gain or
    offset that varies ACROSS the frame — the rolling bands of a lamp
    flickering under a rolling shutter, vignetting that follows the view — is
    not removed by a global gain/offset (:func:`photometric_match`) and a
    translational LK reads it as motion (measured by the review probes: ±10 %
    bands gave a still camera 6.4 px of 'parallax' and two quantum
    keyframes). Removing each window's own mean removes any offset that is
    locally constant; a box filter is symmetric, so it cannot shift a
    texture. Used by the chain link and the consistency match (the refinement
    normalises every window itself)."""
    cv2 = _cv2()
    g = np.asarray(gray, dtype=np.float32)
    mean = cv2.blur(g, (int(win), int(win)), borderType=cv2.BORDER_REFLECT)
    return np.clip(np.rint(g - mean + (GRAY_MAX + 1) / 2.0), 0, GRAY_MAX).astype(np.uint8)


def tile_side(cfg: ParallaxConfig) -> int:
    """Side of the square tile one track is checked in (process px): the LK
    window plus half a window of room on each side (the rectified tile is
    centred on the refined position, so the consistency match moves little)."""
    return int(cfg.lk_win) + 2 * (int(cfg.lk_win) // 2 + 1)


def _mosaic_shape(n: int) -> Tuple[int, int]:
    cols = max(1, int(math.ceil(math.sqrt(max(n, 1)))))
    rows = max(1, int(math.ceil(n / float(cols))))
    return rows, cols


def _to_mosaic(tiles: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(n, T, T) tiles → one (rows·T, cols·T) image, row-major."""
    n, T = tiles.shape[0], tiles.shape[1]
    pad = rows * cols - n
    if pad:
        tiles = np.concatenate([tiles, np.full((pad, T, T), 128, dtype=tiles.dtype)], axis=0)
    return np.ascontiguousarray(tiles.reshape(rows, cols, T, T).transpose(0, 2, 1, 3)
                                .reshape(rows * T, cols * T))


def anchor_tiles(anchor: np.ndarray, seeds: np.ndarray, cfg: ParallaxConfig
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The anchor's own pixels around every seed — no resampling: each tile
    is centred on the seed's nearest pixel and the seed keeps its sub-pixel
    position inside it. Returns ``(tiles (n, T, T) uint8, centres (n, 2)
    int, seed position inside its tile (n, 2))``."""
    T = tile_side(cfg)
    c = T // 2
    s = np.asarray(seeds, dtype=np.float64)
    centres = np.floor(s + 0.5).astype(np.int64)
    pad = np.pad(anchor, c, mode="reflect")
    iy = centres[:, 1][:, None] + np.arange(T)[None, :]           # padded rows of each tile
    ix = centres[:, 0][:, None] + np.arange(T)[None, :]
    tiles = pad[iy[:, :, None], ix[:, None, :]]
    return tiles, centres, s - centres + c


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact (float64) bilinear samples of ``img`` at ``(x, y)`` (pixel
    centres at integers) and the mask of the points that lie inside the
    image (outside ones are sampled at the clamped position)."""
    h, w = img.shape
    fin = np.isfinite(x) & np.isfinite(y)
    xs = np.where(fin, x, 0.0)
    ys = np.where(fin, y, 0.0)
    inside = fin & (xs >= 0.0) & (xs <= w - 1.0) & (ys >= 0.0) & (ys <= h - 1.0)
    xc = np.clip(xs, 0.0, w - 1.0)
    yc = np.clip(ys, 0.0, h - 1.0)
    x0 = np.minimum(np.floor(xc).astype(np.int64), max(w - 2, 0))
    y0 = np.minimum(np.floor(yc).astype(np.int64), max(h - 2, 0))
    fx = xc - x0
    fy = yc - y0
    flat = np.asarray(img, dtype=np.float64).ravel()
    i00 = y0 * w + x0
    x1 = min(1, w - 1)
    y1 = min(1, h - 1) * w
    top = flat.take(i00) + fx * (flat.take(i00 + x1) - flat.take(i00))
    bot = flat.take(i00 + y1) + fx * (flat.take(i00 + y1 + x1) - flat.take(i00 + y1))
    return top + fy * (bot - top), inside


def window_offsets(cfg: ParallaxConfig) -> Tuple[np.ndarray, np.ndarray]:
    """(u, v) offsets of the pixels of one LK window (``lk_win`` × ``lk_win``
    around the seed, process px), row-major."""
    h = int(cfg.lk_win) // 2
    r = np.arange(-h, h + 1, dtype=np.float64)
    V, U = np.meshgrid(r, r, indexing="ij")
    return U.ravel(), V.ravel()


def _gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference gradients (d/dx, d/dy) of a gray image, float64."""
    g = np.asarray(img, dtype=np.float64)
    gy, gx = np.gradient(g)
    return gx, gy


def window_templates(anchor: np.ndarray, seeds: np.ndarray, cfg: ParallaxConfig
                     ) -> Dict[str, np.ndarray]:
    """Everything the inverse-compositional refinement needs from the
    ANCHOR, computed once per anchor: every seed's window (``lk_win`` ×
    ``lk_win``, bilinear at the seed's sub-pixel position) normalised to zero
    mean and unit variance, its steepest-descent images for the eight
    parameters of a homography acting on the window's coordinates
    (normalised by the half-window, for conditioning) and the inverse of
    their Gauss-Newton Hessian (constant for the whole window of frames — the
    point of the inverse-compositional form). ``usable`` is False where the
    window does not lie inside the anchor, has no contrast or a singular
    Hessian."""
    U, V = window_offsets(cfg)
    half = float(max(int(cfg.lk_win) // 2, 1))
    s = np.asarray(seeds, dtype=np.float64)
    X = s[:, 0][:, None] + U[None, :]
    Y = s[:, 1][:, None] + V[None, :]
    T, ins = bilinear(anchor, X, Y)
    gx, gy = _gradients(anchor)
    Tx, _i = bilinear(gx, X, Y)
    Ty, _i = bilinear(gy, X, Y)
    mean = T.mean(axis=1, keepdims=True)
    std = T.std(axis=1, keepdims=True)
    usable = np.all(ins, axis=1) & (std[:, 0] > 0.0)
    safe = np.where(std > 0.0, std, 1.0)
    Tn = (T - mean) / safe
    gxn = Tx / safe * half                  # gradients w.r.t. the normalised coordinates
    gyn = Ty / safe * half
    x = U / half
    y = V / half
    r = gxn * x + gyn * y
    SD = np.stack([gxn * x, gxn * y, gxn, gyn * x, gyn * y, gyn, -r * x, -r * y], axis=2)
    Hs = np.einsum("nki,nkj->nij", SD, SD)
    Hinv = np.zeros_like(Hs)
    det = np.linalg.det(Hs)
    usable &= np.isfinite(det) & (np.abs(det) > 0.0)
    if usable.any():
        Hinv[usable] = np.linalg.inv(Hs[usable])
    usable &= np.all(np.isfinite(Hinv.reshape(len(s), -1)), axis=1)
    return {"Tn": Tn, "SD": SD, "Hinv": Hinv, "usable": usable, "U": U, "V": V,
            "x": x, "y": y, "half": half}


def seed_warp(seeds: np.ndarray, half: float) -> np.ndarray:
    """(n, 3, 3) warps of the windows onto the anchor itself: normalised window
    coordinates (u / half, v / half) → the anchor pixel (seed + (u, v))."""
    s = np.asarray(seeds, dtype=np.float64)
    W = np.zeros((len(s), 3, 3))
    W[:, 0, 0] = half
    W[:, 1, 1] = half
    W[:, 0, 2] = s[:, 0]
    W[:, 1, 2] = s[:, 1]
    W[:, 2, 2] = 1.0
    return W


def warp_centre(W: np.ndarray) -> np.ndarray:
    """(n, 2) where each window's centre (its seed) lands: W · (0, 0, 1)."""
    Wm = np.asarray(W, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Wm[:, :2, 2] / Wm[:, 2:3, 2]


def warp_points(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Frame positions of normalised window coordinates (broadcast (n, k))."""
    Wm = np.asarray(W, dtype=np.float64)
    den = Wm[:, 2, 0][:, None] * x + Wm[:, 2, 1][:, None] * y + Wm[:, 2, 2][:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        X = (Wm[:, 0, 0][:, None] * x + Wm[:, 0, 1][:, None] * y + Wm[:, 0, 2][:, None]) / den
        Y = (Wm[:, 1, 0][:, None] * x + Wm[:, 1, 1][:, None] * y + Wm[:, 1, 2][:, None]) / den
    return X, Y


def move_centre(W: np.ndarray, to: np.ndarray) -> np.ndarray:
    """``W`` pre-composed with the translation that sends its centre to ``to``
    (the chain's predicted position), shape kept."""
    c = warp_centre(W)
    T = np.tile(np.eye(3), (len(W), 1, 1))
    T[:, 0, 2] = to[:, 0] - c[:, 0]
    T[:, 1, 2] = to[:, 1] - c[:, 1]
    return np.einsum("nij,njk->nik", T, W)


def _adjugate3(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(adjugate, determinant) of a stack of 3×3 matrices — the inverse is
    adj / det; written out because the refinement inverts one small update
    per window and iteration."""
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    d, e, f = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    g, h, i = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]
    adj = np.empty_like(M)
    adj[:, 0, 0] = e * i - f * h
    adj[:, 0, 1] = c * h - b * i
    adj[:, 0, 2] = b * f - c * e
    adj[:, 1, 0] = f * g - d * i
    adj[:, 1, 1] = a * i - c * g
    adj[:, 1, 2] = c * d - a * f
    adj[:, 2, 0] = d * h - e * g
    adj[:, 2, 1] = b * g - a * h
    adj[:, 2, 2] = a * e - b * d
    det = a * adj[:, 0, 0] + b * adj[:, 1, 0] + c * adj[:, 2, 0]
    return adj, det


def refine_windows(img: np.ndarray, tpl: Dict[str, np.ndarray], idx: np.ndarray,
                   W0: np.ndarray, cfg: ParallaxConfig, noise: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-compositional Gauss-Newton refinement of a HOMOGRAPHY per
    window (normalised window coordinates → frame t, process px) that brings
    frame t's window onto the ANCHOR's own window, for the tracks ``idx``
    started at ``W0``. A homography — not an affine map — because a PROD
    window spans ~80 native px, and a floor or ceiling patch near the camera
    changes its foreshortening across it on a forward walk (an affine window
    lost those tracks, which carry the parallax). The residual is taken
    between windows normalised to zero mean and unit variance, so a gain or
    offset that differs between the two windows — an exposure change, the
    rolling bands of a flickering lamp — cancels. Iterates until the update
    moves the window's centre by no more than ``refine_eps_px`` or
    ``refine_max_iter`` iterations (``cfg``). ``noise`` ((len(idx), window
    pixels), grey levels) is added to frame t's samples. Returns ``(W,
    converged, rms, nrms, residual windows)``: rms / nrms the residual at
    convergence in grey levels of frame t's window / in units of the
    window's own spread, and the residual windows themselves (grey levels,
    (len(idx), window pixels))."""
    x, y = tpl["x"], tpl["y"]
    half = float(tpl["half"])
    SD = tpl["SD"][idx]
    Hinv = tpl["Hinv"][idx]
    Tn = tpl["Tn"][idx]
    W = np.array(W0, dtype=np.float64, copy=True)
    n = len(idx)
    conv = np.zeros(n, dtype=bool)
    rms = np.full(n, np.nan)
    nrms = np.full(n, np.nan)
    res_win = np.full((n, len(x)), np.nan)
    active = np.all(np.isfinite(W.reshape(n, 9)), axis=1)
    for _it in range(int(cfg.refine_max_iter)):
        a = np.flatnonzero(active)
        if len(a) == 0:
            break
        X, Y = warp_points(W[a], x[None, :], y[None, :])
        Iw, ins = bilinear(img, X, Y)
        if noise is not None:
            Iw = Iw + noise[a]
        m = Iw.mean(axis=1, keepdims=True)
        sd = Iw.std(axis=1, keepdims=True)
        # a window is matched only while it lies whole in frame t: part of a window
        # cannot testify to the anchor's patch (refined on the part inside, eight
        # parameters slid along the border; with the shape frozen, the position was
        # biased by a shape that kept changing)
        outside = ~np.all(ins, axis=1)
        flat = ~(sd[:, 0] > 0.0) | outside
        e = (Iw - m) / np.where(sd > 0.0, sd, 1.0) - Tn[a]
        nrms[a] = np.sqrt(np.mean(e * e, axis=1))
        rms[a] = nrms[a] * sd[:, 0]
        res_win[a] = e * sd
        dp = np.einsum("nij,nj->ni", Hinv[a], np.einsum("nki,nk->ni", SD[a], e))
        D = np.zeros((len(a), 3, 3))
        D[:, 0, :] = dp[:, 0:3]
        D[:, 1, :] = dp[:, 3:6]
        D[:, 2, 0:2] = dp[:, 6:8]
        D[:, 0, 0] += 1.0
        D[:, 1, 1] += 1.0
        D[:, 2, 2] = 1.0
        adj, detD = _adjugate3(D)
        bad = flat | ~np.isfinite(detD) | (detD <= 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Dinv = np.where(bad[:, None, None], np.eye(3)[None], adj / detD[:, None, None])
        Wn = np.einsum("nij,njk->nik", W[a], Dinv)
        # convergence is judged on what the refinement measures — where the
        # window's centre (the track) lands in frame t; the perspective terms of a
        # window's shape are weakly determined and kept moving its corners by more
        # than the bound while the centre had long settled
        step = np.hypot(*(warp_centre(Wn) - warp_centre(W[a])).T)
        good = (~bad & np.all(np.isfinite(Wn.reshape(-1, 9)), axis=1) & np.isfinite(step))
        W[a[good]] = Wn[good]
        done = good & (step <= cfg.refine_eps_px)
        conv[a[done]] = True
        active[a[done | ~good]] = False
    return W, conv, rms, nrms, res_win


def rectified_tiles(img: np.ndarray, seeds: np.ndarray, centres: np.ndarray, W: np.ndarray,
                    half: float, ref_tiles: np.ndarray, cfg: ParallaxConfig) -> np.ndarray:
    """Frame t resampled onto every anchor tile's grid through the track's
    refined window homography, each tile then given its anchor tile's mean
    and standard deviation (a per-window gain/offset — the refinement's own
    normalisation) — the input of the consistency match (:func:`tile_match`;
    OpenCV's bilinear remap — the tile only CHECKS, the position comes
    from the exact refinement). (n, T, T) uint8."""
    Tside = tile_side(cfg)
    c = Tside // 2
    n = len(seeds)
    off = np.arange(Tside, dtype=np.float64) - c
    s = np.asarray(seeds, dtype=np.float64)
    dx = np.broadcast_to((centres[:, 0] - s[:, 0])[:, None, None] + off[None, None, :],
                         (n, Tside, Tside))
    dy = np.broadcast_to((centres[:, 1] - s[:, 1])[:, None, None] + off[None, :, None],
                         (n, Tside, Tside))
    X, Y = warp_points(W, (dx / half).reshape(n, -1), (dy / half).reshape(n, -1))
    X = np.where(np.isfinite(X), X, -1.0).reshape(n, Tside, Tside)
    Y = np.where(np.isfinite(Y), Y, -1.0).reshape(n, Tside, Tside)
    rows, cols = _mosaic_shape(n)
    MX = _to_mosaic(X.astype(np.float32), rows, cols)
    MY = _to_mosaic(Y.astype(np.float32), rows, cols)
    cv2 = _cv2()
    v = cv2.remap(np.asarray(img, dtype=np.float32), MX, MY, interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT_101)
    v = (v.reshape(rows, Tside, cols, Tside).transpose(0, 2, 1, 3)
         .reshape(rows * cols, Tside, Tside)[:n].astype(np.float64))
    ref = np.asarray(ref_tiles, dtype=np.float64)
    m, sd = v.mean(axis=(1, 2), keepdims=True), v.std(axis=(1, 2), keepdims=True)
    rm, rsd = ref.mean(axis=(1, 2), keepdims=True), ref.std(axis=(1, 2), keepdims=True)
    out = (v - m) * np.where(sd > 0.0, rsd / np.where(sd > 0.0, sd, 1.0), 1.0) + rm
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def neighbour_warps(seeds: np.ndarray, W: np.ndarray, donors: np.ndarray, grid_side: int,
                    half: float) -> Tuple[np.ndarray, np.ndarray]:
    """For every seed, what its 8 grid neighbours that are DONORS (carried by
    the chain link) say its window warp is: each donor's own window
    homography shifted by the seed offset, normalised, elementwise median over
    the donors; and the number of donors. A track the refinement lost for a
    frame is predicted by its own surface, not by the dominant plane — the
    plane's homography sends a near floor or ceiling patch several pixels off
    on a forward walk and the refinement never finds it again.
    ((n, 3, 3), (n,))."""
    import warnings
    g = int(grid_side)
    S = np.asarray(seeds, dtype=np.float64).reshape(g, g, 2)
    Wn = np.where(donors[:, None, None], W / W[:, 2:3, 2:3], np.nan).reshape(g, g, 3, 3)
    padS = np.pad(S, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padW = np.pad(Wn, ((1, 1), (1, 1), (0, 0), (0, 0)), constant_values=np.nan)
    cands = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sn = padS[1 + dy:1 + dy + g, 1 + dx:1 + dx + g]
            wn = padW[1 + dy:1 + dy + g, 1 + dx:1 + dx + g]
            d = (S - sn) / half
            Tm = np.zeros((g, g, 3, 3))
            Tm[..., 0, 0] = 1.0
            Tm[..., 1, 1] = 1.0
            Tm[..., 2, 2] = 1.0
            Tm[..., 0, 2] = d[..., 0]
            Tm[..., 1, 2] = d[..., 1]
            M = np.einsum("yxij,yxjk->yxik", wn, Tm)
            cands.append(M / M[..., 2:3, 2:3])
    C = np.stack(cands, axis=0).reshape(8, g * g, 3, 3)
    count = np.sum(np.all(np.isfinite(C.reshape(8, g * g, 9)), axis=2), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(C, axis=0)
    return med, count


def tile_match(a_tiles: np.ndarray, t_tiles: np.ndarray, inside: np.ndarray,
               cfg: ParallaxConfig, scale_xy: ScaleXY
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The CONSISTENCY match of every track: LK at full resolution (no pyramid:
    the rectified tile is already centred on the refined position) from the
    seed's position in its anchor tile into the rectified tile, started at
    the same position, then back with no initial guess. Returns ``(delta, ok, fb_px, err)``: the
    match's offset on the ANCHOR grid (process px), the OK mask (both
    directions found, forward–backward disagreement ≤ ``fb_max_px`` native
    px), the disagreement (native px) and LK's photometric residual at
    convergence (mean |Δgrey| over the window, process-scale grey levels)."""
    cv2 = _cv2()
    sx, sy = scale_xy
    n = len(inside)
    if n == 0:
        z = np.zeros((0, 2))
        return z, np.zeros(0, dtype=bool), np.zeros(0), np.zeros(0)
    T = tile_side(cfg)
    rows, cols = _mosaic_shape(n)
    A = _to_mosaic(a_tiles, rows, cols)
    B = _to_mosaic(t_tiles, rows, cols)
    k = np.arange(n)
    origin = np.c_[(k % cols) * T, (k // cols) * T].astype(np.float64)
    p0 = np.ascontiguousarray((origin + inside).astype(np.float32).reshape(-1, 1, 2))
    win = (int(cfg.lk_win), int(cfg.lk_win))
    fwd, st_f, err = cv2.calcOpticalFlowPyrLK(A, B, p0, p0.copy(), winSize=win, maxLevel=0,
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(B, A, np.ascontiguousarray(fwd.astype(np.float32)),
                                            None, winSize=win, maxLevel=0)
    q = fwd.reshape(-1, 2).astype(np.float64)
    back = bwd.reshape(-1, 2).astype(np.float64)
    a = p0.reshape(-1, 2).astype(np.float64)
    ok = (st_f.reshape(-1) == 1) & (st_b.reshape(-1) == 1)
    ok &= np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(back), axis=1)
    # the match must stay inside its own tile's search area
    rel = q - origin
    ok &= np.all((rel >= 0.0) & (rel <= T - 1.0), axis=1)
    with np.errstate(invalid="ignore"):
        fb = np.hypot((back[:, 0] - a[:, 0]) / sx, (back[:, 1] - a[:, 1]) / sy)
    ok &= np.isfinite(fb) & (fb <= cfg.fb_max_px)
    e = np.full(n, np.nan)
    e[ok] = err.reshape(-1)[ok]
    return q - a, ok, np.where(ok, fb, np.nan), e


# ── two-view geometry ────────────────────────────────────────────────────

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


def fit_homography(a: np.ndarray, b: np.ndarray, cfg: ParallaxConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC homography a → b (bound ``cfg.ransac_px``, native px) and its
    inlier mask; ``(None, all-False)`` when the fit fails."""
    cv2 = _cv2()
    n = len(a)
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    H, mask = cv2.findHomography(a.astype(np.float64), b.astype(np.float64), cv2.RANSAC,
                                 float(cfg.ransac_px))
    if H is None or mask is None:
        return None, np.zeros(n, dtype=bool)
    return H, mask.reshape(-1).astype(bool)


def fit_fundamental(a: np.ndarray, b: np.ndarray, cfg: ParallaxConfig
                    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC fundamental matrix a → b (bound ``cfg.ransac_px``, native px)
    and its inlier mask; ``(None, all-False)`` when the fit fails (fewer than
    eight tracks, or a configuration the estimator rejects)."""
    cv2 = _cv2()
    n = len(a)
    if n < 8:
        return None, np.zeros(n, dtype=bool)
    try:
        F, mask = cv2.findFundamentalMat(a.astype(np.float64), b.astype(np.float64),
                                         cv2.FM_RANSAC, float(cfg.ransac_px))
    except cv2.error:
        return None, np.zeros(n, dtype=bool)
    if F is None or mask is None or F.shape != (3, 3):
        return None, np.zeros(n, dtype=bool)
    return F, mask.reshape(-1).astype(bool)


def _similarity(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: the similarity that moves the points' centroid
    to the origin and their mean distance to √2 (conditioning only)."""
    c = pts.mean(axis=0)
    d = float(np.mean(np.hypot(*(pts - c).T)))
    s = math.sqrt(2.0) / d if d > 0.0 else 1.0
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def _det_normalised(M: np.ndarray) -> np.ndarray:
    det = float(np.linalg.det(M))
    return M / np.cbrt(det) if det != 0.0 else M


def _off_image_residual(a: np.ndarray, b: np.ndarray) -> float:
    """The residual a model pays for sending a point to infinity during a
    least-squares iteration: the span of the coordinates involved — at least
    as bad as mapping it off the image, and finite so the solver can step
    back."""
    return float(np.abs(a).max() + np.abs(b).max())


def pair_discriminant(M: np.ndarray) -> float:
    """Signed evidence of a rotation in a 3×3 (det-normalised, similarity-
    conjugated) homography: ((λᵢ − λⱼ)/2)² for its two CLOSEST eigenvalues —
    −Im² < 0 for a complex pair (a rotation: eigenvalues 1, e^{±iθ} give
    −sin²θ), ≥ 0 when all three are real (a translation of a plane: a
    homology {1, 1, μ} or an elation {1, 1, 1}). Continuous across the two
    cases, so noise straddles zero where there is no rotation. K-free:
    eigenvalues do not change under conjugation."""
    ev = np.linalg.eigvals(np.asarray(M, dtype=np.float64))
    if np.any(ev.imag != 0.0):
        k = np.argsort(-np.abs(ev.imag))[:2]
        d = (ev[k[0]] - ev[k[1]]) / 2.0
        return float((d * d).real)
    r = np.sort(ev.real)
    gaps = np.diff(r)
    j = int(np.argmin(gaps))
    return float((gaps[j] / 2.0) ** 2)


def _dlt_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 2, 9) DLT rows of the correspondences a → b (x' ≅ H x)."""
    n = len(a)
    ah = np.c_[a, np.ones(n)]
    rows = np.zeros((n, 2, 9))
    rows[:, 0, 3:6] = -ah
    rows[:, 0, 6:9] = b[:, 1:2] * ah
    rows[:, 1, 0:3] = ah
    rows[:, 1, 6:9] = -b[:, 0:1] * ah
    return rows


def bootstrap_homographies(a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
                           ) -> np.ndarray:
    """``n_boot`` least-squares (normalised DLT) homographies of the tracks
    resampled with replacement: every resample's normal matrix is the
    multinomial-weighted sum of the per-track DLT outer products, solved at
    once (the smallest eigenvector). (n_boot, 3, 3), in the input frame."""
    Ta, Tb = _similarity(a), _similarity(b)
    an = _apply_h(Ta, a)
    bn = _apply_h(Tb, b)
    rows = _dlt_rows(an, bn)
    outer = np.einsum("nki,nkj->nij", rows, rows).reshape(len(a), 81)
    n = len(a)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    counts = np.zeros((int(n_boot), n))
    np.add.at(counts, (np.repeat(np.arange(int(n_boot)), n), idx.ravel()), 1.0)
    M = (counts @ outer).reshape(int(n_boot), 9, 9)
    _w, V = np.linalg.eigh(M)
    Hn = V[:, :, 0].reshape(int(n_boot), 3, 3)
    return np.einsum("ij,njk,kl->nil", np.linalg.inv(Tb), Hn, Ta)


def rotation_evidence(a: np.ndarray, b: np.ndarray, H: np.ndarray, cfg: ParallaxConfig,
                      seed: Sequence[int]) -> Dict[str, Any]:
    """Is there a rotation in the homography the tracks follow? The pair
    discriminant of H (conjugated by the tracks' Hartley similarity, det 1)
    and its bootstrap: ``cfg.rigidity_bootstrap`` resamples of the tracks
    with replacement, each refitted by least squares (the normalised DLT,
    all resamples solved together; fixed seed → reproducible); ``evident``
    when the one-sided upper bound at ``cfg.rigidity_confidence`` stays below
    zero. ``sin_theta`` is the rotation's |Im λ| (NaN without a complex
    pair)."""
    T = _similarity(a)
    T_inv = np.linalg.inv(T)
    D0 = pair_discriminant(_det_normalised(T @ H @ T_inv))
    rng = np.random.default_rng(list(seed))
    Hb = bootstrap_homographies(a, b, int(cfg.rigidity_bootstrap), rng)
    Ds: List[float] = []
    failed = 0
    for k in range(len(Hb)):
        if not np.all(np.isfinite(Hb[k])) or float(np.linalg.det(Hb[k])) == 0.0:
            failed += 1
            continue
        Ds.append(pair_discriminant(_det_normalised(T @ Hb[k] @ T_inv)))
    upper = float(np.quantile(Ds, cfg.rigidity_confidence)) if Ds else float("nan")
    return {"D": D0, "D_upper": upper, "n_boot": len(Ds), "n_boot_failed": failed,
            "sin_theta": math.sqrt(-D0) if D0 < 0.0 else float("nan"),
            "evident": bool(Ds) and upper < 0.0}


def _rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    R, _ = _cv2().Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def _nearest_rotation(M: np.ndarray) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(_det_normalised(M))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    return R


def _focal_from_conic(H: np.ndarray, pp: Tuple[float, float], a: np.ndarray) -> Optional[float]:
    """Closed-form focal length of a pure rotation with a known principal
    point: the image of the absolute conic diag(1, 1, f²) (pp-centred,
    zero skew, square pixels) is invariant under H (Mᵀ·ω·M = ω), linear in
    f². None when no positive f² solves it."""
    u, v = pp
    d = float(np.mean(np.hypot(a[:, 0] - u, a[:, 1] - v)))
    s = 1.0 / d if d > 0.0 else 1.0
    T = np.array([[s, 0.0, -s * u], [0.0, s, -s * v], [0.0, 0.0, 1.0]])
    M = _det_normalised(T @ H @ np.linalg.inv(T))
    A0 = M.T @ np.diag([1.0, 1.0, 0.0]) @ M - np.diag([1.0, 1.0, 0.0])
    e3 = np.outer([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    A1 = M.T @ e3 @ M - e3
    den = float(np.sum(A1 * A1))
    if den <= 0.0:
        return None
    c = -float(np.sum(A0 * A1)) / den
    return math.sqrt(c) / s if c > 0.0 else None


def fit_pinhole_rotation(a: np.ndarray, b: np.ndarray, H0: np.ndarray,
                         pp: Tuple[float, float], sin_theta: float) -> Dict[str, Any]:
    """The pure rotation of a zero-skew, square-pixel pinhole whose focal
    length AND principal point are unknown: H = K·R·K⁻¹ with K = [[f, 0, cx],
    [0, f, cy], [0, 0, 1]], 6 parameters, fitted to the tracks' forward
    transfer error by Levenberg–Marquardt. ``pp`` (the image centre) is only
    the starting point: a principal point 2 % of the image off-centre —
    ordinary on phones and on stabilised crops — failed a centre-pinned fit,
    the verdict fell back to epipolar and a person crossing a pure pan read
    as parallax. Two starts for f: the invariant-conic closed form and
    median displacement / sin θ (a rotation by θ moves the image by ≈ f·θ);
    the lower residual wins. Returns ``{"H", "f_px", "pp_px", "angle_deg"}``
    or ``{"reason"}``."""
    from scipy.optimize import least_squares
    starts = []
    fc = _focal_from_conic(H0, pp, a)
    if fc is not None and math.isfinite(fc):
        starts.append(fc)
    disp = float(np.median(np.hypot(*(b - a).T)))
    if math.isfinite(sin_theta) and sin_theta > 0.0 and disp > 0.0:
        starts.append(disp / sin_theta)
    if not starts:
        return {"reason": "no_focal_start"}
    big = _off_image_residual(a, b)

    def model(x):
        f_, cx, cy = x[0], x[1], x[2]
        K = np.array([[f_, 0.0, cx], [0.0, f_, cy], [0.0, 0.0, 1.0]])
        K_inv = np.array([[1.0 / f_, 0.0, -cx / f_], [0.0, 1.0 / f_, -cy / f_], [0.0, 0.0, 1.0]])
        return K @ _rotvec_to_matrix(x[3:6]) @ K_inv

    def resid(x):
        r_ = (_apply_h(model(x), a) - b).ravel()
        return np.where(np.isfinite(r_), r_, big)

    best = None
    for f0 in starts:
        K0 = np.array([[f0, 0.0, pp[0]], [0.0, f0, pp[1]], [0.0, 0.0, 1.0]])
        w0, _ = _cv2().Rodrigues(_nearest_rotation(np.linalg.inv(K0) @ H0 @ K0))
        sol = least_squares(resid, np.array([f0, pp[0], pp[1], *w0.ravel()]), method="lm")
        rss = float(np.sum(sol.fun ** 2))
        if sol.x[0] > 0.0 and np.all(np.isfinite(sol.x)) and (best is None or rss < best[0]):
            best = (rss, sol.x)
    if best is None:
        return {"reason": "no_positive_focal"}
    x = best[1]
    return {"H": model(x), "f_px": float(x[0]), "pp_px": [float(x[1]), float(x[2])],
            "angle_deg": float(np.degrees(np.linalg.norm(x[3:6])))}


def fit_plane_translation(a: np.ndarray, b: np.ndarray, H0: np.ndarray) -> Dict[str, Any]:
    """The homography a plane induces under a pure TRANSLATION of any pinhole:
    H ∝ I + u·vᵀ (a planar homology, or an elation when vᵀu = 0 — a rank-one
    update of the identity whatever K is), fitted to the tracks' forward
    transfer error by Levenberg–Marquardt in one Hartley frame shared by both
    images (the structure survives the conjugation). Initialised from ``H0``
    scaled by its median eigenvalue (the homology's repeated one) and the
    best rank-one approximation of the remainder. Returns ``{"H"}`` or
    ``{"reason"}``. A pure rotation (K·R·K⁻¹ − I has rank two) is not in this
    family; a small translation of a plane is not a rotation — the two tests
    together are what separates them past the affine limit."""
    from scipy.optimize import least_squares
    T = _similarity(a)
    T_inv = np.linalg.inv(T)
    M = T @ np.asarray(H0, dtype=np.float64) @ T_inv
    ev = np.linalg.eigvals(M)
    lam = float(np.median(ev.real))
    if not (math.isfinite(lam) and lam != 0.0):
        return {"reason": "no_plane_translation_start"}
    U, sv, Vt = np.linalg.svd(M / lam - np.eye(3))
    x0 = np.r_[U[:, 0] * math.sqrt(sv[0]), Vt[0] * math.sqrt(sv[0])]
    big = _off_image_residual(a, b)

    def model(x):
        return T_inv @ (np.eye(3) + np.outer(x[:3], x[3:6])) @ T

    def resid(x):
        r_ = (_apply_h(model(x), a) - b).ravel()
        return np.where(np.isfinite(r_), r_, big)

    sol = least_squares(resid, x0, method="lm")
    return {"H": model(sol.x)}


def _q_symmetric_error(H: np.ndarray, a: np.ndarray, b: np.ndarray, q: float) -> float:
    try:
        err = symmetric_transfer_error(H, np.linalg.inv(H), a, b)
    except np.linalg.LinAlgError:
        return float("inf")
    err = err[np.isfinite(err)]
    return float(np.quantile(err, q)) if err.size else float("inf")


def kept_share(M: np.ndarray, a: np.ndarray, b: np.ndarray, bound: float) -> float:
    """Share of the tracks a model M keeps within ``bound``: forward transfer
    error |M·a − b| ≤ bound — the RANSAC inlier test, applied to any model."""
    if len(a) == 0:
        return float("nan")
    e = np.hypot(*(_apply_h(M, a) - b).T)
    return float(np.mean(np.isfinite(e) & (e <= bound)))


def rigidity(a: np.ndarray, b: np.ndarray, H: np.ndarray, in_h: np.ndarray,
             in_f: np.ndarray, cfg: ParallaxConfig, seed: Sequence[int],
             pp: Tuple[float, float], floor_px: float) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """The rigid set and how it was chosen (module docstring, step 3).
    Returns ``(rigid mask, verdict, evidence)``.

      * ``epipolar`` when the off-homography F-inliers are not a minority of
        the homography's inliers (F is fixed by the majority): rigid = F-inliers.
      * ``still`` when the identity keeps at least the ``parallax_quantile``
        share of what the homography keeps within ``ransac_px`` (the majority
        did not move at the resolution the rigid set is defined at).
      * ``rotation`` when the homography carries a rotation (bootstrap
        eigenvalue evidence, :func:`rotation_evidence`), the pure rotation of
        a pinhole (:func:`fit_pinhole_rotation`) explains the majority at its
        own noise floor — the ``parallax_quantile`` quantile of its symmetric
        transfer error over the homography's inliers ≤
        ``rotation_floor_factor`` × ``floor_px``, the pure-rotation test of
        the warnings applied to the majority — AND the translation of a plane
        (:func:`fit_plane_translation`) does NOT explain it at that floor. At
        the RANSAC resolution a rotation with a free focal length mimics a
        small translation of a plane (the affine limit: measured, a sideways
        walk past a wall fitted a "rotation" with f 2 000–6 000 px for 228);
        the measured tracker error and the competing model do not. Both
        explaining it is ambiguous, and ambiguity keeps the parallax
        (``epipolar``).
      * ``epipolar`` otherwise.
    In ``still`` / ``rotation`` the rigid set is the homography's inliers."""
    in_h = np.asarray(in_h, dtype=bool)
    in_f = np.asarray(in_f, dtype=bool)
    off = in_f & ~in_h
    q = cfg.parallax_quantile
    ev: Dict[str, Any] = {"n_h_inliers": int(in_h.sum()), "n_f_inliers": int(in_f.sum()),
                          "n_off_h_epipolar": int(off.sum()), "floor_px": floor_px}
    if int(in_h.sum()) <= int(off.sum()):
        ev["why"] = ("the off-homography F-inliers are not a minority: F is fixed by the "
                     "majority")
        return in_f, "epipolar", ev
    ai, bi = a[in_h], b[in_h]
    kept_h = kept_share(H, ai, bi, cfg.ransac_px)
    ev["kept_homography"] = kept_h
    kept_i = kept_share(np.eye(3), ai, bi, cfg.ransac_px)
    still = {"kept_identity": kept_i, "consistent": bool(kept_i >= q * kept_h)}
    ev["still"] = still
    if still["consistent"]:
        ev["why"] = ("the homography majority did not move: the identity keeps its inliers "
                     "within ransac_px as the homography does")
        return in_h, "still", ev
    rot = rotation_evidence(ai, bi, H, cfg, seed)
    bar = cfg.rotation_floor_factor * floor_px
    rot["bar_px"] = bar
    if not rot["evident"]:
        rot.update(consistent=False, reason="no_rotation_evidence")
    elif not (math.isfinite(bar) and bar > 0.0):
        rot.update(consistent=False, reason="floor_not_measured")
    else:
        fit = fit_pinhole_rotation(ai, bi, H, pp, rot["sin_theta"])
        if "reason" in fit:
            rot.update(consistent=False, reason=fit["reason"])
        else:
            qerr = _q_symmetric_error(fit["H"], ai, bi, q)
            rot.update(rotation_error_px=qerr, f_px=fit["f_px"], pp_px=fit["pp_px"],
                       angle_deg=fit["angle_deg"])
            plane = fit_plane_translation(ai, bi, H)
            perr = (_q_symmetric_error(plane["H"], ai, bi, q) if "H" in plane
                    else float("inf"))
            rot["plane_translation_error_px"] = perr
            if qerr > bar:
                rot.update(consistent=False, reason="rotation_error_above_the_floor")
            elif perr <= bar:
                rot.update(consistent=False,
                           reason="a_plane_translation_explains_it_as_well")
            else:
                rot["consistent"] = True
    ev["rotation"] = rot
    if rot["consistent"]:
        ev["why"] = ("the homography majority moved by a pure rotation of a pinhole, at its "
                     "own noise floor: no rigid parallax exists relative to it")
        return in_h, "rotation", ev
    ev["why"] = "the homography majority translated (neither still nor a pure rotation)"
    return in_f, "epipolar", ev


def fit_two_view(a: np.ndarray, b: np.ndarray, cfg: ParallaxConfig,
                 follow: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """RANSAC homography + fundamental matrix of tracks a → b (native px).

    The homography is the parallax's REFERENCE PLANE. ``follow`` (a mask over
    the tracks: the previous frame's homography inliers) keeps the same plane
    along the window — RANSAC restricted to those tracks, inliers then read
    over every track at ``ransac_px`` — so the reading grows with the
    baseline instead of jumping when another plane's support overtakes it
    (``reference`` = "followed"). With no mask, or fewer than ``min_tracks``
    of its tracks left, or no fit on them, RANSAC runs over every track
    (``reference`` = "selected"). ``lost`` / ``reason`` (LOST_REASONS) when
    fewer than ``min_tracks`` tracks or no usable homography; a fundamental
    matrix that cannot be estimated leaves the F-inliers = the H-inliers
    (recorded)."""
    n = len(a)
    out: Dict[str, Any] = {"n_trusted": n,
                           "disp_px": float(np.median(np.hypot(*(b - a).T))) if n else float("nan")}
    if n < cfg.min_tracks:
        return dict(out, lost=True, reason="too_few_tracks")
    H, reference = None, "selected"
    if follow is not None and int(np.count_nonzero(follow)) >= cfg.min_tracks:
        Hf, _m = fit_homography(a[follow], b[follow], cfg)
        if Hf is not None:
            e = np.hypot(*(_apply_h(Hf, a) - b).T)
            in_h = np.isfinite(e) & (e <= cfg.ransac_px)
            if int(in_h.sum()) >= 4:
                H, reference = Hf, "followed"
    if H is None:
        H, in_h = fit_homography(a, b, cfg)
    if H is None:
        return dict(out, lost=True, reason="homography_failed")
    out["reference"] = reference
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return dict(out, lost=True, reason="homography_singular")
    F, in_f = fit_fundamental(a, b, cfg)
    note = None
    if F is None:
        in_f = in_h.copy()
        note = "fundamental_failed: the F-inliers are taken as the H-inliers"
    return dict(out, lost=False, reason=None, H=H, H_inv=H_inv, in_h=in_h, in_f=in_f, F=F,
                fundamental_note=note)


def read_parallax(a: np.ndarray, b: np.ndarray, fit: Dict[str, Any], cfg: ParallaxConfig, *,
                  pp: Tuple[float, float], floor_px: float, seed: Sequence[int]
                  ) -> Dict[str, Any]:
    """The rigid set (:func:`rigidity`) and the parallax read on it, from a
    successful :func:`fit_two_view`. Returns ``fit`` extended with
    ``rigid``, ``verdict``, ``rigidity``, ``sym``, ``parallax_px``,
    ``residual_median_px`` — or ``lost`` with ``too_few_rigid_tracks``."""
    rigid, verdict, ev = rigidity(a, b, fit["H"], fit["in_h"], fit["in_f"], cfg, seed, pp,
                                  floor_px)
    if fit.get("fundamental_note"):
        ev["fundamental"] = fit["fundamental_note"]
    sym = symmetric_transfer_error(fit["H"], fit["H_inv"], a, b)
    rig = rigid & np.isfinite(sym)
    base = dict(fit, n_h_inliers=int(fit["in_h"].sum()), n_f_inliers=int(fit["in_f"].sum()),
                n_rigid=int(rig.sum()), verdict=verdict, rigidity=ev)
    if int(rig.sum()) < cfg.min_tracks:
        return dict(base, lost=True, reason="too_few_rigid_tracks")
    q = cfg.parallax_quantile
    ref, ref_sym = least_squares_reference(a[rig], b[rig])
    if ref is None:
        return dict(base, lost=True, reason="homography_failed")
    ev["parallax_on_the_plane_px"] = float(np.quantile(sym[rig], q))
    return dict(base, rigid=rig, sym=sym, H_ls=ref, parallax_px=float(np.quantile(ref_sym, q)),
                residual_median_px=float(np.median(ref_sym)))


def least_squares_reference(a: np.ndarray, b: np.ndarray
                            ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """The parallax's REFERENCE: the homography that best explains the rigid
    tracks as a whole (least squares over all of them — no inlier choice)
    and the symmetric transfer error of each track w.r.t. it: the image
    motion NO homography explains. Unique for a given set of tracks, so the
    reading does not depend on which plane a RANSAC happened to follow (on
    the synthetic room the plane-referenced reading of the same frames
    differed by 2x with the plane RANSAC picked, and the keyframe count of
    one walk changed with its frame density). ``(None, empty)`` when the fit
    fails."""
    cv2 = _cv2()
    H, _m = cv2.findHomography(a.astype(np.float64), b.astype(np.float64), 0)
    if H is None or not np.all(np.isfinite(H)):
        return None, np.zeros(0)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, np.zeros(0)
    sym = symmetric_transfer_error(H, H_inv, a, b)
    return H, sym[np.isfinite(sym)]


def epipolar_distance(F: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """First-order (Sampson) distance of each correspondence a → b to the
    epipolar geometry F, native px."""
    ah = np.concatenate([a, np.ones((len(a), 1))], axis=1)
    bh = np.concatenate([b, np.ones((len(b), 1))], axis=1)
    Fa = ah @ F.T
    Ftb = bh @ F
    num = np.sum(bh * Fa, axis=1)
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(num) / np.sqrt(den)


def measure_tracks(a: np.ndarray, b: np.ndarray, cfg: ParallaxConfig, *,
                   pp: Tuple[float, float], floor_px: float,
                   seed: Sequence[int] = (0,)) -> Dict[str, Any]:
    """The whole two-view measurement of trusted tracks a → b (native px)
    given the frame's measured noise floor: :func:`fit_two_view` then
    :func:`read_parallax`. Pure: the same tracks and floor give the same
    result."""
    fit = fit_two_view(a, b, cfg)
    if fit["lost"]:
        return fit
    return read_parallax(a, b, fit, cfg, pp=pp, floor_px=floor_px, seed=seed)


# ── the anchor chain ─────────────────────────────────────────────────────

class AnchorChain:
    """The anchor's seeds followed frame by frame and MATCHED TO THE ANCHOR at
    every frame (module docstring, THE MEASURE). ``step(frame, small)``
    measures the next usable frame.

    Each frame: (1) the windows matched at the previous frame follow one LK
    link (t−1 → t, locally zero-mean frames) — a PREDICTION only; the others
    take the shape their carried neighbours' windows say (their own surface)
    or the link's frame-to-frame homography; (2) every window still in view
    is REFINED DIRECTLY against the anchor's own window — an
    inverse-compositional homography per window (:func:`refine_windows`),
    started at its prediction and its last refined shape, on windows
    normalised to zero mean and unit variance; (3) the refined window is
    CHECKED by OpenCV's LK on the rectified tile (status, and
    forward–backward ≤ ``fb_max_px``, :func:`tile_match`); (4) its residual
    must sit at the photometric floor the warp twin measured (APPEARANCE,
    :meth:`step`). A track's position at t is its refined window's centre:
    its error is one match's error whatever the number of links behind it,
    and whether a track is usable at t is decided at t alone — never by the
    product of 50 per-link checks. A track leaves for good only when its
    prediction leaves the view (never re-seeded); a window is matched only
    while it lies whole in frame t. The seeds are inset by half a window so
    every window lies in the anchor.

    The WARP TWIN (:meth:`_twin`) is a zero-parallax video following the
    real one — the anchor warped by each frame's homography — tracked the
    same way, with the frame's own measured disagreement as its noise.

    DYNAMIC MARKS (window-scoped): when a frame's verdict says the majority
    did not translate (``still`` / ``rotation``), a trusted track whose
    transfer error w.r.t. the majority's homography exceeds
    ``witness_min_parallax_px`` — the smallest parallax the intake acts on —
    is moving on its own and leaves the trusted set: an object that stays in
    view while the anchor's static tracks leave it cannot become the
    majority later in the window. The first ``epipolar`` verdict (the
    majority translated) releases every mark: the evidence they rested on
    is gone, and a track marked while a far majority hid the translation
    comes back as parallax."""

    def __init__(self, anchor: int, anchor_small: np.ndarray, scale: ScaleXY,
                 cfg: ParallaxConfig, *, native_wh: Tuple[int, int]):
        self.anchor = int(anchor)
        self.cfg = cfg
        self.scale = _scale_xy(scale)
        self.anchor_small = anchor_small
        self.anchor_stats = gray_stats(anchor_small)     # every frame is matched to these
        w, h = int(native_wh[0]), int(native_wh[1])
        self.pp = ((w - 1) / 2.0, (h - 1) / 2.0)     # pixel-centre convention (fit start)
        self.seeds = seed_grid(anchor_small.shape, cfg.grid_side, margin=int(cfg.lk_win) // 2)
        self.seeds_small = self.seeds.astype(np.float64)
        self.seeds_native = to_native(self.seeds, self.scale)
        self.tpl = window_templates(anchor_small, self.seeds_small, cfg)
        self.half = float(self.tpl["half"])
        self.anchor_zm = zero_mean_local(anchor_small, cfg.lk_win)
        self.a_tiles, self.centres, self.inside = anchor_tiles(self.anchor_zm, self.seeds_small,
                                                               cfg)
        n = len(self.seeds)
        self.pos = self.seeds_small.copy()     # position in the last frame (process px)
        self.W = seed_warp(self.seeds_small, self.half)   # last refined warp of every window
        self.alive = self.tpl["usable"].copy()  # its prediction is still in view
        self.valid = self.alive.copy()          # matched to the anchor at the last frame
        self.dynamic = np.zeros(n, dtype=bool)
        self.fb = np.full(n, np.nan)           # the last frame's consistency forward-backward
        self.err = np.full(n, np.nan)          # the last frame's refinement residual (grey)
        self.reference: Optional[np.ndarray] = None   # the reference plane's tracks (seed mask)
        self.last_rigid: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.tw_prev = self.anchor_zm          # the warp twin's own previous frame
        self.tw_pos = self.seeds_small.copy()  # the twin's tracks (process px)
        self.tw_valid = self.alive.copy()
        self.prev = self.anchor_zm             # the chain link runs on zero-mean frames
        self.n_links = 0

    def trusted(self) -> np.ndarray:
        """The tracks the last frame was measured on: matched to the anchor and
        not marked dynamic."""
        return self.valid & ~self.dynamic

    def positions_native(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(anchor seeds, positions at the last frame) of ``mask``, native px."""
        return self.seeds_native[mask], to_native(self.pos[mask], self.scale)

    def _in_view(self, p: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        h, w = shape_hw
        return (np.all(np.isfinite(p), axis=1) & (p[:, 0] >= 0.0) & (p[:, 0] <= w - 1.0)
                & (p[:, 1] >= 0.0) & (p[:, 1] <= h - 1.0))

    def _track(self, cur: np.ndarray) -> None:
        """Steps (1)–(3) of the class docstring; updates the track state."""
        cfg = self.cfg
        cur_zm = zero_mean_local(cur, cfg.lk_win)
        link = np.flatnonzero(self.alive & self.valid)
        _pa, pb, ok, _fb = track_step(self.prev, cur_zm, self.pos[link], cfg, self.scale)
        # every window follows the frame-to-frame homography of the linked tracks ...
        G = None
        if int(ok.sum()) >= 4:
            G, _m = fit_homography(to_native(self.pos[link[ok]], self.scale),
                                   to_native(pb[ok], self.scale), cfg)
        if G is not None and np.all(np.isfinite(G)):
            W0 = np.einsum("ij,njk->nik", homography_to_process(G, self.scale), self.W)
        else:
            W0 = self.W.copy()
        carried = np.zeros(len(self.pos), dtype=bool)
        carried[link[ok]] = True
        # ... a window the link did not carry takes what its carried neighbours' own
        # warps say about it (its own surface) where it has any ...
        nb_W, nb_n = neighbour_warps(self.seeds_small, W0, carried, int(cfg.grid_side),
                                     self.half)
        use = ~carried & (nb_n > 0)
        W0[use] = nb_W[use]
        # ... and a carried one is centred on its own chain prediction
        W0[link[ok]] = move_centre(W0[link[ok]], pb[ok])
        fin = np.all(np.isfinite(W0.reshape(-1, 9)), axis=1)
        W0 = np.where(fin[:, None, None], W0, self.W)
        pred = warp_centre(W0)
        live = np.flatnonzero(self.alive)
        gone = ~self._in_view(pred[live], cur.shape)
        img = np.asarray(cur, dtype=np.float64)
        W, conv, rms, nrms, rwin = refine_windows(img, self.tpl, live, W0[live], cfg)
        t = warp_centre(W)
        tiles = rectified_tiles(cur_zm, self.seeds_small[live], self.centres[live], W, self.half,
                                self.a_tiles[live], cfg)
        _d, lok, lfb, _le = tile_match(self.a_tiles[live], tiles, self.inside[live], cfg,
                                       self.scale)
        v = conv & lok & ~gone & self._in_view(t, cur.shape)
        self.alive[live[gone]] = False
        self.valid[:] = False
        self.valid[live[v]] = True
        self.pos = np.where(np.all(np.isfinite(pred), axis=1)[:, None], pred, self.pos)
        self.pos[live[v]] = t[v]
        self.W = W0
        self.W[live[v]] = W[v]
        self.fb[:] = np.nan
        self.err[:] = np.nan
        self.fb[live[v]] = lfb[v]
        self.err[live[v]] = rms[v]
        self.nerr = np.full(len(self.pos), np.nan)
        self.nerr[live[v]] = nrms[v]
        self.res_win = np.full((len(self.pos), rwin.shape[1]), np.nan)
        self.res_win[live[v]] = rwin[v]
        self.prev = cur_zm
        self.n_links += 1

    def _twin(self, H: np.ndarray, frame: int) -> Tuple[float, int, Optional[str], np.ndarray]:
        """The warp twin of frame ``frame`` (class docstring): a zero-parallax
        video that follows the real one — frame t of it is the anchor warped
        by ``H`` onto frame t's pixel grid — tracked EXACTLY like the real
        frames (its own chain link from its previous frame gives the
        prediction, the direct affine refinement against the anchor's windows
        the position), with the frame's OWN measured disagreement as its
        noise: every twin window carries the residual window of another real
        track of the reference plane at convergence (a residual bootstrap,
        fixed seed per anchor and frame). Sensor noise, compression and the
        downscale leave spatially correlated residuals that no synthetic
        Gaussian reproduces — at the measured level, pixel noise read 0.7× the
        real error on a well-textured wall and 0.3× on a low-contrast one; the
        real residuals carry whatever the camera did. Returns the reading —
        the ``parallax_quantile`` of the transfer error w.r.t. the
        least-squares homography of its tracks over the windows the real
        frame was measured on: the tracker's error for THIS warp of THIS
        texture at THIS frame's noise — the number of its tracks, the reason
        when it could not be read, and every window's residual at convergence
        (grey levels, NaN where not refined)."""
        cfg = self.cfg
        n = len(self.seeds)
        resid = np.full(n, np.nan)
        self.twin_nerr = np.full(n, np.nan)
        self.twin_pos_err = np.full(n, np.nan)
        Ht = homography_to_process(H, self.scale)
        truth = _apply_h(Ht, self.seeds_small)
        live = np.flatnonzero(self.alive & self._in_view(truth, self.anchor_small.shape))
        if len(live) < cfg.min_tracks:
            return float("nan"), int(len(live)), "twin_too_few_tracks", resid
        pool = np.flatnonzero(self.valid & (self.reference if self.reference is not None
                                            else self.valid))
        pool = pool[np.all(np.isfinite(self.res_win[pool]), axis=1)]
        if len(pool) == 0:
            return float("nan"), 0, "twin_no_residuals", resid
        rng = np.random.default_rng([self.anchor, int(frame)])
        noise = self.res_win[pool[rng.integers(0, len(pool), size=len(live))]]
        clean = twin_image(self.anchor_small, Ht).astype(np.float64)
        img = zero_mean_local(np.clip(np.rint(clean), 0, 255).astype(np.uint8), cfg.lk_win)
        # the twin's chain link, from its previous frame
        link = live[self.tw_valid[live]]
        _pa, pb, ok, _fb = track_step(self.tw_prev, img, self.tw_pos[link], cfg, self.scale)
        pred = truth.copy()                    # a window the twin link did not carry starts
        pred[link[ok]] = pb[ok]                # at its truth
        W0 = move_centre(np.einsum("ij,njk->nik", Ht, seed_warp(self.seeds_small[live],
                                                                  self.half)), pred[live])
        W, conv, rms, nrms, _rw = refine_windows(clean, self.tpl, live, W0, cfg, noise=noise)
        t = warp_centre(W)
        ok_t = conv & self._in_view(t, self.anchor_small.shape)
        self.tw_valid[:] = False
        self.tw_valid[live[ok_t]] = True
        self.tw_pos[live] = np.where(ok_t[:, None], t, pred[live])
        self.tw_prev = img
        use = ok_t & self.valid[live]
        resid[live[use]] = rms[use]
        self.twin_nerr[live[use]] = nrms[use]
        self.twin_pos_err[live[use]] = np.hypot(*(to_native(t[use], self.scale)
                                                 - to_native(truth[live[use]], self.scale)).T)
        a = self.seeds_native[live[use]]
        b = to_native(t[use], self.scale)
        if len(a) < cfg.min_tracks:
            return float("nan"), int(len(a)), "twin_too_few_tracks", resid
        ref, sym = least_squares_reference(a, b)
        if ref is None:
            return float("nan"), int(len(a)), "twin_homography_failed", resid
        if sym.size == 0:
            return float("nan"), int(len(a)), "twin_residuals_undefined", resid
        return float(np.quantile(sym, cfg.parallax_quantile)), int(len(a)), None, resid

    def step(self, frame: int, small: np.ndarray) -> FrameMeasure:
        cfg = self.cfg
        if small.shape != self.anchor_small.shape:
            raise ParallaxError(f"frame {frame} downscaled to {small.shape}, its anchor "
                                f"{self.anchor} to {self.anchor_small.shape} — the session "
                                f"mixes resolutions")
        self._track(photometric_match(small, self.anchor_stats))
        self.last_rigid = None
        self.dynamic &= self.alive
        t_idx = np.flatnonzero(self.trusted())
        a = self.seeds_native[t_idx]
        b = to_native(self.pos[t_idx], self.scale)
        n_valid = int(self.valid.sum())
        base = dict(frame=int(frame), anchor=self.anchor, n_links=self.n_links,
                    n_seeds=len(self.seeds), n_surviving=n_valid)

        def lost(m):
            ev = dict(m.get("rigidity", {}), n_trusted=int(len(t_idx)),
                      n_in_view=int(self.alive.sum()),
                      n_dynamic_marked=int(self.dynamic.sum()))
            return FrameMeasure(lost=True, reason=m["reason"], disp_px=m["disp_px"],
                                n_h_inliers=m.get("n_h_inliers", 0),
                                n_f_inliers=m.get("n_f_inliers", 0),
                                n_rigid=m.get("n_rigid", 0), verdict=m.get("verdict"),
                                rigidity=_finite_or_none(ev), **base)

        fit = fit_two_view(a, b, cfg, follow=self.reference[t_idx] if self.reference is not None
                           else None)
        if fit["lost"]:
            return lost(fit)
        H, H_inv = fit["H"], fit["H_inv"]
        self.reference = np.zeros(len(self.seeds), dtype=bool)
        self.reference[t_idx[fit["in_h"]]] = True
        # the measured noise reference of this frame: the consistency match's
        # forward-backward error and the warp twin (its noise: this frame's residuals)
        q = cfg.parallax_quantile
        fb_px = float(np.quantile(self.fb[self.valid], q)) / math.sqrt(2.0)
        e_ref = self.err[t_idx[fit["in_h"]]]
        e_ref = e_ref[np.isfinite(e_ref)]
        # evidence: the refinement's residual RMS at convergence on the reference plane
        sigma = float(np.median(e_ref)) if e_ref.size else float("nan")
        twin_px, twin_n, twin_reason, _twin_resid = self._twin(H, frame)
        # APPEARANCE: a match whose normalised residual leaves the photometric noise
        # floor the twin measured over the frame's windows — above rotation_floor_factor
        # × the twin's parallax_quantile — is not the anchor's patch (an occlusion edge,
        # a texture aliasing as the view moves, a reflection): it may pass
        # forward-backward and converge, but it is not the same surface point
        v_idx = np.flatnonzero(self.valid)
        tn = self.twin_nerr[np.isfinite(self.twin_nerr)]
        appear_bar = (cfg.rotation_floor_factor * float(np.quantile(tn, q))
                      if tn.size else float("inf"))
        judged = np.isfinite(self.nerr[v_idx])
        off = judged & (self.nerr[v_idx] > appear_bar)
        n_appearance = int(off.sum())
        if n_appearance:
            self.valid[v_idx[off]] = False
            n_valid = int(self.valid.sum())
            base["n_surviving"] = n_valid
            t_idx = np.flatnonzero(self.trusted())
            a = self.seeds_native[t_idx]
            b = to_native(self.pos[t_idx], self.scale)
            fit = fit_two_view(a, b, cfg, follow=self.reference[t_idx])
            if fit["lost"]:
                return lost(dict(fit, rigidity={"n_appearance_rejected": n_appearance}))
            H, H_inv = fit["H"], fit["H_inv"]
            self.reference = np.zeros(len(self.seeds), dtype=bool)
            self.reference[t_idx[fit["in_h"]]] = True
        if math.isfinite(twin_px):
            floor, source = float(math.hypot(twin_px, fb_px)), "twin+fb"
        else:
            floor, source = fb_px, "fb"
        m = read_parallax(a, b, fit, cfg, pp=self.pp, floor_px=floor,
                          seed=(self.anchor, int(frame)))
        if m["lost"]:
            return lost(m)
        # the rigid tracks the reading was taken on (native px), kept for the
        # exclusion audit of the keyframe this frame may become
        self.last_rigid = (a[m["rigid"]].copy(), b[m["rigid"]].copy())
        # every matched track — the trusted ones and those marked dynamic — read
        # against the same homography, for the evidence
        all_idx = np.flatnonzero(self.valid)
        a_all = self.seeds_native[all_idx]
        b_all = to_native(self.pos[all_idx], self.scale)
        sym_all = symmetric_transfer_error(H, H_inv, a_all, b_all)
        rigid_all = np.zeros(len(all_idx), dtype=bool)
        rigid_all[np.searchsorted(all_idx, t_idx[m["rigid"]])] = True
        fin = np.isfinite(sym_all)
        parallax_all = float(np.quantile(sym_all[fin], q)) if fin.any() else float("nan")
        disp_all = float(np.median(np.hypot(*(b_all - a_all).T)))
        # deviation of every track from the static model the verdict chose: the
        # epipolar geometry (epipolar verdict) or the majority's homography
        if m["verdict"] == "epipolar" and fit.get("F") is not None:
            dev = epipolar_distance(fit["F"], a_all, b_all)
        else:
            dev = sym_all
        moving_bar = max(cfg.witness_min_parallax_px, cfg.rotation_floor_factor * floor)
        moving = (~rigid_all) & np.isfinite(dev) & (dev > moving_bar)
        # under a verdict that says the camera did not translate there is no parallax to
        # occlude anything: a window the appearance test rejected this frame is covered
        # by something that moved on its own — dynamic evidence too
        occluded = n_appearance if m["verdict"] in ("still", "rotation") else 0
        n_judged_dyn = n_valid + (n_appearance if occluded else 0)
        # window-scoped dynamic marks (class docstring)
        if m["verdict"] in ("still", "rotation"):
            new = t_idx[np.isfinite(m["sym"]) & (m["sym"] > cfg.witness_min_parallax_px)]
            self.dynamic[new] = True
            n_new, released = int(len(new)), 0
        else:
            n_new, released = 0, int(self.dynamic.sum())
            self.dynamic[:] = False
        ev = dict(m["rigidity"], n_trusted=int(len(t_idx)), n_in_view=int(self.alive.sum()),
                  n_appearance_rejected=n_appearance, n_judged=int(judged.sum()),
                  n_moving=int(moving.sum()), n_occluded_as_dynamic=int(occluded),
                  appearance_bar=appear_bar,
                  n_marked_now=n_new, n_dynamic_marked=int(self.dynamic.sum()),
                  n_released=released, reference=fit["reference"],
                  photometric_sigma=sigma)
        return FrameMeasure(
            lost=False, n_h_inliers=m["n_h_inliers"], n_f_inliers=m["n_f_inliers"],
            n_rigid=m["n_rigid"], verdict=m["verdict"],
            parallax_px=m["parallax_px"], parallax_all_px=parallax_all,
            residual_median_px=m["residual_median_px"], disp_px=disp_all,
            dynamic_share=1.0 - m["n_rigid"] / float(n_valid),
            dynamic_moving_share=float(moving.sum() + occluded) / float(n_judged_dyn),
            twin_px=twin_px, twin_n_tracks=twin_n, twin_reason=twin_reason, fb_px=fb_px,
            floor_px=floor, floor_source=source, rigidity=_finite_or_none(ev), **base)


# ── the quality table ────────────────────────────────────────────────────

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


def _chain_frames(rows: List[Dict[str, Any]]) -> List[int]:
    return [int(r["frame"]) for r in rows if r["usable"]]


def _runs(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    """[start, end) index ranges of consecutive True values."""
    out: List[Tuple[int, int]] = []
    i, n = 0, len(flags)
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


def _median_of(values: Sequence[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.median(v)) if v else None


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# ── flags and warnings (pure over the frame records) ─────────────────────

def frame_flags(rec: Dict[str, Any], cfg: ParallaxConfig) -> Dict[str, bool]:
    """The warning flags of one frame record (a :func:`measure_dict` plus the
    window fields :func:`run_parallax` adds). ``tracking_lost``: the frame
    was lost, or it was measured from an anchor whose tracks were lost
    before ``warn_min_run_frames`` frames — the anchor could not hold (a
    texture the tracker keeps losing re-anchors every frame or two, each
    measurement then succeeds and nothing else would say it)."""
    short = (rec.get("window_closed_by") in LOSS_CLOSERS
             and rec.get("window_n_frames") is not None
             and rec["window_n_frames"] < cfg.warn_min_run_frames)
    if rec["lost"]:
        return {"static": False, "pure_rotation": False, "dynamic_content": False,
                "tracking_lost": True}
    disp, par, floor = rec["disp_px"], rec["parallax_px"], rec["floor_px"]
    # still: the image did not move beyond warn_static_disp_px — or beyond its own
    # measured noise floor, where the tracker's error is larger than that bar (a
    # low-contrast wall seen by a still camera displaces its tracks by their noise)
    still_bar = max(cfg.warn_static_disp_px,
                    cfg.rotation_floor_factor * floor if floor is not None else 0.0)
    still = disp is not None and disp < still_bar
    rot = (disp is not None and par is not None and floor is not None
           and disp >= cfg.warn_rotation_min_disp_px
           and par <= cfg.rotation_floor_factor * floor)
    # enough tracks moving on their own (or, under a no-translation verdict, windows
    # covered by something that moved) to have moved the q-quantile had they counted
    dyn_share = rec.get("dynamic_moving_share")
    dyn = dyn_share is not None and dyn_share > 1.0 - cfg.parallax_quantile
    return {"static": bool(still), "pure_rotation": bool(rot), "dynamic_content": bool(dyn),
            "tracking_lost": bool(short)}


def coverage_warnings(records: Sequence[Dict[str, Any]], rows: List[Dict[str, Any]],
                      cfg: ParallaxConfig) -> List[Dict[str, Any]]:
    """Runs of ≥ ``warn_min_run_frames`` consecutive measured frames (chain
    order) flagged static / pure_rotation / dynamic_content / tracking_lost,
    and of consecutive exposure-rejected frames (exposure). Advisory: each
    names the frames it covers and what was measured there. Pure over the
    records, so the bars can be re-judged without re-tracking."""
    warnings: List[Dict[str, Any]] = []
    flags = [frame_flags(r, cfg) for r in records]
    for kind in ("static", "pure_rotation", "dynamic_content", "tracking_lost"):
        for i, j in _runs([f[kind] for f in flags]):
            if j - i < cfg.warn_min_run_frames:
                continue
            run = list(records[i:j])
            disp = _median_of([r["disp_px"] for r in run])
            par = _median_of([r["parallax_px"] for r in run])
            floor = _median_of([r["floor_px"] for r in run])
            if kind == "static":
                detail = (f"{j - i} consecutive frame(s) whose median displacement from their "
                          f"anchor is {_fmt(disp)} px < max(warn_static_disp_px "
                          f"{cfg.warn_static_disp_px:g}, rotation_floor_factor "
                          f"{cfg.rotation_floor_factor:g} × floor, median floor {_fmt(floor)} px): "
                          f"the camera did not move")
            elif kind == "pure_rotation":
                verdicts: Dict[str, int] = {}
                for r in run:
                    verdicts[str(r["verdict"])] = verdicts.get(str(r["verdict"]), 0) + 1
                detail = (f"{j - i} consecutive frame(s) displaced {_fmt(disp)} px (median) from "
                          f"their anchor, ≥ warn_rotation_min_disp_px "
                          f"{cfg.warn_rotation_min_disp_px:g}, whose parallax sits at its own "
                          f"noise floor (median parallax {_fmt(par)} px ≤ rotation_floor_factor "
                          f"{cfg.rotation_floor_factor:g} × floor, median floor {_fmt(floor)} px; "
                          f"rigidity verdicts "
                          f"{', '.join(f'{k}={v}' for k, v in sorted(verdicts.items()))}): the "
                          f"image moved, the geometry did not (rotation, zoom or a single plane)")
            elif kind == "dynamic_content":
                share = _median_of([r["dynamic_moving_share"] for r in run])
                pall = _median_of([r["parallax_all_px"] for r in run])
                detail = (f"{j - i} consecutive frame(s) where a median {_fmt(share)} of the "
                          f"matched tracks moved on their own beyond max(witness_min_parallax_px, "
                          f"rotation_floor_factor × floor) and were left out of the rigid set, "
                          f"or — while the camera did not translate — stopped showing the "
                          f"anchor's patch (> 1 − parallax_quantile = "
                          f"{1.0 - cfg.parallax_quantile:g}); over every matched track the "
                          f"reading would have been {_fmt(pall)} px instead of {_fmt(par)} px — "
                          f"an object moving on its own (person, vehicle, hand)")
            else:
                reasons: Dict[str, int] = {}
                n_lost_here = 0
                for r in run:
                    for loss in ([r] if r["lost"] else []) + list(r.get("losses") or []):
                        key = loss.get("reason") or "unknown"
                        reasons[key] = reasons.get(key, 0) + 1
                    n_lost_here += int(bool(r["lost"]))
                n_short = sum(1 for r in run if not r["lost"])
                tracks = _median_of([float(r["n_surviving"]) for r in run])
                detail = (f"{j - i} consecutive frame(s): {n_lost_here} lost and {n_short} "
                          f"measured from an anchor whose tracks were lost within "
                          f"warn_min_run_frames {cfg.warn_min_run_frames} frames (losses: "
                          f"{', '.join(f'{k}={v}' for k, v in sorted(reasons.items())) or 'none'}); "
                          f"median matched tracks {_fmt(tracks)} vs min_tracks {cfg.min_tracks}: "
                          f"the tracker cannot hold an anchor here — no keyframe is made without a "
                          f"measured baseline, so this stretch has no keyframe of its own")
            warnings.append({"kind": kind, "frame_start": int(run[0]["frame"]),
                             "frame_end": int(run[-1]["frame"]), "n_frames": j - i,
                             "detail": detail})
    unusable = [not r["usable"] for r in rows]
    for i, j in _runs(unusable):
        if j - i < cfg.warn_min_run_frames:
            continue
        reasons = {}
        for r in rows[i:j]:
            key = r.get("reject_reason") or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
        warnings.append({"kind": "exposure", "frame_start": int(rows[i]["frame"]),
                         "frame_end": int(rows[j - 1]["frame"]), "n_frames": j - i,
                         "detail": (f"{j - i} consecutive frame(s) unusable by exposure "
                                    f"({', '.join(f'{k}={v}' for k, v in sorted(reasons.items()))}): "
                                    f"the chain bridges them when the tracks survive")})
    warnings.sort(key=lambda w: (w["frame_start"], WARNING_KINDS.index(w["kind"])))
    return warnings


# ── selection ────────────────────────────────────────────────────────────

class _Frames:
    """Loads a usable frame at the tracking scale (checked against the
    session size); nothing is cached — memory stays O(1) in the frame count,
    re-reads are the price (runtime is not a criterion)."""

    def __init__(self, by_frame: Dict[int, Path], native_wh: Tuple[int, int], cfg: ParallaxConfig):
        self.by_frame = by_frame
        self.native_w, self.native_h = native_wh
        self.cfg = cfg
        self.scale: Optional[ScaleXY] = None
        self.n_reads = 0

    def small(self, frame: int) -> np.ndarray:
        try:
            g = read_gray(self.by_frame[frame])
        except QualityError as e:
            raise ParallaxError(str(e)) from e
        if g.shape != (self.native_h, self.native_w):
            raise ParallaxError(
                f"frame {self.by_frame[frame].name} is {g.shape[1]}x{g.shape[0]} but the "
                f"quality report says the session is {self.native_w}x{self.native_h}")
        small, scale = downscale(g, self.cfg.process_scale)
        if self.scale is None:
            self.scale = scale
        elif scale != self.scale:
            raise ParallaxError(f"frame {frame} downscaled with scale {scale}, the session "
                                f"with {self.scale} — the session mixes resolutions")
        self.n_reads += 1
        return small


class _Progress:
    def __init__(self, log: Callable, what: str, total: int, heartbeat_s: float,
                 cancelled: Cancelled):
        self.log, self.what, self.total = log, what, max(total, 1)
        self.heartbeat_s, self.cancelled = heartbeat_s, cancelled
        self.t0 = self.last = time.monotonic()
        self.n = 0

    def tick(self, position: int) -> None:
        check_cancelled(self.cancelled, f"intake I1 (parallax, {self.what}) at frame "
                                        f"{position}/{self.total}")
        self.n += 1
        now = time.monotonic()
        if now - self.last >= self.heartbeat_s:
            rate = self.n / max(now - self.t0, 1e-9)
            self.log(f"{LOG_TAG} {self.what}: at frame {position}/{self.total} "
                     f"({rate:.1f} frames/s measured, re-tracks included)")
            self.last = now


def _choose(cands: List[FrameMeasure], by_frame: Dict[int, Dict[str, Any]]) -> FrameMeasure:
    """Highest sharp_rank (I0); ties → the earliest frame."""
    return max(cands, key=lambda m: (float(by_frame[m.frame]["sharp_rank"]), -m.frame))


def baseline_bar(m: FrameMeasure, cfg: ParallaxConfig) -> float:
    """The smallest reading that is a MEASURED baseline: at least
    ``witness_min_parallax_px`` (the smallest parallax the intake acts on)
    and above ``rotation_floor_factor`` × the frame's own measured floor (a
    reading at its floor is the tracker's error, whatever the view did)."""
    floor = m.floor_px if math.isfinite(m.floor_px) else 0.0
    return max(float(cfg.witness_min_parallax_px), float(cfg.rotation_floor_factor) * floor)


def is_view_change(m: FrameMeasure, cfg: ParallaxConfig) -> bool:
    """A track-loss keyframe marks a VIEW CHANGE with a measured baseline: the
    image moved (median displacement from the anchor ≥
    ``warn_rotation_min_disp_px``, the bar at which the warnings call a view
    displaced) AND the reading leaves its floor (:func:`baseline_bar`). Tracks
    that die while the view stays put — a texture the tracker cannot hold,
    noise — are not a coverage event, whatever they read."""
    return (math.isfinite(m.disp_px) and m.disp_px >= cfg.warn_rotation_min_disp_px
            and math.isfinite(m.parallax_px) and m.parallax_px >= baseline_bar(m, cfg))


def _loss_record(m: FrameMeasure) -> Dict[str, Any]:
    return {"anchor": int(m.anchor), "reason": m.reason, "n_surviving": int(m.n_surviving),
            "n_rigid": int(m.n_rigid)}


def select_keyframes(chain: List[int], frames: _Frames, by_frame: Dict[int, Dict[str, Any]],
                     cfg: ParallaxConfig, progress: _Progress,
                     cache: Optional[Dict[Tuple[int, int], FrameMeasure]] = None
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                Dict[int, Tuple[FrameMeasure, int]], Dict[int, List[Dict]],
                                Dict[int, Dict[str, Any]]]:
    """The keyframe pass (module docstring, KEYFRAMES). Returns (keyframes,
    windows, records, losses, keyframe_tracks) — records: the last
    measurement of every frame after the first with the index of the window
    it belongs to; losses: every measurement that was LOST, per frame (a
    frame lost from one anchor and measured again from the next keeps its
    loss on record); keyframe_tracks: for every keyframe after the first, the
    rigid tracks its reading was taken on (anchor and keyframe positions,
    native px) — what the exclusion audit re-reads under I2's masks."""
    quantum = float(cfg.parallax_quantum_px)
    lo = (1.0 - float(cfg.keyframe_band_frac)) * quantum
    hi = (1.0 + float(cfg.keyframe_band_frac)) * quantum
    records: Dict[int, Tuple[FrameMeasure, int]] = {}
    losses: Dict[int, List[Dict[str, Any]]] = {}
    kf_tracks: Dict[int, Dict[str, Any]] = {}
    keyframes: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    pos = {f: i for i, f in enumerate(chain)}

    def kf_record(frame: int, reason: str, m: Optional[FrameMeasure], window: Optional[Dict],
                  anchor_is_kf: Optional[bool]):
        par = None if m is None else _finite_or_none(m.parallax_px)
        keyframes.append({
            "frame": int(frame), "file": by_frame[frame]["file"],
            "sharp_rank": float(by_frame[frame]["sharp_rank"]), "reason": reason,
            "anchor": None if m is None else int(m.anchor),
            "anchor_is_keyframe": anchor_is_kf,
            "parallax_from_anchor_px": par,
            "parallax_over_quantum": None if par is None else float(par) / quantum,
            "disp_from_anchor_px": None if m is None else _finite_or_none(m.disp_px),
            "floor_px": None if m is None else _finite_or_none(m.floor_px),
            "window": window})

    kf_record(chain[0], "first_usable_frame", None, None, None)
    i = 0                       # position of the anchor in the chain
    is_kf = True                # the anchor is a keyframe (False: a pending anchor after a break)
    hard_loss_at: Optional[int] = None   # the frame a track_loss close was made for
    while i < len(chain) - 1:
        anchor = chain[i]
        tracker = AnchorChain(anchor, frames.small(anchor), frames.scale, cfg,
                              native_wh=(frames.native_w, frames.native_h))
        wid = len(windows)
        window: List[FrameMeasure] = []
        win_tracks: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        reached = False         # a reading ≥ the quantum was measured
        lost_m: Optional[FrameMeasure] = None
        ended = "end"
        for j in range(i + 1, len(chain)):
            f = chain[j]
            progress.tick(j)
            m = tracker.step(f, frames.small(f))
            if cache is not None:
                cache[(int(anchor), int(f))] = m
            records[f] = (m, wid)
            if m.lost:
                losses.setdefault(f, []).append(_loss_record(m))
                lost_m, ended = m, "tracking_lost"
                break
            window.append(m)
            if tracker.last_rigid is not None:
                win_tracks[f] = tracker.last_rigid
            if m.parallax_px >= quantum:
                reached = True
            if reached and m.parallax_px >= hi:
                ended = "band_passed"
                break
        base = {"anchor": anchor, "anchor_is_keyframe": is_kf,
                "frame_start": window[0].frame if window else (lost_m.frame if lost_m else None),
                "frame_end": (lost_m.frame if lost_m is not None else
                              (window[-1].frame if window else None)),
                "n_frames": len(window),
                "max_parallax_px": (_finite_or_none(max(w.parallax_px for w in window))
                                    if window else None)}
        if reached:
            cands = [w for w in window if lo <= w.parallax_px <= hi]
            if cands:
                k, choice = _choose(cands, by_frame), "sharpest_in_band"
            else:
                k = min(window, key=lambda w: (abs(w.parallax_px - quantum), w.frame))
                choice = "closest_to_quantum"
            win = dict(base, n_candidates=len(cands), closed_by="quantum", keyframe=k.frame,
                       band_px=[lo, hi], choice=choice, ended=ended, reason=None)
            windows.append(win)
            kf_record(k.frame, "quantum", k, win, is_kf)
            if k.frame in win_tracks:
                kf_tracks[k.frame] = {"anchor": int(anchor), "a": win_tracks[k.frame][0],
                                      "b": win_tracks[k.frame][1]}
            hard_loss_at = None
            i, is_kf = pos[k.frame], True
            continue
        if lost_m is None:
            # the sequence ended before the quantum
            if window:
                last = window[-1]
                windows.append(dict(
                    base, n_candidates=0, closed_by="end", keyframe=None,
                    reason=(f"the sequence ended with the largest reading from the anchor at "
                            f"{max(w.parallax_px for w in window):.2f} px < parallax_quantum_px "
                            f"{quantum:g} (last frame {last.frame}: {last.parallax_px:.2f} px on "
                            f"{last.n_rigid}/{last.n_surviving} rigid/matched tracks, floor "
                            f"{last.floor_px:.2f} px): no keyframe")))
            break
        f = lost_m.frame
        # a loss before the quantum: a keyframe only where a baseline was MEASURED
        measured = [w for w in window if is_view_change(w, cfg)]
        if measured and hard_loss_at != f:
            pmax = max(w.parallax_px for w in measured)
            cands = [w for w in measured if w.parallax_px >= (1.0 - cfg.keyframe_band_frac) * pmax]
            k = _choose(cands, by_frame)
            win = dict(base, n_candidates=len(cands), closed_by="track_loss", keyframe=k.frame,
                       reason=(f"tracking lost at frame {f} ({lost_m.reason}: "
                               f"{lost_m.n_surviving} matched, {lost_m.n_rigid} rigid < min_tracks "
                               f"{cfg.min_tracks}) before the quantum, after a view change with a "
                               f"measured baseline (displacement ≥ warn_rotation_min_disp_px "
                               f"{cfg.warn_rotation_min_disp_px:g} and a reading ≥ "
                               f"max(witness_min_parallax_px, rotation_floor_factor × floor); "
                               f"largest {pmax:.2f} px); the window closes on the sharpest frame "
                               f"within the band of that reading"))
            windows.append(win)
            kf_record(k.frame, "track_loss", k, win, is_kf)
            if k.frame in win_tracks:
                kf_tracks[k.frame] = {"anchor": int(anchor), "a": win_tracks[k.frame][0],
                                      "b": win_tracks[k.frame][1]}
            hard_loss_at = f
            i, is_kf = pos[k.frame], True
            continue
        hard_loss_at = None
        if window:
            # the view left the anchor's tracks with no baseline measured (a pan past
            # the field of view, a texture the tracks cannot hold): no keyframe — the
            # measurement continues from the last measured frame, a PENDING anchor
            nxt = window[-1].frame
            best = max(window, key=lambda w: (w.parallax_px, -w.frame))
            windows.append(dict(
                base, n_candidates=0, closed_by="coverage_break", keyframe=None, next_anchor=nxt,
                reason=(f"tracking lost at frame {f} ({lost_m.reason}: {lost_m.n_surviving} "
                        f"matched, {lost_m.n_rigid} rigid < min_tracks {cfg.min_tracks}) with no "
                        f"view change carrying a measured baseline (largest reading "
                        f"{best.parallax_px:.2f} px at frame {best.frame}, displaced "
                        f"{best.disp_px:.2f} px; a view change needs displacement ≥ "
                        f"warn_rotation_min_disp_px {cfg.warn_rotation_min_disp_px:g} and a "
                        f"reading ≥ max(witness_min_parallax_px {cfg.witness_min_parallax_px:g}, "
                        f"rotation_floor_factor {cfg.rotation_floor_factor:g} × floor "
                        f"{best.floor_px:.2f} px)): no keyframe; the measurement continues from "
                        f"frame {nxt}, a pending anchor")))
            i, is_kf = pos[nxt], False
        else:
            # nothing measured from this anchor: a hard break — the lost frame itself
            # becomes the pending anchor
            windows.append(dict(
                base, n_candidates=0, closed_by="tracking_lost", keyframe=None, next_anchor=f,
                reason=(f"tracking lost at frame {f} ({lost_m.reason}) on the first link from "
                        f"the anchor: a hard break; the measurement continues from frame {f}, "
                        f"a pending anchor, and no keyframe is made without a measured baseline")))
            i, is_kf = pos[f], False
    keyframes.sort(key=lambda k: k["frame"])
    return keyframes, windows, records, losses, kf_tracks


def select_witnesses(chain: List[int], kf_frames: Sequence[int], frames: _Frames,
                     by_frame: Dict[int, Dict[str, Any]], cfg: ParallaxConfig,
                     progress: _Progress,
                     cache: Optional[Dict[Tuple[int, int], FrameMeasure]] = None
                     ) -> List[Dict[str, Any]]:
    """The witness pass (module docstring, WITNESSES). ``cache`` holds the
    keyframe pass's measurements by (anchor, frame): a window of this pass
    that starts on the same anchor reads them instead of tracking again —
    the tracker is deterministic, so the reading is the same — until the
    first frame the keyframe pass did not measure from that anchor, from
    which it tracks (replaying the window from its anchor)."""
    wmin = float(cfg.witness_min_parallax_px)
    kf_set = set(int(k) for k in kf_frames)
    pos = {f: i for i, f in enumerate(chain)}
    witnesses: List[Dict[str, Any]] = []

    def add(frame: int, reason: str, m: Optional[FrameMeasure]):
        witnesses.append({"frame": int(frame), "file": by_frame[frame]["file"],
                          "is_keyframe": frame in kf_set, "reason": reason,
                          "anchor": None if m is None else int(m.anchor),
                          "parallax_from_anchor_px": None if m is None else _finite_or_none(m.parallax_px),
                          "verdict": None if m is None else m.verdict})

    add(chain[0], "first_usable_frame", None)
    i = 0
    while i < len(chain) - 1:
        anchor = chain[i]
        tracker: Optional[AnchorChain] = None
        window: List[FrameMeasure] = []
        advanced = False
        for j in range(i + 1, len(chain)):
            f = chain[j]
            progress.tick(j)
            if f in kf_set:
                # keyframes are witnesses and reset the anchor
                add(f, "keyframe", None)
                i = j
                advanced = True
                break
            hit = None if (cache is None or tracker is not None) else cache.get((int(anchor), int(f)))
            if hit is not None:
                m = hit
            else:
                if tracker is None:
                    tracker = AnchorChain(anchor, frames.small(anchor), frames.scale, cfg,
                                          native_wh=(frames.native_w, frames.native_h))
                    for jj in range(i + 1, j):           # replay the cached part of the window
                        tracker.step(chain[jj], frames.small(chain[jj]))
                m = tracker.step(f, frames.small(f))
            if m.lost:
                # no witness without a measured parallax: the next anchor is pending
                i = pos[window[-1].frame] if window else j
                advanced = True
                break
            window.append(m)
            if m.parallax_px >= wmin:
                add(f, "dedup", m)
                i = j
                advanced = True
                break
        if not advanced:
            break
    witnesses.sort(key=lambda w: w["frame"])
    return witnesses


# ── the stage ────────────────────────────────────────────────────────────

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
                 log: Callable = print, *, heartbeat_s: float, geometry_epoch: int = 0,
                 camera_epoch: int = 0, cancelled: Cancelled = None) -> Dict[str, Any]:
    """Select keyframes and witnesses over the usable frames of ``frames_dir``
    and measure the coverage warnings. ``quality`` is the I0 report of the
    SAME frame inventory (checked). ``heartbeat_s`` (``intake.runtime.
    heartbeat_s``, passed by the caller) paces the progress lines;
    ``cancelled()`` is polled once per measured frame; ``geometry_epoch`` /
    ``camera_epoch`` are the session's stamps (``intake.quality.
    read_session_epochs``; 0 = none).

    Returns ``keyframes``, ``witnesses``, ``warnings``, ``windows``, ``frames``
    (every usable frame after the first as last measured by the keyframe
    chain, with its flags), counts, ``floor`` (a summary of the measured
    noise reference), plus ``version``, ``provenance``, the two epochs,
    ``method``, ``native_w`` / ``native_h``, ``params`` (the effective
    ParallaxConfig) and ``inputs`` (the frame inventory identity)."""
    if heartbeat_s <= 0:
        raise ParallaxError(f"heartbeat_s must be positive, got {heartbeat_s}")
    frames_dir = Path(frames_dir)
    try:
        paths = list_frames(frames_dir)
    except QualityError as e:
        raise ParallaxError(str(e)) from e
    rows = _frame_table(quality)
    _check_inventory(paths, rows, frames_dir)
    by_row = {int(r["frame"]): r for r in rows}
    chain = _chain_frames(rows)
    if not chain:
        rejected = quality.get("rejected", {})
        raise ParallaxError(
            f"no usable frame in {frames_dir}: every one of the {len(rows)} frame(s) was "
            f"rejected by exposure ({', '.join(f'{k}={v}' for k, v in rejected.items())})")
    native_w, native_h = int(quality["native_w"]), int(quality["native_h"])
    log(f"{LOG_TAG} {len(chain)}/{len(rows)} usable frame(s); LK grid {cfg.grid_side}x"
        f"{cfg.grid_side} at scale {cfg.process_scale:g}, chained from each anchor; "
        f"homography / fundamental bound {cfg.ransac_px:g} px; parallax = "
        f"{cfg.parallax_quantile:g}-quantile of the symmetric transfer error over the rigid "
        f"tracks (rigidity tests at confidence {cfg.rigidity_confidence:g}, "
        f"{cfg.rigidity_bootstrap} bootstrap resamples); quantum {cfg.parallax_quantum_px:g} "
        f"px (band {cfg.keyframe_band_frac:g}), witness {cfg.witness_min_parallax_px:g} px "
        f"(native px throughout)")
    frames = _Frames({int(p.stem): p for p in paths}, (native_w, native_h), cfg)
    t0 = time.monotonic()
    prog = _Progress(log, "keyframes", len(chain), heartbeat_s, cancelled)
    cache: Dict[Tuple[int, int], FrameMeasure] = {}
    keyframes, windows, records, losses, kf_tracks = select_keyframes(chain, frames, by_row,
                                                                      cfg, prog, cache=cache)
    kf_frames = [k["frame"] for k in keyframes]
    log(f"{LOG_TAG} keyframes: {len(keyframes)} in {time.monotonic() - t0:.1f} s "
        f"({prog.n} frame measurements, {prog.n / max(time.monotonic() - t0, 1e-9):.1f}/s)")
    t1 = time.monotonic()
    prog_w = _Progress(log, "witnesses", len(chain), heartbeat_s, cancelled)
    witnesses = select_witnesses(chain, kf_frames, frames, by_row, cfg, prog_w, cache=cache)
    cache.clear()
    log(f"{LOG_TAG} witnesses: {len(witnesses)} in {time.monotonic() - t1:.1f} s "
        f"({prog_w.n} frame measurements, {prog_w.n / max(time.monotonic() - t1, 1e-9):.1f}/s)")

    frame_recs = []
    for f in chain[1:]:
        if f not in records:
            continue
        m, wid = records[f]
        d = measure_dict(m)
        w = windows[wid] if wid < len(windows) else None
        d["window_closed_by"] = None if w is None else w["closed_by"]
        d["window_n_frames"] = None if w is None else int(w["n_frames"])
        d["anchor_is_keyframe"] = None if w is None else w["anchor_is_keyframe"]
        d["losses"] = losses.get(f, [])
        d.update(frame_flags(d, cfg))
        frame_recs.append(d)
    warnings = coverage_warnings(frame_recs, rows, cfg)
    measured = [r for r in frame_recs if not r["lost"]]
    verdicts = {v: sum(1 for r in measured if r["verdict"] == v) for v in VERDICTS}
    floor = {
        "median_px": _median_of([r["floor_px"] for r in measured]),
        "min_px": (min(r["floor_px"] for r in measured) if measured else None),
        "max_px": (max(r["floor_px"] for r in measured) if measured else None),
        "median_twin_px": _median_of([r["twin_px"] for r in measured]),
        "median_fb_px": _median_of([r["fb_px"] for r in measured]),
        "n_twin_measured": sum(1 for r in measured if r["floor_source"] == "twin+fb"),
        "n_measured": len(measured),
    }
    result: Dict[str, Any] = {
        # the rigid tracks of every keyframe's reading (native px, float64 arrays —
        # written to intake/keyframe_tracks.npz, never into the JSON documents)
        "keyframe_tracks": {int(k): {"anchor": int(v["anchor"]), "a": v["a"], "b": v["b"]}
                            for k, v in sorted(kf_tracks.items())},
        "keyframes": keyframes,
        "witnesses": witnesses,
        "warnings": warnings,
        "windows": windows,
        "frames": frame_recs,
        "floor": floor,
        "verdicts": verdicts,
        "n_frames": len(rows),
        "n_usable_frames": len(chain),
        "n_measured": len(measured),
        "n_lost": sum(1 for r in frame_recs if r["lost"]),
        "n_still": sum(1 for r in frame_recs if r["static"]),
        "n_pure_rotation": sum(1 for r in frame_recs if r["pure_rotation"]),
        "n_dynamic": sum(1 for r in frame_recs if r["dynamic_content"]),
        "n_keyframes": len(keyframes),
        "n_witness": len(witnesses),
        "n_frame_reads": frames.n_reads,
    }
    log(f"{LOG_TAG} {result['n_keyframes']} keyframe(s), {result['n_witness']} witness(es), "
        f"{len(warnings)} coverage warning(s); floor median {_fmt(floor['median_px'])} px "
        f"(twin {_fmt(floor['median_twin_px'])}, fb {_fmt(floor['median_fb_px'])}); rigidity "
        + ", ".join(f"{k}={v}" for k, v in verdicts.items())
        + f"; {result['n_lost']} lost, {result['n_still']} still, {result['n_pure_rotation']} "
          f"pure rotation, {result['n_dynamic']} dynamic")
    for w in warnings:
        log(f"{LOG_TAG} ⚠ {w['kind']} frames {w['frame_start']}–{w['frame_end']}: {w['detail']}")
    result.update({
        "version": PARALLAX_VERSION,
        "provenance": PROVENANCE,
        "geometry_epoch": int(geometry_epoch),
        "camera_epoch": int(camera_epoch),
        "method": METHOD,
        "native_w": native_w,
        "native_h": native_h,
        "params": asdict(cfg),
        "inputs": {
            "frames_dir": str(frames_dir),
            "n_frames": len(paths),
            "first": paths[0].name,
            "last": paths[-1].name,
            "bytes_total": int(sum(p.stat().st_size for p in paths)),
            "quality_n_frames": int(quality.get("n_frames", len(rows))),
            "quality_n_usable": int(quality.get("n_usable", len(chain))),
        },
    })
    return result


def _stamps(result: Dict[str, Any]) -> Dict[str, Any]:
    """The four artifact stamps + the stage's effective parameters and inputs
    (conventions rules 4 and 6) — carried by every I1 document."""
    return {
        "provenance": PROVENANCE,
        "geometry_epoch": int(result["geometry_epoch"]),
        "camera_epoch": int(result["camera_epoch"]),
        "params": result["params"],
        "inputs": result["inputs"],
    }


def selected_frames_document(result: Dict[str, Any], cfg: ParallaxConfig) -> Dict[str, Any]:
    """frames/selected_frames.json: the v2 contract (``version`` "2.0",
    ``method``, ``total_frames``, ``selected_count``, ``selected_files``
    sorted by frame number) plus this stage's records. ``reduction`` is the
    legacy share the v2 loader prints."""
    files = sorted((k["file"] for k in result["keyframes"]), key=lambda f: int(Path(f).stem))
    total = int(result["n_frames"])
    return {
        "version": SELECTED_CONTRACT_VERSION,
        "method": f"{METHOD}_{cfg.parallax_quantum_px:g}",
        "total_frames": total,
        "selected_count": len(files),
        "selected_files": files,
        "reduction": (1.0 - len(files) / float(total)) if total else 0.0,
        **_stamps(result),
        "intake_version": PARALLAX_VERSION,
        "parallax_quantum_px": float(cfg.parallax_quantum_px),
        "parallax_quantile": float(cfg.parallax_quantile),
        "keyframe_band_frac": float(cfg.keyframe_band_frac),
        "keyframes": result["keyframes"],
        "windows": result["windows"],
        "n_witness": int(result["n_witness"]),
        "n_usable_frames": int(result["n_usable_frames"]),
        "n_lost": int(result["n_lost"]),
    }


def witness_frames_document(result: Dict[str, Any], cfg: ParallaxConfig) -> Dict[str, Any]:
    files = sorted((w["file"] for w in result["witnesses"]), key=lambda f: int(Path(f).stem))
    return {
        "version": PARALLAX_VERSION,
        **_stamps(result),
        "method": METHOD,
        "witness_min_parallax_px": float(cfg.witness_min_parallax_px),
        "frames": result["witnesses"],
        "total_frames": int(result["n_frames"]),
        "selected_count": len(files),
        "selected_files": files,
    }


def coverage_warnings_document(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": PARALLAX_VERSION,
        **_stamps(result),
        "method": METHOD,
        "warnings": result["warnings"],
        "warning_kinds": list(WARNING_KINDS),
        "floor": result["floor"],
        "verdicts": result["verdicts"],
        "n_measured": int(result["n_measured"]),
        "n_lost": int(result["n_lost"]),
        "n_still": int(result["n_still"]),
        "n_pure_rotation": int(result["n_pure_rotation"]),
        "n_dynamic": int(result["n_dynamic"]),
        "frames": result["frames"],
    }


def write_selection(session_dir: os.PathLike, result: Dict[str, Any], cfg: ParallaxConfig
                    ) -> Tuple[Path, Path, Path]:
    """Write ``frames/selected_frames.json``, ``frames/witness_frames.json``
    and ``<session>/intake/coverage_warnings.json`` (each tmp + os.replace).
    Returns the three paths in that order."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        raise ParallaxError(f"{frames_dir} is not a directory — the selection belongs next "
                            f"to the frames it selects")
    p_kf = _write_json_atomic(frames_dir / SELECTED_FRAMES_NAME,
                              selected_frames_document(result, cfg))
    p_w = _write_json_atomic(frames_dir / WITNESS_FRAMES_NAME,
                             witness_frames_document(result, cfg))
    write_keyframe_tracks(session_dir, result)
    p_warn = _write_json_atomic(session_dir / INTAKE_SUBDIR / COVERAGE_WARNINGS_NAME,
                                coverage_warnings_document(result))
    return p_kf, p_w, p_warn


def keyframe_tracks_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_SUBDIR / KEYFRAME_TRACKS_NAME


def write_keyframe_tracks(session_dir: os.PathLike, result: Dict[str, Any]) -> Path:
    """``<session>/intake/keyframe_tracks.npz``: for every keyframe after the
    first, the rigid tracks its reading was taken on — ``k<frame>_a`` (anchor
    positions) and ``k<frame>_b`` (keyframe positions), native px, and
    ``k<frame>_anchor`` — plus ``meta`` (JSON: the four stamps, the version,
    the parameters). Written atomically."""
    path = keyframe_tracks_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for k, v in result["keyframe_tracks"].items():
        key = f"k{int(k):06d}"
        arrays[f"{key}_a"] = np.asarray(v["a"], dtype=np.float64).reshape(-1, 2)
        arrays[f"{key}_b"] = np.asarray(v["b"], dtype=np.float64).reshape(-1, 2)
        arrays[f"{key}_anchor"] = np.asarray(int(v["anchor"]), dtype=np.int64)
    meta = {"version": PARALLAX_VERSION, **_stamps(result), "method": METHOD,
            "native_w": int(result["native_w"]), "native_h": int(result["native_h"])}
    arrays["meta"] = np.asarray(json.dumps(meta, sort_keys=True))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return path


def load_keyframe_tracks(session_dir: os.PathLike) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """(meta, {keyframe: {anchor, a, b}}) from ``keyframe_tracks.npz``;
    ParallaxError when it is absent or unreadable."""
    path = keyframe_tracks_path(session_dir)
    if not path.exists():
        raise ParallaxError(f"{path} does not exist — run intake I1 first")
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            out: Dict[int, Dict[str, Any]] = {}
            for name in z.files:
                if name.endswith("_anchor"):
                    key = name[:-len("_anchor")]
                    out[int(key[1:])] = {"anchor": int(z[name]), "a": np.array(z[f"{key}_a"]),
                                         "b": np.array(z[f"{key}_b"])}
    except (OSError, ValueError, KeyError) as e:
        raise ParallaxError(f"{path} is unreadable ({e})") from e
    return meta, out


def _read_mask(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    arr = _cv2().imread(str(path), _cv2().IMREAD_GRAYSCALE)
    if arr is None:
        raise ParallaxError(f"exclusion mask {path} is unreadable")
    if arr.shape != tuple(shape_hw):
        raise ParallaxError(f"exclusion mask {path} is {arr.shape[1]}x{arr.shape[0]}, the "
                            f"session {shape_hw[1]}x{shape_hw[0]}")
    return arr == MASK_EXCLUDED


def _in_mask(mask: Optional[np.ndarray], pts: np.ndarray) -> np.ndarray:
    if mask is None or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    h, w = mask.shape
    c = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, w - 1)
    r = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, h - 1)
    return mask[r, c]


def audit_exclusions(session_dir: os.PathLike, masks_dir: os.PathLike, cfg: ParallaxConfig,
                     log: Callable = print, *, geometry_epoch: int = 0,
                     camera_epoch: int = 0) -> Dict[str, Any]:
    """The EXCLUSION AUDIT (module docstring, DECLARED LIMITS): I1 runs
    before I2's masks exist, and an object moving along the epipolar lines of
    the camera's own motion is indistinguishable from structure in two views.
    Once I2 has segmented the keyframes, every keyframe's reading is taken
    again on its own rigid tracks (``keyframe_tracks.npz``) WITHOUT those
    whose anchor position lies in the anchor's exclusion mask or whose
    keyframe position lies in the keyframe's: the same statistic, the
    least-squares reference of what is left. A keyframe whose baseline then
    falls under the keyframe band's lower edge ((1 − ``keyframe_band_frac``)
    × quantum; for a track-loss keyframe, under max(``witness_min_parallax_px``,
    ``rotation_floor_factor`` × its floor)) closed its window on an object
    moving on its own: a coverage warning of kind ``excluded_parallax`` names
    it with both readings. Advisory — the selection is not changed. Writes
    ``<session>/intake/exclusion_audit.json`` and merges its warnings into
    ``coverage_warnings.json`` (replacing a previous audit's)."""
    session_dir = Path(session_dir)
    masks_dir = Path(masks_dir)
    meta, tracks = load_keyframe_tracks(session_dir)
    sel = load_selection(session_dir / "frames")
    p_warn = session_dir / INTAKE_SUBDIR / COVERAGE_WARNINGS_NAME
    if not p_warn.exists():
        raise ParallaxError(f"{p_warn} does not exist — run intake I1 first")
    with open(p_warn) as f:
        cov = json.load(f)
    native_hw = (int(meta["native_h"]), int(meta["native_w"]))
    kf_by_frame = {int(k["frame"]): k for k in sel.get("keyframes", [])}
    quantum = float(cfg.parallax_quantum_px)
    lo = (1.0 - float(cfg.keyframe_band_frac)) * quantum
    q = float(cfg.parallax_quantile)
    records: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    masks_used: Dict[str, int] = {}
    cache: Dict[int, Optional[np.ndarray]] = {}

    def mask_of(frame: int) -> Optional[np.ndarray]:
        if frame not in cache:
            mp = masks_dir / f"{int(frame):06d}.png"
            cache[frame] = _read_mask(mp, native_hw) if mp.exists() else None
            if cache[frame] is not None:
                masks_used[str(frame)] = int(mp.stat().st_size)
        return cache[frame]

    for frame in sorted(tracks):
        tr = tracks[frame]
        a, b, anchor = tr["a"], tr["b"], tr["anchor"]
        kf = kf_by_frame.get(frame, {})
        mA, mK = mask_of(anchor), mask_of(frame)
        excl = _in_mask(mA, a) | _in_mask(mK, b)
        rec: Dict[str, Any] = {"keyframe": int(frame), "anchor": int(anchor),
                               "reason": kf.get("reason"), "n_rigid": int(len(a)),
                               "n_excluded": int(excl.sum()),
                               "anchor_mask": mA is not None, "keyframe_mask": mK is not None}
        _H, sym = least_squares_reference(a, b) if len(a) >= 4 else (None, np.zeros(0))
        rec["reading_px"] = float(np.quantile(sym, q)) if sym.size else None
        kept = ~excl
        if not excl.any():
            rec.update(reading_without_excluded_px=rec["reading_px"], verdict="unaffected")
        elif int(kept.sum()) < cfg.min_tracks:
            rec.update(reading_without_excluded_px=None, verdict="too_few_tracks_left")
        else:
            _H2, sym2 = least_squares_reference(a[kept], b[kept])
            r2 = float(np.quantile(sym2, q)) if sym2.size else None
            if kf.get("reason") == "track_loss":
                floor = kf.get("floor_px") or 0.0
                bar = max(float(cfg.witness_min_parallax_px), cfg.rotation_floor_factor * floor)
            else:
                bar = lo
            rec.update(reading_without_excluded_px=r2, bar_px=bar,
                       verdict=("rests_on_excluded" if r2 is None or r2 < bar
                                else "baseline_kept"))
        records.append(rec)
        if rec["verdict"] in ("rests_on_excluded", "too_few_tracks_left"):
            without = rec.get("reading_without_excluded_px")
            warnings.append({
                "kind": "excluded_parallax", "frame_start": int(anchor), "frame_end": int(frame),
                "n_frames": None,
                "detail": (f"keyframe {frame} (from anchor {anchor}) read "
                           f"{_fmt(rec['reading_px'])} px on {rec['n_rigid']} rigid tracks, but "
                           f"{rec['n_excluded']} of them lie in I2's exclusion masks (anchor mask "
                           f"{'yes' if mA is not None else 'no'}, keyframe mask "
                           f"{'yes' if mK is not None else 'no'}); without them it reads "
                           f"{_fmt(without)} px"
                           + (f" < {rec['bar_px']:.2f} px" if rec.get("bar_px") is not None else
                              f" (fewer than min_tracks {cfg.min_tracks} left)")
                           + ": the window closed on an object moving on its own — the keyframe "
                             "carries less static baseline than the quantum")})
    doc = {
        "version": PARALLAX_VERSION,
        "provenance": PROVENANCE,
        "geometry_epoch": int(geometry_epoch),
        "camera_epoch": int(camera_epoch),
        "method": METHOD,
        "params": asdict(cfg),
        "inputs": {"keyframe_tracks": str(keyframe_tracks_path(session_dir)),
                   "keyframe_tracks_bytes": int(keyframe_tracks_path(session_dir).stat().st_size),
                   "masks_dir": str(masks_dir), "masks": masks_used},
        "keyframes": records,
        "warnings": warnings,
        "n_rests_on_excluded": sum(1 for r in records if r["verdict"] in
                                   ("rests_on_excluded", "too_few_tracks_left")),
    }
    p_audit = _write_json_atomic(session_dir / INTAKE_SUBDIR / EXCLUSION_AUDIT_NAME, doc)
    merged = [w for w in cov.get("warnings", []) if w.get("kind") != "excluded_parallax"]
    merged += warnings
    merged.sort(key=lambda w: (w["frame_start"], WARNING_KINDS.index(w["kind"])))
    cov["warnings"] = merged
    cov["exclusion_audit"] = {"path": str(p_audit), "n_keyframes": len(records),
                              "n_rests_on_excluded": doc["n_rests_on_excluded"]}
    _write_json_atomic(p_warn, cov)
    log(f"{LOG_TAG} exclusion audit: {len(records)} keyframe reading(s) re-read without I2's "
        f"excluded tracks, {doc['n_rests_on_excluded']} rest(s) on excluded content")
    for w in warnings:
        log(f"{LOG_TAG} ⚠ {w['kind']} frames {w['frame_start']}–{w['frame_end']}: {w['detail']}")
    return doc


def load_selection(frames_dir: os.PathLike) -> Dict[str, Any]:
    """The ``selected_frames.json`` this module wrote; ParallaxError when it
    is absent or was written by another selector."""
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
        description="I1 — keyframes by measured parallax, witness frames, coverage warnings.")
    ap.add_argument("--session", required=True, help="session directory (frames in "
                                                      "<session>/frames, I0 already run)")
    args = ap.parse_args(argv)
    session_dir = Path(args.session)
    frames_dir = session_dir / "frames"
    icfg = load_intake_config()
    try:
        quality = load_quality(frames_dir)
    except QualityError as e:
        raise ParallaxError(str(e)) from e
    result = run_parallax(frames_dir, quality, icfg.parallax, log=print,
                          heartbeat_s=icfg.runtime.heartbeat_s,
                          **read_session_epochs(session_dir))
    p_kf, p_w, p_warn = write_selection(session_dir, result, icfg.parallax)
    print(f"{LOG_TAG} wrote {p_kf}, {p_w}, {p_warn}")
    print(f"{LOG_TAG} {result['n_keyframes']} keyframe(s), {result['n_witness']} witness(es), "
          f"{len(result['warnings'])} warning(s); floor median "
          f"{_fmt(result['floor']['median_px'])} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
