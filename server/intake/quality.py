"""I0 — per-frame quality FEATURES (claude_stac.txt §4-F1).

Every frame gets measured, none gets ranked out: ``fft`` (the legacy
high-frequency energy, motion blur), ``laplacian`` (variance of the Laplacian
on the NATIVE gray, defocus), ``luma_mean``, ``clip_frac`` (share of pixels at
the ends of the 8-bit range), ``inter_frame_diff`` (the legacy 'motion'
quantity, kept for that selector) and ``sharp_rank`` — the rank of ``fft``
among ALL frames of the session, in [0, 1] (1 = sharpest). Downstream stages
(I1 picks the sharpest frame per parallax window) read the features; nothing
here rejects a frame for being blurrier than its neighbours — the legacy
percentile gate (``frames/quality.py``, bottom 15 % by construction) is what
this module replaces.

The ONLY absolute rejection is an exposure the camera cannot have measured
through: mean luma under ``luma_lo`` ("dark"), over ``luma_hi`` ("bright") or
a clipped share over ``clip_frac_max`` ("clipped") — BOUNDS from
``intake.quality`` in config.yaml, each rejection recorded with its reason.

Artifacts (both in <session>/frames/, which a reconstruction replace leaves
alone; both written atomically):
  * ``quality_features.json`` — the report of :func:`analyze_frames`; it names
    its inputs relative to the session and carries the CPU environment it was
    measured with (:func:`intake.stamps.cpu_environment_record`: library
    versions, CPU, BLAS core, JPEG decoder — docs/plan_determinismo.md 74, 79);
    no wall clock, no absolute path and no session epoch is written into it
    (point 70: the same frames give the same bytes whatever ran before);
  * ``frame_quality.json`` — the LEGACY shape the existing consumers read
    (``workers.map_worker._motion_keyframes``, ``frames.selector.
    _load_valid_frame_list``, the vendor's ``_stac_extra_frames``) with
    ``valid`` == usable (exposure only) and every threshold at 0.0.

No decision literal lives here (``tests/test_intake_config.py`` and
``tests/test_precision_config.py`` scan this package); the analysis scales
(``fft_max_side``, ``diff_max_side``) come from config so the scores stay
comparable with the sessions measured before this module existed.

CLI: ``python -m intake.quality --session <dir>`` (or ``--frames-dir <dir>``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from intake.config import QualityConfig, load_intake_config

QUALITY_FEATURES_NAME = "quality_features.json"
LEGACY_FRAME_QUALITY_NAME = "frame_quality.json"
QUALITY_VERSION = 2                     # 2: FFT in explicit float64, environment recorded, no
                                        #    epochs / absolute paths in the product (2026-10-07)
# The epoch every intake product belongs to BY CONSTRUCTION (plan point 63, "numbered relative
# to the reconstruction"): the intake measures frames, before any reconstruction, so what it
# writes is the input of the ORIGINAL reconstruction — epoch 0 — whatever epoch the session's
# output/ holds when it runs (that one goes to the run record, intake/intake_state.timing.json).
INTAKE_EPOCH = 0
PROVENANCE = "tool_measured"
METHOD = "intake_features"
REJECT_REASONS = ("dark", "bright", "clipped")
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")
FEATURE_NAMES = ("fft", "laplacian", "luma_mean", "clip_frac", "inter_frame_diff",
                 "sharp_rank")
LOG_TAG = "[intake.quality]"


class QualityError(RuntimeError):
    """A structural impossibility of I0 (no frames, unreadable frame, mixed
    resolutions, a frame name that is not a video frame number) — always with
    the exact reason."""


class IntakeCancelled(RuntimeError):
    """The caller's ``cancelled()`` returned True inside an intake loop. The
    message names the step and how far it got; nothing half-written is left
    behind (every artifact is written atomically after its loop)."""


Cancelled = Optional[Callable[[], bool]]


def check_cancelled(cancelled: Cancelled, where: str) -> None:
    """Raise :class:`IntakeCancelled` naming ``where`` when ``cancelled()``
    says so (``None`` = the caller offers no cancel)."""
    if cancelled is not None and cancelled():
        raise IntakeCancelled(f"cancelled during {where}")


def intake_epochs() -> Dict[str, int]:
    """The epoch stamps of every intake product: :data:`INTAKE_EPOCH` for both (the intake
    precedes every reconstruction; its products are epoch 0's input by construction)."""
    return {"geometry_epoch": INTAKE_EPOCH, "camera_epoch": INTAKE_EPOCH}


def read_session_epochs(session_dir: os.PathLike) -> Dict[str, int]:
    """The session's current ``geometry_epoch`` / ``camera_epoch`` AT RUN TIME — recorded in
    the intake's run record (``intake_state.timing.json``), never in a product: the intake
    measures FRAMES only, so none of its numbers depends on either epoch, and a product that
    carried them changed bytes with the session's history (point 70). geometry_epoch =
    ``precision.camera.read_geometry_epoch(<session>/output)`` (0 = none);
    camera_epoch = ``output/camera.json``'s ``camera_epoch`` when F0 wrote one,
    else 0. A camera.json that exists but cannot be read fails (CameraError)."""
    from precision.camera import CAMERA_JSON_NAME, load_camera_json, read_geometry_epoch
    out_dir = Path(session_dir) / "output"
    cam = out_dir / CAMERA_JSON_NAME
    return {"geometry_epoch": int(read_geometry_epoch(out_dir)),
            "camera_epoch": int(load_camera_json(cam).camera_epoch) if cam.exists() else 0}


@dataclass(frozen=True)
class FrameQuality:
    """One frame's measured features and its exposure verdict."""
    frame: int                          # video frame number = int(stem)
    file: str                           # basename in frames/
    fft: float                          # high-frequency energy at ≤ fft_max_side (legacy formula)
    laplacian: float                    # var(Laplacian) on the NATIVE gray
    luma_mean: float                    # mean gray, native
    clip_frac: float                    # share of pixels ≤ clip_lo or ≥ clip_hi
    inter_frame_diff: Optional[float]   # mean |Δgray| vs the previous frame at ≤ diff_max_side; None first
    sharp_rank: float                   # rank of fft among ALL frames / (n − 1), 1 = sharpest
    usable: bool                        # exposure verdict (the ONLY rejection)
    reject_reason: Optional[str]        # dark | bright | clipped | None


# ── helpers ──────────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2


def _write_json_atomic(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


def list_frames(frames_dir: os.PathLike) -> List[Path]:
    """Frame files of ``frames_dir`` sorted by video frame number (int(stem)).
    Raises QualityError when the directory holds no frame, a frame whose
    stem is not an integer (the frame-naming contract, conventions rule 10), or
    two files with the same frame number (``000123.jpg`` + ``000123.png``, or
    ``123.jpg`` + ``000123.jpg`` — docs/plan_determinismo.md point 67: which one
    a reader took was the directory listing's order)."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise QualityError(f"{frames_dir} is not a directory — no frames to measure")
    files = [p for p in frames_dir.iterdir()
             if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES]
    keyed = []
    for p in files:
        try:
            keyed.append((int(p.stem), p))
        except ValueError:
            raise QualityError(
                f"frame file {p.name} in {frames_dir} is not named by its video frame "
                f"number (<frame:06d>.jpg) — the intake cannot attribute it") from None
    if not keyed:
        raise QualityError(f"{frames_dir} holds no frame ({', '.join(FRAME_SUFFIXES)})")
    keyed.sort(key=lambda t: (t[0], t[1].name))
    dup = next((i for i in range(1, len(keyed)) if keyed[i][0] == keyed[i - 1][0]), None)
    if dup is not None:
        same = [p.name for k, p in keyed if k == keyed[dup][0]]
        raise QualityError(f"two frame files of {frames_dir} share video frame number "
                           f"{keyed[dup][0]}: {', '.join(same)} — remove one (a frame number "
                           f"names ONE frame)")
    return [p for _, p in keyed]


def read_gray(path: os.PathLike) -> np.ndarray:
    """The frame as an 8-bit gray image at its native resolution."""
    cv2 = _cv2()
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise QualityError(f"frame {path} is unreadable (cv2.imread returned None)")
    return g


def downscale_gray(gray: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    """``gray`` resized so its longest side is ≤ ``max_side`` (never upscaled),
    with the legacy arithmetic (scale = max_side / max(h, w), fx = fy = scale)
    so the scores stay comparable with the sessions measured before."""
    h, w = gray.shape[:2]
    if max(h, w) <= max_side:
        return gray
    scale = max_side / max(h, w)
    return _cv2().resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)


def fft_score(gray_small: np.ndarray) -> float:
    """Legacy high-frequency energy: mean |F| outside the centred square of
    half-side min(h, w) // 8 (the low-frequency block zeroed).

    The FFT runs in EXPLICIT float64 (plan point 74): the legacy formula fed a
    float32 image, which numpy 1.x upcasts to complex128 and numpy 2.x computes
    natively in complex64 — a ~1e-7 relative change that reorders near-tied
    ``sharp_rank`` values and re-routes I1's chain. ``astype(np.float64)`` is
    bit-identical to the numpy 1.26 result of the float32 call (a uint8 image is
    exact in both; ``tests/test_intake_quality.py`` proves it)."""
    f = np.fft.fftshift(np.fft.fft2(gray_small.astype(np.float64)))
    magnitude = np.abs(f)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    magnitude[cy - r:cy + r, cx - r:cx + r] = 0
    return float(np.mean(magnitude))


def laplacian_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian (CV_64F) — defocus blur, on the NATIVE gray."""
    return float(_cv2().Laplacian(gray, _cv2().CV_64F).var())


def frame_features(gray_native: np.ndarray, prev_gray_small: Optional[np.ndarray],
                   cfg: QualityConfig) -> Dict[str, Any]:
    """Features of one frame.

    Returns ``fft`` (on the ≤ ``cfg.fft_max_side`` INTER_AREA downscale),
    ``laplacian`` (native), ``luma_mean`` (native), ``clip_frac`` (share of
    pixels ≤ ``cfg.clip_lo`` or ≥ ``cfg.clip_hi``), ``inter_frame_diff``
    (mean |Δgray| against ``prev_gray_small`` at the ≤ ``cfg.diff_max_side``
    INTER_LINEAR downscale — the legacy 'motion' arithmetic; None when there is
    no previous frame) and ``gray_small`` — THIS frame's diff-scale gray, to be
    handed to the next call as ``prev_gray_small`` (not serialised)."""
    cv2 = _cv2()
    if gray_native.ndim != 2:
        raise QualityError(f"frame_features expects a 2-D gray image, got shape "
                           f"{gray_native.shape}")
    fft_small = downscale_gray(gray_native, cfg.fft_max_side, cv2.INTER_AREA)
    diff_small = downscale_gray(gray_native, cfg.diff_max_side, cv2.INTER_LINEAR)
    clipped = np.count_nonzero((gray_native <= cfg.clip_lo) | (gray_native >= cfg.clip_hi))
    inter = None
    if prev_gray_small is not None:
        if prev_gray_small.shape != diff_small.shape:
            raise QualityError(
                f"previous frame's diff-scale gray {prev_gray_small.shape} does not match "
                f"this frame's {diff_small.shape} — frames of one session share one "
                f"native resolution")
        inter = float(cv2.absdiff(prev_gray_small, diff_small).mean())
    return {
        "fft": fft_score(fft_small),
        "laplacian": laplacian_score(gray_native),
        "luma_mean": float(gray_native.mean()),
        "clip_frac": float(clipped) / float(gray_native.size),
        "inter_frame_diff": inter,
        "gray_small": diff_small,
    }


def exposure_verdict(luma_mean: float, clip_frac: float, cfg: QualityConfig) -> Optional[str]:
    """The reject reason of an impossible exposure, or None when usable. The
    reasons are tested in the order dark, bright, clipped; the first that
    holds is recorded (a black frame is dark before it is clipped)."""
    if luma_mean < cfg.luma_lo:
        return "dark"
    if luma_mean > cfg.luma_hi:
        return "bright"
    if clip_frac > cfg.clip_frac_max:
        return "clipped"
    return None


def sharp_ranks(fft: np.ndarray) -> np.ndarray:
    """Rank of every fft value among all of them, ties averaged, mapped to
    [0, 1] by /(n − 1) (1 = sharpest). One frame alone ranks 1."""
    v = np.asarray(fft, dtype=np.float64)
    n = v.size
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ordinal = np.empty(n, dtype=np.float64)
    ordinal[order] = np.arange(n, dtype=np.float64)
    _, inverse = np.unique(v, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums = np.bincount(inverse, weights=ordinal)
    counts = np.bincount(inverse)
    return (sums / counts)[inverse] / float(n - 1)


# ── the stage ────────────────────────────────────────────────────────────

def analyze_frames(frames_dir: os.PathLike, cfg: QualityConfig, log: Callable = print, *,
                   heartbeat_s: float, cancelled: Cancelled = None,
                   environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Measure every frame of ``frames_dir`` and return the I0 report.

    ``heartbeat_s`` (``intake.runtime.heartbeat_s``, passed by the caller —
    no default, no global config read here) paces the progress lines, which
    carry the rate. ``cancelled()`` is polled once per frame. ``environment``
    is the CPU environment record to write (default: measured now —
    :func:`intake.stamps.cpu_environment_record`).

    Report: ``version``, ``provenance`` ("tool_measured"), ``geometry_epoch`` /
    ``camera_epoch`` (:data:`INTAKE_EPOCH`), ``native_w``/``native_h``,
    ``n_frames``, ``n_usable``, ``rejected`` {reason: count}, ``frames``
    (FrameQuality dicts sorted by frame), ``params`` (the effective
    QualityConfig), ``inputs`` (the frame inventory, paths relative to the
    session) and ``environment``. No time, no absolute path (point 70)."""
    if heartbeat_s <= 0:
        raise QualityError(f"heartbeat_s must be positive, got {heartbeat_s}")
    from intake.stamps import cpu_environment_record
    env = dict(environment) if environment is not None else cpu_environment_record()
    frames_dir = Path(frames_dir)
    paths = list_frames(frames_dir)
    n = len(paths)
    log(f"{LOG_TAG} measuring {n} frame(s) in {frames_dir} (features only — no "
        f"percentile rejection; exposure bounds luma [{cfg.luma_lo:g}, {cfg.luma_hi:g}], "
        f"clipped share ≤ {cfg.clip_frac_max:g})")

    native_hw = None
    prev_small = None
    rows: List[Dict[str, Any]] = []
    bytes_total = 0
    t0 = time.monotonic()
    last_beat = t0
    for i, p in enumerate(paths):
        check_cancelled(cancelled, f"intake I0 (quality) at frame {i}/{n}")
        gray = read_gray(p)
        if native_hw is None:
            native_hw = gray.shape[:2]
        elif gray.shape[:2] != native_hw:
            raise QualityError(
                f"frame {p.name} is {gray.shape[1]}x{gray.shape[0]} but the session's "
                f"frames are {native_hw[1]}x{native_hw[0]} — frames/ mixes resolutions")
        feats = frame_features(gray, prev_small, cfg)
        prev_small = feats.pop("gray_small")
        bytes_total += p.stat().st_size
        rows.append({"frame": int(p.stem), "file": p.name, **feats})
        now = time.monotonic()
        if now - last_beat >= heartbeat_s or i + 1 == n:
            elapsed = max(now - t0, 1e-9)
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / max(rate, 1e-9)
            log(f"{LOG_TAG} {i + 1}/{n} frames ({rate:.1f} frames/s, eta {eta:.0f} s)")
            last_beat = now

    ranks = sharp_ranks(np.array([r["fft"] for r in rows], dtype=np.float64))
    frames: List[FrameQuality] = []
    rejected = {reason: 0 for reason in REJECT_REASONS}
    for r, rank in zip(rows, ranks):
        reason = exposure_verdict(r["luma_mean"], r["clip_frac"], cfg)
        if reason is not None:
            rejected[reason] += 1
        frames.append(FrameQuality(
            frame=r["frame"], file=r["file"], fft=float(r["fft"]),
            laplacian=float(r["laplacian"]), luma_mean=float(r["luma_mean"]),
            clip_frac=float(r["clip_frac"]), inter_frame_diff=r["inter_frame_diff"],
            sharp_rank=float(rank), usable=reason is None, reject_reason=reason))
    n_usable = sum(1 for f in frames if f.usable)
    for reason in REJECT_REASONS:
        if rejected[reason]:
            log(f"{LOG_TAG} {rejected[reason]} frame(s) unusable by exposure: {reason}")
    log(f"{LOG_TAG} done: {n_usable}/{n} usable, native {native_hw[1]}x{native_hw[0]}")

    return {
        "version": QUALITY_VERSION,
        "provenance": PROVENANCE,
        **intake_epochs(),
        "method": METHOD,
        "native_w": int(native_hw[1]),
        "native_h": int(native_hw[0]),
        "n_frames": n,
        "n_usable": n_usable,
        "rejected": rejected,
        "features": list(FEATURE_NAMES),
        "reject_reasons": list(REJECT_REASONS),
        "params": asdict(cfg),
        "inputs": {
            "frames_dir": frames_dir.name,          # relative to the session (point 70)
            "n_frames": n,
            "first": paths[0].name,
            "last": paths[-1].name,
            "bytes_total": int(bytes_total),
        },
        "environment": env,
        "frames": [asdict(f) for f in frames],
    }


def write_quality(frames_dir: os.PathLike, report: Dict[str, Any]) -> Path:
    """frames/quality_features.json (tmp + os.replace)."""
    return _write_json_atomic(Path(frames_dir) / QUALITY_FEATURES_NAME, report)


def legacy_frame_quality(report: Dict[str, Any]) -> Dict[str, Any]:
    """The LEGACY ``frame_quality.json`` shape built from an I0 report.

    Consumers read ``frames[].file``, ``fft_score`` (the sharpness they rank
    by), ``inter_frame_diff`` (the 'motion' quantum) and ``valid`` — here
    ``valid`` == usable, an EXPOSURE verdict only, and every threshold is 0.0
    because nothing is thresholded. ``index`` is the ordinal in frame order
    (the legacy meaning); the first frame's ``inter_frame_diff`` is 0.0 as the
    legacy writer had it. ``blur_score`` carries the native Laplacian
    variance (the legacy one was computed on the ≤ 640 downscale; no consumer
    gates on it). It also carries the artifact stamps (``version``,
    ``provenance``, ``geometry_epoch`` / ``camera_epoch`` = :data:`INTAKE_EPOCH`)
    — extra top-level keys every legacy reader ignores (they read ``frames[]``
    only)."""
    frames = report["frames"]
    n_usable = int(report["n_usable"])
    return {
        "version": QUALITY_VERSION,
        "provenance": PROVENANCE,
        **intake_epochs(),
        "method": METHOD,
        "threshold_fft": 0.0,
        "threshold": 0.0,
        "threshold_percentile": 0.0,
        "total_frames": int(report["n_frames"]),
        "valid_frames": n_usable,
        "rejected_frames": int(report["n_frames"]) - n_usable,
        "rejected_by_exposure": dict(report["rejected"]),
        "source": QUALITY_FEATURES_NAME,
        "frames": [{
            "index": i,
            "file": f["file"],
            "fft_score": float(f["fft"]),
            "blur_score": float(f["laplacian"]),
            "inter_frame_diff": (0.0 if f["inter_frame_diff"] is None
                                 else float(f["inter_frame_diff"])),
            "valid": bool(f["usable"]),
        } for i, f in enumerate(frames)],
    }


def write_legacy_frame_quality(frames_dir: os.PathLike, report: Dict[str, Any]) -> Path:
    """frames/frame_quality.json in the legacy shape (tmp + os.replace)."""
    return _write_json_atomic(Path(frames_dir) / LEGACY_FRAME_QUALITY_NAME,
                              legacy_frame_quality(report))


def load_quality(frames_dir: os.PathLike) -> Dict[str, Any]:
    """The I0 report written by :func:`write_quality`; QualityError when absent
    or of another version."""
    p = Path(frames_dir) / QUALITY_FEATURES_NAME
    if not p.exists():
        raise QualityError(f"{p} does not exist — run intake I0 "
                           f"(python -m intake.quality --session <dir>) first")
    with open(p) as f:
        report = json.load(f)
    if report.get("version") != QUALITY_VERSION:
        raise QualityError(f"{p} is version {report.get('version')!r}, this module reads "
                           f"version {QUALITY_VERSION} — re-run intake I0")
    return report


def run_quality(frames_dir: os.PathLike, cfg: QualityConfig, log: Callable = print, *,
                heartbeat_s: float, cancelled: Cancelled = None,
                environment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """analyze → write both artifacts; returns the report."""
    report = analyze_frames(frames_dir, cfg, log=log, heartbeat_s=heartbeat_s,
                            cancelled=cancelled, environment=environment)
    p1 = write_quality(frames_dir, report)
    p2 = write_legacy_frame_quality(frames_dir, report)
    log(f"{LOG_TAG} wrote {p1.name} and legacy {p2.name} in {Path(frames_dir)}")
    return report


# ── CLI ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m intake.quality",
        description="I0 — per-frame quality features (no percentile rejection).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="session directory (frames in <session>/frames)")
    g.add_argument("--frames-dir", help="frames directory")
    args = ap.parse_args(argv)
    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(args.session) / "frames"
    # the configuration of this job: the session's frozen copy, else the server's frozen now
    # (plan point 69) — <session>/frames → <session>
    from intake.run_config import cli_intake_config
    icfg, _sha = cli_intake_config(frames_dir.parent, log=print)
    report = run_quality(frames_dir, icfg.quality, log=print,
                         heartbeat_s=icfg.runtime.heartbeat_s)
    rejected = ", ".join(f"{k}={v}" for k, v in report["rejected"].items())
    print(f"{LOG_TAG} {report['n_usable']}/{report['n_frames']} usable "
          f"({rejected}); native {report['native_w']}x{report['native_h']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
