"""Typed, validated load of the top-level ``intake:`` section (server/config.yaml).

Decides nothing: it guarantees that every parameter the intake stages (I0
quality features, I1 parallax keyframes) use exists, has the right type and
sits inside its declared bounds BEFORE a single frame is read. There is NO
default in code — a missing key aborts the load naming it (the YAML documents
every value and its provenance: BOUND / MEASURED / USER DECISION). The loader
pattern is ``precision/config.py``.

I2 (VLM content tags + SAM3 exclusion masks, ``intake.content``) was REMOVED
from the pipeline (USER 2026-10-08, docs/plan_determinismo.md point 75: "ya no
corre VLM ahí"): a leftover ``intake.content`` section fails the load naming
it (:data:`REMOVED_SECTIONS`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SECTION = "intake"


class IntakeConfigError(RuntimeError):
    """Raised when ``intake:`` is missing, incomplete or out of range. The
    message always names the offending key."""


# ── generic validated readers (the precision/config.py pattern) ──────────

def _require(section: Any, key: str, path: str) -> Any:
    if not isinstance(section, dict) or key not in section:
        raise IntakeConfigError(
            f"config.yaml is missing mandatory key '{SECTION}.{path}.{key}' — "
            f"add it (see the intake: documentation); there is no hidden "
            f"default in code")
    return section[key]


def _num(section, key, path, lo=None, hi=None, integer=False, lo_excl=False):
    v = _require(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' = {v} is below the valid range "
            f"(min {lo}{' exclusive' if lo_excl else ''})")
    if hi is not None and v > hi:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' = {v} is above the valid range (max {hi})")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _require(section, key, path)
    if not isinstance(v, bool):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a boolean, got {v!r}")
    return v


def _str(section, key, path) -> str:
    v = _require(section, key, path)
    if not isinstance(v, str) or not v:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a non-empty string, got {v!r}")
    return v


def _num_list(section, key, path, lo=None, integer=False, min_len=1,
              increasing=False) -> Tuple[float, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a list of at least {min_len} "
            f"number(s), got {v!r}")
    out: List[float] = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be a number, got {x!r}")
        if integer and int(x) != x:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be an integer, got {x!r}")
        if lo is not None and x < lo:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' = {x} is below the valid range "
                f"(min {lo})")
        out.append(int(x) if integer else float(x))
    if increasing and any(b <= a for a, b in zip(out, out[1:])):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be strictly increasing, got {v!r}")
    return tuple(out)


def _sub(section: Any, key: str, path: str) -> Dict[str, Any]:
    v = _require(section, key, path)
    if not isinstance(v, dict):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a mapping, got {type(v).__name__}")
    return v


# ── dataclasses (exact field names — the other intake modules code against them)

GRAY_MAX = 255          # 8-bit gray: the range luma / clip bounds live in

# Keys the loader REFUSES because the code that read them is gone — a leftover in a
# config or a session YAML fails the load naming the key and the reason (a key
# nothing reads would only pretend to configure something).
_CUT_BACK = ("removed 2026-09-28 when I1 was cut back to the claude_stac.txt §4-F1 spec "
             "(quantile of the residual w.r.t. the rotation reference; no rigidity bootstrap, "
             "window refinement or noise-floor twin) — intake/parallax.py")
REMOVED_KEYS: Dict[str, Dict[str, str]] = {
    "parallax": {
        **{k: _CUT_BACK for k in ("rotation_floor_factor",
                                  "rigidity_confidence", "rigidity_bootstrap",
                                  "refine_max_iter", "refine_eps_px")},
        "reference_ftol": ("removed 2026-09-28: it STOPPED the rotation fit before convergence and "
                           "the reading depended on where it stopped (with scipy 1.15's MINPACK "
                           "the same inputs gave different keyframes run to run) — the fit now "
                           "runs to reference_tol (intake/parallax.py fit_rotation)"),
        "step_null": ("removed 2026-09-27 with the per-step accumulation: parallax is "
                      "measured against the anchor keyframe and the warnings' noise "
                      "reference is always the warp twin + the chain's forward-backward "
                      "error (intake/parallax.py)"),
    },
}
# Whole SECTIONS the loader refuses, for the same reason (the stage is gone).
REMOVED_SECTIONS: Dict[str, str] = {
    "content": ("intake I2 — VLM content tags + SAM3 exclusion masks — was removed from the "
                "pipeline (USER 2026-10-08, docs/plan_determinismo.md point 75: the VLM runs "
                "once, as the stage after the cloud stage; no exclusion masks, no "
                "content_tags.json) — delete the section"),
}


@dataclass(frozen=True)
class RuntimeConfig:
    heartbeat_s: float          # BOUND: seconds between progress lines of a long loop


@dataclass(frozen=True)
class QualityConfig:
    """I0 — features and the ONLY absolute rejection (impossible exposure)."""
    luma_lo: float              # BOUND: mean luma below this = unusable ("dark")
    luma_hi: float              # BOUND: mean luma above this = unusable ("bright")
    clip_frac_max: float        # BOUND: share of clipped pixels above this = unusable ("clipped")
    clip_lo: int                # BOUND: gray value at or below which a pixel counts as clipped dark
    clip_hi: int                # BOUND: gray value at or above which a pixel counts as clipped bright
    fft_max_side: int           # analysis downscale of the FFT sharpness feature (legacy scale)
    diff_max_side: int          # analysis downscale of inter_frame_diff (legacy 'motion' selector)


@dataclass(frozen=True)
class ParallaxConfig:
    """I1 — tracks matched to the anchor, parallax w.r.t. the rotation reference,
    quantum, witnesses, warnings (intake/parallax.py)."""
    grid_side: int                      # seed grid per anchor (grid_side² points at process scale)
    process_scale: float                # tracking scale (LK runs on the downscaled gray)
    ransac_px: float                    # BOUND: RANSAC bound of the homography that starts the
                                        # rotation fit, native px
    parallax_quantum_px: float          # BOUND: parallax (native px) from the anchor that makes a
                                        # keyframe
    witness_min_parallax_px: float      # BOUND: parallax from the last chosen witness that makes a
                                        # witness; also the least parallax that counts as a baseline
                                        # (track-loss keyframe, pure_rotation warning)
    min_tracks: int                     # BOUND: fewer matched tracks = frame not measured (lost)
    lk_win: int                         # LK window side at process scale
    lk_levels: int                      # LK pyramid levels
    fb_max_px: float                    # BOUND: forward-backward LK disagreement tolerated (native px)
    warn_static_disp_px: float          # BOUND: median displacement from the anchor under this = still
    warn_rotation_min_disp_px: float    # BOUND: displacement from which the view moved (pure_rotation)
    warn_min_run_frames: int            # BOUND: consecutive flagged frames that make a coverage warning
    parallax_quantile: float            # BOUND: parallax = this quantile of the tracks' residuals
                                        # w.r.t. the rotation reference
    keyframe_band_frac: float           # BOUND: the keyframe's reading lies in
                                        # [(1 − band), (1 + band)] × quantum
    reference_max_eval: int             # BOUND: residual evaluations of the rotation fit per frame;
                                        # reaching it = NOT converged = the frame is not measured
    reference_tol: float                # declared convergence tolerance of the rotation fit (step,
                                        # cost change and gradient, scipy trf xtol/ftol/gtol)
    focal_probe_frames: int             # BOUND: frames of the DA3 focal probe (spread over the video)
    focal_probe_res: Any                # DA3 process_res of the focal probe: an int or "native"
    focal_probe_model: str              # DA3 model of the focal probe
    vram_calibration_frames: int        # BOUND: frames of the one DA3 window the calibration CLI measures the
                                        # card's footprint on (server/card_table.json; no run measures it)
    vram_margin_frac: float             # BOUND: share of the card's TOTAL memory left unused when sizing a
                                        # DA3 window / Omega's resolution (never the free memory)


@dataclass(frozen=True)
class IntakeConfig:
    runtime: RuntimeConfig
    quality: QualityConfig
    parallax: ParallaxConfig


# ── section readers ──────────────────────────────────────────────────────

def _load_runtime(sec: Dict[str, Any]) -> RuntimeConfig:
    rn = _sub(sec, "runtime", "")
    _refuse_unknown(rn, "runtime", _fields(RuntimeConfig))
    return RuntimeConfig(
        heartbeat_s=_num(rn, "heartbeat_s", "runtime", lo=0.0, lo_excl=True),
    )


def _load_quality(sec: Dict[str, Any]) -> QualityConfig:
    q = _sub(sec, "quality", "")
    _refuse_unknown(q, "quality", _fields(QualityConfig))
    out = QualityConfig(
        luma_lo=_num(q, "luma_lo", "quality", lo=0.0, hi=float(GRAY_MAX)),
        luma_hi=_num(q, "luma_hi", "quality", lo=0.0, hi=float(GRAY_MAX)),
        clip_frac_max=_num(q, "clip_frac_max", "quality", lo=0.0, hi=1.0),
        clip_lo=_num(q, "clip_lo", "quality", lo=0, hi=GRAY_MAX, integer=True),
        clip_hi=_num(q, "clip_hi", "quality", lo=0, hi=GRAY_MAX, integer=True),
        fft_max_side=_num(q, "fft_max_side", "quality", lo=1, integer=True),
        diff_max_side=_num(q, "diff_max_side", "quality", lo=1, integer=True),
    )
    if not out.luma_lo < out.luma_hi:
        raise IntakeConfigError(
            f"'{SECTION}.quality.luma_lo' ({out.luma_lo}) must be below "
            f"'{SECTION}.quality.luma_hi' ({out.luma_hi}) — the usable exposure "
            f"band would be empty")
    if not out.clip_lo < out.clip_hi:
        raise IntakeConfigError(
            f"'{SECTION}.quality.clip_lo' ({out.clip_lo}) must be below "
            f"'{SECTION}.quality.clip_hi' ({out.clip_hi}) — every pixel would "
            f"count as clipped")
    return out


def _refuse_unknown(section: Dict[str, Any], path: str, known: Sequence[str]) -> None:
    """A key the section's dataclass does not declare fails the load, naming
    it — and, for a removed key, why it was removed (REMOVED_KEYS)."""
    removed = REMOVED_KEYS.get(path, {})
    for key in section:
        if key in removed:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}' is no longer a parameter — {removed[key]}; "
                f"delete it from the configuration")
        if key not in known:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}' is not a parameter of {SECTION}.{path} "
                f"(known: {sorted(known)}) — a key nothing reads cannot configure anything")


def _fields(cls) -> Tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(cls))


def _load_parallax(sec: Dict[str, Any]) -> ParallaxConfig:
    p = _sub(sec, "parallax", "")
    _refuse_unknown(p, "parallax", _fields(ParallaxConfig))
    out = ParallaxConfig(
        grid_side=_num(p, "grid_side", "parallax", lo=2, integer=True),
        process_scale=_num(p, "process_scale", "parallax", lo=0.0, hi=1.0, lo_excl=True),
        ransac_px=_num(p, "ransac_px", "parallax", lo=0.0, lo_excl=True),
        parallax_quantum_px=_num(p, "parallax_quantum_px", "parallax", lo=0.0, lo_excl=True),
        witness_min_parallax_px=_num(p, "witness_min_parallax_px", "parallax", lo=0.0,
                                     lo_excl=True),
        # a homography needs four correspondences: the mathematical minimum, not a decision
        min_tracks=_num(p, "min_tracks", "parallax", lo=4, integer=True),
        lk_win=_num(p, "lk_win", "parallax", lo=3, integer=True),
        lk_levels=_num(p, "lk_levels", "parallax", lo=0, integer=True),
        fb_max_px=_num(p, "fb_max_px", "parallax", lo=0.0, lo_excl=True),
        warn_static_disp_px=_num(p, "warn_static_disp_px", "parallax", lo=0.0),
        warn_rotation_min_disp_px=_num(p, "warn_rotation_min_disp_px", "parallax", lo=0.0,
                                       lo_excl=True),
        warn_min_run_frames=_num(p, "warn_min_run_frames", "parallax", lo=1, integer=True),
        # a quantile lives in (0, 1]; 1.0 is the maximum residual
        parallax_quantile=_num(p, "parallax_quantile", "parallax", lo=0.0, hi=1.0,
                               lo_excl=True),
        # a band of 1 would admit the anchor's own neighbour; 0 = the closing frame only
        keyframe_band_frac=_num(p, "keyframe_band_frac", "parallax", lo=0.0, hi=1.0),
        reference_max_eval=_num(p, "reference_max_eval", "parallax", lo=1, integer=True),
        # MINPACK needs a positive tolerance; 1 would stop on the first step
        reference_tol=_num(p, "reference_tol", "parallax", lo=0.0, hi=1.0, lo_excl=True),
        focal_probe_frames=_num(p, "focal_probe_frames", "parallax", lo=2, integer=True),
        focal_probe_res=(_str(p, "focal_probe_res", "parallax")
                         if p.get("focal_probe_res") == "native"
                         else _num(p, "focal_probe_res", "parallax", lo=14, integer=True)),
        focal_probe_model=_str(p, "focal_probe_model", "parallax"),
        vram_calibration_frames=_num(p, "vram_calibration_frames", "parallax", lo=2, integer=True),
        vram_margin_frac=_num(p, "vram_margin_frac", "parallax", lo=0.0, hi=0.9),
    )
    if out.keyframe_band_frac >= 1.0:
        raise IntakeConfigError(
            f"'{SECTION}.parallax.keyframe_band_frac' = {out.keyframe_band_frac} must be "
            f"below 1 — the keyframe would be allowed to keep no baseline at all")
    if not out.warn_static_disp_px < out.warn_rotation_min_disp_px:
        raise IntakeConfigError(
            f"'{SECTION}.parallax.warn_static_disp_px' ({out.warn_static_disp_px}) must "
            f"be below '{SECTION}.parallax.warn_rotation_min_disp_px' "
            f"({out.warn_rotation_min_disp_px}) — a step cannot be both still and a "
            f"pure rotation")
    return out


def _refuse_removed_sections(sec: Dict[str, Any]) -> None:
    """A section of ``intake:`` whose stage is gone fails the load naming it and why
    (:data:`REMOVED_SECTIONS`) — a section nothing reads would only pretend to configure a
    stage that does not run."""
    for key in sec:
        if key in REMOVED_SECTIONS:
            raise IntakeConfigError(
                f"'{SECTION}.{key}' is no longer a section — {REMOVED_SECTIONS[key]}")


def load_intake_config(raw: Optional[Dict[str, Any]] = None) -> IntakeConfig:
    """Validated IntakeConfig from the raw config dict (defaults to the
    server-wide ``config.cfg``). Raises IntakeConfigError naming the first
    offending key."""
    if raw is None:
        from config import cfg as raw_cfg     # server/config.py
        raw = raw_cfg
    sec = (raw or {}).get(SECTION) if isinstance(raw, dict) else None
    if not isinstance(sec, dict):
        raise IntakeConfigError(
            "config.yaml has no top-level 'intake:' section — the intake "
            "(I0 quality, I1 parallax keyframes) cannot run without its parameters")
    _refuse_removed_sections(sec)
    return IntakeConfig(
        runtime=_load_runtime(sec),
        quality=_load_quality(sec),
        parallax=_load_parallax(sec),
    )
