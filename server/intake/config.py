"""Typed, validated load of the top-level ``intake:`` section (server/config.yaml).

Decides nothing: it guarantees that every parameter the intake stages (I0
quality features, I1 parallax keyframes, I2 content tags) use exists, has the
right type and sits inside its declared bounds BEFORE a single frame is read.
There is NO default in code — a missing key aborts the load naming it (the
YAML documents every value and its provenance: BOUND / MEASURED / USER
DECISION). The loader pattern is ``precision/config.py``.
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


def _enum(section, key, path, allowed: Sequence[str]) -> str:
    v = _str(section, key, path)
    if v not in allowed:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be one of {tuple(allowed)}, got {v!r}")
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


def _str_list(section, key, path, allowed: Optional[Sequence[str]] = None,
              min_len=0) -> Tuple[str, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len:
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a list of at least {min_len} "
            f"string(s), got {v!r}")
    for i, x in enumerate(v):
        if not isinstance(x, str) or not x:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be a non-empty string, got {x!r}")
        if allowed is not None and x not in allowed:
            raise IntakeConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be one of {tuple(allowed)}, "
                f"got {x!r}")
    return tuple(v)


def _sub(section: Any, key: str, path: str) -> Dict[str, Any]:
    v = _require(section, key, path)
    if not isinstance(v, dict):
        raise IntakeConfigError(
            f"'{SECTION}.{path}.{key}' must be a mapping, got {type(v).__name__}")
    return v


# ── dataclasses (exact field names — the other intake modules code against them)

GRAY_MAX = 255          # 8-bit gray: the range luma / clip bounds live in

CONTENT_CLASSES = ("dynamic", "occluder", "reflective", "low_info")
SAM3_SCOPES = ("flagged_ranges", "all")
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
        "step_null": ("removed 2026-09-27 with the per-step accumulation: parallax is "
                      "measured against the anchor keyframe and the warnings' noise "
                      "reference is always the warp twin + the chain's forward-backward "
                      "error (intake/parallax.py)"),
    },
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
    reference_max_eval: int             # BOUND: residual evaluations of the rotation fit per frame
    reference_ftol: float               # BOUND: the rotation fit stops when a step lowers its cost by
                                        # less than this (relative)


@dataclass(frozen=True)
class ContentConfig:
    """I2 — VLM content tags (vlm_proposed) and SAM3 exclusion masks."""
    enabled: bool
    backend: str                        # semantic-service backend name (semantic.backends.<name>)
    batch: int                          # BOUND: images per VLM call (≤ the backend's max_images_per_prompt)
    max_tokens: int                     # BOUND: VLM generation cap per call
    exclusion_classes: Tuple[str, ...]  # classes whose SAM3 masks EXCLUDE observations
    weight_classes: Tuple[str, ...]     # classes that only WEIGH (provenance + report)
    sam3_scope: str                     # flagged_ranges | all
    prompts: Dict[str, Tuple[str, ...]]  # SAM3 prompts per exclusion class (keys ⊇ exclusion_classes)
    sam3_batch: int                     # frames per SAM3 session — models.segmentation.batch_size
                                        # (the segmentation stage's own key, validated here)


@dataclass(frozen=True)
class IntakeConfig:
    runtime: RuntimeConfig
    quality: QualityConfig
    parallax: ParallaxConfig
    content: ContentConfig


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
        reference_ftol=_num(p, "reference_ftol", "parallax", lo=0.0, hi=1.0, lo_excl=True),
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


def _load_content(sec: Dict[str, Any], raw: Optional[Dict[str, Any]]) -> ContentConfig:
    c = _sub(sec, "content", "")
    # sam3_batch is read from models.segmentation.batch_size, not from this section
    _refuse_unknown(c, "content", tuple(k for k in _fields(ContentConfig) if k != "sam3_batch"))
    exclusion = _str_list(c, "exclusion_classes", "content", allowed=CONTENT_CLASSES)
    weight = _str_list(c, "weight_classes", "content", allowed=CONTENT_CLASSES)
    overlap = sorted(set(exclusion) & set(weight))
    if overlap:
        raise IntakeConfigError(
            f"'{SECTION}.content.exclusion_classes' and "
            f"'{SECTION}.content.weight_classes' must be disjoint — {overlap} appear in "
            f"both (a class either excludes observations or only weighs them)")
    if len(set(exclusion)) != len(exclusion):
        raise IntakeConfigError(
            f"'{SECTION}.content.exclusion_classes' lists a class twice: {exclusion}")
    if len(set(weight)) != len(weight):
        raise IntakeConfigError(
            f"'{SECTION}.content.weight_classes' lists a class twice: {weight}")

    prompts_raw = _sub(c, "prompts", "content")
    prompts: Dict[str, Tuple[str, ...]] = {}
    for cls in prompts_raw:
        if cls not in CONTENT_CLASSES:
            raise IntakeConfigError(
                f"'{SECTION}.content.prompts.{cls}' names an unknown class — allowed "
                f"{CONTENT_CLASSES}")
        prompts[cls] = _str_list(prompts_raw, cls, "content.prompts", min_len=1)
    missing = [cls for cls in exclusion if cls not in prompts]
    if missing:
        raise IntakeConfigError(
            f"'{SECTION}.content.prompts' lacks the SAM3 prompts of exclusion class(es) "
            f"{missing} — every exclusion class needs at least one prompt")

    backend = _str(c, "backend", "content")
    batch = _num(c, "batch", "content", lo=1, integer=True)
    # cross-section consistency: the VLM call carries `batch` images and the
    # semantic service caps images per prompt per backend — a batch above that
    # cap fails at the first call, so it fails here, naming both keys.
    sem = (raw or {}).get("semantic") if isinstance(raw, dict) else None
    backends = (sem or {}).get("backends") if isinstance(sem, dict) else None
    if isinstance(backends, dict):
        if backend not in backends:
            raise IntakeConfigError(
                f"'{SECTION}.content.backend' = {backend!r} is not declared under "
                f"'semantic.backends' ({sorted(backends)})")
        cap = backends[backend].get("max_images_per_prompt") \
            if isinstance(backends[backend], dict) else None
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and batch > cap:
            raise IntakeConfigError(
                f"'{SECTION}.content.batch' = {batch} exceeds "
                f"'semantic.backends.{backend}.max_images_per_prompt' = {cap}")

    return ContentConfig(
        enabled=_bool(c, "enabled", "content"),
        backend=backend,
        batch=batch,
        max_tokens=_num(c, "max_tokens", "content", lo=1, integer=True),
        exclusion_classes=exclusion,
        weight_classes=weight,
        sam3_scope=_enum(c, "sam3_scope", "content", SAM3_SCOPES),
        prompts=prompts,
        sam3_batch=_sam3_batch(raw),
    )


def _sam3_batch(raw: Optional[Dict[str, Any]]) -> int:
    """``models.segmentation.batch_size`` — the SAM3 session length the
    segmentation stage uses; I2 segments its exclusion masks with the same one.
    It lives outside ``intake:`` (one key, one owner) and is validated here so
    a bad value fails at load, before the VLM has tagged a single frame."""
    key = "models.segmentation.batch_size"
    models = raw.get("models") if isinstance(raw, dict) else None
    seg = models.get("segmentation") if isinstance(models, dict) else None
    if not isinstance(seg, dict) or "batch_size" not in seg:
        raise IntakeConfigError(
            f"config.yaml is missing '{key}' — intake I2 segments its exclusion masks "
            f"with the segmentation stage's SAM3 batch ('{SECTION}.content' reads it); "
            f"there is no hidden default in code")
    v = seg["batch_size"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or v < 1:
        raise IntakeConfigError(
            f"'{key}' must be a positive integer (read by '{SECTION}.content' for the SAM3 "
            f"exclusion masks), got {v!r}")
    return int(v)


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
            "(I0 quality, I1 parallax keyframes, I2 content) cannot run without "
            "its parameters")
    return IntakeConfig(
        runtime=_load_runtime(sec),
        quality=_load_quality(sec),
        parallax=_load_parallax(sec),
        content=_load_content(sec, raw),
    )
