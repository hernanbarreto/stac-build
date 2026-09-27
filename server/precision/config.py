"""Typed, validated load of ``reconstruction.precision`` (server/config.yaml).

Decides nothing: it guarantees that every parameter the precision stages use
exists, has the right type and sits inside its declared bounds BEFORE any
geometry is touched. There is NO default in code — a missing key aborts the
load naming it (the YAML documents every value and its provenance: BOUND /
MEASURED / USER DECISION). Each phase of claude_stac.txt §4 adds its section
here and in config.yaml in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

SECTION = "reconstruction.precision"


class PrecisionConfigError(RuntimeError):
    """Raised when ``reconstruction.precision`` is missing, incomplete or out
    of range. The message always names the offending key."""


# ── generic validated readers (the correction/config.py pattern) ─────────

def _require(section: Any, key: str, path: str) -> Any:
    if not isinstance(section, dict) or key not in section:
        raise PrecisionConfigError(
            f"config.yaml is missing mandatory key '{SECTION}.{path}.{key}' — "
            f"add it (see the reconstruction.precision: documentation); there "
            f"is no hidden default in code")
    return section[key]


def _num(section, key, path, lo=None, hi=None, integer=False, lo_excl=False):
    v = _require(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' = {v} is below the valid range "
            f"(min {lo}{' exclusive' if lo_excl else ''})")
    if hi is not None and v > hi:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' = {v} is above the valid range (max {hi})")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _require(section, key, path)
    if not isinstance(v, bool):
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a boolean, got {v!r}")
    return v


def _str(section, key, path) -> str:
    v = _require(section, key, path)
    if not isinstance(v, str) or not v:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a non-empty string, got {v!r}")
    return v


def _enum(section, key, path, allowed: Sequence[str]) -> str:
    v = _str(section, key, path)
    if v not in allowed:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be one of {tuple(allowed)}, got {v!r}")
    return v


def _num_list(section, key, path, lo=None, integer=False, min_len=1,
              increasing=False) -> Tuple[float, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a list of at least {min_len} "
            f"number(s), got {v!r}")
    out: List[float] = []
    for i, x in enumerate(v):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise PrecisionConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be a number, got {x!r}")
        if integer and int(x) != x:
            raise PrecisionConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be an integer, got {x!r}")
        if lo is not None and x < lo:
            raise PrecisionConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' = {x} is below the valid range "
                f"(min {lo})")
        out.append(int(x) if integer else float(x))
    if increasing and any(b <= a for a, b in zip(out, out[1:])):
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be strictly increasing, got {v!r}")
    return tuple(out)


def _str_list(section, key, path, allowed: Optional[Sequence[str]] = None,
              min_len=0) -> Tuple[str, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or len(v) < min_len:
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a list of at least {min_len} "
            f"string(s), got {v!r}")
    for i, x in enumerate(v):
        if not isinstance(x, str):
            raise PrecisionConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be a string, got {x!r}")
        if allowed is not None and x not in allowed:
            raise PrecisionConfigError(
                f"'{SECTION}.{path}.{key}[{i}]' must be one of {tuple(allowed)}, "
                f"got {x!r}")
    return tuple(v)


def _sub(section: Any, key: str, path: str) -> Dict[str, Any]:
    v = _require(section, key, path)
    if not isinstance(v, dict):
        raise PrecisionConfigError(
            f"'{SECTION}.{path}.{key}' must be a mapping, got {type(v).__name__}")
    return v


# ── F0: session camera ───────────────────────────────────────────────────

CAMERA_MODELS = ("OPENCV",)
CAMERA_INIT_SOURCES = ("auto", "omega", "stray")


@dataclass(frozen=True)
class CameraConfig:
    model: str              # OPENCV = [fx, fy, cx, cy, k1, k2, p1, p2]
    init_from: str          # auto (stray when camera_matrix.csv exists) | omega | stray
    fx_spread_warn_pct: float   # BOUND: per-frame omega focal spread above which the
                                # camera.json carries a warning (advisory)
    aspect_tol: float       # BOUND: relative pixel-aspect disagreement tolerated when a
                            # K is rescaled between two resolutions of the same sensor
    undistort_max_iter: int     # BOUND: iteration cap of the point undistortion solver
                                # (cv2.undistortPointsIter) — it STOPS the solver
    undistort_eps_px: float     # BOUND: the solver stops when a point's reprojection
                                # through the lens moves it less than this (native px)


# ── F9: runner (declared in F0 so every stage heartbeats the same way) ───

@dataclass(frozen=True)
class RunnerConfig:
    heartbeat_s: float          # BOUND: seconds between progress lines of a long stage
    perf_checkpoint_s: float    # BOUND: seconds into a GPU stage at which the measured
                                # rate is compared with the expected one (abort with reason)


@dataclass(frozen=True)
class PrecisionConfig:
    enabled: bool
    camera: CameraConfig
    runner: RunnerConfig


def load_precision_config(raw: Optional[Dict[str, Any]] = None) -> PrecisionConfig:
    """Validated PrecisionConfig from the raw config dict (defaults to the
    server-wide ``config.cfg``). Raises PrecisionConfigError naming the first
    offending key."""
    if raw is None:
        from config import cfg as raw_cfg     # server/config.py
        raw = raw_cfg
    recon = (raw or {}).get("reconstruction")
    if not isinstance(recon, dict) or not isinstance(recon.get("precision"), dict):
        raise PrecisionConfigError(
            "config.yaml has no 'reconstruction.precision:' section — the "
            "precision pipeline cannot run without its parameters")
    sec = recon["precision"]

    enabled = _require(sec, "enabled", "")
    if not isinstance(enabled, bool):
        raise PrecisionConfigError(
            f"'{SECTION}.enabled' must be a boolean, got {enabled!r}")

    cam = _sub(sec, "camera", "")
    camera = CameraConfig(
        model=_enum(cam, "model", "camera", CAMERA_MODELS),
        init_from=_enum(cam, "init_from", "camera", CAMERA_INIT_SOURCES),
        fx_spread_warn_pct=_num(cam, "fx_spread_warn_pct", "camera", lo=0.0, lo_excl=True),
        aspect_tol=_num(cam, "aspect_tol", "camera", lo=0.0, lo_excl=True),
        undistort_max_iter=_num(cam, "undistort_max_iter", "camera", lo=1, integer=True),
        undistort_eps_px=_num(cam, "undistort_eps_px", "camera", lo=0.0, lo_excl=True),
    )

    rn = _sub(sec, "runner", "")
    runner = RunnerConfig(
        heartbeat_s=_num(rn, "heartbeat_s", "runner", lo=0.0, lo_excl=True),
        perf_checkpoint_s=_num(rn, "perf_checkpoint_s", "runner", lo=0.0, lo_excl=True),
    )

    return PrecisionConfig(enabled=enabled, camera=camera, runner=runner)
