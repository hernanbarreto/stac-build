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


# ── F2: metric gauge along the walk ──────────────────────────────────────

GAUGE_INSTRUMENTS = ("da3_windows", "da3_mono", "vio", "stray", "known_dims")


@dataclass(frozen=True)
class GaugeConfig:
    window_frames: int          # BOUND: keyframes per DA3 multi-view window (one joint inference)
    window_overlap_frac: float  # BOUND: share of a window shared with the next (the frames
                                # that chain two windows' poses into one walk)
    process_res: int            # BOUND: DA3 processing resolution (long side, px)
    model_id: str               # DA3 checkpoint (the NESTED model: metric multi-view)
    knot_walk_m: float          # BOUND: knot spacing of the continuous scale model (m of walk)
    heldout_confidence: float   # declared confidence of the held-out comparisons
    smooth_grid: Tuple[float, ...]  # BOUND: the dimensionless smoothness weights searched by
                                    # leave-one-window-out (× the rows' weight per knot)
    huber_k: float              # Huber's tuning constant of the per-frame gain (statistics:
                                # 1.345 = 95 % efficiency under normal noise)
    instruments: Tuple[str, ...]    # scale instruments that may enter the model


# ── F3: Omega ─────────────────────────────────────────────────────────────

OMEGA_MODES = ("balanced", "max_size")


@dataclass(frozen=True)
class ResolutionProbeConfig:
    enabled: bool
    resolutions: Tuple[int, ...]    # the resolutions compared on the same window
    mode: str                       # balanced (≈ res² pixels) | max_size (longest side = res)
    window_frames: int              # BOUND: keyframes of the probe window
    pair_samples: int               # BOUND: surface samples per frame pair


@dataclass(frozen=True)
class OmegaConfig:
    resolution_probe: ResolutionProbeConfig


# ── F4: native-pixel tracks ───────────────────────────────────────────────

DENSE_MATCHERS = ("none", "roma")


@dataclass(frozen=True)
class TracksConfig:
    tracker_long_side: int      # BOUND: the tracker's input long side (native when smaller —
                                # never upsampled)
    tracker_stride: int         # the tracker network's stride: its input sides are multiples
    window_frames: int          # BOUND: frames per tracking window
    window_overlap_frac: float  # BOUND: share of a keyframe window shared with the next
    query_frames_per_window: int    # BOUND: query frames spread over a keyframe window
    loop_half_window: int       # BOUND: keyframes on each side of a loop pair's ends
    grid_side: int              # BOUND: query grid columns (rows follow the aspect)
    max_corners: int            # BOUND: Shi-Tomasi corners per query frame
    corner_quality: float       # BOUND: Shi-Tomasi quality level (share of the best corner)
    corner_min_distance_px: float   # BOUND: minimum corner spacing (tracker px)
    depth_edge_tol_rel: float   # BOUND: 2×2 depth spread / nearest depth above this = an edge
    vis_thresh: float           # BOUND: tracker visibility below this = not observed
    heldout_frac: float         # BOUND: share of the TRACKS held out of every fit
    seed: int                   # the split's fixed seed
    dense_matcher: str          # none | roma (optional, off)


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
    gauge: GaugeConfig
    omega: OmegaConfig
    tracks: TracksConfig
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

    g = _sub(sec, "gauge", "")
    gauge = GaugeConfig(
        # two windows chain through at least two shared frames: a window needs four
        window_frames=_num(g, "window_frames", "gauge", lo=4, integer=True),
        window_overlap_frac=_num(g, "window_overlap_frac", "gauge", lo=0.0, hi=1.0,
                                 lo_excl=True),
        process_res=_num(g, "process_res", "gauge", lo=14, integer=True),
        model_id=_str(g, "model_id", "gauge"),
        knot_walk_m=_num(g, "knot_walk_m", "gauge", lo=0.0, lo_excl=True),
        heldout_confidence=_num(g, "heldout_confidence", "gauge", lo=0.0, hi=1.0,
                                lo_excl=True),
        smooth_grid=tuple(_num_list(g, "smooth_grid", "gauge", lo=0.0, min_len=1)),
        huber_k=_num(g, "huber_k", "gauge", lo=0.0, lo_excl=True),
        instruments=tuple(_str_list(g, "instruments", "gauge", allowed=GAUGE_INSTRUMENTS)),
    )
    if gauge.window_overlap_frac >= 1.0 or gauge.heldout_confidence >= 1.0:
        raise PrecisionConfigError(
            f"'{SECTION}.gauge.window_overlap_frac' and '.heldout_confidence' must be below 1")
    if int(round(gauge.window_frames * gauge.window_overlap_frac)) < 2:
        raise PrecisionConfigError(
            f"'{SECTION}.gauge.window_frames' x '.window_overlap_frac' shares fewer than 2 "
            f"frames between windows — two windows cannot be chained")

    om = _sub(sec, "omega", "")
    rp = _sub(om, "resolution_probe", "omega")
    omega = OmegaConfig(resolution_probe=ResolutionProbeConfig(
        enabled=_bool(rp, "enabled", "omega.resolution_probe"),
        # Omega's patch is 16 px: a resolution below one patch has no token
        resolutions=tuple(int(v) for v in _num_list(rp, "resolutions", "omega.resolution_probe",
                                                     lo=16, integer=True)),
        mode=_enum(rp, "mode", "omega.resolution_probe", OMEGA_MODES),
        window_frames=_num(rp, "window_frames", "omega.resolution_probe", lo=2, integer=True),
        # depth_pair_samples starves a pair under 500 samples
        pair_samples=_num(rp, "pair_samples", "omega.resolution_probe", lo=500, integer=True),
    ))

    tk = _sub(sec, "tracks", "")
    tracks = TracksConfig(
        tracker_long_side=_num(tk, "tracker_long_side", "tracks", lo=16, integer=True),
        tracker_stride=_num(tk, "tracker_stride", "tracks", lo=1, integer=True),
        window_frames=_num(tk, "window_frames", "tracks", lo=3, integer=True),
        window_overlap_frac=_num(tk, "window_overlap_frac", "tracks", lo=0.0, hi=1.0),
        query_frames_per_window=_num(tk, "query_frames_per_window", "tracks", lo=1, integer=True),
        loop_half_window=_num(tk, "loop_half_window", "tracks", lo=0, integer=True),
        grid_side=_num(tk, "grid_side", "tracks", lo=2, integer=True),
        max_corners=_num(tk, "max_corners", "tracks", lo=0, integer=True),
        corner_quality=_num(tk, "corner_quality", "tracks", lo=0.0, hi=1.0, lo_excl=True),
        corner_min_distance_px=_num(tk, "corner_min_distance_px", "tracks", lo=0.0),
        depth_edge_tol_rel=_num(tk, "depth_edge_tol_rel", "tracks", lo=0.0, lo_excl=True),
        vis_thresh=_num(tk, "vis_thresh", "tracks", lo=0.0, hi=1.0),
        heldout_frac=_num(tk, "heldout_frac", "tracks", lo=0.0, hi=1.0),
        seed=_num(tk, "seed", "tracks", lo=0, integer=True),
        dense_matcher=_enum(tk, "dense_matcher", "tracks", DENSE_MATCHERS),
    )
    if tracks.window_overlap_frac >= 1.0 or tracks.heldout_frac >= 1.0:
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.window_overlap_frac' and '.heldout_frac' must be below 1")
    if tracks.dense_matcher != "none":
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.dense_matcher' = {tracks.dense_matcher!r}: no dense matcher is "
            f"vendored (claude_stac.txt §4-F4: optional, off; its licence must be checked and "
            f"recorded in vendor/VENDORS.lock.md before it can be enabled)")

    rn = _sub(sec, "runner", "")
    runner = RunnerConfig(
        heartbeat_s=_num(rn, "heartbeat_s", "runner", lo=0.0, lo_excl=True),
        perf_checkpoint_s=_num(rn, "perf_checkpoint_s", "runner", lo=0.0, lo_excl=True),
    )

    return PrecisionConfig(enabled=enabled, camera=camera, gauge=gauge, omega=omega,
                           tracks=tracks, runner=runner)
