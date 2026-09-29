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
    undistort_roundtrip_ulps: int   # BOUND: float64 resolution of a round trip through the
                                    # lens, in ulps of the camera's largest pixel magnitude —
                                    # a point converged when re-distorted it lands within
                                    # eps_px + this × ulp of where it was observed


# ── F2: metric gauge along the walk ──────────────────────────────────────

GAUGE_INSTRUMENTS = ("da3_windows", "da3_mono", "vio", "stray", "known_dims")


@dataclass(frozen=True)
class GaugeConfig:
    window_frames: int          # BOUND: keyframes per DA3 multi-view window (one joint inference)
    window_overlap_frac: float  # BOUND: share of a window shared with the next (the frames
                                # that chain two windows' poses into one walk)
    process_res: Any            # DA3 processing resolution (long side, px) or "native"
    model_id: str               # DA3 checkpoint (the NESTED model: metric multi-view)
    knot_walk_m: float          # BOUND: knot spacing of the continuous scale model (m of walk)
    heldout_confidence: float   # declared confidence of the held-out comparisons
    smooth_grid: Tuple[float, ...]  # BOUND: the dimensionless smoothness weights searched by
                                    # leave-one-window-out (× the rows' weight per knot)
    huber_k: float              # Huber's tuning constant of the per-frame gain (statistics:
                                # 1.345 = 95 % efficiency under normal noise)
    huber_tol: float            # declared tolerance: the IRLS has converged when the log gain
                                # moves less than this (× max(1, |gain|)) in one step
    huber_max_iter: int         # BOUND: generous; a gain still moving here is NOT converged —
                                # its row is excluded and reported, never used
    instruments: Tuple[str, ...]    # scale instruments that may enter the model; the FIRST one
                                    # judged is the default — another replaces it only when its
                                    # held-out error is lower beyond the sample's noise


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
class CoherenceProbeConfig:
    enabled: bool
    lengths: Tuple[int, ...]        # BOUND: nested window lengths (keyframes) tried, plus the whole set
    heldout_confidence: float       # declared confidence of the half-to-half drift verdict
    bootstrap: int                  # BOUND: resamples of the steps per half
    seed: int                       # the bootstrap's fixed seed


@dataclass(frozen=True)
class OmegaConfig:
    resolution_probe: ResolutionProbeConfig
    coherence_probe: CoherenceProbeConfig


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
    seed: int                   # the split's fixed seed (and the tracker run's torch seed)
    dense_matcher: str          # none | roma (optional, off)
    tracker_weights_url: str    # the tracker checkpoint at a PINNED revision (never 'main')
    tracker_weights_sha256: str     # its content: a file that does not hash to this is refused


# ── F5: joint refinement + witness localisation ─────────────────────────

@dataclass(frozen=True)
class RefineConfig:
    min_tri_deg: float          # BOUND: a landmark's rays must span at least this (degrees)
    huber_px: float             # BOUND: Huber scale of the reprojection loss (native px)
    max_iterations: int         # BOUND: Ceres iterations per rung (reconstruction.colmap_ba)
    ceres_threads: int          # must stay 1 for bit-identical results (validated at load)
    focal_block_frames: int     # BOUND: keyframes per temporal focal block (rung R3)
    heldout_confidence: float   # declared confidence of the held-out comparisons
    permutations: int           # BOUND: permutation / bootstrap resamples of the tests
    seed: int                   # their fixed seed
    min_witness_corr: int       # BOUND: 2D-3D correspondences a witness needs for PnP


# ── F6: native-resolution depth ─────────────────────────────────────────

PRIOR_FILL_MODES = ("keep", "drop")


@dataclass(frozen=True)
class ColmapConfig:
    enabled: bool               # COLMAP PatchMatch runs in the chain
    as_tier: bool               # its geometric depth enters the fusion as TIER 2 (else A/B only)
    binary: str                 # the COLMAP executable built WITH CUDA (PatchMatch needs it)
    window_radius: int          # COLMAP's own defaults (PatchMatchOptions)
    num_iterations: int
    geom_consistency: bool
    prior_points_per_image: int  # BOUND: prior samples per keyframe feeding its depth range


@dataclass(frozen=True)
class DepthConfig:
    n_views: int                # BOUND: photometric / consistency views per keyframe
    min_tri_deg: float          # view's median triangulation angle range (degrees)
    max_tri_deg: float
    n_hyp: int                  # hypotheses in inverse depth (plus the prior itself)
    beta_max: float             # BOUND: cap of the relative search half-width
    beta_quantile: float        # declared confidence: β = this |error| quantile
    patch_px: int               # ZNCC window side (odd)
    best_k: int                 # views aggregated per pixel (best k of n_views)
    propagation_iters: int      # edge-aware propagation rounds
    null_frames: int            # BOUND: keyframes the ZNCC null distribution is measured on
    null_confidence: float      # declared confidence: floor = this null quantile
    null_texture_bins: int      # texture bins of the floor
    tau_px_k: float             # τ_px = this × F5's held-out RMS
    min_consistent_views: int   # views that must confirm a tier-0 depth
    prior_fill: str             # keep | drop
    prior_fill_min_views: int   # views that must confirm a tier-1 prior
    view_samples: int           # BOUND: prior points per keyframe for view selection
    min_scale_samples: int      # BOUND: track depths that measure one keyframe's s_k
    calib_conf_bins: int        # calibration bins (equal count)
    calib_dist_bins: int
    calib_min_bin_samples: int  # BOUND: a calibration cell speaks for itself from this count
    calib_samples_per_frame: int  # BOUND: calibration samples per keyframe
    seed: int
    colmap: ColmapConfig


# ── F7: witness fusion ───────────────────────────────────────────────────

@dataclass(frozen=True)
class FuseConfig:
    min_witness_views: int      # consistent views a tier-0 point needs to enter the cloud
    voxel_m: float              # dedup cell: one point (the best evidence) per voxel
    cleaning: bool              # the CloudCompy stage's recipe (postprocessing: voxel + SOR, the
                                # same functions and parameters) on the fused cloud, inside F7


# ── F9: runner (declared in F0 so every stage heartbeats the same way) ───

@dataclass(frozen=True)
class RunnerConfig:
    heartbeat_s: float          # BOUND: seconds between progress lines of a long stage
    perf_checkpoint_s: float    # BOUND: seconds into a GPU stage at which the measured
                                # rate is compared with the expected one (abort with reason)
    python_da3: str             # the interpreter of the da3 env (F0, F2, F3 measure, F6, F7)
    python_mapanything: str     # the interpreter of the mapanything env (F4, F3 probe, F5)
    threads: int                # BOUND: fixed CPU threads of every step (determinism)


@dataclass(frozen=True)
class PrecisionConfig:
    enabled: bool
    camera: CameraConfig
    gauge: GaugeConfig
    omega: OmegaConfig
    tracks: TracksConfig
    refine: RefineConfig
    depth: DepthConfig
    fuse: FuseConfig
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
        undistort_roundtrip_ulps=_num(cam, "undistort_roundtrip_ulps", "camera", lo=1,
                                      integer=True),
    )

    g = _sub(sec, "gauge", "")
    gauge = GaugeConfig(
        # two windows chain through at least two shared frames: a window needs four
        window_frames=_num(g, "window_frames", "gauge", lo=4, integer=True),
        window_overlap_frac=_num(g, "window_overlap_frac", "gauge", lo=0.0, hi=1.0,
                                 lo_excl=True),
        process_res=(_str(g, "process_res", "gauge") if g.get("process_res") == "native"
                     else _num(g, "process_res", "gauge", lo=14, integer=True)),
        model_id=_str(g, "model_id", "gauge"),
        knot_walk_m=_num(g, "knot_walk_m", "gauge", lo=0.0, lo_excl=True),
        heldout_confidence=_num(g, "heldout_confidence", "gauge", lo=0.0, hi=1.0,
                                lo_excl=True),
        smooth_grid=tuple(_num_list(g, "smooth_grid", "gauge", lo=0.0, min_len=1)),
        huber_k=_num(g, "huber_k", "gauge", lo=0.0, lo_excl=True),
        huber_tol=_num(g, "huber_tol", "gauge", lo=0.0, lo_excl=True),
        huber_max_iter=_num(g, "huber_max_iter", "gauge", lo=1, integer=True),
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
    ), coherence_probe=CoherenceProbeConfig(
        enabled=_bool(_sub(om, "coherence_probe", "omega"), "enabled", "omega.coherence_probe"),
        lengths=tuple(int(v) for v in _num_list(_sub(om, "coherence_probe", "omega"), "lengths",
                                                 "omega.coherence_probe", lo=4, integer=True)),
        heldout_confidence=_num(_sub(om, "coherence_probe", "omega"), "heldout_confidence",
                                "omega.coherence_probe", lo=0.5, hi=1.0),
        bootstrap=_num(_sub(om, "coherence_probe", "omega"), "bootstrap", "omega.coherence_probe",
                       lo=100, integer=True),
        seed=_num(_sub(om, "coherence_probe", "omega"), "seed", "omega.coherence_probe", lo=0,
                  integer=True),
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
        tracker_weights_url=_str(tk, "tracker_weights_url", "tracks"),
        tracker_weights_sha256=_str(tk, "tracker_weights_sha256", "tracks").lower(),
    )
    if len(tracks.tracker_weights_sha256) != 64 or \
            any(c not in "0123456789abcdef" for c in tracks.tracker_weights_sha256):
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.tracker_weights_sha256' must be a sha256 (64 hex digits), got "
            f"{tracks.tracker_weights_sha256!r}")
    if tracks.window_overlap_frac >= 1.0 or tracks.heldout_frac >= 1.0:
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.window_overlap_frac' and '.heldout_frac' must be below 1")
    if tracks.dense_matcher != "none":
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.dense_matcher' = {tracks.dense_matcher!r}: no dense matcher is "
            f"vendored (claude_stac.txt §4-F4: optional, off; its licence must be checked and "
            f"recorded in vendor/VENDORS.lock.md before it can be enabled)")

    rf = _sub(sec, "refine", "")
    refine = RefineConfig(
        min_tri_deg=_num(rf, "min_tri_deg", "refine", lo=0.0, lo_excl=True),
        huber_px=_num(rf, "huber_px", "refine", lo=0.0, lo_excl=True),
        max_iterations=_num(rf, "max_iterations", "refine", lo=1, integer=True),
        ceres_threads=_num(rf, "ceres_threads", "refine", lo=1, integer=True),
        focal_block_frames=_num(rf, "focal_block_frames", "refine", lo=2, integer=True),
        heldout_confidence=_num(rf, "heldout_confidence", "refine", lo=0.0, hi=1.0, lo_excl=True),
        permutations=_num(rf, "permutations", "refine", lo=1, integer=True),
        seed=_num(rf, "seed", "refine", lo=0, integer=True),
        # PnP needs four points (P3P + one to choose)
        min_witness_corr=_num(rf, "min_witness_corr", "refine", lo=4, integer=True),
    )
    if refine.heldout_confidence >= 1.0:
        raise PrecisionConfigError(f"'{SECTION}.refine.heldout_confidence' must be below 1")
    if refine.ceres_threads != 1:
        raise PrecisionConfigError(
            f"'{SECTION}.refine.ceres_threads' = {refine.ceres_threads}: it must stay 1 — Ceres' "
            f"multi-threaded evaluation and Schur accumulation sum in a run-dependent order, so "
            f"the refined poses, camera and landmarks would not be bit-identical run to run")

    dp = _sub(sec, "depth", "")
    cm = _sub(dp, "colmap", "depth")
    depth = DepthConfig(
        n_views=_num(dp, "n_views", "depth", lo=1, integer=True),
        min_tri_deg=_num(dp, "min_tri_deg", "depth", lo=0.0),
        max_tri_deg=_num(dp, "max_tri_deg", "depth", lo=0.0, hi=180.0, lo_excl=True),
        n_hyp=_num(dp, "n_hyp", "depth", lo=3, integer=True),
        beta_max=_num(dp, "beta_max", "depth", lo=0.0, hi=1.0, lo_excl=True),
        beta_quantile=_num(dp, "beta_quantile", "depth", lo=0.0, hi=1.0, lo_excl=True),
        patch_px=_num(dp, "patch_px", "depth", lo=3, integer=True),
        best_k=_num(dp, "best_k", "depth", lo=1, integer=True),
        propagation_iters=_num(dp, "propagation_iters", "depth", lo=0, integer=True),
        null_frames=_num(dp, "null_frames", "depth", lo=1, integer=True),
        null_confidence=_num(dp, "null_confidence", "depth", lo=0.0, hi=1.0, lo_excl=True),
        null_texture_bins=_num(dp, "null_texture_bins", "depth", lo=1, integer=True),
        tau_px_k=_num(dp, "tau_px_k", "depth", lo=0.0, lo_excl=True),
        min_consistent_views=_num(dp, "min_consistent_views", "depth", lo=1, integer=True),
        prior_fill=_enum(dp, "prior_fill", "depth", PRIOR_FILL_MODES),
        prior_fill_min_views=_num(dp, "prior_fill_min_views", "depth", lo=1, integer=True),
        view_samples=_num(dp, "view_samples", "depth", lo=8, integer=True),
        min_scale_samples=_num(dp, "min_scale_samples", "depth", lo=1, integer=True),
        calib_conf_bins=_num(dp, "calib_conf_bins", "depth", lo=1, integer=True),
        calib_dist_bins=_num(dp, "calib_dist_bins", "depth", lo=1, integer=True),
        calib_min_bin_samples=_num(dp, "calib_min_bin_samples", "depth", lo=1, integer=True),
        calib_samples_per_frame=_num(dp, "calib_samples_per_frame", "depth", lo=1, integer=True),
        seed=_num(dp, "seed", "depth", lo=0, integer=True),
        colmap=ColmapConfig(
            enabled=_bool(cm, "enabled", "depth.colmap"),
            as_tier=_bool(cm, "as_tier", "depth.colmap"),
            binary=_str(cm, "binary", "depth.colmap"),
            window_radius=_num(cm, "window_radius", "depth.colmap", lo=1, integer=True),
            num_iterations=_num(cm, "num_iterations", "depth.colmap", lo=1, integer=True),
            geom_consistency=_bool(cm, "geom_consistency", "depth.colmap"),
            prior_points_per_image=_num(cm, "prior_points_per_image", "depth.colmap", lo=2,
                                        integer=True),
        ),
    )
    if depth.patch_px % 2 == 0:
        raise PrecisionConfigError(f"'{SECTION}.depth.patch_px' must be odd, got {depth.patch_px}")
    if depth.max_tri_deg <= depth.min_tri_deg:
        raise PrecisionConfigError(f"'{SECTION}.depth.max_tri_deg' must exceed min_tri_deg")
    if depth.best_k > depth.n_views:
        raise PrecisionConfigError(f"'{SECTION}.depth.best_k' ({depth.best_k}) exceeds "
                                   f"n_views ({depth.n_views})")
    if depth.beta_max >= 1.0 or depth.beta_quantile >= 1.0 or depth.null_confidence >= 1.0:
        raise PrecisionConfigError(f"'{SECTION}.depth.beta_max', '.beta_quantile' and "
                                   f"'.null_confidence' must be below 1")

    fu = _sub(sec, "fuse", "")
    fuse = FuseConfig(
        min_witness_views=_num(fu, "min_witness_views", "fuse", lo=1, integer=True),
        voxel_m=_num(fu, "voxel_m", "fuse", lo=0.0, lo_excl=True),
        cleaning=_bool(fu, "cleaning", "fuse"),
    )
    for stale in ("sor", "sor_knn", "sor_std", "sor_cell_m", "noise_filter"):
        if stale in fu:
            raise PrecisionConfigError(f"reconstruction.precision.fuse.{stale} no longer exists — the "
                              f"cleaning of the fused cloud is ONE switch, fuse.cleaning, and "
                              f"its parameters are the postprocessing: block's (USER 2026-09-29)")

    rn = _sub(sec, "runner", "")
    runner = RunnerConfig(
        heartbeat_s=_num(rn, "heartbeat_s", "runner", lo=0.0, lo_excl=True),
        perf_checkpoint_s=_num(rn, "perf_checkpoint_s", "runner", lo=0.0, lo_excl=True),
        python_da3=_str(rn, "python_da3", "runner"),
        python_mapanything=_str(rn, "python_mapanything", "runner"),
        threads=_num(rn, "threads", "runner", lo=1, integer=True),
    )

    return PrecisionConfig(enabled=enabled, camera=camera, gauge=gauge, omega=omega,
                           tracks=tracks, refine=refine, depth=depth, fuse=fuse,
                           runner=runner)
