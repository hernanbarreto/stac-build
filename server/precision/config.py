"""Typed, validated load of ``reconstruction.precision`` (server/config.yaml).

Decides nothing: it guarantees that every parameter the precision stages use
exists, has the right type and sits inside its declared bounds BEFORE any
geometry is touched. There is NO default in code — a missing key aborts the
load naming it (the YAML documents every value and its provenance: BOUND /
MEASURED / USER DECISION). Each phase of claude_stac.txt §4 adds its section
here and in config.yaml in the same commit.

WHICH CONFIGURATION (docs/plan_determinismo.md point 69, DECIDIDO USER 2026-10-08: the
precision chain takes the FROZEN configuration of the run, never the disk): every step of
the chain runs as its own process (``python -m precision.<module> --session <dir>``) and
used to call ``load_precision_config()`` with no argument — ``from config import cfg``,
config.yaml AS IT STOOD ON DISK when that process started — while the worker that launched
it held the job's dict; the steps' own ``raw:`` reads (the confidence floor, the cloud
recipe, the user's factor) came from the disk too. Edit config.yaml during a run and F0,
F2 and F6 ran on three configurations, none recorded. Now ONE accessor,
:func:`raw_config_of`, answers "the configuration of this session's job": the frozen
``output/run_config.yaml`` (``intake.run_config.load_run_config`` verifies its digest; the
stage's overrides applied) when the session holds one, else the server's configuration,
DECLARED in the log (a CLI on a session nobody froze). :func:`load_precision_config`
builds from it (``session_dir=``), keeps the dict it was built from on the result
(``PrecisionConfig.raw``, the convention of ``correction.config.CorrectionConfig.raw``),
and :func:`raw_of` is where a step reads its ``raw:`` sections from — never
``config.cfg``. The worker verifies the dict it was handed against the frozen file
(:func:`frozen_config`, the same check the certification worker runs) before anything runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SECTION = "reconstruction.precision"
# the pipeline stage the precision chain runs INSIDE (USER 2026-09-28: "f0 a f7 es etapa de
# reconstrucción"): the stage whose overrides (``_stages`` of run_config.yaml) apply to it
STAGE = "reconstruction"
LOG_TAG = "[precision.config]"


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


def _str_or_empty(section, key, path) -> str:
    v = _require(section, key, path)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise PrecisionConfigError(f"'{SECTION}.{path}.{key}' must be a string, got {v!r}")
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


def _gone(section: Any, key: str, path: str, why: str) -> None:
    """A key that was REMOVED by a decision: a leftover in config.yaml fails the load naming it
    and the decision — it would otherwise sit there looking like it still decided something."""
    if isinstance(section, dict) and key in section:
        raise PrecisionConfigError(f"'{SECTION}.{path}.{key}' no longer exists — {why}; remove it")


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
    delete_windows_after_chain: bool    # the I3 windows (F0's only reader) go when the chain is done;
                                        # run_gauge regenerates them from windows.json on a re-run


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
LOOP_BRIDGES = ("none", "roma", "mast3r")


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
    loop_bridge: str            # phase 3d: none | roma | mast3r — the wide-baseline matcher that bridges
                                # the two visits of every loop pair (precision/loop_bridge.py)
    loop_bridge_weights_url: str        # its checkpoint at a pinned revision (ignored when none)
    loop_bridge_weights_sha256: str     # and its sha256
    loop_bridge_max_matches: int        # BOUND (cost): the most confident matches verified per pair
    loop_bridge_min_inliers: int        # a pair bridges with at least this many verified matches
    loop_bridge_ransac_px: float        # the essential-matrix RANSAC threshold (native px)
    loop_bridge_ransac_prob: float      # its confidence
    loop_bridge_min_inlier_ratio: float # a pair bridges when at least this share of its verified matches are inliers


# ── F3b: the classical cross-check (COLMAP / GLOMAP) ──────────────────────

CLASSICAL_MAPPERS = ("global", "incremental")


@dataclass(frozen=True)
class ClassicalConfig:
    enabled: bool               # phase 3d: COLMAP 4.x SIFT + vocab-tree loops + global mapper on the
                                # keyframes before F4 (evaluation.colmap's own commands and binary):
                                # loop candidates for F4, the pose cross-check in F5's report
    loop_min_inliers: int       # a verified two-view geometry with at least this many inliers …
    loop_min_gap_keyframes: int # … between keyframes at least this far apart is a loop candidate
    mappers: tuple              # the sparse mappers run on the ONE database, in this order — "global" (GLOMAP,
                                # `colmap global_mapper`) and/or "incremental" (COLMAP's mapper); every one is
                                # cross-checked against F5 (refine.json classical_cross_check.models), the first
                                # named gives the cross-check's top-level numbers
    camera_prior: bool          # the session's camera (camera.json at this step: F0's) handed to COLMAP as the
                                # single camera's prior in NATIVE pixels; false = COLMAP's default focal (1.2 × the
                                # long side — on pccr 998 px against 364 measured: GLOMAP's rotation averaging broke
                                # the walk into segments turned 177° / 90° / 66°, 2026-10-31)


# ── F5: joint refinement + witness localisation ─────────────────────────

PRIOR_MODELS = ("relative", "absolute")
ROLLING_SHUTTER = ("judged", "never")


@dataclass(frozen=True)
class RefineConfig:
    enabled: bool               # false = NO bundle adjustment: the gauge's poses and F0's camera pass
                                # through unchanged (an identity F5 epoch, the witnesses localised
                                # against the initial landmarks) — the evaluation battery's variant
                                # "sin ajuste de haces F5" (claude_stac.txt 2026-10-14, rule 4: the
                                # default reproduces the chain exactly)
    min_tri_deg: float          # BOUND: a landmark's rays must span at least this (degrees)
    huber_px: float             # BOUND: Huber scale of the reprojection loss (native px)
    max_iterations: int         # BOUND: Ceres iterations per rung (reconstruction.colmap_ba)
    ceres_threads: int          # must stay 1 for bit-identical results (validated at load)
    focal_block_frames: int     # BOUND: keyframes per temporal focal block (rung R3)
    heldout_confidence: float   # declared confidence of the held-out comparisons
    permutations: int           # BOUND: permutation / bootstrap resamples of the tests
    seed: int                   # their fixed seed
    min_witness_corr: int       # BOUND: 2D-3D correspondences a witness needs for PnP
    prior_model: str            # phase 3b: "relative" (odometry priors between consecutive keyframes,
                                # the first keyframe fixed — the default) or "absolute" (the gauge
                                # priors on every keyframe's position, the pre-phase-3b model)
    prior_sigma_from_loops: bool  # phase 3b, absolute model only: the gauge priors relaxed by the factor the OPEN loops
                                # measure (loop_gap.prior_relaxation), judged by the loop gaps and the
                                # held-out tracks; false = the priors as the gauge measured them
    scale_lock: bool            # after the rungs: the metre re-read from the gauge's step lengths and
                                # re-imposed by the user's rule (precision.scale_lock)
    rolling_shutter: str        # phase 3f: "judged" — the readout measured on the held-out residuals, the
                                # observations rectified, the solve repeated and judged by the user's rule
                                # (precision.rolling_shutter); "never" — declared only (the diagnostic)
    rs_passes: int              # BOUND: refinement levels of the readout search (the grid step halves each)
    rs_max_frames: float        # BOUND: the readout searched up to this many frame periods
    rs_grid_frames: float       # BOUND: the search grid's step (frame periods)


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
    contradiction_views: int    # BOUND: the most covisible views asked "do you see through it?"
                                # — a cost cap, not a decision (all of them cost 195 min on pccr)
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
    prior_fill_min_views: int   # views that must confirm a tier-1 prior (only while
                                # prior_requires_confirmation)
    prior_requires_confirmation: bool   # false = maximum coverage: every prior pixel that is
                                        # not tier 0 is tier 1 unless contradicted / under the
                                        # confidence floor / excluded
    contradiction: bool         # the majority own-error contradiction vote (the flyer filter)
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


# ── the product: corrected Omega cloud (default) or the witness fusion ─────

DEVICES = ("cuda", "cpu")

@dataclass(frozen=True)
class CloudConfig:
    source: str                 # "omega_corrected" (F6's depth — tier 0 + tier 1 — with F5's camera +
                                # poses, the cloud stage's recipe — precision/corrected_cloud.py) |
                                # "fusion" (F6 sweep → F7)
    witness_filter: bool        # the witness filter on the corrected frames (drop_statuses leave)
    silhouette_filter: bool     # the SAM3-silhouette flyer filter (precision/silhouette_filter.py)
    silhouette_max_views: int   # BOUND (cost cap): views tested per point
    silhouette_block: int       # BOUND (device memory): (point, view) pairs projected per step
    silhouette_min_votes: int   # eligible views a point needs before its own silhouette judges it
    silhouette_min_inside_frac: float   # share of those views inside its label's dilated mask
                                        # under which it leaves
    silhouette_device: str      # cuda | cpu — where the projection runs (no fallback)
    consolidate: bool           # the scene consolidation, normals from the corrected depth
    confidence_gate: bool       # USER 2026-09-30: false = no gate of this step's own — F6 already applied
                                # the pipeline's ONE confidence floor (its DISCARD_LOW_CONF)
    repair_contradicted: bool   # USER 2026-09-30: a pixel F6 contradicted takes the depth its neighbours
                                # agree on along its ray instead of leaving (it leaves when they do not)
    repair_neighbors: tuple     # keyframe offsets whose surfaces are carried onto the pixel's ray
    repair_tau_quantile: float  # agreement tolerance = this quantile of the session's own neighbour
                                # disagreement (measured, never a fixed number)
    repair_min_views: int       # neighbours that must agree before a depth is given back


# ── depth on F5 (USER 2026-10-01: the epoch-7 recipe, precision/depth_on_f5.py) ──

EDGE_RULES = ("inverse_second_difference", "window")
COLMAP_ROWS = ("judged", "always", "never")


@dataclass(frozen=True)
class BendConfig:
    enabled: bool               # false = NO per-keyframe bend: Omega's depth stands (k = 1) on F5's
                                # poses; the vote and the cleaner still run — the evaluation battery's
                                # variant "sin refinamiento de profundidad F6" (rule 4: default = the
                                # validated recipe bit for bit)
    model: str                  # the depth model of the bend (precision/bend_model.py, audit 2026-10-23):
                                # 'judged' (both families fitted, the held-out chooses, the plane-preserving
                                # one wins a tie) | 'inverse_depth' (1/z' = a/z + b0 + b1·u' + b2·v', planes
                                # stay planes) | 'affine_pixel' (epoch 7: z' = z·(c0 + c1·u' + c2·v'), bends
                                # every oblique plane whenever c1 or c2 enters)
    windows: Tuple[int, ...]    # BOUND (search): smoothing windows (± keyframes) the held-out chooses among
    neighbors: Tuple[int, ...]  # keyframe offsets that vote on a pixel
    judges_exclude_mixed: bool  # audit 2026-10-23: a neighbour's MIXED pixel (on neither surface of a step —
                                # the vote's own definition) neither confirms nor contradicts anything; false =
                                # epoch 8's vote, where Omega's contour ramps testified against the background
                                # next to every object (halos repaired to the ramp's depth, holes)
    colmap_rows: str            # phase 3: whether the bend is fitted on COLMAP's dense tier-0 depth too — 'judged'
                                # (landmarks alone and landmarks + COLMAP rows are both fitted; the half-A held-out
                                # landmarks, which COLMAP never saw, choose; the denser support wins a tie) | 'always'
                                # | 'never' (F5's landmarks alone, the recipe before phase 3)
    edge_rule: str              # audit 2026-10-23: how a depth STEP is told from a smooth oblique surface —
                                # 'inverse_second_difference' (a step is a kink of 1/z, which is affine on a
                                # plane whatever its slant: a floor or a tunnel wall near the horizon is not an
                                # edge) | 'window' (epoch 8: the 3x3 window's range, which flags every pixel
                                # whose neighbours differ by more than tau — oblique surfaces far away included)
    tau_quantile: float         # agreement tolerance = this percentile of the session's OWN neighbour
                                # disagreement (measured every run, never a fixed number)
    irls_iterations: int        # Huber IRLS steps (pccr epoch 7's fit, USER 2026-10-01)
    min_rows: int               # fewer landmark rows in the window → k = 1 (epoch 7)
    min_depth_m: float          # landmark rows on Omega depth under this are not used (epoch 7)
    heldout_confidence: float   # declared confidence of the per-keyframe c0 verification and of the
                                # window choice's error bar (docs/plan_determinismo.md points 48 / 49)
    bootstrap: int              # BOUND (cost): resamples of those bootstraps
    seed: int                   # their fixed seed


# ── owner fusion: one point per surface element (audit 2026-10-23, precision/owner_fusion.py) ──

@dataclass(frozen=True)
class FusionConfig:
    enabled: bool               # false = every voted pixel of every keyframe is published (the layered
                                # cloud of every run before 2026-10-23; the A/B)
    covis_stride: int           # BOUND (cost): every this-th row and column of a keyframe measures its
                                # frustum overlap with the others (covisibility)
    min_covis_frac: float       # two keyframes are candidates of each other when at least this share of
                                # one's sampled points lands in the other
    max_candidates: int         # BOUND (cost, memory): the most covisible keyframes a keyframe is fused with
    evidence_cap: int           # the confirmed tier: members beyond this count no further in the owner's
                                # rank; among equally confirmed observations the nearest camera owns
    interpolate: bool           # the neighbour's depth read at the exact landing point when its 2x2 patch
                                # is one surface (within tau); at the nearest pixel otherwise
    nearest_tie_rel: float      # two cameras within this relative band of the same distance to the point
                                # are a tie (the earliest keyframe owns) — a sub-pixel landing offset never
                                # decides who is nearer
    rejected_file: bool         # the consumed observations → rejected_points.npz (reason fused, with the winner)
    colmap_tier: bool           # phase 3: COLMAP PatchMatch with F5's geometry (precision/colmap_tier.py, the chain's
                                # f6_colmap_tier step) enters the fusion as TIER 0 — the photometric reference,
                                # with its own measured tolerance; false = the learned depth alone
    colmap_min_views: int       # a COLMAP depth is tier 0 with at least this many consistent COLMAP views (and
                                # fewer contradictions than confirmations — the sweep's rule)
    colmap_max_sources: int     # BOUND (cost): source images per reference keyframe, the most covisible first
    free_space: bool            # phase 3c: an observation more covisible views see FREE SPACE through than
                                # agree with (instruments at least as good as its own) is rejected
                                # (reason contradicted) — the floater's rule, over every covisible keyframe
    min_evidence: int           # phase 3c: an owner confirmed by fewer views than this (itself included) is
                                # rejected as unverified (reason insufficient_witnesses); 1 = publish all
    free_space_factor: float    # a view sees free space through a point only when its surface lies beyond the
                                # point by more than this x the pair's tolerance (tau is the same-surface p75:
                                # one tau is noise, a floater is many); >= 1
    colmap_rows_stride: int     # BOUND (cost): every this-th tier-0 pixel is a row of the bend's fit when
                                # bend.colmap_rows takes them
    colmap_tau_quantile: float  # the instrument's tolerance = this percentile of the disagreement between
                                # COLMAP maps (measured on every run, never a declared number)


# ── flyer diagnosis (claude_stac.txt 2026-10-04, Phase 0) ──────────────────

@dataclass(frozen=True)
class FlyersConfig:
    knn_k: int                  # BOUND: neighbours of the isolation distance
    isolated_quantile: float    # isolated above this percentile of the session's OWN k-NN distance
    step_quantile: float        # a depth step = adjacent-pixel |dz|/z above this percentile of the session's steps
    band_px: int                # BOUND (resolution): half-width of the discontinuity band around a step
    view_quantile: float        # view agreement bar = this percentile of the session's neighbour disagreement
    texture_quantile: float     # low texture = gradient energy under this percentile of the session's pixels
    texture_window_px: int      # BOUND: window of the gradient energy
    specular_gray: float        # BOUND (sensor): mean gray at or above this = saturated highlight
    sample_points: int          # BOUND (cost): points sampled for the session percentiles
    seed: int


# ── mono detail: PointDiT refines the bent depth (claude_stac.txt 2026-10-04) ─────

@dataclass(frozen=True)
class MonoDetailConfig:
    enabled: bool               # feature flag — false keeps the baseline (pccr epoch 8) reproducible
    model: str                  # "H" (PointDiT-H/16 + DINOv3 ViT-H+/16, default) | "L"
    steps: int                  # Euler steps of the ODE from zeros (deterministic)
    repo_dir: str               # the vendored PointDiT checkout (relative to the repo root)
    weights_dir: str            # the released checkpoints
    dinov3_dir: str             # the gated DINOv3 encoder weights
    norm_max: float             # output norm above this = invalid pixel (the paper's sky dome sits at 3.0)
    verify_sha256: bool         # check every checkpoint against the prefix in its name (once per file)
    seed: int                   # the torch seed of repro.deterministic_torch around every inference
                                # (point 54; the ODE draws nothing — the record is complete)
    tile_px: int                # BOUND (model): the native tile side PointDiT runs at (512 = its training size)
    tile_overlap_frac: float    # share of a tile its neighbour repeats (feathered)
    context_scale: float        # 1.0 = native tile; 2.0 = a window twice as large downscaled to the tile
    fit_space: str              # 'depth' (z_cal ≈ s·z_mono + b) | 'inverse' (1/z_cal ≈ s·z_mono + b)
    irls_iterations: int        # Huber IRLS steps of the per-tile affine fit (epoch 7's construction)
    min_support_frac: float     # a tile with less support than this share of its pixels is rejected
    tile_residual_quantile: float  # a tile whose residual is above this percentile of the session's tiles is rejected
    lowpass_patch_fraction: float  # lowpass sigma = Omega's patch on the camera grid × this
    detail_scope: str           # 'edges' (detail only near discontinuities, within tau) | 'surfaces' (everywhere)
    detail_zone_px: int         # BOUND (resolution): how far from a discontinuity the detail reaches ('edges')
    side_reach_px: int          # BOUND (resolution): how far past the band the front/back surfaces are read
    overlays: bool              # per-keyframe PNG overlays (bent | aligned | detail | band + status) for the eye
    ab_edge_gradient_quantile: float  # the A/B's edge band: image gradient energy above this percentile of the
                                      # session's pixels, or a step of the CALIBRATED depth — never PointDiT's
    tile_batch: int             # tiles of a keyframe per PointDiT forward pass (0 = all of them at once — USER
                                # 2026-10-08: speed; part of the run's config, so the batch's bits repeat)
    heldout_gate: bool          # audit 2026-10-23: the refinement is JUDGED per keyframe on the half-A held-out
                                # rows it changed (depth_on_f5.mono_heldout_gate) and reverted where the user's
                                # rule says it worsens them; false = applied unjudged (the 2026-10-04 recipe)


# ── cloud metrics of every run (floor + edges; pending A "run checks", 2026-10-04) ──

@dataclass(frozen=True)
class CloudMetricsConfig:
    ransac_iterations: int      # BOUND (search) of the global floor plane
    floor_max_tilt_deg: float   # the floor normal lies within this of the cameras' up
    camera_min_height_m: float  # the cameras stand at least this high over the floor …
    camera_above_frac: float    # … for at least this share of them
    inlier_m: float             # plane inliers (RANSAC score + refit)
    band_m: float               # the floor band around the plane
    cell_m: float               # the cell of the per-cell statistics
    cell_min_points: int        # BOUND: band points a cell needs to be measured
    mode_bin_m: float           # histogram bin of the cell's dominant layer
    mode_band_m: float          # points within this of the mode fit the cell's plane
    plane_min_points: int       # BOUND: points the cell plane needs
    max_points: int             # BOUND (cost): subsample of the cloud
    score_stride: int           # BOUND (cost): every n-th point scores a RANSAC candidate
    seed: int
    edge_max_objects: int       # BOUND (work): the largest objects (by point count, ties by instance id)
                                # the edge metric measures
    edge_max_points_per_object: int   # BOUND (work): points per object the edge metric sees — a stable
                                      # per-point key sample above it (docs/plan_determinismo.md point 138:
                                      # a limit by WORK, never by clock)


# ── chunk / keyframe floor check (USER 2026-09-29: "verificación interna e intrachunk") ──

@dataclass(frozen=True)
class ChunkCheckConfig:
    low_pct: float              # percentile of a keyframe's heights that seeds its FLOOR band
    high_pct: float             # … its CEILING band
    band_m: float               # half-height of a band around that percentile (a measurement
                                # definition, like silhouette_cell_m)
    min_points: int             # BOUND: fewer band samples than this = the keyframe does not
                                # see that surface (reported as such, never guessed)
    pixel_stride: int           # BOUND (cost): every n-th pixel of the depth maps
    confidence: float           # declared confidence of the bootstrap intervals that decide
    bootstrap: int              # BOUND (cost): bootstrap resamples
    seed: int                   # determinism


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
    cloud: CloudConfig
    classical: ClassicalConfig
    bend: BendConfig
    fusion: FusionConfig
    flyers: FlyersConfig
    mono_detail: MonoDetailConfig
    cloud_metrics: CloudMetricsConfig
    chunk_check: ChunkCheckConfig
    runner: RunnerConfig
    # the WHOLE configuration dict this one was loaded from (point 69, 2026-10-08): the
    # ``raw:`` sections a step reads outside its own block — reconstruction.simple's
    # confidence floor, postprocessing's cloud recipe, correction_graph's factor, the loops
    # and correction configs — come from HERE (:func:`raw_of`), never from ``config.cfg``
    # re-read from disk by a step process. Not compared: two configs with the same sections
    # are the same configuration
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)


def raw_config_of(session_dir: os.PathLike, *, stage: str = STAGE,
                  log: Callable[[str], Any] = print) -> Tuple[Dict[str, Any], Optional[str]]:
    """THE configuration of this session's job (point 69): (the raw dict, its sha256).

    The frozen ``output/run_config.yaml`` when the session holds one — read through
    ``intake.run_config.load_run_config`` (the digest file must match: an edited copy is
    REFUSED naming it) with the ``stage``'s overrides applied (``effective_config``) — else the
    server's ``config.cfg``, DECLARED in the log with sha256 None: that is a command-line step
    on a session nobody froze (the chain CLI, ``precision.runner``, freezes its own first, as
    the intake and certification CLIs do). Never ``config.cfg`` while a frozen file exists."""
    from intake.run_config import (OUTPUT_DIRNAME, RUN_CONFIG_NAME, effective_config,
                                   has_run_config, load_run_config)
    sd = Path(session_dir)
    if has_run_config(sd):
        frozen, sha = load_run_config(sd)
        log(f"{LOG_TAG} configuration: the job's frozen {OUTPUT_DIRNAME}/{RUN_CONFIG_NAME} "
            f"(sha256 {sha[:12]}; stage {stage!r} overrides applied)")
        return effective_config(frozen, stage), sha
    from config import cfg as raw_cfg                     # server/config.py, read ONCE here
    log(f"{LOG_TAG} DECLARED: {sd / OUTPUT_DIRNAME / RUN_CONFIG_NAME} does not exist — no job "
        f"froze this session's configuration; the server's config.yaml is read (a CLI on an "
        f"unfrozen session; point 69)")
    return raw_cfg, None


def frozen_config(session_dir: os.PathLike, handed: Dict[str, Any], *, stage: str = STAGE,
                  log: Callable[[str], Any] = print) -> Tuple[Dict[str, Any], str]:
    """(the configuration a stage worker runs with, its sha256): the job's frozen
    ``output/run_config.yaml`` with the ``stage``'s overrides, verified against the dict the
    pipeline manager handed the worker (the same job's) — a difference FAILS naming the first
    key (two configurations in one run is what points 69 / 139 end). The same check
    ``workers.certify_worker._frozen_config`` runs for the certification; the precision worker
    runs it before the chain starts, so the steps' processes (which read the frozen file through
    :func:`raw_config_of`) and the worker that launched them hold ONE configuration."""
    from intake.run_config import canonical_yaml, effective_config, load_run_config
    frozen, sha = load_run_config(session_dir)
    raw = effective_config(frozen, stage)
    mine = {k: v for k, v in dict(handed).items() if not str(k).startswith("_")}
    for key in sorted(set(mine) | set(raw)):
        a, b = mine.get(key), raw.get(key)
        if canonical_yaml({key: a}) != canonical_yaml({key: b}):
            raise PrecisionConfigError(
                f"the configuration handed to the precision core differs from the job's frozen "
                f"output/run_config.yaml at '{key}' — a step never runs on a configuration nobody "
                f"froze (points 69 / 139)")
    log(f"{LOG_TAG} configuration: the job's frozen run_config.yaml (sha256 {sha[:12]}), verified "
        f"against the dict the manager handed this worker")
    return raw, sha


_RAW_FALLBACK_DECLARED = False       # the DECLARED line of raw_of's fallback, once per process


def raw_of(pcfg: PrecisionConfig, log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """The configuration dict ``pcfg`` was loaded from — where a step reads every ``raw:``
    section (point 69). A PrecisionConfig built by hand without one (a test's) completes itself
    from the server's ``config.cfg``, as ``correction.config.precision_raw`` does — DECLARED in
    the log the first time it happens in a process: on the pipeline path (the worker's dict, a
    step's ``session_dir=``) ``raw`` is always set, so that line in a step's log is the one
    sign a step read the disk again, and a regression is visible instead of silent."""
    global _RAW_FALLBACK_DECLARED
    raw = getattr(pcfg, "raw", None)
    if isinstance(raw, dict) and isinstance(raw.get("reconstruction"), dict) \
            and "precision" in raw["reconstruction"]:
        return raw
    from config import cfg as raw_cfg
    if not _RAW_FALLBACK_DECLARED:
        _RAW_FALLBACK_DECLARED = True
        log(f"{LOG_TAG} DECLARED: PrecisionConfig carries no configuration dict — the server's "
            f"config.cfg is read (a config built by hand; never the pipeline's path, whose steps "
            f"load the job's frozen run_config.yaml — point 69)")
    return raw_cfg


def load_precision_config(raw: Optional[Dict[str, Any]] = None, *,
                          session_dir: Optional[os.PathLike] = None, stage: str = STAGE,
                          log: Callable[[str], Any] = print) -> PrecisionConfig:
    """Validated PrecisionConfig from ``raw``; with ``session_dir`` and no ``raw``, from the
    session's job configuration (:func:`raw_config_of`: the frozen run_config.yaml, else the
    server's, declared) — what every step CLI passes (``--session``). With neither: the
    server-wide ``config.cfg`` (tests, tools that hold no session) — DECLARED here: that path
    is never the pipeline's. The dict read is kept on the result (``raw``). Raises
    PrecisionConfigError naming the first offending key."""
    if raw is None:
        if session_dir is not None:
            raw, _sha = raw_config_of(session_dir, stage=stage, log=log)
        else:
            from config import cfg as raw_cfg     # server/config.py: no session, no job
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
        delete_windows_after_chain=_bool(g, "delete_windows_after_chain", "gauge"),
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
    if "coherence_probe" in om:
        raise PrecisionConfigError(
            f"'{SECTION}.omega.coherence_probe' was DELETED 2026-09-30 (USER: Omega drifts; the "
            f"chunks are planned before Omega — since 2026-10-06 by co-visibility) — remove the key")

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
        loop_bridge=_enum(tk, "loop_bridge", "tracks", LOOP_BRIDGES),
        loop_bridge_weights_url=_str_or_empty(tk, "loop_bridge_weights_url", "tracks"),
        loop_bridge_weights_sha256=_str_or_empty(tk, "loop_bridge_weights_sha256", "tracks").lower(),
        loop_bridge_max_matches=_num(tk, "loop_bridge_max_matches", "tracks", lo=8, integer=True),
        loop_bridge_min_inliers=_num(tk, "loop_bridge_min_inliers", "tracks", lo=8, integer=True),
        loop_bridge_ransac_px=_num(tk, "loop_bridge_ransac_px", "tracks", lo=0.0, lo_excl=True),
        loop_bridge_ransac_prob=_num(tk, "loop_bridge_ransac_prob", "tracks", lo=0.0, hi=1.0, lo_excl=True),
        loop_bridge_min_inlier_ratio=_num(tk, "loop_bridge_min_inlier_ratio", "tracks", lo=0.0, hi=1.0),
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
    lb_sha = (tracks.loop_bridge_weights_sha256 or "").lower()
    if tracks.loop_bridge != "none" and not tracks.loop_bridge_weights_url:
        raise PrecisionConfigError(f"'{SECTION}.tracks.loop_bridge_weights_url' is empty with the loop bridge on")
    if tracks.loop_bridge != "none" and (len(lb_sha) != 64 or any(c not in "0123456789abcdef" for c in lb_sha)):
        raise PrecisionConfigError(
            f"'{SECTION}.tracks.loop_bridge_weights_sha256' must be a sha256 (64 hex digits) when the loop "
            f"bridge is on, got {tracks.loop_bridge_weights_sha256!r}")

    cl = _sub(sec, "classical", "")
    classical = ClassicalConfig(enabled=_bool(cl, "enabled", "classical"),
                                loop_min_inliers=_num(cl, "loop_min_inliers", "classical", lo=8, integer=True),
                                loop_min_gap_keyframes=_num(cl, "loop_min_gap_keyframes", "classical", lo=2, integer=True),
                                mappers=tuple(_str_list(cl, "mappers", "classical", allowed=CLASSICAL_MAPPERS, min_len=1)),
                                camera_prior=_bool(cl, "camera_prior", "classical"))
    rf = _sub(sec, "refine", "")
    refine = RefineConfig(
        enabled=_bool(rf, "enabled", "refine"),
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
        prior_model=_enum(rf, "prior_model", "refine", PRIOR_MODELS),
        prior_sigma_from_loops=_bool(rf, "prior_sigma_from_loops", "refine"),
        scale_lock=_bool(rf, "scale_lock", "refine"),
        rolling_shutter=_enum(rf, "rolling_shutter", "refine", ROLLING_SHUTTER),
        rs_passes=_num(rf, "rs_passes", "refine", lo=0, integer=True),
        rs_max_frames=_num(rf, "rs_max_frames", "refine", lo=0.0, lo_excl=True),
        rs_grid_frames=_num(rf, "rs_grid_frames", "refine", lo=0.0, lo_excl=True),
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
        contradiction_views=_num(dp, "contradiction_views", "depth", lo=1, integer=True),
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
        prior_requires_confirmation=_bool(dp, "prior_requires_confirmation", "depth"),
        contradiction=_bool(dp, "contradiction", "depth"),
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

    cl = _sub(sec, "cloud", "")
    src = _require(cl, "source", "cloud")
    if src not in ("omega_bent", "omega_corrected", "fusion"):
        raise PrecisionConfigError(f"'{SECTION}.cloud.source' must be omega_bent | omega_corrected | fusion, "
                                   f"got {src!r}")
    cloud = CloudConfig(source=str(src), witness_filter=_bool(cl, "witness_filter", "cloud"),
                        silhouette_filter=_bool(cl, "silhouette_filter", "cloud"),
                        silhouette_max_views=_num(cl, "silhouette_max_views", "cloud", lo=1, integer=True),
                        silhouette_block=_num(cl, "silhouette_block", "cloud", lo=1, integer=True),
                        silhouette_min_votes=_num(cl, "silhouette_min_votes", "cloud", lo=1, integer=True),
                        silhouette_min_inside_frac=_num(cl, "silhouette_min_inside_frac", "cloud",
                                                        lo=0.0, hi=1.0, lo_excl=True),
                        silhouette_device=_enum(cl, "silhouette_device", "cloud", DEVICES),
                        consolidate=_bool(cl, "consolidate", "cloud"),
                        confidence_gate=_bool(cl, "confidence_gate", "cloud"),
                        repair_contradicted=_bool(cl, "repair_contradicted", "cloud"),
                        repair_neighbors=tuple(int(x) for x in _require(cl, "repair_neighbors", "cloud")),
                        repair_tau_quantile=_num(cl, "repair_tau_quantile", "cloud", lo=0.0, hi=100.0,
                                                 lo_excl=True),
                        repair_min_views=_num(cl, "repair_min_views", "cloud", lo=1, integer=True))
    cc = _sub(sec, "chunk_check", "")
    _gone(cc, "plane_min_inlier_frac", "chunk_check",
          "USER 2026-10-07 (docs/plan_determinismo.md point 140): no acceptance bar on the floor plane — "
          "it is always fitted and reported with its inlier fraction and interval")
    chunk_check = ChunkCheckConfig(
        low_pct=_num(cc, "low_pct", "chunk_check", lo=0.0, hi=50.0),
        high_pct=_num(cc, "high_pct", "chunk_check", lo=50.0, hi=100.0),
        band_m=_num(cc, "band_m", "chunk_check", lo=0.0, lo_excl=True),
        min_points=_num(cc, "min_points", "chunk_check", lo=1, integer=True),
        pixel_stride=_num(cc, "pixel_stride", "chunk_check", lo=1, integer=True),
        confidence=_num(cc, "confidence", "chunk_check", lo=0.5, hi=1.0),
        bootstrap=_num(cc, "bootstrap", "chunk_check", lo=10, integer=True),
        seed=_num(cc, "seed", "chunk_check", lo=0, integer=True),
    )
    bd = _sub(sec, "bend", "")
    windows = tuple(int(x) for x in _require(bd, "windows", "bend"))
    neighbors = tuple(int(x) for x in _require(bd, "neighbors", "bend"))
    if not windows or min(windows) < 0:
        raise PrecisionConfigError(f"'{SECTION}.bend.windows' must be non-negative keyframe counts, got {windows}")
    if not neighbors or 0 in neighbors:
        raise PrecisionConfigError(f"'{SECTION}.bend.neighbors' must be non-zero keyframe offsets, got {neighbors}")
    from precision.bend_model import MODEL_CHOICES
    bend_model = str(_require(bd, "model", "bend"))
    if bend_model not in MODEL_CHOICES:
        raise PrecisionConfigError(f"'{SECTION}.bend.model' must be one of {MODEL_CHOICES}, got {bend_model!r}")
    colmap_rows = str(_require(bd, "colmap_rows", "bend"))
    if colmap_rows not in COLMAP_ROWS:
        raise PrecisionConfigError(f"'{SECTION}.bend.colmap_rows' must be one of {COLMAP_ROWS}, got {colmap_rows!r}")
    edge_rule = str(_require(bd, "edge_rule", "bend"))
    if edge_rule not in EDGE_RULES:
        raise PrecisionConfigError(f"'{SECTION}.bend.edge_rule' must be one of {EDGE_RULES}, got {edge_rule!r}")
    bend = BendConfig(enabled=_bool(bd, "enabled", "bend"), model=bend_model, windows=windows, neighbors=neighbors,
                      judges_exclude_mixed=_bool(bd, "judges_exclude_mixed", "bend"), edge_rule=edge_rule,
                      colmap_rows=colmap_rows,
                      tau_quantile=_num(bd, "tau_quantile", "bend", lo=0.0, hi=100.0, lo_excl=True),
                      irls_iterations=_num(bd, "irls_iterations", "bend", lo=1, integer=True),
                      min_rows=_num(bd, "min_rows", "bend", lo=3, integer=True),
                      min_depth_m=_num(bd, "min_depth_m", "bend", lo=0.0),
                      heldout_confidence=_num(bd, "heldout_confidence", "bend", lo=0.0, hi=1.0, lo_excl=True),
                      bootstrap=_num(bd, "bootstrap", "bend", lo=1, integer=True),
                      seed=_num(bd, "seed", "bend", lo=0, integer=True))
    if bend.heldout_confidence >= 1.0:
        raise PrecisionConfigError(f"'{SECTION}.bend.heldout_confidence' must be below 1")
    fu = _sub(sec, "fusion", "")
    fusion = FusionConfig(enabled=_bool(fu, "enabled", "fusion"),
                          covis_stride=_num(fu, "covis_stride", "fusion", lo=1, integer=True),
                          min_covis_frac=_num(fu, "min_covis_frac", "fusion", lo=0.0, hi=1.0),
                          max_candidates=_num(fu, "max_candidates", "fusion", lo=1, integer=True),
                          evidence_cap=_num(fu, "evidence_cap", "fusion", lo=0, integer=True),
                          interpolate=_bool(fu, "interpolate", "fusion"),
                          nearest_tie_rel=_num(fu, "nearest_tie_rel", "fusion", lo=0.0, hi=0.5),
                          rejected_file=_bool(fu, "rejected_file", "fusion"),
                          free_space=_bool(fu, "free_space", "fusion"),
                          min_evidence=_num(fu, "min_evidence", "fusion", lo=1, integer=True),
                          free_space_factor=_num(fu, "free_space_factor", "fusion", lo=1.0),
                          colmap_tier=_bool(fu, "colmap_tier", "fusion"),
                          colmap_min_views=_num(fu, "colmap_min_views", "fusion", lo=1, integer=True),
                          colmap_max_sources=_num(fu, "colmap_max_sources", "fusion", lo=1, integer=True),
                          colmap_rows_stride=_num(fu, "colmap_rows_stride", "fusion", lo=1, integer=True),
                          colmap_tau_quantile=_num(fu, "colmap_tau_quantile", "fusion", lo=0.0, hi=100.0, lo_excl=True))
    fl = _sub(sec, "flyers", "")
    flyers = FlyersConfig(knn_k=_num(fl, "knn_k", "flyers", lo=1, integer=True),
                          isolated_quantile=_num(fl, "isolated_quantile", "flyers", lo=0.0, hi=100.0, lo_excl=True),
                          step_quantile=_num(fl, "step_quantile", "flyers", lo=0.0, hi=100.0, lo_excl=True),
                          band_px=_num(fl, "band_px", "flyers", lo=0, integer=True),
                          view_quantile=_num(fl, "view_quantile", "flyers", lo=0.0, hi=100.0, lo_excl=True),
                          texture_quantile=_num(fl, "texture_quantile", "flyers", lo=0.0, hi=100.0, lo_excl=True),
                          texture_window_px=_num(fl, "texture_window_px", "flyers", lo=1, integer=True),
                          specular_gray=_num(fl, "specular_gray", "flyers", lo=0.0, hi=255.0),
                          sample_points=_num(fl, "sample_points", "flyers", lo=1, integer=True),
                          seed=_num(fl, "seed", "flyers", lo=0, integer=True))
    md = _sub(sec, "mono_detail", "")
    model = str(_require(md, "model", "mono_detail"))
    if model not in ("H", "L"):
        raise PrecisionConfigError(f"'{SECTION}.mono_detail.model' must be 'H' or 'L', got {model!r}")
    fit_space = str(_require(md, "fit_space", "mono_detail"))
    detail_scope = str(_require(md, "detail_scope", "mono_detail"))
    if detail_scope not in ("edges", "surfaces"):
        raise PrecisionConfigError(f"'{SECTION}.mono_detail.detail_scope' must be 'edges' or 'surfaces', got {detail_scope!r}")
    if fit_space not in ("depth", "inverse"):
        raise PrecisionConfigError(f"'{SECTION}.mono_detail.fit_space' must be 'depth' or 'inverse', got {fit_space!r}")
    mono_detail = MonoDetailConfig(enabled=_bool(md, "enabled", "mono_detail"), model=model,
                                   steps=_num(md, "steps", "mono_detail", lo=1, integer=True),
                                   repo_dir=str(_require(md, "repo_dir", "mono_detail")),
                                   weights_dir=str(_require(md, "weights_dir", "mono_detail")),
                                   dinov3_dir=str(_require(md, "dinov3_dir", "mono_detail")),
                                   norm_max=_num(md, "norm_max", "mono_detail", lo=0.0, lo_excl=True),
                                   verify_sha256=_bool(md, "verify_sha256", "mono_detail"),
                                   seed=_num(md, "seed", "mono_detail", lo=0, integer=True),
                                   tile_px=_num(md, "tile_px", "mono_detail", lo=16, integer=True),
                                   tile_overlap_frac=_num(md, "tile_overlap_frac", "mono_detail", lo=0.0, hi=0.95),
                                   context_scale=_num(md, "context_scale", "mono_detail", lo=1.0),
                                   fit_space=fit_space,
                                   irls_iterations=_num(md, "irls_iterations", "mono_detail", lo=1, integer=True),
                                   min_support_frac=_num(md, "min_support_frac", "mono_detail", lo=0.0, hi=1.0),
                                   tile_residual_quantile=_num(md, "tile_residual_quantile", "mono_detail", lo=0.0, hi=100.0, lo_excl=True),
                                   lowpass_patch_fraction=_num(md, "lowpass_patch_fraction", "mono_detail", lo=0.0, lo_excl=True),
                                   detail_scope=detail_scope,
                                   detail_zone_px=_num(md, "detail_zone_px", "mono_detail", lo=1, integer=True),
                                   side_reach_px=_num(md, "side_reach_px", "mono_detail", lo=1, integer=True),
                                   overlays=_bool(md, "overlays", "mono_detail"),
                                   ab_edge_gradient_quantile=_num(md, "ab_edge_gradient_quantile", "mono_detail", lo=0.0, hi=100.0, lo_excl=True),
                                   tile_batch=_num(md, "tile_batch", "mono_detail", lo=0, integer=True),
                                   heldout_gate=_bool(md, "heldout_gate", "mono_detail"))
    cm = _sub(sec, "cloud_metrics", "")
    cloud_metrics = CloudMetricsConfig(
        ransac_iterations=_num(cm, "ransac_iterations", "cloud_metrics", lo=1, integer=True),
        floor_max_tilt_deg=_num(cm, "floor_max_tilt_deg", "cloud_metrics", lo=0.0, hi=90.0),
        camera_min_height_m=_num(cm, "camera_min_height_m", "cloud_metrics", lo=0.0),
        camera_above_frac=_num(cm, "camera_above_frac", "cloud_metrics", lo=0.0, hi=1.0),
        inlier_m=_num(cm, "inlier_m", "cloud_metrics", lo=0.0, lo_excl=True),
        band_m=_num(cm, "band_m", "cloud_metrics", lo=0.0, lo_excl=True),
        cell_m=_num(cm, "cell_m", "cloud_metrics", lo=0.0, lo_excl=True),
        cell_min_points=_num(cm, "cell_min_points", "cloud_metrics", lo=3, integer=True),
        mode_bin_m=_num(cm, "mode_bin_m", "cloud_metrics", lo=0.0, lo_excl=True),
        mode_band_m=_num(cm, "mode_band_m", "cloud_metrics", lo=0.0, lo_excl=True),
        plane_min_points=_num(cm, "plane_min_points", "cloud_metrics", lo=3, integer=True),
        max_points=_num(cm, "max_points", "cloud_metrics", lo=1000, integer=True),
        score_stride=_num(cm, "score_stride", "cloud_metrics", lo=1, integer=True),
        seed=_num(cm, "seed", "cloud_metrics", lo=0, integer=True),
        edge_max_objects=_num(cm, "edge_max_objects", "cloud_metrics", lo=0, integer=True),
        edge_max_points_per_object=_num(cm, "edge_max_points_per_object", "cloud_metrics", lo=1, integer=True))
    _gone(cm, "edge_timeout_s", "cloud_metrics",
          "docs/plan_determinismo.md point 138 (2026-10-07): the edge metric is bounded by WORK "
          "(edge_max_objects x edge_max_points_per_object), never by a wall clock")
    return PrecisionConfig(enabled=enabled, camera=camera, gauge=gauge, omega=omega,
                           tracks=tracks, refine=refine, depth=depth, fuse=fuse,
                           cloud=cloud, bend=bend, fusion=fusion, flyers=flyers, mono_detail=mono_detail,
                           cloud_metrics=cloud_metrics, chunk_check=chunk_check, runner=runner,
                           classical=classical, raw=(raw if isinstance(raw, dict) else None))
