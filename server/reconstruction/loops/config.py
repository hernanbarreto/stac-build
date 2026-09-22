"""Typed, validated load of the ``loops:``, ``scale:`` and ``correction_graph:``
sections of server/config.yaml (claude_stac.txt §13: "todo en config.yaml con
validación; cero literales").

What it decides: nothing. It guarantees every decision parameter exists, has
the right type and sits in a sane range BEFORE any candidate is judged. No
default value lives in code: a missing key aborts naming it (the YAML
documents the defaults and their provenance). ``fork_model_loops`` /
``fork_model_scale`` flatten the validated sections into the plain dicts the
VGGT-Long fork consumes as ``Model.loops`` / ``Model.scale``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


class LoopsConfigError(RuntimeError):
    """Missing / out-of-range key in loops:/scale:/correction_graph:. The
    message always names the offending key."""


# ── validation primitives (same discipline as correction/config.py) ───────

def _require(section: Optional[Dict[str, Any]], key: str, path: str) -> Any:
    if section is None or key not in section:
        raise LoopsConfigError(f"config key {path}.{key} is missing")
    return section[key]


def _num(section, key, path, lo=None, hi=None, integer=False, lo_excl=False):
    v = _require(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise LoopsConfigError(f"config key {path}.{key} must be a number, got {v!r}")
    if integer and int(v) != v:
        raise LoopsConfigError(f"config key {path}.{key} must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise LoopsConfigError(f"config key {path}.{key}={v} is below the minimum {lo}")
    if hi is not None and v > hi:
        raise LoopsConfigError(f"config key {path}.{key}={v} is above the maximum {hi}")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _require(section, key, path)
    if not isinstance(v, bool):
        raise LoopsConfigError(f"config key {path}.{key} must be true/false, got {v!r}")
    return v


def _str_list(section, key, path) -> Tuple[str, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) for x in v):
        raise LoopsConfigError(f"config key {path}.{key} must be a list of strings")
    return tuple(str(x).lower() for x in v)


def _choice(section, key, path, allowed: Tuple[str, ...]) -> str:
    v = _require(section, key, path)
    if not isinstance(v, str) or v not in allowed:
        raise LoopsConfigError(f"config key {path}.{key} must be one of {'|'.join(allowed)}, got {v!r}")
    return v


# ── correction_graph.loop — bridge measurement + verification (§4.1/§4.2) ─

@dataclass(frozen=True)
class LoopEdgeConfig:
    anchors_per_bridge: int         # DA3 anchor frames planned inside each bridge window
    max_edge_sigma_m: float         # an edge with σ above this never enters the optimizer
    max_residual_m: float           # geometric acceptance: exact-fit median residual
    min_correspondences: int        # geometric acceptance: exact correspondences per side
    corr_per_frame: int             # exact correspondences kept per shared frame
    fit_sample: int                 # IRLS sample per fit
    scale_tol_log: float            # |log s_ab| beyond this → scale_break (kept, σ inflated)
    scale_break_sigma_factor: float
    starved_sigma_m: float          # σ of the vendor coarse-fit fallback (recorded, low confidence)
    ambiguous_sigma_factor: float   # σ inflation for a spatially ambiguous candidate
    movable_labels: Tuple[str, ...]  # labels that never count for/against a semantic match
    min_shared_structural_labels: int
    intra_chunk_loops: bool         # keep candidates inside one chunk (pose edges, no scale row)


# ── loops — detector + spatial gate (§4.4/§4.5/§4.9) ───────────────────────

@dataclass(frozen=True)
class SpatialGateConfig:
    min_walk_m: float               # a pair whose keyframes are closer than this ALONG THE WALK
                                    # is odometry, not a revisit: the camera never left, so
                                    # "returning" observes nothing the chain does not already
                                    # know. Rejected before any frustum work
    drift_floor_m: float
    drift_rate_m_per_m: float
    frustum_tolerance_factor: float  # factor × δ(L) the frustum rule tolerates
    drift_floor_deg: float
    drift_rate_deg_per_m: float
    frustum_margin_px: float
    occlusion_tol_m: float          # behind the frame's measured surface by more than δ + this → occluded
    min_depth_m: float
    max_depth_m: float
    min_frustum_frames: int
    frustum_window_kf: int          # ± keyframes around i / j tested for visibility
    frustum_points: int             # cluster/frame points projected per test
    min_visible_frac: float         # fraction of points inside the (widened) frustum
    size_tol: float                 # relative tolerance between the two clusters' model dims
    repetitive_labels: Tuple[str, ...]
    min_context_instances: int
    corridor_width_m: float         # lateral extent below which the walk is a corridor
    corridor_min_frustum_frames: int  # stricter reciprocity demanded inside a corridor
    dims_pct_lo: float              # supported-extent percentile band for the size rule
    dims_pct_hi: float
    same_surface_angle_deg: float   # two clusters on one plane/axis (normals or axes within
    same_surface_offset_m: float    # this angle, offset within this) are ONE object
    same_surface_planar_ratio: float  # λ3/λ1 below this → planar cluster
    same_surface_axis_ratio: float    # λ2/λ1 below this → elongated cluster


@dataclass(frozen=True)
class SemanticClassConfig:
    enabled: bool
    max_tokens: int
    crops_per_instance: int
    default_class: str              # class when the VLM is unavailable (recorded, never silent)
    # The class NEVER discards a candidate (USER 2026-09-13: "nunca debe
    # descartarse un duplicado detectado por SAM3"): geometry (the spatial
    # gate) decides loop|ambiguous|split; the class is recorded with the
    # candidate and only sets the σ inflation below for non-structural ones.
    nonstructural_sigma_factor: float   # σ inflation of a loop proposed by a movable/dynamic instance (≥ 1)


@dataclass(frozen=True)
class SaladConfig:
    """§4.4 visual candidates: the DINOv2-SALAD retrieval the fork runs over the
    keyframes (written into the Omega config's ``Loop.SALAD``)."""
    similarity_threshold: float     # cosine similarity a keyframe pair needs to be proposed
    top_k: int                      # retrieved neighbours per keyframe
    min_gap_keyframes: int          # floor: pairs closer than this (in keyframes) are odometry
    min_gap_frac: float             # and so are pairs closer than this FRACTION of the keyframe
                                    # count — the detector ranks its top-k among the frames
                                    # outside the band, so a band that is too narrow spends
                                    # every slot on near-odometry neighbours. The fork takes
                                    # max(min_gap_keyframes, ceil(min_gap_frac × n_keyframes))
    nms_threshold: int              # keyframes suppressed around an accepted pair (0 = off)
    image_size: Tuple[int, int]     # SALAD input (h, w)
    batch_size: int


@dataclass(frozen=True)
class ReprojectionConfig:
    """§4.5 — the frames arbitrate a `split`. Project one copy of an instance
    into the keyframes where the other was observed and compare with the mask
    there: the same object displaced by drift lands under a single consistent
    rigid shift, two objects under none."""
    enabled: bool
    dilate_px: int                  # the projection is a point set, the mask a filled
                                    # region: close the gaps so the overlap is between AREAS
    max_frames: int                 # keyframes tested per direction
    min_self_recall: float          # a copy must reach THIS much of its own mask in its own
                                    # frames or no verdict is trustworthy (declared unusable)
    min_cross_recall: float         # ...and this much of the other's, after the shift
    min_agreeing_frac: float        # ...in at least this fraction of the frames tested; the
                                    # scatter of the shift VECTORS ranks the real cases backwards


@dataclass(frozen=True)
class LoopsConfig:
    reprojection: "ReprojectionConfig"
    min_gap_keyframes: int          # two windows of one instance must be this far apart
    duplicate_min_sep_m: float      # centroid separation of two 3-D clusters = duplicate
    dbscan_eps_m: float
    dbscan_min_samples: int
    cluster_min_points: int
    bridge_extra_frames: int        # §4.9 non-keyframe frames per bridge window
    coverage_radius_m: float        # §4.6a: trajectory within this of a loop endpoint = covered
    min_coverage: float             # below this the acta declares the loop density insufficient
    spatial: SpatialGateConfig
    semantic: SemanticClassConfig
    salad: SaladConfig


# ── scale — the scale graph's extra sources (§5) ────────────────────────────

@dataclass(frozen=True)
class ScaleConfig:
    sigma_loop: float               # σ of a loop row (log units)
    sigma_vio: float                # σ of a VIO absolute row (log units)
    sigma_regulated: float          # σ of a regulated-dimension row (log units)
    verify_max_dev: float           # |s_align − 1| beyond this = session FAILURE (§5.4)
    vio_segment_s: float            # VIO↔chunk ratio segment length (s)
    vio_min_segments_chunk: int     # per-chunk minimum voting segments for a VIO row
    vio_min_seg_disp_m: float       # a segment votes when VIO walked at least this
    break_localisation_gap: float   # Δχ² between the best and second leave-one-seam-out
                                    # candidate below which a scale break is NOT localised


# ── correction_graph.graph — keyframe SE(3) graph (§4.3) ─────────────────

@dataclass(frozen=True)
class GraphConfig:
    sigma_odo_intra_m: float        # odometry σ between consecutive keyframes inside a chunk
    sigma_odo_intra_deg: float
    loop_sigma_rot_deg: float       # rotation σ of a loop edge (its translation σ is measured)
    odo_sigma_from_drift: bool      # derive the odometry σ from the MEASURED drift rate
    odo_sigma_min_m: float          # floor / ceiling of the derived odometry σ
    odo_sigma_max_m: float
    outlier_overlap_frac: float     # loops sharing this much of a stretch measure the same drift
    outlier_mad_k: float            # deviation, in MADs, above which a loop's σ is inflated
    outlier_mad_floor_m_per_m: float  # MAD floor so a unanimous set does not divide by zero
    outlier_max_sigma_factor: float # cap of that inflation (the edge is never dropped)
    drift_model: bool               # fit the accumulated drift as a smooth function of the walk
    drift_degree: int               # terms of that model (1 = the ramp E(d)=eps*d, 2 adds curvature)
    drift_iters: int
    drift_max_step: float           # cap of one Gauss-Newton step, in the coefficient norm
    drift_prior_rot_deg: float      # σ of the prior on each coefficient's rotation half
    drift_prior_trans_m: float      # ...and its translation half: the unobserved directions
                                    # stay at zero instead of walking the flat valley
    drift_rel_tol: float            # a step buying less than this fraction of the cost is the
                                    # valley, not convergence — stop
    drift_min_gain: float           # the model must explain this share of what the graph left
                                    # open, or it is fitting noise and is not applied
    huber_delta_m: float            # Huber on loop + structural edges (never odometry)
    huber_delta_deg: float
    dense_max_unknowns: int         # 6·n_kf ≤ this → dense Cholesky, else block-Jacobi PCG
    lambda_init: float
    lambda_max: float
    lm_diag_floor: float
    tol: float
    rel_tol: float
    max_iters: int
    pcg_tol: float
    pcg_max_iters: int
    min_loop_gain: float            # gate: total loop residual must drop by this fraction
    max_seam_degradation_m: float   # gate: held-out surface pairs may not worsen beyond this
    gate_mode: str                  # advisory | veto — advisory: the gates (gain, held-out,
                                    # authority) are measured and declared, the closure is
                                    # APPLIED (USER 2026-09-13); veto: a failed gate → identity
    holdout_offsets: Tuple[int, ...]  # frame offsets of the held-out pairs (chunk_field_verdict)
    holdout_stride: int
    holdout_samples: int
    holdout_max_nn_m: float         # cloud held-out judge: a NN pair beyond this is not a correspondence


# ── authority (§4.7) ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuthorityConfig:
    saturation_warn: float          # fraction of the declared authority that raises the flag
    pose_graph_max_m: float
    pose_graph_max_deg: float
    scale_graph_max_log: float
    depth_graph_max_log_a: float
    depth_graph_max_b_m: float
    intra_chunk_max_m: float


# ── structural (§4.6) ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class WallPlanarityConfig:
    enabled: bool
    wall_tol_m: float               # tolerated residual of a wall patch to its plane (Huber δ)
    sigma_angle_deg: float
    sigma_offset_m: float
    min_span_m: float               # only walls this long constrain the lateral bend
    min_points_per_kf: int
    labels: Tuple[str, ...]
    reference_span_kf: int          # the wall plane NODE starts at the plane of its earliest N patches
    planar_ratio: float             # λ3/λ1 of a keyframe patch above this → not a plane, no edge
    min_patch_extent_m: float       # a patch narrower than this (2nd axis) is a sliver, no edge


@dataclass(frozen=True)
class ColumnVerticalConfig:
    enabled: bool
    sigma_deg: float
    labels: Tuple[str, ...]
    min_points_per_kf: int
    axis_ratio: float               # λ2/λ1 below this → the patch has an axis


@dataclass(frozen=True)
class RepeatedParallelConfig:
    enabled: bool
    sigma_deg: float
    labels: Tuple[str, ...]
    min_points_per_kf: int
    axis_ratio: float               # λ2/λ1 below this → the patch has an axis (a partial face has none)


@dataclass(frozen=True)
class RegulatedDim:
    label: str
    dimension: str                  # width | diameter | height | gauge
    value_m: float
    tol_m: float


@dataclass(frozen=True)
class StructuralConfig:
    wall_planarity: WallPlanarityConfig
    column_vertical: ColumnVerticalConfig
    repeated_parallel: RepeatedParallelConfig
    regulated_dims: Tuple[RegulatedDim, ...]   # empty by default; per project


# ── witness (§6) ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WitnessRules:
    verified_min_mv_votes: int
    verified_max_mask_conflicts: int
    conflict_min: int               # mask_conflict needs at least this many conflicting views


@dataclass(frozen=True)
class TracksConfig:
    enabled: bool
    python: str                     # interpreter of the env carrying the vendored VGGSfM tracker
    win: int
    stride: int
    loop_window: int
    min_views: int                  # views a track needs to be triangulated
    reproj_max_px: float            # triangulation reprojection gate
    sigma_rel: float                # σ (relative) of a track depth observation
    min_obs_per_frame: int          # frames with fewer observations get no track row
    depth_edge_tol_rel: float       # a track pixel whose 2×2 depth neighbourhood spreads more than
                                    # this (relative) sits on a depth edge — no observation (an
                                    # interpolated depth across an edge is a blend, not a surface)


@dataclass(frozen=True)
class ContourConfig:
    enabled: bool
    min_gradient: float             # image gradient at the contour (0–255 scale) below → no observation
    samples_per_instance: int
    search_rel: float               # depth search range ±this fraction along the ray
    search_steps: int
    sigma_rel: float


@dataclass(frozen=True)
class DepthStageConfig:
    pair_offsets: Tuple[int, ...]   # frame offsets of the pairwise depth sensor (fitted)
    pair_samples: int
    holdout_fraction: float         # fraction of pairs held out for the verdict (never fitted)
    min_pairs: int
    improve: float                  # held-out disagreement must fall to ≤ this × before
    bound: float                    # corrections within this × the pairwise signal
    zref_m: float
    scale_only: bool
    pair_sigma_floor_rel: float     # σ floor of a projection-sensor row (pixel quantisation noise)
    refine_iters: int               # re-measure the pair rows on the corrected depth and re-solve
    prior_sigma_rel: float          # weak prior a=1, b=0 per frame (σ, relative): the (a, b) pair of a
                                    # frame whose depths span a narrow range is ill-conditioned —
                                    # what the evidence cannot tell stays identity
    pair_scatter_clip_sigma: float  # a pair row's precision is its RMS residual after the fit with
                                    # gross outliers clipped at this many robust σ — a structured
                                    # association error (a corner) must count, an occlusion must not


@dataclass(frozen=True)
class WitnessConfig:
    n_neighbors: int
    tau_rel: float
    cpu_threads: int                # torch threads on the CPU path (252-core box: unbounded
                                    # threads make every small op crawl — measured 50 ms/op)
    at_merge: bool                  # compute mv_votes + provisional status inside the GPU clean
                                    # (before voxel/SOR, which then run only on clean_statuses)
    mask_erosion_px: int
    occlusion_tol_rel: float
    rules: WitnessRules
    clean_statuses: Tuple[str, ...]         # voxel/SOR may only drop these
    mls_excluded_statuses: Tuple[str, ...]  # scene_consolidate never moves these
    drop_statuses: Tuple[str, ...]          # REMOVED from the cloud at merge time, before
                                            # the masks, the OBBs and every comparison
                                            # (USER 2026-09-17); empty = nothing removed
    tracks: TracksConfig
    contours: ContourConfig
    depth: DepthStageConfig


# ── certify (§4.8 ensemble witness + §9 loop, §10 harness) ──────────────────

@dataclass(frozen=True)
class ObjectiveWeights:
    loop_residual_m: float
    seam_residual_m: float
    closure_m: float
    depth_disagreement_frac: float
    duplicates: float


@dataclass(frozen=True)
class CertifyGates:
    mode: str                       # advisory | veto — advisory: every gate is measured and
                                    # recorded as a warning in the acta, the epoch is APPLIED
                                    # and the epoch selector is the verdict (USER 2026-09-13);
                                    # veto: a failed gate rejects the iteration (evaluation)
    max_seam_degradation_m: float
    max_loop_residual_increase_m: float
    max_depth_disagreement_increase: float
    max_verified_drop_frac: float
    duplicates_must_not_increase: bool


@dataclass(frozen=True)
class CertifyScale:
    sigma_loop_min_log: float
    sigma_seam_log: float
    sigma_anchor_log: float
    min_copy_points: int
    max_copy_residual_m: float
    icp_iters: int
    icp_trim: float
    max_correction_log: float       # |log r_k| beyond this is DECLARED (gate in the report);
                                    # it blocks the apply only under certify.gates.mode veto


@dataclass(frozen=True)
class VisitLoopsConfig:
    sigma_floor_m: float            # translation σ floor of a visit closure (the ICP rms is the measurement)
    unobserved_sigma_m: float       # σ of a translation direction the visit does not observe
    unobserved_sigma_deg: float     # σ of a rotation the visit does not observe (roll / pitch)
    window_kf: int                  # an instance's copy = its points within ± this of the visit keyframe
    max_pairs_per_instance: int     # how many copy pairs of ONE object are worth the ICP


@dataclass(frozen=True)
class KnownAnswerConfig:
    chunk: str                      # first | middle | last | an index — where the perturbation goes
                                    # (the revisited chunk is where the loops can see it)
    yaw_deg: float
    t_m: float
    scale: float


@dataclass(frozen=True)
class EnvelopeConfig:
    levels_t_m: Tuple[float, ...]
    levels_scale_pct: Tuple[float, ...]
    loop_densities: Tuple[float, ...]


@dataclass(frozen=True)
class DeterminismConfig:
    tol_m: float
    tol_frac: float
    seed: int


@dataclass(frozen=True)
class CertifyConfig:
    ensemble_offset_frames: int     # >0 → a second Omega pass with shifted chunk boundaries
    keep_aligned_chunks: bool
    geometric_cleanup: bool         # the retired second deleter — OFF, see config.yaml
                                    # after the scale (the post-hoc graph, the A/B harness and
                                    # the certification loop read them)
    max_iters: int
    eps: float                      # relative objective improvement below this → converged
    regression_eps: float           # an improvement BELOW -regression_eps is a REGRESSION, not
                                    # convergence: the iteration left the session worse than it
                                    # found it. Declared in the acta and in the attention list;
                                    # under gates.mode advisory the epoch is still applied and
                                    # the epoch selector remains the verdict (USER 2026-09-13)
    auto_after_segmentation: bool
    deliverable_only: bool          # the stage runs the CORRECTION and publishes its
                                    # epoch, and nothing else: no §9 before/after
                                    # measurement, no iteration loop, no acta metrics
                                    # (USER 2026-09-22 — the deliverable is epoch 0 and
                                    # the corrected epoch 1, judged by eye)
    objective: ObjectiveWeights
    gates: CertifyGates
    scale: CertifyScale
    visit_loops: VisitLoopsConfig
    known_answer: KnownAnswerConfig
    envelope: EnvelopeConfig
    determinism: DeterminismConfig


@dataclass(frozen=True)
class MetricGraphConfig:
    loop: LoopEdgeConfig
    loops: LoopsConfig
    scale: ScaleConfig
    graph: GraphConfig
    authority: AuthorityConfig
    structural: StructuralConfig
    certify: CertifyConfig
    witness: WitnessConfig


def load_loops_config(raw: Optional[Dict[str, Any]] = None) -> MetricGraphConfig:
    """Validate loops:/scale:/correction_graph: (raw = the whole config dict;
    None → server/config.py's ``cfg``)."""
    if raw is None:
        from config import cfg as _cfg
        raw = _cfg
    cg = raw.get("correction_graph")
    if not isinstance(cg, dict):
        raise LoopsConfigError("config section correction_graph is missing")
    lp = cg.get("loop")
    if not isinstance(lp, dict):
        raise LoopsConfigError("config section correction_graph.loop is missing")
    P = "correction_graph.loop"
    loop = LoopEdgeConfig(
        anchors_per_bridge=_num(lp, "anchors_per_bridge", P, lo=0, integer=True),
        max_edge_sigma_m=_num(lp, "max_edge_sigma_m", P, lo=0, lo_excl=True),
        max_residual_m=_num(lp, "max_residual_m", P, lo=0, lo_excl=True),
        min_correspondences=_num(lp, "min_correspondences", P, lo=1, integer=True),
        corr_per_frame=_num(lp, "corr_per_frame", P, lo=1, integer=True),
        fit_sample=_num(lp, "fit_sample", P, lo=1, integer=True),
        scale_tol_log=_num(lp, "scale_tol_log", P, lo=0, lo_excl=True),
        scale_break_sigma_factor=_num(lp, "scale_break_sigma_factor", P, lo=1.0),
        starved_sigma_m=_num(lp, "starved_sigma_m", P, lo=0, lo_excl=True),
        ambiguous_sigma_factor=_num(lp, "ambiguous_sigma_factor", P, lo=1.0),
        movable_labels=_str_list(lp, "movable_labels", P),
        min_shared_structural_labels=_num(lp, "min_shared_structural_labels", P, lo=0, integer=True),
        intra_chunk_loops=_bool(lp, "intra_chunk_loops", P),
    )
    if loop.max_edge_sigma_m >= loop.starved_sigma_m:
        raise LoopsConfigError("correction_graph.loop.max_edge_sigma_m must be below "
                               "starved_sigma_m (a starved fallback edge must never "
                               "enter the optimizer)")

    ls = raw.get("loops")
    if not isinstance(ls, dict):
        raise LoopsConfigError("config section loops is missing")
    sp = ls.get("spatial")
    if not isinstance(sp, dict):
        raise LoopsConfigError("config section loops.spatial is missing")
    S = "loops.spatial"
    spatial = SpatialGateConfig(
        min_walk_m=_num(sp, "min_walk_m", S, lo=0),
        drift_floor_m=_num(sp, "drift_floor_m", S, lo=0),
        drift_rate_m_per_m=_num(sp, "drift_rate_m_per_m", S, lo=0),
        frustum_tolerance_factor=_num(sp, "frustum_tolerance_factor", S, lo=1.0),
        drift_floor_deg=_num(sp, "drift_floor_deg", S, lo=0),
        drift_rate_deg_per_m=_num(sp, "drift_rate_deg_per_m", S, lo=0),
        frustum_margin_px=_num(sp, "frustum_margin_px", S, lo=0),
        occlusion_tol_m=_num(sp, "occlusion_tol_m", S, lo=0),
        min_depth_m=_num(sp, "min_depth_m", S, lo=0),
        max_depth_m=_num(sp, "max_depth_m", S, lo=0, lo_excl=True),
        min_frustum_frames=_num(sp, "min_frustum_frames", S, lo=1, integer=True),
        frustum_window_kf=_num(sp, "frustum_window_kf", S, lo=0, integer=True),
        frustum_points=_num(sp, "frustum_points", S, lo=1, integer=True),
        min_visible_frac=_num(sp, "min_visible_frac", S, lo=0, hi=1.0),
        size_tol=_num(sp, "size_tol", S, lo=0),
        repetitive_labels=_str_list(sp, "repetitive_labels", S),
        min_context_instances=_num(sp, "min_context_instances", S, lo=0, integer=True),
        corridor_width_m=_num(sp, "corridor_width_m", S, lo=0, lo_excl=True),
        corridor_min_frustum_frames=_num(sp, "corridor_min_frustum_frames", S, lo=1, integer=True),
        dims_pct_lo=_num(sp, "dims_pct_lo", S, lo=0, hi=100.0),
        dims_pct_hi=_num(sp, "dims_pct_hi", S, lo=0, hi=100.0),
        same_surface_angle_deg=_num(sp, "same_surface_angle_deg", S, lo=0, hi=90.0),
        same_surface_offset_m=_num(sp, "same_surface_offset_m", S, lo=0),
        same_surface_planar_ratio=_num(sp, "same_surface_planar_ratio", S, lo=0, hi=1.0),
        same_surface_axis_ratio=_num(sp, "same_surface_axis_ratio", S, lo=0, hi=1.0),
    )
    if spatial.dims_pct_hi <= spatial.dims_pct_lo:
        raise LoopsConfigError("loops.spatial.dims_pct_hi must exceed dims_pct_lo")
    if spatial.max_depth_m <= spatial.min_depth_m:
        raise LoopsConfigError("loops.spatial.max_depth_m must exceed min_depth_m")
    sm = ls.get("semantic")
    if not isinstance(sm, dict):
        raise LoopsConfigError("config section loops.semantic is missing")
    M = "loops.semantic"
    semantic = SemanticClassConfig(
        enabled=_bool(sm, "enabled", M),
        max_tokens=_num(sm, "max_tokens", M, lo=1, integer=True),
        crops_per_instance=_num(sm, "crops_per_instance", M, lo=1, integer=True),
        default_class=str(_require(sm, "default_class", M)),
        nonstructural_sigma_factor=_num(sm, "nonstructural_sigma_factor", M, lo=1.0),
    )
    if semantic.default_class not in ("structural", "movable", "dynamic"):
        raise LoopsConfigError("loops.semantic.default_class must be structural|movable|dynamic")
    sa = _sub(ls, "salad", "loops")
    A = "loops.salad"
    isz = _require(sa, "image_size", A)
    if (not isinstance(isz, (list, tuple)) or len(isz) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in isz)):
        raise LoopsConfigError("config key loops.salad.image_size must be [h, w] positive integers")
    salad = SaladConfig(
        similarity_threshold=_num(sa, "similarity_threshold", A, lo=-1.0, hi=1.0),
        top_k=_num(sa, "top_k", A, lo=1, integer=True),
        min_gap_keyframes=_num(sa, "min_gap_keyframes", A, lo=1, integer=True),
        min_gap_frac=_num(sa, "min_gap_frac", A, lo=0),
        nms_threshold=_num(sa, "nms_threshold", A, lo=0, integer=True),
        image_size=(int(isz[0]), int(isz[1])),
        batch_size=_num(sa, "batch_size", A, lo=1, integer=True),
    )
    L = "loops"
    rp = _sub(ls, "reprojection", L)
    reprojection = ReprojectionConfig(
        enabled=_bool(rp, "enabled", f"{L}.reprojection"),
        dilate_px=_num(rp, "dilate_px", f"{L}.reprojection", lo=0, integer=True),
        max_frames=_num(rp, "max_frames", f"{L}.reprojection", lo=1, integer=True),
        min_self_recall=_num(rp, "min_self_recall", f"{L}.reprojection", lo=0),
        min_cross_recall=_num(rp, "min_cross_recall", f"{L}.reprojection", lo=0),
        min_agreeing_frac=_num(rp, "min_agreeing_frac", f"{L}.reprojection", lo=0))

    loops = LoopsConfig(
        min_gap_keyframes=_num(ls, "min_gap_keyframes", L, lo=1, integer=True),
        duplicate_min_sep_m=_num(ls, "duplicate_min_sep_m", L, lo=0, lo_excl=True),
        dbscan_eps_m=_num(ls, "dbscan_eps_m", L, lo=0, lo_excl=True),
        dbscan_min_samples=_num(ls, "dbscan_min_samples", L, lo=1, integer=True),
        cluster_min_points=_num(ls, "cluster_min_points", L, lo=1, integer=True),
        bridge_extra_frames=_num(ls, "bridge_extra_frames", L, lo=0, integer=True),
        coverage_radius_m=_num(ls, "coverage_radius_m", L, lo=0, lo_excl=True),
        min_coverage=_num(ls, "min_coverage", L, lo=0, hi=1.0),
        spatial=spatial, semantic=semantic, salad=salad, reprojection=reprojection,
    )

    sc = raw.get("scale")
    if not isinstance(sc, dict):
        raise LoopsConfigError("config section scale is missing")
    C = "scale"
    scale = ScaleConfig(
        sigma_loop=_num(sc, "sigma_loop", C, lo=0, lo_excl=True),
        sigma_vio=_num(sc, "sigma_vio", C, lo=0, lo_excl=True),
        sigma_regulated=_num(sc, "sigma_regulated", C, lo=0, lo_excl=True),
        verify_max_dev=_num(sc, "verify_max_dev", C, lo=0, lo_excl=True),
        vio_segment_s=_num(sc, "vio_segment_s", C, lo=0, lo_excl=True),
        vio_min_segments_chunk=_num(sc, "vio_min_segments_chunk", C, lo=1, integer=True),
        vio_min_seg_disp_m=_num(sc, "vio_min_seg_disp_m", C, lo=0),
        break_localisation_gap=_num(sc, "break_localisation_gap", C, lo=0),
    )
    gp = cg.get("graph")
    if not isinstance(gp, dict):
        raise LoopsConfigError("config section correction_graph.graph is missing")
    G = "correction_graph.graph"
    ho = _require(gp, "holdout_offsets", G)
    if not isinstance(ho, (list, tuple)) or not ho or not all(isinstance(x, int) and x > 0 for x in ho):
        raise LoopsConfigError("correction_graph.graph.holdout_offsets must be a non-empty list "
                               "of positive integers")
    graph = GraphConfig(
        sigma_odo_intra_m=_num(gp, "sigma_odo_intra_m", G, lo=0, lo_excl=True),
        sigma_odo_intra_deg=_num(gp, "sigma_odo_intra_deg", G, lo=0, lo_excl=True),
        loop_sigma_rot_deg=_num(gp, "loop_sigma_rot_deg", G, lo=0, lo_excl=True),
        odo_sigma_from_drift=_bool(gp, "odo_sigma_from_drift", G),
        odo_sigma_min_m=_num(gp, "odo_sigma_min_m", G, lo=0, lo_excl=True),
        odo_sigma_max_m=_num(gp, "odo_sigma_max_m", G, lo=0, lo_excl=True),
        outlier_overlap_frac=_num(gp, "outlier_overlap_frac", G, lo=0, hi=1),
        outlier_mad_k=_num(gp, "outlier_mad_k", G, lo=0, lo_excl=True),
        outlier_mad_floor_m_per_m=_num(gp, "outlier_mad_floor_m_per_m", G, lo=0, lo_excl=True),
        outlier_max_sigma_factor=_num(gp, "outlier_max_sigma_factor", G, lo=1),
        drift_model=_bool(gp, "drift_model", G),
        drift_degree=_num(gp, "drift_degree", G, lo=1, integer=True),
        drift_iters=_num(gp, "drift_iters", G, lo=1, integer=True),
        drift_max_step=_num(gp, "drift_max_step", G, lo=0, lo_excl=True),
        drift_prior_rot_deg=_num(gp, "drift_prior_rot_deg", G, lo=0, lo_excl=True),
        drift_prior_trans_m=_num(gp, "drift_prior_trans_m", G, lo=0, lo_excl=True),
        drift_rel_tol=_num(gp, "drift_rel_tol", G, lo=0, lo_excl=True),
        drift_min_gain=_num(gp, "drift_min_gain", G, lo=0, hi=1),
        huber_delta_m=_num(gp, "huber_delta_m", G, lo=0, lo_excl=True),
        huber_delta_deg=_num(gp, "huber_delta_deg", G, lo=0, lo_excl=True),
        dense_max_unknowns=_num(gp, "dense_max_unknowns", G, lo=6, integer=True),
        lambda_init=_num(gp, "lambda_init", G, lo=0, lo_excl=True),
        lambda_max=_num(gp, "lambda_max", G, lo=0, lo_excl=True),
        lm_diag_floor=_num(gp, "lm_diag_floor", G, lo=0),
        tol=_num(gp, "tol", G, lo=0, lo_excl=True),
        rel_tol=_num(gp, "rel_tol", G, lo=0, lo_excl=True),
        max_iters=_num(gp, "max_iters", G, lo=1, integer=True),
        pcg_tol=_num(gp, "pcg_tol", G, lo=0, lo_excl=True),
        pcg_max_iters=_num(gp, "pcg_max_iters", G, lo=1, integer=True),
        min_loop_gain=_num(gp, "min_loop_gain", G, lo=0, hi=1.0),
        max_seam_degradation_m=_num(gp, "max_seam_degradation_m", G, lo=0),
        gate_mode=_choice(gp, "gate_mode", G, ("advisory", "veto")),
        holdout_offsets=tuple(int(x) for x in ho),
        holdout_stride=_num(gp, "holdout_stride", G, lo=1, integer=True),
        holdout_samples=_num(gp, "holdout_samples", G, lo=100, integer=True),
        holdout_max_nn_m=_num(gp, "holdout_max_nn_m", G, lo=0, lo_excl=True),
    )

    au = raw.get("authority")
    if not isinstance(au, dict):
        raise LoopsConfigError("config section authority is missing")
    A = "authority"
    authority = AuthorityConfig(
        saturation_warn=_num(au, "saturation_warn", A, lo=0, hi=1.0),
        pose_graph_max_m=_num(au, "pose_graph_max_m", A, lo=0, lo_excl=True),
        pose_graph_max_deg=_num(au, "pose_graph_max_deg", A, lo=0, lo_excl=True),
        scale_graph_max_log=_num(au, "scale_graph_max_log", A, lo=0, lo_excl=True),
        depth_graph_max_log_a=_num(au, "depth_graph_max_log_a", A, lo=0, lo_excl=True),
        depth_graph_max_b_m=_num(au, "depth_graph_max_b_m", A, lo=0, lo_excl=True),
        intra_chunk_max_m=_num(au, "intra_chunk_max_m", A, lo=0, lo_excl=True),
    )

    st = raw.get("structural")
    if not isinstance(st, dict):
        raise LoopsConfigError("config section structural is missing")
    wp = st.get("wall_planarity")
    if not isinstance(wp, dict):
        raise LoopsConfigError("config section structural.wall_planarity is missing")
    W = "structural.wall_planarity"
    wall = WallPlanarityConfig(
        enabled=_bool(wp, "enabled", W),
        wall_tol_m=_num(wp, "wall_tol_m", W, lo=0, lo_excl=True),
        sigma_angle_deg=_num(wp, "sigma_angle_deg", W, lo=0, lo_excl=True),
        sigma_offset_m=_num(wp, "sigma_offset_m", W, lo=0, lo_excl=True),
        min_span_m=_num(wp, "min_span_m", W, lo=0),
        min_points_per_kf=_num(wp, "min_points_per_kf", W, lo=3, integer=True),
        labels=_str_list(wp, "labels", W),
        reference_span_kf=_num(wp, "reference_span_kf", W, lo=1, integer=True),
        planar_ratio=_num(wp, "planar_ratio", W, lo=0, hi=1.0),
        min_patch_extent_m=_num(wp, "min_patch_extent_m", W, lo=0),
    )
    cv = st.get("column_vertical")
    if not isinstance(cv, dict):
        raise LoopsConfigError("config section structural.column_vertical is missing")
    V = "structural.column_vertical"
    column = ColumnVerticalConfig(
        enabled=_bool(cv, "enabled", V),
        sigma_deg=_num(cv, "sigma_deg", V, lo=0, lo_excl=True),
        labels=_str_list(cv, "labels", V),
        min_points_per_kf=_num(cv, "min_points_per_kf", V, lo=3, integer=True),
        axis_ratio=_num(cv, "axis_ratio", V, lo=0, hi=1.0),
    )
    rp = st.get("repeated_parallel")
    if not isinstance(rp, dict):
        raise LoopsConfigError("config section structural.repeated_parallel is missing")
    R = "structural.repeated_parallel"
    repeated = RepeatedParallelConfig(
        enabled=_bool(rp, "enabled", R),
        sigma_deg=_num(rp, "sigma_deg", R, lo=0, lo_excl=True),
        labels=_str_list(rp, "labels", R),
        min_points_per_kf=_num(rp, "min_points_per_kf", R, lo=3, integer=True),
        axis_ratio=_num(rp, "axis_ratio", R, lo=0, hi=1.0),
    )
    rd = st.get("regulated_dims")
    if rd is None or not isinstance(rd, (list, tuple)):
        raise LoopsConfigError("config key structural.regulated_dims must be a list (empty by default)")
    dims = []
    for k, e in enumerate(rd):
        D = f"structural.regulated_dims[{k}]"
        if not isinstance(e, dict):
            raise LoopsConfigError(f"{D} must be a mapping")
        dim = str(_require(e, "dimension", D))
        if dim not in ("width", "diameter", "height", "gauge"):
            raise LoopsConfigError(f"{D}.dimension must be width|diameter|height|gauge")
        dims.append(RegulatedDim(label=str(_require(e, "label", D)).lower(), dimension=dim,
                                 value_m=_num(e, "value_m", D, lo=0, lo_excl=True),
                                 tol_m=_num(e, "tol_m", D, lo=0, lo_excl=True)))
    structural = StructuralConfig(wall_planarity=wall, column_vertical=column,
                                  repeated_parallel=repeated, regulated_dims=tuple(dims))

    ce = raw.get("certify")
    if not isinstance(ce, dict):
        raise LoopsConfigError("config section certify is missing")
    certify = _parse_certify(ce)
    wi = raw.get("witness")
    if not isinstance(wi, dict):
        raise LoopsConfigError("config section witness is missing")
    witness = _parse_witness(wi)
    return MetricGraphConfig(loop=loop, loops=loops, scale=scale, graph=graph,
                             authority=authority, structural=structural, certify=certify,
                             witness=witness)


def _sub(section, key, path) -> Dict[str, Any]:
    v = _require(section, key, path)
    if not isinstance(v, dict):
        raise LoopsConfigError(f"config section {path}.{key} must be a mapping")
    return v


def _num_list(section, key, path, lo=None) -> Tuple[float, ...]:
    v = _require(section, key, path)
    if not isinstance(v, (list, tuple)) or not v or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        raise LoopsConfigError(f"config key {path}.{key} must be a non-empty list of numbers")
    if lo is not None and any(x < lo for x in v):
        raise LoopsConfigError(f"config key {path}.{key} must be ≥ {lo} everywhere")
    return tuple(float(x) for x in v)


def _parse_witness(wi: Dict[str, Any]) -> WitnessConfig:
    from reconstruction.witness.status import STATUS_CODES
    P = "witness"
    ru = _sub(wi, "rules", P)
    rules = WitnessRules(
        verified_min_mv_votes=_num(ru, "verified_min_mv_votes", P + ".rules", lo=1, integer=True),
        verified_max_mask_conflicts=_num(ru, "verified_max_mask_conflicts", P + ".rules", lo=0, integer=True),
        conflict_min=_num(ru, "conflict_min", P + ".rules", lo=1, integer=True))
    tr = _sub(wi, "tracks", P)
    T = P + ".tracks"
    tracks = TracksConfig(
        enabled=_bool(tr, "enabled", T), python=str(_require(tr, "python", T)),
        win=_num(tr, "win", T, lo=2, integer=True), stride=_num(tr, "stride", T, lo=1, integer=True),
        loop_window=_num(tr, "loop_window", T, lo=1, integer=True),
        min_views=_num(tr, "min_views", T, lo=2, integer=True),
        reproj_max_px=_num(tr, "reproj_max_px", T, lo=0, lo_excl=True),
        sigma_rel=_num(tr, "sigma_rel", T, lo=0, lo_excl=True),
        min_obs_per_frame=_num(tr, "min_obs_per_frame", T, lo=1, integer=True),
        depth_edge_tol_rel=_num(tr, "depth_edge_tol_rel", T, lo=0, lo_excl=True))
    co = _sub(wi, "contours", P)
    C = P + ".contours"
    contours = ContourConfig(
        enabled=_bool(co, "enabled", C), min_gradient=_num(co, "min_gradient", C, lo=0),
        samples_per_instance=_num(co, "samples_per_instance", C, lo=1, integer=True),
        search_rel=_num(co, "search_rel", C, lo=0, lo_excl=True, hi=0.9),
        search_steps=_num(co, "search_steps", C, lo=3, integer=True),
        sigma_rel=_num(co, "sigma_rel", C, lo=0, lo_excl=True))
    de = _sub(wi, "depth", P)
    D = P + ".depth"
    depth = DepthStageConfig(
        pair_offsets=tuple(int(x) for x in _num_list(de, "pair_offsets", D, lo=1)),
        pair_samples=_num(de, "pair_samples", D, lo=100, integer=True),
        holdout_fraction=_num(de, "holdout_fraction", D, lo=0, hi=0.9, lo_excl=True),
        min_pairs=_num(de, "min_pairs", D, lo=1, integer=True),
        improve=_num(de, "improve", D, lo=0, hi=1.0, lo_excl=True),
        bound=_num(de, "bound", D, lo=1.0), zref_m=_num(de, "zref_m", D, lo=0, lo_excl=True),
        scale_only=_bool(de, "scale_only", D),
        pair_sigma_floor_rel=_num(de, "pair_sigma_floor_rel", D, lo=0, lo_excl=True),
        refine_iters=_num(de, "refine_iters", D, lo=1, integer=True),
        prior_sigma_rel=_num(de, "prior_sigma_rel", D, lo=0, lo_excl=True),
        pair_scatter_clip_sigma=_num(de, "pair_scatter_clip_sigma", D, lo=1.0))
    clean = _str_list(wi, "clean_statuses", P)
    mls = _str_list(wi, "mls_excluded_statuses", P)
    drop = _str_list(wi, "drop_statuses", P)
    if set(drop) & set(clean):
        raise LoopsConfigError(
            f"witness.drop_statuses {sorted(set(drop) & set(clean))} also listed in "
            f"clean_statuses: a status cannot both be removed and be what the net is "
            f"allowed to thin")
    for name in clean + mls + drop:
        if name not in STATUS_CODES:
            raise LoopsConfigError(f"witness status {name!r} unknown (one of {sorted(STATUS_CODES)})")
    return WitnessConfig(
        n_neighbors=_num(wi, "n_neighbors", P, lo=1, integer=True),
        tau_rel=_num(wi, "tau_rel", P, lo=0, lo_excl=True),
        cpu_threads=_num(wi, "cpu_threads", P, lo=1, integer=True),
        at_merge=_bool(wi, "at_merge", P),
        mask_erosion_px=_num(wi, "mask_erosion_px", P, lo=0, integer=True),
        occlusion_tol_rel=_num(wi, "occlusion_tol_rel", P, lo=0),
        rules=rules, clean_statuses=clean, mls_excluded_statuses=mls, drop_statuses=drop,
        tracks=tracks, contours=contours, depth=depth)


def _parse_certify(ce: Dict[str, Any]) -> CertifyConfig:
    P = "certify"
    ow = _sub(ce, "objective_weights", P)
    O = P + ".objective_weights"
    objective = ObjectiveWeights(
        loop_residual_m=_num(ow, "loop_residual_m", O, lo=0), seam_residual_m=_num(ow, "seam_residual_m", O, lo=0),
        closure_m=_num(ow, "closure_m", O, lo=0),
        depth_disagreement_frac=_num(ow, "depth_disagreement_frac", O, lo=0),
        duplicates=_num(ow, "duplicates", O, lo=0))
    ga = _sub(ce, "gates", P)
    G = P + ".gates"
    gates = CertifyGates(
        mode=_choice(ga, "mode", G, ("advisory", "veto")),
        max_seam_degradation_m=_num(ga, "max_seam_degradation_m", G, lo=0),
        max_loop_residual_increase_m=_num(ga, "max_loop_residual_increase_m", G, lo=0),
        max_depth_disagreement_increase=_num(ga, "max_depth_disagreement_increase", G, lo=0),
        max_verified_drop_frac=_num(ga, "max_verified_drop_frac", G, lo=0, hi=1.0),
        duplicates_must_not_increase=_bool(ga, "duplicates_must_not_increase", G))
    sc = _sub(ce, "scale", P)
    S = P + ".scale"
    scale = CertifyScale(
        sigma_loop_min_log=_num(sc, "sigma_loop_min_log", S, lo=0, lo_excl=True),
        sigma_seam_log=_num(sc, "sigma_seam_log", S, lo=0, lo_excl=True),
        sigma_anchor_log=_num(sc, "sigma_anchor_log", S, lo=0, lo_excl=True),
        min_copy_points=_num(sc, "min_copy_points", S, lo=10, integer=True),
        max_copy_residual_m=_num(sc, "max_copy_residual_m", S, lo=0, lo_excl=True),
        icp_iters=_num(sc, "icp_iters", S, lo=1, integer=True),
        icp_trim=_num(sc, "icp_trim", S, lo=0.1, hi=1.0),
        max_correction_log=_num(sc, "max_correction_log", S, lo=0, lo_excl=True))
    vl = _sub(ce, "visit_loops", P)
    V = P + ".visit_loops"
    visit_loops = VisitLoopsConfig(
        sigma_floor_m=_num(vl, "sigma_floor_m", V, lo=0, lo_excl=True),
        unobserved_sigma_m=_num(vl, "unobserved_sigma_m", V, lo=0, lo_excl=True),
        unobserved_sigma_deg=_num(vl, "unobserved_sigma_deg", V, lo=0, lo_excl=True),
        window_kf=_num(vl, "window_kf", V, lo=1, integer=True),
        max_pairs_per_instance=_num(vl, "max_pairs_per_instance", V, lo=0,
                                    integer=True))
    ka = _sub(ce, "known_answer", P)
    K = P + ".known_answer"
    chunk = str(_require(ka, "chunk", K))
    if chunk not in ("first", "middle", "last") and not chunk.lstrip("-").isdigit():
        raise LoopsConfigError(f"{K}.chunk must be first|middle|last or a chunk index, got {chunk!r}")
    known = KnownAnswerConfig(
        chunk=chunk,
        yaw_deg=_num(ka, "yaw_deg", K, lo=0), t_m=_num(ka, "t_m", K, lo=0),
        scale=_num(ka, "scale", K, lo=0, lo_excl=True),
        )
    en = _sub(ce, "envelope", P)
    E = P + ".envelope"
    envelope = EnvelopeConfig(
        levels_t_m=_num_list(en, "levels_t_m", E, lo=0),
        levels_scale_pct=_num_list(en, "levels_scale_pct", E, lo=0),
        loop_densities=_num_list(en, "loop_densities", E, lo=0))
    if any(x > 1.0 for x in envelope.loop_densities):
        raise LoopsConfigError("certify.envelope.loop_densities are fractions in (0, 1]")
    dt = _sub(ce, "determinism", P)
    Dm = P + ".determinism"
    determinism = DeterminismConfig(
        tol_m=_num(dt, "tol_m", Dm, lo=0), tol_frac=_num(dt, "tol_frac", Dm, lo=0),
        seed=_num(dt, "seed", Dm, lo=0, integer=True))
    return CertifyConfig(
        ensemble_offset_frames=_num(ce, "ensemble_offset_frames", P, lo=0, integer=True),
        keep_aligned_chunks=_bool(ce, "keep_aligned_chunks", P),
        geometric_cleanup=_bool(ce, "geometric_cleanup", P),
        max_iters=_num(ce, "max_iters", P, lo=1, integer=True),
        eps=_num(ce, "eps", P, lo=0),
        regression_eps=_num(ce, "regression_eps", P, lo=0),
        auto_after_segmentation=_bool(ce, "auto_after_segmentation", P),
        deliverable_only=_bool(ce, "deliverable_only", P),
        objective=objective, gates=gates, scale=scale, visit_loops=visit_loops,
        known_answer=known, envelope=envelope, determinism=determinism)


# ── fork-facing dicts ────────────────────────────────────────────────────────

def fork_model_loops(cfg: MetricGraphConfig, stac_server_dir: str) -> Dict[str, Any]:
    """``Model.loops`` for vendor/VGGT-Long: the edge/verification keys flat,
    the spatial gate as a sub-dict, plus the server dir the fork imports the
    gate from."""
    d = asdict(cfg.loop)
    d["movable_labels"] = list(cfg.loop.movable_labels)
    d["bridge_extra_frames"] = cfg.loops.bridge_extra_frames
    # σ inflation of a bridge proposed by a non-structural SAM3 instance
    # (candidate source ``instance:<class>``) — the class tags, never drops
    d["nonstructural_sigma_factor"] = float(cfg.loops.semantic.nonstructural_sigma_factor)
    sp = asdict(cfg.loops.spatial)
    sp["repetitive_labels"] = list(cfg.loops.spatial.repetitive_labels)
    d["spatial"] = sp
    d["stac_server_dir"] = str(stac_server_dir)
    return d


def fork_model_scale(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return asdict(cfg.scale)


def fork_loop_salad(cfg: MetricGraphConfig) -> Dict[str, Any]:
    """``Loop.SALAD`` for vendor/VGGT-Long's LoopDetector (its own keys; the
    fork's base_config carries the vendor's driving-video values, this
    overrides every one of them from config.yaml)."""
    s = cfg.loops.salad
    return {"image_size": [int(s.image_size[0]), int(s.image_size[1])],
            "batch_size": int(s.batch_size),
            "similarity_threshold": float(s.similarity_threshold),
            "top_k": int(s.top_k),
            "use_nms": bool(s.nms_threshold > 0),
            "nms_threshold": int(s.nms_threshold),
            "min_gap": int(s.min_gap_keyframes),
            "min_gap_frac": float(s.min_gap_frac)}


def fork_model_graph(cfg: MetricGraphConfig) -> Dict[str, Any]:
    d = asdict(cfg.graph)
    d["holdout_offsets"] = list(cfg.graph.holdout_offsets)
    return d


def fork_model_authority(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return asdict(cfg.authority)


def _plain(obj):
    """Tuples → lists recursively: the Omega config is written with yaml.dump
    and read back by every consumer with safe_load — a tuple would land as a
    ``!!python/tuple`` tag (the elastic A/B harness died on it, pccr 2026-09-13)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def fork_model_certify(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return _plain(asdict(cfg.certify))
