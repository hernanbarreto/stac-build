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
from typing import Any, Dict, List, Optional, Tuple


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
    attention_verify: bool          # §4.2.3 (OFF until an A/B shows fewer false positives)
    attention_min_score: float
    movable_labels: Tuple[str, ...]  # labels that never count for/against a semantic match
    min_shared_structural_labels: int
    intra_chunk_loops: bool         # keep candidates inside one chunk (pose edges, no scale row)


# ── loops — detector + spatial gate (§4.4/§4.5/§4.9) ───────────────────────

@dataclass(frozen=True)
class SpatialGateConfig:
    drift_floor_m: float
    drift_rate_m_per_m: float
    drift_floor_deg: float
    drift_rate_deg_per_m: float
    identity_reject_factor: float   # separation > factor × δ(L) → not drift: split
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


@dataclass(frozen=True)
class LoopsConfig:
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
    sigma_gravity_deg: float        # weak per-node prior: camera down stays the chain consensus
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
    holdout_offsets: Tuple[int, ...]  # frame offsets of the held-out pairs (chunk_field_verdict)
    holdout_stride: int
    holdout_samples: int
    holdout_max_nn_m: float         # cloud held-out judge: a NN pair beyond this is not a correspondence
    run_without_loops: bool         # solve with odometry + priors only (no loop edge)


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
class FloorDatumConfig:
    enabled: bool
    sigma_angle_deg: float
    sigma_offset_m: float
    max_tilt_deg: float             # a low-band plane tilted more than this is not a floor patch
    reference_span_kf: int          # the datum plane is fitted on the first N keyframes' patches
    step_demote_m: float            # a patch farther than this from the datum is a real step
    low_band_pct: float             # per-keyframe low height percentile that seeds the patch
    band_m: float                   # patch band half-height
    min_points: int
    ransac_tol_m: float
    ransac_iters: int


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
    floor_datum: FloorDatumConfig
    wall_planarity: WallPlanarityConfig
    column_vertical: ColumnVerticalConfig
    repeated_parallel: RepeatedParallelConfig
    regulated_dims: Tuple[RegulatedDim, ...]   # empty by default; per project


# ── certify (§4.8 ensemble witness; F3 adds the loop keys) ──────────────────

@dataclass(frozen=True)
class CertifyConfig:
    ensemble_offset_frames: int     # >0 → a second Omega pass with shifted chunk boundaries
    keep_aligned_chunks: bool       # keep maplong_run/_tmp_results_aligned + _tmp_results_loop
                                    # after the scale (the post-hoc graph, the A/B harness and
                                    # the certification loop read them)


@dataclass(frozen=True)
class MetricGraphConfig:
    loop: LoopEdgeConfig
    loops: LoopsConfig
    scale: ScaleConfig
    graph: GraphConfig
    authority: AuthorityConfig
    structural: StructuralConfig
    certify: CertifyConfig


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
        attention_verify=_bool(lp, "attention_verify", P),
        attention_min_score=_num(lp, "attention_min_score", P, lo=-1.0, hi=1.0),
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
        drift_floor_m=_num(sp, "drift_floor_m", S, lo=0),
        drift_rate_m_per_m=_num(sp, "drift_rate_m_per_m", S, lo=0),
        drift_floor_deg=_num(sp, "drift_floor_deg", S, lo=0),
        drift_rate_deg_per_m=_num(sp, "drift_rate_deg_per_m", S, lo=0),
        identity_reject_factor=_num(sp, "identity_reject_factor", S, lo=1.0),
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
    )
    if semantic.default_class not in ("structural", "movable", "dynamic"):
        raise LoopsConfigError("loops.semantic.default_class must be structural|movable|dynamic")
    L = "loops"
    loops = LoopsConfig(
        min_gap_keyframes=_num(ls, "min_gap_keyframes", L, lo=1, integer=True),
        duplicate_min_sep_m=_num(ls, "duplicate_min_sep_m", L, lo=0, lo_excl=True),
        dbscan_eps_m=_num(ls, "dbscan_eps_m", L, lo=0, lo_excl=True),
        dbscan_min_samples=_num(ls, "dbscan_min_samples", L, lo=1, integer=True),
        cluster_min_points=_num(ls, "cluster_min_points", L, lo=1, integer=True),
        bridge_extra_frames=_num(ls, "bridge_extra_frames", L, lo=0, integer=True),
        coverage_radius_m=_num(ls, "coverage_radius_m", L, lo=0, lo_excl=True),
        min_coverage=_num(ls, "min_coverage", L, lo=0, hi=1.0),
        spatial=spatial, semantic=semantic,
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
        sigma_gravity_deg=_num(gp, "sigma_gravity_deg", G, lo=0, lo_excl=True),
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
        holdout_offsets=tuple(int(x) for x in ho),
        holdout_stride=_num(gp, "holdout_stride", G, lo=1, integer=True),
        holdout_samples=_num(gp, "holdout_samples", G, lo=100, integer=True),
        holdout_max_nn_m=_num(gp, "holdout_max_nn_m", G, lo=0, lo_excl=True),
        run_without_loops=_bool(gp, "run_without_loops", G),
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
    fd = st.get("floor_datum")
    if not isinstance(fd, dict):
        raise LoopsConfigError("config section structural.floor_datum is missing")
    F = "structural.floor_datum"
    floor = FloorDatumConfig(
        enabled=_bool(fd, "enabled", F),
        sigma_angle_deg=_num(fd, "sigma_angle_deg", F, lo=0, lo_excl=True),
        sigma_offset_m=_num(fd, "sigma_offset_m", F, lo=0, lo_excl=True),
        max_tilt_deg=_num(fd, "max_tilt_deg", F, lo=0, hi=90.0),
        reference_span_kf=_num(fd, "reference_span_kf", F, lo=1, integer=True),
        step_demote_m=_num(fd, "step_demote_m", F, lo=0, lo_excl=True),
        low_band_pct=_num(fd, "low_band_pct", F, lo=0, hi=100.0),
        band_m=_num(fd, "band_m", F, lo=0, lo_excl=True),
        min_points=_num(fd, "min_points", F, lo=3, integer=True),
        ransac_tol_m=_num(fd, "ransac_tol_m", F, lo=0, lo_excl=True),
        ransac_iters=_num(fd, "ransac_iters", F, lo=1, integer=True),
    )
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
    structural = StructuralConfig(floor_datum=floor, wall_planarity=wall, column_vertical=column,
                                  repeated_parallel=repeated, regulated_dims=tuple(dims))

    ce = raw.get("certify")
    if not isinstance(ce, dict):
        raise LoopsConfigError("config section certify is missing")
    certify = CertifyConfig(
        ensemble_offset_frames=_num(ce, "ensemble_offset_frames", "certify", lo=0, integer=True),
        keep_aligned_chunks=_bool(ce, "keep_aligned_chunks", "certify"))
    return MetricGraphConfig(loop=loop, loops=loops, scale=scale, graph=graph,
                             authority=authority, structural=structural, certify=certify)


# ── fork-facing dicts ────────────────────────────────────────────────────────

def fork_model_loops(cfg: MetricGraphConfig, stac_server_dir: str) -> Dict[str, Any]:
    """``Model.loops`` for vendor/VGGT-Long: the edge/verification keys flat,
    the spatial gate as a sub-dict, plus the server dir the fork imports the
    gate from."""
    d = asdict(cfg.loop)
    d["movable_labels"] = list(cfg.loop.movable_labels)
    d["bridge_extra_frames"] = cfg.loops.bridge_extra_frames
    sp = asdict(cfg.loops.spatial)
    sp["repetitive_labels"] = list(cfg.loops.spatial.repetitive_labels)
    d["spatial"] = sp
    d["stac_server_dir"] = str(stac_server_dir)
    return d


def fork_model_scale(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return asdict(cfg.scale)


def fork_model_graph(cfg: MetricGraphConfig) -> Dict[str, Any]:
    d = asdict(cfg.graph)
    d["holdout_offsets"] = list(cfg.graph.holdout_offsets)
    return d


def fork_model_authority(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return asdict(cfg.authority)


def fork_model_certify(cfg: MetricGraphConfig) -> Dict[str, Any]:
    return asdict(cfg.certify)
