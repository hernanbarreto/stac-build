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


@dataclass(frozen=True)
class MetricGraphConfig:
    loop: LoopEdgeConfig
    loops: LoopsConfig
    scale: ScaleConfig


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
    return MetricGraphConfig(loop=loop, loops=loops, scale=scale)


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
