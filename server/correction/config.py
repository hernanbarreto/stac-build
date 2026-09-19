"""Typed, validated load of the ``correction:`` section of server/config.yaml.

What it decides: nothing — it only guarantees that every decision parameter the
module uses exists, has the right type and sits inside a sane range BEFORE any
geometry is touched. There is deliberately NO default value in code: a missing
key aborts the load naming the key (fail-fast doctrine; the YAML documents the
defaults and their provenance).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


class CorrectionConfigError(RuntimeError):
    """Raised when the ``correction:`` config section is missing, incomplete
    or out of range. The message always names the offending key."""


_FLOOR_MODELS = ("level", "plane", "profile")
_DEPTH_MODES = ("sidecar", "rewrite")


@dataclass(frozen=True)
class EvidenceConfig:
    obb_margin_m: float
    obb_core_pct: float
    min_object_points_solve: int
    min_object_points_fingerprint: int
    min_baseline_m: float
    min_objects_for_depth: int
    visit_gap_kf: int


@dataclass(frozen=True)
class ObservabilityConfig:
    pca_ratio_planar: float
    pca_ratio_cylindrical: float
    bounded_extent_tol: float
    yaw_anisotropy_max: float


@dataclass(frozen=True)
class SolveConfig:
    icp_iters: int
    icp_trim: float
    icp_sample: int
    eval_sample: int
    icp_min_corr: int
    icp_converge_deg: float
    icp_converge_m: float
    plane_ransac_tol_m: float
    plane_ransac_iters: int
    plane_ransac_sample: int
    depth_compress_tol: float
    seed: int


@dataclass(frozen=True)
class GatesConfig:
    mode: str                 # advisory | veto
    max_object_residual_m: float
    residual_improvement_ratio: float
    collapse_floor_m: float
    max_rot_deg: float
    max_translation_m: float
    heldout_floor_tol_m: float
    heldout_floor_abs_m: float
    heldout_object_tol_m: float
    max_step_mm: float
    max_step_deg: float
    scale_agree_tol: float
    scale_mad_tol: float


@dataclass(frozen=True)
class FloorConfig:
    model_default: str
    band_m: float
    low_band_pct: float
    min_tilt_deg: float
    max_tilt_deg: float
    min_inliers: int
    min_inlier_ratio: float
    ransac_tol_m: float
    ransac_refit_band_m: float
    ransac_iters: int
    ransac_sample: int
    smooth_window_kf: int
    step_demote_m: float
    reference_span_kf: int


@dataclass(frozen=True)
class RevisitConfig:
    sample_per_kf: int
    min_gap_kf: int
    min_covis: float
    depth_min_m: float
    depth_max_m: float
    zbuffer_cell_px: int
    same_surface_tol_m: float
    same_surface_tol_rel: float
    region_cell_m: float
    region_sample: int
    offset_min_m: float
    image_max_px: int


@dataclass(frozen=True)
class ConsistencyConfig:
    cell_m: float             # block size of the region-centric check
    block_min_points: int     # blocks with fewer points are not verifiable
    block_sample: int         # consensus points per block (cap)
    writer_min_points: int    # a keyframe must have written this many in the block
    writer_query_sample: int  # points of the writer queried against the others
    viewer_probe: int         # probe points per block for the viewer test
    viewer_min_frac: float    # fraction of probes a keyframe must see unoccluded
    near_kf: int              # |i-j| ≤ this = same pass (neighbours); beyond = distant


@dataclass(frozen=True)
class KfGraphConfig:
    pair_src_sample: int      # source points per pair (i's points in shared blocks)
    pair_tgt_sample: int      # target points per pair (j's points in shared blocks)
    normal_k: int             # neighbours for the target normals (information matrix)
    gn_iters: int             # Gauss-Newton iterations
    gn_converge: float        # max step (m or rad) below which the solve stops
    damping: float            # initial Levenberg damping (relative to the diagonal)
    damping_min: float        # floor of the damping as steps keep being accepted
    lm_factor: float          # damping multiplier when a step is rejected
    lm_tries: int             # rejected steps allowed per iteration
    floor_m: float            # neighbour floor: pairs still above it after the solve are reported


@dataclass(frozen=True)
class PhotoConfig:
    candidate_sample: int     # points per keyframe projected to find candidate pairs
    candidate_min: int        # samples of j inside k's image to make (k, j) a pair
    pair_min_points: int      # rendered points below which a pair is skipped
    points_per_pair: int      # correspondences kept per pair for the solve
    splat_radius: int         # splat radius (px) of the synthetic photo
    outer_iters: int          # render → flow → solve rounds
    huber_px: float           # Huber threshold on the reprojection residual (px)


@dataclass(frozen=True)
class PoseGraphConfig:
    loop_weight: float        # weight of a loop-closure edge vs the seam prior
    rot_lever_m: float        # metres per radian: converts yaw residuals to m
    min_blocks_improved: float  # fraction of a pair's blocks the joint closure must improve


@dataclass(frozen=True)
class ApplyConfig:
    depth_correction_mode: str
    potree_rebuild: bool
    reconsolidate: bool   # re-run the scene consolidation on the WARPED cloud inside the
                          # transaction. The epoch moves each point by its keyframe's
                          # correction and nobody cleans afterwards, so where a duplicate
                          # closes the two copies land on top of each other and stay two
                          # point sets: geometrically right, visually double density — the
                          # user still sees the object twice. Consolidation moves points
                          # without adding or removing any, so globalIndices, colours and
                          # provenance all survive it.


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int          # resolved: 'auto' becomes min(8, affinity)


@dataclass(frozen=True)
class FloorConsensusConfig:
    """The floor as a per-keyframe vertical constraint (USER 2026-09-19).

    Real terrain belongs to the PLACE and a pose error to the MOMENT: the same
    cell of floor measured from two keyframes far apart in the walk reads the
    same height if the difference is a ramp or a step, and a different one if
    it is drift. Ramps and steps survive untouched.
    """
    enabled: bool
    cell_m: float
    texture_window_m: float
    min_shared_cells: int


@dataclass(frozen=True)
class VisitDriftConfig:
    """The correction that measures on the object's own visits and its three
    views (USER 2026-09-18). Every number the algorithm uses lives here."""
    min_points: int                 # an object under this cannot be measured
    min_visit_share: float          # a visit contributing less did not observe it
    min_walk_m: float
    voxel_m: float               # two visits closer than this on the walk
                                    # are one pass with an occlusion in between
    max_ambiguity: int            # if none reaches it, take the best few
    silhouette_cell_m: float        # raster cell of the projected silhouette
    silhouette_close_px: int        # morphological closing, in cells
    silhouette_blur_px: float       # gaussian blur, in cells
    search_margin_m: float          # margin around the pair, bounds the shift
    default_repeatability_m: float  # used only when uncertainty.json is absent
    max_epochs: int


@dataclass(frozen=True)
class CorrectionConfig:
    evidence: EvidenceConfig
    visit_drift: "VisitDriftConfig"
    floor_consensus: "FloorConsensusConfig"
    observability: ObservabilityConfig
    solve: SolveConfig
    gates: GatesConfig
    floor: FloorConfig
    revisit: RevisitConfig
    consistency: ConsistencyConfig
    kfgraph: KfGraphConfig
    photo: PhotoConfig
    posegraph: PoseGraphConfig
    apply: ApplyConfig
    runtime: RuntimeConfig


def _require(section: Dict[str, Any], key: str, path: str) -> Any:
    if not isinstance(section, dict) or key not in section:
        raise CorrectionConfigError(
            f"config.yaml is missing mandatory key 'correction.{path}.{key}' "
            f"— add it (see the correction: section documentation); there is "
            f"no hidden default in code")
    return section[key]


def _num(section, key, path, lo=None, hi=None, integer=False,
         lo_excl=False) -> float:
    v = _require(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CorrectionConfigError(
            f"'correction.{path}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise CorrectionConfigError(
            f"'correction.{path}.{key}' must be an integer, got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise CorrectionConfigError(
            f"'correction.{path}.{key}' = {v} is below the valid range "
            f"(min {lo}{' exclusive' if lo_excl else ''})")
    if hi is not None and v > hi:
        raise CorrectionConfigError(
            f"'correction.{path}.{key}' = {v} is above the valid range "
            f"(max {hi})")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _require(section, key, path)
    if not isinstance(v, bool):
        raise CorrectionConfigError(
            f"'correction.{path}.{key}' must be a boolean, got {v!r}")
    return v


def _resolve_workers(v: Any) -> int:
    if v == "auto":
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:      # non-Linux dev box
            avail = os.cpu_count() or 1
        return max(1, min(8, avail))
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise CorrectionConfigError(
            "'correction.runtime.workers' must be 'auto' or a positive "
            f"integer, got {v!r}")
    return v


def load_correction_config(raw: Optional[Dict[str, Any]] = None) -> CorrectionConfig:
    """Build a validated CorrectionConfig from the raw config dict (defaults
    to the server-wide ``config.cfg``). Raises CorrectionConfigError naming
    the first offending key."""
    if raw is None:
        from config import cfg as raw_cfg     # server/config.py
        raw = raw_cfg
    section = (raw or {}).get("correction")
    if not isinstance(section, dict):
        raise CorrectionConfigError(
            "config.yaml has no 'correction:' section — the correction module "
            "cannot run without its parameters")

    fc_ = section.get("floor_consensus")
    floor_consensus = FloorConsensusConfig(
        enabled=_bool(fc_, "enabled", "floor_consensus"),
        cell_m=_num(fc_, "cell_m", "floor_consensus", lo=0, lo_excl=True),
        texture_window_m=_num(fc_, "texture_window_m", "floor_consensus", lo=0, lo_excl=True),
        min_shared_cells=_num(fc_, "min_shared_cells", "floor_consensus", lo=1, integer=True))
    vd_ = section.get("visit_drift")
    visit_drift = VisitDriftConfig(
        min_points=_num(vd_, "min_points", "visit_drift", lo=1, integer=True),
        min_visit_share=_num(vd_, "min_visit_share", "visit_drift", lo=0.0, hi=1.0),
        min_walk_m=_num(vd_, "min_walk_m", "visit_drift", lo=0.0),
        voxel_m=_num(vd_, "voxel_m", "visit_drift", lo=0, lo_excl=True),
        max_ambiguity=_num(vd_, "max_ambiguity", "visit_drift", lo=1, integer=True),
        silhouette_cell_m=_num(vd_, "silhouette_cell_m", "visit_drift", lo=0.0, lo_excl=True),
        silhouette_close_px=_num(vd_, "silhouette_close_px", "visit_drift", lo=0, integer=True),
        silhouette_blur_px=_num(vd_, "silhouette_blur_px", "visit_drift", lo=0.0),
        search_margin_m=_num(vd_, "search_margin_m", "visit_drift", lo=0.0, lo_excl=True),
        default_repeatability_m=_num(vd_, "default_repeatability_m", "visit_drift",
                                     lo=0.0, lo_excl=True),
        max_epochs=_num(vd_, "max_epochs", "visit_drift", lo=1, integer=True),
    )

    ev = section.get("evidence")
    evidence = EvidenceConfig(
        obb_margin_m=_num(ev, "obb_margin_m", "evidence", lo=0.0, hi=1.0),
        obb_core_pct=_num(ev, "obb_core_pct", "evidence", lo=50.0, hi=100.0),
        min_object_points_solve=_num(ev, "min_object_points_solve", "evidence",
                                     lo=10, integer=True),
        min_object_points_fingerprint=_num(ev, "min_object_points_fingerprint",
                                           "evidence", lo=10, integer=True),
        min_baseline_m=_num(ev, "min_baseline_m", "evidence", lo=0.0,
                            lo_excl=True),
        min_objects_for_depth=_num(ev, "min_objects_for_depth", "evidence",
                                   lo=2, integer=True),
        visit_gap_kf=_num(ev, "visit_gap_kf", "evidence", lo=0, integer=True),
    )

    ob = section.get("observability")
    observability = ObservabilityConfig(
        pca_ratio_planar=_num(ob, "pca_ratio_planar", "observability",
                              lo=0.0, hi=1.0, lo_excl=True),
        pca_ratio_cylindrical=_num(ob, "pca_ratio_cylindrical",
                                   "observability", lo=0.0, hi=1.0,
                                   lo_excl=True),
        bounded_extent_tol=_num(ob, "bounded_extent_tol", "observability",
                                lo=0.0, hi=1.0, lo_excl=True),
        yaw_anisotropy_max=_num(ob, "yaw_anisotropy_max", "observability",
                                lo=0.0, hi=1.0, lo_excl=True),
    )
    if observability.pca_ratio_planar >= observability.pca_ratio_cylindrical:
        raise CorrectionConfigError(
            "'correction.observability.pca_ratio_planar' must be below "
            "'pca_ratio_cylindrical' (planar is the stricter shape)")

    so = section.get("solve")
    solve = SolveConfig(
        icp_iters=_num(so, "icp_iters", "solve", lo=1, integer=True),
        icp_trim=_num(so, "icp_trim", "solve", lo=0.0, hi=1.0, lo_excl=True),
        icp_sample=_num(so, "icp_sample", "solve", lo=100, integer=True),
        eval_sample=_num(so, "eval_sample", "solve", lo=100, integer=True),
        icp_min_corr=_num(so, "icp_min_corr", "solve", lo=3, integer=True),
        icp_converge_deg=_num(so, "icp_converge_deg", "solve", lo=0.0,
                              lo_excl=True),
        icp_converge_m=_num(so, "icp_converge_m", "solve", lo=0.0,
                            lo_excl=True),
        plane_ransac_tol_m=_num(so, "plane_ransac_tol_m", "solve", lo=0.0,
                                lo_excl=True),
        plane_ransac_iters=_num(so, "plane_ransac_iters", "solve", lo=1,
                                integer=True),
        plane_ransac_sample=_num(so, "plane_ransac_sample", "solve", lo=100,
                                 integer=True),
        depth_compress_tol=_num(so, "depth_compress_tol", "solve", lo=0.0,
                                hi=1.0, lo_excl=True),
        seed=_num(so, "seed", "solve", lo=0, integer=True),
    )

    ga = section.get("gates")
    gmode = _require(ga, "mode", "gates")
    if gmode not in ("advisory", "veto"):
        raise CorrectionConfigError(
            f"'correction.gates.mode' must be 'advisory' or 'veto', got "
            f"{gmode!r}")
    gates = GatesConfig(
        mode=gmode,
        max_object_residual_m=_num(ga, "max_object_residual_m", "gates",
                                   lo=0.0, lo_excl=True),
        residual_improvement_ratio=_num(ga, "residual_improvement_ratio",
                                        "gates", lo=0.0, hi=1.0,
                                        lo_excl=True),
        collapse_floor_m=_num(ga, "collapse_floor_m", "gates", lo=0.0,
                              lo_excl=True),
        max_rot_deg=_num(ga, "max_rot_deg", "gates", lo=0.0, hi=180.0,
                         lo_excl=True),
        max_translation_m=_num(ga, "max_translation_m", "gates", lo=0.0,
                               lo_excl=True),
        heldout_floor_tol_m=_num(ga, "heldout_floor_tol_m", "gates", lo=0.0,
                                 lo_excl=True),
        heldout_floor_abs_m=_num(ga, "heldout_floor_abs_m", "gates", lo=0.0,
                                 lo_excl=True),
        heldout_object_tol_m=_num(ga, "heldout_object_tol_m", "gates", lo=0.0,
                                  lo_excl=True),
        max_step_mm=_num(ga, "max_step_mm", "gates", lo=0.0, lo_excl=True),
        max_step_deg=_num(ga, "max_step_deg", "gates", lo=0.0, lo_excl=True),
        scale_agree_tol=_num(ga, "scale_agree_tol", "gates", lo=0.0,
                             lo_excl=True),
        scale_mad_tol=_num(ga, "scale_mad_tol", "gates", lo=0.0,
                           lo_excl=True),
    )

    fl = section.get("floor")
    model_default = _require(fl, "model_default", "floor")
    if model_default not in _FLOOR_MODELS:
        raise CorrectionConfigError(
            f"'correction.floor.model_default' must be one of "
            f"{_FLOOR_MODELS}, got {model_default!r}")
    floor = FloorConfig(
        model_default=model_default,
        band_m=_num(fl, "band_m", "floor", lo=0.0, lo_excl=True),
        low_band_pct=_num(fl, "low_band_pct", "floor", lo=0.0, hi=50.0),
        min_tilt_deg=_num(fl, "min_tilt_deg", "floor", lo=0.0, hi=90.0),
        max_tilt_deg=_num(fl, "max_tilt_deg", "floor", lo=0.0, hi=90.0,
                          lo_excl=True),
        min_inliers=_num(fl, "min_inliers", "floor", lo=10, integer=True),
        min_inlier_ratio=_num(fl, "min_inlier_ratio", "floor", lo=0.0,
                              hi=1.0, lo_excl=True),
        ransac_tol_m=_num(fl, "ransac_tol_m", "floor", lo=0.0, lo_excl=True),
        ransac_refit_band_m=_num(fl, "ransac_refit_band_m", "floor", lo=0.0,
                                 lo_excl=True),
        ransac_iters=_num(fl, "ransac_iters", "floor", lo=1, integer=True),
        ransac_sample=_num(fl, "ransac_sample", "floor", lo=100,
                           integer=True),
        smooth_window_kf=_num(fl, "smooth_window_kf", "floor", lo=0,
                              integer=True),
        step_demote_m=_num(fl, "step_demote_m", "floor", lo=0.0,
                           lo_excl=True),
        reference_span_kf=_num(fl, "reference_span_kf", "floor", lo=1,
                               integer=True),
    )

    rv = section.get("revisit")
    revisit = RevisitConfig(
        sample_per_kf=_num(rv, "sample_per_kf", "revisit", lo=50, integer=True),
        min_gap_kf=_num(rv, "min_gap_kf", "revisit", lo=1, integer=True),
        min_covis=_num(rv, "min_covis", "revisit", lo=0.0, hi=1.0,
                       lo_excl=True),
        depth_min_m=_num(rv, "depth_min_m", "revisit", lo=0.0),
        depth_max_m=_num(rv, "depth_max_m", "revisit", lo=0.0, lo_excl=True),
        zbuffer_cell_px=_num(rv, "zbuffer_cell_px", "revisit", lo=1,
                             integer=True),
        same_surface_tol_m=_num(rv, "same_surface_tol_m", "revisit", lo=0.0,
                                lo_excl=True),
        same_surface_tol_rel=_num(rv, "same_surface_tol_rel", "revisit",
                                  lo=0.0, hi=1.0, lo_excl=True),
        region_cell_m=_num(rv, "region_cell_m", "revisit", lo=0.0,
                           lo_excl=True),
        region_sample=_num(rv, "region_sample", "revisit", lo=100,
                           integer=True),
        offset_min_m=_num(rv, "offset_min_m", "revisit", lo=0.0, lo_excl=True),
        image_max_px=_num(rv, "image_max_px", "revisit", lo=64, integer=True),
    )
    if revisit.depth_max_m <= revisit.depth_min_m:
        raise CorrectionConfigError("'correction.revisit.depth_max_m' must "
                                    "exceed depth_min_m")

    cn = section.get("consistency")
    consistency = ConsistencyConfig(
        cell_m=_num(cn, "cell_m", "consistency", lo=0.0, lo_excl=True),
        block_min_points=_num(cn, "block_min_points", "consistency", lo=1,
                              integer=True),
        block_sample=_num(cn, "block_sample", "consistency", lo=100,
                          integer=True),
        writer_min_points=_num(cn, "writer_min_points", "consistency", lo=1,
                               integer=True),
        writer_query_sample=_num(cn, "writer_query_sample", "consistency",
                                 lo=10, integer=True),
        viewer_probe=_num(cn, "viewer_probe", "consistency", lo=1,
                          integer=True),
        viewer_min_frac=_num(cn, "viewer_min_frac", "consistency", lo=0.0,
                             hi=1.0, lo_excl=True),
        near_kf=_num(cn, "near_kf", "consistency", lo=1, integer=True),
    )

    kg = section.get("kfgraph")
    kfgraph = KfGraphConfig(
        pair_src_sample=_num(kg, "pair_src_sample", "kfgraph", lo=100, integer=True),
        pair_tgt_sample=_num(kg, "pair_tgt_sample", "kfgraph", lo=100, integer=True),
        normal_k=_num(kg, "normal_k", "kfgraph", lo=3, integer=True),
        gn_iters=_num(kg, "gn_iters", "kfgraph", lo=1, integer=True),
        gn_converge=_num(kg, "gn_converge", "kfgraph", lo=0.0, lo_excl=True),
        damping=_num(kg, "damping", "kfgraph", lo=0.0, lo_excl=True),
        damping_min=_num(kg, "damping_min", "kfgraph", lo=0.0, lo_excl=True),
        lm_factor=_num(kg, "lm_factor", "kfgraph", lo=1.0, lo_excl=True),
        lm_tries=_num(kg, "lm_tries", "kfgraph", lo=1, integer=True),
        floor_m=_num(kg, "floor_m", "kfgraph", lo=0.0, lo_excl=True),
    )

    ph = section.get("photo")
    photo = PhotoConfig(
        candidate_sample=_num(ph, "candidate_sample", "photo", lo=10, integer=True),
        candidate_min=_num(ph, "candidate_min", "photo", lo=1, integer=True),
        pair_min_points=_num(ph, "pair_min_points", "photo", lo=1, integer=True),
        points_per_pair=_num(ph, "points_per_pair", "photo", lo=10, integer=True),
        splat_radius=_num(ph, "splat_radius", "photo", lo=0, integer=True),
        outer_iters=_num(ph, "outer_iters", "photo", lo=1, integer=True),
        huber_px=_num(ph, "huber_px", "photo", lo=0.0, lo_excl=True),
    )

    pg = section.get("posegraph")
    posegraph = PoseGraphConfig(
        loop_weight=_num(pg, "loop_weight", "posegraph", lo=0.0, lo_excl=True),
        rot_lever_m=_num(pg, "rot_lever_m", "posegraph", lo=0.0, lo_excl=True),
        min_blocks_improved=_num(pg, "min_blocks_improved", "posegraph",
                                 lo=0.0, hi=1.0, lo_excl=True),
    )

    ap = section.get("apply")
    depth_mode = _require(ap, "depth_correction_mode", "apply")
    if depth_mode not in _DEPTH_MODES:
        raise CorrectionConfigError(
            f"'correction.apply.depth_correction_mode' must be one of "
            f"{_DEPTH_MODES}, got {depth_mode!r}")
    if depth_mode == "rewrite":
        raise CorrectionConfigError(
            "'correction.apply.depth_correction_mode: rewrite' is reserved "
            "and NOT implemented — use 'sidecar' (the accessor in "
            "segmentation/session_io.py serves corrected depth to every "
            "consumer)")
    potree_rebuild = _require(ap, "potree_rebuild", "apply")
    if not isinstance(potree_rebuild, bool):
        raise CorrectionConfigError(
            f"'correction.apply.potree_rebuild' must be a boolean, got "
            f"{potree_rebuild!r}")
    reconsolidate = _require(ap, "reconsolidate", "apply")
    if not isinstance(reconsolidate, bool):
        raise CorrectionConfigError(
            f"'correction.apply.reconsolidate' must be a boolean, got "
            f"{reconsolidate!r}")
    apply_cfg = ApplyConfig(depth_correction_mode=depth_mode,
                            potree_rebuild=potree_rebuild,
                            reconsolidate=reconsolidate)

    rt = section.get("runtime")
    runtime = RuntimeConfig(
        workers=_resolve_workers(_require(rt, "workers", "runtime")))

    return CorrectionConfig(evidence=evidence, visit_drift=visit_drift,
                            floor_consensus=floor_consensus,
                            observability=observability,
                            solve=solve, gates=gates, floor=floor,
                            revisit=revisit, consistency=consistency,
                            kfgraph=kfgraph, photo=photo, posegraph=posegraph,
                            apply=apply_cfg, runtime=runtime)
