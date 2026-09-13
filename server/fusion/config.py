"""Typed, validated load of ``metric_validation:`` and ``fusion:``.

Decides nothing geometric — it guarantees every decision parameter exists,
is typed and sits in range BEFORE any cloud is touched. No defaults in code:
a missing key aborts naming the key (the YAML documents defaults and their
why). Mirrors ``correction/config.py`` deliberately: one loader style across
the metric chain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class FusionConfigError(RuntimeError):
    """Missing/invalid key in metric_validation:/fusion: — message names it."""


_SPREAD_MODES = ("warn", "veto")


@dataclass(frozen=True)
class ConsensusConfig:
    min_scans: int
    min_objects: int
    huber_delta: float
    residual_outlier_sigma: float
    nonuniform_corr_min: float


@dataclass(frozen=True)
class PrimitivesConfig:
    floater_sor_k: int
    floater_sor_sigma: float
    keep_main_component: bool
    component_radius_m: float
    min_points: int


@dataclass(frozen=True)
class MetricValidationConfig:
    min_measure_length_m: float
    max_sigma_s: float
    max_metric_correction: float
    uniformity_tol: float
    fx_mismatch_warn: float
    consensus: ConsensusConfig
    primitives: PrimitivesConfig


@dataclass(frozen=True)
class PairsConfig:
    min_pairs: int
    recommended_pairs: int
    min_pair_spread_ratio: float
    pair_spread_mode: str
    movable_labels: List[str]
    shoot_dist_ratio_max: float
    auto_exclude_outlier_pairs: bool
    pair_outlier_sigma: float
    min_pair_points: int


@dataclass(frozen=True)
class ScaleConfig:
    scale_tol: float
    scale_tol_unvalidated: float
    allow_reference_correction: bool


@dataclass(frozen=True)
class AlignConfig:
    max_tilt_deg: float
    min_nonparallel_normals: int
    min_normal_angle_deg: float


@dataclass(frozen=True)
class RefineConfig:
    overlap_radius_m: float
    icp_trim: float
    icp_max_iters: int
    icp_convergence_m: float
    icp_samples: int


@dataclass(frozen=True)
class FusionGatesConfig:
    pair_residual_max_m: float
    heldout_nn_max_m: float
    heldout_floor_tol_m: float
    heldout_object_tol_m: float
    heldout_trim: float


@dataclass(frozen=True)
class CrossCorrectConfig:
    enabled: bool
    trigger_pair_residual_m: float


@dataclass(frozen=True)
class BuildConfig:
    potree: bool


@dataclass(frozen=True)
class FusionRuntimeConfig:
    workers: int
    seed: int


@dataclass(frozen=True)
class FusionConfig:
    metric: MetricValidationConfig
    pairs: PairsConfig
    scale: ScaleConfig
    align: AlignConfig
    refine: RefineConfig
    gates: FusionGatesConfig
    crosscorrect: CrossCorrectConfig
    build: BuildConfig
    runtime: FusionRuntimeConfig


def _req(section: Any, key: str, path: str) -> Any:
    if not isinstance(section, dict) or key not in section:
        raise FusionConfigError(
            f"config.yaml is missing mandatory key '{path}.{key}' — add it "
            f"(see the {path.split('.')[0]}: section documentation); there "
            f"is no hidden default in code")
    return section[key]


def _num(section, key, path, lo=None, hi=None, integer=False,
         lo_excl=False):
    v = _req(section, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FusionConfigError(f"'{path}.{key}' must be a number, got {v!r}")
    if integer and int(v) != v:
        raise FusionConfigError(f"'{path}.{key}' must be an integer, "
                                f"got {v!r}")
    if lo is not None and (v <= lo if lo_excl else v < lo):
        raise FusionConfigError(
            f"'{path}.{key}' = {v} is below the valid range (min {lo}"
            f"{' exclusive' if lo_excl else ''})")
    if hi is not None and v > hi:
        raise FusionConfigError(
            f"'{path}.{key}' = {v} is above the valid range (max {hi})")
    return int(v) if integer else float(v)


def _bool(section, key, path) -> bool:
    v = _req(section, key, path)
    if not isinstance(v, bool):
        raise FusionConfigError(f"'{path}.{key}' must be a boolean, "
                                f"got {v!r}")
    return v


def _workers(v: Any, path: str) -> int:
    if v == "auto":
        try:
            avail = len(os.sched_getaffinity(0))
        except AttributeError:
            avail = os.cpu_count() or 1
        return max(1, min(8, avail))
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise FusionConfigError(
            f"'{path}.workers' must be 'auto' or a positive integer, "
            f"got {v!r}")
    return v


def load_fusion_config(raw: Optional[Dict[str, Any]] = None) -> FusionConfig:
    """Validated FusionConfig from the raw config dict (defaults to the
    server-wide ``config.cfg``). Raises FusionConfigError naming the first
    offending key."""
    if raw is None:
        from config import cfg as raw_cfg
        raw = raw_cfg
    mv = (raw or {}).get("metric_validation")
    fu = (raw or {}).get("scan_fusion")
    if not isinstance(mv, dict):
        raise FusionConfigError("config.yaml has no 'metric_validation:' "
                                "section")
    if not isinstance(fu, dict):
        raise FusionConfigError("config.yaml has no 'scan_fusion:' section")

    co = mv.get("consensus")
    consensus = ConsensusConfig(
        min_scans=_num(co, "min_scans", "metric_validation.consensus",
                       lo=2, integer=True),
        min_objects=_num(co, "min_objects", "metric_validation.consensus",
                         lo=1, integer=True),
        huber_delta=_num(co, "huber_delta", "metric_validation.consensus",
                         lo=0.0, lo_excl=True),
        residual_outlier_sigma=_num(co, "residual_outlier_sigma",
                                    "metric_validation.consensus", lo=0.0,
                                    lo_excl=True),
        nonuniform_corr_min=_num(co, "nonuniform_corr_min",
                                 "metric_validation.consensus", lo=0.0,
                                 hi=1.0),
    )
    pr = mv.get("primitives")
    primitives = PrimitivesConfig(
        floater_sor_k=_num(pr, "floater_sor_k",
                           "metric_validation.primitives", lo=1,
                           integer=True),
        floater_sor_sigma=_num(pr, "floater_sor_sigma",
                               "metric_validation.primitives", lo=0.0,
                               lo_excl=True),
        keep_main_component=_bool(pr, "keep_main_component",
                                  "metric_validation.primitives"),
        component_radius_m=_num(pr, "component_radius_m",
                                "metric_validation.primitives", lo=0.0,
                                lo_excl=True),
        min_points=_num(pr, "min_points", "metric_validation.primitives",
                        lo=10, integer=True),
    )
    metric = MetricValidationConfig(
        min_measure_length_m=_num(mv, "min_measure_length_m",
                                  "metric_validation", lo=0.0, lo_excl=True),
        max_sigma_s=_num(mv, "max_sigma_s", "metric_validation", lo=0.0,
                         lo_excl=True),
        max_metric_correction=_num(mv, "max_metric_correction",
                                   "metric_validation", lo=0.0,
                                   lo_excl=True),
        uniformity_tol=_num(mv, "uniformity_tol", "metric_validation",
                            lo=0.0, lo_excl=True),
        fx_mismatch_warn=_num(mv, "fx_mismatch_warn", "metric_validation",
                              lo=0.0, lo_excl=True),
        consensus=consensus, primitives=primitives,
    )

    pa = fu.get("pairs")
    labels = _req(pa, "movable_labels", "scan_fusion.pairs")
    if not isinstance(labels, list) or not all(isinstance(x, str)
                                              for x in labels):
        raise FusionConfigError("'fusion.pairs.movable_labels' must be a "
                                "list of strings")
    spread_mode = _req(pa, "pair_spread_mode", "scan_fusion.pairs")
    if spread_mode not in _SPREAD_MODES:
        raise FusionConfigError(
            f"'fusion.pairs.pair_spread_mode' must be one of "
            f"{_SPREAD_MODES}, got {spread_mode!r}")
    pairs = PairsConfig(
        min_pairs=_num(pa, "min_pairs", "scan_fusion.pairs", lo=1, integer=True),
        recommended_pairs=_num(pa, "recommended_pairs", "scan_fusion.pairs",
                               lo=1, integer=True),
        min_pair_spread_ratio=_num(pa, "min_pair_spread_ratio",
                                   "scan_fusion.pairs", lo=0.0, hi=1.0),
        pair_spread_mode=spread_mode,
        movable_labels=[str(x).strip().lower() for x in labels],
        shoot_dist_ratio_max=_num(pa, "shoot_dist_ratio_max", "scan_fusion.pairs",
                                  lo=1.0),
        auto_exclude_outlier_pairs=_bool(pa, "auto_exclude_outlier_pairs",
                                         "scan_fusion.pairs"),
        pair_outlier_sigma=_num(pa, "pair_outlier_sigma", "scan_fusion.pairs",
                                lo=0.0, lo_excl=True),
        min_pair_points=_num(pa, "min_pair_points", "scan_fusion.pairs", lo=10,
                             integer=True),
    )

    sc = fu.get("scale")
    scale = ScaleConfig(
        scale_tol=_num(sc, "scale_tol", "scan_fusion.scale", lo=0.0,
                       lo_excl=True),
        scale_tol_unvalidated=_num(sc, "scale_tol_unvalidated",
                                   "scan_fusion.scale", lo=0.0, lo_excl=True),
        allow_reference_correction=_bool(sc, "allow_reference_correction",
                                         "scan_fusion.scale"),
    )

    al = fu.get("align")
    align = AlignConfig(
        max_tilt_deg=_num(al, "max_tilt_deg", "scan_fusion.align", lo=0.0,
                          hi=90.0, lo_excl=True),
        min_nonparallel_normals=_num(al, "min_nonparallel_normals",
                                     "scan_fusion.align", lo=1, integer=True),
        min_normal_angle_deg=_num(al, "min_normal_angle_deg", "scan_fusion.align",
                                  lo=0.0, hi=90.0, lo_excl=True),
    )

    re_ = fu.get("refine")
    refine = RefineConfig(
        overlap_radius_m=_num(re_, "overlap_radius_m", "scan_fusion.refine",
                              lo=0.0, lo_excl=True),
        icp_trim=_num(re_, "icp_trim", "scan_fusion.refine", lo=0.0, hi=1.0,
                      lo_excl=True),
        icp_max_iters=_num(re_, "icp_max_iters", "scan_fusion.refine", lo=1,
                           integer=True),
        icp_convergence_m=_num(re_, "icp_convergence_m", "scan_fusion.refine",
                               lo=0.0, lo_excl=True),
        icp_samples=_num(re_, "icp_samples", "scan_fusion.refine", lo=100,
                         integer=True),
    )

    ga = fu.get("gates")
    gates = FusionGatesConfig(
        pair_residual_max_m=_num(ga, "pair_residual_max_m", "scan_fusion.gates",
                                 lo=0.0, lo_excl=True),
        heldout_nn_max_m=_num(ga, "heldout_nn_max_m", "scan_fusion.gates",
                              lo=0.0, lo_excl=True),
        heldout_floor_tol_m=_num(ga, "heldout_floor_tol_m", "scan_fusion.gates",
                                 lo=0.0, lo_excl=True),
        heldout_object_tol_m=_num(ga, "heldout_object_tol_m", "scan_fusion.gates",
                                  lo=0.0, lo_excl=True),
        heldout_trim=_num(ga, "heldout_trim", "scan_fusion.gates", lo=0.0,
                          hi=1.0, lo_excl=True),
    )

    cc = fu.get("crosscorrect")
    crosscorrect = CrossCorrectConfig(
        enabled=_bool(cc, "enabled", "scan_fusion.crosscorrect"),
        trigger_pair_residual_m=_num(cc, "trigger_pair_residual_m",
                                     "scan_fusion.crosscorrect", lo=0.0,
                                     lo_excl=True),
    )

    bu = fu.get("build")
    build = BuildConfig(potree=_bool(bu, "potree", "scan_fusion.build"))

    rt = fu.get("runtime")
    runtime = FusionRuntimeConfig(
        workers=_workers(_req(rt, "workers", "scan_fusion.runtime"),
                         "scan_fusion.runtime"),
        seed=_num(rt, "seed", "scan_fusion.runtime", lo=0, integer=True),
    )

    return FusionConfig(metric=metric, pairs=pairs, scale=scale, align=align,
                        refine=refine, gates=gates, crosscorrect=crosscorrect,
                        build=build, runtime=runtime)
