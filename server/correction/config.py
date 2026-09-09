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
class ApplyConfig:
    depth_correction_mode: str
    potree_rebuild: bool


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int          # resolved: 'auto' becomes min(8, affinity)


@dataclass(frozen=True)
class CorrectionConfig:
    evidence: EvidenceConfig
    observability: ObservabilityConfig
    solve: SolveConfig
    gates: GatesConfig
    floor: FloorConfig
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
    apply_cfg = ApplyConfig(depth_correction_mode=depth_mode,
                            potree_rebuild=potree_rebuild)

    rt = section.get("runtime")
    runtime = RuntimeConfig(
        workers=_resolve_workers(_require(rt, "workers", "runtime")))

    return CorrectionConfig(evidence=evidence, observability=observability,
                            solve=solve, gates=gates, floor=floor,
                            apply=apply_cfg, runtime=runtime)
