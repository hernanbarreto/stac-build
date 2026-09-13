"""Fusion gates — pure, configurable, ALL veto (F2/F7 closed).

A tolerance is not a fix (doctrine): an out-of-threshold value is reported
with its diagnosis and blocks the build; the only way past a failed scale
gate is an explicit override recorded in the report and the ledger. Each
gate returns ``{"name", "passed", "detail", ...numbers}`` and the whole list
lands in the fusion report, also on rejection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from fusion.config import FusionConfig

PROFILE_VALIDATED_BOTH = "validated_both"
PROFILE_VALIDATED_REFERENCE = "validated_reference"
PROFILE_UNVALIDATED = "unvalidated"

# scale sources that count as physically validated (bim_registration is
# reserved for the future and is a requirement of nothing)
VALIDATED_SOURCES = ("user_measurement", "multiscan_consensus", "vio")


def scale_profile(ref_source: Optional[str],
                  scan_source: Optional[str]) -> str:
    ref_ok = ref_source in VALIDATED_SOURCES
    scan_ok = scan_source in VALIDATED_SOURCES
    if ref_ok and scan_ok:
        return PROFILE_VALIDATED_BOTH
    if ref_ok:
        return PROFILE_VALIDATED_REFERENCE
    return PROFILE_UNVALIDATED


def gate_scale(s_fusion: float, profile: str, cfg: FusionConfig,
               sigma_ref: Optional[float], sigma_scan: Optional[float],
               override: bool, operator: str) -> dict:
    """|s_fusion − 1| against the active profile's tolerance. Always
    MEASURED and reported, applied or not. When both scans are validated and
    the residual exceeds σ_ref + σ_scan + tol, the report must say one of
    the two measurements is wrong (the caller adds which is consistent with
    the rest of the evidence)."""
    tol = (cfg.scale.scale_tol if profile == PROFILE_VALIDATED_BOTH
           else cfg.scale.scale_tol_unvalidated)
    dev = abs(s_fusion - 1.0)
    passed = dev <= tol
    detail = (f"s_fusion {s_fusion:.4f} (|s-1| {dev:.4f}, tol ±{tol:g}, "
              f"profile {profile})")
    result = {"name": "scale", "passed": passed, "s_fusion": round(s_fusion, 5),
              "tol": tol, "profile": profile, "detail": detail}
    if profile == PROFILE_VALIDATED_BOTH and sigma_ref is not None \
            and sigma_scan is not None \
            and dev > sigma_ref + sigma_scan + tol:
        result["measurement_conflict"] = (
            f"both scans are validated yet they disagree by {dev:.1%} > "
            f"σ_ref+σ_scan+tol — one of the two user measurements is wrong")
        result["detail"] += " — " + result["measurement_conflict"]
    if not passed and override:
        result["passed"] = True
        result["overridden_by"] = operator
        result["detail"] += f" — OVERRIDDEN by {operator} (recorded)"
    return result


def gate_tilt(tilt_deg: float, cfg: FusionConfig) -> dict:
    passed = tilt_deg <= cfg.align.max_tilt_deg
    return {"name": "tilt", "passed": passed,
            "tilt_deg": round(tilt_deg, 3),
            "detail": f"pitch/roll {tilt_deg:.2f}° (max "
                      f"{cfg.align.max_tilt_deg:g}°; yaw is free between "
                      f"days)"}


def gate_pair_residuals(pair_residuals: List[dict],
                        cfg: FusionConfig) -> dict:
    """Median NN per pair on the PRIMITIVE SUPPORTS (cleaned inliers), never
    the raw subclouds (F3). pair_residuals entries:
    {label, residual_m_before, residual_m_after, excluded}."""
    kept = [p for p in pair_residuals if not p.get("excluded")]
    bad = [p for p in kept
           if p["residual_m_after"] > cfg.gates.pair_residual_max_m]
    passed = bool(kept) and not bad
    detail = (f"{len(kept)} pair(s), residuals "
              + str([(p['label'], round(p['residual_m_after'] * 100, 1))
                     for p in kept])
              + f" cm (max {cfg.gates.pair_residual_max_m*100:.0f} cm)")
    if bad:
        detail += f"; FAILED: {[p['label'] for p in bad]}"
    if not kept:
        detail = "no pair left as evidence"
    return {"name": "pair_residuals", "passed": passed,
            "failed_pairs": [p["label"] for p in bad], "detail": detail}


def gate_heldout(nn_median_m: Optional[float],
                 floor_dist_m: Optional[float],
                 unpaired_objects: List[dict],
                 cfg: FusionConfig) -> dict:
    """The rest of the scene is the exam: unpaired overlap points, the
    scan's floor vs the reference floor plane, and common unpaired
    instances. ``unpaired_objects`` entries: {label, residual_m}."""
    why = []
    if nn_median_m is not None and nn_median_m > cfg.gates.heldout_nn_max_m:
        why.append(f"unpaired overlap median NN {nn_median_m*100:.1f} cm > "
                   f"{cfg.gates.heldout_nn_max_m*100:.0f} cm")
    if floor_dist_m is not None \
            and floor_dist_m > cfg.gates.heldout_floor_tol_m:
        why.append(f"floor vs reference floor plane {floor_dist_m*100:.1f} "
                   f"cm > {cfg.gates.heldout_floor_tol_m*100:.0f} cm")
    bad_obj = [o for o in unpaired_objects
               if o["residual_m"] > cfg.gates.heldout_object_tol_m]
    if bad_obj:
        why.append(f"common unpaired instance(s) off: "
                   f"{[(o['label'], round(o['residual_m']*100, 1)) for o in bad_obj]} cm "
                   f"> {cfg.gates.heldout_object_tol_m*100:.0f} cm")
    detail = ("; ".join(why) if why else
              f"unpaired overlap "
              f"{(nn_median_m or 0)*100:.1f} cm, floor "
              f"{(floor_dist_m or 0)*100:.1f} cm, "
              f"{len(unpaired_objects)} unpaired witness instance(s) ok")
    return {"name": "heldout", "passed": not why,
            "nn_median_m": nn_median_m, "floor_dist_m": floor_dist_m,
            "unpaired_objects": unpaired_objects, "detail": detail}


def gate_uniformity(diagnosis: dict, crosscorrected: bool) -> dict:
    """diagnose declared non-uniform and crosscorrect could not (or was not
    allowed to) resolve it → veto."""
    uniform = bool(diagnosis.get("uniform", True))
    passed = uniform or crosscorrected
    detail = diagnosis.get("detail", "uniform")
    if not uniform:
        detail += (" — resolved by the per-keyframe cross-correction"
                   if crosscorrected else
                   " — NOT resolved; correct the scan before fusing")
    return {"name": "uniformity", "passed": passed, "uniform": uniform,
            "crosscorrected": crosscorrected, "detail": detail}


def gate_coverage(n_usable_pairs: int, spread_ratio: float,
                  cfg: FusionConfig) -> dict:
    why = []
    if n_usable_pairs < cfg.pairs.min_pairs:
        why.append(f"{n_usable_pairs} usable pair(s) < min_pairs "
                   f"{cfg.pairs.min_pairs}")
    spread_bad = spread_ratio < cfg.pairs.min_pair_spread_ratio
    if spread_bad and cfg.pairs.pair_spread_mode == "veto":
        why.append(f"pair spread {spread_ratio:.0%} < "
                   f"{cfg.pairs.min_pair_spread_ratio:.0%} (veto mode)")
    detail = ("; ".join(why) if why else
              f"{n_usable_pairs} pair(s), spread {spread_ratio:.0%}"
              + (f" (below {cfg.pairs.min_pair_spread_ratio:.0%} — warning)"
                 if spread_bad else ""))
    return {"name": "coverage", "passed": not why,
            "n_pairs": n_usable_pairs,
            "spread_ratio": round(spread_ratio, 3), "detail": detail}


def gate_integrity(n_ref: int, n_scan: int, n_fused: int,
                   provenance_ok: bool) -> dict:
    counts_ok = (n_fused == n_ref + n_scan)
    passed = counts_ok and provenance_ok
    return {"name": "integrity", "passed": passed,
            "detail": (f"{n_ref:,} + {n_scan:,} = {n_fused:,} points, "
                       f"provenance {'intact' if provenance_ok else 'BROKEN'}"
                       if counts_ok else
                       f"point counts do not add up: {n_ref:,} + {n_scan:,} "
                       f"≠ {n_fused:,}")}
