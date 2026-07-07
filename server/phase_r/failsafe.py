# STAC-Builder — Phase R.9: A/B validation + mandatory fail-safe.
#
# If the semantic-anchoring refinement WORSENS any reference metric, the pipeline
# falls back automatically to the no-anchor path and reports it. Anchoring may
# NEVER leave the reconstruction worse than the baseline without saying so.
#
# Reference metrics (spec R.9): seam error, vote-entropy, onion metric, per-window
# scale S stability, post-BA reprojection. Each is "lower is better".
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass, field

# reference metrics and whether lower is better (all True here)
REFERENCE_METRICS = {
    "seam_error_median": True,
    "vote_entropy_mean": True,
    "onion_separation_median": True,
    "scale_S_std": True,
    "reprojection_rmse": True,
}


@dataclass
class GateResult:
    use_anchoring: bool
    regressions: list[dict] = field(default_factory=list)
    improvements: list[dict] = field(default_factory=list)
    report: str = ""


def compare_and_gate(baseline: dict[str, float], refined: dict[str, float],
                     rel_tolerance: float = 0.0) -> GateResult:
    """Compare refined vs baseline on the reference metrics. If refined is worse
    on ANY metric (beyond rel_tolerance), fall back to baseline and report it."""
    regressions, improvements = [], []
    for key, lower_better in REFERENCE_METRICS.items():
        if key not in baseline or key not in refined:
            continue
        b, r = baseline[key], refined[key]
        if b is None or r is None:
            continue
        # normalize to "worse = positive delta"
        delta = (r - b) if lower_better else (b - r)
        rel = delta / (abs(b) + 1e-9)
        rec = {"metric": key, "baseline": b, "refined": r, "rel_change": rel}
        if rel > rel_tolerance:
            regressions.append(rec)
        elif rel < -1e-9:
            improvements.append(rec)

    use_anchoring = len(regressions) == 0
    if use_anchoring:
        report = (f"anchoring KEPT — improved {len(improvements)} metric(s), "
                  f"no regressions")
    else:
        worst = max(regressions, key=lambda x: x["rel_change"])
        report = (f"anchoring REJECTED (fail-safe) — regressed on "
                  f"{[r['metric'] for r in regressions]}; worst {worst['metric']} "
                  f"{worst['rel_change']*100:+.1f}%. Falling back to no-anchor path.")
    return GateResult(use_anchoring=use_anchoring, regressions=regressions,
                      improvements=improvements, report=report)
