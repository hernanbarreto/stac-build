"""Diagnosis: pose vs depth via the internal fingerprint, and the DA3 anchor
cross-check.

Internal fingerprint (USER 2026-09-06 matrix): if the relations between the
marked objects inside a displaced visit match the reference relations, the
error is pure POSE (rigid); if compressed/expanded, it is DEPTH first (expand
along the shooting rays), then rigid.

DA3 cross-check (prompt §5.5): the session's metric scale comes from
scale_align's DA3 anchors. A fingerprint-derived k is a LOCAL depth
correction, never a session re-scale — so after solving we verify it does not
contradict the anchors. Key property: per-anchor agreement is a per-frame
depth RATIO, exactly divided by k on that keyframe (per-keyframe rigid warps
move a camera and its points together and cannot change camera-frame depth),
so the check is ANALYTIC over ``scale_diagnostics.json`` — no depth files are
required, freed sessions still gate.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from correction.config import CorrectionConfig

DIAGNOSTICS_NAME = "scale_diagnostics.json"


def fingerprint_k(seen_centroids: Dict[int, np.ndarray],
                  ref_fingerprint: Dict[tuple, float],
                  cfg: CorrectionConfig) -> Tuple[float, Optional[float], int]:
    """(k, median ratio, n_pairs) from the inter-object centroid distances of
    a displaced visit vs the reference. k = 1 with fewer than one usable
    baseline pair."""
    fps: List[float] = []
    for (a, b), dref in ref_fingerprint.items():
        if a in seen_centroids and b in seen_centroids \
                and dref >= cfg.evidence.min_baseline_m:
            fps.append(float(np.linalg.norm(
                seen_centroids[a] - seen_centroids[b])) / dref)
    if not fps:
        return 1.0, None, 0
    ratio = float(np.median(fps))
    return float(np.median([1.0 / f for f in fps])), ratio, len(fps)


def diagnose_visit(k: float, ratio: Optional[float], n_pairs: int,
                   depth_allowed: bool, cfg: CorrectionConfig) -> dict:
    """The visit's diagnosis record. Depth is only diagnosed when the
    observability analysis allowed it AND the compression is beyond
    tolerance."""
    depth_needed = (depth_allowed and n_pairs > 0
                    and abs(k - 1.0) > cfg.solve.depth_compress_tol)
    if n_pairs == 0:
        label = "pose (single-object evidence)"
    elif depth_needed:
        label = "depth+pose"
    else:
        label = "pose"
    return {
        "diagnosis": label,
        "depth_needed": depth_needed,
        "k": (round(k, 4) if depth_needed else 1.0),
        "fingerprint_ratio": (round(ratio, 4) if ratio is not None else None),
        "fingerprint_pairs": n_pairs,
    }


# ── DA3 anchor cross-check ───────────────────────────────────────────────

def _load_diag(output_dir: Path) -> Optional[dict]:
    p = Path(output_dir) / DIAGNOSTICS_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError as e:
        raise RuntimeError(
            f"corrupt {DIAGNOSTICS_NAME} at {p}: {e} — regenerate it by "
            f"re-running the scale stage or restore it from backup") from e


def _current_agreements(diag: dict) -> Optional[Dict[int, float]]:
    """Per-anchor agreement ratio a_f (≈1 when the anchor agrees with the
    applied scale) at the CURRENT epoch: the last epochs-history entry when
    present, else s_f / s_applied from the original estimate."""
    epochs = diag.get("epochs") or []
    if epochs:
        last = epochs[-1].get("agreement") or {}
        return {int(f): float(a) for f, a in last.items()} or None
    s_applied = diag.get("s_applied")
    frames = ((diag.get("anchors") or {}).get("frames")) or []
    if not s_applied or not frames:
        return None
    return {int(fr["num"]): float(fr["s_f"]) / float(s_applied)
            for fr in frames if fr.get("s_f")}


def scale_check(output_dir, k_by_frame: Dict[int, float],
                cfg: CorrectionConfig) -> dict:
    """Gate record for the depth correction vs the DA3 anchors.
    k_by_frame: real frame number → k (only ≠1 entries matter)."""
    k_active = {f: k for f, k in k_by_frame.items() if k != 1.0}
    if not k_active:
        return {"name": "scale_vs_da3", "passed": True,
                "detail": "rigid-only correction — per-keyframe rigid warps "
                          "cannot change camera-frame depth; DA3 agreement "
                          "is unchanged"}
    diag = _load_diag(output_dir)
    agreements = _current_agreements(diag) if diag else None
    if agreements is None:
        return {"name": "scale_vs_da3", "passed": False,
                "detail": "the correction changes depth (k ≠ 1) but "
                          f"{DIAGNOSTICS_NAME} holds no usable anchor ratios "
                          "— re-run the scale stage (or "
                          "tools/extract_da3_anchors.py) before applying a "
                          "depth correction, or send override_scale_check "
                          "explicitly"}
    affected = {f: k for f, k in k_active.items() if f in agreements}
    before_all = list(agreements.values())
    after_all = [a / k_active.get(f, 1.0) for f, a in agreements.items()]

    worst = None
    for f, k in affected.items():
        a0, a1 = agreements[f], agreements[f] / k
        worsening = abs(math.log(a1)) - abs(math.log(a0))
        if worst is None or worsening > worst["worsening"]:
            worst = {"frame": f, "before": round(a0, 4),
                     "after": round(a1, 4),
                     "worsening": round(worsening, 4)}

    def _mad(v):
        m = float(np.median(v))
        return float(np.median(np.abs(np.asarray(v) - m)))

    mad_before, mad_after = _mad(before_all), _mad(after_all)
    fail_agree = (worst is not None
                  and worst["worsening"] > cfg.gates.scale_agree_tol)
    fail_mad = (mad_after - mad_before) > cfg.gates.scale_mad_tol
    passed = not (fail_agree or fail_mad)
    detail = (f"{len(affected)} DA3 anchor(s) inside the corrected span; "
              f"worst agreement worsening "
              f"{worst['worsening'] if worst else 0} "
              f"(tol {cfg.gates.scale_agree_tol}); anchor MAD "
              f"{round(mad_before, 4)} → {round(mad_after, 4)} "
              f"(tol +{cfg.gates.scale_mad_tol})")
    if not affected:
        detail = ("no DA3 anchor falls inside the corrected keyframes; MAD "
                  f"{round(mad_before, 4)} → {round(mad_after, 4)}")
    return {"name": "scale_vs_da3", "passed": passed, "detail": detail,
            "worst_anchor": worst,
            "mad_before": round(mad_before, 4),
            "mad_after": round(mad_after, 4)}


def regenerate_scale_diagnostics(output_dir, k_by_frame: Dict[int, float],
                                 epoch: int, correction_id: str
                                 ) -> Optional[dict]:
    """New scale_diagnostics content for the given epoch: the original
    estimate is preserved untouched; an ``epochs`` history entry appends the
    per-anchor agreement AFTER this correction. Returns the new dict (the
    caller persists it inside the transaction) or None when the session has
    no diagnostics (declared upstream — the scale gate already dealt with
    it)."""
    diag = _load_diag(output_dir)
    if diag is None:
        return None
    agreements = _current_agreements(diag)
    if agreements is None:
        return None
    new_agreement = {str(f): round(a / k_by_frame.get(f, 1.0), 6)
                     for f, a in agreements.items()}
    entry = {"epoch": int(epoch), "correction_id": correction_id,
             "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "agreement": new_agreement}
    diag.setdefault("epochs", []).append(entry)
    return diag
