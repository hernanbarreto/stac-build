"""Invariant pairs between the reference scan and a scan to fuse.

Identity of a pair = the NORMALIZED LABEL the user gave to the same physical
object in both scans (USER 2026-09-06: "you chose them as invariant objects
on purpose"; an IFC identity may join in the future — it is a requirement of
nothing). What this module decides:

  * which labels form usable pairs (shared, not excluded, enough points),
  * per-pair evidence indices in each scan (classification.npy first,
    globalIndices second — same precedence as the retired fuse_scans),
  * per-pair shooting distance in each scan (provenance → camera centre;
    pairs shot from very different distances are DOWN-WEIGHTED, F4),
  * warnings, never silent drops: movable-looking labels (configurable
    list), pairs concentrated in one zone of the walk
    (``min_pair_spread_ratio`` over the trajectory chainage), fewer than the
    recommended count.

Culling is NOT done here (F4): a pair that does not fit is diagnosed by
``align``/``gates`` and excluded only by explicit user confirmation or the
``auto_exclude_outlier_pairs`` config, always recorded in the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from correction.session import CorrectionSession
from fusion.config import FusionConfig


def norm_label(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def load_instances(output_dir: Path) -> Dict[str, dict]:
    p = Path(output_dir) / "segmentation_result.json"
    if not p.exists():
        raise RuntimeError(
            f"{p} does not exist — segment the invariant objects before "
            f"fusing (both scans need the same labels)")
    res = json.loads(p.read_text())
    out: Dict[str, dict] = {}
    for inst in res.get("instances", []):
        out[norm_label(inst.get("label", ""))] = inst
    return out


def instance_indices(output_dir: Path, inst: dict,
                     n_points: int) -> np.ndarray:
    """Cloud indices of an instance: classification.npy first (v4 membership
    source of truth), globalIndices second (v3)."""
    iid = int(inst.get("instance_id", inst.get("id", -1)))
    cls_p = Path(output_dir) / "classification.npy"
    if cls_p.exists():
        cls = np.load(cls_p, mmap_mode="r")
        if len(cls) == n_points:
            idx = np.flatnonzero(np.asarray(cls) == iid)
            if len(idx):
                return idx.astype(np.int64)
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    return gi[(gi >= 0) & (gi < n_points)]


def _chainage_fraction(session: CorrectionSession, idx: np.ndarray) -> float:
    """Median position of an object's observing keyframes along the scan's
    walked trajectory, as a 0..1 fraction."""
    centers = session.poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    chain = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(chain[-1])
    ks = session.ks[idx]
    ks = ks[ks >= 0]
    if not len(ks) or total <= 0:
        return 0.0
    return float(np.median(chain[ks]) / total)


def _shoot_dist(session: CorrectionSession, idx: np.ndarray) -> Optional[float]:
    pts = session.xyz[idx]
    cen = np.median(pts, axis=0)
    frames = sorted(set(int(f) for f in session.fg[idx]))
    d = [float(np.linalg.norm(session.cam_center[f] - cen))
         for f in frames if f in session.cam_center]
    return float(np.median(d)) if d else None


def _is_movable(label_norm: str, movable: List[str]) -> bool:
    toks = set(label_norm.replace("_", " ").split())
    return any(m in label_norm or m in toks for m in movable)


@dataclass
class Pair:
    label: str                      # display label (reference side)
    norm: str
    ref_idx: np.ndarray
    scan_idx: np.ndarray
    ref_shoot_dist_m: Optional[float]
    scan_shoot_dist_m: Optional[float]
    scan_chainage: float
    weight: float = 1.0             # down-weighted on shoot-dist mismatch
    excluded: bool = False
    excluded_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "ref_points": int(len(self.ref_idx)),
            "scan_points": int(len(self.scan_idx)),
            "ref_shoot_dist_m": (round(self.ref_shoot_dist_m, 2)
                                 if self.ref_shoot_dist_m else None),
            "scan_shoot_dist_m": (round(self.scan_shoot_dist_m, 2)
                                  if self.scan_shoot_dist_m else None),
            "scan_chainage": round(self.scan_chainage, 3),
            "weight": round(self.weight, 3),
            "excluded": self.excluded,
            "excluded_reason": self.excluded_reason,
            "warnings": self.warnings,
        }


@dataclass
class PairSet:
    pairs: List[Pair]
    warnings: List[str]
    spread_ratio: float
    only_in_scan: List[str]
    only_in_ref: List[str]

    @property
    def usable(self) -> List[Pair]:
        return [p for p in self.pairs if not p.excluded]


def build_pairs(ref_session: CorrectionSession,
                scan_session: CorrectionSession,
                cfg: FusionConfig,
                exclude: Optional[Set[str]] = None) -> PairSet:
    """Pairs between the two sessions, with per-pair evidence, weights and
    warnings. ``exclude`` holds normalized labels the USER excluded in the
    UI (recorded, not silent)."""
    exclude = {norm_label(x) for x in (exclude or set())}
    ref_inst = load_instances(ref_session.output_dir)
    scan_inst = load_instances(scan_session.output_dir)
    shared = sorted(set(ref_inst) & set(scan_inst))
    pairs: List[Pair] = []
    warnings: List[str] = []
    for l in shared:
        ref_idx = instance_indices(ref_session.output_dir, ref_inst[l],
                                   ref_session.n_points)
        scan_idx = instance_indices(scan_session.output_dir, scan_inst[l],
                                    scan_session.n_points)
        p = Pair(label=str(ref_inst[l].get("label") or l), norm=l,
                 ref_idx=ref_idx, scan_idx=scan_idx,
                 ref_shoot_dist_m=_shoot_dist(ref_session, ref_idx),
                 scan_shoot_dist_m=_shoot_dist(scan_session, scan_idx),
                 scan_chainage=_chainage_fraction(scan_session, scan_idx))
        if l in exclude:
            p.excluded = True
            p.excluded_reason = "excluded by the user in the UI"
        if len(ref_idx) < cfg.pairs.min_pair_points \
                or len(scan_idx) < cfg.pairs.min_pair_points:
            p.excluded = True
            p.excluded_reason = (f"too few points for a primitive "
                                 f"({len(ref_idx)}/{len(scan_idx)} < "
                                 f"{cfg.pairs.min_pair_points})")
        if _is_movable(l, cfg.pairs.movable_labels):
            p.warnings.append(
                "label looks like a MOVABLE object — a moved object between "
                "days is not an invariant; prefer structural elements")
        if p.ref_shoot_dist_m and p.scan_shoot_dist_m:
            r = max(p.ref_shoot_dist_m, p.scan_shoot_dist_m) / \
                max(min(p.ref_shoot_dist_m, p.scan_shoot_dist_m), 1e-6)
            if r > cfg.pairs.shoot_dist_ratio_max:
                p.weight = 1.0 / r
                p.warnings.append(
                    f"shot from very different distances (×{r:.1f}) — "
                    f"down-weighted to {p.weight:.2f}")
        pairs.append(p)

    usable = [p for p in pairs if not p.excluded]
    if len(usable) < cfg.pairs.recommended_pairs:
        warnings.append(
            f"only {len(usable)} usable pair(s) — "
            f"{cfg.pairs.recommended_pairs}+ spread along the walk are "
            f"recommended")
    chain = sorted(p.scan_chainage for p in usable)
    spread = float(chain[-1] - chain[0]) if len(chain) >= 2 else 0.0
    if usable and spread < cfg.pairs.min_pair_spread_ratio:
        warnings.append(
            f"pairs cover only {spread:.0%} of the scan's walk "
            f"(< {cfg.pairs.min_pair_spread_ratio:.0%}) — add an invariant "
            f"in the uncovered stretch")
    return PairSet(
        pairs=pairs, warnings=warnings, spread_ratio=spread,
        only_in_scan=sorted(str(scan_inst[l].get("label") or l)
                            for l in set(scan_inst) - set(ref_inst)),
        only_in_ref=sorted(str(ref_inst[l].get("label") or l)
                           for l in set(ref_inst) - set(scan_inst)))
