"""Evidence extraction for the marked instances.

USER 2026-09-06: "lo que yo dejé dentro del bbox es lo correcto que debe
corregirse; lo que esté afuera aunque esté en la máscara no es correcto" — the
curated OBB (+ configurable margin) locates every copy of the object in the
cloud; the MATCHING evidence is always the instance's own curated points
(MEJORAS §3.4: the bbox must not inflate the object — on ccc1 68% of the bbox
points were floor and ICP matched floor to floor).

What it decides:
  * copies: per marked instance, per VISIT (contiguous keyframe runs of the
    points inside its curated OBB) — index sets, centroids, shoot distance.
  * the REFERENCE visit group: the earliest overlapping group of visits
    (USER 2026-09-06: drift accumulates along the trajectory; the first
    sighting is the anchor, the late revisit is what moves).
  * the displaced visits (everything outside the reference keyframe span).

Everything is tool_measured; nothing is modified here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from correction.config import CorrectionConfig
from correction.session import CorrectionSession
from correction.units import visits_from_keyframes


@dataclass
class Copy:
    iid: int
    label: str
    visit: int                    # visit ordinal within the instance
    kfs: List[int]                # keyframes of this visit (sorted)
    idx: np.ndarray               # bbox cloud indices in this visit
    seg_idx: np.ndarray           # curated instance indices in this visit
    centroid: np.ndarray          # median of curated points (bbox fallback)
    shoot_dist_m: Optional[float]
    is_reference: bool = False

    @property
    def kf_range(self) -> List[int]:
        return [self.kfs[0], self.kfs[-1]]

    def summary(self) -> dict:
        return {
            "iid": self.iid, "label": self.label, "visit": self.visit,
            "kf_range": self.kf_range, "n_points": int(len(self.idx)),
            "n_object_points": int(len(self.seg_idx)),
            "shoot_dist_m": (round(self.shoot_dist_m, 2)
                             if self.shoot_dist_m is not None else None),
            "is_reference": self.is_reference,
        }


@dataclass
class Evidence:
    copies: List[Copy]
    ref: Dict[int, Copy]                  # iid → reference copy
    displaced: List[Copy]
    ref_kf_end: int                       # last keyframe of the reference group
    target_idx: np.ndarray                # curated reference points (cloud idx)
    ref_fingerprint: Dict[tuple, float]   # (iid_a, iid_b) → centroid distance
    labels: Dict[int, str]


def _load_instances(output_dir: Path) -> Dict[int, dict]:
    path = output_dir / "segmentation_result.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist — segment the scene before marking "
            f"duplicated objects")
    res = json.loads(path.read_text())
    return {int(i.get("instance_id", i.get("id"))): i
            for i in res.get("instances", [])}


def extract_evidence(session: CorrectionSession, instance_ids: List[int],
                     cfg: CorrectionConfig, log=print) -> Evidence:
    inst_by_iid = _load_instances(session.output_dir)
    labels = {k: (v.get("label") or v.get("name") or str(k))
              for k, v in inst_by_iid.items()}
    xyz, ks = session.xyz, session.ks
    margin = cfg.evidence.obb_margin_m

    copies: List[Copy] = []
    for iid in instance_ids:
        inst = inst_by_iid.get(int(iid))
        if inst is None:
            raise RuntimeError(
                f"instance {iid} is not in segmentation_result.json — "
                f"refresh the segmentation and mark again")
        gidx = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gidx = gidx[(gidx >= 0) & (gidx < len(xyz))]
        if not len(gidx):
            raise RuntimeError(
                f"instance {iid} ({labels.get(int(iid))}) has no points — "
                f"it cannot anchor a correction")
        P = xyz[gidx]
        c_obb = P.mean(0)
        axes = np.linalg.svd(P - c_obb, full_matrices=False)[2]
        loc = (P - c_obb) @ axes.T
        lo = loc.min(0) - margin
        hi = loc.max(0) + margin
        rel = (xyz - c_obb) @ axes.T
        inside = np.all((rel >= lo) & (rel <= hi), axis=1)
        idx_all = np.where(inside)[0]
        log(f"  {labels.get(int(iid))}: bbox holds {len(idx_all):,} cloud pts "
            f"(curated segment: {len(gidx):,})")

        visits = visits_from_keyframes(ks[idx_all], cfg.evidence.visit_gap_kf)
        gidx_set = gidx
        for vi, vk in enumerate(visits):
            in_visit = np.isin(ks[idx_all], vk)
            idx = idx_all[in_visit]
            seg_idx = np.intersect1d(idx, gidx_set)
            base = seg_idx if len(seg_idx) else idx
            cen = np.median(xyz[base], axis=0)
            vframes = sorted(set(int(f) for f in session.fg[idx]))
            dists = [float(np.linalg.norm(session.cam_center[f] - cen))
                     for f in vframes if f in session.cam_center]
            cp = Copy(iid=int(iid), label=labels.get(int(iid), str(iid)),
                      visit=vi, kfs=vk, idx=idx, seg_idx=seg_idx,
                      centroid=cen,
                      shoot_dist_m=(float(np.median(dists)) if dists else None))
            copies.append(cp)
            log(f"  {cp.label} visit {vi}: {len(idx):,} pts "
                f"({len(seg_idx):,} curated), kf {vk[0]}..{vk[-1]}, "
                f"shoot dist {(cp.shoot_dist_m or -1):.2f} m")

    if not copies:
        raise RuntimeError("no copies extracted — the marked bboxes are empty")

    # Reference group: merge visits whose keyframe SPANS overlap (transitive);
    # the group holding the earliest keyframe is the reference.
    spans = [(cp.kfs[0], cp.kfs[-1]) for cp in copies]
    groups: List[set] = []
    for i, (a, b) in enumerate(spans):
        merged = None
        for g in groups:
            if any(not (b < spans[j][0] or a > spans[j][1]) for j in g):
                g.add(i)
                merged = g
                break
        if merged is None:
            groups.append({i})
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                si = {k for m in groups[i] for k in range(spans[m][0],
                                                          spans[m][1] + 1)}
                sj = {k for m in groups[j] for k in range(spans[m][0],
                                                          spans[m][1] + 1)}
                if si & sj:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    ref_group = min(groups, key=lambda g: min(spans[m][0] for m in g))
    ref_kf_set = set()
    for m in ref_group:
        ref_kf_set.update(range(spans[m][0], spans[m][1] + 1))

    # a copy is evidence ONLY through its curated points: the bbox alone
    # (floor, neighbours, floaters) is never a reference nor an ICP source
    # (pccr 2026-09-09: a bbox-only "reference" with 0 curated points sent
    # the ICP to 92°)
    min_pts = cfg.evidence.min_object_points_solve
    ref: Dict[int, Copy] = {}
    for iid in instance_ids:
        cands = [copies[m] for m in ref_group
                 if copies[m].iid == int(iid)
                 and len(copies[m].seg_idx) >= min_pts]
        if cands:
            ref[int(iid)] = max(cands, key=lambda c: len(c.seg_idx))
        else:
            log(f"  {labels.get(int(iid))}: no curated copy in the "
                f"reference visit — its copies are displaced only")
    if not ref:
        raise RuntimeError(
            "no marked instance is observed in the earliest visit — there is "
            "no reference to correct against; mark an object that appears in "
            "both visits")
    for r in ref.values():
        r.is_reference = True
        log(f"  reference for {r.label}: visit {r.visit} "
            f"({len(r.seg_idx):,} curated pts, kf {r.kf_range})")

    displaced = [cp for cp in copies if not cp.is_reference
                 and not set(cp.kfs) <= ref_kf_set
                 and len(cp.seg_idx) > 0]
    if not displaced:
        raise RuntimeError(
            "every copy lies inside the reference visit — nothing to "
            "correct; if the duplicates are visually apart, check that the "
            "curated bbox actually covers both copies")

    target_idx = np.unique(np.concatenate(
        [r.seg_idx for r in ref.values() if len(r.seg_idx)]))
    if len(target_idx) < cfg.evidence.min_object_points_solve:
        raise RuntimeError(
            f"the reference copies hold only {len(target_idx)} curated "
            f"points (< {cfg.evidence.min_object_points_solve}) — too little "
            f"to anchor an ICP; mark a larger object")

    ref_fp: Dict[tuple, float] = {}
    iids = sorted(ref.keys())
    for i in range(len(iids)):
        for j in range(i + 1, len(iids)):
            ref_fp[(iids[i], iids[j])] = float(np.linalg.norm(
                ref[iids[i]].centroid - ref[iids[j]].centroid))

    ref_kf_end = max(k for r in ref.values() for k in r.kfs)
    return Evidence(copies=copies, ref=ref, displaced=displaced,
                    ref_kf_end=ref_kf_end, target_idx=target_idx,
                    ref_fingerprint=ref_fp, labels=labels)
