"""The co-visible split's pair judges, BATCHED on the card (USER 2026-10-15: "debe correr rápido en
gpu todo" — observatorio's split sat for hours in a Python loop over every pair of fragments,
each pair paying a KD-tree query and its own ray walk).

The same rule as `segmentation.pipeline._split_covisible_components`, pair for pair:
- co-visibility: the keyframes of the component seen in fewer keyframes judge (1 when the other
  is seen too) — the judge vectors come from one keyframe matrix, the user's decision rule
  (`decide_change`, a seeded bootstrap) is evaluated once per DISTINCT judge vector (the same
  judges give the same verdict: memoised, exact);
- the camera came back: one matrix product over the keyframe pairs that are near in space and
  far along the walk;
- free space: the closest points of the two components (exact: chunked distances on the card
  when the pair is small enough, else a KD-tree built once per component), the samples of the gap, every camera's rays walked in front and behind, every radius
  test on the FreeSpaceIndex — all pairs of an instance in a few batches; the judges in the
  recipe's order (camera, sample).
Deterministic: fixed sample order, first-minimum argmins, the bootstrap's own seed."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from segmentation.gap_rays_torch import FreeSpaceIndex

BATCH_SAMPLES = 1.5e7           # ray samples per any_within call


SAMPLE_N = 4096                 # the closest pair is searched on each component's stable sample


def _closest_pairs(P: np.ndarray, comp: np.ndarray, ki: np.ndarray, pairs: Sequence[Tuple[int, int]],
                   device: str, gap_m: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(a, b, mode) per pair: the closest points of the two components over their STABLE samples
    (`_stable_sample`, ≤ 4096 points each — exact when a component has no more): one 4096²
    distance matrix per pair on the card, the recipe's rule (for every b the nearest a, the b
    with the smallest distance, the first). DECLARED: two 180 k-point strips facing each other
    along 15 m have no cheap exact closest pair (every cell faces the other strip); the sample's
    closest pair moves the gap's sample segment by at most the sample spacing (centimetres on
    a 10 cm-sampled gap). The record says which mode judged each pair."""
    from segmentation.pipeline import _stable_sample
    a_out = np.zeros((len(pairs), 3)); b_out = np.zeros((len(pairs), 3)); modes = []
    cache: Dict[int, tuple] = {}

    def _comp(c):
        if c not in cache:
            Q = P[comp == c]
            sel = _stable_sample(Q, SAMPLE_N)
            cache[c] = (torch.as_tensor(np.ascontiguousarray(Q[sel], dtype=np.float64), device=device),
                        len(Q) > SAMPLE_N)
        return cache[c]

    for k, (c1, c2) in enumerate(pairs):
        A, sa = _comp(c1)
        B, sb = _comp(c2)
        D = torch.cdist(B, A, compute_mode="donot_use_mm_for_euclid_dist")          # (nB, nA)
        dmn, amn = torch.min(D, dim=1)
        jb = int(torch.argmin(dmn))
        a_out[k] = A[int(amn[jb])].cpu().numpy(); b_out[k] = B[jb].cpu().numpy()
        modes.append("sampled_4096" if (sa or sb) else "exact")
    return a_out, b_out, modes


def gap_rays_batch(a: np.ndarray, b: np.ndarray, cams: List[np.ndarray], index: FreeSpaceIndex,
                   gap_m: float, reach_m: float) -> List[List[int]]:
    """The ray judges of many pairs at once (each pair: its closest points, its cameras in
    keyframe order) — `_gap_rays`' samples, rays, steps and order; the radius tests batched."""
    gap_m = float(gap_m)
    n_pairs = len(a)
    steps = np.arange(1, max(int(reach_m / gap_m), 1) + 1) * gap_m
    T = len(steps)
    per_pair = []                       # (samples (S,3), cams (C,3)) or None
    for k in range(n_pairs):
        n = int(np.floor(np.linalg.norm(b[k] - a[k]) / gap_m))
        C = np.asarray(cams[k], np.float64).reshape(-1, 3)
        if n < 2 or len(C) == 0:
            per_pair.append(None)
            continue
        samples = a[k] + (b[k] - a[k]) * (np.linspace(1, n - 1, min(n - 1, 20))[:, None] / n)
        per_pair.append((samples, C))
    out: List[List[int]] = [[] for _ in range(n_pairs)]
    live = [k for k in range(n_pairs) if per_pair[k] is not None]
    # batches of pairs bounded by the number of samples they put on the card
    start = 0
    while start < len(live):
        batch, n_samp = [], 0
        while start < len(live):
            k = live[start]
            S, C = per_pair[k]
            L = np.linalg.norm(S[None, :, :] - C[:, None, :], axis=2)
            kf = int(np.maximum(np.ceil((L - 2.0 * gap_m) / gap_m), 0).max()) if L.size else 0
            cost = len(C) * len(S) * (kf + T)
            if batch and n_samp + cost > BATCH_SAMPLES:
                break
            batch.append(k); n_samp += cost; start += 1
        # build every front/behind sample of the batch, flattened, with its (pair, cam, sample) key
        front_pts, front_key, behind_pts, behind_key, meta = [], [], [], [], {}
        for k in batch:
            S, C = per_pair[k]
            rays = S[None, :, :] - C[:, None, :]
            L = np.linalg.norm(rays, axis=2)
            ok = L > gap_m
            dirs = rays / np.maximum(L, 1e-12)[:, :, None]
            n_front = np.where(ok, np.maximum(np.ceil((L - 2.0 * gap_m) / gap_m), 0), 0).astype(np.int64)
            meta[k] = (ok, len(C), len(S))
            ci, si = np.nonzero(n_front > 0)
            for c_, s_ in zip(ci, si):
                t = (np.arange(n_front[c_, s_]) + 1.0) * gap_m
                front_pts.append(C[c_] + dirs[c_, s_][None, :] * t[:, None])
                front_key.append(np.full(len(t), k * 10 ** 9 + c_ * 10 ** 4 + s_, dtype=np.int64))
            cb = (S[None, :, None, :] + dirs[:, :, None, :] * steps[None, None, :, None]).reshape(-1, 3)
            behind_pts.append(cb)
            kk = (k * 10 ** 9 + np.arange(len(C))[:, None] * 10 ** 4 + np.arange(len(S))[None, :])
            behind_key.append(np.repeat(kk.reshape(-1), T))
        blocked: Dict[int, bool] = {}
        if front_pts:
            fp = np.concatenate(front_pts); fk = np.concatenate(front_key)
            w = index.any_within(fp, gap_m / 2.0)
            for key in np.unique(fk[w]):
                blocked[int(key)] = True
        bp = np.concatenate(behind_pts); bk = np.concatenate(behind_key)
        hitw = index.any_within(bp, gap_m / 2.0)
        hits = set(int(x) for x in np.unique(bk[hitw]))
        for k in batch:
            ok, nC, nS = meta[k]
            judges = []
            for c_ in range(nC):
                for s_ in range(nS):
                    key = k * 10 ** 9 + c_ * 10 ** 4 + s_
                    if not ok[c_, s_] or blocked.get(key, False):
                        continue
                    judges.append(1 if key in hits else 0)
            out[k] = judges
    return out


def judge_pairs(P: np.ndarray, comp: np.ndarray, ki: np.ndarray, cand: List[int], size: np.ndarray,
                kfs: Dict[int, set], cam_centre: Optional[Dict[int, np.ndarray]], chain: Dict[int, float],
                min_walk_m: Optional[float], covis_share: float, decision: Tuple[float, float, int],
                gap_m: float, reach_m: float, index: Optional[FreeSpaceIndex], iid: int,
                record: Optional[list]) -> Dict[int, int]:
    """Every pair of candidates judged as the recipe does, batched; returns the union-find
    parent map (a pair that is NOT distinct joins its components)."""
    from segmentation.pipeline import _decide_above_bar
    factor, confidence, min_judges = decision
    n_c = len(cand)
    pos = {c: i for i, c in enumerate(cand)}
    frames_all = sorted(set().union(*[kfs[c] for c in cand]))
    f_pos = {f: i for i, f in enumerate(frames_all)}
    K = np.zeros((n_c, len(frames_all)), dtype=bool)
    for c in cand:
        for f in kfs[c]:
            K[pos[c], f_pos[f]] = True
    # the camera came back: keyframes with a camera AND a walk position
    cam_frames = [f for f in frames_all if cam_centre and f in cam_centre and f in chain]
    R = np.zeros((n_c, n_c), dtype=bool)
    if cam_frames and min_walk_m is not None:
        Cc = np.array([cam_centre[f] for f in cam_frames]); wc = np.array([chain[f] for f in cam_frames])
        near = np.linalg.norm(Cc[:, None, :] - Cc[None, :, :], axis=2) <= float(min_walk_m)
        far = np.abs(wc[:, None] - wc[None, :]) > float(min_walk_m)
        M = (near & far).astype(np.int64)
        Kc = K[:, [f_pos[f] for f in cam_frames]].astype(np.int64)
        R = (Kc @ M @ Kc.T) > 0
    parent = {c: c for c in cand}

    def _root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    memo: Dict[tuple, dict] = {}

    def _decide(judges, bar, kind):
        key = (kind, tuple(judges), float(bar))
        if key not in memo:
            p = float(np.mean(judges)) if judges else 0.0
            err = float(np.sqrt(max(p * (1.0 - p), 0.0) / max(len(judges), 1)))
            d = _decide_above_bar(list(judges), bar, err, factor=factor, confidence=confidence,
                                  min_judges=min_judges)
            if kind == "covis":
                d.update({"n_judges": len(judges), "share": p})
            else:
                d.update({"n_rays": len(judges), "share_through": p})
            memo[key] = d
        return memo[key]
    # 1) co-visibility and the camera's return, every pair
    pairs = [(c1, c2) for i, c1 in enumerate(cand) for c2 in cand[i + 1:]]
    cv_of, dup_of, need_rays = {}, {}, []
    for (c1, c2) in pairs:
        small, other = (c1, c2) if len(kfs[c1]) <= len(kfs[c2]) else (c2, c1)
        sf = sorted(kfs[small])
        judges = [1 if K[pos[other], f_pos[f]] else 0 for f in sf]
        cv = _decide(judges, float(covis_share), "covis")
        covis = bool(cv["improves"])
        dup_possible = (not covis) and bool(chain) and bool(R[pos[c1], pos[c2]])
        cv_of[(c1, c2)] = cv; dup_of[(c1, c2)] = dup_possible
        if not dup_possible and cam_centre:
            need_rays.append((c1, c2))
    # 2) the free-space judges of every pair that needs them, batched on the card
    fs_of: Dict[Tuple[int, int], Optional[dict]] = {p_: None for p_ in pairs}
    if need_rays and index is not None:
        cams = []
        for (c1, c2) in need_rays:
            common = kfs[c1] & kfs[c2]
            fs = sorted(f for f in (common or (kfs[c1] | kfs[c2])) if f in cam_centre)
            cams.append(np.array([cam_centre[f] for f in fs]) if fs else np.zeros((0, 3)))
        a, b, modes = _closest_pairs(P, comp, ki, need_rays, index.device, gap_m)
        rays = gap_rays_batch(a, b, cams, index, gap_m, reach_m)
        for (c1, c2), r, mode, C in zip(need_rays, rays, modes, cams):
            if len(C) == 0:
                continue
            d = dict(_decide(r, 0.5, "rays"))
            d["closest_pair"] = mode
            fs_of[(c1, c2)] = d
    # 3) the verdicts in the recipe's order; a pair already one group is spared (its verdict
    #    cannot change the transitive closure either way)
    for (c1, c2) in pairs:
        if _root(c1) == _root(c2):
            if record is not None:
                record.append({"kind": "covisible_split", "instance_id": int(iid),
                               "components": [int(size[c1]), int(size[c2])], "skipped": "already one group"})
            continue
        cv, dup_possible, fs_rec = cv_of[(c1, c2)], dup_of[(c1, c2)], fs_of[(c1, c2)]
        distinct = bool(fs_rec["improves"]) if fs_rec is not None else False
        if record is not None:
            record.append({"kind": "covisible_split", "instance_id": int(iid),
                           "components": [int(size[c1]), int(size[c2])],
                           "covisible": {k_: cv.get(k_) for k_ in ("improves", "n_judges", "share", "ci_margin",
                                                                  "judges_margin", "error_margin", "reason")},
                           "camera_returned": bool(dup_possible),
                           "free_space": ({k_: fs_rec.get(k_) for k_ in ("improves", "n_rays", "share_through",
                                                                         "ci_margin", "judges_margin",
                                                                         "error_margin", "reason", "closest_pair")}
                                          if fs_rec is not None else None),
                           "distinct": bool(distinct)})
        if not distinct:
            parent[_root(c2)] = _root(c1)
    return parent
