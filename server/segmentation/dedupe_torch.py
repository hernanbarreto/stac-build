"""The space dedupe's voxel-set arithmetic on the device (USER 2026-11-01: "todo debe correr en GPU").

`segmentation.pipeline._space_dedupe` walks every pair of instances, largest first, and asks three things
of their 5 cm voxel sets: how many voxels they share, which keyframes saw both, and — per such keyframe
— what share of one instance's voxels born in that keyframe lies inside the other instance's space (the
judges of docs/plan_determinismo.md point 104). The numpy recipe answered with `np.unique` and `np.isin`
over Python lists of arrays, pair by pair: on pccr audit5 (1003 instances, ~500 k pairs) 2 min 15 s of the
projection; 70.7 s on the 454 final instances of pccr.

Here the sets live on one device as SORTED unique int64 keys (the recipe's packing,
`segmentation.pipeline._pack_voxels`) plus, per instance, its (voxel, keyframe) pairs sorted by voxel, and
the three questions are answered for EVERY candidate j of an outer instance i at once (`prepare`):

* |vox_j ∩ vox_i| — one `searchsorted` of the candidates' concatenated keys into vox_i, counts per
  candidate by an integer index sum;
* sj(j, f) = |vox_j,f ∩ vox_i| — the candidates' (voxel, keyframe) pairs whose voxel is in vox_i, counted
  per (candidate, keyframe);
* si(j, f) = |vox_i,f ∩ vox_j| — every (voxel, keyframe) pair of i expanded over the candidates holding
  that voxel (its range in the candidates' concatenated keys sorted by voxel), counted per (candidate,
  keyframe).

The shares and the judges are the same quotients of the same integers as the numpy twin
(`segmentation.pipeline.VoxelSetsNumpy`), so `decide_change` sees the same numbers in the same order and
the same instances merge; a merge re-prepares the outer instance's enlarged sets for its later pairs, as
the recipe's loop saw them. tests/test_dedupe_torch.py holds the two backends equal decision for decision.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def _pack_t(v: torch.Tensor) -> torch.Tensor:
    """`segmentation.pipeline._pack_voxels` on int64 cell coordinates (n, 3) — the same key."""
    half = 1 << 20
    if int(v.shape[0]) and int(v.abs().max()) >= half:
        raise ValueError(f"voxel coordinates reach {int(v.abs().max())} cells from the origin — beyond the "
                         f"{half} the packing holds")
    return ((v[:, 0] + half) << 42) | ((v[:, 1] + half) << 21) | (v[:, 2] + half)


def _member(a_sorted: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per element of ``b``: True when it is in the sorted unique ``a_sorted``."""
    if int(a_sorted.shape[0]) == 0 or int(b.shape[0]) == 0:
        return torch.zeros(int(b.shape[0]), dtype=torch.bool, device=b.device)
    pos = torch.searchsorted(a_sorted, b)
    pos_c = torch.clamp(pos, max=int(a_sorted.shape[0]) - 1)
    return (pos < int(a_sorted.shape[0])) & (a_sorted[pos_c] == b)


def _sorted_pairs(voxel: torch.Tensor, kf: torch.Tensor):
    """Unique (voxel, keyframe) pairs sorted by voxel then keyframe."""
    o = torch.argsort(kf, stable=True)
    o = o[torch.argsort(voxel[o], stable=True)]
    v, f = voxel[o], kf[o]
    if int(v.shape[0]) == 0:
        return v, f
    first = torch.ones(int(v.shape[0]), dtype=torch.bool, device=v.device)
    first[1:] = (v[1:] != v[:-1]) | (f[1:] != f[:-1])
    return v[first], f[first]


class VoxelSetsTorch:
    """The instances' voxel sets (all points, and per birth keyframe) on ``device``."""

    def __init__(self, instances: Sequence[dict], xyz_display: np.ndarray, frame_arr: np.ndarray,
                 vox: float, device: str):
        self.device = str(device)
        dev = self.device
        X = torch.as_tensor(np.ascontiguousarray(np.asarray(xyz_display, np.float64)), device=dev)
        F = torch.as_tensor(np.ascontiguousarray(np.asarray(frame_arr)), device=dev).to(torch.int64)
        self.vox_all: List[torch.Tensor] = []
        self.pv: List[torch.Tensor] = []            # (voxel, keyframe) pairs: voxels, sorted by voxel then kf
        self.pf: List[torch.Tensor] = []            # … their keyframes
        self.kfs: List[set] = []                    # the keyframes with at least one voxel
        for inst in instances:
            idxs = np.asarray(inst["globalIndices"], dtype=np.int64)
            if len(idxs) == 0:
                e = torch.zeros(0, dtype=torch.int64, device=dev)
                self.vox_all.append(e); self.pv.append(e); self.pf.append(e); self.kfs.append(set())
                continue
            it = torch.as_tensor(idxs, device=dev)
            keys = _pack_t(torch.floor(X[it] / float(vox)).to(torch.int64))
            self.vox_all.append(torch.unique(keys, sorted=True))
            v, f = _sorted_pairs(keys, F[it])
            self.pv.append(v); self.pf.append(f)
            self.kfs.append(set(torch.unique(f).tolist()))
        self._cache: Dict[int, dict] = {}
        self._cache_i = -1

    def n(self, k: int) -> int:
        return int(self.vox_all[k].shape[0])

    # ── the batch of an outer instance ───────────────────────────────────────────────────────────────
    def prepare(self, i: int, js: Sequence[int]) -> Dict[int, int]:
        """Everything the pair loop asks about i against each j of ``js``, in three joins; returns
        {j: |vox_j ∩ vox_i|} and caches the per-keyframe counts for :meth:`common_keyframes` / :meth:`judges`."""
        self._cache_i = i
        self._cache = {}
        if not js:
            return {}
        dev = self.device
        vi = self.vox_all[i]
        sizes = torch.tensor([self.n(j) for j in js], device=dev, dtype=torch.int64)
        owner_of = torch.repeat_interleave(torch.arange(len(js), device=dev), sizes)
        cat = torch.cat([self.vox_all[j] for j in js])
        hit = _member(vi, cat).to(torch.int64)
        inter = torch.zeros(len(js), dtype=torch.int64, device=dev).index_add_(0, owner_of, hit)
        inter_l = inter.tolist()
        counts = {j: int(c) for j, c in zip(js, inter_l)}
        active = [k for k, c in enumerate(inter_l) if c]
        for k in range(len(js)):
            self._cache[js[k]] = None
        if not active:
            return counts
        # the keyframes that matter: the union of the kfs of i and of the active candidates, as a dense index
        kf_all = sorted(set().union(self.kfs[i], *(self.kfs[js[k]] for k in active)))
        kf_index = {f: n for n, f in enumerate(kf_all)}
        K = len(kf_all)
        lut_src = torch.tensor(kf_all, device=dev, dtype=torch.int64)
        if K == 0:
            return counts
        kmax = int(lut_src.max()) + 1
        lut = torch.full((kmax,), -1, dtype=torch.int64, device=dev)
        lut[lut_src] = torch.arange(K, device=dev)
        A = len(active)
        # sj(j, f): j's (voxel, kf) pairs whose voxel is in vox_i
        pv_c = torch.cat([self.pv[js[k]] for k in active])
        pf_c = torch.cat([self.pf[js[k]] for k in active])
        own_c = torch.repeat_interleave(torch.arange(A, device=dev),
                                        torch.tensor([int(self.pv[js[k]].shape[0]) for k in active], device=dev))
        nj = torch.zeros(A * K, dtype=torch.int64, device=dev)
        nj.index_add_(0, own_c * K + lut[pf_c], torch.ones(int(pf_c.shape[0]), dtype=torch.int64, device=dev))
        hj = torch.zeros(A * K, dtype=torch.int64, device=dev)
        m = _member(vi, pv_c)
        if bool(m.any()):
            hj.index_add_(0, (own_c * K + lut[pf_c])[m], torch.ones(int(m.sum()), dtype=torch.int64, device=dev))
        # si(j, f): i's (voxel, kf) pairs expanded over the candidates holding that voxel
        ni = torch.zeros(K, dtype=torch.int64, device=dev)
        ni.index_add_(0, lut[self.pf[i]], torch.ones(int(self.pf[i].shape[0]), dtype=torch.int64, device=dev))
        cat_a = torch.cat([self.vox_all[js[k]] for k in active])
        own_a = torch.repeat_interleave(torch.arange(A, device=dev),
                                        torch.tensor([self.n(js[k]) for k in active], device=dev))
        o = torch.argsort(cat_a, stable=True)
        cat_s, own_s = cat_a[o], own_a[o]
        lo = torch.searchsorted(cat_s, self.pv[i])
        hi = torch.searchsorted(cat_s, self.pv[i], right=True)
        cnt = hi - lo
        tot = int(cnt.sum())
        hi_counts = torch.zeros(A * K, dtype=torch.int64, device=dev)
        if tot:
            src = torch.repeat_interleave(torch.arange(int(cnt.shape[0]), device=dev), cnt)
            first = torch.cumsum(cnt, 0) - cnt
            local = torch.arange(tot, device=dev) - first[src]
            owners = own_s[lo[src] + local]
            hi_counts.index_add_(0, owners * K + lut[self.pf[i][src]],
                                 torch.ones(tot, dtype=torch.int64, device=dev))
        nj_np = nj.view(A, K).cpu().numpy()
        hj_np = hj.view(A, K).cpu().numpy()
        hi_np = hi_counts.view(A, K).cpu().numpy()
        ni_np = ni.cpu().numpy()
        for a, k in enumerate(active):
            j = js[k]
            common = [f for f in kf_all if f in self.kfs[j] and f in self.kfs[i]]
            idx = [kf_index[f] for f in common]
            self._cache[j] = {"common": common, "sj": [(int(hj_np[a, q]), int(nj_np[a, q])) for q in idx],
                              "si": [(int(hi_np[a, q]), int(ni_np[q])) for q in idx]}
        return counts

    def inter_counts(self, i: int, js: Sequence[int]) -> Dict[int, int]:
        return self.prepare(i, js)

    def common_keyframes(self, i: int, j: int) -> List[int]:
        c = self._cache.get(j) if self._cache_i == i else None
        if c is None:
            return sorted(self.kfs[i] & self.kfs[j])
        return list(c["common"])

    def judges(self, i: int, j: int, common: Sequence[int], mutual: bool) -> List[float]:
        """Per common keyframe f (in order): sj = |vox_j,f ∩ vox_i| / |vox_j,f| and, mutual, min with
        si = |vox_i,f ∩ vox_j| / |vox_i,f| — the recipe's quotients of the same integers."""
        c = self._cache.get(j) if self._cache_i == i else None
        if c is None:
            self.prepare(i, [j])
            c = self._cache[j]
            if c is None:                       # no shared voxel at all: every share is 0 (the recipe's zeros)
                return [0.0] * len(common)
        out = []
        for (a, b), (cc, d) in zip(c["sj"], c["si"]):
            sj = float(a) / b
            si = float(cc) / d
            out.append(min(si, sj) if mutual else sj)
        return out

    def absorb(self, i: int, j: int) -> None:
        """j's voxels join i's: the union of the sets, and of the (voxel, keyframe) pairs."""
        self.vox_all[i] = torch.unique(torch.cat([self.vox_all[i], self.vox_all[j]]), sorted=True)
        v, f = _sorted_pairs(torch.cat([self.pv[i], self.pv[j]]), torch.cat([self.pf[i], self.pf[j]]))
        self.pv[i], self.pf[i] = v, f
        self.kfs[i] = self.kfs[i] | self.kfs[j]
        self._cache = {}
        self._cache_i = -1
