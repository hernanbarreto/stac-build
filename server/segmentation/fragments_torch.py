"""The fragment consolidation's contiguity on the device (USER 2026-11-11: "todo lo que me marcaste que puede
pasarse a GPU hay que hacerlo en GPU ... Nada debe hacerse en cpu").

``segmentation.pipeline._merge_label_fragments`` welds the pieces of ONE support-surface label (a floor SAM3
returned in many masks) when they touch through at least ``min_adjacent_pairs`` distinct adjacent occupied
voxels (point 104). The search — every instance's occupied cells at ``gap_m``, the 26 neighbours of every
cell looked up in the label's sorted packed cells, the distinct (instance pair, voxel pair) adjacencies
counted per instance pair — ran in numpy (``_fragment_pairs_numpy``, the record), and the welded index lists
were ``np.union1d`` on the CPU. Here both run on the evidence device with the record's integer arithmetic:
the same cells (a row-unique per instance), the same packing, the same stable order (so a cell shared by two
instances resolves to the same first owner), the same distinct pairs and counts — identical results
(tests/test_fragments_torch.py).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)]


def fragment_pairs_t(instances, members: Sequence[int], xyz_display: np.ndarray, gap_m: float,
                     device) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``segmentation.pipeline._fragment_pairs_numpy`` on ``device``."""
    dev = torch.device(device)
    n_xyz = len(xyz_display)
    keys, owners = [], []
    for k in members:
        gi = np.asarray(instances[k].get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < n_xyz)]
        if not len(gi):
            continue
        P = torch.as_tensor(np.ascontiguousarray(xyz_display[gi]), device=dev)       # the cloud's own dtype
        v = torch.unique(torch.floor(P / float(gap_m)).to(torch.int64), dim=0)        # np.unique(axis=0)
        keys.append(v)
        owners.append(torch.full((int(v.shape[0]),), int(k), dtype=torch.int64, device=dev))
    if len(keys) < 2:
        return None
    allv = torch.cat(keys)
    allo = torch.cat(owners)
    base = allv.min(0).values
    span = allv.max(0).values - base + 3
    if float(span[0]) * float(span[1]) * float(span[2]) > 9.0e18:                    # the record's skip
        return None
    s1, s2 = int(span[1]), int(span[2])

    def pack(v):
        d = v - base + 1
        return (d[:, 0] * s1 + d[:, 1]) * s2 + d[:, 2]

    packed = pack(allv)
    order = torch.argsort(packed, stable=True)
    packed_sorted, owner_sorted = packed[order], allo[order]
    m = int(packed_sorted.shape[0])
    pairs = []
    for off in _OFFSETS:
        nb = pack(allv + torch.as_tensor(off, dtype=torch.int64, device=dev))
        pos = torch.searchsorted(packed_sorted, nb)                                    # numpy's 'left'
        ok = pos < m
        pos = torch.where(ok, pos, torch.zeros_like(pos))
        hit = ok & (packed_sorted[pos] == nb)
        if not bool(hit.any()):
            continue
        a, b = allo[hit], owner_sorted[pos[hit]]
        diff = a != b
        if bool(diff.any()):
            va, vb = packed[hit][diff], nb[hit][diff]
            pairs.append(torch.stack([torch.minimum(a[diff], b[diff]), torch.maximum(a[diff], b[diff]),
                                      torch.minimum(va, vb), torch.maximum(va, vb)], 1))
    if not pairs:
        return None
    allp = torch.unique(torch.cat(pairs), dim=0)
    inst_pairs, counts = torch.unique(allp[:, :2], dim=0, return_counts=True)
    return inst_pairs.cpu().numpy(), counts.cpu().numpy()


def union_indices_t(index_lists: Sequence, device) -> np.ndarray:
    """The sorted unique union of the index lists (the record's chained ``np.union1d``), on ``device``."""
    dev = torch.device(device)
    parts: List[torch.Tensor] = [torch.as_tensor(np.asarray(x, dtype=np.int64), device=dev) for x in index_lists]
    return torch.unique(torch.cat(parts)).cpu().numpy() if parts else np.zeros(0, np.int64)
