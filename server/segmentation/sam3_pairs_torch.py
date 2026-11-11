"""SAM3's per-batch identity arithmetic on the device (USER 2026-11-11: "todo debe ser GPU nada en cpu").

Two places of the SAM3 stage compare masks pixel by pixel, both on the CPU until now:

* ``segmentation.pipeline._match_ids_iou`` — the ids of one batch against the previous batch's in the
  overlap frames (point 93: the IoUs of the frames where both appear are the judges). The record,
  ``_overlap_pair_ious_numpy``, walks every (previous, current) pair of every overlap frame with
  ``np.logical_and / logical_or``;
* ``segmentation.pipeline._dedupe_masks_per_frame`` — ids SAM3 handed out for the SAME observation.
  The record, ``_frame_pair_ious_numpy``, sorts each frame's masks by area and touches the pixels of
  the pairs whose areas could reach the bar.

Measured on pccr 2026-08-24's densest concept ('tile', 2892 masklets; a 250-keyframe batch of 6909
masks, the real masks of the store): 2.7 s for the dedupe and 2.0 s for one overlap of 10 frames on
one CPU thread, per batch.

Here the pixels are counted on the device: a frame's masks uploaded once as booleans; their areas are
pixel sums; the intersections are products of 0/1 rows in float32 — every partial sum an integer no
larger than the frame's pixel count (< 2^24), so exact — and the IoU is the same division of the same
two integers in float64 as the record's. The pairs, their frame order, the zeros of the frames judged
without touching pixels and the bar are the record's; only the pairs some frame takes to the bar are
returned (the only ones the judge reads). A frame whose masks are not all on one grid (the record
resizes one of the pair) is handed to the record's own code.

tests/test_sam3_pairs_torch.py holds both equal to the records, value for value and in order.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

#: masks per side of one intersection product (float32 rows of H*W on the device)
CHUNK = 64


def _upload(masks: Sequence[np.ndarray], device) -> torch.Tensor:
    """(k, H*W) boolean rows on ``device`` (one byte a pixel; the float rows exist a block at a time)."""
    st = np.stack([np.asarray(m).astype(bool, copy=False).reshape(-1) for m in masks])
    if st.shape[1] >= (1 << 24):
        raise ValueError(f"a mask of {st.shape[1]} pixels: the float32 counts are exact below 2^24")
    return torch.as_tensor(st, device=device)


def _products(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """|a_i ∧ b_j| for every boolean row of ``a`` and ``b``, as int64 — blocks of CHUNK rows as float32
    0/1, every partial sum an integer below 2^24: exact."""
    out = np.zeros((int(a.shape[0]), int(b.shape[0])), np.int64)
    for i in range(0, int(a.shape[0]), CHUNK):
        fa = a[i:i + CHUNK].to(torch.float32)
        for j in range(0, int(b.shape[0]), CHUNK):
            fb = b[j:j + CHUNK].to(torch.float32)
            out[i:i + CHUNK, j:j + CHUNK] = (fa @ fb.T).round().to(torch.int64).cpu().numpy()
    return out


def overlap_pair_ious_t(prev_masks, curr_masks, overlap_frames, prev_obj_ids, curr_obj_ids,
                        iou_threshold: float, device) -> Dict[Tuple[int, int], List[float]]:
    """``segmentation.pipeline._overlap_pair_ious_numpy`` with the pixels counted on ``device``."""
    from segmentation.pipeline import _overlap_pair_ious_numpy
    dev = torch.device(device)
    ious: Dict[Tuple[int, int], List[float]] = {}
    for fidx in overlap_frames:
        prev_fm = prev_masks.get(fidx, {})
        curr_fm = curr_masks.get(fidx, {})
        P = [pid for pid in prev_obj_ids if pid in prev_fm]
        C = [cid for cid in curr_obj_ids if cid in curr_fm]
        if not P or not C:
            continue
        pm = [prev_fm[pid] for pid in P]
        cm = [curr_fm[cid] for cid in C]
        shape = np.shape(pm[0])
        if any(np.shape(m) != shape for m in pm + cm):
            # not one grid: the record's own code for this frame (it resizes one mask of the pair)
            one = _overlap_pair_ious_numpy({fidx: prev_fm}, {fidx: curr_fm}, [fidx], prev_obj_ids,
                                           curr_obj_ids, float("-inf"))
            for k, v in one.items():
                ious.setdefault(k, []).extend(v)
            continue
        a, b = _upload(pm, dev), _upload(cm, dev)
        inter = _products(a, b)
        area_p = a.sum(1).cpu().numpy().astype(np.int64)                # boolean sums: exact counts
        area_c = b.sum(1).cpu().numpy().astype(np.int64)
        union = area_p[:, None] + area_c[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = inter / union                                   # the record's int64 / int64
        for i, pid in enumerate(P):
            for j in np.flatnonzero(union[i] > 0):
                ious.setdefault((pid, C[j]), []).append(float(iou[i, j]))
    return {k: v for k, v in ious.items() if max(v) >= iou_threshold}


def frame_pair_ious_t(batch_masks, iou_threshold: float, device):
    """``segmentation.pipeline._frame_pair_ious_numpy`` with the areas and the pixels on ``device``:
    (the judgeable pairs and their IoUs in the record's order, every mask's area)."""
    dev = torch.device(device)
    areas: Dict[Tuple[int, int], int] = {}
    computed: Dict[Tuple[int, int], List[float]] = {}
    frames_of: Dict[int, set] = {}
    for frame_idx in sorted(batch_masks):
        frame_masks = batch_masks[frame_idx]
        keys = list(frame_masks)
        if not keys:
            continue
        for oid in keys:
            frames_of.setdefault(int(oid), set()).add(frame_idx)
        shapes = [np.shape(frame_masks[o]) for o in keys]
        # the masks of each grid uploaded once; the areas are their pixel sums
        groups: Dict[tuple, Tuple[torch.Tensor, List[int]]] = {}
        row_of: Dict[int, int] = {}
        area_of: Dict[int, int] = {}
        for shp in sorted(set(shapes)):
            idx = [i for i, sh in enumerate(shapes) if sh == shp]
            t = _upload([frame_masks[keys[i]] for i in idx], dev)
            groups[shp] = (t, idx)
            ar = t.sum(1).cpu().numpy().astype(np.int64)
            for r, i in enumerate(idx):
                row_of[i] = r
                area_of[i] = int(ar[r])
                areas[(int(frame_idx), int(keys[i]))] = int(ar[r])
        items = sorted(((area_of[i], int(keys[i]), i) for i in range(len(keys))), key=lambda x: (x[0], x[1]))
        want: List[Tuple[int, int, Tuple[int, int], int, int]] = []
        for a in range(len(items)):
            area_a, oid_a, ia = items[a]
            if area_a == 0:
                continue
            for b in range(a + 1, len(items)):
                area_b, oid_b, ib = items[b]
                if area_a < iou_threshold * area_b:          # the record's break: sorted by area
                    break
                if shapes[ia] != shapes[ib]:
                    continue
                want.append((ia, ib, (min(oid_a, oid_b), max(oid_a, oid_b)), area_a, area_b))
        if not want:
            continue
        # the intersections of every pair of the grid's masks in one blocked product, read at the pairs
        prod: Dict[tuple, np.ndarray] = {}
        for shp in {shapes[ia] for ia, *_ in want}:
            t, _idx = groups[shp]
            prod[shp] = _products(t, t)
        inter = [int(prod[shapes[ia]][row_of[ia], row_of[ib]]) for ia, ib, *_ in want]
        for (ia, ib, pair, area_a, area_b), n in zip(want, inter):
            iou = n / float(area_a + area_b - n) if n else 0.0
            computed.setdefault(pair, []).append(iou)
    out: Dict[Tuple[int, int], List[float]] = {}
    for pair, vals in computed.items():
        n_both = len(frames_of[pair[0]] & frames_of[pair[1]])
        if n_both > len(vals):                                  # frames judged without touching pixels
            vals = vals + [0.0] * (n_both - len(vals))
        if vals and max(vals) >= iou_threshold:
            out[pair] = vals
    return out, areas
