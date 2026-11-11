"""Phase 1 of the mask→cloud projection on the device (USER 2026-11-11: "todo debe ser GPU nada en cpu").

``segmentation.pipeline._match_points_numpy`` — the path of record — walks every masklet and, for each
keyframe of the cloud, reads that masklet's mask from the npz, looks up the pixels of the points born in
the keyframe and keeps, per point, the SMALLEST covering mask area (a tie: the lower id) and the
runner-up (point 117). On pccr 2026-08-24 that is 14,438 masklets x 2765 keyframes, 290,418 masks
inflated one after the other on one thread, the card idle.

Here the walk is by FRAME. A point is born in ONE keyframe, so the only masks that can cover it are
that keyframe's. Per cloud frame, every mask of the frame (of the masklets projected) is inflated on
reader threads — the store read ONCE, one ZipFile per thread over its bytes (the reader of
``autoprompt.rounds.segmented_masks``) — and stacked on the device: the areas are pixel counts, the
coverage is the stack at the points' (row, col), and the winner and the runner-up are the two smallest
keys ``area * M + id`` (M = largest id + 1; the areas are integers, so the keys are exact and unique).
That is what the record's strict ``<`` updates in ascending id leave: of two equal areas the first
(lower id) stays the winner, and the runner-up is likewise the next in (area, id) order. Erosion
(``segmentation.mask_erosion_iterations``; 0 since USER 2026-08-29) runs where the masks are inflated,
with the record's own cv2 call, so its pixels are the record's. The per-object area bookkeeping is the
record's arithmetic in the record's order (the mean over the object's frames, ascending).

tests/test_mask_match_torch.py holds the two equal, array for array.
"""
from __future__ import annotations

import io
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

#: reader threads inflating the masks (the store's own reader count: rounds.UNION_THREADS)
READ_THREADS = 8
#: frames inflated ahead of the device (bounded RAM: ~frames x masks x grid bytes)
READ_AHEAD = 24
#: (masks x points) per coverage block on the device
BLOCK = 50_000_000
_MASK_KEY = re.compile(r"^f(\d+)_o(\d+)$")


def _frame_masks(names: Sequence[str], obj_ids: Sequence[int]) -> Dict[int, List[int]]:
    """{mask frame: its masklet ids among ``obj_ids``, ascending} from the store's member names."""
    want = set(int(o) for o in obj_ids)
    by: Dict[int, List[int]] = {}
    for key in names:
        m = _MASK_KEY.match(key)
        if m and int(m.group(2)) in want:
            by.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return {f: sorted(v) for f, v in by.items()}


def match_points_t(masks_path, obj_ids: Sequence[int], frame_groups: Dict[int, np.ndarray],
                   mask_frame_of: Dict[int, int], rows_m: np.ndarray, cols_m: np.ndarray, n_pts: int,
                   mask_h_ref: int, mask_w_ref: int, erosion_iterations: int, erosion_kernel,
                   device: str, log=print):
    """``segmentation.pipeline._match_points_numpy`` on ``device``: the same six results."""
    import cv2
    raw = Path(masks_path).read_bytes()
    with np.load(io.BytesIO(raw)) as npz:
        by_mf = _frame_masks(npz.files, obj_ids)
    dev = torch.device(device)
    obj_ids = [int(o) for o in obj_ids]
    mul = (max(obj_ids) + 1) if obj_ids else 1
    big = torch.iinfo(torch.int64).max
    local = threading.local()

    def _read(cloud_frame: int):
        """(cloud frame, ids, stacked boolean masks) — the record's conversion and erosion."""
        zf = getattr(local, "zf", None)
        if zf is None:
            zf = local.zf = zipfile.ZipFile(io.BytesIO(raw))
        mf = int(mask_frame_of[cloud_frame])
        ids = by_mf.get(mf, [])
        out = []
        for oid in ids:
            key = f"f{mf}_o{oid}"
            with zf.open(key + ".npy") as fh:
                a = np.lib.format.read_array(fh, allow_pickle=False)
            mask = a.astype(np.uint8)
            if mask.shape[0] != mask_h_ref or mask.shape[1] != mask_w_ref:
                raise RuntimeError(f"mask {key} is {mask.shape[0]}x{mask.shape[1]} and the store "
                                   f"declares {mask_h_ref}x{mask_w_ref} — one store, one grid")
            raw_area = float(np.sum(mask > 0))
            if erosion_iterations <= 0 or raw_area < 2000:
                pass
            elif raw_area < 10000:
                mask = cv2.erode(mask, erosion_kernel, iterations=1)
            else:
                mask = cv2.erode(mask, erosion_kernel, iterations=erosion_iterations)
            out.append(mask.astype(bool))
        stack = np.stack(out) if out else np.zeros((0, mask_h_ref, mask_w_ref), bool)
        return cloud_frame, ids, stack

    point_obj_id = np.full(n_pts, -1, dtype=np.int32)
    point_mask_area = np.full(n_pts, np.inf, dtype=np.float64)
    second_obj_id = np.full(n_pts, -1, dtype=np.int32)
    second_area = np.full(n_pts, np.inf, dtype=np.float64)
    areas_of: Dict[int, List[Tuple[int, float]]] = {}       # id -> [(cloud frame, area)], frames ascending

    frames = sorted(int(f) for f in frame_groups if by_mf.get(int(mask_frame_of[int(f)])))
    n_f = len(frames)
    step = max(1, n_f // 20)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=READ_THREADS, thread_name_prefix="mask-match") as pool:
        pending = [pool.submit(_read, f) for f in frames[:READ_AHEAD]]
        nxt = len(pending)
        for i in range(n_f):
            cf, ids, stack = pending.pop(0).result()
            if nxt < n_f:
                pending.append(pool.submit(_read, frames[nxt]))
                nxt += 1
            done = i + 1
            if done == 1 or done % step == 0 or done == n_f:
                el = time.time() - t0
                log(f"[SegPipeline]   matching frame {done}/{n_f} ({100.0 * done / max(n_f, 1):.0f}%, "
                    f"{el:.0f}s elapsed" + (f", ~{el / done * (n_f - done):.0f}s left)" if done < n_f else ")")
                    + f" on {device}")
            m = torch.from_numpy(stack).to(dev)
            areas = m.flatten(1).sum(1)                                  # int64 pixel counts
            areas_np = areas.cpu().numpy()
            for k, oid in enumerate(ids):
                if areas_np[k] > 0:
                    areas_of.setdefault(oid, []).append((cf, float(areas_np[k])))
            pts = frame_groups[cf]
            r, c = rows_m[pts], cols_m[pts]
            ok = (r >= 0) & (c >= 0)
            pts = pts[ok]
            if not len(pts) or not len(ids):
                continue
            ids_t = torch.as_tensor(ids, dtype=torch.int64, device=dev)
            keyed = torch.where(areas > 0, areas * mul + ids_t, torch.full_like(areas, big))
            rr = torch.as_tensor(r[ok], dtype=torch.int64, device=dev)
            cc = torch.as_tensor(c[ok], dtype=torch.int64, device=dev)
            per = max(1, BLOCK // max(len(ids), 1))
            best_l, second_l = [], []
            for a in range(0, len(pts), per):
                cov = m[:, rr[a:a + per], cc[a:a + per]]                  # (K, P) bool
                key = torch.where(cov, keyed[:, None], torch.full((1, 1), big, device=dev,
                                                                   dtype=torch.int64))
                srt = torch.sort(key, dim=0).values
                best_l.append(srt[0])
                second_l.append(srt[1] if srt.shape[0] > 1 else torch.full_like(srt[0], big))
            best = torch.cat(best_l).cpu().numpy()
            second = torch.cat(second_l).cpu().numpy()
            h1 = best != big
            point_obj_id[pts[h1]] = (best[h1] % mul).astype(np.int32)
            point_mask_area[pts[h1]] = (best[h1] // mul).astype(np.float64)
            h2 = second != big
            second_obj_id[pts[h2]] = (second[h2] % mul).astype(np.int32)
            second_area[pts[h2]] = (second[h2] // mul).astype(np.float64)

    obj_mask_areas: Dict[int, object] = {}
    areas_by_oid_frame: Dict[int, Dict[int, float]] = {}
    for oid in obj_ids:
        fa = areas_of.get(oid, [])
        if fa:
            areas_by_oid_frame[oid] = {int(f): a for f, a in fa}
        obj_mask_areas[oid] = np.mean([a for _, a in fa]) if fa else 0
    return (point_obj_id, point_mask_area, second_obj_id, second_area, obj_mask_areas,
            areas_by_oid_frame)
