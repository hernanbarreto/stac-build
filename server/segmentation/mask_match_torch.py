"""Phase 1 of the mask→cloud projection on the device (USER 2026-11-11: "todo debe ser GPU nada en cpu").

``segmentation.pipeline._match_points_numpy`` — the path of record — walks every masklet and, for each
keyframe of the cloud, reads that masklet's mask from the npz, looks up the pixels of the points born in
the keyframe and keeps, per point, the SMALLEST covering mask area (a tie: the lower id) and the
runner-up (point 117). On pccr 2026-08-24 that is 14,438 masklets x 2765 keyframes, 290,418 masks
inflated one after the other on one thread, the card idle.

Here the walk is by FRAME. A point is born in ONE keyframe, so the only masks that can cover it are
that keyframe's. Per cloud frame, every mask of the frame (of the masklets projected) is inflated ON THE
DEVICE in one batch (segmentation/mask_store_gpu.DeviceMaskStore — nvCOMP's Deflate on the card, USER
2026-11-11: "Nada debe hacerse en cpu") and stays there: the areas are pixel counts, the coverage is the
stack at the points' (row, col), and the winner and the runner-up are the two smallest keys
``area * M + id`` (M = largest id + 1; the areas are integers, so the keys are exact and unique). That is
what the record's strict ``<`` updates in ascending id leave: of two equal areas the first (lower id) stays
the winner, and the runner-up is likewise the next in (area, id) order. Erosion
(``segmentation.mask_erosion_iterations``; 0 since USER 2026-08-29) runs on the device too
(``erode_like_cv2_t``: cv2.erode's pixels, the record's area branches). The per-object area bookkeeping is
the record's arithmetic in the record's order (the mean over the object's frames, ascending).

tests/test_mask_match_torch.py holds the two equal, array for array.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

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
    """``segmentation.pipeline._match_points_numpy`` on ``device``: the same six results. The masks are
    inflated on the device (segmentation/mask_store_gpu.DeviceMaskStore: nvCOMP on the card, zlib on a
    ``cpu`` device), a frame's masks in one batch; the record's erosion runs there too."""
    from segmentation.mask_store_gpu import DeviceMaskStore, erode_like_cv2_t
    dev = torch.device(device)
    store = DeviceMaskStore(masks_path, dev)
    by_mf = _frame_masks(store.files, obj_ids)
    obj_ids = [int(o) for o in obj_ids]
    mul = (max(obj_ids) + 1) if obj_ids else 1
    big = torch.iinfo(torch.int64).max

    def _read(cloud_frame: int):
        """(cloud frame, ids, the frame's masks (k, H, W) on the device) — the record's erosion."""
        mf = int(mask_frame_of[cloud_frame])
        ids = by_mf.get(mf, [])
        keys = [f"f{mf}_o{oid}" for oid in ids]
        stack = store.stack(keys)
        if int(stack.shape[1]) != int(mask_h_ref) or int(stack.shape[2]) != int(mask_w_ref):
            raise RuntimeError(f"mask {keys[0]} is {int(stack.shape[1])}x{int(stack.shape[2])} and the store "
                               f"declares {mask_h_ref}x{mask_w_ref} — one store, one grid")
        if erosion_iterations > 0:
            raw_area = stack.flatten(1).sum(1).tolist()
            for k, ra in enumerate(raw_area):
                if ra < 2000:
                    continue                                        # the record: small masks untouched
                stack[k] = erode_like_cv2_t(stack[k], erosion_kernel, 1 if ra < 10000 else erosion_iterations)
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
    for i in range(n_f):
        cf, ids, m = _read(frames[i])
        done = i + 1
        if done == 1 or done % step == 0 or done == n_f:
            el = time.time() - t0
            log(f"[SegPipeline]   matching frame {done}/{n_f} ({100.0 * done / max(n_f, 1):.0f}%, "
                f"{el:.0f}s elapsed" + (f", ~{el / done * (n_f - done):.0f}s left)" if done < n_f else ")")
                + f" on {device}")
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


def near_ties_t(point_obj_id: np.ndarray, point_mask_area: np.ndarray, second_obj_id: np.ndarray,
                second_area: np.ndarray, area_var: Dict[int, float], error_factor: float, device: str,
                block: int = 50_000_000) -> Tuple[int, int, int]:
    """``segmentation.pipeline._near_ties_numpy`` on ``device``: the per-object area variations as a table
    indexed by id (0.0 for an id without one — the record's ``dict.get`` default), the same float64
    elementwise arithmetic, counted in blocks of points."""
    dev = torch.device(device)
    top = max([int(point_obj_id.max(initial=-1)), int(second_obj_id.max(initial=-1))]
              + [int(k) for k in area_var]) + 1
    table_np = np.zeros(max(top, 1), np.float64)
    for k, v in area_var.items():
        table_np[int(k)] = float(v)
    table = torch.as_tensor(table_np, device=dev)
    n_cont = n_near = n_exact = 0
    for a in range(0, len(second_area), block):
        sa = torch.as_tensor(second_area[a:a + block], device=dev)
        cont = torch.isfinite(sa)
        if not bool(cont.any()):
            continue
        pa = torch.as_tensor(point_mask_area[a:a + block], device=dev)[cont]
        ob = torch.as_tensor(point_obj_id[a:a + block], device=dev)[cont].to(torch.int64)
        os_ = torch.as_tensor(second_obj_id[a:a + block], device=dev)[cont].to(torch.int64)
        gap = sa[cont] - pa
        bar = float(error_factor) * torch.maximum(table[ob], table[os_])
        n_cont += int(cont.sum())
        n_near += int((gap < bar).sum())
        n_exact += int((gap == 0).sum())
    return n_cont, n_near, n_exact
