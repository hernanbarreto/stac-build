"""Resume / cancel partially-propagated segmentation (USER 2026-09-06).

An interactive segmentation propagates all queued objects across every
keyframe in one pass. With many objects that pass can OOM (31 objects x
1329 frames filled the whole 47 GB) and only the frames reached before the
crash get saved — leaving instances propagated incompletely.

On opening the Segmentation Manager, if any instance is not fully
propagated the UI shows Resume / Cancel:

  * CANCEL deletes every not-fully-propagated instance EVERYWHERE — the
    list, segmentation.json, seg_masks.npz, segmentation_result.json,
    classification.npy, the instance store — they no longer exist.
  * RESUME re-seeds each incomplete instance with MANY interior points
    sampled from ITS OWN saved masks across MANY of its frames (the floor,
    marked over hundreds of frames, gets the most), then re-propagates in
    OBJECT BATCHES (bounded VRAM) and saves — completing them.

Completeness marker: instances carry ``propagated: true`` in
segmentation.json once a full propagation saved them. Legacy instances with
no flag are migrated as complete iff they already matched into
segmentation_result.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_seg(output_dir: Path) -> dict:
    p = output_dir / "segmentation.json"
    return json.loads(p.read_text()) if p.exists() else {"instances": []}


def _mask_obj_ids(output_dir: Path) -> Dict[int, List[int]]:
    """obj_id → sorted list of real frame numbers with a non-empty mask."""
    npz = output_dir / "seg_masks.npz"
    out: Dict[int, List[int]] = {}
    if not npz.exists():
        return out
    z = np.load(npz)
    for k in z.files:
        m = re.match(r"f(\d+)_o(\d+)$", k)
        if not m:
            continue
        arr = z[k]
        if arr.any():
            out.setdefault(int(m.group(2)), []).append(int(m.group(1)))
    for o in out:
        out[o].sort()
    return out


def migrate_flags(output_dir: Path) -> dict:
    """Ensure every instance has a ``propagated`` flag. Legacy instances
    (no flag) are complete iff they appear in segmentation_result.json."""
    seg = _load_seg(output_dir)
    changed = False
    matched_ids = set()
    res_p = output_dir / "segmentation_result.json"
    if res_p.exists():
        try:
            res = json.loads(res_p.read_text())
            for i in res.get("instances", []):
                matched_ids.add(int(i.get("instance_id", i.get("id", -1))))
        except Exception:  # noqa: BLE001
            pass
    for inst in seg.get("instances", []):
        if "propagated" not in inst:
            iid = int(inst.get("instance_id", -1))
            inst["propagated"] = iid in matched_ids
            changed = True
    if changed:
        (output_dir / "segmentation.json").write_text(json.dumps(seg, indent=2))
    return seg


def status(output_dir: Path) -> dict:
    """{needs_resume, incomplete:[{instance_id,label,frames}], complete:n}."""
    seg = migrate_flags(output_dir)
    cover = _mask_obj_ids(output_dir)
    incomplete = []
    complete = 0
    for inst in seg.get("instances", []):
        if inst.get("propagated"):
            complete += 1
            continue
        iid = int(inst.get("instance_id", -1))
        incomplete.append({
            "instance_id": iid,
            "label": inst.get("label") or inst.get("name") or str(iid),
            "frames": len(cover.get(iid - 1, [])),
        })
    return {
        "needs_resume": bool(incomplete),
        "incomplete": incomplete,
        "complete": complete,
    }


def cancel(output_dir: Path, log=print) -> dict:
    """Delete every not-fully-propagated instance from ALL artifacts."""
    seg = migrate_flags(output_dir)
    kill = {int(i["instance_id"]) for i in seg.get("instances", [])
            if not i.get("propagated")}
    if not kill:
        return {"deleted": 0}
    log(f"[Resume] CANCEL — deleting {len(kill)} incomplete instance(s): "
        f"{sorted(kill)}")

    # segmentation.json
    seg["instances"] = [i for i in seg.get("instances", [])
                        if int(i["instance_id"]) not in kill]
    (output_dir / "segmentation.json").write_text(json.dumps(seg, indent=2))

    # seg_masks.npz — drop every f*_o<oid> (oid = instance_id - 1)
    npz = output_dir / "seg_masks.npz"
    if npz.exists():
        z = np.load(npz)
        kill_oids = {i - 1 for i in kill}
        keep = {}
        for k in z.files:
            m = re.match(r"f(\d+)_o(\d+)$", k)
            if m and int(m.group(2)) in kill_oids:
                continue
            if k == "obj_ids":
                keep[k] = np.array([o for o in z[k]
                                    if int(o) not in kill_oids])
            else:
                keep[k] = z[k]
        np.savez_compressed(npz, **keep)

    # segmentation_result.json + classification.npy
    res_p = output_dir / "segmentation_result.json"
    if res_p.exists():
        try:
            res = json.loads(res_p.read_text())
            res["instances"] = [
                i for i in res.get("instances", [])
                if int(i.get("instance_id", i.get("id", -1))) not in kill]
            res_p.write_text(json.dumps(res))
        except Exception as e:  # noqa: BLE001
            log(f"[Resume] result prune failed (non-fatal): {e}")
    cls_p = output_dir / "classification.npy"
    if cls_p.exists():
        try:
            cls = np.load(cls_p)
            for i in kill:
                cls[cls == i] = 0
            np.save(cls_p, cls)
        except Exception as e:  # noqa: BLE001
            log(f"[Resume] classification prune failed (non-fatal): {e}")

    # instance store rebuild (display-frame Q&A/OBBs)
    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        log(f"[Resume] store rebuild skipped: {e}")

    log(f"[Resume] CANCEL done — {len(kill)} instance(s) no longer exist")
    return {"deleted": len(kill), "instance_ids": sorted(kill)}


# ── multi-point seeding from a saved mask ────────────────────────────────
def seed_points_from_mask(mask: np.ndarray, grid: int = 4,
                          erode_px: int = 4, max_pts: int = 10
                          ) -> List[Tuple[int, int]]:
    """Interior positive points spread across the mask: erode to stay off
    the border, grid the bbox, take the most-interior pixel (max distance
    transform) of each occupied cell. Returns [(x, y), ...] in mask pixels."""
    import cv2
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 20:
        return []
    # The image border must count as a boundary: erosion/distance transform
    # otherwise treat off-image as "inside" and pick border pixels as the
    # most interior ones (seeds landed on x=W-1). Pad with zeros.
    mp = np.pad(m, 1, mode="constant", constant_values=0)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (erode_px * 2 + 1, erode_px * 2 + 1))
        e = cv2.erode(mp, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if e.sum() >= 20:
            mp = e
    dist = cv2.distanceTransform(mp, cv2.DIST_L2, 3)[1:-1, 1:-1]
    m = mp[1:-1, 1:-1]
    ys, xs = np.where(m > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pts = []
    gy = max(1, min(grid, y1 - y0))
    gx = max(1, min(grid, x1 - x0))
    for iy in range(gy):
        for ix in range(gx):
            ry0 = y0 + (y1 - y0) * iy // gy
            ry1 = y0 + (y1 - y0) * (iy + 1) // gy
            rx0 = x0 + (x1 - x0) * ix // gx
            rx1 = x0 + (x1 - x0) * (ix + 1) // gx
            sub = dist[ry0:ry1 + 1, rx0:rx1 + 1]
            if sub.size == 0 or sub.max() <= 0:
                continue
            ly, lx = np.unravel_index(int(sub.argmax()), sub.shape)
            pts.append((int(rx0 + lx), int(ry0 + ly)))
    # dedupe (thin masks make neighbouring cells pick the same pixel), then
    # cap keeping the most-interior ones
    pts = list(dict.fromkeys(pts))
    if len(pts) > max_pts:
        pts.sort(key=lambda p: -float(dist[p[1], p[0]]))
        pts = pts[:max_pts]
    return pts


def pick_seed_frames(frames: List[int], min_f: int = 3, max_f: int = 15
                     ) -> List[int]:
    """Evenly-spread subset of an object's frames — more for objects seen in
    many frames (the floor gets the max), few for small ones."""
    n = len(frames)
    if n <= min_f:
        return list(frames)
    target = int(np.clip(round(n / 20), min_f, max_f))
    idx = np.linspace(0, n - 1, target).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def build_resume_seeds(output_dir: Path, grid: int = 4, erode_px: int = 4,
                       max_pts_per_frame: int = 10, min_frames: int = 3,
                       max_frames: int = 15, log=print
                       ) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
    """For every incomplete instance: {oid: {real_frame: [(x,y),...]}}
    sampled from its saved masks. Positives only (a mask alone cannot tell
    background from another identical object — no invented negatives)."""
    seg = migrate_flags(output_dir)
    incomplete_oids = {int(i["instance_id"]) - 1
                       for i in seg.get("instances", [])
                       if not i.get("propagated")}
    npz = output_dir / "seg_masks.npz"
    if not npz.exists() or not incomplete_oids:
        return {}
    z = np.load(npz)
    by_oid: Dict[int, List[int]] = {}
    for k in z.files:
        m = re.match(r"f(\d+)_o(\d+)$", k)
        if m and int(m.group(2)) in incomplete_oids and z[k].any():
            by_oid.setdefault(int(m.group(2)), []).append(int(m.group(1)))
    seeds: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    for oid, frames in by_oid.items():
        frames.sort()
        sel = pick_seed_frames(frames, min_frames, max_frames)
        fseeds = {}
        for rf in sel:
            pts = seed_points_from_mask(z[f"f{rf}_o{oid}"], grid, erode_px,
                                        max_pts_per_frame)
            if pts:
                fseeds[rf] = pts
        if fseeds:
            seeds[oid] = fseeds
            npt = sum(len(v) for v in fseeds.values())
            log(f"[Resume] o{oid}: {len(fseeds)} seed frames, {npt} points")
    return seeds
