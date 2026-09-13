"""Instance split — the geometry declared a SAM3 identity impossible (§4.4/§4.5).

A re-identification that fails the spatial gate is NOT a loop: SAM3 fused two
distinct objects under one id. The fused instance is divided in two, in BOTH
identity spaces (``segmentation_result.json`` instance_id and the mask object
id of ``segmentation.json`` / ``seg_masks.npz``), the moved points are
repainted onto the new mask id, ``classification.npy`` and the instance store
are rebuilt, and the operation is appended to ``segmentation_ledger.jsonl``.
Same recipe as ``segmentation.erase.erase_spheres`` (create + reassign), driven
by an explicit point-index set instead of brush zones. Nothing is deleted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np

SEG_LEDGER = "segmentation_ledger.jsonl"


def split_instance(output_dir, iid: int, idx_new: np.ndarray, reason: dict,
                   log: Callable[[str], None] = print) -> Dict:
    """Move the cloud points ``idx_new`` (global indices) out of instance
    ``iid`` into a NEW instance carrying the same label. Returns
    {new_instance_id, new_oid, n_moved, ledger}."""
    from segmentation.pipeline import _load_ply_origins, _compute_obb
    from segmentation.erase import (_mask_obj_by_iid, _atomic_savez, _write_classification,
                                    raw_to_display, _NEW_SEGMENT_COLORS)

    output_dir = Path(output_dir)
    res_path = output_dir / "segmentation_result.json"
    ply = output_dir / "cleaned_cloud.ply"
    if not res_path.exists() or not ply.exists():
        raise FileNotFoundError("session has no cleaned cloud / segmentation to split")
    origins = _load_ply_origins(ply)
    if origins is None:
        raise RuntimeError("cloud has no origin fields — masks cannot be repainted")
    xyz, fg, pr, pc = origins
    N = len(xyz)
    idx_new = np.unique(np.asarray(idx_new, dtype=np.int64))
    idx_new = idx_new[(idx_new >= 0) & (idx_new < N)]

    result = json.loads(res_path.read_text())
    instances = result.get("instances") or []
    src = next((i for i in instances
                if int(i.get("instance_id", i.get("id"))) == int(iid)), None)
    if src is None:
        raise ValueError(f"instance {iid} not found")
    gi = np.asarray(src.get("globalIndices") or [], dtype=np.int64)
    moving = np.intersect1d(gi, idx_new)
    if len(moving) == 0:
        raise ValueError(f"none of the {len(idx_new)} indices belong to instance {iid}")
    if len(moving) == len(gi):
        raise ValueError(f"split would empty instance {iid} — the two clusters must both "
                         f"keep points")
    remaining = np.setdiff1d(gi, moving)

    # identity in both spaces (erase.py recipe)
    new_iid = 1 + max([int(i.get("instance_id", i.get("id", 0))) for i in instances], default=0)
    seg_json = output_dir / "segmentation.json"
    meta = json.loads(seg_json.read_text()) if seg_json.exists() else {}
    meta_insts = meta.get("instances") or []
    new_oid = 1 + max([int(e.get("id", 0)) for e in meta_insts], default=0)
    color = _NEW_SEGMENT_COLORS[(new_iid - 1) % len(_NEW_SEGMENT_COLORS)]
    label = str(src.get("label", "segment"))
    new_inst = {"id": int(new_iid), "label": label, "instance_id": int(new_iid),
                "color": color, "total_points": int(len(moving)),
                "globalIndices": moving.tolist(),
                "split_from": int(iid)}
    if len(moving) >= 4:
        new_inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[moving]))
    src["globalIndices"] = remaining.tolist()
    src["total_points"] = int(len(remaining))
    if len(remaining) >= 4:
        src["obb"] = _compute_obb(raw_to_display(output_dir, xyz[remaining]))
    instances.append(new_inst)
    meta_insts.append({"id": int(new_oid), "instance_id": int(new_iid),
                       "label": label, "color": color})
    meta["instances"] = meta_insts
    seg_json.write_text(json.dumps(meta, indent=2))

    # masks: pixels of the moved points leave the source oid and join the new one
    masks_path = output_dir / "seg_masks.npz"
    n_px = 0
    if masks_path.exists():
        z = np.load(masks_path, allow_pickle=True)
        masks = {k: z[k] for k in z.files}
        oid_map = _mask_obj_by_iid(output_dir)
        src_oid = oid_map.get(int(iid))
        if "obj_ids" in masks:
            masks["obj_ids"] = np.append(np.asarray(masks["obj_ids"], dtype=np.int32),
                                         np.int32(new_oid))
        orig_h, orig_w = float(pr.max() + 1), float(pc.max() + 1)
        sr = masks.get("scaled_res")
        mh0, mw0 = ((int(sr[0]), int(sr[1])) if sr is not None
                    else next(m.shape[:2] for k, m in masks.items() if k.startswith("f")))
        for f in np.unique(fg[moving]):
            sel = moving[fg[moving] == f]
            dst_key = f"f{int(f)}_o{int(new_oid)}"
            m_new = masks.get(dst_key)
            if m_new is None:
                m_new = np.zeros((mh0, mw0), dtype=np.uint8)
            mh, mw = m_new.shape[:2]
            rr = np.clip((pr[sel] * (mh / orig_h)).astype(np.int64), 0, mh - 1)
            cc = np.clip((pc[sel] * (mw / orig_w)).astype(np.int64), 0, mw - 1)
            m_new = m_new.copy()
            m_new[rr, cc] = 1
            masks[dst_key] = m_new
            n_px += int(len(rr))
            if src_oid is not None:
                src_key = f"f{int(f)}_o{int(src_oid)}"
                m_src = masks.get(src_key)
                if m_src is not None:
                    m_src = m_src.copy()
                    m_src[rr, cc] = 0
                    masks[src_key] = m_src
        _atomic_savez(masks_path, masks)

    result["segmented_points"] = sum(int(i.get("total_points") or 0) for i in instances)
    result["coverage"] = round(result["segmented_points"] / max(1, int(result.get("total_points") or N)), 4)
    res_path.write_text(json.dumps(result))
    _write_classification(output_dir, instances, N)
    from segmentation.pipeline import rebuild_instance_store
    rebuild_instance_store(output_dir)

    entry = {"type": "instance_split", "at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "source_instance": int(iid), "new_instance": int(new_iid), "new_oid": int(new_oid),
             "label": label, "n_moved": int(len(moving)), "n_remaining": int(len(remaining)),
             "mask_pixels_repainted": int(n_px), "reason": reason,
             "provenance": "tool_measured"}
    with open(output_dir / SEG_LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log(f"[instance-split] instance {iid} ({label}) → {iid} + {new_iid}: {len(moving)} "
        f"point(s) moved, {n_px} mask px repainted — {reason.get('reason', '')}")
    return {"new_instance_id": int(new_iid), "new_oid": int(new_oid),
            "n_moved": int(len(moving)), "ledger": entry}
