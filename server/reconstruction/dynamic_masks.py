"""
Dynamic-class masks for the PGSR photometric loss (precision task, Phase D).
================================================================================
The SAM3 stage runs BEFORE the PGSR stage in the pipeline (RECONSTRUCTION →
VLM → SAM3 → CLOUDCOMPY → [PGSR] → TSDF), so its masks are available when the
photometric optimization starts. This module unions, per keyframe, the SAM3
masks of every instance whose canonical label is a DYNAMIC class in the
segmentation vocabulary (person / mobile equipment / train, …) and writes one
PNG per keyframe (255 = dynamic = EXCLUDED from the photometric loss):

    output/pgsr_scene/masks/<frame>.jpg.png     (masks/<image_name>.png)

Sessions segmented manually through the Segmentation Manager produce the same
seg_masks.npz/segmentation.json artifacts, so manual sessions work identically.
No SAM3 artifacts ⇒ no masks (returns 0) — a STATIC scene trains unmasked;
this is logged, never silent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("DynamicMasks")


def _dynamic_labels() -> set:
    from segmentation.autoprompt.vocabulary import get_vocabulary
    try:
        return set(get_vocabulary().dynamic_labels())
    except Exception:  # noqa: BLE001 — vocabulary is config-driven; fall back to core set
        return {"person", "worker", "vehicle", "truck", "excavator", "crane",
                "train", "forklift", "machine"}


def generate(output_dir: Path, dst_dir: Path, log=None,
             _labels_override: Optional[set] = None) -> int:
    """Union the dynamic-class SAM3 masks per keyframe → dst_dir/<name>.png.
    Returns the number of mask files written (0 when no SAM3 artifacts or no
    dynamic instances — logged)."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir, dst_dir = Path(output_dir), Path(dst_dir)
    npz_path = output_dir / "seg_masks.npz"
    meta_path = output_dir / "segmentation.json"
    if not npz_path.exists() or not meta_path.exists():
        _log("no SAM3 artifacts (seg_masks.npz/segmentation.json) — training "
             "unmasked (static-scene assumption, explicit)")
        return 0
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        raise RuntimeError(f"dynamic_masks: segmentation.json unreadable: {e}")
    dyn = _labels_override if _labels_override is not None else _dynamic_labels()
    obj_label: Dict[int, str] = {}
    for inst in meta.get("instances", []):
        try:
            obj_label[int(inst.get("id"))] = str(inst.get("label", "")).lower()
        except Exception:
            continue
    dyn_ids = {i for i, lab in obj_label.items()
               if any(d in lab for d in dyn)}
    if not dyn_ids:
        _log(f"SAM3 artifacts present but no DYNAMIC-class instances "
             f"({len(obj_label)} instances, dynamic vocab {sorted(dyn)[:6]}…) — "
             f"training unmasked")
        return 0

    from PIL import Image
    data = np.load(npz_path)
    per_frame: Dict[int, np.ndarray] = {}
    for key in data.files:
        if not (key.startswith("f") and "_o" in key):
            continue
        try:
            fpart, opart = key[1:].split("_o")
            fnum, oid = int(fpart), int(opart)
        except Exception:
            continue
        if oid not in dyn_ids:
            continue
        m = np.asarray(data[key]).astype(bool)
        if fnum in per_frame:
            if per_frame[fnum].shape == m.shape:
                per_frame[fnum] |= m
        else:
            per_frame[fnum] = m
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fnum, m in per_frame.items():
        Image.fromarray((m * 255).astype(np.uint8)) \
            .save(dst_dir / f"{fnum:06d}.jpg.png")
        n += 1
    _log(f"dynamic masks: {n} keyframes masked ({len(dyn_ids)} dynamic instances)")
    return n
