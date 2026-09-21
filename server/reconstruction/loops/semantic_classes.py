"""Instance class for loop evidence: structural | movable | dynamic (§4.4).

Qwen (server/semantic) looks at an isolated crop of the instance in one of its
keyframes and answers a JSON with the class — nothing else. It never moves a
point: the class only decides whether an instance may PROPOSE a loop
candidate (structural), is ignored (movable) or is excluded (dynamic), and
feeds the per-session movable label list of the verifier. Classes are cached
in the instance store (``scene_r.db`` meta ``loop_class_<iid>``), provenance
``vlm_proposed``. When the VLM is unavailable the configured default class is
recorded with provenance ``default`` — never silently structural.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

CLASSES = ("structural", "movable", "dynamic")

_SYSTEM = ("You classify one object of a construction / infrastructure site for a "
           "measurement system. Answer ONLY the JSON requested.")

_SCHEMA = {
    "type": "object",
    "properties": {
        "class": {"type": "string", "enum": list(CLASSES)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "why": {"type": "string"},
    },
    "required": ["class", "confidence"],
}


def _prompt(label: str) -> str:
    return (f"The highlighted object was segmented with the label '{label}'. Classify it:\n"
            f"- structural: fixed part of the building or infrastructure (wall, column, "
            f"floor, ceiling, beam, platform, track, rail, duct, stair, fixed door frame)\n"
            f"- movable: an object that can be moved between visits (cart, ladder, box, "
            f"chair, table, tool, vehicle, door leaf)\n"
            f"- dynamic: something moving during the capture (person, worker, machine "
            f"in motion)\n"
            f"Reply with JSON: {{\"class\": ..., \"confidence\": ..., \"why\": ...}}")


def _mask_crop(output_dir: Path, session_dir: Path, iid: int, oid: Optional[int],
               frames: List[int], crops: int, cloud_to_mask: Dict[int, int]):
    """Isolated crops (image with the mask kept, background darkened) for up to
    ``crops`` frames where the instance's mask is largest.

    ``frames`` are the cloud's frame_global — REAL video frame numbers — while
    the masks are keyed by KEYFRAME POSITION. ``_load_frame_rgb`` wants the
    first and ``z[key]`` the second, so ``cloud_to_mask`` translates for the
    key and only for the key. Without it the lookup hit only the handful of
    numbers that collide by accident: pccr 2026-09-14 classified 4 instances of
    61 and the other 57 fell to the default class, while the few that did
    "hit" paired an image with another keyframe's mask and raised
    IndexError out of the crop.
    """
    from segmentation.shape_proposer import _load_frame_rgb, _isolated_crop
    p = output_dir / "seg_masks.npz"
    if oid is None or not p.exists():
        return []
    z = np.load(p, allow_pickle=True)
    cand = []
    for f in frames:
        key = f"f{cloud_to_mask.get(int(f), int(f))}_o{int(oid)}"
        if key in z.files:
            m = z[key]
            cand.append((int(m.sum()), int(f), key))
    cand.sort(reverse=True)
    out = []
    for _area, f, key in cand[:crops]:
        img = _load_frame_rgb(session_dir, f)
        if img is None:
            continue
        m = z[key]
        mask_rgb = np.repeat((m > 0)[..., None], 3, axis=2).astype(np.uint8) * 255
        try:
            out.append(_isolated_crop(img, mask_rgb))
        except Exception as e:  # noqa: BLE001 — a bad crop is skipped, the VLM sees the rest
            print(f"[loop-class] crop failed for instance {iid} frame {f}: {e}")
    return out


def classify_instances(output_dir, session_dir, instances: List[dict], cfg,
                       oid_of: Dict[int, Optional[int]], frames_of: Dict[int, List[int]],
                       cloud_to_mask: Optional[Dict[int, int]] = None,
                       log: Callable[[str], None] = print) -> Dict[int, dict]:
    """{instance_id: {class, confidence, provenance}} for every instance,
    cached in the instance store. ``cfg`` = LoopsConfig.semantic."""
    from phase_r.instance_store import InstanceStore
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    store = InstanceStore(output_dir / "scene_r.db")
    out: Dict[int, dict] = {}
    client = None
    # The class of an object is a property of the OBJECT: a desk is still a
    # desk after the cloud is corrected. Every instance already classified is
    # read from the store below — so the service is only worth starting when
    # something is actually missing. It used to start unconditionally: on pccr
    # 2026-09-21 certify booted vLLM (24 GB, ~4 min) and then classified ZERO
    # instances, every one of them already cached.
    _pending = [i for i in instances
                if not store.get_meta(
                    f"loop_class_{int(i.get('instance_id', i.get('id')))}")]
    if not _pending:
        log(f"[loop-class] all {len(instances)} instance(s) already classified "
            f"— the semantic service is not started")
    if cfg.enabled and _pending:
        try:
            # the service may be DOWN here (SAM3 stops vLLM for its exclusive
            # window): bring it up and wait, exactly as the VLM stage does —
            # pccr 2026-09-13 21:03: with it down every instance fell to the
            # default class → 0 instance loops, 0 structural constraints
            from config import cfg as _server_cfg
            from semantic.service import ensure_service
            if not ensure_service(_server_cfg, log=log):
                raise RuntimeError("semantic service did not come up")
            from semantic.client import get_semantic_client
            client = get_semantic_client(consumer="loops.classify")
            if not client.health().get("ok", True):
                raise RuntimeError("semantic service unhealthy after start")
        except Exception as e:  # noqa: BLE001 — declared below, never silent
            log(f"[loop-class] semantic service unavailable ({e}) — default class "
                f"'{cfg.default_class}' recorded for unclassified instances")
            client = None
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        label = str(inst.get("label", "segment"))
        cached = store.get_meta(f"loop_class_{iid}")
        if cached:
            try:
                out[iid] = json.loads(cached)
                continue
            except json.JSONDecodeError:
                pass
        rec = {"class": cfg.default_class, "confidence": 0.0, "provenance": "default",
               "label": label}
        if client is not None:
            crops = _mask_crop(output_dir, session_dir, iid, oid_of.get(iid),
                               frames_of.get(iid, []), cfg.crops_per_instance,
                               cloud_to_mask or {})
            if crops:
                from segmentation.shape_proposer import _chat_json
                from semantic.types import system, user
                parsed, _raw = _chat_json(
                    client, [system(_SYSTEM), user(_prompt(label), images=crops)],
                    _SCHEMA, cfg.max_tokens, log=log)
                if isinstance(parsed, dict) and parsed.get("class") in CLASSES:
                    rec = {"class": str(parsed["class"]),
                           "confidence": float(parsed.get("confidence", 0.0)),
                           "why": str(parsed.get("why", "")), "provenance": "vlm_proposed",
                           "label": label}
        store.set_meta(f"loop_class_{iid}", json.dumps(rec))
        out[iid] = rec
        log(f"[loop-class] instance {iid} '{label}' → {rec['class']} ({rec['provenance']})")
    return out
