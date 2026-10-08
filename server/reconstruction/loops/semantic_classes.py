"""Instance class for loop evidence: structural | movable | dynamic (§4.4).

Qwen (server/semantic) looks at an isolated crop of the instance in one of its
keyframes and answers a JSON with the class — nothing else. It never moves a
point: the class only decides whether an instance may PROPOSE a loop
candidate (structural), is ignored (movable) or is excluded (dynamic), and
feeds the per-session movable label list of the verifier. Classes are cached
in the instance store (``scene_r.db``), provenance ``vlm_proposed``.

DETERMINISM (docs/plan_determinismo.md point 95 — 2026-10-08): the cache is keyed by the
CONTENT the verdict depends on — the sha1 of every crop sent, the label, the prompt text,
the served model's identity, the token bound — never by the instance id (an id reassigned
by a re-segmentation inherited another object's class). When the semantic service cannot
be brought up, or a call answers nothing usable, the classification FAILS: a 'default'
class is never recorded for a service that was down. The configured default class is
recorded only for an instance that has NO crop to show (no mask on disk) — a fact of the
data, provenance ``default``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

CLASSES = ("structural", "movable", "dynamic")
CACHE_PREFIX = "loop_class_"               # scene_r.db meta key: loop_class_<content sha>
CACHE_VERSION = 2                          # v1 keyed by instance id (never read again)

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


class LoopClassError(RuntimeError):
    """The classes could not be measured (service down, a call that answered nothing
    usable) — declared, never a default."""


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
    IndexError out of the crop. Ties of mask area break by frame number (the
    sort key), so the same masks pick the same crops on every run.
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
    cand.sort(key=lambda t: (-t[0], t[1]))
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


def _model_identity() -> Optional[dict]:
    """The stable identity of the engine answering (semantic.serve's record of the live
    vLLM), None when none is recorded — recorded as such in the cache key."""
    try:
        from semantic.service import service_identity
        doc = service_identity()
    except Exception:  # noqa: BLE001
        return None
    return dict(doc["identity"]) if doc else None


def cache_key(label: str, crop_sha1s: List[str], max_tokens: int,
              model: Optional[dict]) -> str:
    """The content a verdict depends on (point 95): crops, label, prompt, model, bound."""
    from repro import stable_id
    return CACHE_PREFIX + stable_id(CACHE_VERSION, str(label), _prompt(label), _SYSTEM,
                                    list(crop_sha1s), int(max_tokens), model, n_hex=24)


def classify_instances(output_dir, session_dir, instances: List[dict], cfg,
                       oid_of: Dict[int, Optional[int]], frames_of: Dict[int, List[int]],
                       cloud_to_mask: Optional[Dict[int, int]] = None,
                       log: Callable[[str], None] = print) -> Dict[int, dict]:
    """{instance_id: {class, confidence, provenance}} for every instance,
    cached in the instance store by the CONTENT of the question (:func:`cache_key`).
    ``cfg`` = LoopsConfig.semantic. Raises :class:`LoopClassError` when the service
    cannot answer (never a default for a service that was down)."""
    from phase_r.instance_store import InstanceStore
    from semantic.types import system, user
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    store = InstanceStore(output_dir / "scene_r.db")
    out: Dict[int, dict] = {}
    if not cfg.enabled:
        for inst in instances:
            iid = int(inst.get("instance_id", inst.get("id")))
            out[iid] = {"class": cfg.default_class, "confidence": 0.0, "provenance": "default",
                        "label": str(inst.get("label", "segment")),
                        "reason": "loops.semantic.enabled is false"}
        return out
    # the crops first: they are the cache key, and an instance without any crop (no mask
    # on disk) is classified by the data, not by the model
    model = _model_identity()
    prepared = []
    pending = 0
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        label = str(inst.get("label", "segment"))
        crops = _mask_crop(output_dir, session_dir, iid, oid_of.get(iid),
                           frames_of.get(iid, []), cfg.crops_per_instance, cloud_to_mask or {})
        msg = user(_prompt(label), images=crops) if crops else None
        sha1s = [r.sha1 for r in msg.images] if msg is not None else []
        key = cache_key(label, sha1s, cfg.max_tokens, model)
        cached = store.get_meta(key)
        rec = None
        if cached:
            try:
                rec = json.loads(cached)
            except json.JSONDecodeError:
                rec = None
        if rec is None and crops:
            pending += 1
        prepared.append((iid, label, msg, key, rec))
    client = None
    if pending:
        # the service may be DOWN here (SAM3 stops vLLM for its exclusive window): bring
        # it up and wait, exactly as the VLM stage does — pccr 2026-09-13 21:03: with it
        # down every instance fell to the default class → 0 instance loops, 0 structural
        # constraints. Since 2026-10-08 that outcome is a FAILURE, not a default.
        from config import cfg as _server_cfg
        from semantic.service import ensure_service
        if not ensure_service(_server_cfg, log=log):
            raise LoopClassError("the semantic service did not come up — the instance classes "
                                 "cannot be measured and no default is recorded (point 95)")
        from semantic.client import get_semantic_client
        client = get_semantic_client(consumer="loops.classify")
        if not client.health().get("ok", True):
            raise LoopClassError("the semantic service is unhealthy after its start — the "
                                 "instance classes cannot be measured (point 95)")
    else:
        log(f"[loop-class] all {len(instances)} instance(s) already classified under their "
            f"content key (or have no crop) — the semantic service is not started")
    for iid, label, msg, key, rec in prepared:
        if rec is None:
            if msg is None:
                rec = {"class": cfg.default_class, "confidence": 0.0, "provenance": "default",
                       "label": label, "reason": "no SAM3 mask crop on disk for this instance"}
            else:
                from segmentation.shape_proposer import _chat_json
                parsed, raw = _chat_json(client, [system(_SYSTEM), msg], _SCHEMA,
                                         cfg.max_tokens, log=log)
                if not (isinstance(parsed, dict) and parsed.get("class") in CLASSES):
                    raise LoopClassError(f"instance {iid} '{label}': the VLM answered nothing "
                                         f"usable ({raw[:120]!r}) — no default is recorded "
                                         f"(point 95)")
                rec = {"class": str(parsed["class"]),
                       "confidence": float(parsed.get("confidence", 0.0)),
                       "why": str(parsed.get("why", "")), "provenance": "vlm_proposed",
                       "label": label}
            rec["cache_key"] = key
            store.set_meta(key, json.dumps(rec, sort_keys=True))
        out[iid] = rec
        log(f"[loop-class] instance {iid} '{label}' → {rec['class']} ({rec['provenance']})")
    return out
