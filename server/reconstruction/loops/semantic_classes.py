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
The engine (close of wave 2, 2026-10-08): the classes are measured on THIS JOB's own
engine — semantic.service.job_engine launched from the job's frozen configuration (the
``config`` the caller hands in; None: the session's frozen run configuration when it holds
one, else the server's, for a CLI) — and the model identity in the cache key is the
VERIFIED identity of that engine, never one read before a launch (it was None then: every
key said 'no model'). The identity is computed WITHOUT a launch first (semantic.serve.
expected_identity: the weights' sha256, the card, the versions); vLLM comes up only for an
instance the store does not hold under that key, and must come up AS that identity. A
caller that already holds an engine passes ``client`` and ``service`` (its verified
identity).
"""

from __future__ import annotations

import contextlib
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


def _live_identity() -> Optional[dict]:
    """The stable identity of the engine answering NOW (semantic.serve's record of the live
    vLLM, read through semantic.service.service_identity), None when none is recorded."""
    from semantic.service import service_identity
    doc = service_identity()
    return dict(doc["identity"]) if doc else None


def cache_key(label: str, crop_sha1s: List[str], max_tokens: int,
              model: Optional[dict]) -> str:
    """The content a verdict depends on (point 95): crops, label, prompt, model, bound."""
    from repro import stable_id
    return CACHE_PREFIX + stable_id(CACHE_VERSION, str(label), _prompt(label), _SYSTEM,
                                    list(crop_sha1s), int(max_tokens), model, n_hex=24)


def _cached_record(store, key: str) -> Optional[dict]:
    """The verdict the instance store holds under ``key`` (None: not classified under this
    content, or a record that is not readable JSON — asked again, never trusted)."""
    cached = store.get_meta(key)
    if not cached:
        return None
    try:
        rec = json.loads(cached)
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) else None


def _job_configuration(config: Optional[dict], session_dir: Path,
                       log: Callable[[str], None]) -> tuple:
    """ONE configuration for this job's engine: ``config`` (the dict the stage received)
    with the session's frozen output/run_config.yaml beside it when there is one (what the
    launcher reads its block from); with None, the FROZEN file's own content when the
    session holds one — the engine is launched from it, so its live identity is verified
    against the same block, never against a server config.yaml that drifted since the
    freeze — else the server's config.yaml (a CLI on a session nobody froze). Returns
    (config, run_config_path or None)."""
    from intake.run_config import (effective_config, has_run_config, load_run_config,
                                   run_config_path)
    run_cfg = run_config_path(session_dir) if has_run_config(session_dir) else None
    if config is None:
        if run_cfg is not None:
            config = effective_config(load_run_config(session_dir)[0])
            log(f"[loop-class] configuration: the session's frozen output/{run_cfg.name}")
        else:
            from config import cfg as config
            log("[loop-class] configuration: the server's config.yaml (no frozen run "
                "configuration in the session)")
    return config, run_cfg


def _no_crop_record(cfg, label: str) -> dict:
    """The record of an instance that has NO crop to show — a fact of the data (no SAM3 mask
    on disk), the one case the configured default class is ever written."""
    return {"class": cfg.default_class, "confidence": 0.0, "provenance": "default",
            "label": label, "reason": "no SAM3 mask crop on disk for this instance"}


def classify_instances(output_dir, session_dir, instances: List[dict], cfg,
                       oid_of: Dict[int, Optional[int]], frames_of: Dict[int, List[int]],
                       cloud_to_mask: Optional[Dict[int, int]] = None,
                       log: Callable[[str], None] = print, *,
                       config: Optional[dict] = None, client=None,
                       service: Optional[dict] = None,
                       cancelled: Optional[Callable[[], bool]] = None) -> Dict[int, dict]:
    """{instance_id: {class, confidence, provenance}} for every instance,
    cached in the instance store by the CONTENT of the question (:func:`cache_key`).
    ``cfg`` = LoopsConfig.semantic. ``config`` = the job's frozen configuration (the dict
    the stage received; None → the server's config.yaml, for a CLI): what this job's own
    engine is launched from (semantic.service.job_engine — lease, verified stop, launch
    from the frozen file, identity verified, stopped at the end). A caller that already
    holds an engine passes ``client`` with ``service`` (that engine's verified identity;
    without it the live launcher record is read, and none FAILS).

    The crops are prepared FIRST: an instance without any crop (no mask on disk) is
    classified by the data without an engine, and when no instance has a crop no engine is
    launched at all. Otherwise the engine comes up, its identity enters every cache key,
    and only the instances whose key is not in the store are asked. Raises
    :class:`LoopClassError` when the engine cannot come up, answers nothing usable or has
    no identity — never a default for a service that was down (point 95)."""
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
    prepared = []
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        label = str(inst.get("label", "segment"))
        crops = _mask_crop(output_dir, session_dir, iid, oid_of.get(iid),
                           frames_of.get(iid, []), cfg.crops_per_instance, cloud_to_mask or {})
        msg = user(_prompt(label), images=crops) if crops else None
        prepared.append((iid, label, msg))
    if not any(msg is not None for _i, _l, msg in prepared):
        log(f"[loop-class] none of the {len(instances)} instance(s) has a SAM3 mask crop on "
            f"disk — classified by the data (default class), no engine is started")
        for iid, label, _msg in prepared:
            out[iid] = _no_crop_record(cfg, label)
            log(f"[loop-class] instance {iid} '{label}' → {out[iid]['class']} (default)")
        return out

    # THE ENGINE'S IDENTITY, then the keys (point 95): the model identity a verdict is cached
    # under is the VERIFIED identity of the engine that answers — never one read before a
    # launch. The engine is this job's own, launched from the job's frozen configuration; it
    # may be DOWN here (SAM3 stops vLLM for its exclusive window) — pccr 2026-09-13 21:03:
    # with it down every instance fell to the default class → 0 instance loops, 0 structural
    # constraints. Since 2026-10-08 that outcome is a FAILURE, not a default.
    # WITHOUT A LAUNCH when nothing is to be asked (close of wave 2): the identity this job's
    # engine WOULD have is computed by semantic.serve.expected_identity (the weights verified
    # by sha256, the card, the versions — exactly what a launch records, in the semantic env);
    # the keys are built with it and looked up; only an instance missing from the store
    # brings vLLM up, and the engine that comes up must then BE that identity
    # (engine_as_expected). Every verdict cached: no load, no verified stop (~3 min of vLLM
    # per certification, the same cost the VLM stage's reuse removed). DECLARED: an instance
    # to ask costs the weights' verification twice (the identity-only run, then the launch's
    # own) — the price of knowing whether a launch is needed at all.
    from semantic.serve import ServeError, engine_as_expected, expected_identity
    from semantic.service import EngineBusy, EngineIdentityError, EngineUnavailable
    launched = False
    try:
        if client is None:
            config, run_cfg = _job_configuration(config, session_dir, log)
            backend = (config.get("autoprompt", {}) or {}).get("backend", "qwen_local")
            expected = expected_identity(backend, run_cfg, log=log)
            n_crops = sum(1 for _i, _l, msg in prepared if msg is not None)
            uncached = [iid for iid, label, msg in prepared if msg is not None
                        and _cached_record(store, cache_key(label, [r.sha1 for r in msg.images],
                                                            cfg.max_tokens, expected)) is None]
            if not uncached:
                log(f"[loop-class] DECLARED: every one of the {n_crops} instance(s) with a crop "
                    f"is already classified under its content key with engine identity "
                    f"{str(expected.get('sha256', ''))[:12]} (verified without a launch: "
                    f"weights sha256, card, versions) — no vLLM is launched, hashed again or "
                    f"stopped")
                engine = contextlib.nullcontext(expected)
            else:
                from semantic.service import job_engine
                log(f"[loop-class] {len(uncached)} of {n_crops} instance(s) with a crop not yet "
                    f"classified under their content key — this job's vLLM is launched "
                    f"(identity {str(expected.get('sha256', ''))[:12]} expected)")
                engine = engine_as_expected(
                    job_engine(config, backend=backend, owner="loops.classify",
                               stage="loop classes", session_dir=session_dir,
                               run_config_path=run_cfg, log=log, cancelled=cancelled),
                    expected, stage="loop classes")
                launched = True
        else:
            engine = contextlib.nullcontext(service)
        with engine as identity:
            if launched:
                from semantic.client import get_semantic_client
                client = get_semantic_client(consumer="loops.classify")
            model = dict(identity) if isinstance(identity, dict) else _live_identity()
            if not model:
                raise LoopClassError("the engine answering has no identity record (semantic."
                                     "serve writes one before exec) — the classes would be "
                                     "cached under no model; nothing is asked (point 95)")
            n_cached = 0
            for iid, label, msg in prepared:
                was_cached = False
                if msg is None:
                    rec = _no_crop_record(cfg, label)
                else:
                    sha1s = [r.sha1 for r in msg.images]
                    key = cache_key(label, sha1s, cfg.max_tokens, model)
                    rec = _cached_record(store, key)
                    if rec is not None:
                        n_cached += 1
                        was_cached = True
                    elif client is None:
                        raise LoopClassError(f"instance {iid} '{label}' is not classified under "
                                             f"its content key and no engine was launched for "
                                             f"it — nothing is asked (point 95)")
                    else:
                        from segmentation.shape_proposer import _chat_json
                        parsed, raw = _chat_json(client, [system(_SYSTEM), msg], _SCHEMA,
                                                 cfg.max_tokens, log=log)
                        if not (isinstance(parsed, dict) and parsed.get("class") in CLASSES):
                            raise LoopClassError(f"instance {iid} '{label}': the VLM answered "
                                                 f"nothing usable ({raw[:120]!r}) — no default "
                                                 f"is recorded (point 95)")
                        rec = {"class": str(parsed["class"]),
                               "confidence": float(parsed.get("confidence", 0.0)),
                               "why": str(parsed.get("why", "")), "provenance": "vlm_proposed",
                               "label": label, "cache_key": key}
                        store.set_meta(key, json.dumps(rec, sort_keys=True))
                out[iid] = rec
                log(f"[loop-class] instance {iid} '{label}' → {rec['class']} "
                    f"({rec['provenance']}{', cached' if was_cached else ''})")
            log(f"[loop-class] {len(prepared)} instance(s): {n_cached} already classified under "
                f"their content key (engine {str(model.get('sha256', ''))[:12]})")
    except LoopClassError:
        raise
    except (EngineBusy, EngineUnavailable, EngineIdentityError, ServeError) as e:
        # the engine: busy, could not come up, another identity than this job's, or an
        # identity that cannot be computed — a programming error is not wrapped (it travels)
        raise LoopClassError(f"the instance classes cannot be measured ({type(e).__name__}: "
                             f"{e}) — no default is recorded (point 95)") from e
    return out
