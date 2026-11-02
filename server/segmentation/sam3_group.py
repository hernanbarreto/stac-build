"""SAM 3.1's concepts advanced together over one frame set, the vision backbone computed once
(USER 2026-11-02: "la etapa más lenta de todo el pipeline es sam3 … tenés que investigar cuál es la
mejor estrategia de sam3 para mejorar esto, porque tarda más que todo el pipeline completo"; "ok"
to sharing the vision backbone and to not keeping the masks the batch flow never reads).

MEASURED on station 2026-11-02 (438 keyframes of 1024x576, 75 concepts, batches 0–249 / 240–437):
~90 s per concept, ~110 min in all — longer than the whole reconstruction before it.

1. SAM 3.1's text pathway is ONE concept per propagation (``add_prompt`` starts with
   ``reset_state``: "since it's a semantic prompt, we start over"; Object Multiplex tracks the many
   OBJECTS of that one concept jointly, never several concepts). Its detector computes the VISION
   BACKBONE — the 32-block ViT at 1008 px and its three necks (detector, interactive, propagation)
   — for every 16-frame chunk again for every concept (``Sam3Image._get_img_feats``, called from the
   batched grounding), and the image features do not depend on the text. :class:`SharedVision`
   computes them once per chunk for a GROUP of concepts that advance through the same frames
   together and hands every member the SAME tensors; the text encoder, the detector, the tracker
   and every decision after them run exactly as before, one concept per session. SAM 3.1's
   inference draws no random number (bucket assignment ``random=False`` / ``self.training``), so
   interleaving the members changes nothing either.

2. The vendor caches every yielded frame's video-resolution mask of every object in the session
   (``cached_frame_outputs``) for its interactive demo's re-fetch and refinement; the batch flow
   never reads it (the backward pass from the prompt frame 0 is empty: ``range(-1, -1, -1)``), and
   a session kept open for the next concept held the previous concept's whole state until that
   concept's ``add_prompt`` reset it. On station 'light' (181 objects) ran out of memory at 69 GB
   allocated with 74.5 GB free once the sessions were released — the out-of-memory stop cut its
   batch into two with new ids at the seam. :func:`frame_outputs_not_kept` stops that cache during
   a batch propagation, and the wrapper resets a session as soon as its masks are copied out (the
   next ``add_prompt`` resets it anyway: ``reset_state`` is idempotent). Equal masks wherever the
   card did not run out of memory; fewer out-of-memory stops where it did (USER: "ok").
"""
from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple

import torch


class SharedVision:
    """The detector's vision-backbone output of a chunk of frames, computed once and served to
    every concept of a group over the same frame set.

    Installed on the detector INSTANCE (``detector._get_img_feats``) for the duration of a
    ``with`` block; outside it the detector is untouched. The key is the frame set (the session's
    frame tensor or loader, the SAME object for every member: :func:`clone_session` shares it) and
    the sorted unique frame ids of the call; an entry holds what ``forward_image`` produced plus
    the vendor's ``id_mapping``. On a hit the vendor's own code indexes the cached features (the
    branch it takes when ``backbone_fpn`` is present), so the features each member reads are the
    ones a recomputation would have produced, bit for bit."""

    def __init__(self, detector, capacity: int):
        if int(capacity) < 1:
            raise ValueError(f"SharedVision capacity must be >= 1 (got {capacity})")
        self.detector = detector
        self.capacity = int(capacity)
        self._entries: "OrderedDict[Tuple[int, Tuple[int, ...]], Tuple[Any, Dict[str, Any]]]" = OrderedDict()
        self._installed = False
        self._had_attr = False
        self._prev_attr = None
        self.computed = 0          # chunks whose backbone ran
        self.served = 0            # chunks served from an entry

    def __enter__(self) -> "SharedVision":
        det = self.detector
        self._had_attr = "_get_img_feats" in vars(det)
        self._prev_attr = vars(det).get("_get_img_feats")
        orig = det._get_img_feats

        def shared(backbone_out, img_ids):
            return self._get(orig, backbone_out, img_ids)

        det._get_img_feats = shared
        self._installed = True
        return self

    def __exit__(self, *exc) -> None:
        det = self.detector
        if self._installed:
            if self._had_attr:
                det._get_img_feats = self._prev_attr
            else:
                del det._get_img_feats
            self._installed = False
        self._entries.clear()

    def _get(self, orig, backbone_out, img_ids):
        if "backbone_fpn" in backbone_out:
            return orig(backbone_out, img_ids)          # already computed in this call chain
        img_batch = backbone_out["img_batch_all_stages"]
        frames = getattr(img_batch, "tensors", img_batch)
        ids = torch.unique(img_ids) if img_ids.numel() > 1 else img_ids.reshape(-1)
        key = (id(frames), tuple(int(i) for i in ids.tolist()))
        hit = self._entries.get(key)
        if hit is not None and hit[0] is frames:
            self._entries.move_to_end(key)
            self.served += 1
            return orig({**backbone_out, **hit[1]}, img_ids)
        result = orig(backbone_out, img_ids)
        computed = {k: v for k, v in result[0].items() if k not in backbone_out}
        self._entries[key] = (frames, computed)       # the frames are held: the id stays theirs
        self.computed += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return result


@contextlib.contextmanager
def frame_outputs_not_kept(model) -> Iterator[None]:
    """The model's per-frame cache of every yielded object mask (``_cache_frame_outputs``) is not
    written while the block runs — the batch flow never reads it (module docstring, 2)."""
    had = "_cache_frame_outputs" in vars(model)
    prev = vars(model).get("_cache_frame_outputs")
    model._cache_frame_outputs = _no_cache
    try:
        yield
    finally:
        if had:
            model._cache_frame_outputs = prev
        else:
            del model._cache_frame_outputs


def _no_cache(*_args, **_kwargs) -> None:
    return None


def clone_session(predictor, base_session_id: str) -> str:
    """A new session over the SAME decoded frames as ``base_session_id``: the model's own
    ``init_state`` builds every field of a fresh state, with the frame loader handing it the base
    session's frames (no second decode, no second copy on the card — and the frame object
    :class:`SharedVision` keys on is the same for every member). Closed like any session."""
    import importlib
    store = getattr(predictor, "_all_inference_states", None)
    if store is None:
        store = getattr(predictor, "_ALL_INFERENCE_STATES")
    base = store[base_session_id]["state"]
    frames = base["input_batch"].img_batch.tensors
    height, width = base["orig_height"], base["orig_width"]
    # the loader is looked up by name in the module of the init_state that calls it
    owners = {importlib.import_module(c.__module__) for c in type(predictor.model).__mro__}
    owners = [m for m in owners if hasattr(m, "load_resource_as_video_frames")]
    if not owners:
        raise RuntimeError(f"clone_session: no frame loader in the modules of "
                           f"{type(predictor.model).__name__}")
    saved = [(m, m.load_resource_as_video_frames) for m in owners]

    def _same_frames(*_args, **_kwargs):
        return frames, height, width

    try:
        for m, _f in saved:
            m.load_resource_as_video_frames = _same_frames
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=f"<frames of {base_session_id}>"))
    finally:
        for m, f in saved:
            m.load_resource_as_video_frames = f
    return response["session_id"]
