"""I2 — VLM content tags (``vlm_proposed``) + SAM3 exclusion masks
(claude_stac.txt §4-F1).

The VLM looks at every KEYFRAME and answers four yes/no questions about its
CONTENT — ``dynamic`` (a person, a moving machine, a train), ``occluder``
(a hand, a helmet brim, something stuck to the lens), ``reflective`` (glass,
mirror, water, screen) and ``low_info`` (a featureless surface filling the
frame). Every answer is a PROPOSAL: it is stamped ``vlm_proposed``, the VLM
fixes no parameter and measures nothing.

The two EXCLUSION classes (``intake.content.exclusion_classes``) then go to
SAM3 with a minimal vocabulary (``intake.content.prompts``, strings from the
config, not thresholds): for every frame in scope — every keyframe AND witness
frame under ``sam3_scope: all`` (claude_stac.txt §4-F1, the default: the tags
stay evidence and SAM3 decides the masks), or only the flagged keyframes plus
the witnesses between a flagged keyframe and its neighbours under
``flagged_ranges`` — the union of every object SAM3 finds for every prompt of
every exclusion class becomes
``<session>/intake/exclusion_masks/<frame:06d>.png`` (uint8, 255 = excluded)
on the NATIVE pixel grid. Those masks EXCLUDE observations downstream (F4 /
F6 / F7); the WEIGHT classes (``weight_classes``) only reach the report and
the provenance — they never exclude anything.

Artifact: ``<session>/intake/content_tags.json`` (atomic; ``version``,
``provenance`` "vlm_proposed", ``geometry_epoch`` / ``camera_epoch``, the
tags per keyframe, the parse failures, the mask inventory, the weight lists,
the effective parameters and the inputs' identity). With
``intake.content.enabled: false`` the JSON is still written (``enabled``
false, empty maps, the reason) and neither the VLM nor SAM3 is touched.

A malformed VLM answer FAILS the stage naming the batch (docs/plan_determinismo.md
point 75, 2026-10-07): it used to tag every frame of the batch all-False — a
truncated or garbled answer silently decided which frames SAM3 segmented
(pccr 2026-10-05: "24 from unparsed answers" on both runs). Nothing is tagged
'nothing' for the whole batch any more; the
report counts it. Structural impossibilities (a keyframe file that does not
exist, a segmenter returning a mask off the native grid, a tagger breaking
the protocol) raise :class:`ContentError` with the exact reason.

GPU exclusivity: the VLM (vLLM) and SAM3 never share the card. The tags are
all written before SAM3 is touched, and ``run_content`` calls the caller's
``before_sam3`` hook right before the first SAM3 call — the map_worker stops
the semantic service there (``workers.base.stop_semantic_service``); a later
VLM consumer restarts it. The hook is injected, so this module never imports
the worker.

Heavy dependencies (the semantic client, SAM3, the segmentation package) are
imported lazily INSIDE the production classes, so tests inject mocks through
the two protocols and never load a model. No decision literal lives here
(``tests/test_intake_config.py`` scans this package).

CLI: ``python -m intake.content --session <dir>`` (keyframes from
``frames/selected_frames.json``, witnesses from ``frames/witness_frames.json``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol,
                    Sequence, Tuple)

import numpy as np

from intake.config import CONTENT_CLASSES, SAM3_SCOPES, ContentConfig, load_intake_config
from intake.quality import Cancelled, check_cancelled, intake_epochs

INTAKE_DIRNAME = "intake"
FRAMES_DIRNAME = "frames"
OUTPUT_DIRNAME = "output"
CONTENT_TAGS_NAME = "content_tags.json"
EXCLUSION_MASKS_DIRNAME = "exclusion_masks"
SELECTED_FRAMES_NAME = "selected_frames.json"       # I1 keyframes (v2 contract)
WITNESS_FRAMES_NAME = "witness_frames.json"         # I1 witnesses
CONTENT_VERSION = 2                                 # 2: stamped mask inventory, relative dir,
                                                    #    no epochs / absolute paths, a VLM parse
                                                    #    failure fails the stage (2026-10-07)
PROVENANCE = "vlm_proposed"
METHOD = "intake_content"
CONSUMER = "intake.content"                         # semantic call-log attribution
EXCLUDED = 255                                      # PNG value of an excluded pixel
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")
TAG_KEYS = tuple(CONTENT_CLASSES) + ("notes",)
SCOPE_FLAGGED_RANGES, SCOPE_ALL = SAM3_SCOPES
LOG_TAG = "[intake.content]"
SAM3_LOGGER_NAME = "SAM3Wrapper"                    # segmentation/sam3_wrapper.py's logger

# What the VLM is asked. These are the DEFINITIONS of the four classes (fixed
# vocabulary of intake.config.CONTENT_CLASSES), not thresholds; the SAM3
# prompts that turn the two exclusion classes into masks come from config.
SYSTEM_PROMPT = (
    "You classify the CONTENT of video frames for a photogrammetry pipeline. You only "
    "say what is visible; you never measure anything. Return STRICT JSON only - no "
    "prose, no markdown fences."
)
CLASS_DEFINITIONS: Dict[str, str] = {
    "dynamic": ("a moving element is visible - a person, a worker, a moving vehicle "
                "or machine, an animal, a train"),
    "occluder": ("something is stuck to or right in front of the lens - a hand, a "
                 "glove, a helmet brim, a strap, dirt, an object covering part of "
                 "the image"),
    "reflective": ("a reflective or transparent surface occupies a large region - "
                   "glass, mirror, water, a screen, polished metal"),
    "low_info": ("most of the frame is a featureless surface - a bare wall, a "
                 "uniform floor, sky, fog or darkness with no texture"),
}


class ContentError(RuntimeError):
    """A structural impossibility of I2 (a keyframe with no file, a mask off
    the native grid, a tagger or segmenter breaking its protocol, a missing
    I1 artifact) — always with the exact reason."""


# ── protocols ────────────────────────────────────────────────────────────

class VLMTagger(Protocol):
    """Tags a batch of keyframe images.

    ``tag`` receives ≤ ``ContentConfig.batch`` RGB uint8 (H, W, 3) images and
    their video frame numbers and returns ONE dict per image, in order, with
    keys exactly ``dynamic``, ``occluder``, ``reflective``, ``low_info``
    (bool) and ``notes`` (str). A production tagger records how many of its
    answers failed to parse in ``parse_failures`` (calls) and ``n_calls``."""

    def tag(self, images: List[np.ndarray], frames: List[int]) -> List[Dict[str, Any]]:
        ...


class Segmenter(Protocol):
    """Segments one open-vocabulary prompt over a list of frames.

    ``masks`` returns ``{frame: bool (H, W) native}`` — the union of every
    object found for ``prompt`` in that frame — for the frames that got a
    mask. CONTRACT: every requested frame WAS segmented; a frame absent from
    the result is a frame where nothing was found (recorded as
    ``segmented_no_object``). A segmenter that could not segment a frame must
    raise naming it — an absence is never allowed to mean a failure."""

    def masks(self, frames_dir: os.PathLike, frame_ids: List[int],
              prompt: str) -> Dict[int, np.ndarray]:
        ...


# ── small helpers ────────────────────────────────────────────────────────

def _cv2():
    import cv2
    return cv2


def _heartbeat(heartbeat_s: float) -> float:
    """``intake.runtime.heartbeat_s`` as passed by the caller, validated."""
    if heartbeat_s <= 0:
        raise ContentError(f"heartbeat_s must be positive, got {heartbeat_s}")
    return float(heartbeat_s)


def _write_json_atomic(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


def _sorted_unique_ints(values: Iterable[Any], what: str) -> List[int]:
    out = set()
    for v in values:
        if isinstance(v, bool) or int(v) != v:
            raise ContentError(f"{what} must be video frame numbers (ints), got {v!r}")
        out.add(int(v))
    return sorted(out)


def frame_file(frames_dir: os.PathLike, frame: int) -> Path:
    """``frames/<frame:06d>.<ext>`` for the first existing suffix of
    FRAME_SUFFIXES; ContentError when the frame has no file."""
    frames_dir = Path(frames_dir)
    for suffix in FRAME_SUFFIXES:
        p = frames_dir / f"{int(frame):06d}{suffix}"
        if p.exists():
            return p
    raise ContentError(
        f"frame {int(frame)} has no file {int(frame):06d}{{{','.join(FRAME_SUFFIXES)}}} in "
        f"{frames_dir} — the keyframe / witness lists name a frame that is not on disk")


def read_rgb(path: os.PathLike) -> np.ndarray:
    """The frame as RGB uint8 (H, W, 3) at its native resolution."""
    cv2 = _cv2()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ContentError(f"frame {path} is unreadable (cv2.imread returned None)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def native_size(frames_dir: os.PathLike, frame: int) -> Tuple[int, int]:
    """(width, height) of ``frame`` on disk — the native grid the masks must match."""
    img = read_rgb(frame_file(frames_dir, frame))
    return int(img.shape[1]), int(img.shape[0])


def empty_tags(notes: str = "") -> Dict[str, Any]:
    """All-False tags with the given notes."""
    out: Dict[str, Any] = {cls: False for cls in CONTENT_CLASSES}
    out["notes"] = str(notes)
    return out


class VLMParseError(ContentError):
    """The VLM's answer for a batch did not follow the contract (point 75): the stage FAILS
    naming the frames and the head of the answer — a batch is never tagged 'nothing'."""


def _as_bool(v: Any) -> bool:
    """A tolerant boolean read of what a VLM writes (true/false, yes/no, 0/1);
    anything else is a ValueError → the batch is a parse failure."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    raise ValueError(f"not a boolean: {v!r}")


def normalize_tags(item: Mapping[str, Any]) -> Dict[str, Any]:
    """One VLM object → the tag dict of the protocol. ValueError when a class
    key is missing or not a boolean (the answer did not follow the contract)."""
    out: Dict[str, Any] = {}
    for cls in CONTENT_CLASSES:
        if cls not in item:
            raise ValueError(f"missing key {cls!r}")
        out[cls] = _as_bool(item[cls])
    notes = item.get("notes", "")
    out["notes"] = "" if notes is None else str(notes)
    return out


def validate_tags(tags: Any, frame: int) -> Dict[str, Any]:
    """The protocol contract of a tagger's answer for ``frame``: a mapping with
    every class a bool and ``notes`` a str. ContentError otherwise."""
    if not isinstance(tags, Mapping):
        raise ContentError(
            f"tagger returned {type(tags).__name__} for frame {frame}; the VLMTagger "
            f"protocol returns one dict per image with keys {TAG_KEYS}")
    out: Dict[str, Any] = {}
    for cls in CONTENT_CLASSES:
        if cls not in tags or not isinstance(tags[cls], (bool, np.bool_)):
            raise ContentError(
                f"tagger's answer for frame {frame} lacks a boolean {cls!r}: {dict(tags)!r}")
        out[cls] = bool(tags[cls])
    notes = tags.get("notes", None)
    if not isinstance(notes, str):
        raise ContentError(
            f"tagger's answer for frame {frame} lacks a string 'notes': {dict(tags)!r}")
    out["notes"] = notes
    return out


# ── the VLM prompt and its parsing ───────────────────────────────────────

def build_tag_prompt(n_images: int, frames: Sequence[int]) -> str:
    """The user prompt for one batch: one JSON object per image, in order."""
    lines = [
        f"You are given {n_images} image(s), video frames {list(int(f) for f in frames)}, "
        f"in that order.",
        "For EACH image answer the four yes/no questions below.",
        "Questions:",
    ]
    lines += [f"  - {cls}: {CLASS_DEFINITIONS[cls]}?" for cls in CONTENT_CLASSES]
    lines += [
        "Return a JSON array with EXACTLY one object per image, in the same order as the "
        "images, with keys exactly: \"image\" (the 0-based index of the image), "
        + ", ".join(f"\"{cls}\"" for cls in CONTENT_CLASSES)
        + " (booleans) and \"notes\" (a short string, may be empty).",
        "Return ONLY the JSON array.",
    ]
    return "\n".join(lines)


def parse_tag_answer(text: str, n_images: int) -> Optional[List[Dict[str, Any]]]:
    """The VLM's answer → ``n_images`` tag dicts in image order, or ``None``
    when the answer is malformed: no JSON array, not exactly one object per
    image, an ``image`` index that is missing/duplicated/out of range, or a
    class value that is not a boolean. Objects without ``image`` are taken in
    the order they appear. Parsing is the auto-prompter's own tolerant reader
    (``segmentation.autoprompt.detector._extract_json_array``: fences, wrapped
    arrays, salvage of truncated answers); imported lazily — the segmentation
    package is heavy."""
    from segmentation.autoprompt.detector import _extract_json_array
    arr = _extract_json_array(text or "")
    if arr is None:
        return None
    items = [x for x in arr if isinstance(x, dict)]
    if len(items) != n_images:
        return None
    by_index: Dict[int, Dict[str, Any]] = {}
    for pos, item in enumerate(items):
        idx = item.get("image", pos)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= n_images or idx in by_index:
            return None
        try:
            by_index[idx] = normalize_tags(item)
        except ValueError:
            return None
    return [by_index[i] for i in range(n_images)]


# ── production implementations ───────────────────────────────────────────

class QwenTagger:
    """The production VLMTagger: ``semantic.client.get_semantic_client`` on
    ``cfg.backend`` (a local vLLM Qwen3-VL — no paid API), one chat call per
    batch of ≤ ``cfg.batch`` images attached to ONE user message, generation
    capped at ``cfg.max_tokens``. A malformed answer FAILS the call
    (:class:`VLMParseError`, point 75) — never a batch tagged all-False;
    ``parse_failures`` counts the failures seen before the raise (for the
    report of a caller that catches it). ``client`` may be injected (tests);
    otherwise it is created lazily on the first call."""

    def __init__(self, cfg: ContentConfig, client: Any = None):
        self.cfg = cfg
        self._client = client
        self.n_calls = 0
        self.parse_failures = 0          # VLM answers that did not parse (calls, not frames)

    def _get_client(self):
        if self._client is None:
            from semantic.client import get_semantic_client     # lazy: needs the service config
            self._client = get_semantic_client(backend=self.cfg.backend, consumer=CONSUMER)
        return self._client

    def tag(self, images: List[np.ndarray], frames: List[int]) -> List[Dict[str, Any]]:
        if len(images) != len(frames):
            raise ContentError(
                f"QwenTagger.tag: {len(images)} image(s) but {len(frames)} frame number(s)")
        if not images:
            return []
        if len(images) > self.cfg.batch:
            raise ContentError(
                f"QwenTagger.tag: {len(images)} images exceed intake.content.batch = "
                f"{self.cfg.batch} — the caller batches")
        from semantic.types import system, user                 # lazy: consumer-agnostic
        prompt = build_tag_prompt(len(images), list(frames))
        resp = self._get_client().chat(
            [system(SYSTEM_PROMPT), user(prompt, images=list(images))],
            max_tokens=self.cfg.max_tokens, consumer=CONSUMER)
        self.n_calls += 1
        text = resp.content or ""
        parsed = parse_tag_answer(text, len(images))
        if parsed is None:
            self.parse_failures += 1
            raise VLMParseError(
                f"the VLM's answer for frames {list(int(f) for f in frames)} did not follow the "
                f"contract (one JSON object per image with boolean {list(CONTENT_CLASSES)}; "
                f"max_tokens {self.cfg.max_tokens}) — intake I2 fails instead of tagging the "
                f"batch 'nothing' (plan point 75); answer head: {text[:200]!r}")
        return parsed


def union_masks(masks: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    """OR of boolean masks (the per-frame union of ``reconstruction/
    dynamic_masks.py``, fail-hard on a shape mismatch instead of skipping);
    ``None`` for an empty iterable."""
    out: Optional[np.ndarray] = None
    for m in masks:
        mb = np.asarray(m).astype(bool)
        if out is None:
            out = mb.copy()
        elif mb.shape != out.shape:
            raise ContentError(
                f"masks of one frame disagree in shape: {out.shape} vs {mb.shape}")
        else:
            out |= mb
    return out


class _WrapperLog(logging.Handler):
    """Collects the records the SAM3 wrapper logs while one call runs: the
    wrapper catches its own exceptions (``process_batch`` logs and returns
    what it has; a prompt it could not add is a warning), so its log is the
    only evidence of a failure it swallowed."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


# what, among the wrapper's own WARNING+ records, says the call did not do its work
# (every ERROR, and the swallowed prompt-add failure); other warnings are the
# wrapper's non-fatal cleanup notes and are only reported
SAM3_FAILURE_PREFIXES = ("Could not add prompt",)


def sam3_failures(records: Sequence[logging.LogRecord]) -> List[str]:
    """The messages of ``records`` that report a swallowed SAM3 failure."""
    out = []
    for r in records:
        msg = r.getMessage()
        if r.levelno >= logging.ERROR or msg.startswith(SAM3_FAILURE_PREFIXES):
            out.append(f"{r.levelname}: {msg}")
    return out


class Sam3Segmenter:
    """The production Segmenter: ``segmentation.sam3_wrapper.get_sam3_wrapper()``
    driven through the segmentation pipeline's own batch machinery
    (``_prepare_batch_dir`` → ``process_batch`` → ``_parse_raw_masks``), the
    objects of every frame OR-ed. Frames are fed in chunks of ``batch_size``
    (``ContentConfig.sam3_batch`` = ``models.segmentation.batch_size``, the
    segmentation stage's own SAM3 session length, validated at config load —
    no overlap: an exclusion mask carries no identity to stitch).
    ``_prepare_batch_dir`` maps local index → int(stem) = the video frame
    number, so the masks come back keyed by our frame ids at the frames'
    ORIGINAL (native) resolution — the caller asserts the shape.

    FAIL-HARD on a partial result (the Segmenter contract): the wrapper
    swallows its exceptions and returns whatever it propagated, so every
    chunk is checked — a result entry for EVERY frame of the chunk (a
    successful propagation yields one per frame), no frame that was not
    asked for, and no failure in the wrapper's own log during the call
    (:func:`sam3_failures`) — and a ContentError names the prompt, the
    session and the frames otherwise. What passes is complete: an absent
    frame then means "segmented, nothing found". One vendor call can take
    minutes: a line is logged before it (chunk, size, prompt) and after it
    (elapsed, frames/s). ``close()`` releases the batch session, unloads the
    model and drops the symlink dirs."""

    def __init__(self, batch_size: int, log: Callable = print):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ContentError(f"Sam3Segmenter batch_size must be a positive integer "
                               f"(intake.content -> models.segmentation.batch_size), got "
                               f"{batch_size!r}")
        self.log = log
        self.batch_size = int(batch_size)

    def masks(self, frames_dir: os.PathLike, frame_ids: List[int],
              prompt: str) -> Dict[int, np.ndarray]:
        frames_dir = Path(frames_dir)
        ids = _sorted_unique_ints(frame_ids, "frame_ids")
        if not ids:
            return {}
        files = [frame_file(frames_dir, f).name for f in ids]
        from segmentation.sam3_wrapper import get_sam3_wrapper       # lazy: SAM3 + torch
        from segmentation.pipeline import _parse_raw_masks, _prepare_batch_dir
        sam3 = get_sam3_wrapper()
        bs = self.batch_size
        n_chunks = (len(files) + bs - 1) // bs
        out: Dict[int, np.ndarray] = {}
        wrapper_logger = logging.getLogger(SAM3_LOGGER_NAME)
        for s in range(0, len(files), bs):
            chunk = files[s:s + bs]
            k = s // bs + 1
            self.log(f"{LOG_TAG} SAM3 '{prompt}': session {k}/{n_chunks} starting — "
                     f"{len(chunk)} frame(s) ({chunk[0]}..{chunk[-1]})")
            t0 = time.monotonic()
            batch_dir, index_mapping = _prepare_batch_dir(frames_dir, chunk, s)
            expected = sorted(int(v) for v in index_mapping.values())
            if expected != sorted(int(Path(c).stem) for c in chunk):
                raise ContentError(
                    f"SAM3 '{prompt}' session {k}/{n_chunks}: the batch index mapping names "
                    f"frames {expected[:5]}… for the chunk {chunk[0]}..{chunk[-1]} — the "
                    f"local-to-frame mapping broke")
            handler = _WrapperLog()
            wrapper_logger.addHandler(handler)
            try:
                raw = sam3.process_batch(str(batch_dir), prompt, index_mapping,
                                         boxes_by_local=None)
            finally:
                wrapper_logger.removeHandler(handler)
            failures = sam3_failures(handler.records)
            if failures:
                raise ContentError(
                    f"SAM3 '{prompt}' session {k}/{n_chunks} (frames {chunk[0]}..{chunk[-1]}) "
                    f"reported a failure it swallowed — {failures[:3]} — the exclusion masks "
                    f"of these frames cannot be told from 'nothing found'; fix SAM3 and re-run "
                    f"intake I2")
            got = sorted(int(f) for f in (raw or {}))
            missing = sorted(set(expected) - set(got))
            extra = sorted(set(got) - set(expected))
            if missing or extra:
                raise ContentError(
                    f"SAM3 '{prompt}' session {k}/{n_chunks} returned a partial result: "
                    f"{len(got)}/{len(expected)} frame(s) — missing {missing[:10]}"
                    f"{'…' if len(missing) > 10 else ''}"
                    + (f", not requested {extra[:10]}" if extra else "")
                    + " — a frame without a result cannot be told from 'nothing found'; "
                      "re-run intake I2")
            parsed = _parse_raw_masks(raw or {})
            for frame, objs in parsed.items():
                union = union_masks(objs.values())
                if union is not None and union.any():
                    out[int(frame)] = union
            elapsed = max(time.monotonic() - t0, 1e-9)
            self.log(f"{LOG_TAG} SAM3 '{prompt}': session {k}/{n_chunks} done in "
                     f"{elapsed:.1f} s ({len(chunk) / elapsed:.2f} frames/s), "
                     f"{len(got)}/{len(chunk)} frame(s) segmented, {len(parsed)} with a mask"
                     + (f"; wrapper notes: {[r.getMessage() for r in handler.records]}"
                        if handler.records else ""))
        return out

    def close(self) -> None:
        from segmentation.sam3_wrapper import get_sam3_wrapper
        from segmentation.pipeline import _clear_batch_dirs
        sam3 = get_sam3_wrapper()
        sam3.release_batch_session()
        sam3.unload_model()
        _clear_batch_dirs()


# ── stage functions ──────────────────────────────────────────────────────

def tag_keyframes(frames_dir: os.PathLike, keyframes: Sequence[int], tagger: VLMTagger,
                  cfg: ContentConfig, log: Callable = print, *, heartbeat_s: float,
                  cancelled: Cancelled = None) -> Dict[int, Dict[str, Any]]:
    """Tag every keyframe in batches of ``cfg.batch`` → ``{frame: tags}``.

    Frames are read from disk as RGB uint8 at native resolution; the tagger's
    answers are validated against the protocol (ContentError on a breach —
    a production tagger never breaches it: a failed parse is a well-formed
    all-False answer). Progress lines carry the rate; ``cancelled()`` is
    polled before every batch."""
    heartbeat_s = _heartbeat(heartbeat_s)
    frames_dir = Path(frames_dir)
    kfs = _sorted_unique_ints(keyframes, "keyframes")
    n = len(kfs)
    if n == 0:
        raise ContentError("tag_keyframes: no keyframes to tag")
    log(f"{LOG_TAG} tagging {n} keyframe(s) in batches of {cfg.batch} "
        f"(classes {list(CONTENT_CLASSES)}; every tag is vlm_proposed)")
    tags: Dict[int, Dict[str, Any]] = {}
    t0 = time.monotonic()
    last_beat = t0
    for s in range(0, n, cfg.batch):
        check_cancelled(cancelled, f"intake I2 (VLM tags) at keyframe {s}/{n}")
        batch = kfs[s:s + cfg.batch]
        images = [read_rgb(frame_file(frames_dir, f)) for f in batch]
        answers = tagger.tag(images, list(batch))
        if not isinstance(answers, (list, tuple)) or len(answers) != len(batch):
            got = len(answers) if isinstance(answers, (list, tuple)) else type(answers).__name__
            raise ContentError(
                f"tagger returned {got} answer(s) for a batch of {len(batch)} frame(s) "
                f"{batch} — the VLMTagger protocol returns one dict per image")
        for f, t in zip(batch, answers):
            tags[f] = validate_tags(t, f)
        done = s + len(batch)
        now = time.monotonic()
        if now - last_beat >= heartbeat_s or done == n:
            elapsed = max(now - t0, 1e-9)
            rate = done / elapsed
            eta = (n - done) / max(rate, 1e-9)
            log(f"{LOG_TAG} {done}/{n} keyframes tagged ({rate:.2f} frames/s, eta {eta:.0f} s)")
            last_beat = now
    counts = {cls: sum(1 for t in tags.values() if t[cls]) for cls in CONTENT_CLASSES}
    log(f"{LOG_TAG} tagged: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return tags


def flagged_ranges(keyframes: Sequence[int], tags: Mapping[int, Mapping[str, Any]],
                   witnesses: Sequence[int], cls: str) -> List[int]:
    """Frames to segment for ``cls`` under ``sam3_scope: flagged_ranges``.

    Every keyframe tagged ``cls``, plus every witness frame lying between a
    tagged keyframe and its neighbouring keyframes (inclusive): the VLM only
    saw the keyframes, so what it flagged at keyframe k is assumed present in
    the witnesses up to the previous and the next keyframe. A tagged FIRST
    keyframe claims the witnesses before it and a tagged LAST keyframe the
    witnesses after it: no other keyframe was seen there to say otherwise,
    and I1 leaves the tail after the last keyframe inside that keyframe's
    open window — the safe direction (more exclusion, never less). Sorted,
    unique; ``[]`` when nothing is tagged."""
    if cls not in CONTENT_CLASSES:
        raise ContentError(f"unknown content class {cls!r}; allowed {CONTENT_CLASSES}")
    kfs = _sorted_unique_ints(keyframes, "keyframes")
    wits = _sorted_unique_ints(witnesses, "witnesses")
    missing = [k for k in kfs if k not in tags]
    if missing:
        raise ContentError(f"keyframe(s) {missing} have no tags — tag_keyframes did not "
                           f"cover the keyframe list")
    out = set()
    for i, k in enumerate(kfs):
        if not bool(tags[k][cls]):
            continue
        out.add(k)
        lo = kfs[i - 1] if i > 0 else None
        hi = kfs[i + 1] if i + 1 < len(kfs) else None
        for w in wits:
            if (lo is None or w >= lo) and (hi is None or w <= hi):
                out.add(w)
    return sorted(out)


def scope_frames(keyframes: Sequence[int], tags: Mapping[int, Mapping[str, Any]],
                 witnesses: Sequence[int], cls: str, scope: str) -> List[int]:
    """The frames SAM3 segments for ``cls``: :func:`flagged_ranges` under
    ``flagged_ranges``; every keyframe and witness under ``all``."""
    if scope == SCOPE_FLAGGED_RANGES:
        return flagged_ranges(keyframes, tags, witnesses, cls)
    if scope == SCOPE_ALL:
        return _sorted_unique_ints(list(keyframes) + list(witnesses), "frames")
    raise ContentError(f"unknown sam3_scope {scope!r}; allowed {SAM3_SCOPES}")


def write_mask_png(path: os.PathLike, mask: np.ndarray) -> Path:
    """``mask`` (bool H×W) → uint8 PNG, EXCLUDED (255) where True, 0 elsewhere;
    encoded in memory and written tmp + os.replace."""
    cv2 = _cv2()
    path = Path(path)
    arr = np.where(np.asarray(mask).astype(bool), EXCLUDED, 0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ContentError(f"cv2 could not encode the exclusion mask for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)
    return path


def read_mask_png(path: os.PathLike) -> np.ndarray:
    """An exclusion PNG back as bool (H, W): True where the pixel is EXCLUDED."""
    cv2 = _cv2()
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ContentError(f"exclusion mask {path} is unreadable")
    return arr == EXCLUDED


def mask_path(masks_dir: os.PathLike, frame: int) -> Path:
    return Path(masks_dir) / f"{int(frame):06d}.png"


def clear_masks_dir(masks_dir: os.PathLike, log: Callable = print) -> int:
    """Remove the PNGs a previous run left in ``masks_dir`` (a re-run replaces
    its output: a frame no longer in scope must not keep a stale mask).
    Returns the number removed — logged with the reason."""
    masks_dir = Path(masks_dir)
    if not masks_dir.is_dir():
        return 0
    stale = sorted(p for p in masks_dir.iterdir()
                   if p.is_file() and (p.suffix.lower() == ".png" or p.name.endswith(".tmp")))
    for p in stale:
        p.unlink()
    if stale:
        log(f"{LOG_TAG} removed {len(stale)} file(s) of a previous run from {masks_dir} "
            f"(reason: the exclusion masks are rebuilt for the current keyframes/tags)")
    return len(stale)


def build_exclusion_masks(frames_dir: os.PathLike, out_dir: os.PathLike,
                          frame_ids_by_class: Mapping[str, Sequence[int]],
                          prompts: Mapping[str, Sequence[str]], segmenter: Segmenter,
                          native_wh: Tuple[int, int], log: Callable = print, *,
                          heartbeat_s: float, cancelled: Cancelled = None) -> Dict[int, int]:
    """Segment every (class, prompt) over that class's frames and write the OR
    over classes and prompts as ``<out_dir>/<frame:06d>.png`` (255 = excluded)
    for every frame with at least one excluded pixel.

    Every mask the segmenter returns must be (native_h, native_w) — a mask on
    another grid is a broken contract (ContentError naming frame, prompt and
    shapes). Returns ``{frame: n_excluded_px}`` for the frames written."""
    heartbeat_s = _heartbeat(heartbeat_s)
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    w, h = int(native_wh[0]), int(native_wh[1])
    jobs: List[Tuple[str, str, List[int]]] = []
    for cls in sorted(frame_ids_by_class):
        if cls not in CONTENT_CLASSES:
            raise ContentError(f"unknown content class {cls!r}; allowed {CONTENT_CLASSES}")
        if cls not in prompts or not prompts[cls]:
            raise ContentError(
                f"no SAM3 prompt for exclusion class {cls!r} — intake.content.prompts.{cls} "
                f"must list at least one")
        ids = _sorted_unique_ints(frame_ids_by_class[cls], f"frame_ids_by_class[{cls!r}]")
        for prompt in prompts[cls]:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ContentError(f"intake.content.prompts.{cls} holds an empty prompt")
            jobs.append((cls, prompt, ids))
    total = sum(len(ids) for _, _, ids in jobs)
    log(f"{LOG_TAG} exclusion masks: {len(jobs)} (class, prompt) job(s) over "
        f"{total} frame-prompt pair(s), native {w}x{h}, dir {out_dir}")

    union: Dict[int, np.ndarray] = {}
    done = 0
    t0 = time.monotonic()
    last_beat = t0
    for k, (cls, prompt, ids) in enumerate(jobs):
        if not ids:
            log(f"{LOG_TAG} {cls}/'{prompt}': no frame in scope — skipped")
            continue
        check_cancelled(cancelled, f"intake I2 (SAM3 masks) at job {k}/{len(jobs)}")
        wanted = set(ids)
        masks = segmenter.masks(frames_dir, list(ids), prompt)
        if not isinstance(masks, Mapping):
            raise ContentError(
                f"segmenter returned {type(masks).__name__} for '{prompt}'; the Segmenter "
                f"protocol returns {{frame: bool mask}}")
        n_px_prompt = 0
        for f, m in masks.items():
            f = int(f)
            if f not in wanted:
                raise ContentError(
                    f"segmenter returned frame {f} for '{prompt}', which was not requested")
            m = np.asarray(m)
            if m.shape != (h, w):
                raise ContentError(
                    f"mask of frame {f} for '{prompt}' ({cls}) is {m.shape}, the native grid "
                    f"is {(h, w)} — masks must come back at the frames' own resolution")
            mb = m.astype(bool)
            n_px_prompt += int(mb.sum())
            union[f] = mb.copy() if f not in union else (union[f] | mb)
        log(f"{LOG_TAG} {cls}/'{prompt}': {len(masks)}/{len(ids)} frame(s) masked, "
            f"{n_px_prompt} px")
        done += len(ids)
        now = time.monotonic()
        if now - last_beat >= heartbeat_s or done == total:
            elapsed = max(now - t0, 1e-9)
            rate = done / elapsed
            eta = (total - done) / max(rate, 1e-9)
            log(f"{LOG_TAG} {done}/{total} frame-prompt pairs ({rate:.2f}/s, eta {eta:.0f} s)")
            last_beat = now

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[int, int] = {}
    for f in sorted(union):
        n_px = int(union[f].sum())
        if n_px == 0:
            continue
        write_mask_png(mask_path(out_dir, f), union[f])
        counts[f] = n_px
    log(f"{LOG_TAG} wrote {len(counts)} exclusion mask(s) ({EXCLUDED} = excluded) to {out_dir}")
    return counts


# ── the I2 report ────────────────────────────────────────────────────────

def _params(cfg: ContentConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["exclusion_classes"] = list(cfg.exclusion_classes)
    d["weight_classes"] = list(cfg.weight_classes)
    d["prompts"] = {k: list(v) for k, v in cfg.prompts.items()}
    return d


def _inputs(frames_dir: Path, kfs: List[int], wits: List[int]) -> Dict[str, Any]:
    """The inputs' identity, paths relative to the session (point 70)."""
    return {
        "frames_dir": frames_dir.name,
        "n_keyframes": len(kfs),
        "keyframes_first": kfs[0] if kfs else None,
        "keyframes_last": kfs[-1] if kfs else None,
        "n_witnesses": len(wits),
        "witnesses_first": wits[0] if wits else None,
        "witnesses_last": wits[-1] if wits else None,
    }


def content_tags_path(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_DIRNAME / CONTENT_TAGS_NAME


def exclusion_masks_dir(session_dir: os.PathLike) -> Path:
    return Path(session_dir) / INTAKE_DIRNAME / EXCLUSION_MASKS_DIRNAME


EXCLUSION_MASKS_REL = f"{INTAKE_DIRNAME}/{EXCLUSION_MASKS_DIRNAME}"   # the product's 'dir'


def masks_stamp(session_dir: os.PathLike, frames_written: Sequence[int], kfs: Sequence[int],
                wits: Sequence[int], cfg: ContentConfig) -> Dict[str, Any]:
    """The stamp of the exclusion-mask inventory (plan points 68 / 84): the BYTES of every
    mask PNG listed (keyed ``intake/exclusion_masks/<frame>.png``), the intake code, the I2
    parameters and the keyframe / witness lists the masks were built for. Written at the
    report's ROOT (``stamp`` — the key ``precision.tracks.exclusion_mask_paths``, F4's and the
    depth sweep's reader, checks: every listed PNG's sha256 against the file on disk) and under
    ``exclusion_masks.stamp`` (what :func:`valid_exclusion_masks` re-computes in full). With I2
    off the inventory is empty and so is the stamp's input list — the stamp still says WHICH
    masks are valid: none."""
    from intake import stamps as St
    session_dir = Path(session_dir)
    masks_dir = exclusion_masks_dir(session_dir)
    inputs = {f"{EXCLUSION_MASKS_REL}/{int(f):06d}.png": mask_path(masks_dir, int(f))
              for f in sorted(int(x) for x in frames_written)}
    return St.step_stamp(inputs, {"params": _params(cfg),
                                  "keyframes": [int(k) for k in kfs],
                                  "witnesses": [int(w) for w in wits],
                                  "enabled": bool(cfg.enabled)})


def _content_config_of(params: Mapping[str, Any]) -> ContentConfig:
    """The ContentConfig a report's ``params`` describe (lists back to tuples)."""
    kw: Dict[str, Any] = {}
    for k, v in dict(params).items():
        if isinstance(v, list):
            kw[k] = tuple(v)
        elif isinstance(v, dict):
            kw[k] = {kk: tuple(vv) for kk, vv in v.items()}
        else:
            kw[k] = v
    return ContentConfig(**kw)


def valid_exclusion_masks(session_dir: os.PathLike, log: Callable = print
                          ) -> Tuple[Dict[int, Path], Dict[str, Any]]:
    """THE way a consumer reads I2's exclusion masks (plan points 68 / 84): ``({frame: PNG
    path}, record)`` of exactly the masks ``content_tags.json`` lists, and only when its stamp
    re-computed now — the listed PNGs' bytes, the intake code, the I2 parameters and the
    keyframe / witness lists on disk (I1's products) — matches the one the report carries.
    Otherwise NO mask is taken and the record says why (``reason``, ``differences``); a PNG on
    disk that the report does not list is never read. No report → no masks, declared."""
    from intake import stamps as St
    session_dir = Path(session_dir)
    p = content_tags_path(session_dir)
    rec: Dict[str, Any] = {"report": f"{INTAKE_DIRNAME}/{CONTENT_TAGS_NAME}", "taken": False,
                           "n_listed": 0, "n_taken": 0, "reason": None, "differences": []}

    def _none(reason: str, diffs: Optional[List[str]] = None) -> Tuple[Dict[int, Path], Dict[str, Any]]:
        rec["reason"], rec["differences"] = reason, list(diffs or [])
        log(f"{LOG_TAG} exclusion masks: none taken ({reason}"
            + ("" if not diffs else ": " + "; ".join(diffs[:5])) + ")")
        return {}, rec

    if not p.exists():
        return _none(f"{p.name} does not exist — intake I2 wrote no mask inventory")
    try:
        doc = load_content(session_dir)
        kfs, wits = read_i1_frames(session_dir)
    except ContentError as e:
        return _none(str(e))
    masks = doc.get("exclusion_masks") or {}
    listed = sorted(int(f) for f in (masks.get("frames") or {}))
    rec["n_listed"] = len(listed)
    if not isinstance(masks.get("stamp"), dict):
        return _none(f"{p.name} carries no mask-inventory stamp (version {doc.get('version')})")
    masks_dir = exclusion_masks_dir(session_dir)
    missing = [f for f in listed if not mask_path(masks_dir, f).exists()]
    if missing:
        return _none(f"{len(missing)} listed mask PNG(s) are missing from {EXCLUSION_MASKS_REL} "
                     f"(first: {missing[0]:06d}.png)")
    try:
        cfg = _content_config_of(doc.get("params") or {})
    except TypeError as e:
        return _none(f"{p.name} params do not read as an I2 configuration ({e})")
    diffs = St.stamp_differences(masks["stamp"], masks_stamp(session_dir, listed, kfs, wits, cfg))
    if diffs:
        return _none("the mask inventory's stamp does not match the session now", diffs)
    out = {f: mask_path(masks_dir, f) for f in listed}
    rec.update({"taken": True, "n_taken": len(out)})
    log(f"{LOG_TAG} exclusion masks: {len(out)} taken (stamp matches)" if out else
        f"{LOG_TAG} exclusion masks: the inventory is empty (I2 "
        f"{'wrote none' if doc.get('enabled') else 'is off'}) — none to take")
    return out, rec


def write_content(session_dir: os.PathLike, report: Dict[str, Any]) -> Path:
    """``<session>/intake/content_tags.json`` (tmp + os.replace)."""
    return _write_json_atomic(content_tags_path(session_dir), report)


def load_content(session_dir: os.PathLike) -> Dict[str, Any]:
    """The I2 report; ContentError when absent or of another version."""
    p = content_tags_path(session_dir)
    if not p.exists():
        raise ContentError(f"{p} does not exist — run intake I2 "
                           f"(python -m intake.content --session <dir>) first")
    with open(p) as f:
        report = json.load(f)
    if report.get("version") != CONTENT_VERSION:
        raise ContentError(f"{p} is version {report.get('version')!r}, this module reads "
                           f"version {CONTENT_VERSION} — re-run intake I2")
    return report


def run_content(session_dir: os.PathLike, keyframes: Sequence[int], witnesses: Sequence[int],
                cfg: ContentConfig, tagger: Optional[VLMTagger] = None,
                segmenter: Optional[Segmenter] = None, log: Callable = print, *,
                heartbeat_s: float, before_sam3: Optional[Callable[[], Any]] = None,
                cancelled: Cancelled = None) -> Dict[str, Any]:
    """I2 end to end: tag the keyframes, choose the frames in scope per
    exclusion class, hand the GPU over (``before_sam3``), build the exclusion
    masks, write ``content_tags.json``.

    ``tagger`` / ``segmenter`` None → the production :class:`QwenTagger` /
    :class:`Sam3Segmenter` (batch ``cfg.sam3_batch``), constructed only when
    ``cfg.enabled`` (a segmenter this function constructs is also closed by
    it; an injected one belongs to the caller). ``cfg.enabled`` False → the
    JSON is written with ``enabled`` false, empty maps and the reason;
    nothing is constructed or called. ``before_sam3()`` runs after every tag
    is in hand and right before the first SAM3 call — only when SAM3 has a
    frame to segment (the map_worker stops vLLM there: the two never share
    the card). ``cancelled()`` is polled by the tag and mask loops.

    Report keys: ``version`` 2, ``provenance`` "vlm_proposed", ``method``,
    ``enabled``, ``backend``, ``geometry_epoch`` / ``camera_epoch``
    (``intake.quality.INTAKE_EPOCH``), ``sam3_handover`` {called,
    verified, check (what ``before_sam3`` returned: the map_worker's and the
    CLIs' stopper verify with pgrep that no ``vllm serve`` is left and fail
    naming the PIDs otherwise), reason}, ``classes``
    {exclusion, weight}, ``native_w``/``native_h``, ``frames`` {str(frame):
    tags}, ``parse_failures`` (always 0: an answer that does not parse FAILS
    the stage, point 75), ``vlm_calls`` {n_calls, n_parse_failed} when the
    tagger exposes its counters (None otherwise), ``exclusion_masks`` {dir
    (session-relative), frames {str(frame): n_px}, prompts, scope, requested
    {cls: n_frames}, n_frames_written, frame_status
    {str(frame): masked | segmented_no_object} for every frame SAM3 was asked
    about, n_segmented_no_object, stamp (:func:`masks_stamp` — what a consumer
    checks before taking a mask)}, ``weights`` {cls: [frames]},
    ``summary`` {tagged: {cls: n}}, ``params``, ``inputs``."""
    session_dir = Path(session_dir)
    frames_dir = session_dir / FRAMES_DIRNAME
    masks_dir = exclusion_masks_dir(session_dir)
    kfs = _sorted_unique_ints(keyframes, "keyframes")
    wits = _sorted_unique_ints(witnesses, "witnesses")
    base: Dict[str, Any] = {
        "version": CONTENT_VERSION,
        "provenance": PROVENANCE,
        "method": METHOD,
        "enabled": bool(cfg.enabled),
        "backend": cfg.backend,
        **intake_epochs(),
        "classes": {"exclusion": list(cfg.exclusion_classes),
                    "weight": list(cfg.weight_classes)},
        "params": _params(cfg),
        "inputs": _inputs(frames_dir, kfs, wits),
    }

    if not cfg.enabled:
        reason = "intake.content.enabled is false — no VLM or SAM3 call was made"
        # the masks a previous run with I2 ON left behind go too (plan points 68 / 84: F4 read
        # every PNG on disk by existence, so a resumed session kept excluding what an older
        # run had segmented); the inventory below lists exactly the valid masks — none
        stale = clear_masks_dir(masks_dir, log=log)
        st_masks = masks_stamp(session_dir, [], kfs, wits, cfg)
        report = {
            **base,
            "stamp": st_masks,
            "reason": reason,
            "native_w": None, "native_h": None,
            "frames": {},
            "parse_failures": 0,
            "vlm_calls": None,
            "exclusion_masks": {"dir": EXCLUSION_MASKS_REL, "frames": {}, "prompts": {},
                                "scope": cfg.sam3_scope, "requested": {},
                                "n_frames_written": 0,
                                "frame_status": {}, "n_segmented_no_object": 0,
                                "stamp": st_masks},
            "sam3_handover": {"called": False, "verified": False, "check": None,
                              "reason": "content disabled"},
            "weights": {cls: [] for cls in cfg.weight_classes},
            "summary": {"n_keyframes": len(kfs), "n_witnesses": len(wits),
                        "tagged": {cls: 0 for cls in CONTENT_CLASSES}},
        }
        p = write_content(session_dir, report)
        log(f"{LOG_TAG} {reason}; {stale} stale mask file(s) removed; wrote {p}")
        return report

    if not kfs:
        raise ContentError("run_content: no keyframes — I1 produced none, nothing to tag")
    heartbeat_s = _heartbeat(heartbeat_s)
    if tagger is None:
        tagger = QwenTagger(cfg)
    own_segmenter = segmenter is None
    if segmenter is None:
        segmenter = Sam3Segmenter(cfg.sam3_batch, log=log)

    tags = tag_keyframes(frames_dir, kfs, tagger, cfg, log=log, heartbeat_s=heartbeat_s,
                         cancelled=cancelled)
    native_w, native_h = native_size(frames_dir, kfs[0])
    frame_ids_by_class = {cls: scope_frames(kfs, tags, wits, cls, cfg.sam3_scope)
                          for cls in cfg.exclusion_classes}
    prompts = {cls: list(cfg.prompts[cls]) for cls in cfg.exclusion_classes}
    for cls in cfg.exclusion_classes:
        log(f"{LOG_TAG} {cls}: {len(frame_ids_by_class[cls])} frame(s) in scope "
            f"({cfg.sam3_scope}), prompts {prompts[cls]}")
    stale = clear_masks_dir(masks_dir, log=log)
    n_in_scope = sum(len(ids) for ids in frame_ids_by_class.values())
    if n_in_scope == 0:
        handover = {"called": False, "verified": False, "check": None,
                    "reason": "no frame in scope — SAM3 is not called"}
    elif before_sam3 is None:
        handover = {"called": False, "verified": False, "check": None,
                    "reason": "no before_sam3 hook given by the caller"}
    else:
        log(f"{LOG_TAG} every keyframe is tagged; GPU handover before SAM3 "
            f"({n_in_scope} frame-class pair(s) in scope)")
        check = before_sam3()
        verified = isinstance(check, Mapping) and bool(check.get("service_stopped"))
        handover = {"called": True, "verified": verified,
                    "check": dict(check) if isinstance(check, Mapping) else None,
                    "reason": ("before the first SAM3 call" if verified else
                               "before the first SAM3 call; the hook returned no verification "
                               "of the stop")}
    try:
        counts = build_exclusion_masks(frames_dir, masks_dir, frame_ids_by_class, prompts,
                                       segmenter, (native_w, native_h), log=log,
                                       heartbeat_s=heartbeat_s, cancelled=cancelled)
    finally:
        if own_segmenter:
            segmenter.close()

    requested_all = sorted({int(f) for ids in frame_ids_by_class.values() for f in ids})
    weights = {cls: [k for k in kfs if tags[k][cls]] for cls in cfg.weight_classes}
    parse_failed_frames = 0          # a parse failure fails the stage (point 75): none survive
    n_calls = getattr(tagger, "n_calls", None)
    n_parse_failed = getattr(tagger, "parse_failures", None)
    vlm_calls = (None if n_calls is None and n_parse_failed is None
                 else {"n_calls": n_calls, "n_parse_failed": n_parse_failed})
    st_masks = masks_stamp(session_dir, sorted(counts), kfs, wits, cfg)
    report = {
        **base,
        "stamp": st_masks,                 # the readers' key (precision.tracks, point 68)
        "native_w": native_w, "native_h": native_h,
        "frames": {str(k): tags[k] for k in kfs},
        "parse_failures": parse_failed_frames,
        "vlm_calls": vlm_calls,
        "exclusion_masks": {
            "dir": EXCLUSION_MASKS_REL,
            "frames": {str(f): n for f, n in sorted(counts.items())},
            "prompts": prompts,
            "scope": cfg.sam3_scope,
            "requested": {cls: len(ids) for cls, ids in frame_ids_by_class.items()},
            "n_frames_written": len(counts),
            # every frame SAM3 was asked about, and what came back (the Segmenter
            # contract: a frame without a mask was segmented and holds nothing)
            "frame_status": {str(f): ("masked" if f in counts else "segmented_no_object")
                             for f in requested_all},
            "n_segmented_no_object": sum(1 for f in requested_all if f not in counts),
            # what makes these masks valid for a consumer (points 68 / 84): see masks_stamp
            "stamp": st_masks,
        },
        "sam3_handover": handover,
        "weights": weights,
        "summary": {
            "n_keyframes": len(kfs), "n_witnesses": len(wits),
            "tagged": {cls: sum(1 for k in kfs if tags[k][cls]) for cls in CONTENT_CLASSES},
        },
    }
    p = write_content(session_dir, report)
    log(f"{LOG_TAG} wrote {p}: {len(kfs)} keyframes tagged, {len(counts)} exclusion mask(s) "
        f"({stale} stale mask file(s) of a previous run removed first)")
    return report


# ── CLI ──────────────────────────────────────────────────────────────────

def cli_before_content(log: Callable = print) -> Callable[[], None]:
    """The CLIs' pre-I2 hook: the semantic service must be up for the tags —
    ``semantic.service.ensure_service`` (healthcheck, auto-start, wait); a
    service that does not come up fails with the reason it gave (no untagged
    fallback). The map_worker passes its own equivalent."""
    def _ensure() -> None:
        from config import cfg                                    # server/config.py
        from semantic.service import ensure_service
        said: List[str] = []

        def _log(m: Any) -> None:
            said.append(str(m))
            log(m)

        if not ensure_service(cfg, log=_log):
            raise ContentError(
                f"intake I2 (content tags; intake.content.enabled: true) needs the semantic "
                f"service and it is not reachable — "
                f"{said[-1] if said else 'ensure_service returned False without a message'}. "
                f"Start it (bash scripts/serve_semantic.sh) or set intake.content.enabled: "
                f"false; the intake does not fall back to untagged frames")
    return _ensure


def cli_before_sam3(log: Callable = print) -> Callable[[], Dict[str, Any]]:
    """The CLIs' GPU handover before SAM3 — the same verified stopper the
    map_worker passes (``workers.base.stop_semantic_service_verified``: stop,
    then fail naming the PIDs if a ``vllm serve`` process is left), imported
    lazily so the intake package itself never imports the worker. Returns the
    check, which run_content records."""
    def _stop() -> Dict[str, Any]:
        from workers.base import stop_semantic_service_verified
        return stop_semantic_service_verified(None, stage="intake I2 SAM3", log=log)
    return _stop


def _frame_of(basename: str) -> int:
    try:
        return int(Path(basename).stem)
    except ValueError:
        raise ContentError(f"{basename!r} is not named by its video frame number "
                           f"(<frame:06d>.jpg)") from None


def read_i1_frames(session_dir: os.PathLike) -> Tuple[List[int], List[int]]:
    """(keyframes, witnesses) from the I1 artifacts: ``frames/selected_frames.json``
    (``selected_files``, the v2 contract) and ``frames/witness_frames.json``
    (``frames[].frame``). ContentError naming the missing file."""
    frames_dir = Path(session_dir) / FRAMES_DIRNAME
    p_kf = frames_dir / SELECTED_FRAMES_NAME
    p_w = frames_dir / WITNESS_FRAMES_NAME
    for p in (p_kf, p_w):
        if not p.exists():
            raise ContentError(f"{p} does not exist — run intake I1 first "
                               f"(python -m intake.run --session <dir>)")
    with open(p_kf) as f:
        sel = json.load(f)
    with open(p_w) as f:
        wit = json.load(f)
    if not isinstance(sel.get("selected_files"), list):
        raise ContentError(f"{p_kf} lacks 'selected_files' (the v2 contract)")
    if not isinstance(wit.get("frames"), list):
        raise ContentError(f"{p_w} lacks 'frames' (the witness_frames.json contract)")
    keyframes = [_frame_of(b) for b in sel["selected_files"]]
    witnesses = [int(r["frame"]) for r in wit["frames"]]
    return keyframes, witnesses


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m intake.content",
        description="I2 — VLM content tags (vlm_proposed) + SAM3 exclusion masks.")
    ap.add_argument("--session", required=True, help="session directory (frames in "
                    "<session>/frames; I1 artifacts must exist)")
    args = ap.parse_args(argv)
    session_dir = Path(args.session)
    from intake.run_config import cli_intake_config       # the frozen configuration (point 69)
    icfg, _sha = cli_intake_config(session_dir, log=print)
    keyframes, witnesses = read_i1_frames(session_dir)
    if icfg.content.enabled:
        cli_before_content(print)()
    report = run_content(session_dir, keyframes, witnesses, icfg.content, log=print,
                         heartbeat_s=icfg.runtime.heartbeat_s,
                         before_sam3=cli_before_sam3(print))
    tagged = ", ".join(f"{k}={v}" for k, v in report["summary"]["tagged"].items())
    print(f"{LOG_TAG} enabled={report['enabled']} keyframes={report['summary']['n_keyframes']} "
          f"witnesses={report['summary']['n_witnesses']} tagged: {tagged}; "
          f"parse_failures={report['parse_failures']}; exclusion masks "
          f"{report['exclusion_masks']['n_frames_written']} in {report['exclusion_masks']['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
