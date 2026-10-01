"""
Object Captioner — the ShapeR description of every segmented object.
====================================================================

The description ShapeR conditions on (its T5 embedder) is a NARRATIVE caption
with four fields behind it:

    CATEGORY  — one or two words (wall, floor, column, pipe, potted plant, ...)
    SHAPE     — dominant geometric form (flat vertical surface, vertical cylinder, ...)
    MATERIAL  — main material(s)
    DETAIL    — one short sentence with the most distinctive visible features

    caption = "{category}, {shape}, {material}. {detail}"   (NO field labels —
    ShapeR was trained on Objaverse-style captions)

USER 2026-10-01: *"la descripción de los objetos la que armó el VLM como
descripción para ShapeR … el VLM, para preparar los prompts, pasa SAM3 y las
descripciones para ShapeR"* — every segmented object carries one, from TWO
sources that produce the SAME dict (``shape_caption``):

  * ``concept`` — the scene-understanding pass that NAMES the SAM3 prompts
    (autoprompt/scene_understanding.py) also describes each KIND it names; the
    descriptions travel in ``vlm_analysis.json`` (``shape_descriptions``, keyed
    by the SAM3 prompt) and every instance the projection creates under that
    concept's label inherits it (``concept_caption_lookup`` — called where
    segmentation/pipeline.py assembles the projected instances);
  * ``object`` — after the certification (the last mutation of the instances
    and the last GPU-exclusive stage) ``caption_session_objects`` shows each
    instance ISOLATED in its best SAM3-mask views to the session's Qwen3-VL and
    refines the description per object. An object caption is never downgraded
    to its concept's on a re-projection (``carry_object_captions``).

The interface every consumer reads (segmentation/shaper_export.py):

    inst["shape_caption"] = {"caption": str, "category": str, "shape": str,
                             "material": str, "detail": str,
                             "provenance": "vlm_proposed",
                             "source": "concept" | "object",
                             "generated": "<ISO timestamp>"}

Nothing here measures; everything the VLM writes is ``vlm_proposed`` and a
missing answer stays MISSING — the label is never recorded as if the VLM had
described it.

CLI:  python -m segmentation.object_captioner --session <dir> [--views N] [--refresh]

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("ObjectCaptioner")

PROVENANCE = "vlm_proposed"
SOURCE_CONCEPT = "concept"
SOURCE_OBJECT = "object"
CAPTION_KEY = "shape_caption"           # on every instance of segmentation_result.json
DESCRIPTIONS_KEY = "shape_descriptions"  # in vlm_analysis.json, keyed by SAM3 prompt
SECTION = "segmentation.object_captions"

# ── Structured prompt ───────────────────────────────────────────────

_CAPTION_PROMPT = """<image>
You are describing a single isolated 3D object (everything else in the crop is darkened).
Reply using EXACTLY these four fields, one per line, in this order, nothing else:

CATEGORY: one or two words naming what this is — e.g. wall, floor, ceiling, column, beam, pipe, duct, railing, stair, door, window, potted plant, chair, table, cabinet, shelf, equipment, fixture, sign. If unsure, your best single guess.
SHAPE: the dominant geometric form — e.g. flat vertical surface, flat horizontal surface, vertical cylinder, horizontal cylinder, rectangular box, L-shaped prism, irregular organic volume, thin elongated bar, dome.
MATERIAL: the main material(s) — e.g. painted concrete, bare concrete, drywall, plaster, brick, brushed steel, painted steel, wood, glass, plastic, ceramic, green foliage and ceramic.
DETAIL: ONE short sentence (no more than ~20 words) with the most distinctive visible features — recesses, openings, attachments, fittings, color, finish, wear, construction state.

Rules: no markdown, no bullets, no extra lines, no commentary. Describe only what is actually visible. Do not invent details.
The element is tagged as: {label}"""

# Prefixes the model commonly emits in defiance of the prompt — strip them
# from the DETAIL line.
_BANNED_PREFIXES = (
    "the image shows", "this image shows", "the image depicts",
    "this image depicts", "the picture shows", "this picture shows",
    "the photo shows", "this photo shows", "i see", "i can see",
    "i observe", "i notice", "in the image", "in the picture",
    "in this image", "in this picture", "this is", "this appears to be",
    "this looks like", "here is", "here we have", "we can see",
    "we see", "the object is", "the object shown", "the element is",
    "the element shown", "shown in the image", "shown is",
    "the image presents", "depicted is", "depicted in the image",
    "it is", "it appears", "it looks like", "the crop shows",
)

# Field name → regexes accepted in model output (tolerant of markdown/bullets).
_FIELD_KEYS = ("category", "shape", "material", "detail")
_MASK_KEY_RE = re.compile(r"^f(\d+)_o(\d+)$")


class ObjectCaptionsConfigError(KeyError):
    """A missing / invalid ``segmentation.object_captions`` key — always named."""


class SemanticServiceUnavailable(RuntimeError):
    """The session's Qwen3-VL could not be brought up — declared, never a label."""


@dataclass(frozen=True)
class ObjectCaptionsConfig:
    enabled: bool               # the per-object refinement after the certification
    views: int                  # BOUND (cost): mask keyframes shown per object
    understand_max_tokens: int  # BOUND: the understanding answer (now carrying the shapes)


def _req(sec, key: str):
    if not isinstance(sec, dict) or key not in sec:
        raise ObjectCaptionsConfigError(
            f"config.yaml is missing mandatory key '{SECTION}.{key}' — there is no "
            f"hidden default in code")
    return sec[key]


def load_object_captions(config: dict) -> ObjectCaptionsConfig:
    """Typed, strict read of ``segmentation.object_captions`` (a missing key
    fails naming it)."""
    seg = (config or {}).get("segmentation")
    if not isinstance(seg, dict):
        raise ObjectCaptionsConfigError("config.yaml has no 'segmentation' section")
    sec = _req(seg, "object_captions")
    enabled = _req(sec, "enabled")
    if not isinstance(enabled, bool):
        raise ObjectCaptionsConfigError(
            f"'{SECTION}.enabled' must be true or false, got {enabled!r}")
    out = {}
    for key in ("views", "understand_max_tokens"):
        v = _req(sec, key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < 1:
            raise ObjectCaptionsConfigError(
                f"'{SECTION}.{key}' must be an integer >= 1, got {v!r}")
        out[key] = int(v)
    return ObjectCaptionsConfig(enabled=enabled, **out)


def _strip_banned_prefixes(text: str) -> str:
    """Strip leading meta-prefixes (re-run for compound openings)."""
    out = (text or "").strip().strip('"').strip("'")
    for _ in range(5):
        low = out.lower()
        matched = False
        for pref in _BANNED_PREFIXES:
            if low.startswith(pref):
                out = out[len(pref):].lstrip(" ,;:-")
                low2 = out.lower()
                for art in ("a ", "an ", "the "):
                    if low2.startswith(art):
                        out = out[len(art):]
                        break
                matched = True
                break
        if not matched:
            break
    return out.strip().rstrip(".").strip()


def _parse_fields(raw: str, fallback_label: str) -> Dict[str, str]:
    """Parse the four structured fields out of the model output.

    Tolerant: accepts ``CATEGORY: x``, ``- CATEGORY: x``, ``**Category**: x``,
    case-insensitive, in any order. If the model ignored the format entirely,
    the whole output becomes ``detail`` and ``category`` falls back to the label.
    """
    fields: Dict[str, str] = {}
    if raw:
        for line in raw.splitlines():
            ln = line.strip().lstrip("-*•·").strip()
            ln = ln.replace("**", "").replace("__", "")
            m = re.match(r"^\s*([A-Za-z ]+?)\s*[:\-–]\s*(.+?)\s*$", ln)
            if not m:
                continue
            key = m.group(1).strip().lower()
            val = m.group(2).strip().strip('"').strip("'")
            for fk in _FIELD_KEYS:
                if key == fk or key.startswith(fk):
                    if fk not in fields and val:
                        fields[fk] = val
                    break

    label_l = (fallback_label or "object").strip()
    category = fields.get("category", label_l) or label_l
    shape = fields.get("shape", "")
    material = fields.get("material", "")
    detail = _strip_banned_prefixes(fields.get("detail", ""))

    # Total fallback: model produced free text with no recognisable fields.
    if not fields:
        detail = _strip_banned_prefixes(raw or "")
        if len(detail) < 10:
            detail = ""

    return {
        "category": category[:60].strip(),
        "shape": shape[:80].strip(),
        "material": material[:80].strip(),
        "detail": detail[:240].strip(),
    }


def _compose_caption(fields: Dict[str, str], fallback_label: str) -> str:
    """Build the narrative caption fed to ShapeR's T5 embedder (no field labels)."""
    parts = [p for p in (fields.get("category"), fields.get("shape"),
                         fields.get("material")) if p]
    head = ", ".join(parts) if parts else (fallback_label or "object")
    detail = fields.get("detail", "")
    caption = f"{head}. {detail}".strip() if detail else head
    caption = caption.strip()
    if not caption or len(caption) < 3:
        return fallback_label or "object"
    return caption[0].upper() + caption[1:]


# ── the ONE dict both sources produce ───────────────────────────────

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def shape_caption(fields: Dict[str, str], label: str, source: str,
                  generated: Optional[str] = None) -> dict:
    """The instance's ``shape_caption`` from parsed fields — the interface
    segmentation/shaper_export.py reads. ``source`` is ``concept`` (inherited
    from the prompt pass) or ``object`` (refined per instance)."""
    if source not in (SOURCE_CONCEPT, SOURCE_OBJECT):
        raise ValueError(f"unknown shape_caption source {source!r}")
    f = {k: str(fields.get(k) or "").strip() for k in _FIELD_KEYS}
    if not f["category"]:
        f["category"] = (label or "object").strip()
    return {
        "caption": _compose_caption(f, label),
        "category": f["category"],
        "shape": f["shape"],
        "material": f["material"],
        "detail": f["detail"],
        "provenance": PROVENANCE,
        "source": source,
        "generated": generated or _now_iso(),
    }


def fields_from_shape_entry(entry, category: str) -> Optional[Dict[str, str]]:
    """The four fields out of the ``shape`` entry the understanding call returns
    per kind — a dict ``{"form"|"shape", "material", "detail"}`` or one
    structured string in the captioner's own ``FIELD: value`` lines. None when
    there is nothing usable: a missing description is never invented."""
    cat = (category or "").strip()
    if isinstance(entry, dict):
        form = entry.get("form") or entry.get("shape") or ""
        material = entry.get("material") or ""
        detail = entry.get("detail") or entry.get("details") or ""
        fields = {"category": cat,
                  "shape": str(form).strip()[:80],
                  "material": str(material).strip()[:80],
                  "detail": _strip_banned_prefixes(str(detail))[:240]}
    elif isinstance(entry, str) and entry.strip():
        fields = _parse_fields(entry, cat)
        fields["category"] = cat or fields["category"]
    else:
        return None
    if not any(fields[k] for k in ("shape", "material", "detail")):
        return None
    return fields


# ── concept inheritance at the projection ───────────────────────────

def concept_caption_lookup(output_dir) -> Callable[[str], Optional[dict]]:
    """``label -> shape_caption (source 'concept')`` for the session, read ONCE
    from ``vlm_analysis.json`` (``shape_descriptions``, keyed by the SAM3
    prompt; the projected instance carries ``census.concept_label(prompt)``).
    Returns a function that hands out a FRESH copy per instance, None when the
    concept was not described. A session without the file describes nothing."""
    from segmentation.census import concept_label
    by_label: Dict[str, dict] = {}
    p = Path(output_dir) / "vlm_analysis.json"
    if p.exists():
        try:
            raw = (json.loads(p.read_text()).get(DESCRIPTIONS_KEY) or {})
            for prompt, rec in raw.items():
                if isinstance(rec, dict) and rec.get("caption"):
                    by_label.setdefault(concept_label(prompt), rec)
        except Exception as e:  # noqa: BLE001 — declared, nothing inherited
            logger.warning(f"vlm_analysis.json unreadable ({e}) — no concept captions")

    def lookup(label: str) -> Optional[dict]:
        rec = by_label.get(concept_label(label))
        return dict(rec) if rec else None

    return lookup


def carry_object_captions(prev_instances: List[dict], instances: List[dict]) -> int:
    """A per-object caption survives a re-projection: an instance that keeps
    its ``instance_id`` and label takes the ``object`` caption its previous
    self carried instead of the concept's. Never the other way round. Returns
    how many were carried."""
    prev: Dict[tuple, dict] = {}
    for inst in prev_instances or []:
        cap = inst.get(CAPTION_KEY)
        if isinstance(cap, dict) and cap.get("source") == SOURCE_OBJECT:
            prev[(int(inst.get("instance_id", inst.get("id", -1))), inst.get("label"))] = cap
    n = 0
    for inst in instances or []:
        key = (int(inst.get("instance_id", inst.get("id", -1))), inst.get("label"))
        cap = prev.get(key)
        if cap is None:
            continue
        cur = inst.get(CAPTION_KEY)
        if isinstance(cur, dict) and cur.get("source") == SOURCE_OBJECT:
            continue
        inst[CAPTION_KEY] = dict(cap)
        n += 1
    return n


# ── isolated crops ──────────────────────────────────────────────────

def _create_isolated_crop(image: Image.Image, mask: np.ndarray,
                           padding_ratio: float = 0.15,
                           bg_darken: float = 0.15) -> Image.Image:
    """Crop tightly around the mask bbox (with padding) and darken the background."""
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # The bbox is taken from the mask and applied to the IMAGE, so the two have
    # to live on the same grid. SAM3 saves its masks at its own resolution
    # (832x464 on pccr) while the frames on disk are at another, and the
    # mismatch surfaced as "index 255 is out of bounds for axis 0 with size
    # 100" out of the boolean indexing below — 32 of 61 instances lost their
    # crops that way and fell to the default class.
    if mask.shape[:2] != (h, w):
        import cv2 as _cv2
        m2 = mask.astype(np.uint8)
        if m2.ndim == 3:
            m2 = m2[..., 0]
        m2 = _cv2.resize(m2, (w, h), interpolation=_cv2.INTER_NEAREST).astype(bool)
        mask = np.repeat(m2[..., None], img_np.shape[2], axis=2) if img_np.ndim == 3 else m2
    mask = mask.astype(bool)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image  # empty mask

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    bbox_h = rmax - rmin
    bbox_w = cmax - cmin
    pad_r = int(bbox_h * padding_ratio)
    pad_c = int(bbox_w * padding_ratio)
    rmin = max(0, rmin - pad_r)
    rmax = min(h - 1, rmax + pad_r)
    cmin = max(0, cmin - pad_c)
    cmax = min(w - 1, cmax + pad_c)

    crop = img_np[rmin:rmax + 1, cmin:cmax + 1].copy().astype(np.float32)
    mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]
    crop[~mask_crop] *= bg_darken
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    return Image.fromarray(crop)


def _select_best_views(frames: List[str], masks: Dict[str, np.ndarray],
                        max_views: int = 4) -> List[str]:
    """Frames with the largest mask area first (most of the object visible)."""
    scored = []
    for fp in frames:
        fname = Path(fp).name
        if fname in masks:
            scored.append((fp, int(masks[fname].sum())))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [fp for fp, _ in scored[:max_views]]


def _fallback(label: str) -> Dict[str, str]:
    return {"category": label or "object", "shape": "", "material": "",
            "detail": "", "caption": label or "object"}


def _crops_for(frames: List[str], masks: Dict[str, np.ndarray], max_views: int
               ) -> List[Image.Image]:
    crops = []
    for fp in _select_best_views(frames, masks, max_views):
        mask = masks.get(Path(fp).name)
        if mask is None:
            continue
        img = Image.open(fp).convert("RGB")
        if mask.shape != (img.height, img.width):
            mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (img.width, img.height), Image.NEAREST)) > 127
        crops.append(_create_isolated_crop(img, mask))
    return crops


def caption_object(
    frames: List[str],
    masks: Dict[str, np.ndarray],
    label: str,
    max_views: int = 4,
    model_id: str = None,
) -> Dict[str, str]:
    """Generate a structured description of a segmented object (InternVL3,
    the stand-alone path — the pipeline uses :func:`caption_object_qwen`).

    Returns a dict ``{category, shape, material, detail, caption}``. ``caption``
    is the narrative string for ShapeR conditioning; the rest feed the
    reconstruction classifier. On any failure, returns the label across all
    fields so callers never crash.
    """
    if not frames or not masks:
        logger.warning(f"No frames/masks for '{label}' — using label")
        return _fallback(label)

    t0 = time.time()
    crops = _crops_for(frames, masks, max_views)
    if not crops:
        logger.warning(f"No valid views for '{label}' — using label")
        return _fallback(label)
    logger.info(f"Captioning '{label}': {len(crops)} best views selected")

    try:
        from segmentation.scene_analyzer import (
            _load_model, _build_transform, _dynamic_preprocess,
        )
        import torch

        model, tokenizer = _load_model(model_id)
        prompt = _CAPTION_PROMPT.format(label=label)

        best_crop = crops[0]  # largest mask area
        transform = _build_transform(input_size=448)
        images = _dynamic_preprocess(best_crop, image_size=448,
                                     use_thumbnail=True, max_num=4)
        pixel_values = torch.stack([transform(im) for im in images])
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(dtype).to(device)

        generation_config = dict(max_new_tokens=512, do_sample=False)
        raw = model.chat(tokenizer, pixel_values, prompt, generation_config)

        fields = _parse_fields(raw, fallback_label=label)
        fields["caption"] = _compose_caption(fields, fallback_label=label)

        elapsed = time.time() - t0
        logger.info(f"Caption for '{label}' ({elapsed:.1f}s): "
                    f"[{fields['category']}] {fields['caption'][:90]}...")
        logger.debug(f"  raw VLM output: {raw[:200]!r}")

        del model, tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return fields

    except Exception as e:  # noqa: BLE001
        logger.error(f"InternVL captioning failed for '{label}': {e}")
        import traceback
        traceback.print_exc()
        return _fallback(label)


# ── the session's Qwen3-VL ──────────────────────────────────────────

def _service_client(config: Optional[dict] = None,
                    log: Callable[[str], None] = logger.info,
                    cancelled: Optional[Callable[[], bool]] = None):
    """The session's semantic client, bringing vLLM UP when it is down (SAM3
    and the reconstruction stop it for their exclusive GPU window; the cold
    start is ~4.5 min — declared in the log, never a silent fallback).
    Raises :class:`SemanticServiceUnavailable` when it cannot come up."""
    from semantic.service import ensure_service
    if not ensure_service(config, log=log, cancelled=cancelled):
        raise SemanticServiceUnavailable(
            "the semantic service (Qwen3-VL) did not come up — "
            "semantic.service.ensure_service returned False")
    from semantic.client import get_semantic_client
    return get_semantic_client(consumer="shaper.caption")


def caption_object_qwen(
    frames: List[str],
    masks: Dict[str, np.ndarray],
    label: str,
    max_views: int = 4,
    *,
    client=None,
    log: Callable[[str], None] = logger.info,
) -> Dict[str, str]:
    """The structured description from the session's Qwen3-VL service (the
    pipeline's VLM — no second VLM on the GPU). The object is shown ISOLATED
    in its ``max_views`` best views (everything else darkened).

    Returns ``{category, shape, material, detail, caption}``. It RAISES when
    it cannot describe — no usable view, the service cannot be brought up
    (``semantic.service.ensure_service``), the call fails: the label is never
    returned as if the VLM had written it (USER 2026-10-01). ``client`` lets a
    session-wide caller start the service once for every object."""
    if not frames or not masks:
        raise ValueError(f"no frames/masks for '{label}'")
    t0 = time.time()
    crops = _crops_for(frames, masks, max_views)
    if not crops:
        raise ValueError(f"no usable mask view for '{label}'")
    if client is None:
        client = _service_client(None, log=log)
    from semantic.types import user
    prompt = _CAPTION_PROMPT.replace("<image>\n", "").format(label=label)
    if len(crops) > 1:
        prompt = ("The images are views of the SAME object from different viewpoints.\n"
                  + prompt)
    resp = client.chat([user(prompt, images=crops)], max_tokens=256,
                       consumer="shaper.caption")
    raw = resp.content or ""
    if not raw.strip():
        raise RuntimeError(f"empty answer for '{label}'")
    fields = _parse_fields(raw, fallback_label=label)
    fields["caption"] = _compose_caption(fields, fallback_label=label)
    log(f"Caption (Qwen3-VL) for '{label}' ({time.time() - t0:.1f}s, "
        f"{len(crops)} view(s)): [{fields['category']}] {fields['caption'][:90]}")
    return fields


# ── every object of a session ───────────────────────────────────────

def _oids_by_instance(parent: dict, absorbed: dict) -> Dict[int, List[int]]:
    """``instance_id -> npz object ids`` from the parent (segmentation.json:
    one masklet per entry, ``id`` = oid). A masklet the matcher ABSORBED into
    a survivor that the parent fusion has not folded yet still testifies for
    that survivor (``absorbed[<mask instance_id>].into``)."""
    own: Dict[int, int] = {}
    for e in parent.get("instances") or []:
        iid = e.get("instance_id", e.get("id"))
        if iid is None or e.get("id") is None:
            continue
        own[int(iid)] = int(e["id"])
    out: Dict[int, List[int]] = {}
    for iid, oid in own.items():
        out.setdefault(iid, []).append(oid)
    for k, rec in (absorbed or {}).items():
        try:
            src, into = int(k), rec.get("into")
        except (TypeError, ValueError, AttributeError):
            continue
        if into is None or src not in own:
            continue
        lst = out.setdefault(int(into), [])
        if own[src] not in lst:
            lst.append(own[src])
    return out


def _instance_views(z, space, frames_dir: Path, oids: List[int], n_views: int):
    """The instance's ``n_views`` largest SAM3 masks (one per keyframe, the
    masks of its oids OR-ed), as ``(frame paths, {filename: mask})`` for
    :func:`caption_object_qwen`. The npz keys are KEYFRAME POSITIONS; the
    JPEGs are named by the video frame number — ``mask_space`` translates."""
    want = {int(o) for o in oids}
    by_pos: Dict[int, List[str]] = {}
    for key in z.files:
        m = _MASK_KEY_RE.match(key)
        if m and int(m.group(2)) in want:
            by_pos.setdefault(int(m.group(1)), []).append(key)
    scored = []
    for pos, keys in by_pos.items():
        mask = None
        for k in keys:
            a = np.asarray(z[k]) > 0
            mask = a if mask is None else (mask | a)
        area = int(mask.sum())
        if area > 0:
            scored.append((area, pos, mask))
    scored.sort(key=lambda t: t[0], reverse=True)
    frames: List[str] = []
    masks: Dict[str, np.ndarray] = {}
    for _area, pos, mask in scored:
        cf = space.to_cloud(pos)
        if cf is None:
            continue
        p = frames_dir / f"{int(cf):06d}.jpg"
        if not p.exists():
            continue
        frames.append(str(p))
        masks[p.name] = mask
        if len(frames) >= n_views:
            break
    return frames, masks


def caption_session_objects(output_dir, session_dir, *, views: int, refresh: bool = False,
                            log: Callable[[str], None] = print,
                            cancelled: Optional[Callable[[], bool]] = None,
                            client=None, config: Optional[dict] = None) -> dict:
    """One ``object`` caption per instance of ``segmentation_result.json`` that
    lacks one (every instance with ``refresh``): its ``views`` largest SAM3
    masks, isolated crops, ONE Qwen3-VL call per instance, written back under
    the session's matching lock with ``atomic_write_json``.

    When the semantic service cannot be brought up NOTHING is recorded — the
    concept captions stay, the reason is logged and returned (``skipped``).
    Returns ``{generated, kept, failed, skipped, n_instances, views}``."""
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    views = int(views)
    if views < 1:
        raise ValueError(f"views must be >= 1, got {views}")
    res_p = output_dir / "segmentation_result.json"
    if not res_p.exists():
        log(f"[captions] no segmentation_result.json in {output_dir} — nothing to describe")
        return {"generated": 0, "kept": 0, "failed": 0, "skipped": 0, "n_instances": 0,
                "views": views, "reason": "no segmentation_result.json"}
    result = json.loads(res_p.read_text())
    instances = result.get("instances") or []
    out = {"generated": 0, "kept": 0, "failed": 0, "skipped": 0,
           "n_instances": len(instances), "views": views}

    pending = []
    for inst in instances:
        cap = inst.get(CAPTION_KEY)
        if (not refresh and isinstance(cap, dict) and cap.get("source") == SOURCE_OBJECT):
            out["kept"] += 1
        else:
            pending.append(inst)
    if not pending:
        log(f"[captions] every one of the {len(instances)} instance(s) already carries an "
            f"object description — nothing to do")
        return out

    parent_p, masks_p = output_dir / "segmentation.json", output_dir / "seg_masks.npz"
    if not parent_p.exists() or not masks_p.exists():
        out["skipped"] = len(pending)
        out["reason"] = "no segmentation.json / seg_masks.npz — no mask views"
        log(f"[captions] {out['reason']}; the concept descriptions stay")
        return out
    parent = json.loads(parent_p.read_text())
    oids_of = _oids_by_instance(parent, result.get("absorbed") or {})

    if client is None:
        try:
            client = _service_client(config, log=log, cancelled=cancelled)
        except Exception as e:  # noqa: BLE001 — declared, nothing invented
            out["skipped"] = len(pending)
            out["reason"] = f"semantic service unavailable: {e}"
            log(f"[captions] {out['reason']} — {len(pending)} instance(s) keep their concept "
                f"description, nothing recorded")
            return out

    from segmentation import mask_space
    z = np.load(masks_p, allow_pickle=True)
    space = mask_space.resolve(output_dir, masks=z, log=lambda m: log(f"[captions]   {m}"))
    frames_dir = session_dir / "frames"
    new_caps: Dict[int, dict] = {}
    t0 = time.time()
    try:
        for n, inst in enumerate(pending, 1):
            if cancelled is not None and cancelled():
                log(f"[captions] cancelled after {n - 1} of {len(pending)}")
                out["skipped"] = len(pending) - (n - 1)
                break
            iid = int(inst.get("instance_id", inst.get("id", -1)))
            label = str(inst.get("label", "object"))
            oids = list(oids_of.get(iid, []))
            if not oids and inst.get("split_from") is not None:
                # a co-visible split child is a piece of its parent's masklet:
                # the parent's masks are the only views SAM3 drew of it
                oids = list(oids_of.get(int(inst["split_from"]), []))
            frames, masks = (_instance_views(z, space, frames_dir, oids, views)
                             if oids else ([], {}))
            if not frames:
                out["failed"] += 1
                log(f"[captions] #{iid} '{label}': no SAM3 mask view on disk — not described")
                continue
            try:
                fields = caption_object_qwen(frames, masks, label, max_views=views,
                                             client=client, log=lambda m: None)
            except Exception as e:  # noqa: BLE001 — declared per object, the rest go on
                out["failed"] += 1
                log(f"[captions] #{iid} '{label}': VLM description failed ({e}) — not described")
                continue
            cap = shape_caption(fields, label, SOURCE_OBJECT)
            new_caps[iid] = cap
            out["generated"] += 1
            log(f"[captions] #{iid} '{label}' ({n}/{len(pending)}, {len(frames)} view(s)): "
                f"{cap['caption'][:100]}")
    finally:
        try:
            z.close()
        except Exception:  # noqa: BLE001
            pass
        if new_caps:
            _write_captions(output_dir, new_caps, log)
    log(f"[captions] {out['generated']} described, {out['kept']} kept, {out['failed']} failed, "
        f"{out['skipped']} skipped in {time.time() - t0:.0f} s ({views} view(s) per object)")
    return out


def _write_captions(output_dir: Path, caps: Dict[int, dict],
                    log: Callable[[str], None]) -> int:
    """Merge the captions into the result ON DISK under the matching lock — the
    file is re-read there, so the minutes of VLM calls cannot overwrite a
    result another writer produced meanwhile."""
    from atomic_io import atomic_write_json
    from segmentation.match_lock import matching_lock
    res_p = output_dir / "segmentation_result.json"
    with matching_lock(output_dir, log=lambda m: log(f"[captions]{m}")):
        result = json.loads(res_p.read_text())
        n = 0
        for inst in result.get("instances") or []:
            iid = int(inst.get("instance_id", inst.get("id", -1)))
            if iid in caps:
                inst[CAPTION_KEY] = dict(caps[iid])
                n += 1
        atomic_write_json(res_p, result)
    missing = len(caps) - n
    log(f"[captions] 💾 {n} object description(s) written to segmentation_result.json"
        + (f" ({missing} instance(s) no longer in the file)" if missing else ""))
    return n


def _main(argv=None) -> int:
    import argparse
    import sys
    server_dir = str(Path(__file__).resolve().parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from config import cfg
    ccfg = load_object_captions(cfg)
    ap = argparse.ArgumentParser(description="ShapeR description per segmented object (Qwen3-VL)")
    ap.add_argument("--session", required=True, help="session dir (holds frames/ and output/)")
    ap.add_argument("--views", type=int, default=ccfg.views,
                    help=f"mask keyframes shown per object (config: {ccfg.views})")
    ap.add_argument("--refresh", action="store_true", help="describe every object again")
    args = ap.parse_args(argv)
    session = Path(args.session)
    res = caption_session_objects(session / "output", session, views=args.views,
                                  refresh=args.refresh, log=print, config=cfg)
    print(json.dumps(res, indent=2))
    return 0 if res.get("skipped", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
