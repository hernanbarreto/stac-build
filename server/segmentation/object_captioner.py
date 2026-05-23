"""
Object Captioner — InternVL3 structured description of isolated objects.
=======================================================================

After SAM3 segmentation identifies an object across multiple frames,
generates isolated crops (object only, background darkened) and sends
them to InternVL3 for a STRUCTURED description with four fields:

    CATEGORY  — one or two words (wall, floor, column, pipe, potted plant, ...)
    SHAPE     — dominant geometric form (flat vertical surface, vertical cylinder, irregular organic volume, ...)
    MATERIAL  — main material(s)
    DETAIL    — one short sentence with the most distinctive visible features

Two consumers:
  - `category` / `shape` feed the reconstruction classifier (Stage A).
  - the derived natural-language `caption` ("{category}, {shape}, {material}. {detail}")
    is the text conditioning for ShapeR's T5 embedder — kept narrative
    (NO field labels) because ShapeR was trained on Objaverse-style captions.

Usage:
    from segmentation.object_captioner import caption_object
    fields = caption_object(frames, masks, label="concrete column")
    # → {"category": "...", "shape": "...", "material": "...",
    #    "detail": "...", "caption": "..."}

Authors: Hernán Barreto — Ingerop IN3
"""

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("ObjectCaptioner")

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


def _create_isolated_crop(image: Image.Image, mask: np.ndarray,
                           padding_ratio: float = 0.15,
                           bg_darken: float = 0.15) -> Image.Image:
    """Crop tightly around the mask bbox (with padding) and darken the background."""
    img_np = np.array(image)
    h, w = img_np.shape[:2]

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


def caption_object(
    frames: List[str],
    masks: Dict[str, np.ndarray],
    label: str,
    max_views: int = 4,
    model_id: str = None,
) -> Dict[str, str]:
    """Generate a structured description of a segmented object.

    Returns a dict ``{category, shape, material, detail, caption}``. ``caption``
    is the narrative string for ShapeR conditioning; the rest feed the
    reconstruction classifier. On any failure, returns the label across all
    fields so callers never crash.
    """
    if not frames or not masks:
        logger.warning(f"No frames/masks for '{label}' — using label")
        return _fallback(label)

    t0 = time.time()
    best_frames = _select_best_views(frames, masks, max_views)
    if not best_frames:
        logger.warning(f"No valid views for '{label}' — using label")
        return _fallback(label)

    logger.info(f"Captioning '{label}': {len(best_frames)} best views selected")

    crops = []
    for fp in best_frames:
        fname = Path(fp).name
        mask = masks.get(fname)
        if mask is None:
            continue
        img = Image.open(fp).convert("RGB")
        if mask.shape != (img.height, img.width):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((img.width, img.height), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        crops.append(_create_isolated_crop(img, mask))

    if not crops:
        return _fallback(label)

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
