"""The second VLM round: the fitted form DRAWN over the views that disagree most.

Only when the converged consensus is in doubt (clearly worse than the cloud's own agreement
with the masks, or losing to an alternative) the VLM sees the real views again with the fitted
form's visible surface drawn on them (red dots) and the object's mask outline (green) and
answers a structured verdict: it matches, or what is wrong — too big, too small, wrong form,
wrong orientation, missing or extra parts, two objects in one segment — and the corrected form.
A size complaint without a form change changes nothing (the optimiser already converged on the
masks: declared); a new form or part list is fitted and judged against the current one by the
same per-view consensus (``silhouette.compare_forms``). Everything the VLM says is
``vlm_proposed`` and recorded, round by round.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .catalog import (CONFIDENCES, FORMS, ORIENTATIONS, OUTLINES, PART_FORMS, PROFILES, AnswerError,
                      FormAnswer, Part, answer_schema, parse_answer)

PROBLEMS = ("ok", "too_big", "too_small", "wrong_form", "wrong_orientation", "missing_parts",
            "extra_parts", "mixed_objects", "other")
CONSUMER = "forms.review"
MAX_SIDE_PX = 896


def review_schema(catalog: List[str]) -> dict:
    base = answer_schema(catalog)
    props = {"matches": {"type": "boolean"},
             "problem": {"type": "string", "enum": list(PROBLEMS)},
             **{k: v for k, v in base["properties"].items()
                if k in ("form", "orientation", "axis_aligned", "hollow", "outline", "profile",
                         "parts", "confidence", "notes")}}
    return {"type": "object", "properties": props,
            "required": ["matches", "problem", "form", "orientation", "axis_aligned", "hollow",
                         "outline", "profile", "parts", "confidence", "notes"]}


_PROMPT = """You proposed that this object is a {form} ({orientation}{extra}). It was fitted to the 3-D scan and the fitted form was DRAWN over {n} real view(s) of the object:
- RED dots = the visible surface of the fitted form, projected into the photo;
- GREEN outline = the object's own mask in that photo.
Where the red dots fill the green outline the model matches the object; red outside the green means the form is too big or wrong there; green with no red means the form is too small or a part is missing.
Measured on the scan (for your information only): the form explains {explained:.0f} % of the object's points within the scan's noise; in the views shown its outline misses the mask by about {px:.0f} pixel(s) on average.

Judge the drawn form and answer ONLY a JSON object:
- matches: true when the form is the right one for this object (small size mismatches are fine);
- problem: "ok" | "too_big" | "too_small" | "wrong_form" | "wrong_orientation" | "missing_parts" | "extra_parts" | "mixed_objects" (the segment holds two different objects) | "other";
- form, orientation, axis_aligned, hollow, outline, profile, parts: the CORRECT description of the object (repeat the current one when it is right; for a composite list every part with its count);
- confidence: "high" | "medium" | "low";
- notes: what a modeller must know, or "".
Never give a dimension; JSON only.
The element is tagged as: {label}."""


def review_prompt(ans: FormAnswer, label: str, n_views: int, explained_frac: float, miss_px: float) -> str:
    extra = ""
    if ans.form == "panel":
        extra = f", outline {ans.outline}"
    elif ans.form == "profile":
        extra = f", section {ans.profile}"
    elif ans.form == "composite" and ans.parts:
        extra = ", parts " + " + ".join(f"{p.count} x {p.form} ({p.name})" for p in ans.parts)
    if ans.hollow:
        extra += ", hollow"
    return _PROMPT.format(form=ans.form, orientation=ans.orientation, extra=extra, n=int(n_views),
                          explained=100.0 * float(explained_frac), px=float(miss_px), label=label)


# ── the overlays ───────────────────────────────────────────────────────────────

def _mask_to_native(mg):
    """Mask-grid pixel (col, row) arrays → native-frame pixels through the declared grid map."""
    def f(gx: np.ndarray, gy: np.ndarray):
        xn = (gx - mg.pad_left + 0.5) * mg.scale_x - 0.5 + mg.crop_x
        yn = (gy - mg.pad_top + 0.5) * mg.scale_y - 0.5 + mg.crop_y
        return xn, yn
    return f


def overlay_image(frame_path: Path, mask: np.ndarray, form_px: np.ndarray, to_native=None,
                  pad_frac: float = 0.25):
    """The photo cropped around the object's mask with the form's visible samples (red) and the
    mask's outline (green) drawn on it. ``form_px``: (N, 2) mask-grid (col, row) of the visible
    samples; ``to_native``: the mask-grid → native map (None: the mask is on the photo's grid)."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as _ndi
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    m = np.asarray(mask, bool)
    if to_native is None:
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        to_native = lambda gx, gy: (gx * (W / mask.shape[1]), gy * (H / mask.shape[0]))  # noqa: E731
    ring = m & ~_ndi.binary_erosion(m)
    ry, rx = np.nonzero(ring)
    gx_r, gy_r = to_native(rx.astype(np.float64), ry.astype(np.float64))
    fx, fy = (to_native(form_px[:, 0], form_px[:, 1]) if len(form_px) else (np.zeros(0), np.zeros(0)))
    draw = ImageDraw.Draw(img)
    for x, y in zip(fx, fy):
        draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(255, 40, 40))
    for x, y in zip(gx_r, gy_r):
        draw.point((float(x), float(y)), fill=(40, 230, 60))
    ys, xs = np.nonzero(m)
    if len(xs):
        x0, x1 = to_native(np.array([xs.min(), xs.max()], float), np.array([ys.min(), ys.max()], float))
        bx0, bx1 = float(min(x0)), float(max(x0))
        by0, by1 = float(min(x1)), float(max(x1))
        if len(fx):
            bx0, bx1 = min(bx0, float(fx.min())), max(bx1, float(fx.max()))
            by0, by1 = min(by0, float(fy.min())), max(by1, float(fy.max()))
        pw, ph = (bx1 - bx0) * pad_frac, (by1 - by0) * pad_frac
        box = (int(max(0, bx0 - pw)), int(max(0, by0 - ph)), int(min(W, bx1 + pw + 1)), int(min(H, by1 + ph + 1)))
        if box[2] > box[0] + 4 and box[3] > box[1] + 4:
            img = img.crop(box)
    w, h = img.size
    if max(w, h) > MAX_SIDE_PX:
        sc = MAX_SIDE_PX / float(max(w, h))
        img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BILINEAR)
    return img


def overlays_for(sil, template, fidxs: Sequence[int], frames_dir: Path, n_samples: int,
                 to_native=None, cloud_frame_of: Optional[Callable[[int], int]] = None) -> List[dict]:
    """The overlay of ``template`` on each requested view (by mask fidx): ``[{fidx, image}]``."""
    out = []
    by_fidx = {v.fidx: v for v in sil.views}
    with torch.no_grad():
        X = template.samples(int(n_samples))
    for fidx in fidxs:
        v = by_fidx.get(int(fidx))
        if v is None:
            continue
        cf = cloud_frame_of(int(fidx)) if cloud_frame_of else v.cloud_frame
        fp = Path(frames_dir) / f"{int(cf):06d}.jpg"
        if not fp.exists():
            continue
        with torch.no_grad():
            px, py, z, visible, inside = sil._project(v, X)
            pts = torch.stack([px[visible], py[visible]], 1).detach().cpu().numpy()
        out.append({"fidx": int(fidx), "frame": int(cf),
                    "image": overlay_image(fp, v.mask.detach().cpu().numpy(), pts, to_native)})
    return out


# ── the call ───────────────────────────────────────────────────────────────────

def ask_review(client, prompt: str, images: list, catalog: List[str], max_tokens: int,
               log: Callable[[str], None] = lambda m: None) -> Dict[str, object]:
    """One structured review call → ``{verdict (dict|None), raw, error, stamp}``."""
    from segmentation.shape_proposer import _extract_json
    from semantic.types import _encode_image, user as user_msg
    refs = [_encode_image(im) for im in images]
    schema = review_schema(catalog)
    msgs = [user_msg(prompt, images=refs)]
    resp = None
    try:
        resp = client.chat(msgs, temperature=0.1, max_tokens=int(max_tokens), consumer=CONSUMER,
                           extra_body={"response_format": {"type": "json_schema",
                                                           "json_schema": {"name": "review",
                                                                           "schema": schema}}})
    except Exception as e:  # noqa: BLE001
        log(f"[forms] review: structured output rejected ({type(e).__name__}: {e}) — plain retry")
        resp = None
    if resp is None:
        resp = client.chat(msgs, temperature=0.1, max_tokens=int(max_tokens), consumer=CONSUMER)
    raw = resp.content or ""
    stamp = {"images_sha1": [str(getattr(r, "sha1", "")) for r in refs],
             "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    doc = _extract_json(raw)
    if not isinstance(doc, dict):
        return {"verdict": None, "raw": raw, "error": "the answer is not JSON", "stamp": stamp}
    v = {"matches": bool(doc.get("matches", False)),
         "problem": str(doc.get("problem") or "other"),
         "notes": str(doc.get("notes") or "")[:500],
         "confidence": str(doc.get("confidence") or "medium")}
    if v["problem"] not in PROBLEMS:
        v["problem"] = "other"
    if v["confidence"] not in CONFIDENCES:
        v["confidence"] = "medium"
    v["answer"] = None
    try:
        v["answer"] = parse_answer({k: doc.get(k) for k in ("form", "orientation", "axis_aligned", "hollow",
                                                           "outline", "profile", "parts", "proportions",
                                                           "alternatives", "confidence", "what_would_help",
                                                           "notes")}, catalog)
    except AnswerError as e:
        v["answer_error"] = f"AnswerError: {e}"
    return {"verdict": v, "raw": raw, "error": None, "stamp": stamp}


def apply_review(current: FormAnswer, verdict: Optional[dict]) -> Optional[FormAnswer]:
    """The corrected form the review asks for, or None when nothing changes: it matches, a
    size complaint with the same form (the optimiser already converged on the masks), or an
    answer that did not parse."""
    if not verdict or verdict.get("matches") or verdict.get("problem") == "ok":
        return None
    new: Optional[FormAnswer] = verdict.get("answer")
    if new is None:
        return None
    same_form = (new.form == current.form and new.orientation == current.orientation
                 and new.hollow == current.hollow and new.outline == current.outline
                 and new.profile == current.profile
                 and [p.to_dict() for p in new.parts] == [p.to_dict() for p in current.parts])
    if same_form:
        return None
    return new


def verdict_record(v: Optional[dict]) -> Optional[dict]:
    if v is None:
        return None
    out = {k: v.get(k) for k in ("matches", "problem", "notes", "confidence", "answer_error")}
    out["answer"] = v["answer"].to_dict() if v.get("answer") is not None else None
    return out
