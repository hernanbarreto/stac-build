"""Autosegment — the segmentation chain on demand, with the prompts in the user's hands.

USER 2026-10-05: "Reconstruir" carries a per-scan check *segment when done*; off, the run
ends at the published cloud. Later, the Instances panel's **Autosegment** button opens a
window that shows the VLM prompt (scene understanding — the SAM3 vocabulary) and the
SAM3 prompts the session has, both editable, and a checkbox per remaining stage (VLM,
SAM3 + mask projection, certification, object descriptions). An edited VLM prompt is
SAVED IN THE SESSION (``output/autosegment.json``) and used by every later VLM pass of
that session; edited SAM3 prompts go straight into ``vlm_analysis.json`` (what the SAM3
stage reads), stamped human_validated.

Provenance rule: the prompt is an instruction, never a measurement — what the VLM
answers is still vlm_proposed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

AUTOSEGMENT_FILE = "autosegment.json"
VLM_ANALYSIS_FILE = "vlm_analysis.json"


def default_vlm_prompt() -> str:
    """The understanding prompt shipped with the auto-prompter."""
    from segmentation.autoprompt.scene_understanding import _PROMPT
    return _PROMPT


def load_autosegment(output_dir) -> dict:
    p = Path(output_dir) / AUTOSEGMENT_FILE
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def vlm_prompt_for(output_dir) -> tuple[str, bool]:
    """(prompt, overridden): the session's saved VLM prompt when it has one,
    else the default."""
    saved = load_autosegment(output_dir).get("vlm_prompt")
    if isinstance(saved, str) and saved.strip():
        return saved, True
    return default_vlm_prompt(), False


def save_vlm_prompt(output_dir, prompt: Optional[str]) -> bool:
    """Persist the session's VLM prompt. Empty, None or the default itself
    REMOVES the override (the session goes back to the shipped prompt).
    Returns whether an override is now stored."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = load_autosegment(out)
    text = (prompt or "").strip()
    if not text or text == default_vlm_prompt().strip():
        doc.pop("vlm_prompt", None)
        doc.pop("vlm_prompt_saved_at", None)
        overridden = False
    else:
        doc["vlm_prompt"] = text
        doc["vlm_prompt_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        doc["provenance"] = "human_validated"
        overridden = True
    (out / AUTOSEGMENT_FILE).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return overridden


def sam3_prompts(output_dir) -> list[str]:
    """The SAM3 prompts the session has (``vlm_analysis.json`` ``prompt``,
    ';'-separated), [] when the VLM never ran."""
    p = Path(output_dir) / VLM_ANALYSIS_FILE
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    return [c.strip() for c in str(d.get("prompt", "")).split(";") if c.strip()]


def set_sam3_prompts(output_dir, prompts: list[str]) -> list[str]:
    """Write the SAM3 prompts the SAM3 stage will read (``vlm_analysis.json``
    ``prompt``) — the rest of the file (shape descriptions, fallbacks) stays; a
    session with no VLM analysis gets a minimal one. Unchanged prompts write
    nothing (the file's mtime drives the SAM3 resume probe)."""
    clean: list[str] = []
    for c in prompts:
        c = " ".join(str(c).split()).strip().strip(";")
        if c and c.lower() not in {x.lower() for x in clean}:
            clean.append(c)
    out = Path(output_dir)
    p = out / VLM_ANALYSIS_FILE
    doc: dict = {}
    if p.exists():
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            doc = {}
    if sam3_prompts(out) == clean and p.exists():
        return clean
    out.mkdir(parents=True, exist_ok=True)
    doc["prompt"] = ";".join(clean)
    doc.setdefault("frame_map", {})
    doc["prompt_source"] = "human_validated"
    doc["prompt_edited_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return clean


def state(output_dir) -> dict:
    """What the Autosegment window shows."""
    out = Path(output_dir)
    prompt, overridden = vlm_prompt_for(out)
    return {
        "vlm_prompt": prompt,
        "vlm_prompt_default": default_vlm_prompt(),
        "vlm_prompt_overridden": overridden,
        "sam3_prompts": sam3_prompts(out),
        "has_vlm_analysis": (out / VLM_ANALYSIS_FILE).exists(),
        "has_cloud": (out / "cleaned_cloud.ply").exists(),
        "has_segmentation": (out / "segmentation_result.json").exists(),
    }
