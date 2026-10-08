"""Autosegment — the segmentation chain on demand, with the prompts in the user's hands.

USER 2026-10-05: "Reconstruir" carries a per-scan check *segment when done*; off, the run
ends at the published cloud. Later, the Instances panel's **Autosegment** button opens a
window that shows the VLM prompt (scene understanding — the SAM3 vocabulary) and the
SAM3 prompts the session has, both editable, and a checkbox per remaining stage (VLM,
SAM3 + mask projection, certification, object descriptions). An edited VLM prompt is
SAVED IN THE SESSION (``output/autosegment.json``) and used by every later VLM pass of
that session; edited SAM3 prompts go into ``vlm_analysis.json`` (what the SAM3 stage reads)
AND ``autoprompt_concepts.json`` (the session's vocabulary record) — one edit, one source
(docs/plan_determinismo.md point 159) — stamped human_validated, with the sha256 of the
text (point 97). No clock enters either file: when a prompt was saved or edited goes to
``autosegment.timing.json`` (points 86 / 166).

Provenance rule: the prompt is an instruction, never a measurement — what the VLM
answers is still vlm_proposed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

AUTOSEGMENT_FILE = "autosegment.json"
AUTOSEGMENT_TIMING_FILE = "autosegment.timing.json"   # when prompts were saved — not compared
VLM_ANALYSIS_FILE = "vlm_analysis.json"
CONCEPTS_FILE = "autoprompt_concepts.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _write_json(path: Path, doc: dict) -> None:
    from atomic_io import atomic_write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc, indent=2, ensure_ascii=False)


def _note_time(output_dir: Path, key: str) -> None:
    """The wall clock of a save, in the record nobody compares."""
    p = Path(output_dir) / AUTOSEGMENT_TIMING_FILE
    doc: dict = {}
    if p.exists():
        try:
            doc = json.loads(p.read_text())
            if not isinstance(doc, dict):
                doc = {}
        except (OSError, ValueError):
            doc = {}
    doc[key] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(p, doc)


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
        doc.pop("vlm_prompt_sha256", None)
        doc.pop("vlm_prompt_saved_at", None)      # a record written before 2026-10-08
        overridden = False
    else:
        doc["vlm_prompt"] = text
        doc["vlm_prompt_sha256"] = _sha256(text)
        doc.pop("vlm_prompt_saved_at", None)
        doc["provenance"] = "human_validated"
        overridden = True
    _write_json(out / AUTOSEGMENT_FILE, doc)
    _note_time(out, "vlm_prompt_saved_at" if overridden else "vlm_prompt_restored_at")
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
    session with no VLM analysis gets a minimal one. The session's vocabulary record
    (``autoprompt_concepts.json``) takes the same list, so there is ONE source (point
    159). Unchanged prompts write nothing (the file's mtime drives the SAM3 resume
    probe). The edit is stamped human_validated with the sha256 of the list; the clock
    goes to the timing record."""
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
    joined = ";".join(clean)
    doc["prompt"] = joined
    doc.setdefault("frame_map", {})
    doc["prompt_source"] = "human_validated"
    doc["prompt_sha256"] = _sha256(joined)
    doc.pop("prompt_edited_at", None)             # a record written before 2026-10-08
    _write_json(p, doc)
    cp = out / CONCEPTS_FILE
    rec: dict = {}
    if cp.exists():
        try:
            rec = json.loads(cp.read_text())
            if not isinstance(rec, dict):
                rec = {}
        except (OSError, ValueError):
            rec = {}
    rec["prompts"] = list(clean)
    rec["prompt_source"] = "human_validated"
    rec["prompt_sha256"] = doc["prompt_sha256"]
    _write_json(cp, rec)
    _note_time(out, "sam3_prompts_edited_at")
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
