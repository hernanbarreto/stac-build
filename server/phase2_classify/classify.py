# STAC-Builder — Phase 2: segment classification & enrichment.
#
# For each instance in the R.8 store: crop its best keyframe (largest visible
# area, best score) -> Qwen3-VL classifies final class, apparent material,
# apparent state, notes (strict JSON). CONSUMES and ENRICHES the store; creates
# NO new objects. The VLM class is vlm_proposed; if it contradicts the prompting
# label a conflict is flagged. The final class updates R.5 whitelist eligibility
# and feeds the class->route table (architectural -> surface_fitting; object ->
# visual). Persisted in the same store the Q&A (Phase 5) consumes.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from phase_r.instance_store import InstanceStore

STRUCTURAL_WHITELIST = {"wall", "slab", "column", "vault", "platform", "floor", "beam"}

_SYSTEM = ("You classify a single cropped construction/scene object for a "
           "supervision report. You describe; you never measure. STRICT JSON only.")


def _prompt(proposed_label: str) -> str:
    return (
        f'This crop was detected as "{proposed_label}". Classify it. Return JSON:\n'
        '{"class": "<refined 1-2 word class>", '
        '"material": "<apparent material, e.g. concrete/steel/wood/glass/unknown>", '
        '"state": "<apparent condition, e.g. good/cracked/corroded/stained/damaged/unknown>", '
        '"notes": "<short remark, or empty>", '
        '"confidence": <0..1>}\n'
        "Base it ONLY on what is visible. Output ONLY the JSON."
    )


def is_structural(class_final: str) -> bool:
    """True if the (possibly multi-word) class names a structural element,
    e.g. 'tiled floor' -> floor, 'arched door' stays non-structural."""
    c = (class_final or "").lower()
    return any(w in c.split() or w in c for w in STRUCTURAL_WHITELIST)


def route_for(class_final: str) -> str:
    """class -> reconstruction route (aligns with config class routing)."""
    return "surface_fitting" if is_structural(class_final) else "visual"


def _crop(frames_dir: Path, frame_id: int, box_xywh, pad: float = 0.08):
    p = frames_dir / f"{frame_id:06d}.jpg"
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    w, h = img.size
    x, y, bw, bh = box_xywh
    x1 = max(0, (x - pad) * w); y1 = max(0, (y - pad) * h)
    x2 = min(w, (x + bw + pad) * w); y2 = min(h, (y + bh + pad) * h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img  # degenerate box -> whole frame
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def _best_masklet(store: InstanceStore, iid: int):
    best = None
    for m in store.get_masklets(iid):
        if m["box_xywh"] is None:
            continue
        x, y, bw, bh = m["box_xywh"]
        area = bw * bh
        score = area * (0.5 + (m["score"] or 0.0))
        if best is None or score > best[0]:
            best = (score, m["frame_id"], m["box_xywh"])
    return best


def _parse(txt: str) -> dict | None:
    m = re.search(r"\{.*\}", txt or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class SegmentClassifier:
    def __init__(self, store_path: str, session_dir, backend: str = "qwen_local"):
        self.store = InstanceStore(store_path)
        self.frames_dir = Path(session_dir) / "frames"
        self.backend = backend

    def run(self, on_progress=None) -> dict:
        from semantic.client import get_semantic_client
        from semantic.types import system, user
        client = get_semantic_client(backend=self.backend, consumer="phase2.classify")

        insts = self.store.list_instances()
        n_done = n_conflict = 0
        for i, inst in enumerate(insts):
            iid = inst["instance_id"]
            bm = _best_masklet(self.store, iid)
            if bm is None:
                continue
            crop = _crop(self.frames_dir, bm[1], bm[2])
            if crop is None:
                continue
            resp = client.chat([system(_SYSTEM), user(_prompt(inst["label"]), images=[crop])],
                               max_tokens=256, consumer="phase2.classify")
            d = _parse(resp.content or "")
            if not d:
                continue
            class_final = str(d.get("class", inst["label"])).strip().lower()
            conflict = bool(class_final and inst["label"] and
                            class_final not in inst["label"] and inst["label"] not in class_final)
            eligible = class_final in STRUCTURAL_WHITELIST
            self.store.set_classification(
                iid, class_final=class_final, material=str(d.get("material", "")).lower(),
                state=str(d.get("state", "")).lower(), notes=str(d.get("notes", "")),
                confidence=float(d.get("confidence", 0.5)), conflict=conflict,
                whitelist_eligible=eligible, best_frame=bm[1])
            n_done += 1
            n_conflict += int(conflict)
            if on_progress:
                on_progress(int(100 * (i + 1) / max(1, len(insts))),
                            f"classified {inst['label']} -> {class_final}")
        self.store.close()
        return {"n_instances": len(insts), "n_classified": n_done, "n_conflicts": n_conflict}
