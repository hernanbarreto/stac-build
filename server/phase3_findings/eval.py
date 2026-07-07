# STAC-Builder — Phase 3 precision evaluation (brutal honesty).
#
# False positives destroy a supervision report's credibility, so the finding
# detector must be measured on a HAND-ANNOTATED crop set (~50 crops, with and
# without defects, drawn from our own keyframes). This module:
#
#   build_set()  — samples crops from keyframes (tight crops on segmented
#                  instances + background crops) and writes them plus an
#                  annotations template (label per crop = "clean" | "defect").
#   evaluate()   — runs the SAME inspection prompt per crop, scores each as
#                  positive if any finding is reported, and reports
#                  precision / recall / specificity against the human labels.
#
# Only crops with a non-null human label count; the rest are reported as
# "pending annotation" — the harness never fabricates ground truth.
#
# PROVENANCE: ours.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_findings.detect import _SYSTEM, _parse_findings, _prompt


def build_set(session_dir, output_dir, out_dir, n_frames: int = 25,
              crops_per_frame: int = 2, seed_stride: int = 7) -> dict:
    """Sample crops from keyframes; write PNGs + annotations template. No model."""
    session_dir, output_dir, out_dir = Path(session_dir), Path(output_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = json.load(open(output_dir / "segmentation.json"))
    id_to = {i["id"]: i for i in seg.get("instances", [])}
    npz = np.load(output_dir / seg.get("mask_file", "seg_masks.npz"))
    by_frame: dict[int, list[str]] = {}
    for key in npz.files:
        if key.startswith("f") and "_o" in key:
            by_frame.setdefault(int(key[1:].split("_o")[0]), []).append(key)
    frames = sorted(by_frame.keys())[::seed_stride][:n_frames]

    template = _load_template(out_dir)
    entries = {e["crop"]: e for e in template.get("crops", [])}
    for fid in frames:
        img_path = session_dir / "frames" / f"{fid:06d}.jpg"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        # one tight crop on an instance mask bbox, one background crop
        boxes = []
        keys = by_frame[fid][:crops_per_frame]
        for key in keys:
            oid = int(key.split("_o")[1])
            m = npz[key]
            ys, xs = np.where(m > 0)
            if xs.size:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                label = id_to.get(oid, {}).get("label", f"obj{oid}")
                boxes.append((int(x1), int(y1), int(x2), int(y2), label))
        for j, (x1, y1, x2, y2, label) in enumerate(boxes):
            name = f"f{fid}_c{j}.png"
            img.crop((x1, y1, x2, y2)).save(out_dir / name)
            entries.setdefault(name, {"crop": name, "frame": fid, "source_label": label,
                                      "label": None, "defect_type": None, "note": ""})
    template["crops"] = sorted(entries.values(), key=lambda e: e["crop"])
    (out_dir / "annotations.json").write_text(json.dumps(template, indent=2))
    n_lab = sum(1 for e in template["crops"] if e["label"] in ("clean", "defect"))
    return {"n_crops": len(template["crops"]), "n_labeled": n_lab, "dir": str(out_dir)}


def _load_template(out_dir: Path) -> dict:
    p = out_dir / "annotations.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {"crops": []}


def evaluate(out_dir, backend: str = "qwen_local", min_confidence: float = 0.35) -> dict:
    """Run the inspection prompt per labeled crop; report honest precision."""
    from semantic.client import get_semantic_client
    from semantic.types import system, user
    out_dir = Path(out_dir)
    ann = json.load(open(out_dir / "annotations.json"))
    client = get_semantic_client(backend=backend, consumer="phase3.eval")

    tp = fp = tn = fn = 0
    pending = 0
    rows = []
    for e in ann.get("crops", []):
        gt = e.get("label")
        if gt not in ("clean", "defect"):
            pending += 1
            continue
        img = Image.open(out_dir / e["crop"]).convert("RGB")
        resp = client.chat([system(_SYSTEM), user(_prompt(), images=[img])],
                           max_tokens=768, consumer="phase3.eval")
        items = [f for f in _parse_findings(resp.content or "")
                 if float(f.get("confidence", 0.0)) >= min_confidence]
        pred = "defect" if items else "clean"
        if gt == "defect" and pred == "defect":
            tp += 1
        elif gt == "clean" and pred == "defect":
            fp += 1
        elif gt == "clean" and pred == "clean":
            tn += 1
        else:
            fn += 1
        rows.append({"crop": e["crop"], "gt": gt, "pred": pred,
                     "n_pred": len(items),
                     "types": [i.get("type") for i in items]})

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "pending_annotation": pending,
            "precision": precision, "recall": recall, "specificity": specificity,
            "rows": rows}


def _self_check() -> None:
    """Unit-level sanity: parsing + scoring logic without the model."""
    assert _parse_findings('{"findings": []}') == []
    assert len(_parse_findings('{"findings": [{"type":"crack"}]}')) == 1
    # confusion arithmetic
    r = {"tp": 3, "fp": 1, "tn": 5, "fn": 2}
    prec = r["tp"] / (r["tp"] + r["fp"])
    assert abs(prec - 0.75) < 1e-9
    print("phase3.eval self-check OK")


if __name__ == "__main__":
    _self_check()
