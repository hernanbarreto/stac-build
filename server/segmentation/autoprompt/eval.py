# STAC-Builder — Auto-prompter evaluation (Phase 1).
#
# Reports per-class recall/precision of the Qwen3-VL detector against a
# hand-segmented ground truth (Manager format: segmentation.json + seg_masks.npz),
# matching boxes at IoU>0.5, plus a time-vs-manual estimate. Documents the
# classes where the 8B is weak (small objects) — the datum for deciding whether
# to escalate to the larger model.
#
#   conda activate da3 && cd server
#   python -m segmentation.autoprompt.eval --session <dir> --gt <gt_output_dir>
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parents[2]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PIL import Image  # noqa: E402

from segmentation.autoprompt.detector import GroundedDetector  # noqa: E402
from segmentation.autoprompt.vocabulary import load_vocabulary  # noqa: E402


def _mask_bbox_norm(mask: np.ndarray) -> tuple | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    return (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def load_gt_boxes(gt_output_dir: Path) -> dict[int, list[tuple[str, tuple]]]:
    """From Manager-format GT: {frame_idx: [(label, box_norm), ...]}."""
    seg = json.load(open(gt_output_dir / "segmentation.json"))
    id_to_label = {inst["id"]: inst["label"] for inst in seg.get("instances", [])}
    npz = np.load(gt_output_dir / seg.get("mask_file", "seg_masks.npz"))
    out: dict[int, list[tuple[str, tuple]]] = {}
    for key in npz.files:
        if not key.startswith("f") or "_o" not in key:
            continue
        fpart, opart = key[1:].split("_o")
        frame_idx, obj_id = int(fpart), int(opart)
        label = id_to_label.get(obj_id)
        if label is None:
            continue
        bb = _mask_bbox_norm(npz[key])
        if bb:
            out.setdefault(frame_idx, []).append((label, bb))
    return out


def evaluate(session_dir: Path, gt_dir: Path, backend: str, iou_thr: float,
             conf_thr: float, manual_sec_per_click: float, max_frames: int = 0) -> dict:
    from semantic.client import get_semantic_client

    frames_dir = session_dir / "frames"
    gt = load_gt_boxes(gt_dir)
    if max_frames and len(gt) > max_frames:
        keep = sorted(gt.keys())[:: max(1, len(gt) // max_frames)][:max_frames]
        gt = {k: gt[k] for k in keep}
    vocab = load_vocabulary()
    detector = GroundedDetector(get_semantic_client(backend=backend, consumer="phase1.eval"), vocab)

    from collections import defaultdict
    # GT may contain labels outside our vocabulary (e.g. a manual "ladder"/"wall1"
    # session); count them too so recall isn't silently inflated.
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # index frames by their numeric stem — sessions are not guaranteed to use
    # zero-padded 6-digit names, and a silent skip here inflated the metrics
    # (skipped frames looked like FN=0). Frames that truly can't be resolved
    # are REPORTED, not swallowed.
    frame_by_num: dict[int, Path] = {}
    for p in sorted(frames_dir.glob("*.jpg")):
        try:
            frame_by_num[int(p.stem)] = p
        except ValueError:
            continue

    t0 = time.time()
    n_frames = 0
    n_gt = 0
    n_unresolved = 0
    for frame_idx, gt_items in sorted(gt.items()):
        fpath = frame_by_num.get(int(frame_idx))
        if fpath is None:
            n_unresolved += 1
            print(f"[eval] WARNING: GT frame {frame_idx} has no matching file in "
                  f"{frames_dir} — excluded from metrics")
            continue
        n_frames += 1
        img = Image.open(fpath).convert("RGB")
        preds = [d for d in detector.detect(img, frame_id=frame_idx)
                 if d.confidence >= vocab.min_confidence_for(d.label, conf_thr)]
        # per class greedy matching
        by_class_gt: dict[str, list] = {}
        for lbl, bb in gt_items:
            by_class_gt.setdefault(lbl, []).append(bb)
            n_gt += 1
        by_class_pred: dict[str, list] = {}
        for d in preds:
            by_class_pred.setdefault(d.label, []).append(d.box)

        for c in set(list(by_class_gt) + list(by_class_pred)):
            gts = list(by_class_gt.get(c, []))
            prs = list(by_class_pred.get(c, []))
            matched_gt = set()
            for pb in prs:
                best_j, best_iou = -1, iou_thr
                for j, gb in enumerate(gts):
                    if j in matched_gt:
                        continue
                    v = _iou(pb, gb)
                    if v >= best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    tp[c] += 1
                    matched_gt.add(best_j)
                else:
                    fp[c] += 1
            fn[c] += len(gts) - len(matched_gt)
    elapsed = time.time() - t0

    # per-class metrics
    rows = []
    tot_tp = tot_fp = tot_fn = 0
    all_classes = sorted(set(tp) | set(fp) | set(fn))
    for c in all_classes:
        t, f, n = tp[c], fp[c], fn[c]
        if t + f + n == 0:
            continue
        prec = t / (t + f) if (t + f) else 0.0
        rec = t / (t + n) if (t + n) else 0.0
        rows.append({"class": c, "tp": t, "fp": f, "fn": n,
                     "precision": round(prec, 3), "recall": round(rec, 3),
                     "small": c in vocab.small_labels()})
        tot_tp += t; tot_fp += f; tot_fn += n
    overall = {
        "precision": round(tot_tp / (tot_tp + tot_fp), 3) if (tot_tp + tot_fp) else 0.0,
        "recall": round(tot_tp / (tot_tp + tot_fn), 3) if (tot_tp + tot_fn) else 0.0,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
    }
    manual_estimate_s = n_gt * manual_sec_per_click
    return {
        "n_frames": n_frames, "n_gt_instances": n_gt,
        "n_unresolved_gt_frames": n_unresolved,
        "per_class": rows, "overall": overall,
        "timing": {
            "auto_seconds": round(elapsed, 1),
            "manual_estimate_seconds": round(manual_estimate_s, 1),
            "speedup_x": round(manual_estimate_s / elapsed, 1) if elapsed else None,
            "note": f"manual estimate = {manual_sec_per_click}s/instance-click over {n_gt} GT instances",
        },
        "weak_small_classes": [r["class"] for r in rows if r["small"] and r["recall"] < 0.5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 auto-prompter evaluation")
    ap.add_argument("--session", required=True)
    ap.add_argument("--gt", required=True, help="ground-truth output dir (segmentation.json + seg_masks.npz)")
    ap.add_argument("--backend", default="qwen_local")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--manual-sec-per-click", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all GT frames")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    report = evaluate(Path(args.session), Path(args.gt), args.backend,
                      args.iou, args.conf, args.manual_sec_per_click, args.max_frames)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
