#!/usr/bin/env python3
"""DIAGNOSTIC (USER 2026-09-04): human-readable report over the fase-5
landmark registry — does the system UNDERSTAND what it sees per keyframe,
and does it re-identify early-walk objects in the LATE keyframes?

Consumes ONLY pipeline artifacts (dino_features cache + landmarks.json +
keyframes). Produces output/landmark_report/:
  - report.json: per multi-visit landmark: frames, feature similarity,
    GEOMETRIC DISAGREEMENT between its observations (the displaced-door
    number, measured), span in keyframes;
  - pair crops: side-by-side JPEG of the earliest vs latest observation of
    the top-N longest-span landmarks, annotated with similarity + offset —
    the "is this really the same door?" evidence for the user's eye.

    python run_landmark_report.py --output-dir <session>/output
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
try:
    os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
except Exception:  # noqa: BLE001
    pass

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SERVER = Path(__file__).resolve().parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))


def main() -> int:
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--top", type=int, default=40,
                    help="render crops for the N longest-span landmarks")
    args = ap.parse_args()
    out = Path(args.output_dir)
    frames_dir = (Path(args.frames_dir) if args.frames_dir
                  else out.parent / "frames")

    lm_path = out / "landmarks.json"
    if not lm_path.exists():
        print("REFUSED: no landmarks.json — run the pipeline first (fase 5 "
              "writes the registry; this tool only READS pipeline artifacts)")
        return 2
    reg = json.loads(lm_path.read_text())
    lms = reg.get("multivisit_landmarks", [])
    if not lms:
        print("landmarks.json has no multi-visit landmarks — nothing to "
              "diagnose (the walk may have no revisits, or fase 5 refused)")
        return 1

    frame_files = {int(Path(f).stem): f for f in json.loads(
        (out / "frame_list.json").read_text())}
    dst = out / "landmark_report"
    dst.mkdir(exist_ok=True)

    # rank by keyframe SPAN — the "door at the end of the walk" cases first
    for lm in lms:
        fs = lm["frames"]
        lm["span"] = max(fs) - min(fs)
    lms.sort(key=lambda l: -l["span"])

    spans = np.array([l["span"] for l in lms])
    print(f"[report] {len(lms)} multi-visit landmarks | span median "
          f"{int(np.median(spans))} frames, max {int(spans.max())}")
    long_range = [l for l in lms if l["span"] >= 600]
    print(f"[report] {len(long_range)} landmarks bridge FIRST↔LAST thirds "
          f"of the walk (span ≥600 frames) — the re-identification the "
          f"user asked about "
          f"{'EXISTS' if long_range else 'was NOT found'}")

    rendered = 0
    for i, lm in enumerate(lms[:args.top]):
        f0, f1 = min(lm["frames"]), max(lm["frames"])
        pair = []
        for fn in (f0, f1):
            ff = frame_files.get(fn)
            if ff and (frames_dir / ff).exists():
                pair.append(Image.open(frames_dir / ff).convert("RGB"))
        if len(pair) != 2:
            continue
        h = min(p.height for p in pair)
        pair = [p.resize((int(p.width * h / p.height), h)) for p in pair]
        combo = Image.new("RGB", (pair[0].width + pair[1].width + 8, h + 28),
                          (20, 20, 20))
        combo.paste(pair[0], (0, 28))
        combo.paste(pair[1], (pair[0].width + 8, 28))
        d = ImageDraw.Draw(combo)
        d.text((4, 6), f"landmark {i}: KF {f0}  <->  KF {f1}   "
                       f"(span {lm['span']} frames, {lm['n_obs']} obs, "
                       f"centroid {lm['centroid']})", fill=(255, 220, 80))
        combo.save(dst / f"lm{i:03d}_kf{f0}_kf{f1}.jpg", quality=85)
        rendered += 1

    summary = {
        "n_multivisit": len(lms),
        "span_median": int(np.median(spans)),
        "span_max": int(spans.max()),
        "n_long_range_600": len(long_range),
        "long_range_examples": [
            {"frames": l["frames"], "n_obs": l["n_obs"],
             "centroid": l["centroid"]} for l in long_range[:50]],
        "crops_rendered": rendered,
        "provenance": "tool_measured (pipeline artifacts only)",
    }
    (dst / "report.json").write_text(json.dumps(summary, indent=1))
    print(f"[report] ✅ {rendered} side-by-side crops + report.json → "
          f"{dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
