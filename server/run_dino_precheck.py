#!/usr/bin/env python3
"""PRE-CHECK DIAGNOSTIC (USER 2026-09-04): 'ya tenemos los KF — correrlo
aparte y ver qué determina DE ANTEMANO'.

Appearance-only analysis over the session's keyframes, BEFORE any geometry:
  1. encodes every keyframe (writes/reuses the SAME dino_features cache
     fase 5 consumes — pre-work, not duplicate work);
  2. global-descriptor similarity across the whole walk;
  3. appearance revisit candidates (|i-j| >= gap) VERIFIED by mutual-NN
     patch matches — does the END of the walk re-identify the BEGINNING?
  4. writes output/dino_precheck/: report.json + side-by-side crops of the
     strongest verified long-range pairs, for the user's eye.

    python run_dino_precheck.py --output-dir <session>/output
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
import time
from pathlib import Path

import numpy as np

_SERVER = Path(__file__).resolve().parent
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))


def _fine_verify(frames_dir, ff_a, ff_b, sift_cache):
    """FINE full-resolution verification (USER 2026-09-04: DINO confuses
    places any human eye separates instantly — its 16px/PCA features lose
    the micro-detail). SIFT pixel matches + FUNDAMENTAL RANSAC: the same
    place yields many pixel-accurate epipolar inliers; different places
    collapse. Returns (n_inliers, inlier_ratio)."""
    import cv2
    def _feats(key):
        if key not in sift_cache:
            if len(sift_cache) > 60:
                sift_cache.pop(next(iter(sift_cache)))
            g = cv2.imread(str(frames_dir / key), cv2.IMREAD_GRAYSCALE)
            sift = cv2.SIFT_create(nfeatures=3000)
            sift_cache[key] = sift.detectAndCompute(g, None)
        return sift_cache[key]
    kp_a, des_a = _feats(ff_a)
    kp_b, des_b = _feats(ff_b)
    if des_a is None or des_b is None or len(kp_a) < 50 or len(kp_b) < 50:
        return 0, 0.0
    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n2 in (p for p in knn if len(p) == 2)
            if m.distance < 0.75 * n2.distance]
    if len(good) < 20:
        return 0, 0.0
    pa = np.float32([kp_a[m.queryIdx].pt for m in good])
    pb = np.float32([kp_b[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or mask is None:
        return 0, 0.0
    n_in = int(mask.sum())
    return n_in, n_in / max(len(good), 1)


def main() -> int:
    from PIL import Image, ImageDraw
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--min-gap", type=int, default=300,
                    help="long-range only: pairs at least this many KFs apart")
    ap.add_argument("--min-sim", type=float, default=0.55)
    ap.add_argument("--match-cos", type=float, default=0.60)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.output_dir)
    frames_dir = (Path(args.frames_dir) if args.frames_dir
                  else out.parent / "frames")
    flist = out / "frame_list.json"
    if not flist.exists():
        # early mode: keyframes are already SELECTED before Ω writes
        # frame_list.json — use the selector's own output
        sel = frames_dir / "selected_frames.json"
        if not sel.exists():
            print("REFUSED: neither frame_list.json nor selected_frames.json"
                  " — keyframes not selected yet")
            return 2
        files = json.loads(sel.read_text()).get("selected_files", [])
        if not files:
            print("REFUSED: selected_frames.json is empty")
            return 2
        (out / "dino_precheck").mkdir(parents=True, exist_ok=True)
        flist = out / "dino_precheck" / "frame_list.json"
        flist.write_text(json.dumps(files))

    t0 = time.time()
    from config import cfg as _cfg
    dcfg = dict(_cfg.get("dino_features") or {})
    from reconstruction.dino_features import (FeatureCache,
                                              extract_session_features,
                                              layout_ransac,
                                              sequence_coherent)
    from reconstruction.pose_refine_fm import _mutual_matches
    # ISOLATED cache (USER 2026-09-04): this diagnostic is for HUMANS only —
    # the pipeline generates and consumes ITS OWN dino_features/, untouched.
    _iso = out / "dino_precheck" / "cache"
    dcfg["batch"] = min(int(dcfg.get("batch", 6)), 2)  # coexist with Ω
    extract_session_features(out, frames_dir, dcfg, log=print,
                             cache_dir=_iso, frame_list_path=flist)
    fc = FeatureCache(out, cache_dir=_iso)
    frames = [int(f) for f in fc.meta["frame_nums"]]
    n = len(frames)
    ncell = fc.hp * fc.wp

    # global descriptors
    desc = np.zeros((n, fc.dim), np.float32)
    for k in range(n):
        g = fc.grid(frames[k]).reshape(ncell, fc.dim)
        d = g.mean(axis=0)
        desc[k] = d / max(np.linalg.norm(d), 1e-8)
    S = desc @ desc.T

    # long-range candidates, verified by patch matches
    cands = []
    for i in range(n):
        for j in np.flatnonzero(S[i] >= args.min_sim):
            if j - i >= args.min_gap:
                cands.append((i, int(j), float(S[i, j])))
    cands.sort(key=lambda t: -t[2])
    seen = set()
    verified = []
    n_seq_rejected = n_layout_rejected = n_fine_rejected = 0
    sift_cache = {}
    frame_files_g = {int(Path(f).stem): f for f in json.loads(
        flist.read_text())}
    for i, j, s in cands:
        key = (i // 10, j // 10)
        if key in seen:
            continue
        seen.add(key)
        # ALIASING KILLERS (USER 2026-09-04: 'parecidos pero no lo mismo'):
        # 1) a true revisit matches in a RUN of neighbouring KFs
        if not sequence_coherent(S, i, j):
            n_seq_rejected += 1
            continue
        fa = fc.grid(frames[i]).reshape(ncell, fc.dim)
        fb = fc.grid(frames[j]).reshape(ncell, fc.dim)
        ia, ib, cos = _mutual_matches(fa, fb, args.match_cos)
        # 2) the matches must follow ONE coherent layout transform
        rc_a = np.column_stack([ia // fc.wp, ia % fc.wp])
        rc_b = np.column_stack([ib // fc.wp, ib % fc.wp])
        inl = layout_ransac(rc_a, rc_b)
        if inl.sum() < 12:
            n_layout_rejected += 1
            continue
        # FINE full-res verification — the twin killer
        ff_a, ff_b = frame_files_g.get(frames[i]), frame_files_g.get(frames[j])
        if not ff_a or not ff_b:
            continue
        n_sift, sift_ratio = _fine_verify(frames_dir, ff_a, ff_b,
                                          sift_cache)
        if n_sift < 40 or sift_ratio < 0.30:
            n_fine_rejected += 1
            continue
        verified.append({"kf_a": frames[i], "kf_b": frames[j],
                         "sift_inliers": n_sift,
                         "sift_ratio": round(sift_ratio, 3),
                         "gap": j - i, "global_sim": round(s, 3),
                         "n_patch_matches": int(inl.sum()),
                         "n_raw_matches": int(len(ia)),
                         "match_cos_median": round(float(np.median(
                             cos[inl])), 3) if inl.any() else None})
        if len(verified) >= 200:
            break
    print(f"[precheck] aliasing rejected: {n_seq_rejected} by sequence, "
          f"{n_layout_rejected} by layout RANSAC, {n_fine_rejected} by "
          f"FINE full-res verification (SIFT epipolar)")
    strong = [v for v in verified if v["n_patch_matches"] >= 30]
    print(f"[precheck] {len(cands)} long-range candidates (gap≥"
          f"{args.min_gap}) → {len(verified)} checked → {len(strong)} "
          f"STRONG re-identifications (≥30 patch matches)")
    thirds = [v for v in strong
              if v["kf_a"] <= frames[n // 3] and v["kf_b"] >= frames[2 * n // 3]]
    print(f"[precheck] FIRST↔LAST third bridges: {len(thirds)} — "
          f"{'the end of the walk DOES re-identify the beginning' if thirds else 'NOT FOUND (revisit may not exist or features too weak)'}")

    # crops of the strongest pairs
    dst = out / "dino_precheck"
    dst.mkdir(exist_ok=True)
    frame_files = {int(Path(f).stem): f for f in json.loads(
        flist.read_text())}
    strong.sort(key=lambda v: -v["n_patch_matches"])
    rendered = 0
    for r, v in enumerate(strong[:args.top]):
        imgs = []
        for fn in (v["kf_a"], v["kf_b"]):
            ff = frame_files.get(fn)
            if ff and (frames_dir / ff).exists():
                imgs.append(Image.open(frames_dir / ff).convert("RGB"))
        if len(imgs) != 2:
            continue
        h = min(p.height for p in imgs)
        imgs = [p.resize((int(p.width * h / p.height), h)) for p in imgs]
        combo = Image.new("RGB", (imgs[0].width + imgs[1].width + 8, h + 28),
                          (20, 20, 20))
        combo.paste(imgs[0], (0, 28))
        combo.paste(imgs[1], (imgs[0].width + 8, 28))
        d = ImageDraw.Draw(combo)
        d.text((4, 6), f"KF {v['kf_a']} <-> KF {v['kf_b']}  gap "
                       f"{v['gap']}  sim {v['global_sim']}  SIFT inliers "
                       f"{v.get('sift_inliers')} (ratio "
                       f"{v.get('sift_ratio')})  patches "
                       f"{v['n_patch_matches']}", fill=(120, 255, 120))
        combo.save(dst / f"pair{r:03d}_kf{v['kf_a']}_kf{v['kf_b']}.jpg",
                   quality=85)
        rendered += 1

    (dst / "report.json").write_text(json.dumps({
        "n_keyframes": n, "min_gap": args.min_gap,
        "n_candidates": len(cands), "n_verified": len(verified),
        "n_strong": len(strong), "n_first_last_bridges": len(thirds),
        "strong_pairs": strong[:100],
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured (appearance only, pre-geometry)",
    }, indent=1))
    print(f"[precheck] ✅ {rendered} crops + report.json → {dst} "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
