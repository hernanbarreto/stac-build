"""DA3 conditioned on LoGeR's poses (runs in the da3 env): multi-view windows with the METRIC
LoGeR cameras (c2w with translation × s, K on the native grid) given to DA3, which then predicts
depth in the scale of those poses (align_to_input_ext_scale) — maps born coherent with the poses.
Each keyframe takes its depth from the window where it is most central. One npz per keyframe:
depth, conf (DA3's processed grid), K (that grid), c2w (metric)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "depth-anything-3" / "src"))


def windows(n: int, size: int, overlap: int):
    if n <= size:
        return [(0, n)]
    step = size - overlap
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append((b - size if b - a < size else a, b))
        if b == n:
            return out
        a += step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loger_pdit.da3_cond")
    ap.add_argument("--lp_dir", required=True, help="output/loger_pdit (loger.npz, fuse_report.json, images.txt)")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--process_res", type=int, default=840)
    a = ap.parse_args(argv)
    import torch
    from depth_anything_3.api import DepthAnything3
    lp = Path(a.lp_dir); out = lp / "da3_cond"; out.mkdir(exist_ok=True)
    L = np.load(lp / "loger.npz"); s = float(json.load(open(lp / "fuse_report.json"))["metric_scale"])
    paths = [l.strip() for l in open(lp / "images.txt") if l.strip()]
    H, W = (int(x) for x in L["grid_hw"]); H0, W0 = (int(x) for x in L["native_hw"])
    K = L["K"].copy(); K[0] *= W0 / W; K[1] *= H0 / H
    c2w = L["c2w"].copy(); c2w[:, :3, 3] *= s
    w2c = np.linalg.inv(c2w).astype(np.float32)
    n = len(paths)
    wins = windows(n, a.window, a.overlap)
    # the window where each keyframe is most central
    best = {}
    for wi, (b0, b1) in enumerate(wins):
        c = 0.5 * (b0 + b1 - 1)
        for i in range(b0, b1):
            d = abs(i - c)
            if i not in best or d < best[i][1]:
                best[i] = (wi, d)
    print(f"[da3-cond] {n} keyframe(s), {len(wins)} window(s) of {a.window} (overlap {a.overlap}), "
          f"process_res {a.process_res}, metric poses (s = {s:.4f})", flush=True)
    model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
    t0 = time.time()
    for wi, (b0, b1) in enumerate(wins):
        mine = [i for i in range(b0, b1) if best[i][0] == wi]
        if all((out / (Path(paths[i]).stem + ".npz")).exists() for i in mine):
            continue
        with torch.no_grad():
            pred = model.inference(paths[b0:b1], extrinsics=w2c[b0:b1], intrinsics=np.repeat(K[None], b1 - b0, 0).astype(np.float32),
                                   align_to_input_ext_scale=True, process_res=a.process_res)
        dep = np.asarray(pred.depth); conf = np.asarray(pred.conf); Ks = np.asarray(pred.intrinsics)
        for i in mine:
            k = i - b0
            np.savez(out / (Path(paths[i]).stem + ".npz"), depth=dep[k].astype(np.float32),
                     conf=conf[k].astype(np.float32), K=Ks[k].astype(np.float64), c2w=c2w[i])
        el = time.time() - t0
        print(f"[da3-cond] window {wi + 1}/{len(wins)} ({el:.0f} s, ~{el / (wi + 1) * (len(wins) - wi - 1):.0f} s left)", flush=True)
    print(f"[da3-cond] done in {time.time() - t0:.0f} s, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
