"""PointDiT depth maps per frame (runs in the da3 env): z up to an affine map + validity, at
PointDiT's working size for the frame (its training token budget), one npz per frame."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loger_pdit.pdit_maps")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    import cv2
    from config import cfg
    from precision.config import load_precision_config
    from precision.pointdit_runner import PointDiTRunner, working_size
    md = load_precision_config(cfg).mono_detail
    runner = PointDiTRunner(md, log=print).load()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = [l.strip() for l in open(a.images) if l.strip()]
    t0 = time.time()
    for i, p in enumerate(paths):
        dst = out / (Path(p).stem + ".npz")
        if dst.exists():
            continue
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = working_size(img.shape[0], img.shape[1])
        z, valid = runner.depth(img, (h, w))
        np.savez(dst, z=z.astype(np.float32), valid=valid)
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            el = time.time() - t0
            print(f"[pointdit] {i + 1}/{len(paths)} frames ({el:.0f} s, ~{el / (i + 1) * (len(paths) - i - 1):.0f} s left)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
