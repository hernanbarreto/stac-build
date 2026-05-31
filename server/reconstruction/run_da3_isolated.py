#!/usr/bin/env python3
"""
Isolated DA3 metric depth (no streaming, no SLAM, no poses). Runs Depth-Anything-3
per image (N=1, independent) over all frames → da3_depth/{frame:06d}.npz with
{depth, conf, intrinsics}. This is ONLY the per-frame metric reference used to
calibrate ViPE depth; DA3 poses are not produced/used.

Runs in the da3 conda env (torch + depth_anything_3 on PYTHONPATH).

    python run_da3_isolated.py --frames-dir <frames> --out <out>/da3_depth \
        [--model depth-anything/DA3NESTED-GIANT-LARGE-1.1] [--process-res 504]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selected-frames", default=None,
                    help="selected_frames.json — restrict inference to these keyframes")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # depth_anything_3 package (vendored)
    da3_src = Path("/workspace/stac-build/vendor/depth-anything-3/src")
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))
    import torch
    from depth_anything_3.api import DepthAnything3

    print(f"[da3-iso] loading {args.model} on {args.device}", flush=True)
    model = DepthAnything3.from_pretrained(args.model).to(args.device)
    model.device = torch.device(args.device)
    model.eval()

    frames = sorted(glob.glob(str(args.frames_dir / "*.jpg")) +
                    glob.glob(str(args.frames_dir / "*.png")))
    if args.selected_frames and Path(args.selected_frames).exists():
        import json
        sf = json.load(open(args.selected_frames))
        keep = set(sf if isinstance(sf, list) else sf.get("selected_files", []))
        frames = [f for f in frames if Path(f).name in keep]
        print(f"[da3-iso] restricted to {len(frames)} keyframes (selected_frames)", flush=True)
    print(f"[da3-iso] {len(frames)} frames", flush=True)

    done = 0
    for fp in frames:
        frame_no = int(Path(fp).stem)
        out_npz = args.out / f"{frame_no:06d}.npz"
        if out_npz.exists():
            done += 1
            continue
        # N=1 → isolated monocular metric depth (no cross-frame influence).
        # export_dir=None → nothing written; export_format must stay a valid
        # string (DA3 does `"gs" in export_format` before the export_dir check).
        pred = model.inference([fp], process_res=args.process_res,
                               export_dir=None, export_format="mini_npz")
        depth = np.asarray(pred.depth[0], dtype=np.float32)            # (H,W)
        K = (np.asarray(pred.intrinsics[0], dtype=np.float64)
             if pred.intrinsics is not None else None)
        save = {"depth": depth, "is_metric": np.int32(getattr(pred, "is_metric", 1))}
        if K is not None:
            save["intrinsics"] = K
        if pred.conf is not None:
            save["conf"] = np.asarray(pred.conf[0], dtype=np.float32)
        np.savez_compressed(out_npz, **save)
        done += 1
        if done % 100 == 0:
            print(f"[da3-iso] {done}/{len(frames)} depth maps", flush=True)

    print(f"[da3-iso] DONE: {done} depth maps in {args.out}", flush=True)


if __name__ == "__main__":
    main()
