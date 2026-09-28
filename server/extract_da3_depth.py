#!/usr/bin/env python3
"""
Simple script to extract DA3 Giant depth maps to NumPy arrays.
Designed to run in the `da3` conda environment to avoid VGGT-Long dependency clashes.
"""
import argparse
import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image

def main():
    parser = argparse.ArgumentParser("Extract DA3 relative depth to NPY")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--per_frame", action="store_true",
                        help="Run inference one image at a time (ISOLATED monocular depth "
                             "— no cross-frame attention). Used for the metric scale "
                             "anchor, where frames are seconds apart and must not be "
                             "treated as a multi-view set.")
    parser.add_argument("--process_res", type=int, default=None,
                        help="DA3 processing resolution (upper-bound long side; model "
                             "default 504). Phase C detail transfer uses ~1008 so the "
                             "depth carries detail above the omega grid's Nyquist. "
                             "ViT cost grows ~quadratically — keyframes only.")
    parser.add_argument("--windows_json", type=str, default=None,
                        help="MULTI-VIEW WINDOWS (intake I3, claude_stac.txt §4-F2): a JSON "
                             "{'windows': [[image path, ...], ...]}; each window is ONE joint "
                             "inference and lands in <output_dir>/window_<i:04d>.npz (frames, "
                             "depth, conf, extrinsics w2c, intrinsics, scale_factor, "
                             "is_metric). Existing window files are kept (resume).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.windows_json:
        return run_windows(args)

    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) + 
                    glob.glob(os.path.join(args.image_dir, "*.png")))

    # Skip entirely if all outputs already exist (before loading model)
    missing = []
    for img_path in images:
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")
        if not os.path.exists(depth_path) or not os.path.exists(conf_path):
            missing.append(img_path)

    if not missing:
        print(f"[DA3 Extractor] All {len(images)} depth+conf maps already exist. Skipping.")
        return

    print(f"[DA3 Extractor] {len(missing)} of {len(images)} need processing")

    # Only now load the heavy model
    import sys
    da3_src = os.path.join(os.path.dirname(__file__), "../vendor/depth-anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
        
    from depth_anything_3.api import DepthAnything3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DA3 Extractor] Loading {args.model} on {device}")
    # PyTorchModelHubMixin.from_pretrained SILENTLY IGNORES a `device=` kwarg —
    # the model stayed on CPU (23 cores pinned, minutes per frame, GPU at 0%).
    # Move it explicitly and verify, so this can never regress quietly.
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    p = next(model.parameters())
    print(f"[DA3 Extractor] model device: {p.device} (dtype {p.dtype})")
    if device.type == "cuda" and p.device.type != "cuda":
        raise RuntimeError("model did not reach the GPU — aborting instead of "
                           "silently burning CPU")

    # Optional processing-resolution override (Phase C hi-res detail source)
    _res_kw = {}
    if args.process_res:
        _res_kw["process_res"] = int(args.process_res)
        print(f"[DA3 Extractor] process_res override: {args.process_res}")

    # Run inference: one joint batch (default) or strictly per-frame (--per_frame)
    if args.per_frame:
        d_list, c_list, k_list = [], [], []
        with torch.no_grad():
            for i, img_path in enumerate(images):
                print(f"[DA3 Extractor] isolated inference {i+1}/{len(images)}: "
                      f"{os.path.basename(img_path)}")
                pred = model.inference([img_path], **_res_kw)
                d = pred.depth
                c = pred.conf
                k = getattr(pred, "intrinsics", None)
                d_list.append(d.cpu().numpy() if isinstance(d, torch.Tensor) else np.asarray(d))
                c_list.append(c.cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c))
                if k is not None:
                    k_list.append(k.cpu().numpy() if isinstance(k, torch.Tensor) else np.asarray(k))
        depths = np.concatenate(d_list, axis=0)
        confs = np.concatenate(c_list, axis=0)
        intrinsics = np.concatenate(k_list, axis=0) if len(k_list) == len(images) else None
    else:
        with torch.no_grad():
            prediction = model.inference(images, **_res_kw)
        # prediction.depth has shape [N, H, W], prediction.conf has shape [N, H, W]
        depths = prediction.depth
        confs = prediction.conf
        intrinsics = getattr(prediction, "intrinsics", None)
        if isinstance(depths, torch.Tensor):
            depths = depths.cpu().numpy()
        if isinstance(confs, torch.Tensor):
            confs = confs.cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()

    # DA3 conf uses expp1 activation (exp(x)+1), range ~1-60+
    # Subtract 1.0 so minimum is 0 (same as DA3-streaming does)
    confs = confs - 1.0
    confs = np.clip(confs, 0, None)

    print(f"[DA3 Extractor] Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
    print(f"[DA3 Extractor] Conf range:  [{confs.min():.3f}, {confs.max():.3f}]")
    if device.type == "cuda":
        print(f"[DA3 Extractor] peak VRAM: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    os.makedirs(args.output_dir, exist_ok=True)   # own your output directory
    for i, img_path in enumerate(images):
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]
        depth_path = os.path.join(args.output_dir, stem + "_depth.npy")
        conf_path = os.path.join(args.output_dir, stem + "_conf.npy")

        if os.path.exists(depth_path) and os.path.exists(conf_path):
            continue

        print(f"[{i+1}/{len(images)}] {basename} → depth + conf")
        np.save(depth_path, depths[i])
        np.save(conf_path, confs[i])
        if intrinsics is not None:
            # per-frame predicted K — dense_pose_fusion unprojects DA3 depth with it
            np.save(os.path.join(args.output_dir, stem + "_intrinsics.npy"), intrinsics[i])

    print("[DA3 Extractor] Finished successfully.")

def _np(x):
    if x is None:
        return None
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def run_windows(args):
    """One joint DA3 inference per window. The NESTED model aligns its multi-view
    depth to its own metric branch with one least-squares factor per window and
    scales the extrinsics' translations by the same factor (model/da3.py
    _apply_depth_alignment: is_metric = 1), so each window's poses are metric."""
    import json
    import sys
    import time
    with open(args.windows_json) as f:
        windows = json.load(f)["windows"]
    todo = [i for i in range(len(windows))
            if not os.path.exists(os.path.join(args.output_dir, f"window_{i:04d}.npz"))]
    if not todo:
        print(f"[DA3 windows] all {len(windows)} windows already exist. Skipping.")
        return
    print(f"[DA3 windows] {len(todo)} of {len(windows)} windows need processing")
    da3_src = os.path.join(os.path.dirname(__file__), "../vendor/depth-anything-3/src")
    if da3_src not in sys.path:
        sys.path.insert(0, da3_src)
    from depth_anything_3.api import DepthAnything3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DepthAnything3.from_pretrained(args.model).to(device)
    model.eval()
    if device.type == "cuda" and next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model did not reach the GPU — aborting instead of "
                           "silently burning CPU")
    res_kw = {"process_res": int(args.process_res)} if args.process_res else {}
    t0 = time.time()
    for n, i in enumerate(todo):
        paths = windows[i]
        with torch.no_grad():
            pred = model.inference(paths, **res_kw)
        frames = np.array([int("".join(ch for ch in os.path.splitext(os.path.basename(p))[0]
                                       if ch.isdigit())) for p in paths], dtype=np.int64)
        conf = np.clip(_np(pred.conf) - 1.0, 0, None)       # expp1 activation, as above
        ext = _np(pred.extrinsics)
        out = os.path.join(args.output_dir, f"window_{i:04d}.npz")
        tmp = out + ".tmp.npz"
        np.savez(tmp, frames=frames, depth=_np(pred.depth).astype(np.float32),
                 conf=conf.astype(np.float32), extrinsics=ext.astype(np.float64),
                 intrinsics=_np(pred.intrinsics).astype(np.float64),
                 scale_factor=np.float64(pred.scale_factor if pred.scale_factor is not None
                                         else np.nan),
                 is_metric=np.int64(pred.is_metric))
        os.replace(tmp, out)
        el = time.time() - t0
        print(f"[DA3 windows] window {n + 1}/{len(todo)} ({len(paths)} frames, "
              f"is_metric={int(pred.is_metric)}, scale {pred.scale_factor}) — "
              f"{el:.0f}s, ~{el / (n + 1) * (len(todo) - n - 1):.0f}s left", flush=True)
    if device.type == "cuda":
        print(f"[DA3 windows] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("[DA3 windows] Finished successfully.")


if __name__ == "__main__":
    main()
