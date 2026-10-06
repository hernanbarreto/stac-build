"""LoGeR inference on an ordered image list (runs in the da3 env; LoGeR's code in vendor/LoGeR).

Writes one npz: names, c2w (N,4,4) float64 camera-to-world, depth (N,h,w) float32 = camera z of
LoGeR's local point map, conf (N,h,w) float32 in 0..1, K (3,3) float64 on the (h,w) grid fitted on
the local point maps (Pi3 style: u = fx·X/Z + cx), K_per_frame (N,3,3), grid_hw.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "LoGeR"))


def load_model(ckpt: str, config: str):
    import torch
    import yaml
    from loger.models.pi3 import Pi3
    cfg = yaml.safe_load(open(config)) or {}
    mcfg = cfg.get("model", {}) or {}
    valid = {n for n, p in inspect.signature(Pi3.__init__).parameters.items()
             if n not in {"self", "args", "kwargs"}
             and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    kw = {k: mcfg[k] for k in sorted(valid) if k in mcfg}
    model = Pi3(**kw)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval(), bool(mcfg.get("se3", cfg.get("se3", False)))


def load_images(paths, W: int, H: int):
    import torch
    from PIL import Image
    from torchvision import transforms
    out = torch.empty((len(paths), 3, H, W), dtype=torch.float32)
    tt = transforms.ToTensor()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i].copy_(tt(im.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)))
    return out


def grid_for(paths, long_side: int = 504, patch: int = 14):
    """LoGeR's grid for these frames: the paper's 504 px long side, the aspect kept, multiples of 14."""
    from PIL import Image
    with Image.open(paths[0]) as im:
        W0, H0 = im.size
    if W0 >= H0:
        W = long_side
        H = max(patch, int(round(long_side * H0 / W0 / patch)) * patch)
    else:
        H = long_side
        W = max(patch, int(round(long_side * W0 / H0 / patch)) * patch)
    return W, H, W0, H0


def fit_K(local_points: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """K on the grid from one frame's local point map: u + 0.5 = fx·X/Z + cx (least squares over the
    most confident half of the pixels with Z > 0)."""
    h, w = local_points.shape[:2]
    X, Y, Z = local_points[..., 0], local_points[..., 1], local_points[..., 2]
    ok = (Z > 1e-6) & (conf >= np.median(conf))
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    x = (X / np.maximum(Z, 1e-9))[ok]; y = (Y / np.maximum(Z, 1e-9))[ok]
    fx, cx = np.linalg.lstsq(np.c_[x, np.ones_like(x)], uu[ok], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[y, np.ones_like(y)], vv[ok], rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m loger_pdit.loger_infer")
    ap.add_argument("--images", required=True, help="text file: one image path per line, in walk order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=3)
    ap.add_argument("--reset_every", type=int, default=0)
    ap.add_argument("--long_side", type=int, default=504)
    a = ap.parse_args(argv)
    import torch
    paths = [l.strip() for l in open(a.images) if l.strip()]
    W, H, W0, H0 = grid_for(paths, a.long_side)
    print(f"[loger] {len(paths)} frame(s), native {W0}x{H0} → grid {W}x{H}, window {a.window}, "
          f"overlap {a.overlap}, reset_every {a.reset_every}", flush=True)
    t0 = time.time()
    model, se3 = load_model(a.ckpt, a.config)
    model = model.to("cuda")
    imgs = load_images(paths, W, H).to("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    kw = {"window_size": a.window, "overlap_size": a.overlap, "reset_every": a.reset_every,
          "num_iterations": 1, "sim3": False, "se3": se3}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        pred = model(imgs[None], **kw)
    c2w = pred["camera_poses"][0].float().cpu().numpy().astype(np.float64)
    lp = pred["local_points"][0].float().cpu().numpy()
    conf = torch.sigmoid(pred["conf"][0].float()).cpu().numpy()
    conf = conf[..., 0] if conf.ndim == 4 else conf
    Ks = np.stack([fit_K(lp[i], conf[i]) for i in range(len(paths))])
    K = np.median(Ks, axis=0)
    np.savez(a.out, names=np.array([Path(p).name for p in paths]), c2w=c2w,
             depth=lp[..., 2].astype(np.float32), conf=conf.astype(np.float32),
             K=K, K_per_frame=Ks, grid_hw=np.array([H, W]), native_hw=np.array([H0, W0]))
    path_len = float(np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1).sum())
    print(f"[loger] done in {time.time() - t0:.1f} s, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB; K fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} "
          f"cx {K[0, 2]:.1f} cy {K[1, 2]:.1f} (grid); camera path {path_len:.2f} (LoGeR units)", flush=True)
    json.dump({"n": len(paths), "grid_hw": [H, W], "native_hw": [H0, W0], "seconds": round(time.time() - t0, 1),
               "path_length_loger_units": path_len, "K_grid": K.tolist()},
              open(str(a.out) + ".json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
