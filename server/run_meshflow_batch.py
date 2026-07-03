#!/usr/bin/env python3
"""
MeshFlow batch worker — runs in the `meshflow` conda env (torch 2.8 cu126).
Launched by run_meshflow.sh from the backend subprocess (main.py).

Loads MeshFlowPipeline ONCE, then generates one mesh per input segment PLY.
The GLB is written NEXT TO the input as ``<stem>_visual.glb`` — the
``_visual`` suffix plus ``"metric": false`` in meta.json mark the output as a
GENERATIVE visual asset, never a metric deliverable (project charter: the
metric surface comes from surface_fit; MeshFlow only replaces ShapeR's role
of pretty per-object meshes for furniture/equipment/signage).

Protocol (stdout) — IDENTICAL to what the backend's [BATCH] parser already
speaks (main.py maps status starting/done/error to UI phases):
    [BATCH] start n=<N>
    [BATCH] item idx=<i> name=<stem> status=starting
    [BATCH] item idx=<i> name=<stem> status=done elapsed=<t> out=<glb> vram_gb=<v>
    [BATCH] item idx=<i> name=<stem> status=error msg=<...>
    [BATCH] done n=<N> ok=<K>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="MeshFlow batch inference (segment PLY → visual GLB)")
    ap.add_argument("--plys", nargs="+", required=True, help="segment point clouds (.ply)")
    ap.add_argument("--model_path", default=None,
                    help="MeshFlow bundle dir (config.yaml + model.pth); "
                         "default: <repo>/vendor/meshflow/ckpt/meshflow")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=2.5,
                    help="only effective when --image is provided (CFG on visual cond)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--image", default=None,
                    help="override reference image applied to ALL items (debug); "
                         "normally each item auto-discovers '<stem>_ref.jpg' next "
                         "to its PLY — and that image is MANDATORY")
    ap.add_argument("--allow_no_image", action="store_true",
                    help="permit geometry-only generation (default: an item "
                         "without reference image FAILS — project decision)")
    ap.add_argument("--num_verts", type=int, default=None,
                    help="target vertex budget (needs use_proj_cond_on_temb in model config)")
    args = ap.parse_args()

    model_path = args.model_path
    if model_path is None:
        model_path = str(Path(__file__).resolve().parent.parent
                         / "vendor" / "meshflow" / "ckpt" / "meshflow")

    import torch
    from meshflow.pipelines import MeshFlowPipeline

    print(f"[BATCH] loading model_path={model_path} dtype={args.dtype}", flush=True)
    pipeline = MeshFlowPipeline.from_pretrained(
        model_path, device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=args.dtype, num_verts=args.num_verts)

    plys = [Path(p) for p in args.plys]
    print(f"[BATCH] start n={len(plys)}", flush=True)
    n_ok = 0
    from PIL import Image

    for i, ply in enumerate(plys):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[BATCH] item idx={i} name={ply.stem} status=starting", flush=True)
        try:
            # Reference image: per-item '<stem>_ref.jpg' (exporter writes it
            # from the best-coverage view) — MANDATORY unless --allow_no_image.
            img_path = Path(args.image) if args.image else ply.parent / f"{ply.stem}_ref.jpg"
            image = None
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
            elif not args.allow_no_image:
                raise FileNotFoundError(
                    f"reference image {img_path.name} missing — image "
                    "conditioning is mandatory (re-run the export)")

            mesh = pipeline.run(
                mesh=str(ply),
                image=image,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                disable_prog=True,   # tqdm \r output corrupts the [BATCH] protocol
            )
            tm = mesh.to_trimesh()

            # ── back to WORLD coordinates ──
            # MeshFlow normalizes the input by bounding sphere
            # (meshflow.utils.mesh._normalize_points: center = bbox centre,
            # radius = 2·max‖p−c‖, p' = (p−c)·(2/radius)) and generates in that
            # normalized space — nothing un-normalizes. Invert it with the
            # SAME formula over the input PLY so the asset lands exactly on
            # the segment it represents.
            import numpy as np
            import trimesh as _tr
            pts_in = np.asarray(_tr.load(str(ply)).vertices, dtype=np.float64)
            center = (pts_in.min(0) + pts_in.max(0)) / 2.0
            radius = float(np.linalg.norm(pts_in - center, axis=1).max() * 2.0)
            tm.vertices = np.asarray(tm.vertices, dtype=np.float64) * (radius / 2.0) + center

            glb = ply.parent / f"{ply.stem}_visual.glb"
            tm.export(str(glb))
            secs = time.time() - t0
            vram = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
            _update_meta(ply.parent / "meta.json", glb.name, args, secs, vram,
                         image_used=(img_path.name if image is not None else None),
                         world_center=center.tolist(), world_radius=radius)
            n_ok += 1
            print(f"[BATCH] item idx={i} name={ply.stem} status=done "
                  f"elapsed={secs:.1f} out={glb} vram_gb={vram:.2f}", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"[BATCH] item idx={i} name={ply.stem} status=error "
                  f"msg={e}", flush=True)
    print(f"[BATCH] done n={len(plys)} ok={n_ok}", flush=True)
    return 0 if n_ok else 1


def _update_meta(meta_path: Path, glb_name: str, args, secs: float, vram: float,
                 image_used=None, world_center=None, world_radius=None):
    """Complete the exporter's meta.json with generation facts. The metric:false
    flag is (re)asserted here so even a hand-made input folder ends labeled."""
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    meta.update({
        "method": "meshflow",
        "metric": False,
        "generative": True,
        "glb": glb_name,
        "generation": {
            "steps": int(args.steps),
            "guidance_scale": float(args.guidance_scale) if image_used else None,
            "ref_image": image_used,
            "world_bsphere": {"center": world_center, "radius": world_radius},
            "seed": int(args.seed),
            "dtype": args.dtype,
            "num_verts": args.num_verts,
            "inference_secs": round(secs, 2),
            "peak_vram_gb": round(vram, 2),
        },
    })
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    sys.exit(main())
