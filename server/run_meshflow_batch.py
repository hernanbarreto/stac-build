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
                    help="optional reference image (.png/.jpg/.webp) applied to ALL items")
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
    for i, ply in enumerate(plys):
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"[BATCH] item idx={i} name={ply.stem} status=starting", flush=True)
        try:
            mesh = pipeline.run(
                mesh=str(ply),
                image=args.image,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
            glb = ply.parent / f"{ply.stem}_visual.glb"
            mesh.to_trimesh().export(str(glb))
            secs = time.time() - t0
            vram = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
            _update_meta(ply.parent / "meta.json", glb.name, args, secs, vram)
            n_ok += 1
            print(f"[BATCH] item idx={i} name={ply.stem} status=done "
                  f"elapsed={secs:.1f} out={glb} vram_gb={vram:.2f}", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"[BATCH] item idx={i} name={ply.stem} status=error "
                  f"msg={e}", flush=True)
    print(f"[BATCH] done n={len(plys)} ok={n_ok}", flush=True)
    return 0 if n_ok else 1


def _update_meta(meta_path: Path, glb_name: str, args, secs: float, vram: float):
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
            "guidance_scale": float(args.guidance_scale) if args.image else None,
            "ref_image": bool(args.image),
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
