"""
NKSR worker — runs in the `nksr` conda env (torch 2.1 + NVIDIA NKSR).
================================================================================
cleaned_cloud.ply → NKSR implicit field → dual-marching-cubes mesh PLY.

The cloud's per-point TRACEABILITY (frame_global) gives each point its ORIGIN
CAMERA position, which NKSR consumes as the `sensor` input — its strongest
orientation signal (no normal estimation guesswork; this is exactly the LiDAR
regime the kitchen-sink model was trained for).

Deterministic contract: NKSR is a LEARNED prior — everything it outputs is
re-gated in the server-env orchestrator (faces far from the cloud are dropped)
before delivery. This script only produces the raw candidate mesh.

Called by run_nksr.sh; prints [nksr] progress lines (streamed to run.log).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _log(msg: str):
    print(f"[nksr] {msg}", flush=True)


def _load_cloud(ply_path: Path):
    """xyz + rgb + frame_global from the binary PLY (plyfile keeps all fields)."""
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = (np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255.0
           if all(c in names for c in ("red", "green", "blue"))
           else np.full((len(xyz), 3), 0.5, np.float32))
    fg = None
    for cand in ("frame_global", "scalar_frame_global"):
        if cand in names:
            fg = np.asarray(v[cand]).astype(np.int64)
            break
    return xyz, rgb, fg


def _camera_centers(output_dir: Path):
    """frame number → camera center (c2w translation) from camera_poses.txt."""
    lines = [ln for ln in (output_dir / "camera_poses.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    nums = [int(ln.split()[0]) for ln in
            (output_dir / "camera_frames.txt").read_text().splitlines() if ln.strip()]
    centers = {}
    for n, ln in zip(nums, lines):
        t = [float(x) for x in ln.split()]
        if len(t) == 16:
            T = np.array(t, np.float64).reshape(4, 4)
            centers[n] = T[:3, 3]
    return centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--output-dir", required=True,
                    help="session output/ (camera_poses.txt + camera_frames.txt)")
    ap.add_argument("--out", required=True, help="raw mesh PLY destination")
    ap.add_argument("--detail-level", type=float, default=0.0,
                    help="NKSR detail level 0..1 (vendor default 0.0; higher=finer)")
    ap.add_argument("--voxel-size", type=float, default=0.0,
                    help="explicit finest voxel size in metres (0 = use "
                         "detail-level; overrides it when set)")
    ap.add_argument("--chunk-size", type=float, default=0.0,
                    help="metres; >0 = chunked out-of-core reconstruction")
    ap.add_argument("--max-points", type=int, default=0,
                    help="optional subsample cap (0 = all points)")
    args = ap.parse_args()

    t0 = time.time()
    import torch
    import nksr
    device = torch.device("cuda")
    _log(f"NKSR {getattr(nksr, '__version__', '?')} | torch {torch.__version__} "
         f"| {torch.cuda.get_device_name(0)}")

    xyz, rgb, fg = _load_cloud(Path(args.cloud))
    _log(f"cloud: {len(xyz):,} points ({Path(args.cloud).name})")
    if args.max_points and len(xyz) > args.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), args.max_points,
                                              replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        fg = fg[idx] if fg is not None else None
        _log(f"subsampled to {len(xyz):,} points (--max-points)")

    centers = _camera_centers(Path(args.output_dir))
    if fg is None or not centers:
        raise RuntimeError(
            "nksr: the cloud has no frame_global traceability or no poses — "
            "the sensor input (origin camera per point) is REQUIRED (it is the "
            "orientation signal; without it NKSR guesses normals)")
    default_c = np.mean(np.stack(list(centers.values())), axis=0)
    lut_max = int(max(centers.keys())) + 1
    lut = np.tile(default_c, (lut_max, 1))
    for n, c in centers.items():
        lut[int(n)] = c
    sensor = lut[np.clip(fg, 0, lut_max - 1)]
    n_miss = int((~np.isin(fg, list(centers.keys()))).sum())
    if n_miss:
        _log(f"{n_miss:,} points reference frames without a pose → mean camera "
             f"center used as their sensor")

    xyz_t = torch.from_numpy(xyz).float().to(device)
    sensor_t = torch.from_numpy(sensor).float().to(device)
    rgb_t = torch.from_numpy(rgb).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")
    # sensor mode requires a normal preprocessor (vendor waymo example):
    # kNN normal estimation oriented toward each point's ORIGIN camera
    kwargs = dict(preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0))
    unscale = 1.0
    if args.chunk_size and args.chunk_size > 0:
        # CHUNKED (out-of-core) mode: the vendor path ignores voxel_size, so we
        # keep the target resolution by PRE-SCALING the scene ourselves (model
        # native voxel / target voxel) and unscaling the mesh afterwards.
        if args.voxel_size and args.voxel_size > 0:
            native = float(reconstructor.hparams.voxel_size)
            s = native / float(args.voxel_size)
            xyz_t = xyz_t * s
            sensor_t = sensor_t * s
            unscale = 1.0 / s
            kwargs["chunk_size"] = float(args.chunk_size) * s
            _log(f"chunked mode: pre-scale ×{s:.1f} (native voxel {native:g} m "
                 f"→ effective {args.voxel_size:g} m)")
        else:
            kwargs["chunk_size"] = float(args.chunk_size)
    elif args.voxel_size and args.voxel_size > 0:
        kwargs["voxel_size"] = float(args.voxel_size)
    else:
        kwargs["detail_level"] = float(args.detail_level)
    _log(f"reconstructing ({'voxel_size=%.3f' % args.voxel_size if args.voxel_size > 0 else 'detail_level=%s' % args.detail_level}"
         + (f", chunk_size={args.chunk_size}m" if args.chunk_size else "")
         + ") …")
    field = reconstructor.reconstruct(xyz_t, sensor=sensor_t, **kwargs)
    field.set_texture_field(nksr.fields.PCNNField(xyz_t, rgb_t))
    mesh = field.extract_dual_mesh()
    if unscale != 1.0:
        mesh.v = mesh.v * unscale
    V = mesh.v.detach().cpu().numpy().astype(np.float64)
    F = mesh.f.detach().cpu().numpy().astype(np.int32)
    C = (np.clip(mesh.c.detach().cpu().numpy(), 0, 1)
         if getattr(mesh, "c", None) is not None else None)
    _log(f"mesh: {len(V):,} verts / {len(F):,} tris in {time.time() - t0:.0f}s "
         f"(peak vram {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)")

    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    if C is not None and len(C) == len(V):
        m.vertex_colors = o3d.utility.Vector3dVector(C.astype(np.float64))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), m)
    _log(f"raw mesh → {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — the orchestrator needs the reason
        import traceback
        traceback.print_exc()
        print(f"[nksr] FATAL: {e}", flush=True)
        sys.exit(1)
