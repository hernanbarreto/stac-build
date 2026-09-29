"""
ShapeR Batch Inference — process N PKLs in one process to amortize model load.
==============================================================================

Loads ShapeR denoiser + VAE + InternVL text encoder once, then iterates over
all input PKLs. Reports per-PKL progress to stdout in a parseable format so the
parent worker can update the UI in real time.

Output: one .glb per input PKL, written next to the source pkl (same folder).

Stdout protocol (one event per line, prefixed):
    [BATCH] start n=<N>
    [BATCH] item idx=<i> name=<pkl_stem> status=<starting|running|done|error> [extra]
    [BATCH] done n=<N>

Run inside the `shaper` conda env. Auto-detects CUDA, falls back to CPU.

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import omegaconf
import trimesh
from scipy.spatial import cKDTree

# vendor/ShapeR is added to PYTHONPATH by run_shaper.sh
from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import (
    DummyTextFeatureExtractor,
    MemoryEfficientTextFeatureExtractor,
)
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import remove_floating_geometry


# Same presets as vendor/ShapeR/infer_shape.py.
# `cpu` previously matched `speed` (4 views, 10 steps) but produced low-quality
# output even on canonical PKLs. Now matches `balance` so CPU runs deliver the
# vendor's "recommended" quality at the cost of much longer wall time.
# (num_views, token_multiplier, num_steps, cfg_value)
#   num_views        — images fed to the model. At inference EVERY view's DINO
#                      tokens are kept (the 1024//V subsample is training-only,
#                      shaper_denoiser.forward_impl), so more views = more evidence
#                      AND more image-condition tokens. Useful up to the number of
#                      frames the object was actually seen in (the vendor pads with
#                      repeats beyond that — no new information).
#   token_multiplier — latent tokens = token_scales[6] (2048) × this → mesh detail.
#   num_steps        — flow-matching ODE steps (midpoint = 2 model evals/step).
#   cfg_value        — classifier-free guidance. -1 disables it (the vendor presets'
#                      default). >1 runs cond+uncond each step (2× evals) and pushes
#                      the mesh to ADHERE to the point cloud / images / text — the
#                      strongest quality lever the stock presets leave switched off.
PRESETS = {
    "quality": (16, 4, 50, -1.0),
    "speed":   (4,  2, 10, -1.0),
    "balance": (16, 4, 25, -1.0),
    "cpu":     (16, 4, 25, -1.0),  # full views + tokens, fewer denoise steps than quality
    # A100 "give me the best": more views, more ODE steps, CFG ON. token_multiplier
    # stays at 4 — raising it explodes the transformer sequence (latent tokens run
    # through every attention layer alongside num_views×400 image tokens).
    "max":     (32, 4, 100, 4.0),  # all views, most steps, strong CFG. token×4 kept
    #                                (raising it OOMs alongside 32×400 image tokens).
}


def emit(msg: str):
    print(f"[BATCH] {msg}", flush=True)


# ── ICP refinement ──────────────────────────────────────────────────

def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Closed-form rigid alignment (P→Q). Returns (R, t)."""
    cp = P.mean(axis=0)
    cq = Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    return R, t


def _icp_refine(mesh: trimesh.Trimesh,
                target_points: np.ndarray,
                max_iter: int = 30,
                sample_n: int = 5000,
                voxel_target_m: float = 0.005,
                tol_m: float = 1e-4) -> tuple[np.ndarray, float, int]:
    """Point-to-point ICP. Returns (T_4x4, residual_median_m, iters_run).

    `target_points` is the original sub-cloud in world coords (already aligned
    coarsely with the mesh after T_model_world). We refine the rigid 6DoF.
    """
    if len(target_points) < 100 or mesh is None or len(mesh.vertices) < 50:
        return np.eye(4), float("nan"), 0

    # Voxel-downsample target so density matches a uniform mesh sample
    if voxel_target_m > 0 and len(target_points) > 5000:
        keys = np.floor(target_points / voxel_target_m).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        target = target_points[np.sort(idx)]
    else:
        target = target_points

    # Sample the mesh surface uniformly (more stable than using vertices)
    try:
        src, _ = trimesh.sample.sample_surface(mesh, sample_n)
    except Exception:
        src = np.asarray(mesh.vertices, dtype=np.float64)
        if len(src) > sample_n:
            src = src[np.random.choice(len(src), sample_n, replace=False)]
    src = np.asarray(src, dtype=np.float64)

    tree = cKDTree(target)
    T_total = np.eye(4)
    cur = src.copy()
    prev_residual = np.inf
    iters_run = 0

    for it in range(max_iter):
        dists, nn_idx = tree.query(cur, k=1, workers=-1)
        Q = target[nn_idx]
        # Outlier rejection: drop pairs beyond 3*median
        med = np.median(dists)
        keep = dists < (3.0 * med + 1e-6)
        if keep.sum() < 50:
            break
        R, t = _kabsch(cur[keep], Q[keep])
        cur = cur @ R.T + t

        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        T_total = T_step @ T_total
        iters_run = it + 1

        residual = float(np.median(dists[keep]))
        if abs(prev_residual - residual) < tol_m:
            break
        prev_residual = residual

    # Final residual on the converged mesh
    dists, _ = tree.query(cur, k=1, workers=-1)
    return T_total, float(np.median(dists)), iters_run


# ── Coverage-aware fit-to-cloud ─────────────────────────────────────
#
# ShapeR's autoregressive prior produces meshes whose extents are independent
# of the source segment cloud — they reflect "what the model thinks a trash
# can / wall / chair looks like." After ICP, the mesh is rigidly aligned but
# can still exceed the cloud's bounding box (or float above the floor) when
# the prior disagrees with the actual scan.
#
# The fit step below honors three priors derived from the segment cloud:
#
#   1. The cloud is a SIZE prior — but only on faces that are actually
#      well-scanned. Per-face coverage in [0,1] tells us whether each side
#      of the bbox is a real boundary or an artifact of partial scanning.
#   2. The base (face opposite world-up) is a STRONG prior — gravity says
#      objects rest on the ground. Always anchor regardless of coverage.
#   3. On low-coverage (occluded) faces, ShapeR's prior fills in. We allow
#      the mesh to extend beyond the cloud bbox, capped at max_grow so the
#      generation can't run away.
#
# All operations happen in world frame, after T_model_world and T_icp are
# baked in.

# Order of faces in coverage arrays: [-x, +x, -y, +y, -z, +z]
_FACE_ORDER = ("-x", "+x", "-y", "+y", "-z", "+z")


def _face_coverage(points: np.ndarray, bb_min: np.ndarray, bb_max: np.ndarray,
                   near_thresh: float, grid: int = 8) -> np.ndarray:
    """Per-face coverage score in [0,1], measuring how much of each bbox face
    has a nearby point. Returns array of length 6, ordered as ``_FACE_ORDER``.

    For each face, project the points within ``near_thresh`` of that face onto
    the face plane, bin into a ``grid x grid`` histogram, and report the
    fraction of cells that contain at least one point. This handles "boundary
    is just outliers" correctly — a single floating point near a face has near
    zero coverage even though it exists.
    """
    coverage = np.zeros(6, dtype=np.float64)
    if len(points) == 0:
        return coverage
    eps = 1e-9
    for axis in range(3):
        u_axis = (axis + 1) % 3
        v_axis = (axis + 2) % 3
        u0, u1 = bb_min[u_axis], bb_max[u_axis]
        v0, v1 = bb_min[v_axis], bb_max[v_axis]
        u_span = max(u1 - u0, eps)
        v_span = max(v1 - v0, eps)
        for sign_idx, face_val in enumerate((bb_min[axis], bb_max[axis])):
            d = np.abs(points[:, axis] - face_val)
            mask = d < near_thresh
            if not mask.any():
                continue
            pts2 = points[mask]
            ui = np.clip(((pts2[:, u_axis] - u0) / u_span * grid).astype(int), 0, grid - 1)
            vi = np.clip(((pts2[:, v_axis] - v0) / v_span * grid).astype(int), 0, grid - 1)
            cells = np.unique(ui * grid + vi)
            coverage[2 * axis + sign_idx] = len(cells) / float(grid * grid)
    return coverage


def _world_up_from_T_zup(T_zup_obj: np.ndarray) -> tuple[int, int]:
    """Decode ``T_zup_obj`` (rotates world-up direction onto +Z) to recover
    which world-frame axis is "up", and its sign.

    Returns ``(axis, sign)`` where axis is 0/1/2 and sign is ±1.
    """
    R = T_zup_obj[:3, :3]
    # The world-up vector is whatever R takes to +Z; equivalently R.T @ +Z
    up_dir = R.T @ np.array([0.0, 0.0, 1.0])
    axis = int(np.argmax(np.abs(up_dir)))
    sign = 1 if up_dir[axis] > 0 else -1
    return axis, sign


def _fit_mesh_to_cloud(
    mesh: trimesh.Trimesh,
    cloud_pts: np.ndarray,
    up_axis: int,
    up_sign: int,
    cov_threshold: float = 0.30,
    near_thresh_frac: float = 0.10,
    max_grow_factor: float = 1.5,
) -> tuple[trimesh.Trimesh, dict]:
    """Adaptive shrink + base-anchor of the mesh AABB to honor the segment
    cloud's geometry priors.

    Args:
        mesh: trimesh.Trimesh in world frame (post-ICP).
        cloud_pts: (N,3) segment cloud in same world frame.
        up_axis, up_sign: world-up direction (decoded from T_zup_obj).
        cov_threshold: faces with coverage >= this are treated as strict bounds.
        near_thresh_frac: a point counts as "near" a face if within this
            fraction of the median cloud extent (so the threshold scales with
            object size — 1cm for a phone, 30cm for a wall).
        max_grow_factor: even on occluded (low-coverage) faces, cap the mesh
            extent at this multiple of the cloud extent on that axis. Prevents
            ShapeR's prior from running away when the cloud is sparse.

    Returns: (mesh, stats_dict). The mesh is mutated in-place.
    """
    if len(cloud_pts) < 4:
        return mesh, {"applied": False, "reason": "cloud_too_small"}

    cloud_min = cloud_pts.min(axis=0)
    cloud_max = cloud_pts.max(axis=0)
    cloud_ext = cloud_max - cloud_min

    near_thresh = max(0.02, float(np.median(cloud_ext)) * near_thresh_frac)
    coverage = _face_coverage(cloud_pts, cloud_min, cloud_max, near_thresh)

    # Base = face opposite of world-up. For up_sign>0 (e.g. +y up), base is
    # the min face on the up-axis (lowest y). For up_sign<0, base is the max.
    base_face = 2 * up_axis + (0 if up_sign > 0 else 1)
    constrained = (coverage >= cov_threshold).copy()
    constrained[base_face] = True  # gravity prior — always strict

    mesh_min = mesh.bounds[0].copy()
    mesh_max = mesh.bounds[1].copy()
    mesh_ext = mesh_max - mesh_min

    # Step 1: anisotropic shrink. Per-axis target extent comes from constrained
    # faces. If both faces of an axis are constrained, target = cloud extent.
    # If neither, target = max_grow_factor * cloud extent (loose cap). If just
    # one, we take the constrained-side distance from the centroid — but for
    # simplicity treat it the same as "loose" since translation will pin the
    # mesh later.
    eps = 1e-9
    target_ext = np.empty(3)
    for axis in range(3):
        c_min = constrained[2 * axis]
        c_max = constrained[2 * axis + 1]
        if c_min and c_max:
            target_ext[axis] = cloud_ext[axis]
        else:
            target_ext[axis] = cloud_ext[axis] * max_grow_factor
    scale = np.minimum(1.0, target_ext / np.maximum(mesh_ext, eps))

    if np.any(scale < 0.999):
        centroid = (mesh_min + mesh_max) / 2.0
        T_s = np.eye(4)
        T_s[:3, :3] = np.diag(scale)
        T_s[:3, 3] = centroid * (1.0 - scale)
        mesh.apply_transform(T_s)

    # Step 2: translate to honor positional priors (base anchor + lateral
    # constraints). Recompute bounds since we may have scaled.
    new_min = mesh.bounds[0].copy()
    new_max = mesh.bounds[1].copy()
    delta = np.zeros(3)
    for axis in range(3):
        if axis == up_axis:
            # Base anchor: pin the appropriate face exactly.
            if up_sign > 0:
                delta[axis] = cloud_min[axis] - new_min[axis]
            else:
                delta[axis] = cloud_max[axis] - new_max[axis]
        else:
            c_min = constrained[2 * axis]
            c_max = constrained[2 * axis + 1]
            if c_min and c_max:
                target_c = (cloud_min[axis] + cloud_max[axis]) / 2.0
                mesh_c = (new_min[axis] + new_max[axis]) / 2.0
                delta[axis] = target_c - mesh_c
            elif c_min:
                delta[axis] = cloud_min[axis] - new_min[axis]
            elif c_max:
                delta[axis] = cloud_max[axis] - new_max[axis]
            # else: no lateral constraint — leave as ICP placed it

    if np.any(np.abs(delta) > 1e-6):
        T_t = np.eye(4)
        T_t[:3, 3] = delta
        mesh.apply_transform(T_t)

    final_ext = mesh.bounds[1] - mesh.bounds[0]
    return mesh, {
        "applied": bool(np.any(scale < 0.999) or np.any(np.abs(delta) > 1e-6)),
        "world_up_axis": int(up_axis),
        "world_up_sign": int(up_sign),
        "base_face": _FACE_ORDER[base_face],
        "coverage_per_face": [round(float(c), 3) for c in coverage],
        "coverage_face_order": list(_FACE_ORDER),
        "cov_threshold": float(cov_threshold),
        "near_thresh_m": round(float(near_thresh), 4),
        "max_grow_factor": float(max_grow_factor),
        "constrained_faces": [bool(c) for c in constrained],
        "scale_per_axis": [round(float(s), 4) for s in scale],
        "translate_m": [round(float(d), 4) for d in delta],
        "cloud_bbox_min": [round(float(v), 4) for v in cloud_min],
        "cloud_bbox_max": [round(float(v), 4) for v in cloud_max],
        "mesh_extent_before": [round(float(v), 4) for v in mesh_ext],
        "mesh_extent_after": [round(float(v), 4) for v in final_ext],
    }


# ── Sidecar metadata ────────────────────────────────────────────────

def _write_meta(meta_path: Path, pkl_sample: dict, glb_path: Path,
                T_model_world: np.ndarray, T_icp: np.ndarray,
                icp_residual_m: float, icp_iters: int,
                preset: str, num_views: int, num_steps: int,
                elapsed_s: float,
                fit_stats: Optional[dict] = None):
    """Write per-mesh metadata for auditability + UI loading."""
    # T_model_world maps world(Y-up) → model_zup (rotation + translation).
    # Recover world centroid + AABB by inverting and projecting points.
    T_world_from_model = np.linalg.inv(T_model_world)
    centroid_world = T_world_from_model[:3, 3].tolist()
    pts_m = pkl_sample["points_model"].numpy().astype(np.float64)
    pts_h = np.concatenate(
        [pts_m, np.ones((len(pts_m), 1), dtype=np.float64)], axis=1
    )
    pts_world = (T_world_from_model @ pts_h.T).T[:, :3]
    bb_min = pts_world.min(axis=0).tolist()
    bb_max = pts_world.max(axis=0).tolist()

    meta = {
        "schema_version": 1,
        "instance_id": int(pkl_sample.get("instance_id", pkl_sample.get("global_id", -1))),
        "label": pkl_sample.get("label", ""),
        "caption": pkl_sample.get("caption", ""),
        "glb_file": glb_path.name,
        "centroid_world": centroid_world,
        "bbox_min_world": bb_min,
        "bbox_max_world": bb_max,
        "T_model_world": T_model_world.tolist(),     # centering applied first
        "T_icp_refine": T_icp.tolist(),              # 6DoF refinement applied second
        "icp_residual_m": None if not np.isfinite(icp_residual_m) else float(icp_residual_m),
        "icp_iterations": int(icp_iters),
        "fit_to_cloud": fit_stats if fit_stats is not None else {"applied": False},
        "n_source_points": int(pkl_sample.get("n_source_points", 0)),
        "n_views": int(pkl_sample.get("n_views", 0)),
        "source_frames": pkl_sample.get("source_frames", []),
        "shaper_preset": preset,
        "shaper_num_views": int(num_views),
        "shaper_num_steps": int(num_steps),
        "elapsed_s": round(float(elapsed_s), 2),
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser("ShapeR Batch Inference")
    ap.add_argument("--pkls", nargs="+", required=True,
                    help="List of input .pkl paths (absolute).")
    ap.add_argument("--output_dir", required=True,
                    help="Optional output dir override. If omitted, .glb is written next to the .pkl.")
    ap.add_argument("--config", default="balance",
                    help="quality | balance | speed | cpu | max")
    ap.add_argument("--num_views", type=int, default=None,
                    help="Override the preset's view count (images fed to the model). "
                         "Useful up to the number of frames the object was seen in.")
    ap.add_argument("--num_steps", type=int, default=None,
                    help="Override the preset's flow-matching ODE steps.")
    ap.add_argument("--token_multiplier", type=int, default=None,
                    help="Override the preset's latent-token multiplier (mesh detail). "
                         "Raising it is the most memory-hungry knob.")
    ap.add_argument("--cfg_value", type=float, default=None,
                    help="Override classifier-free guidance. -1 disables it; values "
                         "~2-5 push the mesh to follow the point cloud / images / text.")
    ap.add_argument("--no_remove_floating", action="store_true")
    ap.add_argument("--no_simplify", action="store_true")
    ap.add_argument("--use_text", action="store_true", default=True)
    ap.add_argument("--ckpt_root", default=None,
                    help="Override path to vendor/ShapeR (where checkpoints/ lives).")
    ap.add_argument("--no_fit_to_cloud", action="store_true",
                    help="Skip the coverage-aware fit-to-cloud step. By default "
                         "the mesh is shrunk anisotropically on well-scanned faces "
                         "(coverage >= --fit_cov_threshold) and its base is "
                         "anchored to the cloud bbox along world-up.")
    ap.add_argument("--fit_cov_threshold", type=float, default=0.30,
                    help="Per-face coverage in [0,1] above which a bbox face is "
                         "treated as a hard upper bound. Lower = more aggressive "
                         "clamp; higher = more permissive (default: 0.30).")
    ap.add_argument("--fit_max_grow", type=float, default=1.5,
                    help="Cap on mesh extent for axes where both faces are "
                         "below the coverage threshold (occluded). Allows ShapeR "
                         "to extend beyond the cloud where the scan is sparse, "
                         "but never more than this multiple (default: 1.5).")
    args = ap.parse_args()

    pkl_paths = [Path(p).resolve() for p in args.pkls]
    for p in pkl_paths:
        if not p.exists():
            emit(f"item idx=-1 name={p.stem} status=error msg=missing_pkl")
            sys.exit(2)

    n = len(pkl_paths)
    emit(f"start n={n}")

    # Auto-detect device — GPU may report available but be broken
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            torch.zeros(1).cuda()  # smoke test — fails fast on broken HW
        except Exception as e:
            emit(f"warn cuda_unusable={type(e).__name__}")
            use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    emit(f"device={device.type}")

    # On CPU, `balance` is downgraded to `cpu` (identical numbers, clearer name);
    # `quality` is honoured — slower (50 denoise steps vs 25) but it's the best
    # the released ckpt can give. `speed` stays `speed`.
    cfg_name = args.config
    if device.type == "cpu" and cfg_name == "balance":
        cfg_name = "cpu"
    num_views, token_multiplier, num_steps, cfg_value = PRESETS[cfg_name]
    # Per-run overrides (so views/steps/CFG/detail can be pushed without editing
    # the preset table). None = keep the preset value.
    if args.num_views is not None:
        num_views = int(args.num_views)
    if args.num_steps is not None:
        num_steps = int(args.num_steps)
    if args.token_multiplier is not None:
        token_multiplier = int(args.token_multiplier)
    if args.cfg_value is not None:
        cfg_value = float(args.cfg_value)
    emit(f"preset={cfg_name} views={num_views} tokens_x={token_multiplier} "
         f"steps={num_steps} cfg={cfg_value}")

    # ── Resolve checkpoints ──
    if args.ckpt_root:
        os.chdir(args.ckpt_root)  # vendor expects relative paths from its root

    setup_checkpoints()
    yaml_file = "checkpoints/config.yaml"
    config = omegaconf.OmegaConf.load(yaml_file)

    # ── Build dataset over all PKLs ──
    dataset = InferenceDataset(config, paths=[str(p) for p in pkl_paths],
                               override_num_views=num_views)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=dataset.custom_collate,
    )

    # ── Pre-compute all text embeddings, then free T5+CLIP ──
    if args.use_text:
        captions = []
        for i in range(len(dataset)):
            captions.append(dataset[i].get("caption", "a 3D object"))
        text_extractor = MemoryEfficientTextFeatureExtractor(device=device)
        text_extractor.precompute_and_free(captions)
        emit(f"text_embeddings_ready n={len(captions)}")
    else:
        text_extractor = DummyTextFeatureExtractor(device=device)

    # ── Load denoiser + VAE ──
    emit("loading_models")
    state_dict = torch.load("checkpoints/019-0-bfloat16.ckpt",
                            map_location=device, weights_only=False)
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)

    if device.type == "cuda":
        model = torch.compile(model, fullgraph=True)
    model = model.eval()

    vae = MichelangeloLikeAutoencoderWrapper(
        "checkpoints/vae-088-0-bfloat16.ckpt", device,
    )
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())

    use_shifted_sampling = (
        getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"
    )
    emit("models_ready")

    # ── Inference loop ──
    output_root = Path(args.output_dir).resolve() if args.output_dir else None
    success = 0
    t_total = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            pkl_path = pkl_paths[idx]
            name = pkl_path.stem
            t0 = time.time()
            emit(f"item idx={idx} name={name} status=starting "
                 f"views={num_views} steps={num_steps}")

            try:
                batch = InferenceDataset.move_batch_to_device(
                    batch, device, dtype=torch.bfloat16,
                )
                latents = model.infer_latents(
                    batch,
                    token_shape=token_shape,
                    text_feature_extractor=text_extractor,
                    num_steps=num_steps,
                    cfg_value=cfg_value,
                    use_shifted_sampling=use_shifted_sampling,
                )
                mesh = vae.infer_mesh_from_latents(latents)[0]

                if not args.no_remove_floating:
                    mesh = remove_floating_geometry(mesh)
                if not args.no_simplify:
                    try:
                        mesh = mesh.simplify_quadric_decimation(face_count=125000)
                    except Exception as e:
                        emit(f"item idx={idx} name={name} status=warn msg=simplify_failed:{e}")

                # Place the mesh in world coords (same frame as cleaned_cloud.ply).
                # rescale_back(do_transform_to_world=True) un-normalizes and applies
                # T_model_world.inverse() so the mesh sits on the original sub-cloud.
                mesh = dataset.rescale_back(idx, mesh, do_transform_to_world=True)

                # ── ICP fine refinement against the original sub-cloud ──
                # points_model is in the centered Z-up model frame; invert
                # T_model_world (which now encodes both centering and the
                # Y-down→Z-up rotation) to recover world-coord points.
                pkl_sample = pickle.load(open(pkl_paths[idx], "rb"))
                T_model_world = pkl_sample["T_model_world"].numpy().astype(np.float64)
                T_world_from_model = np.linalg.inv(T_model_world)
                pts_m = pkl_sample["points_model"].numpy().astype(np.float64)
                pts_h = np.concatenate(
                    [pts_m, np.ones((len(pts_m), 1), dtype=np.float64)], axis=1
                )
                sub_cloud_world = (T_world_from_model @ pts_h.T).T[:, :3]

                T_icp, icp_residual, icp_iters = _icp_refine(mesh, sub_cloud_world)
                if icp_iters > 0 and np.isfinite(icp_residual):
                    if icp_residual < 0.30:  # accept refinement up to 30cm residual
                        mesh.apply_transform(T_icp)
                        emit(f"item idx={idx} name={name} status=icp_ok "
                             f"residual_m={icp_residual:.4f} iters={icp_iters}")
                    else:
                        emit(f"item idx={idx} name={name} status=icp_skip "
                             f"residual_m={icp_residual:.4f} (too high — keeping coarse pose)")
                        T_icp = np.eye(4)
                else:
                    emit(f"item idx={idx} name={name} status=icp_skip reason=no_iters")
                    T_icp = np.eye(4)

                # ── Coverage-aware fit-to-cloud ──
                # Honor the segment cloud as a size + base-position prior, but
                # only on faces where the scan is dense enough to trust. The
                # base (face opposite world-up) is always pinned — no floating.
                fit_stats: dict = {"applied": False, "skipped": True}
                if not args.no_fit_to_cloud:
                    try:
                        T_zup_obj = pkl_sample["T_zup_obj"].numpy().astype(np.float64)
                        up_axis, up_sign = _world_up_from_T_zup(T_zup_obj)
                        mesh, fit_stats = _fit_mesh_to_cloud(
                            mesh, sub_cloud_world,
                            up_axis=up_axis, up_sign=up_sign,
                            cov_threshold=args.fit_cov_threshold,
                            max_grow_factor=args.fit_max_grow,
                        )
                        if fit_stats.get("applied"):
                            emit(f"item idx={idx} name={name} status=fit_applied "
                                 f"scale={fit_stats['scale_per_axis']} "
                                 f"translate={fit_stats['translate_m']} "
                                 f"cov={fit_stats['coverage_per_face']}")
                        else:
                            emit(f"item idx={idx} name={name} status=fit_noop")
                    except Exception as e:
                        emit(f"item idx={idx} name={name} status=fit_error msg={e}")
                        fit_stats = {"applied": False, "error": str(e)}

                # Output: same folder as the input PKL
                if output_root is not None:
                    out_dir = output_root / pkl_path.parent.name
                    out_dir.mkdir(parents=True, exist_ok=True)
                else:
                    out_dir = pkl_path.parent

                tmp_obj = out_dir / f".{name}.tmp.obj"
                mesh.export(str(tmp_obj))
                glb_mesh = trimesh.load(str(tmp_obj), force="mesh")
                out_path = out_dir / f"{name}.glb"
                glb_mesh.export(str(out_path), include_normals=True)
                tmp_obj.unlink(missing_ok=True)

                elapsed = time.time() - t0

                # Sidecar metadata for auditability + UI auto-load
                meta_path = out_dir / f"{name}.meta.json"
                _write_meta(
                    meta_path, pkl_sample, out_path,
                    T_model_world=T_model_world,
                    T_icp=T_icp,
                    icp_residual_m=icp_residual,
                    icp_iters=icp_iters,
                    preset=cfg_name,
                    num_views=num_views,
                    num_steps=num_steps,
                    elapsed_s=elapsed,
                    fit_stats=fit_stats,
                )

                emit(f"item idx={idx} name={name} status=done "
                     f"elapsed={elapsed:.1f} out={out_path} meta={meta_path.name}")
                success += 1

            except Exception as e:
                tb = traceback.format_exc().replace("\n", " | ")[-300:]
                emit(f"item idx={idx} name={name} status=error msg={e} tb={tb}")
                continue

    emit(f"done n={n} success={success} elapsed={time.time()-t_total:.1f}")


if __name__ == "__main__":
    main()
