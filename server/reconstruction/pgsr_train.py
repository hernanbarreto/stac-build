#!/usr/bin/env python3
"""
PGSR trainer — STAC adaptation (precision task, Phase D).
================================================================================
Runs the PGSR per-scene photometric optimization (planar-regularized Gaussian
surface reconstruction, vendor/pgsr @ de24f1a) on a scene exported by
reconstruction/pgsr_export.py, in the `pgsr` conda env. Forked from the vendor
train.py (Inria research license, credits in vendor/pgsr/LICENSE.md) with the
STAC-specific behaviour the vendor loop does not have:

  1. DYNAMIC-CLASS MASK EXCLUSION — masks/<image>.png (255 = dynamic) are
     excluded from the photometric L1/SSIM, the single-view normal loss and the
     multi-view NCC sampling (people / mobile equipment must not sculpt the
     scene surface).
  2. OPTIONAL PHOTOMETRIC POSE REFINEMENT (--pose_refine, default off — A/B
     decides): a per-camera se(3) delta, applied DIFFERENTIABLY to the Gaussian
     means/rotations before rasterization (world-side transform ≡ camera-side
     correction; the rasterizer backpropagates through means3D, so no CUDA
     changes). This is the dense-photometric pose path that replaces the failed
     track-based BA. Refined poses are exported NEXT TO the rendered depths —
     the pipeline's own camera_poses.txt is never touched.
  3. RENDERED-DEPTH EXPORT — after training, every camera's plane_depth is
     written to output/pgsr_render/frame_<num>.npz (+ poses_refined.txt when
     pose refinement ran, + report.json with time / peak VRAM / PSNR / points).
     The existing TSDF stage consumes these via depth_source: "pgsr_render".

Vendor modules are imported from STAC_PGSR_VENDOR (default vendor/pgsr).
Usage (pgsr env):
    python -m reconstruction.pgsr_train --scene <output>/pgsr_scene \
        --model_dir <output>/pgsr_model --render_dir <output>/pgsr_render \
        [--iterations 30000] [--pose_refine] [--quick]
"""
from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from random import randint

import numpy as np

_SERVER = Path(__file__).resolve().parent.parent
_VENDOR = Path(os.environ.get("STAC_PGSR_VENDOR",
                              _SERVER.parent / "vendor" / "pgsr"))
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
if str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))


# ── se(3) helpers (differentiable, small-delta regime) ────────────────────────

def se3_exp(phi_rho):
    """(N,6) [rot_vec | trans] → (N,3,3) R, (N,3) t. Rodrigues, differentiable.
    Small-delta refinement: t is used directly (no V matrix — standard)."""
    import torch
    phi, rho = phi_rho[:, :3], phi_rho[:, 3:]
    theta = phi.norm(dim=1, keepdim=True).clamp(min=1e-12)
    axis = phi / theta
    K = torch.zeros(phi.shape[0], 3, 3, device=phi.device, dtype=phi.dtype)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    st = torch.sin(theta)[..., None]
    ct = torch.cos(theta)[..., None]
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype)[None]
    R = eye + st * K + (1 - ct) * (K @ K)
    return R, rho


def quat_mul(q1, q2):
    """Hamilton product, (…,4) wxyz — differentiable."""
    import torch
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rotmat_to_quat_t(R):
    """(3,3) torch → (4,) wxyz (no grad needed — used once per render)."""
    import torch
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = torch.sqrt(tr + 1.0) * 2
        return torch.stack([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                            (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(torch.argmax(torch.diagonal(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = torch.sqrt((m[i, i] - m[j, j] - m[k, k] + 1.0).clamp(min=1e-12)) * 2
    q = [None] * 4
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return torch.stack(q)


class PosedGaussians:
    """Differentiable world-side pose delta around a GaussianModel: means and
    rotations transformed by (R_d, t_d); everything else delegated. Rendering
    world' = D·world through the fixed camera V equals rendering the original
    world through V·D — a camera correction c2w' = D⁻¹·c2w."""

    def __init__(self, gaussians, R_d, t_d):
        object.__setattr__(self, "_g", gaussians)
        object.__setattr__(self, "_R", R_d)
        object.__setattr__(self, "_t", t_d)
        object.__setattr__(self, "_qd", None)

    @property
    def get_xyz(self):
        return self._g.get_xyz @ self._R.T + self._t

    @property
    def get_rotation(self):
        qd = self._qd
        if qd is None:
            qd = rotmat_to_quat_t(self._R)
            object.__setattr__(self, "_qd", qd)
        q = self._g.get_rotation
        return quat_mul(qd.expand_as(q), q)

    def __getattr__(self, name):
        return getattr(self._g, name)


# ── mask handling ─────────────────────────────────────────────────────────────

def load_keep_masks(mask_dir, cameras, device):
    """image_name → float tensor (1,H,W): 1 keep / 0 excluded (dynamic)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    keep = {}
    if not mask_dir or not Path(mask_dir).exists():
        return keep
    for cam in cameras:
        p = Path(mask_dir) / f"{cam.image_name}.png"
        if not p.exists():
            continue
        m = np.asarray(Image.open(p).convert("L"), np.float32) / 255.0
        t = torch.from_numpy(m)[None, None].to(device)
        t = F.interpolate(t, size=(cam.image_height, cam.image_width),
                          mode="nearest")[0]
        keep[cam.image_name] = (t < 0.5).float()          # 255 = dynamic = drop
    return keep


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torch.nn.functional as F
    import cv2  # noqa: F401 — vendor utils expect it importable

    parser = ArgumentParser(description="STAC PGSR trainer")
    from arguments import ModelParams, PipelineParams, OptimizationParams
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--scene", required=True, help="pgsr_scene dir (exported)")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--pose_refine", action="store_true")
    parser.add_argument("--pose_refine_from_iter", type=int, default=7000)
    parser.add_argument("--pose_lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true",
                        help="15k iterations (half schedule) for A/B turnaround")
    args = parser.parse_args()

    args.source_path = str(Path(args.scene).resolve())
    args.model_path = str(Path(args.model_dir).resolve())
    args.images = "images"
    args.resolution = 1                    # NATIVE resolution — the whole point
    if args.quick:
        args.iterations = 15000
        for k in ("position_lr_max_steps", "densify_until_iter"):
            setattr(args, k, min(getattr(args, k), 15000 if k.endswith("steps")
                                 else 9000))
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)

    from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
    from utils.graphics_utils import patch_offsets, patch_warp
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from scene.app_model import AppModel
    from utils.general_utils import safe_state
    from utils.image_utils import psnr
    import random as _random

    safe_state(False)   # True SILENCES stdout (vendor wrapper) — we stream progress
    os.makedirs(dataset.model_path, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    cams = scene.getTrainCameras()
    keep_masks = load_keep_masks(args.mask_dir or
                                 (Path(args.scene) / "masks"), cams, "cuda")
    print(f"[stac-pgsr] {len(cams)} cameras, {len(keep_masks)} dynamic masks, "
          f"pose_refine={'ON' if args.pose_refine else 'off'}, "
          f"iterations={opt.iterations}", flush=True)

    # per-camera se(3) deltas (identity unless --pose_refine)
    cam_index = {c.image_name: i for i, c in enumerate(cams)}
    pose_delta = torch.zeros(len(cams), 6, device="cuda", requires_grad=bool(args.pose_refine))
    pose_opt = (torch.optim.Adam([pose_delta], lr=args.pose_lr)
                if args.pose_refine else None)

    def posed(g, cam):
        if not args.pose_refine:
            return g
        i = cam_index.get(cam.image_name)
        if i is None:
            return g
        R_d, t_d = se3_exp(pose_delta[i: i + 1])
        return PosedGaussians(g, R_d[0], t_d[0])

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema = 0.0
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image, gt_gray = cam.get_image()
        keep = keep_masks.get(cam.image_name)

        do_pose = args.pose_refine and iteration > args.pose_refine_from_iter
        g_render = posed(gaussians, cam) if do_pose else gaussians
        ret_plane = iteration > opt.single_view_weight_from_iter
        render_pkg = render(cam, g_render, pipe, bg, app_model=app_model,
                            return_plane=ret_plane, return_depth_normal=ret_plane)
        image = render_pkg["render"]
        visibility_filter, radii = render_pkg["visibility_filter"], render_pkg["radii"]

        # photometric loss (dynamic pixels excluded on BOTH sides)
        if keep is not None:
            image_l, gt_l = image * keep, gt_image * keep
        else:
            image_l, gt_l = image, gt_image
        ssim_loss = 1.0 - ssim(image_l, gt_l)
        if "app_image" in render_pkg and ssim_loss < 0.5:
            app_im = render_pkg["app_image"]
            Ll1 = l1_loss(app_im * keep if keep is not None else app_im, gt_l)
        else:
            Ll1 = l1_loss(image_l, gt_l)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            loss = loss + opt.scale_loss_weight * torch.sort(scale, -1)[0][..., 0].mean()

        # single-view planar/normal regularization
        if ret_plane and "rendered_normal" in render_pkg:
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            w_img = (1.0 - get_img_grad_weight(gt_image)).clamp(0, 1).detach() ** 2
            if keep is not None:
                w_img = w_img * keep[0]
            loss = loss + opt.single_view_weight * (
                w_img * (depth_normal - normal).abs().sum(0)).mean()

        # multi-view geometric + NCC regularization (vendor logic, mask-aware)
        if iteration > opt.multi_view_weight_from_iter and len(cam.nearest_id):
            nearest_cam = cams[_random.sample(cam.nearest_id, 1)[0]]
            patch_size = opt.multi_view_patch_size
            total_patch_size = (patch_size * 2 + 1) ** 2
            H, W = render_pkg["plane_depth"].squeeze().shape
            ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
            pixels = torch.stack([ix, iy], -1).float().cuda()
            g_near = posed(gaussians, nearest_cam) if do_pose else gaussians
            near_pkg = render(nearest_cam, g_near, pipe, bg, app_model=app_model,
                              return_plane=True, return_depth_normal=False)
            pts = gaussians.get_points_from_depth(cam, render_pkg["plane_depth"])
            pts_near = pts @ nearest_cam.world_view_transform[:3, :3] \
                + nearest_cam.world_view_transform[3, :3]
            map_z, d_mask = gaussians.get_points_depth_in_depth_map(
                nearest_cam, near_pkg["plane_depth"], pts_near)
            pts_near = pts_near / pts_near[:, 2:3] * map_z.squeeze()[..., None]
            Rn = torch.tensor(nearest_cam.R).float().cuda()
            Tn = torch.tensor(nearest_cam.T).float().cuda()
            pts_ = (pts_near - Tn) @ Rn.transpose(-1, -2)
            pts_view = pts_ @ cam.world_view_transform[:3, :3] \
                + cam.world_view_transform[3, :3]
            proj = torch.stack([pts_view[:, 0] * cam.Fx / pts_view[:, 2] + cam.Cx,
                                pts_view[:, 1] * cam.Fy / pts_view[:, 2] + cam.Cy],
                               -1).float()
            pixel_noise = torch.norm(proj - pixels.reshape(*proj.shape), dim=-1)
            d_mask = d_mask & (pixel_noise < opt.multi_view_pixel_noise_th)
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0
            if keep is not None:                       # dynamic pixels don't vote
                weights = weights * keep[0].reshape(-1)
            if d_mask.sum() > 0:
                loss = loss + opt.multi_view_geo_weight * (
                    (weights * pixel_noise)[d_mask]).mean()
                with torch.no_grad():
                    d_flat = d_mask.reshape(-1) & (weights.reshape(-1) > 0)
                    valid_indices = torch.arange(d_flat.shape[0],
                                                 device=d_flat.device)[d_flat]
                    if d_flat.sum() > opt.multi_view_sample_num:
                        idx = np.random.choice(int(d_flat.sum().cpu()),
                                               opt.multi_view_sample_num,
                                               replace=False)
                        valid_indices = valid_indices[idx]
                    w_s = weights.reshape(-1)[valid_indices]
                    px = pixels.reshape(-1, 2)[valid_indices]
                    offsets = patch_offsets(patch_size, px.device)
                    ori_patch = px.reshape(-1, 1, 2) / cam.ncc_scale + offsets.float()
                    Hg, Wg = gt_gray.squeeze().shape
                    pp_ = ori_patch.clone()
                    pp_[:, :, 0] = 2 * pp_[:, :, 0] / (Wg - 1) - 1
                    pp_[:, :, 1] = 2 * pp_[:, :, 1] / (Hg - 1) - 1
                    ref_gray = F.grid_sample(gt_gray.unsqueeze(1),
                                             pp_.view(1, -1, 1, 2),
                                             align_corners=True)
                    ref_gray = ref_gray.reshape(-1, total_patch_size)
                    r2n_r = nearest_cam.world_view_transform[:3, :3].transpose(-1, -2) \
                        @ cam.world_view_transform[:3, :3]
                    r2n_t = -r2n_r @ cam.world_view_transform[3, :3] \
                        + nearest_cam.world_view_transform[3, :3]
                if valid_indices.numel() > 0:
                    n_loc = render_pkg["rendered_normal"].permute(1, 2, 0) \
                        .reshape(-1, 3)[valid_indices]
                    d_loc = render_pkg["rendered_distance"].squeeze() \
                        .reshape(-1)[valid_indices]
                    H_r2n = r2n_r[None] - torch.matmul(
                        r2n_t[None, :, None].expand(d_loc.shape[0], 3, 1),
                        n_loc[:, :, None].expand(d_loc.shape[0], 3, 1)
                        .permute(0, 2, 1)) / d_loc[..., None, None]
                    H_r2n = torch.matmul(
                        nearest_cam.get_k(nearest_cam.ncc_scale)[None]
                        .expand(d_loc.shape[0], 3, 3), H_r2n) \
                        @ cam.get_inv_k(cam.ncc_scale)
                    grid = patch_warp(H_r2n.reshape(-1, 3, 3), ori_patch)
                    grid[:, :, 0] = 2 * grid[:, :, 0] / (Wg - 1) - 1
                    grid[:, :, 1] = 2 * grid[:, :, 1] / (Hg - 1) - 1
                    _, near_gray = nearest_cam.get_image()
                    samp = F.grid_sample(near_gray[None], grid.reshape(1, -1, 1, 2),
                                         align_corners=True)
                    samp = samp.reshape(-1, total_patch_size)
                    ncc, ncc_mask = lncc(ref_gray, samp)
                    m = ncc_mask.reshape(-1)
                    ncc = (ncc.reshape(-1) * w_s)[m].squeeze()
                    if m.sum() > 0:
                        loss = loss + opt.multi_view_ncc_weight * ncc.mean()

        loss.backward()

        with torch.no_grad():
            ema = 0.4 * float(loss.item()) + 0.6 * ema
            if iteration % 500 == 0:
                print(f"[stac-pgsr] iter {iteration}/{opt.iterations} "
                      f"loss {ema:.4f} points {len(gaussians.get_xyz):,} "
                      f"vram {torch.cuda.max_memory_allocated() / 1e9:.1f}GB",
                      flush=True)
            if iteration < opt.densify_until_iter:
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask],
                                                        radii[mask])
                gaussians.add_densification_stats(
                    render_pkg["viewspace_points"],
                    render_pkg["viewspace_points_abs"], visibility_filter)
                if iteration > opt.densify_from_iter \
                        and iteration % opt.densification_interval == 0:
                    size_th = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                        opt.opacity_cull_threshold, scene.cameras_extent, size_th)
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)
                if pose_opt is not None and do_pose:
                    pose_opt.step()
                    pose_opt.zero_grad(set_to_none=True)

    # ── save model + export rendered depths for the TSDF (depth_source pgsr_render)
    scene.save(opt.iterations)
    app_model.save_weights(scene.model_path, opt.iterations)
    render_dir = Path(args.render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    psnrs = []
    pose_lines = []
    with torch.no_grad():
        for cam in cams:
            g_render = posed(gaussians, cam)
            pkg = render(cam, g_render, pipe, bg, app_model=app_model,
                         return_plane=True, return_depth_normal=False)
            depth = pkg["plane_depth"].squeeze().detach().cpu().numpy() \
                .astype(np.float32)
            alpha = pkg.get("rendered_alpha")
            valid = (depth > 1e-4)
            if alpha is not None:
                valid &= (alpha.squeeze().detach().cpu().numpy() > 0.5)
            num = int(Path(cam.image_name).stem) if cam.image_name.split(".")[0] \
                .isdigit() else int("".join(ch for ch in cam.image_name if ch.isdigit()))
            np.savez_compressed(render_dir / f"frame_{num}.npz",
                                depth=depth, valid=valid)
            psnrs.append(float(psnr((pkg["render"].clamp(0, 1)),
                                    cam.get_image()[0].cuda().clamp(0, 1)).mean()))
            if args.pose_refine:
                i = cam_index[cam.image_name]
                R_d, t_d = se3_exp(pose_delta[i: i + 1])
                D = torch.eye(4, device="cuda")
                D[:3, :3], D[:3, 3] = R_d[0], t_d[0]
                # rendered world' = D·world at fixed cam V  ⇒  c2w' = D⁻¹ · c2w
                Rt = torch.eye(4)
                Rt[:3, :3] = torch.tensor(cam.R).T.float()
                Rt[:3, 3] = torch.tensor(cam.T).float()
                c2w = torch.linalg.inv(Rt.cuda())
                c2w_ref = torch.linalg.inv(D) @ c2w
                pose_lines.append((num, " ".join(f"{x:.9g}" for x in
                                                 c2w_ref.reshape(-1).tolist())))
    if pose_lines:
        pose_lines.sort()
        (render_dir / "poses_refined.txt").write_text(
            "\n".join(f"{n} {ln}" for n, ln in pose_lines) + "\n")
    report = {
        "iterations": int(opt.iterations),
        "n_cameras": len(cams),
        "n_gaussians": int(len(gaussians.get_xyz)),
        "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
        "pose_refine": bool(args.pose_refine),
        "pose_delta_max_t_m": (float(pose_delta[:, 3:].norm(dim=1).max())
                               if args.pose_refine else 0.0),
        "elapsed_s": round(time.time() - t_start, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "masks_used": len(keep_masks),
    }
    (render_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[stac-pgsr] DONE {report}", flush=True)


if __name__ == "__main__":
    main()
