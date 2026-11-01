"""The trainer — runs in env ``gs`` (torch + gsplat, Apache-2.0) on the training set the da3 worker
wrote (``gaussians.dataset``); needs nothing else of the server.

    python -m gaussians.train --trainset <dir> --out <dir> --config <json>

What it holds fixed: every camera (the certified poses and the session camera, distortion included
through gsplat's unscented projection when the camera has a lens). What it optimises: the gaussians
(means, scales, rotations, opacities, spherical harmonics) and, when asked, one affine colour
correction per training keyframe (the phone's exposure) pulled toward identity. Losses: the
reference recipe (1 − w)·L1 + w·(1 − SSIM), plus an L1 on the rendered EXPECTED depth against the
depth the chain measured, where it measured one, normalised by the frame's median depth.

Held-out keyframes are never trained on; their PSNR / SSIM (identity exposure) is the report's
measured number. Determinism: seeds fixed and recorded; gsplat's CUDA backward uses atomic adds,
so two runs agree to float precision, not bit for bit — declared in the report.

Outputs in ``--out``: gaussians.ply (the 3DGS record), gaussians.splat (when asked), seed_index.bin
(the cloud row of every gaussian's seed — the instance is resolved live by the server), report.json.
Progress on stdout as ``[gs] …`` lines the worker reads.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gaussians import label as GL
from gaussians.dataset import TrainingSet, read_depth_map, read_trainset
from gaussians.plyio import SH_C0, write_gaussians_ply, write_instances, write_splat, read_ply_vertices

LOG = "[gs]"
PLY_NAME = "gaussians.ply"
SPLAT_NAME = "gaussians.splat"
SEED_INDEX_NAME = "seed_index.bin"
REPORT_NAME = "report.json"


# ── pure functions (tested on the CPU) ─────────────────────────────────────────────────────

def rgb_to_sh0(rgb01: np.ndarray) -> np.ndarray:
    return (np.asarray(rgb01, np.float32) - 0.5) / SH_C0


def logit(p: float) -> float:
    return float(math.log(p / (1.0 - p)))


def scene_extent(c2w: np.ndarray) -> float:
    """gsplat's convention: 1.1 × the largest distance of a camera centre to their mean."""
    C = np.asarray(c2w)[:, :3, 3]
    d = np.linalg.norm(C - C.mean(axis=0), axis=1)
    return float(1.1 * max(d.max(), 1e-3))


def depth_pixel_map(K_native: np.ndarray, K_depth: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """For every native pixel (u, v) the pixel of the depth grid that sees the same ray — the two
    grids are the same camera at another resolution / crop: u_d = (u − cx_n)·fx_d/fx_n + cx_d."""
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    ud = (u - K_native[0, 2]) * (K_depth[0, 0] / K_native[0, 0]) + K_depth[0, 2]
    vd = (v - K_native[1, 2]) * (K_depth[1, 1] / K_native[1, 1]) + K_depth[1, 2]
    return np.rint(ud).astype(np.int64), np.rint(vd).astype(np.int64)


def grid_pixel_map(grid_map: Dict[str, float], width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """The record-grid pixel of every native pixel through the session's grid map (the inverse of
    precision.camera.grid_to_native): u_g = (u_n − crop_x + 0.5) / scale_x − 0.5 + pad_left."""
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    ug = (u - grid_map["crop_x"] + 0.5) / grid_map["scale_x"] - 0.5 + grid_map["pad_left"]
    vg = (v - grid_map["crop_y"] + 0.5) / grid_map["scale_y"] - 0.5 + grid_map["pad_top"]
    return np.rint(ug).astype(np.int64), np.rint(vg).astype(np.int64)


def resample_depth(depth: np.ndarray, valid: np.ndarray, ud: np.ndarray, vd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(native depth, native validity) by nearest lookup; a native pixel whose depth pixel falls
    off the map is invalid."""
    h, w = depth.shape
    inside_u = (ud >= 0) & (ud < w)
    inside_v = (vd >= 0) & (vd < h)
    uu = np.clip(ud, 0, w - 1)
    vv = np.clip(vd, 0, h - 1)
    out = depth[vv[:, None], uu[None, :]].astype(np.float32)
    ok = valid[vv[:, None], uu[None, :]] & inside_v[:, None] & inside_u[None, :] & np.isfinite(out) & (out > 0)
    return out, ok


def gaussian_window(size: int, sigma: float):
    import torch
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])


def ssim(img1, img2, window_size: int = 11, sigma: float = 1.5):
    """SSIM of two (H, W, 3) images in [0, 1] — the reference 11×11 gaussian window, per channel."""
    import torch
    import torch.nn.functional as F
    a = img1.permute(2, 0, 1)[None]
    b = img2.permute(2, 0, 1)[None]
    C = a.shape[1]
    w = gaussian_window(window_size, sigma).to(a)[None, None].repeat(C, 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(a, w, padding=pad, groups=C)
    mu2 = F.conv2d(b, w, padding=pad, groups=C)
    s11 = F.conv2d(a * a, w, padding=pad, groups=C) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=pad, groups=C) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=pad, groups=C) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return m.mean()


def psnr(img1, img2) -> float:
    import torch
    mse = torch.mean((img1 - img2) ** 2).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


class Exposure:
    """One 3×4 affine colour correction per training keyframe, identity at start."""

    def __init__(self, n: int, device):
        import torch
        A = torch.zeros(n, 3, 4, device=device)
        A[:, :, :3] = torch.eye(3, device=device)
        self.A = torch.nn.Parameter(A)

    def apply(self, img, i: int):
        A = self.A[i]
        return img @ A[:, :3].T + A[:, 3]

    def penalty(self):
        import torch
        eye = torch.zeros_like(self.A)
        eye[:, :, :3] = torch.eye(3, device=self.A.device)
        return ((self.A - eye) ** 2).mean()


def sh_degree_at(step: int, max_degree: int, interval: int) -> int:
    return min(step // max(int(interval), 1), int(max_degree))


# ── the run ────────────────────────────────────────────────────────────────────────────────

def _load_images(ts: TrainingSet, device):
    import torch
    from PIL import Image
    imgs = []
    for p in ts.image_paths:
        im = Image.open(p).convert("RGB")
        if im.size != (ts.width, ts.height):
            raise RuntimeError(f"{p}: {im.size} is not the session's native {ts.width}x{ts.height}")
        imgs.append(torch.from_numpy(np.asarray(im, np.uint8)).to(device))
    return imgs


def _depth_targets(ts: TrainingSet, trainset_dir: Path, device, log) -> Dict[int, Tuple[Any, Any]]:
    """Per training keyframe index: (native depth, native validity) as tensors — resampled from the
    chain's grid through the depth file's own intrinsics when it carries them, else the grid map."""
    import torch
    out: Dict[int, Tuple[Any, Any]] = {}
    if ts.depth_grid is None:
        return out
    n_ok = 0
    for i, f in enumerate(ts.frames.tolist()):
        if ts.holdout[i]:
            continue
        got = read_depth_map(trainset_dir, int(f))
        if got is None:
            continue
        depth, valid = got
        kd_p = Path(trainset_dir) / "depth" / f"K_{int(f)}.npy"
        if kd_p.exists():
            ud, vd = depth_pixel_map(ts.K, np.load(kd_p), ts.width, ts.height)
        elif depth.shape == (ts.height, ts.width):
            ud, vd = np.arange(ts.width), np.arange(ts.height)
        else:
            ud, vd = grid_pixel_map(ts.grid_map, ts.width, ts.height)
        d, ok = resample_depth(depth, valid, ud, vd)
        out[i] = (torch.from_numpy(d).to(device), torch.from_numpy(ok).to(device))
        n_ok += 1
    log(f"{LOG} depth targets for {n_ok} training keyframe(s)")
    return out


def strategy_choice(requested: str, lens: bool) -> Tuple[str, str]:
    """The densification strategy actually run, with the reason. gsplat's DefaultStrategy densifies by the
    gradient of the PROJECTED means (``info["means2d"].retain_grad()``); the unscented-transform projection
    that renders a distorted lens (``with_ut=True``, the phone's OPENCV k1 k2 p1 p2) exposes no such gradient
    (pccr 2026-10-31: "can't retain_grad on Tensor that has requires_grad=False"), so a lens session runs the
    MCMC strategy — capped at train.max_gaussians, seeded — and the report says so. A pinhole session runs
    what was configured."""
    if requested == "default" and lens:
        return "mcmc", ("the session camera has lens distortion, rendered through gsplat's unscented transform, "
                        "whose projected means carry no gradient for the default densification")
    return requested, "as configured"


def init_for(strategy_used: str, init_cfg: Dict[str, Any], mcmc_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """(initial opacity, initial scale factor × the knn distance): gsplat simple_trainer's presets — the mcmc
    preset's init_opa 0.5 / init_scale 0.1 for mcmc, the default preset's (config init.opacity / init.scale)."""
    if strategy_used == "mcmc":
        return float(mcmc_cfg["init_opacity"]), float(mcmc_cfg["init_scale"])
    return float(init_cfg["opacity"]), float(init_cfg["scale"])


def means_lr_gamma(decay: float, iterations: int) -> float:
    """ExponentialLR's gamma so the means' lr reaches decay × its start by the last iteration (gsplat: 0.01)."""
    return float(decay) ** (1.0 / max(1, int(iterations)))


def depth_disparity_loss(d_pred, d_meas, ok, scene_extent: float, depth_lambda: float):
    """gsplat simple_trainer's depth term: L1 of the disparity (1/z) against the measured depth, × the scene
    extent × depth_lambda, over the pixels where the session measured a depth and the render has one."""
    import torch
    ok2 = ok & (d_meas > 0) & (d_pred > 0)
    if not bool(ok2.any()):
        return d_pred.sum() * 0.0
    return torch.abs(1.0 / d_pred[ok2] - 1.0 / d_meas[ok2]).mean() * float(scene_extent) * float(depth_lambda)


_CELL_OFF = 1 << 20   # cell indices live in [-2^20, 2^20): ±31 km at a 3 cm tolerance


def clamp_scales_(scales_log_param, tol: float) -> int:
    """In place: no gaussian wider than tol (σ per axis) — the extent stays inside the confinement band like
    its centre. Returns how many axes were clamped."""
    import math
    import torch
    with torch.no_grad():
        over = scales_log_param.data > math.log(float(tol))
        n = int(over.sum())
        if n:
            scales_log_param.data.clamp_(max=math.log(float(tol)))
    return n


def over_cap_mask(opacities_logit, cap: int):
    """(N,) bool: the gaussians to remove so at most `cap` stay — the lowest opacities go (a deterministic
    stable sort), the BOUND train.max_gaussians applied to the default strategy too (gsplat's has none)."""
    import torch
    n = int(opacities_logit.shape[0])
    mask = torch.zeros(n, dtype=torch.bool, device=opacities_logit.device)
    if n > int(cap):
        order = torch.argsort(opacities_logit.detach(), stable=True)      # ascending: the faintest first
        mask[order[: n - int(cap)]] = True
    return mask


def cloud_cell_keys(xyz, tol: float):
    """The sorted unique keys of the cells (size tol) the cloud occupies — one int64 per cell, exact (no hash)."""
    import torch
    c = torch.floor(xyz / float(tol)).to(torch.int64) + _CELL_OFF
    keys = (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]
    return torch.unique(keys)


def confinement_keep(means, cloud_keys, tol: float):
    """(N,) bool: the gaussian's cell or one of its 26 neighbours holds a cloud point. A gaussian with NO cloud
    point within tol per axis is therefore False (removed); a True one has a cloud point within 2·tol per axis."""
    import torch
    c = torch.floor(means / float(tol)).to(torch.int64) + _CELL_OFF
    keep = torch.zeros(len(means), dtype=torch.bool, device=means.device)
    n = len(cloud_keys)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                k = ((c[:, 0] + dx) << 42) | ((c[:, 1] + dy) << 21) | (c[:, 2] + dz)
                idx = torch.searchsorted(cloud_keys, k).clamp_(max=n - 1)
                keep |= cloud_keys[idx] == k
    return keep


def train(trainset_dir: Path, out_dir: Path, tcfg: Dict[str, Any], log=print) -> Dict[str, Any]:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy
    from gsplat.strategy.ops import remove as _remove_gaussians

    t_start = time.time()
    trainset_dir, out_dir = Path(trainset_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, ex, lb = tcfg["train"], tcfg["export"], tcfg["label"]
    seed = int(tr["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    ts = read_trainset(trainset_dir)
    N0 = ts.n_seeds
    train_idx = [i for i in range(ts.n_frames) if not ts.holdout[i]]
    held_idx = [i for i in range(ts.n_frames) if ts.holdout[i]]
    log(f"{LOG} {ts.n_frames} keyframes ({len(train_idx)} train, {len(held_idx)} held out), {N0:,} seeds, "
        f"{ts.width}x{ts.height}, sh degree {tr['sh_degree']}, {tr['iterations']} iterations, strategy {tr['strategy']}")
    images = _load_images(ts, device)
    depths = _depth_targets(ts, trainset_dir, device, log) if float(tr["depth_lambda"]) > 0 else {}
    extent = scene_extent(ts.c2w)
    # the lens decides the strategy (strategy_choice), the strategy its init preset (gsplat default / mcmc)
    dist = np.asarray(ts.dist, np.float64)
    lens = bool(np.any(np.abs(dist) > 0))
    strategy_used, strategy_reason = strategy_choice(str(tr["strategy"]), lens)
    if strategy_used != tr["strategy"]:
        log(f"{LOG} strategy {tr['strategy']} → {strategy_used}: {strategy_reason}")
    ini, mc, cf = tcfg["init"], tr["mcmc"], tr["confine"]
    init_opacity, init_scale = init_for(strategy_used, ini, mc)
    log(f"{LOG} init: opacity {init_opacity}, scale ×{init_scale} of the knn distance (gsplat {strategy_used} preset)")

    # the parameters
    K_sh = (int(tr["sh_degree"]) + 1) ** 2 - 1
    means = torch.from_numpy(np.asarray(ts.seeds_xyz, np.float32)).to(device)
    scales = torch.from_numpy(np.asarray(ts.seeds_scale_log, np.float32)).to(device) + float(np.log(init_scale))
    quats = torch.zeros(N0, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.full((N0,), logit(init_opacity), device=device)
    sh0 = torch.from_numpy(rgb_to_sh0(np.asarray(ts.seeds_rgb, np.float32) / 255.0)).to(device)[:, None, :]
    shN = torch.zeros(N0, K_sh, 3, device=device)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means), "scales": torch.nn.Parameter(scales), "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities), "sh0": torch.nn.Parameter(sh0), "shN": torch.nn.Parameter(shN)})
    lr = tr["lr"]
    lrs = {"means": float(lr["means"]) * extent, "scales": float(lr["scales"]), "quats": float(lr["quats"]),
           "opacities": float(lr["opacities"]), "sh0": float(lr["sh0"]), "shN": float(lr["shN"])}
    optimizers = {k: torch.optim.Adam([{"params": [params[k]], "lr": lrs[k], "name": k}], eps=1e-15) for k in params}
    exposure = Exposure(ts.n_frames, device) if bool(tr["exposure"]) else None
    exp_opt = torch.optim.Adam([exposure.A], lr=float(lr["exposure"])) if exposure is not None else None

    iters = int(tr["iterations"])
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=means_lr_gamma(float(tr["means_lr_decay"]), iters))

    # confinement to the measured surface (USER 2026-10-31): the published cloud's occupied cells at the tolerance
    cloud_p = Path(ts.meta["session_dir"]) / "output" / "cleaned_cloud.ply"
    cv = read_ply_vertices(cloud_p)
    cloud_xyz = np.stack([np.asarray(cv["x"]), np.asarray(cv["y"]), np.asarray(cv["z"])], 1).astype(np.float32)
    tol = float(cf["tolerance_m"]); confine_every = int(cf["every"])
    cloud_keys = cloud_cell_keys(torch.from_numpy(cloud_xyz).to(device), tol)
    confine_log: List[Dict[str, Any]] = []
    log(f"{LOG} confinement: {len(cloud_xyz):,} cloud points in {len(cloud_keys):,} cells of {tol} m; checked every {confine_every} iterations")

    def confine(at_step: int) -> int:
        with torch.no_grad():
            keep = confinement_keep(params["means"].detach(), cloud_keys, tol)
        n_rm = int((~keep).sum())
        n_cl = clamp_scales_(params["scales"], tol)
        if n_rm:
            _remove_gaussians(params=params, optimizers=optimizers, state=(state if strategy_used == "default" else {}), mask=~keep)
        n_cap = 0
        if strategy_used == "default":           # the cap: gsplat's default strategy has none
            with torch.no_grad():
                over = over_cap_mask(params["opacities"], int(tr["max_gaussians"]))
            n_cap = int(over.sum())
            if n_cap:
                _remove_gaussians(params=params, optimizers=optimizers, state=state, mask=over)
        if n_rm or n_cl or n_cap:
            confine_log.append({"step": int(at_step), "removed": n_rm, "clamped_axes": n_cl, "over_cap": n_cap, "left": int(len(params["means"]))})
        return n_rm
    r0, r1 = int(round(float(tr["refine_start_frac"]) * iters)), int(round(float(tr["refine_stop_frac"]) * iters))
    # cameras
    w2c = torch.from_numpy(np.linalg.inv(np.asarray(ts.c2w, np.float64)).astype(np.float32)).to(device)
    K_t = torch.from_numpy(np.asarray(ts.K, np.float32)).to(device)
    if strategy_used == "mcmc":
        strategy = MCMCStrategy(cap_max=int(tr["max_gaussians"]), noise_lr=float(mc["noise_lr"]),
                                refine_start_iter=int(mc["refine_start_iter"]), refine_stop_iter=int(mc["refine_stop_iter"]),
                                refine_every=int(mc["refine_every"]), min_opacity=float(mc["min_opacity"]), verbose=False)
    else:
        strategy = DefaultStrategy(refine_start_iter=r0, refine_stop_iter=r1, verbose=False)
    strategy.check_sanity(params, optimizers)
    # gsplat 1.5.3: DefaultStrategy.initialize_state(scene_scale) / MCMCStrategy.initialize_state(); only the
    # default strategy has a step_pre_backward (it retains the projected means' gradient there)
    state = strategy.initialize_state(scene_scale=extent) if strategy_used == "default" else strategy.initialize_state()

    radial = torch.tensor([[dist[0], dist[1], 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device) if lens else None
    tangential = torch.tensor([[dist[2], dist[3]]], dtype=torch.float32, device=device) if lens else None
    W, H = ts.width, ts.height
    bg = torch.zeros(1, 3, device=device)

    def render(i: int, degree: int, need_depth: bool):
        kw: Dict[str, Any] = {}
        if lens:
            kw.update(with_ut=True, with_eval3d=True, radial_coeffs=radial, tangential_coeffs=tangential)
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        renders, alphas, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=w2c[i][None], Ks=K_t[None],
            width=W, height=H, sh_degree=degree, packed=False, absgrad=False, backgrounds=bg,
            render_mode="RGB+ED" if need_depth else "RGB", **kw)
        return renders[0], alphas[0], info

    def evaluate(step: int) -> Dict[str, Any]:
        if not held_idx:
            return {"n_heldout": 0, "reason": "no held-out keyframes"}
        ps, ss = [], []
        with torch.no_grad():
            for i in held_idx:
                img, _a, _info = render(i, int(tr["sh_degree"]), False)
                pred = img[..., :3].clamp(0, 1)
                gt = images[i].float() / 255.0
                ps.append(psnr(pred, gt)); ss.append(float(ssim(pred, gt)))
        rec = {"step": int(step), "n_heldout": len(held_idx), "psnr_mean": float(np.mean(ps)), "psnr_median": float(np.median(ps)),
               "psnr_min": float(np.min(ps)), "ssim_mean": float(np.mean(ss)), "ssim_min": float(np.min(ss)),
               "exposure": "identity (the held-out frames have none)"}
        log(f"{LOG} eval it {step}: psnr {rec['psnr_mean']:.2f} (min {rec['psnr_min']:.2f}) ssim {rec['ssim_mean']:.4f} "
            f"on {len(held_idx)} held-out keyframes; {len(params['means']):,} gaussians")
        return rec

    rng = np.random.default_rng(seed)
    order: List[int] = []
    evals: List[Dict[str, Any]] = []
    w_ssim, w_depth, w_exp = float(tr["ssim_weight"]), float(tr["depth_lambda"]), float(tr["exposure_reg"])
    n_confined_total = 0
    loss_hist: List[float] = []
    t_train = time.time()
    for step in range(iters):
        if not order:
            order = [train_idx[k] for k in rng.permutation(len(train_idx))]
        i = order.pop()
        degree = sh_degree_at(step, int(tr["sh_degree"]), int(tr["sh_degree_interval"]))
        need_depth = w_depth > 0 and i in depths
        renders, alphas, info = render(i, degree, need_depth)
        if strategy_used == "default":
            strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = renders[..., :3]
        if exposure is not None:
            pred = exposure.apply(pred, i)
        gt = images[i].float() / 255.0
        l1 = torch.abs(pred - gt).mean()
        loss = (1.0 - w_ssim) * l1 + w_ssim * (1.0 - ssim(pred.clamp(0, 1), gt)) if w_ssim > 0 else l1
        if need_depth:
            d_meas, ok = depths[i]
            loss = loss + depth_disparity_loss(renders[..., 3], d_meas, ok, extent, w_depth)
        if strategy_used == "mcmc":          # the mcmc preset's regularisers (gsplat simple_trainer)
            loss = loss + float(mc["opacity_reg"]) * torch.sigmoid(params["opacities"]).mean() \
                        + float(mc["scale_reg"]) * torch.exp(params["scales"]).mean()
        if exposure is not None and w_exp > 0:
            loss = loss + w_exp * exposure.penalty()
        loss.backward()
        for opt in optimizers.values():      # gsplat's order: optimizer step, scheduler, then the strategy
            opt.step(); opt.zero_grad(set_to_none=True)
        if exp_opt is not None:
            exp_opt.step(); exp_opt.zero_grad(set_to_none=True)
        sched.step()
        if strategy_used == "mcmc":
            strategy.step_post_backward(params, optimizers, state, step, info, lr=sched.get_last_lr()[0])
        else:
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if (step + 1) % confine_every == 0:
            n_confined_total += confine(step + 1)
        loss_hist.append(float(loss))
        if (step + 1) % 100 == 0 or step == 0:
            pct = 100.0 * (step + 1) / iters
            log(f"{LOG} it {step + 1}/{iters} ({pct:.1f}%) loss {np.mean(loss_hist[-100:]):.4f} gaussians {len(params['means']):,} "
                f"sh {degree}", flush=True)
        if (step + 1) % int(tr["eval_every"]) == 0 and step + 1 < iters:
            evals.append(evaluate(step + 1))
    n_confined_final = confine(iters)
    n_confined_total += n_confined_final
    log(f"{LOG} confinement: {n_confined_total:,} gaussians removed off the surface over {len(confine_log)} prune(s), "
        f"{n_confined_final:,} at the end; {len(params['means']):,} stay")
    evals.append(evaluate(iters))
    t_fit = time.time() - t_train

    # the record
    with torch.no_grad():
        means_np = params["means"].detach().cpu().numpy().astype(np.float32)
        scales_np = params["scales"].detach().cpu().numpy().astype(np.float32)
        quats_np = params["quats"].detach().cpu().numpy().astype(np.float32)
        quats_np = quats_np / np.maximum(np.linalg.norm(quats_np, axis=1, keepdims=True), 1e-12)
        opac_np = params["opacities"].detach().cpu().numpy().astype(np.float32)
        sh0_np = params["sh0"].detach().cpu().numpy()[:, 0, :].astype(np.float32)
        shN_np = params["shN"].detach().cpu().numpy().astype(np.float32)
    N = len(means_np)
    seed_index = GL.seed_index_by_nearest(means_np, cloud_xyz, np.arange(len(cloud_xyz), dtype=np.uint32), float(lb["nearest_radius_m"]))
    n_no_seed = int((seed_index == GL.NO_SEED).sum())
    files: Dict[str, str] = {}
    comments = [f"stac gaussians: {N} gaussians from {N0} seeds of the published cloud; poses and camera fixed (certified)"]
    files["ply"] = write_gaussians_ply(out_dir / PLY_NAME, means_np, scales_np, quats_np, opac_np, sh0_np, shN_np, comments).name
    if "splat" in ex["formats"]:
        files["splat"] = write_splat(out_dir / SPLAT_NAME, means_np, scales_np, quats_np, opac_np, sh0_np).name
    files["seed_index"] = write_instances(out_dir / SEED_INDEX_NAME, seed_index).name
    with torch.no_grad():
        s_lin = torch.exp(params["scales"].detach()).cpu().numpy().astype(np.float64)
        op_lin = torch.sigmoid(params["opacities"].detach()).cpu().numpy().astype(np.float64)
    s_hi = s_lin.max(1); aniso = s_hi / np.maximum(s_lin.min(1), 1e-9)
    shape = {"sigma_max_m": {k: float(v) for k, v in zip(("p50", "p90", "p99", "max"), [*np.percentile(s_hi, [50, 90, 99]), s_hi.max()])},
             "anisotropy": {k: float(v) for k, v in zip(("p50", "p90", "p99"), np.percentile(aniso, [50, 90, 99]))},
             "opacity": {k: float(v) for k, v in zip(("p10", "p50", "p90"), np.percentile(op_lin, [10, 50, 90]))},
             "needles_share": float(((s_hi > 0.10) | (aniso > 20)).mean()),
             "reading": "needles = sigma over 10 cm or anisotropy over 20 (the second pccr run: 93 %)"}
    log(f"{LOG} shape: sigma max p50 {shape['sigma_max_m']['p50'] * 100:.2f} cm, anisotropy p50 {shape['anisotropy']['p50']:.1f}, "
        f"opacity p50 {shape['opacity']['p50']:.2f}, needles {100 * shape['needles_share']:.1f} %")
    import gsplat
    report = {
        "version": 1, "stage": "gaussians", "provenance": "tool_measured",
        "n_seeds": int(N0), "n_gaussians": int(N), "n_without_seed_within_radius": n_no_seed,
        "label_radius_m": float(lb["nearest_radius_m"]),
        "iterations": iters, "final_loss_mean_last_100": float(np.mean(loss_hist[-100:])) if loss_hist else None,
        "evaluations": evals, "heldout": evals[-1] if evals else None,
        "depth_supervision": {"weight": w_depth, "n_keyframes_with_depth": len(depths)},
        "exposure": {"learned": exposure is not None, "reg": w_exp},
        "strategy": {"requested": tr["strategy"], "used": strategy_used, "reason": strategy_reason},
        "recipe": "gsplat 1.5.3 examples/simple_trainer.py: default / mcmc presets, depth = disparity L1 x scene extent, "
                  "means lr ExponentialLR; plus the confinement prune to the published cloud",
        "init": {"opacity": init_opacity, "scale_factor": init_scale},
        "confine": {"tolerance_m": tol, "kept_within_m_per_axis": 2 * tol, "sigma_max_m": tol, "every": confine_every,
                    "n_cloud_points": int(len(cloud_xyz)), "removed_total": int(n_confined_total),
                    "removed_at_end": int(n_confined_final), "prunes": confine_log},
        "shape": shape,
        "undistort": ts.meta.get("undistort", {"applied": False}),
        "scene_extent_m": extent, "sh_degree": int(tr["sh_degree"]),
        "camera": {"lens": lens, "dist": dist.tolist(), "K_native": ts.K.tolist(), "size": [W, H]},
        "seconds": {"total": round(time.time() - t_start, 1), "fit": round(t_fit, 1)},
        "versions": {"torch": torch.__version__, "gsplat": getattr(gsplat, "__version__", "?"),
                     "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
        "determinism": "seeds fixed and recorded; gsplat's CUDA backward accumulates with atomic adds — two runs agree "
                       "to float precision, not bit for bit",
        "files": files, "trainset": ts.meta, "config": tcfg,
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=1, default=str))
    log(f"{LOG} done: {N:,} gaussians ({n_no_seed:,} without a seed within {lb['nearest_radius_m']} m) → {out_dir} "
        f"in {report['seconds']['total']} s")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gaussians.train")
    ap.add_argument("--trainset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True, help="JSON with the train / export / label sections")
    args = ap.parse_args(argv)
    tcfg = json.loads(Path(args.config).read_text())

    def _log(msg, flush=True):
        print(msg, flush=flush)
    train(Path(args.trainset), Path(args.out), tcfg, log=_log)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
