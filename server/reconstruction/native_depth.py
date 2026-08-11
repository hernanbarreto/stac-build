"""
Native-resolution keyframe depth — pre-TSDF (precision task, Phase C).
================================================================================
Omega infers at ~512 (the 688×384 grid); the video is 1080p+. The geometric
detail of those native pixels is discarded before the pipeline even starts.
This module refines each keyframe's depth (post-scale, post-Phase-B-consistency
when enabled) to native resolution BEFORE the TSDF, guided by the RGB frame,
WITHOUT hallucinating geometry. Two variants (A/B decides the default):

  guided_filter       deterministic joint-bilateral upsample ×factor with THREE
                      weight terms: space, guide colour, and DEPTH RANGE. The
                      depth term is what the legacy tsdf upsample (upsample_depth,
                      A/B'd OFF for noisy edges) was missing: neighbours across a
                      depth discontinuity get ~zero weight even when their colour
                      matches, so edges cannot smear. Mandatory baseline — no new
                      models.

  da3_detail_transfer DA3 run at HIGH resolution on the keyframe as the detail
                      source: low frequencies (metric) come from the guided-
                      upsampled omega depth, high frequencies from DA3 hi-res,
                      scaled by a robust local gain and CAPPED relative to the
                      base (never invents structure beyond detail_cap_rel).
                      DA3 hi-res runs on keyframes only; time/VRAM are measured
                      and reported.

Edge-aware is mandatory; the TSDF's long-edge cull (edge_thresh / 0.10 m tri
cull) stays as the safety net downstream.

Outputs (resume-aware, fail-fast):
    output/native_depth/frame_<num>.npz   {depth: float32 (H·f, W·f), valid: bool}
    output/native_depth/report.json       params + per-frame stats + timings

Wired by segmentation/tsdf_export.py (config tsdf.native_depth_method, default
"off" until a variant wins its A/B).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("NativeDepth")

ND_DIRNAME = "native_depth"
REPORT_NAME = "report.json"
HIRES_DIRNAME = "hires_output"          # under output/da3_run/


# ──────────────────────────────────────────────────────────────────────────────
# Guided (joint-bilateral, depth-edge-aware) upsampling
# ──────────────────────────────────────────────────────────────────────────────

def guided_upsample(depth_lo: np.ndarray, valid_lo: Optional[np.ndarray],
                    guide_hi: np.ndarray, factor: int, radius: int = 2,
                    sigma_space: float = 1.2, sigma_color: float = 0.06,
                    sigma_depth_rel: float = 0.05, device=None):
    """×factor joint-bilateral depth upsample with space+colour+DEPTH weights.

    For output pixel p: lo-res neighbours q in a (2r+1)² window around p/f vote
    with w = G(|p/f − q|; σ_space) · G(|g_hi(p) − g_lo(q)|; σ_color)
            · G(|d(q) − d_c| / (σ_depth_rel · d_c))
    where d_c is p's nearest lo-res depth. The depth term kills cross-edge
    mixing (the legacy upsampler's measured failure). Invalid samples never
    vote; pixels with no valid votes stay invalid — hallucination-free.

    Returns (depth_hi float32 (H,W), valid_hi bool)."""
    import torch
    import torch.nn.functional as F
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h, w = depth_lo.shape
    f = int(factor)
    H, W = h * f, w * f
    if guide_hi.shape[:2] != (H, W):
        raise RuntimeError(f"guide {guide_hi.shape[:2]} != target {(H, W)}")

    d = torch.as_tensor(np.ascontiguousarray(depth_lo), dtype=torch.float32,
                        device=dev)[None, None]
    v = (d > 1e-4).float()
    if valid_lo is not None:
        v = v * torch.as_tensor(np.ascontiguousarray(valid_lo.astype(np.float32)),
                                device=dev)[None, None]
    g = torch.as_tensor(np.ascontiguousarray(guide_hi), dtype=torch.float32, device=dev)
    if g.ndim == 3:
        g = g.mean(dim=2)
    g = g / 255.0
    g_lo = F.avg_pool2d(g[None, None], f)                       # neighbour colours

    k = 2 * radius + 1
    dn = F.unfold(d, k, padding=radius).reshape(1, k * k, h, w)   # lo-res windows
    vn = F.unfold(v, k, padding=radius).reshape(1, k * k, h, w)
    gn = F.unfold(g_lo, k, padding=radius).reshape(1, k * k, h, w)

    # spatial term (constant per offset)
    off = torch.arange(-radius, radius + 1, device=dev, dtype=torch.float32)
    oy, ox = torch.meshgrid(off, off, indexing="ij")
    w_sp = torch.exp(-(oy ** 2 + ox ** 2) / (2 * sigma_space ** 2)).reshape(1, k * k, 1, 1)

    # upsample windows to the hi-res grid (nearest: each output px inherits its
    # lo-res cell's window)
    dn = F.interpolate(dn, size=(H, W), mode="nearest")
    vn = F.interpolate(vn, size=(H, W), mode="nearest")
    gn = F.interpolate(gn, size=(H, W), mode="nearest")
    d_c = F.interpolate(d, size=(H, W), mode="nearest")[0, 0]

    w_col = torch.exp(-((gn - g[None, None]) ** 2) / (2 * sigma_color ** 2))
    sig_d = (sigma_depth_rel * d_c).clamp(min=1e-4)[None, None]
    w_dep = torch.exp(-((dn - d_c[None, None]) ** 2) / (2 * sig_d ** 2))
    wgt = w_sp * w_col * w_dep * vn

    num = (wgt * dn).sum(dim=1)
    den = wgt.sum(dim=1)
    out = torch.where(den > 1e-8, num / den.clamp(min=1e-12),
                      torch.zeros_like(den))[0]
    valid = (den[0] > 1e-8) & (F.interpolate(v, size=(H, W), mode="nearest")[0, 0] > 0)
    return out.cpu().numpy().astype(np.float32), valid.cpu().numpy()


def _gaussian_blur(x, sigma: float):
    """Separable Gaussian blur on a (H,W) torch tensor."""
    import torch
    import torch.nn.functional as F
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    ker = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    ker = (ker / ker.sum()).reshape(1, 1, -1)
    y = x[None, None]
    y = F.conv2d(y, ker[:, :, :, None].transpose(2, 3), padding=(0, r))
    y = F.conv2d(y, ker[:, :, :, None], padding=(r, 0))
    return y[0, 0]


def detail_transfer(base_hi: np.ndarray, valid_hi: np.ndarray,
                    da3_hi: np.ndarray, da3_conf: Optional[np.ndarray],
                    factor: int, detail_cap_rel: float = 0.10,
                    conf_drop_pct: float = 20.0, device=None):
    """High-frequency detail from DA3 hi-res grafted onto the metric base.

        gain   = lowpass(base) / lowpass(da3)      (robust local scale match)
        detail = (da3 − lowpass(da3)) · gain       (metric-scaled high freq)
        out    = base + clip(detail, ±detail_cap_rel · base)

    lowpass σ = factor: everything the omega grid could already represent comes
    from omega; ONLY the frequencies above its Nyquist come from DA3. The cap
    bounds hallucination structurally; low-confidence DA3 pixels contribute no
    detail. Returns (depth float32, mean_abs_detail_rel float)."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.as_tensor(base_hi, dtype=torch.float32, device=dev)
    d3 = torch.as_tensor(da3_hi, dtype=torch.float32, device=dev)
    vb = torch.as_tensor(valid_hi.astype(np.float32), device=dev) * (b > 1e-4).float() \
        * (d3 > 1e-4).float()
    sig = float(factor)
    Lb = _gaussian_blur(b * vb, sig) / _gaussian_blur(vb, sig).clamp(min=1e-6)
    Ld = _gaussian_blur(d3 * vb, sig) / _gaussian_blur(vb, sig).clamp(min=1e-6)
    gain = Lb / Ld.clamp(min=1e-6)
    gm = torch.median(gain[vb > 0]) if (vb > 0).any() else torch.tensor(1.0, device=dev)
    gain = gain.clamp(min=0.5 * gm, max=2.0 * gm)       # robust: no far-range blowups
    detail = (d3 - Ld) * gain
    if da3_conf is not None:
        c = torch.as_tensor(da3_conf, dtype=torch.float32, device=dev)
        if c.shape == detail.shape and float(conf_drop_pct) > 0:
            thr = torch.quantile(c[vb > 0].float(), float(conf_drop_pct) / 100.0) \
                if (vb > 0).any() else c.min()
            detail = torch.where(c >= thr, detail, torch.zeros_like(detail))
    cap = float(detail_cap_rel) * b
    detail = torch.clamp(detail, -cap, cap) * vb
    out = b + detail
    rel = (detail.abs() / b.clamp(min=1e-6))[vb > 0]
    mean_rel = float(rel.mean()) if rel.numel() else 0.0
    return out.cpu().numpy().astype(np.float32), mean_rel


# ──────────────────────────────────────────────────────────────────────────────
# DA3 hi-res extraction (subprocess, keyframes only, resume-aware)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_da3_hires(output_dir: Path, frames_dir: Path, frame_nums, res: int,
                      log) -> Dict[int, Path]:
    """Isolated per-frame DA3 at process_res=res for every keyframe missing its
    hi-res npz. Same conversion contract as the anchor extraction. Returns
    num → npz path. Time is measured and logged (the cost the A/B weighs)."""
    hires = output_dir / "da3_run" / HIRES_DIRNAME
    hires.mkdir(parents=True, exist_ok=True)
    missing = [n for n in frame_nums if not (hires / f"frame_{n}.npz").exists()]
    if missing:
        server_dir = Path(__file__).resolve().parent.parent
        tmp = output_dir / "_da3_hires_frames"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for n in missing:
            src = frames_dir / f"{n:06d}.jpg"
            if src.exists():
                os.symlink(str(src), str(tmp / src.name))
        raw = output_dir / "da3_run" / "hires_raw"
        cmd = [sys.executable, str(server_dir / "extract_da3_depth.py"),
               "--image_dir", str(tmp), "--output_dir", str(raw),
               "--per_frame", "--process_res", str(int(res))]
        log(f"DA3 hi-res: extracting {len(missing)} keyframes at process_res={res} "
            f"(isolated per-frame)")
        t0 = time.time()
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"DA3 hi-res extraction exited with code {rc}")
        n_ok = 0
        for n in missing:
            stem = f"{n:06d}"
            dp, cp = raw / f"{stem}_depth.npy", raw / f"{stem}_conf.npy"
            if not dp.exists():
                continue
            arrays = {"depth": np.load(dp).astype(np.float32)}
            if cp.exists():
                arrays["conf"] = np.load(cp).astype(np.float32)
            np.savez_compressed(hires / f"frame_{n}.npz", **arrays)
            n_ok += 1
        shutil.rmtree(tmp, ignore_errors=True)
        log(f"DA3 hi-res: {n_ok}/{len(missing)} maps in {time.time() - t0:.0f}s "
            f"({(time.time() - t0) / max(n_ok, 1):.1f} s/frame)")
        if n_ok < len(missing):
            raise RuntimeError(f"DA3 hi-res produced {n_ok}/{len(missing)} maps — "
                               f"refusing a partially-refined depth set")
    return {n: hires / f"frame_{n}.npz" for n in frame_nums
            if (hires / f"frame_{n}.npz").exists()}


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def run(output_dir: Path, frames_dir: Path, method: str = "guided_filter",
        factor: int = 2, sigma_depth_rel: float = 0.05,
        detail_cap_rel: float = 0.10, da3_res: int = 1008,
        mv_dir: Optional[Path] = None, log=None, device=None,
        _frames_override: Optional[Dict[int, dict]] = None) -> dict:
    """Generate native-resolution refined depth for every keyframe. Resume-aware
    (identical params + complete npz set ⇒ no-op), fail-fast on missing inputs.
    mv_dir: Phase-B masks — when given, the refinement consumes the CONSISTENCY-
    FILTERED depth (task order: post-scale, post-consistency, then upsample)."""
    _log = log if log is not None else (lambda m: logger.info(m))
    output_dir = Path(output_dir)
    nd_dir = output_dir / ND_DIRNAME
    if method not in ("guided_filter", "da3_detail_transfer"):
        raise ValueError(f"native_depth method '{method}' unknown")
    params = {"method": method, "factor": int(factor),
              "sigma_depth_rel": float(sigma_depth_rel),
              "detail_cap_rel": float(detail_cap_rel),
              "da3_res": int(da3_res), "mv": mv_dir is not None}

    rp = nd_dir / REPORT_NAME
    if rp.exists():
        try:
            old = json.loads(rp.read_text())
        except Exception:
            old = None
        if old and old.get("params") == params and all(
                (nd_dir / f"frame_{n}.npz").exists() for n in old.get("frames", {})):
            _log(f"refined depth already on disk with identical params — skipped")
            return old

    t0 = time.time()
    if _frames_override is not None:
        frames = _frames_override
    else:
        from segmentation.tsdf_export import _resolve_mapanything_depth
        src = _resolve_mapanything_depth(output_dir, conf_percentile=None,
                                         mv_dir=mv_dir)
        if src is None:
            raise RuntimeError("native_depth: no omega chunk depth found — "
                               "nothing to refine")
        loader, _ = src
        from reconstruction.scale_align import _read_poses
        _, nums, _ = _read_poses(output_dir)
        frames = {}
        for n in nums:
            d = loader(int(n))
            if d is not None and d.get("K") is not None:
                frames[int(n)] = {"depth": np.asarray(d["depth"], np.float32),
                                  "valid": np.asarray(d["valid"], bool)}
    if len(frames) < 1:
        raise RuntimeError("native_depth: no keyframes with depth")

    hires = {}
    da3_elapsed = 0.0
    if method == "da3_detail_transfer" and _frames_override is None:
        t_da3 = time.time()
        hires = _ensure_da3_hires(output_dir, Path(frames_dir),
                                  sorted(frames.keys()), da3_res, _log)
        da3_elapsed = time.time() - t_da3

    import torch
    import torch.nn.functional as F
    nd_dir.mkdir(parents=True, exist_ok=True)
    per: Dict[str, dict] = {}
    from PIL import Image
    for n in sorted(frames.keys()):
        fr = frames[n]
        d_lo, v_lo = fr["depth"], fr.get("valid")
        h, w = d_lo.shape
        H, W = h * int(factor), w * int(factor)
        if "guide" in fr:
            guide = fr["guide"]
        else:
            jp = Path(frames_dir) / f"{n:06d}.jpg"
            if not jp.exists():
                raise RuntimeError(f"native_depth: frame image {jp.name} missing")
            guide = np.asarray(Image.open(jp).convert("L").resize((W, H),
                                                                  Image.BILINEAR))
        d_hi, valid_hi = guided_upsample(
            d_lo, v_lo, guide, factor, sigma_depth_rel=sigma_depth_rel,
            device=device)
        stats = {"h": H, "w": W}
        if method == "da3_detail_transfer":
            src = hires.get(n) if _frames_override is None else None
            da3_map = fr.get("da3_hi")
            da3_conf = fr.get("da3_conf")
            if src is not None:
                npz = np.load(src)
                da3_map = npz["depth"].astype(np.float32)
                da3_conf = npz["conf"].astype(np.float32) if "conf" in npz.files else None
            if da3_map is None:
                raise RuntimeError(f"native_depth: no DA3 hi-res map for frame {n}")
            if da3_map.shape != (H, W):
                t = torch.as_tensor(da3_map)[None, None]
                da3_map = F.interpolate(t, size=(H, W), mode="bilinear",
                                        align_corners=False)[0, 0].numpy()
                if da3_conf is not None:
                    t = torch.as_tensor(da3_conf)[None, None]
                    da3_conf = F.interpolate(t, size=(H, W), mode="bilinear",
                                             align_corners=False)[0, 0].numpy()
            d_hi, mean_rel = detail_transfer(d_hi, valid_hi, da3_map, da3_conf,
                                             factor, detail_cap_rel=detail_cap_rel,
                                             device=device)
            stats["mean_abs_detail_rel"] = round(mean_rel, 5)
        np.savez_compressed(nd_dir / f"frame_{n}.npz",
                            depth=d_hi.astype(np.float32), valid=valid_hi)
        stats["valid_pct"] = round(100.0 * float(valid_hi.mean()), 2)
        per[str(n)] = stats

    report = {"version": 1, "params": params, "n_keyframes": len(per),
              "frames": per, "elapsed_s": round(time.time() - t0, 1),
              "da3_hires_elapsed_s": round(da3_elapsed, 1)}
    rp.write_text(json.dumps(report, indent=2))
    _log(f"{len(per)} keyframes refined to ×{factor} ({method}) in "
         f"{report['elapsed_s']}s"
         + (f" (DA3 hi-res {da3_elapsed:.0f}s)" if da3_elapsed else ""))
    return report
