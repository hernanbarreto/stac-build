"""
DINOv3 dense-feature infrastructure — shared by the 4-phase feature plan
(USER 2026-09-04): cloud consistency score (fase 1/2), feature-metric pose
refinement (fase 3) and instance feature refinement (fase 4).

One pass of DINOv3 ViT-L/16 (vendor/dinov3 + weights/dinov3) over the
session's keyframes produces a per-frame patch-feature grid, PCA-compressed
and cached under ``output/dino_features/``. Everything downstream reads the
cache — the encoder runs once per session.

Provenance discipline: features only ever SCORE or MATCH measured data;
they never generate geometry.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HUB = _REPO_ROOT / "vendor" / "dinov3"
_DEFAULT_WEIGHTS = (_REPO_ROOT / "weights" / "dinov3" /
                    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
_PATCH = 16
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "dino_features"


def load_dinov3(cfg: Optional[dict] = None, device: Optional[str] = None):
    """Vendor-local load. Imports the backbone factory DIRECTLY
    (dinov3.hub.backbones) instead of torch.hub/hubconf — hubconf drags the
    segmentor/depther stacks in (torchmetrics etc.) that the da3 env does
    not carry, and the backbones module has no such deps."""
    import sys as _sys
    import torch
    cfg = cfg or {}
    hub_dir = str(cfg.get("hub_dir") or _DEFAULT_HUB)
    weights = str(cfg.get("weights") or _DEFAULT_WEIGHTS)
    model_name = str(cfg.get("model") or "dinov3_vitl16")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if hub_dir not in _sys.path:
        _sys.path.insert(0, hub_dir)
    from dinov3.hub import backbones as _bb
    factory = getattr(_bb, model_name)
    model = factory(pretrained=True, weights=weights)
    model.eval().to(dev)
    if dev == "cuda":
        # bfloat16, NOT fp16: the fp16 forward overflowed to NaN on EVERY
        # pccr_v1 keyframe (464x832) — all tokens zeroed, PCA "nan% var".
        # bf16 has fp32's exponent range; A6000 (sm_86) runs it natively.
        model.to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, dev


def _patch_tokens(model, x):
    """(B, Hp*Wp, D) patch tokens, robust to hub output conventions."""
    import torch
    import torch.nn.functional as F
    out = model.forward_features(x)
    if isinstance(out, dict):
        if "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"]
        t = out.get("x_prenorm", out.get("x"))
        t = F.layer_norm(t, t.shape[-1:])
        n_extra = t.shape[1] - (x.shape[-2] // _PATCH) * (x.shape[-1] // _PATCH)
        return t[:, n_extra:]
    n_extra = out.shape[1] - (x.shape[-2] // _PATCH) * (x.shape[-1] // _PATCH)
    return out[:, n_extra:]


def extract_session_features(output_dir: Path, frames_dir: Path,
                             cfg: Optional[dict] = None,
                             log=logger.info,
                             cache_dir: Optional[Path] = None,
                             frame_list_path: Optional[Path] = None) -> Path:
    """Run DINOv3 over every keyframe in frame_list.json; cache the
    L2-normalized, PCA-compressed patch grids. Returns the cache dir.
    Idempotent: an existing complete cache is reused."""
    import torch
    from PIL import Image

    cfg = cfg or {}
    pca_dim = int(cfg.get("pca_dim", 64))
    batch = int(cfg.get("batch", 6))
    output_dir = Path(output_dir)
    frames_dir = Path(frames_dir)
    cache = Path(cache_dir) if cache_dir else _cache_dir(output_dir)
    meta_p = cache / "meta.json"
    feats_p = cache / "features_f16.npy"
    if meta_p.exists() and feats_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            if meta.get("complete"):
                log(f"[dino] cache reused: {meta['n_frames']} frames, "
                    f"{meta['hp']}x{meta['wp']}x{meta['pca_dim']}")
                return cache
        except Exception:  # noqa: BLE001
            pass

    frame_files: List[str] = json.loads(
        (Path(frame_list_path) if frame_list_path
         else output_dir / "frame_list.json").read_text())
    if not frame_files:
        raise RuntimeError("frame_list.json is empty — no keyframes to encode")
    # target size: full keyframe resolution snapped to the patch grid
    with Image.open(frames_dir / frame_files[0]) as im0:
        w0, h0 = im0.size
    ht = max(_PATCH, (h0 // _PATCH) * _PATCH)
    wt = max(_PATCH, (w0 // _PATCH) * _PATCH)
    hp, wp = ht // _PATCH, wt // _PATCH

    model, dev = load_dinov3(cfg)
    log(f"[dino] encoding {len(frame_files)} keyframes at {ht}x{wt} "
        f"(grid {hp}x{wp}) on {dev}")
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def _load_batch(files):
        arrs = []
        for f in files:
            with Image.open(frames_dir / f) as im:
                im = im.convert("RGB").resize((wt, ht), Image.BILINEAR)
                a = np.asarray(im, np.float32) / 255.0
            arrs.append((a - _IMAGENET_MEAN) / _IMAGENET_STD)
        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
        return x.to(dev, torch.bfloat16 if dev == "cuda" else torch.float32)

    # pass 1: raw tokens to a temp f16 memmap at full embed dim is too big —
    # instead stream twice: first fit PCA on a token subsample, then project.
    rng = np.random.default_rng(0)
    sub_tokens = []
    tokens_per_frame = max(1, 60_000 // len(frame_files))
    with torch.no_grad():
        for i in range(0, len(frame_files), batch):
            x = _load_batch(frame_files[i:i + batch])
            t = _patch_tokens(model, x).float()          # (B, HW, D)
            t = torch.nan_to_num(t, 0.0, 0.0, 0.0)       # fp16 overflow on
            t = torch.nn.functional.normalize(t, dim=-1)  # degenerate frames
            t = t.cpu().numpy()
            for b in range(t.shape[0]):
                idx = rng.choice(t.shape[1],
                                 min(tokens_per_frame, t.shape[1]),
                                 replace=False)
                sub_tokens.append(t[b, idx])
    # float64 + finite rows only: f32 SVD on NaN-tainted tokens does not
    # converge (killed pose-fm on pccr_v1 — blurry/black frames overflow
    # the fp16 forward)
    sub = np.concatenate(sub_tokens).astype(np.float64)
    finite = np.isfinite(sub).all(axis=1)
    if not finite.all():
        logger.warning("[dino] %d/%d PCA tokens non-finite — dropped",
                       int((~finite).sum()), len(sub))
        sub = sub[finite]
    # a normalized real token has unit norm — all-zero rows mean the forward
    # produced NaN and nan_to_num silenced it. That is a FAILURE, not data.
    nz = np.linalg.norm(sub, axis=1) > 0.5
    if nz.mean() < 0.9:
        raise RuntimeError(
            f"[dino] {int((~nz).sum())}/{len(sub)} tokens are ZERO — the "
            f"encoder forward is producing NaN (dtype/overflow); refusing "
            f"to fit PCA on garbage (nothing fails silently)")
    sub = sub[nz]
    if len(sub) < pca_dim * 4:
        raise RuntimeError(f"[dino] only {len(sub)} finite tokens — "
                           f"cannot fit PCA (nothing fails silently)")
    mu = sub.mean(axis=0)
    # PCA via SVD on the centered subsample
    _u, _s, vt = np.linalg.svd(sub - mu, full_matrices=False)
    basis = vt[:pca_dim].astype(np.float32)              # (pca_dim, D)
    log(f"[dino] PCA fitted on {len(sub):,} tokens → {pca_dim} dims "
        f"({float((_s[:pca_dim] ** 2).sum() / (_s ** 2).sum()) * 100:.1f}% var)")

    feats = np.lib.format.open_memmap(
        feats_p, mode="w+", dtype=np.float16,
        shape=(len(frame_files), hp, wp, pca_dim))
    with torch.no_grad():
        for i in range(0, len(frame_files), batch):
            x = _load_batch(frame_files[i:i + batch])
            t = _patch_tokens(model, x).float()
            t = torch.nan_to_num(t, 0.0, 0.0, 0.0)
            t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
            proj = (t - mu) @ basis.T
            proj /= np.maximum(
                np.linalg.norm(proj, axis=-1, keepdims=True), 1e-8)
            feats[i:i + x.shape[0]] = proj.reshape(
                x.shape[0], hp, wp, pca_dim).astype(np.float16)
    feats.flush()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    frame_nums = [int(Path(f).stem) for f in frame_files]
    meta = {"complete": True, "n_frames": len(frame_files),
            "frame_nums": frame_nums, "hp": hp, "wp": wp,
            "img_h": ht, "img_w": wt, "pca_dim": pca_dim,
            "model": str(cfg.get("model") or "dinov3_vitl16"),
            "elapsed_s": round(time.time() - t0, 1)}
    np.save(cache / "pca_mu.npy", mu)
    np.save(cache / "pca_basis.npy", basis)
    meta_p.write_text(json.dumps(meta))
    log(f"[dino] cache written: {feats_p.stat().st_size / 1e6:.0f} MB "
        f"({meta['elapsed_s']}s)")
    return cache


class FeatureCache:
    """Lazy reader over the per-session feature cache."""

    def __init__(self, output_dir: Path, cache_dir: Optional[Path] = None):
        cache = Path(cache_dir) if cache_dir else _cache_dir(output_dir)
        self.meta = json.loads((cache / "meta.json").read_text())
        self.feats = np.load(cache / "features_f16.npy", mmap_mode="r")
        self.hp, self.wp = int(self.meta["hp"]), int(self.meta["wp"])
        self.dim = int(self.meta["pca_dim"])
        self.frame_index: Dict[int, int] = {
            int(f): i for i, f in enumerate(self.meta["frame_nums"])}

    def has(self, frame_num: int) -> bool:
        return int(frame_num) in self.frame_index

    def grid(self, frame_num: int) -> np.ndarray:
        """(hp, wp, dim) float32 grid for one keyframe."""
        return np.asarray(
            self.feats[self.frame_index[int(frame_num)]], np.float32)

    def sample(self, grid: np.ndarray, v_norm: np.ndarray,
               u_norm: np.ndarray) -> np.ndarray:
        """Bilinear sample at normalized image coords (v=row/H, u=col/W).
        Returns (N, dim) L2-normalized features."""
        r = np.clip(v_norm * self.hp - 0.5, 0, self.hp - 1)
        c = np.clip(u_norm * self.wp - 0.5, 0, self.wp - 1)
        r0 = np.floor(r).astype(np.int64)
        c0 = np.floor(c).astype(np.int64)
        r1 = np.minimum(r0 + 1, self.hp - 1)
        c1 = np.minimum(c0 + 1, self.wp - 1)
        fr = (r - r0)[:, None]
        fc = (c - c0)[:, None]
        g = grid
        out = (g[r0, c0] * (1 - fr) * (1 - fc) + g[r0, c1] * (1 - fr) * fc
               + g[r1, c0] * fr * (1 - fc) + g[r1, c1] * fr * fc)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
        return out


# ── session geometry helpers (shared conventions with pose_refine) ────────

def load_session_cameras(output_dir: Path):
    """(poses c2w (N,4,4), frame_nums list, Ks (N,3,3)) from the session's
    camera_poses/camera_frames/intrinsic txt trio."""
    output_dir = Path(output_dir)
    poses = [np.array([float(x) for x in ln.split()],
                      np.float64).reshape(4, 4)
             for ln in (output_dir / "camera_poses.txt")
             .read_text().splitlines() if len(ln.split()) == 16]
    frames = [int(float(x)) for x in
              (output_dir / "camera_frames.txt").read_text().split()]
    Ks = []
    for ln in (output_dir / "intrinsic.txt").read_text().splitlines():
        v = [float(x) for x in ln.split()]
        if len(v) == 4:
            Ks.append(np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]],
                               np.float64))
    if not (len(poses) == len(frames) == len(Ks)):
        raise RuntimeError(f"camera trio mismatch: {len(poses)} poses / "
                           f"{len(frames)} frames / {len(Ks)} intrinsics")
    return np.stack(poses), frames, np.stack(Ks)


def calibrate_provenance_grid(xyz: np.ndarray, frame_global: np.ndarray,
                              pixel_row: np.ndarray, pixel_col: np.ndarray,
                              poses: np.ndarray, frames: List[int],
                              Ks: np.ndarray, log=logger.info,
                              n_sample: int = 20_000) -> Tuple[int, int, float]:
    """Self-calibrate the (H, W) grid the per-point pixel provenance lives on:
    project a sample of points into their OWN source frames with the session
    cameras and pick the candidate grid that explains the stored pixels.
    Also validates poses+K+provenance coherence (returns median px error)."""
    fidx = {f: k for k, f in enumerate(frames)}
    rng = np.random.default_rng(0)
    sel = rng.choice(len(xyz), min(n_sample, len(xyz)), replace=False)
    ks = np.array([fidx.get(int(f), -1) for f in frame_global[sel]])
    ok = ks >= 0
    sel, ks = sel[ok], ks[ok]
    P = xyz[sel]
    w2c = np.linalg.inv(poses[ks])
    pc = np.einsum("nij,nj->ni", w2c[:, :3, :3], P) + w2c[:, :3, 3]
    z = pc[:, 2]
    front = z > 1e-6
    sel, ks, pc, z = sel[front], ks[front], pc[front], z[front]
    K = Ks[ks]
    u = K[:, 0, 0] * pc[:, 0] / z + K[:, 0, 2]
    v = K[:, 1, 1] * pc[:, 1] / z + K[:, 1, 2]
    # intrinsics reference resolution ≈ 2*principal point
    Kw = int(round(float(np.median(Ks[:, 0, 2])) * 2))
    Kh = int(round(float(np.median(Ks[:, 1, 2])) * 2))
    row_max = int(pixel_row.max()) + 1
    col_max = int(pixel_col.max()) + 1
    cands = {(Kh, Kw), (row_max, col_max)}
    best = None
    for (Hg, Wg) in cands:
        if Hg < row_max or Wg < col_max:
            continue
        su, sv = Wg / Kw, Hg / Kh
        du = u * su - (pixel_col[sel] + 0.5)
        dv = v * sv - (pixel_row[sel] + 0.5)
        err = float(np.median(np.hypot(du, dv)))
        if best is None or err < best[2]:
            best = (Hg, Wg, err)
    if best is None:
        raise RuntimeError("no provenance grid candidate fits the stored "
                           "pixel_row/col range")
    log(f"[dino] provenance grid {best[0]}x{best[1]} "
        f"(median reprojection {best[2]:.2f} px on {len(sel):,} samples)")
    return best


def layout_ransac(rc_a: np.ndarray, rc_b: np.ndarray,
                  tol_cells: float = 2.5, iters: int = 120,
                  seed: int = 0) -> np.ndarray:
    """PERCEPTUAL-ALIASING KILLER (USER FINDING 2026-09-04: the tunnel's
    repeating segments made appearance-only pairs 'parecidos pero no lo
    mismo'). A TRUE same-place pair's patch matches follow one coherent
    similarity transform between the two images; look-alike DIFFERENT
    places match scattered. RANSAC similarity fit over matched patch
    coords (r,c); returns the boolean inlier mask (empty coherence → all
    False)."""
    n = len(rc_a)
    if n < 6:
        return np.zeros(n, bool)
    A = np.asarray(rc_a, np.float64)
    B = np.asarray(rc_b, np.float64)
    rng = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 2, replace=False)
        a0, a1 = A[idx]
        b0, b1 = B[idx]
        va, vb = a1 - a0, b1 - b0
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-6 or nb < 1e-6:
            continue
        s = nb / na
        if not (0.5 <= s <= 2.0):        # same place → similar scale
            continue
        ca, sa = va / na, np.array([-va[1], va[0]]) / na
        cb_ = vb / nb
        cosr = float(ca @ cb_)
        sinr = float(sa @ cb_)
        R = np.array([[cosr, -sinr], [sinr, cosr]])
        pred = (A - a0) @ R.T * s + b0
        inl = np.linalg.norm(pred - B, axis=1) <= tol_cells
        if inl.sum() > best.sum():
            best = inl
    return best if best.sum() >= 12 else np.zeros(n, bool)


def sequence_coherent(S: np.ndarray, i: int, j: int,
                      radius: int = 2, drop: float = 0.06,
                      min_run: int = 3) -> bool:
    """A TRUE revisit matches in a RUN of neighbours — in EITHER direction:
    (i+δ, j+δ) when the walk repeats the same way, (i+δ, j−δ) when it
    RETURNS (the user's door reappearing on the way back). Aliased
    singletons pass neither. S = global-descriptor similarity."""
    n = len(S)
    base = S[i, j]
    best = 0
    for sign in (1, -1):
        run = 0
        for d in range(-radius, radius + 1):
            a, b = i + d, j + sign * d
            if 0 <= a < n and 0 <= b < n and S[a, b] >= base - drop:
                run += 1
        best = max(best, run)
    return best >= min_run
