"""PointDiT runner — Phase 1 of the mono-detail work (claude_stac.txt, 2026-10-04).

PointDiT (google-research/pointdit, ICML 2026, Apache-2.0; `third_party/pointdit`, pinned submodule)
is a pixel-space diffusion transformer that denoises a 3-D point map from one image, conditioned on
a frozen DINOv3 encoder. Its output is AFFINE-INVARIANT (zero-centred, mean-normalised per image):
it knows where the edges are and which side of them each pixel lies on, not the metre. In this
pipeline it only ever REFINES the detail of a depth map that already carries the metric (Omega
bent to F5 on the DA3 gauge); it never touches scale, gauge, poses, intrinsics or the Ω/DA3 pass.

The runner:
  * builds the model from the vendored code in THIS process (env ``da3``, torch untouched —
    verified 2026-10-04: PointDiT-H + DINOv3 ViT-H+/16 runs in 9.4 GB of VRAM, 0.15-0.5 s per
    384x688 frame);
  * loads the released checkpoint (its EMA weights, as the vendor's evaluation does) and the gated
    DINOv3 encoder AFTER construction, strictly — a missing encoder is a clear error, never a
    randomly initialised one;
  * verifies every checkpoint's sha256 against the 8-hex prefix in its file name (the vendor's own
    convention), once per file (a sidecar ``<file>.sha256`` remembers the digest with the file's
    size and mtime);
  * runs the ODE from ZEROS (``generate_noise_scale`` 0 — deterministic: the same input gives the
    same output bit for bit) for ``steps`` Euler steps (default 2);
  * returns the point map at the size it was run at plus the validity mask: a pixel whose output
    norm exceeds ``norm_max`` (2.9 in the paper: the sky dome sits at 3.0) is invalid.

Tests run with the model MOCKED (a callable in place of ``generate``); the GPU test skips itself
when the weights are not on the machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

LOG_TAG = "[pointdit]"
PATCH = 16
TRAIN_TOKENS = (512 // PATCH) ** 2         # the 32x32-token budget the 512 checkpoints were trained at
_SHA_RE = re.compile(r"-([0-9a-f]{8})\.pth$")

# the released 512 checkpoints and the encoder each one was trained with (MODELS.md of the vendor)
MODELS: Dict[str, Dict[str, str]] = {
    "H": {"arch": "PointDiT-H/16", "features": "dinov3_vith16plus",
          "checkpoint": "pointdith-512-mixdata-nodinov3-cb01dd3b.pth",
          "dinov3": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"},
    "L": {"arch": "PointDiT-L/16", "features": "dinov3_vitl16",
          "checkpoint": "pointditl-512-mixdata-nodinov3-240c1a4f.pth",
          "dinov3": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"},
}


class PointDiTError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    repo: Path          # the vendored checkout (model.py, denoiser.py, third_party/dinov3)
    checkpoint: Path    # the released PointDiT weights (EMA inside)
    dinov3: Path        # the gated encoder weights
    dinov3_repo: Path   # the DINOv3 hub code


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_paths(mcfg) -> Paths:
    """The files of ``mcfg.model`` under ``mcfg.repo_dir`` / ``mcfg.weights_dir`` / ``mcfg.dinov3_dir``
    (relative paths against the repository root)."""
    if mcfg.model not in MODELS:
        raise PointDiTError(f"precision.mono_detail.model must be one of {sorted(MODELS)}, got {mcfg.model!r}")
    m = MODELS[mcfg.model]
    root = repo_root()

    def _abs(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    repo = _abs(mcfg.repo_dir)
    return Paths(repo=repo, checkpoint=_abs(mcfg.weights_dir) / m["checkpoint"],
                 dinov3=_abs(mcfg.dinov3_dir) / m["dinov3"], dinov3_repo=repo / "third_party" / "dinov3")


def sha256_prefix_of_name(path: Path) -> str:
    m = _SHA_RE.search(path.name)
    if not m:
        raise PointDiTError(f"{path.name} carries no sha256 prefix in its name (expected '-<8 hex>.pth')")
    return m.group(1)


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_checkpoint(path: Path, log: Callable = print) -> str:
    """Raise unless the file's sha256 starts with the prefix in its name. The digest is computed
    once; ``<file>.sha256`` keeps it with the size and mtime it was computed for."""
    path = Path(path)
    if not path.is_file():
        raise PointDiTError(f"checkpoint {path} is missing")
    want = sha256_prefix_of_name(path)
    st = path.stat()
    side = path.with_name(path.name + ".sha256")
    digest = None
    if side.exists():
        try:
            rec = json.loads(side.read_text())
            if int(rec.get("size", -1)) == st.st_size and float(rec.get("mtime", -1)) == st.st_mtime:
                digest = str(rec.get("sha256", ""))
        except (ValueError, OSError):
            digest = None
    if not digest:
        t0 = time.time()
        digest = file_sha256(path)
        try:
            side.write_text(json.dumps({"sha256": digest, "size": st.st_size, "mtime": st.st_mtime}))
        except OSError:
            pass
        log(f"{LOG_TAG} sha256 of {path.name} computed in {time.time() - t0:.0f} s")
    if not digest.startswith(want):
        raise PointDiTError(f"{path.name}: sha256 {digest[:16]}… does not start with the name's prefix {want} — "
                            f"the download is corrupt or renamed")
    return digest


def working_size(H: int, W: int, tokens: int = TRAIN_TOKENS) -> Tuple[int, int]:
    """The vendor's rule for an image of any size: resize with the aspect kept to the patch-aligned
    size whose token count is nearest ``tokens`` (never under one patch)."""
    f = ((tokens * PATCH ** 2) / float(H * W)) ** 0.5
    nH = max(PATCH, int(round(H * f / PATCH)) * PATCH)
    nW = max(PATCH, int(round(W * f / PATCH)) * PATCH)
    return nH, nW


def _args(arch: str, features: str, steps: int):
    return types.SimpleNamespace(
        model=arch, img_size=512, attn_dropout=0.0, proj_dropout=0.0, attention_type="torch",
        feature_embedding_type=features, dinov3_use_intermediate_layers=True, dinov3_num_intermediate_layers=4,
        feature_embedding_lr_scale=0.0, P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0,
        ema_decay1=0.9999, ema_decay2=0.9999, num_sampling_steps=int(steps), generate_noise_scale=0.0,
        sample_t_eps=0.0)


class PointDiTRunner:
    """One loaded PointDiT. ``generate`` may be injected (tests): a callable mapping an image tensor
    [B, 3, H, W] in 0..1 to a point map [B, 3, H, W]."""

    def __init__(self, mcfg, log: Callable = print, device: Optional[str] = None,
                 generate: Optional[Callable] = None):
        self.cfg = mcfg
        self.log = log
        self.steps = int(mcfg.steps)
        self.norm_max = float(mcfg.norm_max)
        self.device = device or ("cuda" if self._cuda() else "cpu")
        self._generate = generate
        self._model = None
        self.paths: Optional[Paths] = None if generate is not None else resolve_paths(mcfg)

    @staticmethod
    def _cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    # ── loading ──────────────────────────────────────────────────────

    def load(self) -> "PointDiTRunner":
        if self._generate is not None or self._model is not None:
            return self
        import torch
        p = self.paths
        for q, what in ((p.repo / "denoiser.py", "the PointDiT code (git submodule third_party/pointdit)"),
                        (p.dinov3_repo / "hubconf.py", "the DINOv3 hub code (git clone facebookresearch/dinov3 "
                                                        "third_party/pointdit/third_party/dinov3)")):
            if not q.exists():
                raise PointDiTError(f"{q} is missing — {what}")
        if not p.dinov3.is_file():
            raise PointDiTError(f"the DINOv3 encoder weights {p.dinov3} are missing — PointDiT would run on a "
                                f"randomly initialised encoder; the gated weights go to {p.dinov3.parent}")
        if bool(self.cfg.verify_sha256):
            verify_checkpoint(p.checkpoint, self.log)
            verify_checkpoint(p.dinov3, self.log)
        os.environ["DINOV3_REPO"] = str(p.dinov3_repo)
        os.environ["DINOV3_WEIGHTS_DIR"] = str(p.dinov3.parent)
        if str(p.repo) not in sys.path:
            sys.path.insert(0, str(p.repo))
        from denoiser import Denoiser  # noqa: E402 — the vendored code
        m = MODELS[self.cfg.model]
        t0 = time.time()
        model = Denoiser(_args(m["arch"], m["features"], self.steps))
        ck = torch.load(str(p.checkpoint), map_location="cpu", weights_only=False)
        state = dict(ck["model"])
        n_ema = 0
        for k, v in (ck.get("model_ema1") or {}).items():      # the vendor evaluates the EMA copy
            if k in state:
                state[k] = v; n_ema += 1
        res = model.load_state_dict(state, strict=False)
        missing = [k for k in res.missing_keys if "y_embedder" not in k]
        if missing or res.unexpected_keys:
            raise PointDiTError(f"checkpoint {p.checkpoint.name} does not match {m['arch']}: missing {missing[:5]}, "
                                f"unexpected {list(res.unexpected_keys)[:5]}")
        dv = torch.load(str(p.dinov3), map_location="cpu")
        model.net.y_embedder.load_state_dict(dv, strict=True)   # the frozen encoder, after construction
        model = model.to(self.device).eval()
        for q in model.parameters():
            q.requires_grad_(False)
        self._model = model
        self.log(f"{LOG_TAG} {m['arch']} + {m['features']} loaded on {self.device} ({n_ema} EMA tensors, "
                 f"{self.steps} steps) in {time.time() - t0:.0f} s")
        return self

    # ── inference ────────────────────────────────────────────────────

    def infer(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(point map [3, h, w] float32 in the model's normalised space, valid [h, w]) for one RGB image
        (uint8 or float 0..1, H x W x 3), run at ``size`` = (h, w) — multiples of 16 — or at the image's
        own size when it already is."""
        import torch
        self.load()
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise PointDiTError(f"the image must be H x W x 3, got {img.shape}")
        x = torch.from_numpy(img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)).permute(2, 0, 1)[None]
        H, W = int(x.shape[2]), int(x.shape[3])
        if size is None:
            size = (H, W)
        h, w = int(size[0]), int(size[1])
        if h % PATCH or w % PATCH or h < PATCH or w < PATCH:
            raise PointDiTError(f"PointDiT runs on multiples of {PATCH} px, asked {w}x{h}")
        x = x.to(self.device)
        if (h, w) != (H, W):
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        with torch.no_grad():
            if self._generate is not None:
                out = self._generate(x)
            elif self.device.startswith("cuda"):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self._model.generate(x)
            else:
                out = self._model.generate(x)
        P = out.float()[0].cpu().numpy().astype(np.float32)
        valid = np.linalg.norm(P, axis=0) <= self.norm_max
        return P, valid

    def depth(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(z [h, w] float32 — the point map's z, zero-centred and mean-normalised, i.e. depth up to an
        affine map — and its valid mask)."""
        P, valid = self.infer(image, size)
        return P[2], valid

    def footprint(self) -> dict:
        import torch
        d = {"device": self.device, "model": self.cfg.model, "steps": self.steps}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            d["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return d
