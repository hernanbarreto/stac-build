"""MoGe-3 (microsoft/MoGe, ``moge.model.v3``) as F6's mono-detail model, in place of PointDiT (USER 2026-11-22:
"y moge 3 como lo tenemos seteado ahora en lugar de pointdit" / "reemplazar pointdit por moge 3 con correccion de
orden 10" / "moge 3 si sale bien lo pasamos al pod, no importa que este ahora en le disco").

The interface :func:`precision.mono_detail.run_stage` asks of :class:`precision.pointdit_runner.PointDiTRunner`:
``load()``, ``depth_batch(images, size) -> (z [B, h, w], valid [B, h, w])``, ``footprint()``, ``timing()``,
``card()``. The stage aligns every tile's depth AFFINELY to the bent Omega depth whatever the model, so MoGe-3's
metric depth enters exactly where PointDiT's affine-invariant one did.

THE CALL: ViPE / MetricTrail's MoGe call (vipe/metrictrail/depth.py) — ``resolution_level`` and ``num_tokens``
as configured (9 / None: MoGe-3's most tokens, 3600), ``force_projection=True``, ``apply_mask=False`` — plus
MoGe-3's sparse 3D refinement, ``refine_steps`` (10, the setting of the 2026-11-22 tests: the change per step
falls from 0.07-0.13 % of the depth at step 1 to 0.025-0.028 % at step 10 on bufferStop, never diverging).
``fov_x`` is the crop's horizontal field of view from F5's focal (``set_focal``): MoGe-3 is then conditioned on
the camera the chain measured instead of guessing it. The valid pixels: MoGe's mask (its probability over 0.5),
finite, positive.

THE CODE IS NOT VENDORED (the test): a checkout of microsoft/MoGe (``code_dir``) and the two packages MoGe-3 adds
(``site_dir``: utils3d_moge, flex_gemm built with its CUDA kernels for this env's python and torch) are put on
``sys.path`` when the runner loads; the checkpoint is the HuggingFace repo's ``model.pt`` (HF_HOME's cache)."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[MoGe-3]"


class MoGe3Error(RuntimeError):
    pass


class MoGe3Runner:
    """One loaded MoGe-3. ``mcfg`` is the mono_detail section (its ``moge3`` part, its ``seed``)."""

    def __init__(self, mcfg, log: Callable = print, device: Optional[str] = None):
        self.cfg = mcfg
        self.mc = mcfg.moge3
        self.log = log
        self.seed = int(mcfg.seed)
        self.device = device or "cuda"
        self._model = None
        self._numerics: Optional[dict] = None
        self._card: Optional[dict] = None
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._k90 = 0                       # quarter turns (np.rot90's k) that set the STORED frames upright
        self._checkpoint: Optional[str] = None

    def set_focal(self, fx_px: float, fy_px: float) -> None:
        """F5's focals on the grid the crops are cut from (the undistorted native frame), in pixels."""
        for v in (fx_px, fy_px):
            if not (v and math.isfinite(v) and v > 0):
                raise MoGe3Error(f"MoGe-3 needs the camera's focals: got {fx_px!r}, {fy_px!r}")
        self._fx, self._fy = float(fx_px), float(fy_px)

    def set_upright(self, cv2_flag: Optional[str]) -> None:
        """The scan's stored frames may be sideways (intake.frames_manifest.display_rotation: bufferStop's video
        carries a 90° rotation its frames were stored without). MoGe-3 learned from upright images: every crop
        is turned upright before the forward pass and its depth turned back (an exact permutation)."""
        ks = {None: 0, "ROTATE_90_CLOCKWISE": -1, "ROTATE_90_COUNTERCLOCKWISE": 1, "ROTATE_180": 2}
        if cv2_flag not in ks:
            raise MoGe3Error(f"unknown frame rotation {cv2_flag!r}")
        self._k90 = ks[cv2_flag]

    # ── loading ──
    def load(self) -> "MoGe3Runner":
        if self._model is not None:
            return self
        import torch
        code, site = Path(self.mc.code_dir), Path(self.mc.site_dir)
        for p, what in ((code / "moge" / "model" / "v3.py", "a checkout of microsoft/MoGe"),
                        (site / "flex_gemm", "flex_gemm built for this python (MoGe-3's refiner)"),
                        (site / "utils3d_moge", "utils3d_moge (MoGe-3's geometry helpers)")):
            if not p.exists():
                raise MoGe3Error(f"{p} is missing — {what}")
        for p in (str(site), str(code)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from huggingface_hub import hf_hub_download
        from moge.model.v3 import MoGeModel
        t0 = time.time()
        ckpt = hf_hub_download(repo_id=str(self.mc.checkpoint), filename="model.pt")
        model = MoGeModel.from_pretrained(ckpt)
        if not hasattr(model, "refiner"):
            raise MoGe3Error(f"the MoGe-3 checkpoint {self.mc.checkpoint} carries no refiner")
        model = model.float().to(self.device).eval()
        for q in model.parameters():
            q.requires_grad_(False)
        self._model = model
        self._checkpoint = str(ckpt)
        self.log(f"{LOG_TAG} {self.mc.checkpoint} loaded on {self.device} in {time.time() - t0:.0f} s "
                 f"(refine_steps {int(self.mc.refine_steps)}, resolution_level {int(self.mc.resolution_level)}, "
                 f"fp16 autocast {bool(self.mc.use_fp16)})")
        return self

    # ── inference ──
    def depth_batch(self, images: Sequence[np.ndarray], size: Optional[Tuple[int, int]] = None,
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """(z [B, h, w] float32 in metres, valid [B, h, w]) for B RGB crops of ONE size (uint8 or float 0..1,
        H x W x 3) of the native frame, run at ``size`` = (h, w) (the crops' own size when None)."""
        import torch
        self.load()
        if self._fx is None:
            raise MoGe3Error("set_focal() was not called — MoGe-3 would guess the camera")
        if len(images) == 0:
            raise MoGe3Error("an empty batch")
        imgs = [np.asarray(im) for im in images]
        for img in imgs:
            if img.ndim != 3 or img.shape[2] != 3 or img.shape != imgs[0].shape:
                raise MoGe3Error(f"one H x W x 3 size per batch: {img.shape} next to {imgs[0].shape}")
        Hc, Wc = imgs[0].shape[:2]
        h, w = (Hc, Wc) if size is None else (int(size[0]), int(size[1]))
        quarter = self._k90 % 2 == 1
        if self._k90:                                   # upright for the model (an exact permutation)
            imgs = [np.ascontiguousarray(np.rot90(im, k=self._k90)) for im in imgs]
            if quarter:
                Hc, Wc, h, w = Wc, Hc, w, h
        # the crop's horizontal field of view on F5's camera (a resize changes the pixels, not the angle); a
        # quarter turn makes the stored frame's vertical axis the upright one's horizontal
        fov_x = math.degrees(2.0 * math.atan(Wc / (2.0 * (self._fy if quarter else self._fx))))
        x = torch.stack([torch.from_numpy(im.astype(np.float32) / (255 if im.dtype == np.uint8 else 1))
                         .permute(2, 0, 1) for im in imgs]).to(self.device)     # uint8 → 0..1 (PointDiT's own)
        if (h, w) != (Hc, Wc):
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        from repro import deterministic_torch
        with torch.inference_mode(), deterministic_torch(self.seed) as numerics:
            if self._numerics is None:
                self._numerics = dict(numerics)
            out = self._model.infer(x, num_tokens=None, resolution_level=int(self.mc.resolution_level),
                                    force_projection=True, apply_mask=False, fov_x=fov_x,
                                    refine_steps=int(self.mc.refine_steps), use_fp16=bool(self.mc.use_fp16))
        z = out["depth"].float()
        if z.dim() == 2:
            z = z[None]
        m = out.get("mask")
        valid = torch.isfinite(z) & (z > 0)
        if m is not None:
            m = m.bool()
            valid &= (m if m.dim() == 3 else m[None])
        z = torch.where(valid, z, torch.zeros_like(z))
        z_np, v_np = z.cpu().numpy().astype(np.float32), valid.cpu().numpy()
        if self._k90:                                   # back to the stored frame's orientation
            z_np = np.ascontiguousarray(np.rot90(z_np, k=-self._k90, axes=(1, 2)))
            v_np = np.ascontiguousarray(np.rot90(v_np, k=-self._k90, axes=(1, 2)))
        return z_np, v_np

    def depth(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        z, valid = self.depth_batch([image], size)
        return z[0], valid[0]

    def card(self) -> Optional[dict]:
        if self._card is None and self.device.startswith("cuda"):
            from repro import card_identity
            self._card = card_identity(0)
        return self._card

    def footprint(self) -> dict:
        return {"device": self.device, "model": f"MoGe-3 {self.mc.checkpoint}", "checkpoint": self._checkpoint,
                "refine_steps": int(self.mc.refine_steps), "resolution_level": int(self.mc.resolution_level),
                "fp16_autocast": bool(self.mc.use_fp16), "seed": self.seed, "card": self.card(),
                "numerics": self._numerics}

    def timing(self) -> dict:
        import torch
        d: dict = {}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            d["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1000 ** 3), 2)
        return d
