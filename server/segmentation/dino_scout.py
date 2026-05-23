"""
DINO Frame Scout — Find keyframes containing a specific object.
================================================================

Given a SAM3 mask from one frame, uses DINOv2 features to identify
which other keyframes contain the same object. This avoids running
SAM3 propagation on ALL keyframes for objects that only appear in
a subset of the scan.

Model: DINOv2 ViT-S/14 (Apache 2.0, ~80MB, ~50ms/frame on CPU)

Usage:
    from segmentation.dino_scout import DINOScout
    scout = DINOScout()
    frames = scout.find_object_frames(mask, source_frame, all_keyframes)

Authors: Hernán Barreto — Ingerop IN3
"""

import gc
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("DINOScout")

# ── Feature cache for keyframes ──────────────────────────────────────

_FEATURES_CACHE_FILE = "dino_features.npz"


class DINOScout:
    """DINOv2-based object re-identification across keyframes."""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._transform = None

    def _ensure_model(self):
        """Lazy load DINOv2 model."""
        if self.model is not None:
            return

        logger.info(f"Loading {self.model_name} on {self.device}...")
        t0 = time.time()

        # Disable xFormers on CPU — dinov2 attention.py checks this at import
        import os, sys
        if self.device == "cpu":
            os.environ["XFORMERS_DISABLED"] = "1"
            # Purge cached dinov2 modules so the env var takes effect
            to_remove = [k for k in sys.modules if "dinov2" in k]
            for k in to_remove:
                del sys.modules[k]

        self.model = torch.hub.load(
            "facebookresearch/dinov2", self.model_name,
            pretrained=True,
        )
        self.model = self.model.to(self.device).eval()

        # DINOv2 preprocessing
        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"DINOv2 loaded in {time.time() - t0:.1f}s")

    def unload(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._transform = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("DINOv2 unloaded")

    @torch.no_grad()
    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        """Extract CLS token features from an image. Returns (1, D) tensor."""
        self._ensure_model()
        pixel_values = self._transform(image).unsqueeze(0).to(self.device)
        features = self.model(pixel_values)  # (1, D)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def _extract_masked_features(self, image: Image.Image,
                                  mask: np.ndarray) -> torch.Tensor:
        """Extract features from the masked object region only.
        
        Crops the image to the mask bounding box, darkens non-object pixels,
        then extracts DINOv2 features.
        """
        self._ensure_model()
        img_np = np.array(image)

        # Find bounding box of mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return self._extract_features(image)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add 10% padding
        h, w = img_np.shape[:2]
        pad_r = max(int((rmax - rmin) * 0.1), 5)
        pad_c = max(int((cmax - cmin) * 0.1), 5)
        rmin = max(0, rmin - pad_r)
        rmax = min(h - 1, rmax + pad_r)
        cmin = max(0, cmin - pad_c)
        cmax = min(w - 1, cmax + pad_c)

        # Crop and darken non-object pixels
        crop = img_np[rmin:rmax+1, cmin:cmax+1].copy()
        mask_crop = mask[rmin:rmax+1, cmin:cmax+1]
        crop[~mask_crop] = crop[~mask_crop] // 4  # darken background

        crop_pil = Image.fromarray(crop)
        pixel_values = self._transform(crop_pil).unsqueeze(0).to(self.device)
        features = self.model(pixel_values)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def precompute_keyframe_features(self, keyframe_paths: List[str],
                                      save_dir: str = None) -> np.ndarray:
        """Precompute DINOv2 CLS features for all keyframes.
        
        Args:
            keyframe_paths: List of paths to keyframe JPEGs
            save_dir: If provided, save features to dino_features.npz
            
        Returns:
            (N, D) float32 array of L2-normalized features
        """
        self._ensure_model()

        # Check cache
        if save_dir:
            cache_path = Path(save_dir) / _FEATURES_CACHE_FILE
            if cache_path.exists():
                cached = np.load(str(cache_path))
                if cached["features"].shape[0] == len(keyframe_paths):
                    logger.info(f"Loaded cached DINO features ({len(keyframe_paths)} frames)")
                    return cached["features"]

        logger.info(f"Precomputing DINO features for {len(keyframe_paths)} keyframes...")
        t0 = time.time()

        features_list = []
        for i, fp in enumerate(keyframe_paths):
            img = Image.open(fp).convert("RGB")
            feat = self._extract_features(img)
            features_list.append(feat.cpu().numpy())

            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(keyframe_paths)} frames processed")

        features = np.concatenate(features_list, axis=0)  # (N, D)

        if save_dir:
            cache_path = Path(save_dir) / _FEATURES_CACHE_FILE
            np.savez_compressed(str(cache_path), features=features)
            logger.info(f"Saved DINO features to {cache_path}")

        elapsed = time.time() - t0
        logger.info(f"Precomputed {len(features)} features in {elapsed:.1f}s "
                     f"({elapsed/len(features)*1000:.0f}ms/frame)")
        return features

    def find_object_frames(
        self,
        mask: np.ndarray,
        source_frame_path: str,
        all_keyframe_paths: List[str],
        threshold: float = 0.5,
        precomputed_features: np.ndarray = None,
        min_frames: int = 3,
        max_frames: int = 0,
    ) -> List[str]:
        """Find keyframes where the same object likely appears.
        
        Args:
            mask: Binary mask (H, W) of the object in the source frame
            source_frame_path: Path to the frame where mask was generated
            all_keyframe_paths: Paths to all keyframe JPEGs
            threshold: Cosine similarity threshold (0-1)
            precomputed_features: Optional (N, D) features from precompute_keyframe_features
            min_frames: Minimum frames to return (lower threshold if needed)
            max_frames: Max frames to return (0 = no limit)
            
        Returns:
            List of keyframe paths where the object likely appears (includes source)
        """
        self._ensure_model()
        t0 = time.time()

        # Extract object features from masked region
        source_img = Image.open(source_frame_path).convert("RGB")
        object_features = self._extract_masked_features(source_img, mask)  # (1, D)

        # Get features for all keyframes
        if precomputed_features is not None:
            all_features = torch.from_numpy(precomputed_features).to(self.device)
        else:
            # Compute on-the-fly
            features_list = []
            for fp in all_keyframe_paths:
                img = Image.open(fp).convert("RGB")
                feat = self._extract_features(img)
                features_list.append(feat)
            all_features = torch.cat(features_list, dim=0)  # (N, D)

        # Cosine similarity
        similarities = (object_features @ all_features.T).squeeze(0)  # (N,)
        sim_np = similarities.cpu().numpy()

        # Select frames above threshold
        above = np.where(sim_np >= threshold)[0]

        # Ensure source frame is always included
        source_name = Path(source_frame_path).name
        source_idx = None
        for i, fp in enumerate(all_keyframe_paths):
            if Path(fp).name == source_name:
                source_idx = i
                break

        result_indices = set(above.tolist())
        if source_idx is not None:
            result_indices.add(source_idx)

        # If too few, lower threshold progressively
        if len(result_indices) < min_frames:
            sorted_indices = np.argsort(sim_np)[::-1]
            for idx in sorted_indices[:min_frames]:
                result_indices.add(int(idx))

        # If max_frames set, take top-N
        if max_frames > 0 and len(result_indices) > max_frames:
            scored = [(i, sim_np[i]) for i in result_indices]
            scored.sort(key=lambda x: x[1], reverse=True)
            result_indices = {i for i, _ in scored[:max_frames]}

        result_paths = [all_keyframe_paths[i] for i in sorted(result_indices)]

        elapsed = time.time() - t0
        print(
            f"[DINOScout] {len(result_paths)}/{len(all_keyframe_paths)} frames "
            f"(threshold={threshold}, {elapsed:.1f}s)"
        )
        if len(sim_np) > 0:
            print(
                f"[DINOScout] Similarity range: [{sim_np.min():.3f}, {sim_np.max():.3f}], "
                f"mean={sim_np.mean():.3f}"
            )

        return result_paths
