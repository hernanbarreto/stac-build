"""
Visual Novelty Frame Selector — Dual-mode keyframe selection.
=============================================================

Two methods available (configured via config.yaml frame_selection.method):

1. "dino" (default) — DINOv2 cosine similarity:
   Uses DINOv2 ViT-S/14 embeddings to measure visual similarity between frames.
   More robust than feature matching for construction scenes (handles repetitive
   textures, uniform surfaces, and semantic scene changes).

2. "hf_ratio" (legacy) — ORB-SLAM H/F ratio:
   Computes Homography vs Fundamental Matrix ratio to detect geometric parallax.
   Camera-agnostic but struggles with repetitive textures.

Pipeline position:
  Raw frames → Blur filter (FFT+Laplacian) → **Keyframe Selector** → Reconstruction

Usage:
    from frame_selector import select_keyframes
    result = select_keyframes("/path/to/frames", config)
    # result["selected_files"] → list of geometrically useful frames

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

import os
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#                   DINO COSINE SIMILARITY METHOD
# ═══════════════════════════════════════════════════════════════════

def _load_dino_model(model_name: str = "dinov2_vitb14", device: str = "cuda"):
    """
    Load DINOv2 model for embedding extraction.
    Uses ViT-B/14 by default (~86MB, same backbone MapAnything uses).
    On CPU: disables xFormers (CUDA-only) BEFORE import.
    """
    import torch
    import os
    import sys

    # MUST be set BEFORE torch.hub.load imports dinov2 modules.
    # dinov2/layers/attention.py checks this at import time (line 21).
    if device == "cpu":
        os.environ["XFORMERS_DISABLED"] = "1"
        # Invalidate cached dinov2 modules so env var takes effect
        to_remove = [k for k in sys.modules if "dinov2" in k]
        for k in to_remove:
            del sys.modules[k]

    # NOTE: torch.hub cache is left at its default (~/.cache/torch, local disk).
    # Pointing it at the /workspace network volume made the DINOv2 weight load
    # ~40s per run. Local disk loads in a few seconds; the cost is a one-time
    # re-clone from GitHub after a pod restart.

    print(f"[FrameSelector] Loading DINOv2 ({model_name}) on {device}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
    model = model.to(device).eval()
    print(f"[FrameSelector] DINOv2 ready")
    return model


def _get_dino_transform():
    """Standard DINOv2 image transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_embedding(model, img_path: str, transform, device: str):
    """Extract DINOv2 CLS token embedding from an image. Returns (384,) numpy array."""
    import torch
    from PIL import Image

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)  # (1, embed_dim)
    return emb[0].cpu().numpy()


def _extract_embeddings_batched(model, frames_dir, frame_files, transform, device,
                                batch_size: int = 32, num_workers: int = 4,
                                use_amp: bool = False) -> np.ndarray:
    """Compute L2-normalized DINOv2 embeddings for ALL frames, in batches.

    This is the fast path. Embeddings are independent of the (sequential)
    keyframe decisions, so we batch them: parallel CPU decode via THREADS
    (daemon-safe — the pipeline worker is a daemonic process, so DataLoader's
    child processes would crash), batched GPU forward, and a single device→host
    copy per batch instead of one per frame. Turns the batch-1 + per-frame
    `.cpu()` sync (where GPU overhead dominates and loses to CPU) into proper
    GPU throughput.

    Returns (N, D) float32. Rows that failed to decode are left as NaN.
    """
    import torch
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    frames_dir = Path(frames_dir)
    N = len(frame_files)

    def _decode(fname):
        try:
            img = Image.open(str(frames_dir / fname)).convert("RGB")
            return transform(img)  # (3, 224, 224)
        except Exception:
            return None

    embs = None
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for start in range(0, N, batch_size):
            chunk = frame_files[start:start + batch_size]
            tensors = list(pool.map(_decode, chunk))
            valid_local = [j for j, t in enumerate(tensors) if t is not None]

            if valid_local:
                batch = torch.stack([tensors[j] for j in valid_local]).to(device)
                with torch.no_grad():
                    if use_amp and device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(batch)
                    else:
                        out = model(batch)
                    out = torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy()
                if embs is None:
                    embs = np.full((N, out.shape[1]), np.nan, dtype=np.float32)
                for k, j in enumerate(valid_local):
                    embs[start + j] = out[k]

            done += len(chunk)
            if done % (batch_size * 10) < batch_size or done >= N:
                fps = done / max(time.time() - t0, 1e-6)
                print(f"  [embeddings {done}/{N}] {fps:.0f} img/s", flush=True)

    if embs is None:
        embs = np.full((N, 1), np.nan, dtype=np.float32)
    return embs


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


def _emb_cache_path(frames_dir: Path, model_name: str) -> Path:
    return Path(frames_dir) / f".dino_emb_{model_name}.npz"


def _load_embedding_cache(frames_dir: Path, frame_files: list, model_name: str):
    """Return cached (N,D) embeddings iff a cache for the EXACT frame list + model
    exists. The keyframe pass and the DA3-dense pass run DINO over the SAME frames
    (only the threshold differs), so without this they recompute every embedding
    twice. Embeddings depend only on the images → reuse is exact. Returns None on miss."""
    p = _emb_cache_path(Path(frames_dir), model_name)
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
        if list(z["files"]) == list(frame_files):
            emb = z["emb"]
            if emb.shape[0] == len(frame_files):
                return emb
    except Exception:
        pass
    return None


def _save_embedding_cache(frames_dir: Path, frame_files: list, model_name: str, embs) -> None:
    p = _emb_cache_path(Path(frames_dir), model_name)
    try:
        np.savez(p, emb=np.asarray(embs), files=np.array(list(frame_files), dtype=object))
    except Exception as e:
        print(f"  [embeddings] cache write failed: {e}")


def dino_select_keyframes(frames_dir: str, config: dict = None,
                          file_list: list = None, segment_id: int = None) -> Dict:
    """
    Select keyframes using DINOv2 cosine similarity.

    Embeddings for all frames are precomputed in batches (DINOv2, default
    ViT-B/14 → 768-dim), then each candidate is compared against the last
    accepted keyframe; cosine similarity below threshold → new keyframe.

    Args:
        frames_dir: Directory containing frame images (after blur filter)
        config: frame_selection config from config.yaml
        file_list: Optional explicit list of filenames
        segment_id: Optional segment index for labeling

    Returns:
        Same format as select_keyframes() for drop-in compatibility.
    """
    import torch

    if config is None:
        config = {}

    # Configuration
    model_name = config.get("dino_model", "dinov2_vitb14")
    threshold = config.get("dino_threshold", 0.85)
    device = config.get("dino_device", "cuda")
    force_interval = config.get("force_accept_interval", 30)

    # Fallback to CPU if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[FrameSelector] CUDA not available, falling back to CPU")

    frames_dir = Path(frames_dir)

    # Load frame list
    if file_list is not None:
        frame_files = file_list
    else:
        frame_files = _load_valid_frame_list(frames_dir)

    if len(frame_files) == 0:
        return {"selected_files": [], "total_frames": 0, "selected_count": 0,
                "reduction": 0, "stats": {}, "processing_time": 0}

    seg_label = f" [Seg {segment_id}]" if segment_id is not None else ""
    print(f"\n{'=' * 65}")
    print(f"  DINO COSINE KEYFRAME SELECTOR{seg_label}")
    print(f"  Frames: {len(frame_files)}  |  Model: {model_name}")
    print(f"  Threshold: {threshold}  |  Device: {device}")
    print(f"  Force interval: {force_interval}")
    print(f"{'=' * 65}\n")

    t0 = time.time()

    # ── Precompute ALL embeddings in batches (the expensive part) ──
    # Decoupled from the sequential selection logic below, since a frame's
    # embedding never depends on which frames were accepted.
    # Cache: the keyframe pass and the DA3-dense pass run DINO over the SAME frames
    # with only a different threshold → reuse the cached embeddings on the 2nd pass
    # (skips both the model load AND the forward — the "double cosine" cost).
    batch_size = int(config.get("dino_batch_size", 32))
    num_workers = int(config.get("dino_num_workers", 8))
    use_amp = bool(config.get("dino_fp16", False))
    embs = _load_embedding_cache(frames_dir, frame_files, model_name)
    if embs is not None:
        print(f"  [embeddings] reusing cache for {len(frame_files)} frames "
              f"(skipped model load + forward)")
        model_load_time = 0.0
        embed_time = time.time() - t0
    else:
        model = _load_dino_model(model_name, device)
        transform = _get_dino_transform()
        model_load_time = time.time() - t0
        embs = _extract_embeddings_batched(
            model, frames_dir, frame_files, transform, device,
            batch_size=batch_size, num_workers=num_workers, use_amp=use_amp,
        )  # (N, D) L2-normalized; NaN rows = failed loads
        embed_time = time.time() - t0 - model_load_time
        _save_embedding_cache(frames_dir, frame_files, model_name, embs)
        # Model no longer needed — the rest is pure numpy.
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── Sequential keyframe decision over precomputed embeddings ──
    # Bit-identical logic to the per-frame version: compare each candidate
    # against the last accepted keyframe; accept on cosine < threshold, force
    # accept every `force_interval` frames, accept on decode failure (NaN row).
    # Vectors are L2-normalized, so cosine similarity == dot product.
    def _valid(idx):
        return not np.isnan(embs[idx, 0])

    selected = [frame_files[0]]
    ref_emb = embs[0] if _valid(0) else None

    stats = {
        "ACCEPT": 0,
        "SKIP_SIMILAR": 0,
        "FORCE_ACCEPT": 0,
        "ERROR": 0,
    }

    frames_since_accept = 0
    per_frame_log = []
    similarities = []

    for i in range(1, len(frame_files)):
        frames_since_accept += 1

        # Force accept if too many frames skipped
        if frames_since_accept >= force_interval:
            selected.append(frame_files[i])
            if _valid(i):
                ref_emb = embs[i]
            frames_since_accept = 0
            stats["FORCE_ACCEPT"] += 1
            per_frame_log.append({
                "frame": frame_files[i], "status": "FORCE_ACCEPT",
                "similarity": 0.0,
            })
            continue

        # Failed embedding (or no reference yet) → accept (safety), ref unchanged
        if not _valid(i) or ref_emb is None:
            selected.append(frame_files[i])
            frames_since_accept = 0
            stats["ERROR"] += 1
            per_frame_log.append({
                "frame": frame_files[i], "status": "ERROR",
            })
            continue

        sim = float(np.dot(ref_emb, embs[i]))
        similarities.append(sim)

        if sim < threshold:
            # Sufficient visual change → accept as keyframe
            selected.append(frame_files[i])
            ref_emb = embs[i]
            frames_since_accept = 0
            stats["ACCEPT"] += 1
            per_frame_log.append({
                "frame": frame_files[i], "status": "ACCEPT",
                "similarity": round(sim, 4),
            })
        else:
            stats["SKIP_SIMILAR"] += 1
            per_frame_log.append({
                "frame": frame_files[i], "status": "SKIP_SIMILAR",
                "similarity": round(sim, 4),
            })

    elapsed = time.time() - t0
    reduction = 1.0 - len(selected) / len(frame_files) if frame_files else 0

    print(f"\n{'=' * 65}")
    print(f"  ✅ SELECTION COMPLETE{seg_label} — {elapsed:.1f}s "
          f"(model load: {model_load_time:.1f}s, embeddings: {embed_time:.1f}s, "
          f"batched on {device})")
    print(f"{'=' * 65}")
    print(f"  Input:      {len(frame_files)} frames")
    print(f"  Selected:   {len(selected)} keyframes")
    print(f"  Reduction:  {reduction * 100:.1f}%")
    if similarities:
        print(f"  Similarity:  min={min(similarities):.3f}, "
              f"max={max(similarities):.3f}, "
              f"mean={np.mean(similarities):.3f}")
    print(f"  Breakdown:")
    for status, count in sorted(stats.items()):
        if count > 0:
            print(f"    {status}: {count}")
    print(f"{'=' * 65}\n")

    # Save result
    output = {
        "version": "2.0",
        "method": "dino_cosine",
        "model": model_name,
        "total_frames": len(frame_files),
        "selected_count": len(selected),
        "reduction": round(reduction, 4),
        "config": {
            "dino_model": model_name,
            "dino_threshold": threshold,
            "dino_device": device,
            "force_accept_interval": force_interval,
        },
        "stats": stats,
        "similarity_stats": {
            "min": round(min(similarities), 4) if similarities else 0,
            "max": round(max(similarities), 4) if similarities else 0,
            "mean": round(float(np.mean(similarities)), 4) if similarities else 0,
        },
        "selected_files": selected,
        "processing_time": round(elapsed, 1),
    }

    # Save
    suffix = f"_seg{segment_id}" if segment_id is not None else ""
    output_path = Path(frames_dir) / f"selected_frames{suffix}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")

    return output


# ═══════════════════════════════════════════════════════════════════
#              H/F RATIO METHOD (Legacy ORB-SLAM style)
# ═══════════════════════════════════════════════════════════════════

def _compute_symmetric_transfer_error(pts1: np.ndarray, pts2: np.ndarray,
                                       H: np.ndarray) -> np.ndarray:
    """
    Compute symmetric transfer error for Homography.
    Used by ORB-SLAM for proper H/F scoring (not just inlier count).

    Returns per-match error (lower = better fit to H).
    """
    # Forward: project pts1 → pts2 via H
    ones = np.ones((len(pts1), 1))
    pts1_h = np.hstack([pts1, ones])  # (N, 3)
    proj_fwd = (H @ pts1_h.T).T       # (N, 3)
    proj_fwd = proj_fwd[:, :2] / (proj_fwd[:, 2:3] + 1e-10)
    err_fwd = np.sum((pts2 - proj_fwd) ** 2, axis=1)

    # Backward: project pts2 → pts1 via H^-1
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(pts1), 1e6)
    pts2_h = np.hstack([pts2, ones])
    proj_bwd = (H_inv @ pts2_h.T).T
    proj_bwd = proj_bwd[:, :2] / (proj_bwd[:, 2:3] + 1e-10)
    err_bwd = np.sum((pts1 - proj_bwd) ** 2, axis=1)

    return err_fwd + err_bwd


def _compute_sampson_error(pts1: np.ndarray, pts2: np.ndarray,
                            F: np.ndarray) -> np.ndarray:
    """
    Compute Sampson distance for Fundamental Matrix.
    First-order approximation of geometric distance to epipolar line.
    """
    ones = np.ones((len(pts1), 1))
    p1 = np.hstack([pts1, ones])  # (N, 3)
    p2 = np.hstack([pts2, ones])

    # Epipolar constraint: p2^T F p1
    Fp1 = (F @ p1.T).T      # (N, 3)
    Ftp2 = (F.T @ p2.T).T   # (N, 3)

    numerator = np.sum(p2 * Fp1, axis=1) ** 2
    denominator = (Fp1[:, 0] ** 2 + Fp1[:, 1] ** 2 +
                   Ftp2[:, 0] ** 2 + Ftp2[:, 1] ** 2 + 1e-10)

    return numerator / denominator


def _score_model(errors: np.ndarray, sigma_sq: float) -> Tuple[float, np.ndarray]:
    """
    ORB-SLAM style model scoring using chi-squared threshold.
    """
    # Chi-squared threshold (95% confidence, 2 DOF)
    th = 5.991 * sigma_sq

    inlier_mask = errors < th
    scores = np.where(inlier_mask, th - errors, 0)

    return float(np.sum(scores)), inlier_mask


def compute_novelty(img_ref: np.ndarray, img_cand: np.ndarray,
                     max_features: int = 2000,
                     sigma: float = 1.0) -> Dict:
    """
    Compute visual novelty between two frames using ORB-SLAM H/F ratio.
    """
    sigma_sq = sigma * sigma

    # Feature detection (ORB — fastest robust detector)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_cand, None)

    result = {
        "status": "SKIP_STATIC",
        "matches": 0,
        "median_displacement": 0.0,
        "score_H": 0.0,
        "score_F": 0.0,
        "ratio_H": 1.0,
        "inliers_H": 0,
        "inliers_F": 0,
    }

    if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
        result["status"] = "LOST_TRACKING"
        return result

    # Feature matching (Brute-force Hamming for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 30:
        result["status"] = "LOST_TRACKING"
        result["matches"] = len(matches)
        return result

    matches = sorted(matches, key=lambda m: m.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    result["matches"] = len(matches)

    # Check 1: Minimum displacement
    displacements = np.sqrt(np.sum((pts2 - pts1) ** 2, axis=1))
    median_disp = float(np.median(displacements))
    result["median_displacement"] = median_disp

    if median_disp < 3.0:
        result["status"] = "SKIP_STATIC"
        return result

    # Check 2: H/F ratio test
    H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0 * sigma)
    if H is None:
        result["status"] = "LOST_TRACKING"
        return result

    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0 * sigma, 0.99)
    if F is None or F.shape != (3, 3):
        result["status"] = "LOST_TRACKING"
        return result

    h_errors = _compute_symmetric_transfer_error(pts1, pts2, H)
    f_errors = _compute_sampson_error(pts1, pts2, F)

    score_H, inliers_H = _score_model(h_errors, sigma_sq)
    score_F, inliers_F = _score_model(f_errors, sigma_sq)

    result["score_H"] = score_H
    result["score_F"] = score_F
    result["inliers_H"] = int(np.sum(inliers_H))
    result["inliers_F"] = int(np.sum(inliers_F))

    ratio_H = score_H / (score_H + score_F + 1e-10)
    result["ratio_H"] = float(ratio_H)

    if ratio_H > 0.45:
        result["status"] = "SKIP_ROTATION"
    else:
        result["status"] = "ACCEPT"

    return result


def select_keyframes_hf(frames_dir: str, config: dict = None,
                        file_list: list = None, segment_id: int = None) -> Dict:
    """
    Legacy H/F ratio keyframe selection (ORB-SLAM style).
    Kept for fallback / CPU-only environments.
    """
    if config is None:
        config = {}

    max_features = config.get("max_features", 2000)
    sigma = config.get("sigma", 1.0)
    process_scale = config.get("process_scale", 0.25)
    min_displacement = config.get("min_displacement", 3.0)
    hf_ratio_threshold = config.get("hf_ratio_threshold", 0.45)
    force_interval = config.get("force_accept_interval", 30)

    frames_dir = Path(frames_dir)

    if file_list is not None:
        frame_files = file_list
    else:
        frame_files = _load_valid_frame_list(frames_dir)

    if len(frame_files) == 0:
        return {"selected_files": [], "total_frames": 0, "selected_count": 0,
                "reduction": 0, "stats": {}, "processing_time": 0}

    seg_label = f" [Seg {segment_id}]" if segment_id is not None else ""
    print(f"\n{'=' * 65}")
    print(f"  VISUAL NOVELTY FRAME SELECTOR (H/F Ratio){seg_label}")
    print(f"  Frames: {len(frame_files)}  |  Scale: {process_scale}")
    print(f"  Features: {max_features}  |  H/F threshold: {hf_ratio_threshold}")
    print(f"  Force interval: {force_interval}")
    print(f"{'=' * 65}\n")

    t0 = time.time()

    selected = [frame_files[0]]
    ref_img = _load_gray(frames_dir / frame_files[0], process_scale)

    stats = {
        "ACCEPT": 0,
        "SKIP_STATIC": 0,
        "SKIP_ROTATION": 0,
        "LOST_TRACKING": 0,
        "FORCE_ACCEPT": 0,
    }

    frames_since_accept = 0

    for i in range(1, len(frame_files)):
        cand_img = _load_gray(frames_dir / frame_files[i], process_scale)
        if cand_img is None:
            continue

        frames_since_accept += 1

        if frames_since_accept >= force_interval:
            selected.append(frame_files[i])
            ref_img = cand_img
            frames_since_accept = 0
            stats["FORCE_ACCEPT"] += 1
            continue

        result = compute_novelty(ref_img, cand_img, max_features, sigma)

        if result["median_displacement"] < min_displacement:
            result["status"] = "SKIP_STATIC"

        if (result["status"] == "ACCEPT" and
            result["ratio_H"] > hf_ratio_threshold):
            result["status"] = "SKIP_ROTATION"

        status = result["status"]
        stats[status] = stats.get(status, 0) + 1

        if status in ("ACCEPT", "LOST_TRACKING"):
            selected.append(frame_files[i])
            ref_img = cand_img
            frames_since_accept = 0

        if (i + 1) % 500 == 0 or i == len(frame_files) - 1:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(frame_files)}] Selected: {len(selected)} "
                  f"({fps:.0f} frames/s)")

    elapsed = time.time() - t0
    reduction = 1.0 - len(selected) / len(frame_files) if frame_files else 0

    print(f"\n{'=' * 65}")
    print(f"  ✅ SELECTION COMPLETE{seg_label} — {elapsed:.1f}s")
    print(f"{'=' * 65}")
    print(f"  Input:      {len(frame_files)} frames")
    print(f"  Selected:   {len(selected)} keyframes")
    print(f"  Reduction:  {reduction * 100:.1f}%")
    print(f"  Breakdown:")
    for status, count in sorted(stats.items()):
        if count > 0:
            print(f"    {status}: {count}")
    print(f"{'=' * 65}\n")

    output = {
        "version": "1.0",
        "method": "hf_ratio",
        "total_frames": len(frame_files),
        "selected_count": len(selected),
        "reduction": round(reduction, 4),
        "config": {
            "max_features": max_features,
            "sigma": sigma,
            "process_scale": process_scale,
            "hf_ratio_threshold": hf_ratio_threshold,
            "min_displacement": min_displacement,
            "force_accept_interval": force_interval,
        },
        "stats": stats,
        "selected_files": selected,
        "processing_time": round(elapsed, 1),
    }

    suffix = f"_seg{segment_id}" if segment_id is not None else ""
    output_path = frames_dir / f"selected_frames{suffix}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")

    return output


# ═══════════════════════════════════════════════════════════════════
#                    DISPATCHER (Public API)
# ═══════════════════════════════════════════════════════════════════

def select_keyframes(frames_dir: str, config: dict = None,
                     file_list: list = None, segment_id: int = None) -> Dict:
    """
    Select keyframes using the configured method.

    Dispatches to dino_select_keyframes() or select_keyframes_hf()
    based on config["method"] (default: "dino").

    This is the main entry point — all callers use this function.
    """
    if config is None:
        config = {}

    method = config.get("method", "dino")

    if method == "dino":
        return dino_select_keyframes(frames_dir, config, file_list, segment_id)
    else:
        return select_keyframes_hf(frames_dir, config, file_list, segment_id)


def select_keyframes_parallax(frames_dir: str, npz_dir: str, config: dict = None,
                              file_list: list = None, segment_id: int = None) -> Dict:
    """GEOMETRIC keyframe selection (triangulation angle / parallax) for the SLAM backbone.

    Cosine-DINO measures APPEARANCE, not the geometric BASELINE a SLAM/multi-view backbone
    needs. It fails two ways in a corridor: (a) PURE ROTATION — appearance changes a lot but
    baseline ≈ 0, cosine adds a degenerate (un-triangulable) keyframe; (b) FRONTAL ADVANCE —
    appearance changes slowly but there IS translation, cosine misses it. Parallax angle
    captures both: a new keyframe is taken when the MEDIAN triangulation angle vs the last
    keyframe reaches `theta_parallax`. Pure rotation gives C_i≈C_j → near-identical rays →
    parallax ≈ 0 → never fires alone.

    Needs per-frame metric depth + pose (DA3 npz: depth, intrinsics, extrinsics[3x4, w2c]).
    NO cosine fallback: if poses are missing it RAISES; if the whole clip has no baseline
    (pure rotation) it returns geometric_ok=False so the caller can abort with a clear message.
    Writes the SAME selected_frames.json schema as the cosine selector (drop-in).
    """
    config = config or {}
    theta = float(config.get("theta_parallax", 1.5))            # deg, min triangulation angle
    overlap_min = float(config.get("overlap_min", 0.5))         # force a kf if overlap drops below
    min_baseline = float(config.get("min_baseline_m", 0.05))    # …but ONLY with real translation
    min_global_baseline = float(config.get("min_global_baseline_m", 0.3))  # pure-rotation abort
    stride = int(config.get("parallax_stride", 8))
    dmin = float(config.get("parallax_depth_min", 0.1))
    dmax = float(config.get("parallax_depth_max", config.get("reliable_depth_m", 8.0)))

    frames_dir = Path(frames_dir); npz_dir = Path(npz_dir)
    frame_files = file_list if file_list is not None else _load_valid_frame_list(frames_dir)

    def _npz(fname):
        p = npz_dir / f"frame_{int(os.path.splitext(fname)[0])}.npz"
        return p if p.exists() else None

    if len(frame_files) < 2:
        return {"version": "2.0", "method": f"parallax_{theta}", "total_frames": len(frame_files),
                "selected_count": len(frame_files), "selected_files": list(frame_files),
                "geometric_ok": False, "reason": "fewer than 2 valid frames", "processing_time": 0}

    # Fail-fast: every frame MUST have its DA3 depth+pose (no silent cosine fallback).
    missing = [f for f in frame_files if _npz(f) is None]
    if missing:
        raise RuntimeError(f"parallax selection: {len(missing)}/{len(frame_files)} frames have no "
                           f"DA3 npz (depth+pose) in {npz_dir} — DA3 must run on all blur-valid "
                           f"frames first")

    def _load(p):
        z = np.load(p); ex = np.asarray(z["extrinsics"], np.float64)   # 3x4 world→cam (w2c)
        T = np.eye(4); T[:3] = ex; c2w = np.linalg.inv(T)
        return np.asarray(z["depth"], np.float32), np.asarray(z["intrinsics"], np.float64), c2w, T

    def _backproj(depth, K, c2w):
        H, W = depth.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        d = depth[::stride, ::stride]
        m = (d > dmin) & (d < dmax) & np.isfinite(d)
        u, v, d = uu[m], vv[m], d[m]
        x = (u - K[0, 2]) * d / K[0, 0]; y = (v - K[1, 2]) * d / K[1, 1]
        cam = np.stack([x, y, d], 1)
        return cam @ c2w[:3, :3].T + c2w[:3, 3]                        # world (N,3)

    def _parallax_overlap(P, Ci, Cj, Kj, w2cj, hw):
        vi = P - Ci; vj = P - Cj
        ni = np.linalg.norm(vi, axis=1); nj = np.linalg.norm(vj, axis=1)
        ok = (ni > 1e-6) & (nj > 1e-6)
        if not ok.any():
            return 0.0, 0.0
        cos = np.clip((vi[ok] * vj[ok]).sum(1) / (ni[ok] * nj[ok]), -1, 1)
        ang = np.degrees(np.arccos(cos))
        Pc = P @ w2cj[:3, :3].T + w2cj[:3, 3]; z = Pc[:, 2]
        zc = np.clip(z, 1e-6, None)
        u = Kj[0, 0] * Pc[:, 0] / zc + Kj[0, 2]; vp = Kj[1, 1] * Pc[:, 1] / zc + Kj[1, 2]
        H, W = hw
        inb = (z > 0) & (u >= 0) & (u < W) & (vp >= 0) & (vp < H)
        return float(np.median(ang)), float(inb.mean())

    t0 = time.time()
    selected = [frame_files[0]]
    di, Ki, c2wi, _ = _load(_npz(frame_files[0])); Pi = _backproj(di, Ki, c2wi); Ci = c2wi[:3, 3]
    medians = []; centers = [Ci]
    for j in range(1, len(frame_files)):
        dj, Kj, c2wj, w2cj = _load(_npz(frame_files[j])); Cj = c2wj[:3, 3]; centers.append(Cj)
        med, ov = _parallax_overlap(Pi, Ci, Cj, Kj, w2cj, dj.shape)
        medians.append(med)
        baseline = float(np.linalg.norm(Cj - Ci))
        # primary: enough triangulation angle. secondary: overlap dropped AND we actually
        # translated (so pure rotation, baseline≈0, never forces a degenerate keyframe).
        if med >= theta or (ov < overlap_min and baseline > min_baseline):
            selected.append(frame_files[j])
            Pi = _backproj(dj, Kj, c2wj); Ci = Cj

    centers = np.asarray(centers)
    global_baseline = float(np.linalg.norm(centers - centers[0], axis=1).max())
    # min keyframes is a PERCENTAGE of the evaluated frames (+ an absolute floor), not a fixed
    # count: a flat 20 is far too strict for a 300-frame clip yet lenient for 1500+. Scaling with
    # input size keeps the geometric-coverage bar coherent across capture lengths.
    mk_pct = float(config.get("min_keyframes_pct", 0.015))
    mk_floor = int(config.get("min_keyframes_floor", 8))
    min_keyframes = max(mk_floor, int(round(mk_pct * len(frame_files))))
    if global_baseline < min_global_baseline:
        geometric_ok, reason = False, (
            f"insufficient geometric baseline (max camera displacement {global_baseline:.2f}m < "
            f"{min_global_baseline}m) — the clip is essentially pure rotation / no parallax for "
            f"triangulation")
    elif len(selected) < min_keyframes:
        geometric_ok, reason = False, (
            f"too few keyframes ({len(selected)} < {min_keyframes} = max({mk_floor}, "
            f"{mk_pct:.1%}×{len(frame_files)})) — not enough geometric coverage to reconstruct "
            f"(raise the capture length or lower theta_parallax)")
    else:
        geometric_ok, reason = True, ""
    a = np.asarray(medians) if medians else np.zeros(1)
    elapsed = time.time() - t0
    out = {
        "version": "2.0", "method": f"parallax_{theta}",
        "total_frames": len(frame_files), "selected_count": len(selected),
        "reduction": round(1 - len(selected) / len(frame_files), 4),
        "config": {"theta_parallax": theta, "overlap_min": overlap_min,
                   "min_baseline_m": min_baseline, "parallax_stride": stride,
                   "depth_min": dmin, "depth_max": dmax},
        "parallax_stats": {"p50": round(float(np.median(a)), 3),
                           "p90": round(float(np.percentile(a, 90)), 3),
                           "max": round(float(a.max()), 3),
                           "global_baseline_m": round(global_baseline, 3)},
        "geometric_ok": geometric_ok, "reason": reason,
        "selected_files": selected, "processing_time": round(elapsed, 1),
    }
    suffix = f"_seg{segment_id}" if segment_id is not None else ""
    with open(frames_dir / f"selected_frames{suffix}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Parallax] {len(selected)}/{len(frame_files)} keyframes (θ={theta}°, "
          f"p50={out['parallax_stats']['p50']}°, baseline={global_baseline:.1f}m, "
          f"ok={geometric_ok}) in {elapsed:.0f}s")
    return out


def load_selected_frames(frames_dir: str) -> Optional[List[str]]:
    """
    Load previously computed keyframe selection.
    Returns list of selected filenames, or None if not available.
    """
    sel_path = Path(frames_dir) / "selected_frames.json"
    if not sel_path.exists():
        return None

    try:
        with open(sel_path) as f:
            data = json.load(f)
        files = data.get("selected_files", [])
        method = data.get("method", "unknown")
        if files:
            print(f"[FrameSelector] Loaded {len(files)} selected keyframes "
                  f"({method}, from {data.get('total_frames', '?')} total, "
                  f"{data.get('reduction', 0) * 100:.0f}% reduction)")
        return files
    except Exception as e:
        print(f"[FrameSelector] ⚠️ Could not load selected_frames.json: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#                        HELPERS
# ═══════════════════════════════════════════════════════════════════

def _load_valid_frame_list(frames_dir: Path) -> List[str]:
    """Load frame list, respecting frame_quality.json if available."""
    fq_path = frames_dir / "frame_quality.json"

    if fq_path.exists():
        with open(fq_path) as f:
            fq_data = json.load(f)
        files = sorted(
            [fr["file"] for fr in fq_data["frames"] if fr["valid"]],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        rejected = fq_data.get("rejected_frames", 0)
        print(f"[FrameSelector] Using {len(files)} valid frames "
              f"({rejected} blurry removed)")
        return files

    # No quality filter — use all frames
    files = sorted(
        [f for f in os.listdir(frames_dir)
         if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    print(f"[FrameSelector] Using all {len(files)} frames (no quality filter)")
    return files


def _load_gray(path: Path, scale: float = 0.25) -> Optional[np.ndarray]:
    """Load image as grayscale at reduced resolution for fast processing."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if scale != 1.0:
        w = int(img.shape[1] * scale)
        h = int(img.shape[0] * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


# ═══════════════════════════════════════════════════════════════════
#                       CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Keyframe Selector (DINO cosine or H/F ratio)"
    )
    parser.add_argument("frames_dir", help="Directory with frame images")
    parser.add_argument("--method", choices=["dino", "hf_ratio"], default="dino",
                        help="Selection method (default: dino)")
    # DINO args
    parser.add_argument("--dino-threshold", type=float, default=0.85,
                        help="Cosine similarity threshold (lower = fewer keyframes)")
    parser.add_argument("--dino-device", default="cuda",
                        help="Device for DINO inference")
    # H/F args
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--hf-threshold", type=float, default=0.45)
    parser.add_argument("--min-displacement", type=float, default=3.0)
    # Shared
    parser.add_argument("--force-interval", type=int, default=30,
                        help="Force accept every N frames (safety net)")

    args = parser.parse_args()

    config = {
        "method": args.method,
        "dino_model": "dinov2_vitb14",
        "dino_threshold": args.dino_threshold,
        "dino_device": args.dino_device,
        "max_features": args.max_features,
        "process_scale": args.scale,
        "hf_ratio_threshold": args.hf_threshold,
        "min_displacement": args.min_displacement,
        "force_accept_interval": args.force_interval,
    }

    result = select_keyframes(args.frames_dir, config)
    print(f"\nSelected {result['selected_count']} / {result['total_frames']} frames")
