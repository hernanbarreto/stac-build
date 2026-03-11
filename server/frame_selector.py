"""
Visual Novelty Frame Selector — ORB-SLAM style H/F ratio keyframe selection.
=============================================================================

Selects geometrically useful frames from a video sequence by measuring
real parallax between frames using the Homography vs Fundamental Matrix
ratio test from ORB-SLAM2/3.

Key insight:
  - Homography (H) explains planar scenes and pure rotation
  - Fundamental Matrix (F) explains general 3D motion with parallax
  - If H dominates → no useful parallax → SKIP
  - If F dominates → real translational motion → ACCEPT

Neither H nor F require camera intrinsics (K), making this method
camera-agnostic and suitable for uncalibrated preprocessing.

Pipeline position:
  Raw frames → Blur filter → **Visual Novelty Filter** → Reconstruction / Segmentation

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
#                       CORE ALGORITHM
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
    
    Args:
        errors: Per-match errors
        sigma_sq: Sigma squared (reprojection error variance)
    
    Returns:
        (score, inlier_mask)
    """
    # Chi-squared threshold (95% confidence, 2 DOF for H, 1 DOF for F)
    th = 5.991 * sigma_sq  # 2-DOF chi-squared threshold
    
    inlier_mask = errors < th
    # Score: sum of (threshold - error) for inliers (higher = better)
    scores = np.where(inlier_mask, th - errors, 0)
    
    return float(np.sum(scores)), inlier_mask


def compute_novelty(img_ref: np.ndarray, img_cand: np.ndarray,
                     max_features: int = 2000,
                     sigma: float = 1.0) -> Dict:
    """
    Compute visual novelty between two frames using ORB-SLAM H/F ratio.
    
    Args:
        img_ref: Reference grayscale image (last accepted keyframe)
        img_cand: Candidate grayscale image
        max_features: Number of ORB features to detect
        sigma: Reprojection error standard deviation (pixels)
    
    Returns:
        Dict with keys:
            status: "ACCEPT", "SKIP_STATIC", "SKIP_ROTATION", "LOST_TRACKING"
            matches: Number of valid feature matches
            median_displacement: Median pixel displacement of matches
            score_H: Homography model score
            score_F: Fundamental Matrix model score
            ratio_H: H / (H + F) ratio (< 0.45 = parallax present)
            inliers_H: Homography inlier count
            inliers_F: Fundamental inlier count
    """
    sigma_sq = sigma * sigma
    
    # ── Feature detection (ORB — fastest robust detector) ──
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
    
    # ── Feature matching (Brute-force Hamming for ORB) ──
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 30:
        result["status"] = "LOST_TRACKING"
        result["matches"] = len(matches)
        return result
    
    # Sort by distance (best first)
    matches = sorted(matches, key=lambda m: m.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    result["matches"] = len(matches)
    
    # ── Check 1: Minimum displacement (is the camera moving at all?) ──
    displacements = np.sqrt(np.sum((pts2 - pts1) ** 2, axis=1))
    median_disp = float(np.median(displacements))
    result["median_displacement"] = median_disp
    
    if median_disp < 3.0:
        result["status"] = "SKIP_STATIC"
        return result
    
    # ── Check 2: H/F ratio test (ORB-SLAM parallax detection) ──
    
    # Compute Homography
    H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0 * sigma)
    if H is None:
        result["status"] = "LOST_TRACKING"
        return result
    
    # Compute Fundamental Matrix
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0 * sigma, 0.99)
    if F is None or F.shape != (3, 3):
        result["status"] = "LOST_TRACKING"
        return result
    
    # ORB-SLAM style scoring (symmetric transfer error, not just inlier count)
    h_errors = _compute_symmetric_transfer_error(pts1, pts2, H)
    f_errors = _compute_sampson_error(pts1, pts2, F)
    
    score_H, inliers_H = _score_model(h_errors, sigma_sq)
    score_F, inliers_F = _score_model(f_errors, sigma_sq)
    
    result["score_H"] = score_H
    result["score_F"] = score_F
    result["inliers_H"] = int(np.sum(inliers_H))
    result["inliers_F"] = int(np.sum(inliers_F))
    
    # H/F ratio
    ratio_H = score_H / (score_H + score_F + 1e-10)
    result["ratio_H"] = float(ratio_H)
    
    # Decision
    if ratio_H > 0.45:
        result["status"] = "SKIP_ROTATION"  # H dominates → rotation/planar
    else:
        result["status"] = "ACCEPT"  # F dominates → real parallax
    
    return result


# ═══════════════════════════════════════════════════════════════════
#                    BATCH KEYFRAME SELECTION
# ═══════════════════════════════════════════════════════════════════

def select_keyframes(frames_dir: str, config: dict = None,
                     file_list: list = None, segment_id: int = None) -> Dict:
    """
    Select geometrically useful keyframes from a frame directory.
    
    Processes frames sequentially, comparing each candidate against the
    last accepted keyframe. Uses H/F ratio to detect real parallax.
    
    Args:
        frames_dir: Directory containing frame images (after blur filter)
        config: Configuration dict from config.yaml["frame_selection"]
                If None, uses sensible defaults.
        file_list: Optional explicit list of filenames to process
                   (for zoom-segment mode). If None, scans directory.
        segment_id: Optional segment index for labeling output.
    
    Returns:
        Dict with:
            selected_files: List of selected frame filenames
            total_frames: Total frames analyzed
            selected_count: Number of selected keyframes
            reduction: Percentage of frames removed
            stats: Per-frame decision breakdown
            processing_time: Total processing time in seconds
    """
    if config is None:
        config = {}
    
    # Configuration with defaults
    max_features = config.get("max_features", 2000)
    sigma = config.get("sigma", 1.0)
    process_scale = config.get("process_scale", 0.25)  # Process at 25% resolution
    min_displacement = config.get("min_displacement", 3.0)
    hf_ratio_threshold = config.get("hf_ratio_threshold", 0.45)
    min_match_ratio = config.get("min_match_ratio", 0.30)
    force_interval = config.get("force_accept_interval", 30)  # Force accept every N frames
    
    frames_dir = Path(frames_dir)
    
    # Load frame list (explicit list or directory scan)
    if file_list is not None:
        frame_files = file_list
    else:
        frame_files = _load_valid_frame_list(frames_dir)
    
    if len(frame_files) == 0:
        return {"selected_files": [], "total_frames": 0, "selected_count": 0,
                "reduction": 0, "stats": {}, "processing_time": 0}
    
    seg_label = f" [Seg {segment_id}]" if segment_id is not None else ""
    print(f"\n{'='*65}")
    print(f"  VISUAL NOVELTY FRAME SELECTOR (H/F Ratio){seg_label}")
    print(f"  Frames: {len(frame_files)}  |  Scale: {process_scale}")
    print(f"  Features: {max_features}  |  H/F threshold: {hf_ratio_threshold}")
    print(f"  Force interval: {force_interval}")
    print(f"{'='*65}\n")
    
    t0 = time.time()
    
    # Always accept first frame
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
    per_frame_log = []
    
    for i in range(1, len(frame_files)):
        cand_img = _load_gray(frames_dir / frame_files[i], process_scale)
        if cand_img is None:
            continue
        
        frames_since_accept += 1
        
        # Force accept if too many frames skipped (safety net)
        if frames_since_accept >= force_interval:
            selected.append(frame_files[i])
            ref_img = cand_img
            frames_since_accept = 0
            stats["FORCE_ACCEPT"] += 1
            per_frame_log.append({
                "frame": frame_files[i], "status": "FORCE_ACCEPT",
                "gap": frames_since_accept
            })
            continue
        
        # Compute visual novelty
        result = compute_novelty(ref_img, cand_img, max_features, sigma)
        
        # Override displacement threshold from config
        if result["median_displacement"] < min_displacement:
            result["status"] = "SKIP_STATIC"
        
        # Override H/F threshold from config
        if (result["status"] == "ACCEPT" and 
            result["ratio_H"] > hf_ratio_threshold):
            result["status"] = "SKIP_ROTATION"
        
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        
        per_frame_log.append({
            "frame": frame_files[i],
            "status": status,
            "displacement": round(result["median_displacement"], 1),
            "ratio_H": round(result["ratio_H"], 3),
            "matches": result["matches"],
        })
        
        if status == "ACCEPT":
            selected.append(frame_files[i])
            ref_img = cand_img
            frames_since_accept = 0
        elif status == "LOST_TRACKING":
            # Lost tracking → accept and reset reference
            selected.append(frame_files[i])
            ref_img = cand_img
            frames_since_accept = 0
        
        # Progress logging
        if (i + 1) % 500 == 0 or i == len(frame_files) - 1:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{len(frame_files)}] Selected: {len(selected)} "
                  f"({fps:.0f} frames/s)")
    
    elapsed = time.time() - t0
    reduction = 1.0 - len(selected) / len(frame_files) if frame_files else 0
    
    print(f"\n{'='*65}")
    print(f"  ✅ SELECTION COMPLETE{seg_label} — {elapsed:.1f}s")
    print(f"{'='*65}")
    print(f"  Input:      {len(frame_files)} frames")
    print(f"  Selected:   {len(selected)} keyframes")
    print(f"  Reduction:  {reduction*100:.1f}%")
    print(f"  Breakdown:")
    for status, count in sorted(stats.items()):
        if count > 0:
            print(f"    {status}: {count}")
    print(f"{'='*65}\n")
    
    # Save result
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
    
    # Save to frames directory (with segment suffix if applicable)
    suffix = f"_seg{segment_id}" if segment_id is not None else ""
    output_path = frames_dir / f"selected_frames{suffix}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {output_path}")
    
    return output


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
        if files:
            print(f"[FrameSelector] Loaded {len(files)} selected keyframes "
                  f"(from {data.get('total_frames', '?')} total, "
                  f"{data.get('reduction', 0)*100:.0f}% reduction)")
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
        description="Visual Novelty Frame Selector (H/F Ratio)"
    )
    parser.add_argument("frames_dir", help="Directory with frame images")
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Processing scale (0.25 = 25%% resolution)")
    parser.add_argument("--hf-threshold", type=float, default=0.45,
                        help="H/F ratio threshold (lower = stricter)")
    parser.add_argument("--min-displacement", type=float, default=3.0,
                        help="Min median pixel displacement")
    parser.add_argument("--force-interval", type=int, default=30,
                        help="Force accept every N frames (safety net)")
    
    args = parser.parse_args()
    
    config = {
        "max_features": args.max_features,
        "process_scale": args.scale,
        "hf_ratio_threshold": args.hf_threshold,
        "min_displacement": args.min_displacement,
        "force_accept_interval": args.force_interval,
    }
    
    result = select_keyframes(args.frames_dir, config)
    print(f"\nSelected {result['selected_count']} / {result['total_frames']} frames")
