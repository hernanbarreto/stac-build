"""
Frame Quality Analysis — Combined FFT + Laplacian Blur Detection
=================================================================

Uses two complementary methods to catch all blur types:
  - FFT high-frequency energy: catches motion blur (hand shake during scan)
  - Laplacian variance: catches defocus blur (camera out of focus)

A frame is rejected if EITHER score is below its adaptive threshold.

Generates frame_quality.json with per-frame scores.

Usage:
    python -m frames.quality /path/to/frames/dir [--threshold-pct 15]
"""
import cv2
import numpy as np
import json
import os
import sys
import glob
from pathlib import Path


# ═════════════════════════════════════════════════════════════════
#                       BLUR SCORING
# ═════════════════════════════════════════════════════════════════

def compute_fft_score(gray: np.ndarray) -> float:
    """
    FFT high-frequency energy score.
    Low score = motion blur (high frequencies killed by hand movement).
    Robust against uniform surfaces (walls, floors) because even a sharp
    image of a white wall has SOME high-frequency micro-texture.
    ~1ms per frame at 640px.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Zero out the central low-frequency region
    # This isolates the high-frequency ring where blur is detectable
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8  # ~12.5% of image = low frequencies
    magnitude[cy - r:cy + r, cx - r:cx + r] = 0

    return float(np.mean(magnitude))


def compute_laplacian_score(gray: np.ndarray) -> float:
    """
    Laplacian variance score.
    Low score = defocus blur (smooth transitions, no edges).
    Fast and simple, but gives false positives on uniform surfaces.
    ~0.5ms per frame at 640px.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_blur_scores(image_path: str) -> dict:
    """
    Compute combined blur scores for a single frame.
    Returns dict with 'fft' and 'laplacian' scores, or None on error.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize for consistent scoring and speed
    h, w = img.shape
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return {
        "fft": compute_fft_score(img),
        "laplacian": compute_laplacian_score(img),
    }


# Legacy single-score API (backward compat)
def compute_blur_score(image_path: str) -> float:
    """Compute blur score using Laplacian variance (legacy API)."""
    scores = compute_blur_scores(image_path)
    if scores is None:
        return -1.0
    return scores["laplacian"]


def compute_inter_frame_diff(img_path_a: str, img_path_b: str) -> float:
    """
    Compute structural difference between consecutive frames.
    High diff = sudden motion / scene change.
    """
    a = cv2.imread(img_path_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0

    # Resize for speed
    h, w = a.shape
    if max(h, w) > 320:
        scale = 320 / max(h, w)
        a = cv2.resize(a, None, fx=scale, fy=scale)
        b = cv2.resize(b, None, fx=scale, fy=scale)

    diff = cv2.absdiff(a, b)
    return float(diff.mean())


# ═════════════════════════════════════════════════════════════════
#                    ADAPTIVE THRESHOLDS
# ═════════════════════════════════════════════════════════════════

def auto_threshold(scores: list, factor: float = 0.4) -> float:
    """Legacy: median × factor threshold for single-score mode."""
    valid_scores = [s for s in scores if s >= 0]
    if not valid_scores:
        return 0.0
    return float(np.median(valid_scores) * factor)


def compute_adaptive_thresholds(fft_scores: list, lap_scores: list,
                                 percentile: float = 15.0) -> dict:
    """
    Compute adaptive thresholds using percentile method.

    The bottom N% of scores are considered blurry. This adapts to each
    video's lighting, texture, and capture quality automatically.

    Args:
        fft_scores: List of FFT scores
        lap_scores: List of Laplacian scores
        percentile: Bottom percentile to reject (default 15 = reject bottom 15%)

    Returns:
        dict with 'fft' and 'laplacian' thresholds
    """
    valid_fft = [s for s in fft_scores if s > 0]
    valid_lap = [s for s in lap_scores if s > 0]

    fft_thresh = float(np.percentile(valid_fft, percentile)) if valid_fft else 0.0
    lap_thresh = float(np.percentile(valid_lap, percentile)) if valid_lap else 0.0

    return {"fft": fft_thresh, "laplacian": lap_thresh}


# ═════════════════════════════════════════════════════════════════
#                     BATCH ANALYSIS
# ═════════════════════════════════════════════════════════════════

def analyze_frames(image_dir: str, threshold: float = None,
                   threshold_pct: float = 15.0) -> dict:
    """
    Analyze all frames in a directory for blur using FFT + Laplacian.

    Args:
        image_dir: Path to directory containing frame images
        threshold: Manual Laplacian threshold (legacy, if set disables FFT)
        threshold_pct: Bottom percentile to reject (default 15%)

    Returns:
        dict with scores, thresholds, and per-frame results
    """
    # Find all image files, sorted
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files.sort()

    if not image_files:
        print(f"[FrameQuality] No images found in {image_dir}")
        return {"error": "No images found", "frames": []}

    print(f"[FrameQuality] Analyzing {len(image_files)} frames (FFT + Laplacian)...")

    # Compute blur scores
    fft_scores = []
    lap_scores = []
    frames = []
    for i, path in enumerate(image_files):
        scores = compute_blur_scores(path)
        if scores is None:
            fft_scores.append(-1.0)
            lap_scores.append(-1.0)
        else:
            fft_scores.append(scores["fft"])
            lap_scores.append(scores["laplacian"])

        # Inter-frame difference
        if i > 0:
            diff = compute_inter_frame_diff(image_files[i - 1], path)
        else:
            diff = 0.0

        frames.append({
            "index": i,
            "file": os.path.basename(path),
            "fft_score": round(fft_scores[-1], 2),
            "blur_score": round(lap_scores[-1], 2),  # Keep 'blur_score' for backward compat
            "inter_frame_diff": round(diff, 2),
            "valid": True  # Set below
        })

        if (i + 1) % 100 == 0:
            print(f"[FrameQuality]   {i + 1}/{len(image_files)} analyzed...")

    # Compute thresholds
    if threshold is not None:
        # Legacy single-threshold mode (Laplacian only)
        thresholds = {"fft": 0.0, "laplacian": threshold}
    else:
        thresholds = compute_adaptive_thresholds(fft_scores, lap_scores, threshold_pct)

    # Mark validity (blurry if EITHER score is below its threshold)
    valid_count = 0
    rejected_fft = 0
    rejected_lap = 0
    rejected_both = 0
    for f in frames:
        fft_ok = f["fft_score"] >= thresholds["fft"] and f["fft_score"] >= 0
        lap_ok = f["blur_score"] >= thresholds["laplacian"] and f["blur_score"] >= 0

        f["valid"] = fft_ok and lap_ok

        if f["valid"]:
            valid_count += 1
        else:
            if not fft_ok and not lap_ok:
                rejected_both += 1
            elif not fft_ok:
                rejected_fft += 1
            else:
                rejected_lap += 1

    rejected_count = rejected_fft + rejected_lap + rejected_both

    # Stats
    valid_fft = [s for s in fft_scores if s >= 0]
    valid_lap = [s for s in lap_scores if s >= 0]

    result = {
        "method": "fft+laplacian",
        "threshold_fft": round(thresholds["fft"], 2),
        "threshold": round(thresholds["laplacian"], 2),  # backward compat key
        "threshold_percentile": threshold_pct,
        "total_frames": len(image_files),
        "valid_frames": valid_count,
        "rejected_frames": rejected_count,
        "rejected_by_fft": rejected_fft,
        "rejected_by_laplacian": rejected_lap,
        "rejected_by_both": rejected_both,
        "stats": {
            "fft": {
                "min": round(min(valid_fft), 2) if valid_fft else 0,
                "max": round(max(valid_fft), 2) if valid_fft else 0,
                "mean": round(float(np.mean(valid_fft)), 2) if valid_fft else 0,
                "median": round(float(np.median(valid_fft)), 2) if valid_fft else 0,
            },
            "laplacian": {
                "min": round(min(valid_lap), 2) if valid_lap else 0,
                "max": round(max(valid_lap), 2) if valid_lap else 0,
                "mean": round(float(np.mean(valid_lap)), 2) if valid_lap else 0,
                "median": round(float(np.median(valid_lap)), 2) if valid_lap else 0,
            },
            # Legacy compat keys
            "min": round(min(valid_lap), 2) if valid_lap else 0,
            "max": round(max(valid_lap), 2) if valid_lap else 0,
            "mean": round(float(np.mean(valid_lap)), 2) if valid_lap else 0,
            "median": round(float(np.median(valid_lap)), 2) if valid_lap else 0,
            "std": round(float(np.std(valid_lap)), 2) if valid_lap else 0,
        },
        "frames": frames
    }

    print(f"[FrameQuality] ✅ Done: {valid_count} valid, {rejected_count} rejected "
          f"(FFT={rejected_fft}, Lap={rejected_lap}, Both={rejected_both})")
    print(f"[FrameQuality]   Thresholds: FFT≥{thresholds['fft']:.1f}, "
          f"Lap≥{thresholds['laplacian']:.1f} (P{threshold_pct:.0f})")

    return result


def save_manifest(image_dir: str, result: dict) -> str:
    """Save analysis results as frame_quality.json in the image directory."""
    output_path = os.path.join(image_dir, "frame_quality.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[FrameQuality] Saved: {output_path}")
    return output_path


def load_valid_frames(image_dir: str) -> list:
    """
    Read frame_quality.json and return list of valid frame filenames.
    If no JSON exists, returns all image files (no filtering).
    """
    json_path = os.path.join(image_dir, "frame_quality.json")
    if not os.path.exists(json_path):
        return None  # No manifest = process all

    with open(json_path) as f:
        data = json.load(f)

    valid = [f["file"] for f in data["frames"] if f["valid"]]
    total = data["total_frames"]
    rejected = data["rejected_frames"]
    method = data.get("method", "laplacian")
    print(f"[FrameQuality] Loaded manifest ({method}): "
          f"{len(valid)}/{total} valid ({rejected} rejected)")

    return valid


# ── CLI ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze frame quality (FFT + Laplacian blur detection)")
    parser.add_argument("image_dir", help="Directory containing frame images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual Laplacian threshold (legacy, disables FFT)")
    parser.add_argument("--threshold-pct", type=float, default=15.0,
                        help="Bottom percentile to reject (default: 15)")
    parser.add_argument("--show-rejected", action="store_true",
                        help="Print details of rejected frames")
    args = parser.parse_args()

    result = analyze_frames(args.image_dir, args.threshold,
                           threshold_pct=args.threshold_pct)

    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    save_manifest(args.image_dir, result)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Total:     {result['total_frames']}")
    print(f"Valid:     {result['valid_frames']}")
    print(f"Rejected:  {result['rejected_frames']}")
    print(f"  By FFT:      {result.get('rejected_by_fft', 0)}")
    print(f"  By Laplacian: {result.get('rejected_by_laplacian', 0)}")
    print(f"  By Both:      {result.get('rejected_by_both', 0)}")
    print(f"FFT Threshold:  {result.get('threshold_fft', 'N/A')}")
    print(f"Lap Threshold:  {result['threshold']}")

    if args.show_rejected:
        print(f"\nRejected frames:")
        for f in result["frames"]:
            if not f["valid"]:
                print(f"  {f['file']}: fft={f.get('fft_score', '?')}, "
                      f"lap={f['blur_score']}, diff={f['inter_frame_diff']}")
