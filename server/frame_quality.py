"""
Frame Quality Analysis — Blur Detection via Laplacian Variance
Generates frame_quality.json with per-frame blur scores.
Usage:
    python frame_quality.py /path/to/frames/dir [--threshold 50]
"""
import cv2
import numpy as np
import json
import os
import sys
import glob
from pathlib import Path


def compute_blur_score(image_path: str) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher score = sharper image. Low score = blurry.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    
    # Resize for consistent scoring (and speed)
    h, w = img.shape
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())


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


def auto_threshold(scores: list, factor: float = 0.4) -> float:
    """
    Adaptive threshold: median(scores) × factor.
    Frames below this are considered blurry.
    """
    valid_scores = [s for s in scores if s >= 0]
    if not valid_scores:
        return 0.0
    return float(np.median(valid_scores) * factor)


def analyze_frames(image_dir: str, threshold: float = None) -> dict:
    """
    Analyze all frames in a directory for blur.
    
    Args:
        image_dir: Path to directory containing frame images
        threshold: Manual blur threshold. If None, auto-computed.
    
    Returns:
        dict with scores, threshold, and per-frame results
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
    
    print(f"[FrameQuality] Analyzing {len(image_files)} frames...")
    
    # Compute blur scores
    scores = []
    frames = []
    for i, path in enumerate(image_files):
        score = compute_blur_score(path)
        scores.append(score)
        
        # Inter-frame difference (motion detection)
        if i > 0:
            diff = compute_inter_frame_diff(image_files[i-1], path)
        else:
            diff = 0.0
        
        frames.append({
            "index": i,
            "file": os.path.basename(path),
            "blur_score": round(score, 2),
            "inter_frame_diff": round(diff, 2),
            "valid": True  # Set below
        })
        
        if (i + 1) % 50 == 0:
            print(f"[FrameQuality]   {i+1}/{len(image_files)} analyzed...")
    
    # Compute threshold
    if threshold is None:
        threshold = auto_threshold(scores)
    
    # Mark validity
    valid_count = 0
    rejected_count = 0
    for f in frames:
        f["valid"] = f["blur_score"] >= threshold and f["blur_score"] >= 0
        if f["valid"]:
            valid_count += 1
        else:
            rejected_count += 1
    
    # Stats
    valid_scores = [s for s in scores if s >= 0]
    result = {
        "threshold": round(threshold, 2),
        "total_frames": len(image_files),
        "valid_frames": valid_count,
        "rejected_frames": rejected_count,
        "stats": {
            "min": round(min(valid_scores), 2) if valid_scores else 0,
            "max": round(max(valid_scores), 2) if valid_scores else 0,
            "mean": round(float(np.mean(valid_scores)), 2) if valid_scores else 0,
            "median": round(float(np.median(valid_scores)), 2) if valid_scores else 0,
            "std": round(float(np.std(valid_scores)), 2) if valid_scores else 0,
        },
        "frames": frames
    }
    
    print(f"[FrameQuality] ✅ Done: {valid_count} valid, {rejected_count} rejected "
          f"(threshold={threshold:.1f}, median={result['stats']['median']:.1f})")
    
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
    print(f"[FrameQuality] Loaded manifest: {len(valid)}/{total} valid ({rejected} rejected)")
    
    return valid


# ── CLI ──
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze frame quality (blur detection)")
    parser.add_argument("image_dir", help="Directory containing frame images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Manual blur threshold (default: auto)")
    parser.add_argument("--show-rejected", action="store_true",
                        help="Print details of rejected frames")
    args = parser.parse_args()
    
    result = analyze_frames(args.image_dir, args.threshold)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    save_manifest(args.image_dir, result)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Total:     {result['total_frames']}")
    print(f"Valid:     {result['valid_frames']}")
    print(f"Rejected: {result['rejected_frames']}")
    print(f"Threshold: {result['threshold']}")
    print(f"Stats:     min={result['stats']['min']}, max={result['stats']['max']}, "
          f"mean={result['stats']['mean']}, median={result['stats']['median']}")
    
    if args.show_rejected:
        print(f"\nRejected frames:")
        for f in result["frames"]:
            if not f["valid"]:
                print(f"  {f['file']}: score={f['blur_score']}, diff={f['inter_frame_diff']}")
