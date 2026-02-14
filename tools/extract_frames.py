#!/usr/bin/env python3
"""
Extract frames from a video file.

Usage:
    python extract_frames.py /path/to/video.mp4 --output /path/to/frames/
    python extract_frames.py video.mp4 --output frames/ --fps 5
    python extract_frames.py video.mp4 --output frames/ --every 10
"""

import argparse
import os
import sys
import cv2


def extract_frames(video_path: str, output_dir: str,
                   fps: float = None, every: int = None,
                   format: str = "jpg", quality: int = 95):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Extract at this framerate (None = all frames)
        every: Extract every N-th frame (None = all frames)
        format: Output image format (jpg, png)
        quality: JPEG quality (1-100) or PNG compression (0-9)
    """
    if not os.path.isfile(video_path):
        print(f"Error: video not found: {video_path}")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Output: {output_dir}")
    
    # Determine frame skip interval
    if fps and fps > 0:
        skip = max(1, round(video_fps / fps))
        print(f"  Extracting at ~{fps} fps (every {skip} frames)")
    elif every and every > 1:
        skip = every
        print(f"  Extracting every {every} frames")
    else:
        skip = 1
        print(f"  Extracting all frames")
    
    # Encode params
    if format.lower() == "png":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, min(quality, 9)]
    else:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    
    frame_idx = 0
    saved = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip == 0:
            filename = f"{frame_idx:06d}{ext}"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, params)
            saved += 1
            
            if saved % 100 == 0:
                print(f"  Saved {saved} frames...", end="\r")
        
        frame_idx += 1
    
    cap.release()
    print(f"\n✅ Extracted {saved} frames from {frame_idx} total → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=None,
                        help="Extract at this framerate (default: all frames)")
    parser.add_argument("--every", type=int, default=None,
                        help="Extract every N-th frame")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Output format (default: jpg)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality 1-100 or PNG compression 0-9 (default: 95)")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps, args.every,
                   args.format, args.quality)


if __name__ == "__main__":
    main()
