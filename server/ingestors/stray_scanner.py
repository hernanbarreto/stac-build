# STAC-Builder: Stray Scanner Ingestor
# Converts Stray Scanner iOS LiDAR captures into format for MapAnythingV2.
#
# Stray Scanner output format:
#   camera_matrix.csv  - 3x3 intrinsic matrix K
#   odometry.csv       - per-frame ARKit poses (timestamp, frame, x,y,z, qx,qy,qz,qw)
#   imu.csv            - high-rate IMU data (not used directly)
#   rgb.mp4            - video at capture resolution (e.g., 1920x1440 @ 60fps)
#   depth/             - uint16 PNGs, LiDAR depth in mm (192x256)
#   confidence/        - uint8 PNGs, LiDAR confidence levels (0=low,1=med,2=high)
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation


def load_intrinsics(csv_path: str) -> np.ndarray:
    """Load 3x3 camera intrinsic matrix from Stray Scanner's camera_matrix.csv.
    
    Format: 3 rows of comma-separated values (fx, 0, cx / 0, fy, cy / 0, 0, 1).
    
    Returns:
        K: (3, 3) numpy array
    """
    rows = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x.strip()) for x in line.split(',')]
            rows.append(vals)
    K = np.array(rows[:3], dtype=np.float64)
    assert K.shape == (3, 3), f"Expected 3x3 matrix, got {K.shape}"
    return K


def load_odometry(csv_path: str) -> Tuple[List[int], List[np.ndarray]]:
    """Load per-frame poses from Stray Scanner's odometry.csv.
    
    ARKit quaternion+translation are used DIRECTLY as C2W matrices.
    No coordinate conversion is applied — ARKit's camera intrinsics
    already use the standard pinhole convention (X-right, Y-down, Z-forward)
    for projection, so the poses are compatible with standard backprojection.
    
    See: https://github.com/kekeblom/StrayVisualizer/blob/main/stray_visualize.py
    
    Returns:
        frame_indices: list of int frame indices
        poses: list of (4, 4) C2W matrices
    """
    frame_indices = []
    poses = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        
        for row in reader:
            if len(row) < 9:
                continue
            
            frame_idx = int(row[1].strip())
            x, y, z = float(row[2]), float(row[3]), float(row[4])
            qx, qy, qz, qw = float(row[5]), float(row[6]), float(row[7]), float(row[8])
            
            # Build 4x4 C2W matrix from quaternion + translation
            # No conversion needed — ARKit poses work directly with
            # standard pinhole backprojection (verified against StrayVisualizer)
            rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses (x,y,z,w)
            T_c2w = np.eye(4)
            T_c2w[:3, :3] = rot.as_matrix()
            T_c2w[:3, 3] = [x, y, z]
            
            frame_indices.append(frame_idx)
            poses.append(T_c2w)
    
    return frame_indices, poses


def load_depth(depth_dir: str, frame_idx: int) -> np.ndarray:
    """Load a single depth map from Stray Scanner.
    
    Stray Scanner stores depth as uint16 PNG where values are in millimeters.
    
    Returns:
        depth: (H, W) float32 array in meters
    """
    path = Path(depth_dir) / f"{frame_idx:06d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Depth frame not found: {path}")
    
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"Failed to read depth: {path}")
    
    # Convert uint16 mm → float32 meters
    return d.astype(np.float32) / 1000.0


def load_confidence(conf_dir: str, frame_idx: int) -> np.ndarray:
    """Load confidence map from Stray Scanner.
    
    Values: 0=low, 1=medium, 2=high confidence.
    
    Returns:
        mask: (H, W) bool array (True where confidence >= threshold)
    """
    path = Path(conf_dir) / f"{frame_idx:06d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Confidence frame not found: {path}")
    
    c = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if c is None:
        raise IOError(f"Failed to read confidence: {path}")
    
    return c


def extract_frames(
    mp4_path: str,
    output_dir: str,
    max_frames: int = 0,
    stride: int = 1,
) -> int:
    """Extract RGB frames from Stray Scanner's rgb.mp4.
    
    Args:
        mp4_path: Path to rgb.mp4
        output_dir: Directory to save extracted JPGs
        max_frames: Max frames to extract (0 = all)
        stride: Extract every Nth frame
    
    Returns:
        Number of frames extracted
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % stride == 0:
            fname = out / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
            
            if max_frames > 0 and extracted >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    print(f"[StrayScanner] Extracted {extracted}/{total} frames (stride={stride}) → {output_dir}")
    return extracted


def upsample_depth_to_rgb(
    depth: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Upsample LiDAR depth (192x256) to RGB resolution.
    
    Uses INTER_NEAREST to avoid blending depth values at edges.
    Zero-depth pixels stay zero.
    
    Returns:
        depth_upsampled: (target_h, target_w) float32 in meters
    """
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def rescale_intrinsics(K: np.ndarray, orig_w: int, orig_h: int, 
                       new_w: int, new_h: int) -> np.ndarray:
    """Rescale intrinsic matrix for a different resolution.
    
    Returns:
        K_new: (3, 3) rescaled intrinsic matrix
    """
    sx = new_w / orig_w
    sy = new_h / orig_h
    K_new = K.copy()
    K_new[0, :] *= sx
    K_new[1, :] *= sy
    return K_new


def prepare_stray_data(
    data_dir: str,
    frames_output_dir: str,
    stride: int = 2,
    max_frames: int = 0,
    confidence_threshold: int = 1,
) -> dict:
    """Full preparation pipeline for Stray Scanner data.
    
    Extracts frames, loads all priors, and returns a dict with everything
    needed for MapAnythingV2 multi-modal inference.
    
    Args:
        data_dir: Path to Stray Scanner capture (contains rgb.mp4, depth/, etc.)
        frames_output_dir: Where to save extracted JPGs
        stride: Frame subsampling stride (60fps → 30fps with stride=2)
        max_frames: Max frames to use (0 = all)
        confidence_threshold: Min confidence level (0-2)
    
    Returns:
        dict with keys:
            frame_paths: list of extracted frame paths
            frame_indices: list of original frame indices
            intrinsics: (3, 3) K matrix (scaled to depth resolution)
            intrinsics_rgb: (3, 3) K matrix (original RGB resolution)
            poses: list of (4, 4) C2W matrices (raw ARKit, no conversion)
            depths: list of (H, W) float32 depth maps (native 192x256)
            conf_masks: list of (H, W) bool confidence masks (native 192x256)
            rgb_shape: (H, W) of the RGB frames
            depth_shape: (H, W) of the native depth maps (192, 256)
    """
    data = Path(data_dir)
    
    # 1. Load intrinsics (for RGB resolution)
    K_rgb = load_intrinsics(str(data / "camera_matrix.csv"))
    print(f"[StrayScanner] K (RGB): fx={K_rgb[0,0]:.1f} fy={K_rgb[1,1]:.1f} "
          f"cx={K_rgb[0,2]:.1f} cy={K_rgb[1,2]:.1f}")
    
    # 2. Load odometry (raw ARKit poses, no coordinate conversion)
    odo_frame_indices, odo_poses = load_odometry(str(data / "odometry.csv"))
    pose_map = dict(zip(odo_frame_indices, odo_poses))
    print(f"[StrayScanner] Loaded {len(odo_poses)} odometry poses")
    
    # 3. Extract frames
    n_extracted = extract_frames(
        str(data / "rgb.mp4"),
        frames_output_dir,
        max_frames=max_frames,
        stride=stride,
    )
    
    # 4. Get extracted frame list and their indices
    frames_dir = Path(frames_output_dir)
    frame_files = sorted(frames_dir.glob("*.jpg"))
    
    # Read one frame to get RGB resolution
    sample = cv2.imread(str(frame_files[0]))
    rgb_h, rgb_w = sample.shape[:2]
    print(f"[StrayScanner] RGB resolution: {rgb_w}x{rgb_h}")
    
    # 5. Match frames with depth/pose/confidence at NATIVE depth resolution
    # Stray Scanner depth is 256x192 (W x H). We keep it at native resolution
    # and scale intrinsics accordingly, matching the StrayVisualizer approach.
    frame_paths = []
    frame_indices = []
    poses = []
    depths = []
    conf_masks = []
    depth_h, depth_w = None, None
    
    depth_dir = str(data / "depth")
    conf_dir = str(data / "confidence")
    
    for fpath in frame_files:
        fidx = int(fpath.stem)  # e.g., "000042" → 42
        
        # Check all priors exist for this frame
        try:
            d = load_depth(depth_dir, fidx)
            c = load_confidence(conf_dir, fidx)
        except FileNotFoundError:
            continue
        
        if fidx not in pose_map:
            continue
        
        if depth_h is None:
            depth_h, depth_w = d.shape
        
        # Apply confidence mask at native resolution (no upsampling)
        # Apply confidence mask at native resolution (no upsampling)
        mask = c >= confidence_threshold
        d[~mask] = 0.0
        
        frame_paths.append(str(fpath))
        frame_indices.append(fidx)
        poses.append(pose_map[fidx])
        depths.append(d)
        conf_masks.append(c)  # return raw [0, 1, 2] instead of boolean
    
    # Scale intrinsics from RGB resolution to depth resolution
    K_depth = rescale_intrinsics(K_rgb, rgb_w, rgb_h, depth_w, depth_h)
    print(f"[StrayScanner] K (depth): fx={K_depth[0,0]:.1f} fy={K_depth[1,1]:.1f} "
          f"cx={K_depth[0,2]:.1f} cy={K_depth[1,2]:.1f}")
    print(f"[StrayScanner] Depth resolution: {depth_w}x{depth_h}")
    print(f"[StrayScanner] {len(frame_paths)} frames with complete priors "
          f"(of {n_extracted} extracted)")
    
    return {
        "frame_paths": frame_paths,
        "frame_indices": frame_indices,
        "intrinsics": K_depth,
        "intrinsics_rgb": K_rgb,
        "poses": poses,
        "depths": depths,
        "conf_masks": conf_masks,
        "rgb_shape": (rgb_h, rgb_w),
        "depth_shape": (depth_h, depth_w),
    }
