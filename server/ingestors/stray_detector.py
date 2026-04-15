"""
Stray Scanner Data Detector

Detects whether a session directory contains Stray Scanner LiDAR/ARKit data.
Used by the reconstruction pipeline to determine available backends.

Stray Scanner data layout (in source_dir/session_dir):
    depth/              - LiDAR depth maps (.bin 256×192 float32)
    confidence/         - Depth confidence maps (.bin 256×192 uint8)
    odometry.csv        - Per-frame ARKit poses (timestamp, frame, qx,qy,qz,qw, x,y,z)
    camera_matrix.csv   - Camera intrinsics matrix (3×3)
    rgb.mp4             - RGB video (1920×1440)

Authors: Hernán Barreto — Ingerop IN3
"""

from pathlib import Path
from typing import Optional


def detect_stray_data(session_dir) -> dict:
    """Detect Stray Scanner data in a session directory.

    Checks the session_dir (source_dir) for the presence of Stray Scanner
    artifacts: depth maps, odometry poses, and camera intrinsics.

    Args:
        session_dir: Path to session root (e.g., src_xxx/ or scans/session_id/).
                     Can be str or Path.

    Returns:
        dict with keys:
            has_lidar (bool):       True if depth/ and confidence/ dirs exist
            has_arkit (bool):       True if odometry.csv exists
            has_intrinsics (bool):  True if camera_matrix.csv exists
            is_stray_session (bool): True if all three are present (full Stray Scanner data)
            depth_dir (Path|None):  Path to depth/ directory
            odometry (Path|None):   Path to odometry.csv
            camera_matrix (Path|None): Path to camera_matrix.csv
            depth_count (int):      Number of depth files found
    """
    session = Path(session_dir)

    depth_dir = session / "depth"
    confidence_dir = session / "confidence"
    odometry = session / "odometry.csv"
    camera_matrix = session / "camera_matrix.csv"

    has_lidar = depth_dir.is_dir() and confidence_dir.is_dir()
    has_arkit = odometry.is_file()
    has_intrinsics = camera_matrix.is_file()

    depth_count = 0
    if has_lidar:
        depth_count = len(list(depth_dir.glob("*.bin")))

    return {
        "has_lidar": has_lidar,
        "has_arkit": has_arkit,
        "has_intrinsics": has_intrinsics,
        "is_stray_session": has_lidar and has_arkit and has_intrinsics,
        "depth_dir": depth_dir if has_lidar else None,
        "odometry": odometry if has_arkit else None,
        "camera_matrix": camera_matrix if has_intrinsics else None,
        "depth_count": depth_count,
    }


def require_stray_data(session_dir, backend: str) -> dict:
    """Detect Stray Scanner data and raise if required data is missing.

    Args:
        session_dir: Path to session root
        backend: "lidar" or "hybrid"

    Returns:
        Stray detection dict (same as detect_stray_data)

    Raises:
        FileNotFoundError: If required files are missing
    """
    info = detect_stray_data(session_dir)

    if not info["is_stray_session"]:
        missing = []
        if not info["has_lidar"]:
            missing.append("depth/")
        if not info["has_arkit"]:
            missing.append("odometry.csv")
        if not info["has_intrinsics"]:
            missing.append("camera_matrix.csv")
        raise FileNotFoundError(
            f"Backend '{backend}' requires Stray Scanner data, "
            f"but missing: {', '.join(missing)} in {session_dir}"
        )

    return info
