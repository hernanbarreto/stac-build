"""
STAC Build — Occlusion Ray-Caster
===================================

Classifies BIM surface samples by tracing rays from camera positions
through the scan point cloud to determine visibility.

For each BIM surface sample:
  - COVERED:     Camera has line-of-sight AND scan points confirm construction
  - OCCLUDED:    Something blocks the camera's view of this BIM surface
  - NOT_BUILT:   Camera can see the location but no scan points → not constructed
  - NOT_VISIBLE: No camera had a favorable viewing angle for this surface

Uses cylindrical ray queries on a KDTree for efficient spatial lookups.

Authors: Hernán Barreto — Ingerop IN3
"""
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from coverage_store import SampleStatus, ScanCoverageResult, OccluderType


# ═══════════════════════════════════════════════════════════════════
#  CAMERA POSE LOADING
# ═══════════════════════════════════════════════════════════════════

def load_camera_positions(session_dir: str) -> np.ndarray:
    """
    Load camera world positions from reconstruction extrinsics.

    Reconstruction stores world-to-camera (w2c) matrices [3, 4].
    Camera world position = -R^T @ t = c2w[:3, 3]

    Tries (in order):
      1. output/chunk_*_meta.json → cameras dict (richest, per-frame)
      2. output/camera_poses.txt  (saved by reconstruction pipeline)

    Returns:
        (C, 3) array of camera world positions, or empty (0, 3) if none found.
    """
    session_path = Path(session_dir)
    output_dir = session_path / "output"

    # ── Try chunk metadata first (most complete) ──
    positions = _load_from_chunk_meta(output_dir)
    if positions is not None and len(positions) > 0:
        print(f"[RayCaster] Loaded {len(positions)} camera positions from chunk metadata")
        return positions

    # ── Fallback: camera_poses.txt ──
    poses_txt = output_dir / "camera_poses.txt"
    if poses_txt.exists():
        positions = _load_from_poses_txt(poses_txt)
        if positions is not None and len(positions) > 0:
            print(f"[RayCaster] Loaded {len(positions)} camera positions from camera_poses.txt")
            return positions

    print("[RayCaster] WARNING: No camera pose data found")
    return np.empty((0, 3), dtype=np.float32)


def _load_from_chunk_meta(output_dir: Path) -> Optional[np.ndarray]:
    """Extract camera world positions from chunk_*_meta.json files."""
    meta_files = sorted(output_dir.glob("chunk_*_meta.json"))
    if not meta_files:
        return None

    all_positions = []
    for meta_file in meta_files:
        try:
            meta = json.loads(meta_file.read_text())
            cameras = meta.get("cameras", {})
            if not cameras:
                continue
            for cam_key in sorted(cameras.keys(), key=int):
                cam = cameras[cam_key]
                w2c = np.array(cam["extrinsics"], dtype=np.float32)
                pos = _w2c_to_position(w2c)
                all_positions.append(pos)
        except Exception as e:
            print(f"[RayCaster] Error reading {meta_file.name}: {e}")
            continue

    if not all_positions:
        return None
    return np.array(all_positions, dtype=np.float32)


def _load_from_poses_txt(poses_txt: Path) -> Optional[np.ndarray]:
    """
    Load camera poses from camera_poses.txt.
    Format: one line per camera, 12 values (3x4 w2c matrix, row-major).
    """
    positions = []
    try:
        with open(poses_txt) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 12:
                    w2c = np.array([float(v) for v in vals[:12]], dtype=np.float32).reshape(3, 4)
                    positions.append(_w2c_to_position(w2c))
    except Exception as e:
        print(f"[RayCaster] Error reading camera_poses.txt: {e}")
        return None
    if not positions:
        return None
    return np.array(positions, dtype=np.float32)


def _w2c_to_position(w2c: np.ndarray) -> np.ndarray:
    """
    Convert world-to-camera [3,4] matrix to camera world position.
    c2w = inv(w2c) → camera_pos = c2w[:3, 3] = -R^T @ t
    """
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return -R.T @ t


# ═══════════════════════════════════════════════════════════════════
#  CYLINDRICAL RAY QUERY
# ═══════════════════════════════════════════════════════════════════

def cylindrical_query(
    tree: KDTree,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    max_dist: float,
    radius: float,
    cloud_points: np.ndarray,
) -> np.ndarray:
    """
    Find points within a cylinder along a ray.

    The cylinder extends from ray_origin along ray_dir for max_dist,
    with the given radius. Points are returned as indices into cloud_points.

    Args:
        tree: KDTree built on cloud_points
        ray_origin: (3,) start of ray
        ray_dir: (3,) normalized direction
        max_dist: length of cylinder
        radius: radius of cylinder
        cloud_points: (N, 3) the actual point coordinates

    Returns:
        indices: (K,) indices of points inside the cylinder
    """
    # Query a ball at the midpoint of the ray with radius = half-length + cylinder radius
    # This is a conservative bounding query
    midpoint = ray_origin + ray_dir * (max_dist / 2)
    search_radius = max_dist / 2 + radius

    candidate_indices = tree.query_ball_point(midpoint, search_radius)
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int64)

    candidates = cloud_points[candidate_indices]

    # Project candidates onto ray axis
    # t = dot(candidate - origin, dir)  → parametric position along ray
    diff = candidates - ray_origin
    t = np.dot(diff, ray_dir)

    # Filter: must be within [0, max_dist] along ray
    along_mask = (t >= 0) & (t <= max_dist)

    # Perpendicular distance to ray axis
    proj = ray_origin + t[:, None] * ray_dir
    perp_dist = np.linalg.norm(candidates - proj, axis=1)

    # Filter: must be within cylinder radius
    inside_mask = along_mask & (perp_dist <= radius)

    result_indices = np.array(candidate_indices, dtype=np.int64)[inside_mask]
    return result_indices


# ═══════════════════════════════════════════════════════════════════
#  VECTORIZED BEST CAMERA SELECTION
# ═══════════════════════════════════════════════════════════════════

def find_best_cameras(
    samples: np.ndarray,
    normals: np.ndarray,
    cam_positions: np.ndarray,
    min_cos_angle: float = -0.15,
    top_k: int = 3,
) -> np.ndarray:
    """
    For each BIM surface sample, find the best camera(s) that have a
    favorable viewing angle (camera looks toward the surface, not away).

    A camera is favorable if:
      cos(view_dir, normal) < min_cos_angle  (negative = facing surface)

    Args:
        samples: (M, 3) BIM surface sample positions
        normals: (M, 3) surface normals at each sample
        cam_positions: (C, 3) camera world positions
        min_cos_angle: threshold for favorable angle (default -0.15 ≈ 81°)
        top_k: number of best cameras to return per sample

    Returns:
        best_cameras: (M, top_k) indices into cam_positions, -1 if none found
    """
    M = len(samples)
    C = len(cam_positions)

    if C == 0:
        return np.full((M, top_k), -1, dtype=np.int32)

    # Build KDTree on camera positions for fast nearest-neighbor
    cam_tree = KDTree(cam_positions)

    # Query K nearest cameras per sample (search more than top_k to filter by angle)
    k_search = min(C, top_k * 5)
    distances, indices = cam_tree.query(samples, k=k_search)

    # Handle case where k_search=1 (returns 1D arrays)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    best = np.full((M, top_k), -1, dtype=np.int32)

    for i in range(M):
        count = 0
        for j in range(k_search):
            cam_idx = indices[i, j]
            cam_pos = cam_positions[cam_idx]

            # View direction: camera → sample (normalized)
            view_dir = samples[i] - cam_pos
            dist = np.linalg.norm(view_dir)
            if dist < 1e-6:
                continue
            view_dir /= dist

            # Check angle: cos(view_dir, normal) should be negative
            # (camera facing the front of the surface)
            cos_angle = np.dot(view_dir, normals[i])
            if cos_angle < min_cos_angle:
                best[i, count] = cam_idx
                count += 1
                if count >= top_k:
                    break

    return best


# ═══════════════════════════════════════════════════════════════════
#  MAIN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def classify_bim_surface(
    cam_positions: np.ndarray,
    scan_cloud: np.ndarray,
    bim_samples: np.ndarray,
    bim_normals: np.ndarray,
    proximity_m: float = 0.15,
    ray_radius: float = 0.05,
    seg_labels: Optional[np.ndarray] = None,
) -> ScanCoverageResult:
    """
    Classify each BIM surface sample as COVERED, OCCLUDED, NOT_BUILT, or NOT_VISIBLE.

    Args:
        cam_positions: (C, 3) camera world positions
        scan_cloud: (N, 3) scan point cloud
        bim_samples: (M, 3) BIM surface sample points
        bim_normals: (M, 3) surface normals at each sample
        proximity_m: max distance for a scan point to count as "covering" a BIM sample
        ray_radius: cylinder radius for occlusion ray query
        seg_labels: (N,) optional per-point segment labels for occluder identification

    Returns:
        ScanCoverageResult with per-sample status and metadata
    """
    M = len(bim_samples)
    N = len(scan_cloud)
    C = len(cam_positions)

    t0 = time.time()
    print(f"[RayCaster] Classifying {M} BIM samples with {N} scan points, {C} cameras")

    # Initialize arrays
    status = np.full(M, SampleStatus.NOT_VISIBLE, dtype=np.int8)
    deviation = np.full(M, np.nan, dtype=np.float32)
    occluder_labels = np.array([""] * M, dtype=object)
    occluder_types = np.zeros(M, dtype=np.int8)

    if M == 0 or N == 0 or C == 0:
        return ScanCoverageResult(
            scan_id="", date="",
            status=status, deviation=deviation,
            occluder_labels=occluder_labels, occluder_types=occluder_types,
        )

    # ── Step 1: KDTree on scan cloud ──
    scan_tree = KDTree(scan_cloud[:, :3])

    # ── Step 2: Find best cameras per sample ──
    best_cameras = find_best_cameras(bim_samples, bim_normals, cam_positions)

    # ── Step 3: Proximity check — scan points near BIM surface ──
    # This is fast and tells us which samples have scan data nearby
    prox_dists, prox_indices = scan_tree.query(bim_samples)

    # ── Step 4: Classify each sample ──
    n_covered = 0
    n_occluded = 0
    n_not_built = 0
    n_not_visible = 0

    for i in range(M):
        # Find first valid camera for this sample
        cam_idx = -1
        for k in range(best_cameras.shape[1]):
            if best_cameras[i, k] >= 0:
                cam_idx = best_cameras[i, k]
                break

        if cam_idx < 0:
            status[i] = SampleStatus.NOT_VISIBLE
            n_not_visible += 1
            continue

        # Check proximity first (fast path)
        if prox_dists[i] <= proximity_m:
            # Scan point close to BIM surface → COVERED
            status[i] = SampleStatus.COVERED
            deviation[i] = prox_dists[i]
            n_covered += 1
            continue

        # ── Ray-casting for occlusion detection ──
        cam_pos = cam_positions[cam_idx]
        ray_dir = bim_samples[i] - cam_pos
        ray_length = np.linalg.norm(ray_dir)
        if ray_length < 1e-6:
            status[i] = SampleStatus.NOT_VISIBLE
            n_not_visible += 1
            continue
        ray_dir /= ray_length

        # Check for scan points along the ray (before the BIM surface)
        # Use 90% of ray length to avoid false positives near the BIM surface itself
        blocking_indices = cylindrical_query(
            scan_tree, cam_pos, ray_dir,
            max_dist=ray_length * 0.85,
            radius=ray_radius,
            cloud_points=scan_cloud[:, :3],
        )

        if len(blocking_indices) > 0:
            # Something is blocking the view → OCCLUDED
            status[i] = SampleStatus.OCCLUDED
            n_occluded += 1

            # Identify occluder from SAM3 labels if available
            if seg_labels is not None:
                labels_at_block = seg_labels[blocking_indices]
                # Most common non-empty label
                non_empty = labels_at_block[labels_at_block != ""]
                if len(non_empty) > 0:
                    unique, counts = np.unique(non_empty, return_counts=True)
                    occluder_labels[i] = unique[np.argmax(counts)]
        else:
            # Camera can see the location, no scan data → NOT_BUILT
            status[i] = SampleStatus.NOT_BUILT
            n_not_built += 1

    elapsed = time.time() - t0
    print(f"[RayCaster] Classification done in {elapsed:.2f}s:")
    print(f"  COVERED={n_covered}  OCCLUDED={n_occluded}  "
          f"NOT_BUILT={n_not_built}  NOT_VISIBLE={n_not_visible}")

    return ScanCoverageResult(
        scan_id="",
        date="",
        status=status,
        deviation=deviation,
        occluder_labels=occluder_labels,
        occluder_types=occluder_types,
    )


# ═══════════════════════════════════════════════════════════════════
#  BUILD PER-POINT SEGMENT LABEL ARRAY
# ═══════════════════════════════════════════════════════════════════

def build_segment_labels(
    n_points: int,
    segmentation_path: str,
) -> Optional[np.ndarray]:
    """
    Build a per-point label array from segmentation data.
    
    Args:
        n_points: total points in the scan cloud
        segmentation_path: path to segmentation_result.json

    Returns:
        (N,) object array with label string per point, or None if unavailable
    """
    path = Path(segmentation_path)
    if not path.exists():
        return None

    try:
        seg_data = json.loads(path.read_text())
    except Exception:
        return None

    labels = np.array([""] * n_points, dtype=object)

    for inst in seg_data.get("instances", []):
        label = str(inst.get("label", inst.get("category", "")))
        indices = inst.get("globalIndices", inst.get("point_indices", []))
        for idx in indices:
            if 0 <= idx < n_points:
                labels[idx] = label

    return labels
