"""
STAC Build — BIM vs Scan Registration
Hierarchical alignment: floor planes first, then objects.

Generic system: detects floor elements from IFC type (IfcSlab, IfcCovering,
etc.) automatically. No hardcoded element names.

Hernán Barreto — Ingerop IN3
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

# ── IFC types that represent horizontal surfaces (floors/slabs) ──
FLOOR_IFC_TYPES = frozenset({
    'IfcSlab',
    'IfcSlabStandardCase',
    'IfcCovering',          # floor coverings
    'IfcBuildingElementProxy',  # sometimes used for floors
})

# ── IFC types that are NOT useful for object registration ──
SKIP_IFC_TYPES = frozenset({
    'IfcSpace',
    'IfcOpeningElement',
    'IfcSite',
    'IfcBuilding',
    'IfcBuildingStorey',
    'IfcProject',
})


# ═══════════════════════════════════════════════════════════════════
#  LOAD FLOOR TRANSFORM
# ═══════════════════════════════════════════════════════════════════

def load_floor_transform(session_dir: str) -> Optional[np.ndarray]:
    """
    Load the floor_transform.npz from the session output.
    This contains R (3x3 rotation), t (3 translation), s (scale)
    that together transform raw sensor coordinates to viewer coordinates.
    
    Returns 4x4 transformation matrix, or None if not found.
    """
    transform_path = Path(session_dir) / "output" / "floor_transform.npz"
    if not transform_path.exists():
        print(f"[Registration] No floor_transform.npz found at {transform_path}")
        return None
    
    data = np.load(str(transform_path))
    R = data['R']  # 3x3 rotation
    t = data['t']  # 3 translation
    s = float(data['s'])  # scale
    
    # Build 4x4 matrix: T(p) = s * R @ p + t
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    
    print(f"[Registration] Loaded floor transform: scale={s}, "
          f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    return T


# ═══════════════════════════════════════════════════════════════════
#  PLANE FITTING
# ═══════════════════════════════════════════════════════════════════

def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit a plane to points using SVD (least squares).
    Returns (normal, d) where normal·p + d = 0 for points on the plane.
    Normal is oriented to point "up" (positive Y).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # last row = smallest singular value = normal
    
    # Ensure normal points "up" (positive Y in Three.js convention)
    if normal[1] < 0:
        normal = -normal
    
    d = -np.dot(normal, centroid)
    return normal, d


def fit_plane_ransac(
    points: np.ndarray,
    n_iter: int = 200,
    threshold: float = 0.02,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC plane fitting — robust to outliers.
    Returns (normal, d, inlier_mask).
    """
    best_inliers = 0
    best_normal = np.array([0., 1., 0.])
    best_d = 0.
    best_mask = np.zeros(len(points), dtype=bool)
    
    rng = np.random.default_rng(42)
    n = len(points)
    
    for _ in range(n_iter):
        # Sample 3 random points
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        
        # Compute plane normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        
        # Ensure up-facing (Y-up)
        if normal[1] < 0:
            normal = -normal
        
        d = -np.dot(normal, p0)
        
        # Count inliers
        distances = np.abs(points @ normal + d)
        mask = distances < threshold
        n_inliers = mask.sum()
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d
            best_mask = mask
    
    # Refit with all inliers
    if best_inliers >= 3:
        best_normal, best_d = fit_plane_svd(points[best_mask])
    
    print(f"[Registration] Plane fit: {best_inliers}/{n} inliers "
          f"({best_inliers/n*100:.0f}%), normal={best_normal}")
    return best_normal, best_d, best_mask


# ═══════════════════════════════════════════════════════════════════
#  ROTATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix that rotates v_from to v_to.
    Uses Rodrigues' rotation formula.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    
    R = np.eye(3) + K + K @ K / (1 + dot)
    return R


def _yaw_matrix(theta: float) -> np.ndarray:
    """3x3 rotation matrix around Y axis by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


# ═══════════════════════════════════════════════════════════════════
#  FLOOR ALIGNMENT (Height + Tilt → 3 DOF)
# ═══════════════════════════════════════════════════════════════════

def align_floors(
    scan_floor_pts: np.ndarray,
    bim_floor_normal: np.ndarray,
    bim_floor_d: float,
) -> np.ndarray:
    """
    Compute 4x4 transform that aligns the scan floor plane to the BIM floor plane.
    Solves 3 DOF: height (Y) + tilt (2 rotation axes).
    
    Returns 4x4 homogeneous transformation matrix.
    """
    # Fit plane to scan floor points
    scan_normal, scan_d, _ = fit_plane_ransac(scan_floor_pts)
    
    # Step 1: Rotation to align normals
    R = _rotation_between_vectors(scan_normal, bim_floor_normal)
    
    # Step 2: After rotation, compute height offset
    scan_centroid = scan_floor_pts.mean(axis=0)
    rotated_centroid = R @ scan_centroid
    
    # Distance from rotated centroid to BIM plane
    height_offset = -(np.dot(bim_floor_normal, rotated_centroid) + bim_floor_d)
    
    # Build 4x4: rotate around scan centroid, then translate
    # T(p) = R @ p + (c - R @ c + height_offset * normal)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = scan_centroid - R @ scan_centroid + height_offset * bim_floor_normal
    
    # Verify: transformed scan centroid should be near BIM plane
    tc = T[:3, :3] @ scan_centroid + T[:3, 3]
    residual = abs(np.dot(bim_floor_normal, tc) + bim_floor_d)
    print(f"[Registration] Floor alignment: residual = {residual*1000:.1f}mm")
    
    return T


# ═══════════════════════════════════════════════════════════════════
#  OBJECT ALIGNMENT (XZ Translation + Yaw → 3 DOF)
# ═══════════════════════════════════════════════════════════════════

def _compute_overlap_score(
    scan_pts: np.ndarray,
    bim_verts: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Compute overlap score: fraction of scan points within `threshold`
    distance of the BIM mesh surface.
    """
    from scipy.spatial import KDTree
    tree = KDTree(bim_verts)
    dists, _ = tree.query(scan_pts)
    return float(np.mean(dists < threshold))


def align_objects(
    scan_pts: np.ndarray,
    bim_verts: np.ndarray,
    bim_faces: np.ndarray,
    floor_transform: np.ndarray,
    yaw_steps: int = 72,
    progress_callback=None,
) -> np.ndarray:
    """
    Find XZ translation + yaw rotation that maximizes overlap.
    Assumes floor alignment is already applied.
    
    Returns updated 4x4 transform (floor + object alignment combined).
    """
    # Apply floor transform to scan points
    n = len(scan_pts)
    pts_h = np.hstack([scan_pts, np.ones((n, 1))])
    pts_floor = (floor_transform @ pts_h.T).T[:, :3]
    
    # Initial XZ translation from centroids (projected to horizontal)
    scan_centroid_xz = pts_floor[:, [0, 2]].mean(axis=0)
    bim_centroid_xz = bim_verts[:, [0, 2]].mean(axis=0)
    initial_shift = bim_centroid_xz - scan_centroid_xz
    
    print(f"[Registration] Initial XZ shift: [{initial_shift[0]:.3f}, {initial_shift[1]:.3f}]")
    
    # Yaw sweep: rotate around BIM centroid, looking for best overlap
    bim_center = bim_verts.mean(axis=0)
    best_score = -1.0
    best_theta = 0.0
    
    for i in range(yaw_steps):
        theta = 2 * np.pi * i / yaw_steps
        R_yaw = _yaw_matrix(theta)
        
        # Transform: apply floor, shift XZ, rotate yaw around BIM center
        test_pts = pts_floor.copy()
        test_pts[:, 0] += initial_shift[0]
        test_pts[:, 2] += initial_shift[1]
        
        # Rotate around BIM center
        test_pts -= bim_center
        test_pts = test_pts @ R_yaw.T
        test_pts += bim_center
        
        score = _compute_overlap_score(test_pts, bim_verts)
        
        if score > best_score:
            best_score = score
            best_theta = theta
        
        if progress_callback and i % max(1, yaw_steps // 10) == 0:
            progress_callback(int(i / yaw_steps * 50 + 25),
                            f"Yaw sweep: {i}/{yaw_steps} ({score:.2%})")
    
    print(f"[Registration] Best yaw: {np.degrees(best_theta):.1f}° "
          f"(overlap: {best_score:.2%})")
    
    # Refine around best theta ±5° with fine steps
    fine_range = np.linspace(best_theta - np.radians(5), 
                              best_theta + np.radians(5), 36)
    for theta in fine_range:
        R_yaw = _yaw_matrix(theta)
        test_pts = pts_floor.copy()
        test_pts[:, 0] += initial_shift[0]
        test_pts[:, 2] += initial_shift[1]
        test_pts -= bim_center
        test_pts = test_pts @ R_yaw.T
        test_pts += bim_center
        
        score = _compute_overlap_score(test_pts, bim_verts)
        if score > best_score:
            best_score = score
            best_theta = theta
    
    print(f"[Registration] Refined yaw: {np.degrees(best_theta):.1f}° "
          f"(overlap: {best_score:.2%})")
    
    # Build combined transform: floor → XZ shift → yaw around BIM center
    R_yaw = _yaw_matrix(best_theta)
    
    T_shift = np.eye(4)
    T_shift[0, 3] = initial_shift[0]
    T_shift[2, 3] = initial_shift[1]
    
    T_to_center = np.eye(4)
    T_to_center[:3, 3] = -bim_center
    
    T_from_center = np.eye(4)
    T_from_center[:3, 3] = bim_center
    
    T_yaw = np.eye(4)
    T_yaw[:3, :3] = R_yaw
    
    # Final = from_center @ yaw @ to_center @ shift @ floor
    T_combined = T_from_center @ T_yaw @ T_to_center @ T_shift @ floor_transform
    
    return T_combined


# ═══════════════════════════════════════════════════════════════════
#  FULL REGISTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════

def classify_matches(
    matches: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """
    Classify matches into floor elements and non-floor elements
    based on the IFC type. Fully generic — no hardcoded names.
    """
    floors = []
    objects = []
    
    for m in matches:
        ifc_type = m.get("ifc_type", "")
        if ifc_type in FLOOR_IFC_TYPES:
            floors.append(m)
        elif ifc_type not in SKIP_IFC_TYPES:
            objects.append(m)
    
    print(f"[Registration] Classified: {len(floors)} floor(s), "
          f"{len(objects)} object(s)")
    return floors, objects


def _load_cloud_and_segments(session_dir: str):
    """Load cleaned_cloud.ply and segmentation data."""
    session_path = Path(session_dir)
    output_dir = session_path / "output"
    
    # Load segmentation
    seg_result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"
    if seg_result_path.exists():
        seg_data = json.loads(seg_result_path.read_text())
    elif seg_path.exists():
        seg_data = json.loads(seg_path.read_text())
    else:
        return None, None
    
    label_to_inst = {}
    for inst in seg_data.get("instances", []):
        label_to_inst[str(inst.get("label", ""))] = inst
    
    # Load cloud (binary PLY with float x,y,z + uchar rgb + int frame + 2 short)
    cloud_path = output_dir / "cleaned_cloud.ply"
    if not cloud_path.exists():
        return None, None
    
    import struct
    with open(str(cloud_path), 'rb') as f:
        n_verts = 0
        props = []
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            elif line.startswith('property'):
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == 'end_header':
                break
        
        # Compute stride from properties
        type_sizes = {
            'float': 4, 'double': 8,
            'uchar': 1, 'uint8': 1, 'char': 1, 'int8': 1,
            'ushort': 2, 'uint16': 2, 'short': 2, 'int16': 2,
            'uint': 4, 'uint32': 4, 'int': 4, 'int32': 4,
            'float32': 4, 'float64': 8,
        }
        stride = sum(type_sizes.get(t, 4) for t, _ in props)
        
        # Read all vertices
        raw = f.read(stride * n_verts)
    
    # Extract xyz (first 3 floats)
    xyz = np.empty((n_verts, 3), dtype=np.float32)
    for i in range(n_verts):
        off = i * stride
        xyz[i] = struct.unpack('<3f', raw[off:off+12])
    
    return xyz, label_to_inst


def register(
    session_dir: str,
    matches: List[dict],
    progress_callback=None,
) -> Optional[np.ndarray]:
    """
    Full registration pipeline:
      1. Load floor_transform.npz → apply to raw scan points
      2. Classify matches by IFC type → floors vs objects
      3. Align floor planes (height + tilt)
      4. Align objects (XZ + yaw sweep)
    
    Returns 4x4 transformation matrix, or None if registration fails.
    """
    from bim_comparison import extract_all_ifc_triangles
    
    session_path = Path(session_dir)
    t0 = time.time()
    
    # ── Load floor transform (sensor → viewer coords) ──
    T_floor = load_floor_transform(session_dir)
    if T_floor is None:
        print("[Registration] WARNING: No floor transform — using identity")
        T_floor = np.eye(4)
    
    # ── Load scan points + segments ──
    xyz_raw, label_to_inst = _load_cloud_and_segments(session_dir)
    if xyz_raw is None:
        print("[Registration] Could not load scan data")
        return None
    
    # Apply floor transform to ALL points first (sensor → viewer space)
    n = len(xyz_raw)
    pts_h = np.hstack([xyz_raw, np.ones((n, 1))])
    xyz_viewer = (T_floor @ pts_h.T).T[:, :3]
    
    print(f"[Registration] Scan in viewer space: "
          f"X=[{xyz_viewer[:,0].min():.2f}, {xyz_viewer[:,0].max():.2f}] "
          f"Y=[{xyz_viewer[:,1].min():.2f}, {xyz_viewer[:,1].max():.2f}] "
          f"Z=[{xyz_viewer[:,2].min():.2f}, {xyz_viewer[:,2].max():.2f}]")
    
    if progress_callback:
        progress_callback(5, "Loaded scan data")
    
    # ── Classify matches ──
    floors, objects = classify_matches(matches)
    
    if not floors and not objects:
        print("[Registration] No matchable elements found")
        return None
    
    # ── Extract IFC meshes ──
    # Resolve IFC location (ifcs_dir for new-style, session_path for legacy)
    from project_paths import resolve_session
    _sd = session_path
    if 'projects' in _sd.parts:
        _slug = _sd.parts[_sd.parts.index('projects') + 1]
    else:
        _slug = _sd.name
    _ctx = resolve_session(str(Path(__file__).parent), _slug)
    _ifcs_dir = _ctx.ifcs_dir if hasattr(_ctx, 'ifcs_dir') else session_path
    ifc_files = list(_ifcs_dir.glob("*.ifc"))
    if not ifc_files:
        ifc_files = list(session_path.glob("*.ifc"))
    if not ifc_files:
        print("[Registration] No IFC file found")
        return None
    
    all_keys = [str(m["element_key"]) for m in floors + objects]
    ifc_meshes = extract_all_ifc_triangles(str(ifc_files[0]), all_keys)
    
    # Print BIM info for debugging
    for key, (v, f) in ifc_meshes.items():
        print(f"[Registration] BIM '{key}': "
              f"X=[{v[:,0].min():.2f}, {v[:,0].max():.2f}] "
              f"Y=[{v[:,1].min():.2f}, {v[:,1].max():.2f}] "
              f"Z=[{v[:,2].min():.2f}, {v[:,2].max():.2f}]")
    
    if progress_callback:
        progress_callback(15, "Extracted IFC meshes")
    
    # Start with floor transform as base (scanner → viewer conversion)
    T = T_floor.copy()
    
    # ── Step 1: Floor alignment ──
    if floors:
        print(f"[Registration] Step 1: Aligning {len(floors)} floor(s)")
        
        all_floor_scan = []
        bim_floor_normals = []
        bim_floor_ds = []
        
        for fm in floors:
            label = str(fm["segment_label"])
            key = str(fm["element_key"])
            
            inst = label_to_inst.get(label)
            if inst is None:
                continue
            
            indices = inst.get("globalIndices", inst.get("point_indices", []))
            if not indices:
                continue
            
            # Use viewer-space scan points
            scan_pts = xyz_viewer[indices]
            all_floor_scan.append(scan_pts)
            
            if key in ifc_meshes:
                bim_v, bim_f = ifc_meshes[key]
                # For floor slabs, use the TOP surface (max Y) not centroid
                # The slab is a box — SVD on all 8 verts gives the centroid plane
                bim_floor_y = float(bim_v[:, 1].max())  # top surface Y
                bim_normal = np.array([0., 1., 0.])
                bim_d = -bim_floor_y  # normal·p + d = 0 → y + d = 0 → d = -y_top
                bim_floor_normals.append(bim_normal)
                bim_floor_ds.append(bim_d)
                print(f"[Registration] BIM floor '{key}' top surface: Y={bim_floor_y:.3f}")
        
        if all_floor_scan and bim_floor_normals:
            combined_scan_floor = np.vstack(all_floor_scan)
            avg_normal = np.mean(bim_floor_normals, axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            avg_d = np.mean(bim_floor_ds)
            
            T_floor_align = align_floors(combined_scan_floor, avg_normal, avg_d)
            # Compose: T_floor_align @ T_floor_transform
            # But our scan points in T_floor_align are already in viewer space
            # So the final transform takes raw points → viewer → aligned
            T = T_floor_align @ T_floor
            print(f"[Registration] Floor alignment done")
        
        if progress_callback:
            progress_callback(25, "Floor aligned")
    else:
        print("[Registration] No floor elements — skipping floor alignment")
    
    # ── Step 2: Object alignment ──
    if objects:
        print(f"[Registration] Step 2: Aligning with {len(objects)} object(s)")
        
        all_obj_scan_raw = []
        all_obj_bim_v = []
        all_obj_bim_f = []
        
        for om in objects:
            label = str(om["segment_label"])
            key = str(om["element_key"])
            
            inst = label_to_inst.get(label)
            if inst is None:
                continue
            
            indices = inst.get("globalIndices", inst.get("point_indices", []))
            if not indices:
                continue
            
            scan_pts = xyz_raw[indices]
            all_obj_scan_raw.append(scan_pts)
            
            if key in ifc_meshes:
                bim_v, bim_f = ifc_meshes[key]
                all_obj_bim_v.append(bim_v)
                all_obj_bim_f.append(bim_f)
        
        if all_obj_scan_raw and all_obj_bim_v:
            combined_scan = np.vstack(all_obj_scan_raw)
            
            # Merge BIM meshes
            bim_v_combined = np.vstack(all_obj_bim_v)
            face_offset = 0
            bim_f_list = []
            for i, (v, f_arr) in enumerate(zip(all_obj_bim_v, all_obj_bim_f)):
                bim_f_list.append(f_arr + face_offset)
                face_offset += len(v)
            bim_f_combined = np.vstack(bim_f_list)
            
            T = align_objects(
                combined_scan, bim_v_combined, bim_f_combined,
                T, progress_callback=progress_callback,
            )
            print(f"[Registration] Object alignment done")
        
        if progress_callback:
            progress_callback(75, "Objects aligned")
    else:
        print("[Registration] No non-floor objects — using floor alignment only")
    
    # Verify final transform
    # Apply T to a sample of scan points and measure distances
    sample_n = min(1000, len(xyz_raw))
    sample_idx = np.random.choice(len(xyz_raw), sample_n, replace=False)
    sample_raw = xyz_raw[sample_idx]
    sample_h = np.hstack([sample_raw, np.ones((sample_n, 1))])
    sample_transformed = (T @ sample_h.T).T[:, :3]
    print(f"[Registration] Transformed scan sample: "
          f"X=[{sample_transformed[:,0].min():.2f}, {sample_transformed[:,0].max():.2f}] "
          f"Y=[{sample_transformed[:,1].min():.2f}, {sample_transformed[:,1].max():.2f}] "
          f"Z=[{sample_transformed[:,2].min():.2f}, {sample_transformed[:,2].max():.2f}]")
    
    elapsed = time.time() - t0
    print(f"[Registration] ✅ Registration complete in {elapsed:.1f}s")
    
    if progress_callback:
        progress_callback(80, "Registration complete")
    
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) points."""
    n = len(points)
    pts_h = np.hstack([points, np.ones((n, 1))])
    return (T @ pts_h.T).T[:, :3]
