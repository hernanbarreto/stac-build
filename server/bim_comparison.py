"""
STAC Build — BIM vs Scan Comparison
Cloud-to-Mesh (C2M) deviation analysis.

For each matched pair (scan segment ↔ IFC element), computes the
shortest distance from every scan point to the BIM mesh surface.
Returns per-point deviations for heatmap rendering.

Hernán Barreto — Ingerop IN3
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

# Default tolerance (meters) — will be configurable per element type later
DEFAULT_TOLERANCE = 0.015  # 15 mm


# ═══════════════════════════════════════════════════════════════════
#  IFC MESH EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _build_name_index(ifc_file) -> dict:
    """
    Build a mapping from the numeric suffix in element names
    to the IFC element objects.
    
    IFC element names follow the pattern:
      "Type:Description:UniqueId"  e.g. "Furniture_Chair_Viper:1120x940x350mm:292127"
    
    The last colon-separated segment is a Revit UniqueId suffix
    that serves as our matching key.
    """
    index = {}
    for el in ifc_file.by_type('IfcProduct'):
        name = getattr(el, 'Name', None)
        if not name:
            continue
        name = str(name)
        # Extract last colon-separated segment
        parts = name.split(':')
        suffix = parts[-1].strip()
        if suffix.isdigit():
            index[suffix] = el
    return index


def extract_ifc_triangles(
    ifc_path: str,
    element_key: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract mesh triangles for an IFC element identified by its
    Revit UniqueId suffix (e.g., "292127").
    
    Returns:
        (vertices, faces) where vertices is (N,3) float64 and
        faces is (M,3) int32, or None if element not found / has no geometry.
    """
    import ifcopenshell
    import ifcopenshell.geom

    ifc_file = ifcopenshell.open(ifc_path)
    name_index = _build_name_index(ifc_file)
    
    element = name_index.get(str(element_key))
    if element is None:
        print(f"[BIM-Compare] Element key '{element_key}' not found in {ifc_path}")
        return None
    
    print(f"[BIM-Compare] Extracting mesh for '{element_key}': "
          f"#{element.id()} {element.is_a()} — {getattr(element, 'Name', '?')}")
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
    except Exception as e:
        print(f"[BIM-Compare] Could not create shape for '{element_key}': {e}")
        return None
    
    geom = shape.geometry
    verts_flat = geom.verts
    faces_flat = geom.faces
    
    if len(verts_flat) == 0 or len(faces_flat) == 0:
        print(f"[BIM-Compare] Element '{element_key}' has no geometry")
        return None
    
    vertices = np.array(verts_flat, dtype=np.float64).reshape(-1, 3)
    faces = np.array(faces_flat, dtype=np.int32).reshape(-1, 3)
    
    print(f"[BIM-Compare] Mesh: {len(vertices)} verts, {len(faces)} triangles")
    return vertices, faces


def extract_all_ifc_triangles(
    ifc_path: str,
    element_keys: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract mesh triangles for multiple IFC elements by Revit UniqueId suffix.
    Opens the IFC file only once.
    """
    import ifcopenshell
    import ifcopenshell.geom

    ifc_file = ifcopenshell.open(ifc_path)
    name_index = _build_name_index(ifc_file)
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    
    results = {}
    for key in element_keys:
        key_str = str(key)
        element = name_index.get(key_str)
        if element is None:
            print(f"[BIM-Compare] Key '{key_str}' not found in IFC")
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            geom = shape.geometry
            
            if len(geom.verts) == 0 or len(geom.faces) == 0:
                continue
            
            vertices = np.array(geom.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.array(geom.faces, dtype=np.int32).reshape(-1, 3)
            
            # Convert IFC Z-up to Three.js Y-up: [x,y,z] → [x, z, -y]
            vertices = vertices[:, [0, 2, 1]]
            vertices[:, 2] = -vertices[:, 2]
            
            results[key_str] = (vertices, faces)
            
            print(f"[BIM-Compare] '{key_str}' (#{element.id()} {element.is_a()} — "
                  f"{getattr(element, 'Name', '?')}): "
                  f"{len(vertices)} verts, {len(faces)} tris")
        except Exception as e:
            print(f"[BIM-Compare] '{key_str}': {e}")
    
    return results


# IFC types to SKIP in full-BIM comparison (non-physical elements)
_SKIP_IFC_TYPES = frozenset({
    'IfcSpace', 'IfcOpeningElement', 'IfcSite', 'IfcBuilding',
    'IfcBuildingStorey', 'IfcProject', 'IfcAnnotation',
    'IfcGrid', 'IfcGridAxis',
})


def extract_all_ifc_geometry(
    ifc_path: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Extract meshes for ALL IFC elements that have geometry.
    Skips non-physical types (IfcSpace, IfcOpeningElement, etc.).
    
    Returns dict: element_key → (vertices, faces, ifc_type, element_name)
    where element_key is the Revit UniqueId suffix from the element Name.
    """
    import ifcopenshell
    import ifcopenshell.geom
    
    ifc_file = ifcopenshell.open(ifc_path)
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    
    results = {}
    processed = 0
    skipped = 0
    
    for element in ifc_file:
        ifc_type = element.is_a()
        
        # Skip non-physical types
        if ifc_type in _SKIP_IFC_TYPES:
            continue
        
        # Only process IfcProduct subtypes (things with geometry)
        if not element.is_a('IfcProduct'):
            continue
        
        # Skip elements without representation
        if not hasattr(element, 'Representation') or element.Representation is None:
            continue
        
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            geom = shape.geometry
            
            if len(geom.verts) == 0 or len(geom.faces) == 0:
                continue
            
            vertices = np.array(geom.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.array(geom.faces, dtype=np.int32).reshape(-1, 3)
            
            # Convert IFC Z-up to Three.js Y-up: [x,y,z] → [x, z, -y]
            vertices = vertices[:, [0, 2, 1]]
            vertices[:, 2] = -vertices[:, 2]
            
            # Build element key from Name (Revit UniqueId suffix)
            name = getattr(element, 'Name', '') or ''
            # Extract last colon-separated part as key
            parts = name.split(':')
            element_key = parts[-1] if parts else str(element.id())
            if not element_key:
                element_key = str(element.id())
            
            results[element_key] = (vertices, faces, ifc_type, name)
            processed += 1
            
        except Exception:
            skipped += 1
    
    print(f"[BIM-Compare] Extracted {processed} elements with geometry "
          f"({skipped} skipped)")
    return results


# ═══════════════════════════════════════════════════════════════════
#  CLOUD-TO-MESH DISTANCE (C2M)
# ═══════════════════════════════════════════════════════════════════

def _point_to_triangle_distance_batch(
    points: np.ndarray,
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
) -> np.ndarray:
    """
    Compute shortest distance from each point to a triangle (v0,v1,v2).
    Uses vectorized barycentric projection.
    
    points: (N, 3)
    v0, v1, v2: (3,) each — triangle vertices
    Returns: (N,) distances
    """
    edge0 = v1 - v0
    edge1 = v2 - v0
    diff = points - v0  # (N, 3)
    
    a = np.dot(edge0, edge0)
    b = np.dot(edge0, edge1)
    c = np.dot(edge1, edge1)
    d = diff @ edge0  # (N,)
    e = diff @ edge1  # (N,)
    
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e
    
    # Clamp to triangle (6 regions)
    # This is the standard GJK-style closest point on triangle
    # For production, we use the simpler KDTree approach below
    
    # Project onto triangle plane, clamp barycentric coords
    inv_det = 1.0 / max(det, 1e-30)
    s_clamped = np.clip(s * inv_det, 0, 1)
    t_clamped = np.clip(t * inv_det, 0, 1)
    
    # Ensure s + t <= 1
    over = s_clamped + t_clamped > 1.0
    if np.any(over):
        scale = 1.0 / (s_clamped[over] + t_clamped[over])
        s_clamped[over] *= scale
        t_clamped[over] *= scale
    
    closest = v0 + np.outer(s_clamped, edge0) + np.outer(t_clamped, edge1)
    return np.linalg.norm(points - closest, axis=1)


def compute_c2m_distances(
    scan_points: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    progress_callback=None,
) -> np.ndarray:
    """
    Compute Cloud-to-Mesh distances using a hybrid approach:
    1. KDTree on mesh vertices for fast approximate nearest
    2. Refine with exact point-to-triangle projection
    
    scan_points: (N, 3) scan point coordinates
    mesh_verts: (V, 3) mesh vertex coordinates  
    mesh_faces: (F, 3) triangle face indices
    
    Returns: (N,) signed distances (positive = point is outside mesh)
    """
    from scipy.spatial import KDTree
    
    n_points = len(scan_points)
    n_faces = len(mesh_faces)
    
    print(f"[BIM-Compare] Computing C2M: {n_points} points vs {n_faces} triangles")
    t0 = time.time()
    
    # Step 1: Build KDTree on triangle centroids for fast lookup
    centroids = mesh_verts[mesh_faces].mean(axis=1)  # (F, 3)
    centroid_tree = KDTree(centroids)
    
    # Step 2: For each point, find k nearest triangle centroids
    # then compute exact distance to those triangles
    k = min(8, n_faces)  # check 8 nearest triangles per point
    
    # Process in chunks for memory efficiency + progress
    chunk_size = 10000
    distances = np.empty(n_points, dtype=np.float64)
    
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = scan_points[start:end]
        
        # Find k nearest triangle centroids
        _, tri_indices = centroid_tree.query(chunk, k=k)
        if k == 1:
            tri_indices = tri_indices.reshape(-1, 1)
        
        # For each point, compute exact distance to candidate triangles
        chunk_dists = np.full(end - start, np.inf)
        
        for j in range(k):
            face_idx = tri_indices[:, j]
            v0 = mesh_verts[mesh_faces[face_idx, 0]]
            v1 = mesh_verts[mesh_faces[face_idx, 1]]
            v2 = mesh_verts[mesh_faces[face_idx, 2]]
            
            # Vectorized point-to-triangle distance
            d = _point_to_tri_batch(chunk, v0, v1, v2)
            chunk_dists = np.minimum(chunk_dists, d)
        
        distances[start:end] = chunk_dists
        
        if progress_callback and n_points > chunk_size:
            pct = round(end / n_points * 100)
            progress_callback(pct)
    
    elapsed = time.time() - t0
    print(f"[BIM-Compare] C2M done: {elapsed:.1f}s, "
          f"mean={np.mean(distances)*1000:.1f}mm, "
          f"max={np.max(distances)*1000:.1f}mm")
    
    return distances


def _point_to_tri_batch(
    P: np.ndarray,  # (N, 3)
    A: np.ndarray,  # (N, 3) - vertex 0 per point
    B: np.ndarray,  # (N, 3) - vertex 1 per point
    C: np.ndarray,  # (N, 3) - vertex 2 per point
) -> np.ndarray:
    """
    Vectorized closest-point-on-triangle for N points vs N triangles.
    Returns (N,) distances.
    
    Implementation of the Ericson real-time collision detection algorithm,
    fully vectorized with numpy.
    """
    AB = B - A
    AC = C - A
    AP = P - A
    
    d1 = np.sum(AB * AP, axis=1)
    d2 = np.sum(AC * AP, axis=1)
    
    # Region 1: closest to vertex A
    mask_a = (d1 <= 0) & (d2 <= 0)
    
    BP = P - B
    d3 = np.sum(AB * BP, axis=1)
    d4 = np.sum(AC * BP, axis=1)
    
    # Region 2: closest to vertex B
    mask_b = (d3 >= 0) & (d4 <= d3)
    
    CP = P - C
    d5 = np.sum(AB * CP, axis=1)
    d6 = np.sum(AC * CP, axis=1)
    
    # Region 3: closest to vertex C
    mask_c = (d6 >= 0) & (d5 <= d6)
    
    vc = d1 * d4 - d3 * d2
    # Region 4: closest to edge AB
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v_ab = d1 / np.maximum(d1 - d3, 1e-30)
    
    vb = d5 * d2 - d1 * d6
    # Region 5: closest to edge AC
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w_ac = d2 / np.maximum(d2 - d6, 1e-30)
    
    va = d3 * d6 - d5 * d4
    # Region 6: closest to edge BC
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), 1e-30)
    
    # Region 0: inside triangle
    denom = 1.0 / np.maximum(va + vb + vc, 1e-30)
    v_in = vb * denom
    w_in = vc * denom
    
    # Compute closest point for each region
    N = len(P)
    closest = np.empty((N, 3), dtype=np.float64)
    
    closest[mask_a] = A[mask_a]
    closest[mask_b] = B[mask_b]
    closest[mask_c] = C[mask_c]
    
    if np.any(mask_ab):
        closest[mask_ab] = A[mask_ab] + v_ab[mask_ab, None] * AB[mask_ab]
    if np.any(mask_ac):
        closest[mask_ac] = A[mask_ac] + w_ac[mask_ac, None] * AC[mask_ac]
    if np.any(mask_bc):
        closest[mask_bc] = B[mask_bc] + w_bc[mask_bc, None] * (C[mask_bc] - B[mask_bc])
    
    # Inside triangle
    mask_inside = ~(mask_a | mask_b | mask_c | mask_ab | mask_ac | mask_bc)
    if np.any(mask_inside):
        closest[mask_inside] = (A[mask_inside] 
                                + v_in[mask_inside, None] * AB[mask_inside]
                                + w_in[mask_inside, None] * AC[mask_inside])
    
    return np.linalg.norm(P - closest, axis=1)


# ═══════════════════════════════════════════════════════════════════
#  DEVIATION STATISTICS & REPORT
# ═══════════════════════════════════════════════════════════════════

def build_deviation_report(
    distances: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    n_bins: int = 50,
) -> dict:
    """
    Build a deviation report with statistics and histogram.
    """
    within = np.sum(distances <= tolerance)
    total = len(distances)
    
    stats = {
        "min_mm": round(float(np.min(distances)) * 1000, 2),
        "max_mm": round(float(np.max(distances)) * 1000, 2),
        "mean_mm": round(float(np.mean(distances)) * 1000, 2),
        "std_mm": round(float(np.std(distances)) * 1000, 2),
        "median_mm": round(float(np.median(distances)) * 1000, 2),
        "p95_mm": round(float(np.percentile(distances, 95)) * 1000, 2),
        "within_tolerance": int(within),
        "total_points": total,
        "pass_rate": round(within / max(total, 1) * 100, 1),
        "tolerance_mm": round(tolerance * 1000, 1),
    }
    
    # Histogram
    max_dist = min(float(np.max(distances)), tolerance * 5)  # cap at 5x tolerance
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    counts, edges = np.histogram(distances, bins=bin_edges)
    
    histogram = {
        "counts": counts.tolist(),
        "bin_edges_mm": (edges * 1000).tolist(),
    }
    
    return {"stats": stats, "histogram": histogram}


def deviation_to_rgb(distance_mm: float, tolerance_mm: float) -> Tuple[float, float, float]:
    """
    Map a deviation distance to an RGB color.
    Green (0) → Yellow (tolerance) → Red (2x tolerance).
    Gray for no-data.
    """
    if distance_mm < 0:  # no data
        return (0.3, 0.3, 0.3)
    
    t = distance_mm / max(tolerance_mm, 0.1)
    t = min(t, 2.0)  # clamp at 2x tolerance
    
    if t <= 1.0:
        # Green → Yellow
        r = t
        g = 1.0
    else:
        # Yellow → Red
        r = 1.0
        g = max(0.0, 2.0 - t)
    
    return (r, g, 0.0)


def compute_per_face_deviation(
    scan_points: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-face deviation for the 'sábana' visualization.
    
    Inverted approach: for each scan point, find its nearest face centroid,
    then use C2M distances (already computed) to assign deviation per face.
    Uses vectorized numpy — handles 1M+ points efficiently.
    
    Returns:
        face_distances: (F,) average distance per face (-1 = no data)
        face_colors: (F, 3) RGB colors per face
    """
    from scipy.spatial import KDTree
    
    n_faces = len(mesh_faces)
    n_pts = len(scan_points)
    
    if n_pts == 0 or n_faces == 0:
        return np.full(n_faces, -1.0), np.full((n_faces, 3), 0.3)
    
    # Compute face centroids → KDTree
    face_centroids = mesh_verts[mesh_faces].mean(axis=1)  # (F, 3)
    face_tree = KDTree(face_centroids)
    
    # Assign each scan point to its nearest face centroid
    _, face_indices = face_tree.query(scan_points, k=1)  # (N,)
    face_indices = face_indices.ravel().astype(np.int32)
    
    # Compute per-point C2M distance to its assigned face
    # Batch: get triangle vertices for each point's assigned face
    v0 = mesh_verts[mesh_faces[face_indices, 0]]  # (N, 3)
    v1 = mesh_verts[mesh_faces[face_indices, 1]]  # (N, 3)
    v2 = mesh_verts[mesh_faces[face_indices, 2]]  # (N, 3)
    
    pt_dists = _point_to_tri_batch(scan_points, v0, v1, v2)  # (N,)
    
    # Aggregate per face using bincount
    face_sum = np.bincount(face_indices, weights=pt_dists, minlength=n_faces)
    face_count = np.bincount(face_indices, minlength=n_faces)
    
    # Compute averages and colors
    face_distances = np.full(n_faces, -1.0)
    face_colors = np.full((n_faces, 3), 0.3)  # gray = no data
    tolerance_mm = tolerance * 1000
    
    covered_mask = face_count > 0
    face_distances[covered_mask] = face_sum[covered_mask] / face_count[covered_mask]
    
    for i in np.where(covered_mask)[0]:
        face_colors[i] = deviation_to_rgb(face_distances[i] * 1000, tolerance_mm)
    
    covered = int(covered_mask.sum())
    print(f"[BIM-Compare] Per-face deviation: "
          f"{covered}/{n_faces} faces covered "
          f"({covered/max(n_faces,1)*100:.0f}%)")
    
    return face_distances, face_colors, face_centroids


# ═══════════════════════════════════════════════════════════════════
#  LOAD SCAN POINTS
# ═══════════════════════════════════════════════════════════════════

def load_scan_points(
    cloud_path: str,
    indices: List[int],
) -> np.ndarray:
    """Load specific points from cleaned_cloud.ply by global indices."""
    from segmentation_pipeline import _load_ply_origins
    
    result = _load_ply_origins(Path(cloud_path))
    if result is None:
        raise ValueError(f"Could not load {cloud_path}")
    
    xyz = result[0]  # (N, 3)
    return xyz[indices]


# ═══════════════════════════════════════════════════════════════════
#  FULL COMPARISON PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_comparison(
    session_dir: str,
    matches: List[dict],
    tolerance: float = DEFAULT_TOLERANCE,
    progress_callback=None,
) -> dict:
    """
    Run full BIM vs Scan comparison:
      1. Registration using matched segments (alignment only)
      2. Extract ALL BIM element geometry
      3. For matched elements: C2M deviation → green/yellow/red sábana
      4. For unmatched elements: gray transparent (not yet built)
    
    matches: [{"segment_label": "292127", "element_key": "292127", "ifc_type": "..."}]
    """
    from bim_registration import register, transform_points, _load_cloud_and_segments
    
    session_path = Path(session_dir)
    output_dir = session_path / "output"
    
    # Find IFC file
    ifc_files = list(session_path.glob("*.ifc"))
    if not ifc_files:
        return {"error": "No IFC file found in session"}
    ifc_path = str(ifc_files[0])
    
    # Load segmentation
    seg_result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"
    if seg_result_path.exists():
        seg_data = json.loads(seg_result_path.read_text())
    elif seg_path.exists():
        seg_data = json.loads(seg_path.read_text())
    else:
        return {"error": "No segmentation data found"}
    
    label_to_inst = {}
    for inst in seg_data.get("instances", []):
        label_to_inst[str(inst.get("label", ""))] = inst
    
    # Build match index: element_key → segment_label
    match_index = {}
    for m in matches:
        match_index[str(m["element_key"])] = str(m["segment_label"])
    
    # ── Step 1: Registration (alignment using matched segments) ──
    if progress_callback:
        progress_callback(5, "Registering scan to BIM...")
    
    T = register(session_dir, matches, progress_callback=progress_callback)
    if T is None:
        T = np.eye(4)
        print("[BIM-Compare] Registration failed, using identity transform")
    
    # ── Step 2: Extract ALL BIM geometry ──
    if progress_callback:
        progress_callback(80, "Extracting all BIM geometry...")
    
    all_bim = extract_all_ifc_geometry(ifc_path)
    
    # Load scan cloud (raw points)
    xyz_all, _ = _load_cloud_and_segments(session_dir)
    if xyz_all is None:
        return {"error": "Could not load cleaned_cloud.ply"}
    
    if progress_callback:
        progress_callback(85, "Computing deviations...")
    
    # ── Step 3: Evaluate each BIM element ──
    results = []
    total = len(all_bim)
    tolerance_mm = tolerance * 1000
    
    for i, (key, (mesh_verts, mesh_faces, ifc_type, elem_name)) in enumerate(all_bim.items()):
        n_faces = len(mesh_faces)
        
        if key in match_index:
            # ── MATCHED: compare segment points vs BIM mesh ──
            label = match_index[key]
            inst = label_to_inst.get(label)
            
            if inst is None:
                # Matched but segment not found
                results.append({
                    "element_key": key, "label": elem_name,
                    "ifc_type": ifc_type, "status": "error",
                    "error": "Segment not found",
                    "face_colors": [0.3, 0.3, 0.3] * n_faces,
                    "n_faces": n_faces,
                })
                continue
            
            indices = inst.get("globalIndices", inst.get("point_indices", []))
            if not indices:
                results.append({
                    "element_key": key, "label": elem_name,
                    "ifc_type": ifc_type, "status": "error",
                    "error": "No point indices",
                    "face_colors": [0.3, 0.3, 0.3] * n_faces,
                    "n_faces": n_faces,
                })
                continue
            
            # Transform segment points with registration
            scan_points = transform_points(xyz_all[indices], T)
            
            # C2M distances for statistics
            distances = compute_c2m_distances(scan_points, mesh_verts, mesh_faces)
            report = build_deviation_report(distances, tolerance)
            
            # Per-face deviation for sábana coloring
            face_dists, face_colors, face_centroids = compute_per_face_deviation(
                scan_points, mesh_verts, mesh_faces, tolerance
            )
            
            results.append({
                "element_key": key,
                "label": elem_name,
                "ifc_type": ifc_type,
                "status": "evaluated",
                "total_points": len(distances),
                **report,
                "face_colors": face_colors.flatten().round(3).tolist(),
                "face_centroids": face_centroids.flatten().round(4).tolist(),
                "n_faces": n_faces,
            })
        else:
            # ── UNMATCHED: not built / not identified ──
            # Gray transparent — no segment for this BIM element
            # Send centroids so frontend can match by proximity
            centroids = mesh_verts[mesh_faces].mean(axis=1)
            results.append({
                "element_key": key,
                "label": elem_name,
                "ifc_type": ifc_type,
                "status": "unmatched",
                "face_colors": [0.4, 0.4, 0.4] * n_faces,
                "face_centroids": centroids.flatten().round(4).tolist(),
                "n_faces": n_faces,
            })
        
        if progress_callback and i % max(1, total // 20) == 0:
            pct = 85 + round(15 * (i + 1) / total)
            progress_callback(min(pct, 99), f"Element {i+1}/{total}")
    
    # Summary stats
    n_evaluated = sum(1 for r in results if r.get("status") == "evaluated")
    n_unmatched = sum(1 for r in results if r.get("status") == "unmatched")
    n_error = sum(1 for r in results if r.get("status") == "error")
    
    print(f"[BIM-Compare] Summary: {n_evaluated} evaluated, "
          f"{n_unmatched} unmatched (not built), {n_error} errors, "
          f"{total} total elements")
    
    if progress_callback:
        progress_callback(100, "Comparison complete")
    
    return {
        "ok": True,
        "tolerance_mm": round(tolerance * 1000, 1),
        "transform": T.tolist(),
        "summary": {
            "total_elements": total,
            "evaluated": n_evaluated,
            "unmatched": n_unmatched,
            "errors": n_error,
        },
        "results": results,
    }

