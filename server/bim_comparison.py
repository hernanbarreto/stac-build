"""
STAC Build — BIM vs Scan Comparison
Cloud-to-Mesh (C2M) deviation analysis.

For each matched pair (scan segment ↔ IFC element), computes the
shortest distance from every scan point to the BIM mesh surface.
Returns per-point deviations for heatmap rendering.

Hernán Barreto — Ingerop IN3
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from scipy.spatial import KDTree

# Default tolerance (meters) — configurable via config.yaml bim.deviation
DEFAULT_TOLERANCE = 0.050  # 50 mm


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


def compute_c2m_with_projections(
    scan_points: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute C2M distances AND projected positions on the mesh surface.
    
    For each scan point, finds the closest point on the nearest mesh triangle.
    Returns both the distances and the projected positions (on the BIM surface).
    
    Returns:
        distances: (N,) C2M distances
        projections: (N, 3) positions on the mesh surface
    """
    from scipy.spatial import KDTree
    
    n_points = len(scan_points)
    n_faces = len(mesh_faces)
    
    print(f"[BIM-Compare] Computing C2M + projections: {n_points} points")
    t0 = time.time()
    
    centroids = mesh_verts[mesh_faces].mean(axis=1)
    centroid_tree = KDTree(centroids)
    k = min(8, n_faces)
    
    chunk_size = 10000
    distances = np.empty(n_points, dtype=np.float64)
    projections = np.empty((n_points, 3), dtype=np.float64)
    
    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        chunk = scan_points[start:end]
        n_chunk = end - start
        
        _, tri_indices = centroid_tree.query(chunk, k=k)
        if k == 1:
            tri_indices = tri_indices.reshape(-1, 1)
        
        chunk_dists = np.full(n_chunk, np.inf)
        chunk_proj = np.zeros((n_chunk, 3))
        
        for j in range(k):
            face_idx = tri_indices[:, j]
            v0 = mesh_verts[mesh_faces[face_idx, 0]]
            v1 = mesh_verts[mesh_faces[face_idx, 1]]
            v2 = mesh_verts[mesh_faces[face_idx, 2]]
            
            d, proj = _point_to_tri_batch(chunk, v0, v1, v2, return_closest=True)
            better = d < chunk_dists
            chunk_dists[better] = d[better]
            chunk_proj[better] = proj[better]
        
        distances[start:end] = chunk_dists
        projections[start:end] = chunk_proj
    
    elapsed = time.time() - t0
    print(f"[BIM-Compare] C2M+proj done: {elapsed:.1f}s, "
          f"mean={np.mean(distances)*1000:.1f}mm")
    
    return distances, projections


def _point_to_tri_batch(
    P: np.ndarray,  # (N, 3)
    A: np.ndarray,  # (N, 3) - vertex 0 per point
    B: np.ndarray,  # (N, 3) - vertex 1 per point
    C: np.ndarray,  # (N, 3) - vertex 2 per point
    return_closest: bool = False,
) -> np.ndarray:
    """
    Vectorized closest-point-on-triangle for N points vs N triangles.
    Returns (N,) distances. If return_closest=True, also returns (N,3) projected positions.
    
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
    
    dists = np.linalg.norm(P - closest, axis=1)
    if return_closest:
        return dists, closest
    return dists


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


def compute_sabana_cloud(
    scan_points: np.ndarray,
    distances: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate deviation-colored point cloud for the 'sábana' visualization.
    
    Each scan point gets an RGBA color based on its C2M distance.
    This IS the sábana — no mesh subdivision, no interpolation.
    Point-level precision matching the scanner's native resolution.
    
    Args:
        scan_points: (N, 3) scan point positions (world space, meters)
        distances: (N,) C2M distances per point (meters)
        tolerance: deviation tolerance threshold (meters)
    
    Returns:
        positions: (N, 3) point positions
        colors: (N, 4) RGBA colors per point
    """
    n_pts = len(scan_points)
    
    if n_pts == 0:
        return np.empty((0, 3)), np.empty((0, 4))
    
    # Vectorized color mapping: green → yellow → red
    tolerance_mm = tolerance * 1000
    dist_mm = distances * 1000
    t = np.clip(dist_mm / max(tolerance_mm, 0.1), 0, 2.0)
    
    r = np.where(t <= 1.0, t, 1.0)
    g = np.where(t <= 1.0, 1.0, np.clip(2.0 - t, 0, 1))
    b = np.zeros(n_pts)
    a = np.full(n_pts, 0.85)
    
    colors = np.column_stack([r, g, b, a])
    
    print(f"[BIM-Compare] Sábana cloud: {n_pts} points, "
          f"mean={np.mean(dist_mm):.1f}mm, max={np.max(dist_mm):.1f}mm")
    
    return scan_points, colors


def compute_mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of a triangle mesh in m²."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    crosses = np.cross(v1 - v0, v2 - v0)
    return float(np.sum(np.linalg.norm(crosses, axis=1)) * 0.5)


def compute_coverage_pct(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    scan_points: np.ndarray,
    proximity_m: float,
) -> float:
    """
    Estimate what % of the BIM mesh surface is covered by scan points.
    
    Samples the mesh surface uniformly (not just centroids) to handle
    large triangles correctly. Uses KDTree for efficient proximity check.
    """
    if len(scan_points) == 0 or len(mesh_faces) == 0:
        return 0.0
    
    v0 = mesh_verts[mesh_faces[:, 0]]
    v1 = mesh_verts[mesh_faces[:, 1]]
    v2 = mesh_verts[mesh_faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    total_area = np.sum(areas)
    
    if total_area < 1e-10:
        return 0.0
    
    # Sample mesh surface: ~4 samples per m² (min 1 per face, cap at 50k total)
    samples_per_m2 = 4.0
    all_samples = []
    all_weights = []
    
    for i in range(len(mesh_faces)):
        n = max(1, int(areas[i] * samples_per_m2))
        # Random barycentric coordinates for uniform sampling
        r1 = np.sqrt(np.random.rand(n))
        r2 = np.random.rand(n)
        pts = ((1 - r1)[:, None] * v0[i] +
               (r1 * (1 - r2))[:, None] * v1[i] +
               (r1 * r2)[:, None] * v2[i])
        all_samples.append(pts)
        all_weights.append(np.full(n, areas[i] / n))
    
    sample_pts = np.vstack(all_samples)
    sample_wts = np.concatenate(all_weights)
    
    # Cap at 50k samples for performance
    if len(sample_pts) > 50000:
        idx = np.random.choice(len(sample_pts), 50000, replace=False)
        sample_pts = sample_pts[idx]
        sample_wts = sample_wts[idx]
        # Re-normalize weights
        sample_wts *= total_area / np.sum(sample_wts)
    
    # KDTree on scan points → query surface samples
    tree = KDTree(scan_points)
    dists, _ = tree.query(sample_pts)
    
    covered_area = np.sum(sample_wts[dists <= proximity_m])
    result = round(float(covered_area / total_area * 100), 1)
    
    print(f"[Coverage] {len(mesh_faces)} faces → {len(sample_pts)} samples, "
          f"{len(scan_points)} scan pts, proximity={proximity_m:.3f}m")
    print(f"[Coverage]   dist min={np.min(dists):.4f} median={np.median(dists):.4f} "
          f"max={np.max(dists):.4f} | covered: {np.sum(dists <= proximity_m)}/{len(sample_pts)}")
    print(f"[Coverage]   covered_area={covered_area:.2f}/{total_area:.2f} m² → {result}%")
    
    return result


def save_sabana(
    session_dir: str,
    results: list,
    tolerance: float,
    summary: dict,
):
    """
    Save sábana data to sabana.npz for later loading.
    Combines all evaluated element positions/colors into one cloud.
    """
    import json
    
    all_pos = []
    all_col = []
    metrics = []
    
    for r in results:
        if r.get("status") != "evaluated":
            metrics.append({
                "element_key": r["element_key"],
                "label": r.get("label", ""),
                "ifc_type": r.get("ifc_type", ""),
                "status": r["status"],
            })
            continue
        
        n = r["sabana_n_points"]
        if n > 0:
            pos = np.array(r["sabana_positions"]).reshape(-1, 3)
            col = np.array(r["sabana_colors"]).reshape(-1, 4)
            all_pos.append(pos)
            all_col.append(col)
        
        metrics.append({
            "element_key": r["element_key"],
            "label": r.get("label", ""),
            "ifc_type": r.get("ifc_type", ""),
            "status": "evaluated",
            "coverage_pct": r.get("coverage_pct", 0),
            "correctness_pct": r.get("stats", {}).get("pass_rate", 0),
            "total_points": r.get("total_points", 0),
            "mean_mm": r.get("stats", {}).get("mean_mm", 0),
            "bim_surface_m2": r.get("bim_surface_m2", 0),
        })
    
    if all_pos:
        positions = np.concatenate(all_pos, axis=0).astype(np.float32)
        colors = np.concatenate(all_col, axis=0).astype(np.float32)
    else:
        positions = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 4), dtype=np.float32)
    
    # Save compressed npz (archival)
    out_path = os.path.join(session_dir, "sabana.npz")
    np.savez_compressed(
        out_path,
        positions=positions,
        colors=colors,
    )
    
    # Save as binary PLY for Potree streaming (same format as cleaned_cloud.ply)
    ply_path = os.path.join(session_dir, "sabana_cloud.ply")
    n = len(positions)
    # Convert RGBA float (0-1) → RGB uint8
    rgb = np.clip(colors[:, :3] * 255, 0, 255).astype(np.uint8)
    # Build structured array: x,y,z (float32) + r,g,b (uint8)
    ply_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    ply_data = np.empty(n, dtype=ply_dtype)
    ply_data['x'] = positions[:, 0]
    ply_data['y'] = positions[:, 1]
    ply_data['z'] = positions[:, 2]
    ply_data['r'] = rgb[:, 0]
    ply_data['g'] = rgb[:, 1]
    ply_data['b'] = rgb[:, 2]
    
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(ply_data.tobytes())
    
    print(f"[BIM-Compare] Saved sábana PLY: {n} points → {ply_path}")
    
    # Save metadata JSON alongside
    # Load quality thresholds from config
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        quality_cfg = cfg.get("bim", {}).get("deviation", {}).get("quality", {})
    except Exception:
        quality_cfg = {}
    good_thresh = quality_cfg.get("good_threshold", 80)
    regular_thresh = quality_cfg.get("regular_threshold", 50)
    
    # Compute quality + advance for each element
    advance_values = []
    for m in metrics:
        if m["status"] != "evaluated":
            m["quality"] = "not_built"
            m["advance_pct"] = 0.0
            advance_values.append(0.0)
            continue
        
        cpct = m.get("correctness_pct", 0)
        coverage = m.get("coverage_pct", 0)
        
        if cpct >= good_thresh:
            m["quality"] = "good"
            m["advance_pct"] = round(coverage, 1)
        elif cpct >= regular_thresh:
            m["quality"] = "regular"
            m["advance_pct"] = round(coverage, 1)
        else:
            m["quality"] = "bad"
            m["advance_pct"] = 0.0  # Bad quality = no accepted advance
        
        advance_values.append(m["advance_pct"])
    
    global_advance = round(sum(advance_values) / len(advance_values), 1) if advance_values else 0.0
    
    meta = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerance_mm": round(tolerance * 1000, 1),
        "total_points": len(positions),
        "quality_thresholds": {
            "good_pct": good_thresh,
            "regular_pct": regular_thresh,
        },
        "global_advance_pct": global_advance,
        "summary": summary,
        "elements": metrics,
    }
    meta_path = os.path.join(session_dir, "sabana_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[BIM-Compare] Saved sábana: {len(positions)} points → {out_path}")
    print(f"[BIM-Compare] Global advance: {global_advance}% ({len(advance_values)} elements)")


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
      5. Cumulative coverage tracking with occlusion detection
    
    matches: [{"segment_label": "292127", "element_key": "292127", "ifc_type": "..."}]
    """
    from bim_registration import register, transform_points, _load_cloud_and_segments
    import yaml as _yaml
    
    # Coverage engine imports (graceful fallback if not yet available)
    try:
        from coverage_store import CoverageStore, SampleStatus, ElementState
        from occlusion_raycaster import (
            load_camera_positions, classify_bim_surface, build_segment_labels
        )
        COVERAGE_ENGINE_AVAILABLE = True
    except ImportError as _ce:
        print(f"[BIM-Compare] Coverage engine not available: {_ce}")
        COVERAGE_ENGINE_AVAILABLE = False
    
    # Read coverage proximity from config (once, not per element)
    _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(_cfg_path) as _f:
        _cfg = _yaml.safe_load(_f)
    proximity_m = _cfg.get("bim", {}).get("deviation", {}).get("coverage_proximity_m", 0.15)
    
    # Coverage engine config
    ce_cfg = _cfg.get("coverage_engine", {})
    ce_enabled = ce_cfg.get("enabled", True) and COVERAGE_ENGINE_AVAILABLE
    ray_radius = ce_cfg.get("ray_radius", 0.05)
    completion_threshold = ce_cfg.get("completion_threshold", 80.0)
    quality_threshold = ce_cfg.get("quality_threshold", 80.0)
    
    session_path = Path(session_dir)
    output_dir = session_path / "output"
    
    # Clean up previous sábana artifacts before regenerating
    import shutil
    for old_file in ["sabana.npz", "sabana_cloud.ply", "sabana_meta.json"]:
        p = session_path / old_file
        if p.exists():
            p.unlink()
            print(f"[BIM] 🗑️ Deleted old {old_file}")
    sabana_potree_dir = session_path / "sabana_potree"
    if sabana_potree_dir.exists():
        shutil.rmtree(sabana_potree_dir)
        print("[BIM] 🗑️ Deleted old sabana_potree/")
    
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
    
    # ── Step 2b: Coverage engine setup ──
    cam_positions = None
    seg_labels = None
    cov_store = None
    scan_id = time.strftime("%Y%m%d_%H%M%S")
    
    if ce_enabled:
        if progress_callback:
            progress_callback(82, "Loading camera poses for coverage engine...")
        
        cam_positions = load_camera_positions(session_dir)
        if len(cam_positions) == 0:
            print("[BIM-Compare] No camera poses → coverage engine disabled for this run")
            ce_enabled = False
        else:
            # Transform camera positions with registration matrix
            cam_homo = np.hstack([cam_positions, np.ones((len(cam_positions), 1))])
            cam_positions = (T @ cam_homo.T).T[:, :3].astype(np.float32)
            
            # Build per-point segment labels for occluder identification
            seg_file = str(seg_result_path if seg_result_path.exists() else seg_path)
            seg_labels = build_segment_labels(len(xyz_all), seg_file)
            
            # Initialize coverage store
            cov_store = CoverageStore(session_dir)
            print(f"[BIM-Compare] Coverage engine active: {len(cam_positions)} cameras")
    
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
                })
                continue
            
            indices = inst.get("globalIndices", inst.get("point_indices", []))
            if not indices:
                results.append({
                    "element_key": key, "label": elem_name,
                    "ifc_type": ifc_type, "status": "error",
                    "error": "No point indices",
                })
                continue
            
            # Transform segment points with registration
            scan_points = transform_points(xyz_all[indices], T)
            
            # C2M distances: how far each constructed point is from the BIM design
            distances = compute_c2m_distances(scan_points, mesh_verts, mesh_faces)
            report = build_deviation_report(distances, tolerance)
            
            # Sábana: scan points at their real positions, colored by C2M deviation
            # Shows the CONSTRUCTION, colored by how well it matches the design
            sabana_pos, sabana_colors = compute_sabana_cloud(
                scan_points, distances, tolerance
            )
            
            # Coverage: what % of this BIM element's surface has scan data
            coverage = compute_coverage_pct(
                mesh_verts, mesh_faces, scan_points, proximity_m
            )
            
            # BIM element surface area (m²)
            v0 = mesh_verts[mesh_faces[:, 0]]
            v1 = mesh_verts[mesh_faces[:, 1]]
            v2 = mesh_verts[mesh_faces[:, 2]]
            face_areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
            bim_surface_m2 = float(np.sum(face_areas))
            
            # ── Coverage engine: cumulative + occlusion ──
            coverage_cumulative = coverage
            occluded_pct = 0.0
            element_state = "IN_PROGRESS" if coverage > 0 else "NOT_STARTED"
            
            if ce_enabled and cov_store is not None:
                try:
                    # Initialize or load existing coverage for this element
                    ec = cov_store.init_element(key, mesh_verts, mesh_faces)
                    
                    # Transform scan cloud for ray-casting (already registered)
                    scan_registered = transform_points(xyz_all, T)
                    
                    # Classify BIM surface samples with ray-casting
                    scan_result = classify_bim_surface(
                        cam_positions=cam_positions,
                        scan_cloud=scan_registered,
                        bim_samples=ec.surface_samples,
                        bim_normals=ec.surface_normals,
                        proximity_m=proximity_m,
                        ray_radius=ray_radius,
                        seg_labels=seg_labels,
                    )
                    scan_result.scan_id = scan_id
                    scan_result.date = time.strftime("%Y-%m-%dT%H:%M:%S")
                    
                    # Also store per-sample deviation from C2M
                    # For COVERED samples, use nearest scan point distance
                    for si in range(ec.n_samples):
                        if scan_result.status[si] == SampleStatus.COVERED:
                            # Find nearest scan point to this BIM sample
                            d_idx = np.argmin(np.linalg.norm(
                                scan_points - ec.surface_samples[si], axis=1
                            )) if len(scan_points) > 0 else -1
                            if d_idx >= 0 and d_idx < len(distances):
                                scan_result.deviation[si] = distances[d_idx]
                    
                    # Merge into cumulative store
                    ec = cov_store.update_element(
                        key, scan_result,
                        completion_threshold=completion_threshold,
                        quality_threshold=quality_threshold,
                    )
                    cov_store.append_timeline(scan_id, key, ec)
                    
                    # Use cumulative values
                    coverage_cumulative = ec.coverage_cumulative
                    occluded_pct = ec.occluded_pct
                    element_state = ElementState(ec.element_state).name
                    
                except Exception as _cov_err:
                    print(f"[BIM-Compare] Coverage engine error for {key}: {_cov_err}")
            
            results.append({
                "element_key": key,
                "label": elem_name,
                "ifc_type": ifc_type,
                "status": "evaluated",
                "total_points": len(distances),
                "coverage_pct": coverage,
                "coverage_cumulative": coverage_cumulative,
                "occluded_pct": occluded_pct,
                "element_state": element_state,
                "bim_surface_m2": round(bim_surface_m2, 4),
                **report,
                "sabana_positions": sabana_pos.flatten().round(4).tolist(),
                "sabana_colors": sabana_colors.flatten().round(3).tolist(),
                "sabana_n_points": len(sabana_pos),
            })
        else:
            # ── UNMATCHED: not built / not identified ──
            results.append({
                "element_key": key,
                "label": elem_name,
                "ifc_type": ifc_type,
                "status": "unmatched",
                "n_faces": n_faces,
            })
        
        if progress_callback and i % max(1, total // 20) == 0:
            pct = 85 + round(15 * (i + 1) / total)
            progress_callback(min(pct, 99), f"Element {i+1}/{total}")
    
    # Summary stats
    n_evaluated = sum(1 for r in results if r.get("status") == "evaluated")
    n_unmatched = sum(1 for r in results if r.get("status") == "unmatched")
    n_error = sum(1 for r in results if r.get("status") == "error")
    
    # ── Post-process: VLM occluder classification ──
    if ce_enabled and cov_store is not None:
        try:
            from scene_analyzer import classify_occluders
            from coverage_store import OccluderType
            
            # Collect unique occluder labels across all elements
            all_occluder_labels = set()
            for key in cov_store.get_all_elements():
                ec = cov_store.load_element(key)
                if ec is not None:
                    for lbl in ec.occluder_labels:
                        if lbl and lbl != "":
                            all_occluder_labels.add(str(lbl))
            
            if all_occluder_labels:
                print(f"[BIM-Compare] Classifying {len(all_occluder_labels)} occluder labels...")
                classifications = classify_occluders(
                    list(all_occluder_labels), session_dir
                )
                
                # Update coverage store with occluder types
                for key in cov_store.get_all_elements():
                    ec = cov_store.load_element(key)
                    if ec is None:
                        continue
                    updated = False
                    for i in range(ec.n_samples):
                        lbl = str(ec.occluder_labels[i])
                        if lbl and lbl in classifications:
                            otype = (OccluderType.TEMPORARY 
                                    if classifications[lbl] == "temporary"
                                    else OccluderType.PERMANENT)
                            if ec.occluder_types[i] != otype:
                                ec.occluder_types[i] = otype
                                updated = True
                    if updated:
                        # Recompute state with updated occluder types
                        ec.element_state = CoverageStore._compute_state(
                            ec, completion_threshold, quality_threshold
                        )
                        cov_store.save_element(ec)
                
                print(f"[BIM-Compare] Occluder classification: {classifications}")
        except Exception as _vlm_err:
            print(f"[BIM-Compare] VLM occluder classification skipped: {_vlm_err}")
    
    print(f"[BIM-Compare] Summary: {n_evaluated} evaluated, "
          f"{n_unmatched} unmatched (not built), {n_error} errors, "
          f"{total} total elements")
    
    if progress_callback:
        progress_callback(100, "Comparison complete")
    
    summary = {
        "total_elements": total,
        "evaluated": n_evaluated,
        "unmatched": n_unmatched,
        "errors": n_error,
    }
    
    # Add coverage engine summary if available
    if ce_enabled and cov_store is not None:
        cov_summary = cov_store.get_summary()
        summary["coverage_engine"] = {
            "scan_id": scan_id,
            "elements_tracked": len(cov_summary),
            "elements_by_state": {},
        }
        for key, data in cov_summary.items():
            state = data["element_state"]
            summary["coverage_engine"]["elements_by_state"][state] = \
                summary["coverage_engine"]["elements_by_state"].get(state, 0) + 1
    
    # Save sábana to session for later loading
    save_sabana(session_dir, results, tolerance, summary)
    
    return {
        "ok": True,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerance_mm": round(tolerance * 1000, 1),
        "transform": T.tolist(),
        "summary": summary,
        "results": results,
    }

