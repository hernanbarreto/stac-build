"""
STAC Build — BIM vs Scan Comparison
Cloud-to-Mesh (C2M) deviation analysis.

For each matched pair (scan object ↔ IFC element), computes the
shortest distance from every scan point to the BIM mesh surface.
Returns per-point deviations for heatmap rendering.

Hernán Barreto — Ingerop IN3

ADAPTED TO THE SCANS OF TODAY (2026-11-18, his "create un IFC vos a partir de PCCR3108, para
comparar contra ese mismo"): written in March for clouds of a million points whose segments were
labelled by hand with their BIM element's number, it now reads a scan through bim/scan_data.py —
the objects from the instance store, their points from the per-point ids, matched by INSTANCE ID
to an IFC that STAC wrote from a scan and by label to any other — and holds at today's sizes
(pccr 2026-08-31: 18.8 M points, 281 elements): the cloud is read once, the deviation sheet is
kept as arrays (never as JSON lists in the HTTP answer), the IFC's products are asked for by type.
THE DISTANCE IS EXACT (compute_c2m_distances) and the coverage's surface samples are drawn from a
generator seeded by the element (the same comparison gives the same numbers).
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


def _faceset_mesh(element, unit_scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(vertices in metres, world frame; faces) of an element whose body is made of
    IfcTriangulatedFaceSet only — READ FROM THE FILE: its own points and triangles, placed by the
    element's placement. None for anything else (the geometry kernel then builds it).

    Why not the kernel for these too (measured 2026-11-18 on pccr's IFC, 281 elements, 1.4 M
    triangles): it turns each triangle into a face of a shell — 0.17 ms a triangle, minutes for the
    file — and welds the result (a pipe of 3,409 triangles came back with 3,393). The file's own
    triangles are the element."""
    import ifcopenshell.util.placement
    representation = getattr(element, "Representation", None)
    if representation is None or getattr(element, "HasOpenings", None):
        return None                                               # (an opening is cut by the kernel)
    items = [item for rep in representation.Representations for item in rep.Items]
    if not items or any(not item.is_a("IfcTriangulatedFaceSet") or item.PnIndex for item in items):
        return None
    vertices, faces, base = [], [], 0
    for item in items:
        coords = np.asarray(item.Coordinates.CoordList, dtype=np.float64).reshape(-1, 3)
        index = np.asarray(item.CoordIndex, dtype=np.int64).reshape(-1, 3) - 1
        if len(coords) == 0 or len(index) == 0:
            continue
        vertices.append(coords)
        faces.append(index + base)
        base += len(coords)
    if not faces:
        return None
    M = np.asarray(ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement), dtype=np.float64)
    world = (np.concatenate(vertices) @ M[:3, :3].T + M[:3, 3]) * unit_scale
    return world, np.concatenate(faces).astype(np.int32)


def _element_mesh(element, settings, unit_scale: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(vertices, faces) of one IFC element in the viewer's frame (Y up) — None when it has no
    geometry. A triangulated element is read from the file, any other built by the kernel."""
    import ifcopenshell.geom
    mesh = _faceset_mesh(element, unit_scale)
    if mesh is None:
        shape = ifcopenshell.geom.create_shape(settings, element)   # (kept: the geometry's arrays live in it)
        geom = shape.geometry
        if len(geom.verts) == 0 or len(geom.faces) == 0:
            return None
        mesh = (np.array(geom.verts, dtype=np.float64).reshape(-1, 3), np.array(geom.faces, dtype=np.int32).reshape(-1, 3))
    vertices, faces = mesh
    # Convert IFC Z-up to Three.js Y-up: [x,y,z] → [x, z, -y]
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 2] = -vertices[:, 2]
    return np.ascontiguousarray(vertices), faces


def _unit_scale(ifc_file) -> float:
    import ifcopenshell.util.unit
    return float(ifcopenshell.util.unit.calculate_unit_scale(ifc_file))


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
    
    unit_scale = _unit_scale(ifc_file)
    results = {}
    for key in element_keys:
        key_str = str(key)
        element = name_index.get(key_str)
        if element is None:
            print(f"[BIM-Compare] Key '{key_str}' not found in IFC")
            continue
        try:
            mesh = _element_mesh(element, settings, unit_scale)
            if mesh is None:
                continue
            results[key_str] = mesh
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
    ifc_file=None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, str, str]]:
    """
    Extract meshes for ALL IFC elements that have geometry.
    Skips non-physical types (IfcSpace, IfcOpeningElement, etc.).
    
    Returns dict: element_key → (vertices, faces, ifc_type, element_name)
    where element_key is the Revit UniqueId suffix from the element Name.

    ``ifc_file``: the file already opened (it is not opened twice). The products are asked for BY
    TYPE: walking every entity of an IFC whose elements are meshes (millions of points and faces)
    to find its few hundred products took longer than their geometry.
    """
    import ifcopenshell
    import ifcopenshell.geom
    
    if ifc_file is None:
        ifc_file = ifcopenshell.open(ifc_path)
    
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    
    results = {}
    processed = 0
    skipped = 0
    
    unit_scale = _unit_scale(ifc_file)
    for element in ifc_file.by_type('IfcProduct'):
        ifc_type = element.is_a()
        
        # Skip non-physical types
        if ifc_type in _SKIP_IFC_TYPES:
            continue
        
        # Skip elements without representation
        if not hasattr(element, 'Representation') or element.Representation is None:
            continue
        
        try:
            mesh = _element_mesh(element, settings, unit_scale)
            if mesh is None:
                continue
            vertices, faces = mesh
            
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


# WHERE THE DISTANCES ARE COMPUTED (bim.deviation.device): numpy — the path of record — | cpu | cuda
# (bim/c2m_grid.py: the search and the distances on the device, the same distances —
# tests/test_bim_scan_comparison.py).
_DEVICE_OVERRIDE: Optional[str] = None


def use_device(dev: Optional[str]) -> Optional[str]:
    """The device of every C2M distance after this call ('cpu' in the tests — the torch path on the
    CPU, the same code as the card); None = the configuration's (``bim.deviation.device``)."""
    global _DEVICE_OVERRIDE
    _DEVICE_OVERRIDE = None if dev is None else str(dev)
    return _DEVICE_OVERRIDE


_CONFIGURED_DEVICE: Optional[str] = None


def device_of() -> str:
    """:func:`use_device`'s, else config.yaml ``bim.deviation.device`` — a missing key FAILS naming it.
    The file is read ONCE a process: the distance of each of a scan's hundreds of elements asks
    this, and parsing the whole configuration each time doubled pccr's comparison (measured)."""
    global _CONFIGURED_DEVICE
    from bim.c2m_torch import resolve_device
    if _DEVICE_OVERRIDE is not None:
        return resolve_device(_DEVICE_OVERRIDE)
    if _CONFIGURED_DEVICE is None:
        import yaml
        with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        try:
            _CONFIGURED_DEVICE = resolve_device(cfg["bim"]["deviation"]["device"])
        except (KeyError, TypeError):
            raise KeyError("config.yaml: bim.deviation.device is missing (numpy | cpu | cuda)") from None
    return _CONFIGURED_DEVICE


C2M_CHUNK = 200_000          # points per pass, at most
C2M_MIN_CHUNK = 20_000       # … and at least (an object of a few thousand points is one pass)
C2M_FIRST_K = 8              # triangles looked up first, nearest centroid first
C2M_BLOCK = 8                # … and measured in blocks of columns: 8, 8, 16, 32, …
C2M_THREADS = 8              # passes at once (the pod's cgroup grants ~26 CPUs, not the host's 252)
C2M_PAIRS = 20_000_000       # point × candidate pairs held at once by a wider search
C2M_CLASS_SHARE = 0.02       # a size class under this share of the triangles joins the next larger one


def _column_blocks(k: int):
    """The k candidates of a look-up (nearest centroid first), in blocks: [0, 8), [8, 16), [16, 32),
    [32, 64) … — the first triangles settle most points, the wide blocks are for the few left."""
    j0 = 0
    while j0 < k:
        j1 = min(k, max(C2M_BLOCK, 2 * j0))
        yield j0, j1
        j0 = j1


def _nearest_in_class(P, best, A, B, C, tree, reach, radius):
    """Lower ``best`` (the distance of each point of P to the nearest triangle found so far) with
    the triangles (A, B, C) of ONE size class — exactly. A triangle lies within ``reach`` of its
    centroid, so one whose centroid is at c from the point holds no point nearer than c − reach:
    it is worth looking further only while c < best + reach (``reach``: the class's largest such
    distance), and a triangle is worth its exact distance only if c − its own ``radius`` < best.
    Centroids are taken nearest first, k at a time and in blocks of columns (_column_blocks): every
    triangle of a block that can still be nearer — by the distances the block began with — is
    measured at once, the points' distances lowered, and the points whose last centroid of the
    block is beyond the bound are done. (Measuring a triangle the lowered distance would have
    ruled out changes nothing: it cannot be nearer.) A point with all k within the bound is asked
    for 4 k."""
    F = len(A)
    todo = np.arange(len(P))
    k = min(C2M_FIRST_K, F)
    while len(todo):
        step = max(1, C2M_PAIRS // k)
        left = []
        for s0 in range(0, len(todo), step):
            part = todo[s0:s0 + step]
            cd, ci = tree.query(P[part], k=k)
            if k == 1:
                cd, ci = cd[:, None], ci[:, None]
            d = best[part]
            live = np.arange(len(part))                           # rows still worth another triangle
            for j0, j1 in _column_blocks(k):
                cdb, cib = cd[live, j0:j1], ci[live, j0:j1]
                bound = d[live][:, None]
                r, c = np.nonzero((cdb < bound + reach) & (cdb - radius[cib] < bound))
                if len(r):
                    f = cib[r, c]
                    rows = live[r]                                 # (ascending: one run per row)
                    dist = _point_to_tri_batch(P[part[rows]], A[f], B[f], C[f])
                    first = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
                    d[rows[first]] = np.minimum(d[rows[first]], np.minimum.reduceat(dist, first))
                live = live[cd[live, j1 - 1] < d[live] + reach]
                if not len(live):
                    break
            best[part] = d
            if k < F and len(live):
                left.append(part[live])
        if k >= F or not left:
            break
        todo = np.concatenate(left)
        k = min(k * 4, F)
    return best


def compute_c2m_distances(
    scan_points: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    progress_callback=None,
    quiet: bool = False,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    EXACT Cloud-to-Mesh distance: for every point, the distance to the nearest point of the mesh
    surface (unsigned), in float64. ``device``: numpy | cpu | cuda (None: :func:`device_of`) — the
    same distances on each.

    scan_points: (N, 3) scan point coordinates
    mesh_verts: (V, 3) mesh vertex coordinates  
    mesh_faces: (F, 3) triangle face indices

    Until 2026-11-18 this took the 8 triangles with the nearest CENTROIDS and the smallest of their
    distances: right for a mesh of small even triangles, wrong for a BIM solid (a wall is 12
    triangles metres long — the face a point lies on can be the one whose centroid is farthest).
    Now the search is proven (_nearest_in_class), and the triangles are taken in SIZE CLASSES (a
    factor of two in the centroid-to-vertex distance), each with its own centroid tree: a few huge
    triangles in a fine mesh do not widen the search of the fine ones.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.spatial import cKDTree

    P = np.ascontiguousarray(scan_points, dtype=np.float64)
    V = np.asarray(mesh_verts, dtype=np.float64)
    F = np.asarray(mesh_faces)
    n_points, n_faces = len(P), len(F)
    if n_points == 0:
        return np.empty(0, dtype=np.float64)
    if n_faces == 0:
        raise ValueError("the mesh has no triangle: no distance to it")
    t0 = time.time()

    tri = V[F]                                                    # (F, 3, 3)
    centroids = tri.mean(axis=1)
    radius = np.linalg.norm(tri - centroids[:, None, :], axis=2).max(axis=1)
    size_class = np.floor(np.log2(np.maximum(radius, 1e-12))).astype(np.int64)
    classes, counts = np.unique(size_class, return_counts=True)
    # a class of a handful of triangles joins the next larger one (each class costs every point a
    # look-up; its few small triangles are then searched with the larger reach, which stays exact)
    for i in range(len(classes) - 1):
        if counts[i] < C2M_CLASS_SHARE * n_faces:
            size_class[size_class == classes[i]] = classes[i + 1]
            counts[i + 1] += counts[i]
            counts[i] = 0
    classes, counts = classes[counts > 0], counts[counts > 0]
    order = np.argsort(-counts, kind="stable")                    # the class with most triangles first
    device = device_of() if device is None else device
    if device != "numpy":
        # the search AND the distances on the device (bim/c2m_grid.py — USER 2026-11-19: "la comparacion debe
        # hacerse en GPU"): pccr 2026-08-31's 280 elements in 28 s, 98 s with the candidates from scipy's tree
        from bim import c2m_grid
        from bim.c2m_torch import resolve_device
        best = c2m_grid.distances(P, tri, resolve_device(device),
                                  (lambda pct: progress_callback(pct)) if progress_callback else None)
        if not quiet:
            print(f"[BIM-Compare] C2M ({device}): {n_points} points vs {n_faces} triangles in {len(classes)} size "
                  f"class(es), {time.time() - t0:.1f}s — mean={np.mean(best)*1000:.1f}mm, max={np.max(best)*1000:.1f}mm")
        return best
    best = np.full(n_points, np.inf)
    done = 0
    chunk = min(C2M_CHUNK, max(C2M_MIN_CHUNK, -(-n_points // C2M_THREADS)))
    for c in classes[order]:
        sel = np.flatnonzero(size_class == c)
        A, B, C = tri[sel, 0], tri[sel, 1], tri[sel, 2]
        tree = cKDTree(centroids[sel])
        own = radius[sel]
        reach = float(own.max())
        def one_pass(start, A=A, B=B, C=C, tree=tree, reach=reach, own=own):
            end = min(start + chunk, n_points)
            best[start:end] = _nearest_in_class(P[start:end], best[start:end], A, B, C, tree, reach, own)
            return end - start

        # the passes are independent (each writes its own points): in threads — numpy and the tree
        # release the interpreter in their loops
        with ThreadPoolExecutor(C2M_THREADS) as pool:
            for n in pool.map(one_pass, range(0, n_points, chunk)):
                done += n
                if progress_callback and n_points > chunk:
                    progress_callback(round(done / (n_points * len(classes)) * 100))

    if not quiet:
        print(f"[BIM-Compare] C2M: {n_points} points vs {n_faces} triangles in {len(classes)} size "
              f"class(es), {time.time() - t0:.1f}s — mean={np.mean(best)*1000:.1f}mm, max={np.max(best)*1000:.1f}mm")
    return best


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


def _dot3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N,) row-wise dot product of two (N, 3) arrays, its three products added left to right."""
    return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]


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

    THE PATH OF RECORD of the C2M distance; bim/c2m_torch.point_to_tri is its twin on a device. The
    sums of three products and the final norm are written out term by term, left to right (since
    2026-11-18; np.sum and np.linalg.norm before, the same values to the last bit or two): the twin
    then computes the SAME operations in the same order and its distances are these, bit for bit.
    """
    AB = B - A
    AC = C - A
    AP = P - A
    
    d1 = _dot3(AB, AP)
    d2 = _dot3(AC, AP)
    
    # Region 1: closest to vertex A
    mask_a = (d1 <= 0) & (d2 <= 0)
    
    BP = P - B
    d3 = _dot3(AB, BP)
    d4 = _dot3(AC, BP)
    
    # Region 2: closest to vertex B
    mask_b = (d3 >= 0) & (d4 <= d3)
    
    CP = P - C
    d5 = _dot3(AB, CP)
    d6 = _dot3(AC, CP)
    
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
    
    D = P - closest
    dists = np.sqrt(D[:, 0] * D[:, 0] + D[:, 1] * D[:, 1] + D[:, 2] * D[:, 2])
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
    quiet: bool = False,
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
    
    if not quiet:
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


COVERAGE_SAMPLES_PER_M2 = 4.0
COVERAGE_MAX_SAMPLES = 50000


def compute_coverage_pct(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    scan_points: np.ndarray,
    proximity_m: float,
    seed: int = 0,
    quiet: bool = False,
    device: Optional[str] = None,
) -> float:
    """
    Estimate what % of the BIM mesh surface is covered by scan points.

    ``device``: numpy (scipy's tree — the record) | cpu | cuda (bim/c2m_grid.within: the same
    question asked on the device); None: :func:`device_of`.
    
    Samples the mesh surface (~4 samples per m², at least one per triangle, 50,000 at most — each
    weighted by the area it stands for) and counts the area whose sample has a scan point within
    ``proximity_m``.

    ``seed``: the samples are drawn from a generator seeded with it (run_comparison passes the
    element's own key) — until 2026-11-18 they came from numpy's global generator and the same
    comparison gave another coverage at every run. One draw for the whole element (the loop over
    its triangles took seconds on an element of 300,000 of them).
    """
    if len(scan_points) == 0 or len(mesh_faces) == 0:
        return 0.0
    drawn = _coverage_samples(mesh_verts, mesh_faces, seed)
    if drawn is None:
        return 0.0
    sample_pts, sample_wts, total_area = drawn

    # has each surface sample a scan point within proximity_m?
    device = device_of() if device is None else device
    if device == "numpy":
        from scipy.spatial import cKDTree
        dists, _ = cKDTree(scan_points).query(sample_pts, workers=C2M_THREADS)
        near = dists <= proximity_m
    else:
        from bim import c2m_grid
        from bim.c2m_torch import resolve_device
        near = c2m_grid.within(np.ascontiguousarray(scan_points, dtype=np.float64), sample_pts, proximity_m,
                               resolve_device(device))
    result = _coverage_of(near, sample_wts, total_area)
    if not quiet:
        print(f"[Coverage] {len(mesh_faces)} faces → {len(sample_pts)} samples, {len(scan_points)} scan pts, "
              f"proximity={proximity_m:.3f}m → {result}%")
    return result


def _coverage_of(near: np.ndarray, sample_wts: np.ndarray, total_area: float) -> float:
    """The covered share of an element's surface, in % to one decimal: the area of its samples
    that have a scan point near, over its area."""
    return round(float(np.sum(sample_wts[near]) / total_area * 100), 1)


def _coverage_samples(mesh_verts: np.ndarray, mesh_faces: np.ndarray, seed: int):
    """The surface samples of an element's coverage — (points (S, 3), the area each stands for
    (S,), the element's area), or None for a surface of no area. Drawn from a generator seeded
    with ``seed``: the same element gives the same samples."""
    v0 = mesh_verts[mesh_faces[:, 0]]
    v1 = mesh_verts[mesh_faces[:, 1]]
    v2 = mesh_verts[mesh_faces[:, 2]]
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    total_area = np.sum(areas)
    
    if total_area < 1e-10:
        return None
    
    rng = np.random.default_rng(seed)
    per_face = np.maximum(1, (areas * COVERAGE_SAMPLES_PER_M2).astype(np.int64))
    face = np.repeat(np.arange(len(mesh_faces)), per_face)
    sample_wts = (areas / per_face)[face]
    if len(face) > COVERAGE_MAX_SAMPLES:
        keep = np.sort(rng.choice(len(face), COVERAGE_MAX_SAMPLES, replace=False))
        face, sample_wts = face[keep], sample_wts[keep]
        sample_wts = sample_wts * (total_area / np.sum(sample_wts))   # the kept ones stand for the whole area
    # uniform in each triangle (barycentric)
    r1 = np.sqrt(rng.random(len(face)))
    r2 = rng.random(len(face))
    sample_pts = ((1 - r1)[:, None] * v0[face] + (r1 * (1 - r2))[:, None] * v1[face] + (r1 * r2)[:, None] * v2[face])
    return sample_pts, sample_wts, total_area


def _evaluate_on_the_device(all_bim: dict, match_index: dict, scan, xyz_all: np.ndarray, T: np.ndarray,
                            proximity_m: float, device: str):
    """EVERY matched element's distances and coverage in ONE pass on the device (bim/c2m_grid.py —
    USER 2026-11-19: "hace que todo se procese en una sola pasada"): the points of all the elements
    together, each with its element's number, against the triangles of all of them.
    Returns ({element key: (cloud indices, its points, their distances, its coverage %)}, the
    seconds of the distances, the seconds of the coverage). An element with no point is not in it."""
    import zlib
    from bim import c2m_grid
    keys, idx, pts, tris, samples = [], [], [], [], []
    for key, (mesh_verts, mesh_faces, _ifc_type, _name) in all_bim.items():
        if key not in match_index or len(mesh_faces) == 0:
            continue
        indices = scan.indices_of_match(match_index[key])
        if len(indices) == 0:
            continue
        keys.append(key)
        idx.append(indices)
        pts.append(_moved(xyz_all[indices], T))                   # (element by element: the record's own numbers)
        tris.append(np.asarray(mesh_verts, dtype=np.float64)[np.asarray(mesh_faces)])
        samples.append(_coverage_samples(mesh_verts, mesh_faces, zlib.crc32(str(key).encode())))
    if not keys:
        return {}, 0.0, 0.0
    bounds = np.cumsum([0] + [len(p) for p in pts])
    P = np.concatenate(pts)
    pe = np.repeat(np.arange(len(keys)), np.diff(bounds))
    t0 = time.time()
    d = c2m_grid.distances_all(P, pe, np.concatenate(tris), np.repeat(np.arange(len(keys)), [len(t) for t in tris]), device)
    t_c2m = time.time() - t0
    t0 = time.time()
    drawn = [e for e, sm in enumerate(samples) if sm is not None]
    near = (c2m_grid.within_all(P, pe, np.concatenate([samples[e][0] for e in drawn]),
                                np.repeat(drawn, [len(samples[e][0]) for e in drawn]), proximity_m, device)
            if drawn else np.zeros(0, dtype=bool))
    t_cov = time.time() - t0
    out, at = {}, 0
    for e, key in enumerate(keys):
        coverage = 0.0
        if samples[e] is not None:
            n = len(samples[e][0])
            coverage = _coverage_of(near[at:at + n], samples[e][1], samples[e][2])
            at += n
        out[key] = (idx[e], P[bounds[e]:bounds[e + 1]], d[bounds[e]:bounds[e + 1]], coverage)
    return out, t_c2m, t_cov


SHEET_REPORT = "sabana_report.json"
SHEET_META = "sabana_meta.json"
SHEET_OCTREE = "sabana_potree"
SHEET_FILES = ("sabana_cloud.ply", "sabana.npz", SHEET_REPORT)   # + the octree's folder; the meta goes last
NEW_SHEET_DIR = ".new_sheet"


def stored_sheet(bim_dir) -> Optional[dict]:
    """The WHOLE sheet stored in a project's bim_comparison folder — its meta — or None. A sheet is
    whole when its meta (written last) and its octree are there: the files of a run that was
    stopped, or that is still writing, are not a sheet (USER 2026-11-19: "si no hay sabana por que
    mierda me muestra esa opcion")."""
    import json
    bim_dir = Path(bim_dir)
    meta_path = bim_dir / SHEET_META
    if not meta_path.is_file() or not (bim_dir / SHEET_OCTREE / "metadata.json").is_file():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None


def publish_sheet(new_dir, bim_dir) -> None:
    """A NEW sheet, whole in ``new_dir``, takes the stored one's place — only now (USER 2026-11-19:
    "la comparacion debe generarse y correrse una vez, y se debe guardar"; a launch used to delete
    the stored sheet at its START, and a run that failed or was stopped left the project with
    none). The stored meta goes first and the new meta comes last: in between there is no sheet,
    never half of one."""
    import shutil
    new_dir, bim_dir = Path(new_dir), Path(bim_dir)
    if not (new_dir / SHEET_META).is_file() or not (new_dir / SHEET_OCTREE / "metadata.json").is_file():
        raise RuntimeError(f"{new_dir} holds no whole sheet (its meta or its octree is missing): the stored one is kept")
    bim_dir.mkdir(parents=True, exist_ok=True)
    (bim_dir / SHEET_META).unlink(missing_ok=True)
    if (bim_dir / SHEET_OCTREE).exists():
        shutil.rmtree(bim_dir / SHEET_OCTREE)
    (new_dir / SHEET_OCTREE).rename(bim_dir / SHEET_OCTREE)
    for name in SHEET_FILES:
        if (new_dir / name).exists():
            os.replace(new_dir / name, bim_dir / name)
    os.replace(new_dir / SHEET_META, bim_dir / SHEET_META)
    shutil.rmtree(new_dir, ignore_errors=True)


def save_sabana(
    session_dir: str,
    results: list,
    tolerance: float,
    summary: dict,
    positions: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    out_dir=None,
    overall: Optional[dict] = None,
):
    """
    ``out_dir``: where the files are written (the project's bim_comparison folder when None).
    sabana_report.json holds every element's numbers and histogram — what the Report tab shows —
    and sabana_meta.json is written LAST: a folder with it holds a whole sheet.

    Save the sheet: sabana_cloud.ply (its points and their deviation colours), sabana_meta.json (the
    numbers of every element) and sabana.npz (how many points, at which tolerance — until 2026-11-19 a
    second, compressed copy of the PLY's points that nothing read: seconds of every comparison).
    ``positions`` (N, 3) and ``colors`` (N, 4, RGBA in 0-1): every evaluated element's points, as
    arrays — a scan of today is millions of points, never a list in the results.
    Returns the (N, 3) uint8 colours as written in the PLY.
    """
    # Resolve correct output directory for sábana artifacts
    sabana_dir = str(out_dir) if out_dir is not None else str(project_dirs(session_dir)[1])
    os.makedirs(sabana_dir, exist_ok=True)
    import json
    
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
    
    if positions is None or colors is None or len(positions) == 0:
        positions = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 4), dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    
    out_path = os.path.join(sabana_dir, "sabana.npz")
    np.savez(out_path, points=np.int64(len(positions)), tolerance_m=np.float64(tolerance))
    
    # Save as binary PLY for Potree streaming (same format as cleaned_cloud.ply)
    ply_path = os.path.join(sabana_dir, "sabana_cloud.ply")
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
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
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
    meta_path = os.path.join(sabana_dir, "sabana_meta.json")
    # The geometry epoch is the SCAN's — the cloud the sheet was computed on. (Until 2026-11-18 it was
    # looked for above the sheet's own folder: in a project that is the project's root, where no
    # scan is — the comparison of a project's scan ended here with an error.)
    _scan = _scan_of(Path(session_dir))
    if _scan is not None:
        meta["scan"] = _scan[1]
    _output = Path(session_dir) / "output"
    if _output.is_dir():
        from correction.epoch import stamp as _stamp_epoch
        _stamp_epoch(meta, _output)
    else:
        from correction.epoch import stamp_nearest as _stamp_epoch
        _stamp_epoch(meta, sabana_dir)
    with open(os.path.join(sabana_dir, SHEET_REPORT), "w") as f:
        json.dump({"ok": True, "date": meta.get("date"), "tolerance_mm": round(tolerance * 1000, 1),
                   "scan": meta.get("scan"), "overall": overall, "results": results}, f)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[BIM-Compare] Saved sábana: {len(positions)} points → {out_path}")
    print(f"[BIM-Compare] Global advance: {global_advance}% ({len(advance_values)} elements)")
    return rgb


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

def _scan_of(session_path: Path) -> Optional[Tuple[str, str]]:
    """(project, "date/source") of a scan directory …/projects/<project>/scans/<date>/src_<source>."""
    parts = session_path.parts
    if "projects" in parts and "scans" in parts:
        i, j = parts.index("projects"), parts.index("scans")
        if len(parts) > j + 2 and parts[j + 2].startswith("src_"):
            return parts[i + 1], f"{parts[j + 1]}/{parts[j + 2][4:]}"
    return None


def project_dirs(session_dir) -> Tuple[Path, Path]:
    """(ifcs, bim_comparison) — the folders of the project a scan belongs to. A scan directory
    …/projects/<project>/scans/<date>/src_<source> names its own project (wherever that tree is);
    a session of the old layout is resolved as before."""
    session_path = Path(session_dir)
    if _scan_of(session_path) is not None:
        from project_paths import ProjectPaths
        i = session_path.parts.index("projects")
        paths = ProjectPaths(str(Path(*session_path.parts[:i + 1])), session_path.parts[i + 1])
        return paths.ifcs_dir, paths.bim_comparison_dir
    from project_paths import resolve_session
    slug = session_path.parts[session_path.parts.index("projects") + 1] if "projects" in session_path.parts else session_path.name
    ctx = resolve_session(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), slug)
    return (ctx.ifcs_dir if hasattr(ctx, "ifcs_dir") else session_path,
            ctx.bim_comparison_dir if hasattr(ctx, "bim_comparison_dir") else session_path)


def _moved(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def run_comparison(
    session_dir: str,
    matches: List[dict],
    tolerance: float = DEFAULT_TOLERANCE,
    progress_callback=None,
    skip_registration: bool = False,
    out_dir=None,
) -> dict:
    """
    ``out_dir``: the folder the sheet is written in. None — the project's bim_comparison folder
    itself, whose previous sheet is removed first (the tests, a direct call). The job gives a
    folder beside it and publishes the sheet when it is whole (:func:`publish_sheet`): the stored
    one stays until then.

    Run full BIM vs Scan comparison:
      1. Registration using matched segments (alignment only)
      2. Extract ALL BIM element geometry
      3. For matched elements: C2M deviation → green/yellow/red sábana
      4. For unmatched elements: gray transparent (not yet built)
      5. Cumulative coverage tracking with occlusion detection
    
    matches: what bim.scan_data.Scan.match gives —
      [{"segment_label", "segment_ids": [instance ids], "element_key", "ifc_type", …}]
    (a match with no ``segment_ids`` names every object labelled ``segment_label``, as in March).

    The answer carries each element's numbers and NO point: the sheet is on disk (save_sabana).
    """
    import zlib

    from bim.registration import register
    from bim.scan_data import Scan, ScanError, first_ifc, stac_source
    import yaml as _yaml
    
    # Coverage engine imports (graceful fallback if not yet available)
    try:
        from bim.coverage_store import CoverageStore, SampleStatus, ElementState
        from bim.occlusion_raycaster import (
            load_camera_positions, classify_bim_surface, build_segment_labels
        )
        COVERAGE_ENGINE_AVAILABLE = True
    except ImportError as _ce:
        print(f"[BIM-Compare] Coverage engine not available: {_ce}")
        COVERAGE_ENGINE_AVAILABLE = False
    
    # Read coverage proximity from config (once, not per element)
    _cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(_cfg_path) as _f:
        _cfg = _yaml.safe_load(_f)
    proximity_m = _cfg.get("bim", {}).get("deviation", {}).get("coverage_proximity_m", 0.15)
    
    # Coverage engine config
    ce_cfg = _cfg.get("coverage_engine", {})
    ce_enabled = ce_cfg.get("enabled", True) and COVERAGE_ENGINE_AVAILABLE
    ray_radius = ce_cfg.get("ray_radius", 0.05)
    completion_threshold = ce_cfg.get("completion_threshold", 80.0)
    quality_threshold = ce_cfg.get("quality_threshold", 80.0)
    
    t_start = time.time()
    session_path = Path(session_dir)
    output_dir = session_path / "output"
    
    # Resolve project-level paths (IFCs, bim_comparison dir)
    ifcs_dir, bim_dir = project_dirs(session_path)
    
    import shutil
    sheet_dir = bim_dir if out_dir is None else Path(out_dir)
    if out_dir is not None:
        # the stored sheet is NOT touched: the new one is written beside it and published whole
        if sheet_dir.exists():
            shutil.rmtree(sheet_dir)
        sheet_dir.mkdir(parents=True)
    # Clean up previous sábana artifacts before regenerating (a direct call writes in place)
    for old_file in ["sabana.npz", "sabana_cloud.ply", "sabana_meta.json", SHEET_REPORT] if out_dir is None else []:
        for search_dir in [bim_dir, session_path]:
            p = search_dir / old_file
            if p.exists():
                p.unlink()
                print(f"[BIM] 🗑️ Deleted old {old_file}")
    sabana_potree_dir = bim_dir / "sabana_potree"
    if out_dir is None and sabana_potree_dir.exists():
        shutil.rmtree(sabana_potree_dir)
        print("[BIM] 🗑️ Deleted old sabana_potree/")
    # Also check session_path for legacy location
    legacy_potree = session_path / "sabana_potree"
    if legacy_potree.exists():
        shutil.rmtree(legacy_potree)
    
    # Find IFC file (check ifcs_dir first, then session_path for legacy)
    ifc_path = first_ifc(ifcs_dir, session_path)
    if ifc_path is None:
        return {"error": "No IFC file found in session"}
    
    # The scan: its objects and (read once, below) its cloud
    try:
        scan = Scan(output_dir)
    except ScanError as e:
        return {"error": str(e)}
    
    import ifcopenshell
    ifc_file = ifcopenshell.open(str(ifc_path))
    
    # Build match index: element_key → the match (its points are asked from the scan)
    match_index = {str(m["element_key"]): m for m in matches}
    
    # ── Step 1: Registration (alignment using matched segments) ──
    if progress_callback:
        progress_callback(5, "Registering scan to BIM...")
    
    # An IFC that STAC wrote from THIS scan is in this scan's display frame by construction: there
    # is nothing to register (and the registration would take its proxies for floors).
    registration = "manual" if skip_registration else "auto"
    own = stac_source(ifc_file)
    if own is not None and own == _scan_of(session_path):
        registration = "own_scan"
        print(f"[BIM-Compare] {ifc_path.name} was written from this scan ({own[1]}): same frame, no registration")
        # … and its elements ARE this scan's objects, one by one: the pairing is the scan's own, by
        # instance id, whatever the request carried. (The Report tab sent its pairing without the
        # object ids: every element was compared with ALL the points of its label — 266 M points in
        # place of 18.6 M, over 20 minutes and a wrong sheet, 2026-11-18.)
        from bim.scan_data import element_keys
        own_index = {str(m["element_key"]): m for m in scan.match(element_keys(ifc_file), by_instance_id=True)}
        if any(not m.get("segment_ids") for m in match_index.values()):
            print(f"[BIM-Compare] the request's pairing carried no object ids: the scan's own pairing is used "
                  f"({len(own_index)} objects by instance id)")
        match_index = own_index
    
    if registration != "auto":
        # Use saved floor_transform (gizmo-aligned position) as the transform
        transform_path = output_dir / "floor_transform.npz"
        if transform_path.exists():
            data = np.load(transform_path)
            s = float(data["s"])
            R = data["R"]
            t = data["t"]
            T = np.eye(4)
            T[:3, :3] = s * R
            T[:3, 3] = t
            if registration == "manual":
                print(f"[BIM-Compare] Using manual alignment (floor_transform), skipping auto-registration")
        else:
            T = np.eye(4)
            print(f"[BIM-Compare] No floor_transform found, using identity")
    else:
        T = register(session_dir, matches, progress_callback=progress_callback)
        if T is None:
            T = np.eye(4)
            print("[BIM-Compare] Registration failed, using identity transform")
    
    # ── Step 2: Extract ALL BIM geometry ──
    if progress_callback:
        progress_callback(80, "Extracting all BIM geometry...")
    
    t_geom = time.time()
    all_bim = extract_all_ifc_geometry(str(ifc_path), ifc_file)
    t_geom = time.time() - t_geom
    
    # Load scan cloud (raw points)
    try:
        xyz_all = scan.xyz
    except ScanError as e:
        return {"error": str(e)}
    print(f"[BIM-Compare] scan: {len(xyz_all)} points, {len(scan.instances)} objects ({scan.source}); "
          f"IFC {ifc_path.name}: {len(all_bim)} elements with geometry ({t_geom:.1f}s); {len(match_index)} matched")
    
    # ── Step 2b: Coverage engine setup ──
    # (OFF by config.yaml coverage_engine.enabled since before the scans of today; its ray casting
    #  is still March's — a tree of the whole cloud per element — and was not adapted to them)
    cam_positions = None
    seg_labels = None
    cov_store = None
    scan_registered = None
    scan_id = time.strftime("%Y%m%d_%H%M%S")
    seg_result_path = output_dir / "segmentation_result.json"
    seg_path = output_dir / "segmentation.json"
    
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
            scan_registered = _moved(xyz_all, T)
            
            # Initialize coverage store
            cov_store = CoverageStore(session_dir)
            print(f"[BIM-Compare] Coverage engine active: {len(cam_positions)} cameras")
    
    if progress_callback:
        progress_callback(85, "Computing deviations...")
    
    # ── Step 3: Evaluate each BIM element ──
    results = []
    sheet_positions = []
    sheet_colors = []
    sheet_indices = []                                            # the cloud index of every point of the sheet
    sheet_distances = []                                          # … and its distance to its element
    total = len(all_bim)
    tolerance_mm = tolerance * 1000
    t_c2m = t_cov = 0.0
    # on a device: every element's distances and coverage in ONE pass, before the loop
    on_device = {}
    if device_of() != "numpy":
        on_device, t_c2m, t_cov = _evaluate_on_the_device(all_bim, match_index, scan, xyz_all, T, proximity_m, device_of())
    
    for i, (key, (mesh_verts, mesh_faces, ifc_type, elem_name)) in enumerate(all_bim.items()):
        n_faces = len(mesh_faces)
        
        if key in match_index:
            # ── MATCHED: compare segment points vs BIM mesh ──
            coverage = None
            if key in on_device:
                indices, scan_points, distances, coverage = on_device[key]
            else:
                indices = scan.indices_of_match(match_index[key])
                if len(indices) == 0:
                    results.append({
                        "element_key": key, "label": elem_name,
                        "ifc_type": ifc_type, "status": "error",
                        "error": "No points: the scan has no object for this match",
                    })
                    continue

                # Transform segment points with registration
                scan_points = _moved(xyz_all[indices], T)

                # C2M distances: how far each constructed point is from the BIM design
                _t = time.time()
                distances = compute_c2m_distances(scan_points, mesh_verts, mesh_faces, quiet=True)
                t_c2m += time.time() - _t
            report = build_deviation_report(distances, tolerance)
            
            # Sábana: scan points at their real positions, colored by C2M deviation
            # Shows the CONSTRUCTION, colored by how well it matches the design
            sabana_pos, sabana_colors = compute_sabana_cloud(
                scan_points, distances, tolerance, quiet=True
            )
            sheet_positions.append(np.asarray(sabana_pos, dtype=np.float32))
            sheet_colors.append(np.asarray(sabana_colors, dtype=np.float32))
            sheet_indices.append(np.asarray(indices, dtype=np.int64))
            sheet_distances.append(distances)
            
            # Coverage: what % of this BIM element's surface has scan data
            if coverage is None:
                _t = time.time()
                coverage = compute_coverage_pct(
                    mesh_verts, mesh_faces, scan_points, proximity_m,
                    seed=zlib.crc32(str(key).encode()), quiet=True,
                )
                t_cov += time.time() - _t
            
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
                "segment_ids": [int(i) for i in (match_index[key].get("segment_ids") or [])],   # the scan's objects compared
                "total_points": len(distances),
                "coverage_pct": coverage,
                "coverage_cumulative": coverage_cumulative,
                "occluded_pct": occluded_pct,
                "element_state": element_state,
                "bim_surface_m2": round(bim_surface_m2, 4),
                **report,
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
    
    # The InternVL3 occluder classification (scene_analyzer.classify_occluders: a heuristic
    # lookup, then a VLM of its own that never served) is GONE — docs/plan_determinismo.md
    # point 156 (2026-10-08): the pipeline has no second VLM and no silent fallback; the
    # coverage store keeps the occluder types its own engine measured.

    print(f"[BIM-Compare] Summary: {n_evaluated} evaluated, "
          f"{n_unmatched} unmatched (not built), {n_error} errors, "
          f"{total} total elements — distances {t_c2m:.1f}s, coverage {t_cov:.1f}s, "
          f"{time.time() - t_start:.1f}s in all")
    
    if progress_callback:
        progress_callback(100, "Comparison complete")
    
    summary = {
        "total_elements": total,
        "evaluated": n_evaluated,
        "unmatched": n_unmatched,
        "errors": n_error,
        "registration": registration,
        "ifc": ifc_path.name,
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
    
    # EVERY COMPARED POINT TOGETHER: its numbers, and its histogram over 0 … 2 × tolerance — the scale the
    # points are painted with (green at 0, the tolerance in the middle, red at twice it) — with the count
    # of those beyond. (The Report's chart was the first element's histogram alone, USER 2026-11-19: "no se
    # entiende el reporte".)
    overall = None
    if sheet_distances:
        every = np.concatenate(sheet_distances)
        edges = np.linspace(0.0, 2.0 * tolerance, 41)
        overall = {"stats": build_deviation_report(every, tolerance)["stats"],
                   "histogram": {"counts": np.histogram(every, bins=edges)[0].tolist(), "bin_edges_mm": (edges * 1000).tolist(),
                                 "beyond": int(np.sum(every > 2.0 * tolerance))}}

    # Save sábana to session for later loading
    _t = time.time()
    sheet_rgb = save_sabana(
        session_dir, results, tolerance, summary,
        np.concatenate(sheet_positions) if sheet_positions else None,
        np.concatenate(sheet_colors) if sheet_colors else None,
        out_dir=sheet_dir, overall=overall,
    )
    t_save = time.time() - _t

    # The sheet's octree FROM THE SCAN'S OWN (bim/sheet_octree.py): the same points, their colours
    # replaced — seconds, where building one from the sheet's cloud took 40. When the scan's octree
    # cannot stand for the cloud the reason is said and the job builds the octree the old way.
    _t = time.time()
    summary["sheet_octree"] = "built from the sheet's cloud"
    if sheet_indices and len(sheet_rgb) and not np.array_equal(T, np.eye(4)):
        # (the sheet's points were MOVED by the alignment: they are not the scan's octree's)
        print("[BIM-Compare] the sheet's octree: the scan was moved to the model — built from the sheet's cloud")
    elif sheet_indices and len(sheet_rgb):
        try:
            from bim import sheet_octree
            from potree_converter import potree_stamp_for, write_potree_stamp
            rgb_cloud = np.empty((len(xyz_all), 3), dtype=np.uint8)
            rgb_cloud[:] = sheet_octree.NO_DATA_RGB8
            rgb_cloud[np.concatenate(sheet_indices)] = sheet_rgb
            made, said = sheet_octree.build(output_dir / "potree", scan.cloud, rgb_cloud, sheet_dir / SHEET_OCTREE, device_of())
            if made:
                write_potree_stamp(sheet_dir / SHEET_OCTREE, potree_stamp_for(sheet_dir / "sabana_cloud.ply"))
                summary["sheet_octree"] = "the scan's own octree, recoloured"
            print(f"[BIM-Compare] the sheet's octree: {said} — {time.time() - _t:.1f}s (the sheet's files {t_save:.1f}s)")
        except Exception as e:  # noqa: BLE001 — said; the job then builds the octree from the sheet's cloud
            print(f"[BIM-Compare] the sheet's octree could not be made from the scan's ({type(e).__name__}: {e})")
    
    return {
        "ok": True,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tolerance_mm": round(tolerance * 1000, 1),
        "transform": T.tolist(),
        "summary": summary,
        "overall": overall,
        "results": results,
    }
