"""AN IFC FROM THE GENRECON MESH, CUT BY THE SEGMENTATION — a scan's as-built model.

USER 2026-11-17, on the first IFC (each object as the box the store holds — "muy grande gordo"):
"por que no exportas el genrecon que esta perfecto, para eso esta" / "asi debe ser ifc desde la
malla genrecon cortada desde la segmentacion, exacto". The boxes were measured too: a wall's box
25 cm thick (median) around points 2 cm thick, the floors' boxes 1.5 m.

  1. the mesh: the scan's published GenRecon GLB, its positions and triangles as they are on disk
     (it is in the scan's display frame — the baker's Y-up);
  2. SIMPLIFICATION of the whole mesh as the one surface it is (``keep``: the share of its
     triangles kept, quadric decimation): an IFC is text and the whole mesh does not open in the
     programs that read it. ``keep = 1`` writes every triangle. What a value costs is MEASURED:
     the distance from the full surface to the simplified one (:func:`deviation`);
  3. the cut, AFTER: a triangle belongs to the object most of the cloud points lying on it
     belong to (:func:`cut` — the cloud and its per-point instance ids are the segmentation's
     own); one with no point farther than one GenRecon voxel from the cloud belongs to none;
  4. one IFC element per object — its part of the mesh as an IfcTriangulatedFaceSet, its class
     from its label (:data:`bim.ifc_from_scan.IFC_CLASS`; any other object is an
     IfcBuildingElementProxy), named ``<label>:<label>:<instance id>`` (the key the comparison
     matches by); the triangles of unsegmented points are ONE proxy, "unsegmented", so the model
     has no holes the mesh does not have;

Coordinates are written Z-up, (x, y, z) → (x, −z, y), rounded to 0.1 mm (the mesh's cells are
4.9 mm): it keeps the file a third smaller than the float32's full digits.

THE TEXTURE (USER 2026-11-24: "es posible generar los IFC con la textura de genrecon?" / "proba mejor con la textura
real primero si no sale vamos por el color dominante"): when the mesh is written whole (``keep`` 1 — a simplified
mesh's triangles are new: they have no texture), each triangle keeps its corner texture coordinates and its material
through the welding, the split of the large triangles (interpolated in their parent triangle) and the cut. An
element is then one IfcTriangulatedFaceSet per atlas image it uses, each with its IfcIndexedTriangleTextureMap
(IFC's t up from the image's bottom: t = 1 − glTF's v) and a style of two layers: the image (IfcImageTexture, the
base-colour images copied out of the GLB as they are, beside the IFC in ``<name>_textures/``) and the element's
DOMINANT colour (IfcSurfaceStyleShading — "pintar cada elemento por el color dominante del elemento, un color plano
con el color del elemento"): a program that reads IFC textures shows the photograph, one that does not (Revit, the
viewer's web-ifc) the colour.
"""
from __future__ import annotations

import io
import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ifc_from_scan import IFC_CLASS, TO_IFC, _guid

PROXY = ("IfcBuildingElementProxy", None)
UNSEGMENTED = 0
_DECIMALS = 4                      # 0.1 mm
_MIN_FACES = 4                     # a surface simplified under a tetrahedron's faces is none


class MeshError(ValueError):
    pass


def read_glb(path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices N×3 float32, triangles M×3 int64) of a binary glTF: every primitive's positions
    and indices, as stored (the nodes of a published GenRecon mesh carry no transform — checked)."""
    d = read_glb_textured(path, textures=False)
    return d["vertices"], d["faces"]


def read_glb_textured(path: Path | str, *, textures: bool = True) -> Dict[str, Any]:
    """read_glb's vertices and faces and — ``textures``, and every primitive with its TEXCOORD_0 and a material with a
    base-colour image — each triangle's corner texture coordinates (M×3×2 float32, glTF's: v down from the image's
    top), its material, and where each material's image lies in the file ({material: {offset, length, mime}})."""
    with open(path, "rb") as f:
        magic, _version, _total = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise MeshError(f"{path}: not a binary glTF")
        clen, _ctype = struct.unpack("<II", f.read(8))
        doc = json.loads(f.read(clen))
        f.read(8)
        base = f.tell()
        if doc.get("extensionsRequired"):
            raise MeshError(f"{path}: requires {doc['extensionsRequired']} (plain accessors are read)")
        for n in doc.get("nodes", []):
            if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
                raise MeshError(f"{path}: node '{n.get('name')}' carries a transform (a baked mesh is read)")
        acc, views = doc["accessors"], doc["bufferViews"]

        def array(index: int, dtype: str, comps: int) -> np.ndarray:
            a, v = acc[index], views[acc[index]["bufferView"]]
            size = np.dtype(dtype).itemsize * comps
            if v.get("byteStride") not in (None, size):
                raise MeshError(f"{path}: an interleaved buffer view")
            f.seek(base + v.get("byteOffset", 0) + a.get("byteOffset", 0))
            return np.frombuffer(f.read(a["count"] * size), dtype).reshape(-1, comps)

        types = {5123: "<u2", 5125: "<u4"}
        verts, faces, uvs, mats, offset = [], [], [], [], 0
        images: Dict[int, Dict[str, Any]] = {}
        textured = textures
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                if prim.get("mode", 4) != 4:
                    raise MeshError(f"{path}: a primitive that is not triangles")
                v = array(prim["attributes"]["POSITION"], "<f4", 3)
                idx = acc[prim["indices"]]
                i = array(prim["indices"], types[idx["componentType"]], 1).reshape(-1, 3)
                verts.append(v)
                faces.append(i.astype(np.int64) + offset)
                offset += len(v)
                if not textured:
                    continue
                m = prim.get("material")
                tc = prim["attributes"].get("TEXCOORD_0")
                if m is None or tc is None or acc[tc].get("componentType") != 5126:
                    textured = False
                    continue
                if m not in images:
                    bct = (((doc.get("materials") or [])[m].get("pbrMetallicRoughness") or {}).get("baseColorTexture") or {})
                    texs, imgs = doc.get("textures") or [], doc.get("images") or []
                    t_i = bct.get("index")
                    src = texs[t_i].get("source") if t_i is not None and t_i < len(texs) else None
                    img = imgs[src] if src is not None and src < len(imgs) else None
                    if not img or "bufferView" not in img:
                        textured = False
                        continue
                    view = views[img["bufferView"]]
                    images[m] = {"offset": base + view.get("byteOffset", 0), "length": view["byteLength"],
                                 "mime": img.get("mimeType", "image/png")}
                uvs.append(array(tc, "<f4", 2)[i])                          # each corner's own coordinates
                mats.append(np.full(len(i), m, np.int32))
    out: Dict[str, Any] = {"vertices": np.vstack(verts), "faces": np.vstack(faces), "uv": None, "material": None,
                           "images": {}}
    if textured and uvs:
        out.update(uv=np.vstack(uvs).astype(np.float32), material=np.concatenate(mats), images=images)
    return out


def _most_frequent(rows: np.ndarray) -> np.ndarray:
    """The most frequent value of each row (a tie: the smaller)."""
    s = np.sort(rows, axis=1)
    run = np.ones_like(s)
    for j in range(1, s.shape[1]):
        run[:, j] = np.where(s[:, j] == s[:, j - 1], run[:, j - 1] + 1, 1)
    return s[np.arange(len(s)), run.argmax(axis=1)]


def cut_by_neighbours(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, point_ids: np.ndarray,
                      max_distance: float, neighbours: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The object of each triangle: the object MOST OF THE ``neighbours`` CLOUD POINTS NEAREST ITS
    CENTRE belong to (a tie: the smaller id); none (−1) when the nearest of them is farther than
    ``max_distance``.

    Why a neighbourhood (measured on pccr 3108, 2026-11-18): THE SEGMENTATION ITSELF IS
    INTERLEAVED — a cloud point and its nearest neighbour, 3.7 mm away, belong to different
    objects 37 % of the time (floor / tile, wall / drywall, panel / wall: two concepts on one
    surface). A triangle cut by the points lying on it alone has a neighbour of another object
    29 % of the time; by 32 neighbours 14 %, by 128 neighbours 9 % — and the wider the
    neighbourhood, the more small objects lose every triangle (58 / 85 / 140 of 1213)."""
    from scipy.spatial import cKDTree
    ids = np.asarray(point_ids, np.int64)
    centres = vertices[faces].mean(axis=1)
    tree = cKDTree(points)
    owner = np.empty(len(faces), np.int64)
    far = 0
    for k in range(0, len(faces), 1_000_000):
        d, idx = tree.query(centres[k:k + 1_000_000], k=int(neighbours), workers=-1)
        d, idx = d.reshape(len(d), -1), idx.reshape(len(idx), -1)
        block = _most_frequent(ids[idx])
        block[d[:, 0] > max_distance] = -1
        far += int((d[:, 0] > max_distance).sum())
        owner[k:k + 1_000_000] = block
    # EVERY OBJECT APPEARS (USER 2026-11-18: "deben aparecer todos los elementos"): an object the
    # majority left with no triangle — a small one on a surface another object dominates — takes
    # the triangles ITS OWN POINTS LIE ON (each point's nearest triangle, exact)
    lost = np.setdiff1d(np.unique(ids[ids > 0]), np.unique(owner[owner > 0]))
    rescued = 0
    if len(lost):
        mine = np.flatnonzero(np.isin(ids, lost))
        own, _ = cut(vertices, faces, points[mine], ids[mine], max_distance, fill=False)
        take = own > 0
        owner[take] = own[take]
        rescued = int(len(np.unique(own[take])))
    return owner, {"faces_no_object": far, "neighbours": int(neighbours), "objects_rescued": rescued,
                   "objects_left_out_by_the_majority": int(len(lost))}


def cut(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray, point_ids: np.ndarray,
        max_distance: float, fill: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The object of each triangle: THE OBJECT MOST OF THE POINTS LYING ON IT BELONG TO.

    Every cloud point lies on its nearest triangle (exact, over the surface); a triangle is the
    object of the majority of its points (a tie: the smaller id). A triangle no point lies on
    takes the object of the cloud point nearest its centre — and belongs to no object (−1) when
    that point is farther than ``max_distance`` (``fill`` False: it is left unassigned, −2).

    Why not the nearest point alone (measured on pccr 3108, 2026-11-18): on one surface the points
    of several objects are interleaved, and by the nearest point 20 % of the neighbouring
    triangles fell in different objects — an element came out as tens of thousands of crumbs."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))
    face_of = np.empty(len(points), np.int64)
    for k in range(0, len(points), 4_000_000):
        hit = scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(points[k:k + 4_000_000], np.float32)))
        face_of[k:k + 4_000_000] = hit["primitive_ids"].numpy()
    ids = np.asarray(point_ids, np.int64)
    # (face, id) pairs counted; per face the most frequent id, the smaller on a tie
    order = np.lexsort((ids, face_of))
    fo, io = face_of[order], ids[order]
    start = np.flatnonzero(np.r_[True, (fo[1:] != fo[:-1]) | (io[1:] != io[:-1])])
    count = np.diff(np.r_[start, len(fo)])
    gf, gi = fo[start], io[start]
    best = np.lexsort((gi, -count, gf))
    first = np.r_[True, gf[best][1:] != gf[best][:-1]]
    owner = np.full(len(faces), -2, np.int64)
    owner[gf[best][first]] = gi[best][first]
    empty = np.flatnonzero(owner == -2)
    if not fill:                                      # (only the triangles points lie on: the rest stay −2)
        return owner, {"faces_without_points": int(len(empty))}
    if len(empty):
        d, k = cKDTree(points).query(vertices[faces[empty]].mean(axis=1), k=1, workers=-1)
        owner[empty] = np.where(d <= max_distance, ids[k], -1)
    return owner, {"faces_without_points": int(len(empty)), "faces_no_object": int((owner == -1).sum())}


def submesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The triangles ``faces`` with their own vertices, WELDED BY POSITION. A textured mesh
    repeats a vertex wherever the texture's charts meet (one copy per chart, the same position):
    by index, an object's surface is tens of thousands of loose pieces (measured on pccr 3108:
    45,793 for one wall, 44,329 of them under 20 triangles) and a simplification deletes them.
    Vertices at the same float32 position are one; a triangle that collapses is dropped."""
    v, f, _kept = weld(vertices, faces)
    return v, f


def weld(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """submesh, and which of ``faces`` were kept (in their order): a triangle's own data (its texture
    coordinates, its material) follows it."""
    used, inverse = np.unique(faces, return_inverse=True)
    v = np.ascontiguousarray(vertices[used])
    _, first, w = np.unique(v.view([("", v.dtype)] * 3).ravel(), return_index=True, return_inverse=True)
    f = w[inverse].reshape(-1, 3)
    kept = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return v[first], f[kept], kept


def simplify(v: np.ndarray, f: np.ndarray, keep: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric decimation to ``keep`` of the triangles (the element as it is when keep = 1 or it
    is too small to simplify)."""
    target = max(_MIN_FACES, int(round(len(f) * keep)))
    if keep >= 1.0 or len(f) <= target:
        return v, f
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)), o3d.utility.Vector3iVector(f.astype(np.int32)))
    m = m.simplify_quadric_decimation(target_number_of_triangles=target)
    m.remove_unreferenced_vertices()
    out_v, out_f = np.asarray(m.vertices), np.asarray(m.triangles).astype(np.int64)
    return (out_v, out_f) if len(out_f) else (v, f)


def split_large(v: np.ndarray, f: np.ndarray, max_edge: float, *, return_parent: bool = False):
    """The triangles whose longest edge is over ``max_edge`` split (in their own plane: the
    surface does not move) until none is. A simplification leaves triangles of metres on a flat
    floor or wall — larger than the objects on it (a 57 cm tile), so the cut could not follow
    them (measured on pccr 3108 at 10 %: the longest edge reached 1.85 m)."""
    import trimesh
    edges = np.linalg.norm(v[f] - v[f[:, [1, 2, 0]]], axis=2).max(axis=1)
    big = edges > max_edge
    if not big.any():
        return (v, f, np.arange(len(f))) if return_parent else (v, f)
    nv, nf, idx = trimesh.remesh.subdivide_to_size(np.asarray(v, np.float64), f[big], max_edge=max_edge, max_iter=30,
                                                   return_index=True)
    out_v, out_f = np.vstack([v, nv.astype(v.dtype)]), np.vstack([f[~big], nf.astype(np.int64) + len(v)])
    if return_parent:                    # each triangle's own one in ``f`` (the split ones: the one they lie in)
        return out_v, out_f, np.concatenate([np.flatnonzero(~big), np.flatnonzero(big)[idx]])
    return out_v, out_f


def corner_uv(v: np.ndarray, f: np.ndarray, parent: np.ndarray, src_v: np.ndarray, src_f: np.ndarray,
              src_uv: np.ndarray) -> np.ndarray:
    """The corner texture coordinates of triangles ``f`` lying in the triangles ``src_f[parent]``: each corner's
    barycentric coordinates in its parent triangle, applied to the parent's corner coordinates."""
    a, b, c = (np.asarray(src_v[src_f[parent, k]], np.float64) for k in range(3))
    p = np.asarray(v[f], np.float64)                                     # P×3 corners×3
    e0, e1, e2 = b - a, c - a, p - a[:, None, :]
    d00, d01, d11 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1)
    d20, d21 = (e2 * e0[:, None]).sum(-1), (e2 * e1[:, None]).sum(-1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-24
    den = np.where(ok, den, 1.0)
    lb = (d11[:, None] * d20 - d01[:, None] * d21) / den[:, None]
    lc = (d00[:, None] * d21 - d01[:, None] * d20) / den[:, None]
    la = 1.0 - lb - lc
    uv = src_uv[parent].astype(np.float64)                               # P×3×2
    out = la[..., None] * uv[:, None, 0] + lb[..., None] * uv[:, None, 1] + lc[..., None] * uv[:, None, 2]
    out[~ok] = uv[~ok].mean(axis=1, keepdims=True)                       # a degenerate parent: its mean
    return out.astype(np.float32)


def deviation(v: np.ndarray, sv: np.ndarray, sf: np.ndarray) -> np.ndarray:
    """The distance from each original vertex ``v`` to the simplified surface (sv, sf), metres."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(sv.astype(np.float32)), o3d.core.Tensor(sf.astype(np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(v.astype(np.float32))).numpy()


def elements_of(output_dir: Path | str, mesh_path: Path | str, *, keep: float, max_distance: float,
                neighbours: int = 0, max_edge: Optional[float] = None, measure: bool = False,
                log: Callable[[str], Any] = print) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The scan's elements, each with its part of the mesh (display frame), and the report.

    THE FRAME (USER 2026-11-19, after bufferStop's export was refused for its floor adjustment: "dale,
    si, hacelo siempre" / "cuando se aplica ajuste de piso debe alcanzar a las mallas tambien, total
    es una rotacion"): the mesh on disk and the cloud on disk are in ONE frame, the scan's stored one
    — the cut happens there. A floor adjustment (floor_transform.npz: the floor-level tool, the
    alignment gizmo) is applied by the viewer to both, the meshes hanging from the cloud's group; it
    is applied HERE to every element's vertices, so the IFC lies on the cloud as it is shown and
    exported. (It used to refuse such a scan.)

    THE ORDER MATTERS: the whole mesh is simplified FIRST, as the one surface it is, and cut
    AFTER. Simplifying each object's part on its own was measured (pccr 3108): the parts' ragged
    borders and loose crumbs are deleted by the decimation — 12 mm of deviation at the 95th
    percentile keeping 25 % of the triangles, metres at the worst."""
    from exporter import cloud as C
    out = Path(output_dir)
    if not (0.0 < keep <= 1.0):
        raise MeshError(f"keep must be in (0, 1], got {keep!r}")
    A, t = C.floor_transform(out)                               # display = A · stored + t
    adjusted = not (np.array_equal(A, np.eye(3)) and not t.any())
    glb = read_glb_textured(mesh_path)
    full_v, full_f, kept = weld(glb["vertices"], glb["faces"])     # welded: one surface
    uv = glb["uv"][kept] if glb["uv"] is not None else None        # each triangle's own texture, kept with it
    mat = glb["material"][kept] if glb["material"] is not None else None
    del glb["vertices"], glb["faces"], glb["uv"], glb["material"]
    log(f"mesh: {len(full_v):,} vertices, {len(full_f):,} triangles (welded)"
        + (f"; textured: {len(glb['images'])} images" if uv is not None else "; no texture"))
    vertices, faces = simplify(full_v, full_f, keep)
    log(f"simplified to {len(faces):,} triangles (keep {keep})")
    if uv is not None and faces is not full_f:
        uv = mat = None                                          # a simplified mesh's triangles are new: no texture
        log("texture: not carried — the mesh was simplified (its triangles are new); the elements get no colour")
    report: Dict[str, Any] = {"mesh_vertices": int(len(full_v)), "mesh_faces": int(len(full_f)), "keep": float(keep),
                              "faces": int(len(faces)), "max_distance": float(max_distance)}
    if measure and len(faces) != len(full_f):
        d = deviation(full_v, vertices, faces)
        report["deviation_m"] = {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                                 "p99": float(np.percentile(d, 99)), "max": float(d.max())}
    del full_v, full_f
    if max_edge:
        if uv is not None:
            v2, f2, parent = split_large(vertices, faces, float(max_edge), return_parent=True)
            uv, mat = corner_uv(v2, f2, parent, vertices, faces, uv), mat[parent]
            vertices, faces = v2, f2
        else:
            vertices, faces = split_large(vertices, faces, float(max_edge))
        report["faces_after_split"] = int(len(faces))
        report["max_edge"] = float(max_edge)
        log(f"triangles over {max_edge * 100:.1f} cm split: {len(faces):,} triangles")
    cloud = C.open_ply(out / "cleaned_cloud.ply")
    ids = np.load(out / "instance_ids.npy")
    if len(ids) != len(cloud):
        raise MeshError(f"{out}: instance_ids.npy holds {len(ids):,} ids for a cloud of {len(cloud):,} points")
    points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]])
    # ``neighbours`` = 0: the points lying ON the triangle; else that many nearest points
    owner, cut_report = (cut_by_neighbours(vertices, faces, points, ids, max_distance, neighbours) if neighbours
                         else cut(vertices, faces, points, ids, max_distance))
    report.update(cut_report)
    log(f"cut: {cut_report}")
    con = sqlite3.connect(f"file:{out / 'scene_r.db'}?mode=ro", uri=True)
    try:
        labels = dict(con.execute("SELECT instance_id, label FROM instances"))
        n_points = dict(con.execute("SELECT instance_id, n_points FROM instance_obb WHERE window_id = 'global'"))
    finally:
        con.close()
    order = np.argsort(owner, kind="stable")
    owners, starts = np.unique(owner[order], return_index=True)
    bounds = list(starts) + [len(order)]
    elements = []
    for k, iid in enumerate(owners):
        iid = int(iid)
        if iid < 0:
            continue
        if iid != UNSEGMENTED and iid not in labels:
            raise MeshError(f"the cloud holds instance {iid}, which the store does not")
        idx = order[bounds[k]:bounds[k + 1]]
        v, f, kept_e = weld(vertices, faces[idx])
        if not len(f):
            continue
        if adjusted:
            v = np.asarray(v, np.float64) @ A.T + t             # the floor adjustment, as the viewer applies it
        label = "unsegmented" if iid == UNSEGMENTED else str(labels[iid])
        el = {"id": iid, "label": label, "ifc": IFC_CLASS.get(label, PROXY), "vertices": v, "faces": f,
              "n_points": int(n_points.get(iid, 0))}
        if uv is not None:
            el["uv"], el["material"] = uv[idx][kept_e], mat[idx][kept_e]
        elements.append(el)
    with_faces = {e["id"] for e in elements}
    report["texture"] = ({"glb": str(mesh_path), "images": {int(m): i for m, i in glb["images"].items()},
                          "materials_used": int(len({int(m) for e in elements for m in np.unique(e["material"])}))}
                         if uv is not None else None)
    report.update(floor_adjusted=bool(adjusted),
                  elements=len(elements), faces_written=int(sum(len(e["faces"]) for e in elements)),
                  faces_unsegmented=int((owner == UNSEGMENTED).sum()),
                  objects_without_faces=sorted(int(i) for i in set(labels) - with_faces))
    return elements, report


# the two corners of the colour cube at 5 bits a channel: every channel under 8, every channel 248 or more
_CLIPPED_BINS = np.array([0, (1 << 15) - 1])


def _dominant_bin(weight: np.ndarray) -> int:
    """The most common colour bin of one element's samples (``weight``: their weight per 5-bit bin). The two corners
    of the colour cube are where an under- or over-exposed photograph's colours COLLAPSE into one bin, while the rest
    of a surface spreads over many: they win only when they are most of the element (an object that is black stays
    black; a floor partly in the dark does not). Measured on pccr 2026-08-31's floor#72: 19 % of its points under a
    brightness of 16, 11.5 % in the dark corner — the plain mode was that corner, (3, 4, 3), where the median of its
    points is (92, 87, 77)."""
    total = float(weight.sum())
    if total > 0 and float(weight[_CLIPPED_BINS].sum()) < total / 2:
        weight = weight.copy()
        weight[_CLIPPED_BINS] = 0
    return int(np.argmax(weight))


def element_colours(elements: List[Dict[str, Any]], texture: Optional[Dict[str, Any]]) -> None:
    """Each textured element's DOMINANT colour, set as ``e["colour"]`` (r, g, b in 0..1): the colour of its triangles in
    the image at their centres, the most common one at 5 bits a channel (_dominant_bin: an exposure's dark or bright
    corner only when it is most of the element), each triangle weighing its area (USER 2026-11-24: "pintar cada
    elemento por el color dominante del elemento, un color plano con el color del elemento"). The images are decoded
    one at a time."""
    if not texture:
        return
    from PIL import Image
    parts: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for k, e in enumerate(elements):
        if "uv" in e:
            for m in np.unique(e["material"]):
                parts.setdefault(int(m), []).append((k, np.flatnonzero(e["material"] == m)))
    samples: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    with open(texture["glb"], "rb") as fh:
        for m, owners in sorted(parts.items()):
            info = texture["images"].get(m)
            if not info:
                continue
            fh.seek(info["offset"])
            img = np.asarray(Image.open(io.BytesIO(fh.read(info["length"]))).convert("RGB"))
            h, w = img.shape[:2]
            for k, fi in owners:
                e = elements[k]
                c = e["uv"][fi].mean(axis=1)                       # each triangle's centre in the image (v down)
                x = np.clip(((c[:, 0] % 1.0) * w).astype(np.int64), 0, w - 1)
                y = np.clip(((c[:, 1] % 1.0) * h).astype(np.int64), 0, h - 1)
                rgb = img[y, x].astype(np.int64)
                p = np.asarray(e["vertices"], np.float64)[np.asarray(e["faces"])[fi]]
                area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
                bins = (rgb[:, 0] >> 3) << 10 | (rgb[:, 1] >> 3) << 5 | (rgb[:, 2] >> 3)
                samples.setdefault(k, []).append((bins, area, rgb))
    for k, got in samples.items():
        bins = np.concatenate([g[0] for g in got]); area = np.concatenate([g[1] for g in got])
        rgb = np.concatenate([g[2] for g in got]).astype(np.float64)
        weight = np.bincount(bins, weights=area, minlength=1 << 15)
        top = _dominant_bin(weight)
        sel = bins == top
        wsum = float(area[sel].sum())
        mean = (rgb[sel] * area[sel, None]).sum(axis=0) / wsum if wsum > 0 else rgb[sel].mean(axis=0)
        elements[k]["colour"] = tuple(float(x) for x in np.clip(mean / 255.0, 0.0, 1.0))


def object_colours_from_cloud(output_dir: Path | str) -> Dict[int, Tuple[float, float, float]]:
    """Each object's DOMINANT colour from its own cloud points (their photographs' colours): the most common colour at
    5 bits a channel (an exposure's dark or bright corner only when it is most of the object: _dominant_bin), then the
    mean of the points in it — for a mesh written simplified, whose triangles carry no texture (the IFC the viewer
    loads light: USER 2026-11-18 "de momento dejemos el liviano")."""
    from exporter import cloud as C
    out = Path(output_dir)
    cloud = C.open_ply(out / "cleaned_cloud.ply")
    ids = np.load(out / "instance_ids.npy").astype(np.int64)
    rgb = np.column_stack([np.asarray(cloud[c], np.int64) for c in ("red", "green", "blue")])
    key = ids << 15 | (rgb[:, 0] >> 3) << 10 | (rgb[:, 1] >> 3) << 5 | (rgb[:, 2] >> 3)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(inv.ravel(), weights=rgb[:, c], minlength=len(uniq)) for c in range(3)], axis=1)
    obj = uniq >> 15
    corner = np.isin(uniq & ((1 << 15) - 1), _CLIPPED_BINS)
    objects, at = np.unique(obj, return_inverse=True)
    total = np.bincount(at, weights=counts)
    clipped = np.bincount(at, weights=np.where(corner, counts, 0))
    vote = ~corner | (clipped >= total / 2)[at]                   # the corners vote only where they are most of the object
    cand = np.flatnonzero(vote)
    order = cand[np.lexsort((counts[cand], obj[cand]))]           # by object, then by count: each object's last = its mode
    last = order[np.r_[np.flatnonzero(np.diff(obj[order])), len(order) - 1]]
    return {int(obj[k]): tuple(float(x) for x in np.clip(sums[k] / counts[k] / 255.0, 0.0, 1.0)) for k in last}


def colour_elements(elements: List[Dict[str, Any]], texture: Optional[Dict[str, Any]], output_dir: Path | str) -> str:
    """Each element's dominant colour: from the texture when its triangles carry one (the mesh written whole), else
    from its object's own cloud points. Returns where the colours came from."""
    element_colours(elements, texture)
    missing = [e for e in elements if "colour" not in e]
    if not missing:
        return "texture"
    by_object = object_colours_from_cloud(output_dir)
    for e in missing:
        if e["id"] in by_object:
            e["colour"] = by_object[e["id"]]
    return "cloud" if len(missing) == len(elements) else "texture, the rest from the cloud"


def write_textures(elements: List[Dict[str, Any]], texture: Optional[Dict[str, Any]], folder: Path) -> Dict[int, str]:
    """The base-colour images the elements use, copied out of the GLB AS THEY ARE (no re-encoding) into ``folder``;
    returns each material's URL as the IFC beside the folder refers to it (``<folder name>/m<material>.<ext>``)."""
    if not texture:
        return {}
    used = sorted({int(m) for e in elements if "uv" in e for m in np.unique(e["material"])})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    urls: Dict[int, str] = {}
    with open(texture["glb"], "rb") as fh:
        for m in used:
            info = texture["images"].get(m)
            if not info:
                continue
            ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(info["mime"], "png")
            fh.seek(info["offset"])
            (folder / f"m{m}.{ext}").write_bytes(fh.read(info["length"]))
            urls[m] = f"{folder.name}/m{m}.{ext}"
    return urls


def build(elements: List[Dict[str, Any]], *, session: str, scan_key: str, geometry_epoch: str, described: str,
          textures: Optional[Dict[int, str]] = None):
    """The IFC4 model of ``elements`` (each a triangulated face set)."""
    import ifcopenshell
    f = ifcopenshell.file(schema="IFC4")
    f.header.file_name.name = f"{session}_{scan_key.replace('/', '_')}_from_mesh.ifc"
    f.header.file_name.time_stamp = ""                         # no clock in the artifact
    key = (session, scan_key, "mesh")
    origin = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, 0.0)), None, None)
    context = f.createIfcGeometricRepresentationContext(None, "Model", 3, 1e-5, origin, None)
    body = f.createIfcGeometricRepresentationSubContext("Body", "Model", None, None, None, None, context, None,
                                                        "MODEL_VIEW", None)
    units = f.createIfcUnitAssignment([
        f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE"), f.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE"),
        f.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE"), f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")])
    project = f.createIfcProject(_guid(*key, "project"), None, f"{session} {scan_key} (from the mesh)", described,
                                 None, None, None, [context], units)
    place = lambda rel=None: f.createIfcLocalPlacement(rel, origin)   # noqa: E731
    site = f.createIfcSite(_guid(*key, "site"), None, "Site", None, None, place(), None, None, "ELEMENT",
                           None, None, None, None, None)
    building = f.createIfcBuilding(_guid(*key, "building"), None, session, None, None,
                                   place(site.ObjectPlacement), None, None, "ELEMENT", None, None, None)
    storey = f.createIfcBuildingStorey(_guid(*key, "storey"), None, "Scanned level", None, None,
                                       place(building.ObjectPlacement), None, None, "ELEMENT", 0.0)
    f.createIfcRelAggregates(_guid(*key, "agg", "project"), None, None, None, project, [site])
    f.createIfcRelAggregates(_guid(*key, "agg", "site"), None, None, None, site, [building])
    f.createIfcRelAggregates(_guid(*key, "agg", "building"), None, None, None, building, [storey])
    images: Dict[int, Any] = {}

    def image(m: int):                                           # one IfcImageTexture per atlas image
        if m not in images:
            images[m] = f.create_entity("IfcImageTexture", RepeatS=False, RepeatT=False, URLReference=textures[m])
        return images[m]

    def styled(item, label: str, colour, texture_entity) -> None:
        """the face set's style: its texture over its element's dominant colour (a program without textures shows the colour)"""
        layers = []
        if colour is not None:
            layers.append(f.createIfcSurfaceStyleShading(f.createIfcColourRgb(None, *[round(c, 4) for c in colour]), 0.0))
        if texture_entity is not None:
            layers.append(f.createIfcSurfaceStyleWithTextures([texture_entity]))
        if layers:
            f.createIfcStyledItem(item, [f.createIfcSurfaceStyle(label, "BOTH", layers)], None)

    products = []
    for e in elements:
        items = []
        colour = e.get("colour")
        if textures and "uv" in e:
            # one face set per atlas image, each with its texture map (IFC's t is up from the image's bottom)
            for m in np.unique(e["material"]):
                sel = e["material"] == m
                gv, gf, kept_g = weld(np.asarray(e["vertices"]), np.asarray(e["faces"])[sel])
                if not len(gf):
                    continue
                guv = e["uv"][sel][kept_g]
                xyz = np.round(np.asarray(gv, np.float64) @ TO_IFC.T, _DECIMALS) + 0.0
                mesh = f.createIfcTriangulatedFaceSet(f.createIfcCartesianPointList3D(xyz.tolist()), None, None,
                                                      (gf + 1).tolist(), None)
                st = np.round(np.stack([guv[..., 0], 1.0 - guv[..., 1]], axis=-1).reshape(-1, 2).astype(np.float64), 6) + 0.0
                uniq, inv = np.unique(st, axis=0, return_inverse=True)
                tex = image(int(m)) if int(m) in textures else None
                if tex is not None:
                    f.createIfcIndexedTriangleTextureMap([tex], mesh, f.createIfcTextureVertexList(uniq.tolist()),
                                                         (inv.reshape(-1, 3) + 1).tolist())
                styled(mesh, e["label"], colour, tex)
                items.append(mesh)
        if not items:
            xyz = np.round(np.asarray(e["vertices"], np.float64) @ TO_IFC.T, _DECIMALS) + 0.0     # (+ 0.0: no "-0.")
            coords = f.createIfcCartesianPointList3D(xyz.tolist())
            mesh = f.createIfcTriangulatedFaceSet(coords, None, None, (np.asarray(e["faces"]) + 1).tolist(), None)
            styled(mesh, e["label"], colour, None)
            items = [mesh]
        shape = f.createIfcProductDefinitionShape(None, None, [
            f.createIfcShapeRepresentation(body, "Body", "Tessellation", items)])
        cls, predefined = e["ifc"]
        product = f.create_entity(cls, GlobalId=_guid(*key, "element", e["id"]), Name=f"{e['label']}:{e['label']}:{e['id']}",
                                  ObjectType=e["label"], ObjectPlacement=place(storey.ObjectPlacement),
                                  Representation=shape, Tag=str(e["id"]))
        if predefined:
            product.PredefinedType = predefined
        props = [
            f.createIfcPropertySingleValue("InstanceId", None, f.createIfcInteger(e["id"]), None),
            f.createIfcPropertySingleValue("Label", None, f.createIfcLabel(e["label"]), None),
            f.createIfcPropertySingleValue("Points", None, f.createIfcInteger(e["n_points"]), None),
            f.createIfcPropertySingleValue("Triangles", None, f.createIfcInteger(int(len(e["faces"]))), None),
            f.createIfcPropertySingleValue("Scan", None, f.createIfcLabel(f"{session} {scan_key}"), None),
            f.createIfcPropertySingleValue("GeometryEpoch", None, f.createIfcLabel(str(geometry_epoch)), None),
            f.createIfcPropertySingleValue("Provenance", None, f.createIfcLabel(
                "shape: vlm_generated (GenRecon mesh); object: the segmentation; label: vlm_proposed"), None),
        ]
        pset = f.createIfcPropertySet(_guid(*key, "pset", e["id"]), None, "STAC_FromMesh", None, props)
        f.createIfcRelDefinesByProperties(_guid(*key, "relpset", e["id"]), None, None, None, [product], pset)
        products.append(product)
    if products:
        f.createIfcRelContainedInSpatialStructure(_guid(*key, "contained"), None, None, None, products, storey)
    return f


def voxel_of(config_cube_m: Optional[float] = None) -> float:
    """One GenRecon voxel (the cube / 32): how far from the cloud a triangle can be and still be
    the cloud's (config.yaml genrecon.cube_size_m — the user's)."""
    if config_cube_m is None:
        from reconstruction.grid_knn import config_section
        config_cube_m = float(config_section(("genrecon",), ("cube_size_m",))["cube_size_m"])
    return float(config_cube_m) / 32.0


def mesh_of(output_dir: Path | str) -> Path:
    """The scan's published GenRecon mesh (the binary glTF its meta names)."""
    folder = Path(output_dir) / "tsdf" / "genrecon_photo"
    meta = folder / "genrecon_photo.meta.json"
    name = json.loads(meta.read_text()).get("glb_file") if meta.exists() else None
    glb = folder / str(name or "genrecon_photo.glb")
    if not glb.exists():
        raise MeshError(f"{output_dir}: no published GenRecon mesh ({glb})")
    return glb


def write(session: str, scan_key: str, *, keep: float, neighbours: int = 0, max_edge_voxels: float = 0.0,
          out: Path | str | None = None, measure: bool = True, with_textures: bool = True,
          log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Build the scan's IFC from its GenRecon mesh and write it (default: the project's IFC
    directory). Returns the report: elements, triangles, the simplification's measured deviation."""
    from project_paths import resolve_session
    server_dir = Path(__file__).resolve().parent.parent
    ctx = resolve_session(str(server_dir), session, scan_key)
    output_dir = Path(ctx.output_dir)
    con = sqlite3.connect(f"file:{output_dir / 'scene_r.db'}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT value FROM scene_meta WHERE key = 'geometry_epoch'").fetchone()
    finally:
        con.close()
    epoch = str(row[0]) if row else "?"
    elements, report = elements_of(output_dir, mesh_of(output_dir), keep=keep, max_distance=voxel_of(),
                                   neighbours=neighbours, max_edge=max_edge_voxels * voxel_of() or None,
                                   measure=measure, log=log)
    out = Path(out) if out is not None else Path(ctx.ifcs_dir) / f"{session}_{scan_key.replace('/', '_')}_from_mesh.ifc"
    report["colours_from"] = colour_elements(elements, report.get("texture"), output_dir)
    # the IFC the viewer loads carries the colours alone (its web-ifc reads no IFC texture); the export's carries both
    urls = write_textures(elements, report.get("texture"), out.parent / f"{out.stem}_textures") if with_textures else {}
    dev = report.get("deviation_m")
    described = (f"STAC Build — built from the GenRecon mesh of the scan {session} {scan_key}, geometry epoch {epoch}, "
                 f"cut by its segmentation: each element is its object's part of the mesh. Triangles kept: "
                 f"{report['faces']:,} of {report['mesh_faces']:,}"
                 + (f"; distance from the full mesh: median {dev['median'] * 1000:.2f} mm, 95 % under {dev['p95'] * 1000:.2f} mm"
                    if dev else ""))
    model = build(elements, session=session, scan_key=scan_key, geometry_epoch=epoch, described=described, textures=urls)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    model.write(str(tmp))
    tmp.replace(out)
    by_class: Dict[str, int] = {}
    for e in elements:
        by_class[e["ifc"][0]] = by_class.get(e["ifc"][0], 0) + 1
    report.update(path=str(out), bytes=out.stat().st_size, by_class=by_class, textures=len(urls),
                  coloured=sum(1 for e in elements if "colour" in e))
    return report


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="bim.ifc_from_mesh")
    ap.add_argument("session")
    ap.add_argument("scan")
    ap.add_argument("--keep", type=float, required=True, help="the share of the mesh's triangles kept, (0, 1]")
    ap.add_argument("--neighbours", type=int, default=0,
                    help="a triangle's object: the majority of this many nearest cloud points (0: of the points on it)")
    ap.add_argument("--max-edge-voxels", type=float, default=0.0,
                    help="split the triangles whose longest edge is over this many GenRecon voxels (0: none)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-textures", action="store_true",
                    help="each element's dominant colour only: no texture maps, no images beside the IFC (the viewer's)")
    a = ap.parse_args(argv)
    r = write(a.session, a.scan, keep=a.keep, neighbours=a.neighbours, max_edge_voxels=a.max_edge_voxels, out=a.out,
              with_textures=not a.no_textures)
    r.pop("texture", None)                                  # (where the images lie in the GLB: not for the console)
    print(json.dumps(r, default=float))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
