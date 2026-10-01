"""Every face of a generated OBJECT points out of it (USER 2026-10-01: "los objetos deben tener siempre
las normales hacia fuera desde todas sus caras").

RAY PARITY: a face is outward when a ray leaving it along its normal crosses the object's surface an EVEN
number of times (it ends outside), inward when ODD. ShapeR's meshes and their textured GLBs (one
sub-mesh per atlas page + the unseen faces) are not watertight, so winding propagation and the signed
volume fail on them (pccr: half the faces stayed wrong); a voxel fill leaks through their holes. A ray
only errs when it slips through a hole, so each face votes with RAYS rays around its normal. Open3D's
RaycastingScene (Embree) counts the crossings.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh

RAYS = 5                 # rays per face, the normal and 4 tilted around it — the majority decides
TILT = 0.25              # tangent of the tilt of the 4 side rays (≈ 14°)
EPS_REL = 1e-4           # the ray starts this fraction of the object's size off the face


def inward_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Boolean mask of the faces whose normal points INTO the object."""
    import open3d as o3d
    if not len(mesh.faces):
        return np.zeros(0, bool)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    n = np.asarray(mesh.face_normals, np.float64)
    c = np.asarray(mesh.triangles_center, np.float64)
    eps = EPS_REL * float(mesh.extents.max())
    # two tangents per face for the tilted rays
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    dirs = [n, n + TILT * t1, n - TILT * t1, n + TILT * t2, n - TILT * t2][:RAYS]
    odd = np.zeros(len(n), np.int32)
    for d in dirs:
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        rays = np.hstack([c + n * eps, d]).astype(np.float32)
        cnt = scene.count_intersections(o3d.core.Tensor(rays)).numpy()
        odd += (cnt % 2 == 1)
    return odd * 2 > len(dirs)


def orient_mesh(mesh: trimesh.Trimesh) -> int:
    """Flip, in place, the faces of `mesh` that point inward; returns how many."""
    fl = inward_faces(mesh)
    if fl.any():
        f = mesh.faces.copy(); f[fl] = f[fl][:, ::-1]; mesh.faces = f
    return int(fl.sum())


def orient_glb(glb: Path) -> Tuple[int, int]:
    """Orient a (textured, multi-sub-mesh) object GLB outward as ONE object, UVs untouched, rewritten in
    place. Returns (faces flipped, faces)."""
    sc = trimesh.load(str(glb), force="scene")
    names = list(sc.geometry.keys())
    geoms = [sc.geometry[n] for n in names]
    world = []
    for n, g in zip(names, geoms):
        nodes = sc.graph.geometry_nodes.get(n, [])
        T = sc.graph.get(nodes[0])[0] if nodes else np.eye(4)
        world.append(trimesh.transformations.transform_points(g.vertices, T))
    offs = np.cumsum([0] + [len(g.vertices) for g in geoms])
    whole = trimesh.Trimesh(vertices=np.vstack(world),
                            faces=np.vstack([g.faces + o for g, o in zip(geoms, offs[:-1])]), process=False)
    flip = inward_faces(whole)
    start = 0
    for g in geoms:
        k = len(g.faces); fl = flip[start:start + k]
        if fl.any():
            f = g.faces.copy(); f[fl] = f[fl][:, ::-1]; g.faces = f
        start += k
    if flip.any():
        sc.export(str(glb))
    return int(flip.sum()), len(flip)
