"""Generated objects carry every face pointing OUT (USER 2026-10-01), even when they are not watertight
and are split into sub-meshes (a textured GLB)."""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction.orient_outward import inward_faces, orient_glb, orient_mesh  # noqa: E402


def _scrambled_sphere(seed=0):
    m = trimesh.creation.icosphere(subdivisions=5, radius=0.3)
    m.update_faces(np.arange(len(m.faces))[np.random.default_rng(seed).random(len(m.faces)) > 0.01])  # holes
    f = m.faces.copy(); flip = np.random.default_rng(seed + 1).random(len(f)) < 0.5
    f[flip] = f[flip][:, ::-1]; m.faces = f
    return m


def test_an_open_scrambled_object_ends_with_every_face_outward():
    m = _scrambled_sphere()
    assert inward_faces(m).mean() > 0.3
    orient_mesh(m)
    assert inward_faces(m).mean() < 0.001
    out = np.einsum("ij,ij->i", m.face_normals, m.triangles_center - m.centroid) > 0
    assert out.mean() >= 0.999, (~out).sum()        # a ray may slip through a hole: ≥ 99.9 %


def test_a_glb_split_in_sub_meshes_is_oriented_as_one_object(tmp_path):
    m = _scrambled_sphere(3)
    half = len(m.faces) // 2
    a = trimesh.Trimesh(m.vertices, m.faces[:half], process=False)
    b = trimesh.Trimesh(m.vertices, m.faces[half:], process=False)
    p = tmp_path / "obj.glb"
    trimesh.Scene({"a": a, "b": b}).export(str(p))
    n, tot = orient_glb(p)
    assert n > 0 and tot == len(m.faces)
    sc = trimesh.load(str(p), force="scene")
    whole = trimesh.util.concatenate(list(sc.geometry.values()))
    out = np.einsum("ij,ij->i", whole.face_normals, whole.triangles_center - whole.centroid) > 0
    assert out.mean() >= 0.999, (~out).sum()        # a ray may slip through a hole: ≥ 99.9 %
