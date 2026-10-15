"""The per-object TSDF integrates the PUBLISHED cloud of the live epoch (USER 2026-10-15: "todo tiene
que correr sobre todo lo corregido") — never the original depth maps — on one declared device, and
the same voxels give the same bytes twice (canonical mesh order)."""
import json

import numpy as np
import open3d as o3d
import pytest
import trimesh

from segmentation import tsdf_export as T
from test_hole_audit_torch import _session


def _run(tmp_path, out, device, monkeypatch):
    seg = {"instances": [{"instance_id": 1, "label": "plate",
                          "globalIndices": list(range(5000, 6200))}]}
    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)
    for name in ("_resolve_stray_depth", "_resolve_pgsr_render_depth",
                 "_resolve_mapanything_depth", "_resolve_da3_depth", "_load_camera_source"):
        monkeypatch.setattr(T, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError(name)))
    glbs = T.export_tsdf_meshes(out, frames, seg, session_dir=tmp_path, obj_ids=[1],
                                voxel_length=0.02, sdf_trunc=0.06, depth_trunc=5.0, depth_min=0.1,
                                depth_source="live_cloud", device=device, live_fill_px=1,
                                weight_threshold=2.0, name_suffix="_tsdf")
    assert len(glbs) == 1
    return glbs[0]


def test_the_live_cloud_is_the_depth_and_the_mesh_lies_on_it(tmp_path, monkeypatch):
    out, xyz, plate = _session(tmp_path)
    glb = _run(tmp_path, out, "cpu", monkeypatch)
    mesh = trimesh.load(str(glb), force="mesh")      # Open3D's own GLB reader rejects its writer's file
    V = np.asarray(mesh.vertices, float)
    assert len(V) > 100 and len(mesh.faces) > 100
    assert np.abs(V[:, 2] - 1.5).max() <= 0.03, "the plate at z = 1.5 m, within a voxel"
    assert V[:, 0].min() < -0.22 and V[:, 0].max() > 0.22 and V[:, 1].min() < -0.12 \
        and V[:, 1].max() > 0.12, "the plate's extent (±0.3 × ±0.2 m)"
    meta = json.loads(glb.with_suffix(".meta.json").read_text())
    assert meta["depth_source"] == "live_cloud" and meta["device"] == "cpu"
    assert meta["n_frames_integrated"] == 3 and "geometry_epoch" in meta
    b1 = glb.read_bytes()
    glb2 = _run(tmp_path, out, "cpu", monkeypatch)
    assert glb2.read_bytes() == b1, "the same bytes twice"


def test_the_device_and_the_keys_are_declared(monkeypatch):
    from config import cfg
    from run_tsdf_objects import object_tsdf_config
    oc = object_tsdf_config(cfg)
    assert oc["depth_source"] == "live_cloud" and oc["device"] in ("cuda", "cpu")
    with pytest.raises(RuntimeError, match="tsdf.object is missing"):
        object_tsdf_config({"tsdf": {"object": {"device": "cuda"}}})
    with pytest.raises(RuntimeError, match="tsdf.object.device"):
        T._live_device("tpu")


def test_canonical_order_is_one_order():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], float)   # a duplicate vertex
    F = np.array([[2, 0, 1], [0, 3, 2], [1, 1, 2]])                      # rotated, dup, degenerate
    m = T.canonical_mesh(o3d, V, F)
    assert len(m.vertices) == 3 and len(m.triangles) == 1
    assert np.array_equal(np.asarray(m.triangles)[0], [0, 2, 1])   # [0,0,0] [1,0,0] [0,1,0], orientation kept


@pytest.mark.skipif(not (o3d.core.cuda.is_available()), reason="no CUDA Open3D")
def test_on_cuda_the_mesh_equals_the_cpu_one(tmp_path, monkeypatch):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA torch")
    out, xyz, plate = _session(tmp_path)
    g_cpu = _run(tmp_path, out, "cpu", monkeypatch)
    v_cpu = np.asarray(trimesh.load(str(g_cpu), force="mesh").vertices, float)
    g_gpu = _run(tmp_path, out, "cuda", monkeypatch)
    v_gpu = np.asarray(trimesh.load(str(g_gpu), force="mesh").vertices, float)
    assert v_cpu.shape == v_gpu.shape and np.allclose(v_cpu, v_gpu, atol=1e-5)
