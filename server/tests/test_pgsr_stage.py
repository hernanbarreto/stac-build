"""Phase D (precision task) — PGSR stage plumbing, synthetic tests.

Covers: COLMAP scene export round-trip (poses/intrinsics/seed cloud), the
se(3)/quaternion math the photometric pose refinement rests on, dynamic-class
mask generation from SAM3 artifacts, and the pgsr_render TSDF depth resolver.
The training itself is exercised on a real session (integration A/B), not here."""
import json
from pathlib import Path

import numpy as np
import pytest

from reconstruction import pgsr_export, dynamic_masks
from reconstruction.pgsr_train import se3_exp, quat_mul


# ── synthetic session ─────────────────────────────────────────────────────────

Hd, Wd = 32, 24          # omega grid
Wn, Hn = 96, 128         # native frame


def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _build(tmp: Path, n=4):
    out = tmp / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    frames = tmp / "frames"
    frames.mkdir()
    from PIL import Image
    lines, nums, intr = [], [], []
    for i in range(n):
        num = 5 * i
        nums.append(num)
        T = np.eye(4)
        T[:3, :3] = _rot("z", 10.0 * i)
        T[:3, 3] = [0.3 * i, 0.1, 0.2]
        lines.append(" ".join(f"{x:.9g}" for x in T.reshape(-1)))
        intr.append(f"{50.0 + i} {51.0 + i} {Wd / 2} {Hd / 2}")
        Image.new("RGB", (Wn, Hn), (i * 30, 80, 120)).save(frames / f"{num:06d}.jpg")
        np.savez(out / "omega_run" / "results_output" / f"frame_{num}.npz",
                 depth=np.full((Hd, Wd), 2.0, np.float32))
    (out / "camera_poses.txt").write_text("\n".join(lines) + "\n")
    (out / "camera_frames.txt").write_text(" ".join(map(str, nums)) + "\n")
    (out / "intrinsic.txt").write_text("\n".join(intr) + "\n")
    # tiny binary cleaned cloud (xyz + rgb)
    pts = np.zeros(500, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
    pts["x"] = np.linspace(0, 5, 500)
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex 500\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    (out / "cleaned_cloud.ply").write_bytes(hdr.encode() + pts.tobytes())
    return out, frames, nums, lines


def _quat_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_export_scene_roundtrip(tmp_path):
    out, frames, nums, lines = _build(tmp_path)
    scene = pgsr_export.export_scene(out, frames, max_seed_pts=200)
    cams = [l for l in (scene / "sparse" / "cameras.txt").read_text().splitlines()
            if not l.startswith("#")]
    imgs = [l for l in (scene / "sparse" / "images.txt").read_text().splitlines()
            if l and not l.startswith("#")]
    assert len(cams) == 4 and len(imgs) == 4          # (POINTS2D lines are empty)
    # intrinsics scaled omega grid → native
    fx = float(cams[0].split()[4])
    assert fx == pytest.approx(50.0 * Wn / Wd)
    # w2c quaternion/translation must invert back to the input c2w
    t = imgs[0].split()
    q = np.array([float(x) for x in t[1:5]])
    tw2c = np.array([float(x) for x in t[5:8]])
    Rw2c = _quat_R(q)
    C = -Rw2c.T @ tw2c
    T_in = np.array([float(x) for x in lines[0].split()]).reshape(4, 4)
    assert np.allclose(Rw2c.T, T_in[:3, :3], atol=1e-6)
    assert np.allclose(C, T_in[:3, 3], atol=1e-6)
    # seed ply exists with the requested subsample
    assert (scene / "sparse" / "points3D.ply").stat().st_size > 0
    meta = json.loads((scene / "scene_meta.json").read_text())
    assert meta["n_images"] == 4 and meta["native_wh"] == [Wn, Hn]


def test_export_fails_without_cloud(tmp_path):
    out, frames, _, _ = _build(tmp_path)
    (out / "cleaned_cloud.ply").unlink()
    with pytest.raises(RuntimeError, match="cleaned_cloud"):
        pgsr_export.export_scene(out, frames)


def test_se3_and_quat_math():
    import torch
    delta = torch.tensor([[0.0, 0.0, np.deg2rad(30.0), 0.1, -0.2, 0.3]])
    R, t = se3_exp(delta)
    Rz = _rot("z", 30.0)
    assert np.allclose(R[0].numpy(), Rz, atol=1e-6)
    assert np.allclose(t[0].numpy(), [0.1, -0.2, 0.3])
    # quaternion product matches rotation composition
    qz = torch.tensor([np.cos(np.deg2rad(15)), 0, 0, np.sin(np.deg2rad(15))],
                      dtype=torch.float64)
    qc = quat_mul(qz[None], qz[None])[0].numpy()
    assert np.allclose(_quat_R(qc), _rot("z", 60.0), atol=1e-9)
    # identity delta → identity transform (pose refinement starts as a no-op)
    R0, t0 = se3_exp(torch.zeros(1, 6))
    assert np.allclose(R0[0].numpy(), np.eye(3), atol=1e-7) and t0.abs().max() == 0


def test_dynamic_masks_union(tmp_path):
    out = tmp_path
    m_person = np.zeros((Hd, Wd), np.uint8)
    m_person[2:6, 3:9] = 1
    m_truck = np.zeros((Hd, Wd), np.uint8)
    m_truck[10:14, 3:9] = 1
    m_wall = np.ones((Hd, Wd), np.uint8)
    np.savez(out / "seg_masks.npz", f5_o0=m_person, f5_o1=m_wall, f5_o2=m_truck,
             f9_o1=m_wall)
    (out / "segmentation.json").write_text(json.dumps({"instances": [
        {"id": 0, "label": "person"}, {"id": 1, "label": "concrete wall"},
        {"id": 2, "label": "truck"}]}))
    n = dynamic_masks.generate(out, out / "masks",
                               _labels_override={"person", "truck"})
    assert n == 1                                       # frame 9 has only static
    from PIL import Image
    m = np.asarray(Image.open(out / "masks" / "000005.jpg.png"))
    assert m[3, 4] == 255 and m[11, 4] == 255           # union of both dynamics
    assert m[20, 20] == 0                               # wall never masked


def test_dynamic_masks_absent_is_explicit_noop(tmp_path):
    msgs = []
    n = dynamic_masks.generate(tmp_path, tmp_path / "masks", log=msgs.append)
    assert n == 0 and any("unmasked" in m for m in msgs)


def test_pgsr_render_resolver(tmp_path):
    out, frames, nums, _ = _build(tmp_path)
    rd = out / "pgsr_render"
    rd.mkdir()
    for n in nums:
        np.savez(rd / f"frame_{n}.npz",
                 depth=np.full((Hn, Wn), 3.0, np.float32),
                 valid=np.ones((Hn, Wn), bool))
    from segmentation.tsdf_export import _resolve_pgsr_render_depth
    src = _resolve_pgsr_render_depth(out)
    assert src is not None
    loader, (h, w) = src
    assert (h, w) == (Hn, Wn)
    d = loader(nums[1])
    assert d["depth"].shape == (Hn, Wn) and d["valid"].all()
    assert d["K"][0, 0] == pytest.approx(51.0 * Wn / Wd)   # scaled per-frame K
    assert loader(99999) is None
