"""After Omega, on threads and without rewriting what is deleted next (station 2026-11-01): the
omega records written on threads are the one-thread records array for array; the scale and the
orientation with ``aligned_chunks=False`` transform the chunk PLYs exactly as with True and leave
the aligned chunks untouched (the caller deletes them), and their markers name what they wrote;
an applied marker hashed on threads reads current, then stale when a recorded file changes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers import map_worker as MW                                  # noqa: E402
from reconstruction import applied_marker as AM                       # noqa: E402
from reconstruction import orient as OR                               # noqa: E402
from reconstruction import scale_align as SA                          # noqa: E402


class _Pipe:
    def send_log(self, *a, **k):
        pass


def _omega_case(root, N=23, chunks=((0, 12), (8, 23)), H=6, W=7, seed=0):
    rng = np.random.default_rng(seed)
    out, save = root / "output", root / "output" / "maplong_run"
    (save / "_tmp_results_aligned").mkdir(parents=True)
    frames = [3 * f + 1 for f in range(N)]
    (root / "sel.json").write_text(json.dumps({"selected_files": [f"{f:06d}.jpg" for f in frames]}))
    poses = []
    for k, (a, b) in enumerate(chunks):
        S = b - a
        c2w = np.tile(np.eye(4), (S, 1, 1))
        wp = rng.normal(0, 1, (S, H, W, 3)); wp[..., 2] += 5.0
        np.save(save / "_tmp_results_aligned" / f"chunk_{k}.npy",
                {"world_points": wp[None].astype(np.float32),
                 "world_points_conf": rng.uniform(0, 9, (1, S, H, W)).astype(np.float32),
                 "extrinsic": c2w[None], "intrinsic": np.tile(np.eye(3), (1, S, 1, 1))}, allow_pickle=True)
    for g in range(N):
        T = np.eye(4); T[:3, 3] = rng.normal(0, 1, 3)
        T[:3, :3] = np.linalg.qr(rng.normal(0, 1, (3, 3)))[0]
        poses.append(T)
    (out / "camera_poses.txt").write_text("\n".join(" ".join(repr(float(v)) for v in m.ravel()) for m in poses) + "\n")
    (out / "camera_frames.txt").write_text(" ".join(str(f) for f in frames))
    return out, save, frames, chunks


def test_omega_records_on_threads_are_the_one_thread_records(tmp_path, monkeypatch):
    recs = {}
    for threads in (1, 8):
        root = tmp_path / f"t{threads}"
        out, save, frames, chunks = _omega_case(root)
        monkeypatch.setattr(MW, "OMEGA_RECORD_THREADS", threads)
        MW._emit_omega_depth(save, out, list(chunks), str(root / "sel.json"), _Pipe())
        recs[threads] = {}
        for f in frames:
            with np.load(out / "omega_run" / "results_output" / f"frame_{f}.npz") as z:
                recs[threads][f] = {k: z[k] for k in z.files}
    assert recs[1].keys() == recs[8].keys() and len(recs[1]) == 23
    for f in recs[1]:
        assert recs[1][f].keys() == recs[8][f].keys()
        for k in recs[1][f]:
            assert recs[1][f][k].dtype == recs[8][f][k].dtype and np.array_equal(recs[1][f][k], recs[8][f][k])


def _ply(path, xyz, rgb):
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
           "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = xyz.T.astype(np.float32)
    a["red"], a["green"], a["blue"] = rgb.T
    path.write_bytes(hdr.encode() + a.tobytes())


def _cloud_case(root, seed=1):
    rng = np.random.default_rng(seed)
    out = root / "output"
    (out / "maplong_run" / "_tmp_results_aligned").mkdir(parents=True)
    for k in range(5):
        xyz = rng.normal(0, 2, (3000 + 500 * k, 3)); xyz[:, 1] -= 1.5
        _ply(out / f"chunk_{k:03d}.ply", xyz, rng.integers(0, 255, (len(xyz), 3)).astype(np.uint8))
        np.save(out / "maplong_run" / "_tmp_results_aligned" / f"chunk_{k}.npy",
                {"world_points": rng.normal(0, 1, (2, 4, 5, 3)).astype(np.float32),
                 "depth": rng.uniform(1, 3, (2, 4, 5)).astype(np.float32)}, allow_pickle=True)
    # cameras looking along +z with their down along a tilted −y (the gravity orient estimates)
    R = np.linalg.qr(np.eye(3) + 0.1 * rng.normal(0, 1, (3, 3)))[0]
    lines = []
    for i in range(30):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0.1 * i, 0.0, 0.05 * i]
        lines.append(" ".join(repr(float(v)) for v in T.ravel()))
    (out / "camera_poses.txt").write_text("\n".join(lines) + "\n")
    return out


@pytest.mark.parametrize("step", ["scale", "orient"])
def test_scale_and_orient_without_the_deleted_chunks(tmp_path, step):
    outs = {}
    for flag in (True, False):
        out = _cloud_case(tmp_path / f"{step}_{flag}")
        before = {p.name: p.read_bytes() for p in (out / "maplong_run" / "_tmp_results_aligned").glob("*.npy")}
        if step == "scale":
            SA.apply_scale(out, 1.0731, aligned_chunks=flag)
            files = SA.transformed_files(out, flag)
        else:
            T = OR.run(out, log=lambda m: None, aligned_chunks=flag)
            assert T is not None
            files = OR.transformed_files(out, flag)
        after = {p.name: p.read_bytes() for p in (out / "maplong_run" / "_tmp_results_aligned").glob("*.npy")}
        assert (after == before) == (not flag)
        assert any(f.suffix == ".npy" for f in files) == flag
        outs[flag] = {p.name: p.read_bytes() for p in out.glob("chunk_*.ply")}
    assert outs[True] == outs[False] and len(outs[True]) == 5


def test_an_applied_marker_hashed_on_threads(tmp_path):
    out = _cloud_case(tmp_path)
    files = sorted(out.glob("chunk_*.ply"))
    m = out / ".marker"
    AM.record(m, "s=1", files, base=out)
    status, info = AM.state(m, base=out)
    assert status == "current" and info["n_recorded"] == 5
    files[2].unlink()                                          # gone: not a difference
    assert AM.state(m, base=out)[0] == "current"
    files[3].write_bytes(files[3].read_bytes()[:-3] + b"xyz")  # changed: stale, named
    status, info = AM.state(m, base=out)
    assert status == "stale" and info["changed"] == [files[3].name] and info["gone"] == [files[2].name]
