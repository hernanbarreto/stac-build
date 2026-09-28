"""Every keyframe's omega record (claude_stac.txt §4-F3): depth, conf, the grid K
and the aligned c2w, from the aligned chunk arrays."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.map_worker import _emit_omega_depth                 # noqa: E402


class _Pipe:
    def send_log(self, *a, **k):
        pass


def test_every_keyframe_gets_depth_conf_k_and_pose(tmp_path):
    out, save = tmp_path / "output", tmp_path / "output" / "maplong_run"
    (save / "_tmp_results_aligned").mkdir(parents=True)
    frames = [3, 9, 15]
    sel = tmp_path / "selected_frames.json"
    sel.write_text(json.dumps({"selected_files": [f"{f:06d}.jpg" for f in frames]}))
    H, W, S = 4, 5, 3
    c2w = np.tile(np.eye(4), (S, 1, 1))
    c2w[:, 0, 3] = [0.0, 1.0, 2.0]
    z = np.array([2.0, 3.0, 4.0])
    wp = np.zeros((S, H, W, 3))
    for j in range(S):
        wp[j, ..., 0] = c2w[j, 0, 3]
        wp[j, ..., 2] = z[j]                                # a plane z metres ahead
    K = np.tile(np.array([[10.0, 0, 2], [0, 10.0, 1.5], [0, 0, 1]]), (S, 1, 1))
    np.save(save / "_tmp_results_aligned" / "chunk_0.npy",
            {"world_points": wp[None], "world_points_conf": np.full((1, S, H, W), 5.0),
             "extrinsic": c2w[None], "intrinsic": K[None]}, allow_pickle=True)
    (out / "camera_poses.txt").write_text(
        "\n".join(" ".join(f"{v:.8g}" for v in m.ravel()) for m in c2w) + "\n")
    (out / "camera_frames.txt").write_text(" ".join(str(f) for f in frames))
    _emit_omega_depth(save, out, S, 0, str(sel), _Pipe())
    for j, f in enumerate(frames):
        with np.load(out / "omega_run" / "results_output" / f"frame_{f}.npz") as r:
            assert set(r.files) == {"depth", "conf", "K_omega", "pose_c2w", "chunk",
                                    "frame_global"}
            assert int(r["chunk"]) == 0 and int(r["frame_global"]) == j
            assert np.allclose(r["depth"], z[j]) and np.allclose(r["conf"], 5.0)
            assert np.allclose(r["K_omega"], K[j]) and np.allclose(r["pose_c2w"], c2w[j])


def test_a_shared_frame_is_recorded_from_its_owner_chunk_in_numeric_order(tmp_path):
    """12 keyframes in chunks of 4 with overlap 2 (5 chunks — chunk_10-style names
    sort wrong lexicographically only past 10, so the owner rule is what is tested
    here): every record carries the chunk whose centre is nearest, and its depth."""
    out, save = tmp_path / "output", tmp_path / "output" / "maplong_run"
    (save / "_tmp_results_aligned").mkdir(parents=True)
    N, CS, OV = 12, 4, 2
    frames = list(range(N))
    (tmp_path / "sel.json").write_text(json.dumps({"selected_files": [f"{f:06d}.jpg" for f in frames]}))
    H, W = 2, 3
    chunks = [(i * 2, min(i * 2 + CS, N)) for i in range((N - OV + 2 - 1) // 2)]
    for k, (a, b) in enumerate(chunks):
        S = b - a
        c2w = np.tile(np.eye(4), (S, 1, 1))
        wp = np.zeros((S, H, W, 3))
        wp[..., 2] = 10.0 + k                          # each chunk's depth names the chunk
        np.save(save / "_tmp_results_aligned" / f"chunk_{k}.npy",
                {"world_points": wp[None], "world_points_conf": np.ones((1, S, H, W)),
                 "extrinsic": c2w[None], "intrinsic": np.tile(np.eye(3), (1, S, 1, 1))},
                allow_pickle=True)
    (out / "camera_poses.txt").write_text("\n".join(" ".join(["1", "0", "0", "0", "0", "1", "0", "0",
                                                              "0", "0", "1", "0", "0", "0", "0", "1"])
                                                    for _ in frames) + "\n")
    (out / "camera_frames.txt").write_text(" ".join(str(f) for f in frames))
    _emit_omega_depth(save, out, CS, OV, str(tmp_path / "sel.json"), _Pipe())
    centres = [(a + b) / 2.0 for a, b in chunks]
    for g in frames:
        own = min((k for k, (a, b) in enumerate(chunks) if a <= g < b),
                  key=lambda k: (abs(g - centres[k]), k))
        with np.load(out / "omega_run" / "results_output" / f"frame_{g}.npz") as r:
            assert int(r["chunk"]) == own, g
            assert np.allclose(r["depth"], 10.0 + own), g
