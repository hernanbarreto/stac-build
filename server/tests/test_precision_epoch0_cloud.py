"""precision.epoch0_cloud: the Omega cloud of epoch 0 rebuilt after the core, from the
records, in epoch-0 coordinates, in the layout the cleaner reads, registered in the
stored epoch's manifest (USER 2026-09-29: keep epoch 0, selectable from the UI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from precision import epoch0_cloud as E0


def test_similarity_recovers_scale_rotation_translation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    dst = 0.9707 * src @ R.T + np.array([1.0, 2.0, -0.5])
    s, Rh, t = E0.similarity_umeyama(src, dst)
    assert abs(s - 0.9707) < 1e-9 and np.allclose(Rh, R) and np.allclose(t, [1.0, 2.0, -0.5])


def test_conf_threshold_is_the_forks_rule():
    c = np.concatenate([np.zeros(10), np.linspace(1.0, 11.0, 100)])     # zeros = sky mask
    thr = E0.conf_threshold(c, 20.0, 0.0)
    assert abs(thr - np.percentile(np.linspace(1.0, 11.0, 100), 20)) < 1e-9
    # the min-max floor on top: 10 % of [1, 11] = 2.0 — the stricter wins
    assert E0.conf_threshold(c, 0.0, 0.10) == pytest.approx(2.0)
    assert E0.conf_threshold(c, 20.0, 0.10) == pytest.approx(max(thr, 2.0))
    assert E0.conf_threshold(np.zeros(5), 20.0, 0.1) == -1.0


def _records_session(tmp_path: Path, n: int = 4) -> Path:
    import cv2
    s = tmp_path / "sess"
    out = s / "output"
    rec = out / "omega_run" / "results_output"
    rec.mkdir(parents=True)
    (s / "frames").mkdir()
    H, W = 8, 12
    K = np.array([[10.0, 0, 6.0], [0, 10.0, 4.0], [0, 0, 1]])
    frames = [10 * i for i in range(n)]
    pre, e0 = [], []
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    for i, f in enumerate(frames):
        c2w = np.eye(4); c2w[0, 3] = 0.1 * i
        depth = np.full((H, W), 2.0 + 0.1 * i, np.float32)
        conf = np.full((H, W), 5.0, np.float32); conf[0, :] = 0.0          # a sky row
        conf[1, :] = 1.0                                                   # a weak row
        np.savez(rec / f"frame_{f}.npz", depth=depth, conf=conf, pose_c2w=c2w, K_omega=K,
                 frame_global=f, chunk=i // 2)
        cv2.imwrite(str(s / "frames" / f"{f:06d}.jpg"), np.full((H, W, 3), 128, np.uint8))
        pre.append(c2w)
        e = c2w.copy(); e[:3, :3] = R @ c2w[:3, :3]; e[:3, 3] = 0.9707 * R @ c2w[:3, 3] + [1, 2, 3]
        e0.append(e)
    (out / "camera_frames.txt").write_text(" ".join(map(str, frames)))
    np.savetxt(out / "camera_poses.txt.prescale", np.stack([p.ravel() for p in pre]))
    (out / "_epoch_0").mkdir()
    np.savetxt(out / "_epoch_0" / "camera_poses.txt", np.stack([p.ravel() for p in e0]))
    (out / "_epoch_0" / "_manifest.json").write_text(json.dumps(
        {"epoch": 0, "epoch_from": 0, "epoch_to": 1,
         "artifacts": [{"rel": "camera_poses.txt", "existed_before": True}]}))
    return s


def test_records_become_cleaner_chunks_in_epoch0_coordinates(tmp_path):
    s = _records_session(tmp_path)
    tmp = s / "output" / "_tx"
    rep = E0.chunks_from_omega_records(s, tmp, conf_percentile=0.0, conf_min_norm=0.0,
                                       log=lambda *a: None)
    assert rep["n_chunks"] == 2 and rep["camera_residual_max_m"] < 1e-9
    plys = sorted(tmp.glob("chunk_*.ply")); orgs = sorted(tmp.glob("chunk_*_origins.npz"))
    assert [p.name for p in plys] == ["chunk_000.ply", "chunk_001.ply"]
    assert [p.name for p in orgs] == ["chunk_000_origins.npz", "chunk_001_origins.npz"]
    z = np.load(orgs[0])
    assert set(z.files) >= {"frame_global", "pixel_row", "pixel_col", "confidence"}
    assert z["frame_global"].dtype == np.int32 and z["pixel_row"].dtype == np.int16
    # the sky row is out, the weak row stays (percentile 0, no floor)
    assert 0 not in set(z["pixel_row"].tolist()) and 1 in set(z["pixel_row"].tolist())
    from correction.session import read_ply
    _, data = read_ply(plys[0])
    assert len(data) == len(z["frame_global"]) and {"x", "red"} <= set(data.dtype.names)
    # a point of keyframe 0 at pixel (2, 6): depth 2.0 along the optical axis → (0,0,2) in
    # the record frame → the similarity puts it at 0.9707·R·(0,0,2) + t
    m = (z["frame_global"] == 0) & (z["pixel_row"] == 4) & (z["pixel_col"] == 6)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1)[np.flatnonzero(m)[0]]
    assert np.allclose(xyz, [1.0, 2.0, 3.0 + 0.9707 * 2.0], atol=1e-5)


def test_register_in_manifest_adds_the_artifacts_once(tmp_path):
    e0 = tmp_path / "_epoch_0"; e0.mkdir()
    (e0 / "_manifest.json").write_text(json.dumps({"epoch": 0, "artifacts": [{"rel": "a", "existed_before": True}]}))
    E0.register_in_manifest(e0, ["cleaned_cloud.ply", "potree"])
    E0.register_in_manifest(e0, ["potree"])
    m = json.loads((e0 / "_manifest.json").read_text())
    assert [a["rel"] for a in m["artifacts"]] == ["a", "cleaned_cloud.ply", "potree"]
    assert all(a["existed_before"] for a in m["artifacts"])


def test_build_runs_the_recipe_and_registers(tmp_path, monkeypatch):
    s = _records_session(tmp_path)
    out = s / "output"
    calls = []

    class _P:
        returncode = 0
        def __init__(self, cmd, **kw):
            calls.append(cmd)
            self.stdout = iter(["[Step 3/6] ok", "✅ done"])
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"ply\n")
        def wait(self):
            return 0

    monkeypatch.setattr(E0.subprocess, "Popen", _P)
    import potree_converter
    def fake_potree(session_dir, force=False, ply_override=None, potree_dir_override=None):
        potree_dir_override.mkdir(parents=True)
        (potree_dir_override / "metadata.json").write_text(json.dumps({"points": 123}))
        return True
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree", fake_potree)
    cfg = {"reconstruction": {"simple": {"conf_percentile": 0.0, "conf_min_norm": 0.0}},
           "postprocessing": {"voxel_size": 0.005, "sor_knn": 8, "sor_sigma": 3.0,
                              "noise_radius": 0.01, "noise_sigma": 1.0, "conf_min_norm": 0.0,
                              "scene_consolidate": {"enabled": False}}}
    rep = E0.build_epoch0_cloud(s, cfg, from_records=True, log=lambda *a: None)
    assert rep["n_points"] == 123 and (out / "_epoch_0" / "cleaned_cloud.ply").exists()
    assert (out / "_epoch_0" / "potree" / "metadata.json").exists()
    m = json.loads((out / "_epoch_0" / "_manifest.json").read_text())
    assert {"cleaned_cloud.ply", "potree"} <= {a["rel"] for a in m["artifacts"]}
    assert "--voxel-size" in calls[0] and calls[0][calls[0].index("--voxel-size") + 1] == "0.005"
    assert not (out / E0.TMP_DIRNAME).exists()
