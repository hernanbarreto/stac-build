"""Phase 3d (audit 2026-10-23): the classical cross-check — COLMAP / GLOMAP as an independent
photogrammetry: its verified two-view geometries far apart in the walk become F4's loop candidates
(stamped to this reconstruction and keyframe list), and its trajectory Sim3-aligned to F5's is
reported keyframe by keyframe. A stand-in runner writes the database and the sparse model."""
import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import classical_check as CK                       # noqa: E402
from precision.config import load_precision_config                # noqa: E402

PC = load_precision_config()
KF = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def _db(path: Path, names, pairs):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, "
                "data BLOB, config INTEGER)")
    for k, n in enumerate(names):
        con.execute("INSERT INTO images VALUES (?, ?, 1)", (k + 1, n))
    for a, b, n in pairs:                           # image ids (1-based), inliers
        con.execute("INSERT INTO two_view_geometries VALUES (?, ?, 2, NULL, 2)", (a * CK.MAX_IMAGE_ID + b, n))
    con.commit(); con.close()


def _model(model_dir: Path, poses_c2w, names):
    from evaluation.colmap_io import ColmapCamera, ColmapImage, rotmat_to_qvec, write_model_text
    cams = {1: ColmapCamera(1, "OPENCV", 640, 480, [500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0])}
    imgs = {}
    for k, (T, n) in enumerate(zip(poses_c2w, names)):
        w2c = np.linalg.inv(T)
        imgs[k + 1] = ColmapImage(k + 1, n, 1, rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3])
    write_model_text(model_dir, cams, imgs, np.zeros((0, 3)))


def _walk(n=len(KF)):
    c2w = []
    for k in range(n):
        T = np.eye(4); a = 0.2 * k
        T[:3, :3] = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        T[:3, 3] = [0.5 * k, 0.02 * np.sin(1.3 * k), 0.1 * k * k / n]
        c2w.append(T)
    return np.array(c2w)


def _session(tmp_path):
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    s = tmp_path / "sess"; out = s / "output"
    (s / "frames").mkdir(parents=True); (out / "precision").mkdir(parents=True)
    names = [f"{f:06d}.jpg" for f in KF]
    (s / "frames" / "selected_frames.json").write_text(json.dumps({"selected_files": names}))
    for n in names:
        (s / "frames" / n).write_bytes(b"")
    g = grid_full_frame_resize(640, 480, 640, 480, "omega")
    save_camera_json(out / "camera.json", CameraModel(640, 480, (500.0, 500.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0),
                                                      "synthetic", 0, g), 0)
    # a reconstruction to name (correction.epoch.reconstruction_id): Omega's records + the keyframe list
    rec = out / "omega_run" / "results_output"; rec.mkdir(parents=True)
    for f in KF:
        np.savez(rec / f"frame_{f}.npz", depth=np.ones((4, 4), np.float32), conf=np.ones((4, 4), np.float32))
    (out / "camera_frames.txt").write_text("\n".join(map(str, KF)) + "\n")
    return s, names


def _runner(poses_c2w, pairs):
    def run(ws, files, ccfg, log):
        _db(ws / "database.db", files, pairs)
        _model(ws / "sparse_global" / "0", poses_c2w, files)
        return ({"features": "SIFT", "n_images": len(files)},
                {"mapper": "global", "models": [{"dir": "sparse_global/0", "n_images": len(files), "n_points3d": 0}]})
    return run


def test_loop_pairs_come_from_the_verified_geometries_far_apart(tmp_path):
    names = [f"{f:06d}.jpg" for f in KF]
    _db(tmp_path / "database.db", names, [(1, 2, 500), (1, 9, 120), (2, 10, 40), (3, 8, 90), (4, 6, 300)])
    pairs = CK.loop_pairs_from_database(tmp_path / "database.db", KF, min_inliers=60, min_gap=4)
    assert pairs == [(0, 8, 120), (2, 7, 90)]


def test_the_step_writes_stamped_loop_candidates_f4_takes_only_its_own(tmp_path):
    s, names = _session(tmp_path)
    c2w = _walk()
    pcfg = replace(PC, classical=replace(PC.classical, loop_min_inliers=60, loop_min_gap_keyframes=4))
    rep = CK.run_classical(s, pcfg, log=lambda *a: None, runner=_runner(c2w, [(1, 9, 120), (3, 8, 90), (1, 2, 400)]))
    assert rep["loop_candidates"] == 2 and rep["registered_share"] == 1.0
    doc = json.loads((s / "output" / "precision" / CK.LOOPS_NAME).read_text())
    assert doc["pairs"] == [[0, 40, 120], [10, 35, 90]] and doc["keyframes_sha256"] == CK.keyframe_list_sha(KF)
    from correction.epoch import reconstruction_id_or_none
    rid = reconstruction_id_or_none(s / "output")
    pairs, info = CK.classical_loops(s / "output", KF, rid)
    assert info["taken"] and pairs == [(0, 8), (2, 7)]
    # another keyframe list: refused by name
    pairs2, info2 = CK.classical_loops(s / "output", KF + [50], rid)
    assert pairs2 == [] and not info2["taken"] and "keyframe list" in info2["reason"]
    # another reconstruction id: refused by the chain's ownership rule
    pairs3, info3 = CK.classical_loops(s / "output", KF, "not-this-reconstruction")
    assert pairs3 == [] and not info3["taken"]


def test_the_cross_check_aligns_the_classical_trajectory_and_names_the_worst_keyframes(tmp_path):
    s, names = _session(tmp_path)
    c2w = _walk()
    # the classical model: F5's truth in ANOTHER metre (x 0.37) and frame (rotated, shifted), two
    # keyframes pushed 8 cm, one unregistered
    rng = np.random.default_rng(0)
    A = np.eye(4); A[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]); A[:3, 3] = [3.0, -1.0, 2.0]
    cl = []
    for k, T in enumerate(c2w):
        Tk = T.copy()
        if k in (3, 7):
            Tk[:3, 3] += 0.08 * rng.normal(size=3) / np.sqrt(3)
        S = A.copy(); S[:3, 3] = A[:3, 3]
        Tc = S @ Tk; Tc[:3, 3] = 0.37 * (S[:3, :3] @ Tk[:3, 3]) + S[:3, 3]
        cl.append(Tc)
    cl = cl[:-1]; names_reg = names[:-1]
    pcfg = replace(PC, classical=replace(PC.classical, loop_min_inliers=60, loop_min_gap_keyframes=4))
    CK.run_classical(s, pcfg, log=lambda *a: None, runner=_runner(cl, [(1, 9, 120)]))
    from correction.epoch import reconstruction_id_or_none
    rep = CK.cross_check(s / "output", KF, np.linalg.inv(c2w), reconstruction_id_or_none(s / "output"))
    assert rep["measured"] and rep["n_registered"] == len(KF) - 1
    assert abs(rep["alignment"]["scale_f5_over_classical"] - 1 / 0.37) < 0.01
    assert rep["position_m"]["median"] < 0.005 and rep["rotation_deg"]["median"] < 0.05
    worst = {w["frame"] for w in rep["worst_keyframes"][:2]}
    assert worst == {KF[3], KF[7]} and rep["position_m"]["max"] > 0.02
    # no model: declared, never a number
    assert not CK.cross_check(tmp_path / "nowhere", KF, np.linalg.inv(c2w), None)["measured"]
