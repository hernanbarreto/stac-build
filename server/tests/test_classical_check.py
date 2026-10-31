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


def _runner(poses_c2w, pairs, broken=None):
    """The stand-in for the COLMAP stages: the database, one model per mapper asked. ``broken``:
    {mapper: (k0, angle_deg)} — that mapper's model turned about x by the angle from keyframe k0 on
    (a broken segment, what GLOMAP produced without the camera prior)."""
    def run(ws, files, ccfg, log, camera_params=None, mappers=("global",)):
        run.calls.append({"camera_params": list(camera_params) if camera_params is not None else None,
                          "mappers": list(mappers)})
        import shutil
        (ws / "database.db").unlink(missing_ok=True)                 # a second run on the same session
        for d in ws.glob("sparse_*"):
            shutil.rmtree(d, ignore_errors=True)
        _db(ws / "database.db", files, pairs)
        mapped = {}
        for m in mappers:
            poses = np.array(poses_c2w, np.float64).copy()
            if broken and m in broken:
                k0, deg = broken[m]
                a = np.radians(deg)
                Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
                T = np.eye(4); T[:3, :3] = Rx
                poses[k0:] = np.einsum("ij,njk->nik", T, poses[k0:])
            _model(ws / f"sparse_{m}" / "0", poses, files)
            mapped[m] = {"mapper": m, "models": [{"dir": f"sparse_{m}/0", "n_images": min(len(files), len(poses)),
                                                  "n_points3d": 0}]}
        return ({"features": "SIFT", "n_images": len(files), "camera_params": camera_params}, mapped)
    run.calls = []
    return run


def test_loop_pairs_come_from_the_verified_geometries_far_apart(tmp_path):
    names = [f"{f:06d}.jpg" for f in KF]
    _db(tmp_path / "database.db", names, [(1, 2, 500), (1, 9, 120), (2, 10, 40), (3, 8, 90), (4, 6, 300)])
    pairs = CK.loop_pairs_from_database(tmp_path / "database.db", KF, min_inliers=60, min_gap=4)
    assert pairs == [(0, 8, 120), (2, 7, 90)]


def test_the_step_writes_stamped_loop_candidates_f4_takes_only_its_own(tmp_path):
    s, names = _session(tmp_path)
    c2w = _walk()
    pcfg = replace(PC, classical=replace(PC.classical, loop_min_inliers=60, loop_min_gap_keyframes=4, mappers=("global",)))
    rep = CK.run_classical(s, pcfg, log=lambda *a: None, runner=_runner(c2w, [(1, 9, 120), (3, 8, 90), (1, 2, 400)]))
    assert rep["loop_candidates"] == 2 and rep["registered_share"] == {"global": 1.0}
    assert rep["camera_prior"]["used"] and rep["camera_prior"]["params_native"][:4] == [500.0, 500.0, 320.0, 240.0]
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
    pcfg = replace(PC, classical=replace(PC.classical, loop_min_inliers=60, loop_min_gap_keyframes=4, mappers=("global",)))
    CK.run_classical(s, pcfg, log=lambda *a: None, runner=_runner(cl, [(1, 9, 120)]))
    from correction.epoch import reconstruction_id_or_none
    rep = CK.cross_check(s / "output", KF, np.linalg.inv(c2w), reconstruction_id_or_none(s / "output"))
    assert rep["measured"] and rep["n_registered"] == len(KF) - 1 and rep["top_level_from"] == "global"
    assert rep["models"]["global"]["n_registered"] == len(KF) - 1 and rep["camera_prior"]["used"]
    assert abs(rep["alignment"]["scale_f5_over_classical"] - 1 / 0.37) < 0.01
    assert rep["position_m"]["median"] < 0.005 and rep["rotation_deg"]["median"] < 0.05
    worst = {w["frame"] for w in rep["worst_keyframes"][:2]}
    assert worst == {KF[3], KF[7]} and rep["position_m"]["max"] > 0.02
    # no model: declared, never a number
    assert not CK.cross_check(tmp_path / "nowhere", KF, np.linalg.inv(c2w), None)["measured"]


def test_the_camera_prior_is_the_session_camera_in_native_pixels(tmp_path):
    """camera.json's parameters live on the record grid; COLMAP gets them in native pixels through
    the grid map (fx, fy scaled; cx, cy through the pixel-centre convention), the distortion as is."""
    s, _names = _session(tmp_path)
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    g = grid_full_frame_resize(1280, 960, 640, 480, "omega")            # the grid is HALF the native frame
    save_camera_json(s / "output" / "camera.json",
                     CameraModel(1280, 960, (500.0, 500.0, 319.5, 239.5, 0.01, -0.002, 0.0003, 0.0), "synthetic", 0, g), 0)
    params, rec = CK.camera_prior_params(s / "output")
    assert params[0] == 1000.0 and params[1] == 1000.0
    assert abs(params[2] - ((319.5 + 0.5) * 2.0 - 0.5)) < 1e-9 and abs(params[3] - ((239.5 + 0.5) * 2.0 - 0.5)) < 1e-9
    assert params[4:] == [0.01, -0.002, 0.0003, 0.0]
    assert rec["native_size"] == [1280, 960] and rec["camera_epoch"] == 0 and rec["used"]


def test_every_mapper_runs_with_the_prior_and_the_cross_check_names_a_broken_segment(tmp_path):
    """Both mappers on the one database, the camera prior handed to the runner; the cross-check
    reports each model, the first configured at the top level, and the consecutive-keyframe STEPS
    name a segment turned 170° in one model (pccr_audit 2026-10-31: GLOMAP without the prior) while
    the other agrees with F5 step by step."""
    s, _names = _session(tmp_path)
    c2w = _walk()
    pcfg = replace(PC, classical=replace(PC.classical, loop_min_inliers=60, loop_min_gap_keyframes=4,
                                         mappers=("global", "incremental"), camera_prior=True))
    runner = _runner(c2w, [(1, 9, 120)], broken={"incremental": (5, 170.0)})
    rep = CK.run_classical(s, pcfg, log=lambda *a: None, runner=runner)
    assert runner.calls[0]["mappers"] == ["global", "incremental"]
    assert runner.calls[0]["camera_params"][:4] == [500.0, 500.0, 320.0, 240.0]
    assert rep["mappers"] == ["global", "incremental"] and set(rep["mapping"]) == {"global", "incremental"}
    assert rep["registered_share"] == {"global": 1.0, "incremental": 1.0}
    from correction.epoch import reconstruction_id_or_none
    cc = CK.cross_check(s / "output", KF, np.linalg.inv(c2w), reconstruction_id_or_none(s / "output"))
    assert cc["measured"] and cc["top_level_from"] == "global" and set(cc["models"]) == {"global", "incremental"}
    g, i = cc["models"]["global"], cc["models"]["incremental"]
    assert g["rotation_deg"]["median"] < 0.01 and g["steps"]["abs_diff_deg"]["max"] < 0.01 and g["steps"]["n_steps"] == len(KF) - 1
    assert i["steps"]["abs_diff_deg"]["max"] > 150.0 and i["steps"]["abs_diff_deg"]["median"] < 0.01
    assert i["steps"]["worst"][0]["from_frame"] == KF[4] and i["steps"]["worst"][0]["to_frame"] == KF[5]
    assert cc["n_registered"] == len(KF) and cc["camera_prior"]["used"]
    # the prior off: COLMAP's own default, said
    pcfg2 = replace(pcfg, classical=replace(pcfg.classical, camera_prior=False))
    runner2 = _runner(c2w, [(1, 9, 120)])
    rep2 = CK.run_classical(s, replace(pcfg2, classical=replace(pcfg2.classical, mappers=("global",))),
                            log=lambda *a: None, runner=runner2)
    assert runner2.calls[0]["camera_params"] is None and rep2["camera_prior"]["used"] is False


def test_colmap_matching_takes_the_camera_prior_and_rebuilds_when_it_changes(tmp_path):
    """evaluation.colmap_baseline.ensure_features_and_matches: the prior becomes COLMAP's
    ImageReader.camera_params; a workspace made with another prior is rebuilt (the two-view
    geometries depend on the camera), one made with the same is reused."""
    from evaluation import colmap_baseline as CB
    fake = tmp_path / "colmap"
    fake.write_text(f"""#!/bin/bash
echo "$@" >> {tmp_path}/argv.log
[ "$1" = "-h" ] && echo "I COLMAP 4.2.0 (fake)"
exit 0
""")
    fake.chmod(0o755)
    ws = tmp_path / "ws"; (ws / "images").mkdir(parents=True); (ws / "images" / "000000.jpg").write_bytes(b"")
    ccfg = {"binary": str(fake), "camera_model": "OPENCV", "use_gpu": False, "sift": {"max_num_features": 100},
            "aliked": {"max_num_features": 100}, "timeout_s": 60, "vocab_tree": {},
            "sequential": {"overlap": 2, "quadratic_overlap": False, "loop_detection": False,
                           "loop_detection_period": 10, "loop_detection_num_images": 5}}
    prior = [364.1, 364.6, 232.0, 416.0, 0.0, 0.0, 0.0, 0.0]
    rec = CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, lambda *a: None, camera_params=prior)

    def extractions():
        return [a for a in (tmp_path / "argv.log").read_text().splitlines() if a.startswith("feature_extractor")]
    fx = extractions()
    assert len(fx) == 1 and "--ImageReader.camera_params 364.1,364.6,232,416,0,0,0,0" in fx[0]
    assert rec["camera_params"] == prior
    CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, lambda *a: None, camera_params=prior)
    assert len(extractions()) == 1                               # reused
    rec2 = CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, lambda *a: None)
    fx = extractions()
    assert len(fx) == 2 and "camera_params" not in fx[1] and rec2["camera_params"] is None


def test_a_matcher_killed_by_a_signal_is_retried_once_and_recorded(tmp_path):
    """FAISS's vocabulary-tree retrieval died by SIGSEGV on the pod (pccr_audit3 2026-10-31) and ran
    through on the next attempt: ONE retry on a signal death, declared in the record; a second death
    is the failure it is."""
    from evaluation import colmap_baseline as CB
    fake = tmp_path / "colmap"
    fake.write_text(f"""#!/bin/bash
echo "$@" >> {tmp_path}/argv.log
[ "$1" = "-h" ] && echo "I COLMAP 4.2.0 (fake)"
if [ "$1" = "sequential_matcher" ]; then
  n=$(grep -c "^sequential_matcher" {tmp_path}/argv.log)
  [ "$n" -le "$(cat {tmp_path}/deaths)" ] && kill -SEGV $$
fi
exit 0
""")
    fake.chmod(0o755)
    ccfg = {"binary": str(fake), "camera_model": "OPENCV", "use_gpu": False, "sift": {"max_num_features": 100},
            "aliked": {"max_num_features": 100}, "timeout_s": 60, "vocab_tree": {},
            "sequential": {"overlap": 2, "quadratic_overlap": False, "loop_detection": False,
                           "loop_detection_period": 10, "loop_detection_num_images": 5}}
    ws = tmp_path / "ws"; (ws / "images").mkdir(parents=True); (ws / "images" / "000000.jpg").write_bytes(b"")
    (tmp_path / "deaths").write_text("1")                      # the first matcher dies, the second lives
    rec = CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, lambda *a: None)
    matcher = [c for c in rec["commands"] if c["argv"][1] == "sequential_matcher"]
    assert len(matcher) == 1 and matcher[0]["attempt"] == 2 and "code -11" in matcher[0]["first_attempt"]
    assert sum(1 for a in (tmp_path / "argv.log").read_text().splitlines() if a.startswith("sequential_matcher")) == 2
    # two deaths: the failure it is
    ws2 = tmp_path / "ws2"; (ws2 / "images").mkdir(parents=True); (ws2 / "images" / "000000.jpg").write_bytes(b"")
    (tmp_path / "argv.log").write_text(""); (tmp_path / "deaths").write_text("2")
    with pytest.raises(CB.ColmapError, match="code -11"):
        CB.ensure_features_and_matches(ws2, "SIFT", "SIFT_BRUTEFORCE", ccfg, lambda *a: None)
