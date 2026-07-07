# STAC-Builder — Phase R writeback unit tests (R.4 pose + R.5 depth apply).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.residuals import Sim3  # noqa: E402
from phase_r.writeback import compose_refined_poses, write_refined_camera_poses  # noqa: E402


def test_identity_correction_is_noop():
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    pm = {0: pose, 1: np.eye(4)}
    out = compose_refined_poses(pm, {}, ["global"], [Sim3.identity()])
    assert np.allclose(out[0], pose) and np.allclose(out[1], np.eye(4))


def test_known_sim3_applied_to_pose():
    # window 1 correction: scale 2, translate +x by 5
    M = Sim3(2.0, np.eye(3), np.array([5.0, 0.0, 0.0]))
    pose = np.eye(4); pose[:3, 3] = [1.0, 0.0, 0.0]
    window_map = {7: "w1"}
    out = compose_refined_poses({7: pose}, window_map, ["w0", "w1"], [Sim3.identity(), M])
    # camera centre: 2*(I@[1,0,0]) + [5,0,0] = [7,0,0]
    assert np.allclose(out[7][:3, 3], [7.0, 0.0, 0.0])
    assert np.allclose(out[7][:3, :3], np.eye(3))


def test_write_refined_camera_poses_roundtrip_and_backup():
    with tempfile.TemporaryDirectory() as d:
        # two keyframes, frame numbers 10 and 20
        p0 = np.eye(4); p1 = np.eye(4); p1[:3, 3] = [9, 9, 9]
        (open(os.path.join(d, "camera_poses.txt"), "w")
         .write(" ".join(map(str, p0.reshape(-1))) + "\n"
                + " ".join(map(str, p1.reshape(-1))) + "\n"))
        open(os.path.join(d, "camera_frames.txt"), "w").write("10 20\n")
        refined = {10: (np.eye(4) * 1), 20: np.eye(4)}
        refined[10][:3, 3] = [3, 3, 3]
        written = write_refined_camera_poses(d, refined)
        assert written and os.path.exists(os.path.join(d, "camera_poses.txt.prephaser"))
        lines = open(os.path.join(d, "camera_poses.txt")).read().splitlines()
        row0 = np.array([float(x) for x in lines[0].split()]).reshape(4, 4)
        assert np.allclose(row0[:3, 3], [3, 3, 3])   # frame 10 got refined
        # backup preserves the original
        bak = open(os.path.join(d, "camera_poses.txt.prephaser")).read().splitlines()
        assert np.allclose(np.array([float(x) for x in bak[0].split()]).reshape(4, 4), p0)


def test_write_skips_when_unaligned():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "camera_poses.txt"), "w").write(
            " ".join(map(str, np.eye(4).reshape(-1))) + "\n")
        open(os.path.join(d, "camera_frames.txt"), "w").write("10 20 30\n")  # mismatch
        written = write_refined_camera_poses(d, {10: np.eye(4)})
        assert written == []  # not aligned -> skipped, nothing written


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name); n += 1
    print(f"\n{n} tests passed")
