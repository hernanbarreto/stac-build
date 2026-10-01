"""The grid the cloud's birth pixels live on is DECLARED by the session (pccr 2026-10-01: the
2·cx / 2·cy guess over F5's refined camera read 470x828 for a 464x832 cloud and the certification
refused epoch 8 for an aspect ratio the cloud never had)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.visit_drift import trace_grid  # noqa: E402

CAM = {"width": 464, "height": 832, "params": [391.87, 388.71, 234.80, 414.06, 0, 0, 0, 0],
       "omega_grid": {"w": 464, "h": 832, "scale_x": 1.0, "scale_y": 1.0}}


def test_a_corrected_cloud_declares_its_grid(tmp_path):
    (tmp_path / "camera.json").write_text(json.dumps(CAM))
    (tmp_path / "corrected_cloud.json").write_text(json.dumps({"grid": [464, 832], "epoch_to": 3}))
    (tmp_path / "intrinsic.txt").write_text("391.87 388.71 234.80 414.06\n" * 3)
    assert trace_grid(tmp_path) == (832, 464)


def test_a_corrected_cloud_without_the_field_is_on_the_camera_grid(tmp_path):
    (tmp_path / "camera.json").write_text(json.dumps(CAM))
    (tmp_path / "corrected_cloud.json").write_text(json.dumps({"epoch_to": 3}))
    assert trace_grid(tmp_path) == (832, 464)


def test_an_omega_cloud_is_on_omega_s_record_grid(tmp_path):
    cam = dict(CAM, width=1920, height=1080, omega_grid={"w": 688, "h": 384, "scale_x": 0.358, "scale_y": 0.356})
    (tmp_path / "camera.json").write_text(json.dumps(cam))
    (tmp_path / "intrinsic.txt").write_text("300 300 344 192\n")
    assert trace_grid(tmp_path) == (384, 688)


def test_a_legacy_session_reads_the_centred_intrinsics(tmp_path):
    np.savetxt(tmp_path / "intrinsic.txt", np.array([[300.0, 300.0, 344.0, 192.0]] * 4))
    assert trace_grid(tmp_path) == (384, 688)


def test_the_refined_camera_no_longer_misreads_the_grid(tmp_path):
    """The case that refused epoch 8: F5's principal point off centre, no corrected report."""
    (tmp_path / "camera.json").write_text(json.dumps(CAM))
    (tmp_path / "intrinsic.txt").write_text("391.87 388.71 234.80 414.06\n" * 3)
    assert trace_grid(tmp_path) == (832, 464)


def test_nothing_declared_is_an_error(tmp_path):
    try:
        trace_grid(tmp_path)
    except RuntimeError as e:
        assert "cannot be guessed" in str(e)
    else:
        raise AssertionError("a grid nobody declared must not be guessed")
