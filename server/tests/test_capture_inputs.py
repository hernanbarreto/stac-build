"""The capture lives in <scan>/inputs/ (docs/plan_determinismo.md points 73 / 77, DECIDIDO
2026-10-07): every Stray / VIO reader looks there FIRST, then in the scan's own legacy places —
never a sibling scan; the replace wipe keeps inputs/ and frames/manifest.json, MOVES legacy
capture data into inputs/ instead of deleting it (refusing two different copies before touching
anything), and is refused — like the whole pipeline — while the scan's frames are being written;
camera.json, scale_diagnostics.json and gauge.json record where K / the scale came from and the
sha256 of the file. A session with no VIO and no Stray (every session today) is a no-op."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                       # noqa: E402
from ingestors import capture_inputs as CI                         # noqa: E402
from intake import frames_manifest as FM                           # noqa: E402

K_CSV = "400,0,232\n0,400,416\n0,0,1\n"
ODO = "timestamp,frame,x,y,z,qx,qy,qz,qw\n"


def _scan(tmp_path, name="src_default"):
    s = tmp_path / "scan" / name
    (s / "frames").mkdir(parents=True)
    (s / "frames" / "000000.jpg").write_bytes(b"jpg")
    return s


def _stray(d: Path, k=K_CSV):
    d.mkdir(parents=True, exist_ok=True)
    (d / "camera_matrix.csv").write_text(k)
    (d / "odometry.csv").write_text(ODO)
    for sub in ("depth", "confidence"):
        (d / sub).mkdir(exist_ok=True)
        (d / sub / "000000.png").write_bytes(sub.encode())
    (d / "rgb.mp4").write_bytes(b"rgb")


# ── readers: inputs/ first, then the scan's own places, never a sibling ─────────────────────

def test_every_stray_reader_prefers_inputs_then_the_scans_own_places(tmp_path):
    from workers.map_worker import _find_stray_dir as mw
    from segmentation.session_io import _find_stray_dir as sio
    from segmentation.shaper_export import _find_stray_dir as shp
    from precision import camera as C
    from precision import gauge as G
    from ingestors.stray_detector import detect_stray_data
    s = _scan(tmp_path)
    _stray(tmp_path / "scan" / "src_sibling")                      # another recording
    for f in (mw, sio, shp):
        assert f(s) is None
    assert C.find_stray_camera_matrix(s) is None and G._stray_depth_dir(s) is None
    assert detect_stray_data(s)["is_stray_session"] is False
    _stray(s)                                                      # legacy: the scan root
    assert mw(s) == sio(s) == shp(s) == s
    _stray(s / "inputs" / "stray")                                 # the capture's place
    ins = s / "inputs" / "stray"
    assert mw(ins.parent.parent) == sio(s) == shp(s) == ins
    assert C.find_stray_camera_matrix(s) == ins / "camera_matrix.csv"   # same bytes: no conflict
    assert G._stray_depth_dir(s) == ins
    inputs = G.chain_inputs(s, ["stray"])
    assert inputs["depth"] == ins / "depth" and inputs["confidence"] == ins / "confidence"
    det = detect_stray_data(s)
    assert det["is_stray_session"] and det["camera_matrix"] == ins / "camera_matrix.csv"
    # two DIFFERENT calibrations of one scan are refused, naming both
    (s / "camera_matrix.csv").write_text("401,0,232\n0,401,416\n0,0,1\n")
    with pytest.raises(C.CameraError, match="two different Stray calibrations"):
        C.find_stray_camera_matrix(s)


def test_vio_is_read_from_inputs_first(tmp_path):
    from ingestors.vio_detector import detect_vio_data
    s = _scan(tmp_path)
    (tmp_path / "scan" / "src_sibling").mkdir()
    (tmp_path / "scan" / "src_sibling" / "vio_trajectory.csv").write_text("0,0,0,0\n")
    assert detect_vio_data(s)["has_vio"] is False                   # never a sibling's
    (s / "vio_trajectory.csv").write_text("legacy")
    assert detect_vio_data(s)["vio_path"] == s / "vio_trajectory.csv"
    (s / "inputs" / "vio").mkdir(parents=True)
    (s / "inputs" / "vio" / "trajectory.json").write_text("{}")
    det = detect_vio_data(s)
    assert det["vio_path"] == s / "inputs" / "vio" / "trajectory.json" and det["format"] == "json"
    assert CI.vio_record(s) == {"file": "inputs/vio/trajectory.json",
                                "sha256": repro.sha256_bytes(b"{}")}


# ── the replace wipe ────────────────────────────────────────────────────────────────────────

def _left(s):
    return sorted(str(p.relative_to(s)) for p in s.rglob("*"))


def test_wipe_keeps_the_seal_and_inputs_and_moves_legacy_capture_data(tmp_path):
    from pipeline_manager import PipelineManager
    s = _scan(tmp_path)
    (s / "source_video.mp4").write_bytes(b"video")
    FM.adopt(s, {"000000.jpg": repro.sha256_bytes(b"jpg")}, log=lambda *a: None)
    seal = FM.manifest_path(s).read_bytes()
    (s / "frames" / "selected_frames.json").write_text("{}")     # an intake product: goes
    (s / "inputs").mkdir()
    (s / "inputs" / "vio_trajectory.json").write_text('{"kept": 1}')
    _stray(s)                                                      # legacy root Stray export
    (s / "imu.csv").write_text("imu")
    (s / "vio_trajectory.csv").write_text("0,0,0,0\n")            # legacy root VIO
    (s / "camera_data").mkdir()
    (s / "camera_data" / "000000.json").write_text("{}")         # legacy WebXR capture data
    (s / "scan_meta.json").write_text("{}")
    (s / "output" / "potree").mkdir(parents=True)
    (s / "intake").mkdir()
    (s / "intake" / "walk.json").write_text("{}")

    PipelineManager._wipe_outputs_for_replace(s, s / "output")

    assert _left(s) == [
        "frames", "frames/000000.jpg", "frames/manifest.json",
        "inputs", "inputs/stray", "inputs/stray/camera_matrix.csv", "inputs/stray/confidence",
        "inputs/stray/confidence/000000.png", "inputs/stray/depth", "inputs/stray/depth/000000.png",
        "inputs/stray/imu.csv", "inputs/stray/odometry.csv", "inputs/stray/rgb.mp4",
        "inputs/vio_trajectory.csv", "inputs/vio_trajectory.json",
        "inputs/webxr", "inputs/webxr/camera_data", "inputs/webxr/camera_data/000000.json",
        "inputs/webxr/scan_meta.json",
        "output", "source_video.mp4"], _left(s)
    assert FM.manifest_path(s).read_bytes() == seal
    assert (s / "inputs" / "stray" / "camera_matrix.csv").read_text() == K_CSV
    assert (s / "inputs" / "vio_trajectory.json").read_text() == '{"kept": 1}'
    # a second replace is a no-op on the capture
    PipelineManager._wipe_outputs_for_replace(s, s / "output")
    assert "inputs/stray/rgb.mp4" in _left(s)


def test_wipe_moves_a_legacy_stray_subdir_and_drops_identical_duplicates(tmp_path):
    from pipeline_manager import PipelineManager
    s = _scan(tmp_path)
    _stray(s / "stray")
    _stray(s / "inputs" / "stray")                                 # the same export, already moved
    (s / "depth").mkdir()                                          # root depth/ with no Stray
    (s / "depth" / "x.png").write_bytes(b"x")                      # marker: not capture data
    PipelineManager._wipe_outputs_for_replace(s, s / "output")
    left = _left(s)
    assert not (s / "stray").exists() and not (s / "depth").exists()
    assert "inputs/stray/camera_matrix.csv" in left and "inputs/stray/depth/000000.png" in left


def test_two_different_copies_of_capture_data_refuse_the_wipe_before_anything_is_touched(tmp_path):
    from pipeline_manager import PipelineManager
    s = _scan(tmp_path)
    _stray(s)
    _stray(s / "inputs" / "stray", k="999,0,1\n0,999,1\n0,0,1\n")
    (s / "output").mkdir()
    (s / "output" / "cleaned_cloud.ply").write_bytes(b"ply")
    before = _left(s)
    with pytest.raises(CI.CaptureInputsError, match="different content"):
        PipelineManager._wipe_outputs_for_replace(s, s / "output")
    assert _left(s) == before                                      # nothing moved, nothing deleted


def test_wipe_is_refused_while_the_frames_are_being_written(tmp_path):
    from pipeline_manager import PipelineManager
    s = _scan(tmp_path)
    (s / "output").mkdir()
    (s / "output" / "x").write_text("x")
    FM.extracting_dir(s).mkdir()
    with pytest.raises(RuntimeError, match="Replace refused"):
        PipelineManager._wipe_outputs_for_replace(s, s / "output")
    assert (s / "output" / "x").exists()


def test_a_session_without_capture_data_is_untouched_but_for_the_wipe(tmp_path):
    s = _scan(tmp_path)
    assert CI.move_legacy_capture(s) == [] and not (s / "inputs").exists()


# ── the pipeline does not start on a scan whose frames are being written ───────────────────

def test_the_pipeline_refuses_a_scan_with_an_extraction_in_progress(tmp_path):
    from pipeline_manager import (JobStatus, PipelineJob, PipelineManager, PipelineStage,
                                  StageId, StageState)
    s = _scan(tmp_path)
    (s / "output").mkdir()
    (s / "output" / "keep.txt").write_text("x")
    events = {"progress": [], "complete": []}

    async def on_progress(sid, d):
        events["progress"].append(d)

    async def on_complete(sid, ok):
        events["complete"].append(ok)

    def go():
        pm = PipelineManager()
        job = PipelineJob(session_id="sess", stages=[StageState(stage=PipelineStage(
            id=StageId.RECONSTRUCTION))])
        asyncio.run(pm._run_pipeline(job, str(s), {}, on_progress, on_complete, replace=True))
        return job

    for busy in ("temp dir", "claim"):
        if busy == "temp dir":
            FM.extracting_dir(s).mkdir()
            token = None
        else:
            FM.extracting_dir(s).rmdir()
            token = FM.claim_writer(s, FM.ORIGIN_VIDEO)
        try:
            job = go()
        finally:
            if token:
                FM.release_writer(s, token)
        assert job.status == JobStatus.FAILED
        assert "not ready" in job.stages[0].message and job.stages[0].status == JobStatus.FAILED
        assert events["complete"][-1] is False
        assert (s / "output" / "keep.txt").exists()               # no wipe, no stage ran
        assert not (s / "output" / "run_config.yaml").exists()


# ── provenance: where K and the scale came from ─────────────────────────────────────────────

def test_camera_json_records_the_k_source_and_its_sha(tmp_path):
    from precision import camera as C
    from tests.test_precision_camera import _CamCfg, _fake_session
    sess, _g, _rows = _fake_session(tmp_path)
    C.build_session_camera(sess, _CamCfg(), log=lambda *a: None)
    rep = json.loads((sess / "output" / "camera.json").read_text())["report"]
    assert rep["k_source"] == {"source": "omega", "file": "output/intrinsic.txt",
                               "sha256": repro.sha256_file(sess / "output" / "intrinsic.txt")}
    ins = sess / "inputs" / "stray"
    ins.mkdir(parents=True)
    np.savetxt(ins / "camera_matrix.csv", np.array([[400.0, 0, 232], [0, 400.0, 416], [0, 0, 1]]),
               delimiter=",")
    (ins / "odometry.csv").write_text(ODO)
    cam = C.build_session_camera(sess, _CamCfg("auto"), log=lambda *a: None)
    rep = json.loads((sess / "output" / "camera.json").read_text())["report"]
    assert cam.source == "stray"
    assert rep["k_source"] == {"source": "stray", "file": "inputs/stray/camera_matrix.csv",
                               "sha256": repro.sha256_file(ins / "camera_matrix.csv")}


def test_scale_diagnostics_record_the_vio_file_and_its_sha(tmp_path, monkeypatch):
    from reconstruction import scale_align
    from tests.test_scale_v2 import _vio_csv, build_session
    out = build_session(tmp_path, n_frames=12, s_true=2.0, noise=0.01)
    _s, diag = scale_align.estimate_v2(out, cfg={"mode": "global_median", "vio": True},
                                       session_dir=tmp_path)
    assert diag["scale_source"] == "da3" and diag["vio_sha256"] is None
    assert diag["scale_source_file"] is None
    vt = np.arange(0.0, 111.0, 0.1)
    vp = np.stack([0.01 * 2.1 * vt, np.zeros_like(vt), np.zeros_like(vt)], 1)
    (tmp_path / "inputs").mkdir()
    _vio_csv(tmp_path / "inputs" / "vio_trajectory.csv", vt, vp)
    monkeypatch.setattr("reconstruction.vio_scale.video_fps", lambda _s: 1.0)
    _s, diag = scale_align.estimate_v2(
        out, cfg={"mode": "global_median", "vio": True, "vio_segment_s": 10.0,
                  "vio_min_segments": 5, "vio_min_coverage": 0.5}, session_dir=tmp_path)
    sha = repro.sha256_file(tmp_path / "inputs" / "vio_trajectory.csv")
    assert diag["scale_source"] == "vio" and diag["vio_sha256"] == sha
    assert diag["scale_source_file"] == {"file": "inputs/vio_trajectory.csv", "sha256": sha}
    assert diag["vio"]["file"] == "inputs/vio_trajectory.csv"


def test_gauge_scale_inputs_name_the_vio_and_stray_files_used(tmp_path):
    from precision import gauge as G
    s = _scan(tmp_path)
    assert G._scale_inputs(s, ("vio", "stray"), []) == \
        {"vio": None, "vio_sha256": None, "stray_depth_dir": None}
    (s / "inputs").mkdir()
    (s / "inputs" / "vio_trajectory.csv").write_text("0,0,0,0\n")
    _stray(s / "inputs" / "stray")
    rec = G._scale_inputs(s, ("da3_windows", "vio", "stray"), [])
    sha = repro.sha256_file(s / "inputs" / "vio_trajectory.csv")
    assert rec == {"vio": {"file": "inputs/vio_trajectory.csv", "sha256": sha}, "vio_sha256": sha,
                   "stray_depth_dir": "inputs/stray/depth"}
    # an instrument that gave no rows did not set anything: null
    assert G._scale_inputs(s, ("vio", "stray"), ["vio", "stray"])["vio_sha256"] is None


def test_video_fps_reads_the_video_the_frames_came_from(tmp_path, monkeypatch):
    from reconstruction import vio_scale
    cv2 = pytest.importorskip("cv2")
    s = _scan(tmp_path)
    for name, fps in (("source_video.avi", 10.0), ("source_video.mkv", 25.0)):
        w = cv2.VideoWriter(str(s / name), cv2.VideoWriter_fourcc(*"MJPG"), fps, (32, 24))
        for _ in range(3):
            w.write(np.zeros((24, 32, 3), np.uint8))
        w.release()
    assert vio_scale.video_fps(s) == pytest.approx(10.0)            # alphabetical: .avi
    FM.adopt(s, {"000000.jpg": repro.sha256_bytes(b"jpg")}, log=lambda *a: None)
    m = json.loads(FM.manifest_path(s).read_text())
    assert m["video"] is None and m["videos_present"] == ["source_video.avi", "source_video.mkv"]
    m["video"] = {"name": "source_video.mkv", "size": 0, "sha256": "0" * 64}
    FM.manifest_path(s).write_text(json.dumps(m))
    assert vio_scale.video_fps(s) == pytest.approx(25.0)            # the manifest's video
