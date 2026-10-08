"""intake.stamps (docs/plan_determinismo.md 66, 70, 74, 78, 79): the CPU environment record
names the libraries, the CPU, the BLAS core and BOTH JPEG decoders with their libjpeg-turbo
builds; frames are stamped by their bytes under session-relative paths (a copied session keeps
its stamps, a changed byte does not); every intake step is keyed on the intake + DA3 code; (the single
visible card of point 78 is card_table.require_one_visible_card's, tested in test_card_table)."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                    # noqa: E402
from intake import stamps as St                                 # noqa: E402

SERVER = Path(__file__).resolve().parents[1]


def _frames(root: Path, n=3):
    fd = root / "frames"
    fd.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (fd / f"{i * 3:06d}.jpg").write_bytes(bytes([i, 7, 9]) * 10)
    return fd


# ── the environment ─────────────────────────────────────────────────────────────────────────

def test_cpu_environment_record_names_libraries_cpu_blas_and_both_jpeg_decoders():
    import cv2
    import PIL
    import scipy
    rec = St.cpu_environment_record()
    assert rec["libs"]["numpy"] == np.__version__ and rec["libs"]["scipy"] == scipy.__version__
    assert rec["libs"]["pillow"] == PIL.__version__
    assert (rec["libs"]["opencv-python"] or "").startswith(cv2.__version__) or \
        (rec["libs"]["opencv-python-headless"] or "").startswith(cv2.__version__)
    assert rec["cpu_model"] == repro.cpu_model()
    assert any(b["internal_api"] == "openblas" for b in rec["blas"])
    assert all("num_threads" not in b for b in rec["blas"])          # a thread count decides nothing
    jd = rec["jpeg_decoders"]
    assert jd["opencv"]["version"] == cv2.__version__ and "libjpeg" in jd["opencv"]["libjpeg"]
    assert jd["opencv"]["dispatched"] and jd["opencv"]["baseline"]
    assert jd["pillow"]["version"] == PIL.__version__ and jd["pillow"]["jpg_codec"]
    assert rec["numpy_fft_dtype"] == "complex128"
    # the same twice, JSON-able, no time / pid / path in it — and the same after I1's lazy
    # scipy.optimize import loaded scipy's own OpenBLAS (it is loaded by the record itself)
    assert rec == St.cpu_environment_record()
    from scipy.optimize import least_squares
    least_squares(lambda x: x - 1.0, np.zeros(2), method="trf")
    assert rec == St.cpu_environment_record()
    assert len(rec["blas"]) >= 2 and len({b["library"] for b in rec["blas"]}) == len(rec["blas"])
    txt = json.dumps(rec)
    assert "updated_at" not in txt and str(SERVER) not in txt


def test_opencv_build_record_fails_without_a_jpeg_line(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "getBuildInformation", lambda: "  PNG: libpng\n")
    with pytest.raises(St.IntakeStampError, match="JPEG"):
        St.opencv_build_record()


# ── paths and frames ────────────────────────────────────────────────────────────────────────

def test_paths_are_session_relative_and_an_outsider_fails(tmp_path):
    sess = tmp_path / "s"
    fd = _frames(sess)
    assert St.rel_to_session(fd / "000003.jpg", sess) == "frames/000003.jpg"
    assert St.in_session("frames/000003.jpg", sess) == fd / "000003.jpg"
    with pytest.raises(St.IntakeStampError, match="not inside"):
        St.rel_to_session(tmp_path / "other.jpg", sess)


def test_frames_are_stamped_by_bytes_under_relative_keys(tmp_path):
    sess = tmp_path / "s"
    fd = _frames(sess)
    st = St.frames_stamp(fd, sess)
    assert set(st["inputs"]) == {"frames/000000.jpg", "frames/000003.jpg", "frames/000006.jpg"}
    assert st["inputs"]["frames/000003.jpg"] == repro.sha256_file(fd / "000003.jpg")
    assert st["code"] == {} and st["config"] == {}
    # a copied session: the same stamp
    dst = tmp_path / "copy" / "s"
    shutil.copytree(sess, dst)
    assert St.frames_stamp(dst / "frames", dst) == st
    # one byte changed at the same size: another stamp, the frame named
    (fd / "000003.jpg").write_bytes(bytes([1, 7, 8]) * 10)
    st2 = St.frames_stamp(fd, sess)
    assert st2["sha256"] != st["sha256"]
    assert St.stamp_differences(st, st2) == [
        f"input 'frames/000003.jpg' changed ({st['inputs']['frames/000003.jpg'][:12]} -> "
        f"{st2['inputs']['frames/000003.jpg'][:12]})"]
    # a subset by name, and a missing frame fails
    sub = St.frame_inputs(fd, sess, files=["000000.jpg"])
    assert list(sub) == ["frames/000000.jpg"]
    with pytest.raises(St.IntakeStampError, match="000099"):
        St.frame_inputs(fd, sess, files=["000099.jpg"])
    with pytest.raises(St.IntakeStampError):
        St.frames_stamp(tmp_path / "nowhere")


def test_step_stamps_are_keyed_on_the_intake_and_da3_code(tmp_path):
    names = {p.name for p in St.INTAKE_CODE_FILES}
    assert {"quality.py", "parallax.py", "content.py", "run.py", "focal.py", "walk.py", "vram.py",
            "stamps.py", "run_config.py", "focal_probe.py", "extract_da3_depth.py", "repro.py",
            "da3_weights.py", "card_table.py"} <= names
    assert all(p.is_file() for p in St.INTAKE_CODE_FILES)
    st = St.step_stamp({}, {"params": {"a": 1}})
    assert "server/intake/quality.py" in st["code"] and "server/extract_da3_depth.py" in st["code"]
    assert st["code"]["server/intake/quality.py"] == repro.sha256_file(SERVER / "intake" / "quality.py")
    assert set(st["config"]) == {"params"}
    # another parameter: only that section differs
    st2 = St.step_stamp({}, {"params": {"a": 2}})
    assert St.stamp_differences(st, st2) == [
        f"config 'params' changed ({st['config']['params'][:12]} -> {st2['config']['params'][:12]})"]
    assert St.stamp_differences(None, st) == ["no stamp saved with the product (or it is unreadable)"]
