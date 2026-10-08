"""The scan's frames are SEALED (docs/plan_determinismo.md points 67 / 76): every writer fills
<scan>/frames.extracting/ and renames it to frames/ only when complete, with frames/manifest.json
(decoder, encoder, the video's sha256, every frame's sha256) written last; the bytes of the
manifest are deterministic; a duplicate frame number is refused; an interrupted extraction
leaves the temp dir, which blocks every reader until the next upload removes it; a scan that has
frames refuses an upload; a manifest-less frames/ is adopted (DECLARED) and checked against the
frames on disk. Plus the source-level guarantees of the writers in main.py, which these tests do
not import (the backend module loads the whole server)."""

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

import repro                                                       # noqa: E402
from intake import frames_manifest as FM                           # noqa: E402
from intake import quality as Q                                    # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
W, H, N = 64, 48, 7


def _video(path: Path, n: int = N, seed: int = 0) -> Path:
    """A small MJPG .avi of ``n`` noise frames (OpenCV's own writer)."""
    rng = np.random.default_rng(seed)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (W, H))
    assert w.isOpened()
    for _ in range(n):
        w.write(rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    w.release()
    return path


def _scan(tmp_path: Path, name: str = "src_default") -> Path:
    s = tmp_path / "scan" / name
    (s / "frames").mkdir(parents=True)               # ProjectPaths.ensure_source_dirs makes it empty
    (s / "output").mkdir()
    return s


# ── the extraction ──────────────────────────────────────────────────────────────────────────

def test_extraction_is_sealed_complete_and_identical_to_the_legacy_frames(tmp_path):
    s = _scan(tmp_path)
    v = _video(s / "source_video.avi")
    doc = FM.extract_video(v, s, log=lambda *a: None)
    assert not FM.extracting_dir(s).exists()                          # renamed, nothing left
    names = sorted(p.name for p in (s / "frames").iterdir())
    assert names == [f"{i:06d}.jpg" for i in range(N)] + [FM.MANIFEST_NAME]
    on_disk = json.loads((s / "frames" / FM.MANIFEST_NAME).read_text())
    assert on_disk == doc and doc["complete"] is True and doc["origin"] == FM.ORIGIN_VIDEO
    assert doc["video"] == {"name": "source_video.avi", "size": v.stat().st_size,
                            "sha256": repro.sha256_file(v)}
    assert doc["n_frames"] == N and doc["frame_count_reported"] == N and doc["frames_decoded"] == N
    for r in doc["frames"]:
        assert r["sha256"] == repro.sha256_file(s / "frames" / r["name"])
    dec, enc = doc["decoder"], doc["encoder"]
    assert dec["backend"] == "FFMPEG" and dec["api_preference"] == "CAP_FFMPEG"
    assert dec["orientation_auto"] == 0.0 and "orientation_meta" in dec
    assert dec["opencv_version"] == cv2.__version__ and "avcodec" in dec["video_io"]
    assert enc["params"]["IMWRITE_JPEG_QUALITY"] == FM.JPEG_QUALITY == 95
    assert "libjpeg" in enc["jpeg_library"] or "jpeg" in enc["jpeg_library"].lower()
    # the bytes are the old extractor's (default VideoCapture + cv2.imwrite quality 95)
    cap = cv2.VideoCapture(str(v))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        legacy = tmp_path / "legacy.jpg"
        cv2.imwrite(str(legacy), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert legacy.read_bytes() == (s / "frames" / f"{i:06d}.jpg").read_bytes()
        i += 1
    cap.release()
    assert i == N


def test_manifest_bytes_are_deterministic_and_carry_no_clock_or_absolute_path(tmp_path):
    a, b = _scan(tmp_path, "src_a"), _scan(tmp_path, "src_b")
    _video(a / "source_video.avi")
    shutil.copy(a / "source_video.avi", b / "source_video.avi")
    FM.extract_video(a / "source_video.avi", a, log=lambda *x: None)
    FM.extract_video(b / "source_video.avi", b, log=lambda *x: None)
    ma, mb = (x / "frames" / FM.MANIFEST_NAME for x in (a, b))
    assert ma.read_bytes() == mb.read_bytes()
    txt = ma.read_text()
    assert str(tmp_path) not in txt and "time" not in json.loads(txt)
    assert txt == FM.manifest_bytes(json.loads(txt)).decode()          # sorted keys, fixed form


def test_stride_and_max_frames_keep_the_decode_order_numbers(tmp_path):
    s = _scan(tmp_path)
    v = _video(s / "rgb.avi", n=9)
    doc = FM.extract_video(v, s, origin=FM.ORIGIN_STRAY, stride=4, log=lambda *a: None)
    assert [r["name"] for r in doc["frames"]] == ["000000.jpg", "000004.jpg", "000008.jpg"]
    assert doc["decoder"]["stride"] == 4 and doc["frames_decoded"] == 9


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI to tag rotation")
def test_orientation_policy_is_pinned_unrotated(tmp_path):
    """A video carrying a 90° display matrix decodes UNROTATED (CAP_PROP_ORIENTATION_AUTO 0, set
    explicitly) — how bufferStop's and fosa_pan's frames on disk are — and the manifest records
    the container's orientation metadata."""
    src = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    for k in range(3):
        w.write(np.full((H, W, 3), 40 * k, np.uint8))
    w.release()
    s = _scan(tmp_path)
    rot = s / "source_video.mp4"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i",
                        str(src), "-c", "copy", str(rot)], capture_output=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"this ffmpeg cannot tag a display rotation: {r.stderr[-200:]!r}")
    doc = FM.extract_video(rot, s, log=lambda *a: None)
    assert doc["decoder"]["orientation_meta"] != 0.0 and doc["decoder"]["orientation_auto"] == 0.0
    assert cv2.imread(str(s / "frames" / "000000.jpg")).shape[:2] == (H, W)   # not rotated


def test_an_interrupted_extraction_blocks_every_reader_until_the_next_upload(tmp_path, monkeypatch):
    s = _scan(tmp_path)
    v = _video(s / "source_video.avi")
    real = cv2.imencode
    calls = {"n": 0}

    def dies_at_frame_3(ext, img, params):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt("backend killed mid-extraction")
        return real(ext, img, params)

    monkeypatch.setattr(cv2, "imencode", dies_at_frame_3)
    with pytest.raises(KeyboardInterrupt):
        FM.extract_video(v, s, log=lambda *a: None)
    monkeypatch.setattr(cv2, "imencode", real)
    # frames/ never saw a frame; the temp dir holds the partial set and no manifest
    assert FM.frames_present(s) == {"n_frames": 0, "manifest": False, "other": []}
    assert sorted(p.name for p in FM.extracting_dir(s).iterdir()) == \
        ["000000.jpg", "000001.jpg", "000002.jpg"]
    why = FM.extraction_in_progress(s)
    assert why and FM.EXTRACTING_DIRNAME in why
    with pytest.raises(FM.FramesManifestError, match="cannot be read"):
        FM.check_ready(s)
    with pytest.raises(FM.FramesManifestError, match="not adopted"):
        FM.adopt(s, {"000000.jpg": "0" * 64})
    with pytest.raises(FM.FramesManifestError, match="interrupted"):
        FM.extract_video(v, s, log=lambda *a: None)
    # the next upload removes it (it holds the writer claim), then extracts sealed
    token = FM.claim_writer(s, FM.ORIGIN_VIDEO)
    assert FM.discard_partial(s) == [FM.EXTRACTING_DIRNAME + "/"]
    doc = FM.extract_video(v, s, log=lambda *a: None)
    FM.release_writer(s, token)
    assert doc["n_frames"] == N and FM.extraction_in_progress(s) is None
    assert FM.check_ready(s) == doc


def test_a_writer_claim_refuses_readers_and_other_writers(tmp_path):
    s = _scan(tmp_path)
    t = FM.claim_writer(s, FM.ORIGIN_VIDEO)
    try:
        assert "being written" in FM.extraction_in_progress(s)
        with pytest.raises(FM.FramesManifestError, match="being written"):
            FM.claim_writer(s, FM.ORIGIN_VIDEO)              # a second upload
        with pytest.raises(FM.FramesManifestError, match="being written"):
            FM.claim_writer(s, FM.ORIGIN_WEBXR)
        with pytest.raises(FM.FramesManifestError, match="being written"):
            FM.check_ready(s)
    finally:
        FM.release_writer(s, t)
    assert FM.extraction_in_progress(s) is None
    # a WebXR capture that reconnects takes its own claim over; the old token releases nothing
    t1 = FM.claim_writer(s, FM.ORIGIN_WEBXR)
    t2 = FM.claim_writer(s, FM.ORIGIN_WEBXR)
    FM.release_writer(s, t1)
    assert FM.extraction_in_progress(s)
    FM.release_writer(s, t2)
    assert FM.extraction_in_progress(s) is None


def test_upload_is_refused_into_a_scan_that_has_frames(tmp_path):
    s = _scan(tmp_path)
    assert FM.upload_refusal(s) is None                               # empty frames/: fine
    _video(s / "source_video.avi")
    FM.extract_video(s / "source_video.avi", s, log=lambda *a: None)
    why = FM.upload_refusal(s)
    assert why and f"already has {N} frame(s)" in why
    s2 = _scan(tmp_path, "src_b")
    (s2 / "frames" / "selected_frames.json").write_text("{}")
    assert "holds no frame but 1 other file" in FM.upload_refusal(s2)
    with pytest.raises(FM.FramesManifestError, match="already holds"):
        FM.extract_video(s / "source_video.avi", s, log=lambda *a: None)


# ── duplicates ──────────────────────────────────────────────────────────────────────────────

def test_duplicate_frame_numbers_are_refused_by_writers_and_readers(tmp_path):
    s = _scan(tmp_path)
    sealer = FM.FrameSealer(s)
    sealer.add(3, b"a", ".jpg")
    with pytest.raises(FM.FramesManifestError, match="duplicate"):
        sealer.add(3, b"b", ".png")                                   # same number, other suffix
    with pytest.raises(FM.FramesManifestError, match="negative"):
        sealer.add(-1, b"c")
    # a continued temp dir keeps its numbers taken (a WebXR capture that reconnected)
    again = FM.FrameSealer(s)
    with pytest.raises(FM.FramesManifestError, match="duplicate"):
        again.add(3, b"d")
    again.add(4, b"e")
    doc = again.seal(origin=FM.ORIGIN_WEBXR)
    assert [r["name"] for r in doc["frames"]] == ["000003.jpg", "000004.jpg"]
    assert (s / "frames" / "000003.jpg").read_bytes() == b"a"
    # the intake's lister refuses two files with one number (iterdir order used to decide)
    d = tmp_path / "f"
    d.mkdir()
    for n in ("000001.jpg", "000002.jpg", "000002.png"):
        (d / n).write_bytes(b"x")
    with pytest.raises(Q.QualityError, match="share video frame number 2: 000002.jpg, 000002.png"):
        Q.list_frames(d)
    (d / "000002.png").unlink()
    (d / "2.jpg").write_bytes(b"x")
    with pytest.raises(Q.QualityError, match="share video frame number 2"):
        Q.list_frames(d)
    assert FM.duplicate_numbers(["1.jpg", "000001.jpg", "000002.jpg"]) == \
        {1: ["000001.jpg", "1.jpg"]}


def test_sealing_never_overwrites_a_frames_dir_that_is_not_empty(tmp_path):
    s = _scan(tmp_path)
    (s / "frames" / "stray_note.txt").write_text("x")
    sealer = FM.FrameSealer(s)
    sealer.add(0, b"a")
    with pytest.raises(FM.FramesManifestError, match="not empty"):
        sealer.seal(origin=FM.ORIGIN_VIDEO)
    assert (s / "frames" / "stray_note.txt").exists() and (FM.extracting_dir(s) / "000000.jpg").exists()


# ── adoption and verification ───────────────────────────────────────────────────────────────

def _legacy(tmp_path, n=4, video=True):
    s = _scan(tmp_path)
    for i in range(n):
        (s / "frames" / f"{i:06d}.jpg").write_bytes(bytes([i]) * 10)
    if video:
        (s / "source_video.mp4").write_bytes(b"video bytes")
    return s


def test_legacy_frames_are_adopted_declared_and_then_verified(tmp_path):
    s = _legacy(tmp_path)
    assert FM.check_ready(s) is None                                  # predates manifests
    shas = {p.name: repro.sha256_file(p) for p in FM.frame_files(s / "frames")}
    log = []
    doc = FM.adopt(s, shas, log=log.append)
    assert doc["origin"] == FM.ORIGIN_ADOPTED and doc["decoder"] is None and doc["encoder"] is None
    assert doc["frame_count_reported"] is None and "unknown" in doc["declared"]
    assert doc["video"] == {"name": "source_video.mp4", "size": 11,
                            "sha256": repro.sha256_bytes(b"video bytes")}
    assert any("DECLARED" in m and "ADOPTED" in m for m in log)
    assert FM.check_ready(s) == doc
    with pytest.raises(FM.FramesManifestError, match="never adopted"):
        FM.adopt(s, shas)
    FM.verify_shas(doc, shas)                                         # the same frames pass
    with pytest.raises(FM.FramesManifestError, match="000001.jpg differs from the manifest"):
        FM.verify_shas(doc, dict(shas, **{"000001.jpg": "f" * 64}))
    with pytest.raises(FM.FramesManifestError, match="000009.jpg is on disk but not in"):
        FM.verify_names(doc, list(shas) + ["000009.jpg"])
    with pytest.raises(FM.FramesManifestError, match="000002.jpg is in the manifest but not on disk"):
        FM.verify_names(doc, [n for n in shas if n != "000002.jpg"])


def test_adoption_without_a_video_and_a_broken_manifest(tmp_path):
    s = _legacy(tmp_path, video=False)
    doc = FM.adopt(s, {p.name: repro.sha256_file(p) for p in FM.frame_files(s / "frames")},
                   log=lambda *a: None)
    assert doc["video"] is None and doc["videos_present"] == []
    p = FM.manifest_path(s)
    p.write_text(json.dumps(dict(doc, complete=False)))
    with pytest.raises(FM.FramesManifestError, match="not a complete manifest"):
        FM.check_ready(s)
    p.write_text("{ not json")
    with pytest.raises(FM.FramesManifestError, match="unreadable"):
        FM.check_ready(s)
    p.write_text(json.dumps(dict(doc, manifest_version=99)))
    with pytest.raises(FM.FramesManifestError, match="version 99"):
        FM.check_ready(s)


# ── the Stray rgb.mp4 writer ────────────────────────────────────────────────────────────────

def test_stray_extraction_is_sealed_and_never_rewrites_existing_frames(tmp_path):
    from ingestors import stray_scanner as SS
    s = _scan(tmp_path)
    (s / "inputs" / "stray").mkdir(parents=True)
    v = _video(s / "inputs" / "stray" / "rgb.avi", n=6)
    n = SS.extract_frames(str(v), str(s / "frames"), stride=2)
    doc = FM.load_manifest(s)
    assert n == 3 and doc["origin"] == FM.ORIGIN_STRAY and doc["video"]["name"] == "inputs/stray/rgb.avi"
    with pytest.raises(FM.FramesManifestError, match="already holds"):
        SS.extract_frames(str(v), str(s / "frames"), stride=2)
    with pytest.raises(ValueError, match="not a scan's frames/"):
        SS.extract_frames(str(v), str(tmp_path / "elsewhere"), stride=2)
    assert FM.extraction_in_progress(s) is None                      # its claim was released


# ── the writers in main.py (source level: the backend module is not imported here) ─────────

def _fn(tree, name):
    return next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def test_main_writers_go_through_the_sealer():
    src = (SERVER / "main.py").read_text()
    tree = ast.parse(src)
    ext = ast.get_source_segment(src, _fn(tree, "_extract_video_frames_sync"))
    assert "FM.extract_video(" in ext and "imwrite" not in ext and "VideoCapture" not in ext
    up = ast.get_source_segment(src, _fn(tree, "upload_video"))
    # refused before anything is written; the temp dir exists before the video is streamed
    assert up.index("FM.upload_refusal(") < up.index("FM.claim_writer(") < \
        up.index("FM.extracting_dir(scan_dir).mkdir(") < up.index("await file.read(")
    assert "status_code=409, detail=refusal" in up and "FM.release_writer(" in up
    assert "FM.UPLOADING_VIDEO_NAME" in up and "os.replace(uploading, video_path)" in up
    ws = ast.get_source_segment(src, _fn(tree, "scan_websocket"))
    assert "FM.FrameSealer(" in ws and "sealer.add(" in ws and "sealer.seal(" in ws
    assert 'frames_dir / f"{filename}.jpg"' not in ws and "webxr_inputs_dir" in ws
    cam = ast.get_source_segment(src, _fn(tree, "_camera_frame_capture"))
    assert "frame_storage.add_frame(" not in cam and "refused" in cam
    prog = ast.get_source_segment(src, _fn(tree, "video_extract_progress"))
    assert "FM.extraction_in_progress(" in prog


def test_no_other_server_code_writes_frame_images_into_a_scan_unsealed():
    """Every place that encodes frame images into a scan's frames/ goes through the sealer: the
    video extractor, the Stray extractors (stray_scanner, convert_stray_to_da3) and the WebXR
    socket. A new writer that calls cv2.imwrite on a frames path fails here."""
    for rel in ("ingestors/stray_scanner.py", "convert_stray_to_da3.py"):
        src = (SERVER / rel).read_text()
        assert "FrameSealer" in src or "FM.extract_video(" in src, rel
        assert "cv2.imwrite(str(fname)" not in src and "cv2.imwrite(str(frames_dst)" not in src, rel
