"""The scan's frames are SEALED: ``frames/manifest.json`` says where they came from, how they were
decoded and encoded, and every frame's sha256 (docs/plan_determinismo.md points 67 and 76,
2026-10-07).

Until today a video upload decoded straight into ``frames/`` one JPEG at a time, with whatever
OpenCV / FFmpeg / libjpeg was installed and no record of any of it: a backend restart or a
reconstruction started mid-extraction measured a truncated video (the last JPEG cut), an upload
into a scan that already had frames left the longer video's tail behind, and two files with the
same frame number (``000123.jpg`` + ``000123.png``) were both listed. Now:

- EVERY writer of a scan's frames writes into ONE sibling temp dir, ``<scan>/frames.extracting/``
  (:data:`EXTRACTING_DIRNAME`), through :class:`FrameSealer`, which refuses a duplicate frame
  number; ``frames/manifest.json`` is written LAST inside it and the directory is renamed to
  ``frames/`` (``os.replace``, one syscall) only then. A crash leaves the temp dir, never a
  half-filled ``frames/``; the next upload removes it (:func:`discard_partial`). Writers:
  the video upload (``main.upload_video`` → :func:`extract_video`), the Stray ``rgb.mp4``
  extraction (``ingestors.stray_scanner.extract_frames`` → :func:`extract_video`) and the WebXR
  capture socket (``main.scan_websocket`` → :class:`FrameSealer`). The legacy ``/ws/camera``
  stream (no client in this repo) is refused.
- While a writer is at work the scan is CLAIMED in this process (:func:`claim_writer`) and its
  temp dir exists; :func:`extraction_in_progress` answers both, and the pipeline does not start
  on such a scan (``pipeline_manager``), nor does the intake read it (:func:`check_ready`).
- The manifest pins and records the decoder (OpenCV's version, the FFMPEG backend requested
  explicitly, the video I/O libraries from ``cv2.getBuildInformation()``, the orientation policy
  — the decoder's own rotation OFF (``CAP_PROP_ORIENTATION_AUTO`` 0, set explicitly) and THE
  CONTAINER'S ROTATION APPLIED HERE, explicitly, to every frame (:data:`UPRIGHT`): the frames on
  disk are UPRIGHT, as the video is watched (USER 2026-11-13: "las imagenes de bufferstop, estan
  rotadas, deben estar correctamente orientadas … verifica que siempre no solo en esta sesion …
  queden correctamente orientadas"). Until 2026-11-14 the rotation was only recorded: a video
  carrying a 90° display matrix was extracted unrotated, which is how bufferStop's frames of
  2026-08-11 are — and the container's own orientation metadata), the encoder
  (``cv2.imencode('.jpg')`` with every JPEG option given explicitly at the values that produced
  the frames on disk, plus the libjpeg build), the reported vs decoded frame count, the video's
  name / size / sha256 and each frame's name + sha256. Its bytes are deterministic: sorted keys,
  no wall clock, no absolute path (times go to the log only — nothing is written beside the
  frames that would enter the precision chain's ``frames`` stamp).
- A ``frames/`` without a manifest can only predate this module (no writer here fills
  ``frames/`` in place any more): the intake ADOPTS it once (:func:`adopt`, origin
  ``adopted``, decoder / encoder unknown — said so — each frame's sha256 taken from the I0
  stamp the intake computes anyway, the video's sha256 when ``source_video.*`` exists), logged
  as DECLARED. Adoption is refused while a writer is at work or the temp dir exists.
- The intake checks the frames on disk against the manifest — names and count first, then the
  sha256 of every frame from its own I0 stamp (no second hashing pass) — and fails naming the
  first difference (:func:`verify_names`, :func:`verify_shas`).
"""

from __future__ import annotations

import json
import os
import shutil
import itertools
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FRAMES_DIRNAME = "frames"
EXTRACTING_DIRNAME = "frames.extracting"       # the ONE temp dir every frame writer fills
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
UPLOADING_VIDEO_NAME = ".source_video.uploading"   # an upload in flight (not 'source_video*': the
                                                    # replace wipe keeps that prefix)
SOURCE_VIDEO_STEM = "source_video"
# the frame images a scan holds (the same suffixes the intake lists, intake.quality.FRAME_SUFFIXES)
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")

ORIGIN_VIDEO = "video_extraction"        # main.upload_video
ORIGIN_STRAY = "stray_rgb_video"         # ingestors.stray_scanner.extract_frames (rgb.mp4)
ORIGIN_WEBXR = "webxr_capture"           # main.scan_websocket: the client's JPEG bytes as sent
ORIGIN_ADOPTED = "adopted"               # a frames/ that predates the manifest, adopted once

# The JPEG quality every frame on disk was encoded with: the extractor's value since it existed
# (main._extract_video_frames_sync and ingestors.stray_scanner.extract_frames, both 95).
JPEG_QUALITY = 95

# Writer kinds whose claim a new writer of the SAME kind takes over: a WebXR capture socket that
# reconnects (camera.html reconnects on its own mid-capture) continues its own capture while the
# server may not yet have noticed the old socket died. A video upload is never taken over.
RECLAIMABLE_KINDS = frozenset({ORIGIN_WEBXR})


class FramesManifestError(RuntimeError):
    """A scan's frames cannot be written, adopted or trusted — always with the exact reason."""


# ── paths ───────────────────────────────────────────────────────────────────────────────────

def frames_dir(scan_dir: os.PathLike) -> Path:
    return Path(scan_dir) / FRAMES_DIRNAME


def extracting_dir(scan_dir: os.PathLike) -> Path:
    return Path(scan_dir) / EXTRACTING_DIRNAME


def manifest_path(scan_dir: os.PathLike) -> Path:
    return frames_dir(scan_dir) / MANIFEST_NAME


def _key(scan_dir: os.PathLike) -> str:
    return str(Path(scan_dir).resolve())


# ── who is writing a scan's frames (this process) ───────────────────────────────────────────

_LOCK = threading.Lock()
_WRITERS: Dict[str, Tuple[str, str]] = {}          # resolved scan dir -> (kind, token)
_TOKENS = itertools.count(1)                       # claim tokens: in memory only, never written


def claim_writer(scan_dir: os.PathLike, kind: str) -> str:
    """Claim the scan's frames for a writer of ``kind``; returns the token :func:`release_writer`
    takes. FAILS when another writer holds the claim — except a writer of the same kind in
    :data:`RECLAIMABLE_KINDS`, which takes it over."""
    k = _key(scan_dir)
    with _LOCK:
        cur = _WRITERS.get(k)
        if cur is not None and not (cur[0] == kind and kind in RECLAIMABLE_KINDS):
            raise FramesManifestError(f"the frames of {scan_dir} are being written right now "
                                      f"({cur[0]}) — wait until it finishes")
        token = f"{kind}#{next(_TOKENS)}"
        _WRITERS[k] = (kind, token)
        return token


def release_writer(scan_dir: os.PathLike, token: str) -> None:
    """Release a claim — only the holder's token releases it (a taken-over claim stays)."""
    k = _key(scan_dir)
    with _LOCK:
        cur = _WRITERS.get(k)
        if cur is not None and cur[1] == token:
            del _WRITERS[k]


def extraction_in_progress(scan_dir: os.PathLike) -> Optional[str]:
    """Why the scan's frames are not to be read now, or None: a writer of this process holds the
    claim, or the temp dir exists (an extraction running elsewhere, or one a crash or a restart
    left — the next upload removes it)."""
    with _LOCK:
        cur = _WRITERS.get(_key(scan_dir))
    if cur is not None:
        return f"the frames are being written right now ({cur[0]})"
    tmp = extracting_dir(scan_dir)
    if tmp.exists():
        return (f"{tmp} exists: a frame extraction is in progress, or one was interrupted "
                f"(a backend restart mid-extraction) — wait for it, or upload the video again "
                f"(the upload removes the interrupted one)")
    return None


def discard_partial(scan_dir: os.PathLike) -> List[str]:
    """Remove what an interrupted writer left (the temp dir, a video upload in flight). Only for
    a caller that holds the writer claim. Returns what it removed."""
    gone = []
    tmp = extracting_dir(scan_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
        gone.append(EXTRACTING_DIRNAME + "/")
    up = Path(scan_dir) / UPLOADING_VIDEO_NAME
    if up.exists():
        up.unlink()
        gone.append(UPLOADING_VIDEO_NAME)
    return gone


# ── frame names ─────────────────────────────────────────────────────────────────────────────

def frame_number(name: str) -> int:
    """The video frame number a frame file is named by (``<frame:06d>.jpg``)."""
    stem, suffix = os.path.splitext(name)
    if suffix.lower() not in FRAME_SUFFIXES:
        raise FramesManifestError(f"{name} is not a frame image ({', '.join(FRAME_SUFFIXES)})")
    try:
        return int(stem)
    except ValueError:
        raise FramesManifestError(f"frame file {name} is not named by its video frame number "
                                  f"(<frame:06d>.jpg)") from None


def frame_files(directory: os.PathLike) -> List[Path]:
    """The frame images of ``directory`` (any frame suffix), sorted by name."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES)


def duplicate_numbers(names: Iterable[str]) -> Dict[int, List[str]]:
    """{frame number: [names]} of every number two or more files share."""
    by: Dict[int, List[str]] = {}
    for n in names:
        by.setdefault(frame_number(n), []).append(n)
    return {k: sorted(v) for k, v in sorted(by.items()) if len(v) > 1}


def frames_present(scan_dir: os.PathLike) -> Dict[str, Any]:
    """What ``frames/`` holds now: ``n_frames`` (frame images), ``manifest`` (one exists),
    ``other`` (the names of anything else in it)."""
    d = frames_dir(scan_dir)
    if not d.is_dir():
        return {"n_frames": 0, "manifest": False, "other": []}
    n, other = 0, []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES:
            n += 1
        elif p.name != MANIFEST_NAME:
            other.append(p.name)
    return {"n_frames": n, "manifest": (d / MANIFEST_NAME).exists(), "other": other}


def upload_refusal(scan_dir: os.PathLike) -> Optional[str]:
    """Why a video cannot be uploaded into the scan (the upload's 409), or None: the scan already
    holds frames (or a manifest) — a video goes into a scan WITHOUT frames only, so the frames
    and the video they come from are never mixed (point 67) — or its ``frames/`` holds other
    files. A writer at work is refused by :func:`claim_writer`; an interrupted extraction's temp
    dir is not a refusal (the upload removes it)."""
    have = frames_present(scan_dir)
    if have["n_frames"] or have["manifest"]:
        return (f"This scan already has {have['n_frames']} frame(s) — a video is uploaded into a "
                f"scan without frames only (the frames and the video they come from are never "
                f"mixed). Upload it into a new scan.")
    if have["other"]:
        return (f"frames/ of this scan holds no frame but {len(have['other'])} other file(s) "
                f"({', '.join(have['other'][:5])}) — remove them before uploading a video")
    return None


# ── what the frames were made with ──────────────────────────────────────────────────────────

def _sha256_file(path: os.PathLike) -> str:
    import repro
    return repro.sha256_file(path)


def _sha256_bytes(data: bytes) -> str:
    import repro
    return repro.sha256_bytes(data)


def video_record(video_path: os.PathLike, scan_dir: os.PathLike) -> Dict[str, Any]:
    """The video the frames come from: its path relative to the scan, its size, its sha256."""
    p = Path(video_path)
    try:
        name = p.resolve().relative_to(Path(scan_dir).resolve()).as_posix()
    except ValueError:
        raise FramesManifestError(f"{p} is not inside the scan {scan_dir} — the manifest names "
                                  f"its video relative to the scan") from None
    return {"name": name, "size": int(p.stat().st_size), "sha256": _sha256_file(p)}


def video_io_build_record() -> Dict[str, str]:
    """The ``Video I/O:`` section of ``cv2.getBuildInformation()`` (FFMPEG and the avcodec /
    avformat / avutil / swscale versions OpenCV decodes with), as {name: value}."""
    import cv2
    out: Dict[str, str] = {}
    inside = False
    for ln in cv2.getBuildInformation().splitlines():
        s = ln.strip()
        if s.startswith("Video I/O:"):
            inside = True
            continue
        if inside:
            if not s:
                break
            if ":" in s:
                k, v = s.split(":", 1)
                out[k.strip()] = v.strip()
    if "FFMPEG" not in out:
        raise FramesManifestError("cv2.getBuildInformation() has no FFMPEG line in 'Video I/O' — "
                                  "the decoder of the frames cannot be recorded")
    return out


def jpeg_params(cv2) -> List[int]:
    """Every JPEG option of the encoder, given EXPLICITLY at the value that produced the frames on
    disk (quality :data:`JPEG_QUALITY`; progressive, optimize and restart interval off; 4:2:0
    chroma — OpenCV 4.11.0's defaults: measured 2026-10-07, these options give the same bytes as
    ``cv2.imwrite(path, frame, [IMWRITE_JPEG_QUALITY, 95])``), so a change of OpenCV's defaults
    cannot change a frame silently."""
    return [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
            cv2.IMWRITE_JPEG_RST_INTERVAL, 0,
            cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]


def jpeg_encoder_record() -> Dict[str, Any]:
    """The encoder of the frames: the call, its options and the libjpeg OpenCV was built with."""
    import cv2
    from intake.stamps import opencv_build_record
    return {"library": "opencv", "opencv_version": str(cv2.__version__),
            "call": "cv2.imencode('.jpg', frame, params); the bytes are written unchanged",
            "params": {"IMWRITE_JPEG_QUALITY": JPEG_QUALITY, "IMWRITE_JPEG_PROGRESSIVE": 0,
                       "IMWRITE_JPEG_OPTIMIZE": 0, "IMWRITE_JPEG_RST_INTERVAL": 0,
                       "IMWRITE_JPEG_SAMPLING_FACTOR": "IMWRITE_JPEG_SAMPLING_FACTOR_420"},
            "jpeg_library": opencv_build_record()["jpeg"]}


# The orientation policy. The DECODER's automatic rotation stays OFF (set explicitly: OpenCV
# 4.11.0's default, measured 2026-10-07) — the frames it delivers are the video's coded pixels —
# and the container's rotation (``CAP_PROP_ORIENTATION_META``, degrees) is APPLIED HERE to every
# frame with ``cv2.rotate``: an exact permutation of the pixels, recorded in the manifest.
# USER 2026-11-13: "las imagenes de bufferstop, estan rotadas, deben estar correctamente
# orientadas … verifica que siempre … queden correctamente orientadas".
# MEASURED 2026-11-14 on bufferStop's video (meta 90): cv2.rotate(frame, ROTATE_90_CLOCKWISE) of
# the unrotated decode IS the decoder's own automatic rotation, bit for bit; 180 and 270 are held
# equal to it by the tests on tagged videos. A video with no rotation (every other session) is
# extracted exactly as before.
ORIENTATION_AUTO = 0
UPRIGHT = {0: None, 90: "ROTATE_90_CLOCKWISE", 180: "ROTATE_180", 270: "ROTATE_90_COUNTERCLOCKWISE"}


def upright_rotation(orientation_meta: float) -> Optional[str]:
    """The ``cv2.rotate`` flag (its name) that sets a frame of a video with this container
    rotation upright, None for no rotation. FAILS on an angle that is not a quarter turn."""
    deg = int(round(float(orientation_meta))) % 360
    if deg not in UPRIGHT:
        raise FramesManifestError(f"the video's rotation is {orientation_meta!r}°: not a quarter turn "
                                  f"— it cannot be set upright by a rotation of the pixels")
    return UPRIGHT[deg]


# ── scans whose frames were stored sideways: upright FOR DISPLAY ─────────────────────────────
# USER 2026-11-14, on bufferStop (extracted before the frames were set upright, every product
# computed on its sideways frames): "solo gira las imagenes cuando se muestran en las poses de
# camara en el viewer y en el video nomas, no hace falta mas nada". The stored frames and every
# product stay as they are; what is SHOWN is turned.
DEGREES_CLOCKWISE = {"ROTATE_90_CLOCKWISE": 90, "ROTATE_180": 180, "ROTATE_90_COUNTERCLOCKWISE": 270}
_DISPLAY_MEMO: Dict[tuple, Optional[str]] = {}


def display_rotation(scan_dir: os.PathLike) -> Optional[str]:
    """The ``cv2.rotate`` flag (its name) that sets this scan's STORED frames upright for
    display; None when they are upright already — extracted upright (the manifest says
    ``orientation_applied``), or from a video with no rotation. A scan with no manifest (its
    frames predate them) is answered by its source video's own rotation. Memoised on the
    manifest's and the video's stamps."""
    scan_dir = Path(scan_dir)
    mp, videos = manifest_path(scan_dir), sorted(scan_dir.glob("source_video.*"))

    def _stamp(p: Path):
        try:
            st = p.stat()
            return (str(p), st.st_size, st.st_mtime_ns)
        except OSError:
            return (str(p), None, None)
    key = (_stamp(mp), tuple(_stamp(v) for v in videos))
    if key in _DISPLAY_MEMO:
        return _DISPLAY_MEMO[key]
    meta: Optional[float] = None
    try:
        man = load_manifest(scan_dir)
    except FramesManifestError:
        man = None
    if man is not None:
        dec = man.get("decoder") or {}
        if dec.get("orientation_applied"):
            _DISPLAY_MEMO[key] = None
            return None
        if isinstance(dec.get("orientation_meta"), (int, float)):
            meta = float(dec["orientation_meta"])
    if meta is None and videos:
        import cv2
        cap = cv2.VideoCapture(str(videos[0]), cv2.CAP_FFMPEG)
        try:
            if cap.isOpened():
                meta = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        finally:
            cap.release()
    try:
        flag = upright_rotation(meta) if meta is not None else None
    except FramesManifestError:
        flag = None                                   # not a quarter turn: shown as stored
    _DISPLAY_MEMO[key] = flag
    return flag


def upright_jpeg(frame_path: os.PathLike, flag: str, quality: int = 90) -> bytes:
    """One stored frame turned upright, as JPEG bytes — for DISPLAY only (the frame on disk,
    which every product was computed on, is not touched)."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(Path(frame_path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FramesManifestError(f"{frame_path} cannot be decoded")
    ok, buf = cv2.imencode(".jpg", cv2.rotate(img, getattr(cv2, flag)), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise FramesManifestError(f"{frame_path} could not be encoded upright")
    return buf.tobytes()


# ── the manifest's bytes ────────────────────────────────────────────────────────────────────

def manifest_bytes(doc: Mapping[str, Any]) -> bytes:
    """Deterministic: sorted keys, ASCII, fixed indent, one trailing newline."""
    return (json.dumps(doc, sort_keys=True, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def _frames_block(shas: Mapping[str, str]) -> Dict[str, Any]:
    """``n_frames``, ``frames`` (name, number, sha256 — by frame number) and ``frames_sha256``
    (the inventory's digest: canonical JSON of the [name, sha256] list)."""
    import repro
    dups = duplicate_numbers(shas)
    if dups:
        k, v = next(iter(dups.items()))
        raise FramesManifestError(f"two frame files share frame number {k}: {', '.join(v)}")
    rows = sorted(((frame_number(n), n, s) for n, s in shas.items()))
    return {"n_frames": len(rows),
            "frames": [{"name": n, "number": k, "sha256": s} for k, n, s in rows],
            "frames_sha256": repro.sha256_json([[n, s] for _k, n, s in rows])}


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# ── writing: the sealer ─────────────────────────────────────────────────────────────────────

class FrameSealer:
    """Writes a scan's frames into the temp dir and seals them: ``add`` one frame (a duplicate
    frame number is REFUSED), ``seal`` writes the manifest LAST and renames the temp dir to
    ``frames/``. Opening on an existing temp dir continues it (a WebXR capture that reconnected);
    the frames already in it are hashed once so the manifest covers them and their numbers stay
    taken. No per-frame fsync: a frame lost by a kernel crash after the rename is caught by the
    intake, which verifies every frame's sha256 against the manifest."""

    def __init__(self, scan_dir: os.PathLike):
        self.scan_dir = Path(scan_dir)
        self.tmp = extracting_dir(self.scan_dir)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.shas: Dict[str, str] = {}
        self.numbers: Dict[int, str] = {}
        for p in frame_files(self.tmp):
            self._take(p.name)
            self.shas[p.name] = _sha256_file(p)

    def _take(self, name: str) -> int:
        k = frame_number(name)
        if k in self.numbers:
            raise FramesManifestError(f"frame number {k} is already taken by "
                                      f"{self.numbers[k]} — {name} is refused (duplicate)")
        self.numbers[k] = name
        return k

    def add(self, number: int, data: bytes, suffix: str = ".jpg") -> str:
        number = int(number)
        if number < 0:
            raise FramesManifestError(f"frame number {number} is negative")
        name = f"{number:06d}{suffix}"
        self._take(name)
        with open(self.tmp / name, "wb") as f:
            f.write(data)
        self.shas[name] = _sha256_bytes(data)
        return name

    def seal(self, *, origin: str, video: Optional[Mapping[str, Any]] = None,
             decoder: Optional[Mapping[str, Any]] = None,
             encoder: Optional[Mapping[str, Any]] = None,
             extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if not self.shas:
            raise FramesManifestError(f"no frame was written into {self.tmp} — nothing to seal")
        doc: Dict[str, Any] = {"manifest_version": MANIFEST_VERSION, "complete": True,
                               "origin": origin, "video": dict(video) if video else None,
                               "decoder": dict(decoder) if decoder else None,
                               "encoder": dict(encoder) if encoder else None}
        doc.update(dict(extra or {}))
        doc.update(_frames_block(self.shas))
        _write_atomic(self.tmp / MANIFEST_NAME, manifest_bytes(doc))     # LAST inside the temp dir
        _fsync_dir(self.tmp)
        dst = frames_dir(self.scan_dir)
        if dst.exists():
            leftover = [p.name for p in dst.iterdir()] if dst.is_dir() else [dst.name]
            if leftover:
                raise FramesManifestError(f"{dst} is not empty ({', '.join(sorted(leftover)[:5])}"
                                          f"{' …' if len(leftover) > 5 else ''}) — the sealed "
                                          f"frames stay in {self.tmp}; nothing was overwritten")
        os.replace(self.tmp, dst)                      # an empty frames/ is replaced atomically
        _fsync_dir(self.scan_dir)
        return doc


def reduced_size(w: int, h: int, max_long_side: int) -> Tuple[int, int]:
    """(w, h) of a frame of ``w`` x ``h`` extracted with the long-side cap ``max_long_side`` (USER 2026-11-09, "a 1024
    todo"): unchanged when the cap is 0 or not exceeded, else scaled to the cap (the ONE computation — the extractor
    resizes with it and F0 checks the frames against it)."""
    w, h, cap = int(w), int(h), int(max_long_side)
    if cap <= 0 or max(w, h) <= cap:
        return w, h
    sc = cap / float(max(w, h))
    return max(1, int(round(w * sc))), max(1, int(round(h * sc)))


def extract_video(video_path: os.PathLike, scan_dir: os.PathLike, *, origin: str = ORIGIN_VIDEO,
                  stride: int = 1, max_frames: int = 0, max_long_side: int = 0,
                  progress: Optional[Callable[[int, int], Any]] = None,
                  log: Callable[[str], Any] = print, upright: bool = True) -> Dict[str, Any]:
    """Decode ``video_path`` into the scan's frames, sealed (module doc): frame ``i`` of the
    decode order → ``<i:06d>.jpg`` (only ``i % stride == 0``, at most ``max_frames`` when > 0 —
    the Stray extractor's legacy stride). The decoder is OpenCV's FFMPEG backend, requested
    explicitly, orientation policy :data:`ORIENTATION_AUTO` set and read back; the encoder
    :func:`jpeg_params`. Refused when the scan already holds frames or a manifest, or when the
    temp dir holds anything (an interrupted writer: :func:`discard_partial` first). Returns the
    manifest. ``upright`` False keeps the coded pixels whatever the container's rotation — for a
    capture whose OTHER products are aligned to the coded frame (the Stray scanner's depth maps
    and intrinsics)."""
    import cv2
    scan_dir = Path(scan_dir)
    have = frames_present(scan_dir)
    if have["n_frames"] or have["manifest"]:
        raise FramesManifestError(f"{frames_dir(scan_dir)} already holds {have['n_frames']} frame(s)"
                                  f"{' and a manifest' if have['manifest'] else ''} — frames are "
                                  f"extracted into a scan without frames only")
    if extracting_dir(scan_dir).exists() and any(extracting_dir(scan_dir).iterdir()):
        raise FramesManifestError(f"{extracting_dir(scan_dir)} is not empty — an interrupted "
                                  f"extraction: remove it first (a new upload does)")
    stride, max_frames = int(stride), int(max_frames)
    if stride < 1:
        raise FramesManifestError(f"stride {stride} < 1")
    video = video_record(video_path, scan_dir)
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise FramesManifestError(f"OpenCV's FFMPEG backend cannot open {video_path}")
    try:
        if not cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, ORIENTATION_AUTO) or \
                int(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)) != ORIENTATION_AUTO:
            raise FramesManifestError(f"the decoder refuses the orientation policy "
                                      f"CAP_PROP_ORIENTATION_AUTO={ORIENTATION_AUTO}")
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoder = {"library": "opencv", "opencv_version": str(cv2.__version__),
                   "api_preference": "CAP_FFMPEG", "backend": str(cap.getBackendName()),
                   "video_io": video_io_build_record(),
                   "orientation_auto": float(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
                   "orientation_meta": float(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
                   # the container's rotation, applied here to every frame: they are UPRIGHT
                   "orientation_applied": upright_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META))
                   if upright else None,
                   "stride": stride, "max_frames": max_frames,
                   # USER 2026-11-09 ("a 1024 todo"): frames longer than this are reduced (INTER_AREA)
                   "max_long_side": int(max_long_side), "resize": "INTER_AREA" if int(max_long_side) > 0 else None}
        params = jpeg_params(cv2)
        turn = getattr(cv2, decoder["orientation_applied"]) if decoder["orientation_applied"] else None
        sealer = FrameSealer(scan_dir)
        decoded = kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i = decoded
            decoded += 1
            if i % stride:
                continue
            if turn is not None:
                frame = cv2.rotate(frame, turn)
            if int(max_long_side) > 0 and max(frame.shape[:2]) > int(max_long_side):
                h0, w0 = frame.shape[:2]
                frame = cv2.resize(frame, reduced_size(w0, h0, max_long_side), interpolation=cv2.INTER_AREA)
            ok_enc, buf = cv2.imencode(".jpg", frame, params)
            if not ok_enc:
                raise FramesManifestError(f"the JPEG encoder failed on frame {i} of {video_path}")
            sealer.add(i, buf.tobytes(), ".jpg")
            kept += 1
            if progress is not None and kept % 25 == 0:
                progress(kept, reported)
            if max_frames > 0 and kept >= max_frames:
                break
    finally:
        cap.release()
    if decoded != reported:
        log(f"[frames] {Path(video_path).name}: the container reports {reported} frame(s), the "
            f"decoder delivered {decoded} — both recorded in the manifest")
    doc = sealer.seal(origin=origin, video=video, decoder=decoder, encoder=jpeg_encoder_record(),
                      extra={"frame_count_reported": reported, "frames_decoded": decoded})
    log(f"[frames] sealed {doc['n_frames']} frame(s) of {video['name']} → "
        f"{frames_dir(scan_dir)} (manifest {doc['frames_sha256'][:12]})")
    return doc


# ── reading: check, adopt, verify ───────────────────────────────────────────────────────────

def load_manifest(scan_dir: os.PathLike) -> Optional[Dict[str, Any]]:
    """The scan's manifest, or None when ``frames/`` has none. FAILS on one that cannot be read,
    is of another version or is not complete."""
    p = manifest_path(scan_dir)
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise FramesManifestError(f"{p} is unreadable ({type(e).__name__}: {e})") from e
    if not isinstance(doc, dict):
        raise FramesManifestError(f"{p} holds a {type(doc).__name__}, not a JSON object")
    if doc.get("manifest_version") != MANIFEST_VERSION:
        raise FramesManifestError(f"{p} is version {doc.get('manifest_version')!r}, this code "
                                  f"reads {MANIFEST_VERSION}")
    if doc.get("complete") is not True or not isinstance(doc.get("frames"), list):
        raise FramesManifestError(f"{p} is not a complete manifest — the frames were never sealed")
    return doc


def check_ready(scan_dir: os.PathLike) -> Optional[Dict[str, Any]]:
    """Before anything reads the scan's frames: FAILS while a writer is at work or the temp dir
    exists, or on a manifest that is not a complete one; returns the manifest, or None when
    ``frames/`` predates manifests (the caller adopts it once it has hashed the frames)."""
    why = extraction_in_progress(scan_dir)
    if why:
        raise FramesManifestError(f"the frames of {scan_dir} cannot be read: {why}")
    return load_manifest(scan_dir)


def verify_names(manifest: Mapping[str, Any], names: Sequence[str], where: str = "") -> None:
    """The frames on disk are exactly the manifest's: FAILS naming the first difference (by frame
    number), then the count."""
    listed = {str(r["name"]) for r in manifest["frames"]}
    disk = set(names)
    diff = sorted(listed ^ disk, key=lambda n: (frame_number(n), n))
    if diff:
        n = diff[0]
        side = ("is on disk but not in the manifest" if n in disk
                else "is in the manifest but not on disk")
        raise FramesManifestError(f"{where or 'frames/'}{n} {side} ({len(disk)} frame(s) on disk, "
                                  f"{len(listed)} in the manifest) — the frames changed after "
                                  f"they were sealed")
    if int(manifest.get("n_frames", -1)) != len(listed):
        raise FramesManifestError(f"the manifest lists {len(listed)} frame(s) but says n_frames "
                                  f"{manifest.get('n_frames')}")


def verify_shas(manifest: Mapping[str, Any], shas: Mapping[str, str], where: str = "") -> None:
    """Every frame's sha256 (``shas``: name → sha256, from the I0 stamp) is the manifest's: FAILS
    naming the first frame (by number) whose bytes differ."""
    verify_names(manifest, list(shas), where)
    want = {str(r["name"]): str(r["sha256"]) for r in manifest["frames"]}
    for n in sorted(shas, key=lambda n: (frame_number(n), n)):
        if shas[n] != want[n]:
            raise FramesManifestError(f"{where or 'frames/'}{n} differs from the manifest (sha256 "
                                      f"{shas[n][:12]} on disk, {want[n][:12]} sealed) — the "
                                      f"frames changed after they were sealed")


def source_videos(scan_dir: os.PathLike) -> List[Path]:
    """``source_video.*`` files of the scan (sorted)."""
    d = Path(scan_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.stem == SOURCE_VIDEO_STEM)


def adopt(scan_dir: os.PathLike, shas: Mapping[str, str],
          log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Seal a ``frames/`` that predates manifests, as found (module doc): origin ``adopted``,
    decoder / encoder / counts null — unknown, and said so — each frame's sha256 from ``shas``
    (the caller's I0 stamp), the video's when ``source_video.*`` exists. Refused while a writer
    is at work or the temp dir exists, and when a manifest already exists. Logged DECLARED."""
    scan_dir = Path(scan_dir)
    why = extraction_in_progress(scan_dir)
    if why:
        raise FramesManifestError(f"the frames of {scan_dir} are not adopted: {why}")
    p = manifest_path(scan_dir)
    if p.exists():
        raise FramesManifestError(f"{p} exists — a sealed scan is never adopted")
    vids = source_videos(scan_dir)
    doc: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION, "complete": True, "origin": ORIGIN_ADOPTED,
        "declared": ("frames/ predates the frames manifest (docs/plan_determinismo.md points 67, "
                     "76): adopted as found on disk. The decoder, the encoder, the video's "
                     "reported frame count and whether the extraction completed were never "
                     "recorded: they are unknown (null)."),
        "video": video_record(vids[0], scan_dir) if len(vids) == 1 else None,
        "videos_present": [v.name for v in vids],
        "decoder": None, "encoder": None,
        "frame_count_reported": None, "frames_decoded": None,
    }
    doc.update(_frames_block(shas))
    _write_atomic(p, manifest_bytes(doc))
    log(f"[frames] DECLARED: {frames_dir(scan_dir)} had no manifest (it predates them) — ADOPTED "
        f"as found: {doc['n_frames']} frame(s), video "
        f"{doc['video']['name'] if doc['video'] else (vids and [v.name for v in vids] or 'none')}"
        f", decoder and encoder unknown → {p}")
    return doc
