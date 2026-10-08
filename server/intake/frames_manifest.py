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
  SET explicitly — ``CAP_PROP_ORIENTATION_AUTO`` 0, measured to be OpenCV 4.11.0's default: a
  video carrying a 90° display matrix decodes unrotated, which is how bufferStop's and
  fosa_pan's frames on disk are — and the container's own orientation metadata), the encoder
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


# The orientation policy of the decoder: OpenCV's automatic rotation OFF — the frames are the
# video's coded pixels. Measured 2026-10-07 on env da3 (OpenCV 4.11.0, FFMPEG backend): the
# default IS off (a video with a 90° display matrix decodes unrotated; bufferStop's and fosa_pan's
# videos carry -90° and their frames on disk are the unrotated 1024x576 / 768x576), so setting it
# explicitly keeps every extraction identical to the frames already on disk.
ORIENTATION_AUTO = 0


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


def extract_video(video_path: os.PathLike, scan_dir: os.PathLike, *, origin: str = ORIGIN_VIDEO,
                  stride: int = 1, max_frames: int = 0,
                  progress: Optional[Callable[[int, int], Any]] = None,
                  log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """Decode ``video_path`` into the scan's frames, sealed (module doc): frame ``i`` of the
    decode order → ``<i:06d>.jpg`` (only ``i % stride == 0``, at most ``max_frames`` when > 0 —
    the Stray extractor's legacy stride). The decoder is OpenCV's FFMPEG backend, requested
    explicitly, orientation policy :data:`ORIENTATION_AUTO` set and read back; the encoder
    :func:`jpeg_params`. Refused when the scan already holds frames or a manifest, or when the
    temp dir holds anything (an interrupted writer: :func:`discard_partial` first). Returns the
    manifest."""
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
                   "stride": stride, "max_frames": max_frames}
        params = jpeg_params(cv2)
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
