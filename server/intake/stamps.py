"""What an intake product depends on beyond its frames and its parameters, and the stamps that
say so (docs/plan_determinismo.md points 66, 70, 74, 78, 79 — 2026-10-07).

The intake's resume markers used to be reused on keys with no code, no library, no weights and
no card in them: a frame was identified by its NAME and its SIZE, a step by a version number
bumped by hand. A session resumed under newer code, another numpy or another card kept the old
keyframes while a fresh run computed new ones. Everything here is built on :mod:`repro`:

- :data:`INTAKE_CODE_FILES` — the code every intake stamp is keyed on: the intake package, the
  DA3 extractor the focal probe and I3 run, and the shared modules they import. A change in any
  of them re-runs the step; no version number is compared.
- :func:`frames_stamp` — the sha256 of EVERY frame's bytes, keyed by its session-relative path
  (``frames/000123.jpg``), so a copied session keeps its stamps and re-extracted frames under the
  same names do not.
- :func:`cpu_environment_record` — the CPU-side libraries whose numerics reach the readings
  (numpy / scipy / OpenCV / Pillow / threadpoolctl versions, the CPU model, the BLAS core
  OpenBLAS dispatched to, OpenCV's dispatched instruction sets and IPP), and the two JPEG
  decoders of this pipeline with their libjpeg-turbo builds (:func:`jpeg_decoder_record`:
  OpenCV decodes the frames for I0, I1 and F4; Pillow decodes the same files inside DA3).

The card a GPU step of the intake runs on is checked ONE and read through torch by
``card_table.require_one_visible_card`` / ``repro.card_identity`` (point 78), called by the DA3
extractor's identity probe, the calibration CLI and the Omega planner — not here.

Nothing here reads a wall clock. Paths in every record are relative to the session.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import repro

SERVER_DIR = Path(__file__).resolve().parent.parent
INTAKE_DIR = Path(__file__).resolve().parent

# The code an intake product depends on (plan point 66): the intake package itself, the DA3
# extractor (the focal probe's and I3's instrument), and the shared modules they import. Keyed
# in the stamp by the repo-relative path; a file that does not exist FAILS the stamp.
INTAKE_CODE_FILES: Sequence[Path] = tuple(sorted(
    [p for p in INTAKE_DIR.glob("*.py")]
    + [SERVER_DIR / "extract_da3_depth.py", SERVER_DIR / "repro.py",
       SERVER_DIR / "da3_weights.py", SERVER_DIR / "card_table.py"]))

# The CPU-side distributions whose numerics reach the intake readings (point 74 / 79). One that
# is not installed is recorded as null — a fact, not an unknown.
CPU_LIBS: Sequence[str] = ("numpy", "scipy", "opencv-python", "opencv-python-headless",
                           "pillow", "pillow-heif", "threadpoolctl")

FRAMES_DIRNAME = "frames"


class IntakeStampError(RuntimeError):
    """A stamp cannot be made (an input is missing, a decoder cannot be identified, the card
    is not exactly one) — with the exact reason, never a placeholder."""


# ── paths ───────────────────────────────────────────────────────────────────────────────────

def rel_to_session(path: os.PathLike, session_dir: os.PathLike) -> str:
    """``path`` as a posix path relative to the session (point 70: products and reuse keys
    carry no absolute path). A path outside the session FAILS — nothing an intake product
    names lives elsewhere."""
    p, s = Path(path), Path(session_dir)
    try:
        return p.resolve().relative_to(s.resolve()).as_posix()
    except ValueError as e:
        raise IntakeStampError(f"{p} is not inside the session {s}") from e


def in_session(rel: str, session_dir: os.PathLike) -> Path:
    """The absolute path of a session-relative one (the inverse of :func:`rel_to_session`)."""
    return Path(session_dir) / Path(rel)


# ── the frames ──────────────────────────────────────────────────────────────────────────────

def frame_inputs(frames_dir: os.PathLike, session_dir: Optional[os.PathLike] = None,
                 files: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    """``{session-relative path: path}`` of the frames (every frame of ``frames_dir`` in frame
    order, or only ``files`` — basenames), the mapping :func:`repro.stamp` takes as inputs."""
    from intake.quality import QualityError, list_frames
    frames_dir = Path(frames_dir)
    session_dir = Path(session_dir) if session_dir is not None else frames_dir.parent
    if files is None:
        try:
            paths = list_frames(frames_dir)
        except QualityError as e:
            raise IntakeStampError(str(e)) from e
    else:
        paths = [frames_dir / f for f in files]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise IntakeStampError(f"frame(s) {missing[:5]} are not in {frames_dir}")
    return {rel_to_session(p, session_dir): p for p in paths}


def frames_stamp(frames_dir: os.PathLike, session_dir: Optional[os.PathLike] = None,
                 files: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """:func:`repro.stamp` over the frames' BYTES only (no code, no config): ``inputs`` maps
    each session-relative frame path to its sha256; ``sha256`` is the inventory's digest."""
    return repro.stamp(inputs=frame_inputs(frames_dir, session_dir, files))


# ── the CPU environment ─────────────────────────────────────────────────────────────────────

_CV_BUILD_KEYS = {
    "JPEG:": "jpeg", "PNG:": "png", "Intel IPP:": "ipp", "Baseline:": "baseline",
    "Dispatched code generation:": "dispatched", "Parallel framework:": "parallel",
}


def opencv_build_record() -> Dict[str, Any]:
    """OpenCV's version and, from ``cv2.getBuildInformation()``, its JPEG and PNG codecs, the
    IPP it dispatches to, and the baseline / dispatched instruction sets. A build information
    without the JPEG line FAILS — the decoder of every frame must be identified."""
    import cv2
    out: Dict[str, Any] = {"version": str(cv2.__version__)}
    for ln in cv2.getBuildInformation().splitlines():
        s = ln.strip()
        for key, name in _CV_BUILD_KEYS.items():
            if s.startswith(key) and name not in out:
                out[name] = s[len(key):].strip()
    if "jpeg" not in out:
        raise IntakeStampError("cv2.getBuildInformation() names no JPEG codec — the decoder of "
                               "the frames cannot be recorded")
    return out


def pillow_record() -> Dict[str, Any]:
    """Pillow's version, its JPEG codec version and its libjpeg-turbo build (the decoder DA3
    reads the same frames with: vendor/depth-anything-3 input_processor, ``Image.open``)."""
    import PIL
    from PIL import features
    if not features.check_codec("jpg"):
        raise IntakeStampError("Pillow has no JPEG codec — the DA3 decoder cannot be recorded")
    turbo = bool(features.check_feature("libjpeg_turbo"))
    return {"version": str(PIL.__version__),
            "jpg_codec": str(features.version_codec("jpg")),
            "libjpeg_turbo": (str(features.version_feature("libjpeg_turbo")) if turbo else None)}


def jpeg_decoder_record() -> Dict[str, Any]:
    """The two JPEG decoders of this pipeline (point 79): OpenCV (I0, I1, F4, the content
    masks) and Pillow (DA3: the focal probe, the I3 windows, the anchors)."""
    cv = opencv_build_record()
    return {"opencv": {"version": cv["version"], "libjpeg": cv["jpeg"],
                       "ipp": cv.get("ipp"), "dispatched": cv.get("dispatched"),
                       "baseline": cv.get("baseline")},
            "pillow": pillow_record()}


def _fft_dtype() -> str:
    """The dtype numpy's FFT produces for the float64 input I0 feeds it (point 74: numpy 1.x
    upcast a float32 input to complex128, numpy 2.x computes float32 natively)."""
    import numpy as np
    return str(np.fft.fft2(np.zeros((2, 2), np.float64)).dtype)


def cpu_environment_record() -> Dict[str, Any]:
    """What the CPU-side intake readings depend on beyond frames, parameters and code (points
    74, 79): python, platform, the CPU model, the BLAS libraries loaded (API, version, the core
    OpenBLAS dispatched to, threading layer — NOT the thread count: I1 pins it to one and no
    reading depends on it), the library versions (:data:`CPU_LIBS`), the JPEG decoders with
    their libjpeg-turbo builds, OpenCV's dispatch and numpy's FFT dtype. Deterministic on one
    machine and stack; a different stack differs, which is the point.

    The libraries the intake computes with are imported FIRST, so the BLAS list is the same
    whatever was imported before the call: scipy brings its own OpenBLAS build, and I1 loads
    scipy.optimize lazily — a record taken before I1 ran listed one BLAS, one taken after
    listed two, and the same session re-ran every step on its second pass."""
    import cv2  # noqa: F401
    import scipy.optimize  # noqa: F401 — loads scipy's bundled OpenBLAS (I1's least_squares)
    from PIL import Image  # noqa: F401
    blas = [{k: r.get(k) for k in ("user_api", "internal_api", "version", "architecture",
                                   "threading_layer", "library")}
            for r in repro.blas_record()]
    return {
        "python": sys.version.split()[0],
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "cpu_model": repro.cpu_model(),
        "blas": blas,
        "libs": {name: repro._dist_version(name) for name in CPU_LIBS},
        "jpeg_decoders": jpeg_decoder_record(),
        "numpy_fft_dtype": _fft_dtype(),
    }


# ── stamps ──────────────────────────────────────────────────────────────────────────────────

def step_stamp(inputs: Mapping[str, os.PathLike], config: Mapping[str, Any]) -> Dict[str, Any]:
    """:func:`repro.stamp` of an intake step: its ``inputs`` (session-relative names → paths),
    the intake code (:data:`INTAKE_CODE_FILES`) and its named ``config`` sections."""
    return repro.stamp(inputs=dict(inputs), code=list(INTAKE_CODE_FILES), config=dict(config))


def stamp_differences(saved: Any, now: Mapping[str, Any]) -> List[str]:
    """:func:`repro.check_stamp` — [] when the saved product is this stamp's."""
    return repro.check_stamp(saved, now)
