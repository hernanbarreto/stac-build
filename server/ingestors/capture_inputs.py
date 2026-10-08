"""Where a scan's CAPTURE data lives: ``<scan>/inputs/`` (docs/plan_determinismo.md points 73
and 77, DECIDIDO 2026-10-07).

The user's rule for "Reconstruir" with replace (2026-09-28: the wipe leaves nothing but the
frames and the original video) is applied by taking the WHOLE CAPTURE as the original: the
Stray Scanner export (camera_matrix.csv, odometry.csv, depth/, confidence/, rgb.mp4, imu.csv),
the VIO trajectory and the WebXR capture's per-frame camera data are inputs the wipe never
touches. Until today they sat in the scan directory beside the outputs and the wipe deleted
them: a session with VIO was scaled by VIO until its first replace and by DA3 after it, and F0
took Stray's K or Omega's depending on whether a replace had ever run.

- ``inputs/stray/`` — the Stray Scanner export (:data:`STRAY_ENTRIES`);
- ``inputs/vio_trajectory.csv|json`` or ``inputs/vio/trajectory.csv|json`` — the VIO
  trajectory (docs/VIO_FORMAT.md);
- ``inputs/webxr/`` — the WebXR capture's ``camera_data/`` and ``scan_meta.json``.

READERS (map_worker / session_io / shaper_export ``_find_stray_dir``, ``precision.camera``,
``precision.gauge``, ``ingestors.stray_detector``, ``ingestors.vio_detector`` → scale_align,
the gauge, the in-run VIO rows) look in ``inputs/`` FIRST, then in the scan's own legacy
places — the scan directory itself and its ``stray/`` subdirectory, never a sibling scan
(point 35). The replace wipe MOVES legacy capture data into ``inputs/`` instead of deleting it
(:func:`move_legacy_capture`). No writer of Stray data or VIO trajectories exists in the server
today (they are copied into a scan by hand); one that is added writes into :func:`inputs_dir`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

INPUTS_DIRNAME = "inputs"
STRAY_DIRNAME = "stray"
WEBXR_DIRNAME = "webxr"
# the Stray Scanner export layout (ingestors/stray_scanner.py, ingestors/stray_detector.py)
STRAY_ENTRIES = ("camera_matrix.csv", "odometry.csv", "imu.csv", "rgb.mp4", "depth", "confidence")
# a scan directory IS a (legacy) Stray export when it holds one of these
STRAY_MARKERS = ("odometry.csv", "camera_matrix.csv")
# the VIO trajectory's accepted names, relative to inputs/ (and, legacy, to the scan directory)
VIO_CANDIDATES = ("vio_trajectory.csv", "vio_trajectory.json", "vio/trajectory.csv",
                  "vio/trajectory.json")
VIO_ENTRIES = ("vio_trajectory.csv", "vio_trajectory.json", "vio")
# the WebXR capture socket's capture data (main.scan_websocket)
WEBXR_ENTRIES = ("camera_data", "scan_meta.json")


class CaptureInputsError(RuntimeError):
    """Capture data that cannot be placed or read unambiguously — with the exact reason."""


def inputs_dir(scan_dir: os.PathLike) -> Path:
    return Path(scan_dir) / INPUTS_DIRNAME


def stray_inputs_dir(scan_dir: os.PathLike) -> Path:
    return inputs_dir(scan_dir) / STRAY_DIRNAME


def webxr_inputs_dir(scan_dir: os.PathLike) -> Path:
    return inputs_dir(scan_dir) / WEBXR_DIRNAME


def scan_dir_of(stray_dir: os.PathLike) -> Path:
    """The scan a Stray directory belongs to: ``<scan>/inputs/stray`` → ``<scan>``, a legacy
    ``<scan>/stray`` → ``<scan>``, the scan directory itself → itself."""
    d = Path(stray_dir)
    if d.name == STRAY_DIRNAME and d.parent.name == INPUTS_DIRNAME:
        return d.parent.parent
    if d.name == STRAY_DIRNAME:
        return d.parent
    return d


# ── readers ─────────────────────────────────────────────────────────────────────────────────

def stray_candidates(scan_dir: os.PathLike) -> List[Path]:
    """Where a scan's Stray export may sit, in reading order: ``inputs/stray/`` first, then the
    scan's own legacy places (the scan directory, its ``stray/``) — never a sibling scan."""
    s = Path(scan_dir)
    return [stray_inputs_dir(s), s, s / STRAY_DIRNAME]


def _holds(d: Path, names: Sequence[str]) -> bool:
    return all((d / n).exists() for n in names)


def stray_dirs(scan_dir: os.PathLike, required: Sequence[str] = (),
               any_of: Sequence[str] = ()) -> List[Path]:
    """Every candidate (reading order) holding ALL of ``required`` and, when given, one of
    ``any_of``."""
    return [d for d in stray_candidates(scan_dir)
            if d.is_dir() and _holds(d, required)
            and (not any_of or any((d / n).exists() for n in any_of))]


def find_stray_dir(scan_dir: os.PathLike, required: Sequence[str] = (),
                   any_of: Sequence[str] = ()) -> Optional[Path]:
    """The first candidate of :func:`stray_candidates` holding the reader's files, or None."""
    found = stray_dirs(scan_dir, required, any_of)
    return found[0] if found else None


def vio_candidates(scan_dir: os.PathLike) -> List[Path]:
    """Where a scan's VIO trajectory may sit, in reading order: under ``inputs/`` first, then the
    scan directory's legacy names (docs/VIO_FORMAT.md)."""
    s = Path(scan_dir)
    return [inputs_dir(s) / rel for rel in VIO_CANDIDATES] + [s / rel for rel in VIO_CANDIDATES]


def find_vio_file(scan_dir: os.PathLike) -> Optional[Path]:
    return next((p for p in vio_candidates(scan_dir) if p.is_file()), None)


def rel_to_scan(path: os.PathLike, scan_dir: os.PathLike) -> str:
    """``path`` relative to the scan (posix); FAILS for a path outside it."""
    try:
        return Path(path).resolve().relative_to(Path(scan_dir).resolve()).as_posix()
    except ValueError:
        raise CaptureInputsError(f"{path} is not inside the scan {scan_dir}") from None


def file_record(path: Optional[os.PathLike], scan_dir: os.PathLike) -> Optional[Dict[str, Any]]:
    """``{"file": scan-relative path, "sha256": …}`` of a capture file, None for None."""
    if path is None:
        return None
    import repro
    return {"file": rel_to_scan(path, scan_dir), "sha256": repro.sha256_file(path)}


def vio_record(scan_dir: os.PathLike) -> Optional[Dict[str, Any]]:
    """The VIO trajectory the scan's readers take (``find_vio_file``), as a :func:`file_record`;
    None when the scan has none."""
    return file_record(find_vio_file(scan_dir), scan_dir)


# ── the replace wipe: legacy capture data is MOVED into inputs/ ────────────────────────────

def _same_content(a: Path, b: Path) -> bool:
    import repro
    if a.is_file() != b.is_file():
        return False
    return repro.sha256_path(a) == repro.sha256_path(b)


def _move(src: Path, dst: Path, scan_dir: Path, moved: List[str], log: Callable) -> None:
    rel_s, rel_d = rel_to_scan(src, scan_dir), rel_to_scan(dst, scan_dir)
    if dst.exists() or dst.is_symlink():
        if not _same_content(src, dst):
            raise CaptureInputsError(
                f"capture data in two places with different content: {rel_s} and {rel_d} — "
                f"keep the one of this recording (the replace wipe moves {rel_s} into "
                f"{INPUTS_DIRNAME}/ and will not overwrite {rel_d})")
        if src.is_dir() and not src.is_symlink():
            shutil.rmtree(src)
        else:
            src.unlink()
        moved.append(f"{rel_s} (identical to {rel_d}: the duplicate removed)")
        log(f"[inputs] {rel_s} is identical to {rel_d} — the duplicate removed")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    moved.append(f"{rel_s} -> {rel_d}")
    log(f"[inputs] capture data kept: {rel_s} -> {rel_d}")


def legacy_capture_entries(scan_dir: os.PathLike) -> List[tuple]:
    """``[(legacy path, its place under inputs/)]`` of the capture data sitting in the scan's
    legacy places: the VIO names at the scan root, a root-level Stray export (only when the root
    holds odometry.csv or camera_matrix.csv), every entry of a legacy ``stray/`` subdirectory,
    the WebXR capture's camera_data/ and scan_meta.json."""
    s = Path(scan_dir)
    out = []
    for n in VIO_ENTRIES:
        if (s / n).exists():
            out.append((s / n, inputs_dir(s) / n))
    if any((s / m).is_file() for m in STRAY_MARKERS):
        for n in STRAY_ENTRIES:
            if (s / n).exists():
                out.append((s / n, stray_inputs_dir(s) / n))
    legacy_stray = s / STRAY_DIRNAME
    if legacy_stray.is_dir() and not legacy_stray.is_symlink():
        for p in sorted(legacy_stray.iterdir()):
            out.append((p, stray_inputs_dir(s) / p.name))
    for n in WEBXR_ENTRIES:
        if (s / n).exists():
            out.append((s / n, webxr_inputs_dir(s) / n))
    return out


def move_legacy_capture(scan_dir: os.PathLike, log: Callable[[str], Any] = print) -> List[str]:
    """Move the scan's legacy capture data into ``inputs/`` (DECIDIDO 73 / 77: capture data is
    the original, the wipe keeps it). An identical copy already under ``inputs/`` removes the
    legacy duplicate; a DIFFERENT one FAILS naming both — nothing is overwritten, nothing is
    deleted. An emptied legacy ``stray/`` is removed. Returns what moved."""
    s = Path(scan_dir)
    entries = legacy_capture_entries(s)
    if not entries:
        return []
    # every conflict is found BEFORE anything moves: the wipe either moves all of it or none
    for src, dst in entries:
        if (dst.exists() or dst.is_symlink()) and not _same_content(src, dst):
            raise CaptureInputsError(
                f"capture data in two places with different content: {rel_to_scan(src, s)} and "
                f"{rel_to_scan(dst, s)} — keep the one of this recording; nothing was moved")
    moved: List[str] = []
    for src, dst in entries:
        _move(src, dst, s, moved, log)
    legacy_stray = s / STRAY_DIRNAME
    if legacy_stray.is_dir() and not legacy_stray.is_symlink() and not any(legacy_stray.iterdir()):
        legacy_stray.rmdir()
    return moved
