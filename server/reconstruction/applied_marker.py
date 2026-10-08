"""Idempotency markers that KNOW what they transformed (docs/plan_determinismo.md point 10).

``scale_align``'s ``.metric_scale_applied`` and ``orient``'s ``.orientation_applied`` used to be
existence-only: a resume with the marker on disk skipped the transform whatever the files were.
After a completed run the pipeline's cleanup deletes Omega's chunk PLYs, so the next replace-OFF
Reconstruir re-ran the fork, which rewrote camera_poses.txt in Omega's raw frame — both markers
were still there, and the new poses and chunks were left unscaled and unoriented (audit M-05 /
omega-10, pccr: s = 0.984577 and a 165.48° orientation skipped over raw products).

Now every marker has a SIDECAR ``<marker>.sha256.json`` naming the files the transform wrote,
with their sha256 AFTER it (paths relative to output/). On the next pass the marker is

- ``current``  — every recorded file that is still on disk hashes the same: the products ARE the
  transformed ones, the transform is not repeated (repeating it would apply s twice);
- ``stale``    — a recorded file is on disk with other bytes (a fresh fork pass rewrote it): the
  caller REDOES the transform from what is on disk — the raw poses of that pass — and refreshes
  the ``.prescale`` / ``.preorient`` backups;
- ``unstamped`` — a marker written before the sidecar existed (sessions reconstructed before
  2026-10-07): nothing to verify against. It is REUSED and DECLARED in the log (redoing blindly
  would double-scale a validated session); a Replace makes a fresh, stamped pass;
- ``absent``   — no marker: a fresh pass.

A recorded file that is GONE is not a difference: the cleanup deletes the chunk PLYs and the
aligned copies after every completed run while the poses they were transformed with stay.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SIDECAR_SUFFIX = ".sha256.json"
SIDECAR_VERSION = 1


def sidecar_path(marker: Path) -> Path:
    marker = Path(marker)
    return marker.with_name(marker.name + SIDECAR_SUFFIX)


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(p).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def record(marker: Path, text: str, files: Iterable[Path], *, base: Path,
           extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write ``marker`` with ``text`` and its sidecar: the sha256 of every file in ``files``
    (as they are NOW — after the transform), keyed by the path relative to ``base``, plus
    ``extra`` (what was applied: s, the rotation …). Returns the sidecar path."""
    import repro
    marker = Path(marker)
    shas: Dict[str, str] = {}
    for f in files:
        f = Path(f)
        if f.is_file():
            shas[_rel(f, base)] = repro.sha256_file(f)
    doc = {"version": SIDECAR_VERSION, "marker": marker.name, "files": dict(sorted(shas.items())),
           "extra": dict(extra or {})}
    side = sidecar_path(marker)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, side)
    marker.write_text(text if text.endswith("\n") else text + "\n")
    return side


def state(marker: Path, *, base: Path) -> Tuple[str, Dict[str, Any]]:
    """(status, info) of ``marker``: ``absent`` | ``current`` | ``stale`` | ``unstamped``.
    ``info`` carries the marker text, the sidecar's ``extra``, and for ``stale`` the list of
    files that differ (``changed``) — every difference named, nothing guessed."""
    import repro
    marker = Path(marker)
    if not marker.exists():
        return "absent", {}
    text = marker.read_text().strip()
    side = sidecar_path(marker)
    if not side.exists():
        return "unstamped", {"text": text}
    try:
        doc = json.loads(side.read_text())
    except (OSError, ValueError) as e:
        return "stale", {"text": text, "changed": [f"{side.name}: unreadable ({e})"]}
    files = doc.get("files") if isinstance(doc, dict) else None
    if doc.get("version") != SIDECAR_VERSION or not isinstance(files, dict):
        return "stale", {"text": text, "changed": [f"{side.name}: not a version-{SIDECAR_VERSION} record"]}
    changed: List[str] = []
    gone: List[str] = []
    for rel, sha in sorted(files.items()):
        p = Path(base) / rel
        if not p.is_file():
            gone.append(rel)
        elif repro.sha256_file(p) != sha:
            changed.append(rel)
    info = {"text": text, "extra": doc.get("extra") or {}, "n_recorded": len(files),
            "gone": gone, "changed": changed}
    return ("stale" if changed else "current"), info


def clear(marker: Path) -> None:
    """Remove the marker and its sidecar (a fresh pass starts clean)."""
    marker = Path(marker)
    for p in (marker, sidecar_path(marker)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def describe(status: str, info: Dict[str, Any]) -> str:
    if status == "current":
        return (f"marker current ({info.get('text')}): {info.get('n_recorded', 0)} transformed "
                f"file(s) recorded, every one still on disk unchanged"
                + (f", {len(info['gone'])} gone (deleted by the cleanup)" if info.get("gone") else ""))
    if status == "stale":
        return (f"marker STALE ({info.get('text')}): transformed file(s) rewritten since — "
                f"{', '.join(info.get('changed') or [])}")
    if status == "unstamped":
        return (f"marker predates the sha record ({info.get('text')}): it cannot be verified "
                f"and is reused as is, DECLARED — a Replace makes a fresh, stamped pass")
    return "no marker"
