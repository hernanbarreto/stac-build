"""The ONE rule for reading ``floor_transform.npz`` outside a stage (docs/plan_determinismo.md
point 158, USER 2026-10-08): the viewer (session load, every ``potree_ready``, the scan list,
the camera markers, the raw-cloud send), the erase / paint tools and the alignment gizmo apply
the transform ONLY under the seal the stages verify — `segmentation.pipeline._floor_seal_status`:
``floor_level.json`` beside the npz names THIS reconstruction, THIS cloud and (when its writer
recorded them) THESE npz bytes. Otherwise the cloud is shown in its own frame (identity) and the
reason is DECLARED in the log — never the transform of another cloud, of another reconstruction,
or one nobody sealed. The stages FAIL on such a file (`_display_transform`); the viewer, which
only shows, ignores it and says so — so the frame a cloud is SHOWN in is the frame it was
PROJECTED in, and an erase zone drawn on the screen lands on the points it covers.

The verdict is memoised per process by the SIGNATURE of every file it is a function of — the
cloud, the npz, floor_level.json and the reconstruction's identity files — because the seal
hashes the cloud and the Omega records (seconds on a real session) and the viewer asks several
times per load and on every ``potree_ready``. A file's signature is (inode, size, mtime_ns); a
DIRECTORY's (``omega_run/results_output``, whose records `correction.epoch.reconstruction_id`
hashes) is its own triple PLUS the triple of every regular file under it, because a directory's
own triple moves on entry create / delete / rename only — an Omega record rewritten in place by
a resume would have left it unchanged and the memo serving a verdict for a reconstruction that
no longer exists, while the stages (which call `_floor_seal_status` directly) judged right: the
one disagreement point 158 exists to prevent. One stat per file, no hashing. ONE entry per
session directory, replaced when the signature differs — the memo never grows past the sessions
this process touched. SINCE 2026-10-09 the memo is also kept ON DISK, outside every session
(``server/.cache/floor_seal/``, one JSON per session directory: the signature and the verdict),
because a process starts cold and the first open after a backend restart hashed zaragoza's 2.6 GB
cloud + 1.6 GB of Omega records on the event loop before the cloud could be shown (USER: *"las
nubes deben cargarse inmediatamente"*). It is the same memo under the same key — the verdict is
reused only for the IDENTICAL signature, recomputed on any change — and opening still writes
nothing into the session. Computations of the same directory are serialised (a lock per
directory), so the scan list and the session load never hash the same cloud twice at once.
Declared limit: a
file rewritten IN PLACE to the same size within the clock's granularity of the mtime is not
told apart; every file this module reads is replaced atomically (a new inode) and the Omega
records are written by a run that takes minutes, so none of them is written that way.

The log says what the viewer does ONCE PER CHANGE of the verdict per session directory — the
erase tools ask per zone and per stroke, the scan list on every poll; a line per call would be
noise, a line per change is the declaration.
"""
from __future__ import annotations

import json
import os
import stat as _stat
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

FloorTransform = Tuple[float, np.ndarray, np.ndarray]

#: per session directory (its resolved path): (signature of the files the verdict is a function
#: of, verdict) — one entry per directory, replaced when the signature differs
_VERDICTS: Dict[str, Tuple[tuple, Tuple[bool, str]]] = {}
#: per session directory: the last reason logged — the next line only when it changes
_LOGGED: Dict[str, str] = {}
#: per session directory: the lock that serialises its verdict computations
_LOCKS: Dict[str, "threading.Lock"] = {}
_LOCKS_GUARD = threading.Lock()
#: the on-disk memo (outside every session — opening never writes into one)
_DISK = Path(__file__).resolve().parent.parent / ".cache" / "floor_seal"


_RULE: Optional[str] = None


def _rule_identity() -> str:
    """The identity of the CODE the verdict is a function of — the seal check and the hashes
    it compares — so a verdict saved by another version of that code is never served (the
    stat signature only follows the files). Computed once per process."""
    global _RULE
    if _RULE is None:
        import hashlib
        import inspect
        import repro
        from correction import epoch as _epoch
        from segmentation import pipeline as _pipeline
        try:
            parts = [inspect.getsource(_pipeline._floor_seal_status),
                     inspect.getsource(_epoch.reconstruction_id_or_none),
                     inspect.getsource(_epoch.reconstruction_id),
                     repr(_epoch.RECONSTRUCTION_ID_FILES),
                     inspect.getsource(repro.stamp), inspect.getsource(repro.sha256_path),
                     inspect.getsource(repro.sha256_file),
                     repr(getattr(repro, "STAMP_VERSION", None))]
            _RULE = hashlib.sha256("\n\0".join(parts).encode("utf-8")).hexdigest()
        except (OSError, TypeError) as e:
            # the code cannot be read (a stand-in, a frozen build): a token no saved verdict
            # carries, so nothing on disk is reused by this process
            _RULE = f"unreadable:{os.getpid()}:{id(e)}"
    return _RULE


def _disk_path(key: str) -> Path:
    import hashlib
    return _DISK / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")


def _plain(x):
    """A signature as JSON gives it back (tuples → lists), so the saved and the live compare."""
    return json.loads(json.dumps(x))


def _disk_get(key: str, sig: tuple) -> Optional[Tuple[bool, str]]:
    try:
        doc = json.loads(_disk_path(key).read_text())
    except (OSError, ValueError):
        return None
    if (doc.get("dir") != key or doc.get("rule") != _rule_identity()
            or doc.get("signature") != _plain(sig)):
        return None
    v = doc.get("verdict")
    return (bool(v[0]), str(v[1])) if isinstance(v, list) and len(v) == 2 else None


def _disk_put(key: str, sig: tuple, verdict: Tuple[bool, str]) -> None:
    try:
        from atomic_io import atomic_write_json
        _DISK.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_disk_path(key), {"dir": key, "rule": _rule_identity(),
                                            "signature": _plain(sig),
                                            "verdict": [bool(verdict[0]), str(verdict[1])]})
    except Exception as e:  # noqa: BLE001 — a cache that cannot be written is only slower
        print(f"[Floor] the on-disk seal memo was not written ({e}) — the next process hashes again")


def _sig(path: Path) -> Optional[tuple]:
    """The signature of what a reader would see at ``path``: (inode, size, mtime_ns) of a
    file; of a directory, its own triple plus the sorted (relative name, inode, size,
    mtime_ns) of every regular file under it (a record rewritten in place changes its own
    triple, never the directory's). None when the path does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    own = (st.st_ino, st.st_size, st.st_mtime_ns)
    if not _stat.S_ISDIR(st.st_mode):
        return own
    rows: List[tuple] = []
    stack = [str(path)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            s = e.stat(follow_symlinks=False)
                            rows.append((os.path.relpath(e.path, str(path)),
                                         s.st_ino, s.st_size, s.st_mtime_ns))
                    except OSError:    # an entry that vanished while listing: not part of the signature
                        continue
        except OSError:
            continue
    return own + (tuple(sorted(rows)),)


def _signature(output_dir: Path, cloud_path: Path) -> tuple:
    from correction.epoch import RECONSTRUCTION_ID_FILES
    out = Path(output_dir)
    return (str(Path(cloud_path).resolve()), _sig(cloud_path), _sig(out / "floor_transform.npz"),
            _sig(out / "floor_level.json"),
            tuple(_sig(out / rel) for rel in RECONSTRUCTION_ID_FILES))


def floor_seal_verdict(output_dir, cloud_path=None) -> Tuple[bool, str]:
    """(sealed for this geometry, reason) — `segmentation.pipeline._floor_seal_status` on the
    session's published cloud (``cleaned_cloud.ply`` unless given), memoised by the signatures
    of the files it reads. A session without the cloud has nothing to hold the seal to."""
    from segmentation.pipeline import _floor_seal_status
    out = Path(output_dir)
    cloud = Path(cloud_path) if cloud_path is not None else out / "cleaned_cloud.ply"
    if not cloud.is_file():
        return False, f"no {cloud.name} to hold the seal to (the cloud does not exist yet)"
    key = str(out.resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        sig = _signature(out, cloud)
        hit = _VERDICTS.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        saved = _disk_get(key, sig)
        if saved is not None:
            _VERDICTS[key] = (sig, saved)
            return saved
        verdict = _floor_seal_status(out, cloud)
        # memoised only when nothing moved while it was computed: a verdict read across a
        # transaction's renames is a mix of two states, and a rename back would bring the
        # first signature (and that mixed verdict) back
        if _signature(out, cloud) == sig:
            _VERDICTS[key] = (sig, verdict)
            _disk_put(key, sig, verdict)
        return verdict


def _log_once(output_dir: Path, reason: str, log: Callable[[str], object], line: str) -> None:
    """``line`` through ``log`` only when ``reason`` differs from the last one logged for this
    session directory: a declaration per change of verdict, not per call."""
    key = str(Path(output_dir).resolve())
    if _LOGGED.get(key) == reason:
        return
    _LOGGED[key] = reason
    log(line)


def sealed_floor_transform(output_dir, *, log: Callable[[str], object] = print,
                           tag: str = "Floor", cloud_path=None) -> Optional[FloorTransform]:
    """(s, R, t) of ``floor_transform.npz`` when it carries the seal of the geometry on disk;
    None when there is no npz (nothing to say) or when it cannot be applied — then the reason
    is DECLARED through ``log`` (once per change of verdict) and the caller shows the cloud in
    its own frame."""
    out = Path(output_dir)
    tp = out / "floor_transform.npz"
    if not tp.is_file():
        return None
    ok, why = floor_seal_verdict(out, cloud_path)
    if not ok:
        _log_once(out, f"ignored: {why}", log,
                  f"[{tag}] DECLARED: {tp.name} ignored — {why} (docs/plan_determinismo.md point 158: a "
                  f"floor transform is applied only with the seal of the geometry it levelled; the "
                  f"cloud is shown in its own frame, as the stages project it; level the floor "
                  f"explicitly or run the cloud stage to seal one)")
        return None
    try:
        data = np.load(tp)
        s, R, t = float(data["s"]), np.asarray(data["R"], np.float64), np.asarray(data["t"], np.float64)
    except Exception as e:  # noqa: BLE001 — an unreadable npz is ignored, declared, never a crash of the viewer
        _log_once(out, f"unreadable: {e}", log,
                  f"[{tag}] DECLARED: {tp.name} ignored — unreadable ({e}); the cloud is shown in its own frame")
        return None
    _log_once(out, f"applied: {why}", log, f"[{tag}] floor transform applied — {why}")
    return s, R, t


def floor_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The 4x4 of ``xyz_display = s·R·xyz + t`` (row-major)."""
    M = np.eye(4)
    M[:3, :3] = float(s) * np.asarray(R, np.float64)
    M[:3, 3] = np.asarray(t, np.float64)
    return M


def sealed_floor_matrix(output_dir, *, log: Callable[[str], object] = print,
                        tag: str = "Floor") -> Optional[np.ndarray]:
    """:func:`sealed_floor_transform` as a 4x4, or None."""
    srt = sealed_floor_transform(output_dir, log=log, tag=tag)
    return None if srt is None else floor_matrix(*srt)
