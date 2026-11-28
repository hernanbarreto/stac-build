"""A SESSION AS ONE ZIP — out of the pod and back in, the same session.

USER 2026-11-18: "Para las sesiones vamos a hacer una función en cada sesión para descargarla como
zip, debe descargar todo lo necesario para que pueda volver a subirse y correr exactamente igual. Y
para subirla … en nueva sesión … crear una vacía o subir un archivo zip que va a comprobar que tenga
la estructura correcta de los datos que contiene el zip y si está todo ok la sube y la descomprime,
elimina el zip y se queda con la sesión perfectamente utilizable."

A session is its project directory (project_paths.py: project.json, scans/<date>/src_<name>/{frames,
output, …}, merged/, ifcs/, segmentation/, exports/ …) and NOTHING of it is left out but what any
use writes again: python bytecode, a notebook's checkpoints, SQLite's shared-memory file and its
EMPTY write-ahead log, an upload in flight. The ZIP holds

    STAC_SESSION.json            the manifest: format, version, the session's id, and every entry —
                                 a file (bytes, mtime in ns, mode), a link (its target), an empty
                                 directory — so that the copy is the SAME tree: sizes, dates, modes
    <session id>/…               the tree

WRITTEN AS A FUNCTION OF THE MANIFEST (Layout): every entry is STORED, so the place of every byte
of the ZIP is known before a file is read — its whole size (the browser shows how much is left) and
any RANGE of it (a download of 100 GB cut by the network is RESUMED, not started again). No second
copy of the session is ever on the volume: the bytes are read from the session as they are sent. A
file that is not, when it is read, the one the manifest lists ENDS the stream with an error: a
half-consistent archive is never finished. (Stored, not deflated: the JSON of a session would shrink
— 12 of pccr's 107 GB — but a deflated entry's size is only known after compressing it, and then
nothing can be resumed.)

READ BACK IN TWO STEPS: validate() opens the uploaded ZIP and holds it to the manifest before a
single byte is extracted — the format, one top-level directory, safe paths, every manifest entry
present with its size and nothing beside them, the project's own shape (project.json, the scans'
layout) — and extract() writes it beside the sessions under a hidden name, checks every file's
CRC and size as it goes, sets dates and modes, and renames it into place last. A failure leaves
no session and no partial tree.
"""
from __future__ import annotations

import bisect
import calendar
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

FORMAT = "stac-session"
VERSION = 1
MANIFEST = "STAC_SESSION.json"
ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
SCAN_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BLOCK = 1 << 20

# what a use of the session writes again, and never part of it (ifcs/.tiles: the viewer's tiles of each IFC, made again
# from the IFC when it is first shown — bim/ifc_tiles.py)
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".tiles"}
SKIP_SUFFIXES = ("-shm", ".uploading", ".pyc")
STAT_THREADS = 16                                             # (the volume is a network one: a file's stat is ~1 ms, in turn)

Progress = Optional[Callable[[int, int, str], None]]          # (bytes done, bytes total, what)


class ArchiveError(ValueError):
    """The session cannot be archived, or the ZIP is not a session — the message says why."""


# ── the tree ───────────────────────────────────────────────────────────────────────────────────
def entries(project_dir: Path) -> List[Dict[str, Any]]:
    """Every file, link and EMPTY directory of the session, sorted — the manifest's entries:
    {"p": path} + {"s": bytes, "m": mtime ns, "x": mode} | {"l": link target} | {"d": 1}."""
    project_dir = Path(project_dir)
    if not (project_dir / "project.json").is_file():
        raise ArchiveError(f"{project_dir.name}: no project.json — not a session of this layout")
    walked: List[Tuple[str, List[str], bool]] = []              # (directory, its names that are not walked directories, has walked directories)
    for root, dirs, files in os.walk(project_dir):
        if "frames.extracting" in dirs or "frames.extracting" in files:
            raise ArchiveError(f"{project_dir.name}: a video is being extracted into it — download it when that ends")
        linked = [d for d in dirs if os.path.islink(os.path.join(root, d))]   # (a link to a directory is an entry, not a walk)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d not in linked)
        walked.append((root, sorted(files + linked), bool(dirs)))
    paths = [os.path.join(root, name) for root, names, _ in walked for name in names]
    try:
        with ThreadPoolExecutor(STAT_THREADS) as pool:
            stats = dict(zip(paths, pool.map(os.lstat, paths)))
    except OSError as e:
        raise ArchiveError(f"{project_dir.name}: {e.filename} changed while the session was being listed — "
                           f"it is being written; try again when that ends") from e
    out: List[Dict[str, Any]] = []
    for root, names, has_dirs in walked:
        rel_root = os.path.relpath(root, project_dir)
        kept = 0
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name)).replace(os.sep, "/")
            st = stats[full]
            if stat.S_ISLNK(st.st_mode):
                out.append({"p": rel, "l": os.readlink(full)})
                kept += 1
                continue
            if name.endswith(SKIP_SUFFIXES) or not stat.S_ISREG(st.st_mode):
                continue                                        # (… and a socket, a pipe: not data)
            if name.endswith("-wal"):
                if st.st_size == 0:
                    continue                                    # (an empty write-ahead log: SQLite makes it again)
                raise ArchiveError(f"{project_dir.name}: {rel} holds changes not yet written into its database "
                                   f"({st.st_size} bytes) — the session is being written; close it and download again")
            out.append({"p": rel, "s": st.st_size, "m": st.st_mtime_ns, "x": stat.S_IMODE(st.st_mode)})
            kept += 1
        if not has_dirs and not kept and rel_root != ".":
            out.append({"p": rel_root.replace(os.sep, "/"), "d": 1})
    return sorted(out, key=lambda e: e["p"])


def scans_of(project_dir: Path) -> List[str]:
    scans = Path(project_dir) / "scans"
    found = []
    for day in sorted(scans.iterdir()) if scans.is_dir() else []:
        if day.is_dir() and SCAN_DAY_RE.match(day.name):
            for src in sorted(day.iterdir()):
                if src.is_dir() and src.name.startswith("src_"):
                    found.append(f"{day.name}/{src.name[4:]}")
    return found


def _scan_dir(key: str) -> str:
    day, source = key.split("/", 1)
    return f"scans/{day}/src_{source}"


def _for_scans(project_dir: Path, listed: List[Dict[str, Any]], every: List[str],
               keep: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """The session with ONLY the scans ``keep`` (USER 2026-11-19: the session's download, "asi completa
    como esta, y eventualmente seleccionar cual de los escaneos quiere agregar o excluir"): the entries
    without the other scans' folders — and without a scan day left with no scan, and without the
    stored BIM comparison when it is of a scan that is left out — and the text project.json has in
    that archive: the same file with the other scans taken off its list, off the composition and
    off "active_scan". The archive is still a whole session: it uploads and opens as one.
    (The composition is of scans TO THE REFERENCE scan: when the reference is left out, what places
    the others on it means nothing any more and is dropped.)"""
    gone = [k for k in every if k not in keep]
    gone_dirs = tuple(_scan_dir(k) for k in gone)
    kept_days = {k.split("/", 1)[0] for k in keep}
    sheet_meta = project_dir / "bim_comparison" / "sabana_meta.json"
    drop_sheet = False
    if sheet_meta.is_file():
        try:
            drop_sheet = json.loads(sheet_meta.read_text()).get("scan") in gone
        except (OSError, ValueError):
            drop_sheet = False

    def stays(p: str) -> bool:
        if any(p == g or p.startswith(g + "/") for g in gone_dirs):
            return False
        parts = p.split("/")
        if parts[0] == "scans" and len(parts) >= 2 and SCAN_DAY_RE.match(parts[1]) and parts[1] not in kept_days:
            return False
        return not (drop_sheet and parts[0] == "bim_comparison")

    out = [dict(e) for e in listed if stays(e["p"])]
    meta = json.loads((project_dir / "project.json").read_text())
    if isinstance(meta.get("scans"), list):
        meta["scans"] = [x for x in meta["scans"] if not (isinstance(x, dict) and x.get("key") in gone)]
    composition = meta.get("composition")
    if isinstance(composition, dict):
        if composition.get("reference") in gone:
            composition["reference"] = keep[0]
            composition["transforms"] = {}
        elif isinstance(composition.get("transforms"), dict):
            composition["transforms"] = {k: v for k, v in composition["transforms"].items() if k not in gone}
    if meta.get("active_scan") in gone:
        meta["active_scan"] = keep[0]
    text = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    for e in out:
        if e["p"] == "project.json":
            e["s"] = len(text.encode("utf-8"))
    return out, {"project.json": text}


def manifest_of(project_dir: Path, session_id: str, commit: str = "", scans: Optional[List[str]] = None) -> Dict[str, Any]:
    """The session's manifest. ``scans``: only these scans of it (None: every one) — the manifest then
    carries ``rewritten`` ({path: text}), the files whose content in the archive is not the disk's
    (project.json without the scans left out): the ZIP stays a function of its manifest."""
    project_dir = Path(project_dir)
    listed = entries(project_dir)
    every = scans_of(project_dir)
    kept, rewritten = every, None
    if scans is not None:
        asked = list(dict.fromkeys(scans))
        unknown = [k for k in asked if k not in every]
        if unknown:
            raise ArchiveError(f"{session_id}: no scan {unknown} (its scans: {every})")
        kept = [k for k in every if k in asked]
        if not kept:
            raise ArchiveError(f"{session_id}: choose at least one scan to download")
        if len(kept) < len(every):
            listed, rewritten = _for_scans(project_dir, listed, every, kept)
    m = {"format": FORMAT, "version": VERSION, "session_id": session_id,
         "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stac_commit": commit,
         "files": sum(1 for e in listed if "s" in e), "bytes": sum(e.get("s", 0) for e in listed),
         "scans": kept, "entries": listed}
    if rewritten:
        m["rewritten"] = rewritten
    return m


def sizes_of(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """How much of a manifest is each scan's (``by_scan``) and how much the project's own (``common``:
    its IFCs, tickets, stored comparison …) — for a download of some of its scans."""
    dirs = {k: _scan_dir(k) + "/" for k in manifest["scans"]}
    by_scan = {k: {"files": 0, "bytes": 0} for k in manifest["scans"]}
    common = {"files": 0, "bytes": 0}
    for e in manifest["entries"]:
        if "s" in e:
            owner = next((k for k, d in dirs.items() if e["p"].startswith(d)), None)
            bucket = by_scan[owner] if owner is not None else common
            bucket["files"] += 1
            bucket["bytes"] += e["s"]
    return {"by_scan": by_scan, "common": common}


def plan(project_dir: Path, session_id: str) -> Dict[str, Any]:
    """What a download of the session holds (no entry list): files, bytes, scans, and each scan's share."""
    m = manifest_of(project_dir, session_id)
    return {**{k: m[k] for k in ("session_id", "files", "bytes", "scans")}, **sizes_of(m)}


# ── write ──────────────────────────────────────────────────────────────────────────────────────
# The ZIP's own numbers (APPNOTE 4.3 … 4.5), written as zipfile writes them so that zipfile — which
# reads the upload back — and any other tool read the same archive.
ZIP64_LIMIT = (1 << 31) - 1                                    # a size or an offset from here on goes to the ZIP64 fields
COUNT_LIMIT = (1 << 16) - 1
FLAG_UTF8, FLAG_DESCRIPTOR = 0x800, 0x08
CRC_STEP = 64 * BLOCK                                          # a file's running CRC is remembered every 64 MB: a resumed
                                                               # download reads at most that much again before it sends
_BYTES, _FILE, _DESCRIPTOR, _DIRECTORY = 0, 1, 2, 3


def _dos(seconds: int) -> Tuple[int, int]:
    """A ZIP's clock (1980 … 2107, 2 s): what its tools show; the exact date is the manifest's."""
    t = time.gmtime(min(max(seconds, 315532800), 4354819199))
    return t[3] << 11 | t[4] << 5 | t[5] // 2, (t[0] - 1980) << 9 | t[1] << 5 | t[2]


class _Item:
    """One entry of the ZIP: known content (``data``: the manifest, a link's target, a directory) or
    a file of the session (``path``), read when it is sent."""
    __slots__ = ("name", "mode", "time", "date", "size", "mtime", "path", "data", "crc", "offset", "big")

    def __init__(self, name: str, mode: int, seconds: int, data: Optional[bytes] = None, path: Optional[str] = None,
                 size: int = 0, mtime: int = 0):
        self.name = name.encode("utf-8")
        self.mode = mode
        self.time, self.date = _dos(seconds)
        self.data, self.path, self.mtime = data, path, mtime
        self.size = len(data) if data is not None else size
        self.crc = zlib.crc32(data) if data is not None else None
        self.big = self.size >= ZIP64_LIMIT
        self.offset = 0

    def local_header(self) -> bytes:
        described = self.data is None                           # (a file's CRC is only known once it was read: after its bytes)
        flags = FLAG_UTF8 | (FLAG_DESCRIPTOR if described else 0)
        size = 0 if described else self.size
        extra = b""
        if self.big:
            extra, size = struct.pack("<HHQQ", 1, 16, size, size), 0xFFFFFFFF
        return struct.pack("<4sHHHHHLLLHH", b"PK\x03\x04", 45 if self.big else 20, flags, 0, self.time, self.date,
                           0 if described else self.crc, size, size, len(self.name), len(extra)) + self.name + extra

    def descriptor(self, crc: int) -> bytes:
        return struct.pack("<4sLQQ" if self.big else "<4sLLL", b"PK\x07\x08", crc, self.size, self.size)

    def _directory_extra(self) -> Tuple[int, ...]:
        return ((self.size, self.size) if self.big else ()) + ((self.offset,) if self.offset > ZIP64_LIMIT else ())

    def directory_length(self) -> int:
        n = len(self._directory_extra())
        return 46 + len(self.name) + (4 + 8 * n if n else 0)

    def directory_entry(self, crc: int) -> bytes:
        wide = self._directory_extra()
        extra = struct.pack("<HH" + "Q" * len(wide), 1, 8 * len(wide), *wide) if wide else b""
        version = 45 if wide else 20
        flags = FLAG_UTF8 | (FLAG_DESCRIPTOR if self.data is None else 0)
        size = 0xFFFFFFFF if self.big else self.size
        attributes = (self.mode & 0xFFFF) << 16 | (0x10 if stat.S_ISDIR(self.mode) else 0)
        return struct.pack("<4s4B4HL2L5H2L", b"PK\x01\x02", version, 3, version, 0, flags, 0, self.time, self.date, crc,
                           size, size, len(self.name), len(extra), 0, 0, 0, attributes,
                           0xFFFFFFFF if self.offset > ZIP64_LIMIT else self.offset) + self.name + extra


def _end_records(count: int, directory_size: int, directory_offset: int) -> bytes:
    out = b""
    if count > COUNT_LIMIT or directory_offset > ZIP64_LIMIT or directory_size > ZIP64_LIMIT:
        out += struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, directory_size, directory_offset)
        out += struct.pack("<4sLQL", b"PK\x06\x07", 0, directory_offset + directory_size, 1)
        count, directory_size, directory_offset = min(count, 0xFFFF), min(directory_size, 0xFFFFFFFF), min(directory_offset, 0xFFFFFFFF)
    return out + struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, directory_size, directory_offset, 0)


class Layout:
    """THE SESSION'S ZIP AS A FUNCTION OF ITS MANIFEST. Everything is stored, so where each byte of
    the archive lies is known from the manifest alone: ``total`` (the ZIP's size), ``etag`` (the same
    manifest = the same bytes) and ``stream(start, stop)`` — any range of it, read from the session
    at that moment. What is not known before a file was read is its CRC-32, written AFTER its bytes
    and again in the ZIP's directory: a stream that passes over a file keeps it (and its running
    value every CRC_STEP, for a download resumed inside a large file); a range that needs one nobody
    read yet reads that file (slower, never wrong).

    One Layout serves every request of one download — its first bytes and its resumptions."""

    def __init__(self, project_dir: Path, session_id: str, manifest: Dict[str, Any]):
        self.project_dir, self.session_id, self.manifest = Path(project_dir), session_id, manifest
        body = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        self.etag = hashlib.sha256(body).hexdigest()[:32]
        try:
            made = calendar.timegm(time.strptime(manifest.get("created_at", ""), "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            made = 0
        self.items: List[_Item] = [_Item(MANIFEST, stat.S_IFREG | 0o644, made, data=body)]
        rewritten = manifest.get("rewritten") or {}              # {path: text}: not the disk's content (manifest_of)
        for e in manifest["entries"]:
            name = f"{session_id}/{e['p']}"
            if "d" in e:
                self.items.append(_Item(name + "/", stat.S_IFDIR | 0o755, made, data=b""))
            elif "l" in e:
                self.items.append(_Item(name, stat.S_IFLNK | 0o777, made, data=e["l"].encode("utf-8")))
            elif e["p"] in rewritten:
                self.items.append(_Item(name, stat.S_IFREG | e["x"], e["m"] // 1_000_000_000, data=rewritten[e["p"]].encode("utf-8")))
            else:
                self.items.append(_Item(name, stat.S_IFREG | e["x"], e["m"] // 1_000_000_000, path=e["p"], size=e["s"], mtime=e["m"]))
        self._starts: List[int] = []
        self._pieces: List[Tuple[int, int, int, Optional[bytes]]] = []       # (kind, item, length, bytes when known)
        pos = 0

        def add(kind: int, item: int, length: int, data: Optional[bytes] = None) -> None:
            nonlocal pos
            if length:
                self._starts.append(pos)
                self._pieces.append((kind, item, length, data))
                pos += length

        for i, it in enumerate(self.items):
            it.offset = pos
            header = it.local_header()
            add(_BYTES, i, len(header), header)
            if it.data is not None:
                add(_BYTES, i, len(it.data), it.data)
            else:
                add(_FILE, i, it.size)
                add(_DESCRIPTOR, i, 24 if it.big else 16)
        self.directory_offset = pos
        for i, it in enumerate(self.items):
            add(_DIRECTORY, i, it.directory_length())
        end = _end_records(len(self.items), pos - self.directory_offset, self.directory_offset)
        add(_BYTES, -1, len(end), end)
        self.total = pos
        self._crc: Dict[int, int] = {i: it.crc for i, it in enumerate(self.items) if it.crc is not None}
        self._marks: Dict[Tuple[int, int], int] = {}             # (item, offset) -> the CRC of the file's bytes before it
        self._lock = threading.Lock()

    def _changed(self, it: _Item) -> ArchiveError:
        return ArchiveError(f"{self.session_id}: {it.path} changed while the session was being sent")

    def _read(self, i: int, start: int, stop: int, cancelled: Optional[Callable[[], bool]]) -> Iterator[bytes]:
        """The bytes [start, stop) of file ``i`` — read from the nearest point whose running CRC is
        known (the file's start when there is none), so that the CRC is kept as the read advances."""
        it = self.items[i]
        full = self.project_dir / it.path
        try:
            st = os.stat(full)
        except OSError:
            raise self._changed(it) from None
        if st.st_size != it.size or st.st_mtime_ns != it.mtime:
            raise self._changed(it)
        base = start - start % CRC_STEP
        with self._lock:
            crc = 0 if base == 0 else self._marks.get((i, base))
        if crc is None:
            base, crc = 0, 0
        pos = base
        with open(full, "rb") as f:
            if base:
                f.seek(base)
            while pos < stop:
                if cancelled is not None and cancelled():
                    raise ArchiveError(f"{self.session_id}: the download was cancelled")
                block = f.read(min(BLOCK, it.size - pos))
                if not block:
                    raise self._changed(it)
                crc = zlib.crc32(block, crc)
                lo, hi = max(start - pos, 0), min(stop - pos, len(block))
                if hi > lo:
                    yield block[lo:hi]
                pos += len(block)
                if pos % CRC_STEP == 0 and pos < it.size:
                    with self._lock:
                        self._marks[(i, pos)] = crc
            if pos == it.size:
                if f.read(1):
                    raise self._changed(it)
                with self._lock:
                    self._crc[i] = crc

    def crc(self, i: int, cancelled: Optional[Callable[[], bool]] = None) -> int:
        with self._lock:
            known = self._crc.get(i)
        if known is None:                                         # (nobody read it to its end yet: read it)
            size = self.items[i].size
            for _ in self._read(i, size, size, cancelled):
                pass
            known = self._crc[i]
        return known

    def _parts(self, start: int, stop: int, cancelled: Optional[Callable[[], bool]]) -> Iterator[bytes]:
        k = bisect.bisect_right(self._starts, start) - 1
        while k < len(self._pieces) and self._starts[k] < stop:
            kind, i, length, data = self._pieces[k]
            lo, hi = max(start - self._starts[k], 0), min(stop - self._starts[k], length)
            if hi <= lo:
                k += 1
                continue
            if kind == _FILE:
                yield from self._read(i, lo, hi, cancelled)
            else:
                if kind == _DESCRIPTOR:
                    data = self.items[i].descriptor(self.crc(i, cancelled))
                elif kind == _DIRECTORY:
                    data = self.items[i].directory_entry(self.crc(i, cancelled))
                yield data[lo:hi]
            k += 1

    def stream(self, start: int = 0, stop: Optional[int] = None, cancelled: Optional[Callable[[], bool]] = None) -> Iterator[bytes]:
        """The ZIP's bytes [start, stop), in blocks (the small pieces — headers, a directory of
        thousands of entries, small files — joined)."""
        stop = self.total if stop is None else min(stop, self.total)
        if start < 0 or start > stop:
            raise ValueError(f"no such range: {start}-{stop} of {self.total}")
        held = bytearray()
        for part in self._parts(start, stop, cancelled):
            if not held and len(part) >= BLOCK:
                yield part
                continue
            held += part
            if len(held) >= BLOCK:
                yield bytes(held)
                held.clear()
        if held:
            yield bytes(held)


def write_zip(project_dir: Path, session_id: str, stream, manifest: Optional[Dict[str, Any]] = None,
              progress: Progress = None, cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Write the session to ``stream`` (any object with write(): it need not seek). Returns the
    manifest. Raises ArchiveError when a file is not, at the moment it is read, the one listed."""
    layout = Layout(project_dir, session_id, manifest or manifest_of(project_dir, session_id))
    sent = 0
    for block in layout.stream(cancelled=cancelled):
        stream.write(block)
        sent += len(block)
        if progress is not None:
            progress(sent, layout.total, "")
    return layout.manifest


# ── read ───────────────────────────────────────────────────────────────────────────────────────
def _safe(rel: str) -> bool:
    return bool(rel) and not rel.startswith("/") and "\\" not in rel and ".." not in rel.split("/") and "\0" not in rel


def _checked(m: Any) -> Dict[str, Any]:
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise ArchiveError("this ZIP is not a STAC session (its manifest names another format)")
    if m.get("version") != VERSION:
        raise ArchiveError(f"this session was written in format version {m.get('version')}; this STAC reads version {VERSION}")
    sid = m.get("session_id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        raise ArchiveError("the manifest's session id is not a valid session name")
    if not isinstance(m.get("entries"), list):
        raise ArchiveError("the manifest lists no entries")
    return m


def read_manifest(zf: zipfile.ZipFile) -> Dict[str, Any]:
    try:
        raw = zf.read(MANIFEST)
    except KeyError:
        raise ArchiveError(f"this ZIP is not a STAC session: it has no {MANIFEST}") from None
    try:
        return _checked(json.loads(raw))
    except ValueError as e:
        if isinstance(e, ArchiveError):
            raise
        raise ArchiveError(f"{MANIFEST} is not readable") from None


def _target(m: Dict[str, Any], projects_dir: Path, name: Optional[str]) -> str:
    target = (name or m["session_id"]).strip()
    if not ID_RE.match(target):
        raise ArchiveError("a session's name can only contain letters, numbers, dashes and underscores")
    if (Path(projects_dir) / target).exists():
        raise ArchiveError(f"a session named '{target}' already exists — give this one another name, or delete that one first")
    return target


def manifest_at_head(zip_path: Path) -> Tuple[Optional[Dict[str, Any]], int]:
    """The manifest read from the FIRST bytes of a ZIP that is still arriving (a session's ZIP
    carries it first, stored): (manifest, 0) once the file holds it whole; (None, n) while its
    first n bytes are needed to say; (None, 0) when this ZIP does not begin with it — the verdict
    then waits for the whole file (validate)."""
    with open(zip_path, "rb") as f:
        head = f.read(30)
        if len(head) < 30:
            return None, 30
        sig, _version, flags, method, _t, _d, crc, size, _usize, name_len, extra_len = struct.unpack("<4sHHHHHLLLHH", head)
        if sig != b"PK\x03\x04" or method != 0 or flags & (FLAG_DESCRIPTOR | 0x1) or size == 0xFFFFFFFF:
            return None, 0
        if name_len != len(MANIFEST):
            return None, 0
        need = 30 + name_len + extra_len + size
        rest = f.read(need - 30)
    if len(rest) >= name_len and rest[:name_len] != MANIFEST.encode():
        return None, 0
    if len(rest) < need - 30:
        return None, need
    body = rest[name_len + extra_len:]
    if zlib.crc32(body) != crc:
        raise ArchiveError(f"the ZIP is damaged: {MANIFEST} does not match its checksum")
    try:
        return _checked(json.loads(body)), 0
    except ValueError as e:
        if isinstance(e, ArchiveError):
            raise
        raise ArchiveError(f"{MANIFEST} is not readable") from None


def check_head(m: Dict[str, Any], projects_dir: Path, name: Optional[str] = None) -> Dict[str, Any]:
    """What can be refused from the manifest alone, while the rest of the ZIP is still on its way:
    the name the session would take (valid, and free) and the manifest's own totals."""
    target = _target(m, projects_dir, name)
    files = sum(1 for e in m["entries"] if isinstance(e, dict) and "s" in e)
    total = sum(e.get("s", 0) for e in m["entries"] if isinstance(e, dict) and isinstance(e.get("s"), int))
    if files != m.get("files") or total != m.get("bytes"):
        raise ArchiveError("the manifest's totals do not match its own entries")
    return {"session_id": target, "source_id": m["session_id"], "files": files, "bytes": total, "scans": m.get("scans", [])}


def validate(zip_path: Path, projects_dir: Path, name: Optional[str] = None) -> Dict[str, Any]:
    """Hold the ZIP to its manifest and to a session's shape BEFORE anything is extracted. Returns
    {"session_id" (the name it will have), "source_id", "files", "bytes", "scans"}; raises
    ArchiveError naming the first thing that is wrong."""
    zip_path, projects_dir = Path(zip_path), Path(projects_dir)
    if not zipfile.is_zipfile(zip_path):
        raise ArchiveError("the file is not a ZIP (or it is cut short: its directory is missing)")
    with zipfile.ZipFile(zip_path) as zf:
        m = read_manifest(zf)
        source_id = m["session_id"]
        target = _target(m, projects_dir, name)
        in_zip: Dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.filename in in_zip:
                raise ArchiveError(f"the ZIP holds {info.filename} twice")
            if info.flag_bits & 0x1:
                raise ArchiveError("the ZIP is encrypted")
            in_zip[info.filename] = info
        prefix = source_id + "/"
        listed = {MANIFEST}
        for e in m["entries"]:
            rel = e.get("p")
            if not isinstance(rel, str) or not _safe(rel):
                raise ArchiveError(f"the manifest names an unsafe path: {rel!r}")
            zname = prefix + rel + ("/" if "d" in e else "")
            info = in_zip.get(zname)
            if info is None:
                raise ArchiveError(f"the ZIP lacks {rel} (the manifest lists it): the archive is incomplete")
            listed.add(zname)
            kind = stat.S_IFMT(info.external_attr >> 16)
            if "l" in e:
                if kind != stat.S_IFLNK:
                    raise ArchiveError(f"{rel} is listed as a link and stored as something else")
                target_link = e["l"]
                if not isinstance(target_link, str) or target_link.startswith("/") or \
                        ".." in os.path.normpath(os.path.join(os.path.dirname(rel), target_link)).split(os.sep):
                    raise ArchiveError(f"the link {rel} points outside the session ({target_link!r})")
            elif "d" in e:
                continue
            else:
                if kind == stat.S_IFLNK:
                    raise ArchiveError(f"{rel} is listed as a file and stored as a link")
                if not isinstance(e.get("s"), int) or info.file_size != e["s"]:
                    raise ArchiveError(f"{rel} is {info.file_size} bytes in the ZIP and {e.get('s')} in the manifest")
        extra = sorted(set(in_zip) - listed)
        if extra:
            raise ArchiveError(f"the ZIP holds {len(extra)} entr{'y' if len(extra) == 1 else 'ies'} the manifest does not list "
                               f"(the first: {extra[0]})")
        # the session's own shape
        paths = {e["p"] for e in m["entries"]}
        if "project.json" not in paths:
            raise ArchiveError("the session has no project.json")
        try:
            meta = json.loads(zf.read(prefix + "project.json"))
            if not isinstance(meta, dict):
                raise ValueError
        except ValueError:
            raise ArchiveError("the session's project.json is not readable") from None
        for p in paths:
            parts = p.split("/")
            if parts[0] == "scans" and len(parts) >= 2:
                if not SCAN_DAY_RE.match(parts[1]):
                    raise ArchiveError(f"scans/{parts[1]} is not a scan day (YYYY-MM-DD)")
                if len(parts) >= 3 and not (parts[2].startswith("src_") or parts[2] == "scan_day.json"):
                    raise ArchiveError(f"scans/{parts[1]}/{parts[2]} is neither a source (src_<name>) nor scan_day.json")
        files = sum(1 for e in m["entries"] if "s" in e)
        total = sum(e.get("s", 0) for e in m["entries"])
        if files != m.get("files") or total != m.get("bytes"):
            raise ArchiveError("the manifest's totals do not match its own entries")
    return {"session_id": target, "source_id": source_id, "files": files, "bytes": total, "scans": m.get("scans", [])}


def extract(zip_path: Path, projects_dir: Path, name: Optional[str] = None, progress: Progress = None,
            cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """validate(), then write the session beside the others and rename it into place LAST. Every
    file's CRC is checked by the reader and its size against the manifest; dates and modes are the
    manifest's. On any failure nothing of it is left."""
    zip_path, projects_dir = Path(zip_path), Path(projects_dir)
    report = validate(zip_path, projects_dir, name)
    target = projects_dir / report["session_id"]
    work = projects_dir / f".incoming-{report['session_id']}-{os.getpid()}"
    if work.exists():
        shutil.rmtree(work)
    done, total = 0, report["bytes"]
    try:
        with zipfile.ZipFile(zip_path) as zf:
            m = read_manifest(zf)
            prefix = m["session_id"] + "/"
            work.mkdir(parents=True)
            links, dated = [], []
            for e in m["entries"]:
                dest = work / e["p"]
                if "d" in e:
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if "l" in e:
                    links.append((dest, e["l"]))
                    continue
                written = 0
                with zf.open(prefix + e["p"]) as src, open(dest, "wb") as out:
                    for block in iter(lambda: src.read(BLOCK), b""):          # (the reader raises on a wrong CRC)
                        if cancelled is not None and cancelled():
                            raise ArchiveError("the upload was cancelled")
                        out.write(block)
                        written += len(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, total, e["p"])
                if written != e["s"]:
                    raise ArchiveError(f"{e['p']}: {written} bytes extracted, the manifest says {e['s']}")
                dated.append((dest, e))
            for dest, link in links:
                os.symlink(link, dest)
            for dest, e in dated:                                              # (last: writing beside a file moves no date)
                os.chmod(dest, e.get("x", 0o644))
                os.utime(dest, ns=(e["m"], e["m"]))
        if target.exists():
            raise ArchiveError(f"a session named '{report['session_id']}' appeared while this one was being written")
        os.rename(work, target)
    except zipfile.BadZipFile as e:
        shutil.rmtree(work, ignore_errors=True)
        raise ArchiveError(f"the ZIP is damaged: {e}") from e
    except OSError as e:
        shutil.rmtree(work, ignore_errors=True)
        if e.errno in (28, 122):                                               # no space / over the volume's quota
            raise ArchiveError("the volume is full: the session was not written (free space and upload it again)") from e
        raise
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return report


def tree_state(project_dir: Path) -> Dict[str, Any]:
    """{path: (bytes, mtime ns, mode) | link target | 'dir'} of a session — what two copies of one
    session have equal (tests, and the proof on a real session)."""
    return {e["p"]: (e["l"] if "l" in e else "dir" if "d" in e else (e["s"], e["m"], e["x"])) for e in entries(project_dir)}
