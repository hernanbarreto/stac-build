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

WRITTEN AS A STREAM (write_zip): no second copy of a 100 GB session is ever on the volume; text is
deflated, everything else (frames, clouds, octrees, meshes: already dense) is stored. A file that
changes while it is being sent ENDS the stream with an error: a half-consistent archive is never
finished.

READ BACK IN TWO STEPS: validate() opens the uploaded ZIP and holds it to the manifest before a
single byte is extracted — the format, one top-level directory, safe paths, every manifest entry
present with its size and nothing beside them, the project's own shape (project.json, the scans'
layout) — and extract() writes it beside the sessions under a hidden name, checks every file's
CRC and size as it goes, sets dates and modes, and renames it into place last. A failure leaves
no session and no partial tree.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

FORMAT = "stac-session"
VERSION = 1
MANIFEST = "STAC_SESSION.json"
ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
SCAN_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
BLOCK = 1 << 20

# what a use of the session writes again, and never part of it
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = ("-shm", ".uploading", ".pyc")
# deflated (text); everything else is stored as it is
TEXT_SUFFIXES = (".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".tsv", ".log", ".md", ".xml", ".ifc", ".html",
                 ".obj", ".mtl", ".svg")

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
    if list(project_dir.rglob("frames.extracting")):
        raise ArchiveError(f"{project_dir.name}: a video is being extracted into it — download it when that ends")
    out: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(project_dir):
        rel_root = os.path.relpath(root, project_dir)
        for d in sorted(dirs):
            full = os.path.join(root, d)
            if os.path.islink(full):
                files.append(d)                                 # (a link to a directory is an entry, not a walk)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not os.path.islink(os.path.join(root, d)))
        kept = 0
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name)).replace(os.sep, "/")
            if os.path.islink(full):
                out.append({"p": rel, "l": os.readlink(full)})
                kept += 1
                continue
            if name.endswith(SKIP_SUFFIXES):
                continue
            st = os.stat(full)
            if not stat.S_ISREG(st.st_mode):
                continue                                        # (a socket, a pipe: not data)
            if name.endswith("-wal"):
                if st.st_size == 0:
                    continue                                    # (an empty write-ahead log: SQLite makes it again)
                raise ArchiveError(f"{project_dir.name}: {rel} holds changes not yet written into its database "
                                   f"({st.st_size} bytes) — the session is being written; close it and download again")
            out.append({"p": rel, "s": st.st_size, "m": st.st_mtime_ns, "x": stat.S_IMODE(st.st_mode)})
            kept += 1
        if not dirs and not kept and rel_root != ".":
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


def manifest_of(project_dir: Path, session_id: str, commit: str = "") -> Dict[str, Any]:
    listed = entries(project_dir)
    return {"format": FORMAT, "version": VERSION, "session_id": session_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stac_commit": commit,
            "files": sum(1 for e in listed if "s" in e), "bytes": sum(e.get("s", 0) for e in listed),
            "scans": scans_of(project_dir), "entries": listed}


def plan(project_dir: Path, session_id: str) -> Dict[str, Any]:
    """What a download of the session holds (no entry list): files, bytes, scans."""
    m = manifest_of(project_dir, session_id)
    return {k: m[k] for k in ("session_id", "files", "bytes", "scans")}


# ── write ──────────────────────────────────────────────────────────────────────────────────────
def _info(name: str, mtime_ns: int, mode: int, size: int = 0) -> zipfile.ZipInfo:
    t = time.gmtime(max(mtime_ns // 1_000_000_000, 315532800))   # (a ZIP's clock starts in 1980; the exact date is the manifest's)
    info = zipfile.ZipInfo(name, date_time=t[:6])
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED if name.lower().endswith(TEXT_SUFFIXES) else zipfile.ZIP_STORED
    info.file_size = size
    return info


def write_zip(project_dir: Path, session_id: str, stream, manifest: Optional[Dict[str, Any]] = None,
              progress: Progress = None, cancelled: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Write the session to ``stream`` (any object with write(): it need not seek). Returns the
    manifest. Raises ArchiveError when a file is not, at the moment it is read, the one listed."""
    project_dir = Path(project_dir)
    manifest = manifest or manifest_of(project_dir, session_id)
    total, done = int(manifest["bytes"]), 0
    with zipfile.ZipFile(stream, "w", allowZip64=True) as zf:
        zf.writestr(_info(MANIFEST, time.time_ns(), stat.S_IFREG | 0o644), json.dumps(manifest, ensure_ascii=False),
                    compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        for e in manifest["entries"]:
            name = f"{session_id}/{e['p']}"
            full = project_dir / e["p"]
            if "d" in e:
                zf.writestr(_info(name + "/", time.time_ns(), stat.S_IFDIR | 0o755), b"")
                continue
            if "l" in e:
                zf.writestr(_info(name, time.time_ns(), stat.S_IFLNK | 0o777), e["l"])
                continue
            st = os.stat(full)
            if st.st_size != e["s"] or st.st_mtime_ns != e["m"]:
                raise ArchiveError(f"{session_id}: {e['p']} changed while the session was being sent")
            info = _info(name, e["m"], stat.S_IFREG | e["x"], e["s"])
            sent = 0
            with open(full, "rb") as src, zf.open(info, "w", force_zip64=e["s"] >= (1 << 31) - 1) as dst:
                for block in iter(lambda: src.read(BLOCK), b""):
                    if cancelled is not None and cancelled():
                        raise ArchiveError(f"{session_id}: the download was cancelled")
                    dst.write(block)
                    sent += len(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total, e["p"])
            if sent != e["s"]:
                raise ArchiveError(f"{session_id}: {e['p']} changed while the session was being sent")
    return manifest


# ── read ───────────────────────────────────────────────────────────────────────────────────────
def _safe(rel: str) -> bool:
    return bool(rel) and not rel.startswith("/") and "\\" not in rel and ".." not in rel.split("/") and "\0" not in rel


def read_manifest(zf: zipfile.ZipFile) -> Dict[str, Any]:
    try:
        raw = zf.read(MANIFEST)
    except KeyError:
        raise ArchiveError(f"this ZIP is not a STAC session: it has no {MANIFEST}") from None
    try:
        m = json.loads(raw)
    except ValueError:
        raise ArchiveError(f"{MANIFEST} is not readable") from None
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
        target = (name or source_id).strip()
        if not ID_RE.match(target):
            raise ArchiveError("a session's name can only contain letters, numbers, dashes and underscores")
        if (projects_dir / target).exists():
            raise ArchiveError(f"a session named '{target}' already exists — give this one another name, or delete that one first")
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


def iter_blocks(path: Path) -> Iterator[bytes]:
    with open(path, "rb") as f:
        yield from iter(lambda: f.read(BLOCK), b"")
