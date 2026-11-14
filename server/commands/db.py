"""The database of commands (sqlite + FTS5): built from the validated declarations, read by the
chat, the voice, the interface and the help.

    commands(id, area, tool, kind, runs_on, handler, model, args_json, needs_json, confirm, core,
             status, since, position)
    command_text(command_id, lang, title, about, say_json, done)
    areas(id, position) + area_text(area_id, lang, title, summary)
    command_search — FTS5 over title / about / say per language, diacritics folded: how the
                     commands a sentence may mean are found among all of them
    meta(catalog_sha256, schema_version)

The file (``commands.db`` next to this module) is DERIVED: it is rebuilt, atomically, whenever the
declarations' sha256 differs from the one it carries; at run time the process works on a copy
in memory (a search per question must cost microseconds, and the volume is a network one).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .catalog import AREAS_DIR, LANGS, Catalog, load_catalog, tool_schema

SCHEMA_VERSION = 1
DB_PATH = Path(__file__).resolve().parent / "commands.db"

_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE areas(id TEXT PRIMARY KEY, position INTEGER NOT NULL);
CREATE TABLE area_text(area_id TEXT NOT NULL REFERENCES areas(id), lang TEXT NOT NULL,
                       title TEXT NOT NULL, summary TEXT NOT NULL, PRIMARY KEY(area_id, lang));
CREATE TABLE commands(
    id TEXT PRIMARY KEY, area TEXT NOT NULL REFERENCES areas(id), tool TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK(kind IN ('action', 'query', 'talk')),
    runs_on TEXT NOT NULL CHECK(runs_on IN ('server', 'ui')),
    handler TEXT NOT NULL, model TEXT NOT NULL, args_json TEXT NOT NULL, needs_json TEXT NOT NULL,
    confirm INTEGER NOT NULL, core INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('available', 'planned')),
    since TEXT NOT NULL, position INTEGER NOT NULL);
CREATE TABLE command_text(command_id TEXT NOT NULL REFERENCES commands(id), lang TEXT NOT NULL,
                          title TEXT NOT NULL, about TEXT NOT NULL, say_json TEXT NOT NULL,
                          done TEXT NOT NULL, PRIMARY KEY(command_id, lang));
CREATE VIRTUAL TABLE command_search USING fts5(
    command_id UNINDEXED, lang UNINDEXED, title, about, say,
    tokenize = 'unicode61 remove_diacritics 2');
"""


def fold(text: str) -> str:
    """Lower case, no diacritics, letters and digits only — as the index compares words."""
    t = unicodedata.normalize("NFKD", str(text or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", t))


def build(catalog: Catalog, path: Path | str | None) -> sqlite3.Connection:
    """The database of ``catalog`` in memory; written to ``path`` too (atomically) unless None.
    Deterministic: the same declarations give the same rows in the same order (no clock)."""
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript(_SCHEMA)
    con.execute("INSERT INTO meta VALUES ('catalog_sha256', ?)", (catalog.sha256,))
    con.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    for a in catalog.areas:
        con.execute("INSERT INTO areas VALUES (?, ?)", (a["id"], a["position"]))
        for lang in LANGS:
            con.execute("INSERT INTO area_text VALUES (?, ?, ?, ?)",
                        (a["id"], lang, a["text"][lang]["title"], a["text"][lang]["summary"]))
    for pos, c in enumerate(catalog.commands):
        con.execute("INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            c["id"], c["area"], c["tool"], c["kind"], c["runs_on"], c["handler"], c["model"],
            json.dumps(c["args"], ensure_ascii=False),          # the declared order: the model reads it
            json.dumps(c["needs"]), int(c["confirm"]), int(c["core"]), c["status"], c["since"], pos))
        for lang in LANGS:
            t = c["text"][lang]
            con.execute("INSERT INTO command_text VALUES (?,?,?,?,?,?)", (
                c["id"], lang, t["title"], t["about"], json.dumps(t["say"], ensure_ascii=False), t["done"]))
            con.execute("INSERT INTO command_search VALUES (?,?,?,?,?)", (
                c["id"], lang, t["title"], t["about"], " . ".join(t["say"])))
    con.commit()
    if path is not None:
        path = Path(path)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        if tmp.exists():
            tmp.unlink()
        con.execute("VACUUM INTO ?", (str(tmp),))
        os.replace(tmp, path)
    return con


def _stamp_of(path: Path) -> Optional[str]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT value FROM meta WHERE key = 'catalog_sha256'").fetchone()
            ver = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return row[0] if row and ver and ver[0] == str(SCHEMA_VERSION) else None


class CommandsDB:
    """The commands, read. Thread-safe (one connection in memory, one lock)."""

    def __init__(self, con: sqlite3.Connection, sha256: str):
        self._con = con
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.sha256 = sha256

    # ── rows ────────────────────────────────────────────────────────────────────────────────
    def _rows(self, sql: str, args: Iterable[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._con.execute(sql, tuple(args)).fetchall()

    @staticmethod
    def _command(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        d["args"] = json.loads(d.pop("args_json"))
        d["needs"] = json.loads(d.pop("needs_json"))
        d["confirm"], d["core"] = bool(d["confirm"]), bool(d["core"])
        if "say_json" in d:
            d["say"] = json.loads(d.pop("say_json"))
        return d

    def count(self) -> int:
        return int(self._rows("SELECT COUNT(*) FROM commands")[0][0])

    def areas(self, lang: str = "en") -> List[Dict[str, Any]]:
        return [dict(r) for r in self._rows(
            "SELECT a.id, a.position, t.title, t.summary, "
            "(SELECT COUNT(*) FROM commands c WHERE c.area = a.id) AS commands "
            "FROM areas a JOIN area_text t ON t.area_id = a.id AND t.lang = ? ORDER BY a.position, a.id",
            (_lang(lang),))]

    def list(self, lang: str = "en", area: Optional[str] = None, runs_on: Optional[str] = None,
             status: Optional[str] = "available") -> List[Dict[str, Any]]:
        """Every command (with its text in ``lang``), in the catalogue's order."""
        sql = ("SELECT c.*, t.title, t.about, t.say_json, t.done FROM commands c "
               "JOIN command_text t ON t.command_id = c.id AND t.lang = ? WHERE 1 = 1")
        args: List[Any] = [_lang(lang)]
        for col, val in (("c.area", area), ("c.runs_on", runs_on), ("c.status", status)):
            if val is not None:
                sql += f" AND {col} = ?"
                args.append(val)
        return [self._command(r) for r in self._rows(sql + " ORDER BY c.position", args)]

    def get(self, cid: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        rows = self._rows(
            "SELECT c.*, t.title, t.about, t.say_json, t.done FROM commands c "
            "JOIN command_text t ON t.command_id = c.id AND t.lang = ? WHERE c.id = ?", (_lang(lang), cid))
        return self._command(rows[0]) if rows else None

    def by_tool(self, tool: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        rows = self._rows("SELECT id FROM commands WHERE tool = ?", (tool,))
        return self.get(rows[0][0], lang) if rows else None

    def core(self) -> List[str]:
        """The commands every sentence is offered whatever it says (conversation, help)."""
        return [r[0] for r in self._rows(
            "SELECT id FROM commands WHERE core = 1 AND status = 'available' ORDER BY position")]

    # ── search ──────────────────────────────────────────────────────────────────────────────
    def search(self, text: str, lang: Optional[str] = None, limit: int = 20,
               runs_on: Optional[str] = None) -> List[Dict[str, Any]]:
        """The commands whose name, description or way of asking share words with ``text``, best
        first (BM25 over the index of ``lang``; every language when None). Deterministic: ties
        go to the catalogue's order."""
        words = sorted(set(fold(text).split()))
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        # the index is asked alone (its ranking function cannot be used through a join or an
        # aggregate); the best row per command, the filters and the order are applied here
        sql = "SELECT command_id, rank FROM command_search WHERE command_search MATCH ?"
        args: List[Any] = [match]
        if lang is not None:
            sql += " AND lang = ?"
            args.append(_lang(lang))
        best: Dict[str, float] = {}
        for r in self._rows(sql, args):
            cid, score = r[0], float(r[1])
            if cid not in best or score < best[cid]:
                best[cid] = score
        info = {r["id"]: r for r in self._rows("SELECT id, status, runs_on, position FROM commands")}
        hits = [(score, info[cid]["position"], cid) for cid, score in best.items()
                if info[cid]["status"] == "available"
                and (runs_on is None or info[cid]["runs_on"] == runs_on)]
        hits.sort()
        return [{"id": cid, "score": score} for score, _pos, cid in hits[:int(limit)]]

    # ── what the model is offered ───────────────────────────────────────────────────────────
    def tool_schemas(self, ids: Optional[Iterable[str]] = None, runs_on: Optional[str] = None) -> List[dict]:
        """The commands as functions the model can call, in the catalogue's order — all the
        available ones, or the given ids."""
        rows = self.list(runs_on=runs_on)
        if ids is not None:
            want = set(ids)
            rows = [r for r in rows if r["id"] in want]
        return [tool_schema(r) for r in rows]


def _lang(lang: str) -> str:
    return lang if lang in LANGS else "en"


_DB: Dict[str, Any] = {"db": None, "stamp": None}
_DB_LOCK = threading.Lock()


def _catalog_stamp(directory: Path) -> tuple:
    return tuple(sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in directory.glob("*.yaml")))


def database(directory: Path | str | None = None, path: Path | str | None = None) -> CommandsDB:
    """THE database of commands of this process. The declarations are validated and the file on
    disk rebuilt when their sha256 is not the file's; a change of the declarations while the
    process runs is picked up at the next call (their files' stamps)."""
    directory = Path(directory) if directory is not None else AREAS_DIR
    path = Path(path) if path is not None else DB_PATH
    stamp = (str(directory), str(path), _catalog_stamp(directory))
    with _DB_LOCK:
        if _DB["db"] is not None and _DB["stamp"] == stamp:
            return _DB["db"]
        catalog = load_catalog(directory)
        on_disk = _stamp_of(path) if path.exists() else None
        con = build(catalog, None if on_disk == catalog.sha256 else path)
        _DB.update(db=CommandsDB(con, catalog.sha256), stamp=stamp)
        return _DB["db"]
