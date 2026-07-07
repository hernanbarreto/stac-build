# STAC-Builder — Phase R.8: canonical per-scene instance store (SQLite).
#
# THE single source of objects for Phase 2 (classification), Phase 3 (findings
# anchoring), Phase 5 (Q&A) and Phase 6 (reports). No parallel object structures.
#
# PROVENANCE: schema EXTENDS R3D's per-object stores
#   vendor/r3d/r3d/pipeline/stores/sqlite_store.py
#     scene_objects (:649), object_reconstructions (:657), object_points (:1288)
# Our extensions: masklet refs, vote/onion metrics, per-window OBB history,
# validation status, and vlm_proposed/tool_measured/human_validated provenance.
# numpy arrays are stored as float32 tobytes() blobs (R3D object_points convention).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _blob(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float32).tobytes()


def _arr(b: bytes | None, shape) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32).reshape(shape).copy()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS instances (
    instance_id   INTEGER PRIMARY KEY,
    label         TEXT NOT NULL,
    known         INTEGER DEFAULT 0,        -- maps to a known obra class
    source        TEXT DEFAULT 'autoprompt',
    confidence    REAL DEFAULT 0.0,
    status        TEXT DEFAULT 'proposed',  -- proposed | dubious | validated
    label_origin  TEXT DEFAULT 'vlm_proposed',
    n_views       INTEGER DEFAULT 0,
    first_frame   INTEGER,
    last_frame    INTEGER,
    scene_type    TEXT,
    created_ns    INTEGER
);
CREATE TABLE IF NOT EXISTS instance_points (
    instance_id INTEGER PRIMARY KEY,
    points_blob BLOB,                       -- float32 (N,3) world
    num_points  INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS instance_obb (
    instance_id   INTEGER,
    window_id     TEXT DEFAULT 'global',    -- per-window OBB (R.4) or 'global'
    obb_transform BLOB,                      -- float32 (4,4)
    obb_aabb      BLOB,                      -- float32 (6,)
    position      BLOB,                      -- float32 (3,)
    gravity       BLOB,                      -- float32 (3,) up used
    n_points      INTEGER DEFAULT 0,
    obb_origin    TEXT DEFAULT 'tool_measured',
    PRIMARY KEY (instance_id, window_id)
);
CREATE TABLE IF NOT EXISTS masklet_refs (
    instance_id INTEGER,
    frame_id    INTEGER,
    mask_key    TEXT,                        -- key into seg_masks.npz (f{frame}_o{obj})
    box_xywh    BLOB,                        -- float32 (4,) normalized xywh
    score       REAL,
    PRIMARY KEY (instance_id, frame_id)
);
CREATE TABLE IF NOT EXISTS vote_metrics (
    instance_id    INTEGER PRIMARY KEY,
    mean_entropy   REAL,                     -- R.2 vote entropy (misalignment)
    max_entropy    REAL,
    n_points       INTEGER
);
CREATE TABLE IF NOT EXISTS onion_metrics (
    instance_id  INTEGER,
    seam         TEXT DEFAULT 'global',      -- window seam k->k+1 or 'global'
    bimodal      INTEGER DEFAULT 0,          -- R.3 GMM 2 vs 1 by BIC
    separation_m REAL DEFAULT 0.0,           -- distance between modes IN METRES
    bic_delta    REAL DEFAULT 0.0,
    PRIMARY KEY (instance_id, seam)
);
CREATE TABLE IF NOT EXISTS window_history (
    instance_id      INTEGER,
    window_id        TEXT,
    residual_applied INTEGER DEFAULT 0,
    PRIMARY KEY (instance_id, window_id)
);
CREATE TABLE IF NOT EXISTS instance_classification (
    instance_id   INTEGER PRIMARY KEY,   -- Phase 2 enrichment (no new objects)
    class_final    TEXT,
    material       TEXT,
    state          TEXT,
    notes          TEXT,
    confidence     REAL,
    conflict       INTEGER DEFAULT 0,     -- VLM class contradicts prompt label
    whitelist_eligible INTEGER DEFAULT 0, -- feeds R.5 whitelist for next iteration
    origin         TEXT DEFAULT 'vlm_proposed',
    best_frame     INTEGER
);
CREATE TABLE IF NOT EXISTS findings (
    finding_id    INTEGER PRIMARY KEY AUTOINCREMENT,   -- Phase 3
    instance_id   INTEGER,
    type          TEXT,
    severity      TEXT,
    description   TEXT,
    confidence    REAL,
    frame_id      INTEGER,
    box_xywh      BLOB,
    point3d       BLOB,
    status        TEXT DEFAULT 'proposed',
    origin        TEXT DEFAULT 'vlm_proposed',
    correlated_residual INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scene_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class InstanceStore:
    """SQLite-backed canonical instance store (one file per scene)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── instances ───────────────────────────────────────────────────
    def upsert_instance(self, instance_id: int, label: str, *, known: bool = False,
                        source: str = "autoprompt", confidence: float = 0.0,
                        status: str = "proposed", n_views: int = 0,
                        first_frame: int | None = None, last_frame: int | None = None,
                        scene_type: str | None = None,
                        label_origin: str = "vlm_proposed") -> None:
        self.conn.execute(
            """INSERT INTO instances
               (instance_id,label,known,source,confidence,status,label_origin,
                n_views,first_frame,last_frame,scene_type,created_ns)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(instance_id) DO UPDATE SET
                 label=excluded.label, known=excluded.known, source=excluded.source,
                 confidence=excluded.confidence, status=excluded.status,
                 label_origin=excluded.label_origin, n_views=excluded.n_views,
                 first_frame=excluded.first_frame, last_frame=excluded.last_frame,
                 scene_type=excluded.scene_type""",
            (instance_id, label, int(known), source, confidence, status, label_origin,
             n_views, first_frame, last_frame, scene_type, int(time.time() * 1e9)),
        )
        self.conn.commit()

    def set_status(self, instance_id: int, status: str) -> None:
        self.conn.execute("UPDATE instances SET status=? WHERE instance_id=?",
                          (status, instance_id))
        self.conn.commit()

    # ── points / obb ────────────────────────────────────────────────
    def set_points(self, instance_id: int, points: np.ndarray) -> None:
        pts = np.asarray(points, np.float32).reshape(-1, 3)
        self.conn.execute(
            "INSERT INTO instance_points (instance_id,points_blob,num_points) VALUES (?,?,?) "
            "ON CONFLICT(instance_id) DO UPDATE SET points_blob=excluded.points_blob, "
            "num_points=excluded.num_points",
            (instance_id, _blob(pts), len(pts)))
        self.conn.commit()

    def get_points(self, instance_id: int) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT points_blob,num_points FROM instance_points WHERE instance_id=?",
            (instance_id,)).fetchone()
        if not row or row[0] is None:
            return None
        return _arr(row[0], (row[1], 3))

    def set_obb(self, instance_id: int, transform: np.ndarray, aabb: np.ndarray,
               position: np.ndarray, *, window_id: str = "global",
               gravity: np.ndarray | None = None, n_points: int = 0,
               obb_origin: str = "tool_measured") -> None:
        g = np.asarray(gravity, np.float32) if gravity is not None else np.zeros(3, np.float32)
        self.conn.execute(
            """INSERT INTO instance_obb
               (instance_id,window_id,obb_transform,obb_aabb,position,gravity,n_points,obb_origin)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(instance_id,window_id) DO UPDATE SET
                 obb_transform=excluded.obb_transform, obb_aabb=excluded.obb_aabb,
                 position=excluded.position, gravity=excluded.gravity,
                 n_points=excluded.n_points, obb_origin=excluded.obb_origin""",
            (instance_id, window_id, _blob(np.asarray(transform)), _blob(np.asarray(aabb)),
             _blob(np.asarray(position)), _blob(g), n_points, obb_origin))
        self.conn.commit()

    def get_obb(self, instance_id: int, window_id: str = "global"):
        row = self.conn.execute(
            "SELECT obb_transform,obb_aabb,position FROM instance_obb "
            "WHERE instance_id=? AND window_id=?", (instance_id, window_id)).fetchone()
        if not row:
            return None
        return (_arr(row[0], (4, 4)), _arr(row[1], (6,)), _arr(row[2], (3,)))

    def list_obb_windows(self, instance_id: int) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT window_id FROM instance_obb WHERE instance_id=?", (instance_id,))]

    # ── masklets / metrics ──────────────────────────────────────────
    def add_masklet_ref(self, instance_id: int, frame_id: int, mask_key: str,
                       box_xywh: np.ndarray | None = None, score: float = 0.0) -> None:
        b = _blob(np.asarray(box_xywh)) if box_xywh is not None else None
        self.conn.execute(
            "INSERT OR REPLACE INTO masklet_refs (instance_id,frame_id,mask_key,box_xywh,score) "
            "VALUES (?,?,?,?,?)", (instance_id, frame_id, mask_key, b, score))
        self.conn.commit()

    def set_vote_metrics(self, instance_id: int, mean_entropy: float,
                       max_entropy: float, n_points: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vote_metrics (instance_id,mean_entropy,max_entropy,n_points) "
            "VALUES (?,?,?,?)", (instance_id, mean_entropy, max_entropy, n_points))
        self.conn.commit()

    def set_onion_metric(self, instance_id: int, bimodal: bool, separation_m: float,
                       bic_delta: float, seam: str = "global") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO onion_metrics "
            "(instance_id,seam,bimodal,separation_m,bic_delta) VALUES (?,?,?,?,?)",
            (instance_id, seam, int(bimodal), separation_m, bic_delta))
        self.conn.commit()

    # ── Phase 2 classification ──────────────────────────────────────
    def set_classification(self, instance_id: int, *, class_final: str,
                          material: str = "", state: str = "", notes: str = "",
                          confidence: float = 0.0, conflict: bool = False,
                          whitelist_eligible: bool = False, best_frame: int | None = None,
                          origin: str = "vlm_proposed") -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO instance_classification
               (instance_id,class_final,material,state,notes,confidence,conflict,
                whitelist_eligible,origin,best_frame)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (instance_id, class_final, material, state, notes, confidence,
             int(conflict), int(whitelist_eligible), origin, best_frame))
        self.conn.commit()

    def get_classification(self, instance_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT class_final,material,state,notes,confidence,conflict,"
            "whitelist_eligible,origin,best_frame FROM instance_classification WHERE instance_id=?",
            (instance_id,)).fetchone()
        if not row:
            return None
        keys = ["class_final", "material", "state", "notes", "confidence", "conflict",
                "whitelist_eligible", "origin", "best_frame"]
        d = dict(zip(keys, row))
        d["conflict"] = bool(d["conflict"]); d["whitelist_eligible"] = bool(d["whitelist_eligible"])
        return d

    # ── Phase 3 findings ────────────────────────────────────────────
    def add_finding(self, *, instance_id: int | None, type: str, severity: str,
                   description: str, confidence: float, frame_id: int | None = None,
                   box_xywh=None, point3d=None, correlated_residual: bool = False,
                   status: str = "proposed", origin: str = "vlm_proposed") -> int:
        b = _blob(np.asarray(box_xywh)) if box_xywh is not None else None
        p = _blob(np.asarray(point3d)) if point3d is not None else None
        cur = self.conn.execute(
            """INSERT INTO findings
               (instance_id,type,severity,description,confidence,frame_id,box_xywh,
                point3d,status,origin,correlated_residual)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (instance_id, type, severity, description, confidence, frame_id, b, p,
             status, origin, int(correlated_residual)))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_findings(self, instance_id: int | None = None) -> list[dict]:
        q = ("SELECT finding_id,instance_id,type,severity,description,confidence,"
             "frame_id,point3d,status,origin,correlated_residual FROM findings")
        args: tuple = ()
        if instance_id is not None:
            q += " WHERE instance_id=?"; args = (instance_id,)
        keys = ["finding_id", "instance_id", "type", "severity", "description",
                "confidence", "frame_id", "point3d", "status", "origin", "correlated_residual"]
        out = []
        for r in self.conn.execute(q, args):
            d = dict(zip(keys, r))
            d["point3d"] = (_arr(d["point3d"], (3,)).tolist() if d["point3d"] else None)
            d["correlated_residual"] = bool(d["correlated_residual"])
            out.append(d)
        return out

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO scene_meta (key,value) VALUES (?,?)",
                          (key, str(value)))
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM scene_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    # ── reads for downstream phases ─────────────────────────────────
    def list_instances(self, status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT instance_id,label,known,source,confidence,status,label_origin,n_views,first_frame,last_frame,scene_type FROM instances"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        cols = ["instance_id", "label", "known", "source", "confidence", "status",
                "label_origin", "n_views", "first_frame", "last_frame", "scene_type"]
        return [dict(zip(cols, r)) for r in self.conn.execute(q, args)]

    def get_masklets(self, instance_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT frame_id,mask_key,box_xywh,score FROM masklet_refs WHERE instance_id=? ORDER BY frame_id",
            (instance_id,))
        return [{"frame_id": r[0], "mask_key": r[1],
                 "box_xywh": (_arr(r[2], (4,)).tolist() if r[2] else None), "score": r[3]}
                for r in rows]

    def get_metrics(self, instance_id: int) -> dict[str, Any]:
        v = self.conn.execute(
            "SELECT mean_entropy,max_entropy,n_points FROM vote_metrics WHERE instance_id=?",
            (instance_id,)).fetchone()
        o = self.conn.execute(
            "SELECT bimodal,separation_m,bic_delta FROM onion_metrics WHERE instance_id=? AND seam='global'",
            (instance_id,)).fetchone()
        return {
            "vote": {"mean_entropy": v[0], "max_entropy": v[1], "n_points": v[2]} if v else None,
            "onion": {"bimodal": bool(o[0]), "separation_m": o[1], "bic_delta": o[2]} if o else None,
        }

    def close(self) -> None:
        self.conn.close()
