"""THE REAL RESCALE OF A SCAN — its geometry rewritten at another scale, as a new geometry epoch.

USER 2026-11-13, on the calibration's factor: "pero ojo, el reescalado debe ser real, porque hay
humano de referencia, comparacion contra otras nubes, debe generarse la nube real escalada … sino
solo sirve localmente pero para comparar contra objetos glb de la coleccion, contra el humano de
referencia, contra otras sesiones, etc".

A uniform scale ``s`` about the origin of the scan's frame:

  THE EPOCH (correction/apply.py — the same transaction as every other correction: staged, sealed,
  swapped under the journal, the previous epoch kept and selectable):
    cloud, raw cloud            every point x → s·x (the points with no keyframe too)
    camera poses, every copy    centre c → s·c, the rotation as it is (the intrinsics are pixels)
    floor transform             display = s_f·R·x + t  →  t' = s·t, re-sealed on the new cloud, so
                                the DISPLAY frame scales about its own origin: the floor stays at 0
    object boxes                centre and half extents × s, exactly (never refit)
    instance store              points, boxes and findings follow (correction/invalidate.py)
    depth sidecar, octree       k = s for every frame; the octree rebuilt inside the transaction
  Nothing is deleted or re-fitted: the mask filter and the re-consolidation of the epoch
  transaction are OFF here — a rescale moves every point and changes none.

  WHAT AN EPOCH DOES NOT CARRY, rewritten here, each declared in the report:
    evaluation volumes          the centre follows the scene (× s); the SIZE stays — it is what
                                the user stated in metres
    placed objects              the position follows the scene (× s); the object's own size stays
                                — a reference human or a collection mesh IS at real size
    meshes (GLB, 3D tiles)      scaled by a transform at their root (the tiles' root.transform, a
                                wrapper node of the GLB): the same mesh, exactly, at the new scale
    calibration measurements    re-expressed on the new epoch (points and model lengths × s; the
                                planes refit on the scaled points) and the application recorded —
                                the calibration that was applied then reads a factor of 1
    the project's composition   the scan's transform to the reference composed with 1/s (the
                                reference itself: every other scan's with s)

  LEFT AS THEY ARE, declared: the intake's products (their own probe frame), the precision
  stages' internal records, chat logs and notes (history), a fused cloud made before (to fuse
  again), a BIM deviation computed before (to compare again).

UNDO: applying 1/s (another epoch) — every artifact above follows it. Selecting the previous
epoch restores the epoch's own artifacts only.

PER-AXIS FACTORS ARE NOT APPLIED HERE: an anisotropic scale is not a similarity — the cameras
cannot follow it (the scene would no longer project onto its own images). On his word.
"""
from __future__ import annotations

import dataclasses
import json
import os
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

KIND = "scale"


class RescaleError(ValueError):
    pass


# ── the per-keyframe form of a uniform scale ────────────────────────────────────────────────
def uniform_scale_arrays(session, scale: float):
    """(R_kf, t_kf, k_kf) of x → s·x in the epoch transaction's own terms: the depth of each
    point scaled by s about its camera (k = s), then the keyframe moved by t = (s − 1)·c:
    c + s·(x − c) + (s − 1)·c = s·x, and the pose's centre c → s·c."""
    s = float(scale)
    n = session.n_kf
    R = np.tile(np.eye(3), (n, 1, 1))
    k = np.full(n, s, dtype=np.float64)
    t = np.stack([(s - 1.0) * np.asarray(session.cam_center[int(f)], np.float64) for f in session.frames]) \
        if n else np.zeros((0, 3))
    return R, t, k


# ── what the epoch does not carry ───────────────────────────────────────────────────────────
def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def scale_user_volumes(db_path: Path, scale: float) -> int:
    """The evaluation volumes' centres × s (their sizes are the user's own metres)."""
    import sqlite3
    if not Path(db_path).exists():
        return 0
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT volume_id, center FROM user_volumes").fetchall()
        for vid, blob in rows:
            c = np.frombuffer(blob, dtype=np.float32).astype(np.float64) * float(scale)
            con.execute("UPDATE user_volumes SET center = ? WHERE volume_id = ?",
                        (np.ascontiguousarray(c, dtype=np.float32).tobytes(), int(vid)))
        con.commit()
    except sqlite3.OperationalError:
        return 0                                   # a store with no volumes table
    finally:
        con.close()
    return len(rows)


def scale_scene_objects(path: Path, scale: float) -> int:
    """The placed objects' positions × s (matrix: 16 numbers, column-major — the translation is
    its last column); their own scale and rotation are not touched."""
    path = Path(path)
    if not path.exists():
        return 0
    doc = json.loads(path.read_text())
    n = 0
    for obj in doc.get("objects", []):
        m = obj.get("matrix")
        if isinstance(m, list) and len(m) == 16:
            for k in (12, 13, 14):
                m[k] = float(m[k]) * float(scale)
            n += 1
    _atomic_write(path, json.dumps(doc, indent=1).encode())
    return n


_GLB_MAGIC, _JSON_CHUNK, _WRAPPER = 0x46546C67, 0x4E4F534A, "stac_scale"


def scale_glb(path: Path, scale: float) -> None:
    """Scale a binary glTF by a node at its root: the scene's root nodes become the children of
    one node named ``stac_scale`` whose ``scale`` is [s, s, s] (an existing one is multiplied).
    The JSON chunk alone is rewritten; the binary buffers are copied as they are."""
    path = Path(path)
    with open(path, "rb") as f:
        magic, version, _total = struct.unpack("<III", f.read(12))
        if magic != _GLB_MAGIC:
            raise RescaleError(f"{path}: not a binary glTF")
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != _JSON_CHUNK:
            raise RescaleError(f"{path}: its first chunk is not JSON")
        doc = json.loads(f.read(clen).decode("utf-8"))
        rest_offset = f.tell()
        nodes = doc.setdefault("nodes", [])
        scenes = doc.setdefault("scenes", [{"nodes": []}])
        scene = scenes[int(doc.get("scene", 0))]
        roots = list(scene.get("nodes", []))
        s = float(scale)
        if len(roots) == 1 and nodes[roots[0]].get("name") == _WRAPPER:
            w = nodes[roots[0]]
            w["scale"] = [float(v) * s for v in w.get("scale", [1.0, 1.0, 1.0])]
        else:
            nodes.append({"name": _WRAPPER, "scale": [s, s, s], "children": roots})
            scene["nodes"] = [len(nodes) - 1]
        payload = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        payload += b" " * (-len(payload) % 4)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as out:
            rest_len = path.stat().st_size - rest_offset
            out.write(struct.pack("<III", magic, version, 12 + 8 + len(payload) + rest_len))
            out.write(struct.pack("<II", len(payload), _JSON_CHUNK))
            out.write(payload)
            while True:
                block = f.read(16 << 20)
                if not block:
                    break
                out.write(block)
    os.replace(tmp, path)


def scale_tileset(path: Path, scale: float) -> None:
    """Scale a 3D-Tiles set by its root transform: world = transform · local (16 numbers,
    column-major), so world' = s·world is transform' = S·transform. The tiles' own contents,
    their bounding volumes (local) and geometric errors (scaled by the renderer with the
    transform) are not touched."""
    path = Path(path)
    doc = json.loads(path.read_text())
    root = doc.get("root")
    if not isinstance(root, dict):
        raise RescaleError(f"{path}: no root tile")
    T = np.array(root.get("transform", np.eye(4).reshape(-1, order="F").tolist()), dtype=np.float64).reshape(4, 4, order="F")
    S = np.diag([float(scale)] * 3 + [1.0])
    root["transform"] = [float(v) for v in (S @ T).reshape(-1, order="F")]
    _atomic_write(path, json.dumps(doc).encode())


def scale_meshes(output_dir: Path, scale: float, epoch_to: int, log: Callable[[str], Any] = print) -> List[str]:
    """Every published mesh of the scan (``tsdf/*`` and the GenRecon raw ones): its GLB and its
    tiles scaled at their root, the bounds of its record × s, and the record stamped with the
    epoch it now belongs to."""
    out: List[str] = []
    output_dir = Path(output_dir)
    for base in (output_dir / "tsdf", output_dir / "genrecon"):
        if not base.is_dir():
            continue
        for glb in sorted(base.glob("*/*.glb")):
            scale_glb(glb, scale)
            out.append(str(glb.relative_to(output_dir)))
        for tileset in sorted(base.glob("*/tiles/tileset.json")):
            scale_tileset(tileset, scale)
            out.append(str(tileset.relative_to(output_dir)))
        for meta in sorted(base.glob("*/*.meta.json")):
            doc = json.loads(meta.read_text())
            for key, val in list(doc.items()):
                if key.startswith("bounds") and isinstance(val, (list, dict)):
                    doc[key] = _scaled(val, scale)
            doc["geometry_epoch"] = int(epoch_to)
            doc.setdefault("rescaled", []).append({"scale": float(scale), "epoch_to": int(epoch_to)})
            _atomic_write(meta, json.dumps(doc, indent=1).encode())
    if out:
        log(f"  meshes scaled at their root: {len(out)} file(s)")
    return out


def _scaled(value: Any, scale: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) * float(scale)
    if isinstance(value, list):
        return [_scaled(v, scale) for v in value]
    if isinstance(value, dict):
        return {k: _scaled(v, scale) for k, v in value.items()}
    return value


def rescale_calibration(output_dir: Path, scale: float, epoch_from: int, epoch_to: int,
                        mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The calibration document re-expressed on the new epoch: the measurements and planes that
    were live on ``epoch_from`` have their points × s (the display frame scales about its
    origin), their model lengths recomputed, their planes refit, and their epoch set to
    ``epoch_to`` — so the calibration that was applied still STANDS, and reads a factor of 1.
    Entries of older epochs stay stale, untouched. The application is appended to ``applied``."""
    import calibration3d as C
    output_dir = Path(output_dir)
    if not C.path_of(output_dir).exists():
        return None
    doc = C.load(output_dir)
    s = float(scale)
    n_m = n_p = 0
    for m in doc.get("measurements", []):
        if int(m.get("geometry_epoch", -1)) != int(epoch_from):
            continue
        m["a"] = [float(v) * s for v in m["a"]]
        m["b"] = [float(v) * s for v in m["b"]]
        m["measured_m"] = C.length_of(m["a"], m["b"])
        m["geometry_epoch"] = int(epoch_to)
        n_m += 1
    planes = doc.get("planes") or []
    for k, pl in enumerate(planes):
        if int(pl.get("geometry_epoch", -1)) != int(epoch_from):
            continue
        body = {"points": [[float(v) * s for v in p] for p in pl["points"]],
                "label": pl.get("label", ""), "note": pl.get("note", "")}
        planes[k] = C.make_plane(body, pid=pl["id"], geometry_epoch=int(epoch_to), created=pl.get("created"))
        n_p += 1
    doc.setdefault("applied", []).append({"scale": s, "mode": mode or doc.get("mode", "global"),
                                          "epoch_from": int(epoch_from), "epoch_to": int(epoch_to)})
    C.save(output_dir, doc)
    return {"measurements": n_m, "planes": n_p}


def rescale_composition(paths, scan_key: str, scale: float) -> Optional[Dict[str, Any]]:
    """The project's composition after one of its scans changed scale. A scan's transform M maps
    its RAW cloud into the reference scan's: the rescaled scan's becomes M·S⁻¹ (its points are
    now s·x); when the rescaled scan IS the reference, every other scan's becomes S·M. A fused
    cloud made before holds the old geometry: it is to be fused again (declared)."""
    from project_scans import load_meta
    meta = load_meta(paths)
    comp = meta.get("composition") or {}
    transforms = comp.get("transforms") or {}
    if not transforms:
        return None
    S = np.diag([float(scale)] * 3 + [1.0])
    S_inv = np.diag([1.0 / float(scale)] * 3 + [1.0])
    changed: List[str] = []
    for key, entry in transforms.items():
        if not isinstance(entry, dict) or not entry.get("matrix"):
            continue
        M = np.array(entry["matrix"], dtype=np.float64).reshape(4, 4)
        if scan_key == comp.get("reference"):
            M = S @ M
        elif key == scan_key:
            M = M @ S_inv
        else:
            continue
        entry["matrix"] = [float(v) for v in M.reshape(-1)]
        entry["scale_scan_to_ref"] = float(abs(np.linalg.det(M[:3, :3])) ** (1.0 / 3.0))
        entry.setdefault("rescaled", []).append({"scan": scan_key, "scale": float(scale)})
        changed.append(key)
    if not changed:
        return None
    comp.setdefault("rescaled", []).append({"scan": scan_key, "scale": float(scale)})
    paths.save_project_meta(meta)
    return {"transforms": changed,
            "fused_to_redo": sorted({str(e.get("fused")) for e in transforms.values()
                                     if isinstance(e, dict) and e.get("fused")})}


# ── the rescale ─────────────────────────────────────────────────────────────────────────────
def apply_scale(output_dir, scale: float, *, source: Optional[Dict[str, Any]] = None,
                paths=None, scan_key: Optional[str] = None, cfg=None,
                log: Callable[[str], Any] = print, progress=None) -> Dict[str, Any]:
    """Rewrite the scan in ``output_dir`` at ``scale`` × its present size (see the module).
    ``source``: what decided the factor (the calibration's summary), sealed into the epoch's
    record. ``paths`` / ``scan_key``: the project and the scan's key in it, for the composition.
    Returns the report (also written to ``corrections/report_<id>.json``)."""
    from correction import diagnose as diag_mod
    from correction import ledger
    from correction.apply import stage_transaction, swap_transaction
    from correction.chain import applied_depth_factor
    from correction.config import load_correction_config
    from correction.epoch import current_epoch
    from correction.invalidate import update_instance_store
    from correction.session import load_session

    s = float(scale)
    if not np.isfinite(s) or s <= 0.0:
        raise RescaleError(f"the scale must be a positive number, got {scale!r}")
    if s == 1.0:
        raise RescaleError("the scale is exactly 1: there is nothing to rescale")
    output_dir = Path(output_dir)
    cfg = cfg if cfg is not None else load_correction_config()
    # a rescale moves every point and changes none: no point is deleted, nothing is re-fitted
    cfg = dataclasses.replace(cfg, apply=dataclasses.replace(cfg.apply, mask_filter=False, reconsolidate=False))
    session = load_session(output_dir)
    epoch_from = int(current_epoch(output_dir))
    R_kf, t_kf, k_kf = uniform_scale_arrays(session, s)

    # the display frame scales about its own origin: display = s_f·R·x + t  →  t' = s·t
    floor_npz = None
    ft = output_dir / "floor_transform.npz"
    if ft.exists():
        d = np.load(ft)
        floor_npz = {"s": np.asarray(d["s"], np.float64), "R": np.asarray(d["R"], np.float64),
                     "t": np.asarray(d["t"], np.float64) * s}

    cid = ledger.new_correction_id(*ledger.transform_parts(KIND, epoch_from, session.frames, R_kf, t_kf, k_kf))
    extra = {"stage": "rescale", "scale": s, "source": source or {}}
    k_applied = {int(f): float(k) for f, k in zip(session.frames, applied_depth_factor(output_dir, session.frames))}
    tx = stage_transaction(
        session, cfg, R_kf, t_kf, k_kf, correction_id=cid,
        scale_diag_new=diag_mod.regenerate_scale_diagnostics(
            output_dir, {int(f): s for f in session.frames}, epoch_from + 1, cid, k_applied_by_frame=k_applied),
        floor_npz=floor_npz, log=log, progress=progress, record_extra=extra, global_scale=s,
        floor_leveled_by=(f"rescale × {s:.9g}: the floor transform of epoch {epoch_from} with its translation "
                          f"scaled (correction.rescale; correction_id {cid})"))
    swap_transaction(output_dir, tx, log=log)
    epoch_to = int(tx["epoch_to"])
    store = update_instance_store(output_dir, R_kf, t_kf, k_kf, session.frames, log=log)

    # ── what the epoch does not carry ───────────────────────────────────────────────────────
    report: Dict[str, Any] = {
        "correction_id": cid, "kind": KIND, "scale": s, "epoch_from": epoch_from, "epoch_to": epoch_to,
        "points": int(tx["points_moved"]), "instance_store": store, "source": source or {},
        "provenance": "tool_measured",
        "volumes_moved": scale_user_volumes(output_dir / "scene_r.db", s),
        "placed_objects_moved": scale_scene_objects(output_dir / "scene_objects.json", s),
        "meshes_scaled": scale_meshes(output_dir, s, epoch_to, log=log),
        "calibration": rescale_calibration(output_dir, s, epoch_from, epoch_to, mode=(source or {}).get("mode")),
        "composition": rescale_composition(paths, scan_key, s) if paths is not None and scan_key else None,
        "left_as_they_are": ["intake products (their own probe frame)", "precision stages' internal records",
                             "chat logs and notes", "a fused cloud made before (fuse again)",
                             "a BIM deviation computed before (compare again)"],
        "undo": f"apply the scale {1.0 / s:.12g} (another epoch)",
    }
    rp = output_dir / "corrections" / f"report_{cid}.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    ledger.record_run(
        output_dir, correction_id=cid, epoch_from=epoch_from, epoch_to=epoch_to, kind=KIND,
        operator="user", instance_ids=[], visits=[], observability=[], anchors=[],
        diagnosis=[{"scale": s, "source": source or {}}], gates=[], overrides={}, verdict="applied",
        report_path=str(rp.relative_to(output_dir)))
    log(f"[rescale] × {s:.9g} applied → epoch {epoch_to} ({report['points']:,} points)")
    return report
