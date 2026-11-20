"""WHAT A SESSION KEEPS OF ITS RECONSTRUCTION WHEN THE HEAVY PART IS RELEASED.

USER 2026-11-19: "Esto de liberar los intermedios debería hacerse de forma automática, se podría hacer una vez
que termina la nube o recién cuando termina genrecon? Pero como sea y cuando se pueda el sistema debe liberar
ese espacio" / "Ok lo de la huella con archivos chicos dale hacelo y no te mandes más cagadas que son muchas
horas de trabajo y un costo enorme si tenemos que rehacer todas las sesiones nuevamente".

The Omega records (``omega_run/results_output/frame_<n>.npz``: depth, confidence, pose, K, chunk — 2.6 MB a
keyframe) were removed by hand from five scans that day as "intermediates". They are not only that: after the
cloud is published two things are still read FROM them, and both are small —

  the reconstruction's IDENTITY   correction.epoch.reconstruction_id hashes them; the floor transform's seal
                                  (point 158) and every artifact read across runs name that id. Kept in
                                  ``reconstruction_id.json`` (correction.epoch.RECONSTRUCTION_ID_RECORD).
  the CHUNK that owns a keyframe  correction.floor.omega_chunk_of_frames reads it from each record (the floor
                                  solved by chunk, precision.cloud_metrics). Kept in ``omega_chunks.json``.

:func:`keep` writes both FROM the records, while they are on disk — the only order in which nothing is taken on
trust. :func:`recover` is for a session whose records are already gone: the id is the one its own artifacts
carry (every one of them must name the same id, and one beside the floor seal must be among them — a seal does
not vouch for itself), the chunks are the cube plan's owner list (``maplong_run/chunk_sim3.json``; a record's
``chunk`` IS that owner: workers.map_worker._emit_omega_depth writes a keyframe's record from its owner cube
alone, with ``chunk = k``).

    python -m correction.release recover <output dir>            what would be written, and from what
    python -m correction.release recover <output dir> --write    write it
    python -m correction.release keep <output dir>               from the records on disk
    python -m correction.release release <output dir> [--dry]    free what a published session no longer reads

THE RELEASE (:func:`release`, :data:`RELEASED`): once the cloud is PUBLISHED, what only a precision chain that
starts again would read goes — the way back to epoch 0, F6's context views, the DA3 per-frame depth. Each entry
names its readers, found by reading every place of server/ that names the folder (2026-11-19); nothing enters
that list on a description. A session released this way cannot restart or resume its reconstruction: a new one
is a Replace.

NOT RELEASED, declared: the Omega records themselves. The identity and the chunks are kept beside them (so the
day they go nothing is lost), but their per-keyframe DEPTH has one more reader after the cloud — the VLM stage's
association (segmentation.autoprompt ``use_depth``: the depth at a detection's centre, else an assumed 5 m) —
and that stage runs whenever he asks for it. They stay until that reader has another source.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from correction import epoch as E

KEPT_CHUNKS_FILE = "omega_chunks.json"
KEPT_CHUNKS_VERSION = 1
#: folders of output/ no artifact that carries the id lives in (heavy, or another session state)
_SKIP_DIRS = ("potree", "tsdf", "genrecon", "sky_masks", "uncert", "context_views", "_restart_epochs",
              "omega_run", "da3_run", "da3_windows")
_CARRIER_MAX_BYTES = 64 * 1024 * 1024


class ReleaseError(RuntimeError):
    """What was asked cannot be done without taking something on trust — the message says what."""


#: (what it is, folder relative to output/, who reads it) — what a session whose cloud is PUBLISHED no longer
#: reads. Readers as found 2026-11-19 by reading every file of server/ that names the folder.
RELEASED: Tuple[Tuple[str, str, str], ...] = (
    ("the way back to epoch 0", "_restart_epochs",
     "correction.restart.restore_epoch0 alone: a precision chain that starts again from its first step"),
    ("F6's context views", "maplong_run/context_views",
     "precision.depth_on_f5 alone: the vote's witnesses, read while the cloud is being made"),
    ("the DA3 per-frame depth", "da3_run/results_output",
     "the reconstruction's own steps (the metric lock, the anchors, the gauge); elsewhere only as a fallback "
     "of paths that exist (session_io, mask_space, tsdf_export, correction.revisit)"),
)


def _keyframes(out: Path) -> List[int]:
    p = out / "camera_frames.txt"
    if not p.is_file():
        raise ReleaseError(f"{p} is missing — the keyframe list names the frames of the reconstruction")
    return [int(x) for x in p.read_text().split()]


# ── the chunk that owns a keyframe ────────────────────────────────────────────────────────────────
def write_kept_chunks(output_dir, chunk_of_frame: Dict[int, int], rid: str, *, source: str) -> Path:
    from atomic_io import atomic_write_json
    out = Path(output_dir)
    doc = {"version": KEPT_CHUNKS_VERSION, E.RECONSTRUCTION_ID_KEY: rid, "source": str(source),
           "provenance": "tool_measured",
           "chunk_of_frame": {str(int(f)): int(c) for f, c in sorted(chunk_of_frame.items())}}
    p = out / KEPT_CHUNKS_FILE
    try:
        if p.is_file() and json.loads(p.read_text()) == doc:
            return p
    except (OSError, ValueError):
        pass
    atomic_write_json(p, doc, indent=None, sort_keys=True)
    return p


def kept_chunks(output_dir) -> Dict[int, int]:
    """{video frame: the Omega chunk that owns it} as kept for THIS reconstruction; {} when the session
    keeps none (or keeps another reconstruction's — its id is held to the session's, as every artifact read
    across runs is)."""
    out = Path(output_dir)
    p = out / KEPT_CHUNKS_FILE
    if not p.is_file():
        return {}
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    ok, _why = E.same_reconstruction(doc, E.reconstruction_id_or_none(out))
    if not ok or doc.get("version") != KEPT_CHUNKS_VERSION or not isinstance(doc.get("chunk_of_frame"), dict):
        return {}
    return {int(f): int(c) for f, c in doc["chunk_of_frame"].items()}


def _chunks_from_records(out: Path) -> Dict[int, int]:
    import numpy as np
    rec = out / E.RECORDS_REL
    got: Dict[int, int] = {}
    for f in _keyframes(out):
        p = rec / f"frame_{int(f)}.npz"
        if not p.is_file():
            raise ReleaseError(f"{p} is missing — keyframe {f} has no Omega record; nothing is kept from an "
                               f"incomplete set")
        with np.load(p) as z:
            got[int(f)] = int(z["chunk"]) if "chunk" in z.files else 0
    return got


def _chunks_from_plan(out: Path) -> Tuple[Optional[Dict[int, int]], str]:
    """({video frame: owner cube}, how) from the cube plan the run recorded; (None, why) when the session is
    not a cube run or the plan does not name every keyframe."""
    p = out / "maplong_run" / "chunk_sim3.json"
    if not p.is_file():
        return None, f"no {p.relative_to(out)}"
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return None, f"{p.relative_to(out)} is unreadable ({e})"
    if doc.get("layout") != "cubes" or not isinstance(doc.get("owner"), list):
        return None, f"{p.relative_to(out)} is not a cube run's (layout {doc.get('layout')!r})"
    frames = _keyframes(out)
    owner = [int(x) for x in doc["owner"]]
    if len(owner) != len(frames):
        return None, (f"{p.relative_to(out)} names the owner of {len(owner)} keyframes, camera_frames.txt "
                      f"lists {len(frames)}")
    n_cubes = len(doc.get("poses") or [])
    if n_cubes and (min(owner) < 0 or max(owner) >= n_cubes):
        return None, f"{p.relative_to(out)}: an owner outside its {n_cubes} cubes"
    return dict(zip(frames, owner)), f"the cube plan's owner of each of {len(frames)} keyframes ({p.relative_to(out)})"


# ── keep: from the records ───────────────────────────────────────────────────────────────────────
def keep(output_dir, log: Callable[[str], object] = print) -> Dict[str, Any]:
    """Write the identity and the keyframes' chunks FROM the Omega records on disk. The records are not
    touched. Returns {"reconstruction_id", "keyframes"}."""
    out = Path(output_dir)
    chunks = _chunks_from_records(out)
    rid = E.keep_reconstruction_id(out)
    write_kept_chunks(out, chunks, rid, source="records")
    log(f"[release] kept from the {len(chunks)} Omega records: the identity {rid[:12]}… "
        f"({E.RECONSTRUCTION_ID_RECORD}) and each keyframe's chunk ({KEPT_CHUNKS_FILE})")
    return {"reconstruction_id": rid, "keyframes": len(chunks)}


# ── recover: the records are gone ────────────────────────────────────────────────────────────────
def _is_repro_stamp(doc: Any) -> bool:
    """A ``repro.stamp`` record: its ``config`` / ``inputs`` / ``code`` hold DIGESTS of what the step read,
    never the values (pccr 2026-08-31's segmentation.json: the id at its top level, sha256 of the id's JSON
    under stamp.config — one reconstruction, not two)."""
    return isinstance(doc, dict) and "stamp_version" in doc and "sha256" in doc and isinstance(doc.get("config"), dict)


def _id_digest(rid: str) -> str:
    """What a ``repro.stamp`` record keeps of an identity it was computed with: the sha256 of its JSON."""
    import hashlib
    return hashlib.sha256(json.dumps(rid).encode("utf-8")).hexdigest()


def _ids_in(doc: Any, depth: int = 0):
    """("id", value) for every value stored under the identity's key in ``doc`` (the document's own, a seal's,
    a row's) and ("digest", sha256) for the digest a ``repro.stamp`` record keeps of the identity."""
    if depth > 6:
        return
    if _is_repro_stamp(doc):
        d = doc["config"].get(E.RECONSTRUCTION_ID_KEY)
        if isinstance(d, str):
            yield "digest", d
        return
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == E.RECONSTRUCTION_ID_KEY and (v is None or isinstance(v, str)):
                yield "id", v
            elif isinstance(v, (dict, list)):
                yield from _ids_in(v, depth + 1)
    elif isinstance(doc, list):
        for v in doc:
            if isinstance(v, (dict, list)):
                yield from _ids_in(v, depth + 1)


def carriers(output_dir) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """{file relative to output/: the distinct ("id", value) / ("digest", sha256) it carries} over every JSON
    artifact of the session (the kept files themselves left out)."""
    out = Path(output_dir)
    found: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for base, dirs, files in os.walk(out):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("_epoch_"))
        for name in sorted(files):
            if not name.endswith(".json") or name in (E.RECONSTRUCTION_ID_RECORD, KEPT_CHUNKS_FILE):
                continue
            p = Path(base) / name
            try:
                if p.stat().st_size > _CARRIER_MAX_BYTES:
                    continue
                text = p.read_text()
                if f'"{E.RECONSTRUCTION_ID_KEY}"' not in text:
                    continue
                ids = sorted(set(_ids_in(json.loads(text))), key=lambda kv: (kv[0], kv[1] is None, kv[1] or ""))
            except (OSError, ValueError):
                continue
            if ids:
                found[str(p.relative_to(out))] = ids
    return found


def recover(output_dir, *, write: bool = False, log: Callable[[str], object] = print) -> Dict[str, Any]:
    """The identity and the keyframes' chunks of a session whose Omega records are GONE, from what the session
    itself still holds. Refuses — writing nothing — unless every artifact that carries an id names the SAME
    one and an artifact other than the floor seal is among them. Returns the report; writes with ``write``."""
    out = Path(output_dir)
    rec = out / E.RECORDS_REL
    if rec.is_dir() and any(rec.glob("frame_*.npz")):
        raise ReleaseError(f"{rec} holds Omega records — the identity is computed from them (keep), never "
                           f"recovered beside them")
    frames = _keyframes(out)
    found = carriers(out)
    named: Dict[str, List[str]] = {}
    digests: Dict[str, List[str]] = {}
    unstamped: List[str] = []
    for rel, ids in found.items():
        for kind, v in ids:
            if kind == "digest":
                digests.setdefault(str(v), []).append(rel)
            elif v is None:
                unstamped.append(rel)
            else:
                named.setdefault(v, []).append(rel)
    report: Dict[str, Any] = {"output": str(out), "keyframes": len(frames), "ids": {k: sorted(v) for k, v in named.items()},
                              "stamped_without_an_id": sorted(unstamped), "written": False}
    if not named:
        raise ReleaseError(f"{out}: no artifact carries a reconstruction id — there is nothing to recover it from")
    if len(named) > 1:
        raise ReleaseError(f"{out}: its artifacts name {len(named)} different reconstructions "
                           + "; ".join(f"{k[:12]}… in {len(v)} file(s) ({', '.join(sorted(v)[:3])})"
                                       for k, v in sorted(named.items())) + " — nothing is written")
    rid, files = next(iter(named.items()))
    if len(rid) != 64 or any(c not in "0123456789abcdef" for c in rid):
        raise ReleaseError(f"{out}: {rid!r} is not a reconstruction id")
    others = sorted(f for f in files if f != "floor_level.json")
    if not others:
        raise ReleaseError(f"{out}: only the floor seal carries the id — a seal does not vouch for itself")
    # the stamps computed WITH an identity keep its digest: every one of them must be this identity's
    want = _id_digest(rid)
    foreign = {d: v for d, v in digests.items() if d != want}
    if foreign:
        raise ReleaseError(f"{out}: a step's stamp was computed with another identity than {rid[:12]}… — "
                           + "; ".join(f"digest {d[:12]}… in {', '.join(sorted(v)[:3])}" for d, v in sorted(foreign.items()))
                           + " — nothing is written")
    stamped_with = sorted(digests.get(want, []))
    chunks, how = _chunks_from_plan(out)
    report.update({"reconstruction_id": rid, "carriers": sorted(files), "stamped_with_it": stamped_with, "chunks": how})
    log(f"[release] {out}: {len(files)} artifact(s) name ONE reconstruction, {rid[:12]}… "
        f"({', '.join(sorted(files))})"
        + (f"; {len(stamped_with)} step stamp(s) were computed with it ({', '.join(stamped_with)})" if stamped_with else ""))
    if unstamped:
        log(f"[release]   stamped WITHOUT an id (sealed while the session had none — to be sealed again): "
            f"{', '.join(sorted(unstamped))}")
    log(f"[release]   chunks: {how}" if chunks is not None else f"[release]   chunks NOT recovered: {how}")
    if write:
        E.write_reconstruction_id_record(out, rid, source="recovered", carriers=sorted(files),
                                         stamped_with_it=stamped_with)
        if chunks is not None:
            # (the identity must answer first: kept_chunks holds the file to it)
            write_kept_chunks(out, chunks, rid, source="maplong_run/chunk_sim3.json")
        got = E.reconstruction_id(out)
        if got != rid:
            raise ReleaseError(f"{out}: the session answers {got[:12]}… after the identity {rid[:12]}… was written")
        report["written"] = True
        log(f"[release]   written: {E.RECONSTRUCTION_ID_RECORD}" + (f", {KEPT_CHUNKS_FILE}" if chunks is not None else ""))
    return report


# ── release: what a published session no longer reads ───────────────────────────────────────────
def _tree_bytes(path: Path) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            try:
                st = os.lstat(os.path.join(base, name))
            except OSError:
                continue
            total += st.st_size
    return total


def _remove_tree(path: Path) -> None:
    """Remove ``path`` so that its bytes are free AT ONCE: a regular file that nothing else links is emptied
    before it is unlinked (the volume charges allocated size; measured 2026-11-19: 35.5 GB removed this way
    were writable right after). A file with another link is only unlinked — its bytes belong to that link
    too — and a symlink is removed, never followed."""
    import shutil
    import stat as _stat
    if path.is_symlink() or not path.is_dir():
        raise ReleaseError(f"{path} is not a directory of the session")
    for base, _dirs, files in os.walk(path):
        for name in files:
            p = os.path.join(base, name)
            try:
                st = os.lstat(p)
                if _stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and st.st_size:
                    os.truncate(p, 0)
            except OSError:
                continue
    shutil.rmtree(path)


def release(output_dir, *, dry: bool = False, log: Callable[[str], object] = print) -> Dict[str, Any]:
    """Free what a session whose cloud is PUBLISHED no longer reads (:data:`RELEASED`). Before anything goes,
    the identity and the keyframes' chunks are kept FROM the Omega records (which stay); a session that could
    not name its reconstruction afterwards is refused. Returns {"freed": {folder: bytes}, "bytes",
    "reconstruction_id"}; ``dry`` only measures."""
    from precision.product import product_is_live
    out = Path(output_dir)
    live, why = product_is_live(out)
    if not live:
        raise ReleaseError(f"{out}: nothing is released before the cloud is published — {why}")
    rec = out / E.RECORDS_REL
    if rec.is_dir() and any(rec.glob("frame_*.npz")):
        if not dry:
            keep(out, log=log)
    rid = E.reconstruction_id_or_none(out)
    if rid is None and not dry:
        raise ReleaseError(f"{out}: the session cannot name its reconstruction (no Omega records, no kept "
                           f"identity) — nothing is released from it")
    freed: Dict[str, int] = {}
    for what, rel, _readers in RELEASED:
        d = out / rel
        if d.is_symlink() or not d.is_dir():
            continue
        n = _tree_bytes(d)
        if not dry:
            _remove_tree(d)
        freed[rel] = n
        log(f"[release] {'would free' if dry else 'freed'} {n / 1e9:.2f} GB — {what} ({rel})")
    total = sum(freed.values())
    log(f"[release] {out}: {total / 1e9:.2f} GB {'would be' if dry else ''} freed; the session keeps its identity"
        + (f" {rid[:12]}…" if rid else "") + ", its cloud and everything a later stage reads; its reconstruction "
        "can no longer be resumed or restarted (a new one is a Replace)")
    return {"freed": freed, "bytes": total, "reconstruction_id": rid}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("keep", "recover", "release"):
        print(__doc__)
        return 2
    try:
        if argv[0] == "keep":
            keep(argv[1])
        elif argv[0] == "release":
            release(argv[1], dry="--dry" in argv[2:])
        else:
            recover(argv[1], write="--write" in argv[2:])
    except (ReleaseError, E.ReconstructionIdError) as e:
        print(f"[release] REFUSED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
