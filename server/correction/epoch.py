"""Geometry epoch — the version number of the session's geometry.

What it persists: ``output/geometry_epoch.json``
``{"epoch": N, "correction_id": ..., "parent_epoch": N-1, "kind": ...}``.
A session with no file is epoch 0 (the original reconstruction). No wall clock
since 2026-10-07 (docs/plan_determinismo.md point 36: the record is compared
byte for byte between two runs; when a run happened is in
``corrections.timing.jsonl``, correction.ledger).

It also names the RECONSTRUCTION an artifact was measured on
(:func:`reconstruction_id`): epoch numbers restart at 0 with every new
reconstruction, so "measured on epoch 0" alone cannot tell this reconstruction
from the previous one (points 33 / 34).

What it decides: nothing geometric. It gives every derived artifact a way to
say WHICH geometry it was computed from (``stamp``) and every consumer a way to
refuse or flag geometry from another epoch (``check``). Kept dependency-free
(json + pathlib only) so any server module can import it without cycles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EPOCH_FILE = "geometry_epoch.json"
# a stored epoch lives in output/_epoch_<N>/ (correction.apply re-exports this)
EPOCH_DIR_PREFIX = "_epoch_"
LEDGER_FILE = "corrections.jsonl"

# What an epoch IS relative to its parent:
#   transform — the previous geometry warped per keyframe; the exact warp is
#               persisted in corrections/epoch_<N>.npz, so the instance store
#               can follow it and `correction.replay` can reproduce it;
#   new_cloud — a cloud REBUILT from the depth maps under the same cameras
#               (F7 witness fusion, precision/fuse.py). No camera moves, so no
#               warp exists: the store follows it with the identity and refits
#               its points from the swapped cloud, and replay cannot cross it.
EPOCH_KIND_TRANSFORM = "transform"
EPOCH_KIND_NEW_CLOUD = "new_cloud"
EPOCH_KINDS = (EPOCH_KIND_TRANSFORM, EPOCH_KIND_NEW_CLOUD)
# ledger `kind`s whose run publishes a new cloud — read only for records written
# before `kind` existed (pccr epoch 3, 2026-09-29), never for a record that
# carries its own kind
_LEDGER_KINDS_NEW_CLOUD = ("fuse",)


def read_epoch(output_dir) -> Dict[str, Any]:
    """The session's current epoch record (epoch 0 when no file exists).
    A corrupt file is an error — it means the last swap was interrupted."""
    p = Path(output_dir) / EPOCH_FILE
    if not p.exists():
        return {"epoch": 0, "created_at": None, "correction_id": None,
                "parent_epoch": None}
    try:
        data = json.loads(p.read_text())
        int(data["epoch"])
        return data
    except (ValueError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"corrupt {EPOCH_FILE} at {p}: {e} — the last correction swap "
            f"may have been interrupted; inspect output/_epoch_*/ and the "
            f"ledger before touching the session") from e


def current_epoch(output_dir) -> int:
    return int(read_epoch(output_dir)["epoch"])


def make_epoch_record(epoch: int, correction_id: str,
                      parent_epoch: int,
                      kind: str = EPOCH_KIND_TRANSFORM, **extra: Any) -> Dict[str, Any]:
    """The epoch record. ``extra``: what the writer seals into it — the certification writes
    the sha256 of the frozen run configuration and its input stamp (points 126 / 139); the
    reconstruction's callers pass nothing and get the record they always got."""
    if kind not in EPOCH_KINDS:
        raise ValueError(f"unknown epoch kind {kind!r} (one of {EPOCH_KINDS})")
    rec: Dict[str, Any] = {"epoch": int(epoch),
                           "correction_id": correction_id,
                           "parent_epoch": int(parent_epoch),
                           "kind": kind}
    for k, v in sorted(extra.items()):
        if k in rec:
            raise ValueError(f"make_epoch_record: {k!r} is a fixed field of the record")
        rec[k] = v
    return rec


# What makes a reconstruction THIS reconstruction, as Omega left it (paths relative to output/):
# the keyframe list, the Omega run's own config, its per-keyframe records (depth, confidence, pose,
# K — written once by map_worker._emit_omega_depth, never rewritten by an epoch) and the global
# scale / orientation markers scale_align and orient left on epoch 0. Every epoch's poses and
# clouds DERIVE from these; none of them is touched by a correction.
RECONSTRUCTION_ID_FILES = ("camera_frames.txt", "vggt_omega_config.yaml",
                           "omega_run/results_output", ".metric_scale_applied",
                           ".orientation_applied")
RECONSTRUCTION_ID_KEY = "reconstruction_id"


class ReconstructionIdError(RuntimeError):
    """The session has no reconstruction to name — with the exact reason."""


def reconstruction_id(output_dir) -> str:
    """The identity of the session's reconstruction: sha256 (``repro.stamp``) of
    :data:`RECONSTRUCTION_ID_FILES` — the ones present, and which are absent. An artifact that
    other stages read across runs (scale rows, known dimensions, instance loops, the confidence
    calibration) carries it under :data:`RECONSTRUCTION_ID_KEY`; a reader takes the artifact only
    when it equals this one (:func:`same_reconstruction`). RAISES when the session holds no Omega
    records — there is then no reconstruction to name."""
    from repro import stamp
    out = Path(output_dir)
    rec = out / "omega_run" / "results_output"
    if not rec.is_dir() or not any(rec.glob("frame_*.npz")):
        raise ReconstructionIdError(f"{rec} holds no Omega record (frame_*.npz) — the session has "
                                    f"no reconstruction to name")
    if not (out / "camera_frames.txt").is_file():
        raise ReconstructionIdError(f"{out / 'camera_frames.txt'} is missing — the keyframe list "
                                    f"is part of the reconstruction's identity")
    present = {rel: out / rel for rel in RECONSTRUCTION_ID_FILES if (out / rel).exists()}
    absent = sorted(set(RECONSTRUCTION_ID_FILES) - set(present))
    return stamp(inputs=present, config={"absent": absent})["sha256"]


def reconstruction_id_or_none(output_dir) -> Optional[str]:
    """:func:`reconstruction_id`, or None for a session with no Omega records (a synthetic or
    partial one). For WRITERS: an artifact stamped None is taken by no reader."""
    try:
        return reconstruction_id(output_dir)
    except ReconstructionIdError:
        return None


def same_reconstruction(doc: Optional[Dict[str, Any]], rid: Optional[str]) -> Tuple[bool, str]:
    """(taken, reason): an artifact ``doc`` belongs to the reconstruction ``rid`` only when it
    carries that very id — no id, another id, or no reconstruction to compare with is a refusal,
    with the reason for the report (points 33 / 34)."""
    if rid is None:
        return False, "the session has no reconstruction id (no Omega records)"
    if not isinstance(doc, dict) or RECONSTRUCTION_ID_KEY not in doc:
        return False, f"it carries no {RECONSTRUCTION_ID_KEY} (written before 2026-10-07 or by hand)"
    got = doc.get(RECONSTRUCTION_ID_KEY)
    if got != rid:
        return False, (f"it was measured on another reconstruction ({str(got)[:12]}…, this one is "
                       f"{rid[:12]}…)")
    return True, "same reconstruction"


def _epoch_record_path(output_dir: Path, epoch: int) -> Path:
    """Where epoch ``epoch``'s record lives: the live file when it is the
    epoch being shown, its stored directory otherwise."""
    if int(epoch) == current_epoch(output_dir):
        return output_dir / EPOCH_FILE
    return output_dir / f"{EPOCH_DIR_PREFIX}{int(epoch)}" / EPOCH_FILE


def epoch_kind(output_dir, epoch: int) -> str:
    """``transform`` or ``new_cloud`` for epoch ``epoch`` (see EPOCH_KINDS).

    Read from the epoch's own record. A record written before ``kind`` existed
    is classified from the ledger run that published it (``kind: fuse`` is a
    new cloud); with no such run it is a transform — what every epoch was
    until F7. Epoch 0 is the original reconstruction, a transform of nothing.
    """
    output_dir = Path(output_dir)
    epoch = int(epoch)
    if epoch == 0:
        return EPOCH_KIND_TRANSFORM
    rec_p = _epoch_record_path(output_dir, epoch)
    if rec_p.is_file():
        try:
            kind = json.loads(rec_p.read_text()).get("kind")
        except (OSError, ValueError, TypeError):
            kind = None
        if kind in EPOCH_KINDS:
            return kind
    ledger_p = output_dir / LEDGER_FILE
    if ledger_p.exists():
        for line in ledger_p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if (entry.get("type") == "run"
                    and entry.get("verdict") != "rejected"
                    and entry.get("epoch_to") == epoch
                    and entry.get("kind") in _LEDGER_KINDS_NEW_CLOUD):
                return EPOCH_KIND_NEW_CLOUD
    return EPOCH_KIND_TRANSFORM


def epoch_lineage(output_dir, epoch: int,
                  live: Optional[int] = None) -> List[int]:
    """The ancestry of ``epoch``, root first — [0, …, epoch].

    Each state carries its own ``geometry_epoch.json`` (the live one in the
    session, a stored one inside ``_epoch_<N>/``), and that record names the
    ``parent_epoch`` the correction ran on top of. A session that never
    branched gives the plain [0, 1, …, N]; one where a correction ran on top of
    a re-selected older epoch gives the real line.
    """
    output_dir = Path(output_dir)
    live = current_epoch(output_dir) if live is None else int(live)
    chain: List[int] = []
    e: Optional[int] = int(epoch)
    seen = set()
    while e is not None and e not in seen:
        seen.add(e)
        chain.append(e)
        if e == 0:
            break
        rec_p = ((output_dir / EPOCH_FILE) if e == live
                 else (output_dir / f"{EPOCH_DIR_PREFIX}{e}" / EPOCH_FILE))
        parent: Optional[int] = e - 1          # a session written before the
        if rec_p.is_file():                    # parent was recorded is linear
            try:
                data = json.loads(rec_p.read_text())
                if data.get("parent_epoch") is not None:
                    parent = int(data["parent_epoch"])
            except (OSError, ValueError, TypeError):
                pass
        e = parent
    return list(reversed(chain))


def epoch_path(output_dir, frm: int, to: int) -> List[Tuple[int, bool]]:
    """The edges to travel from epoch ``frm`` to epoch ``to``, in order.

    Each edge is ``(epoch, inverse)``: the transform stored as
    ``corrections/epoch_<epoch>.npz``, applied inverted while climbing from
    ``frm`` to the common ancestor and forward while descending to ``to``.
    On a session that never branched this is exactly the old arithmetic
    (cur…to+1 inverted, or cur+1…to forward).
    """
    live = current_epoch(output_dir)
    a = epoch_lineage(output_dir, frm, live)
    b = epoch_lineage(output_dir, to, live)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    up = [(e, True) for e in reversed(a[i:])]     # undo frm → ancestor
    down = [(e, False) for e in b[i:]]            # apply ancestor → to
    return up + down


def corrections_summary(output_dir) -> Dict[str, Any]:
    """{applied: N, overridden: [correction_id…]} of the geometry being SHOWN.

    ``applied`` counts the transform epochs of the LIVE LINEAGE over the reconstruction's
    PRODUCT epoch — the corrections the geometry on screen actually holds over the published
    cloud — never the ledger's line count (docs/plan_determinismo.md
    point 136, 2026-10-08: the ledger is append-only history, so its count grew with every run
    that ever reached the session, rejected branches and superseded certifications included,
    and that number went into every derived artifact). ``overridden`` lists the ledger runs of
    those epochs that carried an override (none since the manual flow went, 2026-09-24).
    There is no approving (USER 2026-09-16): ``approved`` equals ``applied``.
    """
    output_dir = Path(output_dir)
    live = current_epoch(output_dir)
    lineage = epoch_lineage(output_dir, live)
    # over the PRODUCT epoch (the most recent new-cloud epoch of the lineage: the cloud the
    # reconstruction published) — the corrections the certification applied to the deliverable
    base = 0
    for e in reversed(lineage):
        if int(e) == 0 or epoch_kind(output_dir, int(e)) == EPOCH_KIND_NEW_CLOUD:
            base = int(e)
            break
    after = lineage[lineage.index(base) + 1:]
    applied_epochs = [int(e) for e in after if epoch_kind(output_dir, int(e)) == EPOCH_KIND_TRANSFORM]
    overridden: List[str] = []
    p = output_dir / LEDGER_FILE
    if p.exists() and applied_epochs:
        wanted = set(applied_epochs)
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if (entry.get("type") == "run" and entry.get("verdict") != "rejected"
                    and entry.get("epoch_to") in wanted and entry.get("overrides")):
                overridden.append(str(entry.get("correction_id")))
    n = len(applied_epochs)
    return {"applied": n, "approved": n, "overridden": sorted(set(overridden))}


def stamp(meta: Dict[str, Any], output_dir) -> Dict[str, Any]:
    """Add the provenance trio to a derived artifact's metadata (in place and
    returned): geometry_epoch, human_directed_corrections,
    corrections_overridden. Every measurement/export that leaves the module
    must carry these — a supervision report never hides human intervention."""
    summary = corrections_summary(output_dir)
    meta["geometry_epoch"] = current_epoch(output_dir)
    meta["human_directed_corrections"] = summary["applied"]
    meta["corrections_overridden"] = summary["overridden"]
    return meta


def stamp_nearest(meta: Dict[str, Any], start_dir) -> Dict[str, Any]:
    """``stamp`` for writers that only know their own artifact directory
    (e.g. output/surface_fit/<name>/): walks up the tree to the session's
    output dir (recognised by geometry_epoch.json, cleaned_cloud.ply or
    camera_frames.txt) and stamps against it. A writer outside any session
    tree raises — a deliverable must always declare its geometry epoch."""
    d = Path(start_dir).resolve()
    for _ in range(8):
        if (d / EPOCH_FILE).exists() or (d / "cleaned_cloud.ply").exists() \
                or (d / "camera_frames.txt").exists():
            return stamp(meta, d)
        if d.parent == d:
            break
        d = d.parent
    raise RuntimeError(
        f"stamp_nearest: no session output dir found above {start_dir} — "
        f"cannot stamp the geometry epoch on this artifact")


def check(artifact_meta: Optional[Dict[str, Any]], output_dir) -> Dict[str, Any]:
    """Compare an artifact's stamped epoch with the session's current one.
    Returns {stale: bool, artifact_epoch, current_epoch}. An artifact with no
    stamp predates the epoch system and counts as epoch 0."""
    cur = current_epoch(output_dir)
    art = 0
    if artifact_meta:
        try:
            art = int(artifact_meta.get("geometry_epoch", 0))
        except (TypeError, ValueError):
            art = 0
    return {"stale": art != cur, "artifact_epoch": art, "current_epoch": cur}
