"""Append-only correction ledger + exact per-epoch transforms.

Persists:
  * ``output/corrections.jsonl`` — one ``run`` record per correction attempt
    that reached apply. History is NEVER deleted, and since 2026-09-16 there
    is nothing to undo: every epoch of the session stays on disk and the user
    selects which one is shown.
  * ``output/corrections/epoch_<N>.npz`` — the exact per-keyframe transform
    (R_kf, t_kf, k_kf, real frame numbers) for bit-faithful replay without
    recomputing slerp, and for re-applying corrections to a
    re-reconstruction of the same scene (re-keyed by frame_global).
  * ``output/corrections.timing.jsonl`` — WHEN each run was recorded (wall
    clock) and how long it took, keyed by its correction id. Kept apart from
    the ledger since 2026-10-07 (docs/plan_determinismo.md point 36): the
    ledger, the epoch record and the per-run report are compared byte for byte
    between two runs of the same session, and a clock never repeats.

The ledger is HISTORY: it is never mirrored into scene_r.db any more (point
136, 2026-10-08 — the mirror made the store's bytes depend on every run that
ever reached the session) and no compared product counts its lines
(``correction.epoch.corrections_summary`` reads the live LINEAGE instead).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from correction.epoch import LEDGER_FILE

ALGORITHM_VERSION = "correction/2.0.0 (2026-09-08 redesign)"
EPOCH_NPZ_DIR = "corrections"
TIMING_FILE = "corrections.timing.jsonl"
# the ledger's id format (8 hex characters) — kept so every existing session's ids still read
CORRECTION_ID_HEX = 8


def new_correction_id(*parts: Any) -> str:
    """The id of a correction run. With ``parts`` (what makes the run what it is: its kind, the
    epochs it goes from / to, the digest of the poses it started from and of the transform it
    applies ...) the id is DERIVED from them (``repro.stable_id``, the ledger's 8 hex characters)
    — a re-run of the same correction on the same session writes the same id, the same ledger
    line and the same report file (docs/plan_determinismo.md points 36 / 56 / 137: every producer
    of the Omega → F6 chain AND every writer of the correction / certification passes its parts).
    Without parts it is still the random id of the callers outside both (precision/fuse.py,
    reconstruction/witness/run.py, reconstruction/witness/depth_tracks.py,
    reconstruction/loops/kf_graph.py — to convert in their own packages); nothing in this
    package calls it that way."""
    if parts:
        from repro import stable_id
        return stable_id("correction", *parts, n_hex=CORRECTION_ID_HEX)
    return str(uuid.uuid4())[:CORRECTION_ID_HEX]


def transform_parts(kind: str, epoch_from: int, frames: Sequence[int], R_kf: np.ndarray,
                    t_kf: np.ndarray, k_kf: np.ndarray, b_kf: Optional[np.ndarray] = None,
                    *extra: Any) -> List[Any]:
    """The parts of a transform epoch's id: what it applies, to which keyframes, on top of which
    epoch — the same inputs give the same id (:func:`new_correction_id`)."""
    return [str(kind), int(epoch_from), [int(f) for f in frames], np.asarray(R_kf, np.float64),
            np.asarray(t_kf, np.float64), np.asarray(k_kf, np.float64),
            (np.asarray(b_kf, np.float64) if b_kf is not None else np.zeros(len(k_kf))), *extra]


def record_timing(output_dir, correction_id: str, **times: Any) -> None:
    """Append when a correction run happened (wall clock and any durations) to
    ``corrections.timing.jsonl`` — the one place a run's times live, outside every compared
    artifact (point 36). ``created_at`` is added here."""
    entry = {"correction_id": correction_id,
             "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), **times}
    with open(Path(output_dir) / TIMING_FILE, "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=float) + "\n")


def _ledger_path(output_dir) -> Path:
    return Path(output_dir) / LEDGER_FILE


def read_ledger(output_dir) -> List[dict]:
    p = _ledger_path(output_dir)
    if not p.exists():
        return []
    entries = []
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError as e:
            raise RuntimeError(
                f"corrupt ledger {p} at line {i + 1}: {e} — the ledger is "
                f"append-only evidence; repair the file manually (do not "
                f"delete it)") from e
    return entries


def _append(output_dir, entry: dict) -> None:
    # the ledger line is written with sorted keys and no clock: the same run writes the
    # same bytes (point 36); no mirror into scene_r.db (point 136)
    with open(_ledger_path(output_dir), "a") as f:
        f.write(json.dumps(entry, sort_keys=True, default=float) + "\n")


def record_run(output_dir, *, correction_id: str, epoch_from: int,
               epoch_to: int, kind: str, operator: str,
               instance_ids: List[int], visits: list, observability: list,
               diagnosis: list, anchors: list, gates: list,
               overrides: Optional[dict], report_path: str,
               verdict: str = "applied", elapsed_s: Optional[float] = None) -> dict:
    if verdict not in ("applied", "rejected"):
        raise RuntimeError(f"invalid verdict {verdict!r}")
    # no wall clock in the ledger line (point 36): when the run happened — and how long it took
    # (``elapsed_s``, point 166) — go to corrections.timing.jsonl, keyed by the same id
    entry = {
        "type": "run", "correction_id": correction_id,
        "epoch_from": int(epoch_from), "epoch_to": int(epoch_to),
        "kind": kind, "operator": operator,
        "instance_ids": instance_ids, "visits": visits,
        "observability": observability, "diagnosis": diagnosis,
        "anchors": anchors, "gates": gates,
        "overrides": overrides or {}, "verdict": verdict,
        "report": report_path, "algorithm_version": ALGORITHM_VERSION,
    }
    _append(output_dir, entry)
    times: Dict[str, Any] = {"epoch_to": int(epoch_to), "kind": kind}
    if elapsed_s is not None:
        times["elapsed_s"] = round(float(elapsed_s), 1)
    record_timing(output_dir, correction_id, **times)
    return entry


def ledger_view(output_dir) -> List[dict]:
    """Run records, with the verdict of a LEGACY session folded in.

    Nothing writes verdicts any more (USER 2026-09-16: every epoch stays and is
    selected, never approved or undone), but a session recorded before that
    keeps its approved/undone rows and its history is never rewritten.
    """
    entries = read_ledger(output_dir)
    verdicts = {e["correction_id"]: e for e in entries
                if e.get("type") == "verdict"}
    view = []
    for e in entries:
        if e.get("type") != "run":
            continue
        v = verdicts.get(e["correction_id"])
        row = dict(e)
        if v:
            row["verdict"] = v["verdict"]
            row["verdict_at"] = v["at"]
        view.append(row)
    return view


def applied_runs(output_dir) -> List[dict]:
    """Every run whose epoch is on disk, oldest first (the certification loop
    leaves one per applied iteration). "pending" is the legacy spelling of an
    applied run in sessions written before 2026-09-16."""
    return [row for row in ledger_view(output_dir)
            if row["verdict"] in ("applied", "pending")]


def last_run(output_dir) -> Optional[dict]:
    """The most recent applied run (what the UI shows as the current report)."""
    runs = applied_runs(output_dir)
    return runs[-1] if runs else None


def save_epoch_npz(output_dir, epoch: int, R_kf: np.ndarray,
                   t_kf: np.ndarray, k_kf: np.ndarray,
                   frames: List[int], dir_override: Optional[Path] = None,
                   b_kf: Optional[np.ndarray] = None,
                   dropped: Optional[np.ndarray] = None) -> Path:
    """The exact, re-appliable definition of one epoch.

    ``dropped`` are the row indices, into the cloud as that epoch FOUND it,
    that the epoch deleted — the visit-drift filter removes points for real
    (USER 2026-09-18) and an epoch that moved 22 million points and deleted
    eight thousand is not reproduced by the motion alone.
    """
    d = (Path(dir_override) if dir_override
         else Path(output_dir) / EPOCH_NPZ_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"epoch_{int(epoch)}.npz"
    b = (np.asarray(b_kf, np.float64) if b_kf is not None
         else np.zeros(len(k_kf), np.float64))
    np.savez(p, R_kf=R_kf, t_kf=t_kf, k_kf=k_kf, b_kf=b,
             frames=np.asarray(frames, dtype=np.int64),
             dropped=(np.asarray(dropped, np.int64) if dropped is not None
                      else np.zeros(0, np.int64)))
    return p


def load_epoch_npz(output_dir, epoch: int) -> dict:
    p = Path(output_dir) / EPOCH_NPZ_DIR / f"epoch_{int(epoch)}.npz"
    if not p.exists():
        raise RuntimeError(f"{p} does not exist — epoch {epoch} has no "
                           f"persisted transform — it cannot be selected "
                           f"and the store cannot follow it)")
    d = np.load(p)
    return {"R_kf": d["R_kf"], "t_kf": d["t_kf"], "k_kf": d["k_kf"],
            "b_kf": (d["b_kf"] if "b_kf" in d.files else np.zeros(len(d["k_kf"]))),
            "frames": [int(f) for f in d["frames"]],
            "dropped": (d["dropped"] if "dropped" in d.files
                        else np.zeros(0, np.int64))}
