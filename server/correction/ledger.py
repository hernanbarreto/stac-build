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
  * a mirror of the ledger in the instance store's ``scene_meta`` (the jsonl
    stays authoritative; the mirror is refreshed on every append so a store
    rebuild only loses it until the next entry).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from correction.epoch import LEDGER_FILE

ALGORITHM_VERSION = "correction/2.0.0 (2026-09-08 redesign)"
EPOCH_NPZ_DIR = "corrections"


def new_correction_id() -> str:
    return str(uuid.uuid4())[:8]


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
    with open(_ledger_path(output_dir), "a") as f:
        f.write(json.dumps(entry) + "\n")
    _mirror_to_store(output_dir)


def _mirror_to_store(output_dir) -> None:
    """Best-effort mirror into scene_r.db scene_meta (jsonl is authoritative;
    a session without a store simply has no mirror — that is declared here,
    not swallowed)."""
    db = Path(output_dir) / "scene_r.db"
    if not db.exists():
        return
    try:
        import sys
        server_dir = str(Path(__file__).resolve().parents[1])
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        from phase_r.instance_store import InstanceStore
        store = InstanceStore(db)
        store.set_meta("corrections_ledger",
                       json.dumps(read_ledger(output_dir)))
        store.close()
    except Exception as e:  # noqa: BLE001 — mirror only; jsonl already holds the truth
        print(f"[Correction] ledger mirror to instance store failed "
              f"(jsonl is authoritative): {e}", flush=True)


def record_run(output_dir, *, correction_id: str, epoch_from: int,
               epoch_to: int, kind: str, operator: str,
               instance_ids: List[int], visits: list, observability: list,
               diagnosis: list, anchors: list, gates: list,
               overrides: Optional[dict], report_path: str,
               verdict: str = "applied") -> dict:
    if verdict not in ("applied", "rejected"):
        raise RuntimeError(f"invalid verdict {verdict!r}")
    entry = {
        "type": "run", "correction_id": correction_id,
        "epoch_from": int(epoch_from), "epoch_to": int(epoch_to),
        "kind": kind, "operator": operator,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instance_ids": instance_ids, "visits": visits,
        "observability": observability, "diagnosis": diagnosis,
        "anchors": anchors, "gates": gates,
        "overrides": overrides or {}, "verdict": verdict,
        "report": report_path, "algorithm_version": ALGORITHM_VERSION,
    }
    _append(output_dir, entry)
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
                   b_kf: Optional[np.ndarray] = None) -> Path:
    d = (Path(dir_override) if dir_override
         else Path(output_dir) / EPOCH_NPZ_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"epoch_{int(epoch)}.npz"
    b = (np.asarray(b_kf, np.float64) if b_kf is not None
         else np.zeros(len(k_kf), np.float64))
    np.savez(p, R_kf=R_kf, t_kf=t_kf, k_kf=k_kf, b_kf=b,
             frames=np.asarray(frames, dtype=np.int64))
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
            "frames": [int(f) for f in d["frames"]]}
