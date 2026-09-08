"""Structured per-run report.

Every run — applied, rejected or failed by a gate — produces a persistent
report under ``output/corrections/report_<correction_id>.json`` (H5: the old
single ``correction_report.json`` was overwritten each run). The report
carries every stage's numbers, every gate's verdict, and the provenance tags:
measurements are ``tool_measured``; anything applied is ``human_directed``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from correction.epoch import stamp
from correction.ledger import ALGORITHM_VERSION, EPOCH_NPZ_DIR


def build_report(*, correction_id: str, kind: str, operator: str,
                 status: str, instance_ids: Optional[List[int]],
                 visits: Optional[list], observability: Optional[list],
                 diagnosis: Optional[list], solutions: Optional[list],
                 distribution: Optional[dict], gates: List[dict],
                 overrides: Optional[dict], epoch_from: int,
                 epoch_to: Optional[int], extra: Optional[dict] = None,
                 rejection_reason: Optional[str] = None,
                 suggestion: Optional[str] = None,
                 elapsed_s: Optional[float] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "correction_id": correction_id,
        "kind": kind,
        "operator": operator,
        "status": status,                    # pending | rejected
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instance_ids": instance_ids,
        "visits": visits,
        "observability": observability,
        "diagnosis": diagnosis,
        "solutions": solutions,
        "distribution": distribution,
        "gates": gates,
        "overrides": overrides or {},
        "epoch_from": epoch_from,
        "epoch_to": epoch_to,
        "rejection_reason": rejection_reason,
        "suggestion": suggestion,
        "elapsed_s": (round(elapsed_s, 1) if elapsed_s is not None else None),
        "algorithm_version": ALGORITHM_VERSION,
        "provenance": "tool_measured",
        "human_directed": True,
    }
    if extra:
        report.update(extra)
    return report


def save_report(output_dir, report: Dict[str, Any]) -> Path:
    d = Path(output_dir) / EPOCH_NPZ_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"report_{report['correction_id']}.json"
    stamp(report, output_dir)
    p.write_text(json.dumps(report, indent=1))
    return p
