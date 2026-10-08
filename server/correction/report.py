"""Structured per-run report.

Every run — applied, rejected or failed by a gate — produces a persistent
report under ``output/corrections/report_<correction_id>.json`` (H5: the old
single ``correction_report.json`` was overwritten each run). The report
carries every stage's numbers, every gate's verdict, and the provenance tags:
measurements are ``tool_measured``; anything applied is ``human_directed``.

No wall clock inside it (docs/plan_determinismo.md points 137 / 166, 2026-10-08):
the report is compared byte for byte between two runs; when a run happened and
how long it took live in ``corrections.timing.jsonl`` (correction.ledger), keyed
by the same correction id.
"""

from __future__ import annotations

import json
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
    """``elapsed_s`` is accepted for the callers that measure it and is NOT written into the
    report (point 166) — pass it to ``ledger.record_run(elapsed_s=...)`` instead."""
    report: Dict[str, Any] = {
        "correction_id": correction_id,
        "kind": kind,
        "operator": operator,
        "status": status,                    # applied | rejected
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
    p.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    return p
