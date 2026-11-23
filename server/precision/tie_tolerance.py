"""F5's anchor tolerance WITHOUT F2 (USER 2026-11-22: "parala, y lanzala sin F2"; the rule of 2026-11-21 for the
pipeline without F2 — "ok, de acuerdo"): the relative tolerance of F5's position anchors is the tie's own error,
the median over the cube graph's seams of (the seam's residual / the distance it was measured at), written as
``output/gauge.json`` in the shape F5 reads (applied_instrument, sigma_by_instrument) before the chain runs.

Imported by the chain runner only (precision.runner.run_chain), never by a step: it enters no step's code stamp.
The document says that F2 did not run and where the tolerance comes from."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

INSTRUMENT = "tie_seams"


class TieToleranceError(RuntimeError):
    pass


def tolerance_from_seams(output_dir: Path) -> dict:
    """The document: every seam (residual, lever, relative) and their median as the instrument's sigma."""
    cg_path = Path(output_dir) / "maplong_run" / "chunk_graph.json"
    if not cg_path.is_file():
        raise TieToleranceError(f"{cg_path} is missing — without F2 the tolerance of F5's anchors is the tie's seams'")
    cg = json.loads(cg_path.read_text())
    rows = [{"seam": [int(e["b"]), int(e["a"])], "residual_m": float(e["sigma_m"]), "lever_m": float(e["lever_m"]),
             "relative": float(e["sigma_m"]) / float(e["lever_m"])} for e in cg.get("seams") or []
            if float(e.get("lever_m", 0.0)) > 0.0]
    if not rows:
        raise TieToleranceError("the session is ONE cube: it has no seam, so the tie measured no tolerance for F5's "
                                "anchors — without F2 there is no rule for that case")
    sigma = float(np.median([r["relative"] for r in rows]))
    return {"version": "tie_seams_1", "provenance": "tool_measured",
            "note": "F2 did not run (reconstruction.precision.gauge.enabled: false). F5's anchor tolerance is the "
                    "tie's own error: the median over the seams of (residual / the distance it was measured at).",
            "applied_instrument": INSTRUMENT, "sigma_by_instrument": {INSTRUMENT: sigma}, "seams": rows}


def write_tolerance_from_seams(output_dir: Path, log: Callable = print) -> float:
    """output/gauge.json from the seams — rewritten only when its content changes (the steps that read it keep their
    stamps on a resume)."""
    from intake.quality import _write_json_atomic
    doc = tolerance_from_seams(output_dir)
    p = Path(output_dir) / "gauge.json"
    old = None
    if p.is_file():
        try:
            old = json.loads(p.read_text())
        except ValueError:
            old = None
    if old != doc:
        _write_json_atomic(p, doc)
    rows, sigma = doc["seams"], doc["sigma_by_instrument"][INSTRUMENT]
    log("[tie-tolerance] F2 off: F5's anchor tolerance from the tie's "
        f"{len(rows)} seam(s): " + ", ".join(f"{r['seam'][0]}<->{r['seam'][1]} {r['residual_m'] * 100:.1f} cm at "
                                             f"{r['lever_m']:.2f} m = {r['relative'] * 100:.1f} %" for r in rows)
        + f" → median {sigma * 100:.2f} % → output/gauge.json" + ("" if old != doc else " (unchanged)"))
    return sigma
