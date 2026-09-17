"""How precisely can THIS session measure the same geometry twice?

USER 2026-09-16, on `sigma_floor_m = max_residual_m / 4` (2.5 cm): every σ floor
is the same statement — *no edge may claim more precision than the pipeline can
repeat* — and the session already measures exactly that, three different ways.
So the floor stops being a constant divided by another constant and becomes the
measurement, declared in the acta with its source.

Precedence, most direct evidence first:

1. ``uncertainty.json`` → ``session_median_m``: the same frame reconstructed in
   two chunks, how far apart its two copies land (pccr 2026-09-16: 4.77 cm over
   174 shared frames). This is repeatability with nothing in between.
2. ``elastic_seams.json``: the per-shared-frame residual left after the seam
   fit — the non-rigid floor both copies share. Same idea, one stage earlier.
3. ``intra_chunk.json`` → ``held_after_cm``: frames of the SAME chunk agreeing
   about the surfaces they both see, measured on held-out pairs. This is the
   one that exists in a SINGLE-CHUNK session, where there are no shared frames
   between chunks and (1) and (2) never get written.
4. the configured constant, used only when the session measured none of the
   above — and said so in the acta.

The number that came out of (1) on pccr, 4.77 cm, is nearly twice the 2.5 cm the
old formula produced: the edges were claiming more certainty than the
reconstruction can reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# The fork writes its own artifacts into output/maplong_run/, not output/ —
# uncertainty.json, elastic_seams.json and intra_chunk.json all live there
# (pccr 2026-09-17: the first run of this module reported "no evidence" while
# all three files existed one directory down).
_SUBDIRS = ("maplong_run", "")


def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _find(output_dir: Path, name: str) -> Optional[dict]:
    """The artifact, wherever the pipeline wrote it."""
    for sub in _SUBDIRS:
        got = _load(Path(output_dir) / sub / name if sub else Path(output_dir) / name)
        if got:
            return got
    return None


def _from_uncertainty(output_dir: Path) -> Optional[Dict[str, Any]]:
    rep = _find(output_dir, "uncertainty.json")
    if not rep:
        return None
    v = rep.get("session_median_m")
    if v is None or not np.isfinite(float(v)) or float(v) <= 0:
        return None
    return {"sigma_floor_m": float(v), "source": "uncertainty.json",
            "detail": {"what": "two copies of the same shared frame",
                       "n_shared_frames": rep.get("n_shared_frames")}}


def _from_elastic_seams(output_dir: Path) -> Optional[Dict[str, Any]]:
    rep = _find(output_dir, "elastic_seams.json")
    if not rep:
        return None
    res = [float(e["residual_m"]) for d in (rep.get("seams") or {}).values()
           for e in d.values()
           if isinstance(e, dict) and e.get("residual_m") is not None
           and np.isfinite(float(e["residual_m"])) and float(e["residual_m"]) > 0]
    if not res:
        return None
    return {"sigma_floor_m": float(np.median(res)), "source": "elastic_seams.json",
            "detail": {"what": "per-shared-frame residual after the seam fit",
                       "n_frames": len(res)}}


def _from_intra_chunk(output_dir: Path) -> Optional[Dict[str, Any]]:
    """The single-chunk answer: frames of one chunk agreeing with each other."""
    rep = _find(output_dir, "intra_chunk.json")
    if not rep:
        return None
    vals = []
    for k, v in (rep.get("chunks") or rep.get("report") or rep).items():
        if not isinstance(v, dict):
            continue
        cm = v.get("held_after_cm")
        if cm is None:
            continue
        m = float(cm) / 100.0
        if np.isfinite(m) and m > 0:
            vals.append(m)
    if not vals:
        return None
    return {"sigma_floor_m": float(np.median(vals)), "source": "intra_chunk.json",
            "detail": {"what": "held-out agreement between frames of the same chunk",
                       "n_chunks": len(vals)}}


def session_repeatability(output_dir, fallback_m: Optional[float] = None,
                          log=None) -> Dict[str, Any]:
    """The session's own measurement floor, with its provenance.

    Never raises and never guesses silently: when nothing was measured the
    fallback is returned WITH ``source='config_fallback'`` so the acta says the
    number was declared, not observed.
    """
    output_dir = Path(output_dir)
    for probe in (_from_uncertainty, _from_elastic_seams, _from_intra_chunk):
        got = probe(output_dir)
        if got is not None:
            if log:
                log(f"[repeatability] σ floor {got['sigma_floor_m'] * 100:.2f} cm "
                    f"— measured by this session ({got['source']}: "
                    f"{got['detail']['what']})")
            return got
    out = {"sigma_floor_m": (float(fallback_m) if fallback_m is not None else None),
           "source": "config_fallback",
           "detail": {"what": "nothing measured — no shared frames and no "
                              "intra-chunk field in this session"}}
    if log:
        log(f"[repeatability] σ floor {out['sigma_floor_m']} m — DECLARED, not "
            f"measured: this session wrote no uncertainty/seam/intra-chunk "
            f"evidence")
    return out
