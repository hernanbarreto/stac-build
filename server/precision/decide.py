"""THE DECISION BY ZONE — what a stage does with its result before it applies it (USER 2026-11-20: "si empeora ese
paso o esa decisión se vuelve al estado mejor y se sigue con el siguiente paso del proceso. Debemos llegar a la
publicación de la nube habiendo pasado por todas las etapas y habiéndonos quedado con el mejor estado del sistema" /
"hay que tomar decisiones sobre los resultados que se van obteniendo").

The flow (docs/pipeline_checklist.md):
 1. there is always a BEST STATE — the session's cameras, its camera model and the scale they carry;
 2. a stage starts from it and solves the WHOLE session: a CANDIDATE;
 3. the candidate is measured against the best state ZONE BY ZONE (a zone = the keyframes a cube owns) with the
    coherence measure (precision/coherence.py), which depends on no stage, and his rule (95 %, 1.1 x the measured
    error);
 4. no zone worse -> the candidate is the new best state;
 5. some zone worse -> the SAME stage is solved again with those zones HELD to the best state, and measured again;
 6. still worse somewhere -> the stage is discarded: nothing is applied, the best state stays, the reason is recorded.
Two solutions are never cut and pasted: what is applied is always ONE solve of the whole session.

The stage calls :func:`decide_stage` BEFORE it applies anything, so nothing has to be undone. Every decision is
written to ``output/precision/decisions.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from precision import coherence as CO

DECISIONS_NAME = "decisions.json"
VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[decide]"
ACCEPTED, ACCEPTED_HELD, DISCARDED = "accepted", "accepted with zones held", "discarded"

SAME_BODY_TOL = 1e-6            # relative spread of the step ratios under which cameras are "the same body, scaled":
#                                 the rounding of poses written and read back, far under any real change of shape

_DEVICE_OVERRIDE: Optional[str] = None


class DecideError(RuntimeError):
    pass


def use_device(dev: Optional[str]) -> Optional[str]:
    """The device of every judge made after this call; None = the configuration's
    (``reconstruction.precision.coherence.device``). The tests set cpu."""
    global _DEVICE_OVERRIDE
    if dev is not None and str(dev) not in ("cpu", "cuda"):
        raise DecideError(f"the measure's device is cpu or cuda, got {dev!r}")
    _DEVICE_OVERRIDE = None if dev is None else str(dev)
    return _DEVICE_OVERRIDE


def camera_scale(c2w: np.ndarray, rec_c2w: np.ndarray) -> float:
    """THE SCALE A STATE'S CAMERAS CARRY against the keyframes' depth records, for a state whose cameras are the
    records' own moved as ONE body and scaled (what every stage before F5 leaves: the placed cubes, the session's one
    scale): the distance between two consecutive cameras in the state over the same distance between the records'
    poses — the same number for every pair. A record's depth is in the units of its own pose; multiplied by this it
    is in the units of the state's cameras.

    A state whose cameras were moved one by one (F5's) has no such number in its steps — its steps also change when a
    tremor is taken out of the cameras (bufferStop 2026-11-20: x0.982 at the median with the depth agreeing BETTER at
    the old scale) — so this function REFUSES it: the scale is decided once, and such a state keeps the scale of the
    state it came from (:func:`decide_stage`)."""
    a = np.linalg.norm(np.diff(np.asarray(c2w, np.float64)[:, :3, 3], axis=0), axis=1)
    b = np.linalg.norm(np.diff(np.asarray(rec_c2w, np.float64)[:, :3, 3], axis=0), axis=1)
    ok = b > 0
    if not ok.any():
        raise DecideError("the records' cameras do not move between any two consecutive keyframes — the scale of a "
                          "state's cameras against them cannot be measured")
    ratio = a[ok] / b[ok]
    s = float(np.median(ratio))
    if not (np.isfinite(s) and s > 0):
        raise DecideError(f"the scale of the cameras against the depth records came out {s!r}")
    spread = float(np.max(np.abs(ratio - s))) / s          # EVERY step: one camera moved alone is not one body
    if spread > SAME_BODY_TOL:
        raise DecideError(f"these cameras are not the records' cameras moved as one body and scaled (their steps are "
                          f"x{float(ratio.min()):.4f} … x{float(ratio.max()):.4f} of the records', spread "
                          f"{spread:.2e}): the depth's scale in this state must be given, it cannot be read from "
                          f"its cameras")
    return s


class Judge:
    """A session ready to be measured in any state: its maps, its pairs and its zones are the SAME for every state
    and are loaded once; a state is its cameras, its camera model and the scale its cameras carry."""

    def __init__(self, session_dir, pcfg, raw_cfg: dict, cam, log: Callable = print):
        self.session_dir = Path(session_dir)
        self.log = log
        self.P = CO.params(pcfg, raw_cfg, _DEVICE_OVERRIDE)
        self.maps, self.zone_of, self.rec_c2w = CO.session_maps(self.session_dir, cam, self.P, log=log)
        self.pairs, self.kinds = CO.session_pairs(self.session_dir, self.maps.frames, self.P)
        if not self.pairs:
            raise DecideError("no pair of keyframes to judge a state with")
        n_pass = int(sum(1 for k in self.kinds if k == CO.KIND_PASS))
        log(f"{LOG_TAG} judges: {len(self.pairs) - n_pass:,} pairs of neighbours and {n_pass:,} pairs of different "
            f"passes, each in both directions; {len(np.unique(self.zone_of))} zone(s)")

    def state(self, label: str, w2c: np.ndarray, cam, depth_scale: Optional[float] = None) -> CO.State:
        """A state of the session. ``depth_scale``: what multiplies the records' depth in it; None = the scale its
        own cameras carry against the records (:func:`camera_scale` — only for cameras that are the records' moved as
        one body)."""
        c2w = np.linalg.inv(np.asarray(w2c, np.float64))
        if len(c2w) != len(self.maps.frames):
            raise DecideError(f"'{label}' has {len(c2w)} cameras, the session {len(self.maps.frames)} keyframes")
        K, lens, _g = CO.grid_camera(cam, self.maps.hw)
        dev = self.P.device
        f = camera_scale(c2w, self.rec_c2w) if depth_scale is None else float(depth_scale)
        return CO.State(label=label, c2w=torch.as_tensor(c2w, dtype=torch.float64, device=dev), K=K, dist=lens,
                        factor=torch.full((len(c2w),), f, dtype=torch.float64, device=dev))

    def measure(self, state: CO.State) -> Dict[str, np.ndarray]:
        return CO.measure(self.maps, state, self.pairs, self.kinds, self.P)

    def compare(self, before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> dict:
        return CO.judge_zones(before, after, self.zone_of, self.P)

    def keyframes_of(self, zones: Sequence[int]) -> np.ndarray:
        """Per keyframe, True = it belongs to one of ``zones``."""
        return np.isin(self.zone_of, np.asarray(list(zones), np.int64))

    def park(self) -> None:
        """The maps leave the device while the stage solves again (its solve wants the card's memory)."""
        m = self.maps
        m.depth, m.valid, m.grey = m.depth.cpu(), m.valid.cpu(), m.grey.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unpark(self) -> None:
        m, dev = self.maps, self.P.device
        m.depth, m.valid, m.grey = m.depth.to(dev), m.valid.to(dev), m.grey.to(dev)


def _line(tag: str, s: dict) -> str:
    out = []
    for kind, rep in s.items():
        if rep.get("along_cm") and rep.get("across_cm"):
            out.append(f"{kind}: along the ray {rep['along_cm']['median']:.2f} cm, across {rep['across_cm']['median']:.2f} cm "
                       f"(medians of {rep['measured']:,} directions)")
        else:
            out.append(f"{kind}: nothing measured")
    return f"{tag} — " + "; ".join(out)


def _record(attempt: int, s0: CO.State, s1: CO.State, m0, m1, verdict: dict, outcome: Optional[str],
            held_zones: Sequence[int]) -> dict:
    return {"attempt": int(attempt), "before": s0.label, "candidate": s1.label, "outcome": outcome,
            "zones_held": [int(z) for z in held_zones],
            "depth_scale": {"before": float(s0.factor[0]), "candidate": float(s1.factor[0])},
            "measure": {"before": CO.summary(m0), "candidate": CO.summary(m1)},
            "zones": {"worse": [int(z) for z in verdict["worse"]], "better": [int(z) for z in verdict["better"]],
                      "same": [int(z) for z in verdict["same"]], "detail": {str(z): r for z, r in verdict["zones"].items()}}}


def decide_stage(judge: Judge, stage: str, before: Tuple[str, np.ndarray, Any], candidate: Tuple[str, np.ndarray, Any],
                 solve_again: Optional[Callable[[np.ndarray], Optional[Tuple[str, np.ndarray, Any]]]],
                 log: Callable = print, owns_scale: bool = False) -> Dict[str, Any]:
    """The flow of the module's docstring for ONE stage. ``before`` / ``candidate``: (label, w2c (N, 4, 4), camera).
    ``solve_again(held)``: the stage solved again with the keyframes of ``held`` (a mask) kept where ``before`` has
    them -> its new candidate; None when the stage cannot hold a zone.

    THE SCALE IS DECIDED ONCE: the depth of the state the stage started from is at the scale its cameras carry. A
    stage that does not decide the scale (``owns_scale`` False: F5 refines cameras) cannot change it — its candidate
    is measured with the SAME depth, so a walk it made longer or shorter reads as what it is: cameras that no longer
    agree with the depth. ``owns_scale`` True: the candidate's depth is at the scale of its own cameras.

    Returns {"outcome": ACCEPTED | ACCEPTED_HELD | DISCARDED, "attempt": which candidate stands (1, 2 or None),
    "held": the mask of the held keyframes (or None), "records": one per attempt}."""
    s0 = judge.state(*before)
    m0 = judge.measure(s0)
    log(_line(f"{LOG_TAG} {stage}: {s0.label}", CO.summary(m0)))
    kept = None if owns_scale else float(s0.factor[0])
    s1 = judge.state(*candidate, depth_scale=kept)
    m1 = judge.measure(s1)
    v1 = judge.compare(m0, m1)
    log(_line(f"{LOG_TAG} {stage}: {s1.label}", CO.summary(m1)))
    log(f"{LOG_TAG} {stage}: zone by zone against the state it started from — {len(v1['better'])} better, "
        f"{len(v1['same'])} the same, {len(v1['worse'])} WORSE{(' ' + str(v1['worse'])) if v1['worse'] else ''}")
    if not v1["worse"]:
        return {"outcome": ACCEPTED, "attempt": 1, "held": None,
                "records": [_record(1, s0, s1, m0, m1, v1, ACCEPTED, [])]}
    records = [_record(1, s0, s1, m0, m1, v1, None, [])]
    held = judge.keyframes_of(v1["worse"])
    cand2 = None
    if solve_again is not None:
        log(f"{LOG_TAG} {stage}: solved again with the {len(v1['worse'])} zone(s) that got worse held to the state it "
            f"started from ({int(held.sum())} keyframe(s))")
        judge.park()
        cand2 = solve_again(held)
        judge.unpark()
    if cand2 is None:
        records[0]["outcome"] = DISCARDED
        log(f"{LOG_TAG} {stage}: DISCARDED — the stage cannot be solved with zones held; the state it started from stays")
        return {"outcome": DISCARDED, "attempt": None, "held": None, "records": records}
    s2 = judge.state(*cand2, depth_scale=kept)
    m2 = judge.measure(s2)
    v2 = judge.compare(m0, m2)
    log(_line(f"{LOG_TAG} {stage}: {s2.label}", CO.summary(m2)))
    log(f"{LOG_TAG} {stage}: zone by zone — {len(v2['better'])} better, {len(v2['same'])} the same, "
        f"{len(v2['worse'])} WORSE{(' ' + str(v2['worse'])) if v2['worse'] else ''}")
    if v2["worse"]:
        records.append(_record(2, s0, s2, m0, m2, v2, DISCARDED, v1["worse"]))
        log(f"{LOG_TAG} {stage}: DISCARDED — still worse with those zones held; the state it started from stays")
        return {"outcome": DISCARDED, "attempt": None, "held": None, "records": records}
    records.append(_record(2, s0, s2, m0, m2, v2, ACCEPTED_HELD, v1["worse"]))
    return {"outcome": ACCEPTED_HELD, "attempt": 2, "held": held, "records": records}


def write_decisions(output_dir, stage: str, records: List[dict]) -> Path:
    """``output/precision/decisions.json``: the record of every decision of the chain, one entry per stage (a stage
    that runs again replaces its own entry; another reconstruction's file is not continued)."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    out = Path(output_dir)
    p = out / "precision" / DECISIONS_NAME
    rid = reconstruction_id_or_none(out)
    doc = None
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("version") == VERSION and old.get(RECONSTRUCTION_ID_KEY) == rid:
            doc = old
    if doc is None:
        doc = {"version": VERSION, "provenance": PROVENANCE, RECONSTRUCTION_ID_KEY: rid, "stages": {}}
    doc["stages"][str(stage)] = records
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=float))
    return p
