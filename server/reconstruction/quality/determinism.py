"""§10.13 determinism: two runs of the certification loop on two copies of
the same session must produce the same acta within the declared tolerance
(metres for residuals, a fraction for shares and objectives). Seeds are
fixed; where torch has a deterministic mode it is requested.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from reconstruction.quality.known_answer import copy_session


def _scalars(m: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (m or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, (int, float)) and v is not None and np.isfinite(v):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_scalars(v, key + "."))
    return out


def compare_actas(a: dict, b: dict, tol_m: float, tol_frac: float) -> dict:
    """Every scalar of the final metrics compared: metric-unit keys (…_m)
    within tol_m, the rest within tol_frac (relative, floor tol_m)."""
    sa, sb = _scalars(a.get("metrics_final", {})), _scalars(b.get("metrics_final", {}))
    diffs = []
    for k in sorted(set(sa) | set(sb)):
        if k not in sa or k not in sb:
            diffs.append({"key": k, "reason": "missing in one run"})
            continue
        x, y = sa[k], sb[k]
        if k.endswith("_m") or k.endswith("_m_median") or k.endswith("_m_max"):
            ok = abs(x - y) <= tol_m
        else:
            ok = abs(x - y) <= max(tol_frac * max(abs(x), abs(y)), tol_m)
        if not ok:
            diffs.append({"key": k, "a": x, "b": y})
    same_stop = a.get("stopped_at") == b.get("stopped_at") and a.get("epoch_final") == b.get("epoch_final")
    return {"identical_within_tolerance": not diffs and same_stop, "differences": diffs,
            "same_stop": same_stop, "n_scalars": len(sa)}


def determinism(session_dir, cfg=None, correction_cfg=None, work: Optional[Path] = None,
                log: Callable = print, **certify_kw) -> dict:
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.run import certify_session
    import torch
    cfg = cfg or load_loops_config()
    dcfg = cfg.certify.determinism
    t0 = time.time()
    actas = []
    for run in range(2):
        np.random.seed(int(dcfg.seed))
        torch.manual_seed(int(dcfg.seed))
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (TypeError, RuntimeError):
            pass
        copy = copy_session(session_dir, Path(work) / f"run{run}" if work else None)
        actas.append(certify_session(copy, cfg, operator=f"determinism_{run}", log=log,
                                     correction_cfg=correction_cfg, **certify_kw))
    cmp = compare_actas(actas[0], actas[1], dcfg.tol_m, dcfg.tol_frac)
    rep = {"version": 1, "tolerance": {"tol_m": dcfg.tol_m, "tol_frac": dcfg.tol_frac, "seed": dcfg.seed},
           **cmp, "elapsed_s": round(time.time() - t0, 1), "provenance": "tool_measured"}
    log(f"[determinism] {rep['n_scalars']} scalars compared → "
        f"{'identical within tolerance' if rep['identical_within_tolerance'] else 'DIFFERENT: ' + str(rep['differences'][:5])}")
    return rep
