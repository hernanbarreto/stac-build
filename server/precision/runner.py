"""The precision core of "Reconstruir" (claude_stac.txt §3): EVIDENCE + CORE, in order.

    EVIDENCE  visit_drift MEASURE on epoch 0 (relative scale rows, instance loops)
    CORE      F0 session camera → F2 continuous gauge → F4 native tracks → F3 Omega
              resolution probe (report) → F5 joint refinement → F6 native depth
              (+ COLMAP reference) → F7 witness fusion → epoch N

It runs after the semantics of epoch 0 (VLM + SAM3) as the pipeline's PRECISION stage
(workers/precision_worker.py), and by hand as ``python -m precision.runner --session
<dir>`` — the ONE list of steps below serves both.

Each step runs in its own env (the pycolmap-4 / VGGSfM steps need ``mapanything``) as
a subprocess, deterministic (cuBLAS workspace, fixed threads, MKL reproducible mode,
hash seed). Resume: ``output/precision/chain_state.json`` records every finished step
with the geometry epoch the session was left in; a re-run skips the finished prefix
while the session is still in that epoch, and refuses — naming both epochs — when
someone moved it (the steps are not re-run on top of an epoch they did not produce).
Before a GPU step the semantic service is stopped (the card is not shared).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

STATE_NAME = "chain_state.json"
STATE_VERSION = 1
LOG_TAG = "[precision]"


class ChainError(RuntimeError):
    """The chain cannot go on — with the exact reason."""


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    env: str                    # "da3" | "mapanything" (precision.runner.python_*)
    module: str
    args: Sequence[str] = ()
    gpu: bool = False


STEPS: List[Step] = [
    Step("f0_camera", "F0 session camera", "da3", "precision.camera"),
    Step("f3_measure", "F3 visit drift (measure)", "da3", "correction.visit_drift_run",
         ("--mode", "measure")),
    Step("f2_gauge", "F2 continuous gauge", "da3", "precision.gauge"),
    Step("f4_tracks", "F4 native-pixel tracks", "mapanything", "precision.tracks", gpu=True),
    Step("f3_probe", "F3 Omega resolution probe (report)", "mapanything",
         "precision.omega_probe", gpu=True),
    Step("f5_refine", "F5 joint refinement", "mapanything", "precision.refine"),
    Step("f6_sweep", "F6 native depth sweep", "da3", "precision.depth_sweep", gpu=True),
    Step("f6_colmap", "F6 COLMAP reference (A/B)", "da3", "precision.depth_colmap", gpu=True),
    Step("f7_fuse", "F7 witness fusion", "da3", "precision.fuse"),
]


def step_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads), "MKL_CBWR": "COMPATIBLE"})
    return env


def _epoch(session_dir: Path) -> int:
    from precision.camera import read_geometry_epoch
    return int(read_geometry_epoch(Path(session_dir) / "output"))


def load_state(session_dir: Path) -> dict:
    p = Path(session_dir) / "output" / "precision" / STATE_NAME
    if not p.exists():
        return {"version": STATE_VERSION, "done": []}
    d = json.loads(p.read_text())
    if d.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "done": []}
    return d


def _save_state(session_dir: Path, state: dict) -> None:
    p = Path(session_dir) / "output" / "precision" / STATE_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, p)


def resume_point(state: dict, steps: Sequence[Step], epoch_now: int) -> int:
    """Index of the first step to run. The finished prefix counts only while the
    session is in the epoch its last step left it in."""
    done = state.get("done") or []
    keys = [s.key for s in steps]
    n = 0
    for rec in done:
        if n < len(keys) and rec.get("key") == keys[n]:
            n += 1
        else:
            break
    if n == 0:
        return 0
    last = done[n - 1]
    if int(last.get("epoch_after", -1)) != int(epoch_now):
        raise ChainError(f"the precision chain stopped after '{last['key']}' with the session in "
                         f"epoch {last.get('epoch_after')}, and it is now in epoch {epoch_now} — "
                         f"the remaining steps would run on geometry they did not produce; "
                         f"select epoch {last.get('epoch_after')} or reconstruct again")
    return n


def run_chain(session_dir: Path, pcfg, *, log: Callable = print,
              progress: Optional[Callable[[float, str], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              before_gpu: Optional[Callable[[str], None]] = None,
              steps: Sequence[Step] = STEPS) -> dict:
    session_dir = Path(session_dir).resolve()
    rcfg = pcfg.runner
    py = {"da3": rcfg.python_da3, "mapanything": rcfg.python_mapanything}
    for k, v in py.items():
        if not Path(v).exists():
            raise ChainError(f"precision.runner.python_{k} = {v} does not exist")
    server_dir = Path(__file__).resolve().parent.parent
    state = load_state(session_dir)
    first = resume_point(state, steps, _epoch(session_dir))
    state["done"] = list(state.get("done") or [])[:first]
    if first:
        log(f"{LOG_TAG} resuming after '{steps[first - 1].key}' "
            f"({first}/{len(steps)} step(s) already done in epoch {_epoch(session_dir)})")
    env = step_env(int(rcfg.threads))
    t_chain = time.time()
    for i in range(first, len(steps)):
        s = steps[i]
        if cancelled is not None and cancelled():
            raise ChainError("cancelled")
        if progress is not None:
            progress(100.0 * i / len(steps), f"{s.label} ({i + 1}/{len(steps)})")
        if s.gpu and before_gpu is not None:
            before_gpu(s.label)
        epoch_before = _epoch(session_dir)
        cmd = [py[s.env], "-u", "-m", s.module, "--session", str(session_dir), *s.args]
        log(f"{LOG_TAG} ── {s.label}: {' '.join(cmd[2:])}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(server_dir), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
            if cancelled is not None and cancelled():
                proc.terminate()
                proc.wait()
                raise ChainError(f"cancelled during {s.label}")
        rc = proc.wait()
        if rc != 0:
            raise ChainError(f"{s.label} failed (exit {rc}) — see the lines above; the chain "
                             f"resumes from this step")
        rec = {"key": s.key, "epoch_before": epoch_before, "epoch_after": _epoch(session_dir),
               "seconds": round(time.time() - t0, 1)}
        state["done"].append(rec)
        _save_state(session_dir, state)
        log(f"{LOG_TAG} ✓ {s.label} in {rec['seconds'] / 60:.1f} min (epoch "
            f"{rec['epoch_before']} → {rec['epoch_after']})")
    if progress is not None:
        progress(100.0, "precision core done")
    return {"steps": state["done"], "seconds": round(time.time() - t_chain, 1),
            "epoch": _epoch(session_dir)}


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.runner",
                                 description="The precision core F0 → F7 on a session.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    run_chain(Path(args.session), load_precision_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
