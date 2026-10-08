"""The precision core of "Reconstruir" (claude_stac.txt §3): EVIDENCE + CORE, in order.

    CORE      F0 session camera → F2 continuous gauge → F4 native tracks → F3 Omega
              resolution probe (report) → F5 joint refinement → F6 native depth →
              the product: F7-C corrected cloud on F6's depth (cloud.source
              omega_corrected) | F6 COLMAP reference + F7 witness fusion (fusion)
              → the chunk / keyframe floor check (report)

It runs INSIDE the reconstruction stage, right after Omega and the chunk merge
(USER 2026-09-28: "f0 a f7 es etapa de reconstrucción, antes de cloudcompy"), through
workers/precision_worker.py, and by hand as ``python -m precision.runner --session
<dir>`` — the ONE list of steps below serves both. The §3 EVIDENCE step (visit_drift
MEASURE over the SAM3 instances) is not in the chain: the instances are projected onto
the cloud only after F7; ``python -m correction.visit_drift_run --mode measure`` stays
available by hand.

Each step runs in its own env (the pycolmap-4 / VGGSfM steps need ``mapanything``) as
a subprocess, deterministic (docs/plan_determinismo.md, 2026-10-07): the cuBLAS
workspace and Python's hash seed (``repro.deterministic_env``), a fixed thread count,
MKL's reproducible mode and ONE OpenBLAS kernel set (``OPENBLAS_CORETYPE``, point 38 —
OpenBLAS otherwise dispatches by CPU model; the pin is VERIFIED in the step's own
interpreter before it runs, because an unknown name is silently ignored) with the CPU
model recorded in every step record.

Before a GPU step the semantic service is stopped and the card is CHECKED FREE
(``repro.require_exclusive_gpu``, point 4): nothing is lowered to fit a shared card —
a step that finds another process on it fails the chain naming it. F2 counts as a GPU
step whenever it has to regenerate the deleted DA3 windows (point 26).

Resume (point 31): ``output/precision/chain_state.json`` records every finished step
with a STAMP of what it consumed (its input files, the config sections it reads, the
code it runs — ``repro.stamp``) and a stamp of what it produced. A re-run skips the
finished prefix only while every saved input stamp equals the one computed now and
every product is still what its step left (``resume_point``); the first step whose
stamp differs is where the chain starts again. The geometry-epoch number is bookkeeping,
not the key: an epoch that restores byte-identical geometry resumes where those bytes
left off. No wall clock in the state file — step and chain durations go to
``chain_timing.json`` next to it (point 36).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import repro  # noqa: E402  (module attribute access on purpose: tests monkeypatch it)

STATE_NAME = "chain_state.json"
TIMING_NAME = "chain_timing.json"
# 2 (2026-10-07, point 31): every record carries the stamps of its inputs and products; a version-1
# state (epoch numbers only) is not evidence of anything and is not reused
STATE_VERSION = 2
LOG_TAG = "[precision]"

# the sha a stamp records for a declared input that does not exist (presence is part of the
# identity: an evidence file appearing or vanishing changes what the step consumes)
ABSENT = "absent"

# OpenBLAS kernel set pinned for every step (point 38). PROVENANCE, measured 2026-10-07 on this
# box (AMD EPYC 7763; flags fma avx avx2, no avx512) with threadpoolctl: the da3 env's numpy
# (OpenBLAS 0.3.23, DYNAMIC_ARCH) dispatched to 'Zen', the mapanything env's numpy and the three
# OpenBLAS builds pycolmap 4.0.4 bundles (0.3.30 / 0.3.31) to 'Haswell'. HASWELL is the one
# kernel set every one of these builds has and the one any AVX2+FMA CPU runs (Intel since 2013,
# AMD since Zen), so a run on another such machine reproduces this one. With the pin all five
# libraries report 'Haswell'; an unknown name is IGNORED by OpenBLAS (OPENBLAS_CORETYPE=BOGUS
# measured: autodetect), which is why the pin is verified in the step's interpreter (blas_probe).
OPENBLAS_CORETYPE = "HASWELL"
# OpenBLAS' build-time name of the kernel set is what threadpoolctl reports back
OPENBLAS_ARCHITECTURE_REPORTED = "Haswell"

# F5's BLAS threads (point 58): the step runs with runner.threads like every other step. VERIFIED
# 2026-10-07 (tests/test_precision_refine.py::test_the_solve_is_bit_identical_under_the_runners_
# blas_threads, and by hand on the same scene): two F5 ladder solves in two interpreters under
# this env at 8 threads hash identically (a8790622…), and identically to the 1-thread solve —
# pycolmap's bundled OpenBLAS runs CHOLMOD's calls on one thread whatever the env says. The test
# re-proves it on every run of the mapanything env's suite; if it ever fails, pin F5 to 1 thread.


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
    # what the step consumes / produces, as session-relative posix paths (files or directories);
    # a path in both is rewritten in place (poses by an epoch step). ``config``: the sections it
    # reads — a field of PrecisionConfig, or ``raw:<dotted path>`` into config.yaml.
    reads: Sequence[str] = ()
    writes: Sequence[str] = ()
    config: Sequence[str] = ()


# files every pose-moving step rewrites (correction.session.POSE_COPY_RELPATHS + the sidecar)
_POSE_FILES = ("output/camera_poses.txt", "output/maplong_run/camera_poses.txt",
               "output/omega_run/camera_poses.txt", "output/da3_run/camera_poses.txt",
               "output/depth_correction.json")
_GEOMETRY_IN = ("output/camera.json", "output/camera_poses.txt", "output/camera_frames.txt")

STEPS: List[Step] = [
    Step("f0_camera", "F0 session camera", "da3", "precision.camera",
         reads=("frames", "output/vggt_omega_config.yaml", "output/intrinsic.txt",
                "output/maplong_run/intrinsic.txt", "output/camera_frames.txt",
                # the Stray K F0 may take, in every place precision.camera reads it (point 77:
                # inputs/stray/ first, then the scan's legacy places)
                "inputs/stray/camera_matrix.csv", "inputs/stray/odometry.csv",
                "camera_matrix.csv", "odometry.csv", "stray/camera_matrix.csv", "stray/odometry.csv"),
         writes=("output/camera.json",), config=("camera",)),
    Step("f2_gauge", "F2 continuous gauge", "da3", "precision.gauge",
         reads=("intake/walk.json", "output/da3_windows/windows.json", "output/.metric_scale_applied",
                "output/camera_frames.txt", "output/camera_poses.txt", "output/maplong_run/metric_lock.json"),
         writes=("output/gauge.json",) + _POSE_FILES,
         config=("gauge", "raw:correction_graph.graph")),
    Step("f4_tracks", "F4 native-pixel tracks", "mapanything", "precision.tracks", gpu=True,
         reads=("frames", "output/camera.json"),
         writes=("output/precision/tracks.npz", "output/precision/tracks.json"), config=("tracks",)),
    Step("f3_probe", "F3 Omega resolution probe (report)", "mapanything",
         "precision.omega_probe", gpu=True,
         reads=("frames",), writes=("output/omega_probe.json",),
         config=("omega", "raw:reconstruction")),
    Step("f5_refine", "F5 joint refinement", "mapanything", "precision.refine",
         reads=_GEOMETRY_IN + ("output/gauge.json", "output/precision/tracks.npz"),
         writes=("output/precision/refine.json", "output/precision/witness_poses.txt",
                 "output/precision/witness_frames.txt", "output/precision/refine_residuals.npz",
                 "output/camera.json") + _POSE_FILES,
         config=("refine", "camera")),
    Step("f6_bend", "F6 depth on F5: Omega bent to F5's landmarks + multi-view vote (the product)", "da3",
         "precision.depth_on_f5",
         reads=_GEOMETRY_IN + ("frames", "output/precision/tracks.npz", "output/precision/refine.json",
                               "output/gauge.json", "output/depth_correction.json", "output/seg_masks.npz",
                               "output/precision/witness_poses.txt", "output/precision/witness_frames.txt"),
         writes=("output/cleaned_cloud.ply", "output/intrinsic.txt", "output/precision/depth_on_f5.json"),
         config=("bend", "depth", "cloud", "mono_detail", "gauge", "refine", "camera",
                 "raw:reconstruction.simple", "raw:postprocessing")),
    Step("f6_sweep", "F6 native depth sweep", "da3", "precision.depth_sweep", gpu=True,
         reads=_GEOMETRY_IN + ("frames", "output/precision/tracks.npz", "output/precision/refine.json"),
         writes=("output/precision/depth_sweep.json",), config=("depth", "camera", "refine")),
    Step("f6_colmap", "F6 COLMAP reference (A/B)", "da3", "precision.depth_colmap", gpu=True,
         reads=_GEOMETRY_IN + ("frames",), writes=("output/precision/depth_colmap.json",),
         config=("depth", "camera")),
    Step("f7_cloud", "F7-C corrected cloud on F6's depth (the product)", "da3",
         "precision.corrected_cloud", gpu=True,
         reads=_GEOMETRY_IN + ("frames", "output/precision/depth_sweep.json"),
         writes=("output/cleaned_cloud.ply", "output/intrinsic.txt", "output/precision/corrected_cloud.json"),
         config=("cloud", "depth", "camera", "raw:postprocessing")),
    Step("f7_fuse", "F7 witness fusion", "da3", "precision.fuse",
         reads=_GEOMETRY_IN + ("frames", "output/precision/depth_sweep.json"),
         writes=("output/cleaned_cloud.ply", "output/precision/fuse_report.json"),
         config=("fuse", "cloud", "camera")),
    Step("f6_check", "chunk / keyframe floor check (report)", "da3", "precision.chunk_check",
         reads=_GEOMETRY_IN + ("output/cleaned_cloud.ply", "output/precision/depth_on_f5.json",
                               "intake/walk.json", "intake/covis.json", "output/chunk_plan.json",
                               "output/da3_run/results_output"),
         writes=("output/precision/chunk_check.json",), config=("chunk_check", "cloud_metrics")),
]


# ── the step environment (points 38, 58) ────────────────────────────────────────────────────

def step_env(threads: int, base: Optional[Mapping[str, str]] = None) -> dict:
    """The environment of a step subprocess: the deterministic variables every launcher shares
    (``repro.deterministic_env``: cuBLAS workspace, hash seed), ``threads`` CPU threads for OpenMP
    / MKL / OpenBLAS, MKL's reproducible mode and the OpenBLAS kernel pin (point 38)."""
    env = repro.deterministic_env(base)
    env.update({"OMP_NUM_THREADS": str(int(threads)), "MKL_NUM_THREADS": str(int(threads)),
                "OPENBLAS_NUM_THREADS": str(int(threads)), "MKL_CBWR": "COMPATIBLE",
                "OPENBLAS_CORETYPE": OPENBLAS_CORETYPE})
    return env


def step_threads(step: Step, rcfg) -> int:
    """The CPU threads of ``step``: ``runner.threads`` for every step, F5 included (point 58 —
    verified bit-identical at that count, see the note at the top)."""
    return int(rcfg.threads)


_BLAS_PROBE = r"""
import json, numpy
from threadpoolctl import threadpool_info
print(json.dumps(sorted(({k: d.get(k) for k in ("user_api", "internal_api", "version",
      "architecture", "threading_layer", "num_threads")} | {"library": __import__("os").path.basename(
      str(d.get("filepath", "")))} for d in threadpool_info()), key=lambda r: (str(r["user_api"]),
      str(r["library"])))))
"""


def blas_probe(python: str, env: Mapping[str, str], threads: int) -> Dict[str, Any]:
    """What the BLAS of the step's own interpreter dispatches to under ``env`` (threadpoolctl in
    that interpreter), VERIFIED against the pin: every OpenBLAS it loads must report
    :data:`OPENBLAS_ARCHITECTURE_REPORTED` and ``threads`` threads — else the chain FAILS (point
    38: the kernel set decides the last bits; a pin OpenBLAS ignored would hide that). Returns the
    record kept in the step record: the libraries, the pin, the CPU model."""
    try:
        r = subprocess.run([python, "-c", _BLAS_PROBE], capture_output=True, text=True,
                           env=dict(env), timeout=repro.PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise ChainError(f"the BLAS probe of {python} did not answer in {repro.PROBE_TIMEOUT_S} s") from e
    if r.returncode != 0:
        raise ChainError(f"the BLAS probe of {python} failed (exit {r.returncode}): "
                         f"{(r.stderr or r.stdout).strip()[-600:]}")
    try:
        libs = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise ChainError(f"unreadable BLAS probe answer from {python}: {r.stdout!r}") from e
    blas = [d for d in libs if d.get("user_api") == "blas"]
    if not blas:
        raise ChainError(f"{python}: numpy loads no BLAS threadpoolctl can see — the kernel pin "
                         f"cannot be verified")
    bad = [d for d in blas if d.get("internal_api") == "openblas"
           and str(d.get("architecture")).lower() != OPENBLAS_ARCHITECTURE_REPORTED.lower()]
    if bad:
        raise ChainError(f"{python}: OpenBLAS ignored OPENBLAS_CORETYPE={OPENBLAS_CORETYPE} — it "
                         f"dispatched to {[d.get('architecture') for d in bad]} (point 38: the "
                         f"kernel set is pinned, not chosen by the CPU)")
    off = [d for d in blas if int(d.get("num_threads") or 0) != int(threads)]
    if off:
        raise ChainError(f"{python}: a BLAS runs {[d.get('num_threads') for d in off]} thread(s) "
                         f"under OPENBLAS_NUM_THREADS={threads}")
    return {"openblas_coretype": OPENBLAS_CORETYPE, "threads": int(threads), "libraries": libs,
            "cpu_model": repro.cpu_model()}


# ── stamps (point 31): the code closure lives in precision/code_closure.py so that a step
# can stamp itself without importing the runner ──
from precision.code_closure import (  # noqa: E402
    CODE_FOLLOWED_DIRS, CODE_FOLLOWED_FILES, CodeClosureError, FORK_DIR, _CODE_ROOTS,
    _followed, _imports_of, _module_file, code_closure)
def _section(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            raise ChainError(f"config section {dotted!r} is missing from config.yaml — a step's "
                             f"stamp names every section it reads")
    return cur


def _plain(v: Any) -> Any:
    """Dataclasses / tuples → JSON-able (canonical_json handles dataclasses, but a tuple inside
    a dict and a list must stamp alike: tuples become lists)."""
    if is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def step_config(step: Step, pcfg, raw_cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The config sections ``step`` reads, by name: PrecisionConfig fields (a wrong name FAILS)
    and ``raw:`` sections of config.yaml (the section ``reconstruction`` without its ``precision``
    block, which the precision fields already cover)."""
    out: Dict[str, Any] = {}
    for name in step.config:
        if name.startswith("raw:"):
            if raw_cfg is None:
                from config import cfg as raw_cfg  # noqa: F811  (server/config.py, config.yaml)
            sec = _section(raw_cfg, name[4:])
            if name[4:] == "reconstruction" and isinstance(sec, Mapping):
                sec = {k: v for k, v in sec.items() if k != "precision"}
            out[name[4:]] = _plain(sec)
        else:
            if not hasattr(pcfg, name):
                raise ChainError(f"step {step.key}: {name!r} is not a section of "
                                 f"reconstruction.precision")
            out[f"reconstruction.precision.{name}"] = _plain(getattr(pcfg, name))
    return out


def _dynamic_reads(step: Step, session_dir: Path, pcfg, rid: Optional[str]) -> Dict[str, Path]:
    """Inputs only the step's own module can name (the evidence files it TAKES, the instruments
    configured): keyed like the static ones (session-relative)."""
    if step.key == "f2_gauge":
        from precision.gauge import chain_inputs
        return dict(chain_inputs(session_dir, pcfg.gauge, rid))
    if step.key == "f4_tracks":
        from precision.tracks import chain_inputs
        return dict(chain_inputs(session_dir, rid))
    return {}


def step_inputs(step: Step, session_dir: Path, pcfg, rid: Optional[str] = None) -> Dict[str, Path]:
    sd = Path(session_dir)
    ins = {k: sd / k for k in step.reads}
    ins.update(_dynamic_reads(step, sd, pcfg, rid))
    return ins


def step_products(step: Step, session_dir: Path) -> Dict[str, Path]:
    sd = Path(session_dir)
    return {k: sd / k for k in step.writes}


def _files_stamp(paths: Mapping[str, Path], code: Sequence[Path] = (),
                 config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``repro.stamp`` over inputs that may be ABSENT (presence is part of the identity): the
    present ones by content, the missing ones as :data:`ABSENT`; code and config through the
    shared helper. Same shape as a repro stamp, so ``repro.check_stamp`` reads it."""
    base = repro.stamp(inputs={}, code=list(code), config=dict(config or {}))
    ins = {str(k): (repro.sha256_path(p) if Path(p).exists() else ABSENT)
           for k, p in sorted(paths.items())}
    body = {"stamp_version": base["stamp_version"], "inputs": ins, "code": base["code"],
            "config": base["config"]}
    body["sha256"] = repro.sha256_json(body)
    return body


def _with_inputs(stamp: Mapping[str, Any], inputs: Mapping[str, str]) -> Dict[str, Any]:
    body = {"stamp_version": stamp.get("stamp_version"), "inputs": dict(sorted(inputs.items())),
            "code": dict(stamp.get("code") or {}), "config": dict(stamp.get("config") or {})}
    body["sha256"] = repro.sha256_json(body)
    return body


def stamp_differences(step: Step, saved: Any, now: Mapping[str, Any],
                      written: Mapping[str, str]) -> List[str]:
    """What differs between the stamp a finished ``step`` SAVED and the one computed NOW: its
    inputs (a file a previous step of the chain wrote is compared as that step left it —
    ``written`` — and a file the step itself rewrites in place is not compared here: its
    before-state is gone, its after-state is checked as a product), its code, its config."""
    if not isinstance(saved, Mapping) or "sha256" not in saved:
        return ["no stamp saved with the step (a record of an older runner)"]
    own = set(step.writes)
    saved_in = {k: v for k, v in (saved.get("inputs") or {}).items() if k not in own}
    now_in = {k: written.get(k, v) for k, v in (now.get("inputs") or {}).items() if k not in own}
    return repro.check_stamp(_with_inputs(saved, saved_in), _with_inputs(now, now_in))


# ── state ───────────────────────────────────────────────────────────────────────────────────

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


def _record_timing(session_dir: Path, key: str, seconds: float, chain_seconds: Optional[float] = None) -> None:
    """Durations live apart from the compared state file (point 36)."""
    p = Path(session_dir) / "output" / "precision" / TIMING_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        d = {}
    d.setdefault("version", 1)
    d.setdefault("steps", {})
    if key:
        d["steps"][key] = round(float(seconds), 1)
    if chain_seconds is not None:
        d["last_chain_seconds"] = round(float(chain_seconds), 1)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, p)


def _only_new_cloud_epochs(out: Optional[Path], e_from: int, e_to: int) -> bool:
    """True when every epoch after ``e_from`` up to ``e_to`` is a NEW-CLOUD epoch: a published
    cloud moves no camera, so the poses the chain left behind still hold (load_inputs' own rule)."""
    if out is None or e_to <= e_from:
        return False
    try:
        from correction.epoch import epoch_kind
        return all(epoch_kind(out, e) == "new_cloud" for e in range(e_from + 1, e_to + 1))
    except Exception:  # noqa: BLE001
        return False


def resume_point(state: dict, steps: Sequence[Step], epoch_now: int, out: Optional[Path] = None, *,
                 stamp_now: Optional[Callable[[Step], Mapping[str, Any]]] = None,
                 product_sha: Optional[Callable[[str], str]] = None,
                 log: Callable = lambda *a: None) -> int:
    """Index of the first step to run.

    With ``stamp_now`` (the stamp of a step's inputs as they are on disk now) and ``product_sha``
    (a product's sha256 now, or ABSENT): the finished prefix counts step by step while the saved
    input stamp equals the one computed now (``stamp_differences``: inputs, code, config), and
    stops at the first that differs (point 31). Then every product of the prefix must still be
    what its LAST writer left: a product that is what an EARLIER step of the chain left (an epoch
    undone back to that geometry) re-runs the chain from the step after that writer; a product
    changed by something outside the chain FAILS naming it — unless the only epochs since are
    published clouds, which move no camera (the depth stage re-runs over them).

    Without ``stamp_now`` (callers that have no session): the pre-stamp rule — the prefix counts
    only while the session is in the epoch its last step left it in, or in a later epoch that
    only published clouds."""
    done = state.get("done") or []
    keys = [s.key for s in steps]
    if stamp_now is None:
        n = 0
        for rec in done:
            if n < len(keys) and rec.get("key") == keys[n]:
                n += 1
            else:
                break
        if n == 0:
            return 0
        last = done[n - 1]
        if int(last.get("epoch_after", -1)) != int(epoch_now) and not _only_new_cloud_epochs(
                out, int(last.get("epoch_after", -1)), int(epoch_now)):
            raise ChainError(f"the precision chain stopped after '{last['key']}' with the session in "
                             f"epoch {last.get('epoch_after')}, and it is now in epoch {epoch_now} — "
                             f"the remaining steps would run on geometry they did not produce; "
                             f"select epoch {last.get('epoch_after')} or reconstruct again")
        return n
    if product_sha is None:
        raise ChainError("resume_point: stamp_now needs product_sha")
    n = 0
    written: Dict[str, str] = {}                   # product → sha as its last writer left it
    writers: Dict[str, List[Tuple[int, str]]] = {}  # product → [(step index, sha)] in order
    for rec in done:
        if n >= len(keys) or rec.get("key") != keys[n]:
            break
        s = steps[n]
        diffs = stamp_differences(s, rec.get("stamp_in"), stamp_now(s), written)
        so = rec.get("stamp_out")
        if not diffs and not (isinstance(so, Mapping) and isinstance(so.get("inputs"), Mapping)):
            diffs = ["no product stamp saved with the step"]
        if diffs:
            log(f"{LOG_TAG} '{s.key}' runs again — what it consumed changed: "
                + "; ".join(diffs[:6]) + (" …" if len(diffs) > 6 else ""))
            break
        for f, sha in so["inputs"].items():
            writers.setdefault(f, []).append((n, sha))
            written[f] = sha
        n += 1
    if n == 0:
        return 0
    cut, moved = n, []
    for f, sha_last in sorted(written.items()):
        now = product_sha(f)
        if now == sha_last:
            continue
        earlier = [i for i, sha in writers[f] if sha == now]
        if earlier:
            j = max(earlier)
            nxt = min(i for i, _ in writers[f] if i > j)
            log(f"{LOG_TAG} {f} is what '{steps[j].key}' left (a later epoch was undone): "
                f"'{steps[nxt].key}' and the steps after it run again")
            cut = min(cut, nxt)
        else:
            moved.append(f)
    if moved:
        last = done[n - 1]
        e_last = int(last.get("epoch_after", -1))
        if not _only_new_cloud_epochs(out, e_last, int(epoch_now)):
            raise ChainError(f"the chain's product(s) {moved} are not what the steps left them "
                             f"(the chain stopped after '{last['key']}' in epoch {e_last}; the session "
                             f"is now in epoch {epoch_now}) — the remaining steps would run on geometry "
                             f"they did not produce; select epoch {e_last} or reconstruct again")
        log(f"{LOG_TAG} {moved} changed since '{last['key']}' by published-cloud epochs only "
            f"(no camera moved) — the finished prefix still counts")
    return cut


def chain_steps(pcfg, steps: Sequence[Step] = STEPS) -> List[Step]:
    """The steps this configuration runs: the product decides the chain. Both run the
    F6 sweep after F5; ``cloud.source: omega_corrected`` then builds the corrected
    cloud on F6's depth (f7_cloud — never the COLMAP reference nor the fusion),
    ``fusion`` the COLMAP reference (only when precision.depth.colmap.enabled) and the
    witness fusion. The check runs last."""
    fusion = pcfg.cloud.source == "fusion"
    drop = set()
    if pcfg.cloud.source == "omega_bent":
        # USER 2026-10-01 (pccr epoch 7): the depth on F5 IS the product — no plane sweep, no F7.
        # With mono_detail on, PointDiT runs inside it: the step takes the card (vLLM stands down).
        from dataclasses import replace as _replace
        mono = bool(getattr(getattr(pcfg, "mono_detail", None), "enabled", False))
        return [(_replace(s, gpu=True) if (mono and s.key == "f6_bend") else s)
                for s in steps if s.key not in {"f6_sweep", "f6_colmap", "f7_cloud", "f7_fuse"}]
    drop.add("f6_bend")
    if fusion:
        drop.add("f7_cloud")
        if not pcfg.depth.colmap.enabled:
            drop.add("f6_colmap")
    else:
        drop.update({"f6_colmap", "f7_fuse"})
    return [s for s in steps if s.key not in drop]


def step_needs_gpu(step: Step, session_dir: Path) -> bool:
    """``step.gpu``, or F2 when it has to regenerate the deleted DA3 windows (point 26: the
    extractor then takes the card — the chat service stands down and the card is checked free
    like for any GPU step)."""
    if step.gpu:
        return True
    if step.key == "f2_gauge":
        from precision.gauge import needs_window_regeneration
        return bool(needs_window_regeneration(session_dir))
    return False


def _reconstruction_id(out: Path) -> Optional[str]:
    from correction.epoch import reconstruction_id_or_none
    return reconstruction_id_or_none(out)


def run_chain(session_dir: Path, pcfg, *, log: Callable = print,
              progress: Optional[Callable[[float, str], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None,
              before_gpu: Optional[Callable[[str], None]] = None,
              steps: Optional[Sequence[Step]] = None, from_step: Optional[str] = None) -> dict:
    """``from_step``: forget the steps from that one on (its record and the later ones) and run
    them again — a changed depth stage over the same F5 (USER 2026-10-04: "relanzamos desde donde
    requiera esta modificación")."""
    steps = chain_steps(pcfg) if steps is None else list(steps)
    session_dir = Path(session_dir).resolve()
    out = session_dir / "output"
    rcfg = pcfg.runner
    py = {"da3": rcfg.python_da3, "mapanything": rcfg.python_mapanything}
    for k, v in py.items():
        if not Path(v).exists():
            raise ChainError(f"precision.runner.python_{k} = {v} does not exist")
    state = load_state(session_dir)
    if from_step is not None:
        keys = [s.key for s in steps]
        if from_step not in keys:
            raise ChainError(f"--from {from_step!r}: not a step of this chain {keys}")
        state["done"] = [r for r in (state.get("done") or [])[:keys.index(from_step)]]
        _save_state(session_dir, state)
        log(f"{LOG_TAG} re-running from '{from_step}' (the later records forgotten)")
    # the reconstruction's identity enters every step's stamp (its Omega records are its inputs,
    # hashed ONCE here instead of once per step)
    rid = _reconstruction_id(out)
    code_of: Dict[str, Tuple[List[Path], Dict[str, Path]]] = {}

    def _code(s: Step) -> Tuple[List[Path], Dict[str, Path]]:
        if s.key not in code_of:
            code_of[s.key] = code_closure(s.module)
        return code_of[s.key]

    def _config(s: Step) -> Dict[str, Any]:
        cfg = step_config(s, pcfg)
        cfg["reconstruction"] = {"id": rid}
        cfg["step_env"] = {"threads": step_threads(s, rcfg), "OPENBLAS_CORETYPE": OPENBLAS_CORETYPE,
                           "MKL_CBWR": "COMPATIBLE", "env": s.env, "args": list(s.args)}
        return cfg

    def _stamp_in(s: Step) -> Dict[str, Any]:
        files, external = _code(s)
        return _files_stamp(step_inputs(s, session_dir, pcfg, rid) | external, files, _config(s))

    def _product_sha(key: str) -> str:
        p = session_dir / key
        return repro.sha256_path(p) if p.exists() else ABSENT

    epoch_now = _epoch(session_dir)
    first = resume_point(state, steps, epoch_now, out=out, stamp_now=_stamp_in,
                         product_sha=_product_sha, log=log)
    done = list(state.get("done") or [])
    state["done"] = done[:first]
    if first:
        last = done[first - 1]
        note = ""
        if int(last.get("epoch_after", -1)) != int(epoch_now):
            note = (f"; the session is in epoch {epoch_now} and the prefix left it in epoch "
                    f"{last.get('epoch_after')} — its geometry is byte-identical, the number is not "
                    f"the key")
        log(f"{LOG_TAG} resuming after '{steps[first - 1].key}' "
            f"({first}/{len(steps)} step(s) already done: inputs, code and config unchanged{note})")
    _save_state(session_dir, state)
    t_chain = time.time()
    for i in range(first, len(steps)):
        s = steps[i]
        if cancelled is not None and cancelled():
            raise ChainError("cancelled")
        if progress is not None:
            progress(100.0 * i / len(steps), f"{s.label} ({i + 1}/{len(steps)})")
        gpu = step_needs_gpu(s, session_dir)
        if gpu:
            if before_gpu is not None:
                before_gpu(s.label)
            try:
                # point 4: the whole card or nothing — never a smaller resolution / window to fit
                gpu_rec = repro.require_exclusive_gpu(log=log)
            except repro.ReproError as e:
                raise ChainError(f"{s.label}: {e}") from e
        else:
            gpu_rec = None
        threads = step_threads(s, rcfg)
        try:
            env = step_env(threads)
        except repro.ReproError as e:
            raise ChainError(f"{s.label}: {e}") from e
        blas = blas_probe(py[s.env], env, threads)      # point 38: the pin, verified where it acts
        epoch_before = _epoch(session_dir)
        stamp_in = _stamp_in(s)
        cmd = [py[s.env], "-u", "-m", s.module, "--session", str(session_dir), *s.args]
        log(f"{LOG_TAG} ── {s.label}: {' '.join(cmd[2:])}")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(SERVER_DIR), env=env, stdout=subprocess.PIPE,
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
        seconds = time.time() - t0
        if rc != 0:
            raise ChainError(f"{s.label} failed (exit {rc}) — see the lines above; the chain "
                             f"resumes from this step")
        rec = {"key": s.key, "epoch_before": epoch_before, "epoch_after": _epoch(session_dir),
               "gpu": bool(gpu), "stamp_in": stamp_in,
               "stamp_out": _files_stamp(step_products(s, session_dir)),
               "environment": {"blas": blas, "gpu": gpu_rec}}
        state["done"].append(rec)
        _save_state(session_dir, state)
        _record_timing(session_dir, s.key, seconds)
        log(f"{LOG_TAG} ✓ {s.label} in {seconds / 60:.1f} min (epoch "
            f"{rec['epoch_before']} → {rec['epoch_after']})")
        if s.key == "f2_gauge" and bool(pcfg.gauge.delete_windows_after_chain):
            # the I3 window depth (read by F0 and F2 only) goes the moment F2 is through,
            # not hours later at the end of the chain (USER 2026-10-05, the /workspace
            # quota at 92 % on pccr 2408: 18 GB of windows sat there through F4-F6).
            # windows.json, walk.json and the anchors stay; run_gauge regenerates the
            # files when a re-run from F0/F2 needs them.
            from intake.walk import delete_windows
            delete_windows(session_dir, log)
    if progress is not None:
        progress(100.0, "precision core done")
    if bool(pcfg.gauge.delete_windows_after_chain):
        # the I3 window depth (F0's only reader) is dead once the chain is through — walk.json,
        # the anchors and windows.json stay, run_gauge regenerates the files for a re-run from F0
        from intake.walk import delete_windows
        delete_windows(session_dir, log)
    chain_seconds = time.time() - t_chain
    _record_timing(session_dir, "", 0.0, chain_seconds=chain_seconds)
    return {"steps": state["done"], "seconds": round(chain_seconds, 1),
            "epoch": _epoch(session_dir)}


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.runner",
                                 description="The precision core F0 → F7 on a session.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--from", dest="from_step", default=None, help="re-run from this step (f6_bend …)")
    args = ap.parse_args(argv)
    # by hand the card is not shared either: vLLM (the chat) is stopped, verified,
    # before every GPU step — the same rule the pipeline's worker applies
    from workers.base import stop_semantic_service_verified

    def _before_gpu(label: str) -> None:
        stop_semantic_service_verified(None, stage=f"precision {label}", log=print)

    run_chain(Path(args.session), load_precision_config(), before_gpu=_before_gpu, from_step=args.from_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
