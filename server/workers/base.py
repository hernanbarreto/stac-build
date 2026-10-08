# STAC-Builder: Base Worker Protocol
# Defines the IPC message protocol and base helper used by all workers.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import signal
import sys
import time
import traceback
from multiprocessing.connection import Connection
from typing import Optional


class WorkerPipe:
    """Helper wrapper around a multiprocessing.Pipe connection.
    
    Provides typed send methods for the IPC protocol and
    a check_cancel() method workers should call periodically.
    """

    def __init__(self, conn: Connection):
        self._conn = conn
        self._cancelled = False

    # ── Send helpers (Worker → Server) ───────────────────────

    def send_progress(self, pct: float, msg: str, stage: str = ""):
        """Send a progress update.  pct is 0-100."""
        self._conn.send({
            "type": "progress",
            "stage": stage,
            "pct": round(pct, 1),
            "msg": msg,
        })

    def send_log(self, msg: str, level: str = "info"):
        self._conn.send({"type": "log", "level": level, "msg": msg})

    def send_done(self, success: bool, elapsed: float, detail: str = ""):
        self._conn.send({
            "type": "done",
            "success": success,
            "elapsed": round(elapsed, 2),
            "detail": detail,
        })

    def send_error(self, msg: str, tb: Optional[str] = None):
        self._conn.send({
            "type": "error",
            "msg": msg,
            "traceback": tb or "",
        })

    # ── Receive helpers (Server → Worker) ────────────────────

    def check_cancel(self) -> bool:
        """Non-blocking check if server sent a cancel signal."""
        if self._cancelled:
            return True
        while self._conn.poll():
            try:
                msg = self._conn.recv()
                if isinstance(msg, dict) and msg.get("type") == "cancel":
                    self._cancelled = True
                    return True
            except (EOFError, OSError):
                self._cancelled = True
                return True
        return False

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def die_with_parent(poll_s: float = 2.0) -> None:
    """USER 2026-10-06: "cuando mato el proceso debe matar todo lo que tenía corriendo,
    omega, da3, vlm, sam3, pointdit, todo". A daemon thread watches the parent; the
    moment it is gone (the server restarted, killed, crashed) the WHOLE process group
    of this stage dies with SIGKILL — the worker, the inline child, Omega's
    vggt_long.py, DA3, SAM3, the precision steps, PointDiT: everything launched under
    it. PR_SET_PDEATHSIG alone kills only the direct child, and the stage tree used to
    survive a backend restart (pccr 2408: an orphan Omega kept 58 GB of RAM and the
    CPU for an hour; zaragoza: an orphan SAM3 held 21 GB of VRAM for four hours).
    vLLM runs in its own session (a GPU handover must not signal the backend's group)
    and dies with the backend through semantic/watchdog.py and the server shutdown."""
    import os
    import signal
    import threading
    ppid = os.getppid()

    def _watch():
        while True:
            time.sleep(poll_s)
            if os.getppid() != ppid:
                try:
                    os.killpg(os.getpgid(0), signal.SIGKILL)
                except OSError:
                    os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=_watch, name="die-with-parent", daemon=True).start()


def run_worker_safe(worker_fn, conn: Connection, *args, **kwargs):
    """Execute a worker function with standard error handling and cleanup.

    This is the top-level entry point called as a multiprocessing target.
    It wraps the actual worker logic in try/except, sends done/error
    messages, and ensures the pipe is always closed.
    """
    # Become our own process-group leader so a pipeline cancel can kill the
    # WHOLE stage subtree (bash scripts, DA3/CloudCompy children) with one
    # killpg — a plain terminate() on the worker orphaned its children, which
    # kept running and holding GPU memory. Deliberately-persistent services
    # (vLLM) detach with start_new_session and are unaffected.
    # A worker HOSTED by another worker (workers/inline_child.py, env
    # STAC_INLINE_CHILD) stays in its host's group, so that same killpg reaches it.
    try:
        import os
        if not os.environ.get("STAC_INLINE_CHILD"):
            os.setsid()
    except Exception:
        pass
    die_with_parent()          # the stage tree never outlives the server (USER 2026-10-06)
    pipe = WorkerPipe(conn)
    t0 = time.time()
    try:
        worker_fn(pipe, *args, **kwargs)
        pipe.send_done(success=True, elapsed=time.time() - t0)
    except Exception as exc:
        pipe.send_error(str(exc), traceback.format_exc())
        pipe.send_done(success=False, elapsed=time.time() - t0, detail=str(exc))
    finally:
        pipe.close()
        # Release this process's cached VRAM — only when it initialised CUDA (it ran GPU work).
        # A worker that never touched the card must not create a CUDA context on its way out
        # (torch.cuda.synchronize creates one) just to synchronise nothing: for those seconds
        # it is a compute process on the card the next step checks free
        # (repro.require_exclusive_gpu). torch is read from sys.modules: never imported here =
        # never initialised here.
        try:
            _torch = sys.modules.get("torch")
            if _torch is not None and _torch.cuda.is_initialized():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
        except RuntimeError:
            pass  # a CUDA error on the way out changes nothing of what the worker reported
        import gc
        gc.collect()


# ── Exclusive-GPU helpers (shared by reconstruction / SAM3 workers) ──────────

def gpu_total_gb() -> Optional[float]:
    """TOTAL VRAM (GB) of GPU 0 via nvidia-smi — a property of the card, unlike the
    free memory; None when it can't be read."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def gpu_free_gb() -> Optional[float]:
    """Free VRAM (GB) of GPU 0 via nvidia-smi; None when it can't be read."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


# The stop's schedule (the numbers of 2026-10-05, unchanged, named so the VRAM wait below reuses
# them): up to 6 kill attempts — SIGTERM for the first 3, then SIGKILL — each followed by up to
# 10 looks at the process table 2 s apart.
_STOP_ATTEMPTS = 6
_STOP_SIGTERM_ATTEMPTS = 3
_STOP_POLLS = 10
_STOP_POLL_S = 2.0
# The wait for the card to let go of the stopped processes gets the budget the processes had to
# die (6 attempts x 10 looks x 2 s = 120 s). A BOUND, not a decision: past it nothing is assumed
# — the GPU step's own repro.require_exclusive_gpu reads the card and names what still holds it.
_RELEASE_POLLS = _STOP_ATTEMPTS * _STOP_POLLS


def stop_semantic_service(pipe: Optional["WorkerPipe"] = None, stage: str = "",
                          log=None) -> Optional[dict]:
    """EXCLUSIVE GPU for a heavy stage: stop the vLLM semantic service (its ~40 GB
    resident VRAM starves Omega single passes and long SAM3 sessions). Any later
    consumer restarts it (semantic.service.ensure_service — the VLM worker AND the
    spatial-Q&A route), so this is a stage-scoped handover, not a shutdown. No-op
    when vLLM isn't running.

    Once the processes are gone it WAITS for the card to let go of them
    (:func:`await_vram_release`) — the next GPU step's repro.require_exclusive_gpu reads the
    card right after, and a context still being torn down reads as a busy card.

    ``log`` is for callers that are not workers and have no pipe — the epoch
    transaction's re-consolidation is one (correction/apply.py).

    Returns None when no vLLM was running (or the stop itself failed — declared in the log),
    otherwise the release record of :func:`await_vram_release` (None there too when the
    processes outlived every attempt: the verified stopper names them).
    """
    _say = (pipe.send_log if pipe is not None else (log or (lambda m, **k: None)))
    try:
        if not vllm_pids():
            return None
        _say(f"[gpu] stopping vLLM semantic service — {stage or 'this stage'} "
             f"gets the whole GPU (it auto-restarts on next VLM use)")
        # what the card holds and for whom, read BEFORE the first signal (which pids are vLLM's
        # is only known from the process table, and only while they are alive)
        before = _gpu_snapshot()
        # the launcher chain (serve_semantic.sh → semantic.serve → vllm serve) is killed as a
        # whole and the kill is REPEATED until nothing is left: a vLLM still starting turns into
        # a new 'vllm serve' process right after the first signal (zaragoza 2026-10-05, launched
        # one second after the backend came up)
        signalled = set()
        for attempt in range(_STOP_ATTEMPTS):
            signalled.update(kill_vllm_pids(
                signal.SIGTERM if attempt < _STOP_SIGTERM_ATTEMPTS else signal.SIGKILL))
            for _ in range(_STOP_POLLS):
                time.sleep(_STOP_POLL_S)
                if not vllm_pids():
                    break
            if not vllm_pids():
                break
        rec = None
        if not vllm_pids():
            rec = await_vram_release(sorted(signalled), before, _say)
        free = gpu_free_gb()
        if free is not None:
            _say(f"[gpu] vLLM stopped — {free:.0f} GB VRAM free")
        return rec
    except Exception as e:  # noqa: BLE001
        _say(f"[gpu] could not stop vLLM ({e}) — continuing with shared GPU")
        return None


def _gpu_snapshot() -> dict:
    """What nvidia-smi says now, through repro's readers: ``{"used_mib": {card uuid: MiB},
    "procs": [{"pid", "name", "used_mib", "gpu_uuid"}], "here": [listed pids that are
    processes of this pid namespace at this moment]}``, or ``{"error": reason}`` when
    nvidia-smi cannot answer (a box without it, a driver that does not respond)."""
    import os
    import repro
    try:
        used = {c["uuid"]: int(c["used_mib"]) for c in repro.gpu_cards()}
        procs = repro.gpu_compute_processes()
    except repro.ReproError as e:
        return {"error": str(e)}
    return {"used_mib": used, "procs": procs,
            "here": sorted({p["pid"] for p in procs if os.path.exists(f"/proc/{p['pid']}")})}


def await_vram_release(pids, before: dict, say=None, *, polls: int = _RELEASE_POLLS,
                       poll_s: float = _STOP_POLL_S) -> dict:
    """WAIT until the card let go of the stopped vLLM processes ``pids``: nvidia-smi lists none
    of them as a compute process AND every card's used memory has dropped by what nvidia-smi
    attributed to them in ``before`` (a :func:`_gpu_snapshot` taken before the first signal).

    The process table alone does not say it: a process in exit has already lost its command
    line (``pgrep -f`` no longer matches it, so the stop sees nothing left) while the driver is
    still tearing its CUDA context down; the next step's repro.require_exclusive_gpu, run right
    after, would read that card as busy. Checked at once, then every ``poll_s`` up to ``polls``
    times (a BOUND — past it the step's own exclusive-GPU check decides).

    Declared limit: when nvidia-smi listed compute processes before the stop but none of the
    stopped pids, and some listed pid is not a process of this pid namespace (a container whose
    driver reports host pids), the release cannot be attributed and is not awaited.

    Returns ``{"awaited", "released" (True | False | None = could not tell), "pids",
    "held_mib" ({uuid: MiB} attributed to them before the stop), "still_listed" (stopped pids
    nvidia-smi lists at the end), "used_mib_after", "polls", "why"}``."""
    say = say or (lambda m, **k: None)
    pids = sorted({int(p) for p in pids})
    rec = {"awaited": False, "released": None, "pids": pids, "held_mib": {},
           "still_listed": [], "used_mib_after": None, "polls": 0, "why": ""}
    if before.get("error"):
        rec["why"] = f"nvidia-smi could not be read before the stop ({before['error']})"
        say(f"[gpu] ⚠ the release of vLLM's VRAM is not awaited: {rec['why']} — the step's own "
            f"exclusive-GPU check decides")
        return rec
    mine = [p for p in before["procs"] if p["pid"] in pids]
    here = set(before.get("here") or ())
    other_ns = sorted(p["pid"] for p in before["procs"] if p["pid"] not in here)
    if not mine and other_ns:
        rec["why"] = (f"nvidia-smi listed compute process(es) {sorted(p['pid'] for p in before['procs'])} "
                      f"before the stop, none of them a stopped vLLM pid {pids}, and {other_ns} "
                      f"is no process of this pid namespace — the release cannot be attributed")
        say(f"[gpu] ⚠ the release of vLLM's VRAM is not awaited: {rec['why']}; the step's own "
            f"exclusive-GPU check decides")
        return rec
    held: dict = {}
    for p in mine:
        if p["used_mib"] is not None:
            held[p["gpu_uuid"]] = held.get(p["gpu_uuid"], 0) + int(p["used_mib"])
    rec.update(awaited=True, held_mib=held)
    listed: list = []
    pending: dict = {}
    for i in range(int(polls) + 1):
        snap = _gpu_snapshot()
        if snap.get("error"):
            rec.update(released=None, polls=i, why=f"nvidia-smi could not be read ({snap['error']})")
            say(f"[gpu] ⚠ the release of vLLM's VRAM could not be checked: {rec['why']} — the "
                f"step's own exclusive-GPU check decides")
            return rec
        listed = sorted({p["pid"] for p in snap["procs"] if p["pid"] in pids})
        pending = {u: snap["used_mib"].get(u) for u, h in held.items()
                   if snap["used_mib"].get(u, 0) > before["used_mib"].get(u, 0) - h}
        if not listed and not pending:
            rec.update(released=True, polls=i, used_mib_after=snap["used_mib"])
            say(f"[gpu] vLLM's VRAM released: nvidia-smi lists none of the stopped process(es) "
                f"{pids}" + (f", {sum(held.values())} MiB they held are back" if held else "")
                + (f" (after {i * poll_s:.0f} s)" if i else ""))
            return rec
        if i < int(polls):
            time.sleep(poll_s)
    rec.update(released=False, polls=int(polls), still_listed=listed, used_mib_after=snap["used_mib"],
               why=(f"after {int(polls) * poll_s:.0f} s nvidia-smi still lists the stopped "
                    f"process(es) {listed}" if listed else
                    f"after {int(polls) * poll_s:.0f} s the card still uses {pending} MiB, more than "
                    f"before the stop less the {held} MiB vLLM held"))
    say(f"[gpu] ⚠ vLLM's VRAM not released: {rec['why']}")
    return rec


VLLM_PROCESS_PATTERN = "vllm serve"      # what stop_semantic_service kills and pgrep looks for
# the whole launcher chain, each ANCHORED to the start of the command line so a shell whose
# command text merely mentions these words (a terminal, a tool, a test) is never matched:
#   bash [/…/]scripts/serve_semantic.sh  →  python -m semantic.serve  →  /…/envs/semantic/bin/
#   python3.11 /…/bin/vllm serve …  →  its engine (proctitle VLLM::EngineCore)
# (one pid through the first three: each exec()s the next). The launcher's path is relative when
# init_pod.sh's tmux session starts it (`bash scripts/serve_semantic.sh`) and absolute when the
# backend does (semantic/service.py); bash's `exec python -m semantic.serve` keeps argv[0] as
# typed — `python`, no directory. Missing either form, a stop that looked during those first
# seconds found nothing and the launcher became a 'vllm serve' right after (2026-10-07 07:24:
# two chains alive at once — pid 1349 launched from a tmux shell, pid 2387 by the backend's
# boot kick).
VLLM_PATTERNS = (r"^\S+/python[0-9.]* \S+/bin/vllm serve",
                 r"^(\S*/)?python[0-9.]* -m semantic\.serve",
                 r"^(\S*/)?bash (\S*/)?scripts/serve_semantic\.sh",
                 r"^VLLM::")


def kill_vllm_pids(sig) -> list:
    """Signal every vLLM process by PID (the `vllm serve` command lines pgrep finds), never
    the backend or any shell that merely mentions the pattern. Returns the PIDs signalled."""
    import os
    pids = [p for p in vllm_pids() if p != os.getpid()]
    for p in pids:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass
    return pids


def vllm_pids() -> list:
    """PIDs of every process whose command line matches ``vllm serve`` (pgrep -f),
    [] when none. RuntimeError when pgrep itself cannot run — a check that could
    not look is not a check that found nothing."""
    import os
    import subprocess
    pids = set()
    for pat in VLLM_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        except OSError as e:
            raise RuntimeError(f"cannot verify the GPU handover: pgrep failed to run ({e})") from e
        if out.returncode not in (0, 1):
            raise RuntimeError(f"cannot verify the GPU handover: pgrep -f '{pat}' "
                               f"exited {out.returncode} ({out.stderr.strip()})")
        pids.update(int(p) for p in out.stdout.split() if p.strip().isdigit())
    pids.discard(os.getpid())
    return sorted(pids)


def stop_semantic_service_verified(pipe: Optional["WorkerPipe"] = None, stage: str = "",
                                   log=None) -> dict:
    """:func:`stop_semantic_service`, then VERIFY no ``vllm serve`` process is left
    (stop_semantic_service swallows its own failures and, after its wait, logs
    "vLLM stopped" without looking again). Raises RuntimeError naming the PIDs
    still alive — a stage that needs the whole GPU must not start on a shared one —
    and naming the stopped PIDs nvidia-smi still lists on the card once the stop's
    VRAM wait (:func:`await_vram_release`) ran out.
    Returns the check for the caller's report: ``{"service_stopped": True,
    "check": "pgrep -f 'vllm serve'", "remaining_pids": [], "free_gb": float|None}``."""
    # the backend kicks vLLM at its own boot through an HTTP probe that takes seconds: a pipeline
    # launched right after a restart stops nothing, then the launcher appears (zaragoza 2026-10-05,
    # twice) — so stop + verify is REPEATED until a verification finds nothing
    left = []
    release = None                       # the record of the LAST stop that killed something
    for _ in range(_STOP_ATTEMPTS):
        r = stop_semantic_service(pipe, stage=stage, log=log)
        if isinstance(r, dict):
            release = r
        left = vllm_pids()
        if not left:
            break
        time.sleep(_STOP_POLL_S)
    if left:
        raise RuntimeError(
            f"{stage or 'this stage'} needs the GPU without the semantic service, but "
            f"{len(left)} '{VLLM_PROCESS_PATTERN}' process(es) are still running after the "
            f"stop (PIDs {left}) — stop them (pkill -f '{VLLM_PROCESS_PATTERN}') and re-run")
    if release is not None and release.get("still_listed"):
        raise RuntimeError(
            f"{stage or 'this stage'} needs the GPU without the semantic service, but the card "
            f"did not let go of the stopped vLLM: {release['why']} (they left the process table; "
            f"nvidia-smi still lists their CUDA context) — check `nvidia-smi` and re-run")
    return {"service_stopped": True, "check": f"pgrep -f '{VLLM_PROCESS_PATTERN}'",
            "remaining_pids": [], "free_gb": gpu_free_gb()}


# ── Running one worker inside another (the reconstruction stage hosts the
# semantics and the precision core — USER 2026-09-28) ────────────────────────

def _terminate(proc, timeout: float = 5.0) -> None:
    """SIGTERM the hosted child, wait, SIGKILL what is left. The child shares the
    host's process group (no setsid, STAC_INLINE_CHILD), so the pipeline's own
    killpg on a cancelled stage reaches it and its subprocesses as well."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except Exception:  # noqa: BLE001 — subprocess.TimeoutExpired
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            pass


def run_stage_inline(pipe: "WorkerPipe", module_name: str, session_dir: str, config: dict, *,
                     label: str, stage: str = "reconstruction",
                     pct_range: tuple = (0.0, 100.0), cancel_grace_s: float = 30.0) -> None:
    """Run another worker module's ``run(conn, session_dir, config)`` as a child of
    THIS worker and relay what it says: logs verbatim (prefixed with ``label``),
    progress rescaled into ``pct_range`` of ``stage``, a cancel forwarded to the
    child (which stops its own subprocesses and reports) and enforced after
    ``cancel_grace_s``. The child's ``error`` / failed ``done`` — or a child that
    dies without one — raise a RuntimeError naming the label and the reason.

    The child is a plain subprocess (``python -m workers.inline_child``) speaking
    the stage protocol over a multiprocessing Pipe handed down by fd: the pipeline's
    stage workers are daemonic multiprocessing processes and Python forbids those
    to spawn multiprocessing children (pccr 2026-09-28: "daemonic processes are not
    allowed to have children" was the first thing the hosted VLM stage said)."""
    import os
    import subprocess
    import sys
    from multiprocessing import Pipe
    from pathlib import Path

    server_dir = Path(__file__).resolve().parent.parent
    host_conn, child_conn = Pipe()
    fd = child_conn.fileno()
    env = dict(os.environ)
    env["STAC_INLINE_CHILD"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "workers.inline_child", module_name, session_dir, str(fd)],
        cwd=str(server_dir), env=env, pass_fds=(fd,))
    child_conn.close()
    host_conn.send(config)
    lo, hi = float(pct_range[0]), float(pct_range[1])
    error: Optional[dict] = None
    done: Optional[dict] = None
    cancel_sent_at: Optional[float] = None
    try:
        while True:
            if cancel_sent_at is None and pipe.check_cancel():
                try:
                    host_conn.send({"type": "cancel"})
                except Exception:  # noqa: BLE001 — the child may be gone already
                    pass
                cancel_sent_at = time.time()
            if cancel_sent_at is not None and time.time() - cancel_sent_at > cancel_grace_s:
                _terminate(proc)
                raise RuntimeError(f"cancelled during {label}")
            if not host_conn.poll(0.25):
                if proc.poll() is not None and not host_conn.poll(0.0):
                    break
                continue
            try:
                msg = host_conn.recv()
            except EOFError:
                break
            if not isinstance(msg, dict):
                continue
            kind = msg.get("type")
            if kind == "log":
                pipe.send_log(f"[{label}] {msg.get('msg', '')}", level=msg.get("level", "info"))
            elif kind == "progress":
                frac = max(0.0, min(100.0, float(msg.get("pct", 0.0)))) / 100.0
                pipe.send_progress(lo + (hi - lo) * frac, f"{label}: {msg.get('msg', '')}",
                                   stage=stage)
            elif kind == "error":
                error = msg
                if msg.get("traceback"):
                    pipe.send_log(str(msg["traceback"]), level="error")
            elif kind == "done":
                done = msg
                break
    finally:
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            _terminate(proc)
        host_conn.close()
    if cancel_sent_at is not None:
        raise RuntimeError(f"cancelled during {label}")
    if error is not None or done is None or not done.get("success", False):
        why = ((error or {}).get("msg") or (done or {}).get("detail")
               or f"the worker died without reporting (exit code {proc.returncode})")
        raise RuntimeError(f"{label} failed: {why}")
