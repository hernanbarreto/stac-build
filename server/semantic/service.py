# STAC-Builder — semantic service lifecycle (health, auto-start, the ENGINE LEASE and the
# per-job engine).
#
# The reconstruction and SAM3 stages STOP vLLM to get the whole GPU
# (workers/base.py::stop_semantic_service). Every later consumer therefore has
# to be able to bring it back: the user never starts it by hand. This module is
# that single entry point, shared by the VLM worker (subprocess) and by the
# FastAPI routes (spatial Q&A), which previously had no way to restart it and
# answered 500 ConnectionRefused after any pipeline run.
#
# DETERMINISM (docs/plan_determinismo.md points 81, 154, 155, 156 — 2026-10-08):
#   * ONE permit to use the engine AND the card per job — the ENGINE LEASE
#     (logs/semantic_engine_lease.json, a file so the backend process, its stage subprocesses
#     and the chat all see it): while a job's process holds it, only that process starts or
#     queries vLLM; the chat and the background session intel get 'busy' (phase5_qa/api.py,
#     :func:`ensure_service`) and wait. The lease names its holder by pid AND the pid's START
#     TIME (/proc/<pid>/stat field 22) with the kernel's boot id, plus an OWNER (the role:
#     'pipeline' for the manager's job permit, 'vlm_worker', 'object_captioner', 'server' for
#     the server's own GPU tasks) and a TOKEN (the job's id): a lease whose pid was REUSED
#     after a container restart is stale (2026-10-08 — it used to block every job until the
#     file was removed by hand), and a caller of the backend process asks "does a JOB hold
#     the lease?" by its own owner — the manager's job permit lives under the backend's own
#     pid and a pid-only check used to let the chat, the intel and the on-demand captioner
#     through in the middle of a stage (point 81). The server's GPU tasks (meshing, the
#     Segmentation Manager's SAM3, the legacy auto-segmentation) take the SAME lease
#     (:func:`server_gpu_permit`, point 154): a pipeline job refuses them while it holds the
#     card and WAITS for them when they hold it (pipeline_manager).
#   * the VLM stage (and the per-object description pass) runs on ITS OWN vLLM, launched from
#     the job's FROZEN configuration with the deterministic flags of semantic.serve and stopped
#     at the end (:func:`job_engine`): never on a long-lived engine whose prefix cache, launch
#     flags and co-tenants come from other moments. Before launching, any running vLLM is
#     stopped with the VERIFIED stop and the card must hold no foreign compute process
#     (never with SAM3 or the reconstruction on the card).
#   * the engine's IDENTITY (what semantic.serve wrote before exec) is read back from the live
#     process (:func:`service_identity`) and VERIFIED against the job (served model, the sha256
#     of the frozen semantic block, the deterministic flags) before a single call
#     (:func:`verify_service_identity`); the stage writes it into vlm_analysis.json.
#   * the startup wait is a BOUND that FAILS (:class:`EngineUnavailable`) — never a branch to
#     another model or a silent skip (point 156); a double launch cannot happen: a launcher
#     chain already alive is awaited, never duplicated.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional

logger = logging.getLogger(__name__)

_STAC_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHER = _STAC_ROOT / "scripts" / "serve_semantic.sh"
LEASE_NAME = "semantic_engine_lease.json"
# the launcher's health poll (seconds between looks while the weights load) — a cadence, not
# a decision; the only bound is semantic.service.startup_timeout_s, which FAILS
_POLL_S = 5.0


class EngineBusy(RuntimeError):
    """Another live process holds the engine lease: this caller gets 'busy' and waits."""


class EngineUnavailable(RuntimeError):
    """The job's engine could not be brought up (launch error, the startup bound reached,
    a foreign process on the card, the launcher died) — declared, never a fallback."""


class EngineIdentityError(RuntimeError):
    """The vLLM answering is not the one this job launched / expects (another served model,
    another configuration, flags missing, no identity record of a live launcher)."""


def _service_cfg(config: Optional[dict] = None) -> dict:
    if config is None:
        from config import cfg as config  # lazy: keeps this importable from workers
    return (config.get("semantic", {}) or {}).get("service", {}) or {}


def _logs_dir() -> Path:
    from semantic.semantic_config import resolve_path
    return resolve_path("logs")


def is_alive(config: Optional[dict] = None, timeout_s: float = 3.0) -> bool:
    """True when the vLLM endpoint answers /health."""
    import requests
    svc = _service_cfg(config)
    url = f"http://{svc.get('host', '127.0.0.1')}:{svc.get('port', 8799)}/health"
    try:
        return requests.get(url, timeout=timeout_s).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def is_starting() -> bool:
    """True when a vLLM launcher chain exists but is not serving yet (weights loading) —
    every form of the chain (workers.base.VLLM_PATTERNS), not only 'vllm serve'."""
    try:
        from workers.base import vllm_pids
        return bool(vllm_pids())
    except Exception:  # noqa: BLE001
        return False


# ── the engine lease (point 81 / 154) ────────────────────────────────────────────────────

# the owners of a lease — a ROLE, never a free string: the manager's job permit (held under
# the backend's own pid for the stages that run in other processes), the stage workers that
# take the engine themselves, and the server's own GPU tasks (point 154)
JOB_PERMIT_OWNER = "pipeline"
SERVER_TASK_OWNER = "server"
# the token of the Segmentation Manager's SAM3 session — the ONE server task whose permit is
# deliberately re-entered by its own token: a Manager opened again takes over the one SAM3
# model of this process (start_session resets the previous sessions), and its permit goes
# only when the Manager closes (/api/segmentation/refresh) or a session is loaded. Declared
# here (not in main.py) so the pipeline manager can say, while a job waits for it, HOW it is
# released (point 154).
SAM3_MANAGER_TOKEN = "sam3-manager"


def lease_path() -> Path:
    return _logs_dir() / LEASE_NAME


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True            # no /proc: the signal reached it, it is alive


def _boot_id() -> Optional[str]:
    """The kernel's boot id (a reboot changes it; a lease from another boot is stale)."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _proc_start(pid: int) -> Optional[int]:
    """The START TIME of ``pid`` (/proc/<pid>/stat field 22: clock ticks since boot) — what
    tells a process from a later one that got the same pid after a restart. None without
    /proc (the pid alone then identifies the holder, as before 2026-10-08)."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError, TypeError):
        return None
    # the command name (field 2) is parenthesised and may hold spaces: split after it
    tail = stat.rsplit(")", 1)[-1].split()
    try:
        return int(tail[19])              # fields 3.. follow the ')': field 22 is index 19
    except (IndexError, ValueError):
        return None


def _holder_alive(doc: Dict[str, Any]) -> bool:
    """Is the process the lease names still THAT process: alive, from this boot, and started
    when the lease says (a reused pid fails the start-time check)."""
    pid = doc.get("pid", -1)
    if not _pid_alive(pid):
        return False
    boot = doc.get("boot_id")
    if boot is not None and _boot_id() is not None and boot != _boot_id():
        return False
    start = doc.get("pid_start")
    now = _proc_start(pid)
    if start is not None and now is not None and int(start) != int(now):
        return False
    return True


def _same_process(doc: Dict[str, Any], pid: Optional[int] = None) -> bool:
    """Does the lease name ``pid`` (default: this process) — the pid and, when both sides know
    it, the start time."""
    me = int(pid if pid is not None else os.getpid())
    if int(doc.get("pid", -1)) != me:
        return False
    start = doc.get("pid_start")
    now = _proc_start(me)
    return start is None or now is None or int(start) == int(now)


def _is_mine(doc: Dict[str, Any], owner: Optional[str] = None,
             token: Optional[str] = None) -> bool:
    """The lease is this process's AND, when named, this owner's / this token's."""
    if not _same_process(doc):
        return False
    if owner is not None and doc.get("owner") != owner:
        return False
    if token is not None and doc.get("token") != token:
        return False
    return True


def engine_lease_holder() -> Optional[Dict[str, Any]]:
    """The live holder of the engine lease ({pid, pid_start, boot_id, owner, token, stage,
    session, acquired_unix}), or None. A lease whose process is gone — or whose pid now
    belongs to another process (a reused pid after a restart) — is stale: removed, None.

    A lease file that is not a JSON object names no holder at all: it is removed and
    DECLARED in the log (2026-10-08 — it used to be read as 'no lease' while the file stayed,
    so every acquire then lost at the exclusive create and the whole server was refused until
    someone removed it by hand). It cannot be a live holder's: :func:`acquire_engine_lease`
    links a COMPLETE document into place, never a file being written. A lease that exists but
    cannot be READ (permissions, I/O) is a failure that names the file — never 'no lease',
    which would put two users on the card."""
    p = lease_path()
    try:
        raw = p.read_text()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise EngineBusy(f"the engine lease {p} exists but cannot be read ({e}) — nothing takes "
                         f"the engine or the card until it can") from e
    try:
        doc = json.loads(raw)
    except ValueError as e:
        doc = None
        why = f"not JSON ({e})"
    else:
        why = f"a {type(doc).__name__}, not an object"
    if not isinstance(doc, dict):
        logger.warning(f"[semantic] DECLARED: the engine lease {p} is {why} — it names no holder; "
                       f"removed (a lease is linked into place complete, so this is no live "
                       f"holder's file)")
        with contextlib.suppress(OSError):
            p.unlink()
        return None
    if not _holder_alive(doc):
        with contextlib.suppress(OSError):
            p.unlink()
        return None
    return doc


def _busy_reason(holder: Dict[str, Any]) -> str:
    """Why the caller is busy, worded by what holds the lease: a server task (meshing, the
    Segmentation Manager) ends by itself or when the Manager closes; anything else is a
    job's stage."""
    owner = holder.get("owner")
    if owner == SERVER_TASK_OWNER:
        until = ("until the Segmentation Manager closes or a session is loaded"
                 if holder.get("token") == SAM3_MANAGER_TOKEN else "until that task ends")
    else:
        until = "until that job's stage ends"
    return (f"the semantic engine is leased to pid {holder.get('pid')} "
            f"({owner}, {holder.get('stage')}, session "
            f"{holder.get('session')}) — busy {until}")


def engine_available_to(pid: Optional[int] = None, owner: Optional[str] = None,
                        token: Optional[str] = None) -> tuple:
    """(available, reason): whether ``pid`` (default: this process) may start or query the
    engine now — True with no live holder; otherwise only when the holder IS this process
    and, with ``owner`` named, THIS OWNER (and ``token``, when named).

    A caller that names no owner asks by pid, as before — with ONE exception (point 81): the
    manager's job permit (owner ``pipeline``) stands for a job whose stages run in other
    processes; nothing in the backend process may use the engine or the card under it, so
    it is never 'mine' to a pid-only caller (the chat, the intel, the on-demand captioner
    used to pass that check mid-stage). A server GPU task's own permit IS available to a
    pid-only caller of the same process (its captions, say)."""
    holder = engine_lease_holder()
    if holder is None:
        return True, "no lease held"
    if owner is None:
        if _same_process(holder, pid) and holder.get("owner") != JOB_PERMIT_OWNER:
            return True, "this process holds the lease"
        return False, _busy_reason(holder)
    if pid is None and _is_mine(holder, owner, token):
        return True, f"this process holds the lease as {owner}"
    return False, _busy_reason(holder)


def holds_engine_lease(owner: str, token: Optional[str] = None) -> bool:
    """Does THIS process hold the lease as ``owner`` (and ``token``, when named) right now —
    a held permit, not merely an available one (no holder at all is False)."""
    holder = engine_lease_holder()
    return holder is not None and _is_mine(holder, owner, token)


def acquire_engine_lease(owner: str, *, stage: str, session: Optional[str] = None,
                         token: Optional[str] = None) -> Dict[str, Any]:
    """Take the engine lease for THIS process as ``owner`` (idempotent for the same owner —
    and the same ``token``, when the holder's is known — of this process). Another live
    holder RAISES :class:`EngineBusy` — another process, or ANOTHER OWNER of this very
    process (the manager's job permit vs a server GPU task, point 154).

    The lease is taken ATOMICALLY AND COMPLETE: the document is written to a sibling temp
    file of this process and LINKED to the lease path — ``os.link`` fails with EEXIST when
    another holder won, and a lease file can never exist half-written (2026-10-08: the
    O_EXCL create followed by the dump left an empty file behind a crash between the two,
    which blocked every job, the chat and every server task until removed by hand)."""
    p = lease_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": 2, "pid": os.getpid(), "pid_start": _proc_start(os.getpid()),
           "boot_id": _boot_id(), "owner": str(owner),
           "token": (str(token) if token is not None else None), "stage": str(stage),
           "session": (str(session) if session is not None else None),
           "acquired_unix": time.time()}
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    for _ in range(2):
        holder = engine_lease_holder()          # drops a stale lease
        if holder is not None:
            if _is_mine(holder, str(owner), None if token is None else str(token)):
                return holder
            raise EngineBusy(_busy_reason(holder))
        tmp.write_text(json.dumps(doc))
        try:
            os.link(str(tmp), str(p))
        except FileExistsError:
            continue                              # lost the race: read the winner
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()
        return doc
    holder = engine_lease_holder()
    raise EngineBusy(_busy_reason(holder) if holder
                     else f"the lease {p} could not be taken: another holder won it twice in a "
                          f"row (taken and released between two looks)")


def release_engine_lease(owner: Optional[str] = None, token: Optional[str] = None) -> bool:
    """Drop the lease when THIS process holds it — as ``owner`` / with ``token`` when named
    (a server task never drops the manager's job permit, nor the other way round). True
    when something was released."""
    holder = engine_lease_holder()
    if holder is None or not _is_mine(holder, owner, token):
        return False
    with contextlib.suppress(OSError):
        lease_path().unlink()
    return True


# ── the server's own GPU tasks take the SAME permit (point 154) ───────────────────────────

def acquire_server_gpu_permit(task: str, *, session: Optional[str] = None,
                              token: Optional[str] = None) -> Dict[str, Any]:
    """The ONE GPU permit for a GPU task the SERVER process runs itself — the meshing
    endpoints (ShapeR, the per-object meshes), the Segmentation Manager's SAM3, the legacy
    auto-segmentation, the retroactive segmentation (point 154): the engine lease under the
    owner ``server``. Raises :class:`EngineBusy` naming the holder while a pipeline job
    holds the card (the manager's permit) or another server task does; the pipeline
    manager, for its part, WAITS while a server task holds it."""
    return acquire_engine_lease(SERVER_TASK_OWNER, stage=task, session=session, token=token)


def release_server_gpu_permit(token: Optional[str] = None) -> bool:
    """Drop the server's GPU permit (only a ``server`` lease of this process, and only the
    one with ``token`` when named). True when something was released."""
    return release_engine_lease(SERVER_TASK_OWNER, token)


@contextlib.contextmanager
def server_gpu_permit(task: str, *, session: Optional[str] = None,
                      token: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """:func:`acquire_server_gpu_permit` for the block, released on the way out — unless the
    permit was already held by this task (re-entered: left to the outer holder)."""
    held = engine_lease_holder()
    reentered = held is not None and _is_mine(held, SERVER_TASK_OWNER, token)
    doc = acquire_server_gpu_permit(task, session=session, token=token)
    try:
        yield doc
    finally:
        if not reentered:
            release_server_gpu_permit(token)


# ── the identity of the running engine (point 155) ───────────────────────────────────────

def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace")
    except OSError:
        return ""


def service_identity() -> Optional[Dict[str, Any]]:
    """The identity semantic.serve wrote for the vLLM running NOW (logs/semantic_service.json),
    or None when there is no record, or its launcher pid is gone / is no longer a vLLM: the
    identity of a dead engine says nothing about the one answering. Returns the whole document
    ({identity, process}); the stable part a product records is ``doc["identity"]``."""
    from semantic.serve import identity_path
    try:
        doc = json.loads(identity_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("identity"), dict):
        return None
    pid = (doc.get("process") or {}).get("pid")
    if not _pid_alive(pid if pid is not None else -1):
        return None
    cmd = _cmdline(int(pid))
    if "vllm" not in cmd or "serve" not in cmd:
        return None
    return doc


def verify_service_identity(expected_backend: str, semantic_cfg: Mapping[str, Any],
                            log: Callable[[str], Any] = print) -> Dict[str, Any]:
    """The live engine's identity, VERIFIED to be the one this job expects: a live launcher
    record exists, it serves ``expected_backend``'s model, it was configured from exactly
    ``semantic_cfg`` (sha256 of the block), every deterministic flag of semantic.serve is in
    its argv, and /v1/models serves the model. Any difference RAISES
    :class:`EngineIdentityError` naming it. Returns the stable identity."""
    from semantic.semantic_config import _DEFAULTS, _deep_merge, backend_config
    from semantic.serve import deterministic_serve_args, semantic_config_sha256
    doc = service_identity()
    if doc is None:
        raise EngineIdentityError("no identity record of a live vLLM launcher (logs/"
                                  "semantic_service.json): the engine answering was not "
                                  "started by semantic.serve, or it is gone")
    ident = doc["identity"]
    # the launcher hashes the block MERGED with the loader's defaults (semantic.serve.
    # semantic_config_from); the same merge here, or a raw block never matches
    semantic_cfg = _deep_merge(_DEFAULTS, dict(semantic_cfg))
    b = backend_config(expected_backend, dict(semantic_cfg))
    diffs: List[str] = []
    if ident.get("served_model_name") != b["served_model_name"]:
        diffs.append(f"served model {ident.get('served_model_name')!r} != {b['served_model_name']!r}")
    want_sha = semantic_config_sha256(semantic_cfg)
    if ident.get("semantic_config_sha256") != want_sha:
        diffs.append(f"semantic configuration {str(ident.get('semantic_config_sha256'))[:12]} != "
                     f"the job's frozen block {want_sha[:12]}")
    argv = [str(a) for a in (ident.get("argv") or [])]
    want = deterministic_serve_args(int(b.get("max_model_len", 32768)),
                                    int((semantic_cfg.get("generation") or {}).get("seed", 0)))
    # every deterministic flag must be in the engine's argv — with its value, when it has one
    i = 0
    while i < len(want):
        tok = want[i]
        val = want[i + 1] if i + 1 < len(want) and not want[i + 1].startswith("--") else None
        if tok not in argv:
            diffs.append(f"flag {tok} missing from the engine's argv")
        elif val is not None:
            j = argv.index(tok)
            got = argv[j + 1] if j + 1 < len(argv) else None
            if got != val:
                diffs.append(f"{tok} is {got!r} in the engine's argv, this job needs {val!r}")
        i += 2 if val is not None else 1
    if not (ident.get("batch_invariance") or {}).get("enabled"):
        diffs.append("batch invariance not enabled")
    if diffs:
        raise EngineIdentityError("the running vLLM is not this job's engine: " + "; ".join(diffs))
    from semantic.backends import make_backend
    h = make_backend(expected_backend, dict(semantic_cfg)).health()
    if not h.get("ok"):
        raise EngineIdentityError(f"the engine does not serve '{b['served_model_name']}' "
                                  f"({h.get('error')}; served {h.get('served_models')})")
    log(f"[semantic] engine identity verified: {ident.get('sha256', '')[:12]} — "
        f"{ident.get('served_model_name')} on {(ident.get('card') or {}).get('key')}, weights "
        f"{(ident.get('weights') or {}).get('model_id')} @ "
        f"{str((ident.get('weights') or {}).get('revision'))[:12]}, batch invariance "
        f"{(ident.get('batch_invariance') or {}).get('form')}")
    return dict(ident)


# ── launching ────────────────────────────────────────────────────────────────────────────

def release_cached_vram(log: Optional[Callable[[str], Any]] = None) -> Optional[float]:
    """Give back the VRAM PyTorch's caching allocator holds in THIS process — only when this
    process already initialised CUDA (it ran GPU work: the certification's re-consolidation is
    the case this exists for). A process that never touched the card has nothing cached and
    must not create a CUDA context to find that out: ``torch.cuda.mem_get_info`` /
    ``ipc_collect`` create one, and the backend did exactly that when it kicked vLLM at its own
    boot (2026-10-07: nvidia-smi listed uvicorn main:app holding 416 MiB of the A100 for the
    backend's whole life), which makes every later ``repro.require_exclusive_gpu`` refuse the
    card. ``torch`` is read from ``sys.modules``: a process that never imported it never
    initialised CUDA through it. Returns the GB released (None when nothing was asked of the
    card); never fatal."""
    def _log(msg: str) -> None:
        if log:
            log(msg)

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_initialized():
            return None
        free_before = torch.cuda.mem_get_info()[0] / 2 ** 30
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_after = torch.cuda.mem_get_info()[0] / 2 ** 30
        if free_after - free_before > 0.1:
            _log(f"[gpu] released {free_after - free_before:.1f} GB of cached "
                 f"VRAM before starting vLLM ({free_after:.1f} GB free)")
        return free_after - free_before
    except Exception as e:  # noqa: BLE001 — declared, never fatal
        _log(f"[gpu] could not release cached VRAM ({e}) — starting anyway")
        return None


def _die_with_parent():
    # USER ORDER 2026-09-04: the chat must DIE with the backend — twice
    # today orphaned vLLMs (start_new_session) kept 20-45 GB of VRAM after
    # the backend was killed and OOM'd the GPU stages. PR_SET_PDEATHSIG
    # delivers SIGTERM to the (exec-chained bash→python→vllm) child the
    # moment its parent exits, however the parent died; vLLM shuts its
    # EngineCore down on SIGTERM.
    try:
        import ctypes
        import signal as _sig
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _sig.SIGTERM)
    except Exception:  # noqa: BLE001 — best effort, non-Linux fallback
        pass


def _launch(launcher_args: List[str], log: Callable[[str], Any]):
    """Start the launcher chain (scripts/serve_semantic.sh → semantic.serve → vllm serve) in
    its own session, watched by semantic.watchdog so it dies with the backend. Returns the
    Popen of the launcher (its pid is the vLLM's pid after the exec chain)."""
    # USER 2026-10-05 (zaragoza): with vLLM in the backend's own process group, the GPU
    # handover that stops it took the backend down with it (twice it had survived because
    # the vLLM came from an earlier, dead backend). Own session (start_new_session) — no
    # group-wide signal reaches the backend — AND PR_SET_PDEATHSIG keeps the chat dying
    # with the backend as before.
    _proc = subprocess.Popen(["bash", str(_LAUNCHER), *launcher_args], cwd=str(_STAC_ROOT),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             preexec_fn=_die_with_parent, start_new_session=True)
    # USER 2026-10-06 ("todo lo que haya debe morir", the VLM included): a watchdog on
    # the BACKEND's pid kills the whole vLLM session the moment the backend is gone —
    # PR_SET_PDEATHSIG alone follows the launching THREAD and reaches only the launcher
    _backend = int(os.environ.get("STAC_BACKEND_PID") or os.getpid())
    subprocess.Popen([sys.executable, "-m", "semantic.watchdog", str(_backend), str(_proc.pid)],
                     cwd=str(Path(__file__).resolve().parent.parent),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return _proc


def _wait_alive(config: Optional[dict], timeout_s: float, log: Callable[[str], Any],
                cancelled: Optional[Callable[[], bool]] = None,
                proc=None) -> bool:
    """Poll /health until it answers (True), the bound passes (False), the caller cancels
    (False) or the launcher ``proc`` exits first (False — a launcher that died never comes
    up; waiting the whole bound for it would turn a declared refusal into a timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if cancelled and cancelled():
            return False
        if is_alive(config):
            return True
        if proc is not None and proc.poll() is not None and not is_alive(config):
            return False
        time.sleep(_POLL_S)
    return False


def _launcher_log_tail(n: int = 12) -> str:
    """The last lines of logs/semantic_latest.log — the launcher's own words when it refused
    (weights not pinned, a configuration without the block) or vLLM's when it died."""
    try:
        lines = (_logs_dir() / "semantic_latest.log").read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no launcher log)"


def ensure_service(config: Optional[dict] = None,
                   log: Optional[Callable[[str], Any]] = None,
                   cancelled: Optional[Callable[[], bool]] = None,
                   timeout_s: Optional[float] = None,
                   owner: Optional[str] = None) -> bool:
    """Healthcheck the semantic service; if down, start it and wait until it serves.

    Returns True once the endpoint is healthy, False on timeout / launch error — and False,
    declared, when the engine lease is held by someone else (point 81: the chat and the
    session intel wait for the job's stage to end; ``owner`` names the caller — 'chat',
    'intel' — so a lease of THIS process under another owner, the manager's job permit or
    a server GPU task, is 'busy' too; without it the pid-only rule of
    :func:`engine_available_to` applies) or when a launcher chain is already starting
    (awaited, never duplicated: point 155).
    Loading Qwen3-VL weights takes minutes — pass a short `timeout_s` from
    request handlers that must not block, and report the wait to the user.
    """
    def _log(msg: str) -> None:
        if log:
            log(msg)

    ok, why = engine_available_to() if owner is None else engine_available_to(owner=owner)
    if not ok:
        _log(f"Semantic service busy — {why}")
        return False
    if is_alive(config):
        return True
    if not _LAUNCHER.exists():
        _log(f"Semantic service down and launcher missing ({_LAUNCHER})")
        return False

    if timeout_s is None:
        timeout_s = float(_service_cfg(config).get("startup_timeout_s", 900))
    if is_starting():
        # a launcher chain is already alive (another caller kicked it seconds ago): wait for
        # IT — a second chain would fight the first for the card and the loser dies on memory
        _log("Semantic service is starting (a launcher is already running) — not starting "
             "a second instance")
        if timeout_s <= 0:
            return False
        if _wait_alive(config, timeout_s, _log, cancelled):
            _log("Semantic service is up")
            return True
        _log(f"Semantic service did not come up within {timeout_s:.0f}s")
        return False

    _log("Semantic service down — starting vLLM (Qwen3-VL)...")

    # Give back the VRAM THIS process is holding before vLLM asks for its own.
    # A GPU stage that ran here (the epoch re-consolidation is the one that
    # bites) leaves PyTorch's caching allocator sitting on tens of GB it no
    # longer uses, and vLLM refuses to start below its
    # `--gpu-memory-utilization` share. pccr 2026-09-19: the certification
    # stopped vLLM to take the whole GPU, consolidated, asked for it back, and
    # vLLM died with "Free memory on device cuda:0 (16.27/47.41 GiB) … is less
    # than desired GPU memory utilization (0.5, 23.71 GiB)" — then the
    # certification blocked FOREVER waiting for a service that could not start
    # while the certification itself held the memory. A deadlock, not a crash:
    # nothing in the log, 1 % CPU, S (sleeping), for as long as anyone waited.
    release_cached_vram(_log)

    try:
        proc = _launch([], _log)
    except Exception as e:  # noqa: BLE001
        _log(f"Could not launch semantic service: {e}")
        return False

    if timeout_s <= 0:
        # main.py kicks the launcher at boot ON PURPOSE without waiting, so the
        # server never blocks on a model load. Saying "did not come up within
        # 0s" made a deliberate non-wait read as a failure (USER 2026-09-16).
        _log("Semantic service launched — not waiting for it (loads in the "
             "background; the chat reloads when it is ready)")
        return False
    if _wait_alive(config, timeout_s, _log, cancelled, proc=proc):
        _log("Semantic service is up")
        return True
    if proc.poll() is not None:
        _log(f"Semantic service launcher exited (code {proc.returncode}) before serving:\n"
             f"{_launcher_log_tail()}")
        return False
    _log(f"Semantic service did not come up within {timeout_s:.0f}s")
    return False


# ── the job's own engine (point 155) ─────────────────────────────────────────────────────

def _foreign_gpu_tenants(log: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """Compute processes on the visible card(s) other than this process: SAM3, the
    reconstruction, a stray vLLM — anything the engine must never start beside (point 81).
    DECLARED limit (as workers.base.await_vram_release): a pid nvidia-smi lists that is no
    process of this pid namespace (a container whose driver reports host pids) cannot be
    attributed — it is logged, not refused, when THIS process holds a CUDA context (it is
    then most likely ourselves)."""
    import repro
    try:
        cards = {c["uuid"] for c in repro._visible_cards(repro.gpu_cards())}
        procs = [p for p in repro.gpu_compute_processes() if p["gpu_uuid"] in cards]
    except repro.ReproError as e:
        raise EngineUnavailable(f"the card cannot be checked before starting vLLM ({e})") from e
    torch = sys.modules.get("torch")
    mine_ctx = bool(torch is not None and torch.cuda.is_initialized())
    foreign, unattributed = [], []
    for p in procs:
        if p["pid"] == os.getpid():
            continue
        if not os.path.exists(f"/proc/{p['pid']}"):
            unattributed.append(p)
        else:
            foreign.append(p)
    if unattributed:
        if mine_ctx and not foreign:
            log(f"[semantic] DECLARED: nvidia-smi lists {[p['pid'] for p in unattributed]} on "
                f"the card, no process of this pid namespace, while this process holds a CUDA "
                f"context — taken as ourselves")
        else:
            foreign.extend(unattributed)
    return foreign


@contextlib.contextmanager
def job_engine(config: Mapping[str, Any], *, backend: str, owner: str, stage: str,
               session_dir: Optional[os.PathLike] = None,
               run_config_path: Optional[os.PathLike] = None,
               log: Callable[[str], Any] = print,
               cancelled: Optional[Callable[[], bool]] = None) -> Iterator[Dict[str, Any]]:
    """The engine of ONE job stage (point 155): take the lease, stop any running vLLM (the
    verified stop), require a card without foreign tenants, launch vLLM from the job's
    FROZEN configuration (``run_config_path`` — the file semantic.serve reads its block from;
    ``config`` is the same configuration as a dict, the one the stage got), wait within the
    startup bound (fail past it), verify the engine's identity, yield it; on the way out stop
    the engine (verified) and release the lease — whatever happened inside.

    Raises :class:`EngineBusy` (another job holds the lease), :class:`EngineUnavailable`
    (launch / bound / tenants) or :class:`EngineIdentityError` (another engine answered)."""
    from workers.base import stop_semantic_service_verified
    semantic_cfg = dict((config.get("semantic") or {}))
    if not semantic_cfg:
        raise EngineUnavailable("the job's configuration has no 'semantic' block")
    # a lease this owner already holds (a server task that took the GPU permit for the whole
    # of its work and now wants the engine for part of it, point 154) is re-entered and left
    # to that holder on the way out; one taken here is released here
    held = engine_lease_holder()
    reentered = held is not None and _is_mine(held, str(owner))
    acquire_engine_lease(owner, stage=stage, session=(str(session_dir) if session_dir else None))
    launched = None
    try:
        # whatever vLLM is up came from another moment (the chat's, an earlier job's): it is
        # stopped, verified, and the card must then hold nobody else
        stop_semantic_service_verified(None, stage=stage, log=log)
        tenants = _foreign_gpu_tenants(log)
        if tenants:
            raise EngineUnavailable(
                f"{stage}: vLLM is never started with another process on the card (point 81) — "
                + "; ".join(f"pid {p['pid']} {p['name'] or '?'} "
                            f"{p['used_mib'] if p['used_mib'] is not None else 'N/A'} MiB"
                            for p in tenants))
        release_cached_vram(log)
        args = ["--backend", str(backend)]
        if run_config_path is not None:
            args += ["--config", str(run_config_path)]
        log(f"[semantic] {stage}: launching this job's own vLLM ({backend}"
            + (f", configuration {Path(run_config_path).name}" if run_config_path else "") + ")")
        try:
            launched = _launch(args, log)
        except Exception as e:  # noqa: BLE001
            raise EngineUnavailable(f"{stage}: could not launch vLLM ({e})") from e
        timeout_s = float((semantic_cfg.get("service") or {}).get("startup_timeout_s", 900))
        t0 = time.time()
        if not _wait_alive(config, timeout_s, log, cancelled, proc=launched):
            if cancelled and cancelled():
                raise EngineUnavailable(f"{stage}: cancelled while vLLM was starting")
            if launched.poll() is not None:
                raise EngineUnavailable(f"{stage}: the vLLM launcher exited (code "
                                        f"{launched.returncode}) before serving:\n"
                                        f"{_launcher_log_tail()}")
            raise EngineUnavailable(f"{stage}: vLLM did not come up within the startup bound "
                                    f"semantic.service.startup_timeout_s = {timeout_s:.0f} s "
                                    f"(point 156: a bound that fails, never a branch)")
        log(f"[semantic] {stage}: vLLM up after {time.time() - t0:.0f} s")
        identity = verify_service_identity(backend, semantic_cfg, log=log)
        yield identity
    finally:
        if launched is not None:
            try:
                stop_semantic_service_verified(None, stage=f"{stage} (engine shutdown)", log=log)
                log(f"[semantic] {stage}: this job's vLLM stopped (verified)")
            except Exception as e:  # noqa: BLE001 — declared; the lease still goes
                log(f"[semantic] {stage}: the engine shutdown could not be verified: {e}")
        if not reentered:
            release_engine_lease(str(owner))
