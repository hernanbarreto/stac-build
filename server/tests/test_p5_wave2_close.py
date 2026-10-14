"""Wave 2 close, package P5 — the remaining orchestration points of docs/plan_determinismo.md
(81, 111, 112/151, 154, 156, 162 — 2026-10-08), on the REAL functions the server calls, CPU only.

- 81: the engine lease names its holder by pid + start time + boot id, with an owner and a
  token: a reused pid is stale; the manager's job permit under the backend's own pid is
  never 'mine' to the chat / the intel / a pid-only caller; ensure_service asked as 'chat'
  refuses under it.
- 154: the server's GPU tasks take the SAME lease as 'server' — refused while the manager
  holds it, re-entrant, and a job that starts while one holds it WAITS (and can be cancelled
  while waiting).
- 111: the cancel is cooperative with a bound — a stage that polls check_cancel exits at a
  safe point and nothing is killed; one that does not is killed past the bound; the job ends
  CANCELLED either way and the select of an epoch is refused while a job is active.
- 112/151: config.py raises on a broken configuration; the SAM3 worker receives the job's
  configuration.
- 156: no InternVL3 / scene_analyzer in the server.
- 162: the command's prompts become the job's inputs (normalised as the session's writers
  would), the scan's saved prompts only when the command carries none.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_manager as PM  # noqa: E402
import semantic.service as svc  # noqa: E402
from pipeline_manager import (JobStatus, PipelineJob, PipelineManager, PipelineStage,  # noqa: E402
                              StageId, StageState, select_stages)

SERVER_DIR = Path(__file__).resolve().parents[1]
UI_APP = SERVER_DIR.parent / "ui" / "src" / "App.tsx"


# ── fixtures ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def lease(tmp_path, monkeypatch):
    """The engine lease file in tmp (never the backend's logs/)."""
    p = tmp_path / "lease.json"
    monkeypatch.setattr(svc, "lease_path", lambda: p)
    yield p
    with_suppress_unlink(p)


def with_suppress_unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def _base_config(**over) -> dict:
    cfg = {"reconstruction": {"precision": {"runner": {"python_da3": sys.executable,
                                                       "python_mapanything": sys.executable,
                                                       "threads": 2}},
                              "backend": "vggtomega"},
           "pipeline": {"auto_segment": True, "auto_tsdf": False},
           "certify": {"auto_after_segmentation": True},
           "semantic": {"service": {"startup_timeout_s": 5}},
           "segmentation": {"object_captions": {"enabled": True}}}
    cfg.update(over)
    return cfg


def _session(tmp_path, name="sess") -> Path:
    s = tmp_path / name
    (s / "frames").mkdir(parents=True)
    (s / "output").mkdir()
    return s


def _job(session_dir: Path, stage_ids, kind="reconstruct", scan_key=None) -> PipelineJob:
    stages = [StageState(stage=PipelineStage(id=s, enabled=True)) for s in stage_ids]
    job = PipelineJob(session_id=session_dir.name, stages=stages, scan_key=scan_key, kind=kind)
    job.session_dir = str(session_dir)
    return job


# ── 81: the lease names its holder, by pid + start time + boot id, owner and token ────

def test_the_lease_carries_the_holders_start_time_and_a_reused_pid_is_stale(lease):
    me = os.getpid()
    start = svc._proc_start(me)
    assert isinstance(start, int) and start > 0, "/proc/<pid>/stat field 22"
    assert svc._proc_start(-1) is None
    doc = svc.acquire_engine_lease("vlm_worker", stage="VLM stage", session="s1", token="j1")
    assert doc["pid"] == me and doc["pid_start"] == start and doc["token"] == "j1"
    assert doc["boot_id"] == svc._boot_id() and doc["version"] == 2
    assert svc.engine_lease_holder()["pid_start"] == start
    assert svc.release_engine_lease("vlm_worker", "j1") is True
    # the same pid with ANOTHER start time: a process that got this pid after a restart
    lease.write_text(json.dumps({"pid": me, "pid_start": start + 12345, "boot_id": svc._boot_id(),
                                 "owner": "pipeline", "stage": "job", "token": "k"}))
    assert svc.engine_lease_holder() is None and not lease.exists(), "stale: removed"
    # another boot: stale too
    lease.write_text(json.dumps({"pid": me, "pid_start": start, "boot_id": "not-this-boot",
                                 "owner": "pipeline", "stage": "job"}))
    assert svc.engine_lease_holder() is None
    # a lease written before 2026-10-08 (pid only) of a live process is still a holder
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "SAM3"}))
    assert svc.engine_lease_holder() is not None
    assert not svc.engine_available_to()[0] and "leased to pid" in svc.engine_available_to()[1]
    lease.unlink()


def test_the_managers_job_permit_is_never_mine_to_a_pid_only_caller_and_owners_are_told_apart(lease):
    """The 81 bug: the manager holds the job permit under the backend's OWN pid, and a
    pid-only check let the chat, the intel and the captioner through mid-stage."""
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="pccr@x", token="pccr@x")
    ok, why = svc.engine_available_to()
    assert not ok and "leased to pid" in why and "pipeline" in why
    assert not svc.engine_available_to(owner="chat")[0]
    assert not svc.engine_available_to(owner="intel")[0]
    assert svc.engine_available_to(owner=svc.JOB_PERMIT_OWNER, token="pccr@x")[0]
    assert not svc.engine_available_to(owner=svc.JOB_PERMIT_OWNER, token="other")[0]
    # the same pid under ANOTHER owner cannot take it (a server GPU task vs the job's permit)
    with pytest.raises(svc.EngineBusy, match="pipeline"):
        svc.acquire_engine_lease(svc.SERVER_TASK_OWNER, stage="meshing")
    with pytest.raises(svc.EngineBusy):
        svc.acquire_server_gpu_permit("meshing", session="s", token="m")
    # ... nor release it
    assert svc.release_engine_lease(svc.SERVER_TASK_OWNER) is False
    assert svc.release_server_gpu_permit("m") is False
    assert svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "other-token") is False
    assert lease.exists()
    assert svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "pccr@x") is True and not lease.exists()
    # a worker's own lease IS available to a pid-only caller of the same process
    svc.acquire_engine_lease("vlm_worker", stage="VLM stage")
    assert svc.engine_available_to()[0] and "this process" in svc.engine_available_to()[1]
    assert svc.acquire_engine_lease("vlm_worker", stage="VLM stage")["owner"] == "vlm_worker", "idempotent"
    assert not svc.engine_available_to(owner="chat")[0]
    assert svc.release_engine_lease() is True, "the plain release still drops this process's lease"


def test_the_server_gpu_permit_is_the_same_lease_reentrant_for_its_task(lease):
    doc = svc.acquire_server_gpu_permit("ShapeR meshing", session="s1", token="shape:s1")
    assert doc["owner"] == svc.SERVER_TASK_OWNER and doc["stage"] == "ShapeR meshing"
    # its own captions (a pid-only caller of the same process) may use the engine; the chat not
    assert svc.engine_available_to()[0]
    assert not svc.engine_available_to(owner="chat")[0]
    assert svc.engine_available_to(owner=svc.SERVER_TASK_OWNER, token="shape:s1")[0]
    # re-entered by the same task: left to the outer holder on the way out
    with svc.server_gpu_permit("ShapeR meshing", session="s1", token="shape:s1"):
        pass
    assert svc.engine_lease_holder() is not None, "still held by the outer task"
    # another server task cannot take it meanwhile
    with pytest.raises(svc.EngineBusy):
        svc.acquire_server_gpu_permit("per-object meshing", session="s1", token="tsdf:s1")
    # the manager's job permit cannot either (the job WAITS — pipeline_manager._take_job_lease)
    with pytest.raises(svc.EngineBusy):
        svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", token="k")
    assert svc.release_server_gpu_permit("other") is False
    assert svc.release_server_gpu_permit("shape:s1") is True and not lease.exists()
    # a block that CREATED the permit releases it
    with svc.server_gpu_permit("x", session="s1", token="t"):
        assert svc.engine_lease_holder()["token"] == "t"
    assert svc.engine_lease_holder() is None


def test_ensure_service_asked_as_the_chat_refuses_under_this_processs_job_permit(lease, monkeypatch):
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="s", token="s")
    launched = []
    monkeypatch.setattr(svc, "_launch", lambda args, log: launched.append(args))
    monkeypatch.setattr(svc, "is_alive", lambda config=None, timeout_s=3.0: False)
    said = []
    assert svc.ensure_service({}, log=said.append, timeout_s=0.0, owner="chat") is False
    assert any("busy" in m and "leased" in m for m in said) and not launched
    # a pid-only caller of this process is refused under the job permit too
    said.clear()
    assert svc.ensure_service({}, log=said.append, timeout_s=0.0) is False
    assert any("busy" in m for m in said) and not launched
    svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "s")
    # the chat's own check in phase5_qa sees the lease, pid or no pid
    from phase5_qa import api
    fake_main = types.SimpleNamespace(pipeline_manager=types.SimpleNamespace(get_all_jobs=lambda: {}))
    monkeypatch.setitem(sys.modules, "main", fake_main)
    assert api._engine_busy() is None
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="s", token="s")
    assert "leased" in api._engine_busy()
    # the server's permit is 'busy' to the chat as well
    svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "s")
    svc.acquire_server_gpu_permit("meshing", session="s", token="m")
    assert "leased" in api._engine_busy()
    svc.release_server_gpu_permit("m")
    assert api._engine_busy() is None


def test_job_engine_reenters_the_tasks_own_permit_and_leaves_it_held(lease, monkeypatch):
    """A server task holding the GPU permit for the whole of its work (the legacy
    auto-segmentation: VLM, then SAM3) takes the engine for part of it through job_engine
    under the SAME owner: re-entered, not released on the way out."""
    import workers.base as wb
    monkeypatch.setattr(wb, "stop_semantic_service_verified", lambda pipe=None, stage="", log=None: None)
    monkeypatch.setattr(svc, "_foreign_gpu_tenants", lambda log: [])
    monkeypatch.setattr(svc, "release_cached_vram", lambda log=None: None)
    proc = types.SimpleNamespace(poll=lambda: None, returncode=None, pid=4242)
    monkeypatch.setattr(svc, "_launch", lambda args, log: proc)
    monkeypatch.setattr(svc, "_wait_alive", lambda config, t, log, cancelled=None, proc=None: True)
    monkeypatch.setattr(svc, "verify_service_identity", lambda backend, cfg, log=print: {"sha256": "id"})
    cfg = {"semantic": {"service": {"startup_timeout_s": 5}}}
    svc.acquire_server_gpu_permit("auto-segmentation", session="s", token="auto:s")
    with svc.job_engine(cfg, backend="qwen_local", owner=svc.SERVER_TASK_OWNER,
                        stage="auto-segmentation VLM", log=lambda m: None) as ident:
        assert ident["sha256"] == "id"
    assert svc.engine_lease_holder()["token"] == "auto:s", "the task's permit survives the engine"
    assert svc.release_server_gpu_permit("auto:s") is True
    # a job_engine that TOOK the lease releases it, under its owner
    with svc.job_engine(cfg, backend="qwen_local", owner="vlm_worker", stage="VLM stage", log=lambda m: None):
        assert svc.engine_lease_holder()["owner"] == "vlm_worker"
    assert svc.engine_lease_holder() is None


# ── 154: a job waits for a server GPU task; 111: a cancel while waiting ──────────────

def _run(pm: PipelineManager, job: PipelineJob, config: dict, *, replace=False, force=False):
    done = {}

    async def _complete(sid, ok):
        done["ok"] = ok

    async def _body():
        pm._jobs[pm.job_key(job.session_id, job.scan_key, job.kind)] = job
        await pm._run_pipeline(job, job.session_dir, config, None, _complete, replace, force)

    asyncio.run(_body())
    return done.get("ok")


def test_a_job_waits_for_a_server_gpu_task_and_refuses_a_foreign_holder(tmp_path, lease, monkeypatch, caplog):
    s = _session(tmp_path)
    pm = PipelineManager()
    ran = []
    messages = []

    async def _stage(job, st, session_dir, config, on_progress, replace):
        ran.append(st.stage.id)
        return True

    async def _progress(sid, d):
        messages.append(d["stages"][0]["message"])

    monkeypatch.setattr(pm, "_run_stage", _stage)
    monkeypatch.setattr(PM, "LEASE_POLL_S", 0.02)
    svc.acquire_server_gpu_permit("ShapeR meshing", session="other", token="shape:other")
    job = _job(s, [StageId.CLOUDCOMPY])
    done = {}

    async def _complete(sid, ok):
        done["ok"] = ok

    async def _body():
        pm._jobs[pm.job_key(job.session_id, None, job.kind)] = job
        task = asyncio.create_task(pm._run_pipeline(job, job.session_dir, _base_config(), _progress,
                                                    _complete, False, False))
        # the job records its environment first (git, libraries — seconds): wait until it is
        # AT the permit, then hold it a little longer
        for _ in range(600):
            if messages and "waiting for the GPU" in messages[-1]:
                break
            await asyncio.sleep(0.05)
        assert messages and "waiting for the GPU" in messages[-1], messages
        await asyncio.sleep(0.15)
        assert ran == [], "the job waits while the server task holds the card"
        assert svc.release_server_gpu_permit("shape:other") is True
        await task

    with caplog.at_level(logging.INFO, logger="pipeline_manager"):
        asyncio.run(_body())
    assert ran == [StageId.CLOUDCOMPY] and done["ok"] is True and job.status == JobStatus.DONE
    assert any("waiting for the GPU: ShapeR meshing" in m for m in messages)
    assert any("the GPU permit is free again" in r.getMessage() for r in caplog.records)
    assert svc.engine_lease_holder() is None, "the job's permit went with it"
    # a holder of ANOTHER process is a refusal, not a wait (as before)
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "VLM stage"}))
    job2 = _job(s, [StageId.CLOUDCOMPY])
    ran.clear()
    assert _run(pm, job2, _base_config()) is False and ran == []
    assert "leased to pid" in job2.stages[0].message
    lease.unlink()


def test_a_job_cancelled_while_waiting_for_the_permit_ends_cancelled(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    pm = PipelineManager()
    monkeypatch.setattr(pm, "_run_stage", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no stage")))
    monkeypatch.setattr(PM, "LEASE_POLL_S", 0.02)
    svc.acquire_server_gpu_permit("per-object meshing", session="other", token="tsdf:other")
    job = _job(s, [StageId.CLOUDCOMPY])
    done = {}

    async def _complete(sid, ok):
        done["ok"] = ok

    messages = []

    async def _progress(sid, d):
        messages.append(d["stages"][0]["message"])

    async def _body():
        pm._jobs[pm.job_key(job.session_id, None, job.kind)] = job
        job.status = JobStatus.RUNNING
        job._task = asyncio.create_task(pm._run_pipeline(job, job.session_dir, _base_config(), _progress,
                                                         _complete, False, False))
        for _ in range(600):                      # until the job is AT the permit
            if messages and "waiting for the GPU" in messages[-1]:
                break
            await asyncio.sleep(0.05)
        assert messages and "waiting for the GPU" in messages[-1], messages
        await pm._cancel_job(job)
        await job._task

    asyncio.run(_body())
    assert job.status == JobStatus.CANCELLED and done["ok"] is False
    assert job.stages[0].status == JobStatus.CANCELLED and "cancelled while waiting" in job.stages[0].message
    assert svc.release_server_gpu_permit("tsdf:other") is True


# ── 111: the cooperative cancel with a bound, on a REAL stage process ────────────────

_POLLING_WORKER = """
import time
from workers.base import run_worker_safe

def _work(pipe, session_dir, config):
    deadline = time.time() + 60
    while time.time() < deadline:
        if pipe.check_cancel():
            pipe.send_log("cancel acknowledged at a safe point")
            return
        time.sleep(0.02)

def run(conn, session_dir, config):
    run_worker_safe(_work, conn, session_dir, config)
"""

_DEAF_WORKER = """
import time
from workers.base import run_worker_safe

def _work(pipe, session_dir, config):
    time.sleep(60)          # never polls check_cancel: not between transactions

def run(conn, session_dir, config):
    run_worker_safe(_work, conn, session_dir, config)
"""


def _cancel_real_stage(tmp_path, monkeypatch, worker_src: str, modname: str):
    mod = tmp_path / f"{modname}.py"
    mod.write_text(textwrap.dedent(worker_src))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(PM.STAGE_REGISTRY, StageId.CLOUDCOMPY,
                        {"label": "fake", "icon": "x", "module": modname})
    s = _session(tmp_path)
    pm = PipelineManager()
    job = _job(s, [StageId.CLOUDCOMPY])
    done = {}

    async def _complete(sid, ok):
        done["ok"] = ok

    async def _body():
        pm._jobs[pm.job_key(job.session_id, None, job.kind)] = job
        job.status = JobStatus.RUNNING
        job._task = asyncio.create_task(pm._run_pipeline(job, job.session_dir, _base_config(), None,
                                                         _complete, False, False))
        for _ in range(600):
            if job._process is not None and job._process.is_alive():
                break
            await asyncio.sleep(0.05)
        assert job._process is not None and job._process.is_alive(), "the stage process is up"
        proc = job._process
        t0 = time.time()
        await pm._cancel_job(job)
        await job._task
        return proc, time.time() - t0

    proc, dt = asyncio.run(_body())
    return job, done, proc, dt


def test_a_stage_that_polls_check_cancel_exits_at_a_safe_point_and_nothing_is_killed(tmp_path, lease, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="pipeline_manager"):
        job, done, proc, dt = _cancel_real_stage(tmp_path, monkeypatch, _POLLING_WORKER, "p5_polling_worker")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("acknowledged the cancel and exited at a safe point" in m for m in msgs), msgs
    assert not any("killing its process tree" in m for m in msgs)
    assert any("cancel acknowledged at a safe point" in m for m in msgs), "the worker's own words reached the log"
    assert dt < PM.CANCEL_ACK_BOUND_S
    assert not proc.is_alive()
    assert job.status == JobStatus.CANCELLED and job.stages[0].status == JobStatus.CANCELLED
    assert done["ok"] is False, "a cancelled job never completes, even when its worker returned cleanly"
    assert svc.engine_lease_holder() is None


def test_a_stage_deaf_to_the_cancel_is_killed_past_the_bound(tmp_path, lease, monkeypatch, caplog):
    monkeypatch.setattr(PM, "CANCEL_ACK_BOUND_S", 0.6)
    with caplog.at_level(logging.INFO, logger="pipeline_manager"):
        job, done, proc, dt = _cancel_real_stage(tmp_path, monkeypatch, _DEAF_WORKER, "p5_deaf_worker")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("did not exit within the cancel bound" in m and "killing its process tree" in m for m in msgs), msgs
    assert not proc.is_alive() and dt >= 0.6
    assert job.status == JobStatus.CANCELLED and done["ok"] is False


def test_the_cancel_bound_is_derived_from_the_hosted_childs_grace_and_wait():
    """The bound is the inline child's cancel grace (30 s) plus its host's wait for it to go
    (proc.wait(timeout=10)) — read from workers.base, not chosen here — plus the manager's
    own detection latency (two of its polls), so a host exiting at the last instant of its
    cascade is never logged as killed."""
    import inspect
    from workers.base import run_stage_inline
    grace = inspect.signature(run_stage_inline).parameters["cancel_grace_s"].default
    src = inspect.getsource(run_stage_inline)
    assert "proc.wait(timeout=10)" in src
    assert PM.CANCEL_ACK_BOUND_S == grace + 10.0 + 2 * PM.CANCEL_POLL_S
    assert PM.CANCEL_ACK_BOUND_S > grace + 10.0, "a margin, never zero"


# ── the lease file is never half-written, never a silent {} ──────────────────────────

def test_the_lease_is_linked_into_place_complete_and_a_broken_one_is_removed_declared(lease, caplog):
    # taken: no temp sibling survives, the document is complete JSON
    doc = svc.acquire_engine_lease("vlm_worker", stage="VLM stage", token="j")
    assert json.loads(lease.read_text()) == doc
    assert not list(lease.parent.glob(f"{lease.name}.*.tmp"))
    assert svc.release_engine_lease("vlm_worker", "j") is True
    # the failure class of 2026-10-08: a crash between the exclusive create and the dump
    # left an EMPTY file that blocked every acquire and named no holder
    lease.write_text("")
    with caplog.at_level(logging.WARNING, logger="semantic.service"):
        assert svc.engine_lease_holder() is None
    assert not lease.exists(), "removed: it names no holder"
    assert any("DECLARED" in r.getMessage() and str(lease) in r.getMessage() and "not JSON" in r.getMessage()
               for r in caplog.records)
    # ... and the next acquire simply succeeds
    lease.write_text("[1, 2]")
    assert svc.acquire_engine_lease("vlm_worker", stage="VLM stage")["owner"] == "vlm_worker"
    assert svc.release_engine_lease("vlm_worker") is True
    # a lease that EXISTS but cannot be read is a failure naming it — never 'no lease'
    lease.mkdir()
    with pytest.raises(svc.EngineBusy, match=str(lease)):
        svc.engine_lease_holder()
    lease.rmdir()


def test_the_busy_reason_is_worded_by_the_holder(lease):
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="s", token="s")
    assert "that job's stage ends" in svc.engine_available_to(owner="chat")[1]
    svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "s")
    svc.acquire_server_gpu_permit("ShapeR meshing", session="s", token="shape:s#1")
    assert "until that task ends" in svc.engine_available_to(owner="chat")[1]
    svc.release_server_gpu_permit("shape:s#1")
    svc.acquire_server_gpu_permit("Segmentation Manager (SAM3)", session="s", token=svc.SAM3_MANAGER_TOKEN)
    assert "Segmentation Manager closes" in svc.engine_available_to(owner="chat")[1]
    svc.release_server_gpu_permit(svc.SAM3_MANAGER_TOKEN)


def test_a_holder_gone_in_the_window_is_retried_never_a_refusal(tmp_path, lease, monkeypatch):
    """_take_job_lease: the acquire is refused, the holder is looked at — and it has just
    released (a meshing batch ends any moment): the job must start, not fail 'Refused'."""
    s = _session(tmp_path)
    pm = PipelineManager()
    monkeypatch.setattr(PM, "LEASE_POLL_S", 0.01)
    real = pm._acquire_job_lease
    refusals = {"n": 0}

    def _flaky(job):
        if refusals["n"] < 3:
            refusals["n"] += 1
            raise svc.EngineBusy("the lease could not be taken")      # and no holder to look at
        return real(job)

    monkeypatch.setattr(pm, "_acquire_job_lease", _flaky)
    job = _job(s, [StageId.CLOUDCOMPY])

    async def _body():
        await pm._take_job_lease(job, None)

    asyncio.run(_body())
    assert refusals["n"] == 3 and svc.engine_lease_holder()["owner"] == svc.JOB_PERMIT_OWNER
    assert job.stages[0].message == "", "never a refusal"
    pm._release_job_lease(job)


def test_the_wait_message_says_how_the_managers_permit_is_released(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    pm = PipelineManager()
    monkeypatch.setattr(PM, "LEASE_POLL_S", 0.01)
    svc.acquire_server_gpu_permit("Segmentation Manager (SAM3)", session="x", token=svc.SAM3_MANAGER_TOKEN)
    job = _job(s, [StageId.CLOUDCOMPY])
    seen = []

    async def _progress(sid, d):
        seen.append(d["stages"][0]["message"])

    async def _body():
        t = asyncio.create_task(pm._take_job_lease(job, _progress))
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.01)
        svc.release_server_gpu_permit(svc.SAM3_MANAGER_TOKEN)
        await t

    asyncio.run(_body())
    assert seen and "Segmentation Manager closes or a session is loaded" in seen[0]
    pm._release_job_lease(job)


# ── 162: the command's prompts are the job's inputs ─────────────────────────────────

def test_the_commands_prompts_become_the_jobs_inputs_normalised_and_the_disk_is_the_fallback(tmp_path, caplog):
    from segmentation.autoprompt import autosegment as AS
    s = _session(tmp_path)
    out = s / "output"
    AS.save_vlm_prompt(out, "saved on disk")
    AS.set_sam3_prompts(out, ["floor", "wall"])
    sam3_only = select_stages([PipelineStage(id=StageId.SAM3, enabled=True),
                               PipelineStage(id=StageId.CERTIFY, enabled=True)])
    both = select_stages([PipelineStage(id=StageId.VLM, enabled=True),
                          PipelineStage(id=StageId.SAM3, enabled=True)])
    # the command's list, normalised as set_sam3_prompts stores it
    with caplog.at_level(logging.INFO, logger="pipeline_manager"):
        got = PM.snapshot_job_inputs(str(s), sam3_only, False,
                                     command={"sam3_prompts": [" steel  door;", "Door", "", "floor", "FLOOR "]})
    assert got == {"sam3_prompts": ["steel door", "Door", "floor"]}
    assert any("COMMAND's SAM3 prompt list" in r.getMessage() for r in caplog.records)
    # the command's VLM prompt — an override, or the shipped prompt / empty = no override
    assert PM.snapshot_job_inputs(str(s), both, False, command={"vlm_prompt": " Name it all. "}) == {"vlm_prompt": "Name it all."}
    assert PM.snapshot_job_inputs(str(s), both, False, command={"vlm_prompt": ""}) == {"vlm_prompt": None}
    assert PM.snapshot_job_inputs(str(s), both, False, command={"vlm_prompt": AS.default_vlm_prompt()}) == {"vlm_prompt": None}
    # the command's SAM3 list is not an input of a job whose VLM pass makes the list
    assert PM.snapshot_job_inputs(str(s), both, False, command={"sam3_prompts": ["x"]}) == {"vlm_prompt": "saved on disk"}
    # no command prompts: the scan's files, as before
    assert PM.snapshot_job_inputs(str(s), sam3_only, False, command=None) == {"sam3_prompts": ["floor", "wall"]}
    assert PM.snapshot_job_inputs(str(s), both, False) == {"vlm_prompt": "saved on disk"}
    # a wipe carries nothing, the command's prompts included (the user's rule, point 97)
    assert PM.snapshot_job_inputs(str(s), both, True, command={"vlm_prompt": "x"}) == {}
    # the wrong type fails the order, never a default
    with pytest.raises(ValueError, match="list of strings"):
        PM.snapshot_job_inputs(str(s), sam3_only, False, command={"sam3_prompts": "floor;wall"})
    with pytest.raises(ValueError, match="must be a string"):
        PM.snapshot_job_inputs(str(s), both, False, command={"vlm_prompt": 3})
    assert PM.clean_sam3_prompts(["a ", "A", " b;"]) == ["a", "b"]


def test_start_pipeline_takes_the_commands_prompts_into_the_job(tmp_path, lease, monkeypatch):
    from segmentation.autoprompt import autosegment as AS
    s = _session(tmp_path)
    AS.set_sam3_prompts(s / "output", ["on", "disk"])
    pm = PipelineManager()
    launched = []
    monkeypatch.setattr(pm, "_launch", lambda pending: launched.append(pending))
    import project_paths

    class _Ctx:
        session_dir = s

    monkeypatch.setattr(project_paths, "resolve_session", lambda server_dir, sid: _Ctx())
    stages = select_stages([PipelineStage(id=StageId.SAM3, enabled=True)])

    async def _go():
        return await pm.start_pipeline(s.name, stages, _base_config(), kind="autosegment",
                                       force=True, inputs={"sam3_prompts": ["from", "the command"]})

    job = asyncio.run(_go())
    assert job.inputs == {"sam3_prompts": ["from", "the command"]}
    assert stages[0].config["sam3_prompts"] == ["from", "the command"], "frozen under _stages"
    assert launched and launched[0]["job"] is job
    # without them: the snapshot of the scan, as today
    pm2 = PipelineManager()
    monkeypatch.setattr(pm2, "_launch", lambda pending: None)
    stages2 = select_stages([PipelineStage(id=StageId.SAM3, enabled=True)])

    async def _go2():
        return await pm2.start_pipeline(s.name, stages2, _base_config(), kind="autosegment", force=True)

    assert asyncio.run(_go2()).inputs == {"sam3_prompts": ["on", "disk"]}


def test_the_ui_sends_the_prompts_inside_the_order():
    ui = UI_APP.read_text()
    i = ui.index("const runAutosegment = useCallback(")
    body = ui[i:ui.index("const handlePipelineCancel", i)]
    assert "vlm_prompt: vlmPrompt" in body and "autosegment.sam3_prompts = sam3List" in body
    assert "autosegment: { stages }" not in body, "the order no longer carries the stages alone"
    assert "/api/autosegment/" in body, "the save for the session's record stays"
    # a save refused as BUSY (a job of the scan is active with other prompts) never stops the
    # order — the order carries the prompts and the job writes them when it runs
    assert "r.status === 409" in body and "deferred.push(scan)" in body
    assert body.index("deferred.push(scan)") < body.index("fetch('/api/pipeline/run'")


def test_the_segmentation_manager_reads_the_start_session_answer():
    """A 409 from start_session (a job holds the card) is shown with its reason — the poll
    no longer spins on 'loading model' against an init status that never changes."""
    ui = (UI_APP.parent / "components" / "InteractiveSegmentation.tsx").read_text()
    i = ui.index("/api/segmentation/start_session/")
    body = ui[i - 400:i + 1200]
    assert "const startRes = await fetch(" in body and "if (!startRes.ok)" in body
    assert "d.detail?.error" in body and "throw new Error(" in body
    assert ".catch(() => { })" not in body, "no longer fire-and-forget"


# ── 112 / 151: the server does not start on a broken configuration ───────────────────

def test_config_py_raises_on_a_broken_missing_or_non_mapping_configuration(tmp_path, monkeypatch):
    import config as C
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(C, "CONFIG_PATH", p)
    with pytest.raises(C.ConfigError, match="does not exist"):
        C.load_config()
    p.write_text("a: [unclosed\n")
    with pytest.raises(C.ConfigError, match="not readable YAML"):
        C.load_config()
    p.write_text("- a\n- list\n")
    with pytest.raises(C.ConfigError, match="not a mapping"):
        C.load_config()
    p.write_text("a: 1\n")
    assert C.load_config() == {"a": 1}
    src = (SERVER_DIR / "config.py").read_text()
    assert "return {}" not in src, "a broken configuration never becomes {}"


def test_every_stage_worker_receives_the_jobs_configuration_sam3_included(tmp_path, monkeypatch):
    """The manager passes ONE dict — the job's — to every stage process (point 112)."""
    import inspect
    src = inspect.getsource(PipelineManager._run_stage)
    assert "merged_config = {**config, **stage_state.stage.config" in src
    assert "args=(worker_conn, session_dir, merged_config)" in src
    assert PM.STAGE_REGISTRY[StageId.SAM3]["module"] == "workers.sam3_worker"


# ── 156 / 154 / 111(b): the server's text ─────────────────────────────────────────────

def test_the_server_has_no_internvl3_fallback_and_every_gpu_task_takes_the_permit():
    src = (SERVER_DIR / "main.py").read_text()
    assert "scene_analyzer" not in src and "analyze_scene" not in src, "the InternVL3 path is gone (point 156)"
    assert "floor;wall;ceiling;door;window;furniture;object" not in src, "no generic list stands in"
    # the prompt resolution runs on the task's own engine and raises
    r = src.index("def _resolve_segmentation_prompt(")
    rb = src[r:r + 4000]
    assert "job_engine(" in rb and "raise RuntimeError" in rb and "service=identity" in rb
    # every GPU task of the server takes the permit (point 154)
    for endpoint, needle in (('@app.post("/api/segmentation/shape/export")', "_take_gpu_permit(\"ShapeR meshing\""),
                             ('@app.post("/api/segmentation/tsdf/export")', "_take_gpu_permit(\"per-object meshing\""),
                             ('@app.post("/api/segmentation/tsdf/scene_export")', "_take_gpu_permit(\"scene TSDF meshing\""),
                             ('@app.post("/api/segmentation/start_session/{session_id}")', "_take_gpu_permit(\"Segmentation Manager (SAM3)\""),
                             ('@app.post("/api/segmentation/auto/{session_id}")', "_take_gpu_permit(\"auto-segmentation (VLM + SAM3)\"")):
        i = src.index(endpoint)
        assert needle in src[i:i + 6000], endpoint
    for endpoint in ('@app.post("/api/segmentation/add_prompt")', '@app.post("/api/segmentation/propagate")',
                     '@app.post("/api/segmentation/resume/run")'):
        i = src.index(endpoint)
        assert "_require_sam3_manager_permit()" in src[i:i + 3500], endpoint
    i = src.index('elif cmd.get("type") == "set_prompt":')
    assert '_take_gpu_permit("retroactive segmentation"' in src[i:i + 4000] and "_drop_gpu_permit(token" in src[i:i + 6000]
    assert "acquire_server_gpu_permit(" not in src[i:i + 6000], "the retro path takes the permit through the one helper"
    # the Manager's permit goes with its model
    i = src.index('@app.post("/api/segmentation/refresh")')
    assert "_drop_gpu_permit(SAM3_MANAGER_PERMIT" in src[i:i + 2500]
    # the Manager's refusal reaches its poll: the init status says 'error' with the reason
    i = src.index('@app.post("/api/segmentation/start_session/{session_id}")')
    sb = src[i:i + 4000]
    assert sb.index("except HTTPException as e:") < sb.index("async def _background_init():")
    assert '"status": "error"' in sb[:sb.index("async def _background_init():")]
    # every background task whose finally releases the permit is KEPT (never a bare create_task)
    for endpoint in ('@app.post("/api/segmentation/tsdf/export")', '@app.post("/api/segmentation/tsdf/scene_export")',
                     '@app.post("/api/segmentation/shape/export")'):
        i = src.index(endpoint)
        j = src.index("\n@app.", i + 10)
        assert "_keep_bg_task(" in src[i:j] and "asyncio.create_task(_bg" not in src[i:j], endpoint
    # the per-request tokens: one request, one token; the Manager's is the constant
    for kind in ("shape", "tsdf", "tsdf-scene", "auto-segmentation", "retro-segmentation"):
        assert f'_permit_token("{kind}"' in src, kind
    assert 'f"shape:{session_id}"' not in src and 'f"tsdf:{session_id}"' not in src
    # the socket's cancel is scheduled, never awaited in the command loop (point 111's bound)
    i = src.index('elif cmd.get("type") == "cancel_pipeline":')
    cb = src[i:i + 2500]
    assert "await pipeline_manager.cancel_pipeline(session_id)" not in cb
    assert "_keep_bg_task(_cancel_and_report()" in cb and "to stop at a safe point" in cb
    # a cancelled job is reported as cancelled, never as a failure
    j = src.index("async def _on_pipeline_complete(sid, success, restart_chat=True):")
    k = src.index("    # Start pipeline — support sequential multi-scan")
    assert "_job.status == JobStatus.CANCELLED" in src[j:k] and "_tm.cancel(" in src[j:k]
    assert src[j:k].index("_tm.cancel(") < src[j:k].index('_tm.fail(_pipeline_tid, "Pipeline failed")')
    # the intel asks the ONE gate at its entry and hands it to the loop
    assert "_gate = {\"cancelled\": lambda: _intel_engine_busy() is not None}" in src
    # the chat and the intel ask by their own owner
    assert 'engine_available_to(owner="intel")' in src and "owner=CHAT_OWNER" in src
    # the certify select is guarded where the router is mounted (111 b)
    assert "app.include_router(certify_api.router, dependencies=[Depends(_certify_select_guard)])" in src
    # the order carries the prompts (162)
    assert "inputs=_inputs," in src


@pytest.fixture(scope="module")
def main_mod():
    import main
    return main


def test_the_certify_select_guard_refuses_while_a_job_of_the_session_is_active(main_mod):
    from fastapi import HTTPException
    pm = main_mod.pipeline_manager
    sid = "p5-guard-session"
    key = pm.job_key(sid, None, "reconstruct")
    job = PipelineJob(session_id=sid, stages=[], status=JobStatus.RUNNING)
    pm._jobs[key] = job

    def _req(method, path, body):
        async def _json():
            return body
        return types.SimpleNamespace(method=method, url=types.SimpleNamespace(path=path), json=_json)

    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(main_mod._certify_select_guard(_req("POST", "/api/certify/select", {"session_id": sid, "epoch": 1})))
        assert ei.value.status_code == 409 and ei.value.detail["busy"] and sid in ei.value.detail["error"]
        # the reads of the kit pass; another session passes; a queued job refuses too
        asyncio.run(main_mod._certify_select_guard(_req("GET", f"/api/certify/state/{sid}", None)))
        asyncio.run(main_mod._certify_select_guard(_req("POST", "/api/certify/select", {"session_id": "other", "epoch": 1})))
        job.status = JobStatus.QUEUED
        with pytest.raises(HTTPException):
            asyncio.run(main_mod._certify_select_guard(_req("POST", "/api/certify/select", {"session_id": sid, "epoch": 1})))
        job.status = JobStatus.DONE
        asyncio.run(main_mod._certify_select_guard(_req("POST", "/api/certify/select", {"session_id": sid, "epoch": 1})))
    finally:
        pm._jobs.pop(key, None)


def test_a_server_gpu_task_is_refused_naming_the_job_while_the_manager_holds_the_card(main_mod, lease):
    from fastapi import HTTPException
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="pccr@2026-08-31/default",
                             token="pccr@2026-08-31/default")
    try:
        with pytest.raises(HTTPException) as ei:
            main_mod._take_gpu_permit("ShapeR meshing", "pccr", "shape:pccr")
        assert ei.value.status_code == 409
        assert "pipeline job pccr@2026-08-31/default" in ei.value.detail["error"]
        assert ei.value.detail["holder"]["owner"] == "pipeline"
        with pytest.raises(HTTPException):
            main_mod._require_sam3_manager_permit()
    finally:
        svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "pccr@2026-08-31/default")
    # with the card free: taken, held, re-entrant (the Manager — a Manager opened again takes
    # over the one SAM3 model), released
    assert main_mod.SAM3_MANAGER_PERMIT == svc.SAM3_MANAGER_TOKEN
    main_mod._take_gpu_permit("Segmentation Manager (SAM3)", "pccr", main_mod.SAM3_MANAGER_PERMIT)
    assert main_mod._gpu_permit_held(main_mod.SAM3_MANAGER_PERMIT)
    main_mod._require_sam3_manager_permit()
    assert main_mod._gpu_permit_state()["stage"] == "Segmentation Manager (SAM3)"
    main_mod._take_gpu_permit("Segmentation Manager (SAM3)", "pccr", main_mod.SAM3_MANAGER_PERMIT)
    with pytest.raises(HTTPException) as ei:
        main_mod._take_gpu_permit("ShapeR meshing", "pccr", main_mod._permit_token("shape", "pccr"))
    assert "Segmentation Manager (SAM3)" in ei.value.detail["error"]
    assert main_mod._drop_gpu_permit(main_mod.SAM3_MANAGER_PERMIT) is True
    assert main_mod._gpu_permit_state() is None and not main_mod._gpu_permit_held(main_mod.SAM3_MANAGER_PERMIT)


def test_a_second_request_of_the_same_kind_is_refused_while_the_first_is_on_the_card(main_mod, lease):
    """One token per REQUEST: a second ShapeR export on the same session while the first
    batch runs is refused naming it — never re-entered, so the first's release is its own."""
    from fastapi import HTTPException
    t1 = main_mod._permit_token("shape", "pccr")
    t2 = main_mod._permit_token("shape", "pccr")
    assert t1 != t2 and t1.startswith("shape:pccr#")
    main_mod._take_gpu_permit("ShapeR meshing", "pccr", t1)
    with pytest.raises(HTTPException) as ei:
        main_mod._take_gpu_permit("ShapeR meshing", "pccr", t2)
    assert ei.value.status_code == 409 and "ShapeR meshing (session pccr) holds the GPU" in ei.value.detail["error"]
    assert main_mod._drop_gpu_permit(t2) is False, "the second request releases nothing"
    assert main_mod._gpu_permit_held(t1)
    assert main_mod._drop_gpu_permit(t1) is True and main_mod._gpu_permit_state() is None


def test_a_running_job_refuses_every_server_gpu_task_lease_or_no_lease(main_mod, lease):
    """The engine stages hand the manager's lease to their worker, which holds it only around
    its own vLLM — for the rest of the certification NOBODY holds the lease: a RUNNING job
    refuses first, whatever the lease file says."""
    from fastapi import HTTPException
    pm = main_mod.pipeline_manager
    sid = "p5-running-session"
    key = pm.job_key(sid, None, "autosegment")
    pm._jobs[key] = PipelineJob(session_id=sid, stages=[], status=JobStatus.RUNNING)
    try:
        assert svc.engine_lease_holder() is None, "no lease at all"
        for task in ("ShapeR meshing", "Segmentation Manager (SAM3)", "retroactive segmentation"):
            with pytest.raises(HTTPException) as ei:
                main_mod._take_gpu_permit(task, "other", main_mod._permit_token("x", "other"))
            assert ei.value.status_code == 409 and ei.value.detail["busy"]
            assert f"the pipeline job {key} holds the GPU" in ei.value.detail["error"]
            assert ei.value.detail["holder"]["owner"] == "pipeline"
        assert svc.engine_lease_holder() is None, "nothing was taken"
        assert main_mod._intel_engine_busy() == "a pipeline job is running or queued"
        # a QUEUED job does not refuse: it waits for the server task by design
        pm._jobs[key].status = JobStatus.QUEUED
        tok = main_mod._permit_token("shape", "other")
        main_mod._take_gpu_permit("ShapeR meshing", "other", tok)
        assert main_mod._drop_gpu_permit(tok) is True
    finally:
        pm._jobs.pop(key, None)
    assert main_mod._intel_engine_busy() is None
    svc.acquire_engine_lease(svc.JOB_PERMIT_OWNER, stage="job", session="s", token="s")
    assert "engine busy" in main_mod._intel_engine_busy()
    svc.release_engine_lease(svc.JOB_PERMIT_OWNER, "s")


def test_kept_background_tasks_are_dropped_when_they_end_and_a_death_is_printed(main_mod, capsys):
    async def _ok():
        return 1

    async def _dies():
        raise RuntimeError("boom")

    async def _body():
        t1 = main_mod._keep_bg_task(_ok(), "T")
        t2 = main_mod._keep_bg_task(_dies(), "T")
        assert t1 in main_mod._bg_tasks and t2 in main_mod._bg_tasks
        await asyncio.sleep(0.05)
        assert t1 not in main_mod._bg_tasks and t2 not in main_mod._bg_tasks

    asyncio.run(_body())
    assert "background task died before logging: RuntimeError('boom')" in capsys.readouterr().out


def test_the_task_manager_tells_a_cancel_from_a_failure(capsys):
    from task_manager import task_manager
    tid = task_manager.start("s", "pipeline", "Running Pipeline")
    task_manager.cancel(tid, "the user cancelled the pipeline")
    out = capsys.readouterr().out
    assert "⏹ Cancelled pipeline" in out and "❌" not in out
    assert not any(t["task_id"] == tid for t in task_manager.get_active())
