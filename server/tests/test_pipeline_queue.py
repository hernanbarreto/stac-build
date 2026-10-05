"""Reconstructions QUEUE instead of sharing the card (USER ORDER 2026-09-23).

*"si mando 5 reconstrucciones, se empieza con la primera y el resto quedan en
cola y en la medida que van terminando va iniciando el resto hasta que terminan
todas"*.

Until now the refusal was per SESSION: `start_pipeline` rejected a second
command for the same session and said nothing about a different one, so two
scans launched together both ran and shared one GPU — against this repo's own
operating lesson ("One GPU job at a time; A/B timing measured under contention
is INVALID").

These tests drive the manager's real `start_pipeline`/`cancel_pipeline` with the
orchestration loop replaced by a controllable stub, so what is asserted is the
QUEUEING, not the stages.
"""

import asyncio
import time
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_manager import (JobStatus, PipelineManager,  # noqa: E402
                              build_pipeline_stages)


def _mgr(monkeypatch, gate: dict):
    """A manager whose pipeline body waits on a per-session event."""
    pm = PipelineManager()

    async def _fake_run(job, session_dir, config, on_progress, on_complete, replace, force=False):
        gate.setdefault(job.session_id, asyncio.Event())
        await gate[job.session_id].wait()
        job.status = JobStatus.DONE
        if on_complete:
            await on_complete(job.session_id, True)
        await pm._start_next_queued()

    monkeypatch.setattr(pm, "_run_pipeline", _fake_run)
    monkeypatch.setattr(pm, "_resolve_dir", lambda *a, **k: "/tmp", raising=False)
    return pm


async def _start(pm, sid, monkeypatch):
    # resolve_session touches the disk; the queue does not care where it lands
    import pipeline_manager as pmod
    monkeypatch.setattr(pmod, "resolve_session",
                        lambda *a, **k: type("C", (), {"session_dir": f"/tmp/{sid}"})(),
                        raising=False)
    return await pm.start_pipeline(sid, build_pipeline_stages(), {}, scan_key=None)


def test_five_commands_run_one_at_a_time_in_order(monkeypatch):
    asyncio.run(_body_test_five_commands_run_one_at_a_time_in_order(monkeypatch))


async def _body_test_five_commands_run_one_at_a_time_in_order(monkeypatch):
    gate: dict = {}
    pm = _mgr(monkeypatch, gate)
    ids = [f"scan{i}" for i in range(5)]
    for sid in ids:
        await _start(pm, sid, monkeypatch)
    await asyncio.sleep(0)

    assert pm._jobs["scan0"].status == JobStatus.RUNNING
    for k, sid in enumerate(ids[1:], start=1):
        assert pm._jobs[sid].status == JobStatus.QUEUED
        assert pm._jobs[sid].queue_position == k, sid

    # release them one by one — each release must start exactly the next one
    for k, sid in enumerate(ids):
        gate.setdefault(sid, asyncio.Event()).set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert pm._jobs[sid].status == JobStatus.DONE, sid
        running = [s for s, j in pm._jobs.items() if j.status == JobStatus.RUNNING]
        if k + 1 < len(ids):
            assert running == [ids[k + 1]], (k, running)
        else:
            assert running == [], running
    assert pm._queue == []


def test_the_same_session_twice_is_still_refused(monkeypatch):
    asyncio.run(_body_test_the_same_session_twice_is_still_refused(monkeypatch))


async def _body_test_the_same_session_twice_is_still_refused(monkeypatch):
    """USER 2026-09-05: *one and only one* per session. The queue is for OTHER
    scans; re-commanding the one that is already going stays an error."""
    gate: dict = {}
    pm = _mgr(monkeypatch, gate)
    await _start(pm, "scanA", monkeypatch)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="already"):
        await _start(pm, "scanA", monkeypatch)


def test_cancelling_a_waiting_job_never_touches_the_running_one(monkeypatch):
    asyncio.run(_body_test_cancelling_a_waiting_job_never_touches_the_running_one(monkeypatch))


async def _body_test_cancelling_a_waiting_job_never_touches_the_running_one(monkeypatch):
    gate: dict = {}
    pm = _mgr(monkeypatch, gate)
    for sid in ("a", "b", "c"):
        await _start(pm, sid, monkeypatch)
    await asyncio.sleep(0)

    await pm.cancel_pipeline("b")
    assert pm._jobs["a"].status == JobStatus.RUNNING, "the card must not be disturbed"
    assert pm._jobs["b"].status == JobStatus.CANCELLED
    assert pm._jobs["c"].queue_position == 1, "the rest move up"
    assert [q["job"].session_id for q in pm._queue] == ["c"]

    gate.setdefault("a", asyncio.Event()).set()
    await asyncio.sleep(0); await asyncio.sleep(0)
    assert pm._jobs["c"].status == JobStatus.RUNNING, "the cancelled one is skipped"


def test_the_queue_is_visible_to_the_ui(monkeypatch):
    asyncio.run(_body_test_the_queue_is_visible_to_the_ui(monkeypatch))


async def _body_test_the_queue_is_visible_to_the_ui(monkeypatch):
    gate: dict = {}
    pm = _mgr(monkeypatch, gate)
    for sid in ("x", "y"):
        await _start(pm, sid, monkeypatch)
    await asyncio.sleep(0)
    jobs = pm.get_all_jobs()
    assert jobs["x"]["status"] == "running" and jobs["x"]["queue_position"] == 0
    assert jobs["y"]["status"] == "queued" and jobs["y"]["queue_position"] == 1


# ── every scan of a multi-scan order is its own job (USER 2026-10-05) ──────────

def test_two_scans_of_one_session_are_two_jobs_the_second_queued(monkeypatch):
    asyncio.run(_body_test_two_scans_of_one_session(monkeypatch))


async def _body_test_two_scans_of_one_session(monkeypatch):
    """"hay que meterlo en cola ... se debe mostrar en el UI": the second scan of
    the same project is a queued job of its own — visible, cancellable — not a
    callback waiting for the first; the same scan twice is still refused."""
    from pipeline_manager import JobStatus, PipelineManager, build_pipeline_stages
    gate: dict = {}
    pm = PipelineManager()

    async def _fake_run(job, session_dir, config, on_progress, on_complete, replace, force=False):
        key = pm.job_key(job.session_id, job.scan_key)       # one gate PER JOB
        gate.setdefault(key, asyncio.Event())
        await gate[key].wait()
        job.status = JobStatus.DONE
        job.ended_at = time.time()
        if on_complete:
            await on_complete(job.session_id, True)
        await pm._start_next_queued()

    monkeypatch.setattr(pm, "_run_pipeline", _fake_run)
    monkeypatch.setattr(pm, "_resolve_dir", lambda *a, **k: "/tmp", raising=False)
    a = await pm.start_pipeline("pccr", build_pipeline_stages(), {}, scan_key="2026-08-24/default")
    b = await pm.start_pipeline("pccr", build_pipeline_stages(), {}, scan_key="2026-08-31/default")
    assert a.status == JobStatus.RUNNING and b.status == JobStatus.QUEUED and b.queue_position == 1
    keys = set(pm.get_all_jobs())
    assert keys == {"pccr@2026-08-24/default", "pccr@2026-08-31/default"}
    assert pm.get_all_jobs()["pccr@2026-08-31/default"]["scan_key"] == "2026-08-31/default"
    assert pm.is_active("pccr") and pm.is_active("pccr", "2026-08-31/default")
    assert not pm.is_active("pccr", "2026-09-01/default")
    with pytest.raises(RuntimeError, match="one and only one"):
        await pm.start_pipeline("pccr", build_pipeline_stages(), {}, scan_key="2026-08-31/default")
    assert pm.job_of("pccr") is a                       # the one holding the card
    # the first finishes: its completion handler must still see IT, then the second runs
    gate.setdefault("pccr@2026-08-24/default", asyncio.Event()).set()
    await asyncio.sleep(0.05)
    assert a.status == JobStatus.DONE and a.ended_at > 0
    assert b.status == JobStatus.RUNNING and pm.job_of("pccr") is b
    # cancelling by session takes every job of it that is still alive
    await pm.cancel_pipeline("pccr")
    assert b.status == JobStatus.CANCELLED and a.status == JobStatus.DONE
