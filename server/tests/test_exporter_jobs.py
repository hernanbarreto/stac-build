"""AN EXPORT IS A JOB (USER 2026-11-19: "cerre la ventana de exportacion y no figura en jobs la tarea de
exportar, que paso?, se corto? … deberia figurar en jobs, y poder cancelarse o dejarla continuar"): a
running export is a task of the task manager (the Jobs list and the count on the console icon read it
there), it ends there as finished, failed or cancelled, and a cancelled one leaves nothing half-written —
the previous whole export stays."""
from __future__ import annotations

import asyncio
import json
import sys

import pytest
from fastapi import HTTPException

import exporter_api as X
from exporter import package as P
from task_manager import task_manager


def _exports(session: str):
    return [t for t in task_manager.get_active(session) if t["task_type"] == "export"]


def _child(code: str):
    return [sys.executable, "-c", code]


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "scratch_of", lambda target: tmp_path / "scratch" / target.stem)
    monkeypatch.setattr(X, "_RUNS", {})
    monkeypatch.setattr(X, "_PROCS", {})
    return tmp_path


def test_a_running_export_is_a_task_and_ends_as_one(scratch):
    target = scratch / "exports" / "job_a_stac_export.zip"
    code = ("import time, json; print('PROGRESS 40 writing the cloud', flush=True); time.sleep(0.6); "
            "print('RESULT ' + json.dumps({'file': 'job_a_stac_export.zip', 'bytes': 1}), flush=True)")

    async def go():
        state = {"status": "running", "progress": 0, "message": "starting"}
        X._launch("job_a|*", _child(code), state, "job_a", "Export of job_a", lambda: target)
        assert _exports("job_a") and _exports("job_a")[0]["label"] == "Export of job_a"     # listed at once
        seen = None
        for _ in range(200):
            await asyncio.sleep(0.05)
            live = _exports("job_a")
            if live and live[0]["pct"] == 40:
                seen = live[0]
            if state["status"] != "running":
                break
        return state, seen

    state, seen = asyncio.run(go())
    assert state["status"] == "done" and state["report"]["file"] == "job_a_stac_export.zip" and state["progress"] == 100
    assert seen is not None and seen["detail"] == "writing the cloud" and seen["session_id"] == "job_a"
    assert not _exports("job_a")                                                            # ended: off the list
    assert any(t["task_type"] == "export" for t in task_manager.get_active()) is False


def test_a_cancelled_export_stops_and_leaves_nothing_half_written(scratch):
    target = scratch / "exports" / "job_b_stac_export.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"the previous whole export")
    work = P.scratch_of(target)
    # a child in the middle of its work: a ZIP half written, files on the scratch disk, minutes to go
    code = (f"import time, pathlib; pathlib.Path({str(P.partial_of(target))!r}).write_bytes(b'half'); "
            f"pathlib.Path({str(work)!r}).mkdir(parents=True); pathlib.Path({str(work / 'cloud.e57')!r}).write_bytes(b'x'); "
            "print('PROGRESS 10 writing the cloud as E57', flush=True); time.sleep(120)")

    async def go():
        state = {"status": "running", "progress": 0, "message": "starting"}
        X._launch("job_b|*", _child(code), state, "job_b", "Export of job_b", lambda: target)
        for _ in range(400):
            await asyncio.sleep(0.05)
            if state["progress"] == 10:
                break
        assert state["progress"] == 10 and _exports("job_b") and P.partial_of(target).exists() and work.exists()
        ended = await X._cancel("job_b|*")
        with pytest.raises(HTTPException) as e:                       # nothing is running any more
            await X._cancel("job_b|*")
        return ended, e.value.status_code

    state, again = asyncio.run(go())
    assert state["status"] == "cancelled" and again == 409
    assert not P.partial_of(target).exists() and not work.exists()
    assert target.read_bytes() == b"the previous whole export"
    assert not _exports("job_b")


def test_an_export_that_fails_says_why_and_is_off_the_list(scratch):
    target = scratch / "exports" / "job_c_stac_export.zip"

    async def go():
        state = {"status": "running", "progress": 0, "message": "starting"}
        X._launch("job_c|*", _child("import sys; print('ERROR cannot export 2026-01-01/default ply: no published cloud', flush=True); sys.exit(1)"),
                  state, "job_c", "Export of job_c", lambda: target)
        for _ in range(200):
            await asyncio.sleep(0.05)
            if state["status"] != "running":
                break
        return state

    state = asyncio.run(go())
    assert state["status"] == "failed" and "no published cloud" in state["error"] and not _exports("job_c")


def test_a_zip_cut_short_is_removed(tmp_path):
    """What the child does when it is asked to stop while packing (SIGTERM → SystemExit): the ZIP being
    written goes, the files it was reading stay."""
    source = tmp_path / "mesh.glb"
    source.write_bytes(b"x" * 100)
    target = tmp_path / "exports" / "s_stac_export.zip"
    target.parent.mkdir()

    def stop(_frac, _message):
        raise SystemExit(143)

    with pytest.raises(SystemExit):
        P._pack(target, [({"path": source, "arc": "mesh.glb", "key": "glb", "source": True}, "s_export/mesh.glb")],
                [("s_export/README.txt", "hello")], stop)
    assert not P.partial_of(target).exists() and not target.exists() and source.exists()
    assert json.dumps(P.discard_partial(target)) == "[]"                # nothing left to remove
