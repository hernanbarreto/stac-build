"""workers.base.run_stage_inline: one worker hosts another (the reconstruction
stage hosts VLM, SAM3, the merge and the precision core — USER 2026-09-28) and
relays its logs / progress; the child's failure is the host's failure, named."""

from __future__ import annotations

import sys

import pytest

FAKE = '''
from workers.base import run_worker_safe

def _work(pipe, session_dir, config):
    pipe.send_log("hello from the child")
    pipe.send_progress(50, "half way", stage="child")
    if config.get("fail"):
        raise RuntimeError("the child says why")
    pipe.send_progress(100, "done", stage="child")

def run(conn, session_dir, config):
    run_worker_safe(_work, conn, session_dir, config)
'''


class _Pipe:
    def __init__(self):
        self.logs, self.progress = [], []

    def send_log(self, msg, level="info"):
        self.logs.append((level, msg))

    def send_progress(self, pct, msg, stage=""):
        self.progress.append((round(pct, 2), msg, stage))

    def check_cancel(self):
        return False


@pytest.fixture
def fake_module(tmp_path, monkeypatch):
    (tmp_path / "fake_inline_worker.py").write_text(FAKE)
    # the hosted child is a subprocess (python -m workers.inline_child): it finds
    # the fake module through PYTHONPATH
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "fake_inline_worker"


def test_relays_logs_and_rescaled_progress(fake_module, tmp_path):
    from workers.base import run_stage_inline
    pipe = _Pipe()
    run_stage_inline(pipe, fake_module, str(tmp_path), {}, label="child",
                     stage="reconstruction", pct_range=(80.0, 90.0))
    assert ("info", "[child] hello from the child") in pipe.logs
    assert (85.0, "child: half way", "reconstruction") in pipe.progress
    assert (90.0, "child: done", "reconstruction") in pipe.progress


def test_the_childs_failure_is_raised_with_its_reason(fake_module, tmp_path):
    from workers.base import run_stage_inline
    pipe = _Pipe()
    with pytest.raises(RuntimeError, match="child failed: the child says why"):
        run_stage_inline(pipe, fake_module, str(tmp_path), {"fail": True}, label="child")
    assert any(lvl == "error" and "the child says why" in msg for lvl, msg in pipe.logs)
