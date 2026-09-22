"""USER 2026-09-21, after a full reconstruction: "todo el pipeline debe ser
correcto, coherente y no ejecutarse nada de forma innecesaria — más tiempo,
más errores... nada de parches ni conformarnos con guardas que hacen que no se
rompa".

The pipeline's SAM3 worker was matching 22.7 M points when the viewer opened
the session; the viewer found no cached result and launched a SECOND full
matching. The guard in place was a `threading.Lock`, which coordinates threads
inside ONE process and is blind to a `multiprocessing.spawn` worker. Measured
cost: ~10 minutes of duplicated work, three passes computing 121/113/119
instances while a fourth wrote 125, the viewer showing 119 against a file
saying 125, four Potree octrees over a cloud that never changed.

These tests pin the property that makes all of that impossible: ONE matching
per session at a time, across processes, and a caller that finds one in flight
waits for it instead of starting another.
"""

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.match_lock import (LOCK_NAME, MatchingBusy,  # noqa: E402
                                     matching_lock)


def _hold(d, seconds, started, done):
    """Child PROCESS: take the lock and hold it. Threads would not prove it."""
    with matching_lock(d, log=lambda m: None):
        started.set()
        time.sleep(seconds)
    done.set()


def _try_nowait(d, q, ready=None):
    if ready is not None:
        ready.set()          # spawning a process costs ~1-2 s; say when we
        time.sleep(0.05)     # are actually about to try, not when we started
    try:
        with matching_lock(d, timeout_s=0.1, log=lambda m: None):
            q.put("acquired")
    except MatchingBusy:
        q.put("busy")


def test_the_lock_is_held_across_processes(tmp_path):
    """The whole point: a `threading.Lock` would let this through."""
    ctx = mp.get_context("spawn")
    started, done = ctx.Event(), ctx.Event()
    q, ready = ctx.Queue(), ctx.Event()
    c = ctx.Process(target=_try_nowait, args=(str(tmp_path), q, ready))
    # hold for long enough to outlast spawning the challenger (~1-2 s)
    p = ctx.Process(target=_hold, args=(str(tmp_path), 20.0, started, done))
    p.start()
    try:
        assert started.wait(30), "the child never took the lock"
        c.start()
        assert ready.wait(30), "the challenger never started"
        c.join(30)
        assert q.get(timeout=10) == "busy", \
            "a second PROCESS got in while the first was matching"
    finally:
        p.terminate(); p.join(30)


def test_the_second_caller_waits_and_is_told_it_waited(tmp_path):
    """It must not start its own pass: `waited` is how the caller knows to
    read what the other one wrote instead of matching again."""
    ctx = mp.get_context("spawn")
    started, done = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), 1.0, started, done))
    p.start()
    try:
        assert started.wait(30)
        t0 = time.time()
        with matching_lock(tmp_path, log=lambda m: None) as waited:
            assert waited is True, "the caller was not told it had waited"
            assert time.time() - t0 >= 0.4, "it did not actually wait"
    finally:
        p.join(30)


def test_an_uncontended_lock_says_it_did_not_wait(tmp_path):
    with matching_lock(tmp_path, log=lambda m: None) as waited:
        assert waited is False


def test_it_is_released_even_when_the_body_raises(tmp_path):
    with pytest.raises(ValueError):
        with matching_lock(tmp_path, log=lambda m: None):
            raise ValueError("boom")
    with matching_lock(tmp_path, timeout_s=0.1, log=lambda m: None) as waited:
        assert waited is False, "the lock survived a raising body"


def test_the_lock_file_names_its_holder(tmp_path):
    """A stale lock from a killed process must be identifiable, not guessed."""
    with matching_lock(tmp_path, log=lambda m: None):
        txt = (tmp_path / LOCK_NAME).read_text()
    assert f"pid={os.getpid()}" in txt and "since=" in txt


def test_a_busy_wait_reports_who_holds_it(tmp_path):
    ctx = mp.get_context("spawn")
    started, done = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), 20.0, started, done))
    p.start()
    try:
        assert started.wait(30)
        said = []
        with pytest.raises(MatchingBusy, match="pid="):
            with matching_lock(tmp_path, timeout_s=0.2, log=said.append):
                pass
        assert any("waiting for it instead of starting a second pass" in m
                   for m in said), said
    finally:
        p.terminate(); p.join(30)


# ── the callers ─────────────────────────────────────────────────────────

def test_both_matching_paths_take_the_lock():
    """`_match_and_save_result` and `apply_segmentation_to_cloud` are the two
    entry points that match; either one alone leaves the race open."""
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "pipeline.py").read_text()
    body = src[src.index("def _match_and_save_result("):
               src.index("def _match_and_save_result_locked(")]
    assert "matching_lock(" in body, \
        "_match_and_save_result must hold the session lock"
    body = src[src.index("def apply_segmentation_to_cloud("):
               src.index("def _apply_segmentation_slow(")]
    assert "matching_lock(" in body, \
        "apply_segmentation_to_cloud must hold the session lock"


def test_the_dead_threading_lock_is_gone():
    """A guard that looks like a protection and gives none is worse than no
    guard: it is what let this race live."""
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "pipeline.py").read_text()
    assert "_get_matching_lock" not in src
    assert "_matching_locks" not in src
