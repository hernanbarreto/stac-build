"""USER 2026-10-06: when the server dies, the WHOLE stage tree dies with it — the worker,
its children and grandchildren (Omega, DA3, SAM3, the precision steps, PointDiT)."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                   # a zombie is dead for our purpose
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def test_the_stage_tree_dies_when_the_server_dies(tmp_path):
    pids = tmp_path / "pids"
    # "server" → "worker" (setsid + die_with_parent, as run_worker_safe does) → "Omega" grandchild
    worker = textwrap.dedent(f"""
        import os, subprocess, sys, time
        sys.path.insert(0, {str(SERVER)!r})
        os.setsid()
        from workers.base import die_with_parent
        die_with_parent(poll_s=0.2)
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        open({str(pids)!r}, "w").write(f"{{os.getpid()}} {{g.pid}}")
        time.sleep(120)
    """)
    server = subprocess.Popen([sys.executable, "-c",
                               f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {worker!r}]); time.sleep(120)"])
    t0 = time.time()
    while not pids.exists() and time.time() - t0 < 20:
        time.sleep(0.1)
    w_pid, g_pid = (int(x) for x in pids.read_text().split())
    assert _alive(w_pid) and _alive(g_pid)
    server.send_signal(signal.SIGKILL)       # the server dies hard
    server.wait()
    t0 = time.time()
    while (_alive(w_pid) or _alive(g_pid)) and time.time() - t0 < 10:
        time.sleep(0.1)
    assert not _alive(w_pid), "the worker outlived the server"
    assert not _alive(g_pid), "the grandchild (Omega/DA3/SAM3...) outlived the server"


def test_the_server_shutdown_kills_the_pipeline_stage_trees():
    src = (SERVER / "main.py").read_text()
    assert "pipeline_manager.kill_all_stages(\"server shutdown\")" in src
    base = (SERVER / "workers" / "base.py").read_text()
    i_setsid = base.index("os.setsid()")
    i_watch = base.index("die_with_parent()          # the stage tree never outlives the server")
    assert i_setsid < i_watch


def test_vllm_session_dies_with_the_backend(tmp_path):
    """USER 2026-10-06: the VLM too. A fake 'vLLM session' (its own session, a child
    in it) is killed by semantic.watchdog when the 'backend' pid dies."""
    import subprocess as sp
    backend = sp.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    fake_vllm = sp.Popen([sys.executable, "-c",
                          "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); time.sleep(120)"],
                         start_new_session=True)
    time.sleep(0.5)
    dog = sp.Popen([sys.executable, "-c",
                    f"import sys; sys.path.insert(0, {str(SERVER)!r}); from semantic.watchdog import watch; "
                    f"watch({backend.pid}, {fake_vllm.pid}, poll_s=0.2, grace_s=2.0)"])
    time.sleep(0.6)
    assert dog.poll() is None and fake_vllm.poll() is None
    backend.kill(); backend.wait()
    dog.wait(timeout=10)
    fake_vllm.wait(timeout=10)
    try:
        os.killpg(fake_vllm.pid, 0)
        group_left = True
    except OSError:
        group_left = False
    assert not group_left, "a process of the vLLM session outlived the backend"


def test_the_launcher_starts_the_watchdog_and_the_shutdown_stops_vllm():
    svc = (SERVER / "semantic" / "service.py").read_text()
    assert '"-m", "semantic.watchdog"' in svc and "STAC_BACKEND_PID" in svc
    src = (SERVER / "main.py").read_text()
    assert 'os.environ["STAC_BACKEND_PID"] = str(os.getpid())' in src
    assert '_stop_sem(None, stage="server shutdown")' in src
