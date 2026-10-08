"""The backend is never a GPU co-tenant of a reconstruction step, and the vLLM stop hands over a
card that is really free (2026-10-07, pccr's first wave-1 validation run).

Measured that day: nvidia-smi listed the backend itself (uvicorn main:app, env da3) holding
416 MiB of the A100 — a CUDA context created by semantic/service.py when the backend kicked vLLM
at its own boot (torch.cuda.mem_get_info / ipc_collect to "release cached VRAM" in a process that
had never touched the card). repro.require_exclusive_gpu refuses any compute process on the card,
so every GPU step of the run would have failed on the backend's own pid.

CPU-only: torch, nvidia-smi and the processes are faked; no CUDA call, no signal to any process
but the ones a test starts itself."""

import builtins
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import repro
import semantic.service as svc
import workers.base as wb

UUID = "GPU-9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5"
NAME = "NVIDIA A100 80GB PCIe"


class FakeCuda:
    """torch.cuda as far as these paths use it. Every call that would create a CUDA context on a
    real card (mem_get_info, ipc_collect, synchronize, get_device_properties) marks it created."""

    def __init__(self, initialised: bool, free_bytes=(10 * 2 ** 30, 30 * 2 ** 30)):
        self.initialised = initialised
        self.context_created = False
        self.calls = []
        self._free = list(free_bytes)

    def _touch(self, name):
        self.calls.append(name)
        if not self.initialised:
            self.context_created = True
        self.initialised = True

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialised

    def mem_get_info(self):
        self._touch("mem_get_info")
        return (self._free.pop(0) if self._free else 0, 80 * 2 ** 30)

    def ipc_collect(self):
        self._touch("ipc_collect")

    def synchronize(self):
        self._touch("synchronize")

    def get_device_properties(self, i):
        self._touch("get_device_properties")
        return types.SimpleNamespace(total_memory=85094825984)

    def empty_cache(self):
        self.calls.append("empty_cache")        # torch's own empty_cache is a no-op uninitialised


def _fake_torch(cuda):
    return types.SimpleNamespace(cuda=cuda, bfloat16="bf16", float32="f32")


# ── 1. the semantic service never creates a context to measure ─────────────────────────────

def test_release_cached_vram_creates_no_context_in_a_process_that_never_used_cuda(monkeypatch):
    cuda = FakeCuda(initialised=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
    said = []
    assert svc.release_cached_vram(said.append) is None
    assert not cuda.context_created and cuda.calls == [], cuda.calls
    assert said == []


def test_release_cached_vram_does_not_even_import_torch(monkeypatch):
    """A process that never imported torch never initialised CUDA through it — importing it just
    to ask would cost seconds and prove nothing."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__
    asked = []

    def spy(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            asked.append(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", spy)
    assert svc.release_cached_vram(lambda m: None) is None
    assert asked == []


def test_release_cached_vram_still_releases_for_a_process_that_ran_gpu_work(monkeypatch):
    """The 2026-09-19 deadlock fix stays: a process whose allocator holds VRAM gives it back."""
    cuda = FakeCuda(initialised=True)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
    said = []
    assert svc.release_cached_vram(said.append) == pytest.approx(20.0)
    assert cuda.calls == ["mem_get_info", "empty_cache", "ipc_collect", "mem_get_info"]
    assert any("released 20.0 GB of cached VRAM" in m for m in said)


def test_the_backend_boot_kick_launches_vllm_without_touching_cuda(monkeypatch, tmp_path):
    """The exact path of 2026-10-07: lifespan → _semantic_reload_if_idle → ensure_service
    (timeout 0) with the service down. vLLM is launched; the backend creates no context."""
    cuda = FakeCuda(initialised=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
    monkeypatch.setattr(svc, "is_alive", lambda config=None, timeout_s=3.0: False)
    launcher = tmp_path / "serve_semantic.sh"
    launcher.write_text("#!/bin/bash\n")
    monkeypatch.setattr(svc, "_LAUNCHER", launcher)
    launched = []

    def fake_popen(argv, **kw):
        launched.append(list(argv))
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setattr(svc, "subprocess",
                        types.SimpleNamespace(Popen=fake_popen, DEVNULL=subprocess.DEVNULL))
    said = []
    assert svc.ensure_service({}, log=said.append, timeout_s=0.0) is False
    assert not cuda.context_created and cuda.calls == [], cuda.calls
    assert launched[0] == ["bash", str(launcher)]
    assert launched[1][1:4] == ["-m", "semantic.watchdog", str(int(os.environ.get(
        "STAC_BACKEND_PID") or os.getpid()))]
    assert any("not waiting for it" in m for m in said)


def test_scene_analyzer_sizes_the_model_without_a_context(monkeypatch):
    """The InternVL3 fallback runs IN the backend; its card-size probe goes through
    repro.card_identity (a subprocess when this process has no context), never through
    torch.cuda.get_device_properties in-process."""
    import segmentation.scene_analyzer as sa
    cuda = FakeCuda(initialised=False)
    monkeypatch.setattr(sa, "torch", _fake_torch(cuda))
    asked = []

    def ident(device=0):
        asked.append(device)
        return {"total_memory_bytes": 85094825984}

    monkeypatch.setattr(repro, "card_identity", ident)
    assert sa._select_model_and_dtype() == (sa.DEFAULT_MODEL_ID, "bf16", "cuda")
    assert asked == [0]
    assert not cuda.context_created and "get_device_properties" not in cuda.calls
    # a card under 16 GiB still selects the small model (the rule is unchanged)
    monkeypatch.setattr(repro, "card_identity", lambda device=0: {"total_memory_bytes": 12 * 2 ** 30})
    assert sa._select_model_and_dtype()[0] == "OpenGVLab/InternVL3-2B"


def test_a_worker_that_never_used_cuda_creates_no_context_on_its_way_out(monkeypatch):
    monkeypatch.setenv("STAC_INLINE_CHILD", "1")             # no setsid of the test process
    monkeypatch.setattr(wb, "die_with_parent", lambda poll_s=2.0: None)
    for initialised, expected in ((False, []), (True, ["empty_cache", "synchronize"])):
        cuda = FakeCuda(initialised=initialised)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda))
        here, there = multiprocessing.Pipe()
        wb.run_worker_safe(lambda pipe: pipe.send_log("worked"), there)
        msgs = []
        while here.poll():
            try:
                msgs.append(here.recv())
            except EOFError:                 # the worker closed its end, as it must
                break
        assert msgs[-1]["type"] == "done" and msgs[-1]["success"] is True
        assert cuda.calls == expected and cuda.context_created is False
        here.close()


# ── 2. the vLLM stop waits for the processes AND for the card ──────────────────────────────

class FakeSmi:
    """nvidia-smi as repro._nvidia_smi runs it: each card query moves to the next state (the
    last one stays); the process query answers for the current state."""

    def __init__(self, states):
        self.states = states
        self.card_reads = 0
        self.cur = states[0]

    def __call__(self, args):
        q = " ".join(args)
        if "--query-gpu=" in q:
            self.cur = self.states[min(self.card_reads, len(self.states) - 1)]
            self.card_reads += 1
            return f"0, {UUID}, {NAME}, {self.cur['used']}, 81920, 595.91.07\n"
        if "--query-compute-apps=" in q:
            return "".join(f"{pid}, {name}, {mib}, {UUID}\n" for pid, name, mib in self.cur["procs"])
        raise AssertionError(f"unexpected nvidia-smi call {args}")


VLLM_LAUNCHER, VLLM_API, VLLM_ENGINE = 2380, 2387, 2390
BEFORE = {"used": 40960, "procs": [(VLLM_API, "python3.11", 512), (VLLM_ENGINE, "VLLM::EngineCore", 40448)]}
TEARDOWN = {"used": 40448, "procs": [(VLLM_ENGINE, "VLLM::EngineCore", 40448)]}   # gone from ps, not from the card
DRAINING = {"used": 12000, "procs": []}
FREE = {"used": 0, "procs": []}


@pytest.fixture
def fake_chain(monkeypatch):
    """A vLLM chain that leaves the process table at the first signal, a no-op sleep, a fixed
    free-memory line; the test plugs its nvidia-smi."""
    st = {"alive": [VLLM_LAUNCHER, VLLM_API, VLLM_ENGINE], "signals": [], "sleeps": []}
    monkeypatch.setattr(wb, "vllm_pids", lambda: list(st["alive"]))

    def kill(sig):
        pids, st["alive"] = list(st["alive"]), []
        st["signals"].append(sig)
        return pids

    monkeypatch.setattr(wb, "kill_vllm_pids", kill)
    monkeypatch.setattr(wb, "time", types.SimpleNamespace(sleep=st["sleeps"].append, time=time.time))
    monkeypatch.setattr(wb, "gpu_free_gb", lambda: 80.0)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)   # repro reads every card
    return st


def test_the_stop_waits_until_the_card_let_go_before_the_exclusive_check(fake_chain, monkeypatch):
    smi = FakeSmi([BEFORE, TEARDOWN, DRAINING, FREE])
    monkeypatch.setattr(repro, "_nvidia_smi", smi)
    said = []
    rec = wb.stop_semantic_service(stage="intake focal probe (DA3)", log=said.append)
    assert rec["awaited"] is True and rec["released"] is True
    assert rec["pids"] == [VLLM_LAUNCHER, VLLM_API, VLLM_ENGINE]
    assert rec["held_mib"] == {UUID: 512 + 40448}
    assert rec["polls"] == 2 and rec["used_mib_after"] == {UUID: 0}
    # the card was read until it was free — and only then did the stop return
    assert smi.card_reads == 4
    assert any("vLLM's VRAM released" in m for m in said)
    assert said[-1] == "[gpu] vLLM stopped — 80 GB VRAM free"
    # what the GPU step runs next passes: the card it reads is the free one
    assert repro.require_exclusive_gpu(log=said.append)["compute_processes"] == 0


def test_the_verified_stop_returns_its_record_unchanged_once_released(fake_chain, monkeypatch):
    monkeypatch.setattr(repro, "_nvidia_smi", FakeSmi([BEFORE, TEARDOWN, FREE]))
    assert wb.stop_semantic_service_verified(stage="precision f5", log=lambda m: None) == {
        "service_stopped": True, "check": "pgrep -f 'vllm serve'", "remaining_pids": [],
        "free_gb": 80.0}


def test_the_verified_stop_fails_when_the_card_never_lets_go(fake_chain, monkeypatch):
    monkeypatch.setattr(repro, "_nvidia_smi", FakeSmi([BEFORE, TEARDOWN]))
    said = []
    with pytest.raises(RuntimeError, match=r"did not let go of the stopped vLLM.*still lists the "
                                           r"stopped process\(es\) \[2390\]"):
        wb.stop_semantic_service_verified(stage="intake focal probe (DA3)", log=said.append)
    # bounded: the budget the processes had to die (6 attempts x 10 looks x 2 s)
    release_sleeps = [s for s in fake_chain["sleeps"]][1:]   # the first is the kill loop's look
    assert len(release_sleeps) == wb._RELEASE_POLLS == 60 and set(release_sleeps) == {2.0}
    assert any("not released" in m for m in said)


def test_memory_that_another_tenant_holds_is_left_to_the_exclusive_check(fake_chain, monkeypatch):
    """The stopped pids are off the card but the memory stays above 'before less what vLLM
    held' (someone else allocated): the stop declares it and does not fail — the step's own
    require_exclusive_gpu decides and names the tenant."""
    other = {"used": 30000, "procs": [(31337, "python", 30000)]}
    monkeypatch.setattr(repro, "_nvidia_smi", FakeSmi([BEFORE, other]))
    said = []
    out = wb.stop_semantic_service_verified(stage="precision f2", log=said.append)
    assert out["remaining_pids"] == []
    assert any("still uses" in m and "MiB" in m for m in said)
    with pytest.raises(repro.ReproError, match="pid 31337"):
        repro.require_exclusive_gpu(log=said.append)


def test_unattributable_pids_are_declared_not_awaited(fake_chain, monkeypatch):
    """A driver that reports pids of another namespace: none of the stopped pids is listed and
    the listed one is no process here — the release cannot be attributed; declared, no wait."""
    foreign = int(Path("/proc/sys/kernel/pid_max").read_text()) + 1    # no pid can be this
    smi = FakeSmi([{"used": 40960, "procs": [(foreign, "python3.11", 40960)]}, FREE])
    monkeypatch.setattr(repro, "_nvidia_smi", smi)
    said = []
    rec = wb.stop_semantic_service(stage="Omega reconstruction", log=said.append)
    assert rec["awaited"] is False and rec["released"] is None
    assert "cannot be attributed" in rec["why"] and smi.card_reads == 1
    assert any("not awaited" in m for m in said)


def test_a_vllm_with_no_context_yet_is_still_watched(fake_chain, monkeypatch):
    """Killed while loading its weights: before the signal it held no context (only the
    backend-free card); the stop still looks once and returns at once."""
    smi = FakeSmi([FREE, FREE])
    monkeypatch.setattr(repro, "_nvidia_smi", smi)
    rec = wb.stop_semantic_service(stage="SAM3", log=lambda m: None)
    assert rec["awaited"] is True and rec["released"] is True and rec["polls"] == 0
    assert smi.card_reads == 2


def test_no_nvidia_smi_never_breaks_the_stop(fake_chain, monkeypatch):
    def missing(args):
        raise repro.ReproError("nvidia-smi is not installed — the card cannot be checked")

    monkeypatch.setattr(repro, "_nvidia_smi", missing)
    said = []
    rec = wb.stop_semantic_service(stage="cloud cleaning", log=said.append)
    assert rec["awaited"] is False and "not installed" in rec["why"]
    assert fake_chain["alive"] == [] and said[-1].startswith("[gpu] vLLM stopped")


# ── 3. every launcher form is matched (the real pgrep, against processes this test starts) ──

@pytest.mark.skipif(not (shutil.which("perl") and shutil.which("pgrep")),
                    reason="needs perl (to set a command line) and pgrep")
def test_every_form_of_the_launcher_chain_is_found_and_nothing_else():
    """Command lines as /proc shows them (verified 2026-10-07: bash `exec python …` keeps
    argv[0] bare; init_pod.sh's tmux runs the launcher by a relative path). Each is given to a
    sleeping perl through $0; vllm_pids (the real pgrep) must find exactly the chain's."""
    chain = ["bash scripts/serve_semantic.sh",
             "/bin/bash /workspace/stac-build/scripts/serve_semantic.sh",
             "bash /workspace/stac-build/scripts/serve_semantic.sh qwen_local_large",
             "python -m semantic.serve",
             "/workspace/miniforge3/envs/semantic/bin/python -m semantic.serve --backend qwen_local",
             "/workspace/miniforge3/envs/semantic/bin/python3.11 /workspace/miniforge3/envs/"
             "semantic/bin/vllm serve /workspace/stac-build/weights/qwen3vl/8b-instruct --port 8799",
             "VLLM::EngineCore"]
    others = ["tail -f /workspace/stac-build/logs/semantic_latest.log",
              "vim scripts/serve_semantic.sh",
              "bash -c pkill -f 'vllm serve'",
              "grep -rn vllm serve server",
              "python -m semantic.healthcheck",
              "/workspace/miniforge3/envs/da3/bin/python -m pytest -k semantic.serve",
              "less /workspace/stac-build/scripts/serve_semantic.sh"]
    procs = {}
    try:
        for cmd in chain + others:
            procs[cmd] = subprocess.Popen(["perl", "-e", f"$0 = q{{{cmd}}}; sleep 60"])
        deadline = time.time() + 10
        while time.time() < deadline and any(
                not Path(f"/proc/{p.pid}/cmdline").read_bytes().startswith(c.encode()[:6])
                for c, p in procs.items()):
            time.sleep(0.05)
        found = set(wb.vllm_pids())
        missed = [c for c in chain if procs[c].pid not in found]
        wrong = [c for c in others if procs[c].pid in found]
        assert missed == [] and wrong == [], (missed, wrong)
    finally:
        for p in procs.values():
            p.kill()
            p.wait()
