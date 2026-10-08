"""server/repro.py — the shared reproducibility primitives (docs/plan_determinismo.md, 2026-10-07):
deterministic torch, the exclusive-GPU check, the card read through torch, the environment record,
stamps, stable ids and lossless pose text. CPU only: nvidia-smi and the torch card probe are
replaced by canned answers; nothing here touches the GPU."""

import json
import os
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import repro  # noqa: E402

CARD_ROW = "0, GPU-9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5, NVIDIA A100 80GB PCIe, {used}, 81920, 595.91.07\n"
UUID = "GPU-9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5"


def _smi(cards: str, apps: str = ""):
    def fake(args):
        if args[0].startswith("--query-gpu"):
            return cards
        if args[0].startswith("--query-compute-apps"):
            return apps
        raise AssertionError(args)
    return fake


# ── deterministic torch ─────────────────────────────────────────────────────────────────────

def test_deterministic_env_pins_cublas_and_hash_seed_and_refuses_another_value():
    env = repro.deterministic_env({"PATH": "/bin"})
    assert env == {"PATH": "/bin", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
    assert repro.deterministic_env({"PYTHONHASHSEED": "0"})["PYTHONHASHSEED"] == "0"
    with pytest.raises(repro.ReproError, match="CUBLAS_WORKSPACE_CONFIG"):
        repro.deterministic_env({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})
    with pytest.raises(repro.ReproError, match="PYTHONHASHSEED"):
        repro.deterministic_env({"PYTHONHASHSEED": "random"})


def test_ensure_cublas_workspace_sets_it_and_refuses_another(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    repro.ensure_cublas_workspace()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(repro.ReproError, match=":4096:8"):
        repro.ensure_cublas_workspace()


def test_ensure_cublas_workspace_refuses_after_cuda_initialised(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(repro.ReproError, match="already initialised"):
        repro.ensure_cublas_workspace()


def test_deterministic_torch_is_strict_inside_and_restores_outside(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = repro.torch_numerics_record()
    with repro.deterministic_torch(123) as rec:
        assert rec["deterministic_algorithms"] and not rec["deterministic_warn_only"]
        assert rec["cudnn_deterministic"] and not rec["cudnn_benchmark"]
        assert not rec["cudnn_allow_tf32"] and not rec["matmul_allow_tf32"]
        assert rec["float32_matmul_precision"] == "highest" and rec["seed"] == 123
        assert rec["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        a = (torch.rand(5), np.random.rand(3))
        json.dumps(rec)
    with repro.deterministic_torch(123):
        b = (torch.rand(5), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    after = repro.torch_numerics_record()
    keys = ("deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
            "cudnn_benchmark", "cudnn_allow_tf32", "matmul_allow_tf32")
    assert {k: after[k] for k in keys} == {k: before[k] for k in keys}


def test_enable_deterministic_torch_is_process_wide(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prev = repro._torch_state(torch)
    try:
        rec = repro.enable_deterministic_torch(7)
        assert rec["seed"] == 7 and torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    finally:
        repro._restore_torch_state(torch, prev)
    with pytest.raises(repro.ReproError, match="integer"):
        repro.enable_deterministic_torch(1.5)


# ── the card ────────────────────────────────────────────────────────────────────────────────

# torch's readings of the A100 80GB PCIe, 2026-10-07: before (~06:21, when server/card_table.json
# was written) and after (~07:17) a pod restart — 3 MiB apart on the same card model
TORCH_BYTES_BEFORE_RESTART = 85097971712
TORCH_BYTES_AFTER_RESTART = 85094825984
A100_KEY = "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0"


def _a100_probe(total_bytes, calls=None):
    def probe(device):
        if calls is not None:
            calls.append(device)
        return {"name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": int(total_bytes),
                "capability": "8.0", "uuid": "9e73f3cf-a20b-ef36-85c3-9cc06ba2b7d5",
                "multi_processor_count": 108, "l2_cache_bytes": 41943040}
    return probe


def test_card_identity_reads_torch_in_a_probe_and_never_returns_unknown(monkeypatch):
    calls = []
    monkeypatch.setattr(repro, "_card_probe_subprocess", _a100_probe(TORCH_BYTES_BEFORE_RESTART, calls))
    monkeypatch.setattr(repro, "_card_probe_inprocess",
                        lambda d: (_ for _ in ()).throw(AssertionError("in-process probe")))
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=0)))
    monkeypatch.delitem(sys.modules, "torch", raising=False)      # no CUDA in this process
    c = repro.card_identity()
    assert calls == [0] and c["uuid"] == UUID and c["capability"] == "8.0"
    assert c["memory_total_mib"] == 81920 and c["total_memory_bytes"] == TORCH_BYTES_BEFORE_RESTART
    assert c["key"] == A100_KEY == repro.card_key(c)
    for bad in ({"name": "", "total_memory_bytes": 1, "capability": "8.0", "uuid": "x"},
                {"name": "unknown", "total_memory_bytes": 1, "capability": "8.0", "uuid": "x"},
                {"name": "A100", "total_memory_bytes": 0, "capability": "8.0", "uuid": "x"},
                {"name": "A100", "total_memory_bytes": 1, "capability": "8.0", "uuid": ""}):
        monkeypatch.setattr(repro, "_card_probe_subprocess", lambda d, b=bad: b)
        with pytest.raises(repro.ReproError, match="incomplete"):
            repro.card_identity()


def test_card_identity_uses_the_live_context_when_this_process_has_one(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(repro, "_card_probe_inprocess", lambda d: {
        "name": "NVIDIA RTX A6000", "total_memory_bytes": 51041271808, "capability": "8.6",
        "uuid": "GPU-abc", "multi_processor_count": 84, "l2_cache_bytes": 6291456})
    monkeypatch.setattr(repro, "_card_probe_subprocess",
                        lambda d: (_ for _ in ()).throw(AssertionError("subprocess probe")))
    monkeypatch.setattr(repro, "_nvidia_smi",
                        _smi("0, GPU-abc, NVIDIA RTX A6000, 0, 49140, 595.91.07\n"))
    c = repro.card_identity()
    assert c["uuid"] == "GPU-abc" and c["key"] == "NVIDIA RTX A6000 | 49140 MiB | sm_8.6"


def test_the_card_key_is_the_model_not_torchs_usable_bytes(monkeypatch):
    """The run of 2026-10-07 died on it: torch read 3 MiB less after a pod restart, on the same
    A100, and the committed table (keyed on the bytes) no longer knew the card. The key is the
    MODEL: name | board MiB (nvidia-smi) | sm — torch's bytes stay in the identity as a fact."""
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=0)))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    idents = []
    for total in (TORCH_BYTES_BEFORE_RESTART, TORCH_BYTES_AFTER_RESTART):
        monkeypatch.setattr(repro, "_card_probe_subprocess", _a100_probe(total))
        idents.append(repro.card_identity())
    assert idents[0]["key"] == idents[1]["key"] == A100_KEY
    assert idents[0]["total_memory_bytes"] != idents[1]["total_memory_bytes"]  # recorded, not keyed
    # the key never carries torch's bytes nor the instance
    assert str(TORCH_BYTES_AFTER_RESTART) not in A100_KEY and "9e73f3cf" not in A100_KEY


def test_one_name_two_board_sizes_are_two_keys(monkeypatch):
    """A name alone is not the model: the GeForce RTX 3060 ships with 8 GB and 12 GB boards."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(repro, "_card_probe_subprocess", lambda d: {
        "name": "NVIDIA GeForce RTX 3060", "total_memory_bytes": 1 << 33, "capability": "8.6",
        "uuid": "GPU-3060", "multi_processor_count": 28, "l2_cache_bytes": 1 << 21})
    keys = set()
    for mib in (8192, 12288):
        monkeypatch.setattr(repro, "_nvidia_smi",
                            _smi(f"0, GPU-3060, NVIDIA GeForce RTX 3060, 0, {mib}, 595.91.07\n"))
        keys.add(repro.card_identity()["key"])
    assert keys == {"NVIDIA GeForce RTX 3060 | 8192 MiB | sm_8.6",
                    "NVIDIA GeForce RTX 3060 | 12288 MiB | sm_8.6"}


def test_card_identity_fails_when_nvidia_smi_cannot_name_the_board(monkeypatch):
    """Point 13: never an 'unknown' card — a uuid nvidia-smi does not list, or nvidia-smi
    failing, RAISES (the board memory is part of the key)."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(repro, "_card_probe_subprocess", _a100_probe(TORCH_BYTES_AFTER_RESTART))
    other = "0, GPU-11111111-2222-3333-4444-555555555555, NVIDIA A100 80GB PCIe, 0, 81920, 595.91.07\n"
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(other))
    with pytest.raises(repro.ReproError, match="does not list the card " + UUID):
        repro.card_identity()

    def broken(args):
        raise repro.ReproError("nvidia-smi --query-gpu failed (exit 9): NVIDIA-SMI has failed")
    monkeypatch.setattr(repro, "_nvidia_smi", broken)
    with pytest.raises(repro.ReproError, match="NVIDIA-SMI has failed"):
        repro.card_identity()


def test_card_key_round_trips_and_refuses_the_old_format():
    rec = repro.parse_card_key(A100_KEY)
    assert rec == {"name": "NVIDIA A100 80GB PCIe", "memory_total_mib": 81920, "capability": "8.0"}
    assert repro.card_key(rec) == A100_KEY
    for bad in ("NVIDIA A100 80GB PCIe | 85097971712 B | sm_8.0",     # torch's bytes (pre 2026-10-07)
                "cardX", "A | 0 MiB | sm_8.0", "A | 81920 MiB | 8.0", " A | 1 MiB | sm_8.0"):
        with pytest.raises(repro.ReproError, match="not a card key"):
            repro.parse_card_key(bad)
    for bad in ({"name": "A", "capability": "8.0"},                     # no board memory
                {"name": "A", "memory_total_mib": 0, "capability": "8.0"},
                {"name": "unknown", "memory_total_mib": 1, "capability": "8.0"}):
        with pytest.raises(repro.ReproError):
            repro.card_key(bad)


def test_the_card_probe_fails_instead_of_guessing(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1)
    monkeypatch.setattr(repro.subprocess, "run", timeout)
    with pytest.raises(repro.ReproError, match="did not answer"):
        repro._card_probe_subprocess(0)
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=3, stdout="", stderr="torch sees no CUDA device"))
    with pytest.raises(repro.ReproError, match="no CUDA device"):
        repro._card_probe_subprocess(0)


def test_exclusive_gpu_passes_on_a_free_card(monkeypatch):
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=0)))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    logs = []
    rec = repro.require_exclusive_gpu(logs.append)
    assert rec == {"cards": [{"uuid": UUID, "name": "NVIDIA A100 80GB PCIe", "used_mib": 0,
                              "total_mib": 81920, "driver_version": "595.91.07"}],
                   "compute_processes": 0}
    assert logs and "exclusive" in logs[0]


def test_exclusive_gpu_fails_listing_every_other_process(monkeypatch):
    apps = (f"4242, vllm serve, 40000, {UUID}\n"
            f"777, python, [N/A], {UUID}\n")
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=41000), apps))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    logs = []
    with pytest.raises(repro.ReproError) as e:
        repro.require_exclusive_gpu(logs.append)
    msg = str(e.value)
    assert "pid 4242 vllm serve 40000 MiB" in msg and "pid 777 python N/A MiB" in msg
    assert "41000 MiB of 81920 MiB in use" in msg and logs


def test_exclusive_gpu_fails_on_memory_used_by_a_process_it_cannot_see(monkeypatch):
    """Inside a container nvidia-smi may list no process at all while another container holds the
    card: the memory in use says so."""
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=512), ""))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(repro.ReproError, match="512 MiB of 81920 MiB in use"):
        repro.require_exclusive_gpu(lambda m: None)


def test_exclusive_gpu_refuses_a_caller_holding_cuda(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=0)))
    with pytest.raises(repro.ReproError, match="already initialised CUDA"):
        repro.require_exclusive_gpu(lambda m: None)


def test_exclusive_gpu_checks_the_cards_this_process_may_use(monkeypatch):
    other = "GPU-11111111-2222-3333-4444-555555555555"
    cards = CARD_ROW.format(used=0) + f"1, {other}, NVIDIA A100 80GB PCIe, 30000, 81920, 595.91.07\n"
    apps = f"99, someone, 30000, {other}\n"
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(cards, apps))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID[:12])
    assert repro.require_exclusive_gpu(lambda m: None)["cards"][0]["uuid"] == UUID
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")            # by index: every card is checked
    with pytest.raises(repro.ReproError, match="pid 99"):
        repro.require_exclusive_gpu(lambda m: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(repro.ReproError, match="no card"):
        repro.require_exclusive_gpu(lambda m: None)


def test_nvidia_smi_failures_raise(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=1)
    monkeypatch.setattr(repro.subprocess, "run", run)
    with pytest.raises(repro.ReproError, match="did not answer"):
        repro.gpu_cards()
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=9, stdout="", stderr="NVIDIA-SMI has failed"))
    with pytest.raises(repro.ReproError, match="failed"):
        repro.gpu_compute_processes()
    monkeypatch.setattr(repro, "_nvidia_smi", _smi("garbage\n"))
    with pytest.raises(repro.ReproError, match="unreadable"):
        repro.gpu_cards()


# ── the environment ─────────────────────────────────────────────────────────────────────────

def test_environment_record_is_complete_and_the_same_twice(monkeypatch):
    real = repro.environment_record(gpu=False)
    assert len(real["git"]["repo"]["commit"]) == 40
    assert "vendor/VGGT-Long" in real["git"]["submodules"]
    # the same twice (git pinned: other work may edit the tree between two calls of this test;
    # git_state itself is tested on a repository of its own below)
    monkeypatch.setattr(repro, "git_state", lambda root: {"commit": str(root)})
    a = repro.environment_record(gpu=False)
    b = repro.environment_record(gpu=False)
    assert a == b
    json.dumps(a)
    assert a["cpu"]["model"] and a["cpu"]["logical_cpus"] > 0
    assert a["libs"]["numpy"] == np.__version__
    assert any(r["internal_api"] == "openblas" for r in a["blas"])
    assert "OPENBLAS_CORETYPE" in a["env"] and "gpu" not in a and "torch" not in a


def test_environment_record_with_the_card(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(repro, "card_identity", lambda device=0: {
        "device": 0, "name": "NVIDIA A100 80GB PCIe", "total_memory_bytes": 85097971712,
        "capability": "8.0", "uuid": UUID, "multi_processor_count": 108,
        "l2_cache_bytes": 41943040, "key": "k"})
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(CARD_ROW.format(used=0)))
    rec = repro.environment_record(gpu=True)
    assert rec["gpu"]["driver_version"] == "595.91.07" and rec["gpu"]["uuid"] == UUID
    assert rec["torch"]["cudnn"] > 0 and rec["torch"]["version"]
    other = "1, GPU-ffff, X, 0, 1, 1.0\n"
    monkeypatch.setattr(repro, "_nvidia_smi", _smi(other))
    with pytest.raises(repro.ReproError, match="does not list the card"):
        repro.environment_record(gpu=True)


def test_git_state_sees_a_dirty_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "c"], check=True, env=env)
    clean = repro.git_state(tmp_path)
    assert not clean["dirty"] and clean["n_untracked"] == 0
    (tmp_path / "a.py").write_text("x = 2\n")
    dirty = repro.git_state(tmp_path)
    assert dirty["dirty"] and dirty["diff_sha256"] != clean["diff_sha256"]
    assert dirty["commit"] == clean["commit"]
    (tmp_path / "new.py").write_text("y\n")
    u1 = repro.git_state(tmp_path)
    assert u1["n_untracked"] == 1
    (tmp_path / "new.py").write_text("z\n")                   # same size, other content
    assert repro.git_state(tmp_path)["untracked_sha256"] != u1["untracked_sha256"]


# ── hashing, stamps, stable ids ─────────────────────────────────────────────────────────────

def test_canonical_json_is_one_text_per_value():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert repro.canonical_json({"b": 1, "a": [np.int64(2), np.float64(0.1)]}) == \
        repro.canonical_json({"a": [2, 0.1], "b": 1})
    assert repro.sha256_json({"x": a}) == repro.sha256_json({"x": np.asfortranarray(a)})
    assert repro.sha256_json({"x": a}) != repro.sha256_json({"x": a.astype(np.float64)})
    assert repro.canonical_json({3, 1, 2}) == repro.canonical_json({2, 3, 1})
    assert repro.canonical_json(Path("/a/b")) == '"/a/b"'
    with pytest.raises(TypeError):
        repro.canonical_json(object())
    with pytest.raises(TypeError):
        repro.canonical_json(np.array([object()]))


def test_sha256_path_of_a_directory_ignores_bytecode_and_sees_content(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "sub" / "b.txt").write_text("b")
    h0 = repro.sha256_path(d)
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "m.cpython-310.pyc").write_bytes(b"\0")
    (d / "x.pyc").write_bytes(b"\1")
    assert repro.sha256_path(d) == h0
    (d / "sub" / "b.txt").write_text("B")
    assert repro.sha256_path(d) != h0
    with pytest.raises(FileNotFoundError):
        repro.sha256_path(tmp_path / "missing")


def _session(root: Path) -> Path:
    (root / "output").mkdir(parents=True)
    (root / "output" / "camera_poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
    (root / "output" / "walk.json").write_text('{"walk_m": 12.5}')
    return root


def test_stamp_names_every_difference_and_survives_a_copy(tmp_path):
    s = _session(tmp_path / "s1")
    ins = [s / "output" / "camera_poses.txt", s / "output" / "walk.json"]
    cfg = {"refine": {"tol": 1e-8, "rungs": ["R0", "R1"]}, "graph": {"heldout_confidence": 0.95}}
    st = repro.stamp(ins, code=["repro"], config=cfg, root=s)
    assert set(st) == {"stamp_version", "inputs", "code", "config", "sha256"}
    assert set(st["inputs"]) == {"output/camera_poses.txt", "output/walk.json"}
    assert set(st["code"]) == {"server/repro.py"}
    assert repro.check_stamp(st, repro.stamp(ins, code=["repro"], config=cfg, root=s)) == []
    json.loads(json.dumps(st))                                  # persisted as JSON
    # a copied session keeps its stamp
    s2 = tmp_path / "elsewhere" / "s1"
    shutil.copytree(s, s2)
    ins2 = [s2 / "output" / "camera_poses.txt", s2 / "output" / "walk.json"]
    assert repro.stamp(ins2, code=["repro"], config=cfg, root=s2) == st
    # an input changed
    (s / "output" / "walk.json").write_text('{"walk_m": 12.6}')
    d = repro.check_stamp(st, repro.stamp(ins, code=["repro"], config=cfg, root=s))
    assert len(d) == 1 and "input 'output/walk.json' changed" in d[0]
    # a config section changed, another is new, one input gone
    cfg2 = {"refine": {"tol": 1e-9, "rungs": ["R0", "R1"]}, "graph": cfg["graph"], "new": 1}
    d = repro.check_stamp(st, repro.stamp(ins[:1], code=["repro"], config=cfg2, root=s))
    assert any("config 'refine' changed" in x for x in d)
    assert any("config 'new' is new" in x for x in d)
    assert any("input 'output/walk.json' was stamped and is not one now" in x for x in d)


def test_stamp_code_entries_and_their_keys(tmp_path):
    sys.path.insert(0, str(ROOT / "vendor" / "VGGT-Long"))
    import loop_utils.metric_lock as ml
    st = repro.stamp(code=[repro, "loop_utils.metric_lock", ROOT / "server" / "repro.py"])
    assert set(st["code"]) == {"server/repro.py", "vendor/VGGT-Long/loop_utils/metric_lock.py"}
    assert st["code"]["vendor/VGGT-Long/loop_utils/metric_lock.py"] == repro.sha256_file(ml.__file__)
    base = repro.stamp(code=[repro])
    d = repro.check_stamp(base, st)
    assert d == ["code 'vendor/VGGT-Long/loop_utils/metric_lock.py' is new (not in the saved stamp)"]
    with pytest.raises(repro.ReproError, match="outside"):
        repro.stamp(code=[np])
    with pytest.raises(repro.ReproError, match="cannot be found"):
        repro.stamp(code=["no_such_module_xyz"])


def test_stamp_inputs_by_mapping_and_duplicate_names(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "f.txt").write_text("1")
    (tmp_path / "b" / "f.txt").write_text("2")
    with pytest.raises(repro.ReproError, match="share the key"):
        repro.stamp([tmp_path / "a" / "f.txt", tmp_path / "b" / "f.txt"])
    st = repro.stamp({"first": tmp_path / "a" / "f.txt", "second": tmp_path / "b" / "f.txt"})
    assert set(st["inputs"]) == {"first", "second"}
    with pytest.raises(FileNotFoundError):
        repro.stamp([tmp_path / "nope.txt"])
    with pytest.raises(repro.ReproError, match="mapping"):
        repro.stamp(config=[1, 2])


def test_check_stamp_on_no_stamp_or_an_edited_one():
    now = repro.stamp(config={"a": 1})
    assert repro.check_stamp(None, now) == ["no stamp saved with the product (or it is unreadable)"]
    assert repro.check_stamp({"inputs": {}}, now)[0].startswith("no stamp saved")
    old = dict(now, stamp_version=0)
    assert any("stamp version" in x for x in repro.check_stamp(old, now))
    edited = dict(now, sha256="0" * 64)
    assert repro.check_stamp(edited, now) == [
        "the saved stamp's digest does not match its parts (it was edited)"]
    with pytest.raises(repro.ReproError):
        repro.check_stamp(now, {"inputs": {}})


def test_stable_id_is_derived_from_its_parts_only():
    a = repro.stable_id("epoch", 3, {"kf": [1, 2], "b": 0.5})
    assert a == repro.stable_id("epoch", 3, {"b": 0.5, "kf": [1, 2]})
    assert len(a) == 64 and a != repro.stable_id("epoch", 4, {"kf": [1, 2], "b": 0.5})
    assert repro.stable_id("x", n_hex=8) == repro.stable_id("x")[:8]
    with pytest.raises(repro.ReproError):
        repro.stable_id()
    with pytest.raises(repro.ReproError):
        repro.stable_id("x", n_hex=0)


# ── lossless pose text (point 45) ───────────────────────────────────────────────────────────

def _hard_floats(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) * 10.0 ** rng.integers(-12, 12, size=n)
    v[:8] = [0.1, -0.0, 5e-324, 1e-300, 1.7976931348623157e308, 1 / 3, -2.0 / 3.0, 123456789.12345679]
    return v


def test_exact_float_round_trips_every_float64():
    v = _hard_floats(5000)
    back = np.array([float(repro.exact_float(x)) for x in v])
    assert np.array_equal(back.view(np.uint64), v.view(np.uint64))
    # the defect it replaces: '{:.8g}' does not
    lossy = np.array([float(f"{x:.8g}") for x in v])
    assert not np.array_equal(lossy.view(np.uint64), v.view(np.uint64))
    f32 = np.float32(0.1)
    assert np.float32(float(repro.exact_float(f32))) == f32


def test_poses_written_exact_read_back_bit_identical_by_every_reader(tmp_path):
    poses = _hard_floats(16 * 50, seed=1).reshape(50, 4, 4)
    p = repro.write_poses_exact(tmp_path / "out" / "camera_poses.txt", poses)
    via_loadtxt = np.loadtxt(p).reshape(-1, 4, 4)
    assert np.array_equal(via_loadtxt.view(np.uint64), poses.view(np.uint64))
    from correction.session import read_poses
    via_session = read_poses(p)
    assert np.array_equal(via_session.view(np.uint64), poses.view(np.uint64))
    assert p.read_text().count("\n") == 50 and len(p.read_text().splitlines()[0].split()) == 16
    # (N,16) and one (4,4) are accepted; anything else is refused
    assert repro.write_poses_exact(tmp_path / "b.txt", poses.reshape(50, 16)).read_text() == p.read_text()
    assert len(repro.write_poses_exact(tmp_path / "c.txt", np.eye(4)).read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        repro.write_poses_exact(tmp_path / "d.txt", np.zeros((3, 3)))
    # writing twice gives the same bytes; no temporary file is left behind
    a = p.read_bytes()
    repro.write_poses_exact(p, poses)
    assert p.read_bytes() == a and sorted(x.name for x in p.parent.iterdir()) == ["camera_poses.txt"]


def test_intrinsics_written_exact_read_back_bit_identical(tmp_path):
    rows = np.abs(_hard_floats(4 * 30, seed=2).reshape(30, 4)) + 100.0
    rows[0] = [391.87123456789012, 388.71, 234.80000000000001, 414.06]
    p = repro.write_intrinsics_exact(tmp_path / "intrinsic.txt", rows)
    back = np.loadtxt(p, dtype=np.float64, ndmin=2)
    assert np.array_equal(back.view(np.uint64), rows.view(np.uint64))
    from reconstruction.trace_normals import _read_intrinsic_rows
    K = _read_intrinsic_rows(p)
    assert np.array_equal(K[:, 0, 0], rows[:, 0]) and np.array_equal(K[:, 1, 2], rows[:, 3])
    assert len(repro.write_intrinsics_exact(tmp_path / "one.txt", rows[0]).read_text().split()) == 4
    with pytest.raises(ValueError):
        repro.write_intrinsics_exact(tmp_path / "bad.txt", np.zeros((3, 3)))


def test_written_files_keep_the_umask_permissions(tmp_path):
    old = os.umask(0o022)
    try:
        p = repro.write_rows_exact(tmp_path / "r.txt", [[1.0, 2.0]])
    finally:
        os.umask(old)
    assert stat.S_IMODE(p.stat().st_mode) == 0o644
    with pytest.raises(ValueError):
        repro.write_rows_exact(tmp_path / "e.txt", np.zeros((0, 3)))
