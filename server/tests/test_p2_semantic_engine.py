"""Wave 2, package P2 — the semantic engine (docs/plan_determinismo.md points 80, 81, 94,
154, 155, 156): the launcher's deterministic flags and pinned weights, the identity it
writes and the service reads back, the engine lease, ensure_service's refusals, the
per-job engine context, and the chat's 'busy'. CPU only, no vLLM, no network."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import repro  # noqa: E402
import semantic.serve as serve  # noqa: E402
import semantic.service as svc  # noqa: E402
from semantic.semantic_config import load_semantic_config  # noqa: E402


# ── the launcher (point 80 / 94) ─────────────────────────────────────────────────────

def test_the_serve_line_carries_every_deterministic_flag_and_only_local_weights():
    cfg = load_semantic_config()
    argv = serve.build_argv("qwen_local", {}, cfg)
    b = cfg["backends"]["qwen_local"]
    assert argv[:3] == ["vllm", "serve", b["weights_path_abs"]] if "weights_path_abs" in b \
        else argv[2].endswith("weights/qwen3vl/8b-instruct")
    for tok in ("--no-enable-prefix-caching", "--enforce-eager"):
        assert tok in argv
    i = argv.index("--max-num-seqs")
    assert argv[i + 1] == "1"
    i = argv.index("--max-num-batched-tokens")
    assert argv[i + 1] == str(int(b["max_model_len"])), "the token batch IS the model length"
    i = argv.index("--generation-config")
    assert argv[i + 1] == "vllm", "the model's generation_config.json defaults never apply"
    i = argv.index("--seed")
    assert argv[i + 1] == str(int(cfg["generation"]["seed"]))
    i = argv.index("--attention-backend")
    assert argv[i + 1] == serve.ATTENTION_BACKEND
    # the flags are the same object the identity check reads (one definition)
    want = serve.deterministic_serve_args(int(b["max_model_len"]), int(cfg["generation"]["seed"]))
    assert all(w in argv for w in want if w.startswith("--"))


def test_the_service_environment_is_deterministic_offline_and_batch_invariant():
    env = serve.service_environment({"PATH": "/x"})
    assert env["VLLM_BATCH_INVARIANT"] == "1" and env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTHONHASHSEED"] == "0" and env[repro.CUBLAS_WORKSPACE_ENV] == repro.CUBLAS_WORKSPACE_VALUE
    with pytest.raises(repro.ReproError):
        serve.service_environment({"PYTHONHASHSEED": "7"})
    # which form of invariance the card gets is recorded, never guessed
    assert serve.batch_invariance_record("8.0")["form"] == "batch_invariant_matmul"
    assert serve.batch_invariance_record("10.0")["form"] == "batch_invariant_matmul"
    assert serve.batch_invariance_record("8.6")["form"] == "cublas_workspace"
    assert serve.batch_invariance_record("8.6")["enabled"] is True


def test_the_weights_are_served_only_when_every_pinned_file_hashes_to_its_pin(tmp_path, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 10)
    pins = {"test/model": {"revision": "deadbeef", "sha256": {
        "config.json": repro.sha256_bytes(b"{}"),
        "model.safetensors": repro.sha256_bytes(b"\x00" * 10)}}}
    monkeypatch.setattr(serve, "PINNED_WEIGHTS", pins)
    b = {"name": "t", "model_id": "test/model", "weights_path_abs": str(root)}
    ident = serve.weights_identity(b, log=lambda m: None)
    assert ident["model_id"] == "test/model" and ident["revision"] == "deadbeef"
    assert ident["sha256"] == pins["test/model"]["sha256"]
    # a changed byte, a missing file, a missing directory, an unpinned model: refused
    (root / "model.safetensors").write_bytes(b"\x00" * 9 + b"\x01")
    with pytest.raises(serve.ServeError, match="not the weights"):
        serve.weights_identity(b, log=lambda m: None)
    (root / "model.safetensors").unlink()
    with pytest.raises(serve.ServeError, match="missing"):
        serve.weights_identity(b, log=lambda m: None)
    with pytest.raises(serve.ServeError, match="never downloads"):
        serve.weights_identity(dict(b, weights_path_abs=str(tmp_path / "nowhere")), log=lambda m: None)
    with pytest.raises(serve.ServeError, match="no pinned revision"):
        serve.weights_identity(dict(b, model_id="other/model"), log=lambda m: None)


def test_the_production_weights_are_pinned_and_present():
    cfg = load_semantic_config()
    b = cfg["backends"]["qwen_local"]
    pin = serve.PINNED_WEIGHTS[b["model_id"]]
    root = SERVER.parent / b["weights_path"]
    if not root.is_dir():
        pytest.skip("the local Qwen3-VL weights are not on this checkout")
    for name in pin["sha256"]:
        assert (root / name).is_file(), f"pinned file {name} missing from {root}"
    # the launcher refuses a config with no seed (the seed is part of every answer's identity)
    bad = json.loads(json.dumps(cfg))
    bad["generation"].pop("seed")
    with pytest.raises(serve.ServeError, match="seed"):
        serve.build_argv("qwen_local", {}, bad)


def test_the_config_block_is_read_from_the_frozen_file_or_refused(tmp_path):
    p = tmp_path / "run_config.yaml"
    p.write_text("semantic:\n  service:\n    port: 8123\n  default_backend: qwen_local\n")
    cfg = serve.semantic_config_from(p)
    assert cfg["service"]["port"] == 8123 and cfg["backends"]["qwen_local"]["model_id"]
    (tmp_path / "noblock.yaml").write_text("intake: {}\n")
    with pytest.raises(serve.ServeError, match="no 'semantic' section"):
        serve.semantic_config_from(tmp_path / "noblock.yaml")
    with pytest.raises(serve.ServeError):
        serve.semantic_config_from(tmp_path / "missing.yaml")
    # the block's sha256 is canonical: key order does not change it
    a = serve.semantic_config_sha256({"x": 1, "y": {"b": 2, "a": 1}})
    assert a == serve.semantic_config_sha256({"y": {"a": 1, "b": 2}, "x": 1})


# ── the identity (point 155) ─────────────────────────────────────────────────────────

def _identity_doc(tmp_path, cfg, pid, argv=None):
    from semantic.semantic_config import _DEFAULTS, _deep_merge
    b = cfg["backends"]["qwen_local"]
    merged = _deep_merge(_DEFAULTS, cfg)
    argv = argv or serve.build_argv("qwen_local", {}, merged, model_ref="weights/x")
    doc = {"identity": {"identity_version": 1, "backend": "qwen_local",
                        "served_model_name": b["served_model_name"], "argv": argv,
                        "batch_invariance": {"enabled": True, "form": "batch_invariant_matmul"},
                        "semantic_config_sha256": serve.semantic_config_sha256(merged),
                        "weights": {"model_id": b["model_id"], "revision": "r"},
                        "card": {"key": "A | 1 MiB | sm_8.0"}, "sha256": "abc"},
           "process": {"pid": pid}}
    p = tmp_path / "semantic_service.json"
    serve.write_identity(doc, p)
    return p, doc


def test_service_identity_is_read_only_for_a_live_vllm_launcher(tmp_path, monkeypatch):
    cfg = load_semantic_config()
    p, doc = _identity_doc(tmp_path, cfg, os.getpid())
    monkeypatch.setattr(serve, "identity_path", lambda: p)
    # this process is alive but is no vLLM: no identity
    assert svc.service_identity() is None
    monkeypatch.setattr(svc, "_cmdline", lambda pid: "python -m vllm serve weights --port 1")
    got = svc.service_identity()
    assert got["identity"]["served_model_name"] == cfg["backends"]["qwen_local"]["served_model_name"]
    # a dead launcher pid: no identity
    _identity_doc(tmp_path, cfg, 2 ** 22 - 1)
    assert svc.service_identity() is None


def test_verify_service_identity_names_every_difference(tmp_path, monkeypatch):
    cfg = load_semantic_config()
    p, doc = _identity_doc(tmp_path, cfg, os.getpid())
    monkeypatch.setattr(serve, "identity_path", lambda: p)
    monkeypatch.setattr(svc, "_cmdline", lambda pid: "vllm serve")
    monkeypatch.setattr("semantic.backends.QwenLocalBackend.health",
                        lambda self: {"ok": True, "served_models": [self.model]})
    ident = svc.verify_service_identity("qwen_local", cfg, log=lambda m: None)
    assert ident["served_model_name"] == "qwen_local"
    # another configuration: refused
    other = json.loads(json.dumps(cfg))
    other["generation"]["max_tokens"] = 999
    with pytest.raises(svc.EngineIdentityError, match="semantic configuration"):
        svc.verify_service_identity("qwen_local", other, log=lambda m: None)
    # a flag missing from the engine's argv: refused
    argv = [a for a in doc["identity"]["argv"] if a != "--enforce-eager"]
    _identity_doc(tmp_path, cfg, os.getpid(), argv=argv)
    with pytest.raises(svc.EngineIdentityError, match="--enforce-eager missing"):
        svc.verify_service_identity("qwen_local", cfg, log=lambda m: None)
    # no live record at all: refused
    p.unlink()
    with pytest.raises(svc.EngineIdentityError, match="no identity record"):
        svc.verify_service_identity("qwen_local", cfg, log=lambda m: None)


# ── the lease (points 81 / 154) ──────────────────────────────────────────────────────

@pytest.fixture
def lease(tmp_path, monkeypatch):
    p = tmp_path / "lease.json"
    monkeypatch.setattr(svc, "lease_path", lambda: p)
    yield p
    svc.release_engine_lease()


def test_the_lease_is_one_per_process_and_a_dead_holder_is_stale(lease):
    assert svc.engine_lease_holder() is None and svc.engine_available_to()[0]
    doc = svc.acquire_engine_lease("vlm_worker", stage="VLM stage", session="s1")
    assert doc["pid"] == os.getpid() and lease.exists()
    assert svc.acquire_engine_lease("vlm_worker", stage="VLM stage")["pid"] == os.getpid(), "idempotent"
    ok, why = svc.engine_available_to()
    assert ok and "this process" in why
    ok, why = svc.engine_available_to(pid=1)
    assert not ok and "leased to pid" in why and "VLM stage" in why
    assert svc.release_engine_lease() is True and not lease.exists()
    assert svc.release_engine_lease() is False
    # a lease of a process that is gone is stale: ignored and removed
    lease.write_text(json.dumps({"pid": 2 ** 22 - 1, "owner": "ghost", "stage": "x"}))
    assert svc.engine_lease_holder() is None and not lease.exists()
    # a lease of a LIVE other process: busy
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "SAM3"}))
    with pytest.raises(svc.EngineBusy, match="leased to pid"):
        svc.acquire_engine_lease("me", stage="y")
    lease.unlink()


def test_ensure_service_refuses_while_another_process_holds_the_lease(lease, monkeypatch):
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "VLM stage"}))
    launched = []
    monkeypatch.setattr(svc, "_launch", lambda args, log: launched.append(args))
    monkeypatch.setattr(svc, "is_alive", lambda config=None, timeout_s=3.0: True)
    said = []
    assert svc.ensure_service({}, log=said.append, timeout_s=0.0) is False
    assert any("busy" in m and "leased" in m for m in said) and not launched
    lease.unlink()
    # a launcher chain already alive is awaited, never duplicated
    monkeypatch.setattr(svc, "is_alive", lambda config=None, timeout_s=3.0: False)
    monkeypatch.setattr(svc, "is_starting", lambda: True)
    monkeypatch.setattr(svc, "_LAUNCHER", Path(__file__))
    assert svc.ensure_service({}, log=said.append, timeout_s=0.0) is False
    assert any("already running" in m for m in said) and not launched


def test_the_chat_gets_busy_while_a_job_runs_or_holds_the_lease(lease, monkeypatch):
    from phase5_qa import api
    fake_main = SimpleNamespace(pipeline_manager=SimpleNamespace(
        get_all_jobs=lambda: {"s": {"status": "running"}}))
    monkeypatch.setitem(sys.modules, "main", fake_main)
    assert "pipeline" in api._engine_busy()
    fake_main.pipeline_manager.get_all_jobs = lambda: {"s": {"status": "done"}}
    assert api._engine_busy() is None
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "vlm_worker", "stage": "VLM stage"}))
    assert "leased" in api._engine_busy()


# ── the job's own engine (point 155 / 156) ───────────────────────────────────────────

def test_job_engine_stops_the_old_engine_launches_its_own_verifies_and_stops_it(lease, monkeypatch):
    events = []
    import workers.base as wb
    monkeypatch.setattr(wb, "stop_semantic_service_verified",
                        lambda pipe=None, stage="", log=None: events.append(("stop", stage)))
    monkeypatch.setattr(svc, "_foreign_gpu_tenants", lambda log: [])
    monkeypatch.setattr(svc, "release_cached_vram", lambda log=None: None)
    proc = SimpleNamespace(poll=lambda: None, returncode=None, pid=4242)
    monkeypatch.setattr(svc, "_launch", lambda args, log: (events.append(("launch", args)), proc)[1])
    monkeypatch.setattr(svc, "_wait_alive", lambda config, t, log, cancelled=None, proc=None: True)
    monkeypatch.setattr(svc, "verify_service_identity",
                        lambda backend, cfg, log=print: {"sha256": "id", "backend": backend})
    cfg = {"semantic": {"service": {"startup_timeout_s": 5}}}
    with svc.job_engine(cfg, backend="qwen_local", owner="t", stage="VLM stage",
                        run_config_path="/s/output/run_config.yaml", log=lambda m: None) as ident:
        assert ident["sha256"] == "id"
        assert svc.engine_lease_holder()["pid"] == os.getpid(), "the lease is held inside"
        assert events[0] == ("stop", "VLM stage")
        assert events[1] == ("launch", ["--backend", "qwen_local", "--config",
                                        "/s/output/run_config.yaml"])
    assert events[-1][0] == "stop" and "shutdown" in events[-1][1], "its own engine is stopped"
    assert svc.engine_lease_holder() is None, "and the lease released"


def test_job_engine_fails_on_the_startup_bound_a_dead_launcher_or_a_tenant(lease, monkeypatch):
    import workers.base as wb
    monkeypatch.setattr(wb, "stop_semantic_service_verified", lambda pipe=None, stage="", log=None: None)
    monkeypatch.setattr(svc, "release_cached_vram", lambda log=None: None)
    cfg = {"semantic": {"service": {"startup_timeout_s": 1}}}
    # a foreign process on the card: never launched beside it
    monkeypatch.setattr(svc, "_foreign_gpu_tenants",
                        lambda log: [{"pid": 7, "name": "python sam3", "used_mib": 20000}])
    with pytest.raises(svc.EngineUnavailable, match="never started with another process"):
        with svc.job_engine(cfg, backend="qwen_local", owner="t", stage="VLM stage", log=lambda m: None):
            pass
    assert svc.engine_lease_holder() is None
    # the bound: a wait that runs out FAILS (no fallback)
    monkeypatch.setattr(svc, "_foreign_gpu_tenants", lambda log: [])
    alive = SimpleNamespace(poll=lambda: None, returncode=None, pid=1)
    monkeypatch.setattr(svc, "_launch", lambda args, log: alive)
    monkeypatch.setattr(svc, "_wait_alive", lambda config, t, log, cancelled=None, proc=None: False)
    stops = []
    monkeypatch.setattr(wb, "stop_semantic_service_verified",
                        lambda pipe=None, stage="", log=None: stops.append(stage))
    with pytest.raises(svc.EngineUnavailable, match="startup bound"):
        with svc.job_engine(cfg, backend="qwen_local", owner="t", stage="VLM stage", log=lambda m: None):
            pass
    assert any("shutdown" in s for s in stops), "what was launched is stopped"
    # a launcher that died (pinned weights refused, say) fails at once with its words
    dead = SimpleNamespace(poll=lambda: 1, returncode=1, pid=1)
    monkeypatch.setattr(svc, "_launch", lambda args, log: dead)
    monkeypatch.setattr(svc, "_launcher_log_tail", lambda n=12: "ServeError: not the weights")
    with pytest.raises(svc.EngineUnavailable, match="not the weights"):
        with svc.job_engine(cfg, backend="qwen_local", owner="t", stage="VLM stage", log=lambda m: None):
            pass
    # a job that holds the lease blocks another job's engine
    svc.acquire_engine_lease("other", stage="SAM3")
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "SAM3"}))
    with pytest.raises(svc.EngineBusy):
        with svc.job_engine(cfg, backend="qwen_local", owner="t", stage="VLM stage", log=lambda m: None):
            pass
    lease.unlink()


def test_the_vlm_worker_runs_on_its_own_engine_and_has_no_fallback(tmp_path, monkeypatch):
    """points 155 / 156: the worker enters the job engine, hands the identity to the
    auto-prompter, and FAILS — no InternVL3 — when the engine is unavailable."""
    import yaml
    from workers import vlm_worker
    (tmp_path / "frames").mkdir()
    cfg = yaml.safe_load((SERVER / "config.yaml").read_text())
    seen = {}

    @contextlib.contextmanager
    def engine(pipe, config):
        seen["session"] = config.get("_session_dir")
        yield {"sha256": "engine-id", "served_model_name": "qwen_local"}

    class _AP:
        def __init__(self, session, out, backend="qwen_local", config=None):
            pass

        def run(self, run_sam3=False, on_progress=None, service=None):
            seen["service"] = service
            return SimpleNamespace(prompt="desk;floor", frame_map={}, n_accepted=0, n_review=0,
                                   per_class_counts={})

    import segmentation.autoprompt.session_builder as sb
    monkeypatch.setattr(sb, "AutoPrompter", _AP)
    monkeypatch.setattr(vlm_worker, "_ensure_semantic_service", engine)

    class _Pipe:
        def send_log(self, *a, **k): pass
        def send_progress(self, *a, **k): pass
        def check_cancel(self): return False

    vlm_worker._vlm_work(_Pipe(), str(tmp_path), cfg)
    assert seen["session"] == str(tmp_path) and seen["service"]["sha256"] == "engine-id"
    assert "scene_analyzer" not in vlm_worker.__dict__ and "InternVL3" not in \
        Path(vlm_worker.__file__).read_text().split("DETERMINISM")[1].split("Hernán")[0] \
        .replace("the InternVL3 fallback is gone", "")

    @contextlib.contextmanager
    def down(pipe, config):
        raise svc.EngineUnavailable("VLM stage: vLLM did not come up within the startup bound")
        yield None

    monkeypatch.setattr(vlm_worker, "_ensure_semantic_service", down)
    with pytest.raises(svc.EngineUnavailable, match="startup bound"):
        vlm_worker._vlm_work(_Pipe(), str(tmp_path), cfg)
    off = json.loads(json.dumps(cfg))
    off["autoprompt"]["enabled"] = False
    with pytest.raises(RuntimeError, match="InternVL3 fallback is gone"):
        vlm_worker._vlm_work(_Pipe(), str(tmp_path), off)
