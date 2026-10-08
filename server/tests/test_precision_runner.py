"""The precision core as a pipeline stage: the one step list, in order; a step's record carries
the stamp of what it consumed and produced, and a re-run resumes at the first step whose stamp
differs (docs/plan_determinismo.md point 31) — the geometry-epoch number is not the key. Before a
GPU step the card is checked free (point 4; F2 counts as one when it must regenerate the DA3
windows, point 26); every step runs under the pinned OpenBLAS kernel set with the CPU recorded
(point 38); no wall clock in the state file (point 36)."""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

import pytest

import repro
from precision import runner as RN

# a fake step: reads <session>/in_<key>.txt (when present), writes <session>/out_<key>.txt with the
# input's text + its own marker; "pub" also publishes an epoch; "fail_<key>" makes it exit 3
_FAKE = (
    "import sys, json, pathlib; s = pathlib.Path(sys.argv[sys.argv.index('--session') + 1]); k = sys.argv[-1];"
    "(s / 'ran.txt').open('a').write(k + '\\n');"
    "src = s / ('in_' + k + '.txt'); txt = src.read_text() if src.exists() else '';"
    "(s / ('out_' + k + '.txt')).write_text(txt + '|' + k);"
    "p = s / 'output' / 'geometry_epoch.json';"
    "e = json.loads(p.read_text())['epoch'] if p.exists() else 0;"
    "p.write_text(json.dumps({'epoch': e + 1})) if k == 'pub' else None;"
    "sys.exit(3 if (s / ('fail_' + k)).exists() else 0)")


def _steps(tmp, gpu=()):
    (tmp / "fakestep.py").write_text(_FAKE)
    out = []
    for k in ("a", "pub", "c"):
        reads = (f"in_{k}.txt",) + (("out_a.txt",) if k == "pub" else ()) + (("out_pub.txt",) if k == "c" else ())
        out.append(RN.Step(k, k, "da3", "fakestep", (k,), gpu=k in gpu, reads=reads, writes=(f"out_{k}.txt",)))
    return out


@pytest.fixture
def pcfg(monkeypatch, tmp_path):
    from precision.config import load_precision_config
    p = load_precision_config()
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))      # the fake step module, for the subprocess
    monkeypatch.syspath_prepend(str(tmp_path))            # … and for the runner's own code stamp
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    # the card is not touched by these tests: the check is a free card unless a test says otherwise
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: {"cards": [], "compute_processes": 0})
    return replace(p, runner=replace(p.runner, python_da3=sys.executable,
                                     python_mapanything=sys.executable, threads=1))


def _session(tmp_path):
    s = tmp_path / "sess"
    (s / "output").mkdir(parents=True)
    (s / "in_a.txt").write_text("A0")
    return s


def _ran(s):
    return (s / "ran.txt").read_text().split()


def test_runs_in_order_and_resumes_after_a_failure(tmp_path, pcfg):
    s = _session(tmp_path)
    steps = _steps(tmp_path)
    (s / "fail_c").write_text("")
    with pytest.raises(RN.ChainError, match="resumes from this step"):
        RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s) == ["a", "pub", "c"]
    (s / "fail_c").unlink()
    rep = RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    # a and pub are not repeated: their inputs, code and config are what they were, their
    # products are what they left
    assert _ran(s) == ["a", "pub", "c", "c"]
    assert [r["key"] for r in rep["steps"]] == ["a", "pub", "c"] and rep["epoch"] == 1
    state = json.loads((s / "output" / "precision" / RN.STATE_NAME).read_text())
    assert state["version"] == RN.STATE_VERSION
    for r in state["done"]:
        assert "seconds" not in r and "stamp_in" in r and "stamp_out" in r       # point 36 / 31
        assert r["environment"]["blas"]["cpu_model"] and r["environment"]["blas"]["openblas_coretype"] == "HASWELL"
    timing = json.loads((s / "output" / "precision" / RN.TIMING_NAME).read_text())
    assert set(timing["steps"]) == {"a", "pub", "c"} and "last_chain_seconds" in timing
    assert state["done"][1]["stamp_out"]["inputs"]["out_pub.txt"] != RN.ABSENT
    assert state["done"][1]["epoch_before"] == 0 and state["done"][1]["epoch_after"] == 1


def test_a_changed_input_reruns_from_that_step_and_a_changed_product_from_its_writer(tmp_path, pcfg):
    s = _session(tmp_path)
    steps = _steps(tmp_path)
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    # the input of 'pub' changes (an evidence file appears): a is kept, pub and c run again
    (s / "in_pub.txt").write_text("new evidence")
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s) == ["a", "pub", "c", "pub", "c"]
    # the input of 'a' changes: everything runs again
    (s / "in_a.txt").write_text("A1")
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s)[-3:] == ["a", "pub", "c"]
    # nothing changed: nothing runs
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s)[-3:] == ["a", "pub", "c"] and len(_ran(s)) == 8
    # a product of 'pub' is edited by hand: it is not what the chain left — refused, naming it
    (s / "out_pub.txt").write_text("edited")
    with pytest.raises(RN.ChainError, match="out_pub.txt"):
        RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)


def test_a_product_restored_to_an_earlier_writers_bytes_resumes_after_that_writer(tmp_path, pcfg):
    """Plan point 31: an epoch undone back to byte-identical geometry resumes where those bytes
    left off — the epoch number is not the key."""
    s = _session(tmp_path)
    (tmp_path / "fakestep.py").write_text(_FAKE)
    # two writers of the same product: 'pub' writes shared.txt, then 'c' rewrites it
    steps = [RN.Step("a", "a", "da3", "fakestep", ("a",), reads=("in_a.txt",), writes=("out_a.txt",)),
             RN.Step("pub", "pub", "da3", "fakestep", ("pub",), reads=("out_a.txt",), writes=("out_pub.txt", "shared.txt")),
             RN.Step("c", "c", "da3", "fakestep", ("c",), reads=("out_pub.txt",), writes=("out_c.txt", "shared.txt"))]
    # fake 'shared.txt': pub leaves "P", c leaves "C" (simulated by the test around the steps)
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    state_p = s / "output" / "precision" / RN.STATE_NAME
    st = json.loads(state_p.read_text())
    # stamp the shared product as pub = "P" and c = "C" by rewriting the records' product stamps
    (s / "shared.txt").write_text("P")
    sha_p = repro.sha256_file(s / "shared.txt")
    (s / "shared.txt").write_text("C")
    sha_c = repro.sha256_file(s / "shared.txt")
    st["done"][1]["stamp_out"]["inputs"]["shared.txt"] = sha_p
    st["done"][2]["stamp_out"]["inputs"]["shared.txt"] = sha_c
    for rec in st["done"][1:]:
        body = {k: rec["stamp_out"][k] for k in ("stamp_version", "inputs", "code", "config")}
        rec["stamp_out"]["sha256"] = repro.sha256_json(body)
    state_p.write_text(json.dumps(st))
    # the session is "undone" to what pub left: c (the first writer after pub) runs again
    (s / "shared.txt").write_text("P")
    (s / "output" / "geometry_epoch.json").write_text(json.dumps({"epoch": 7}))   # another number
    logs = []
    RN.run_chain(s, pcfg, log=logs.append, steps=steps)
    assert _ran(s) == ["a", "pub", "c", "c"]
    assert any("a later epoch was undone" in m for m in logs)
    assert any("the number is not the key" in m for m in logs)


def test_a_changed_config_or_code_reruns_the_step(tmp_path, pcfg):
    s = _session(tmp_path)
    steps = _steps(tmp_path)
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    # the step's code (the fake module on PYTHONPATH is stamped as an external input) changes
    (tmp_path / "fakestep.py").write_text(_FAKE + "\n# v2\n")
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s) == ["a", "pub", "c"] * 2
    # the config of ONE step changes: only it and the steps after it run again
    steps2 = [replace(steps[0]), replace(steps[1], config=("gauge",)), replace(steps[2])]
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps2)
    assert _ran(s)[-2:] == ["pub", "c"] and len(_ran(s)) == 8
    p2 = replace(pcfg, gauge=replace(pcfg.gauge, knot_walk_m=pcfg.gauge.knot_walk_m * 2))
    RN.run_chain(s, p2, log=lambda *a: None, steps=steps2)
    assert _ran(s)[-2:] == ["pub", "c"] and len(_ran(s)) == 10
    # the thread count is part of every step's environment stamp
    p3 = replace(pcfg, runner=replace(pcfg.runner, threads=2))
    RN.run_chain(s, p3, log=lambda *a: None, steps=steps)
    assert len(_ran(s)) == 13


def test_gpu_steps_stop_the_chat_then_check_the_card_and_a_shared_card_fails_the_chain(tmp_path, pcfg, monkeypatch):
    """Plan point 4: before every GPU step the semantic service is stopped and the card checked
    free; a card in use fails the chain with the message — nothing is lowered to fit."""
    s = _session(tmp_path)
    steps = _steps(tmp_path, gpu=("pub",))
    calls = []
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: calls.append("gpu") or {"cards": []})
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=steps, before_gpu=lambda label: calls.append(f"stop:{label}"))
    assert calls == ["stop:pub", "gpu"]
    st = json.loads((s / "output" / "precision" / RN.STATE_NAME).read_text())
    assert [r["gpu"] for r in st["done"]] == [False, True, False]

    def busy(log=print):
        raise repro.ReproError("the GPU is NOT free — pid 4242 python 21000 MiB")
    monkeypatch.setattr(repro, "require_exclusive_gpu", busy)
    s2 = _session(tmp_path / "two")
    with pytest.raises(RN.ChainError, match="pid 4242"):
        RN.run_chain(s2, pcfg, log=lambda *a: None, steps=steps)
    assert _ran(s2) == ["a"]                                   # the GPU step never started


def test_f2_is_a_gpu_step_when_the_da3_windows_must_be_regenerated(tmp_path, pcfg, monkeypatch):
    """Plan point 26."""
    s = _session(tmp_path)
    (tmp_path / "fakestep.py").write_text(_FAKE)
    f2 = RN.Step("f2_gauge", "F2", "da3", "fakestep", ("f2",))
    import precision.gauge as G
    monkeypatch.setattr(G, "needs_window_regeneration", lambda sd: True)
    assert RN.step_needs_gpu(f2, s)
    monkeypatch.setattr(G, "needs_window_regeneration", lambda sd: False)
    assert not RN.step_needs_gpu(f2, s)
    assert RN.step_needs_gpu(replace(f2, gpu=True), s)
    monkeypatch.setattr(G, "needs_window_regeneration", lambda sd: True)
    calls = []
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: calls.append("gpu") or {})
    RN.run_chain(s, pcfg, log=lambda *a: None, steps=[f2], before_gpu=lambda label: calls.append("stop"))
    assert calls == ["stop", "gpu"]


def test_step_env_pins_the_blas_kernel_and_the_probe_verifies_it(pcfg, monkeypatch):
    """Plan point 38: OPENBLAS_CORETYPE pinned in every step's env, verified in the step's own
    interpreter (an unknown name is silently ignored by OpenBLAS), the CPU model recorded."""
    env = RN.step_env(3)
    assert env["OPENBLAS_CORETYPE"] == "HASWELL" and env["OMP_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3" and env["MKL_NUM_THREADS"] == "3"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == repro.CUBLAS_WORKSPACE_VALUE and env["PYTHONHASHSEED"] == "0"
    rec = RN.blas_probe(sys.executable, env, 3)
    assert rec["cpu_model"] == repro.cpu_model() and rec["openblas_coretype"] == "HASWELL"
    assert all(d["architecture"] == "Haswell" for d in rec["libraries"] if d["internal_api"] == "openblas")
    # a launch environment that disagrees on a deterministic variable is refused, not merged
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    with pytest.raises(repro.ReproError):
        RN.step_env(1)
    monkeypatch.delenv("PYTHONHASHSEED")
    # the probe refuses an interpreter whose OpenBLAS did not take the pin
    monkeypatch.setattr(RN, "OPENBLAS_ARCHITECTURE_REPORTED", "SkylakeX")
    with pytest.raises(RN.ChainError, match="ignored OPENBLAS_CORETYPE"):
        RN.blas_probe(sys.executable, env, 3)


def test_f5_runs_at_the_runners_threads_as_verified(pcfg):
    """Plan point 58: F5 keeps runner.threads — two solves at that count were bit-identical
    (tests/test_precision_refine.py, the mapanything env). A pin to 1 would be the next line."""
    f5 = next(s for s in RN.STEPS if s.key == "f5_refine")
    assert RN.step_threads(f5, pcfg.runner) == pcfg.runner.threads


def test_resume_point_without_stamps_keeps_the_epoch_rule(monkeypatch, tmp_path):
    """Callers without a session (no stamps): the pre-stamp rule — the prefix counts only in the
    epoch it left, or after published-cloud epochs (USER 2026-10-04)."""
    steps = [RN.Step("f5_refine", "F5", "da3", "m"), RN.Step("f6_bend", "F6", "da3", "m"), RN.Step("f6_check", "chk", "da3", "m")]
    state = {"done": [{"key": "f5_refine", "epoch_after": 2}]}
    import correction.epoch as CE
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "new_cloud" if e == 3 else "transform")
    assert RN.resume_point(state, steps, 3, out=tmp_path) == 1
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "transform")
    with pytest.raises(RN.ChainError):
        RN.resume_point(state, steps, 3, out=tmp_path)
    with pytest.raises(RN.ChainError):
        RN.resume_point(state, steps, 3)
    assert RN.resume_point(state, steps, 2) == 1


def test_resume_point_with_stamps_tolerates_published_clouds_only(monkeypatch, tmp_path):
    """A product changed by a NEW-CLOUD epoch (no camera moved) keeps the prefix; a transform
    epoch does not."""
    steps = [RN.Step("f6_bend", "F6", "da3", "m", writes=("output/cleaned_cloud.ply",)),
             RN.Step("f6_check", "chk", "da3", "m")]
    st_in = RN._files_stamp({}, [], {"x": 1})
    rec = {"key": "f6_bend", "epoch_after": 3, "stamp_in": st_in,
           "stamp_out": RN._with_inputs(st_in, {"output/cleaned_cloud.ply": "aaa"})}
    state = {"done": [rec]}
    import correction.epoch as CE
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "new_cloud")
    n = RN.resume_point(state, steps, 4, out=tmp_path, stamp_now=lambda s: st_in,
                        product_sha=lambda k: "bbb")
    assert n == 1
    monkeypatch.setattr(CE, "epoch_kind", lambda out, e: "transform")
    with pytest.raises(RN.ChainError, match="cleaned_cloud.ply"):
        RN.resume_point(state, steps, 4, out=tmp_path, stamp_now=lambda s: st_in, product_sha=lambda k: "bbb")
    # an older record (no stamp) is not reusable
    assert RN.resume_point({"done": [{"key": "f6_bend", "epoch_after": 3}]}, steps, 3, out=tmp_path,
                           stamp_now=lambda s: st_in, product_sha=lambda k: "aaa") == 0


def test_the_code_closure_stops_at_the_precision_core(pcfg):
    """F2's stamp holds the gauge and what it computes with, none of F4–F6's modules — an F6 edit
    never re-runs F2; F6's holds every module up to it. Every real step stamps real sections."""
    f2, _ = RN.code_closure("precision.gauge")
    names = {p.name for p in f2}
    assert {"gauge.py", "poses_epoch.py", "camera.py", "repro.py", "metric_lock.py", "epoch.py"} <= names
    assert not {"depth_on_f5.py", "refine.py", "tracks.py", "corrected_cloud.py"} & names
    f6, _ = RN.code_closure("precision.depth_on_f5")
    assert {"depth_on_f5.py", "refine.py", "tracks.py", "gauge.py", "corrected_cloud.py"} <= {p.name for p in f6}
    for s in RN.STEPS:
        cfg = RN.step_config(s, pcfg)
        assert cfg, s.key
        repro.canonical_json(cfg)                      # stampable
    with pytest.raises(RN.ChainError, match="not a section"):
        RN.step_config(RN.Step("x", "x", "da3", "m", config=("nope",)), pcfg)


def test_the_core_runs_inside_the_reconstruction_stage():
    """USER 2026-09-28: F0-F7 is part of the reconstruction, before the cloud stage;
    there is no separate PRECISION stage and no visit_drift step in the chain."""
    from pipeline_manager import DEFAULT_STAGE_ORDER, STAGE_REGISTRY, StageId
    assert not hasattr(StageId, "PRECISION")
    assert all("precision" not in v["module"] for v in STAGE_REGISTRY.values())
    assert DEFAULT_STAGE_ORDER.index(StageId.RECONSTRUCTION) < DEFAULT_STAGE_ORDER.index(StageId.CLOUDCOMPY)
    import workers.precision_worker as W
    assert callable(W.run)
    import inspect
    import workers.map_worker as M
    assert callable(M._run_precision_core) and not hasattr(M, "_run_semantics_2d")
    # no working cloud: the core merges / filters nothing before F7 (USER 2026-09-29)
    src = inspect.getsource(M._run_precision_core)
    assert "cloudcompy_worker" not in src and "merge" not in src.lower().replace("merged", "")
    for mod in ("precision.gauge", "precision.refine"):
        text = pathlib.Path(mod.replace(".", "/") + ".py").read_text()
        assert "apply_pose_epoch" in text and "apply_transform_epoch(" not in text, mod
    keys = [s.key for s in RN.STEPS]
    assert keys[0] == "f0_camera" and keys[-1] == "f6_check" and "f7_cloud" in keys, keys
    assert not any("measure" in k or "visit_drift" in s.module for k, s in zip(keys, RN.STEPS)), keys
    # the runner's own source: the exclusive-GPU check sits before every GPU step's launch
    text = pathlib.Path(RN.__file__).read_text()
    assert text.index("repro.require_exclusive_gpu(") < text.index("subprocess.Popen(cmd")
    assert "time.strftime" not in text
