"""Wave 2, package P5 — orchestration (docs/plan_determinismo.md points 111, 112, 149, 151,
152, 153, 154, 157, 158, 160, 162): the REAL pipeline manager driven on synthetic sessions,
CPU only.

- 151 / 112: the order's configuration is read ONCE and strictly; a job continuing a session
  frozen with another base configuration is refused before any stage; stage overrides do not
  count.
- 152 / 153: every stage process is spawned on the configured interpreter with the
  deterministic environment (hash seed, cuBLAS workspace, fixed threads, the launchers'
  interpreters); the job verifies its own interpreter and records the environment of the job
  and of every stage.
- 154: ONE engine permit per job — held from start to end, handed over only for the stages
  whose worker takes the engine itself (VLM, certification).
- 149: a resume / Autosegment cleans the previous run's VLM / SAM3 products before the stage
  runs, keeping the stamped carriers (vlm_analysis.json, autoprompt_concepts.json), the
  projection's result, the acta, cleaned_cloud.ply and the reconstruction.
- 162: the prompts travel with the job (snapshotted at enqueue, frozen, delivered at the stage).
- 158 / 160: the projection is a stage of the manager (registered, off in "Reconstruir", on
  demand through service_stages).
- 153: Omega's launcher runs the configured interpreter and fails without it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline_manager as PM  # noqa: E402
from pipeline_manager import (JobStatus, OrderRefused, PipelineJob, PipelineManager,  # noqa: E402
                              PipelineStage, StageId, StageState, build_pipeline_stages,
                              read_config_once, select_stages, service_stages)

SERVER_DIR = Path(__file__).resolve().parents[1]


# ── fixtures ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def lease(tmp_path, monkeypatch):
    """The engine lease file in tmp (never the backend's logs/)."""
    import semantic.service as svc
    p = tmp_path / "lease.json"
    monkeypatch.setattr(svc, "lease_path", lambda: p)
    yield p
    svc.release_engine_lease()


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


def _run(pm: PipelineManager, job: PipelineJob, config: dict, *, replace=False, force=False):
    done = {}

    async def _complete(sid, ok):
        done["ok"] = ok

    async def _body():
        pm._jobs[pm.job_key(job.session_id, job.scan_key, job.kind)] = job
        await pm._run_pipeline(job, job.session_dir, config, None, _complete, replace, force)

    asyncio.run(_body())
    return done.get("ok")


# ── 151 / 112: one configuration, read once ──────────────────────────────────────────

def test_read_config_once_is_strict(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\nb:\n  c: [1, 2]\n")
    assert read_config_once(p) == {"a": 1, "b": {"c": [1, 2]}}
    p.write_text("a: [unclosed\n")
    with pytest.raises(OrderRefused, match="not readable YAML"):
        read_config_once(p)
    p.write_text("- just\n- a list\n")
    with pytest.raises(OrderRefused, match="not a mapping"):
        read_config_once(p)
    with pytest.raises(OrderRefused, match="cannot be read"):
        read_config_once(tmp_path / "missing.yaml")


def test_a_continuation_with_another_base_configuration_is_refused_before_any_stage(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    pm = PipelineManager()
    ran = []

    async def _stage(job, st, session_dir, config, on_progress, replace):
        ran.append(st.stage.id)
        return True

    monkeypatch.setattr(pm, "_run_stage", _stage)
    cfg = _base_config()
    job = _job(s, [StageId.CLOUDCOMPY])
    assert _run(pm, job, cfg) is True and ran == [StageId.CLOUDCOMPY]
    sha1 = (s / "output" / "run_config.sha256").read_text().strip()

    # the same base configuration with a STAGE override: allowed (the override is the job's;
    # the frozen file carries it under _stages, so its digest moves — the base did not)
    job2 = _job(s, [StageId.CERTIFY])
    job2.stages[0].stage.config["segmentation"] = {"object_captions": {"enabled": False}}
    ran.clear()
    assert _run(pm, job2, cfg) is True and ran == [StageId.CERTIFY]
    sha2 = (s / "output" / "run_config.sha256").read_text().strip()
    assert sha2 != sha1
    frozen = yaml.safe_load((s / "output" / "run_config.yaml").read_text())
    assert frozen["_stages"]["certify"]["segmentation"]["object_captions"]["enabled"] is False
    assert frozen["segmentation"]["object_captions"]["enabled"] is True, "the base is untouched"

    # another base configuration: refused, naming the section, nothing runs
    other = _base_config(pipeline={"auto_segment": True, "auto_tsdf": True})
    job3 = _job(s, [StageId.CERTIFY])
    ran.clear()
    assert _run(pm, job3, other) is False
    assert ran == [] and job3.status == JobStatus.FAILED
    assert "ONE configuration" in job3.stages[0].message and "pipeline" in job3.stages[0].message
    assert (s / "output" / "run_config.sha256").read_text().strip() == sha2, "nothing re-frozen"

    # Replace starts the session over on the new configuration
    job4 = _job(s, [StageId.CLOUDCOMPY])
    ran.clear()
    assert _run(pm, job4, other, replace=True) is True and ran == [StageId.CLOUDCOMPY]
    assert (s / "output" / "run_config.sha256").read_text().strip() != sha1


def test_an_edited_frozen_file_refuses_the_continuation(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    pm = PipelineManager()
    monkeypatch.setattr(pm, "_run_stage", _ok_stage)
    cfg = _base_config()
    assert _run(pm, _job(s, [StageId.CLOUDCOMPY]), cfg) is True
    (s / "output" / "run_config.yaml").write_text("pipeline: {auto_segment: false}\n")
    job = _job(s, [StageId.CERTIFY])
    assert _run(pm, job, cfg) is False
    assert "edited after it was frozen" in job.stages[0].message


async def _ok_stage(job, st, session_dir, config, on_progress, replace):
    return True


# ── 153 / 152: the interpreter and the environment record ────────────────────────────

def test_the_job_verifies_its_interpreter_and_the_configured_ones_exist(tmp_path):
    cfg = _base_config()
    assert PM.verify_own_interpreter(cfg["reconstruction"]["precision"]["runner"]["python_da3"])
    with pytest.raises(OrderRefused, match="configured stage interpreter"):
        PM.verify_own_interpreter("/usr/bin/env")
    bad = _base_config()
    bad["reconstruction"]["precision"]["runner"]["python_mapanything"] = "/nonexistent/python"
    with pytest.raises(OrderRefused, match="python_mapanything"):
        PM.runner_config(bad)
    rel = _base_config()
    rel["reconstruction"]["precision"]["runner"]["python_da3"] = "python"
    with pytest.raises(OrderRefused, match="absolute"):
        PM.runner_config(rel)
    nothr = _base_config()
    del nothr["reconstruction"]["precision"]["runner"]["threads"]
    with pytest.raises(OrderRefused, match="threads"):
        PM.runner_config(nothr)


def test_stage_environment_is_the_chains_deterministic_env_plus_the_launchers(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    rcfg = PM.runner_config(_base_config())
    delta = PM.stage_environment(rcfg, StageId.SAM3)
    assert all(os.environ.get(k) != v for k, v in delta.items()), "only what the stage gets on top"
    env = {**os.environ, **delta}                      # what the spawned stage sees
    assert env["PYTHONHASHSEED"] == "0" and env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_NUM_THREADS"):
        assert env[k] == "2", k
    assert env["OPENBLAS_CORETYPE"] == "HASWELL" and env["MKL_CBWR"] == "COMPATIBLE"
    assert env["STAC_PYTHON_MAPANYTHING"] == sys.executable
    assert env["STAC_PYTHON_DA3"] == sys.executable
    # a launcher environment that disagrees on a deterministic variable is REFUSED
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    import repro
    with pytest.raises(repro.ReproError):
        PM.stage_environment(rcfg, StageId.SAM3)


def test_a_real_stage_process_runs_on_the_configured_interpreter_with_the_env(tmp_path, monkeypatch):
    """The REAL _run_stage: a fake worker module spawned like every stage writes what it
    was given — interpreter, hash seed, cuBLAS workspace, threads, the launchers."""
    mod = tmp_path / "p5_fake_worker.py"
    mod.write_text(textwrap.dedent("""
        import json, os, sys
        from workers.base import run_worker_safe

        def _work(pipe, session_dir, config):
            keys = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENCV_NUM_THREADS",
                    "STAC_PYTHON_MAPANYTHING", "STAC_PYTHON_DA3", "OPENBLAS_CORETYPE")
            rec = {k: os.environ.get(k) for k in keys}
            rec["executable"] = sys.executable
            rec["hash_order"] = sorted(set("desk wall floor chair door".split()))
            with open(config["_p5_out"], "w") as fh:
                json.dump(rec, fh)
            pipe.send_log("fake stage ran")

        def run(conn, session_dir, config):
            run_worker_safe(_work, conn, session_dir, config)
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setitem(PM.STAGE_REGISTRY, StageId.CLOUDCOMPY,
                        {"label": "fake", "icon": "x", "module": "p5_fake_worker"})
    s = _session(tmp_path)
    out = tmp_path / "env.json"
    cfg = _base_config(_p5_out=str(out))
    pm = PipelineManager()
    job = _job(s, [StageId.CLOUDCOMPY])
    ok = asyncio.run(pm._run_stage(job, job.stages[0], str(s), cfg, None, False))
    assert ok is True, job.stages[0].message
    rec = json.loads(out.read_text())
    assert rec["executable"] == sys.executable
    assert rec["PYTHONHASHSEED"] == "0" and rec["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rec["OMP_NUM_THREADS"] == rec["OPENBLAS_NUM_THREADS"] == rec["MKL_NUM_THREADS"] == "2"
    assert rec["OPENCV_NUM_THREADS"] == "2" and rec["OPENBLAS_CORETYPE"] == "HASWELL"
    assert rec["STAC_PYTHON_MAPANYTHING"] == sys.executable
    # the server's own environment was restored after the spawn
    assert "STAC_PYTHON_MAPANYTHING" not in os.environ and "OMP_NUM_THREADS" in os.environ
    # the stage was recorded with the variables it ran with
    env_rec = json.loads((s / "output" / PM.RUN_ENVIRONMENT_NAME).read_text())
    st = env_rec["stages"]["cloudcompy"]
    assert st["interpreter"] == sys.executable and st["module"] == "p5_fake_worker"
    assert st["env"]["PYTHONHASHSEED"] == "0" and st["env"]["STAC_PYTHON_MAPANYTHING"] == sys.executable


def test_the_job_records_its_environment_before_the_stages(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    pm = PipelineManager()
    monkeypatch.setattr(pm, "_run_stage", _ok_stage)
    assert _run(pm, _job(s, [StageId.CLOUDCOMPY]), _base_config()) is True
    rec = json.loads((s / "output" / PM.RUN_ENVIRONMENT_NAME).read_text())
    assert rec["job"]["python_da3"] == sys.executable and rec["job"]["threads"] == 2
    assert rec["job"]["run_config_sha256"] == (s / "output" / "run_config.sha256").read_text().strip()
    assert rec["manager"]["python_executable"] == sys.executable
    assert rec["manager"]["git"]["repo"]["commit"] and rec["manager"]["blas"]
    assert not any(k in rec["manager"] for k in ("pid", "started_at", "hostname")), "no clock, no pid"


# ── 154: one engine permit per job ───────────────────────────────────────────────────

def test_the_manager_holds_the_lease_for_the_job_and_hands_it_to_engine_stages(tmp_path, lease, monkeypatch):
    import semantic.service as svc
    s = _session(tmp_path)
    pm = PipelineManager()
    seen = {}

    async def _stage(job, st, session_dir, config, on_progress, replace):
        h = svc.engine_lease_holder()
        seen[st.stage.id] = None if h is None else (h["pid"], h["owner"], h["stage"])
        return True

    monkeypatch.setattr(pm, "_run_stage", _stage)
    job = _job(s, [StageId.CLOUDCOMPY, StageId.VLM, StageId.SAM3, StageId.CERTIFY])
    assert _run(pm, job, _base_config()) is True
    me = os.getpid()
    assert seen[StageId.CLOUDCOMPY] == (me, "pipeline", "job"), "held by the manager"
    assert seen[StageId.SAM3] == (me, "pipeline", "job"), "held again between engine stages"
    assert seen[StageId.VLM] is None, "handed over: the VLM worker takes it under its own pid"
    assert seen[StageId.CERTIFY] is None, "handed over: the description pass takes it"
    assert svc.engine_lease_holder() is None, "released at the job's end"
    assert svc.engine_available_to(pid=1)[0]


def test_a_live_foreign_holder_refuses_the_job_before_any_stage(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    lease.write_text(json.dumps({"pid": os.getppid(), "owner": "other", "stage": "VLM stage"}))
    pm = PipelineManager()
    ran = []

    async def _stage(job, st, session_dir, config, on_progress, replace):
        ran.append(st.stage.id)
        return True

    monkeypatch.setattr(pm, "_run_stage", _stage)
    job = _job(s, [StageId.CLOUDCOMPY])
    assert _run(pm, job, _base_config()) is False and ran == []
    assert "leased to pid" in job.stages[0].message
    lease.unlink()


def test_the_lease_goes_with_a_failed_job(tmp_path, lease, monkeypatch):
    import semantic.service as svc
    s = _session(tmp_path)
    pm = PipelineManager()

    async def _stage(job, st, session_dir, config, on_progress, replace):
        assert svc.engine_lease_holder()["pid"] == os.getpid()
        return False

    monkeypatch.setattr(pm, "_run_stage", _stage)
    job = _job(s, [StageId.CLOUDCOMPY])
    assert _run(pm, job, _base_config()) is False
    assert svc.engine_lease_holder() is None


# ── 149: the resume cleanup ───────────────────────────────────────────────────────────

def test_resume_cleanup_keeps_the_stamped_carriers_the_cloud_and_the_reconstruction(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    keep = ["vlm_analysis.json", "autoprompt_concepts.json", "cleaned_cloud.ply", "autosegment.json",
            "camera_poses.txt", "certify_acta.json", "chunk_plan.json", "geometry_epoch.json"]
    gone = ["scene_understanding.json", "autoprompt_review_queue.json", "vlm_analysis.timing.json",
            "segmentation.json", "seg_masks.npz", "segmentation_result.json", "seg_broadcast.json",
            "classification.npy", "instance_ids.npy", "class_map.json", "scene_r.db", "fusion_map.json",
            "segmentation_census.json", "potree_stamp.json", "mask_audit.json", "out_of_place.npy"]
    for n in keep + gone:
        (out / n).write_text("x")
    (out / "_sam3_raw").mkdir()
    (out / "_sam3_store.staging").mkdir()
    (out / "omega_run").mkdir()
    (out / "quality").mkdir()
    (out / "potree").mkdir()
    (out / "potree" / "metadata.json").write_text("{}")
    removed = PipelineManager._cleanup_for_resume(out, StageId.VLM)
    for n in keep:
        assert (out / n).exists(), n
    for n in gone:
        assert not (out / n).exists(), n
    assert not (out / "_sam3_raw").exists() and not (out / "_sam3_store.staging").exists()
    for d in ("omega_run", "quality", "potree"):
        assert (out / d).exists(), d
    assert set(removed) == set(gone) | {"_sam3_raw", "_sam3_store.staging"}
    # a SAM3 run cleans SAM3's products only (vlm's side files stay)
    (out / "scene_understanding.json").write_text("x")
    (out / "segmentation.json").write_text("x")
    removed = PipelineManager._cleanup_for_resume(out, StageId.SAM3)
    assert "segmentation.json" in removed and (out / "scene_understanding.json").exists()
    # the projection and the certification clean nothing (their stamps decide the reuse)
    for sid in (StageId.PROJECTION, StageId.CERTIFY, StageId.RECONSTRUCTION, StageId.CLOUDCOMPY):
        assert PipelineManager._cleanup_for_resume(out, sid) == []
        assert (out / "cleaned_cloud.ply").exists()


def test_the_resume_cleanup_runs_before_the_stage_only_when_nothing_was_wiped(tmp_path, lease, monkeypatch):
    s = _session(tmp_path)
    out = s / "output"
    (out / "segmentation.json").write_text("old")
    (out / "vlm_analysis.json").write_text("{}")
    pm = PipelineManager()
    seen = {}

    async def _stage(job, st, session_dir, config, on_progress, replace):
        seen[st.stage.id] = sorted(p.name for p in out.iterdir())
        return True

    monkeypatch.setattr(pm, "_run_stage", _stage)
    assert _run(pm, _job(s, [StageId.SAM3], kind="autosegment"), _base_config(), force=True) is True
    assert "segmentation.json" not in seen[StageId.SAM3] and "vlm_analysis.json" in seen[StageId.SAM3]


# ── 162: the prompts travel with the job ──────────────────────────────────────────────

def test_the_prompts_are_snapshotted_at_enqueue_frozen_and_delivered_at_the_stage(tmp_path, lease, monkeypatch):
    from segmentation.autoprompt import autosegment as AS
    s = _session(tmp_path)
    out = s / "output"
    AS.save_vlm_prompt(out, "Name every object. JSON.")
    AS.set_sam3_prompts(out, ["floor", "wall"])
    stages = select_stages([PipelineStage(id=StageId.SAM3, enabled=True),
                            PipelineStage(id=StageId.CERTIFY, enabled=True)])
    inputs = PM.snapshot_job_inputs(str(s), stages, replace=False)
    assert inputs == {"sam3_prompts": ["floor", "wall"]}, "SAM3 without a VLM pass: the list"
    both = select_stages([PipelineStage(id=StageId.VLM, enabled=True),
                          PipelineStage(id=StageId.SAM3, enabled=True)])
    assert PM.snapshot_job_inputs(str(s), both, replace=False) == {"vlm_prompt": "Name every object. JSON."}
    assert PM.snapshot_job_inputs(str(s), both, replace=True) == {}, "a wipe carries nothing"
    PM.apply_job_inputs(stages, inputs)
    assert stages[0].config["sam3_prompts"] == ["floor", "wall"] and stages[1].config == {}

    # the job is enqueued, then ANOTHER job rewrites the session's prompts before it runs
    pm = PipelineManager()
    seen = {}

    async def _stage(job, st, session_dir, config, on_progress, replace):
        seen[st.stage.id] = AS.sam3_prompts(Path(session_dir) / "output")
        return True

    monkeypatch.setattr(pm, "_run_stage", _stage)
    job = PipelineJob(session_id=s.name, stages=[StageState(stage=st) for st in stages], kind="autosegment")
    job.session_dir = str(s)
    job.inputs = inputs
    AS.set_sam3_prompts(out, ["door"])                          # what a VLM pass of another job did
    assert _run(pm, job, _base_config(), force=True) is True
    assert seen[StageId.SAM3] == ["floor", "wall"], "the stage read the job's frozen list"
    frozen = yaml.safe_load((out / "run_config.yaml").read_text())
    assert frozen["_stages"]["sam3"]["sam3_prompts"] == ["floor", "wall"]


def test_deliver_restores_the_shipped_vlm_prompt_when_the_job_carried_none(tmp_path):
    from segmentation.autoprompt import autosegment as AS
    out = tmp_path / "output"
    out.mkdir()
    AS.save_vlm_prompt(out, "saved after the order")
    lines = PM.deliver_job_inputs(out, StageId.VLM, {"vlm_prompt": None})
    assert lines and AS.vlm_prompt_for(out) == (AS.default_vlm_prompt(), False)
    assert PM.deliver_job_inputs(out, StageId.VLM, {"vlm_prompt": None}) == [], "nothing to deliver twice"


# ── 158 / 160: the projection stage; the stage lists ─────────────────────────────────

def test_the_projection_stage_is_registered_off_in_reconstruir_and_on_demand():
    assert PM.DEFAULT_STAGE_ORDER.index(StageId.PROJECTION) == PM.DEFAULT_STAGE_ORDER.index(StageId.SAM3) + 1
    reg = PM.STAGE_REGISTRY[StageId.PROJECTION]
    assert reg["module"] == "pipeline_manager" and callable(getattr(PM, reg["entry"]))
    assert PM.STAGE_REGISTRY[StageId.VLM]["engine"] and PM.STAGE_REGISTRY[StageId.CERTIFY]["engine"]
    assert not PM.STAGE_REGISTRY[StageId.SAM3].get("engine") and not reg.get("engine")
    stages = build_pipeline_stages(backend="vggtomega", config=_base_config())
    assert not next(s for s in stages if s.id == StageId.PROJECTION).enabled
    assert [s.id for s in select_stages(stages, segment=False) if s.enabled] == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY]
    svc = service_stages({StageId.CERTIFY, StageId.PROJECTION})
    assert [s.id for s in svc] == [StageId.PROJECTION, StageId.CERTIFY] and all(s.enabled for s in svc)
    with pytest.raises(ValueError):
        service_stages({"nope"})
    assert StageId.CERTIFY in PipelineManager.CASCADE_INVALIDATION[StageId.PROJECTION]
    assert "potree_stamp.json" in PipelineManager.STAGE_OUTPUT_FILES[StageId.SAM3]
    assert "vlm_analysis.timing.json" in PipelineManager.STAGE_OUTPUT_FILES[StageId.VLM]
    assert "object_captions.timing.json" in PipelineManager.STAGE_OUTPUT_FILES[StageId.CERTIFY]


def test_build_pipeline_stages_reads_the_orders_configuration_not_the_servers():
    off = build_pipeline_stages(backend="vggtomega", config=_base_config(pipeline={"auto_segment": False, "auto_tsdf": False}))
    assert [s.id for s in off if s.enabled] == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY]
    on = build_pipeline_stages(backend="vggtomega", config=_base_config())
    assert [s.id for s in on if s.enabled] == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY, StageId.VLM,
                                                StageId.SAM3, StageId.CERTIFY]


def test_projection_probe_and_the_job_config_reach_the_reconstruction_probe(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.PROJECTION, _base_config())
    assert not done and "no masks" in why
    (out / "segmentation.json").write_text("{}")
    (out / "seg_masks.npz").write_bytes(b"x")
    done, why = PipelineManager._stage_is_complete(out, tmp_path, StageId.PROJECTION, _base_config())
    assert not done and why.startswith("masks on disk but no projection")
    (out / "segmentation_result.json").write_text("{}")
    assert PipelineManager._stage_is_complete(out, tmp_path, StageId.PROJECTION, _base_config())[0]
    # the PGSR probe reads the job's backend, never config.yaml
    cfg = _base_config()
    cfg["reconstruction"]["backend"] = "vggtomega_pgsr"
    assert not PipelineManager._stage_is_complete(out, tmp_path, StageId.PGSR, cfg)[0]
    assert PipelineManager._stage_is_complete(out, tmp_path, StageId.PGSR, _base_config())[0]


def test_replace_declares_the_discarded_vlm_prompt(tmp_path, caplog):
    import logging
    s = _session(tmp_path)
    (s / "frames" / "000000.jpg").write_bytes(b"jpg")
    (s / "output" / "autosegment.json").write_text('{"vlm_prompt": "x"}')
    with caplog.at_level(logging.INFO, logger="pipeline_manager"):
        PipelineManager._wipe_outputs_for_replace(s, s / "output")
    assert any("discards the session's saved VLM prompt" in r.getMessage() for r in caplog.records)
    assert not (s / "output" / "autosegment.json").exists()


# ── 153: Omega's launcher ─────────────────────────────────────────────────────────────

def test_run_mapanything_fails_without_the_configured_interpreter_and_runs_it_when_given(tmp_path):
    script = SERVER_DIR / "run_mapanything.sh"
    env = dict(os.environ, STAC_PYTHON_MAPANYTHING="/nonexistent/python", CONDA_ROOT="/nonexistent")
    r = subprocess.run(["bash", str(script), "--x"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "no fallback" in r.stderr and "/nonexistent/python" in r.stderr
    # no PATH python either: the conda default under a CONDA_ROOT that does not exist
    env2 = dict(os.environ, CONDA_ROOT="/nonexistent")
    env2.pop("STAC_PYTHON_MAPANYTHING", None)
    r = subprocess.run(["bash", str(script), "--x"], env=env2, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "python_mapanything" in r.stderr
    # the configured interpreter runs vggt_long.py with the arguments, by absolute path
    fake = tmp_path / "python"
    fake.write_text("#!/bin/bash\necho \"FAKE $@\"\n")
    fake.chmod(0o755)
    env3 = dict(os.environ, STAC_PYTHON_MAPANYTHING=str(fake), CONDA_ROOT="/nonexistent")
    r = subprocess.run(["bash", str(script), "--arg", "1"], env=env3, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"interpreter: {fake}" in r.stdout
    assert "FAKE -u " in r.stdout and r.stdout.strip().endswith("vggt_long.py --arg 1")
    assert "python -u" not in (SERVER_DIR / "run_mapanything.sh").read_text().replace('"${PY}" -u', "")


def test_start_sh_exports_the_deterministic_environment():
    src = (SERVER_DIR.parent / "scripts" / "start.sh").read_text()
    assert "export PYTHONHASHSEED=0" in src and "export CUBLAS_WORKSPACE_CONFIG=:4096:8" in src
    assert src.index("export PYTHONHASHSEED=0") < src.index("exec python -m uvicorn")
