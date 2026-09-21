"""FINDING 23 + FINDING 21, both opened by the same run on 2026-09-21 10:00.

A run_pipeline was sent with replace=False to relaunch from the VLM. The
manager enables ALL stages, so RECONSTRUCTION was in the list, and STARTING it
wiped its own outputs — cleaned_cloud.ply, cleaned_cloud_raw.ply,
camera_poses.txt, intrinsic.txt, omega_run/, da3_run/, chunk_plan.json,
scale_diagnostics.json, the idempotency markers AND frames/selected_frames.json.
Cancelled 40 s later: too late, 3.5 h of reconstruction gone. The side effect
that gave it away: without selected_frames.json SAM3 fell back to "all frames"
and started segmenting 3,044 video frames in 12 batches instead of the 216
keyframes — 14x the work, with masks in video-frame space.

These tests pin the two properties that make that run impossible:
  * replace=False deletes NOTHING, and a stage that cannot re-run without
    rewriting a FINISHED predecessor refuses and names the flag (replace) that
    authorises it;
  * a caller can name the stages ("desde cloudcompy para abajo sin incluir
    claro a cloudcompy"), the names are validated loudly, and the resume probe
    of the reconstruction does not depend on the chunk_*.ply CloudCompy deletes.

No GPU, no worker: _run_stage is stubbed, so what runs here is the manager's
own bookkeeping.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_manager import (  # noqa: E402
    DEFAULT_STAGE_ORDER,
    JobStatus,
    PipelineManager,
    PipelineSelectionError,
    PipelineStage,
    StageId,
    build_pipeline_stages,
)

BACKEND = "vggtomega_pgsr"   # server/config.yaml reconstruction.backend


# ── a session that reconstructed, was cleaned, and was segmented ──

def _finished_session(tmp_path: Path) -> Path:
    """The shape of the pccr scan the incident destroyed: poses + depth + the
    chunk plan in output/, the cleaned cloud CloudCompy baked (and the
    chunk_*.ply it deleted on its way out), and the keyframe list SAM3 reads."""
    session = tmp_path / "scan"
    out = session / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    (out / "omega_run" / "camera_poses.txt").write_text("1 0 0 0\n")
    (out / "camera_poses.txt").write_text("1 0 0 0\n")
    (out / "intrinsic.txt").write_text("500 0 320\n")
    (out / "chunk_plan.json").write_text('{"chunks": 7}')
    (out / "scale_diagnostics.json").write_text('{"ratio": 1.1401}')
    (out / "cleaned_cloud.ply").write_bytes(b"ply")
    (out / "cleaned_cloud_raw.ply").write_bytes(b"ply")
    (out / ".metric_scale_applied").write_text("")
    (out / "vlm_analysis.json").write_text('{"prompt": "wall; floor"}')
    (out / "segmentation.json").write_text('{"prompts": ["wall", "floor"]}')
    (out / "seg_masks.npz").write_bytes(b"npz")
    # the mask->cloud mapping CloudCompy owns: without it its probe reads
    # "mask->cloud mapping pending" and the whole tail re-runs
    (out / "segmentation_result.json").write_text('{"instances": []}')
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "selected_frames.json").write_text('{"selected_files": ["f0.jpg"]}')
    (frames / "frame_quality.json").write_text("{}")
    return session


def _snapshot(session: Path) -> set:
    return {str(p.relative_to(session)) for p in session.rglob("*")}


def _run(manager: PipelineManager, session: Path, stages, replace=False):
    """Drive _run_pipeline with the stage worker stubbed out. Returns the job
    and the list of stages that actually reached the worker."""
    from pipeline_manager import PipelineJob, StageState

    ran = []
    job = PipelineJob(session_id="s", stages=[StageState(stage=s) for s in stages])

    async def _fake_stage(job_, stage_state, session_dir, config, on_progress, replace_):
        ran.append(stage_state.stage.id)
        return True

    manager._run_stage = _fake_stage   # type: ignore[assignment]
    asyncio.run(manager._run_pipeline(job, str(session), {}, None, None, replace))
    return job, ran


# ── FINDING 23: replace=False deletes nothing ─────────────────

def test_replace_false_deletes_nothing():
    """The incident, replayed: every stage enabled, replace omitted. Not one
    byte of the finished session may disappear before a stage runs."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        before = _snapshot(session)
        stages = build_pipeline_stages(backend=BACKEND)
        _run(PipelineManager(), session, stages, replace=False)
        assert _snapshot(session) == before, (
            "replace=False deleted " + str(sorted(before - _snapshot(session))))


def test_finished_reconstruction_refuses_a_replaceless_rerun():
    """Naming the reconstruction on a finished session is the destructive case:
    map_worker rewrites camera_poses.txt / omega_run / the keyframe list in
    place. It must stop, say what it would have taken, and name `replace`."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        before = _snapshot(session)
        stages = build_pipeline_stages(backend=BACKEND, stages=["reconstruction"])
        job, ran = _run(PipelineManager(), session, stages, replace=False)
        assert ran == [], "the stage must not start"
        assert job.status == JobStatus.FAILED
        msg = job.stages[0].message
        assert "camera_poses.txt" in msg and "frames/selected_frames.json" in msg
        assert "replace=true" in msg
        assert _snapshot(session) == before


def test_a_half_finished_reconstruction_still_resumes():
    """The refusal protects a FINISHED stage, never a crashed one: resuming a
    reconstruction that never wrote its poses is the point of resume mode."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        for name in ("camera_poses.txt", "omega_run/camera_poses.txt"):
            (session / "output" / name).unlink()
        assert PipelineManager._destructive_conflict(
            session / "output", session, StageId.RECONSTRUCTION) == []
        stages = build_pipeline_stages(backend=BACKEND, stages=["reconstruction"])
        job, ran = _run(PipelineManager(), session, stages, replace=False)
        assert ran == [StageId.RECONSTRUCTION] and job.status == JobStatus.DONE


def test_selected_frames_is_nobody_output():
    """frames/selected_frames.json is the INPUT of the segmentation. It must not
    appear in any cleanup list: deleting it is what sent SAM3 over 3,044 video
    frames instead of 216 keyframes."""
    assert "selected_frames.json" not in PipelineManager.FRAMES_DIR_FILES
    for sid, patterns in PipelineManager.STAGE_OUTPUT_FILES.items():
        assert "selected_frames.json" not in patterns, sid
    # the seg-side derived lists and the quality manifest DO stay regenerable
    assert "selected_frames_seg*.json" in PipelineManager.FRAMES_DIR_FILES
    assert "frame_quality.json" in PipelineManager.FRAMES_DIR_FILES


def test_replace_true_still_wipes_the_whole_output():
    """Pre-cleaning is what replace MEANS — the flag keeps its teeth."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        stages = build_pipeline_stages(backend=BACKEND)
        _run(PipelineManager(), session, stages, replace=True)
        assert not (session / "output" / "cleaned_cloud.ply").exists()
        assert not (session / "output" / "camera_poses.txt").exists()
        # map_worker rewrites the keyframe list on a replace run; the manager
        # no longer deletes it out from under a cancelled one
        assert (session / "frames" / "selected_frames.json").exists()


def test_selective_replace_cleans_only_what_it_reruns():
    """Replace + a partial selection must not wipe output/ wholesale: that
    would destroy exactly what the caller did not ask to redo."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        stages = build_pipeline_stages(backend=BACKEND, from_stage="vlm")
        job, ran = _run(PipelineManager(), session, stages, replace=True)
        out = session / "output"
        assert (out / "cleaned_cloud.ply").exists(), "the cloud was not selected"
        assert (out / "camera_poses.txt").exists()
        assert not (out / "vlm_analysis.json").exists(), "the VLM was selected"
        assert StageId.VLM in ran and StageId.RECONSTRUCTION not in ran


# ── FINDING 21: naming the stages ─────────────────────────────

def test_no_selection_is_todays_behaviour():
    """Backward compatibility: the historical call is untouched — the whole
    chain, gated by the config switches, and nothing marked explicit."""
    stages = build_pipeline_stages(backend=BACKEND)
    ids = [s.id for s in stages]
    assert ids == [s for s in DEFAULT_STAGE_ORDER]      # pgsr backend: full order
    enabled = [s.id for s in stages if s.enabled]
    assert enabled[:5] == [StageId.RECONSTRUCTION, StageId.CLOUDCOMPY,
                           StageId.VLM, StageId.SAM3, StageId.CERTIFY]
    assert all(not s.explicit for s in stages)


def test_from_stage_runs_the_tail():
    """USER 2026-09-21: "desde cloudcompy para abajo sin incluir claro a
    cloudcompy" — that is from_stage=vlm."""
    stages = build_pipeline_stages(backend=BACKEND, from_stage="vlm")
    enabled = [s.id for s in stages if s.enabled]
    assert enabled[0] == StageId.VLM
    assert StageId.RECONSTRUCTION not in enabled and StageId.CLOUDCOMPY not in enabled
    assert StageId.SAM3 in enabled and StageId.CERTIFY in enabled
    assert all(s.explicit for s in stages if s.enabled)
    assert all(not s.explicit for s in stages if not s.enabled)


def test_stages_selection_is_sorted_into_pipeline_order():
    stages = build_pipeline_stages(backend=BACKEND, stages=["certify", "sam3"])
    assert [s.id for s in stages if s.enabled] == [StageId.SAM3, StageId.CERTIFY]
    # a StageId is as good as its name
    same = build_pipeline_stages(backend=BACKEND,
                                 stages=[StageId.CERTIFY, StageId.SAM3])
    assert [s.id for s in same if s.enabled] == [StageId.SAM3, StageId.CERTIFY]


def test_an_unknown_stage_fails_loudly():
    with pytest.raises(PipelineSelectionError, match="unknown stage 'segmentation'"):
        build_pipeline_stages(backend=BACKEND, stages=["segmentation"])
    with pytest.raises(PipelineSelectionError, match="unknown stage"):
        build_pipeline_stages(backend=BACKEND, from_stage="cloud")
    with pytest.raises(PipelineSelectionError, match="mutually exclusive"):
        build_pipeline_stages(backend=BACKEND, stages=["vlm"], from_stage="vlm")
    with pytest.raises(PipelineSelectionError, match="empty stage selection"):
        build_pipeline_stages(backend=BACKEND, stages=[])
    # PGSR exists only under vggtomega_pgsr — asking for it elsewhere is not a
    # silent no-op, it is a refusal naming the stages this backend does run
    with pytest.raises(PipelineSelectionError, match="not part of this pipeline"):
        build_pipeline_stages(backend="da3", stages=["pgsr"])


def test_a_named_stage_outvotes_the_config_switch():
    """`pipeline.auto_tsdf: false` configures the AUTOMATIC chain. A caller who
    types "pgsr" has already decided; the override is honoured and logged."""
    auto = build_pipeline_stages(backend=BACKEND)
    assert StageId.PGSR not in [s.id for s in auto if s.enabled]
    picked = build_pipeline_stages(backend=BACKEND, stages=["pgsr"])
    assert [s.id for s in picked if s.enabled] == [StageId.PGSR]


def test_a_named_stage_skips_the_resume_probe():
    """The selection IS the resume decision: SAM3 is complete on disk and the
    automatic chain would resume past it, but a caller who named it gets it."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        done, why = PipelineManager._stage_is_complete(
            session / "output", session, StageId.SAM3)
        assert done, why
        _, ran = _run(PipelineManager(), session,
                      build_pipeline_stages(backend=BACKEND, stages=["sam3"]))
        assert ran == [StageId.SAM3]
        # without the selection the same session resumes past it
        _, ran_auto = _run(PipelineManager(), session,
                           build_pipeline_stages(backend=BACKEND))
        assert StageId.SAM3 not in ran_auto


# ── FINDING 21: the probe must not need what CloudCompy deletes ──

def test_reconstruction_probe_survives_the_chunk_cleanup():
    """CloudCompy ends with "removed 21 redundant chunk files, baked into
    cleaned_cloud": a probe that asks for chunk_*.ply says "never ran" about
    every session that ever reached the cleaned cloud, and redoes 3.5 h."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        session = _finished_session(Path(td))
        out = session / "output"
        assert not list(out.glob("chunk_*.ply")), "the fixture has none, as on disk"
        done, why = PipelineManager._stage_is_complete(out, session,
                                                       StageId.RECONSTRUCTION)
        assert done, why
        # the chunk plan alone carries it once the depth folder is gone too
        import shutil
        shutil.rmtree(out / "omega_run" / "results_output")
        (out / "cleaned_cloud.ply").unlink()
        done, why = PipelineManager._stage_is_complete(out, session,
                                                       StageId.RECONSTRUCTION)
        assert done and "chunk_plan" in why, why
        # nothing at all → it really never ran, and the reason says so
        (out / "camera_poses.txt").unlink()
        (out / "omega_run" / "camera_poses.txt").unlink()
        done, why = PipelineManager._stage_is_complete(out, session,
                                                       StageId.RECONSTRUCTION)
        assert not done and why == "no reconstruction artifacts"


def test_every_stage_name_is_selectable():
    """Every stage the order contains can be named — no stage is reachable only
    through the automatic chain."""
    for sid in DEFAULT_STAGE_ORDER:
        picked = build_pipeline_stages(backend=BACKEND, stages=[sid.value])
        assert [s.id for s in picked if s.enabled] == [sid]


def test_from_stage_is_a_range_not_a_naming():
    """`from_stage` must not resurrect a stage the config switched off.

    Measured 2026-09-21 while preparing the relaunch the user asked for
    ("relanzar al menos desde cloudcompy para abajo completo, excluyendo el
    cloudcompy"): `from_stage="vlm"` answered vlm, sam3, certify AND TSDF on a
    session whose config says `pipeline.auto_tsdf: false` — "the run ends at
    the cleaned cloud, the mesh is on demand" (USER 2026-08-28). A relaunch
    meant to redo the segmentation would have ended in a two-hour mesh nobody
    asked for.

    Naming a stage in `stages=` is the opposite case: that IS asking for it,
    and it still runs whatever the switch says.
    """
    from pipeline_manager import build_pipeline_stages, StageId

    def _ids(**kw):
        return [s.id for s in build_pipeline_stages(backend="vggtomega", **kw)
                if s.enabled]

    auto = _ids()
    frm = _ids(from_stage="vlm")
    assert StageId.TSDF not in auto, "the fixture's config no longer disables TSDF"
    assert StageId.TSDF not in frm, "from_stage re-enabled a stage the config disabled"
    # the range is exactly the tail of the automatic chain
    assert frm == auto[auto.index(StageId.VLM):]
    # naming it explicitly is a decision, and it is honoured
    assert _ids(stages=["tsdf"]) == [StageId.TSDF]
