"""intake.run — I0 → I1 end to end on a synthetic session; the marker (stamps:
every frame's bytes, the intake + DA3 code, parameters, K, the CPU environment,
the frozen run configuration — docs/plan_determinismo.md 66 / 69 / 70 / 74)
makes an identical second run a no-op (JSON mtimes untouched, the log says
skipped); force re-runs; a parameter, code or inventory change re-runs exactly
the steps whose stamp changed and names the difference; no time, epoch or
absolute path in the marker (the run record beside it holds them); the CLI.
I2 (VLM content tags + SAM3 exclusion masks) is GONE (USER 2026-10-08, plan
point 75): a marker of an older run that lists it is read as what it says
about I0 / I1, and its leftover products are deleted, said. Plus the
map_worker side: the frame-selection resolver (rejects an unknown value,
admits parallax_lk) and a branch for every admitted value (hf included), the
stamped, atomic witness ∪ keyframe da3_frames.json, and _run_intake_selection
driven through a fake pipe — reuse / re-run branches, cancel, and the GPU
handover before the focal probe (the intake's one GPU step)."""

import ast
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from intake import frames_manifest as FM                          # noqa: E402
from intake import parallax as P                                  # noqa: E402
from intake import quality as Q                                   # noqa: E402
from intake import run as R                                       # noqa: E402
from intake import run_config as RC                               # noqa: E402
from intake.config import load_intake_config                      # noqa: E402
from tests import synth_precision as S                            # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
with open(SERVER / "config.yaml") as _f:
    BASE_RAW = yaml.safe_load(_f)                                  # the raw server configuration
W, H = 320, 240
N_WALK, STILL_PREFIX, STEP_M = 40, 8, 0.08
NOISE_SIGMA = 2.0


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def icfg():
    """The real config.yaml, with the parallax tracker sized for 320x240 test
    frames (production frames are 1080p+; a sparse seed grid — these tests
    exercise the orchestration, test_intake_parallax the measurement)."""
    with open(SERVER / "config.yaml") as f:
        base = load_intake_config(yaml.safe_load(f))
    par = replace(base.parallax, grid_side=12, process_scale=0.5, lk_win=15)
    return replace(base, parallax=par)


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One rendered session per module (a sideways walk after a still prefix,
    strided video frame numbers); tests copy it."""
    cam = S.default_camera(W, H)
    scene = S.room_scene(seed=0)
    walk = S.walk_poses("translate", N_WALK, start=S.room_start_pose(x_m=1.0), step_m=STEP_M,
                        direction="right")
    poses = np.concatenate([np.repeat(walk[:1], STILL_PREFIX, axis=0), walk])
    frame_numbers = list(range(0, 3 * len(poses), 3))
    root = tmp_path_factory.mktemp("intake_run")
    return S.write_session(root / "sess", scene, cam, poses, frame_numbers=frame_numbers,
                           noise_sigma=NOISE_SIGMA)


@pytest.fixture
def session(rendered, tmp_path):
    """A fresh copy of the rendered session (same names + sizes → same inventory)."""
    dst = tmp_path / "sess"
    shutil.copytree(rendered.session_dir, dst)
    return dst


class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, msg, level="info"):
        self.lines.append(str(msg))

    def has(self, fragment):
        return any(fragment in ln for ln in self.lines)


def _artifacts(session_dir):
    """Every JSON the intake writes (existing ones)."""
    paths = [session_dir / "frames" / Q.QUALITY_FEATURES_NAME,
             session_dir / "frames" / Q.LEGACY_FRAME_QUALITY_NAME,
             session_dir / "frames" / P.SELECTED_FRAMES_NAME,
             session_dir / "frames" / P.WITNESS_FRAMES_NAME,
             session_dir / "intake" / P.COVERAGE_WARNINGS_NAME,
             R.state_path(session_dir)]
    return [p for p in paths if p.exists()]


def _mtimes(session_dir):
    return {str(p): p.stat().st_mtime_ns for p in _artifacts(session_dir)}


def _true_focal(*_a, **_k):
    """The focal probe as DA3 would measure it: the rendering camera's pinhole K."""
    return S.default_camera(W, H).K()


@pytest.fixture(autouse=True)
def _no_da3(monkeypatch):
    """Every path that does not inject ``focal`` (the CLI, the map_worker) gets the
    rendering camera's K instead of a DA3 run."""
    import intake.focal as F
    monkeypatch.setattr(F, "default_probe", lambda python=None: _true_focal)


def _raw(icfg):
    """The raw configuration whose intake section IS ``icfg`` (what a job freezes)."""
    return RC.raw_with_intake(BASE_RAW, icfg)


def _run(session_dir, icfg, **kw):
    kw.setdefault("focal", _true_focal)
    kw.setdefault("log", Log())
    kw.setdefault("run_config", _raw(icfg))            # this call starts the job: it freezes
    return R.run_intake(session_dir, icfg, **kw), kw


def _s(step):
    """A step's record without its 'differences' list (compared on their own where it matters)."""
    return {k: v for k, v in step.items() if k != "differences"}


def _stamped(doc, provenance):
    assert doc["provenance"] == provenance, doc.get("provenance")
    assert doc["geometry_epoch"] == 0 and doc["camera_epoch"] == 0
    assert "version" in doc


# ── end to end + the marker ──────────────────────────────────────────────

def test_end_to_end_artifacts_stamps_and_noop_second_run(session, icfg):
    res, kw = _run(session, icfg)
    log = kw["log"]

    # every step ran, for the reason a fresh session gives
    assert {k: (v["ran"], v["reason"]) for k, v in res["steps"].items()} == {
        "quality": (True, R.REASON_NO_MARKER), "parallax": (True, R.REASON_NO_MARKER)}
    assert R.STEPS == ("quality", "parallax") and "content" not in res["steps"]

    frames_dir = session / "frames"
    quality = json.loads((frames_dir / Q.QUALITY_FEATURES_NAME).read_text())
    _stamped(quality, "tool_measured")
    assert quality["n_frames"] == N_WALK + STILL_PREFIX
    shim = json.loads((frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).read_text())
    assert shim["method"] == "intake_features" and len(shim["frames"]) == quality["n_frames"]

    selected = json.loads((frames_dir / P.SELECTED_FRAMES_NAME).read_text())
    _stamped(selected, "tool_measured")
    assert selected["version"] == "2.0" and selected["method"].startswith("parallax_lk")
    assert selected["selected_count"] == len(selected["selected_files"]) >= 2
    witness = json.loads((frames_dir / P.WITNESS_FRAMES_NAME).read_text())
    _stamped(witness, "tool_measured")
    keyframes = [int(Path(f).stem) for f in selected["selected_files"]]
    witnesses = [int(r["frame"]) for r in witness["frames"]]
    assert set(keyframes) <= set(witnesses)
    warnings = json.loads((session / "intake" / P.COVERAGE_WARNINGS_NAME).read_text())
    _stamped(warnings, "tool_measured")

    # nothing of I2 is written (USER 2026-10-08)
    assert not (session / "intake" / "content_tags.json").exists()
    assert not (session / "intake" / "exclusion_masks").exists()
    assert res["i2_leftovers_removed"] == [] and "content" not in res["artifacts"]

    state_text = R.state_path(session).read_text()
    state = json.loads(state_text)
    assert state["version"] == R.STATE_VERSION and state["provenance"] == "tool_measured"
    assert state["run_config_sha256"] == res["run_config_sha256"]
    assert RC.load_run_config(session)[1] == res["run_config_sha256"]
    assert set(state["steps"]) == set(R.STEPS)
    for step in R.STEPS:
        e = state["steps"][step]
        st = e["stamp"]
        assert e["done"] is True and len(st["sha256"]) == 64
        # every stamp: the intake + DA3 code, every frame's bytes, the environment, the
        # frozen run configuration, the step's parameters (plan point 66)
        assert "server/intake/run.py" in st["code"] and "server/extract_da3_depth.py" in st["code"]
        assert "server/intake/content.py" not in st["code"]
        assert {"environment", "params"} <= set(st["config"])
        assert e["run_config_sha256"] == res["run_config_sha256"]      # recorded, not a key
        assert all((session / a).exists() and not Path(a).is_absolute() for a in e["artifacts"])
    assert set(state["steps"]["quality"]["stamp"]["inputs"]) == {
        f"frames/{p.name}" for p in frames_dir.glob("*.jpg")}
    assert "frames" in state["steps"]["parallax"]["stamp"]["config"]
    assert set(state["steps"]["parallax"]["stamp"]["inputs"]) == {"frames/quality_features.json"}
    # no wall clock, no absolute path, no session epoch in the marker (point 70) ...
    for key in ("updated_at", "finished_at", "elapsed_s", "session_dir", "frames_dir",
                "geometry_epoch", "camera_epoch"):
        assert key not in state_text, key
    assert str(session) not in state_text
    # ... they live in the run record beside it
    timing = json.loads(R.timing_path(session).read_text())
    assert timing["session_dir"] == str(session) and timing["geometry_epoch"] == 0
    assert all("finished_at" in timing["steps"][s_] and "elapsed_s" in timing["steps"][s_]
               for s_ in R.STEPS)
    assert timing["environment"]["libs"]["numpy"] and timing["updated_at"]
    # the products: epoch 0 by construction, session-relative inputs, the environment
    for doc in (quality, selected, witness, warnings):
        assert str(session) not in json.dumps(doc)
    assert quality["environment"]["jpeg_decoders"]["opencv"]["libjpeg"]

    s = res["summary"]
    assert (s["n_frames"], s["n_keyframes"], s["n_witness"]) == (
        quality["n_frames"], len(keyframes), len(witnesses))
    assert set(s) == {"n_frames", "n_usable", "n_keyframes", "n_witness", "n_warnings"}
    assert log.has("quality: running") and log.has("parallax: running")

    # ── second run, identical inputs: nothing is written, nothing is called ──
    mt = _mtimes(session)
    assert len(mt) >= 6
    time.sleep(0.02)
    res2, kw2 = _run(session, icfg)
    assert _mtimes(session) == mt
    assert {k: (v["ran"], v["reason"]) for k, v in res2["steps"].items()} == {
        step: (False, R.REASON_MATCHES) for step in R.STEPS}
    log2 = kw2["log"]
    assert log2.has("quality: skipped") and log2.has("parallax: skipped")
    assert res2["summary"] == res["summary"]
    # the frozen run configuration is the same bytes (its sha is in every stamp)
    assert res2["run_config_sha256"] == res["run_config_sha256"]


def test_a_copied_session_keeps_its_marker(session, icfg, tmp_path):
    """Stamps key frames by their session-relative path and bytes (point 70): the session
    copied elsewhere resumes without re-measuring anything."""
    _run(session, icfg)
    dst = tmp_path / "elsewhere" / "sess"
    shutil.copytree(session, dst)
    res, _kw = _run(dst, icfg)
    assert {k: v["ran"] for k, v in res["steps"].items()} == {s_: False for s_ in R.STEPS}


def test_the_intake_refuses_a_configuration_nobody_froze(session, icfg):
    """Plan point 69: without a frozen output/run_config.yaml the intake does not run; with one,
    the configuration it is handed must be the frozen one (field named) — and the marker's
    stamps carry the frozen sha, so another frozen configuration re-runs the steps."""
    with pytest.raises(R.IntakeRunError, match="run_config.yaml"):
        R.run_intake(session, icfg, focal=_true_focal, log=Log())
    RC.freeze_run_config(session, _raw(icfg), log=lambda m: None)
    res = R.run_intake(session, icfg, focal=_true_focal, log=Log())
    assert all(v["ran"] for v in res["steps"].values())
    other = replace(icfg, quality=replace(icfg.quality, luma_lo=icfg.quality.luma_lo + 1.0))
    with pytest.raises(R.IntakeRunError, match=r"quality\.luma_lo"):
        R.run_intake(session, other, focal=_true_focal, log=Log())
    # the same configuration frozen again: identical bytes, nothing re-runs
    RC.freeze_run_config(session, _raw(icfg), log=lambda m: None)
    res2 = R.run_intake(session, icfg, focal=_true_focal, log=Log())
    assert not any(v["ran"] for v in res2["steps"].values())
    # a frozen configuration that differs OUTSIDE the intake section re-runs nothing (the
    # step's own parameters are the key); the marker RECORDS the new sha on every entry
    raw = _raw(icfg)
    raw["pipeline"] = {**dict(raw.get("pipeline") or {}), "_probe": 1}
    sha3 = RC.freeze_run_config(session, raw, log=lambda m: None)["sha256"]
    res3 = R.run_intake(session, icfg, focal=_true_focal, log=Log())
    assert not any(v["ran"] for v in res3["steps"].values())
    assert res3["run_config_sha256"] == sha3 != res2["run_config_sha256"]
    st = json.loads(R.state_path(session).read_text())
    assert st["run_config_sha256"] == sha3


def test_force_reruns_everything(session, icfg):
    _run(session, icfg)
    mt = _mtimes(session)
    time.sleep(0.02)
    res, _kw = _run(session, icfg, force=True)
    assert all(v["ran"] and v["reason"] == R.REASON_FORCE for v in res["steps"].values())
    after = _mtimes(session)
    for p in (session / "frames" / Q.QUALITY_FEATURES_NAME,
              session / "frames" / P.SELECTED_FRAMES_NAME, R.state_path(session)):
        assert after[str(p)] > mt[str(p)], p.name


def test_a_marker_listing_the_removed_i2_is_read_as_i0_i1_and_its_leftovers_go(session, icfg):
    """USER 2026-10-08 (plan point 75): a session whose intake ran before I2 was removed holds a
    marker with a ``content`` entry, a content_tags.json and exclusion masks. The marker is read
    as what it says about I0 / I1 (same version, same stamps — nothing is re-measured), the
    ``content`` entry is dropped (said) and the leftover products are deleted (said): no reader
    can take them for a product of this intake."""
    res0, _ = _run(session, icfg)
    p = R.state_path(session)
    st = json.loads(p.read_text())
    st["steps"]["content"] = {"done": True, "stamp": {"sha256": "0" * 64}, "artifacts": []}
    p.write_text(json.dumps(st))
    (session / "intake" / "content_tags.json").write_text("{}")
    masks = session / "intake" / "exclusion_masks"
    masks.mkdir()
    (masks / "000003.png").write_bytes(b"\x89PNG")
    log = Log()
    res, _ = _run(session, icfg, log=log)
    assert {k: (v["ran"], v["reason"]) for k, v in res["steps"].items()} == {
        step: (False, R.REASON_MATCHES) for step in R.STEPS}
    assert res["i2_leftovers_removed"] == ["intake/content_tags.json", "intake/exclusion_masks"]
    assert not (session / "intake" / "content_tags.json").exists() and not masks.exists()
    assert log.has("['content']") and log.has("dropped") and log.has("deleted the products of the removed I2")
    on_disk = json.loads(p.read_text())
    assert set(on_disk["steps"]) == set(R.STEPS) and on_disk["version"] == R.STATE_VERSION
    assert res["summary"] == res0["summary"]
    # the second pass has nothing left to delete and says nothing about it
    log2 = Log()
    res2, _ = _run(session, icfg, log=log2)
    assert res2["i2_leftovers_removed"] == [] and not log2.has("removed I2")
    # the intake's public surface carries no I2 hook any more
    import inspect
    params = inspect.signature(R.run_intake).parameters
    assert not {"tagger", "segmenter", "skip_content", "before_content", "before_sam3"} & set(params)
    assert not hasattr(R, "REASON_SKIP_CONTENT")


# ── what re-runs, and what does not ──────────────────────────────────────

def test_parameter_change_reruns_only_downstream(session, icfg):
    res, _ = _run(session, icfg)
    mt = _mtimes(session)
    time.sleep(0.02)
    # parallax change: I0 stands, I1 re-runs with the reason and the differences named
    par = replace(icfg, parallax=replace(icfg.parallax, parallax_quantum_px=6.0))
    res2, _ = _run(session, par)
    assert res2["steps"]["quality"]["ran"] is False
    assert _s(res2["steps"]["parallax"]) == {"ran": True, "reason": R.REASON_PARAMS_CHANGED}
    assert res2["steps"]["parallax"]["differences"] and \
        all(d.startswith("config 'params'") for d in res2["steps"]["parallax"]["differences"])
    n_kf_before, n_kf_after = res["summary"]["n_keyframes"], res2["summary"]["n_keyframes"]
    assert n_kf_after > n_kf_before                    # half the quantum → more keyframes
    after = _mtimes(session)
    assert after[str(session / "frames" / Q.QUALITY_FEATURES_NAME)] == \
        mt[str(session / "frames" / Q.QUALITY_FEATURES_NAME)]
    assert after[str(session / "frames" / P.SELECTED_FRAMES_NAME)] > \
        mt[str(session / "frames" / P.SELECTED_FRAMES_NAME)]
    # a quality parameter: I0 re-runs (params_changed), I1 follows (its input report changed)
    time.sleep(0.02)
    qua = replace(par, quality=replace(par.quality, luma_lo=par.quality.luma_lo + 1.0))
    res3, _ = _run(session, qua)
    assert _s(res3["steps"]["quality"]) == {"ran": True, "reason": R.REASON_PARAMS_CHANGED,
                                            "legacy_shim": {"action": "written", "reason": "I0 ran"}}
    assert res3["steps"]["parallax"]["ran"] and \
        res3["steps"]["parallax"]["reason"] == R.REASON_INPUTS_CHANGED
    assert any("quality_features.json" in d for d in res3["steps"]["parallax"]["differences"])


def test_frame_inventory_change_reruns_everything(session, icfg):
    _run(session, icfg)
    frames_dir = session / "frames"
    last = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))[-1]
    added = f"{int(last.stem) + 3:06d}.jpg"
    shutil.copy(last, frames_dir / added)
    # a frame added behind the seal is refused, naming it (points 67 / 76) ...
    with pytest.raises(R.IntakeRunError, match=f"frames/{added} is on disk but not in the manifest"):
        _run(session, icfg)
    # ... a new inventory comes with its own seal (here: the old one dropped, the frames adopted)
    (frames_dir / FM.MANIFEST_NAME).unlink()
    res, _ = _run(session, icfg)
    assert {k: (v["ran"], v["reason"]) for k, v in res["steps"].items()} == {
        step: (True, R.REASON_INPUTS_CHANGED) for step in R.STEPS}
    assert res["summary"]["n_frames"] == N_WALK + STILL_PREFIX + 1


def test_legacy_frames_are_adopted_once_declared_and_verified_after(session, icfg):
    """A frames/ with no manifest predates the seal (points 67 / 76): the first intake ADOPTS it
    — origin 'adopted', each frame's sha256 from I0's own stamp, DECLARED in the log — and every
    later run checks the frames on disk against it, leaving its bytes alone."""
    import repro
    frames_dir = session / "frames"
    assert not (frames_dir / FM.MANIFEST_NAME).exists()
    res, kw = _run(session, icfg)
    m = json.loads((frames_dir / FM.MANIFEST_NAME).read_text())
    assert m["origin"] == FM.ORIGIN_ADOPTED and m["complete"] is True and m["decoder"] is None
    assert {r["name"]: r["sha256"] for r in m["frames"]} == \
        {p.name: repro.sha256_file(p) for p in Q.list_frames(frames_dir)}
    assert kw["log"].has("DECLARED") and kw["log"].has("ADOPTED")
    assert res["frames_manifest"] == {"origin": FM.ORIGIN_ADOPTED, "n_frames": m["n_frames"],
                                      "frames_sha256": m["frames_sha256"]}
    seal = (frames_dir / FM.MANIFEST_NAME).read_bytes()
    _res2, kw2 = _run(session, icfg)
    assert kw2["log"].has("every frame's sha256 matches") and not kw2["log"].has("ADOPTED")
    assert (frames_dir / FM.MANIFEST_NAME).read_bytes() == seal


def test_frames_being_written_are_refused_and_never_adopted(session, icfg):
    """The intake never reads a scan whose frames are being written (point 76): the temp dir of
    an extraction (running, or left by a restart) refuses the run before any frame is hashed, and
    nothing is adopted; an incomplete manifest is refused too."""
    FM.extracting_dir(session).mkdir()
    with pytest.raises(R.IntakeRunError, match="frames.extracting"):
        _run(session, icfg)
    assert not (session / "frames" / FM.MANIFEST_NAME).exists()
    assert not R.state_path(session).exists()
    FM.extracting_dir(session).rmdir()
    (session / "frames" / FM.MANIFEST_NAME).write_text(json.dumps(
        {"manifest_version": FM.MANIFEST_VERSION, "complete": False, "frames": []}))
    with pytest.raises(R.IntakeRunError, match="not a complete manifest"):
        _run(session, icfg)


def test_replace_deleted_files_rerun_i1_only(session, icfg):
    """pipeline_manager.FRAMES_DIR_FILES deletes selected_frames.json and
    frame_quality.json on a reconstruction with replace. I0 stands (its
    report is intact; the legacy shim is rewritten from it) and I1 re-measures
    the same frames into the same keyframes."""
    res0, _ = _run(session, icfg)
    frames_dir = session / "frames"
    (frames_dir / P.SELECTED_FRAMES_NAME).unlink()
    (frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).unlink()
    res, kw = _run(session, icfg)
    assert res["steps"]["quality"]["ran"] is False and kw["log"].has("rewritten from")
    assert (frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).exists()
    assert res["steps"]["parallax"]["ran"] is True
    assert res["steps"]["parallax"]["reason"].startswith(R.REASON_ARTIFACT)
    assert res["parallax"]["selected_frames"]["selected_files"] == \
        res0["parallax"]["selected_frames"]["selected_files"]


@pytest.mark.parametrize("marker, reason", [
    ({"version": 99, "steps": {}}, R.REASON_MARKER_VERSION),
    ("not json {", R.REASON_MARKER_UNREADABLE),
    ([1, 2, 3], R.REASON_MARKER_UNREADABLE),
])
def test_foreign_or_broken_marker_reruns_with_its_own_reason(session, icfg, marker, reason):
    _run(session, icfg)
    p = R.state_path(session)
    p.write_text(marker if isinstance(marker, str) else json.dumps(marker))
    res, kw = _run(session, icfg)
    assert all(v["ran"] and v["reason"].startswith(reason) for v in res["steps"].values())
    if reason == R.REASON_MARKER_UNREADABLE:
        assert kw["log"].has(R.REASON_MARKER_UNREADABLE) and kw["log"].has(R.STATE_NAME)
    st = json.loads(p.read_text())
    assert st["version"] == R.STATE_VERSION and all(st["steps"][s]["done"] for s in R.STEPS)


def test_code_is_part_of_the_marker(session, icfg):
    """A change of the MEASUREMENT (the code), not only of its parameters, re-runs the step
    (plan point 66: no version number bumped by hand): an I1 marker whose stamp was made by
    another intake/parallax.py re-runs I1 with reason code_changed, naming the file."""
    _run(session, icfg)
    p = R.state_path(session)
    st = json.loads(p.read_text())
    assert st["steps"]["parallax"]["stamp"]["code"]["server/intake/parallax.py"]
    st["steps"]["parallax"]["stamp"]["code"]["server/intake/parallax.py"] = "0" * 64
    p.write_text(json.dumps(st))
    res, _ = _run(session, icfg)
    assert res["steps"]["quality"]["ran"] is False
    assert res["steps"]["parallax"]["ran"] and \
        res["steps"]["parallax"]["reason"] == R.REASON_CODE_CHANGED
    cur = R.St.step_stamp({}, {})["code"]["server/intake/parallax.py"][:12]
    assert res["steps"]["parallax"]["differences"] == [
        f"code 'server/intake/parallax.py' changed (000000000000 -> {cur})"]


def test_a_changed_frame_byte_reruns_everything_and_names_the_frame(session, icfg):
    """Frames are identified by their BYTES, not their name and size (point 66): re-encoding
    one frame to the same size re-runs I0 (and what follows) and the log names the frame."""
    _run(session, icfg)
    frames_dir = session / "frames"
    first = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))[0]
    data = bytearray(first.read_bytes())
    data[-3] ^= 0x01                                   # same size, other bytes
    first.write_bytes(bytes(data))
    # behind the seal: the intake fails naming the frame (points 67 / 76) ...
    with pytest.raises(R.IntakeRunError, match=f"frames/{first.name} differs from the manifest"):
        _run(session, icfg)
    # ... re-sealed (the old manifest dropped, the frames adopted), the change re-runs I0
    (frames_dir / FM.MANIFEST_NAME).unlink()
    res, kw = _run(session, icfg)
    assert res["steps"]["quality"]["ran"] and \
        res["steps"]["quality"]["reason"] == R.REASON_INPUTS_CHANGED
    assert any(f"frames/{first.name}" in d for d in res["steps"]["quality"]["differences"])
    assert kw["log"].has(f"frames/{first.name}")


def test_epochs_are_stamped_and_cancel_names_the_step(session, icfg):
    """Every artifact carries epoch 0 by construction whatever the session's output/ says;
    the session's epochs at run time go to the run record; cancel names the step it
    stopped in."""
    out = session / "output"
    out.mkdir(exist_ok=True)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 5}))
    res, _ = _run(session, icfg)
    # the session's epochs at run time go to the run record; every product carries epoch 0
    # by construction (points 63 / 70: the same frames give the same bytes)
    assert (res["geometry_epoch"], res["camera_epoch"]) == (5, 0)
    assert json.loads(R.timing_path(session).read_text())["geometry_epoch"] == 5
    for path in _artifacts(session):
        if path.suffix == ".json" and path != R.state_path(session):
            doc = json.loads(path.read_text())
            assert doc["geometry_epoch"] == 0, path.name
    with pytest.raises(Q.IntakeCancelled, match="intake I0"):
        R.run_intake(session, icfg, focal=_true_focal, log=Log(), force=True, cancelled=lambda: True)


# ── the CLI ───────────────────────────────────────────────────────────────

def test_run_intake_needs_a_frames_dir(tmp_path, icfg):
    with pytest.raises(R.IntakeRunError, match="frames"):
        R.run_intake(tmp_path, icfg, focal=_true_focal, log=Log())
    (tmp_path / "frames").mkdir()
    with pytest.raises(R.IntakeRunError):
        R.run_intake(tmp_path, icfg, focal=_true_focal, log=Log())


def test_cli_main(session, icfg, monkeypatch, capsys):
    # the CLI reads the session's frozen run configuration (point 69); frozen here as a job
    # would have
    RC.freeze_run_config(session, _raw(icfg), log=lambda m: None)
    assert R.main(["--session", str(session)]) == 0
    out = capsys.readouterr().out
    assert "keyframes=" in out and "marker" in out and "content" not in out.split("[intake.run] keyframes=")[1]
    assert (session / "frames" / P.SELECTED_FRAMES_NAME).exists()
    assert not (session / "intake" / "content_tags.json").exists()
    # the I2 switch is gone from the CLI
    with pytest.raises(SystemExit):
        R.main(["--session", str(session), "--skip-content"])
    capsys.readouterr()
    # a second CLI pass with --force re-runs; without it, nothing
    assert R.main(["--session", str(session), "--force"]) == 0
    out = capsys.readouterr().out
    assert "quality: running (force)" in out
    assert R.main(["--session", str(session)]) == 0
    out = capsys.readouterr().out
    assert "quality: skipped (marker_matches" in out and "parallax: skipped" in out


# ── the map_worker helpers ───────────────────────────────────────────────

def _map_worker():
    try:
        from workers import map_worker
    except ImportError as e:                       # pragma: no cover - environment gap
        if "torch" in str(e):
            raise
        pytest.skip(f"workers.map_worker not importable here: {e}")
    return map_worker


def _map_work_ast(mw):
    tree = ast.parse(Path(mw.__file__).read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_map_work")


def _mode_branch(node, value):
    """True when ``node`` is an ``if``/``elif`` whose test is ``mode == value``."""
    t = getattr(node, "test", None)
    return (isinstance(node, ast.If) and isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name) and t.left.id == "mode"
            and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
            and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == value)


def test_every_admitted_frame_selection_has_its_own_branch():
    """No admitted value falls into the chain's final 'stride'/'none' else (which
    writes every blur-valid frame with method 'none'): each has an explicit
    `mode == ...` branch, and hf's calls the legacy H/F selector."""
    mw = _map_worker()
    fn = _map_work_ast(mw)
    for value in mw.FRAME_SELECTIONS:
        branches = [n for n in ast.walk(fn) if _mode_branch(n, value)]
        assert branches, f"frame selection {value!r} has no branch in _map_work"
    hf = [n for n in ast.walk(fn) if _mode_branch(n, "hf")]
    assert len(hf) == 1
    calls = {c.func.id for c in ast.walk(ast.Module(body=hf[0].body, type_ignores=[]))
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "select_keyframes_hf" in calls


def test_hf_selector_writes_a_selection_the_v2_readers_consume(session):
    """What the hf branch runs, on real frames: selected_frames.json readable by
    the consumers (the legacy selector writes its own 'version' 1.0; no reader
    checks the version, every one reads selected_files)."""
    mw = _map_worker()
    import yaml as _yaml
    from frames.selector import load_selected_frames, select_keyframes_hf
    with open(SERVER / "config.yaml") as f:
        fcfg = _yaml.safe_load(f).get("frame_selection", {})
    sel = select_keyframes_hf(str(session / "frames"), fcfg)
    frames_dir = session / "frames"
    assert sel["method"] == "hf_ratio" and sel["selected_count"] >= 2
    assert load_selected_frames(str(frames_dir)) == sel["selected_files"]
    from segmentation.scene_analyzer import _load_keyframes
    assert [Path(p).name for p in _load_keyframes(frames_dir)] == sorted(
        sel["selected_files"], key=lambda f: int(Path(f).stem))
    assert mw._resolve_frame_selection(
        {"frames_selector": "none", "simple": {"enabled": True, "frame_selection": "hf"}}) == "hf"


# ── map_worker._run_intake_selection through a fake pipe ─────────────────

class FakePipe:
    """The WorkerPipe surface the frame-selection stage uses."""

    def __init__(self, cancel_after=None):
        self.logs, self.progress, self.polls = [], [], 0
        self.cancel_after = cancel_after

    def send_log(self, msg, level="info"):
        self.logs.append(str(msg))

    def send_progress(self, pct, msg, stage=""):
        self.progress.append((pct, msg, stage))

    def check_cancel(self):
        self.polls += 1
        return self.cancel_after is not None and self.polls > self.cancel_after

    def has(self, fragment):
        return any(fragment in m for m in self.logs)


def _worker_config():
    """The raw config.yaml as the worker receives it, the tracker sized for 320x240 frames."""
    with open(SERVER / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["intake"]["parallax"].update(grid_side=24, process_scale=0.5, lk_win=15)
    return raw


def test_worker_hands_the_gpu_over_before_the_focal_probe_and_touches_no_vlm(session, monkeypatch):
    """The real path: the worker's VERIFIED stopper (workers.base.stop_semantic_service)
    runs ONCE, before the DA3 focal probe — the intake's one GPU step (zaragoza 2026-10-04:
    the probe ran with vLLM loading beside it and died in OOM); no semantic service is
    ensured and no SAM3 runs: I2 is gone (USER 2026-10-08)."""
    mw = _map_worker()
    import semantic.service as svc
    import workers.base as wb
    ev = []
    monkeypatch.setattr(svc, "ensure_service",
                        lambda config=None, log=None, cancelled=None, timeout_s=None:
                        ev.append(("ensure",)) or True)
    # the worker's VERIFIED stopper runs for real: its stop and its pgrep are faked
    monkeypatch.setattr(wb, "stop_semantic_service",
                        lambda pipe=None, stage="", log=None: ev.append(("stop", stage)))
    monkeypatch.setattr(wb, "vllm_pids", lambda: ev.append(("pgrep",)) or [])
    monkeypatch.setattr(wb, "gpu_free_gb", lambda: 44.0)
    pipe = FakePipe()
    cfg = _worker_config()
    RC.freeze_run_config(session, cfg, log=lambda m: None)      # the manager's job start
    mw._run_intake_selection(pipe, session, session / "frames", cfg, True)
    kinds = [e[0] for e in ev]
    assert ev[0] == ("stop", "intake focal probe (DA3)") and kinds.count("stop") == 1
    assert "ensure" not in kinds and "pgrep" in kinds
    assert pipe.has("Frame set:") and pipe.has("witness frames")
    assert not (session / "intake" / "content_tags.json").exists()
    assert not hasattr(mw, "_ensure_semantic_or_fail")
    src = Path(mw.__file__).read_text()
    assert "before_content" not in src and "before_sam3" not in src and "intake.content" not in src


def test_a_pgrep_that_cannot_run_is_not_a_check_that_found_nothing(monkeypatch):
    """The verified stopper's pgrep (workers.base.vllm_pids) fails loudly when it cannot run."""
    import subprocess
    import workers.base as wb

    def broken(*a, **k):
        raise OSError("pgrep: not found")
    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(RuntimeError, match="cannot verify the GPU handover"):
        wb.vllm_pids()


def test_worker_always_asks_the_intake_marker(session, icfg, monkeypatch):
    """replace on or off, the worker runs intake.run.run_intake — its marker
    skips what is measured and runs what is missing (the old shortcut reused
    the selection without reading the marker) — with the worker's cancel and
    GPU-handover hooks."""
    mw = _map_worker()
    import intake.run as run_mod
    frames_dir = session / "frames"
    calls = []

    def fake_run_intake(session_dir, icfg_, **kw):
        calls.append(kw)
        return {"steps": {"quality": {"ran": False}, "parallax": {"ran": True}},
                "summary": {"n_keyframes": 3, "n_frames": 48, "n_witness": 9, "n_warnings": 0},
                "artifacts": {"coverage_warnings": str(session_dir / "intake" / "cw.json")}}

    monkeypatch.setattr(run_mod, "run_intake", fake_run_intake)
    cfg = _worker_config()
    for replace_ in (False, True):
        pipe = FakePipe()
        mw._run_intake_selection(pipe, session, frames_dir, cfg, replace_)
        assert calls[-1]["cancelled"] == pipe.check_cancel
        assert callable(calls[-1]["focal"])
        assert set(calls[-1]) == {"log", "progress", "focal", "cancelled"}
        assert pipe.has("the intake marker decides") and pipe.has("['parallax']")
    assert len(calls) == 2

    # fewer than two keyframes → the stage fails naming where to look
    def one_keyframe(session_dir, icfg_, **kw):
        return {"steps": {}, "summary": {"n_keyframes": 1, "n_frames": 48, "n_witness": 1,
                                         "n_warnings": 1},
                "artifacts": {"coverage_warnings": "/s/intake/coverage_warnings.json"}}

    monkeypatch.setattr(run_mod, "run_intake", one_keyframe)
    with pytest.raises(RuntimeError, match=r"1 keyframe\(s\).*coverage_warnings.json"):
        mw._run_intake_selection(FakePipe(), session, frames_dir, cfg, True)


def test_worker_retry_resumes_from_the_marker(session, monkeypatch):
    """A run that died in the focal probe (the DA3 step after I0) left I0 on disk and no
    selection; the retry with replace=off measures I0 nothing again and runs I1."""
    mw = _map_worker()
    import intake.focal as F
    import workers.base as wb
    monkeypatch.setattr(wb, "stop_semantic_service", lambda pipe=None, stage="", log=None: None)
    monkeypatch.setattr(wb, "vllm_pids", lambda: [])

    def probe_down(*_a, **_k):
        raise RuntimeError("DA3 focal probe exited with code 3")

    monkeypatch.setattr(F, "default_probe", lambda python=None: probe_down)
    cfg = _worker_config()
    RC.freeze_run_config(session, cfg, log=lambda m: None)
    with pytest.raises(RuntimeError, match="focal probe exited"):
        mw._run_intake_selection(FakePipe(), session, session / "frames", cfg, False)
    st = json.loads(R.state_path(session).read_text())
    assert st["steps"]["quality"]["done"] and "parallax" not in st["steps"]
    q_mtime = (session / "frames" / Q.QUALITY_FEATURES_NAME).stat().st_mtime_ns
    monkeypatch.setattr(F, "default_probe", lambda python=None: _true_focal)
    pipe = FakePipe()
    mw._run_intake_selection(pipe, session, session / "frames", cfg, False)
    assert (session / "frames" / P.SELECTED_FRAMES_NAME).exists()
    assert (session / "frames" / Q.QUALITY_FEATURES_NAME).stat().st_mtime_ns == q_mtime
    assert pipe.has("quality: skipped") and pipe.has("parallax: running") and pipe.has("['parallax']")


def test_skipped_i0_rewrites_a_foreign_or_stale_legacy_shim(session, icfg):
    """frames/frame_quality.json feeds the legacy readers' 'valid' flags: when I0
    is skipped it must still be THIS intake's shim of THIS report. A foreign
    one (the legacy blur analysis's percentile cull) or a stale one is
    rewritten from quality_features.json, with the reason; an identical one is
    left untouched."""
    res, _ = _run(session, icfg)
    shim = session / "frames" / Q.LEGACY_FRAME_QUALITY_NAME
    good = shim.read_text()
    foreign = json.loads(good)
    foreign.update(method="blur_percentile", threshold_percentile=15.0)
    foreign["frames"][3]["valid"] = False
    shim.write_text(json.dumps(foreign))
    log = Log()
    res2, _ = _run(session, icfg, log=log)
    assert res2["steps"]["quality"]["ran"] is False
    rec = res2["steps"]["quality"]["legacy_shim"]
    assert rec["action"] == "rewritten" and "foreign" in rec["reason"]
    assert "blur_percentile" in rec["reason"] and log.has("foreign")
    assert json.loads(shim.read_text()) == json.loads(good)
    # ours but not the current report's → stale
    stale = json.loads(good)
    stale["frames"][0]["fft_score"] += 1.0
    shim.write_text(json.dumps(stale))
    res3, _ = _run(session, icfg)
    assert res3["steps"]["quality"]["legacy_shim"]["action"] == "rewritten"
    assert "stale" in res3["steps"]["quality"]["legacy_shim"]["reason"]
    # unreadable and missing
    shim.write_text("{not json")
    assert "unreadable" in _run(session, icfg)[0]["steps"]["quality"]["legacy_shim"]["reason"]
    shim.unlink()
    assert "missing" in _run(session, icfg)[0]["steps"]["quality"]["legacy_shim"]["reason"]
    # identical → kept, mtime untouched
    mt = shim.stat().st_mtime_ns
    time.sleep(0.02)
    res4, _ = _run(session, icfg)
    assert res4["steps"]["quality"]["legacy_shim"]["action"] == "kept"
    assert shim.stat().st_mtime_ns == mt


def test_worker_honours_cancel_inside_the_intake(session, monkeypatch):
    """A cancel inside the intake's own loops stops it where it is, naming the step."""
    mw = _map_worker()
    cfg = _worker_config()
    RC.freeze_run_config(session, cfg, log=lambda m: None)
    with pytest.raises(Q.IntakeCancelled, match="intake I0"):
        mw._run_intake_selection(FakePipe(cancel_after=2), session, session / "frames", cfg, True)


def test_resolve_frame_selection_admits_the_five_and_rejects_the_rest():
    mw = _map_worker()
    assert mw.FRAME_SELECTIONS == ("parallax_lk", "motion", "fps", "dino", "hf")
    for sel in mw.FRAME_SELECTIONS:
        cfg = {"frames_selector": "none", "simple": {"enabled": True, "frame_selection": sel}}
        assert mw._resolve_frame_selection(cfg) == sel
    # case-insensitive like the legacy chain
    assert mw._resolve_frame_selection(
        {"frames_selector": "none", "simple": {"enabled": True, "frame_selection": "Parallax_LK"}}
    ) == "parallax_lk"
    # an unknown value names itself — no silent 'motion'
    with pytest.raises(RuntimeError, match="banana"):
        mw._resolve_frame_selection(
            {"frames_selector": "none", "simple": {"enabled": True, "frame_selection": "banana"}})
    with pytest.raises(RuntimeError, match="frame_selection is missing"):
        mw._resolve_frame_selection({"frames_selector": "none", "simple": {"enabled": True}})
    # the SIMPLE block does not override an explicit fps / motion frames_selector
    for legacy in ("fps", "motion"):
        assert mw._resolve_frame_selection(
            {"frames_selector": legacy, "simple": {"enabled": True, "frame_selection": "banana"}}
        ) == legacy
    # SIMPLE off → the legacy selector as-is, whatever the simple block says
    assert mw._resolve_frame_selection(
        {"frames_selector": "stride", "simple": {"enabled": False, "frame_selection": "banana"}}
    ) == "stride"
    assert mw._resolve_frame_selection({}) == "none"
    assert mw._resolve_frame_selection({"frames_selector": "dino"}) == "dino"


def test_intake_da3_frames_is_witness_union_keyframes(tmp_path):
    mw = _map_worker()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for k in range(100):                                   # the inventory the count is taken on
        (frames_dir / f"{k:06d}.jpg").write_bytes(b"")
    with pytest.raises(RuntimeError, match="selected_frames.json"):
        mw._intake_da3_frames(frames_dir)
    sel = {"version": "2.0", "method": "parallax_lk_12", "total_frames": 100,
           "selected_count": 3, "selected_files": ["000000.jpg", "000030.jpg", "000090.jpg"],
           "provenance": "tool_measured", "geometry_epoch": 2, "camera_epoch": 1}
    (frames_dir / "selected_frames.json").write_text(json.dumps(sel))
    with pytest.raises(RuntimeError, match="witness_frames.json"):
        mw._intake_da3_frames(frames_dir)
    wit = {"version": P.PARALLAX_VERSION, "method": "parallax_lk", "total_frames": 100,
           "frames": [{"frame": f} for f in (0, 9, 30, 45, 60)], "selected_count": 5,
           "selected_files": ["000000.jpg", "000009.jpg", "000030.jpg", "000045.jpg",
                              "000060.jpg"]}
    (frames_dir / "witness_frames.json").write_text(json.dumps(wit))
    doc = mw._intake_da3_frames(frames_dir)
    assert doc["version"] == "2.0" and doc["method"] == "parallax_lk_witness"
    assert doc["selected_files"] == ["000000.jpg", "000009.jpg", "000030.jpg", "000045.jpg",
                                     "000060.jpg", "000090.jpg"]
    assert doc["selected_count"] == 6 and doc["total_frames"] == 100
    assert (doc["n_keyframes"], doc["n_witness"]) == (3, 5)
    # the four stamps: provenance and epochs carried from the intake's selection
    assert doc["provenance"] == "tool_measured"
    assert (doc["geometry_epoch"], doc["camera_epoch"]) == (2, 1)
    assert doc["intake_version"] == P.PARALLAX_VERSION
    # total_frames is COUNTED on disk and must agree with both intake documents
    (frames_dir / "000100.jpg").write_bytes(b"")
    with pytest.raises(RuntimeError, match="records total_frames 100 but .* holds 101"):
        mw._intake_da3_frames(frames_dir)
    (frames_dir / "000100.jpg").unlink()
    (frames_dir / "witness_frames.json").write_text(json.dumps(dict(wit, total_frames=99)))
    with pytest.raises(RuntimeError, match="witness_frames.json records total_frames 99"):
        mw._intake_da3_frames(frames_dir)
    (frames_dir / "witness_frames.json").write_text(json.dumps({"frames": []}))
    with pytest.raises(RuntimeError, match="selected_files"):
        mw._intake_da3_frames(frames_dir)
    (frames_dir / "witness_frames.json").write_text(json.dumps(wit))
    (frames_dir / "selected_frames.json").write_text(json.dumps(
        {k: v for k, v in sel.items() if k != "provenance"}))
    with pytest.raises(RuntimeError, match="provenance"):
        mw._intake_da3_frames(frames_dir)


def test_intake_da3_frames_on_a_real_run_written_atomically_and_read_back(session, icfg):
    mw = _map_worker()
    res, _ = _run(session, icfg)
    frames_dir = session / "frames"
    path, doc = mw._write_intake_da3_frames(frames_dir)
    assert path == frames_dir / "da3_frames.json" and not list(frames_dir.glob("*.tmp"))
    assert json.loads(path.read_text()) == doc
    kf = set(res["parallax"]["selected_frames"]["selected_files"])
    wit = set(res["parallax"]["witness_frames"]["selected_files"])
    assert set(doc["selected_files"]) == kf | wit == wit          # keyframes ⊂ witnesses
    assert doc["selected_files"] == sorted(doc["selected_files"], key=lambda f: int(f[:6]))
    assert doc["total_frames"] == res["summary"]["n_frames"]
    assert doc["provenance"] == "tool_measured" and doc["geometry_epoch"] == 0
    # its legacy consumer (the dense BA tracking, map_worker Step 4) reads it as-is
    from reconstruction.vggt_tracks import _load_list_from_json
    listed = _load_list_from_json(path, frames_dir)
    assert [p.name for _, p in listed] == doc["selected_files"]


def test_legacy_readers_accept_the_stamped_shim(session, icfg):
    """frames/frame_quality.json now carries version / provenance / epochs; every
    legacy reader of it still reads it (they read frames[] only)."""
    mw = _map_worker()
    res, _ = _run(session, icfg)
    frames_dir = session / "frames"
    shim = json.loads((frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).read_text())
    assert shim["version"] == Q.QUALITY_VERSION and shim["provenance"] == "tool_measured"
    assert shim["geometry_epoch"] == 0 and shim["camera_epoch"] == 0
    usable = [f["file"] for f in res["quality"]["frames"] if f["usable"]]
    from frames.quality import load_valid_frames
    from frames.selector import _load_valid_frame_list
    assert load_valid_frames(str(frames_dir)) == usable
    assert _load_valid_frame_list(frames_dir) == usable
    chosen, n_total, _ = mw._motion_keyframes(frames_dir, 1e9)
    assert n_total == len(shim["frames"]) and len(chosen) == 1
