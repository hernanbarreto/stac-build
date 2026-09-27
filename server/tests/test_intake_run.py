"""intake.run — I0 → I1 → I2 end to end on a synthetic session with the
tagger / segmenter injected; the marker makes an identical second run a
no-op (JSON mtimes untouched, the log says skipped); force re-runs; a
parameter or inventory change re-runs exactly the steps whose inputs
changed; --skip-content leaves I2 out; the CLI. Plus the map_worker side: the
frame-selection resolver (rejects an unknown value, admits parallax_lk) and a
branch for every admitted value (hf included), the stamped, atomic witness ∪
keyframe da3_frames.json, and _run_intake_selection driven through a fake
pipe — reuse / re-run branches, the semantic-service failure with its reason,
cancel, and the GPU handover: vLLM tags, then the stopper, then SAM3."""

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

from intake import content as Cn                                  # noqa: E402
from intake import parallax as P                                  # noqa: E402
from intake import quality as Q                                   # noqa: E402
from intake import run as R                                       # noqa: E402
from intake.config import CONTENT_CLASSES, load_intake_config     # noqa: E402
from tests import synth_precision as S                            # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
W, H = 320, 240
N_WALK, STILL_PREFIX, STEP_M = 40, 8, 0.08
NOISE_SIGMA = 2.0
RECTS = {"person": (10, 60, 20, 100), "hand": (150, 230, 0, 60)}     # (r0, r1, c0, c1)


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def icfg():
    """The real config.yaml, with the parallax tracker sized for 320x240 test
    frames (production frames are 1080p+; a sparse seed grid — these tests
    exercise the orchestration, test_intake_parallax the measurement) and a
    small content batch / prompt set matched to the rectangle segmenter."""
    with open(SERVER / "config.yaml") as f:
        base = load_intake_config(yaml.safe_load(f))
    par = replace(base.parallax, grid_side=12, process_scale=0.5, lk_win=15)
    con = replace(base.content, enabled=True, batch=3,
                  exclusion_classes=("dynamic", "occluder"),
                  weight_classes=("reflective", "low_info"), sam3_scope="flagged_ranges",
                  prompts={"dynamic": ("person",), "occluder": ("hand",)})
    return replace(base, parallax=par, content=con)


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


class SpyTagger:
    """Tags every keyframe but the first as 'dynamic'; records calls."""

    def __init__(self, rule=lambda f: f > 0, cls="dynamic"):
        self.rule, self.cls = rule, cls
        self.calls, self.n_calls, self.parse_failures = [], 0, 0

    def tag(self, images, frames):
        self.calls.append(list(frames))
        self.n_calls += 1
        out = []
        for f in frames:
            t = {c: False for c in CONTENT_CLASSES}
            t[self.cls] = bool(self.rule(int(f)))
            t["notes"] = f"frame {f}"
            out.append(t)
        return out


class RectSegmenter:
    def __init__(self, hw=(H, W), rects=RECTS):
        self.hw, self.rects, self.calls, self.closed = hw, rects, [], 0

    def masks(self, frames_dir, frame_ids, prompt):
        self.calls.append((prompt, list(frame_ids)))
        r0, r1, c0, c1 = self.rects[prompt]
        m = np.zeros(self.hw, dtype=bool)
        m[r0:r1, c0:c1] = True
        return {int(f): m.copy() for f in frame_ids}

    def close(self):
        self.closed += 1


class Boom:
    def __init__(self, *a, **k):
        raise AssertionError("production class constructed where nothing may run")


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1


def _artifacts(session_dir):
    """Every JSON + PNG the intake writes (existing ones)."""
    paths = [session_dir / "frames" / Q.QUALITY_FEATURES_NAME,
             session_dir / "frames" / Q.LEGACY_FRAME_QUALITY_NAME,
             session_dir / "frames" / P.SELECTED_FRAMES_NAME,
             session_dir / "frames" / P.WITNESS_FRAMES_NAME,
             session_dir / "intake" / P.COVERAGE_WARNINGS_NAME,
             Cn.content_tags_path(session_dir), R.state_path(session_dir)]
    paths += sorted(Cn.exclusion_masks_dir(session_dir).glob("*.png"))
    return [p for p in paths if p.exists()]


def _mtimes(session_dir):
    return {str(p): p.stat().st_mtime_ns for p in _artifacts(session_dir)}


def _run(session_dir, icfg, **kw):
    kw.setdefault("tagger", SpyTagger())
    kw.setdefault("segmenter", RectSegmenter())
    kw.setdefault("log", Log())
    return R.run_intake(session_dir, icfg, **kw), kw


def _stamped(doc, provenance):
    assert doc["provenance"] == provenance, doc.get("provenance")
    assert doc["geometry_epoch"] == 0 and doc["camera_epoch"] == 0
    assert "version" in doc


# ── end to end + the marker ──────────────────────────────────────────────

def test_end_to_end_artifacts_stamps_and_noop_second_run(session, icfg):
    before = Counter()
    res, kw = _run(session, icfg, before_content=before)
    log, tagger, seg = kw["log"], kw["tagger"], kw["segmenter"]

    # every step ran, for the reason a fresh session gives
    assert {k: (v["ran"], v["reason"]) for k, v in res["steps"].items()} == {
        "quality": (True, R.REASON_NO_MARKER), "parallax": (True, R.REASON_NO_MARKER),
        "content": (True, R.REASON_NO_MARKER)}
    assert before.n == 1 and tagger.n_calls >= 1 and seg.calls
    assert seg.closed == 0                       # an injected segmenter is the caller's

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

    content = json.loads(Cn.content_tags_path(session).read_text())
    _stamped(content, "vlm_proposed")
    assert content["enabled"] is True
    assert sorted(int(k) for k in content["frames"]) == keyframes
    assert set(int(k) for k in content["exclusion_masks"]["frames"])   # something flagged
    for f in content["exclusion_masks"]["frames"]:
        png = Cn.mask_path(Cn.exclusion_masks_dir(session), int(f))
        m = Cn.read_mask_png(png)
        assert m.shape == (H, W) and m.dtype == bool
        r0, r1, c0, c1 = RECTS["person"]
        assert m[r0:r1, c0:c1].all() and not m[r1 + 5:, c1 + 5:].any()
    assert (content["summary"]["tagged"]["dynamic"] == len(keyframes) - 1)

    state = json.loads(R.state_path(session).read_text())
    assert state["version"] == R.STATE_VERSION and state["provenance"] == "tool_measured"
    for step in R.STEPS:
        e = state["steps"][step]
        assert e["done"] is True and isinstance(e["params_hash"], str)
        assert e["inputs"]["frames"]["n_frames"] == quality["n_frames"]
        assert all(Path(a).exists() for a in e["artifacts"])
    assert state["steps"]["parallax"]["inputs"]["quality"]["digest"]
    assert state["steps"]["content"]["inputs"]["keyframes"]["n"] == len(keyframes)

    s = res["summary"]
    assert (s["n_frames"], s["n_keyframes"], s["n_witness"]) == (
        quality["n_frames"], len(keyframes), len(witnesses))
    assert s["n_exclusion_masks"] == len(content["exclusion_masks"]["frames"])
    assert log.has("quality: running") and log.has("parallax: running") \
        and log.has("content: running")
    # the exclusion audit ran on I1's keyframe tracks under I2's masks
    assert res["audit_step"]["ran"] and res["audit_step"]["reason"] == "I1 ran"
    audit = json.loads((session / "intake" / P.EXCLUSION_AUDIT_NAME).read_text())
    _stamped(audit, "tool_measured")
    assert len(audit["keyframes"]) == len(keyframes) - 1
    assert res["exclusion_audit"]["n_rests_on_excluded"] == audit["n_rests_on_excluded"]
    assert [w for w in warnings["warnings"] if w["kind"] == "excluded_parallax"] == \
        audit["warnings"]
    assert warnings["exclusion_audit"]["n_keyframes"] == len(audit["keyframes"])

    # ── second run, identical inputs: nothing is written, nothing is called ──
    mt = _mtimes(session)
    assert len(mt) >= 7
    time.sleep(0.02)
    before2 = Counter()
    res2, kw2 = _run(session, icfg, before_content=before2)
    assert _mtimes(session) == mt
    assert {k: (v["ran"], v["reason"]) for k, v in res2["steps"].items()} == {
        step: (False, R.REASON_MATCHES) for step in R.STEPS}
    assert kw2["tagger"].n_calls == 0 and kw2["segmenter"].calls == [] and before2.n == 0
    log2 = kw2["log"]
    assert log2.has("quality: skipped") and log2.has("parallax: skipped") \
        and log2.has("content: skipped")
    assert res2["summary"] == res["summary"]
    assert res2["content"]["frames"] == content["frames"]
    assert res2["audit_step"] == {"ran": False, "reason": "I1 and I2 unchanged, audit on disk"}


def test_force_reruns_everything(session, icfg):
    _run(session, icfg)
    mt = _mtimes(session)
    time.sleep(0.02)
    res, kw = _run(session, icfg, force=True)
    assert all(v["ran"] and v["reason"] == R.REASON_FORCE for v in res["steps"].values())
    assert kw["tagger"].n_calls >= 1
    after = _mtimes(session)
    for p in (session / "frames" / Q.QUALITY_FEATURES_NAME,
              session / "frames" / P.SELECTED_FRAMES_NAME, Cn.content_tags_path(session),
              R.state_path(session)):
        assert after[str(p)] > mt[str(p)], p.name


def test_skip_content_then_content_only(session, icfg, monkeypatch):
    monkeypatch.setattr(Cn, "QwenTagger", Boom)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Boom)
    before = Counter()
    log = Log()
    res = R.run_intake(session, icfg, log=log, skip_content=True, before_content=before)
    assert res["steps"]["quality"]["ran"] and res["steps"]["parallax"]["ran"]
    assert res["steps"]["content"] == {
        "ran": False, "reason": R.REASON_SKIP_CONTENT, "skipped": True,
        "would_have_run": R.REASON_NO_MARKER, "stale_artifact": False}
    assert res["content"] is None and res["artifacts"]["content_tags"] is None
    assert res["summary"]["content_enabled"] is None
    assert not Cn.content_tags_path(session).exists()
    assert not Cn.exclusion_masks_dir(session).exists()
    assert before.n == 0
    st = json.loads(R.state_path(session).read_text())["steps"]["content"]
    assert st["done"] is False and st["skipped"] is True
    assert st["reason"].startswith(R.REASON_SKIP_CONTENT) and st["stale_artifact"] is False
    assert log.has("content: skipped (skip_content")

    # the frame set exists and is the intake's — the map_worker can proceed without I2
    assert P.load_selection(session / "frames")["method"].startswith("parallax_lk")

    # now with content: only I2 runs
    mt = _mtimes(session)
    time.sleep(0.02)
    res2, kw2 = _run(session, icfg, before_content=before)
    assert not res2["steps"]["quality"]["ran"] and not res2["steps"]["parallax"]["ran"]
    assert res2["steps"]["content"] == {"ran": True, "reason": R.REASON_NOT_DONE,
                                        "skipped": False}
    assert before.n == 1 and kw2["tagger"].n_calls >= 1
    after = _mtimes(session)
    for p in (session / "frames" / Q.QUALITY_FEATURES_NAME,
              session / "frames" / P.SELECTED_FRAMES_NAME):
        assert after[str(p)] == mt[str(p)]
    assert Cn.content_tags_path(session).exists()

    # skip_content with a content marker that still matches: the report is reused
    res3 = R.run_intake(session, icfg, log=Log(), skip_content=True, tagger=Boom,
                        segmenter=Boom)
    assert res3["steps"]["content"] == {"ran": False, "reason": R.REASON_MATCHES,
                                        "skipped": False}
    assert res3["content"]["frames"] == res2["content"]["frames"]


def test_skip_content_flags_a_stale_report(session, icfg):
    _run(session, icfg)
    # a parallax parameter changes → the keyframes may change → content_tags.json
    # would have to be re-measured; with skip_content it is left and declared STALE
    changed = replace(icfg, parallax=replace(icfg.parallax, parallax_quantum_px=6.0))
    log = Log()
    res = R.run_intake(session, changed, log=log, skip_content=True, tagger=Boom,
                       segmenter=Boom)
    assert res["steps"]["parallax"]["ran"] and \
        res["steps"]["parallax"]["reason"] == R.REASON_PARAMS_CHANGED
    c = res["steps"]["content"]
    assert c["skipped"] and c["would_have_run"] == R.REASON_INPUTS_CHANGED
    assert c["stale_artifact"] is True and log.has("STALE")
    st = json.loads(R.state_path(session).read_text())["steps"]["content"]
    assert st["done"] is False and st["stale_artifact"] is True
    assert Cn.content_tags_path(session).exists()      # left on disk, never deleted


# ── what re-runs, and what does not ──────────────────────────────────────

def test_parameter_change_reruns_only_downstream(session, icfg):
    _run(session, icfg)
    mt = _mtimes(session)
    time.sleep(0.02)
    # content-only change: I0 and I1 stand, I2 re-runs
    con = replace(icfg, content=replace(icfg.content, batch=2))
    res, kw = _run(session, con)
    assert [v["ran"] for v in res["steps"].values()] == [False, False, True]
    assert res["steps"]["content"]["reason"] == R.REASON_PARAMS_CHANGED
    assert all(len(c) <= 2 for c in kw["tagger"].calls)
    after = _mtimes(session)
    assert after[str(session / "frames" / Q.QUALITY_FEATURES_NAME)] == \
        mt[str(session / "frames" / Q.QUALITY_FEATURES_NAME)]
    assert after[str(session / "frames" / P.SELECTED_FRAMES_NAME)] == \
        mt[str(session / "frames" / P.SELECTED_FRAMES_NAME)]
    assert after[str(Cn.content_tags_path(session))] > mt[str(Cn.content_tags_path(session))]

    # parallax change: I0 stands, I1 re-runs; I2 re-runs only if the keyframes changed
    time.sleep(0.02)
    par = replace(con, parallax=replace(con.parallax, parallax_quantum_px=6.0))
    res2, _ = _run(session, par)
    assert res2["steps"]["quality"]["ran"] is False
    assert res2["steps"]["parallax"] == {"ran": True, "reason": R.REASON_PARAMS_CHANGED}
    n_kf_before, n_kf_after = res["summary"]["n_keyframes"], res2["summary"]["n_keyframes"]
    assert n_kf_after > n_kf_before                    # half the quantum → more keyframes
    assert res2["steps"]["content"] == {"ran": True, "reason": R.REASON_INPUTS_CHANGED,
                                        "skipped": False}


def test_frame_inventory_change_reruns_everything(session, icfg):
    _run(session, icfg)
    frames_dir = session / "frames"
    last = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))[-1]
    shutil.copy(last, frames_dir / f"{int(last.stem) + 3:06d}.jpg")
    res, _ = _run(session, icfg)
    assert {k: (v["ran"], v["reason"]) for k, v in res["steps"].items()} == {
        step: (True, R.REASON_INPUTS_CHANGED) for step in R.STEPS}
    assert res["summary"]["n_frames"] == N_WALK + STILL_PREFIX + 1


def test_replace_deleted_files_rerun_i1_but_not_the_vlm(session, icfg):
    """pipeline_manager.FRAMES_DIR_FILES deletes selected_frames.json and
    frame_quality.json on a reconstruction with replace. I0 stands (its
    report is intact; the legacy shim is rewritten from it), I1 re-measures
    the same frames into the same keyframes, and I2 — the only step that
    needs the VLM and SAM3 — stays skipped because its inputs did not change."""
    res0, _ = _run(session, icfg)
    frames_dir = session / "frames"
    (frames_dir / P.SELECTED_FRAMES_NAME).unlink()
    (frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).unlink()
    before = Counter()
    res, kw = _run(session, icfg, before_content=before)
    assert res["steps"]["quality"]["ran"] is False and kw["log"].has("rewritten from")
    assert (frames_dir / Q.LEGACY_FRAME_QUALITY_NAME).exists()
    assert res["steps"]["parallax"]["ran"] is True
    assert res["steps"]["parallax"]["reason"].startswith(R.REASON_ARTIFACT)
    assert res["steps"]["content"] == {"ran": False, "reason": R.REASON_MATCHES,
                                       "skipped": False}
    assert kw["tagger"].n_calls == 0 and kw["segmenter"].calls == [] and before.n == 0
    assert res["parallax"]["selected_frames"]["selected_files"] == \
        res0["parallax"]["selected_frames"]["selected_files"]


def test_missing_mask_png_reruns_content(session, icfg):
    _run(session, icfg)
    pngs = sorted(Cn.exclusion_masks_dir(session).glob("*.png"))
    assert pngs
    pngs[0].unlink()
    res, kw = _run(session, icfg)
    assert res["steps"]["content"]["ran"] and \
        res["steps"]["content"]["reason"].startswith(R.REASON_ARTIFACT)
    assert kw["tagger"].n_calls >= 1 and pngs[0].exists()


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


def test_content_report_without_its_mask_dir_is_re_measured(session, icfg):
    _run(session, icfg)
    p = Cn.content_tags_path(session)
    doc = json.loads(p.read_text())
    del doc["exclusion_masks"]["dir"]
    p.write_text(json.dumps(doc))
    res, _ = _run(session, icfg)
    assert res["steps"]["content"]["ran"]
    assert res["steps"]["content"]["reason"].startswith(R.REASON_ARTIFACT)
    assert "exclusion_masks.dir" in res["steps"]["content"]["reason"]


def test_stage_version_is_part_of_the_marker(session, icfg):
    """A change of the MEASUREMENT (the stage version), not only of its
    parameters, re-runs the step: an I1 marker from another version re-runs I1."""
    _run(session, icfg)
    p = R.state_path(session)
    st = json.loads(p.read_text())
    assert st["steps"]["parallax"]["inputs"]["stage_version"] == P.PARALLAX_VERSION
    st["steps"]["parallax"]["inputs"]["stage_version"] = P.PARALLAX_VERSION - 1
    p.write_text(json.dumps(st))
    res, _ = _run(session, icfg)
    assert res["steps"]["quality"]["ran"] is False
    assert res["steps"]["parallax"] == {"ran": True, "reason": R.REASON_INPUTS_CHANGED}


def test_hooks_run_in_order_and_epochs_are_stamped(session, icfg):
    """before_content (service up) → the VLM tags → before_sam3 (GPU handed
    over) → SAM3; every artifact carries the session's epochs; cancel names the
    step it stopped in."""
    out = session / "output"
    out.mkdir(exist_ok=True)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 5}))
    ev = []

    class T(SpyTagger):
        def tag(self, images, frames):
            ev.append("tag")
            return super().tag(images, frames)

    class Sg(RectSegmenter):
        def masks(self, frames_dir, frame_ids, prompt):
            ev.append("sam3")
            return super().masks(frames_dir, frame_ids, prompt)

    res, _ = _run(session, icfg, tagger=T(), segmenter=Sg(),
                  before_content=lambda: ev.append("before_content"),
                  before_sam3=lambda: ev.append("before_sam3"))
    first_tag, first_sam3 = ev.index("tag"), ev.index("sam3")
    assert ev[0] == "before_content" and ev.count("before_sam3") == 1
    assert first_tag < ev.index("before_sam3") < first_sam3
    assert "tag" not in ev[ev.index("before_sam3"):]
    assert res["content"]["sam3_handover"]["called"] is True
    assert (res["geometry_epoch"], res["camera_epoch"]) == (5, 0)
    for path in _artifacts(session):
        if path.suffix == ".json":
            doc = json.loads(path.read_text())
            assert doc["geometry_epoch"] == 5, path.name
    with pytest.raises(Q.IntakeCancelled, match="intake I0"):
        R.run_intake(session, icfg, log=Log(), force=True, cancelled=lambda: True)


# ── disabled content, laziness, CLI ──────────────────────────────────────

def test_disabled_content_writes_json_and_constructs_nothing(session, icfg, monkeypatch):
    monkeypatch.setattr(Cn, "QwenTagger", Boom)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Boom)
    off = replace(icfg, content=replace(icfg.content, enabled=False))
    before = Counter()
    res = R.run_intake(session, off, log=Log(), before_content=before)
    assert res["steps"]["content"]["ran"] is True and before.n == 0
    content = json.loads(Cn.content_tags_path(session).read_text())
    assert content["enabled"] is False and content["provenance"] == "vlm_proposed"
    assert content["frames"] == {} and "reason" in content
    assert res["summary"]["content_enabled"] is False and res["summary"]["n_exclusion_masks"] == 0
    # flipping it on is a parameter change: only I2 re-runs
    res2, kw = _run(session, icfg)
    assert [v["ran"] for v in res2["steps"].values()] == [False, False, True]
    assert res2["steps"]["content"]["reason"] == R.REASON_PARAMS_CHANGED
    assert kw["tagger"].n_calls >= 1


def test_run_intake_needs_a_frames_dir(tmp_path, icfg):
    with pytest.raises(R.IntakeRunError, match="frames"):
        R.run_intake(tmp_path, icfg, log=Log())
    (tmp_path / "frames").mkdir()
    with pytest.raises(R.IntakeRunError):
        R.run_intake(tmp_path, icfg, log=Log())


def test_cli_main(session, icfg, monkeypatch, capsys):
    monkeypatch.setattr(R, "load_intake_config", lambda raw=None: icfg)
    assert R.main(["--session", str(session), "--skip-content"]) == 0
    out = capsys.readouterr().out
    assert "keyframes=" in out and "content=skipped" in out
    assert (session / "frames" / P.SELECTED_FRAMES_NAME).exists()
    assert not Cn.content_tags_path(session).exists()
    # a second CLI pass with --force re-runs; without it, nothing
    assert R.main(["--session", str(session), "--skip-content", "--force"]) == 0
    out = capsys.readouterr().out
    assert "quality: running (force)" in out
    assert R.main(["--session", str(session), "--skip-content"]) == 0
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
    """The raw config.yaml as the worker receives it, the tracker sized for
    320x240 frames and the content prompts matched to the rectangle segmenter."""
    with open(SERVER / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["intake"]["parallax"].update(grid_side=24, process_scale=0.5, lk_win=15)
    raw["intake"]["content"].update(enabled=True, batch=3, sam3_scope="all",
                                    prompts={"dynamic": ["person"], "occluder": ["hand"]})
    return raw


def test_worker_hands_the_gpu_over_between_the_tags_and_sam3(session, monkeypatch):
    """The real path: the semantic service is ensured, the VLM tags every
    keyframe, THEN the worker's stopper (workers.base.stop_semantic_service) runs,
    THEN SAM3 segments — vLLM and SAM3 never share the card."""
    mw = _map_worker()
    import semantic.service as svc
    ev = []

    class T(SpyTagger):
        def __init__(self, cfg):
            super().__init__()
            ev.append(("tagger", cfg.backend))

        def tag(self, images, frames):
            ev.append(("tag",))
            return super().tag(images, frames)

    class Sg(RectSegmenter):
        def __init__(self, batch_size, log=print):
            super().__init__()
            ev.append(("segmenter", batch_size))

        def masks(self, frames_dir, frame_ids, prompt):
            ev.append(("sam3", prompt))
            return super().masks(frames_dir, frame_ids, prompt)

    import workers.base as wb
    monkeypatch.setattr(Cn, "QwenTagger", T)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Sg)
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
    mw._run_intake_selection(pipe, session, session / "frames", cfg, True)
    kinds = [e[0] for e in ev]
    stop = kinds.index("stop")
    assert kinds.index("ensure") < kinds.index("tag") < stop < kinds.index("pgrep") \
        < kinds.index("sam3")
    assert "tag" not in kinds[stop:] and "sam3" not in kinds[:stop]
    assert ev[stop] == ("stop", "intake I2 SAM3") and kinds.count("stop") == 1
    assert ("segmenter", cfg["models"]["segmentation"]["batch_size"]) in ev
    assert pipe.has("Frame set:") and pipe.has("witness frames")
    content = json.loads(Cn.content_tags_path(session).read_text())
    assert content["exclusion_masks"]["scope"] == "all"
    assert content["sam3_handover"] == {
        "called": True, "verified": True, "reason": "before the first SAM3 call",
        "check": {"service_stopped": True, "check": "pgrep -f 'vllm serve'",
                  "remaining_pids": [], "free_gb": 44.0}}


def test_worker_refuses_sam3_while_vllm_survives_the_stop(session, monkeypatch):
    """stop_semantic_service swallows its own failures; the verified stopper
    looks again (pgrep) and fails naming the PIDs — SAM3 never loads on a
    shared card."""
    mw = _map_worker()
    import semantic.service as svc
    import workers.base as wb
    sam3 = []

    class Sg(RectSegmenter):
        def __init__(self, batch_size, log=print):
            super().__init__()

        def masks(self, frames_dir, frame_ids, prompt):
            sam3.append(prompt)
            return super().masks(frames_dir, frame_ids, prompt)

    monkeypatch.setattr(Cn, "QwenTagger", lambda cfg: SpyTagger())
    monkeypatch.setattr(Cn, "Sam3Segmenter", Sg)
    monkeypatch.setattr(svc, "ensure_service", lambda *a, **k: True)
    monkeypatch.setattr(wb, "stop_semantic_service", lambda pipe=None, stage="", log=None: None)
    monkeypatch.setattr(wb, "vllm_pids", lambda: [4242, 4243])
    with pytest.raises(RuntimeError, match=r"still running after the stop \(PIDs \[4242, 4243\]\)"):
        mw._run_intake_selection(FakePipe(), session, session / "frames", _worker_config(), True)
    assert sam3 == [] and not Cn.content_tags_path(session).exists()
    # a pgrep that cannot run is not a check that found nothing
    import subprocess

    def broken(*a, **k):
        raise OSError("pgrep: not found")
    monkeypatch.undo()
    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(RuntimeError, match="cannot verify the GPU handover"):
        wb.vllm_pids()


def test_worker_always_asks_the_intake_marker(session, icfg, monkeypatch):
    """replace on or off, the worker runs intake.run.run_intake — its marker
    skips what is measured and runs what is missing (the old shortcut reused
    the selection without reading the marker, and a run that died in I2 went on
    without content tags) — with the worker's cancel and GPU-handover hooks."""
    mw = _map_worker()
    import intake.run as run_mod
    frames_dir = session / "frames"
    calls = []

    def fake_run_intake(session_dir, icfg_, **kw):
        calls.append(kw)
        return {"steps": {"quality": {"ran": False}, "parallax": {"ran": False},
                          "content": {"ran": True}},
                "summary": {"n_keyframes": 3, "n_frames": 48, "n_witness": 9, "n_warnings": 0},
                "artifacts": {"coverage_warnings": str(session_dir / "intake" / "cw.json")}}

    monkeypatch.setattr(run_mod, "run_intake", fake_run_intake)
    cfg = _worker_config()
    for replace_ in (False, True):
        pipe = FakePipe()
        mw._run_intake_selection(pipe, session, frames_dir, cfg, replace_)
        assert calls[-1]["cancelled"] == pipe.check_cancel
        assert callable(calls[-1]["before_sam3"]) and callable(calls[-1]["before_content"])
        assert pipe.has("the intake marker decides") and pipe.has("['content']")
    assert len(calls) == 2

    # fewer than two keyframes → the stage fails naming where to look
    def one_keyframe(session_dir, icfg_, **kw):
        return {"steps": {}, "summary": {"n_keyframes": 1, "n_frames": 48, "n_witness": 1,
                                         "n_warnings": 1},
                "artifacts": {"coverage_warnings": "/s/intake/coverage_warnings.json"}}

    monkeypatch.setattr(run_mod, "run_intake", one_keyframe)
    with pytest.raises(RuntimeError, match=r"1 keyframe\(s\).*coverage_warnings.json"):
        mw._run_intake_selection(FakePipe(), session, frames_dir, cfg, True)


def test_worker_retry_runs_the_missing_i2(session, monkeypatch):
    """The residual the old shortcut left: a run that died in I2 (the semantic
    service did not come up) left I0 + I1 on disk and no content_tags.json; the
    retry with replace=off measures nothing again and runs I2."""
    mw = _map_worker()
    import semantic.service as svc
    import workers.base as wb
    monkeypatch.setattr(Cn, "QwenTagger", lambda cfg: SpyTagger())
    monkeypatch.setattr(Cn, "Sam3Segmenter", lambda batch_size, log=print: RectSegmenter())
    monkeypatch.setattr(wb, "stop_semantic_service", lambda pipe=None, stage="", log=None: None)
    monkeypatch.setattr(wb, "vllm_pids", lambda: [])

    def down(config=None, log=None, cancelled=None, timeout_s=None):
        log("Semantic service down")
        return False

    monkeypatch.setattr(svc, "ensure_service", down)
    cfg = _worker_config()
    with pytest.raises(RuntimeError, match="Semantic service down"):
        mw._run_intake_selection(FakePipe(), session, session / "frames", cfg, False)
    st = json.loads(R.state_path(session).read_text())
    assert st["steps"]["quality"]["done"] and st["steps"]["parallax"]["done"]
    assert "content" not in st["steps"] and not Cn.content_tags_path(session).exists()
    sel_mtime = (session / "frames" / P.SELECTED_FRAMES_NAME).stat().st_mtime_ns
    monkeypatch.setattr(svc, "ensure_service", lambda *a, **k: True)
    pipe = FakePipe()
    mw._run_intake_selection(pipe, session, session / "frames", cfg, False)
    assert Cn.content_tags_path(session).exists()
    assert (session / "frames" / P.SELECTED_FRAMES_NAME).stat().st_mtime_ns == sel_mtime
    assert pipe.has("quality: skipped") and pipe.has("parallax: skipped")
    assert pipe.has("content: running") and pipe.has("['content']")


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


def test_worker_fails_with_the_services_reason_and_honours_cancel(session, monkeypatch):
    mw = _map_worker()
    import semantic.service as svc

    def down(config=None, log=None, cancelled=None, timeout_s=None):
        log("Semantic service down and launcher missing (/nowhere/serve_semantic.sh)")
        return False

    monkeypatch.setattr(svc, "ensure_service", down)
    cfg = _worker_config()
    with pytest.raises(RuntimeError, match=r"launcher missing.*intake\.content\.enabled: false"):
        mw._ensure_semantic_or_fail(FakePipe(), cfg)
    with pytest.raises(RuntimeError, match="cancelled while waiting"):
        mw._ensure_semantic_or_fail(FakePipe(cancel_after=0), cfg)
    # a cancel inside the intake's own loops stops it where it is
    monkeypatch.setattr(Cn, "QwenTagger", Boom)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Boom)
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
