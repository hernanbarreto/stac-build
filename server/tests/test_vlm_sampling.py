"""The VLM sees the WHOLE walk (USER 2026-09-29: "debe segmentar todo,
absolutamente preciso y completo").

In the SIMPLE pipeline the phrases the VLM writes ARE the SAM3 prompts, so a
keyframe it never looks at is a set of objects never segmented. pccr 2026-09-29
was named from 8 linspace keyframes of 289 (the intake has no cloud, the
coverage cover fell back), and `aggregate` then kept ONE phrase per head noun —
'cardboard box' vanished into 'red fire alarm box' without a trace.

Pinned here, all synthetic (no vLLM, no SAM3): the sampling is uniform along
the walk (chainage when intake/walk.json covers the keyframes, else keyframe
index) at the configured density, the call count respects its BOUND, the crops
tile the frame, and the auto-prompter runs exactly the planned calls, records
every one, and folds only true synonyms.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.autoprompt.vlm_sampling import (  # noqa: E402
    AXIS_CHAINAGE, AXIS_INDEX, VLMSampling, VLMSamplingConfigError, crop_for_vlm,
    load_vlm_sampling, plan_vlm_frames, tile_boxes, walk_chainage)

CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"


def _cfg(**kw):
    base = dict(spacing_m=0.5, spacing_kf=8, tile_rows=2, tile_cols=2,
                tile_overlap_frac=0.2, max_calls=10_000)
    base.update(kw)
    return VLMSampling(**base)


def _files(n, step=1):
    return [f"{i * step:06d}.jpg" for i in range(n)]


# ── the strict loader ────────────────────────────────────────────────────

def test_production_config_loads_and_is_bounded():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    cfg = load_vlm_sampling(raw)
    assert cfg.calls_per_frame == 1 + cfg.tile_rows * cfg.tile_cols
    assert cfg.max_calls >= cfg.calls_per_frame
    assert raw["autoprompt"]["understand_cover"] is False, \
        "the cover cannot measure at the intake (no cloud of this run)"


@pytest.mark.parametrize("key", ["spacing_m", "spacing_kf", "tile_rows", "tile_cols",
                                 "tile_overlap_frac", "max_calls"])
def test_a_missing_key_fails_naming_it(key):
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    del raw["autoprompt"]["vlm_sampling"][key]
    with pytest.raises(VLMSamplingConfigError, match=f"autoprompt.vlm_sampling.{key}"):
        load_vlm_sampling(raw)


def test_the_removed_fixed_count_fails_the_load():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    raw["autoprompt"]["understand_sample"] = 8
    with pytest.raises(VLMSamplingConfigError, match="spacing_kf"):
        load_vlm_sampling(raw)


def test_a_bound_that_cannot_hold_one_frame_fails():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    raw["autoprompt"]["vlm_sampling"].update(tile_rows=2, tile_cols=2, max_calls=4)
    with pytest.raises(VLMSamplingConfigError, match="max_calls"):
        load_vlm_sampling(raw)


# ── uniform along the walk ───────────────────────────────────────────────

def test_keyframe_index_axis_is_uniform_and_covers_both_ends():
    """pccr's shape: 289 keyframes, no walk yet — one every 8 keyframes, not 8."""
    files = _files(289)
    plan = plan_vlm_frames(files, _cfg(), chainage=None, axis_reason="no walk")
    idx = [f["keyframe_index"] for f in plan.frames]
    assert plan.axis == AXIS_INDEX
    assert idx[0] == 0 and idx[-1] == 288, "the walk's two ends must be looked at"
    gaps = np.diff(idx)
    assert gaps.max() <= 8 and gaps.min() >= 7, gaps
    assert len(idx) == 37
    assert plan.n_calls == 37 * 5 and not plan.bound_reached


def test_chainage_axis_follows_the_METRES_not_the_index():
    """Half the keyframes crawl through the first metre (turning in place), the
    rest cover 9 m. Index-uniform sampling would spend half its frames on that
    first metre and leave ~0.9 m gaps later; walk-uniform sampling keeps every
    gap near the configured 0.5 m."""
    n = 200
    files = _files(n)
    ch = np.concatenate([np.linspace(0.0, 1.0, 100), np.linspace(1.0 + 0.09, 10.0, 100)])
    chainage = {int(f[:6]): float(c) for f, c in zip(files, ch)}
    plan = plan_vlm_frames(files, _cfg(spacing_m=0.5), chainage=chainage, axis_reason="walk")
    pos = np.array([f["position"] for f in plan.frames])
    assert plan.axis == AXIS_CHAINAGE
    assert pos[0] == 0.0 and pos[-1] == pytest.approx(10.0)
    step = float(np.max(np.diff(ch)))
    assert np.diff(pos).max() <= 0.5 + step + 1e-9, np.diff(pos)
    # the crawl is not over-sampled: ~1 m of it holds ~3 frames, not half of them
    assert sum(1 for p in pos if p <= 1.0) <= 4
    # the same keyframes sampled by INDEX leave metre-long holes on the fast half
    idx_plan = plan_vlm_frames(files, _cfg(spacing_kf=len(files) // len(plan.frames)))
    idx_pos = ch[[f["keyframe_index"] for f in idx_plan.frames]]
    assert np.diff(idx_pos).max() > 0.5 + step


def test_the_bound_thins_the_frames_uniformly_and_says_so():
    files = _files(289)
    plan = plan_vlm_frames(files, _cfg(max_calls=100))       # 5 calls/frame → 20 frames
    assert plan.n_calls <= 100
    assert plan.bound_reached and plan.n_before_bound == 37
    idx = [f["keyframe_index"] for f in plan.frames]
    assert len(idx) == 20 and idx[0] == 0 and idx[-1] == 288
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1, "thinned, but still uniform"
    assert "BOUND REACHED" in plan.summary()


def test_the_bound_also_holds_for_a_coverage_preselection():
    files = _files(100)
    plan = plan_vlm_frames(files, _cfg(max_calls=25), preselected=files[::2])
    assert plan.n_calls <= 25 and plan.bound_reached


def test_walk_json_is_used_only_when_it_covers_every_keyframe(tmp_path):
    files = _files(5, step=10)
    (tmp_path / "intake").mkdir()
    ch, why = walk_chainage(tmp_path, files)
    assert ch is None and "not on disk" in why
    doc = {"chainage": [{"frame": i * 10, "chainage_m": 0.3 * i} for i in range(5)]}
    (tmp_path / "intake" / "walk.json").write_text(json.dumps(doc))
    ch, why = walk_chainage(tmp_path, files)
    assert ch == {0: 0.0, 10: 0.3, 20: 0.6, 30: pytest.approx(0.9), 40: 1.2}
    ch, why = walk_chainage(tmp_path, files + ["000050.jpg"])
    assert ch is None and "another keyframe set" in why


# ── the crops ────────────────────────────────────────────────────────────

def test_crops_tile_the_whole_frame_with_the_declared_overlap():
    W, H = 832, 464
    boxes = tile_boxes(W, H, 2, 2, 0.2)
    assert [t for t, _ in boxes] == ["r0c0", "r0c1", "r1c0", "r1c1"]
    cover = np.zeros((H, W), bool)
    for _t, (x0, y0, x1, y1) in boxes:
        cover[y0:y1, x0:x1] = True
    assert cover.all(), "a pixel no crop shows"
    (_, a), (_, b) = boxes[0], boxes[1]
    tw = a[2] - a[0]
    assert (a[2] - b[0]) == pytest.approx(0.2 * tw, abs=1.5)
    assert tile_boxes(W, H, 1, 1, 0.2) == [], "a 1x1 grid IS the frame"


def test_a_crop_is_shown_at_the_frames_size():
    from PIL import Image
    img = Image.new("RGB", (832, 464))
    _t, box = tile_boxes(832, 464, 2, 2, 0.2)[3]
    crop = crop_for_vlm(img, box)
    assert max(crop.size[0] / 832, crop.size[1] / 464) == pytest.approx(1.0, abs=0.01)


# ── the auto-prompter runs exactly the plan and loses nothing ────────────

class _Reply:
    def __init__(self, content):
        self.content = content


class _FakeVLM:
    """Answers every understanding call; a crop (enlarged, so taller than the
    tiny frame) names a small object the full frame does not."""

    def __init__(self, frame_size, broken_every=0):
        self.frame_size = frame_size
        self.calls = []
        self.broken_every = broken_every

    def chat(self, messages, max_tokens=None, consumer=None):
        self.calls.append(consumer)
        if self.broken_every and len(self.calls) % self.broken_every == 0:
            return _Reply("not json")
        objs = ["White tiled floor", "cardboard box", "red fire alarm box",
                "fire extinguishers"]
        if len(self.calls) % 5 != 1:          # the four crops of every frame
            objs = ["doorbell panel", "fire extinguisher"]
        return _Reply(json.dumps({"scene_type": "server room", "summary": "s",
                                  "objects": objs}))


def _session(tmp_path, n_kf=40):
    from PIL import Image
    frames = tmp_path / "frames"
    frames.mkdir()
    files = []
    for i in range(n_kf):
        fn = f"{i * 7:06d}.jpg"
        Image.new("RGB", (64, 36), (i, i, i)).save(frames / fn)
        files.append(fn)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    return files


def _config(**sampling):
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    raw["autoprompt"]["consolidate_prompts"] = False
    raw["autoprompt"]["vlm_sampling"].update(sampling)
    raw["reconstruction"]["simple"]["enabled"] = True
    return raw


def test_the_autoprompter_runs_the_plan_records_every_call_and_folds_only_synonyms(
        tmp_path, monkeypatch):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    files = _session(tmp_path, n_kf=40)
    fake = _FakeVLM((64, 36), broken_every=7)
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: fake)
    cfg = _config(spacing_kf=8, tile_rows=2, tile_cols=2, max_calls=300)
    res = AutoPrompter(tmp_path, tmp_path / "output", config=cfg).run()

    vlm = json.loads((tmp_path / "output" / "vlm_analysis.json").read_text())
    rec = vlm["census"]
    plan = rec["sampling"]
    # uniform along the keyframe index, both ends, one every 8 → 6 frames x 5 calls
    assert [f["keyframe_index"] for f in plan["frames"]] == [0, 8, 16, 23, 31, 39]
    assert plan["n_calls"] == 30 == len(fake.calls) == rec["n_calls"]
    assert sum(1 for c in rec["calls"] if c["tile"] is not None) == 24
    assert rec["n_parsed"] == 30 - 30 // 7, "a failed parse is a recorded call, not a silent one"

    # true synonyms fold, different names stay — with the reason written down
    prompts = res.prompt.split(";")
    for p in ("cardboard box", "red fire alarm box", "doorbell panel"):
        assert p in prompts, f"'{p}' never reached SAM3"
    assert sum(1 for p in prompts if "extinguisher" in p) == 1
    fates = {c["concept"]: c for c in rec["concepts"]}
    assert set(fates) == {"white tiled floor", "cardboard box", "red fire alarm box",
                          "fire extinguishers", "doorbell panel", "fire extinguisher"}
    merged = [c for c in fates.values() if c["fate"] == "merged"]
    assert len(merged) == 1 and "same name" in merged[0]["reason"]
    assert all(c["fate"] in ("prompt", "merged") for c in fates.values())
    # the doorbell panel was only ever seen in a crop
    assert all(b["tile"] is not None for b in fates["doorbell panel"]["proposed_by"])


def test_a_vocabulary_derived_under_another_sampling_is_not_reused(tmp_path, monkeypatch):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    _session(tmp_path, n_kf=10)
    out = tmp_path / "output"
    out.mkdir()
    # pccr's record: written before the stamp existed
    (out / "autoprompt_concepts.json").write_text(json.dumps(
        {"version": 1, "prompts": ["only the old list"]}))
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: _FakeVLM((64, 36)))
    cfg = _config(spacing_kf=8, tile_rows=1, tile_cols=1, max_calls=300)
    res = AutoPrompter(tmp_path, out, config=cfg).run()
    assert "only the old list" not in res.prompt
    rec = json.loads((out / "autoprompt_concepts.json").read_text())
    assert rec["derived_under"]["vlm_sampling"]["spacing_kf"] == 8
    assert rec["reused"] is False
    # same sampling next run → the pinned list IS the answer, and the census says why
    res2 = AutoPrompter(tmp_path, out, config=cfg).run()
    assert res2.prompt == res.prompt
    rec2 = json.loads((out / "autoprompt_concepts.json").read_text())
    assert rec2["reused"] is True
