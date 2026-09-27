"""I2 — content tags are vlm_proposed and land in intake/content_tags.json;
exclusion PNGs are 255 inside the segmenter's mask on the NATIVE grid;
flagged_ranges claims the witnesses between a tagged keyframe and its
neighbours only; a VLM answer that does not parse tags the batch all-False
with notes 'vlm_parse_failed' and is counted; enabled=false writes the JSON
and constructs / calls nothing; SAM3 prompts come from the config only; the
GPU is handed over (``before_sam3``) after the last tag and before the first
SAM3 call — the VLM and SAM3 never share the card."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from intake import content as Cn                                    # noqa: E402
from intake.config import CONTENT_CLASSES, ContentConfig            # noqa: E402
from tests import synth_precision as S                              # noqa: E402

W, H = 160, 120
FRAME_NUMBERS = list(range(0, 130, 10))          # 13 frames, strided like a real session
KEYFRAMES = [0, 30, 60, 90, 120]
WITNESSES = FRAME_NUMBERS                        # keyframes ⊂ witnesses (the I1 invariant)

CFG = ContentConfig(
    enabled=True, backend="qwen_local", batch=3, max_tokens=256,
    exclusion_classes=("dynamic", "occluder"), weight_classes=("reflective", "low_info"),
    sam3_scope="flagged_ranges",
    prompts={"dynamic": ("person", "train"), "occluder": ("hand",)},
    sam3_batch=250,
)
RECTS = {                                        # (r0, r1, c0, c1) per prompt
    "person": (10, 40, 20, 60),
    "train": (50, 70, 100, 150),
    "hand": (80, 110, 0, 30),
}
QUIET = lambda *a, **k: None                     # noqa: E731


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cam = S.default_camera(W, H)
    scene = S.room_scene(seed=0)
    poses = S.walk_poses("translate", len(FRAME_NUMBERS), start=S.room_start_pose(),
                         step_m=0.05)
    root = tmp_path_factory.mktemp("content")
    return S.write_session(root / "sess", scene, cam, poses, frame_numbers=FRAME_NUMBERS)


class SpyTagger:
    """Tags exactly the frames named in ``flags``; records every call."""

    def __init__(self, flags=None):
        self.flags = flags or {}
        self.calls = []
        self.n_calls = 0
        self.parse_failures = 0

    def tag(self, images, frames):
        self.calls.append((list(frames), [im.shape for im in images], [im.dtype for im in images]))
        self.n_calls += 1
        out = []
        for f in frames:
            t = {cls: bool(self.flags.get(f, {}).get(cls, False)) for cls in CONTENT_CLASSES}
            t["notes"] = f"frame {f}"
            out.append(t)
        return out


class RectSegmenter:
    """Draws one rectangle per prompt on every requested frame; records calls."""

    def __init__(self, hw=(H, W), rects=RECTS):
        self.hw = hw
        self.rects = rects
        self.calls = []
        self.closed = 0

    def masks(self, frames_dir, frame_ids, prompt):
        self.calls.append((prompt, list(frame_ids)))
        r0, r1, c0, c1 = self.rects[prompt]
        m = np.zeros(self.hw, dtype=bool)
        m[r0:r1, c0:c1] = True
        return {int(f): m.copy() for f in frame_ids}

    def close(self):
        self.closed += 1


class Boom:
    """Stands in for a production class: constructing it is the failure."""

    def __init__(self, *a, **k):
        raise AssertionError("production class constructed on the disabled path")


def _rect_mask(prompts):
    m = np.zeros((H, W), dtype=bool)
    for p in prompts:
        r0, r1, c0, c1 = RECTS[p]
        m[r0:r1, c0:c1] = True
    return m


# ── the happy path ───────────────────────────────────────────────────────

def test_run_content_writes_tags_and_masks(session):
    tagger = SpyTagger({30: {"dynamic": True}, 90: {"reflective": True}})
    seg = RectSegmenter()
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=tagger,
                         segmenter=seg, log=QUIET, heartbeat_s=1e-6)

    p = session.session_dir / "intake" / "content_tags.json"
    assert p.exists() and not p.with_name(p.name + ".tmp").exists()
    disk = json.loads(p.read_text())
    assert disk == rep
    assert rep["version"] == 1 and rep["provenance"] == "vlm_proposed"
    assert rep["geometry_epoch"] == 0 and rep["camera_epoch"] == 0
    assert rep["enabled"] is True and rep["backend"] == "qwen_local"
    assert rep["sam3_handover"] == {"called": False, "verified": False, "check": None,
                                    "reason": "no before_sam3 hook given by the caller"}
    # every frame SAM3 was asked about has a recorded outcome (the Segmenter contract:
    # a frame without a mask was segmented and holds nothing)
    status = rep["exclusion_masks"]["frame_status"]
    asked = {f for cls_frames in [rep["exclusion_masks"]["frames"]] for f in cls_frames}
    assert set(status) >= asked and set(status.values()) <= {"masked", "segmented_no_object"}
    assert rep["exclusion_masks"]["n_segmented_no_object"] == \
        sum(1 for v in status.values() if v == "segmented_no_object")
    assert all(status[f] == "masked" for f in rep["exclusion_masks"]["frames"])
    assert rep["classes"] == {"exclusion": ["dynamic", "occluder"],
                              "weight": ["reflective", "low_info"]}
    assert (rep["native_w"], rep["native_h"]) == (W, H)
    assert sorted(int(k) for k in rep["frames"]) == KEYFRAMES
    assert rep["frames"]["30"]["dynamic"] is True and rep["frames"]["30"]["notes"] == "frame 30"
    assert rep["frames"]["0"] == {**{c: False for c in CONTENT_CLASSES}, "notes": "frame 0"}
    assert rep["weights"] == {"reflective": [90], "low_info": []}
    assert rep["parse_failures"] == 0
    assert rep["vlm_calls"] == {"n_calls": 2, "n_parse_failed": 0}      # 5 keyframes / batch 3
    assert rep["summary"]["tagged"] == {"dynamic": 1, "occluder": 0, "reflective": 1,
                                        "low_info": 0}
    assert rep["params"]["prompts"] == {"dynamic": ["person", "train"], "occluder": ["hand"]}
    assert rep["inputs"]["n_keyframes"] == 5 and rep["inputs"]["n_witnesses"] == 13

    # keyframe 30 flagged dynamic → witnesses in [0, 60]; occluder had no flag
    expected = [0, 10, 20, 30, 40, 50, 60]
    em = rep["exclusion_masks"]
    assert em["scope"] == "flagged_ranges"
    assert em["requested"] == {"dynamic": 7, "occluder": 0}
    assert em["prompts"] == {"dynamic": ["person", "train"], "occluder": ["hand"]}
    assert sorted(int(k) for k in em["frames"]) == expected
    assert em["n_frames_written"] == len(expected)
    masks_dir = Path(em["dir"])
    assert masks_dir == session.session_dir / "intake" / "exclusion_masks"
    want = _rect_mask(["person", "train"])
    for f in expected:
        png = masks_dir / f"{f:06d}.png"
        assert png.exists()
        arr = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        assert arr.dtype == np.uint8 and arr.shape == (H, W)          # native, single channel
        assert set(np.unique(arr)) <= {0, 255}
        assert np.array_equal(arr == 255, want)
        assert em["frames"][str(f)] == int(want.sum())
        assert np.array_equal(Cn.read_mask_png(png), want)
    for f in set(FRAME_NUMBERS) - set(expected):
        assert not (masks_dir / f"{f:06d}.png").exists()
    # an injected segmenter belongs to the caller: not closed here
    assert seg.closed == 0
    assert Cn.load_content(session.session_dir) == rep


def test_tag_keyframes_batches_by_config_and_feeds_native_rgb(session):
    tagger = SpyTagger()
    tags = Cn.tag_keyframes(session.frames_dir, KEYFRAMES, tagger, CFG, log=QUIET,
                            heartbeat_s=1e-6)
    assert [c[0] for c in tagger.calls] == [[0, 30, 60], [90, 120]]
    for _, shapes, dtypes in tagger.calls:
        assert all(s == (H, W, 3) for s in shapes) and all(d == np.uint8 for d in dtypes)
    assert sorted(tags) == KEYFRAMES
    assert all(set(t) == set(Cn.TAG_KEYS) for t in tags.values())
    big = replace(CFG, batch=8)
    tagger = SpyTagger()
    Cn.tag_keyframes(session.frames_dir, KEYFRAMES, tagger, big, log=QUIET, heartbeat_s=1e-6)
    assert [c[0] for c in tagger.calls] == [KEYFRAMES]


# ── flagged_ranges ───────────────────────────────────────────────────────

def _tags(flagged, cls="dynamic"):
    return {k: {**Cn.empty_tags(), cls: k in flagged} for k in KEYFRAMES}


def test_flagged_ranges_semantics():
    wits = list(range(0, 125, 5))
    # a middle keyframe claims the witnesses between its two neighbours, inclusive
    assert Cn.flagged_ranges(KEYFRAMES, _tags({60}), wits, "dynamic") == list(range(30, 95, 5))
    # nothing tagged → nothing
    assert Cn.flagged_ranges(KEYFRAMES, _tags(set()), wits, "dynamic") == []
    # the class matters: tagged 'dynamic' says nothing about 'occluder'
    assert Cn.flagged_ranges(KEYFRAMES, _tags({60}), wits, "occluder") == []
    # first / last keyframe: open end on the side without a neighbour
    assert Cn.flagged_ranges(KEYFRAMES, _tags({0}), wits, "dynamic") == list(range(0, 35, 5))
    assert Cn.flagged_ranges(KEYFRAMES, _tags({120}), wits, "dynamic") == list(range(90, 125, 5))
    # two tagged keyframes → union of their ranges
    assert Cn.flagged_ranges(KEYFRAMES, _tags({30, 90}), wits, "dynamic") == \
        sorted(set(range(0, 65, 5)) | set(range(60, 125, 5)))
    # a tagged keyframe is included even when it is not a witness; witnesses
    # outside every range are not
    assert Cn.flagged_ranges(KEYFRAMES, _tags({60}), [3, 47, 118], "dynamic") == [47, 60]
    # order and duplicates in the inputs do not matter
    assert Cn.flagged_ranges([120, 60, 0, 90, 30, 60], _tags({60}), wits[::-1] + [50],
                             "dynamic") == list(range(30, 95, 5))
    with pytest.raises(Cn.ContentError, match="unknown content class"):
        Cn.flagged_ranges(KEYFRAMES, _tags({60}), wits, "sky")
    with pytest.raises(Cn.ContentError, match="no tags"):
        Cn.flagged_ranges(KEYFRAMES + [200], _tags({60}), wits, "dynamic")


def test_scope_all_segments_every_frame(session):
    cfg = replace(CFG, sam3_scope="all")
    tagger = SpyTagger()                                       # nothing flagged
    seg = RectSegmenter()
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, cfg, tagger=tagger,
                         segmenter=seg, log=QUIET, heartbeat_s=1e-6)
    assert rep["exclusion_masks"]["scope"] == "all"
    assert rep["exclusion_masks"]["requested"] == {"dynamic": 13, "occluder": 13}
    assert [c[0] for c in seg.calls] == ["person", "train", "hand"]
    assert all(c[1] == FRAME_NUMBERS for c in seg.calls)
    assert sorted(int(k) for k in rep["exclusion_masks"]["frames"]) == FRAME_NUMBERS
    want = _rect_mask(["person", "train", "hand"])
    arr = Cn.read_mask_png(Path(rep["exclusion_masks"]["dir"]) / "000050.png")
    assert np.array_equal(arr, want)
    assert Cn.scope_frames(KEYFRAMES, _tags(set()), [7, 3], "dynamic", "all") == [0, 3, 7, 30, 60, 90, 120]
    with pytest.raises(Cn.ContentError, match="sam3_scope"):
        Cn.scope_frames(KEYFRAMES, _tags(set()), [], "dynamic", "some")


# ── the VLM parse path (real QwenTagger, fake client) ────────────────────

class FakeClient:
    """Answers from a queue; records every chat call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice=None, temperature=None,
             max_tokens=None, extra_body=None, consumer=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens,
                           "consumer": consumer})
        return SimpleNamespace(content=self.answers.pop(0), tool_calls=[],
                               finish_reason="stop", model="fake", usage={}, latency_ms=0.0)


def _good_answer(n, flags=None, fenced=False, shuffle=False):
    flags = flags or {}
    objs = []
    for i in range(n):
        o = {"image": i, **{c: False for c in CONTENT_CLASSES}, "notes": f"img {i}"}
        o.update(flags.get(i, {}))
        objs.append(o)
    if shuffle:
        objs = objs[::-1]
    txt = json.dumps(objs)
    return f"```json\n{txt}\n```" if fenced else txt


def test_qwen_tagger_parses_and_fails_without_crashing(session):
    imgs = [Cn.read_rgb(Cn.frame_file(session.frames_dir, f)) for f in (0, 30, 60)]
    client = FakeClient([
        _good_answer(3, {1: {"dynamic": True, "notes": "worker"}}, fenced=True, shuffle=True),
        "I am sorry, I cannot help with that.",                     # no JSON at all
        _good_answer(2),                                            # wrong count for 3 images
        json.dumps([{"image": 0, "dynamic": "yes", "occluder": "no", "reflective": 0,
                     "low_info": "false", "notes": None}]),         # tolerant booleans
        json.dumps([{"image": 0, "dynamic": "maybe", "occluder": False, "reflective": False,
                     "low_info": False}]),                          # not a boolean
        json.dumps([{"image": 0, "occluder": False, "reflective": False, "low_info": False}]),
    ])
    tagger = Cn.QwenTagger(CFG, client=client)

    out = tagger.tag(imgs, [0, 30, 60])
    assert [t["dynamic"] for t in out] == [False, True, False]      # re-ordered by image index
    assert out[1]["notes"] == "worker" and out[0]["notes"] == "img 0"
    assert tagger.n_calls == 1 and tagger.parse_failures == 0
    call = client.calls[0]
    assert call["max_tokens"] == CFG.max_tokens and call["consumer"] == "intake.content"
    assert [m.role for m in call["messages"]] == ["system", "user"]
    assert len(call["messages"][1].images) == 3                      # ≤ batch, one user message
    assert "image" in call["messages"][1].text and "low_info" in call["messages"][1].text

    out = tagger.tag(imgs, [0, 30, 60])
    assert out == [Cn.failed_tags()] * 3
    assert all(t["notes"] == "vlm_parse_failed" and not any(t[c] for c in CONTENT_CLASSES)
               for t in out)
    assert tagger.parse_failures == 1

    out = tagger.tag(imgs, [0, 30, 60])                              # 2 objects for 3 images
    assert out == [Cn.failed_tags()] * 3 and tagger.parse_failures == 2

    out = tagger.tag(imgs[:1], [0])
    assert out == [{"dynamic": True, "occluder": False, "reflective": False,
                    "low_info": False, "notes": ""}]
    assert tagger.parse_failures == 2

    assert tagger.tag(imgs[:1], [0]) == [Cn.failed_tags()] and tagger.parse_failures == 3
    assert tagger.tag(imgs[:1], [0]) == [Cn.failed_tags()] and tagger.parse_failures == 4
    assert tagger.n_calls == 6

    with pytest.raises(Cn.ContentError, match="exceed"):
        tagger.tag(imgs + imgs, [0, 30, 60, 1, 2, 3])
    with pytest.raises(Cn.ContentError, match="frame number"):
        tagger.tag(imgs, [0, 30])
    assert tagger.tag([], []) == []


def test_run_content_counts_parse_failures_per_frame(session):
    client = FakeClient(["garbage", _good_answer(2, {0: {"reflective": True}})])
    tagger = Cn.QwenTagger(CFG, client=client)
    seg = RectSegmenter()
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=tagger,
                         segmenter=seg, log=QUIET, heartbeat_s=1e-6)
    assert rep["parse_failures"] == 3                                # the first batch of 3
    assert rep["vlm_calls"] == {"n_calls": 2, "n_parse_failed": 1}
    for k in ("0", "30", "60"):
        assert rep["frames"][k] == Cn.failed_tags()
    assert rep["frames"]["90"]["reflective"] is True and rep["weights"]["reflective"] == [90]
    assert rep["exclusion_masks"]["requested"] == {"dynamic": 0, "occluder": 0}
    assert seg.calls == [] and rep["exclusion_masks"]["frames"] == {}


# ── the disabled path ────────────────────────────────────────────────────

def test_disabled_path_writes_json_and_calls_nothing(session, monkeypatch):
    monkeypatch.setattr(Cn, "QwenTagger", Boom)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Boom)
    cfg = replace(CFG, enabled=False)
    tagger, seg = SpyTagger({30: {"dynamic": True}}), RectSegmenter()
    logs = []
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, cfg, tagger=tagger,
                         segmenter=seg, log=logs.append, heartbeat_s=1e-6)
    assert tagger.calls == [] and seg.calls == [] and seg.closed == 0
    # production defaults are not even constructed
    rep2 = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, cfg, log=QUIET,
                          heartbeat_s=1e-6)
    assert rep2 == rep
    disk = json.loads((session.session_dir / "intake" / "content_tags.json").read_text())
    assert disk == rep
    assert rep["enabled"] is False and rep["provenance"] == "vlm_proposed" and rep["version"] == 1
    assert "enabled is false" in rep["reason"]
    assert rep["frames"] == {} and rep["parse_failures"] == 0 and rep["vlm_calls"] is None
    assert rep["exclusion_masks"]["frames"] == {} and rep["exclusion_masks"]["requested"] == {}
    assert rep["exclusion_masks"]["scope"] == "flagged_ranges"
    assert rep["weights"] == {"reflective": [], "low_info": []}
    assert rep["classes"] == {"exclusion": ["dynamic", "occluder"],
                              "weight": ["reflective", "low_info"]}
    assert rep["inputs"]["n_keyframes"] == 5
    assert any("enabled is false" in m for m in logs)


# ── prompts, contracts, failures with reasons ────────────────────────────

def test_sam3_prompts_come_from_config_only(session):
    tagger = SpyTagger({60: {"dynamic": True, "occluder": True}})
    seg = RectSegmenter()
    Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=tagger,
                   segmenter=seg, log=QUIET, heartbeat_s=1e-6)
    prompts = [c[0] for c in seg.calls]
    assert prompts == ["person", "train", "hand"]                    # cfg.prompts, in order
    assert set(prompts) == set(CFG.prompts["dynamic"]) | set(CFG.prompts["occluder"])
    assert all(c[1] == [30, 40, 50, 60, 70, 80, 90] for c in seg.calls)

    other = replace(CFG, prompts={"dynamic": ("excavator",), "occluder": ("glove", "strap")})
    seg = RectSegmenter(rects={"excavator": RECTS["person"], "glove": RECTS["hand"],
                               "strap": RECTS["train"]})
    Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, other, tagger=SpyTagger(
        {60: {"dynamic": True, "occluder": True}}), segmenter=seg, log=QUIET, heartbeat_s=1e-6)
    assert [c[0] for c in seg.calls] == ["excavator", "glove", "strap"]

    # only the classes with a frame in scope are segmented
    seg = RectSegmenter()
    Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG,
                   tagger=SpyTagger({60: {"occluder": True}}), segmenter=seg, log=QUIET,
                   heartbeat_s=1e-6)
    assert [c[0] for c in seg.calls] == ["hand"]


def test_build_exclusion_masks_direct_and_its_contract(session, tmp_path):
    out = tmp_path / "masks"
    seg = RectSegmenter()
    counts = Cn.build_exclusion_masks(session.frames_dir, out, {"dynamic": [0, 10], "occluder": []},
                                      {"dynamic": ["person"], "occluder": ["hand"]}, seg,
                                      (W, H), log=QUIET, heartbeat_s=1e-6)
    assert counts == {0: int(_rect_mask(["person"]).sum()), 10: int(_rect_mask(["person"]).sum())}
    assert [c[0] for c in seg.calls] == ["person"]                   # occluder: no frame, no call
    assert sorted(p.name for p in out.iterdir()) == ["000000.png", "000010.png"]

    class Empty:
        def masks(self, frames_dir, frame_ids, prompt):
            return {int(f): np.zeros((H, W), bool) for f in frame_ids}

    assert Cn.build_exclusion_masks(session.frames_dir, tmp_path / "m2", {"dynamic": [0]},
                                    {"dynamic": ["person"]}, Empty(), (W, H), log=QUIET,
                                    heartbeat_s=1e-6) == {}
    assert not (tmp_path / "m2" / "000000.png").exists()

    class Wrong(RectSegmenter):
        def masks(self, frames_dir, frame_ids, prompt):
            return {int(f): np.zeros((H // 2, W // 2), bool) for f in frame_ids}

    with pytest.raises(Cn.ContentError, match=r"\(60, 80\).*native grid.*\(120, 160\)"):
        Cn.build_exclusion_masks(session.frames_dir, tmp_path / "m3", {"dynamic": [0]},
                                 {"dynamic": ["person"]}, Wrong(), (W, H), log=QUIET,
                                 heartbeat_s=1e-6)

    class Extra(RectSegmenter):
        def masks(self, frames_dir, frame_ids, prompt):
            m = super().masks(frames_dir, frame_ids, prompt)
            m[999] = np.zeros((H, W), bool)
            return m

    with pytest.raises(Cn.ContentError, match="999.*not requested"):
        Cn.build_exclusion_masks(session.frames_dir, tmp_path / "m4", {"dynamic": [0]},
                                 {"dynamic": ["person"]}, Extra(), (W, H), log=QUIET,
                                 heartbeat_s=1e-6)
    with pytest.raises(Cn.ContentError, match="prompts.dynamic"):
        Cn.build_exclusion_masks(session.frames_dir, tmp_path / "m5", {"dynamic": [0]},
                                 {"occluder": ["hand"]}, seg, (W, H), log=QUIET,
                                 heartbeat_s=1e-6)


def test_tagger_contract_breaches_fail_with_reason(session):
    class Short(SpyTagger):
        def tag(self, images, frames):
            return super().tag(images, frames)[:-1]

    with pytest.raises(Cn.ContentError, match="one dict per image"):
        Cn.tag_keyframes(session.frames_dir, KEYFRAMES, Short(), CFG, log=QUIET, heartbeat_s=1e-6)

    class NoBool(SpyTagger):
        def tag(self, images, frames):
            out = super().tag(images, frames)
            out[0]["dynamic"] = "yes"
            return out

    with pytest.raises(Cn.ContentError, match="boolean 'dynamic'"):
        Cn.tag_keyframes(session.frames_dir, KEYFRAMES, NoBool(), CFG, log=QUIET, heartbeat_s=1e-6)

    with pytest.raises(Cn.ContentError, match="no file"):
        Cn.tag_keyframes(session.frames_dir, [0, 5], SpyTagger(), CFG, log=QUIET, heartbeat_s=1e-6)
    with pytest.raises(Cn.ContentError, match="no keyframes"):
        Cn.tag_keyframes(session.frames_dir, [], SpyTagger(), CFG, log=QUIET, heartbeat_s=1e-6)
    with pytest.raises(Cn.ContentError, match="heartbeat_s"):
        Cn.tag_keyframes(session.frames_dir, KEYFRAMES, SpyTagger(), CFG, log=QUIET, heartbeat_s=0)


def test_rerun_removes_stale_masks_and_closes_an_owned_segmenter(session, monkeypatch):
    seg = RectSegmenter()
    Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG,
                   tagger=SpyTagger({30: {"dynamic": True}}), segmenter=seg, log=QUIET,
                   heartbeat_s=1e-6)
    masks_dir = session.session_dir / "intake" / "exclusion_masks"
    assert (masks_dir / "000000.png").exists()

    owned = RectSegmenter()
    monkeypatch.setattr(Cn, "Sam3Segmenter", lambda *a, **k: owned)
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG,
                         tagger=SpyTagger({120: {"dynamic": True}}), log=QUIET, heartbeat_s=1e-6)
    assert owned.closed == 1                                          # constructed here → closed here
    assert rep["exclusion_masks"]["stale_removed"] == 7
    assert sorted(p.name for p in masks_dir.iterdir()) == ["000090.png", "000100.png",
                                                            "000110.png", "000120.png"]
    assert sorted(int(k) for k in rep["exclusion_masks"]["frames"]) == [90, 100, 110, 120]


def test_epochs_are_read_from_the_session(session):
    out = session.session_dir / "output"
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 3}))
    from precision.camera import save_camera_json
    cam = replace(session.cam.to_precision_camera(), camera_epoch=1)
    save_camera_json(out / "camera.json", cam, geometry_epoch=3)
    try:
        rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=SpyTagger(),
                             segmenter=RectSegmenter(), log=QUIET, heartbeat_s=1e-6)
        assert rep["geometry_epoch"] == 3 and rep["camera_epoch"] == 1
    finally:
        (out / "geometry_epoch.json").unlink()
        (out / "camera.json").unlink()


def test_heartbeat_is_required(session):
    with pytest.raises(TypeError, match="heartbeat_s"):
        Cn.tag_keyframes(session.frames_dir, KEYFRAMES, SpyTagger(), CFG, log=QUIET)
    with pytest.raises(TypeError, match="heartbeat_s"):
        Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=SpyTagger(),
                       segmenter=RectSegmenter(), log=QUIET)


# ── GPU exclusivity: tags → before_sam3 → SAM3 ───────────────────────────

class _Events(list):
    """One ordered record shared by the tagger, the hook and the segmenter."""


class OrderTagger(SpyTagger):
    def __init__(self, events, flags=None):
        super().__init__(flags)
        self.events = events

    def tag(self, images, frames):
        self.events.append(("tag", tuple(frames)))
        return super().tag(images, frames)


class OrderSegmenter(RectSegmenter):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def masks(self, frames_dir, frame_ids, prompt):
        self.events.append(("sam3", prompt))
        return super().masks(frames_dir, frame_ids, prompt)


def test_gpu_handover_happens_after_every_tag_and_before_any_sam3(session):
    ev = _Events()
    cfg = replace(CFG, sam3_scope="all")
    check = {"service_stopped": True, "check": "pgrep -f 'vllm serve'", "remaining_pids": [],
             "free_gb": 44.0}
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, cfg,
                         tagger=OrderTagger(ev), segmenter=OrderSegmenter(ev), log=QUIET,
                         heartbeat_s=1e-6,
                         before_sam3=lambda: ev.append(("before_sam3",)) or check)
    kinds = [e[0] for e in ev]
    assert kinds.count("before_sam3") == 1
    cut = kinds.index("before_sam3")
    assert set(kinds[:cut]) == {"tag"} and set(kinds[cut + 1:]) == {"sam3"}
    assert sum(len(e[1]) for e in ev[:cut]) == len(KEYFRAMES)      # every keyframe tagged first
    assert [e[1] for e in ev[cut + 1:]] == ["person", "train", "hand"]
    assert rep["sam3_handover"] == {"called": True, "verified": True, "check": check,
                                    "reason": "before the first SAM3 call"}
    # a hook that does not report its verification is recorded as unverified
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, cfg,
                         tagger=OrderTagger(_Events()), segmenter=OrderSegmenter(_Events()),
                         log=QUIET, heartbeat_s=1e-6, before_sam3=lambda: None)
    assert rep["sam3_handover"]["called"] is True and rep["sam3_handover"]["verified"] is False
    assert "no verification" in rep["sam3_handover"]["reason"]


def test_gpu_handover_is_skipped_when_sam3_has_nothing_to_do(session):
    calls = []
    # flagged_ranges and nothing tagged: SAM3 is never called, the GPU stays as is
    rep = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, CFG, tagger=SpyTagger(),
                         segmenter=RectSegmenter(), log=QUIET, heartbeat_s=1e-6,
                         before_sam3=lambda: calls.append(1))
    assert calls == [] and rep["sam3_handover"]["called"] is False
    assert "no frame in scope" in rep["sam3_handover"]["reason"]
    # disabled content: nothing at all
    off = Cn.run_content(session.session_dir, KEYFRAMES, WITNESSES, replace(CFG, enabled=False),
                         log=QUIET, heartbeat_s=1e-6, before_sam3=lambda: calls.append(1))
    assert calls == [] and off["sam3_handover"]["called"] is False


def test_sam3_segmenter_batch_is_validated_without_loading_a_model():
    seg = Cn.Sam3Segmenter(250, log=QUIET)
    assert seg.batch_size == 250
    for bad in (0, -3, 2.5, True, None):
        with pytest.raises(Cn.ContentError, match="batch_size"):
            Cn.Sam3Segmenter(bad, log=QUIET)
    assert seg.masks("/nowhere", [], "person") == {}                 # nothing to segment


def test_vlm_prompt_assumes_no_domain():
    text = (Cn.SYSTEM_PROMPT + " " + Cn.build_tag_prompt(2, [0, 1])).lower()
    for word in ("construction", "railway", "tunnel", "site"):
        assert word not in text, word


def test_cancel_stops_the_tag_loop(session):
    from intake.quality import IntakeCancelled
    with pytest.raises(IntakeCancelled, match=r"intake I2 \(VLM tags\) at keyframe 0/5"):
        Cn.tag_keyframes(session.frames_dir, KEYFRAMES, SpyTagger(), CFG, log=QUIET,
                         heartbeat_s=1e-6, cancelled=lambda: True)


def test_cli_disabled_path(session, monkeypatch, capsys):
    frames_dir = session.frames_dir
    (frames_dir / "selected_frames.json").write_text(json.dumps({
        "version": "2.0", "method": "parallax_lk_12", "total_frames": 13, "selected_count": 5,
        "selected_files": [f"{k:06d}.jpg" for k in KEYFRAMES]}))
    (frames_dir / "witness_frames.json").write_text(json.dumps({
        "version": 1, "provenance": "tool_measured", "method": "parallax_lk",
        "frames": [{"frame": f, "file": f"{f:06d}.jpg", "acc_parallax_px": 0.0,
                    "is_keyframe": f in KEYFRAMES} for f in WITNESSES],
        "selected_files": [f"{f:06d}.jpg" for f in WITNESSES]}))
    assert Cn.read_i1_frames(session.session_dir) == (KEYFRAMES, WITNESSES)

    fake = SimpleNamespace(content=replace(CFG, enabled=False),
                           runtime=SimpleNamespace(heartbeat_s=1.0))
    monkeypatch.setattr(Cn, "load_intake_config", lambda raw=None: fake)
    monkeypatch.setattr(Cn, "QwenTagger", Boom)
    monkeypatch.setattr(Cn, "Sam3Segmenter", Boom)
    assert Cn.main(["--session", str(session.session_dir)]) == 0
    out = capsys.readouterr().out
    assert "enabled=False" in out and "keyframes=5" in out and "witnesses=13" in out
    rep = Cn.load_content(session.session_dir)
    assert rep["enabled"] is False

    (frames_dir / "witness_frames.json").unlink()
    with pytest.raises(Cn.ContentError, match="witness_frames.json"):
        Cn.read_i1_frames(session.session_dir)
    (frames_dir / "selected_frames.json").unlink()


# ── the production Segmenter, driven through a fake SAM3 wrapper ──────────

class FakeSam3:
    """Stands in for segmentation.sam3_wrapper's wrapper: reads the batch dir the
    REAL segmentation.pipeline._prepare_batch_dir built, answers per local index
    in the wrapper's raw format ({frame: {out_binary_masks (n, H, W), out_obj_ids}})
    — two objects per frame, placed by the FRAME NUMBER behind the symlink so a
    broken local→frame mapping shows — and can drop frames or log a swallowed
    failure the way the real one does."""

    def __init__(self, hw=(H, W), drop=(), log_error=None):
        self.hw, self.drop, self.log_error = hw, set(drop), log_error
        self.calls, self.released, self.unloaded = [], 0, 0

    def process_batch(self, batch_dir, prompt, index_mapping, prompt_frames=None,
                      boxes_by_local=None):
        import logging
        files = sorted(Path(batch_dir).iterdir())
        self.calls.append((prompt, dict(index_mapping), [f.name for f in files]))
        if self.log_error:
            logging.getLogger(Cn.SAM3_LOGGER_NAME).error(self.log_error)
        out = {}
        for local, link in enumerate(files):
            frame = int(Path(link.resolve()).stem)          # the frame behind the symlink
            assert index_mapping[local] == frame
            if frame in self.drop:
                continue
            m = np.zeros((2,) + self.hw, dtype=bool)
            r = (frame // 10) % (self.hw[0] - 10)
            m[0, r:r + 5, 0:10] = True
            m[1, r:r + 5, 20:30] = True
            out[frame] = {"out_binary_masks": m, "out_obj_ids": np.array([1, 2])}
        return out

    def release_batch_session(self):
        self.released += 1

    def unload_model(self):
        self.unloaded += 1


@pytest.fixture
def fake_sam3(monkeypatch):
    sw = pytest.importorskip("segmentation.sam3_wrapper")   # the real module (no model load)
    holder = {}

    def install(fake):
        holder["fake"] = fake
        monkeypatch.setattr(sw, "get_sam3_wrapper", lambda: fake)
        return fake
    return install


def test_sam3_segmenter_production_path_chunks_maps_and_unions(session, fake_sam3):
    """Chunking by the configured SAM3 batch, the local→frame mapping of the
    real _prepare_batch_dir, the OR over objects, masks keyed by frame at the
    native size, and close() releasing the session and the model."""
    fake = fake_sam3(FakeSam3())
    frames = FRAME_NUMBERS[:7]
    seg = Cn.Sam3Segmenter(3, log=QUIET)
    out = seg.masks(session.frames_dir, frames, "person")
    assert [len(c[1]) for c in fake.calls] == [3, 3, 1]            # chunks of batch_size
    assert [sorted(c[1].values()) for c in fake.calls] == [frames[0:3], frames[3:6], frames[6:7]]
    assert sorted(out) == frames
    for f, m in out.items():
        assert m.shape == (H, W) and m.dtype == bool
        r = (f // 10) % (H - 10)
        assert m[r:r + 5, 0:10].all() and m[r:r + 5, 20:30].all()   # both objects, OR-ed
        assert m.sum() == 2 * 5 * 10
    seg.close()
    assert fake.released == 1 and fake.unloaded == 1


def test_sam3_segmenter_fails_on_a_partial_result(session, fake_sam3):
    """The wrapper swallows its exceptions and returns what it propagated: a
    frame without a result entry is a failure, never 'nothing found'."""
    fake_sam3(FakeSam3(drop={FRAME_NUMBERS[4]}))
    seg = Cn.Sam3Segmenter(3, log=QUIET)
    with pytest.raises(Cn.ContentError, match=rf"partial result: 2/3 frame\(s\) — missing \[{FRAME_NUMBERS[4]}\]"):
        seg.masks(session.frames_dir, FRAME_NUMBERS[:7], "person")


def test_sam3_segmenter_fails_on_a_swallowed_wrapper_error(session, fake_sam3):
    fake_sam3(FakeSam3(log_error="Error during batch processing: CUDA error: device-side assert"))
    seg = Cn.Sam3Segmenter(3, log=QUIET)
    with pytest.raises(Cn.ContentError, match="swallowed .*device-side assert"):
        seg.masks(session.frames_dir, FRAME_NUMBERS[:3], "person")
    assert Cn.sam3_failures([]) == []
