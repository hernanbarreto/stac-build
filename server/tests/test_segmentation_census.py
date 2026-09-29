"""output/segmentation_census.json — no proposed concept disappears without a
recorded reason (USER 2026-09-29: "debe segmentar todo, absolutamente preciso y
completo").

On pccr 'white folding table', 'doorbell panel', 'conduit' and 'exposed ceiling
pipes' reached SAM3 and got 0 masklets, 'cardboard box' never reached it, and
nothing on disk said which was which. The census is built by the SAM3 worker
from what the workers already wrote; these tests build its inputs synthetically
(no vLLM, no SAM3 weights) and pin: the accounting closes (every raw concept has
exactly one fate), zero-masklet prompts are flagged down to their concepts, and
masklet spans / visits are read off the masks with the mask filter's own gap.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space  # noqa: E402
from segmentation.census import (  # noqa: E402
    CENSUS_NAME, build_census, concept_label, visit_gap_kf, visits_of)

KF_FILES = [f"{v:06d}.jpg" for v in (0, 51, 61, 75, 91, 120, 150, 180, 200, 230, 260, 300)]


def _fates():
    """The VLM half, shaped exactly as session_builder._concept_fates writes it."""
    from segmentation.autoprompt.scene_understanding import FrameUnderstanding, aggregate
    from segmentation.autoprompt.session_builder import AutoPrompter
    und = aggregate([
        FrameUnderstanding(0, "server room", "", ["black office chair", "cardboard box",
                                                  "red fire alarm box"]),
        FrameUnderstanding(51, "server room", "", ["black office chairs", "doorbell panel"]),
        FrameUnderstanding(51, "server room", "", ["doorbell panel", "conduit"], tile="r0c1"),
    ])
    fates = AutoPrompter._concept_fates(und, und.objects, reused=False, consolidation=None)
    return und, fates


def _session(tmp_path, prompts, instances, masks, *, space=mask_space.SPACE_KEYFRAME):
    frames = tmp_path / "frames"
    frames.mkdir()
    for f in KF_FILES:
        (frames / f).write_bytes(b"")
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": KF_FILES}))
    out = tmp_path / "output"
    out.mkdir()
    (out / "segmentation.json").write_text(json.dumps(
        {"prompt": ";".join(prompts), "instances": instances, "mask_file": "seg_masks.npz"}))
    data = {k: v for k, v in masks.items()}
    data[mask_space.NPZ_KEY] = mask_space.declaration(space)
    np.savez_compressed(out / "seg_masks.npz", **data)
    return out


def _m(on=True):
    a = np.zeros((4, 4), np.uint8)
    if on:
        a[1:3, 1:3] = 1
    return a


def test_accounting_closes_and_zero_masklet_prompts_are_flagged(tmp_path):
    und, fates = _fates()
    prompts = und.objects
    assert set(prompts) == {"black office chair", "cardboard box", "red fire alarm box",
                            "doorbell panel", "conduit"}, "a different NAME is never folded"
    chair = concept_label("black office chair")
    box = concept_label("cardboard box")
    masks = {f"f{k}_o0": _m() for k in (0, 1, 2, 5, 6)}          # chair: two visits
    masks.update({f"f{k}_o1": _m() for k in (3, 4)})             # box
    masks["f7_o1"] = _m(on=False)                                # stored but EMPTY
    masks["f9_o7"] = _m()                                        # oid of no instance
    out = _session(tmp_path, prompts,
                   [{"id": 0, "label": chair, "instance_id": 1},
                    {"id": 1, "label": box, "instance_id": 2}], masks)
    vlm = {"source": "qwen3vl_autoprompt_simple", "prompt": ";".join(prompts),
           "census": {"sampling": {"n_keyframes": 12, "axis": "keyframe_index",
                                   "max_calls": 300, "bound_reached": False},
                      "calls": [{"frame": 0, "file": "000000.jpg", "keyframe_index": 0,
                                 "tile": None, "parsed": True, "n_objects": 3},
                                {"frame": 51, "file": "000051.jpg", "keyframe_index": 1,
                                 "tile": None, "parsed": True, "n_objects": 2},
                                {"frame": 51, "file": "000051.jpg", "keyframe_index": 1,
                                 "tile": "r0c1", "parsed": True, "n_objects": 2},
                                {"frame": 51, "file": "000051.jpg", "keyframe_index": 1,
                                 "tile": "r1c1", "parsed": False, "n_objects": 0}],
                      "concepts": fates, "vocabulary_reused": False}}
    lines = []
    doc = build_census(out, prompt=vlm["prompt"], vlm_doc=vlm, gap_kf=1,
                       frames_dir=tmp_path / "frames", log=lines.append)
    assert (out / CENSUS_NAME).exists()
    acc = doc["accounting"]
    assert acc["closed"] is True
    assert acc["n_concepts"] == 6 == sum(acc["by_fate"].values())
    assert acc["by_fate"]["merged"] == 1            # 'black office chairs' = same name
    names = {c["concept"] for c in doc["concepts"]}
    assert names == {"black office chair", "black office chairs", "cardboard box",
                     "red fire alarm box", "doorbell panel", "conduit"}
    rows = {r["prompt"]: r for r in doc["prompts"]}
    assert set(doc["zero_masklet_prompts"]) == {"red fire alarm box", "doorbell panel",
                                                 "conduit"}
    assert {c["concept"] for c in doc["concepts"] if c.get("zero_masklets")} == \
        {"red fire alarm box", "doorbell panel", "conduit"}
    ch = rows["black office chair"]
    assert ch["raw_concepts"] == ["black office chair", "black office chairs"]
    (m,) = ch["masklets"]
    assert m["visits"] == [[0, 2], [5, 6]] and m["n_visits"] == 2
    assert (m["first_kf"], m["last_kf"], m["span_kf"]) == (0, 6, 7)
    assert (m["first_frame"], m["last_frame"]) == (0, 150)
    bx = rows["cardboard box"]["masklets"][0]
    assert bx["n_keyframes"] == 2, "an EMPTY stored mask does not count as seeing it"
    assert doc["vlm"]["n_calls"] == 4 and doc["vlm"]["n_parsed"] == 3
    assert [e["frame"] for e in doc["vlm"]["looked_at"]] == [0, 51]
    assert any("ZERO masklets" in ln for ln in lines)
    assert any("accounting closed" in ln for ln in lines)


def test_a_concept_whose_prompt_sam3_never_received_is_unaccounted(tmp_path):
    und, fates = _fates()
    out = _session(tmp_path, ["black office chair"], [], {})
    vlm = {"census": {"concepts": fates, "calls": [], "sampling": None}}
    doc = build_census(out, prompt="black office chair", vlm_doc=vlm, gap_kf=1,
                       frames_dir=tmp_path / "frames", log=lambda m: None)
    acc = doc["accounting"]
    assert acc["closed"] is False
    assert set(acc["concepts_not_reaching_sam3_prompt"]) == {
        "cardboard box", "red fire alarm box", "doorbell panel", "conduit"}
    assert acc["by_fate"]["unaccounted"] == 4


def test_no_concept_record_is_declared_not_assumed(tmp_path):
    out = _session(tmp_path, ["floor"], [], {})
    doc = build_census(out, prompt="floor;wall", vlm_doc={"prompt": "floor;wall"}, gap_kf=1,
                       frames_dir=tmp_path / "frames", log=lambda m: None)
    assert doc["accounting"]["closed"] is False
    assert doc["accounting"]["concept_record"].startswith("MISSING")
    assert doc["zero_masklet_prompts"] == ["floor", "wall"]


def test_a_failed_segmentation_still_writes_the_census(tmp_path):
    und, fates = _fates()
    out = _session(tmp_path, und.objects, [{"id": 0, "label": "stale", "instance_id": 1}],
                   {"f0_o0": _m()})
    doc = build_census(out, prompt=";".join(und.objects),
                       vlm_doc={"census": {"concepts": fates, "calls": []}}, gap_kf=1,
                       frames_dir=tmp_path / "frames", seg_error="No masks generated",
                       log=lambda m: None)
    assert doc["sam3"]["error"] == "No masks generated"
    assert len(doc["zero_masklet_prompts"]) == len(und.objects)


def test_a_video_frame_store_is_translated_to_keyframe_positions(tmp_path):
    out = _session(tmp_path, ["conduit"], [{"id": 3, "label": "conduit", "instance_id": 1}],
                   {"f61_o3": _m(), "f75_o3": _m(), "f200_o3": _m()},
                   space=mask_space.SPACE_VIDEO)
    doc = build_census(out, prompt="conduit", vlm_doc=None, gap_kf=1,
                       frames_dir=tmp_path / "frames", log=lambda m: None)
    (m,) = doc["prompts"][0]["masklets"]
    assert m["visits"] == [[2, 3], [8, 8]]


def test_visits_use_the_mask_filter_gap():
    assert visits_of([3, 4, 5, 9, 10], 1) == [[3, 5], [9, 10]]
    assert visits_of([3, 4, 5, 9, 10], 4) == [[3, 10]]
    assert visits_of([], 1) == []


def test_the_gap_is_read_strictly():
    with pytest.raises(KeyError, match="segmentation.mask_filter.visit_gap_kf"):
        visit_gap_kf({"segmentation": {}})
    assert visit_gap_kf({"segmentation": {"mask_filter": {"visit_gap_kf": 1}}}) == 1


def test_the_label_rule_is_the_one_the_masks_are_saved_under():
    """pipeline._save_masks must label by concept_label, or the census attributes
    masklets to the wrong prompt."""
    src = (Path(__file__).resolve().parents[1] / "segmentation" / "pipeline.py").read_text()
    body = src[src.index("def _save_masks"):]
    body = body[:body.index("\ndef ")]
    assert "concept_label(label)" in body
    assert concept_label("Plastic-wrapped appliance") == "plastic_wrapped_appliance"


# ── the worker writes it on every run ────────────────────────────────────

class _Pipe:
    def __init__(self):
        self.logs = []

    def send_log(self, msg, level="info"):
        self.logs.append(msg)

    def send_progress(self, *a, **k):
        pass

    def check_cancel(self):
        return False


def test_the_sam3_worker_writes_the_census_after_segmentation(tmp_path, monkeypatch):
    import yaml
    import segmentation_pipeline
    from workers import sam3_worker
    und, fates = _fates()
    prompts = und.objects
    chair = concept_label("black office chair")

    def fake_run_segmentation(frames_dir, output_dir, prompt, frame_map, boxes_map,
                              on_progress):
        out = Path(output_dir)
        (out / "segmentation.json").write_text(json.dumps(
            {"prompt": prompt, "instances": [{"id": 0, "label": chair, "instance_id": 1}],
             "mask_file": "seg_masks.npz"}))
        np.savez_compressed(out / "seg_masks.npz", f0_o0=_m(), f1_o0=_m(),
                            **{mask_space.NPZ_KEY: mask_space.declaration(
                                mask_space.SPACE_KEYFRAME)})
        return {"instances": [{"id": 0}]}

    monkeypatch.setattr(segmentation_pipeline, "run_segmentation", fake_run_segmentation)
    session = tmp_path
    out = _session(session, prompts, [], {})
    (out / "segmentation.json").unlink()
    (out / "seg_masks.npz").unlink()
    (out / "vlm_analysis.json").write_text(json.dumps(
        {"prompt": ";".join(prompts), "frame_map": {},
         "census": {"concepts": fates, "calls": [], "sampling": None}}))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    cfg["reconstruction"]["simple"]["exclusive_gpu"] = False
    pipe = _Pipe()
    sam3_worker._sam3_work(pipe, str(session), cfg)
    doc = json.loads((out / CENSUS_NAME).read_text())
    assert doc["accounting"]["closed"] is True
    rows = {r["prompt"]: r for r in doc["prompts"]}
    assert rows["black office chair"]["masklets"][0]["visits"] == [[0, 1]]
    assert "conduit" in doc["zero_masklet_prompts"]
    assert any(ln.startswith("[census]") for ln in pipe.logs)
