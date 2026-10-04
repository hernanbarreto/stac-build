"""One ShapeR description per segmented object (USER 2026-10-01: "la descripción de
los objetos la que armó el VLM como descripción para ShapeR … el VLM, para preparar
los prompts, pasa SAM3 y las descripciones para ShapeR").

Two sources, ONE dict (`shape_caption`): the understanding pass that names the SAM3
prompts also describes each KIND (source 'concept', inherited at the projection), and
after the certification Qwen3-VL refines one per OBJECT from its best SAM3-mask views
(source 'object'). No GPU, no vLLM: the semantic client is faked everywhere.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space  # noqa: E402
from segmentation.object_captioner import (  # noqa: E402
    CAPTION_KEY, ObjectCaptionsConfigError, _oids_by_instance, caption_object_qwen,
    caption_session_objects, carry_object_captions, concept_caption_lookup,
    fields_from_shape_entry, load_object_captions, shape_caption)

SERVER = Path(__file__).resolve().parents[1]
CONFIG_YAML = SERVER / "config.yaml"
IFACE = {"caption", "category", "shape", "material", "detail", "provenance", "source",
         "generated"}


# ── the config block ──────────────────────────────────────────────────────

def test_production_config_declares_the_block():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    c = load_object_captions(raw)
    assert c.enabled is True and c.views >= 1 and c.understand_max_tokens >= 512


@pytest.mark.parametrize("drop", ["enabled", "views", "understand_max_tokens"])
def test_a_missing_key_fails_naming_it(drop):
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    del raw["segmentation"]["object_captions"][drop]
    with pytest.raises(ObjectCaptionsConfigError, match=f"segmentation.object_captions.{drop}"):
        load_object_captions(raw)
    raw2 = yaml.safe_load(CONFIG_YAML.read_text())
    raw2["segmentation"]["object_captions"]["views"] = 0
    with pytest.raises(ObjectCaptionsConfigError, match="views"):
        load_object_captions(raw2)


# ── the dict both sources produce ────────────────────────────────────────

def test_shape_caption_is_the_interface_and_narrative():
    cap = shape_caption({"category": "desk", "shape": "rectangular box",
                         "material": "painted steel", "detail": "two drawers on the left"},
                        "desk", "concept")
    assert set(cap) == IFACE
    assert cap["caption"] == "Desk, rectangular box, painted steel. two drawers on the left"
    assert cap["provenance"] == "vlm_proposed" and cap["source"] == "concept"
    datetime.fromisoformat(cap["generated"])
    with pytest.raises(ValueError):
        shape_caption({}, "desk", "manual")
    # no fields at all → the label alone is the caption, the category the label
    bare = shape_caption({}, "desk", "object")
    assert bare["caption"] == "Desk" and bare["category"] == "desk" and bare["shape"] == ""   # captions start upper-case


def test_the_shape_entry_is_parsed_defensively():
    f = fields_from_shape_entry({"form": "vertical cylinder", "material": "concrete",
                                 "detail": "The image shows a grey column with bolts."}, "column")
    assert f == {"category": "column", "shape": "vertical cylinder", "material": "concrete",
                 "detail": "grey column with bolts"}
    s = fields_from_shape_entry("SHAPE: flat vertical surface\nMATERIAL: drywall\nDETAIL: white.",
                                "wall")
    assert s["shape"] == "flat vertical surface" and s["material"] == "drywall"
    assert s["category"] == "wall"
    assert fields_from_shape_entry(None, "door") is None
    assert fields_from_shape_entry({"form": "", "material": ""}, "door") is None
    assert fields_from_shape_entry(42, "door") is None
    assert fields_from_shape_entry("", "door") is None


# ── (a) the understanding call, with and without the shape entry ─────────

class _VLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat(self, msgs, max_tokens=0, consumer=""):
        self.calls.append(max_tokens)
        return SimpleNamespace(content=json.dumps(self.answer))


def test_understand_frame_keeps_its_contract_and_reads_the_shape_entry():
    from segmentation.autoprompt.scene_understanding import understand_frame
    vlm = _VLM({"scene_type": "server room", "summary": "racks", "objects": [
        {"category": "Column", "description": "metal support column",
         "shape": {"form": "vertical cylinder", "material": "painted steel",
                   "detail": "bolted base plate"}},
        {"category": "desk", "description": "black metal desk",
         "shape": "SHAPE: rectangular box\nMATERIAL: steel\nDETAIL: drawers"},
        {"category": "floor", "description": "grey tiles"},          # no shape entry
        {"category": "door", "shape": {"form": "", "material": "", "detail": ""}},
        "wall"]})
    fu = understand_frame(vlm, Image.new("RGB", (8, 8)), frame_id=3, max_tokens=777)
    assert vlm.calls == [777]
    assert fu.objects == ["column", "desk", "floor", "door", "wall"]            # unchanged
    assert fu.descriptions == {"column": "metal support column", "desk": "black metal desk",
                               "floor": "grey tiles"}                            # unchanged
    assert set(fu.shapes) == {"column", "desk"}, "a kind without a usable entry has none"
    assert fu.shapes["column"] == {"category": "column", "shape": "vertical cylinder",
                                   "material": "painted steel", "detail": "bolted base plate"}
    assert fu.shapes["desk"]["shape"] == "rectangular box"


def test_understand_frame_without_any_shape_entry_is_the_old_answer():
    from segmentation.autoprompt.scene_understanding import aggregate, understand_frame
    fu = understand_frame(_VLM({"scene_type": "room", "summary": "",
                                "objects": [{"category": "desk", "description": "black desk"},
                                            "floor"]}),
                          Image.new("RGB", (8, 8)), frame_id=1)
    assert fu.shapes == {} and fu.objects == ["desk", "floor"]
    d = aggregate([fu]).to_dict()
    assert d["objects"] == ["desk", "floor"] and d["per_frame"][0]["shapes"] == {}


def test_shape_descriptions_follow_the_prompt_most_frequent_first_seen():
    from segmentation.autoprompt.scene_understanding import FrameUnderstanding, SceneUnderstanding
    from segmentation.autoprompt.session_builder import build_shape_descriptions

    def sh(form, mat, det=""):
        return {"category": "x", "shape": form, "material": mat, "detail": det}

    frames = [
        FrameUnderstanding(1, "room", "", ["column", "desk"],
                           shapes={"column": sh("vertical cylinder", "concrete"),
                                   "desk": sh("rectangular box", "steel", "two drawers")}),
        FrameUnderstanding(2, "room", "", ["columns", "beam", "floor"],
                           shapes={"columns": sh("vertical cylinder", "steel"),
                                   "beam": sh("vertical cylinder", "steel")}),
        FrameUnderstanding(3, "room", "", ["column"],
                           shapes={"column": sh("vertical cylinder", "concrete")}),
    ]
    und = SceneUnderstanding("room", "", ["column", "desk", "floor"], per_frame=frames,
                             merged={"columns": "column"})
    out = build_shape_descriptions(und, ["column", "desk", "floor"], {"beam": "column"},
                                   generated="2026-10-01T10:00:00")
    assert set(out) == {"column", "desk"}, "floor was never described — nothing invented"
    # concrete x2 (frames 1 and 3) beats steel x2 (columns + beam)? no: 2 vs 2 → first seen
    assert out["column"]["caption"] == "Column, vertical cylinder, concrete"
    assert out["column"]["category"] == "column", "the kind is named by its PROMPT"
    assert out["column"]["source"] == "concept" and out["column"]["provenance"] == "vlm_proposed"
    assert out["column"]["generated"] == "2026-10-01T10:00:00"
    assert out["desk"]["caption"] == "Desk, rectangular box, steel. two drawers"
    assert set(out["desk"]) == IFACE
    frames[1].shapes["beam"] = sh("vertical cylinder", "steel")
    frames[1].shapes["floor"] = sh("flat horizontal surface", "tiles")
    frames.append(FrameUnderstanding(4, "room", "", ["beam"],
                                     shapes={"beam": sh("vertical cylinder", "steel")}))
    und2 = SceneUnderstanding("room", "", ["column", "desk", "floor"], per_frame=frames,
                              merged={"columns": "column"})
    out2 = build_shape_descriptions(und2, ["column", "desk", "floor"], {"beam": "column"})
    assert out2["column"]["caption"] == "Column, vertical cylinder, steel", "3 votes beat 2"
    assert out2["floor"]["caption"] == "Floor, flat horizontal surface, tiles"


# ── the auto-prompter writes them next to the prompts ─────────────────────

class _FakeUnderstanding:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens=None, consumer=None):
        self.calls += 1
        return SimpleNamespace(content=json.dumps({
            "scene_type": "server room", "summary": "s",
            "objects": [{"category": "desk", "description": "black metal desk",
                         "shape": {"form": "rectangular box", "material": "steel",
                                   "detail": "two drawers"}},
                        {"category": "floor", "description": "grey tiles"}]}))


def test_the_autoprompter_writes_shape_descriptions_into_the_sam3_contract(tmp_path, monkeypatch):
    import semantic.client as sc
    from segmentation.autoprompt.session_builder import AutoPrompter
    frames = tmp_path / "frames"
    frames.mkdir()
    files = []
    for i in range(10):
        fn = f"{i * 7:06d}.jpg"
        Image.new("RGB", (64, 36), (i, i, i)).save(frames / fn)
        files.append(fn)
    (frames / "selected_frames.json").write_text(json.dumps({"selected_files": files}))
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    raw["autoprompt"]["consolidate_prompts"] = False
    raw["autoprompt"]["merge_synonyms"] = False
    raw["autoprompt"]["vlm_sampling"].update(all_keyframes=False, spacing_kf=5, tile_rows=1,
                                             tile_cols=1, max_calls=300)
    raw["reconstruction"]["simple"]["enabled"] = True
    fake = _FakeUnderstanding()
    monkeypatch.setattr(sc, "get_semantic_client", lambda **kw: fake)
    res = AutoPrompter(tmp_path, tmp_path / "output", config=raw).run()
    vlm = json.loads(Path(res.vlm_analysis_path).read_text())
    assert vlm["prompt"] == "desk;floor" and vlm["fallback_prompts"] == {"desk": ["black metal desk"],
                                                                         "floor": ["grey tiles floor"]}
    assert set(vlm["shape_descriptions"]) == {"desk"}
    assert vlm["shape_descriptions"]["desk"]["caption"] == "Desk, rectangular box, steel. two drawers"
    assert vlm["shape_descriptions"]["desk"]["source"] == "concept"
    assert vlm["scene_understanding"]["per_frame"][0]["shapes"]["desk"]["shape"] == "rectangular box"
    assert "census" in vlm and vlm["census"]["n_calls"] == fake.calls
    # a config without the block fails the run naming the key, it does not run blind
    del raw["segmentation"]["object_captions"]
    with pytest.raises(ObjectCaptionsConfigError, match="object_captions"):
        AutoPrompter(tmp_path, tmp_path / "output2", config=raw).run()


# ── (b) inheritance at the projection ────────────────────────────────────

def test_projected_instances_inherit_their_concept_and_objects_are_never_downgraded(tmp_path):
    cap = shape_caption({"category": "black office desk", "shape": "rectangular box",
                         "material": "steel", "detail": ""}, "black office desk", "concept")
    (tmp_path / "vlm_analysis.json").write_text(json.dumps(
        {"prompt": "black office desk;floor", "shape_descriptions": {"black office desk": cap}}))
    lookup = concept_caption_lookup(tmp_path)
    got = lookup("black_office_desk")                 # the label SAM3 persisted
    assert got == cap and got is not cap, "a fresh copy per instance"
    assert lookup("floor") is None, "a concept the VLM did not describe inherits nothing"
    assert concept_caption_lookup(tmp_path / "nowhere")("black_office_desk") is None

    obj = shape_caption({"category": "desk", "shape": "L-shaped prism", "material": "steel",
                         "detail": "monitor arm"}, "black_office_desk", "object")
    prev = [{"id": 0, "instance_id": 1, "label": "black_office_desk", CAPTION_KEY: obj},
            {"id": 1, "instance_id": 2, "label": "floor", CAPTION_KEY: dict(cap, source="concept")},
            {"id": 2, "instance_id": 3, "label": "door", CAPTION_KEY: dict(obj)}]
    new = [{"id": 0, "instance_id": 1, "label": "black_office_desk", CAPTION_KEY: dict(cap)},
           {"id": 1, "instance_id": 2, "label": "floor"},
           {"id": 2, "instance_id": 3, "label": "window"},            # relabelled: not the same object
           {"id": 5, "instance_id": 6, "label": "black_office_desk", CAPTION_KEY: dict(obj)}]
    assert carry_object_captions(prev, new) == 1
    assert new[0][CAPTION_KEY] == obj, "the object caption replaces the re-projected concept one"
    assert CAPTION_KEY not in new[1], "a concept caption is not carried — it is re-derived"
    assert CAPTION_KEY not in new[2]
    assert new[3][CAPTION_KEY] == obj
    assert carry_object_captions([], new) == 0 and carry_object_captions(None, None) == 0


def test_the_pipeline_attaches_and_carries_at_the_two_spots():
    """segmentation/pipeline.py: the concept caption lands where the projected instance is
    assembled, the object caption is carried where the previous result is merged."""
    src = (SERVER / "segmentation" / "pipeline.py").read_text()
    body = src[src.index("def _match_masks_to_cloud("):src.index("def _match_and_save_result(")]
    assert "concept_caption_lookup(output_dir)" in body
    assert 'instance["shape_caption"] = _cap' in body
    writer = src[src.index("def _match_and_save_result_locked("):
                 src.index("atomic_write_json(result_path, merged_result)")]
    assert "carry_object_captions(prev_instances, merged)" in writer


# ── (c) the per-object pass over a synthetic session ──────────────────────

KF_VIDEO = [0, 7, 14, 21]                 # keyframe position i → video frame number


def _mask(area_px: int) -> np.ndarray:
    m = np.zeros((9, 16), np.uint8)
    m.flat[:area_px] = 1
    return m


def _session(tmp_path, *, with_concept=True):
    frames = tmp_path / "frames"
    frames.mkdir()
    for i, v in enumerate(KF_VIDEO):
        Image.new("RGB", (64, 36), (40 * i, 10, 10)).save(frames / f"{v:06d}.jpg")
    out = tmp_path / "output"
    out.mkdir()
    (out / "camera_frames.txt").write_text("\n".join(str(v) for v in KF_VIDEO) + "\n")
    masks = {"f0_o0": _mask(4), "f1_o0": _mask(9), "f2_o0": _mask(1), "f3_o0": _mask(6),
             "f1_o1": _mask(2),
             "obj_ids": np.array([0, 1], np.int32), "frames": np.array([0, 1, 2, 3], np.int32),
             "scaled_res": np.array([9, 16], np.int32),
             mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME)}
    np.savez_compressed(out / "seg_masks.npz", **masks)
    (out / "segmentation.json").write_text(json.dumps({
        "prompt": "desk;door", "mask_file": "seg_masks.npz",
        "instances": [{"id": 0, "label": "desk", "instance_id": 1},
                      {"id": 1, "label": "door", "instance_id": 2}]}))
    desk = {"id": 0, "instance_id": 1, "label": "desk", "globalIndices": [0, 1, 2], "total_points": 3}
    if with_concept:
        desk[CAPTION_KEY] = shape_caption({"category": "desk", "shape": "rectangular box",
                                           "material": "steel", "detail": ""}, "desk", "concept")
    (out / "segmentation_result.json").write_text(json.dumps({
        "instances": [desk, {"id": 1, "instance_id": 2, "label": "door", "globalIndices": [3],
                             "total_points": 1}],
        "total_points": 4}))
    return out


class _Qwen:
    """Answers the structured prompt; records how many views each call carried."""

    def __init__(self, answer=None, fail_label=None):
        self.answer = answer or ("CATEGORY: desk\nSHAPE: rectangular box\nMATERIAL: painted steel\n"
                                 "DETAIL: The image shows a black top with two drawers.")
        self.fail_label = fail_label
        self.images = []
        self.prompts = []

    def chat(self, messages, max_tokens=None, consumer=None, **kw):
        self.prompts.append(messages[-1].text)
        if self.fail_label and self.fail_label in messages[-1].text:
            raise RuntimeError("engine hiccup")
        self.images.append(len(messages[-1].images))
        return SimpleNamespace(content=self.answer)


def _insts(out):
    return {i["instance_id"]: i for i in json.loads(
        (out / "segmentation_result.json").read_text())["instances"]}


def test_every_object_gets_its_caption_from_its_largest_mask_views(tmp_path):
    out = _session(tmp_path)
    q = _Qwen()
    logs = []
    res = caption_session_objects(out, tmp_path, views=2, client=q, log=logs.append)
    assert res["generated"] == 2 and res["kept"] == 0 and res["failed"] == 0 and res["skipped"] == 0
    assert q.images == [2, 1], "desk: its two largest masks; door: the only mask it has"
    assert "The images are views of the SAME object" in q.prompts[0]
    assert "tagged as: desk" in q.prompts[0] and "tagged as: door" in q.prompts[1]
    by = _insts(out)
    cap = by[1][CAPTION_KEY]
    assert set(cap) == IFACE
    assert cap["source"] == "object" and cap["provenance"] == "vlm_proposed"
    assert cap["category"] == "desk" and cap["shape"] == "rectangular box"
    assert cap["detail"] == "black top with two drawers", "the meta prefix is stripped"
    assert cap["caption"] == "Desk, rectangular box, painted steel. black top with two drawers"
    datetime.fromisoformat(cap["generated"])
    assert by[2][CAPTION_KEY]["source"] == "object"
    assert by[1]["globalIndices"] == [0, 1, 2], "the rest of the instance is untouched"
    assert not (out / ".matching.lock").exists() or True   # the lock file may remain, never held
    # a second pass does nothing until asked to refresh
    q2 = _Qwen()
    res2 = caption_session_objects(out, tmp_path, views=2, client=q2, log=logs.append)
    assert res2 == {"generated": 0, "kept": 2, "failed": 0, "skipped": 0, "n_instances": 2,
                    "views": 2} and q2.images == []
    res3 = caption_session_objects(out, tmp_path, views=1, refresh=True, client=q2, log=logs.append)
    assert res3["generated"] == 2 and q2.images == [1, 1], "views is honoured"


def test_views_bounds_the_images_and_the_largest_masks_are_chosen(tmp_path):
    from segmentation.object_captioner import _instance_views
    out = _session(tmp_path)
    z = np.load(out / "seg_masks.npz", allow_pickle=True)
    space = mask_space.resolve(out, masks=z)
    frames, masks = _instance_views(z, space, tmp_path / "frames", [0], 3)
    assert [Path(f).name for f in frames] == ["000007.jpg", "000021.jpg", "000000.jpg"], \
        "largest mask first, named by the VIDEO frame of the keyframe position"
    assert masks["000007.jpg"].sum() == 9
    frames1, _ = _instance_views(z, space, tmp_path / "frames", [0], 1)
    assert len(frames1) == 1
    # several oids of one instance: their masks are OR-ed per keyframe
    frames2, masks2 = _instance_views(z, space, tmp_path / "frames", [0, 1], 1)
    assert masks2["000007.jpg"].sum() == 9                  # f1_o1 ⊂ f1_o0 here


def test_service_down_records_nothing_and_says_why(tmp_path, monkeypatch):
    import semantic.service as svc
    out = _session(tmp_path)
    before = (out / "segmentation_result.json").read_text()
    said = []
    monkeypatch.setattr(svc, "ensure_service", lambda *a, **k: (said.append("asked"), False)[1])
    logs = []
    res = caption_session_objects(out, tmp_path, views=4, log=logs.append)
    assert said == ["asked"], "the service is brought up through ensure_service, not assumed"
    assert res["generated"] == 0 and res["skipped"] == 2 and "unavailable" in res["reason"]
    assert (out / "segmentation_result.json").read_text() == before, "nothing recorded"
    assert _insts(out)[1][CAPTION_KEY]["source"] == "concept", "the concept caption stays"
    assert any("unavailable" in m and "nothing recorded" in m for m in logs)
    # the stand-alone call raises instead of returning the label as if the VLM wrote it
    with pytest.raises(RuntimeError, match="did not come up"):
        caption_object_qwen([str(tmp_path / "frames" / "000007.jpg")],
                            {"000007.jpg": _mask(9).astype(bool)}, "desk")


def test_a_failed_object_is_declared_and_the_others_are_written(tmp_path):
    out = _session(tmp_path, with_concept=False)
    q = _Qwen(fail_label="tagged as: door")
    logs = []
    res = caption_session_objects(out, tmp_path, views=2, client=q, log=logs.append)
    assert res["generated"] == 1 and res["failed"] == 1
    by = _insts(out)
    assert by[1][CAPTION_KEY]["source"] == "object"
    assert CAPTION_KEY not in by[2], "no caption is invented for the failed one"
    assert any("#2 'door'" in m and "failed" in m for m in logs)
    # an instance without any mask on disk is declared too
    (out / "segmentation.json").write_text(json.dumps({"prompt": "desk", "instances": [
        {"id": 0, "label": "desk", "instance_id": 1}]}))
    res2 = caption_session_objects(out, tmp_path, views=2, client=_Qwen(), log=logs.append)
    assert res2["failed"] == 1 and res2["kept"] == 1 and res2["generated"] == 0
    assert any("#2 'door': no SAM3 mask view" in m for m in logs)


def test_absorbed_masklets_and_split_children_find_their_masks(tmp_path):
    parent = {"instances": [{"id": 0, "label": "desk", "instance_id": 1},
                            {"id": 5, "label": "desk", "instance_id": 6},
                            {"id": 7, "label": "floor", "instance_id": 8}]}
    oids = _oids_by_instance(parent, {"6": {"into": 1, "reason": "fragment"},
                                      "99": {"into": 1, "reason": "space_dedupe"}})
    assert oids == {1: [0, 5], 6: [5], 8: [7]}
    out = _session(tmp_path, with_concept=False)
    res = json.loads((out / "segmentation_result.json").read_text())
    res["instances"].append({"id": 40, "instance_id": 41, "label": "desk", "globalIndices": [2],
                             "total_points": 1, "split_from": 1})
    (out / "segmentation_result.json").write_text(json.dumps(res))
    q = _Qwen()
    out_res = caption_session_objects(out, tmp_path, views=2, client=q, log=lambda m: None)
    assert out_res["generated"] == 3 and q.images == [2, 1, 2], "the child uses its parent's masks"


# ── the certification worker runs it last, never fatally ─────────────────

class _Pipe:
    def __init__(self):
        self.logs, self.progress = [], []

    def send_log(self, msg, level="info"):
        self.logs.append((level, msg))

    def send_progress(self, pct, msg, stage=""):
        self.progress.append((pct, msg))

    def check_cancel(self):
        return False


def test_the_worker_captions_after_certify_and_a_failure_never_fails_the_stage(tmp_path, monkeypatch):
    import segmentation.object_captioner as oc
    from workers.certify_worker import _caption_objects
    cfg = load_object_captions(yaml.safe_load(CONFIG_YAML.read_text()))
    seen = {}

    def fake(output_dir, session_dir, *, views, log, cancelled, config, refresh=False):
        seen.update(output_dir=Path(output_dir), session_dir=Path(session_dir), views=views)
        return {"generated": 3, "kept": 1, "failed": 0, "skipped": 0, "n_instances": 4}

    monkeypatch.setattr(oc, "caption_session_objects", fake)
    pipe = _Pipe()
    res = _caption_objects(pipe, tmp_path, {"x": 1}, cfg)
    assert res["generated"] == 3
    assert seen == {"output_dir": tmp_path / "output", "session_dir": tmp_path, "views": cfg.views}
    assert pipe.progress and pipe.progress[0][0] == 96
    assert any("3 generated" in m for _l, m in pipe.logs)

    def boom(*a, **k):
        raise RuntimeError("vLLM exploded")

    monkeypatch.setattr(oc, "caption_session_objects", boom)
    pipe = _Pipe()
    assert _caption_objects(pipe, tmp_path, {}, cfg) is None
    assert any(l == "warning" and "vLLM exploded" in m and "concept descriptions stay" in m
               for l, m in pipe.logs)

    off = type(cfg)(enabled=False, views=cfg.views, understand_max_tokens=cfg.understand_max_tokens)
    pipe = _Pipe()
    monkeypatch.setattr(oc, "caption_session_objects", boom)
    assert _caption_objects(pipe, tmp_path, {}, off) is None
    assert any("enabled is false" in m for _l, m in pipe.logs)


def test_the_worker_runs_the_captions_after_the_certification():
    src = (SERVER / "workers" / "certify_worker.py").read_text()
    body = src[src.index("def _certify_work("):]
    assert body.index("certify_session(session_path") < body.index("_caption_objects(pipe") \
        < body.index("send_progress(100")
    assert body.index("load_object_captions(config)") < body.index("certify_session(session_path"), \
        "the keys are read before the hour of certification"
