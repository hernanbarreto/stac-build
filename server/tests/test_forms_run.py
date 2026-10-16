"""FORMS P6 (USER 2026-10-16): the per-session runner — query, fit, correct, review, mesh,
publish — on a synthetic session with injected points, views and a faked VLM. CPU."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms import run as R  # noqa: E402
from reconstruction.forms.catalog import load_forms_config  # noqa: E402
from segmentation import mask_space  # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
RAW = yaml.safe_load((SERVER / "config.yaml").read_text())


def _fcfg(**over):
    raw = {**RAW, "forms": {**RAW["forms"], "device": "cpu", "max_points": 20000, "surface_samples": 4096,
                            "block_iterations": 15, "max_blocks": 12, "boundary_px_per_view": 256, **over}}
    return load_forms_config(raw)


class _Qwen:
    """Answers the form query by label; the review with a fixed verdict."""

    def __init__(self, forms, review=None):
        self.forms, self.review, self.calls = forms, review, []
        self.backend = SimpleNamespace(model="fake", name="fake")

    def chat(self, messages, temperature=None, max_tokens=None, consumer=None, extra_body=None):
        text = messages[-1].text
        self.calls.append(consumer)
        if consumer == "forms.review":
            return SimpleNamespace(content=json.dumps(self.review), finish_reason="stop")
        label = text.split("The element is tagged as: ")[1].split(".")[0]
        return SimpleNamespace(content=json.dumps(self.forms[label]), finish_reason="stop")


_CYL = {"form": "cylinder", "orientation": "vertical", "axis_aligned": True, "hollow": False,
        "outline": "unknown", "profile": "unknown", "parts": [], "proportions": "tall", "alternatives": ["box"],
        "confidence": "high", "what_would_help": "", "notes": ""}
_PLANT = {**_CYL, "form": "freeform", "alternatives": []}


def _session(tmp_path):
    from test_forms_silhouette import _scene
    truth, cloud, views = _scene()
    frames = tmp_path / "frames"
    frames.mkdir()
    for v in views:
        Image.new("RGB", (960, 720), (50, 50, 50)).save(frames / f"{v.cloud_frame:06d}.jpg")
    out = tmp_path / "output"
    out.mkdir()
    (out / "camera_frames.txt").write_text("\n".join(str(v.cloud_frame) for v in views) + "\n")
    masks = {f"f{i}_o0": np.ones((9, 16), bool) for i in range(len(views))}
    masks["f0_o1"] = np.ones((9, 16), bool)
    masks.update({"obj_ids": np.array([0, 1], np.int32), "frames": np.arange(len(views), dtype=np.int32),
                  "scaled_res": np.array([9, 16], np.int32),
                  mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME)})
    np.savez_compressed(out / "seg_masks.npz", **masks)
    (out / "segmentation.json").write_text(json.dumps({
        "instances": [{"id": 0, "label": "column", "instance_id": 1}, {"id": 1, "label": "plant", "instance_id": 2}]}))
    (out / "segmentation_result.json").write_text(json.dumps({
        "instances": [{"id": 0, "instance_id": 1, "label": "column", "globalIndices": list(range(100)), "total_points": 100},
                      {"id": 1, "instance_id": 2, "label": "plant", "globalIndices": [100, 101], "total_points": 2}],
        "total_points": 102}))
    rng = np.random.default_rng(3)
    plant = rng.normal(0, 0.2, (3000, 3)) + [2.0, 0.5, 0.0]
    points = {1: cloud, 2: plant}
    return out, views, points


def test_the_session_runs_end_to_end_with_forms_and_as_measured(tmp_path, monkeypatch):
    out, views, points = _session(tmp_path)
    fcfg = _fcfg()
    vlm = _Qwen({"column": _CYL, "plant": _PLANT})
    monkeypatch.setattr(R, "mesh_as_measured", lambda *a, **k: (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                                                                  np.array([[0, 1, 2]])))
    summary = R.run_session(tmp_path, instance_ids=None, fcfg=fcfg, raw_cfg=RAW, log=lambda m: None, client=vlm,
                            views_fn=lambda iid, oids: views,
                            points_fn=lambda inst: points[int(inst["instance_id"])], sigma=0.005, texture=False)
    assert [o["instance_id"] for o in summary["ok"]] == [1, 2]      # the largest first
    col, plant = summary["ok"]
    assert col["method"] == "hull" and col["form"] == "cylinder" and col["verdict"] == "coincides"
    assert plant["method"] == "hull" and plant["status"] == "as_measured"   # views given: its envelope
    assert summary["failed"] == [] and summary["skipped"] == []
    # one query per object, no review (the column coincided)
    assert vlm.calls == ["forms.query", "forms.query"]
    # the published copies the viewer lists, with their meta
    meta = json.loads((out / "tsdf" / "column_1" / "column_1.meta.json").read_text())
    assert meta["method"] == "hull" and meta["kind"] == "cylinder" and meta["dims_m"]["radius_m"] == pytest.approx(0.10, abs=0.004)
    assert meta["hull"]["n_faces"] > 500 and "visual hull" in meta["provenance"]["mesh"]
    assert meta["provenance"]["form"] == "vlm_proposed" and meta["provenance"]["pose_and_dims"] == "tool_measured"
    assert (out / "tsdf" / "column_1" / "column_1.glb").exists() and (out / "forms" / "column_1" / "form.glb").exists()
    fit = json.loads((out / "forms" / "column_1" / "form_fit.json").read_text())
    assert fit["chosen"] == 0 and fit["candidates"][0]["corrector"]["n_views"] == 6 and "_tpl" not in fit
    pm = json.loads((out / "tsdf" / "plant_2" / "plant_2.meta.json").read_text())
    assert pm["method"] == "hull" and "freeform" in pm["why"]
    run = json.loads((out / "forms" / "forms_run.json").read_text())
    assert run["n_instances"] == 2 and run["sigma_m"] == 0.005 and "frame" in run


def test_a_doubt_opens_the_review_and_the_views_choose(tmp_path, monkeypatch):
    out, views, points = _session(tmp_path)
    fcfg = _fcfg(max_rounds=3)
    # the VLM first says "box" for the column; the review says "wrong_form → cylinder"
    box = {**_CYL, "form": "box", "alternatives": [], "confidence": "medium"}
    review = {"matches": False, "problem": "wrong_form", "form": "cylinder", "orientation": "vertical",
              "axis_aligned": True, "hollow": False, "outline": "unknown", "profile": "unknown", "parts": [],
              "confidence": "high", "notes": ""}
    vlm = _Qwen({"column": box, "plant": _PLANT}, review=review)
    real_correct = R.correct
    calls = {"n": 0}

    def forced(*a, **k):
        rec = real_correct(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:
            rec["verdict"] = "doubt"           # the box: in doubt by decree, so the review fires
        return rec

    monkeypatch.setattr(R, "correct", forced)
    monkeypatch.setattr(R, "mesh_as_measured", lambda *a, **k: (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
                                                                  np.array([[0, 1, 2]])))
    summary = R.run_session(tmp_path, instance_ids=[1], fcfg=fcfg, raw_cfg=RAW, log=lambda m: None, client=vlm,
                            views_fn=lambda iid, oids: views,
                            points_fn=lambda inst: points[int(inst["instance_id"])], sigma=0.005, texture=False)
    col = summary["ok"][0]
    assert col["form"] == "cylinder" and col["method"] == "hull"
    assert vlm.calls == ["forms.query", "forms.review"]
    fit = json.loads((out / "forms" / "column_1" / "form_fit.json").read_text())
    assert fit["rounds"][0]["kind"] == "review" and fit["rounds"][0]["applied"] is True
    assert [c["answer"]["form"] for c in fit["candidates"]] == ["box", "cylinder"] and fit["chosen"] == 1
    assert fit["rounds"][0]["verdict"]["problem"] == "wrong_form"
