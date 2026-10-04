"""The second VLM pass over the unsegmented points (segmentation/second_pass.py, 2026-10-04): synthetic
session on disk, the VLM, SAM3 and the projection mocked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import second_pass as SP  # noqa: E402

CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"
H, W = 32, 48


def _write_ply(path: Path, xyz, fg, pr, pc):
    n = len(xyz)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\nproperty int frame_global\n"
              "property short pixel_row\nproperty short pixel_col\nend_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")])
    arr = np.zeros(n, dt)
    arr["x"], arr["y"], arr["z"] = xyz.T; arr["frame_global"] = fg; arr["pixel_row"] = pr; arr["pixel_col"] = pc
    with open(path, "wb") as fh:
        fh.write(header.encode()); fh.write(arr.tobytes())


def _session(tmp: Path):
    out = tmp / "output"; out.mkdir(parents=True)
    frames = tmp / "frames"; frames.mkdir()
    kfs = [0, 7, 14]
    for f in kfs:
        Image.fromarray(np.full((H, W, 3), 128, np.uint8)).save(frames / f"{f:06d}.jpg")
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in kfs) + "\n")
    # masks: keyframe position 0 holds one masklet over the LEFT half; positions 1, 2 none
    m = np.zeros((H, W), np.uint8); m[:, :W // 2] = 1
    np.savez(out / "seg_masks.npz", f0_o0=m)
    (out / "segmentation.json").write_text(json.dumps({"version": "3.0", "prompts": ["wall"], "instances": [
        {"id": 1, "label": "wall", "frames": [0]}], "mask_file": "seg_masks.npz", "frame_space": "keyframe_position"}))
    # the cloud: every 2nd pixel of each keyframe, z = 2
    rows, cols = np.meshgrid(np.arange(0, H, 2), np.arange(0, W, 2), indexing="ij")
    pr = np.tile(rows.ravel(), 3); pc = np.tile(cols.ravel(), 3)
    fg = np.repeat(np.array(kfs), rows.size)
    xyz = np.stack([pc * 0.01, pr * 0.01, np.full(len(pr), 2.0)], 1).astype(np.float32)
    _write_ply(out / "cleaned_cloud.ply", xyz, fg, pr, pc)
    (out / "vlm_analysis.json").write_text(json.dumps({"prompt": "wall;floor", "scene_understanding": {"merged": {"walls": "wall"}},
                                                        "consolidation": {"synonyms": {}}, "fallback_prompts": {}}))
    return out, kfs


class _Client:
    def __init__(self):
        self.calls = []

    def chat(self, msgs, max_tokens=0, consumer=""):
        self.calls.append(msgs)
        return SimpleNamespace(content=json.dumps({"scene_type": "room", "summary": "", "objects": [
            {"category": "Cabinet", "description": "grey metal"}, {"category": "walls", "description": "white"},
            {"category": "floor"}, {"category": "cable", "description": "black"}]}))


def _config():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    raw["autoprompt"]["second_pass"] = {"enabled": True, "max_calls": 10, "dim": 0.25, "grow_px": 2}
    raw["autoprompt"]["max_sam3_prompts"] = 150
    return raw


def test_unmasked_pixels_come_from_the_cloud_provenance(tmp_path):
    out, kfs = _session(tmp_path)
    maps, acct = SP.unmasked_pixels(out, log=lambda m: None)
    n_per_kf = (H // 2) * (W // 2)
    assert acct["points"] == 3 * n_per_kf
    assert acct["unmasked_points"] == n_per_kf // 2 + 2 * n_per_kf     # half of kf 0, all of kfs 1 and 2
    assert set(maps) == {0, 1, 2} and maps[0][:, :W // 2].sum() == 0 and maps[0][:, W // 2:].sum() == n_per_kf // 2


def test_highlight_dims_the_segmented_pixels_only():
    img = np.full((H, W, 3), 200, np.uint8)
    un = np.zeros((H, W), bool); un[10, 10] = True
    hi = SP.highlight(img, un, 0.25, 2)
    assert hi[10, 10, 0] == 200 and hi[12, 12, 0] == 200 and hi[0, 0, 0] == 50 and hi[20, 20, 0] == 50


def test_new_concepts_skip_what_the_first_pass_covers():
    from collections import Counter
    new, fate = SP.new_concepts(Counter({"cabinet": 3, "walls": 2, "floor": 2, "cable": 1, "cabinets": 1}),
                                ["wall", "floor"], {"walls": "wall"}, {})
    assert new == ["cabinet", "cable"]
    assert fate["walls"].startswith("covered by 'wall'") and fate["floor"].startswith("covered")
    assert fate["cabinets"] == "same name as 'cabinet'"


def test_the_pass_prompts_sam3_with_the_new_concepts_and_reprojects(tmp_path):
    out, kfs = _session(tmp_path)
    client = _Client()
    seen = {}

    def segment(prompt, status, fallbacks):
        seen["prompt"] = prompt; seen["fallbacks"] = fallbacks
        for c in prompt.split(";"):
            status[c] = {"status": "ran"}
        # SAM3 "finds" the cabinet on the right half of keyframe 1 → the store grows
        z = dict(np.load(out / "seg_masks.npz"))
        m = np.zeros((H, W), np.uint8); m[:, W // 2:] = 1
        z["f1_o1"] = m
        np.savez(out / "seg_masks.npz", **z)
        return {"instances": [{"id": 2}]}

    def project(o):
        seen["projected"] = True
        (out / "segmentation_result.json").write_text(json.dumps({"instances": [{"id": 1}, {"id": 2}], "coverage": 0.6}))
        return {"coverage": 0.6, "instances": [1, 2]}

    rep = SP.run_second_pass(tmp_path, _config(), log=lambda m: None, client_factory=lambda: client,
                             segment=segment, project=project, stop_vllm=lambda: seen.setdefault("stopped", True))
    assert len(client.calls) == 3                                   # the three keyframes with unmasked points
    assert any("bright region" in str(m) for m in client.calls[0])  # the extra instruction reached the VLM
    assert rep["new_prompts"] == ["cabinet", "cable"] and seen["prompt"] == "cabinet;cable"
    assert seen["fallbacks"] == {"cabinet": ["grey metal cabinet"], "cable": ["black cable"]}
    assert seen["stopped"] is True and seen["projected"] is True
    assert rep["after"]["unmasked_points"] < rep["before"]["unmasked_points"]
    vlm = json.loads((out / "vlm_analysis.json").read_text())
    assert vlm["prompt"] == "wall;floor;cabinet;cable" and vlm["second_pass"]["calls"] == 3
    assert (out / "second_pass.json").exists()


def test_disabled_pass_is_recorded_and_touches_nothing(tmp_path):
    out, _ = _session(tmp_path)
    raw = _config(); raw["autoprompt"]["second_pass"]["enabled"] = False
    rep = SP.run_second_pass(tmp_path, raw, log=lambda m: None, client_factory=lambda: None)
    assert rep["skipped"] and json.loads((out / "vlm_analysis.json").read_text())["prompt"] == "wall;floor"


def test_production_config_declares_the_pass():
    raw = yaml.safe_load(CONFIG_YAML.read_text())
    sp = raw["autoprompt"]["second_pass"]
    assert set(sp) >= {"enabled", "max_calls", "dim", "grow_px"} and 0 <= sp["dim"] <= 1
