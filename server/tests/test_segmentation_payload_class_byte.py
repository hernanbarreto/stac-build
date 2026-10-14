"""The segmentation the viewer receives carries each instance's class byte (pccr 2026-10-01).

With instance ids above 255 the octree carries COMPACT bytes (class_map.json); the viewer toggles by
`inst.class_byte ?? instance_id`, and the WebSocket payload had no class_byte — every toggle switched
another object."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.pipeline import apply_segmentation_to_cloud  # noqa: E402
from segmentation.republish import write_classification  # noqa: E402


def test_payload_carries_the_compact_byte(tmp_path):
    insts = [{"id": 299, "instance_id": 300, "label": "desk", "globalIndices": [0, 1]},
             {"id": 49, "instance_id": 50, "label": "door", "globalIndices": [2]}]
    write_classification(tmp_path, insts, 3)                      # ids > 255 -> compact + class_map.json
    (tmp_path / "segmentation_result.json").write_text(json.dumps({"instances": insts, "total_points": 3}))
    out = apply_segmentation_to_cloud(tmp_path)
    by = {i["instance_id"]: i["class_byte"] for i in out["instances"]}
    cmap = json.loads((tmp_path / "class_map.json").read_text())["class_of"]
    assert by == {300: cmap["300"], 50: cmap["50"]}
    assert by[50] != 50, "the toggle of door #50 must not be keyed by its id under the compact encoding"


def test_without_a_map_the_byte_is_the_id(tmp_path):
    insts = [{"id": 4, "instance_id": 5, "label": "x", "globalIndices": [0]}]
    (tmp_path / "segmentation_result.json").write_text(json.dumps({"instances": insts, "total_points": 1}))
    assert apply_segmentation_to_cloud(tmp_path)["instances"][0]["class_byte"] == 5


def test_an_octree_with_the_instance_dimension_is_keyed_by_the_id(tmp_path):
    """USER 2026-10-08 (no 255 ceiling): an octree that carries the per-point `instance` dimension
    is read by the viewer through it, so the key the payload sends is the instance id itself —
    never the compact byte of class_map.json, which that octree no longer drives."""
    insts = [{"id": 299, "instance_id": 300, "label": "desk", "globalIndices": [0, 1]},
             {"id": 49, "instance_id": 50, "label": "door", "globalIndices": [2]}]
    write_classification(tmp_path, insts, 3)                      # ids > 255 -> compact + class_map.json
    (tmp_path / "segmentation_result.json").write_text(json.dumps({"instances": insts, "total_points": 3}))
    (tmp_path / "potree").mkdir()
    (tmp_path / "potree" / "metadata.json").write_text(json.dumps({"attributes": [
        {"name": "position", "size": 12}, {"name": "classification", "size": 1},
        {"name": "instance", "size": 4, "type": "uint32"}]}))
    out = apply_segmentation_to_cloud(tmp_path)
    assert {i["instance_id"]: i["class_byte"] for i in out["instances"]} == {300: 300, 50: 50}
