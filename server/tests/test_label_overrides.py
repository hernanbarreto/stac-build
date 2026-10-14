"""The user's segment names (USER 2026-10-09): kept beside the SAM3 store, never in it; applied on
top of every projection (after the fusion), carried to the chat's store with the user's
provenance, found again by geometry when a new segmentation renumbers the objects, and followed
by the object's mesh files. CPU only, synthetic session (the projection fixture of
test_p3_projection)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import label_overrides as LO                       # noqa: E402
from segmentation import pipeline as P                                # noqa: E402

from test_erase_store_reseal import _seal                             # noqa: E402
from test_p3_projection import _session, fake_potree                  # noqa: E402,F401


def _result(out):
    return json.loads((out / "segmentation_result.json").read_text())


def _store_rows(out):
    from phase_r.instance_store import InstanceStore
    st = InstanceStore(out / "scene_r.db")
    try:
        return {int(r["instance_id"]): r for r in st.list_instances()}
    finally:
        st.conn.close()


def test_a_name_survives_a_reprojection_and_reaches_the_chat_store(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    inst = _result(out)["instances"][0]
    iid, old = int(inst["instance_id"]), inst["label"]
    seg_before = (out / "segmentation.json").read_bytes()

    LO.record(out, iid, "my desk", sam3_label=old, anchors=None)
    assert (out / "segmentation.json").read_bytes() == seg_before, "the SAM3 store is never touched"
    P._match_and_save_result(out)                          # a re-projection
    again = {int(i["instance_id"]): i for i in _result(out)["instances"]}[iid]
    assert again["label"] == "my desk" and again["label_sam3"] == old
    assert again["label_origin"] == LO.HUMAN
    row = _store_rows(out)[iid]
    assert row["label"] == "my desk" and row["label_origin"] == LO.HUMAN, "the chat sees it"
    assert P.check_store_seal(out) == [], "the store's seal still holds"


def test_apply_to_session_rewrites_the_result_and_the_store(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    inst = _result(out)["instances"][-1]
    iid = int(inst["instance_id"])
    LO.record(out, iid, "renamed", sam3_label=inst["label"], anchors=None)
    got = LO.apply_to_session(out, log=lambda m: None)
    assert got == {iid: "renamed"}
    assert {int(i["instance_id"]): i["label"] for i in _result(out)["instances"]}[iid] == "renamed"
    assert _store_rows(out)[iid]["label"] == "renamed"
    # applied twice: nothing changes, the result is not rewritten
    mt = (out / "segmentation_result.json").stat().st_mtime_ns
    LO.apply_to_session(out, log=lambda m: None)
    assert (out / "segmentation_result.json").stat().st_mtime_ns == mt


def test_giving_back_the_segmentation_name_removes_the_override(tmp_path):
    out = tmp_path
    LO.record(out, 5, "chair A", sam3_label="chair", anchors=None)
    assert LO.load(out)["labels"]["5"]["label"] == "chair A"
    LO.record(out, 5, "chair", sam3_label="chair", anchors=None)
    assert not (out / LO.FILE).exists()


def test_a_renumbered_object_is_found_again_by_its_geometry(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.uniform([0, 0, 0], [1, 1, 1], (500, 3))
    b = rng.uniform([3, 0, 0], [4, 1, 1], (500, 3))
    xyz = np.vstack([a, b])

    def obb(p):
        return {"center": p.mean(0).tolist(), "half_extents": ((p.max(0) - p.min(0)) / 2).tolist(),
                "rotation": np.eye(3).tolist()}
    LO.record(tmp_path, 1, "red desk", sam3_label="desk", anchors=LO.anchors_from(b))
    # a NEW segmentation: the desk is now #7 and #1 is another object
    insts = [{"instance_id": 1, "label": "chair", "globalIndices": list(range(500)), "obb": obb(a)},
             {"instance_id": 7, "label": "desk", "globalIndices": list(range(500, 1000)), "obb": obb(b)}]
    got = LO.apply(tmp_path, insts, points_of=lambda i: xyz[np.asarray(i["globalIndices"])],
                   log=lambda m: None)
    assert got == {7: "red desk"} and insts[0]["label"] == "chair"
    assert set(LO.load(tmp_path)["labels"]) == {"7"}, "the override follows the object"


def test_an_override_nothing_matches_is_kept_not_applied(tmp_path):
    LO.record(tmp_path, 3, "pump", sam3_label="machine", anchors=[[10.0, 10.0, 10.0]])
    insts = [{"instance_id": 3, "label": "wall", "globalIndices": [0], "obb": None}]
    got = LO.apply(tmp_path, insts, points_of=lambda i: np.zeros((1, 3)), log=lambda m: None)
    assert got == {} and insts[0]["label"] == "wall"
    assert "3" in LO.load(tmp_path)["labels"]


def test_the_object_files_follow_the_name(tmp_path):
    d = tmp_path / "tsdf" / "wall_12"
    d.mkdir(parents=True)
    (d / "wall_12.glb").write_bytes(b"glb")
    (d / "meta.json").write_text(json.dumps({"label": "wall", "x": 1}))
    p = tmp_path / "tsdf" / "wall_12_poisson"
    p.mkdir()
    (p / "wall_12_poisson.glb").write_bytes(b"glb")
    other = tmp_path / "tsdf" / "wall_123"
    other.mkdir()
    moved = LO.rename_object_files(tmp_path, "wall", "north wall", 12, log=lambda m: None)
    assert len(moved) == 2
    nd = tmp_path / "tsdf" / "north_wall_12"
    assert (nd / "north_wall_12.glb").exists()
    assert json.loads((nd / "meta.json").read_text())["label"] == "north wall"
    assert (tmp_path / "tsdf" / "north_wall_12_poisson" / "north_wall_12_poisson.glb").exists()
    assert other.exists(), "wall_123 is another object"


def _two_objects():
    rng = np.random.default_rng(1)
    a = rng.uniform([0, 0, 0], [1, 1, 1], (400, 3))
    b = rng.uniform([3, 0, 0], [4, 1, 1], (400, 3))
    return a, b, np.vstack([a, b])


def _obb(p):
    return {"center": p.mean(0).tolist(), "half_extents": ((p.max(0) - p.min(0)) / 2).tolist(),
            "rotation": np.eye(3).tolist()}


def test_two_names_whose_ids_swapped_both_follow_their_objects(tmp_path):
    a, b, xyz = _two_objects()
    LO.record(tmp_path, 5, "my desk", sam3_label="desk", anchors=LO.anchors_from(a))
    LO.record(tmp_path, 9, "my chair", sam3_label="chair", anchors=LO.anchors_from(b))
    insts = [{"instance_id": 5, "label": "chair", "globalIndices": list(range(400, 800)), "obb": _obb(b)},
             {"instance_id": 9, "label": "desk", "globalIndices": list(range(400)), "obb": _obb(a)}]
    got = LO.apply(tmp_path, insts, points_of=lambda i: xyz[np.asarray(i["globalIndices"])],
                   log=lambda m: None)
    assert got == {9: "my desk", 5: "my chair"}
    assert {k: v["label"] for k, v in LO.load(tmp_path)["labels"].items()} == \
        {"9": "my desk", "5": "my chair"}, "nothing overwritten, both re-keyed"


def test_the_id_is_verified_among_objects_of_the_same_name(tmp_path):
    a, b, xyz = _two_objects()
    LO.record(tmp_path, 12, "director's chair", sam3_label="chair", anchors=LO.anchors_from(a))
    # a new segmentation numbered the two chairs the other way round
    insts = [{"instance_id": 12, "label": "chair", "globalIndices": list(range(400, 800)), "obb": _obb(b)},
             {"instance_id": 13, "label": "chair", "globalIndices": list(range(400)), "obb": _obb(a)}]
    LO.apply(tmp_path, insts, points_of=lambda i: xyz[np.asarray(i["globalIndices"])],
             log=lambda m: None)
    assert insts[1]["label"] == "director's chair" and insts[0]["label"] == "chair"


def test_renaming_back_gives_the_segmentation_name_back(tmp_path, monkeypatch, fake_potree):
    out, _, _ = _session(tmp_path)
    _seal(out, monkeypatch)
    P._match_and_save_result(out)
    inst = _result(out)["instances"][0]
    iid, old = int(inst["instance_id"]), inst["label"]
    LO.record(out, iid, "renamed", sam3_label=old, anchors=None)
    LO.apply_to_session(out, log=lambda m: None)
    assert LO.record(out, iid, old, sam3_label=old, anchors=None) is None
    LO.apply_to_session(out, log=lambda m: None)
    back = {int(i["instance_id"]): i for i in _result(out)["instances"]}[iid]
    assert back["label"] == old and back.get("label_origin") != LO.HUMAN and "label_sam3" not in back
    assert _store_rows(out)[iid]["label"] == old and _store_rows(out)[iid]["label_origin"] == LO.VLM


def test_folder_names_that_only_start_the_same_are_another_object(tmp_path):
    (tmp_path / "tsdf" / "chair_1").mkdir(parents=True)
    (tmp_path / "tsdf" / "chair_1_7").mkdir()           # object #7 named 'chair 1'
    (tmp_path / "scene_objects.json").write_text(json.dumps(
        {"objects": [{"url": "/api/segmentation/tsdf/file/s/chair_1/chair_1.glb"}]}))
    (tmp_path / "tsdf" / "chair_1" / "chair_1.glb").write_bytes(b"x")
    LO.rename_object_files(tmp_path, "chair", "stool", 1, log=lambda m: None)
    assert (tmp_path / "tsdf" / "stool_1" / "stool_1.glb").exists()
    assert (tmp_path / "tsdf" / "chair_1_7").exists(), "#7's folder is not #1's"
    assert "/stool_1/stool_1.glb" in (tmp_path / "scene_objects.json").read_text()
