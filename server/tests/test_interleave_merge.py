"""One owner per surface (USER 2026-11-18: "hay que arreglarlo de raíz"): two objects whose points
are interleaved millimetres apart are the same surface seen in different keyframes — they are one
object; objects that merely touch, or are apart, stay two."""
from __future__ import annotations

import numpy as np
import pytest

from segmentation import interleave as I


def _scene():
    """A wall owned point by point by two objects (1: 70 %, 2: 30 %), a tile lying wholly among
    the floor's points (floor 3, tile 4), a chair apart (5) and a box that only TOUCHES the wall (6)."""
    rng = np.random.default_rng(0)
    wall = np.column_stack([rng.uniform(0, 4, 6000), rng.uniform(0, 2.5, 6000), np.zeros(6000)])
    wall_owner = np.where(rng.uniform(size=6000) < 0.7, 1, 2)
    floor = np.column_stack([rng.uniform(0, 4, 8000), np.zeros(8000), rng.uniform(0.5, 4, 8000)])
    in_tile = (floor[:, 0] < 0.6) & (floor[:, 2] < 1.1)
    floor_owner = np.where(in_tile & (rng.uniform(size=8000) < 0.5), 4, 3)
    chair = rng.uniform(0, 0.4, (1500, 3)) + [8.0, 0.0, 8.0]
    box = np.column_stack([rng.uniform(1, 1.5, 1500), rng.uniform(0, 0.5, 1500), rng.uniform(0.001, 0.4, 1500)])
    xyz = np.vstack([wall, floor, chair, box])
    owner = np.concatenate([wall_owner, floor_owner, np.full(1500, 5), np.full(1500, 6)])
    labels = {1: "wall", 2: "drywall", 3: "floor", 4: "tile", 5: "chair", 6: "box"}
    instances = [{"id": i, "instance_id": i, "label": labels[i], "globalIndices": np.flatnonzero(owner == i).tolist(),
                  "total_points": int((owner == i).sum())} for i in labels]
    return xyz, owner, instances


def test_the_shares_say_who_sits_among_whom():
    xyz, owner, _ = _scene()
    s = I.neighbour_shares(xyz, owner, 8)
    assert s[(2, 1)] == pytest.approx(0.7, abs=0.05) and s[(1, 2)] == pytest.approx(0.3, abs=0.05)   # the same wall
    assert s[(4, 3)] > 0.4 and s[(3, 4)] < 0.05                 # the tile lies among the floor's points
    assert s.get((5, 3), 0.0) == 0.0 and s.get((6, 1), 0.0) < 0.05 and s.get((1, 6), 0.0) < 0.05   # apart / touching


def test_objects_sharing_a_surface_become_one_and_the_rest_stay(capsys):
    xyz, owner, instances = _scene()
    record = {}
    report = I.merge_instances(instances, xyz, len(xyz), share=0.10, neighbours=8, record=record)
    assert sorted(i["id"] for i in instances) == [1, 3, 5, 6]            # the larger member survives, with its label
    wall = next(i for i in instances if i["id"] == 1)
    assert wall["label"] == "wall" and wall["total_points"] == 6000 == len(wall["globalIndices"])
    assert next(i for i in instances if i["id"] == 3)["total_points"] == 8000
    assert record[2]["into"] == 1 and record[2]["reason"] == "same_surface" and record[2]["label"] == "drywall"
    assert record[4]["into"] == 3 and record[4]["interleaved_share"] > 0.4
    assert report["objects_before"] == 6 and report["objects_after"] == 4 and report["absorbed"] == 2
    assert report["neighbours_of_another_object_before"] > 0.1 > report["neighbours_of_another_object_after"]
    # every point has one owner still, and none was lost
    assert sum(len(i["globalIndices"]) for i in instances) == len(xyz)
    # the same input, the same outcome
    xyz2, _o, again = _scene()
    I.merge_instances(again, xyz2, len(xyz2), share=0.10, neighbours=8, log=lambda _m: None)
    assert [(i["id"], i["total_points"]) for i in again] == [(i["id"], i["total_points"]) for i in instances]


def test_a_high_share_merges_nothing_and_the_config_is_typed():
    xyz, _owner, instances = _scene()
    report = I.merge_instances(instances, xyz, len(xyz), share=0.95, neighbours=8, log=lambda _m: None)
    assert report["absorbed"] == 0 and len(instances) == 6
    assert I.config()["share"] == 0.10 and I.config()["neighbours"] == 8 and isinstance(I.config()["enabled"], bool)
    with pytest.raises(KeyError, match="interleave_merge"):
        I.config({"segmentation": {}})
    with pytest.raises(ValueError, match="share"):
        I.config({"segmentation": {"interleave_merge": {"enabled": True, "share": 0, "neighbours": 8}}})


def test_the_fusion_and_the_mask_fates_know_the_reason():
    from segmentation import fuse_parent, mask_fates
    assert "same_surface" in fuse_parent.FUSING and "same_surface" in mask_fates.REASONS
