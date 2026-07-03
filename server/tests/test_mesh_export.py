"""
MeshFlow input-exporter routing tests: architectural classes must NEVER reach
the generative path (they belong to surface_fit), oversized segments stay on
TSDF, and eligible objects come out as one PLY + non-metric meta each.
"""
import json

import numpy as np
import pytest

from segmentation.mesh_export import export_segment_plys


@pytest.fixture()
def session_output(tmp_path):
    rng = np.random.default_rng(5)
    wall = np.column_stack([rng.normal(0, 0.002, 30_000),
                            rng.uniform(0, 4, 30_000), rng.uniform(0, 3, 30_000)])
    chair = rng.normal(0, 0.25, (10_000, 3)) + np.array([2, 1, 0.5])
    train = np.column_stack([rng.uniform(0, 20, 20_000),
                             rng.uniform(0, 3, 20_000), rng.uniform(0, 3, 20_000)])
    cloud = np.vstack([wall, chair, train])
    out = tmp_path / "output"
    out.mkdir()
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), pcd)
    seg = {"instances": [
        {"id": 1, "instance_id": 1, "label": "wall",
         "globalIndices": list(range(30_000))},
        {"id": 2, "instance_id": 2, "label": "chair",
         "globalIndices": list(range(30_000, 40_000))},
        {"id": 3, "instance_id": 3, "label": "train",
         "globalIndices": list(range(40_000, 60_000))},
    ]}
    (out / "segmentation_result.json").write_text(json.dumps(seg))
    return out, seg


def test_routing(session_output):
    out, seg = session_output
    exported, skipped = export_segment_plys(out, seg, max_extent_m=6.0)
    # only the chair is a MeshFlow-eligible object
    assert len(exported) == 1
    assert exported[0].parent.name == "chair_2"
    reasons = {s["label"]: s["reason"] for s in skipped}
    assert "surface_fit" in reasons["wall"]          # architectural → metric path
    assert "TSDF" in reasons["train"]                # 20 m > 6 m budget

    meta = json.loads((exported[0].parent / "meta.json").read_text())
    assert meta["metric"] is False                   # ⚠ non-metric labeling
    assert meta["generative"] is True
    assert meta["method"] == "meshflow"


def test_explicit_selection_still_routed(session_output):
    """Even if the UI explicitly selects an architectural instance, the
    exporter refuses it — routing is a policy, not a default."""
    out, seg = session_output
    exported, skipped = export_segment_plys(out, seg, obj_ids=[1])
    assert not exported
    assert skipped and skipped[0]["instance_id"] == 1
