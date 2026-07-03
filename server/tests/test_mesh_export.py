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
    # default: NO category routing (user decision) — wall generates too;
    # only the oversized train is skipped (model vertex budget)
    exported, skipped = export_segment_plys(out, seg, max_extent_m=6.0,
                                            require_ref_image=False)
    assert {p.parent.name for p in exported} == {"chair_2", "wall_1"}
    reasons = {s["label"]: s["reason"] for s in skipped}
    assert "TSDF" in reasons["train"]                # 20 m > 6 m budget

    # opt-in category routing still works
    exported2, skipped2 = export_segment_plys(out, seg, max_extent_m=6.0,
                                              require_ref_image=False,
                                              exclude_architectural=True)
    assert {p.parent.name for p in exported2} == {"chair_2"}
    reasons2 = {s["label"]: s["reason"] for s in skipped2}
    assert "surface_fit" in reasons2["wall"]

    meta = json.loads((exported[0].parent / "meta.json").read_text())
    assert meta["metric"] is False                   # ⚠ non-metric labeling
    assert meta["generative"] is True
    assert meta["method"] == "meshflow"


def test_explicit_selection_respects_optin_routing(session_output):
    """With exclude_architectural=True, an explicitly selected architectural
    instance is still refused (routing is a policy when enabled)."""
    out, seg = session_output
    exported, skipped = export_segment_plys(out, seg, obj_ids=[1],
                                            require_ref_image=False,
                                            exclude_architectural=True)
    assert not exported
    assert skipped and skipped[0]["instance_id"] == 1


def test_mandatory_ref_image(session_output, tmp_path):
    """Con require_ref_image=True: sin máscaras/frames se SALTEA; con una
    máscara y un frame válidos se genera el recorte de la mejor vista."""
    import cv2
    out, seg = session_output
    # sin seg_masks.npz → skip con razón explícita
    exported, skipped = export_segment_plys(out, seg, obj_ids=[2],
                                            frames_dir=None,
                                            require_ref_image=True)
    assert not exported
    assert "mandatory" in skipped[0]["reason"]

    # con máscara + frame: se exporta y el ref.jpg existe
    frames = tmp_path / "frames"
    frames.mkdir()
    img = np.zeros((360, 640, 3), np.uint8)
    img[:, :, 1] = 128
    cv2.imwrite(str(frames / "000042.jpg"), img)
    mask = np.zeros((180, 320), np.uint8)
    mask[60:120, 100:220] = 1
    np.savez(out / "seg_masks.npz", **{"f42_o2": mask})
    exported, skipped = export_segment_plys(out, seg, obj_ids=[2],
                                            frames_dir=frames,
                                            require_ref_image=True)
    assert len(exported) == 1
    ref = exported[0].parent / f"{exported[0].parent.name}_ref.jpg"
    assert ref.exists()
    crop = cv2.imread(str(ref))
    assert crop is not None and crop.shape[0] > 50 and crop.shape[1] > 100
