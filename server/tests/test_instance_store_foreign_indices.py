"""The instance store never refits from another cloud's indices (pccr 2026-09-30).

`segmentation_result.json` stores each instance as `globalIndices` into the cloud the
masks were projected on (`total_points` = that cloud's size). Selecting an epoch that is
a NEW cloud swapped `cleaned_cloud.ply` under those indices, and `update_instance_store`
refit every OBB from whatever points the old indices now named: walk-sized cubes.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth_correction import write_cloud  # noqa: E402


def _session(tmp: Path, n_cloud: int, n_seg: int):
    rng = np.random.default_rng(0)
    out = tmp / "output"
    out.mkdir()
    xyz = rng.uniform(-10, 10, (n_cloud, 3))                 # a scene 20 m across
    xyz[:50] = rng.uniform(0, 0.5, (50, 3))                  # the object: a 0.5 m box
    write_cloud(out / "cleaned_cloud.ply", xyz, np.zeros(n_cloud, np.int32), rng)
    (out / "camera_frames.txt").write_text("0\n")
    (out / "camera_poses.txt").write_text(" ".join(f"{v:.1f}" for v in np.eye(4).ravel()) + "\n")
    (out / "segmentation_result.json").write_text(json.dumps({
        "type": "segmentation", "version": "3.0", "total_points": n_seg,
        "instances": [{"instance_id": 1, "label": "desk", "globalIndices": list(range(50))}]}))
    from phase_r.instance_store import InstanceStore
    st = InstanceStore(out / "scene_r.db")
    st.upsert_instance(1, "desk")
    st.set_points(1, np.zeros((3, 3), np.float32))
    T = np.eye(4); T[:3, 3] = 0.25
    st.set_obb(1, T, np.array([-0.25, 0.25, -0.25, 0.25, -0.25, 0.25]), T[:3, 3], n_points=3,
               obb_origin="tool_measured")
    st.close()
    return out


def _update(out):
    from correction.invalidate import update_instance_store
    return update_instance_store(out, np.eye(3)[None], np.zeros((1, 3)), np.ones(1), [0],
                                 log=lambda *a, **k: None)


def _obb_T(out):
    from phase_r.instance_store import InstanceStore
    st = InstanceStore(out / "scene_r.db")
    try:
        o = st.get_obb(1)
        meta = st.get_meta("segmentation_indexes_live_cloud")
    finally:
        st.close()
    return o, meta


def test_a_segmentation_of_another_cloud_leaves_the_store_as_it_was(tmp_path):
    out = _session(tmp_path, n_cloud=5000, n_seg=4000)       # projected on a 4000-pt cloud
    before, _ = _obb_T(out)
    rep = _update(out)
    after, meta = _obb_T(out)
    assert rep["store"] == "stale_segmentation" and rep["instances"] == 0
    assert meta == "false"
    assert json.dumps(before, default=str) == json.dumps(after, default=str), \
        "an OBB was refit from indices of a cloud that is not the live one"


def test_a_segmentation_of_this_cloud_is_refit(tmp_path):
    out = _session(tmp_path, n_cloud=5000, n_seg=5000)
    before, _ = _obb_T(out)
    rep = _update(out)
    after, meta = _obb_T(out)
    assert rep["store"] == "updated" and rep["instances"] == 1 and meta == "true"
    assert json.dumps(before, default=str) != json.dumps(after, default=str)
