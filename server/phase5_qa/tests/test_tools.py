# STAC-Builder — Phase 5 spatial-tools unit tests (synthetic store geometry).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.instance_store import InstanceStore  # noqa: E402
from phase5_qa.tools import SpatialTools  # noqa: E402

rng = np.random.default_rng(0)


def _box_points(center, ext, n=800):
    return center + rng.uniform(-0.5, 0.5, (n, 3)) * ext


def _aabb(ext):
    return np.array([-ext[0]/2, ext[0]/2, -ext[1]/2, ext[1]/2, -ext[2]/2, ext[2]/2])


def _make_store(path):
    st = InstanceStore(path)
    # column A at x=0, 0.4x3x0.4 ; wall B at x=2, 0.2x3x4
    for iid, label, center, ext in [
        (1, "column", np.array([0, 1.5, 0.0]), np.array([0.4, 3.0, 0.4])),
        (2, "wall", np.array([2.0, 1.5, 0.0]), np.array([0.2, 3.0, 4.0])),
    ]:
        st.upsert_instance(iid, label, known=True, confidence=0.9, n_views=5)
        pts = _box_points(center, ext)
        st.set_points(iid, pts)
        T = np.eye(4); T[:3, 3] = center
        st.set_obb(iid, T, _aabb(ext), center, n_points=len(pts))
    return st


def test_size_and_position():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        sz = t.get_object_size(1)
        assert abs(sz["width_m"] - 0.4) < 0.05 and abs(sz["height_m"] - 3.0) < 0.1
        pos = t.get_position(2)
        assert abs(pos["position_m"][0] - 2.0) < 0.05
        st.close()


def test_distance_and_clearance():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        # centers 2 m apart; column half-width 0.2, wall half-width 0.1 -> ~1.7 m gap
        dist = t.get_distance(1, 2)
        assert 1.4 < dist["distance_m"] < 2.0, dist
        clr = t.get_clearance(1, 2)
        assert 1.2 < clr["clearance_m"] < 2.0 and "point_a_m" in clr
        st.close()


def test_count_and_list():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        assert t.count_objects()["count"] == 2
        assert t.count_objects("column")["count"] == 1
        assert len(t.list_objects()["objects"]) == 2
        st.close()


def test_insufficient_data():
    with tempfile.TemporaryDirectory() as d:
        st = InstanceStore(os.path.join(d, "s.db"))
        st.upsert_instance(5, "pipe")  # no points, no OBB
        t = SpatialTools(st)
        assert t.get_position(5).get("insufficient_data")
        assert t.get_clearance(5, 5).get("insufficient_data")
        assert t.get_findings(5).get("insufficient_data")
        st.close()


def test_volume_and_span():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        v = t.get_object_volume(2)  # 0.2*3*4 = 2.4
        assert abs(v["bbox_volume_m3"] - 2.4) < 0.3
        sp = t.get_span(2)
        assert abs(sp["span_m"] - 4.0) < 0.2
        st.close()


def test_define_and_persist_volume():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        r = t.define_volume("bay A", [0.0, 1.5, 0.0], [1.0, 3.0, 1.0], yaw_deg=0.0)
        vid = r["volume_id"]
        assert vid >= 1 and st.get_user_volume(vid)["name"] == "bay A"
        assert len(st.list_user_volumes()) == 1
        st.delete_user_volume(vid)
        assert st.get_user_volume(vid) is None
        st.close()


def test_objects_in_volume():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        # box around the column at x=0 -> contains column (id 1), not wall (id 2)
        r = t.objects_in_volume(center=[0.0, 1.5, 0.0], size=[1.0, 3.2, 1.0])
        ids = [o["id"] for o in r["objects"]]
        assert 1 in ids and 2 not in ids
        assert r["objects"][0]["fraction_inside"] > 0.9
        # box far away -> empty
        far = t.objects_in_volume(center=[50.0, 1.5, 0.0], size=[1.0, 1.0, 1.0])
        assert far["count"] == 0


def test_evaluate_volume_free_fraction():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        r = t.evaluate_volume(center=[0.0, 1.5, 0.0], size=[1.0, 3.0, 1.0], voxel_m=0.1)
        assert abs(r["box_volume_m3"] - 3.0) < 1e-6
        assert 0.0 < r["occupied_fraction"] < 1.0  # column occupies part of the bay
        assert r["free_volume_m3"] > 0.0
        assert any(o["id"] == 1 for o in r["objects_inside"])


def test_fits_in_volume():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        # a small item fits in the free border around the column
        small = t.fits_in_volume(item_size=[0.25, 0.25, 0.25],
                                 center=[0.0, 1.5, 0.0], size=[1.0, 3.0, 1.0], voxel_m=0.1)
        assert small["fits"] is True and "placement_box_local_m" in small
        # an item as wide as the whole bay cannot fit past the column
        big = t.fits_in_volume(item_size=[0.9, 0.9, 0.9],
                               center=[0.0, 1.5, 0.0], size=[1.0, 3.0, 1.0], voxel_m=0.1)
        assert big["fits"] is False
        # an item larger than the box is rejected outright
        huge = t.fits_in_volume(item_size=[5.0, 5.0, 5.0],
                                center=[0.0, 1.5, 0.0], size=[1.0, 3.0, 1.0])
        assert huge["fits"] is False


def test_volume_by_saved_id():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        vid = t.define_volume("bay", [0.0, 1.5, 0.0], [1.0, 3.0, 1.0])["volume_id"]
        r = t.evaluate_volume(volume_id=vid, voxel_m=0.1)
        assert r["points_inside"] > 0 and r["volume_id"] == vid
        st.close()


def test_get_findings_reads_store():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        st.add_finding(instance_id=1, type="crack", severity="high",
                       description="x", confidence=0.9, frame_id=1)
        t = SpatialTools(st)
        r = t.get_findings(1)
        assert r["count"] == 1 and r["findings"][0]["type"] == "crack"
        assert t.get_findings(2).get("insufficient_data")
        st.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all phase 5 tool tests passed")
