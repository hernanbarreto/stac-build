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


# ── new tools + display-frame tests (Phase 5 completion) ────────────

def _make_store_with_cameras(path):
    import json as _json
    st = _make_store(path)
    st.set_meta("camera_positions", _json.dumps({0: [5.0, 1.5, 0.0], 9: [4.0, 1.5, 0.0]}))
    return st


def test_ego_tools():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store_with_cameras(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        me = t.get_my_position()
        assert me["position_m"] == [4.0, 1.5, 0.0] and me["frame"] == 9
        dm = t.get_distance_from_me(2)  # wall at x=2, half-width 0.1 -> ~1.9 m
        assert 1.7 < dm["distance_m"] < 2.1, dm
        st.close()


def test_ego_tools_insufficient_without_cameras():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        assert t.get_my_position().get("insufficient_data")
        st.close()


def test_measure_between_features():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        r = t.measure_between(1, 2, "centroid", "centroid")
        assert abs(r["distance_m"] - 2.0) < 0.05
        top = t.measure_between(1, 1, "top", "bottom")   # column height 3 m
        assert abs(top["distance_m"] - 3.0) < 0.05
        bad = t.measure_between(1, 2, "corner", "centroid")
        assert bad.get("insufficient_data")
        st.close()


def test_fits_through_opening_and_gap():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        # opening: door 0.9 x 2.1 x 0.1 (thickness)
        c = np.array([4.0, 1.05, 0.0]); ext = np.array([0.9, 2.1, 0.1])
        st.upsert_instance(3, "door", n_views=3)
        st.set_points(3, _box_points(c, ext, 300))
        T = np.eye(4); T[:3, 3] = c
        st.set_obb(3, T, _aabb(ext), c, n_points=300)
        t = SpatialTools(st)
        ok = t.fits_through([0.6, 1.8, 0.6], opening_id=3)
        assert ok["fits"] is True and ok["mode"] == "opening", ok
        no = t.fits_through([1.2, 1.8, 1.2], opening_id=3)
        assert no["fits"] is False
        gap = t.fits_through([1.0, 1.0, 1.0], id1=1, id2=2)   # ~1.7 m gap
        assert gap["fits"] is True and gap["mode"] == "gap", gap
        st.close()


def test_plumb_level_and_confidence():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        p = t.get_plumb(1)     # synthetic column is exactly vertical
        assert p["plumb_deviation_deg"] < 2.0 and "confidence" in p
        # a leaning column: tilt points 5° about z
        pts = _box_points(np.zeros(3), np.array([0.3, 3.0, 0.3]), 600)
        ang = np.radians(5)
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        st.upsert_instance(4, "column leaning", n_views=3)
        st.set_points(4, pts @ R.T)
        t2 = SpatialTools(st)
        p2 = t2.get_plumb(4)
        assert 3.0 < p2["plumb_deviation_deg"] < 7.0, p2
        # level of a flat slab
        slab = _box_points(np.array([0, 0, 0]), np.array([4.0, 0.02, 4.0]), 800)
        st.upsert_instance(5, "slab", n_views=3)
        st.set_points(5, slab)
        lv = SpatialTools(st).get_level(5)
        assert lv["level_deviation_deg"] < 1.0 and "confidence" in lv
        st.close()


def test_alignment_health_and_onion_confidence():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        st.set_vote_metrics(1, 0.3, 0.9, 700)
        st.set_onion_metric(1, True, 0.08, 12.0)
        st.set_onion_metric(1, False, 0.0, 0.0, seam="w000|w001")
        t = SpatialTools(st)
        h = t.get_alignment_health(1)
        assert h["vote"]["mean_entropy"] == 0.3 and "confidence" in h
        o = t.get_onion_report(1)
        assert o["bimodal"] is True and abs(o["separation_m"] - 0.08) < 1e-6
        assert "w000|w001" in o["seams_available"]
        oseam = t.get_onion_report(1, seam="w000|w001")
        assert oseam["bimodal"] is False and oseam["seam"] == "w000|w001"
        hist = t.get_instance_history(1)
        assert "confidence" in hist and hist["n_views"] == 5
        st.close()


def test_height_profile_with_floor():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        # vault-like ceiling slab at y=3, floor at y=0
        ceil_pts = _box_points(np.array([0, 3.0, 0]), np.array([6.0, 0.05, 2.0]), 2000)
        floor_pts = _box_points(np.array([0, 0.0, 0]), np.array([6.0, 0.02, 2.0]), 2000)
        st.upsert_instance(6, "vault", n_views=4)
        st.set_points(6, ceil_pts)
        T = np.eye(4); T[:3, 3] = [0, 3.0, 0]
        st.set_obb(6, T, _aabb(np.array([6.0, 0.05, 2.0])), np.array([0, 3.0, 0]))
        st.upsert_instance(7, "floor", n_views=4)
        st.set_points(7, floor_pts)
        t = SpatialTools(st)
        prof = t.get_height_profile(6, n_stations=6, floor_id=7)
        assert "min_clearance_m" in prof, prof
        assert 2.8 < prof["min_clearance_m"] < 3.1, prof["min_clearance_m"]
        st.close()


def test_flatness_report_fallback_and_bridge():
    import json as _json
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        # flat wall -> fallback plane fit passes
        flat = _box_points(np.array([0, 1.5, 3.0]), np.array([4.0, 3.0, 0.002]), 1500)
        st.upsert_instance(8, "wall flat", n_views=4)
        st.set_points(8, flat)
        t = SpatialTools(st)
        r = t.get_flatness_report(8)
        assert r["backend"] == "plane_fit_fallback" and r["flatness_pass"] is True, r
        # surface_fit artifact bridge takes precedence
        art = os.path.join(d, "surface_fit", "wall_8")
        os.makedirs(art)
        with open(os.path.join(art, "residuals.json"), "w") as f:
            _json.dump({"stats": {"rms_mm": 1.2, "p95_mm": 2.5, "max_mm": 4.0,
                                  "flatness_worst_mm": 3.0, "flatness_span_m": 2.0,
                                  "flatness_pass": True}, "findings": []}, f)
        t2 = SpatialTools(st)
        r2 = t2.get_flatness_report(8)
        assert r2["backend"] == "surface_fitting" and r2["rms_mm"] == 1.2, r2
        st.close()


def test_display_transform_applies_to_all_outputs():
    """The misplaced-measurement bug: tools must answer in the DISPLAY frame
    (floor_transform s·R·p+t), matching the rendered cloud."""
    from scipy.spatial.transform import Rotation
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        s = 2.0
        R = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        tvec = np.array([1.0, -2.0, 0.5])
        M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = tvec
        t = SpatialTools(st, display_transform=M)
        # position: column centre (0,1.5,0) -> s·R·p + t
        expect = s * (R @ np.array([0, 1.5, 0.0])) + tvec
        got = np.array(t.get_position(1)["position_m"])
        assert np.allclose(got, expect, atol=1e-3), (got, expect)
        # sizes/volumes scale by s / s^3
        sz = t.get_object_size(1)
        dims = sorted([sz["width_m"], sz["height_m"], sz["depth_m"]])
        assert abs(max(dims) - 3.0 * s) < 0.05
        vol = t.get_object_volume(1)["bbox_volume_m3"]
        assert abs(vol - (0.4 * 3.0 * 0.4) * s ** 3) < 0.2
        # distances scale by s (centres 2 m apart -> gap scales too)
        dist = t.get_distance(1, 2)["distance_m"]
        assert 1.4 * s < dist < 2.0 * s, dist
        st.close()


def test_display_transform_loaded_from_npz():
    from phase5_qa.tools import load_display_transform
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.db")
        st = _make_store(p)
        np.savez(os.path.join(d, "floor_transform.npz"),
                 s=np.float64(1.0), R=np.eye(3), t=np.array([0.0, 2.0, 0.0]))
        M = load_display_transform(p)
        assert M is not None and np.allclose(M[:3, 3], [0, 2, 0])
        t = SpatialTools(st)  # auto-loads from next to the store
        pos = t.get_position(1)["position_m"]
        assert abs(pos[1] - 3.5) < 0.01, pos  # 1.5 + 2.0 offset
        st.close()


def test_findings_envelope_provenance():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        st.add_finding(instance_id=1, type="crack", severity="medium",
                       description="hairline", confidence=0.8, point3d=[0, 1, 0])
        st.add_finding(instance_id=1, type="moisture", severity="low",
                       description="damp patch", confidence=0.7, point3d=[0, 2, 0],
                       status="human_validated")
        t = SpatialTools(st)
        f = t.get_findings(1)
        assert f["source"] == "mixed" and f["n_validated"] == 1, f
        st.close()


def test_evaluate_volume_explicit_reports_no_null_id():
    with tempfile.TemporaryDirectory() as d:
        st = _make_store(os.path.join(d, "s.db"))
        t = SpatialTools(st)
        r = t.evaluate_volume(center=[0, 1.5, 0], size=[1.0, 3.2, 1.0])
        assert r["explicit_box"] is True and r["volume_id"] is None
        assert r["points_inside"] > 0
        st.close()
