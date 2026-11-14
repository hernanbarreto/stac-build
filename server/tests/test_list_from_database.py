"""The Instances list is taken from the database (USER 2026-11-13: "la lista debe ser tomada de la
base de datos")."""
import numpy as np

from phase_r.instance_store import InstanceStore, read_instances
from segmentation.mask_fates import from_database


def test_the_store_is_read_without_writing(tmp_path):
    assert read_instances(tmp_path / "scene_r.db") is None
    store = InstanceStore(tmp_path / "scene_r.db")
    for iid, label, n in ((3, "door", 1500), (1, "floor", 9000), (8, "sign", 0)):
        store.upsert_instance(iid, label, source="test")
        if n:
            store.set_points(iid, np.zeros((n, 3)))
            store.set_obb(iid, np.eye(4), np.zeros(6), np.zeros(3), n_points=n)
    store.close()
    before = (tmp_path / "scene_r.db").stat().st_mtime_ns
    db = read_instances(tmp_path / "scene_r.db")
    assert db == {1: {"label": "floor", "points": 9000}, 3: {"label": "door", "points": 1500},
                  8: {"label": "sign", "points": 0}}
    assert (tmp_path / "scene_r.db").stat().st_mtime_ns == before


def test_the_database_is_the_list():
    db = {1: {"label": "floor", "points": 9000}, 3: {"label": "door (renamed)", "points": 1500},
          9: {"label": "split child", "points": 4000}}
    listed = [{"instance_id": 1, "label": "floor", "total_points": 8990, "color": "#0f0"},
              {"instance_id": 3, "label": "door", "total_points": 1500},
              {"instance_id": 4, "label": "beam", "total_points": 12},        # in the files, not in the store
              {"instance_id": 7, "label": "just propagated"}]                  # no record anywhere
    enriched = {1: {}, 3: {}, 4: {}, 9: {"color": "#00f"}}
    out, hidden = from_database(listed, [{"instance_id": 5, "reason": "too_small"}], enriched, {5: {}}, db)
    assert [(o["instance_id"], o["label"], o.get("total_points")) for o in out] == [
        (1, "floor", 9000), (3, "door (renamed)", 1500), (7, "just propagated", None),
        (9, "split child", 4000)]
    assert out[0]["color"] == "#0f0" and out[3]["color"] == "#00f"
    assert [(h["instance_id"], h["reason"]) for h in hidden] == [(5, "too_small"), (4, "not_in_database")]
    # a scan with no store: the lists come back as they were
    same, hid = from_database(listed, [], enriched, {}, None)
    assert same == listed and hid == []
