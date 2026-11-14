"""The minimum object size applied to a scan already segmented (USER 2026-11-13: "aparecen muchos
elementos con 1 o muy pocos puntos" / "Quién dijo que lo estructural nunca se exime")."""
import json

import numpy as np
import pytest

from phase_r.instance_store import InstanceStore
from segmentation import apply_min_points as A
from segmentation.republish import CLASSIFICATION, INSTANCE_IDS


def _scan(tmp_path, n_points=5000):
    out = tmp_path / "output"
    out.mkdir()
    sizes = {1: ("floor", 3000), 2: ("door", 1), 3: ("beam", 40), 4: ("cabinet", 1200), 9: ("pipe", 999)}
    insts, cursor = [], 0
    store = InstanceStore(out / "scene_r.db")
    for iid, (label, n) in sizes.items():
        gi = list(range(cursor, cursor + n))
        cursor += n
        insts.append({"id": iid, "instance_id": iid, "label": label, "total_points": n,
                      "globalIndices": gi, "kept_structural_under_min_points": n < 1000})
        store.upsert_instance(iid, label, source="test")
        store.set_points(iid, np.zeros((n, 3)))
    store.close()
    (out / "segmentation_result.json").write_text(json.dumps(
        {"total_points": n_points, "instances": insts, "absorbed": {"7": {"reason": "unmatched"}}}))
    (out / "seg_broadcast.json").write_text("{}")
    from segmentation.republish import republish_membership
    republish_membership(out, n_points=n_points, log=lambda m: None)
    return out


def test_every_label_under_the_minimum_is_dropped_and_recorded(tmp_path):
    out = _scan(tmp_path)
    before_cls = np.load(out / INSTANCE_IDS).copy()
    plan = A.plan(out, 1000)
    assert (plan["drop"], plan["drop_points"]) == (3, 1040)
    assert plan["drop_by_label"] == {"beam": 1, "door": 1, "pipe": 1}
    assert np.array_equal(np.load(out / INSTANCE_IDS), before_cls)             # a plan writes nothing
    rep = A.apply(out, 1000, log=lambda m: None)
    assert rep["changed"] and rep["kept"] == 2 and rep["store_removed"] == 3
    res = json.loads((out / "segmentation_result.json").read_text())
    assert [i["instance_id"] for i in res["instances"]] == [1, 4]
    assert res["absorbed"]["2"] == {"into": None, "into_label": None, "reason": "too_small",
                                    "label": "door", "points": 1, "min_points": 1000, "margin": -999}
    assert res["absorbed"]["7"] == {"reason": "unmatched"}                      # what was there stays
    assert res["segmented_points"] == 4200 and res["coverage"] == pytest.approx(4200 / 5000)
    assert all("kept_structural_under_min_points" not in i for i in res["instances"])
    ids = np.load(out / INSTANCE_IDS)
    assert sorted(np.unique(ids).tolist()) == [0, 1, 4]                        # their points: unsegmented
    assert int((ids == 0).sum()) == 5000 - 4200
    assert int((np.load(out / CLASSIFICATION) == 0).sum()) == 5000 - 4200
    store = InstanceStore(out / "scene_r.db")
    assert [i["instance_id"] for i in store.list_instances()] == [1, 4]
    store.close()
    assert json.loads((out / "seg_broadcast.json").read_text())["segmented_points"] == 4200
    # applied again there is nothing left to drop and the backup stays; restored, all is as before
    assert A.apply(out, 1000, log=lambda m: None)["changed"] is False
    assert (out / A.BACKUP_DIR / "segmentation_result.json").exists()
    A.restore(out, log=lambda m: None)
    assert np.array_equal(np.load(out / INSTANCE_IDS), before_cls)
    assert len(json.loads((out / "segmentation_result.json").read_text())["instances"]) == 5
    store = InstanceStore(out / "scene_r.db")
    assert len(store.list_instances()) == 5
    store.close()


def test_nothing_under_the_minimum_changes_nothing(tmp_path):
    out = _scan(tmp_path)
    rep = A.apply(out, 1, log=lambda m: None)
    assert rep["changed"] is False and not (out / A.BACKUP_DIR).exists()


def test_the_minimum_is_the_configured_one():
    assert A._min_points() == 1000
