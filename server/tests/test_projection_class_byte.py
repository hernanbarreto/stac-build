"""The projection's class byte never merges objects (pccr 2026-09-30).

`_match_masks_to_cloud` wrote `min(obj id, 255)` per point: every instance id above
254 shared byte 255 and one viewer toggle switched 50 objects at once (drywalls,
the backpack, beams, windows…). It now encodes through segmentation.republish._encode
and writes class_map.json, like every other writer."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.pipeline import _encode_classification  # noqa: E402


def _inst(iid, idx):
    return {"id": iid - 1, "instance_id": iid, "globalIndices": list(idx)}


def _paint(instances, cls, prev_map=None):
    cls, code, cmap = _encode_classification(instances, cls, prev_map)
    for i in instances:
        cls[np.asarray(i["globalIndices"])] = code[i["instance_id"]]
    return cls, cmap


def test_ids_above_254_keep_their_own_byte():
    insts = [_inst(iid, [k]) for k, iid in enumerate(range(200, 330))]   # 130 objects, ids up to 329
    cls, cmap = _paint(insts, np.zeros(130, np.uint8))
    assert len(np.unique(cls)) == 130, "two objects share a class byte"
    back = {int(k): int(v) for k, v in cmap["instance_of"].items()}
    assert [back[int(c)] for c in cls] == list(range(200, 330))


def test_the_byte_is_the_instance_id_while_ids_fit():
    cls, cmap = _paint([_inst(5, [0, 1]), _inst(9, [2])], np.zeros(3, np.uint8))
    assert cmap["encoding"] == "identity" and cls.tolist() == [5, 5, 9]


def test_incremental_translates_the_previous_bytes():
    # previous run: identity bytes for ids 3 and 7
    cls = np.array([3, 3, 7, 0, 0], np.uint8)
    cls, cmap = _paint([_inst(300, [3])], cls, prev_map={})        # a new id that forces compact
    back = {int(k): int(v) for k, v in cmap["instance_of"].items()}
    assert cmap["encoding"] == "compact"
    assert [back.get(int(c), 0) for c in cls] == [3, 3, 7, 300, 0]
