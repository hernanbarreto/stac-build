"""The fragment consolidation's contiguity on the device (segmentation/fragments_torch.py, USER 2026-11-11:
"Nada debe hacerse en cpu") against the record (segmentation.pipeline._fragment_pairs_numpy): the same
instance pairs and adjacent-voxel counts — float32 and float64 clouds, a cell two instances share, an
empty instance — the same welded index lists, and the same consolidation end to end."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import pipeline as P  # noqa: E402
from segmentation.fragments_torch import fragment_pairs_t, union_indices_t  # noqa: E402


def _floor(seed, dtype):
    """A floor slab cut into pieces (some touching, one apart), one point shared by two pieces' cells."""
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(0, 6, 30000), rng.normal(0, 0.005, 30000), rng.uniform(0, 3, 30000)].astype(dtype)
    pts[:500, 0] += 9.0                                                   # a piece far from the rest
    piece = np.where(pts[:, 0] > 8, 4, (pts[:, 0] // 1.5).astype(int))
    inst = [{"id": k, "instance_id": k + 1, "label": "floor",
             "globalIndices": np.flatnonzero(piece == k).tolist()} for k in range(5)]
    inst.append({"id": 5, "instance_id": 6, "label": "floor", "globalIndices": []})
    return pts, inst


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_pairs_are_the_records(dtype):
    pts, inst = _floor(0, dtype)
    members = list(range(len(inst)))
    ref = P._fragment_pairs_numpy(inst, members, pts, 0.1)
    new = fragment_pairs_t(inst, members, pts, 0.1, "cpu")
    assert ref is not None and new is not None
    assert np.array_equal(ref[0], new[0]) and np.array_equal(ref[1], new[1])


def test_the_union_is_union1d():
    rng = np.random.default_rng(2)
    lists = [rng.integers(0, 1000, 300).tolist() for _ in range(4)]
    ref = np.asarray(lists[0], np.int64)
    for x in lists[1:]:
        ref = np.union1d(ref, np.asarray(x, np.int64))
    assert np.array_equal(union_indices_t(lists, "cpu"), ref)


def test_the_consolidation_end_to_end_is_the_records(monkeypatch):
    from segmentation import fragments_torch as F
    pts, inst = _floor(1, np.float32)
    a = [dict(i) for i in inst]
    b = [dict(i) for i in inst]
    rec_a, rec_b, dec_a, dec_b = {}, {}, [], []
    n_a = P._merge_label_fragments(a, pts, 0.1, record=rec_a, min_adjacent_pairs=5, decisions=dec_a)
    with monkeypatch.context() as m:
        m.setattr(F, "fragment_pairs_t", lambda inst_, mem, xyz, gap, dev: P._fragment_pairs_numpy(inst_, mem, xyz, gap))
        m.setattr(F, "union_indices_t", lambda lists, dev: (lambda acc: acc)(
            __import__("functools").reduce(lambda x, y: np.union1d(x, np.asarray(y, np.int64)),
                                           lists[1:], np.asarray(lists[0], np.int64))))
        n_b = P._merge_label_fragments(b, pts, 0.1, record=rec_b, min_adjacent_pairs=5, decisions=dec_b)
    assert n_a == n_b >= 2 and rec_a == rec_b and dec_a == dec_b
    assert [i["globalIndices"] for i in a] == [i["globalIndices"] for i in b]
