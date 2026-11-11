"""SAM3's per-batch identity arithmetic on the device (segmentation/sam3_pairs_torch.py, USER 2026-11-11:
"todo debe ser GPU nada en cpu") against the paths of record in segmentation.pipeline: the same judgeable
pairs with the same IoUs in the same order, the same areas — and the dedupe / batch match built on them
give the same ids, the same masks and the same decision records."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import pipeline as P  # noqa: E402
from segmentation import sam3_pairs_torch as T  # noqa: E402

H, W = 40, 56


def _rect(rng, h=None, w=None):
    m = np.zeros((H, W), bool)
    h = h or int(rng.integers(4, H)); w = w or int(rng.integers(4, W))
    r0, c0 = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
    m[r0:r0 + h, c0:c0 + w] = True
    return m


def _batch(seed, n_frames=14, n_obj=9):
    """Frames of ids; id 20 is an exact copy of id 3 (the tracker's double id), id 21 a near copy of id 5
    (a few pixels off), id 22 appears on another grid in one frame."""
    rng = np.random.default_rng(seed)
    base = {o: _rect(rng) for o in range(n_obj)}
    batch = {}
    for f in range(n_frames):
        fm = {}
        for o in range(n_obj):
            if rng.random() < 0.8:
                m = base[o].copy()
                if rng.random() < 0.3:
                    m = np.roll(m, int(rng.integers(-2, 3)), axis=1)
                fm[o] = m.astype(np.uint8) if rng.random() < 0.5 else m
        if 3 in fm:
            fm[20] = fm[3].copy()
        if 5 in fm:
            near = fm[5].astype(bool).copy()
            near[np.flatnonzero(near.any(1))[:1]] = False                   # one row less
            fm[21] = near
        if f == 4:
            fm[22] = np.ones((H // 2, W // 2), bool)                          # another grid
        batch[f] = fm
    return batch


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_frame_pair_ious_are_the_records(seed):
    b = _batch(seed)
    ref = P._frame_pair_ious_numpy(b, 0.9)
    new = T.frame_pair_ious_t(b, 0.9, "cpu")
    assert ref[1] == new[1], "the areas"
    assert list(sorted(ref[0])) == list(sorted(new[0])) and all(ref[0][k] == new[0][k] for k in ref[0])
    assert (3, 20) in ref[0], "the exact copy is judged"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_overlap_ious_are_the_records(seed):
    a, b = _batch(seed), _batch(seed + 10)
    # the current batch's ids re-numbered, some of them the previous batch's masks
    curr = {f: {100 + o: (a[f][o] if o in a[f] and o % 2 == 0 else m) for o, m in fm.items()}
            for f, fm in b.items()}
    frames = [f for f in range(3, 13) if f in a and f in curr]
    pids = sorted({o for f in frames for o in a[f]})
    cids = sorted({o for f in frames for o in curr[f]})
    ref = P._overlap_pair_ious_numpy(a, curr, frames, pids, cids, 0.5)
    new = T.overlap_pair_ious_t(a, curr, frames, pids, cids, 0.5, "cpu")
    assert ref == new and len(ref) > 3


def test_the_dedupe_and_the_match_decide_as_the_records(monkeypatch):
    """The callers on the device twin and on the records, side by side: the same masks kept, the same ids,
    the same decision records."""
    from config import cfg as server_cfg
    b = _batch(5, n_frames=16)
    rec_dev, rec_ref = [], []
    dev = P._dedupe_masks_per_frame(b, 0.9, server_cfg, record=rec_dev)
    with monkeypatch.context() as m:
        m.setattr(T, "frame_pair_ious_t", lambda bm, thr, device: P._frame_pair_ious_numpy(bm, thr))
        ref = P._dedupe_masks_per_frame(b, 0.9, server_cfg, record=rec_ref)
    assert dev[1] == ref[1] >= 1, "the exact copy collapses"
    assert rec_dev == rec_ref
    assert {f: sorted(fm) for f, fm in dev[0].items()} == {f: sorted(fm) for f, fm in ref[0].items()}
    assert all(np.array_equal(dev[0][f][o], ref[0][f][o]) for f in dev[0] for o in dev[0][f])
    prev = b
    curr = {f: {100 + o: m for o, m in fm.items()} for f, fm in _batch(5, n_frames=16).items()}
    rec_dev, rec_ref = [], []
    r_dev = P._match_ids_iou(prev, curr, 6, 15, 0.5, 1000, server_cfg, record=rec_dev)
    with monkeypatch.context() as m:
        m.setattr(T, "overlap_pair_ious_t",
                  lambda pm, cm, fr, pi, ci, thr, device: P._overlap_pair_ious_numpy(pm, cm, fr, pi, ci, thr))
        r_ref = P._match_ids_iou(prev, curr, 6, 15, 0.5, 1000, server_cfg, record=rec_ref)
    assert r_dev == r_ref and rec_dev == rec_ref and any(r["linked"] for r in rec_dev)
