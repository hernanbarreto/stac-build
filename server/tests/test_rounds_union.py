"""segmentation.autoprompt.rounds.segmented_masks on threads (station 2026-11-02: a round's VLM
waited 73 s for the store's 19,780 masks inflated one after the other) against the one-thread
loop it replaced, kept here as the reference: the same frames, the same boolean maps."""
import numpy as np

from segmentation import mask_space
from segmentation import pipeline as P
from segmentation.autoprompt import rounds as R


def _reference(out):
    union = {}
    with np.load(out / "seg_masks.npz") as npz:
        space = mask_space.declared_space(npz) or mask_space.SPACE_KEYFRAME
        kfs = mask_space.declared_keyframes(npz) or mask_space.keyframe_numbers(out) or []
        for key in npz.files:
            m = R._MASK_KEY.match(key)
            if not m:
                continue
            f = int(m.group(1))
            if space == mask_space.SPACE_KEYFRAME:
                if f >= len(kfs):
                    continue
                f = int(kfs[f])
            a = np.asarray(npz[key]) > 0
            if f in union:
                union[f] |= a
            else:
                union[f] = a.copy()
    return union


def test_the_union_on_threads_is_the_loops(tmp_path):
    rng = np.random.default_rng(3)
    kfs = [0, 7, 19, 30, 44, 61]
    members = {}
    for o in range(1, 30):
        for pos in rng.choice(len(kfs) + 2, size=4, replace=False):   # positions past the list too
            members[f"f{pos}_o{o}"] = rng.random((12, 20)) > 0.8
    members["scaled_res"] = np.array([12, 20])
    members[mask_space.KEYFRAMES_KEY] = np.asarray(kfs, np.int32)
    members[mask_space.NPZ_KEY] = mask_space.declaration(mask_space.SPACE_KEYFRAME)
    P.write_canonical_store(tmp_path / "seg_masks.npz", members)
    want = _reference(tmp_path)
    for threads in (1, 3, 8):
        R.UNION_THREADS = threads
        got = R.segmented_masks(tmp_path)
        assert got.keys() == want.keys()
        assert all(np.array_equal(got[f], want[f]) and got[f].dtype == np.bool_ for f in want)
    R.UNION_THREADS = 8
