"""Phase 1 of the mask→cloud projection on the device (segmentation/mask_match_torch.py, USER 2026-11-11:
"todo debe ser GPU nada en cpu") against the path of record, segmentation.pipeline._match_points_numpy:
the winner and its area, the runner-up and its area, the per-object mean area and the per-frame areas —
equal, array for array, on stores with overlaps, equal areas (the lower id wins), empty masks, masks of
keyframes no point was born in, points outside the mask grid, a cloud frame mapped to another mask
frame, and erosion on."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import pipeline as P  # noqa: E402
from segmentation.mask_match_torch import match_points_t  # noqa: E402

H, W = 96, 128


def _scene(tmp_path, seed, *, n_obj=12, n_frames=9, n_pts=4000, equal_sizes=False):
    rng = np.random.default_rng(seed)
    members, obj_ids = {}, list(range(3, 3 + n_obj))
    for o in obj_ids:
        for f in rng.choice(n_frames + 2, size=rng.integers(1, n_frames), replace=False):
            m = np.zeros((H, W), np.uint8)
            if rng.random() > 0.1:                                  # some stored masks are EMPTY
                h, w = ((40, 50) if equal_sizes else (int(rng.integers(8, H)), int(rng.integers(8, W))))
                r0, c0 = int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))
                m[r0:r0 + h, c0:c0 + w] = 1
            members[f"f{int(f)}_o{o}"] = m.astype(bool) if rng.random() > 0.5 else m
    members["obj_ids"] = np.array(obj_ids)
    path = tmp_path / f"seg_masks_{seed}.npz"
    np.savez_compressed(path, **members)
    frame_arr = rng.integers(0, n_frames, size=n_pts).astype(np.int32)   # frames n_frames.. have no point
    frame_groups = P._frame_groups_of(frame_arr)
    mask_frame_of = {f: (f + 1 if f == 2 else f) for f in frame_groups}  # one cloud frame reads another
    rows = rng.integers(-3, H, size=n_pts)                                # -1.. : outside the mask grid
    cols = rng.integers(-3, W, size=n_pts)
    rows[rows < 0], cols[cols < 0] = -1, -1
    return path, obj_ids, frame_groups, mask_frame_of, rows, cols, n_pts


def _both(path, obj_ids, frame_groups, mask_frame_of, rows, cols, n_pts, erosion=0):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    with np.load(path) as npz:
        ref = P._match_points_numpy(npz, set(npz.files), obj_ids, frame_groups, mask_frame_of, rows, cols,
                                    n_pts, H, W, erosion, kernel)
    new = match_points_t(path, obj_ids, frame_groups, mask_frame_of, rows, cols, n_pts, H, W, erosion,
                         kernel, "cpu", log=lambda m: None)
    return ref, new


def _assert_equal(ref, new):
    for k in range(4):
        assert ref[k].dtype == new[k].dtype
        assert np.array_equal(ref[k], new[k]), f"array {k} differs"
    assert list(ref[4].items()) == list(new[4].items()), "the per-object mean areas differ"
    assert [(o, list(d.items())) for o, d in ref[5].items()] == \
        [(o, list(d.items())) for o, d in new[5].items()], "the per-frame areas differ"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_device_match_is_the_records(tmp_path, seed):
    sc = _scene(tmp_path, seed)
    ref, new = _both(*sc)
    assert (ref[0] >= 0).sum() > 100 and (ref[2] >= 0).sum() > 20, "the scene must contest points"
    _assert_equal(ref, new)


def test_equal_areas_go_to_the_lower_id_as_in_the_record(tmp_path):
    sc = _scene(tmp_path, 7, equal_sizes=True)
    ref, new = _both(*sc)
    assert np.isfinite(ref[3]).sum() > 0 and (ref[1][np.isfinite(ref[3])] == ref[3][np.isfinite(ref[3])]).any()
    _assert_equal(ref, new)


def test_erosion_runs_where_the_masks_are_inflated(tmp_path):
    global H, W
    H0, W0 = H, W
    H, W = 160, 200                       # masks over 2000 / 10000 pixels reach both erosion branches
    try:
        sc = _scene(tmp_path, 11)
        ref, new = _both(*sc, erosion=2)
        _assert_equal(ref, new)
    finally:
        H, W = H0, W0


def test_a_mask_on_another_grid_fails_naming_it(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, f0_o1=np.ones((H, W + 1), np.uint8), obj_ids=np.array([1]))
    fg = {0: np.arange(5)}
    with pytest.raises(RuntimeError, match="f0_o1 is"):
        match_points_t(path, [1], fg, {0: 0}, np.zeros(5, np.int64), np.zeros(5, np.int64), 5, H, W, 0,
                       None, "cpu", log=lambda m: None)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_near_ties_on_the_device_are_the_records(seed):
    """The near-ties of the smaller-area rule (point 117) counted on the device: the same three counts as
    the record's per-point dictionary lookups — ids without a measured variation (0.0), exact ties,
    uncontested points (inf), blocks smaller than the cloud."""
    from segmentation.mask_match_torch import near_ties_t
    rng = np.random.default_rng(seed)
    n = 20_000
    best = rng.integers(0, 60, n).astype(np.int32)
    second = rng.integers(0, 60, n).astype(np.int32)
    pa = rng.integers(10, 500, n).astype(np.float64)
    sa = pa + rng.integers(0, 40, n)
    sa[rng.random(n) < 0.3] = np.inf                                 # uncontested
    tie = rng.random(n) < 0.05
    sa[tie] = pa[tie]                                                # exact ties
    var = {int(k): float(rng.random() * 30) for k in rng.choice(60, 40, replace=False)}
    ref = P._near_ties_numpy(best, pa, second, sa, var, 1.1)
    new = near_ties_t(best, pa, second, sa, var, 1.1, "cpu", block=3_000)
    assert ref == new and ref[1] > 0 and ref[2] > 0
