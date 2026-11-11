"""segmentation/sam3_fast.py: the planning metadata's shallow copy holds what the vendor's deep copy
holds — the same keys, every frame of the score history its own dict with the same scores — and a
write to the new metadata's current frame never reaches the previous one (USER 2026-11-11). The masks
on the card were compared bit for bit on pccr 2026-08-31 (120 keyframes, wall / cable / light)."""
import copy
from collections import defaultdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from segmentation import sam3_fast as F  # noqa: E402


def _prev(n_frames=6, n_obj=4):
    hist = defaultdict(dict)
    for f in range(n_frames):
        for o in range(n_obj):
            hist[f][o] = torch.tensor(0.1 * f + o, dtype=torch.float32)
    return {"obj_ids_per_gpu": [np.arange(n_obj)], "obj_ids_all_gpu": np.arange(n_obj),
            "num_obj_per_gpu": np.array([n_obj]), "obj_id_to_score": {o: np.float32(o) for o in range(n_obj)},
            "obj_id_to_sam2_score_frame_wise": hist, "obj_id_to_last_occluded": {1: 3},
            "max_obj_id": n_obj - 1, "gpu_metadata": object()}


def _vendor(prev):
    score_key = "obj_id_to_sam2_score_frame_wise"
    return {"obj_ids_per_gpu": copy.deepcopy(prev["obj_ids_per_gpu"]), "obj_ids_all_gpu": None,
            "num_obj_per_gpu": copy.deepcopy(prev["num_obj_per_gpu"]),
            "obj_id_to_score": copy.deepcopy(prev["obj_id_to_score"]),
            score_key: copy.deepcopy(prev[score_key]), "obj_id_to_last_occluded": {},
            "max_obj_id": copy.deepcopy(prev["max_obj_id"])}


def test_the_shallow_copy_holds_what_the_deep_copy_holds():
    prev = _prev()
    a, b = _vendor(prev), F.shallow_planning_metadata(None, prev)
    assert list(a) == list(b)
    ha, hb = a["obj_id_to_sam2_score_frame_wise"], b["obj_id_to_sam2_score_frame_wise"]
    assert isinstance(hb, defaultdict) and hb.default_factory is dict
    assert list(ha) == list(hb)
    for f in ha:
        assert list(ha[f]) == list(hb[f])
        assert all(torch.equal(ha[f][o], hb[f][o]) for o in ha[f])
        assert hb[f] is not prev["obj_id_to_sam2_score_frame_wise"][f]      # each frame its own dict
    for k in ("obj_ids_per_gpu", "num_obj_per_gpu"):
        assert all(np.array_equal(x, y) for x, y in zip(np.atleast_1d(a[k]), np.atleast_1d(b[k])))
    assert a["obj_id_to_score"] == b["obj_id_to_score"] and a["max_obj_id"] == b["max_obj_id"]


def test_a_write_to_the_new_metadata_never_reaches_the_previous_one():
    prev = _prev()
    new = F.shallow_planning_metadata(None, prev)
    hist_new, hist_prev = new["obj_id_to_sam2_score_frame_wise"], prev["obj_id_to_sam2_score_frame_wise"]
    hist_new[6].update({0: torch.tensor(9.0)})                   # the current frame, as every writer does
    hist_new[5][0] = torch.tensor(-1e4)                          # even a past frame's entry
    new["obj_id_to_score"][0] = -1e4
    assert 6 not in hist_prev and float(hist_prev[5][0]) == pytest.approx(0.5)
    assert prev["obj_id_to_score"][0] == np.float32(0)


def test_the_mode_is_validated_and_deep_leaves_the_vendor_alone():
    with pytest.raises(ValueError, match="planning_copy"):
        F.install_planning_copy("fast")
    assert F.install_planning_copy("deep") == "deep"


def test_install_switches_the_vendor_method_and_back():
    pytest.importorskip("sam3.model.sam3_video_base")
    from sam3.model.sam3_video_base import Sam3VideoBase
    original = Sam3VideoBase.__dict__["_create_planning_metadata"]
    try:
        F.install_planning_copy("shallow")
        assert Sam3VideoBase.__dict__["_create_planning_metadata"] is F.shallow_planning_metadata
        F.install_planning_copy("deep")
        assert Sam3VideoBase.__dict__["_create_planning_metadata"] is original
    finally:
        Sam3VideoBase._create_planning_metadata = original
