"""The ray judges of the co-visible split on the card equal the cKDTree recipe, judge for judge
(USER 2026-10-15: observatorio's split ran for hours on a 99 M-point cKDTree)."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from segmentation import pipeline as P
from segmentation.gap_rays_torch import FreeSpaceIndex, gap_rays_t


def _scene():
    rng = np.random.default_rng(5)
    floor = np.column_stack([rng.uniform(-4, 4, 60000), rng.uniform(-3, 3, 60000), rng.normal(0, 0.004, 60000)])
    wall = np.column_stack([rng.uniform(-4, 4, 30000), np.full(30000, 3.0) + rng.normal(0, 0.004, 30000), rng.uniform(0, 2.5, 30000)])
    desk_a = np.column_stack([rng.uniform(-2.0, -1.0, 15000), rng.uniform(0, 0.6, 15000), np.full(15000, 0.75) + rng.normal(0, 0.004, 15000)])
    desk_b = np.column_stack([rng.uniform(0.5, 1.5, 15000), rng.uniform(0, 0.6, 15000), np.full(15000, 0.75) + rng.normal(0, 0.004, 15000)])
    cloud = np.vstack([floor, wall, desk_a, desk_b])
    cams = np.array([[-3.0, -2.5, 1.6], [0.0, -2.5, 1.6], [2.5, -2.0, 1.6], [-0.5, 1.5, 1.7]])
    return cloud, desk_a, desk_b, cams


def test_any_within_is_exact_against_brute_force():
    cloud, da, db, cams = _scene()
    idx = FreeSpaceIndex(cloud, 0.05, device="cpu")
    rng = np.random.default_rng(9)
    S = np.vstack([rng.uniform(-4.5, 4.5, (3000, 3)) * np.array([1, 0.8, 0.6]) + np.array([0, 0, 0.8]),
                   cloud[rng.choice(len(cloud), 500)] + rng.normal(0, 0.03, (500, 3))])
    got = idx.any_within(S, 0.05)
    d, _ = cKDTree(cloud).query(S, k=1, distance_upper_bound=0.05)
    ref = np.isfinite(d)
    assert np.array_equal(got, ref)
    assert 0.1 < got.mean() < 0.9


def test_the_torch_judges_equal_the_ckdtree_judges():
    cloud, da, db, cams = _scene()
    tree = cKDTree(cloud)
    gap, reach = 0.10, float(np.linalg.norm(cloud.max(0) - cloud.min(0)))
    ref = P._gap_rays(da, db, tree, cams, gap, reach)
    idx = FreeSpaceIndex(cloud, gap / 2.0, device="cpu")
    got = gap_rays_t(da, db, idx, cams, gap, reach)
    assert len(ref) > 10 and got == ref, (len(ref), len(got))
    assert P._gap_rays(da, db, idx, cams, gap, reach) == ref, "the pipeline's own entry dispatches to the card"
    # the decision too
    d_ref = P._gap_is_free_space(da, db, tree, cams, gap, reach, factor=2.0, confidence=0.95, min_judges=5)
    d_got = P._gap_is_free_space(da, db, idx, cams, gap, reach, factor=2.0, confidence=0.95, min_judges=5)
    assert d_got == d_ref and d_ref["n_rays"] == len(ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_on_cuda_the_judges_equal_the_cpu_ones():
    cloud, da, db, cams = _scene()
    gap, reach = 0.10, float(np.linalg.norm(cloud.max(0) - cloud.min(0)))
    cpu = gap_rays_t(da, db, FreeSpaceIndex(cloud, gap / 2.0, device="cpu"), cams, gap, reach)
    gpu = [gap_rays_t(da, db, FreeSpaceIndex(cloud, gap / 2.0, device="cuda"), cams, gap, reach) for _ in range(2)]
    assert gpu[0] == cpu and gpu[1] == cpu


def _split_instances():
    """ONE instance holding two desks and a floor patch between them, plus the wall as another: the
    desks are seen together and the gap between them is free space → the split separates them."""
    cloud, da, db, cams = _scene()
    n_floor, n_wall = 60000, 30000
    idx_desks = np.arange(n_floor + n_wall, len(cloud))
    inst = [{"id": 1, "instance_id": 2, "label": "desk", "globalIndices": idx_desks.tolist()},
            {"id": 2, "instance_id": 3, "label": "wall", "globalIndices": np.arange(n_floor, n_floor + n_wall).tolist()}]
    # every point's keyframe = its nearest camera (frames 10, 20, 30, 40)
    frames = np.array([10, 20, 30, 40])
    d = np.linalg.norm(cloud[:, None, :] - cams[None, :, :], axis=2)
    frame_arr = frames[np.argmin(d, axis=1)]
    cam_centre = {int(f): cams[i] for i, f in enumerate(frames)}
    return cloud, inst, frame_arr, cam_centre


def test_the_batched_split_equals_the_reference_loop():
    import copy
    cloud, inst, frame_arr, cam_centre = _split_instances()
    outs = []
    for use_torch in (False, True):
        instances = copy.deepcopy(inst)
        rec = []
        n = P._split_covisible_components(instances, cloud, frame_arr, gap_m=0.10, min_points=1000,
                                          covis_share=0.5, cam_centre=cam_centre, min_walk_m=1.0,
                                          id_base=10, decision=(2.0, 0.95, 5), record=rec, use_torch=use_torch)
        outs.append((n, [(i["id"], i["instance_id"], len(i["globalIndices"]), sorted(i["globalIndices"])[:5]) for i in instances],
                     [(r["components"], r.get("distinct"), r.get("camera_returned"), (r.get("free_space") or {}).get("n_rays"))
                      for r in rec if "skipped" not in r]))
    assert outs[0][0] == outs[1][0] >= 1, "the desks are split in both paths"
    assert outs[0][1] == outs[1][1], "the same children, the same points, the same ids"
    assert outs[0][2] == outs[1][2], "the same verdicts, the same judges"


def test_sparse_cell_labelling_equals_ndimage_label():
    from scipy import ndimage
    rng = np.random.default_rng(11)
    pts = np.vstack([rng.uniform(0, 3, (5000, 3)), rng.uniform(5, 7, (3000, 3)), rng.uniform(0, 3, (200, 3)) + np.array([0, 0, 9.0])])
    ki = np.floor(pts / 0.3).astype(np.int64); ki -= ki.min(0)
    grid = np.zeros(ki.max(0) + 1, dtype=bool); grid[tuple(ki.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    ref = lab[tuple(ki.T)] - 1
    got, n2 = P._label_occupied_cells(ki)
    assert n2 == n >= 3
    # the same partition (labels may be numbered differently)
    pairs = set(zip(ref.tolist(), got.tolist()))
    assert len(pairs) == n and len({a for a, _ in pairs}) == n and len({b for _, b in pairs}) == n
