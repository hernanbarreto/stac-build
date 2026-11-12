"""tools/genrecon/layout_gpu.py: the twins of adapter.py's arithmetic over the cloud give the record's numbers — the
integers, the levels and the percentiles exactly, the transformed points within 1e-9 — on the cpu and, when one is
visible, on the card."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "genrecon"))
import layout_gpu as LG  # noqa: E402
from genrecon_levels import anchored_centers, chunk_levels, height_levels  # noqa: E402
from presence import plan_layout  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUBE = 3.7
V = CUBE / 32


def _t(a, device):
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def _room(rng, n=120_000):
    """A room in GenRecon's frame (z up): a floor, a table top, a ceiling, two walls and loose points."""
    k = n // 6
    xy = rng.uniform([-4.0, -3.0], [5.0, 6.0], (k, 2))
    floor = np.column_stack([xy, rng.normal(0.0, 0.004, k)])
    table = np.column_stack([rng.uniform([0.0, 0.0], [1.6, 0.8], (k, 2)), rng.normal(0.75, 0.003, k)])
    ceil = np.column_stack([rng.uniform([-4.0, -3.0], [5.0, 6.0], (k, 2)), rng.normal(2.9, 0.005, k)])
    wall_a = np.column_stack([np.full(k, -4.0) + rng.normal(0, 0.004, k), rng.uniform(-3, 6, k), rng.uniform(0, 2.9, k)])
    wall_b = np.column_stack([rng.uniform(-4, 5, k), np.full(k, 6.0) + rng.normal(0, 0.004, k), rng.uniform(0, 2.9, k)])
    loose = rng.uniform([-4.0, -3.0, -0.6], [5.0, 6.0, 3.3], (n - 5 * k, 3))
    return np.concatenate([floor, table, ceil, wall_a, wall_b, loose])


def test_the_device_key_is_typed(monkeypatch):
    assert LG.adapter_device({"adapter_device": "numpy"}) == "numpy"
    assert LG.adapter_device({"adapter_device": "cpu"}) == "cpu"
    for cfg in ({}, None, {"presence_device": "cuda"}):
        with pytest.raises(RuntimeError, match="genrecon.adapter_device is missing"):
            LG.adapter_device(cfg)
    for bad in ("gpu", "CUDA", "cuda:0", "", None, 0):
        with pytest.raises(RuntimeError, match="genrecon.adapter_device must be one of"):
            LG.adapter_device({"adapter_device": bad})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="genrecon.adapter_device is cuda and no CUDA device"):
        LG.adapter_device({"adapter_device": "cuda"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert LG.adapter_device({"adapter_device": "cuda"}) == "cuda"


@pytest.mark.parametrize("device", DEVICES)
def test_the_points_in_a_frame_are_the_records_within_1e_9(device):
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-40.0, 40.0, (200_000, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    A, t = 1.0371 * q, rng.uniform(-5.0, 5.0, 3)
    got = LG.to_frame(LG.as_points(xyz, device), A, t)
    assert got.dtype == torch.float64 and got.device.type == device
    assert np.abs(got.cpu().numpy() - (xyz @ A.T + t)).max() <= 1e-9
    # one rounding per operation: the same bits on every device
    assert torch.equal(got.cpu(), LG.to_frame(LG.as_points(xyz, "cpu"), A, t))
    # the adapter's turn y-up → z-up alone (a baked session): the record's bits
    up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    got = LG.to_frame(LG.as_points(xyz, device), up, np.zeros(3)).cpu().numpy()
    assert np.array_equal(got, xyz @ up.T + np.zeros(3))
    lo, hi = LG.extent(LG.as_points(xyz, device))
    assert np.array_equal(lo, xyz.min(0)) and np.array_equal(hi, xyz.max(0))


def _record_keys(P, origin, v, min_points):
    """presence.plan_layout's own lines."""
    ijk = np.clip(np.floor((P - origin) / v).astype(np.int64), 0, None)
    dims = ijk.max(0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    act, cnt = np.unique(key, return_counts=True)
    return act[cnt >= int(min_points)], dims


@pytest.mark.parametrize("device", DEVICES)
def test_the_active_voxels_are_the_records_also_for_points_on_a_voxels_face(device):
    rng = np.random.default_rng(5)
    # points lying on the lattice's planes where a product with 1 / v lands in the voxel before the division's:
    # torch divides a tensor by a Python number that way on the card — the twin must not
    d = np.arange(1, 400_000) * V
    faces = d[np.floor(d / V) != np.floor(d * (1.0 / V))]
    assert len(faces) > 100, "the case this test exists for is not in its data"
    on_planes = np.column_stack([faces[:3000], rng.permutation(faces)[:3000], rng.permutation(faces)[:3000]])
    for origin, P in ((np.zeros(3), np.concatenate([on_planes, np.abs(_room(rng))])),
                      (np.array([-4.43, -3.57, -1.2]), _room(rng))):
        for min_points in (1, 10):
            want, dims = _record_keys(P, origin, V, min_points)
            got, got_dims = LG.active_keys(LG.as_points(P, device), origin, V, min_points)
            assert got.dtype == np.int64 and np.array_equal(got_dims, dims)
            assert np.array_equal(got, want)


@pytest.mark.parametrize("device", DEVICES)
def test_the_levels_are_numpys_histogram(device):
    rng = np.random.default_rng(7)
    y = _room(rng)[:, 2]
    y = np.concatenate([y, np.arange(-12, 70) * 0.05, np.repeat(1.25, 1200), [np.nextafter(0.75, 1), np.nextafter(0.75, 0)]])
    ys = LG.sorted_heights(_t(y, device))
    for bin_m in (0.05, 0.1, 0.013):
        for share in (0.0, 1e-9, 1.0 / len(y), 0.01, 0.05, 0.2, 2.0):
            assert LG.levels_of_sorted(ys, bin_m, share) == height_levels(y, bin_m, share)
    for few in ([], [0.3], [0.05, 0.05, 0.05], [1.0, 1.0], [-0.2, 0.0, 0.2], [2.0, 2.05]):
        few = np.asarray(few, np.float64)
        assert LG.levels_of_sorted(LG.sorted_heights(_t(few, device)), 0.05, 0.01) == height_levels(few, 0.05, 0.01)
    with pytest.raises(ValueError, match="not a number"):
        LG.sorted_heights(_t(np.array([0.0, np.nan, 1.0]), device))


@pytest.mark.parametrize("device", DEVICES)
def test_the_percentiles_are_numpys(device):
    rng = np.random.default_rng(11)
    for n in (1, 2, 3, 7, 50, 51, 1001, 100_000):
        y = np.round(rng.normal(1.0, 2.0, n), 3 if n > 1000 else 9)          # ties in the long ones
        ys = LG.sorted_heights(_t(y, device))
        for q in (0, 0.2, 1, 5, 50, 95, 99, 99.97, 100):
            got = LG.percentile_of_sorted(ys, q)
            assert isinstance(got, np.float64) and got == np.percentile(y, q), (n, q)
        got = LG.percentile_of_sorted(ys, [1, 99])
        assert got.shape == (2,) and np.array_equal(got, np.percentile(y, [1, 99]))
    with pytest.raises(ValueError, match="no percentile"):
        LG.percentile_of_sorted(_t(np.zeros(0), device), 50)


@pytest.mark.parametrize("device", DEVICES)
def test_the_percentiles_of_every_group_are_numpys(device):
    rng = np.random.default_rng(13)
    sizes = [1, 2, 50, 51, 999, 3, 10_000, 100, 101, 64]
    groups = [np.round(rng.normal(rng.uniform(-3, 3), 1.5, n), 2 if n > 500 else 9) for n in sizes]
    values = _t(np.concatenate(groups), device)
    for qs in ([1], [1, 99], [0, 0.2, 50, 95, 99.97, 100]):
        got = LG.group_percentiles(values, sizes, qs)
        want = np.array([[np.percentile(g, q) for q in qs] for g in groups])
        assert got.dtype == np.float64 and np.array_equal(got, want), qs
        assert np.array_equal(got, np.array([np.percentile(g, qs) for g in groups]))     # a list of q, as adapter.py asks
    assert LG.group_percentiles(values[:0], [], [1, 99]).shape == (0, 2)
    for bad in ([len(values) - 1], [len(values), 0], [5, 5]):
        with pytest.raises(ValueError, match="are not the"):
            LG.group_percentiles(values, bad, [1])


@pytest.mark.parametrize("device", DEVICES)
def test_the_levels_json_through_the_twins_is_the_records(device):
    rng = np.random.default_rng(17)
    P = _room(rng)
    y = P[:, 2]
    floor_idx = np.arange(0, len(P) // 6)
    ceilings = [np.arange(2 * (len(P) // 6), 3 * (len(P) // 6)), np.arange(40, 90), np.zeros(0, np.int64)]
    y_t = _t(y, device)

    def at(idx):
        return LG.sorted_heights(y_t[_t(np.asarray(idx, np.int64), device)])

    for support, ceil, share in ((floor_idx, ceilings, 0.01), (floor_idx[:300], [], 0.01),
                                 (floor_idx[:300], ceilings[2:], 2.0),          # no level at all: the percentile 0.2
                                 (np.zeros(0, np.int64), ceilings[:1], 0.004)):
        want = chunk_levels(y, y[support], [y[g] for g in ceil], bin_m=0.05, min_share=share)
        got = chunk_levels(LG.sorted_heights(y_t), at(support), [at(g) for g in ceil], bin_m=0.05, min_share=share,
                           levels_of=LG.levels_of_sorted, percentile_of=LG.percentile_of_sorted)
        assert got == want and list(got) == list(want)
        assert all(type(got[k]) is type(want[k]) for k in want)


@pytest.mark.parametrize("device", DEVICES)
def test_the_layout_on_the_device_is_the_records(device):
    rng = np.random.default_rng(19)
    P = _room(rng)
    quiet = lambda m: None                                                   # noqa: E731
    for min_points, support in ((10, [0.0, 0.75]), (1, None), (10, [0.0])):
        centers = anchored_centers(0.0, float(P[:, 2].min()), float(P[:, 2].max()), CUBE)
        want = plan_layout(P, CUBE, log=quiet, min_points=min_points, z_centers=centers, support_levels=support)
        assert want["cells"].shape[0] > 1000 and (not support or want["ground_filled"] > 0)
        for cloud in (P, LG.as_points(P, device)):                           # an array, or a tensor already there
            got = plan_layout(cloud, CUBE, log=quiet, min_points=min_points, z_centers=centers,
                              support_levels=support, device=device)
            cells = got.pop("cells")
            assert cells.dtype == np.int32 and np.array_equal(cells, want["cells"])
            assert got =={k: v for k, v in want.items() if k != "cells"}
        same = plan_layout(P, CUBE, log=quiet, min_points=min_points, z_centers=centers, support_levels=support,
                           device="numpy")
        assert np.array_equal(same.pop("cells"), want["cells"]) and same == {k: v for k, v in want.items() if k != "cells"}
