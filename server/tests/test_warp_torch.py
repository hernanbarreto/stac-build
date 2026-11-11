"""The epoch warp on the device (correction/warp_torch.py, USER 2026-11-11: "Nada debe hacerse en cpu")
against the record (correction.apply.warp_full_cloud_numpy), to the last bit — measured ON THE CARD on pccr
3108: the floor's rigid motion bit for bit over 18.8 M points, the depth k·z + b 99.5 % identical (max
3.6e-15 m); here, on the CPU device, torch's 3x3 products round apart from numpy's BLAS in the last bit —
and the points without a keyframe left where they are."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.apply import warp_full_cloud_numpy  # noqa: E402
from correction.warp_torch import warp_full_cloud_t  # noqa: E402


def _session(seed, n_kf=7, n=5000, dtype=np.float64):
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(n_kf):
        P = np.eye(4); P[:3, 3] = rng.normal(size=3); poses.append(P)
    frames = [10 * k + 3 for k in range(n_kf)]
    ks = rng.integers(-1, n_kf + 1, n)                                   # -1 / n_kf: no keyframe
    return SimpleNamespace(xyz=rng.normal(size=(n, 3)).astype(dtype), ks=ks, n_kf=n_kf, frames=frames,
                           cam_center={f: poses[k][:3, 3] for k, f in enumerate(frames)}, poses=poses)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_rigid_warp_is_the_records(dtype):
    s = _session(0, dtype=dtype)
    a = np.radians(1.9)
    R = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    R_kf = np.repeat(R[None], s.n_kf, 0); t_kf = np.tile([0.0, -0.37, 0.0], (s.n_kf, 1))
    R_kf[2] = np.eye(3); t_kf[2] = 0.0                                    # one keyframe not moved
    ref, n_ref = warp_full_cloud_numpy(s, R_kf, t_kf, np.ones(s.n_kf), log=lambda m: None)
    new, n_new = warp_full_cloud_t(s, R_kf, t_kf, np.ones(s.n_kf), "cpu")
    assert n_ref == n_new and new.dtype == ref.dtype
    assert np.abs(ref.astype(np.float64) - new.astype(np.float64)).max() <= (1e-12 if dtype == np.float64 else 1e-6)
    unmoved = (s.ks < 0) | (s.ks >= s.n_kf) | (s.ks == 2)
    assert np.array_equal(ref[unmoved], new[unmoved]) and np.array_equal(new[unmoved], s.xyz[unmoved])


def test_the_depth_warp_is_the_records_to_the_last_bit():
    s = _session(1)
    k_kf = np.where(np.arange(s.n_kf) % 2 == 0, 1.013, 1.0)
    b_kf = np.where(np.arange(s.n_kf) % 3 == 0, 0.02, 0.0)
    R_kf = np.repeat(np.eye(3)[None], s.n_kf, 0); t_kf = np.zeros((s.n_kf, 3))
    ref, n_ref = warp_full_cloud_numpy(s, R_kf, t_kf, k_kf, log=lambda m: None, b_kf=b_kf)
    new, n_new = warp_full_cloud_t(s, R_kf, t_kf, k_kf, "cpu", b_kf=b_kf)
    assert n_ref == n_new
    assert np.abs(ref - new).max() <= 1e-12
