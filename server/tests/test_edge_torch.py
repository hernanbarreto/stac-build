"""The edge metric's torch path (precision/edge_torch.py, `reconstruction.precision.cloud_metrics.edge_device`
cpu | cuda) against the numpy path of record: the profile distances of every radius, the same r̂ on the
synthetic sections (sharp, filleted, eroded, other angles, noisier), the normals' bootstrap with the same
seeded resamples, and the whole metric on a synthetic box — the same creases, the same numbers."""
import os

import numpy as np
import pytest

from precision import edge_metric as em
from tests import synth_edges as se
from tests.test_edge_metric import FILLET_R, _raw_cfg, _section

torch = pytest.importorskip("torch")
from precision import edge_torch as ET  # noqa: E402

DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _on(dev, fn):
    em.use_device(dev)
    try:
        return fn()
    finally:
        em.use_device("cpu")


def test_profile_distances_of_every_radius_equal_the_reference():
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.05, 0.2, (3000, 2))
    for phi in (np.radians(60.0), np.pi / 2, np.radians(125.0)):
        grid = 0.002 * np.arange(40)
        D = ET.profile_dist_t(torch.as_tensor(q, device=DEV), torch.as_tensor(grid, device=DEV), phi, 0.002)
        D = D.cpu().numpy()
        for k, r in enumerate(grid):
            assert np.allclose(D[k], em.profile(q, float(r), phi, 0.002).dist, atol=1e-12, rtol=0), (phi, r)


def test_fillet_radius_equals_numpy_on_the_sections():
    delta, z, r_max = 0.002, em.z_of(0.95), 0.15
    cases = [(_section(0.0), np.pi / 2), (_section(FILLET_R), np.pi / 2), (_section(0.0, erode=0.012), np.pi / 2),
             (_section(0.012, phi=np.radians(70.0), seed=3), np.radians(70.0)),
             (_section(0.03, phi=np.radians(110.0), noise=0.002, seed=5), np.radians(110.0)),
             (_section(0.008, noise=0.0015, rows=7, seed=11), np.pi / 2)]
    for Q, phi in cases:
        a = em.fillet_radius(Q, phi, delta, r_max, z)
        b = ET.fillet_radius_t(Q, phi, delta, r_max, z, DEV, block_entries=200_000)   # several blocks
        assert a == b, (a, b, phi)
    assert em.fillet_radius(_section(FILLET_R), np.pi / 2, delta, r_max, z) == pytest.approx(FILLET_R, abs=delta)


def test_bootstrap_normals_draw_the_same_resamples():
    rng = np.random.default_rng(4)
    n = 5000
    P = np.column_stack([rng.uniform(-1, 1, n), rng.normal(0.0, 0.003, n), rng.uniform(-1, 1, n)])
    th = 0.3
    P = P @ np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
    a = em._bootstrap_normals(P, 150, 21)
    b = ET.bootstrap_normals_t(P, 150, 21, DEV, block=32)
    assert a.shape == b.shape == (150, 3)
    dots = np.abs(np.einsum("ij,ij->i", a, b))
    assert dots.min() > 1.0 - 1e-9
    assert np.allclose(np.abs(a), np.abs(b), atol=1e-9, rtol=0)


def _quiet(*_a, **_k):
    pass


def test_the_metric_on_the_device_equals_numpy_on_a_box(tmp_path):
    cfg = em.load_edge_config(_raw_cfg())
    P = se.box(r=FILLET_R)
    ply, seg = se.scene(tmp_path, "fillet", P, None)
    rep_np = _on("numpy", lambda: em.run(tmp_path, ply, seg, cfg=cfg, log=_quiet))
    rep_t = _on(DEV, lambda: em.run(tmp_path, ply, seg, cfg=cfg, log=_quiet))
    a = rep_np["reference"]["objects"][0]
    b = rep_t["reference"]["objects"][0]
    assert a["n_faces"] == b["n_faces"] and len(a["creases"]) == len(b["creases"]) > 0
    for ca, cb in zip(a["creases"], b["creases"]):
        assert ca["measured"] == cb["measured"] and ca["faces"] == cb["faces"] and ca["split"] == cb["split"]
        if not ca["measured"]:
            continue
        assert ca["r_hat_m"] == cb["r_hat_m"]
        for k in ("tau_m", "coverage", "overshoot", "sigma_a_m", "sigma_b_m", "theta_deg", "theta_null_deg"):
            assert np.isclose(ca[k], cb[k], rtol=1e-9, atol=1e-12, equal_nan=True), k
    sa, sb = a["summary"], b["summary"]
    for k in ("r_hat_m", "tau_over_sigma", "coverage", "overshoot"):
        assert np.isclose(sa[k], sb[k], rtol=1e-9, atol=1e-12, equal_nan=True), k
