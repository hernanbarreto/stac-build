"""Variant B of F6's bend (USER 2026-11-22: "B"): the smooth correction field on top of the plane — its basis, its
robust fit, its choice on the held-out half A, and the record kept bit for bit without it."""
import numpy as np
import pytest

from precision import depth_on_f5 as D


def test_field_map_is_the_basis_on_every_pixel():
    rng = np.random.default_rng(0)
    W, H = 37, 23
    for G in (2, 3, 5):
        c = rng.normal(size=G * G)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        direct = (D.field_basis(uu.ravel(), vv.ravel(), W, H, G) @ c).reshape(H, W)
        np.testing.assert_allclose(D.field_map(c, W, H, G), direct, rtol=0, atol=1e-12)


def test_smoothness_is_zero_on_a_plane():
    G = 5
    j, i = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    plane = (0.3 + 0.02 * i - 0.05 * j).ravel()
    assert np.abs(D.field_smoothness(G) @ plane).max() < 1e-12


def test_fit_recovers_a_field_through_noise_and_outliers():
    rng = np.random.default_rng(1)
    W, H, G = 400, 300, 4
    truth = rng.normal(scale=0.02, size=G * G)
    u = rng.uniform(0, W - 1, 6000); v = rng.uniform(0, H - 1, 6000)
    B = D.field_basis(u, v, W, H, G)
    e = B @ truth + rng.normal(scale=0.002, size=len(u))
    bad = rng.choice(len(u), 300, replace=False)
    e[bad] += rng.normal(scale=0.5, size=len(bad))                  # 5 % gross outliers
    c = D.fit_field(B, e, D.field_smoothness(G), 1e-4, 1.345, 10)
    assert np.abs(c - truth).max() < 2e-3


def _session(field_amp, n=12, W=200, H=120, seed=3):
    """n keyframes whose landmark / Omega ratio is a plane times (1 + a smooth bump of amplitude field_amp)."""
    rng = np.random.default_rng(seed)
    rows, rows_A, coefs = {}, {}, {}
    for i in range(n):
        u = rng.uniform(0, W - 1, 3000); v = rng.uniform(0, H - 1, 3000)
        c = np.array([0.95 + 0.01 * i, 0.02, -0.01])
        bump = field_amp * np.sin(np.pi * u / W) * np.sin(np.pi * v / H)
        k = D.design(u, v, W, H) @ c * (1 + bump)
        zo = rng.uniform(1, 6, len(u))
        zl = zo * k * (1 + rng.normal(scale=0.003, size=len(u)))
        half = np.arange(len(u)) % 3
        fit, held = half != 0, half == 0
        rows[i] = (D.design(u[fit], v[fit], W, H), zl[fit] / zo[fit])
        rows_A[i] = (u[held], v[held], zl[held], zo[held])
        coefs[i] = c
    return rows, rows_A, coefs, n, W, H


class _F:
    grids = (2, 3, 4, 6)
    lambdas = (1.0, 0.1, 0.01)


def test_choice_keeps_the_plane_when_nothing_is_left():
    rows, rows_A, coefs, n, W, H = _session(0.0)
    out = D.choose_field(rows, rows_A, coefs, n, W, H, _F, 20, 1.345, 10, 1.1, 200, 0, log=lambda m: None)
    assert out["chosen"] is None and out["coef"] == {}


def test_choice_takes_a_field_when_the_plane_misses_a_bend():
    rows, rows_A, coefs, n, W, H = _session(0.04)
    out = D.choose_field(rows, rows_A, coefs, n, W, H, _F, 20, 1.345, 10, 1.1, 200, 0, log=lambda m: None)
    assert out["chosen"] is not None and len(out["coef"]) == n
    rep = out["report"]["candidates"]
    assert min(v for k, v in rep.items() if k != "plane") < rep["plane"] / 2


def test_without_a_field_the_bent_map_is_the_record_bit_for_bit():
    rng = np.random.default_rng(5)
    W, H = 64, 48
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = D.design(uu.ravel(), vv.ravel(), W, H)
    c = np.array([0.97, 0.013, -0.021])
    zo = rng.uniform(0.5, 9.0, (H, W)).astype(np.float32)
    record = (zo * (Dm @ c).reshape(H, W)).astype(np.float32)
    kmap = Dm @ c
    now = (zo * kmap.reshape(H, W)).astype(np.float32)
    assert np.array_equal(record, now)


def test_the_depth_source_is_omega_unless_moge3_is_asked_for():
    import copy
    from pathlib import Path
    import yaml
    from precision.config import PrecisionConfigError, load_precision_config
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert load_precision_config(raw).bend.depth_source == "omega"
    r2 = copy.deepcopy(raw)
    del r2["reconstruction"]["precision"]["bend"]["depth_source"]           # a session frozen before the key
    assert load_precision_config(r2).bend.depth_source == "omega"
    r2["reconstruction"]["precision"]["bend"]["depth_source"] = "moge3"
    assert load_precision_config(r2).bend.depth_source == "moge3"
    r2["reconstruction"]["precision"]["bend"]["depth_source"] = "midas"
    with pytest.raises(PrecisionConfigError):
        load_precision_config(r2)
