"""`correction.floor.bootstrap_draws: device` — the floor bootstrap's Poisson(1) resamples drawn by torch's
seeded generator on the device instead of numpy's stream (read on the CPU: 15 s per 1 M-point chunk floor
on pccr). The planes, inliers and verdicts are the numpy path's; σ differs from the reference's only by the
bootstrap's own resampling noise (~1/√(2·n_boot) relative); two runs are identical."""
import os

import numpy as np
import pytest

from correction import floor as FL
from tests.test_correction_floor_session import _cfg as _cfg_session, _scene

torch = pytest.importorskip("torch")
DEV = os.environ.get("STAC_TEST_DEVICE", "cpu")


def _on(dev, draws, fn):
    FL.use_device(dev)
    FL.use_draws(draws)
    try:
        return fn()
    finally:
        FL.use_device("cpu")
        FL.use_draws("numpy")


def test_device_draws_give_the_same_decisions_and_a_sigma_within_the_bootstrap_noise(tmp_path):
    s, _ = _scene(tmp_path, tilt_deg=3.0, offset=0.12, frame_rot_deg=25.0, platform=True)
    cfg = _cfg_session()
    solve = lambda: FL.solve_floor_session(s, cfg, log=lambda m: None)   # noqa: E731
    ref = _on("numpy", "numpy", solve)
    got = _on(DEV, "device", solve)
    again = _on(DEV, "device", solve)
    assert np.allclose(ref["R_kf"], got["R_kf"], atol=1e-9, rtol=0)
    assert np.allclose(ref["t_kf"], got["t_kf"], atol=1e-9, rtol=0)
    ra, rb, rc = ref["model_params"], got["model_params"], again["model_params"]
    for k in ("role", "chosen_rank", "inliers", "cameras_under", "tilt_applied", "height_applied"):
        assert ra[k] == rb[k] == rc[k], k
    assert rb["sigma_tilt_deg"] == rc["sigma_tilt_deg"] and rb["sigma_height_m"] == rc["sigma_height_m"]
    for la, lb in zip(ra["levels"], rb["levels"]):
        assert la["support"] == lb["support"] and la["real"] == lb["real"]
    # the raw σ (the record rounds them to 4–5 decimals): within the bootstrap's own noise
    cams = np.asarray(s.poses, np.float64)[:, :3, 3]
    rows, _ = FL.support_rows(s, cfg.floor.support_labels)
    fac, conf = FL._judge_of(cfg)
    rep = FL._session_repeatability(s, cfg)
    levels = lambda: FL.session_floor_levels(s.xyz, s.ks, cams, rows, cfg.floor, rep, fac, conf)   # noqa: E731
    lv_np = _on("numpy", "numpy", levels)
    lv_dev = _on(DEV, "device", levels)
    assert lv_np["chosen"] == lv_dev["chosen"] and len(lv_np["cands"]) == len(lv_dev["cands"]) == 2
    for qa, qb in zip(lv_np["cands"], lv_dev["cands"]):
        assert qa["support"] == qb["support"] and qa["real"] == qb["real"]
        for k in ("sigma_tilt_deg", "sigma_height_m"):
            assert qb[k] > 0 and abs(qb[k] - qa[k]) <= 0.1 * qa[k], (k, qa[k], qb[k])


def test_the_draws_mode_is_validated():
    from types import SimpleNamespace
    FL.use_draws(None)
    try:
        assert FL.draws_of(SimpleNamespace(bootstrap_draws="device")) == "device"
        with pytest.raises(ValueError):
            FL.draws_of(SimpleNamespace(bootstrap_draws="gpu"))
    finally:
        FL.use_draws("numpy")
