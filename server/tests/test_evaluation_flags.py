"""The two chain switches the evaluation battery's variants need (claude_stac.txt 2026-10-14, rule 4):
their default reproduces the chain as it is; false is the variant."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from precision.config import PrecisionConfigError, load_precision_config

SERVER_DIR = Path(__file__).resolve().parent.parent


def test_defaults_are_the_production_chain_and_a_missing_key_is_named():
    pc = load_precision_config()
    assert pc.refine.enabled is True and pc.bend.enabled is True
    raw = yaml.safe_load((SERVER_DIR / "config.yaml").read_text())
    for key in ("refine", "bend"):
        broken = copy.deepcopy(raw)
        del broken["reconstruction"]["precision"][key]["enabled"]
        with pytest.raises(PrecisionConfigError, match=f"reconstruction.precision.{key}.enabled"):
            load_precision_config(broken)


def test_disabled_bend_is_the_identity_for_every_keyframe():
    from precision.depth_on_f5 import IDENTITY, disabled_bend
    wb, rule, coefs, prov = disabled_bend(5)
    assert wb is None and rule["disabled"] and len(coefs) == 5 and len(prov) == 5
    assert all(np.array_equal(c, IDENTITY) for c in coefs.values())
    assert all(p["status"] == "disabled" and p["fit"] is None for p in prov.values())


def test_refine_core_without_solving_keeps_the_initial_state():
    from dataclasses import replace
    from precision import refine as R
    from precision.camera import undistort_solver
    from tests import test_precision_refine as T   # the F5 synthetic scene (skips without pycolmap 4)
    pc = load_precision_config()
    X, c2w = T._scene(n_kf=8, n_pts=400, seed=3)
    gt = [520.0, 520.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]
    kf = list(range(len(c2w)))
    track, frame, uv = T._observe(X, c2w, gt, 0.3, 0, kf)
    w2c0 = np.linalg.inv(T._perturb(c2w, 0.2, 0.02, 2))
    core = R.refine_core(w2c0, gt, T.WH, track, frame, uv, T._split(track), kf, 0.01, pc.refine,
                         undistort_solver(pc.camera), error_factor=T.FAC, log=lambda *a: None, solve=False)
    assert core["best"].name == "init" and np.array_equal(core["best"].w2c, w2c0)
    assert core["ladder"] == [] and core["rungs"] == {} and core["ladder_restricted_by"] == "refine.enabled: false"
    assert "init" in core["held"] and len(core["X"]) > 0 and core["X"] is core["X0"]
    # the production path (solve=True) is untouched: the same call with the flag solves the ladder
    core2 = R.refine_core(w2c0, gt, T.WH, track, frame, uv, T._split(track), kf, 0.01,
                          replace(pc.refine, enabled=True), undistort_solver(pc.camera), error_factor=T.FAC,
                          log=lambda *a: None)
    assert core2["best"].name != "init" and core2["ladder"]
