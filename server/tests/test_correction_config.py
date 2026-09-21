"""Config: incomplete/out-of-range configs fail at load naming the key;
no decision literal lives outside config.py (static scan)."""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.config import (CorrectionConfigError,        # noqa: E402
                               load_correction_config)
from tests.synth_correction import make_correction_raw_cfg   # noqa: E402

PKG = Path(__file__).resolve().parents[1] / "correction"


def test_missing_key_names_the_key():
    raw = make_correction_raw_cfg()
    del raw["correction"]["gates"]["max_step_mm"]
    with pytest.raises(CorrectionConfigError, match="gates.max_step_mm"):
        load_correction_config(raw)


def test_missing_section_fails():
    with pytest.raises(CorrectionConfigError, match="correction"):
        load_correction_config({})


def test_out_of_range_names_the_key():
    with pytest.raises(CorrectionConfigError, match="solve.icp_trim"):
        load_correction_config(
            make_correction_raw_cfg(**{"solve.icp_trim": 1.5}))
    with pytest.raises(CorrectionConfigError, match="floor.model_default"):
        load_correction_config(
            make_correction_raw_cfg(**{"floor.model_default": "banana"}))
    with pytest.raises(CorrectionConfigError, match="runtime.workers"):
        load_correction_config(
            make_correction_raw_cfg(**{"runtime.workers": -2}))


def test_rewrite_depth_mode_fails_loudly():
    with pytest.raises(CorrectionConfigError, match="rewrite"):
        load_correction_config(make_correction_raw_cfg(
            **{"apply.depth_correction_mode": "rewrite"}))


def test_valid_config_loads():
    cfg = load_correction_config(make_correction_raw_cfg())
    assert cfg.solve.icp_iters == 60
    assert cfg.floor.model_default == "plane"


# ── static literal scan ──────────────────────────────────────────────────
# Decision thresholds live in config.yaml only. The scan flags FLOAT
# literals in the package (outside config.py) that are not identity /
# epsilon / unit-conversion values. Integers are indices, counts, rounding
# digits and progress percentages — not decision values — except the
# whitelisted structural ones below.
# 0.6744897501960817 is Phi^-1(3/4): the MAD→sigma conversion, a property of
# the normal distribution and not a value anyone chose.
_FLOAT_WHITELIST = {0.0, 1.0, -1.0, 2.0, 0.5, 1e-9, 1e-6, 100.0, 1000.0,
                    0.6744897501960817}


def _float_literals(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value not in _FLOAT_WHITELIST:
                yield node.lineno, node.value


def test_no_decision_literals_outside_config():
    offenders = []
    for py in sorted(PKG.glob("*.py")):
        if py.name == "config.py":
            continue
        for lineno, val in _float_literals(py):
            offenders.append(f"{py.name}:{lineno} = {val}")
    assert not offenders, ("decision-looking float literals outside "
                           f"config.py: {offenders}")
