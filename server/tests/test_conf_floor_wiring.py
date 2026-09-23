"""The confidence floor must reach EVERY vendor config the reconstruction builds.

USER 2026-09-23: *"deben desaparecer de la nube eh!, porque no quiero que se
hagan ajustes de pose sobre ruido"*. The floor was first written at ONE call
site, inside a branch the production backend never takes, and three complete
reconstructions ran with it silently at 0 — the key simply was not in the YAML
handed to the vendor. This test is the structural guard: a builder that returns
a config without passing through `_apply_conf_floor` fails here, not three runs
later.

Parsed from source rather than imported: map_worker pulls the whole GPU stack.
"""
import ast
import pathlib

import pytest

MAP_WORKER = pathlib.Path(__file__).resolve().parent.parent / "workers" / "map_worker.py"
BUILDERS = ("_build_vggt_config", "_build_vggtomega_config")


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAP_WORKER.read_text())


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {MAP_WORKER}")


@pytest.mark.parametrize("name", BUILDERS)
def test_every_builder_returns_through_the_floor(tree, name):
    fn = _func(tree, name)
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, f"{name} returns nothing"
    for r in returns:
        assert isinstance(r.value, ast.Call) and getattr(r.value.func, "id", None) == "_apply_conf_floor", (
            f"{name} returns a vendor config that never passed through _apply_conf_floor "
            f"(line {r.lineno}) — the floor would silently stay 0 for the whole run")


def _helper(tree):
    src = MAP_WORKER.read_text()
    fn = _func(tree, "_apply_conf_floor")
    ns = {}
    exec(ast.get_source_segment(src, fn), ns)
    return ns["_apply_conf_floor"]


def test_the_floor_is_the_configured_number(tree):
    f = _helper(tree)
    cfg = f({}, {"reconstruction": {"simple": {"conf_min_norm": 0.10}}})
    assert cfg["Model"]["pose_fit_conf_min_norm"] == pytest.approx(0.10)


def test_absent_config_means_historical_behaviour(tree):
    f = _helper(tree)
    for config in ({}, {"reconstruction": {}}, {"reconstruction": {"simple": {}}},
                   {"reconstruction": {"simple": {"conf_min_norm": None}}}):
        assert f({}, config)["Model"]["pose_fit_conf_min_norm"] == 0.0


def test_it_does_not_clobber_the_rest_of_the_model_block(tree):
    f = _helper(tree)
    cfg = f({"Model": {"chunk_size": 60, "overlap": 30}},
            {"reconstruction": {"simple": {"conf_min_norm": 0.10}}})
    assert cfg["Model"]["chunk_size"] == 60 and cfg["Model"]["overlap"] == 30
    assert cfg["Model"]["pose_fit_conf_min_norm"] == pytest.approx(0.10)


def test_the_vendor_reads_the_key_this_test_guards():
    """A guard on a key nothing consumes guards nothing."""
    vendor = (MAP_WORKER.parent.parent.parent / "vendor" / "VGGT-Long" / "vggt_long.py")
    assert "pose_fit_conf_min_norm" in vendor.read_text(), (
        "the vendor stopped reading the key the builders write")
