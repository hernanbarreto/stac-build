"""Every STAC decision must reach the vendor config the reconstruction actually runs.

THE BUG CLASS, found three times on 2026-09-23 and each time silent — the run
succeeds, it simply does not do what config.yaml says:
  · `pose_fit_conf_min_norm` was written inside `_run_mapanything`'s `if cond:`
    branch, which the production backend never takes → three full runs with the
    confidence floor at 0.
  · `Model.mask_sky` was read by the vendor with a default of True and written by
    NOBODY → skyseg ran on every frame of every scene, indoors included.
  · `Model.intra_chunk` was written only inside `_apply_chunked_metric` → the
    single-pass layout lost the one adjustment stage that works on one chunk.

So the guards here are structural: a builder that returns a config which did not
pass through `_apply_stac_model_keys`, or an adjustment stage with a single
writer, fails at test time instead of three reconstructions later.

Parsed from source rather than imported: map_worker pulls the whole GPU stack.
"""
import ast
import pathlib

import pytest

SERVER = pathlib.Path(__file__).resolve().parent.parent
MAP_WORKER = SERVER / "workers" / "map_worker.py"
VENDOR = SERVER.parent / "vendor" / "VGGT-Long" / "vggt_long.py"
BUILDERS = ("_build_vggt_config", "_build_vggtomega_config")


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAP_WORKER.read_text())


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {MAP_WORKER}")


def _helper(tree):
    fn = _func(tree, "_apply_stac_model_keys")
    ns = {}
    exec(ast.get_source_segment(MAP_WORKER.read_text(), fn), ns)
    return ns["_apply_stac_model_keys"]


@pytest.mark.parametrize("name", BUILDERS)
def test_every_builder_returns_through_the_helper(tree, name):
    fn = _func(tree, name)
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, f"{name} returns nothing"
    for r in returns:
        assert isinstance(r.value, ast.Call) and getattr(r.value.func, "id", None) == "_apply_stac_model_keys", (
            f"{name} returns a vendor config that never passed through "
            f"_apply_stac_model_keys (line {r.lineno}) — its STAC keys would "
            f"silently keep the vendor's defaults for the whole run")


def test_the_confidence_floor_is_the_configured_number(tree):
    f = _helper(tree)
    cfg = f({}, {"reconstruction": {"simple": {"conf_min_norm": 0.10}}})
    assert cfg["Model"]["pose_fit_conf_min_norm"] == pytest.approx(0.10)


def test_the_sky_mask_is_declared_on_every_config(tree):
    """Its absence, not its value, was the defect: True was unreachable-by-config."""
    f = _helper(tree)
    assert f({}, {})["Model"]["mask_sky"] is True, "the default must be today's behaviour"
    assert f({}, {"reconstruction": {"simple": {"mask_sky": False}}})["Model"]["mask_sky"] is False, (
        "turning the sky mask off must be possible from config.yaml")


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


def test_intra_chunk_is_written_outside_the_chunked_path():
    """It guards on `len(chunk_indices) < 1` — a single chunk RUNS it.

    Every seam stage guards on `< 2` and stands down by itself; this one was
    written to work on one chunk, and the single-pass layout wrote it nowhere.
    """
    src = MAP_WORKER.read_text()
    writers = src.count('["Model"]["intra_chunk"]')
    assert writers >= 2, (
        "intra_chunk has a single writer again — if that writer is "
        "_apply_chunked_metric, the single-pass layout silently loses the only "
        "adjustment stage it can run")


def test_the_vendor_still_reads_what_the_builders_write():
    """A guard on keys nothing consumes guards nothing."""
    vendor = VENDOR.read_text()
    for key in ("pose_fit_conf_min_norm", "mask_sky", "intra_chunk",
                "sigma_seam", "sigma_anchor"):
        assert key in vendor, f"the vendor stopped reading {key}"


def test_the_single_chunk_guard_is_still_the_one_that_makes_this_true():
    """If the vendor ever raises intra_chunk's guard to `< 2`, this test's
    premise dies and the wiring above becomes pointless — say so loudly."""
    vendor = VENDOR.read_text()
    assert "get('intra_chunk') or len(self.chunk_indices) < 1" in vendor, (
        "intra_chunk's chunk-count guard changed — re-check whether the "
        "single-pass layout should still write it")
