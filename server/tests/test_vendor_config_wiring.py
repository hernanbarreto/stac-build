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


# ── THE CHUNK LAYOUT (USER 2026-10-06): the co-visibility plan's explicit ranges ──
# Every Omega run hands the fork Model.chunk_ranges — the SAME list chunk_plan.json stores —
# and no chunk_size / overlap (a second, uniform layout nobody ran). Same bug class as above:
# a key written on one branch only, or a list rebuilt instead of passed, fails here.

def _run_src(tree):
    return ast.get_source_segment(MAP_WORKER.read_text(), _func(tree, "_run_vggtomega"))


def test_the_builder_writes_no_uniform_layout(tree):
    import yaml
    src = MAP_WORKER.read_text()
    ns = {"Path": pathlib.Path}
    exec(ast.get_source_segment(src, _func(tree, "_apply_stac_model_keys")), ns)
    exec(ast.get_source_segment(src, _func(tree, "_build_vggtomega_config")), ns)
    ns["__file__"] = str(MAP_WORKER)
    cfg = ns["_build_vggtomega_config"]({"reconstruction": {"vggtomega": {"resolution": 512}}})
    assert "chunk_size" not in cfg["Model"] and "overlap" not in cfg["Model"], (
        "the vendor base YAML's chunk_size/overlap must not reach the fork — the layout is "
        "Model.chunk_ranges")
    base = yaml.safe_load((SERVER.parent / "vendor" / "VGGT-Long" / "configs" /
                           "stac_vggtomega.yaml").read_text())
    assert "chunk_size" in base["Model"], "premise: the base YAML carries a uniform size"
    with pytest.raises(RuntimeError, match="chunk_overlap was DELETED"):
        ns["_build_vggtomega_config"]({"reconstruction": {"vggtomega": {"resolution": 512,
                                                                        "chunk_overlap": 30}}})


def test_the_planned_ranges_are_the_ones_the_fork_and_the_plan_file_get(tree):
    run = _run_src(tree)
    # the ONE source of the layout: the co-visibility plan
    assert "_cplan = plan_session(output_dir.parent" in run
    assert '_ranges = [(int(a), int(b)) for a, b in _cplan["ranges"]]' in run
    # chunked: the same list reaches the fork and chunk_plan.json
    assert "_apply_chunked_metric(vggt_config, _ranges)" in run
    assert "_persist_chunk_plan(_ranges, _n_selected, _walk0, _covis_rep, _res)" in run
    applied = run[run.index("def _apply_chunked_metric"):run.index("def _ensure_anchors")]
    assert 'cfg_v["Model"]["chunk_ranges"] = [[int(a), int(b)] for a, b in _ranges]' in applied
    assert 'cfg_v["Model"].pop("chunk_size", None)' in applied
    assert 'cfg_v["Model"].pop("overlap", None)' in applied
    # single pass: the one range [0, n)
    assert 'vggt_config["Model"]["chunk_ranges"] = [[0, int(_n_selected)]]' in run
    # nothing re-derives a uniform layout anywhere in the Omega path
    for gone in ('["Model"]["chunk_size"] =', '["Model"]["overlap"] =', "chunk_ranges(_n_selected"):
        assert gone not in run, gone


def test_the_plan_file_is_the_explicit_list(tree):
    src = MAP_WORKER.read_text()
    ns = {}
    exec(ast.get_source_segment(src, _func(tree, "chunk_plan_doc")), ns)
    import sys
    sys.path.insert(0, str(SERVER))
    ranges = [(0, 63), (30, 169), (63, 189), (169, 211), (189, 289)]
    plan = ns["chunk_plan_doc"](ranges, 289, 17.3, {"D_total": 34.71},
                                {"resolution": 832, "mode": "max_size"})
    assert plan["chunk_ranges"] == [list(r) for r in ranges]
    assert plan["seam_overlaps"] == [33, 106, 20, 22]
    assert "chunk_size" not in plan and "overlap" not in plan


def test_the_fork_reads_the_explicit_layout():
    vendor = VENDOR.read_text()
    assert "self.config['Model'].get('chunk_ranges')" in vendor
    assert "validate_chunk_ranges(self.chunk_ranges_cfg" in vendor
    # every seam sliced by its own overlap, never by a uniform self.overlap
    assert "[-self.overlap:]" not in vendor and "[:self.overlap]" not in vendor


def test_the_resolution_reaches_the_config_f0_reads(tree):
    run = _run_src(tree)
    assert 'vggt_config["Model"]["omega_resolution"] = int(_res["resolution"])' in run
    i_res = run.index('vggt_config["Model"]["omega_resolution"] = int(_res["resolution"])')
    i_pass = run.index("_ok = _omega_pass(vggt_config, _tag1)")
    assert i_res < i_pass, "the resolution must be in the config the pass writes"
    cam = (SERVER / "precision" / "camera.py").read_text()
    assert 'return str(model["omega_mode"]), int(model["omega_resolution"])' in cam, \
        "F0 must keep reading Omega's grid from the run's own config"
