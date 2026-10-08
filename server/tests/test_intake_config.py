"""intake: the real config.yaml loads; a missing / mistyped / out-of-range key
fails at load naming it; class lists are validated against the content
vocabulary and against each other; no decision literal lives in the intake
package outside its config.py (static scan, the test_precision_config.py
pattern)."""

import ast
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake.config import (CONTENT_CLASSES, REMOVED_KEYS, SAM3_SCOPES,   # noqa: E402
                           IntakeConfig, IntakeConfigError, load_intake_config)

SERVER = Path(__file__).resolve().parents[1]
INTAKE = SERVER / "intake"


def _raw():
    with open(SERVER / "config.yaml") as f:
        return yaml.safe_load(f)


def _expect(raw, key_fragment):
    with pytest.raises(IntakeConfigError, match=key_fragment):
        load_intake_config(raw)


# ── the real file ────────────────────────────────────────────────────────

def test_real_config_loads_with_documented_shape():
    cfg = load_intake_config(_raw())
    assert isinstance(cfg, IntakeConfig)
    assert cfg.runtime.heartbeat_s > 0
    q = cfg.quality
    assert 0 <= q.luma_lo < q.luma_hi <= 255
    assert 0 <= q.clip_frac_max <= 1
    assert isinstance(q.clip_lo, int) and isinstance(q.clip_hi, int) and q.clip_lo < q.clip_hi
    assert isinstance(q.fft_max_side, int) and isinstance(q.diff_max_side, int)
    p = cfg.parallax
    assert isinstance(p.grid_side, int) and p.grid_side >= 2
    assert 0 < p.process_scale <= 1
    assert p.parallax_quantum_px > 0 and p.witness_min_parallax_px > 0
    assert isinstance(p.min_tracks, int) and isinstance(p.lk_win, int) and isinstance(p.lk_levels, int)
    assert p.warn_static_disp_px < p.warn_rotation_min_disp_px
    assert isinstance(p.warn_min_run_frames, int)
    assert 0 <= p.keyframe_band_frac < 1 and 0 < p.parallax_quantile <= 1
    assert isinstance(p.reference_max_eval, int) and p.reference_max_eval >= 1
    assert 0 < p.reference_tol < 1
    c = cfg.content
    assert isinstance(c.enabled, bool) and c.backend
    assert isinstance(c.batch, int) and c.batch >= 1 and isinstance(c.max_tokens, int)
    assert c.sam3_scope in SAM3_SCOPES
    assert set(c.exclusion_classes) <= set(CONTENT_CLASSES)
    assert set(c.weight_classes) <= set(CONTENT_CLASSES)
    assert not set(c.exclusion_classes) & set(c.weight_classes)
    assert set(c.prompts) >= set(c.exclusion_classes)
    assert all(isinstance(v, tuple) and v for v in c.prompts.values())
    # the plan's §4-F1 vocabulary
    assert set(c.exclusion_classes) == {"dynamic", "occluder"}
    assert set(c.weight_classes) == {"reflective", "low_info"}
    # the SAM3 session length is the segmentation stage's own key
    raw = _raw()
    assert c.sam3_batch == raw["models"]["segmentation"]["batch_size"]


def test_shipped_values_and_their_provenance_tags():
    """The shipped values and honest tags: sam3_scope cites the plan (keyframes
    AND witness frames), not a user statement; the parallax keys are BOUNDs."""
    cfg = load_intake_config(_raw())
    p = cfg.parallax
    assert p.keyframe_band_frac == 0.25 and p.parallax_quantile == 0.9
    assert p.reference_max_eval == 200 and p.reference_tol == 1.0e-12
    # USER 2026-09-28: the EXCLUSION pass stays on the flagged ranges ("all" measured 31 h on pccr)
    assert cfg.content.sam3_scope == "flagged_ranges"
    lines = (SERVER / "config.yaml").read_text().splitlines()
    keys = ("sam3_scope:", "parallax_quantile:", "keyframe_band_frac:", "reference_max_eval:",
            "reference_tol:")
    tag = {}
    for ln in lines:
        for key in keys:
            if ln.strip().startswith(key):
                tag[key] = ln
    assert "USER 2026-09-28" in tag["sam3_scope:"]          # the exclusion pass stays on the flagged ranges
    for key in keys[1:-1]:
        assert "BOUND" in tag[key], key
    assert "declared convergence tolerance" in tag["reference_tol:"]
    # the removed switches are gone from the shipped file
    for gone in ("step_null:", "reference_ftol:", "rotation_floor_factor:",
                 "rigidity_confidence:", "rigidity_bootstrap:", "refine_max_iter:",
                 "refine_eps_px:"):
        assert not any(ln.strip().startswith(gone) for ln in lines), gone


def test_removed_and_unknown_keys_fail_the_load():
    """A key nothing reads cannot configure anything: step_null (removed with
    the per-step accumulation) fails naming why, and so does any key a
    section's dataclass does not declare."""
    raw = _raw()
    assert "step_null" in REMOVED_KEYS["parallax"]
    _expect(_set(raw, "parallax", "step_null", "warp_twin"),
            r"intake\.parallax\.step_null' is no longer a parameter")
    for sec in ("runtime", "quality", "parallax", "content"):
        _expect(_set(raw, sec, "banana", 1), rf"intake\.{sec}\.banana' is not a parameter")


def test_reconstruction_simple_frame_selection_is_a_known_selector():
    # USER 2026-09-28: parallax_lk (F1, eebebe8) for the F0-F2 validation run on pccr
    # 2026-08-24; 'motion' stays selectable. The key must name a selector map_worker knows.
    from workers.map_worker import FRAME_SELECTIONS
    raw = _raw()
    assert raw["reconstruction"]["simple"]["frame_selection"] in FRAME_SELECTIONS


def test_config_is_frozen():
    cfg = load_intake_config(_raw())
    with pytest.raises(Exception):
        cfg.quality.luma_lo = 1.0          # type: ignore[misc]


# ── missing keys / sections name themselves ──────────────────────────────

def test_missing_section_fails():
    _expect({}, "intake")
    _expect({"intake": "nope"}, "intake")
    _expect({"intake": None}, "intake")
    # raw=None is the "read the server config" sentinel, not an empty config
    assert isinstance(load_intake_config(None), IntakeConfig)


@pytest.mark.parametrize("path", [
    ("runtime", "heartbeat_s"),
    ("quality", "luma_lo"), ("quality", "luma_hi"), ("quality", "clip_frac_max"),
    ("quality", "clip_lo"), ("quality", "clip_hi"), ("quality", "fft_max_side"),
    ("quality", "diff_max_side"),
    ("parallax", "grid_side"), ("parallax", "process_scale"), ("parallax", "ransac_px"),
    ("parallax", "parallax_quantum_px"), ("parallax", "witness_min_parallax_px"),
    ("parallax", "min_tracks"), ("parallax", "lk_win"), ("parallax", "lk_levels"),
    ("parallax", "fb_max_px"), ("parallax", "warn_static_disp_px"),
    ("parallax", "warn_rotation_min_disp_px"), ("parallax", "warn_min_run_frames"),
    ("parallax", "parallax_quantile"), ("parallax", "keyframe_band_frac"),
    ("parallax", "reference_max_eval"),
    ("parallax", "reference_tol"),
    ("parallax", "focal_probe_frames"), ("parallax", "focal_probe_res"),
    ("parallax", "focal_probe_model"),
    ("content", "enabled"), ("content", "backend"), ("content", "batch"),
    ("content", "max_tokens"), ("content", "exclusion_classes"), ("content", "weight_classes"),
    ("content", "sam3_scope"), ("content", "prompts"),
])
def test_missing_key_names_the_key(path):
    raw = _raw()
    sec, key = path
    del raw["intake"][sec][key]
    _expect(raw, rf"intake\.{sec}\.{key}")


def test_missing_subsection_names_it():
    for sec in ("runtime", "quality", "parallax", "content"):
        raw = _raw()
        del raw["intake"][sec]
        _expect(raw, rf"intake\.\.{sec}|intake\.{sec}")


# ── types, ranges, enums ─────────────────────────────────────────────────

def _set(raw, sec, key, value):
    bad = copy.deepcopy(raw)
    bad["intake"][sec][key] = value
    return bad


def test_type_errors_name_the_key():
    raw = _raw()
    _expect(_set(raw, "content", "enabled", "yes"), "content.enabled")
    _expect(_set(raw, "quality", "luma_lo", "40"), "quality.luma_lo")
    _expect(_set(raw, "quality", "clip_lo", 5.5), "quality.clip_lo")
    _expect(_set(raw, "parallax", "grid_side", True), "parallax.grid_side")
    _expect(_set(raw, "content", "backend", ""), "content.backend")
    _expect(_set(raw, "content", "prompts", ["person"]), "content.prompts")


def test_range_errors_name_the_key():
    raw = _raw()
    _expect(_set(raw, "runtime", "heartbeat_s", 0), "runtime.heartbeat_s")
    _expect(_set(raw, "quality", "luma_lo", -1), "quality.luma_lo")
    _expect(_set(raw, "quality", "luma_hi", 300), "quality.luma_hi")
    _expect(_set(raw, "quality", "clip_frac_max", 1.5), "quality.clip_frac_max")
    _expect(_set(raw, "quality", "clip_hi", 256), "quality.clip_hi")
    _expect(_set(raw, "quality", "fft_max_side", 0), "quality.fft_max_side")
    _expect(_set(raw, "parallax", "process_scale", 0), "parallax.process_scale")
    _expect(_set(raw, "parallax", "process_scale", 1.5), "parallax.process_scale")
    _expect(_set(raw, "parallax", "grid_side", 1), "parallax.grid_side")
    _expect(_set(raw, "parallax", "ransac_px", 0), "parallax.ransac_px")
    _expect(_set(raw, "parallax", "parallax_quantum_px", 0.0), "parallax.parallax_quantum_px")
    _expect(_set(raw, "parallax", "min_tracks", 3), "parallax.min_tracks")
    _expect(_set(raw, "parallax", "lk_levels", -1), "parallax.lk_levels")
    _expect(_set(raw, "parallax", "warn_min_run_frames", 0), "parallax.warn_min_run_frames")
    _expect(_set(raw, "parallax", "keyframe_band_frac", -0.1), "parallax.keyframe_band_frac")
    _expect(_set(raw, "parallax", "keyframe_band_frac", 1.0), "parallax.keyframe_band_frac")
    _expect(_set(raw, "parallax", "parallax_quantile", 0.0), "parallax.parallax_quantile")
    _expect(_set(raw, "parallax", "parallax_quantile", 1.5), "parallax.parallax_quantile")
    _expect(_set(raw, "parallax", "reference_max_eval", 0), "parallax.reference_max_eval")
    _expect(_set(raw, "parallax", "reference_tol", 0.0), "parallax.reference_tol")
    _expect(_set(raw, "parallax", "reference_tol", 1.5), "parallax.reference_tol")
    _expect(_set(raw, "content", "batch", 0), "content.batch")
    _expect(_set(raw, "content", "max_tokens", 0), "content.max_tokens")


def test_cross_key_consistency():
    raw = _raw()
    # exposure band must be non-empty
    bad = _set(raw, "quality", "luma_lo", raw["intake"]["quality"]["luma_hi"])
    _expect(bad, "quality.luma_lo")
    bad = _set(raw, "quality", "clip_lo", raw["intake"]["quality"]["clip_hi"])
    _expect(bad, "quality.clip_lo")
    # still vs pure-rotation displacement bars cannot overlap
    bad = _set(raw, "parallax", "warn_static_disp_px",
               raw["intake"]["parallax"]["warn_rotation_min_disp_px"])
    _expect(bad, "warn_static_disp_px")


def test_enum_errors_name_the_key():
    raw = _raw()
    _expect(_set(raw, "content", "sam3_scope", "banana"), "content.sam3_scope")
    _expect(_set(raw, "content", "exclusion_classes", ["dynamic", "banana"]),
            r"content.exclusion_classes\[1\]")
    _expect(_set(raw, "content", "weight_classes", ["banana"]), r"content.weight_classes\[0\]")
    _expect(_set(raw, "content", "exclusion_classes", ["dynamic", 3]),
            r"content.exclusion_classes\[1\]")


def test_class_overlap_and_duplicates_fail():
    raw = _raw()
    _expect(_set(raw, "content", "weight_classes", ["reflective", "dynamic"]), "disjoint")
    _expect(_set(raw, "content", "exclusion_classes", ["dynamic", "dynamic"]), "twice")
    _expect(_set(raw, "content", "weight_classes", ["low_info", "low_info"]), "twice")


def test_prompts_cover_every_exclusion_class():
    raw = _raw()
    bad = copy.deepcopy(raw)
    del bad["intake"]["content"]["prompts"]["occluder"]
    _expect(bad, r"prompts.*occluder")
    bad = copy.deepcopy(raw)
    bad["intake"]["content"]["prompts"]["banana"] = ["x"]
    _expect(bad, "content.prompts.banana")
    bad = copy.deepcopy(raw)
    bad["intake"]["content"]["prompts"]["dynamic"] = []
    _expect(bad, "content.prompts.dynamic")
    bad = copy.deepcopy(raw)
    bad["intake"]["content"]["prompts"]["dynamic"] = ["person", ""]
    _expect(bad, r"content.prompts.dynamic\[1\]")
    # a weight class may carry prompts too (they are strings, not a gate)
    ok = copy.deepcopy(raw)
    ok["intake"]["content"]["prompts"]["reflective"] = ["glass", "mirror"]
    assert load_intake_config(ok).content.prompts["reflective"] == ("glass", "mirror")
    # classes may be empty lists; prompts then owe nothing
    ok = _set(raw, "content", "exclusion_classes", [])
    assert load_intake_config(ok).content.exclusion_classes == ()


def test_content_backend_cross_checked_against_semantic_section():
    raw = _raw()
    assert "backends" in raw["semantic"]
    backend = raw["intake"]["content"]["backend"]
    cap = raw["semantic"]["backends"][backend]["max_images_per_prompt"]
    _expect(_set(raw, "content", "batch", cap + 1), "max_images_per_prompt")
    assert load_intake_config(_set(raw, "content", "batch", cap)).content.batch == cap
    _expect(_set(raw, "content", "backend", "no_such_backend"), "content.backend")
    # without a semantic section there is nothing to cross-check against
    alone = copy.deepcopy(raw)
    del alone["semantic"]
    assert load_intake_config(_set(alone, "content", "batch", 1000)).content.batch == 1000


def test_sam3_batch_comes_from_the_segmentation_section():
    raw = _raw()
    for bad in (0, 2.5, True, "250"):
        bad_raw = copy.deepcopy(raw)
        bad_raw["models"]["segmentation"]["batch_size"] = bad
        _expect(bad_raw, r"models\.segmentation\.batch_size")
    missing = copy.deepcopy(raw)
    del missing["models"]["segmentation"]["batch_size"]
    _expect(missing, r"models\.segmentation\.batch_size")
    no_models = copy.deepcopy(raw)
    del no_models["models"]
    _expect(no_models, r"models\.segmentation\.batch_size")
    ok = copy.deepcopy(raw)
    ok["models"]["segmentation"]["batch_size"] = 32
    assert load_intake_config(ok).content.sam3_batch == 32


def test_numbers_come_back_typed():
    raw = _raw()
    raw["intake"]["quality"]["luma_lo"] = 40          # int in YAML → float field
    raw["intake"]["parallax"]["grid_side"] = 16.0     # integral float → int field
    cfg = load_intake_config(raw)
    assert isinstance(cfg.quality.luma_lo, float) and cfg.quality.luma_lo == 40.0
    assert isinstance(cfg.parallax.grid_side, int) and cfg.parallax.grid_side == 16


# ── static literal scan (the test_precision_config.py pattern) ───────────
_FLOAT_WHITELIST = {0.0, 1.0, -1.0, 2.0, 0.5, 1e-9, 1e-6, 1e-12, 100.0, 1000.0,
                    0.6744897501960817,
                    1e9}            # bytes -> GB in a log line (intake/walk.py), a unit not a decision


def _float_literals(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value not in _FLOAT_WHITELIST:
                yield node.lineno, node.value


def test_no_decision_literals_outside_intake_config():
    offenders = []
    for py in sorted(INTAKE.rglob("*.py")):
        if py.name == "config.py":
            continue
        for lineno, val in _float_literals(py):
            offenders.append(f"{py.relative_to(SERVER)}:{lineno} = {val}")
    assert not offenders, ("decision-looking float literals outside intake/config.py: "
                           f"{offenders}")

