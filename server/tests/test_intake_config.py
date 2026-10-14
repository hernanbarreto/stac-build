"""intake: the real config.yaml loads; a missing / mistyped / out-of-range key
fails at load naming it; a leftover section of the removed I2 (``intake.content``,
USER 2026-10-08, plan point 75) fails naming why; no decision literal lives in the
intake package outside its config.py (static scan, the test_precision_config.py
pattern)."""

import ast
import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intake.config import (REMOVED_KEYS, REMOVED_SECTIONS,                 # noqa: E402
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
    # I2 is gone (USER 2026-10-08): no content section in the typed configuration, none in
    # the shipped file — only the one-line note where it was
    assert not hasattr(cfg, "content")
    assert "content" not in _raw()["intake"]
    lines = (SERVER / "config.yaml").read_text().splitlines()
    assert any("content:" in ln and "REMOVED" in ln and "2026-10-08" in ln for ln in lines)


def test_shipped_values_and_their_provenance_tags():
    """The shipped values and honest tags: the parallax keys are BOUNDs."""
    cfg = load_intake_config(_raw())
    p = cfg.parallax
    assert p.keyframe_band_frac == 0.25 and p.parallax_quantile == 0.9
    assert p.reference_max_eval == 200 and p.reference_tol == 1.0e-12
    lines = (SERVER / "config.yaml").read_text().splitlines()
    keys = ("parallax_quantile:", "keyframe_band_frac:", "reference_max_eval:",
            "reference_tol:")
    tag = {}
    for ln in lines:
        for key in keys:
            if ln.strip().startswith(key):
                tag[key] = ln
    for key in keys[:-1]:
        assert "BOUND" in tag[key], key
    assert "declared convergence tolerance" in tag["reference_tol:"]
    # the removed switches are gone from the shipped file — I2's keys with them
    for gone in ("step_null:", "reference_ftol:", "rotation_floor_factor:",
                 "rigidity_confidence:", "rigidity_bootstrap:", "refine_max_iter:",
                 "refine_eps_px:", "sam3_scope:", "exclusion_classes:", "weight_classes:"):
        assert not any(ln.strip().startswith(gone) for ln in lines), gone


def test_removed_and_unknown_keys_fail_the_load():
    """A key nothing reads cannot configure anything: step_null (removed with
    the per-step accumulation) fails naming why, and so does any key a
    section's dataclass does not declare."""
    raw = _raw()
    assert "step_null" in REMOVED_KEYS["parallax"]
    _expect(_set(raw, "parallax", "step_null", "warp_twin"),
            r"intake\.parallax\.step_null' is no longer a parameter")
    for sec in ("runtime", "quality", "parallax"):
        _expect(_set(raw, sec, "banana", 1), rf"intake\.{sec}\.banana' is not a parameter")


def test_a_leftover_content_section_fails_the_load_naming_why():
    """USER 2026-10-08 (plan point 75): I2 is eliminated; a leftover ``intake.content`` in
    server/config.yaml (or any configuration handed to the loader) fails naming the section
    and the decision — it would only pretend to configure a stage that does not run."""
    assert "content" in REMOVED_SECTIONS and "2026-10-08" in REMOVED_SECTIONS["content"]
    raw = _raw()
    bad = copy.deepcopy(raw)
    bad["intake"]["content"] = {"enabled": False}
    with pytest.raises(IntakeConfigError, match=r"intake\.content' is no longer a section") as e:
        load_intake_config(bad)
    assert "I2" in str(e.value) and "2026-10-08" in str(e.value)
    bad["intake"]["content"] = {}                 # even an empty one
    _expect(bad, r"intake\.content' is no longer a section")
    # no I2 code is left in the intake package: no module, no import of it, no tagger
    assert not (INTAKE / "content.py").exists()
    for py in sorted(INTAKE.glob("*.py")):
        src = py.read_text()
        assert "from intake import content" not in src and "intake.content import" not in src, py.name
        assert "ContentConfig" not in src and "QwenTagger" not in src, py.name


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
])
def test_missing_key_names_the_key(path):
    raw = _raw()
    sec, key = path
    del raw["intake"][sec][key]
    _expect(raw, rf"intake\.{sec}\.{key}")


def test_missing_subsection_names_it():
    for sec in ("runtime", "quality", "parallax"):
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
    _expect(_set(raw, "quality", "luma_lo", "40"), "quality.luma_lo")
    _expect(_set(raw, "quality", "clip_lo", 5.5), "quality.clip_lo")
    _expect(_set(raw, "parallax", "grid_side", True), "parallax.grid_side")
    _expect(_set(raw, "parallax", "focal_probe_model", ""), "parallax.focal_probe_model")


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
    _expect(_set(raw, "parallax", "vram_margin_frac", 0.95), "parallax.vram_margin_frac")


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
