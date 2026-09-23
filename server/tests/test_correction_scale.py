"""Scale rule vs the DA3 anchors: a k that contradicts the anchors is
rejected without override, and applied WITH override — recorded in the
ledger."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.ledger import ledger_view                    # noqa: E402
from correction.run import run_objects                       # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def test_scale_gate_rejects_contradicting_k(tmp_path):
    """DA3 confirms the observed (compressed) depth → the solved k=1/c
    contradicts the anchors → rejected."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=False)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "rejected"
    g = next(g for g in rep["gates"] if g["name"] == "scale_vs_da3")
    assert not g["passed"], g
    assert g["worst_anchor"] is not None


def test_scale_gate_passes_when_da3_agrees(tmp_path):
    """DA3 measured the TRUE depth → the k that re-expands the compressed
    geometry IMPROVES the anchor agreement → passes."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=True)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    g = next(g for g in rep["gates"] if g["name"] == "scale_vs_da3")
    assert g["passed"]


def test_override_applies_and_is_recorded(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=False)
    rep = run_objects(scene.output_dir, [1, 2], "operator-x",
                      override_scale_check=True,
                      cfg=make_correction_cfg())
    assert rep["status"] == "applied", rep.get("rejection_reason")
    assert rep["overrides"]["scale_check"]["overridden_by"] == "operator-x"
    rows = ledger_view(scene.output_dir)
    assert rows[-1]["overrides"].get("scale_check"), "ledger must record it"
    # scale_diagnostics regenerated with the per-epoch history
    diag = json.loads(
        (scene.output_dir / "scale_diagnostics.json").read_text())
    assert diag["epochs"][-1]["epoch"] == 1
    assert diag["epochs"][-1]["correction_id"] == rep["correction_id"]


# ── EARN THE RIGHT + rival pricing (USER 2026-09-22) ─────────────────────

def _closures(rows):
    import numpy as _np
    return [{"chunks": [i, j], "log_r": float(_np.log(v))} for i, j, v in rows]


def test_a_solution_that_explains_nothing_is_not_applied():
    """USER 2026-09-22, after test2's epoch 1 came out torn: *"la 1 es un puto
    desastre"*. Its 18 closures contradicted each other (0.90 to 1.18) and the
    free per-chunk solution zigzagged — a -16.7 % and a +19.2 % step between
    ADJACENT chunks, about 2 m of tearing at 10 m. Nothing in the stage asked
    whether the solution explained closures it had not been fitted on."""
    import numpy as np
    from reconstruction.certify.scale_stage import earn_the_right
    r = [0.9864, 1.0224, 0.9622, 1.0067, 0.9968, 1.0747, 0.8954, 1.0673]
    chain = [0.8, 2.4, 4.0, 5.6, 7.2, 8.8, 10.4, 12.0]
    rows = [(1, 3, 1.0043), (4, 6, 1.0323), (2, 4, 1.0029), (0, 7, 1.0317),
            (1, 7, 0.9813), (2, 7, 1.1098), (0, 7, 0.9947), (2, 6, 1.1774),
            (0, 4, 0.9847), (0, 4, 0.9793), (0, 7, 0.9652), (4, 7, 0.9565),
            (0, 1, 0.8978), (3, 6, 1.02), (1, 5, 0.97), (2, 5, 1.05),
            (3, 7, 0.94), (0, 6, 1.08)]
    rep = earn_the_right(np.asarray(r), _closures(rows), np.asarray(chain),
                         0.02, log=lambda m: None)
    assert rep["verdict"] == "none", rep
    assert rep["err_identity"] <= rep["err_free"], \
        "leaving the geometry alone must be the better explanation here"


def test_a_coherent_session_still_corrects_and_prefers_the_ramp():
    """pccr's 13 closures agree on x1.10-1.15 and its r is monotone. The
    correction must SURVIVE the judge — and the ramp (one parameter along the
    walk) explains the held-out closures better than the free per-chunk
    staircase, which is what CLAUDE.md declared as the real fix."""
    import numpy as np
    from reconstruction.certify.scale_stage import earn_the_right
    r = [0.9559, 0.947, 0.9955, 1.0312, 1.0191, 1.0263, 1.0288]
    chain = [1.3, 3.9, 6.5, 9.1, 11.7, 14.3, 16.9]
    rows = [(0, 6, 1.1194), (0, 5, 1.1169), (1, 6, 1.1432), (0, 6, 1.1338),
            (1, 5, 1.1024), (0, 6, 1.1555), (1, 6, 1.0464), (0, 5, 1.0615),
            (2, 6, 1.0142), (0, 6, 1.1003), (1, 6, 1.2184), (2, 5, 0.9580),
            (0, 4, 1.0381)]
    rep = earn_the_right(np.asarray(r), _closures(rows), np.asarray(chain),
                         0.02, log=lambda m: None)
    assert rep["verdict"] in ("free", "ramp"), rep
    assert rep["err_ramp"] < rep["err_identity"], rep
    if rep["verdict"] == "ramp":
        rr = np.asarray(rep["r_ramp"])
        assert rr.max() / rr.min() > 1.05, "the ramp must still correct"
        jump = float(np.exp(np.abs(np.diff(np.log(rr))).max()) - 1.0)
        assert jump < 0.05, f"a ramp cannot tear a seam, got {jump:.1%}"


def test_the_judge_reads_closures_not_fused_rows():
    """The solver fuses every closure of one chunk PAIR into a single row: on
    pccr that turns 13 independent measurements into 7 and leaves nothing to
    hold out. The judge must see the closures."""
    import numpy as np
    from reconstruction.certify.scale_stage import earn_the_right
    rows = [(0, 6, 1.10 + 0.001 * k) for k in range(12)]      # ONE chunk pair
    rep = earn_the_right(np.asarray([1.0] * 7), _closures(rows),
                         np.asarray([0, 3, 6, 9, 12, 15, 18]), 0.02,
                         log=lambda m: None)
    assert rep["rows"] == 12, "twelve closures, not one fused row"


def test_ambiguity_prices_a_closure_instead_of_vetoing_it():
    """USER 2026-09-22: a repeated scene makes closures that LOOK perfect and
    are two different objects (`metal_track_rails#107`: 99 % radial over
    1.12 m, one rail against another). With n rivals the pairing is one of
    n+1 identities, so sigma grows by sqrt(n+1) — it PRICES, it never vetoes,
    and pccr's floor closures keep speaking because they all agree."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from correction.visit_drift import rival_sigma_factor as f
    assert f(0) == 1.0
    assert f(3) == 2.0
    assert f(0) < f(1) < f(10) < f(30)
    assert abs(1 / f(20) ** 2 - 0.0476) < 1e-3, "20 rivals weigh ~1/21"
