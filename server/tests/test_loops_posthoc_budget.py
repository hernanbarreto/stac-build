"""USER 2026-09-21: "nada que no sirve sirve, porque come computo y tiempo".

Measured on the full pccr run that day, `copy_scale_rows` spent 56 ICP
alignments over a 22.7 M point cloud to produce ONE scale row:

    dark_ceiling#187            19 evaluations -> 0 rows   (later 42 of 56)
    exposed_ceiling_ducts#116    8             -> 0
    computer_monitor#169         1             -> 1 row    (a COMPACT object)

7 of the 56 were a pair already measured, with an identical residual to four
decimals, and 4 ran the ICP with one side holding NO points at all. An
extended surface pairs its own clusters across every keyframe that saw it, so
the work grows as n² and the 42nd answer is the 1st answer.

These tests pin the three guards, all of them BEFORE the work: never the same
pair twice, never more than `max_pairs_per_instance` of one object, never an
alignment against an empty side.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.certify import loops_posthoc as lp  # noqa: E402


class _Session:
    """Two clusters of one instance, far apart along x."""
    def __init__(self, tmp_path, n=400):
        self.output_dir = tmp_path
        a = np.random.default_rng(0).normal(0, .1, (n, 3))
        b = a + np.array([9.0, 0, 0])
        self.xyz = np.vstack([a, b])
        self.ks = np.concatenate([np.full(n, 10), np.full(n, 200)])
        import json
        (tmp_path / "segmentation_result.json").write_text(json.dumps(
            {"instances": [{"instance_id": 7, "id": 7, "label": "ceiling",
                            "globalIndices": list(range(2 * n))}]}))


def _scfg():
    return types.SimpleNamespace(min_copy_points=10, icp_iters=3, icp_trim=0.8,
                                 max_copy_residual_m=0.05)


def _cand(i, j, iid=7):
    return {"instance_id": iid, "label": "ceiling", "i": i, "j": j,
            "verdict": "ambiguous", "kind": "instance"}


def _patch_indices(monkeypatch, session):
    def _idx(sess, inst, i, j, window_kf):
        gi = np.arange(len(sess.xyz))
        ks = sess.ks
        return (gi[np.abs(ks - i) <= int(window_kf)],
                gi[np.abs(ks - j) <= int(window_kf)], ks)
    monkeypatch.setattr(lp, "_copy_indices", _idx)


def test_the_same_pair_is_never_measured_twice(tmp_path, monkeypatch):
    s = _Session(tmp_path)
    _patch_indices(monkeypatch, s)
    calls = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda a, b, cfg, seed=0: calls.append(1) or None)
    lp.copy_scale_rows(s, [_cand(10, 200), _cand(10, 200), _cand(200, 10)],
                       _scfg(), 15, 10, log=lambda m: None)
    assert len(calls) == 1, "the pair (and its mirror) was measured more than once"


def test_one_object_cannot_spend_the_whole_budget(tmp_path, monkeypatch):
    s = _Session(tmp_path)
    _patch_indices(monkeypatch, s)
    calls = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda a, b, cfg, seed=0: calls.append(1) or None)
    cands = [_cand(10, 200 + k) for k in range(12)]     # 12 distinct pairs
    lp.copy_scale_rows(s, cands, _scfg(), 15, 3, log=lambda m: None)
    assert len(calls) == 3, f"the cap was not honoured: {len(calls)} ICP runs"


def test_an_empty_side_never_reaches_the_icp(tmp_path, monkeypatch):
    """It used to align against nothing and discard the answer afterwards —
    the work is skipped, not undone."""
    s = _Session(tmp_path)
    monkeypatch.setattr(lp, "_copy_indices",
                        lambda *a, **k: (np.arange(10), np.array([], int), s.ks))
    calls = []
    monkeypatch.setattr(lp, "measure_copy",
                        lambda a, b, cfg, seed=0: calls.append(1) or None)
    out = lp.copy_scale_rows(s, [_cand(10, 200)], _scfg(), 15, 10,
                             log=lambda m: None)
    assert not calls, "the ICP ran against an empty copy"
    assert "no points" in out[0]["reason"], out[0]


def test_the_cap_is_configured_not_hardcoded():
    from reconstruction.loops.config import load_loops_config
    c = load_loops_config()
    assert c.certify.visit_loops.max_pairs_per_instance >= 1
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "certify" / "loops_posthoc.py").read_text()
    assert "_MAX_PAIRS_PER_INSTANCE" not in src, "the bound must come from config"
