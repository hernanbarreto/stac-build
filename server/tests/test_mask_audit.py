"""The audit of the cloud against the masks (USER 2026-09-15).

"no hay nada que cortar, es la auditoría de tu propia nube contra el ground
truth de la máscara, es eso."

What is asserted here is the RULE, not the projection arithmetic: where an
instance's mass falls relative to its mask is the verdict, how many clusters it
forms is not, and nothing is ever removed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.mask_filter import MaskAudit                      # noqa: E402


class _Audit:
    """MaskAudit's pure rules without a session behind them."""

    def __init__(self, gap=1):
        self.cfg = {"visit_gap_kf": gap}

    _visits = MaskAudit._visits


def _verdict(visits):
    """The classification MaskAudit.audit applies to its per-visit tallies."""
    off = [v for v in visits if v["seen"] and v["off_mask"] > v["on_mask"]]
    on = [v for v in visits if v["seen"] and v["on_mask"] >= v["off_mask"]]
    return "correct" if not off else "drift_duplicate" if on else "unsupported"


def _v(on, off, seen=None):
    return {"seen": seen if seen is not None else on + off, "on_mask": on, "off_mask": off}


# ── where the mass falls is the verdict ──────────────────────────────────

def test_all_the_mass_on_the_mask_is_correct():
    assert _verdict([_v(900, 20), _v(700, 30)]) == "correct"


def test_n_islands_on_the_mask_are_still_correct():
    """Confidence filtering cuts a surface into islands. USER: 'la nube sale
    cortada y genera dos islas del objeto, pero tiene dos concentraciones
    dentro de la máscara, está correcta, solo que incompleta, pero correcta.'
    The count of concentrations must never be the verdict."""
    assert _verdict([_v(400, 10), _v(350, 8), _v(300, 5), _v(120, 3)]) == "correct"


def test_mass_off_the_mask_in_one_visit_and_on_it_in_another_is_a_duplicate():
    """The case that matters: on the mask where it was first seen, off it in
    the frames of the revisit."""
    assert _verdict([_v(900, 30), _v(40, 800)]) == "drift_duplicate"


def test_mass_off_the_mask_in_every_visit_is_junk_not_a_duplicate():
    """Attached by proximity, supported by no mask anywhere. It is declared,
    never corrected — and never cut."""
    assert _verdict([_v(10, 900), _v(5, 700)]) == "unsupported"


def test_a_visit_that_saw_nothing_does_not_vote():
    assert _verdict([_v(900, 20), _v(0, 0, seen=0)]) == "correct"


def test_the_verdict_needs_no_threshold_only_a_comparison():
    """One observation either side of the tie flips it, and the tie itself
    counts as placed. There is no percentage to tune."""
    assert _verdict([_v(500, 500)]) == "correct"
    assert _verdict([_v(499, 501)]) == "unsupported"


# ── a visit ends when the object stops being seen ────────────────────────

def test_consecutive_keyframes_are_one_visit():
    assert _Audit()._visits([3, 4, 5, 6]) == [[3, 4, 5, 6]]


def test_a_single_missing_keyframe_opens_a_new_visit():
    """USER 2026-09-14: 'en un kf deja de verse, y luego se vuelve a ver,
    independientemente de los kf que pasaron sin verse, mínimo 1, pero nada
    más.' A larger gap would assert a continuity nothing observed."""
    assert _Audit()._visits([3, 4, 5, 7, 8]) == [[3, 4, 5], [7, 8]]
    assert _Audit()._visits([3, 4, 60, 61]) == [[3, 4], [60, 61]]


def test_the_frames_are_ordered_before_they_are_grouped():
    assert _Audit()._visits([8, 3, 4, 7, 5]) == [[3, 4, 5], [7, 8]]


# ── nothing is cut, anywhere ─────────────────────────────────────────────

def test_the_audit_has_no_way_to_remove_a_point():
    """The guarantee, enforced at source: the module that used to delete points
    no longer carries a knob or a code path that can. If this fails, someone
    reintroduced trimming."""
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "mask_filter.py").read_text()
    for gone in ("min_inside_frac", "max_drop_frac", "min_votes",
                 "dropped", "outlier", "def judge"):
        assert gone not in src, f"{gone!r} is back in mask_filter.py"
    assert "def audit" in src


def test_an_instance_is_separated_only_when_SAM3_says_it_holds_two_objects():
    """The gate may still SAY split, but geometry alone never cuts: on pccr its
    rule (separation > an invented multiple of an invented drift floor) carved
    the floor into four instances and the ceiling into six. The one admissible
    ground is SAM3's own count of mask object ids under that instance — USER
    2026-09-15, "no hay que desconfiar tanto de SAM3"."""
    src = (Path(__file__).resolve().parents[1]
           / "reconstruction" / "loops" / "instance_loops.py").read_text()
    body = src.split("def detect_instance_loops")[-1]
    assert "split_declined" in body, "the declined path is gone"
    assert "oids_of" in body, "the split no longer consults SAM3's object ids"
    # the separation must sit behind the multi-oid condition, never alone
    guarded = body.split("many = len(oids_of")[-1]
    assert "split_instance(" in guarded, "the separation moved out from behind the SAM3 check"
    assert "split_instance(" not in body.split("many = len(oids_of")[0], \
        "something separates an instance before SAM3 is consulted"


def test_the_pipeline_does_not_trim_by_the_audit():
    src = (Path(__file__).resolve().parents[1]
           / "segmentation" / "pipeline.py").read_text()
    assert "keep_geo" not in src, "the pipeline trims instances by the audit again"
    assert "mask_filter.audit(" in src


def test_the_removed_config_keys_stay_removed():
    import yaml
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    mf = cfg["segmentation"]["mask_filter"]
    for gone in ("min_inside_frac", "max_drop_frac", "min_votes"):
        assert gone not in mf, f"{gone} is back in segmentation.mask_filter"
    assert mf["visit_gap_kf"] == 1
