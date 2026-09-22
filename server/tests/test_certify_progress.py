"""USER 2026-09-21, watching a 3h34 reconstruction: "en el ui el avance de
certificacion sigue al 5%" — with the correction already applied and epoch 1
already published.

The cause was not a mis-calibrated bar. `workers/certify_worker.py` called
`send_progress` exactly twice: 5 on entry and 100 on exit. Everything in
between — instance loops, the VLM classification, the depth correction, the
floor, the transaction, the consolidation, the octree, the acta: 40 minutes —
reported nothing. And the correction's OWN progress was thrown away at the
`stage_transaction` call, which passed `progress=None`.

These tests pin that the advance is wired end to end and monotone.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def test_certify_accepts_and_uses_a_progress_callback():
    src = (ROOT / "reconstruction" / "certify" / "run.py").read_text()
    assert "progress: Optional[Callable[[int, str], None]] = None" in src, \
        "_certify_session must take a progress callback"
    assert src.count("_pc(") >= 5, \
        "certify must report the advance of its stages, not only its ends"


def test_the_corrections_own_progress_is_not_discarded():
    """`progress=None` at the transaction was why the one stage that DOES
    report its advance (correction/apply.py, steps 55..85) reported it to
    nobody."""
    src = (ROOT / "reconstruction" / "certify" / "run.py").read_text()
    assert "floor_npz=None, log=log, progress=None," not in src
    assert "run_correction(session_dir, log=log, cfg=ccfg,\n" in src \
        and "progress=lambda" in src, "the correction must be given a reporter"


def test_the_worker_sends_more_than_its_two_ends():
    src = (ROOT / "workers" / "certify_worker.py").read_text()
    assert "progress=_progress" in src, \
        "the worker must hand certify a reporter, not just bracket it"
    # and it must still declare the end
    assert "send_progress(100" in src


def test_the_reported_percentages_are_monotone_and_bounded():
    """A bar that goes backwards is worse than one that does not move."""
    src = (ROOT / "reconstruction" / "certify" / "run.py").read_text()
    lits = [int(m) for m in re.findall(r"_pc\((\d+),", src)]
    assert lits, "no fixed milestones found"
    assert lits == sorted(lits), f"the milestones go backwards: {lits}"
    assert all(0 < v <= 100 for v in lits), lits
    # the bands the lambdas map into must stay inside the milestones
    for base, span in re.findall(r"_pc\((\d+) \+ int\(p \* ([0-9.]+)\)", src):
        assert int(base) + float(span) * 100 <= 100.5, (base, span)


def test_the_correction_reports_its_two_stages():
    src = (ROOT / "correction" / "visit_drift_run.py").read_text()
    assert "progress: Optional[Callable[[int, str], None]] = None" in src
    assert src.count("_pc(") >= 3, \
        "the depth and the floor must each say when they start"
