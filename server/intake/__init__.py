"""Intake (claude_stac.txt §3) — the frames enter the pipeline here. F1 ships
I0–I1; I3 (DA3 windows, metric walk) and I4 (chunk plan) arrive with F2.

    quality.py   I0  per-frame quality FEATURES (no percentile rejection)
    parallax.py  I1  keyframes by measured parallax + rotation quantum, witness frames
    focal.py     I1's instrument: the session camera K (DA3 focal probe)
    walk.py      I3  DA3 windows → the metric walk, the SALAD revisit reference
    run.py       the I0–I1 command: ``python -m intake.run --session <dir>``
    config.py    typed, validated ``intake:`` section

I2 (VLM content tags + SAM3 exclusion masks, ``intake/content.py``) was REMOVED
(USER 2026-10-08, docs/plan_determinismo.md point 75: "ya no corre VLM ahí"); the
VLM runs once, as the stage after the cloud stage.

No decision literal outside config.yaml (``tests/test_intake_config.py`` scans
this package). The VLM fixes no parameter.
"""
