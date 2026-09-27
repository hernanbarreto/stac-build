"""Intake (claude_stac.txt §3) — the frames enter the pipeline here. F1 ships
I0–I2; I3 (DA3 windows, metric walk) and I4 (chunk plan) arrive with F2.

    quality.py   I0  per-frame quality FEATURES (no percentile rejection)
    parallax.py  I1  keyframes by measured parallax + witness frames
    content.py   I2  VLM content tags + SAM3 exclusion masks (dynamic/occluder)
    run.py       the I0–I2 command: ``python -m intake.run --session <dir>``
    config.py    typed, validated ``intake:`` section

No decision literal outside config.yaml (``tests/test_intake_config.py`` scans
this package). Every VLM output is ``vlm_proposed``; the VLM fixes no parameter.
"""
