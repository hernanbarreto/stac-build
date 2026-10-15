"""Evaluation battery (claude_stac.txt, 2026-10-14): measures the production chain against
hand-measured references, never changes it.

Phases: 0 references (distances + planes anchored on keyframe pixels) · 1 error per
distance · 2 error curves · 3 contribution of each stage (variants) · 4 cost · 5 calibrated
confidence · 6 repeatability and determinism · 7 cost–error frontier · 8 CLI + API + panel.

Every tolerance, price and path lives in ``config.yaml`` under ``evaluation:``
(:mod:`evaluation.config`). Results of a run are JSON, CSV and PNG files under the run's
directory; nothing here declares a result good or bad.
"""
