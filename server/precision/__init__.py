"""Precision pipeline (claude_stac.txt §3) — one stage per module.

    camera.py      F0  session camera model + exact grid-to-native-pixel mappings
    gauge.py       F2  continuous metric gauge along the walk
    omega_probe.py F3  strict omega load report + resolution probe
    tracks.py      F4  native-pixel subpixel tracks with a held-out split
    refine.py      F5  joint pose + session camera refinement, witness localisation
    depth_sweep.py F6  prior-guided native-resolution plane sweep (tiers)
    depth_colmap.py F6 COLMAP PatchMatch reference (comparison only)
    confidence.py  F6  per-session confidence calibration
    fuse.py        F7  witness-based fusion, provenance v2, rejected set
    provenance.py  F7  the single provenance schema and its loader (v1 + v2)
    certify.py     F8  visit-drift verify, known dimensions, precision_report
    runner.py      F9  orchestration, resume per stage, cancel, heartbeat
    config.py      typed, validated ``reconstruction.precision`` section

Rules that hold in every module (claude_stac.txt §0, §5): no decision literal
outside config.yaml (``tests/test_precision_config.py`` scans this package);
every gate is a measurement (held-out + bootstrap), advisory; every artifact
is stamped ``geometry_epoch`` + ``camera_epoch``; nothing is discarded without
a ``reason``; the user validates results visually — this code never writes a
quality verdict.
"""
