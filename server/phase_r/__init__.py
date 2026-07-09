# STAC-Builder — canonical scene-data layer (instance store + deterministic
# geometry), shared by phases 2-6 (classification, findings, QC, spatial Q&A,
# reports).
#
# HISTORY: this package used to host "Phase R" (semantic anchoring — the
# inter-window Sim(3) pose graph, plurality vote, onion detector, fail-safe
# A/B). That machinery was REMOVED on 2026-07-09: the SIMPLE one-pass
# reconstruction has no window seams to anchor, and the anchoring never
# survived its own A/B gate on real scenes. What lives on is its data layer:
#   instance_store.py        — scene_r.db (R3D scene.db schema, extended);
#                              now populated by the SAM3 mask->cloud stage
#   geometry.py              — depth-lift, KNN filter, gravity-aligned OBB
#                              (adapted from R3D) + camera/frame helpers
#   depth_regularization.py  — deterministic plane fit (phase5 tools)
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
