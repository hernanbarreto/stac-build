Figures of the 2026-10-23 precision audit — SYNTHETIC scenes of the test suite (no GPU, no session):
  fig_f5.py     01 F5 poses: absolute priors vs odometry priors on a drifted walk with one loop (test_precision_refine scene)
  fig_fusion.py 02 owner fusion (layers → one surface) and free-space carving of a 3-keyframe floater (test_owner_fusion scenes)
  fig_bend.py   03 bend families on an oblique plane, the judged choice, the published cloud (test_depth_on_f5_compute session)
  fig_rs.py     04 rolling shutter: residual vs row before/after, the judged grid search (test_rolling_shutter scene)
  e2e_synthetic.py  the COMPLETE synthetic run: rendered room + revisit, Omega's errors injected, the real F5 → F6 → cloud →
                    masks → projection; e2e_synthetic/{metrics.json, figures/, run.log}; stages cached in e2e_synthetic/cache
Run from the server env (pycolmap 4 for 01 and 04): python analysis/2026-10-23_precision_audit/fig_<x>.py → figures/<x>.png
Numbers for the eye, never a verdict (claude_stac.txt rule 3).
