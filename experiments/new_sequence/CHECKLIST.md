# Checklist — the new order, on the copy of bufferStop

USER 2026-11-21: *"arma un plan … un plan es un checklist que vas completando a medida que se terminan las etapas"*.
The ten steps are his, word for word; a box is ticked only when its check has been shown with its numbers.
Test session: `server/projects/bufferStop_orden_nuevo` (frames, intake and video of bufferStop; empty output at the start).
Working data: `/workspace/tmp/new_sequence/run/`. Every step runs the pipeline's own code.

- [x] **1. Intake: como hoy.**
  - [x] The intake of bufferStop is in the copy: frames, 174 keyframes, the plan of 5 cubes.

- [x] **2. Omega por cubo y candado métrico DA3 por cubo: como hoy. No se ata nada.**
  - [x] `CubeRun.infer` and `CubeRun.witness_pass` on the 5 cubes (Omega's two passes) — same layout as the pipeline's
        own run on this video (`14d2195b8931`, 54–90 Omega views per cube).
  - [x] `CubeRun.metric_lock` — DA3 scale per cube: 3.1311, 3.0286, 2.4186, 2.2344, 2.3119.
  - [x] Stopped there: no seam, bridge, graph or placement ran.
  - [x] Shown: 5 cubes in `run/maplong_run/_tmp_results_unaligned`, `metric_lock.json`; `_tmp_results_aligned` and
        `_tmp_results_loop` empty, no `chunk_graph.json`, no `chunk_sim3.json`. 487 s on the card (2026-11-21 03:43 UTC).

- [x] **3. F0: como hoy.**
  - [x] The pipeline's F0 (`python -m precision.camera`): the first camera from Omega's focal lengths. Its input
        file (each keyframe's K in its owner cube — today written by the tie's last method) was written from the
        step-2 cubes with the pipeline's own writer (`step3_f0.py`).
  - [x] Shown: `camera.json` — fx 862.63, fy 838.50, cx 512, cy 288, no lens; Omega's fx spread 1.97 % over the 174
        keyframes (830.31 .. 915.70). The same numbers the pipeline's own run gives on this video.

- [x] **4. F2: dentro de cada cubo.**
  - [x] The pipeline's F2 (`python -m precision.gauge`, unchanged) in each of the 5 cubes, each laid out as a session
        of its own (`cube_sessions.py`: its cameras, its Omega records written by the pipeline's
        `_emit_omega_depth`, its stretch of the walk). The I3 window depth was brought back first with F2's own
        `regenerate_windows` (19 windows, bit-identical to the ones the walk was measured on).
  - [x] Shown, per cube — scale along the cube's walk | cameras moved at most | tolerance (relative sigma) | walk:
        0: x0.9256 .. x1.0887 | 17.52 cm | 0.03647 | 6.149 m
        1: x0.9263 .. x1.0366 | 10.92 cm | 0.04009 | 5.685 m
        2: x0.9651 .. x1.0200 |  3.22 cm | 0.07524 | 3.866 m
        3: x0.9678 .. x1.0683 |  7.33 cm | 0.11837 | 4.188 m
        4: x0.8559 .. x1.1173 | 11.09 cm | 0.02475 | 3.223 m
        (instrument `da3_windows` in all five; each cube is at its own epoch 1.)

- [x] **5. F4: como hoy.**
  - [x] The pipeline's F4 (`python -m precision.tracks`), unchanged. FOUND ON THE WAY: F4's depth-edge filter reads
        Omega's per-keyframe records and is skipped WITHOUT A WORD when they are absent (a first launch on the clean
        session counted 865,333 observations at window 25 where the pipeline counts 847,053 — stopped). Each
        keyframe's record (its owner cube's, written by the pipeline's writer in step 4) was put into the session and
        F4 run again.
  - [x] Shown: 4,467,947 observations, 761,874 tracks (152,756 held out), keyframes 174/174, sigma median 1.07 px —
        the pipeline's own totals on this video, and the same counts at windows 25, 50 and 75. 424 s on the card.

- [x] **6. F5 por cubo, con una sola cámara para todos.**
  - [x] F5's adjustment (`refine_core`: R0, R1, R2 with its judge), unchanged, given the five cubes at once: 302
        cameras (a keyframe as each cube holds it), the landmarks separate per cube — the cubes share no geometry
        (`step6_f5.py`). Each cube starts from its cameras after F2.
  - [x] ONE camera shared by the 5 cubes: fx 895.38, fy 889.62, cx 515.70, cy 288.58, k1 +0.0549, k2 -0.0477,
        p1 +0.00012, p2 -0.00052 (F0: 862.63 / 838.50).
  - [x] Each cube's anchors with the tolerance F2 gave it: 0.03647, 0.04009, 0.07524, 0.11837, 0.02475.
  - [x] Shown: held-out median R0 0.2853 px, R1 0.2701 px (taken), R2 0.2587 px (taken, applied) — the pipeline's F5
        on the whole session: 0.2894 / 0.2699 / 0.2550. Per cube: 0.2067, 0.2415, 0.3562, 0.2950, 0.2484 px; its
        cameras moved (median / most) 5.98 / 22.39, 4.44 / 17.28, 1.86 / 7.95, 3.07 / 10.92, 2.96 / 6.20 cm. Each
        cube's result went into its session through F5's own door (camera.json, epoch 2, refine.json). Not done: the
        localisation of the witness frames (F6 does not read their poses). 164 s on the card.

- [x] **7. Profundidad, por cubo: el factor de cada keyframe ajustado a las correspondencias, y después PointDiT en los bordes.**
  - [x] `bend_coefficients` (factor c0 + c1·u + c2·v) for every keyframe of every cube — by the pipeline's own
        `depth_on_f5.compute` run on each cube's session and stopped where `apply_mono_detail` returns
        (`step7_depth.py`); cube 0: held-out depth error 4.28 % unbent -> 1.02 % with the factor.
  - [x] `apply_mono_detail` (PointDiT-H at the edges, 6 tiles of 512 px a keyframe) for every keyframe of every cube.
  - [x] Shown, per cube — factor per keyframe median (min .. max) | how decided | PointDiT pixels: detail / band /
        mixed (unresolved) | tiles accepted:
        0: 0.9668 (0.9382 .. 0.9922) | 64 verified               | 15.09 M / 6.75 M / 3.02 M (1.94 M) | 364 of 384
        1: 0.9926 (0.9787 .. 1.0380) | 63 verified, 1 pooled     | 14.19 M / 8.85 M / 3.70 M (2.01 M) | 364 of 384
        2: 1.0030 (0.9669 .. 1.0589) | 62 verified, 2 pooled     | 16.03 M / 7.10 M / 2.58 M (1.48 M) | 364 of 384
        3: 0.9973 (0.9456 .. 1.0680) | 60 verified, 4 pooled     | 14.01 M / 5.44 M / 1.74 M (0.99 M) | 364 of 384
        4: 1.0292 (1.0035 .. 1.0515) | 44 verified, 2 pooled     | 10.37 M / 4.95 M / 1.77 M (0.95 M) | 262 of 276
        2638 s on the card (the model is loaded once per cube; 302 keyframes instead of the session's 174; each
        keyframe took 8-12 s — slower than the pipeline's own F6 on this video, not explained).

- [x] **8. Atada del pipeline: costuras, cierres con puentes de Omega y grafo Sim(3) exactamente igual que en el pipeline principal**
  - [x] The cubes the tie reads, written again with what steps 4, 6 and 7 corrected in each (`step8_cubes.py`: the
        camera after F2 + F5, the depth after the factor + PointDiT, the points = that depth from that camera, Omega's
        confidence, 0 where step 7 left no depth): 94.4 - 97.2 % of the pixels Omega gave a confidence keep a depth.
  - [x] The fork's own cube stage from the bridges on, call for call (`step8_tie.py`): `CubeRun.bridges` (32 of the
        43 revisits join two cubes that share no keyframe; one joint Omega pass each), `CubeRun.seams`,
        `CubeRun.closures` + `CubeRun.graph`, `CubeRun.apply`, uncertainty, depth graph (verdict SKIP), outputs,
        `CubeRun.camera_poses`. The corrected cubes were taken under the same fork stamp as step 2.
  - [x] Shown — the graph's own numbers, next to the SAME tie on Omega's raw cubes of this video (the pipeline's order):
        seams (4): sigma 1.4 - 1.6 cm (median 1.5)        | pipeline's order: 2.6 - 4.2 cm (median 3.1)
        after the graph, at their surfaces: 1.4 / 1.8 cm  | 3.5 / 5.5 cm (median / max)
        closures in the solve: 26 of 32 (sigma median 6.9 cm; 6 outside the session's scale bound, declared) | 24 (8.2 cm)
        the graph moved the cubes at most 4.3 cm          | 8.3 cm
        cube scales 1.0000, 0.9463, 0.9783, 1.0012, 0.9391. 358 s on the card.

- [x] **9. Escala a metros (scale_align) y orientación.**
  - [x] The worker's own sequence after the cube stage, with its functions and the job's settings (`step9_metres.py`):
        `_postprocess_reconstruction`, `_emit_omega_depth` (174 records from the tied cubes),
        `reconstruction.scale_align.run`, `reconstruction.orient.run`.
  - [x] The pipeline's orientation step ran and REFUSED, as it does on this video in the pipeline's own order
        ("camera-down consensus too weak (alignment 0.475 < 0.7) — leaving the reconstruction frame untouched": the
        video is rotated 90 degrees; the known miss of the orientation step).
  - [x] Shown: ONE factor x1.0573 for the whole session (DA3, median over the 174 keyframes, MAD 3.8 %, spread
        0.878 - 1.195); the walk measures 12.769 m before and 13.501 m after (the I3 walk: 13.40 m).

- [x] **10. Voto, limpiador y publicación.**
  - [x] The pipeline's F6 on the session (`depth_on_f5.compute` + `publish_cloud`, `step10_publish.py`), after the
        files F6 asks of a session were written in F5's format (`step10a_handover.py`: the one camera, the report;
        held-out of the FINAL cameras measured with F5's own function: median 0.2675 px over 83,987 tracks — the
        pipeline's F5 on this video: 0.2550). No factor is fitted and no PointDiT runs in this step
        (`bend.enabled: false`, `mono_detail.enabled: false` in the session's frozen configuration): each keyframe takes
        the depth step 7 gave it in the cube that owns it, in the session's units; the factor of the turn frames is the
        known metre (x1.0573). Check before the vote: step 7's depth / the session's records x metre = 1.000000 on
        every keyframe.
  - [x] Confidence floor per cube, the vote (`edge_keeping_vote`, COLMAP rule: 138 turn frames + a median of 167
        co-visible keyframes per keyframe; 5 views or more, 1.5° - 30°, reprojection 0.301°, 15 m, occlusion veto,
        majority, in place, no repair), the cleaner (voxel 5 mm, SOR 8 / sigma 3), the publication (epoch 1, octree).
  - [x] Shown — next to the pipeline's order on the same video (`bufferStop_prueba`):
        vote coverage 52.9 %            | 61.5 %
        kept 47.4 %, in place 10.8 %    | 61.2 %, 6.0 %
        contradicted 23.3 %             | 14.2 %
        raw points 54,303,956           | 63,117,776
        published points 4,177,718      | 3,919,659
        309 s on the card.
  - MISTAKES OF THE FIRST RUNS, corrected: steps 7 and 10 called `compute()` without the stage entry of the pipeline's
    F6 (`f6_torch.require_cuda`), so their maps ran on the CPU (step 7: 2638 s; step 10's vote: stopped after 47 min).
    With the call added, cube 4 of step 7 was computed again on the card in 253 s and compared with the first result:
    24,903,537 pixels with depth in both, IDENTICAL in 100.000 % (factors, masks and source maps equal) — steps 7, 8
    and 9 stand. The first step 10 also left the turn frames' depth 5.5 % short (factor 1 instead of the metre).

- [x] **Deliverable: the cloud of bufferStop in the copy, for his eye.** Session `bufferStop_orden_nuevo`
      (4,177,718 points, octree). To compare with: `bufferStop_prueba` (the pipeline's order, same video, cloud only)
      and `bufferStop`. The orientation step refused on both (the video is rotated): neither is levelled.
