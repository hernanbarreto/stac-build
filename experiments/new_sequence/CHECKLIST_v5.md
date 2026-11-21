# Checklist — version 5: today's pipeline without I3, F2 and the DA3 lock per cube, on a copy of bufferStop

USER 2026-11-21, to "los keyframes de hoy (los 174) con el pipeline ya limpio, sin I3, sin F2 y sin candado, dejando todo
lo demás como en la original": *"si, dale, veamos como sale"*.

Test session: `server/projects/bufferStop_v5`. Log: `/workspace/tmp/new_sequence/v5/logs/front.log`. Driver:
`v5_front.py` — the pipeline's worker with config.yaml as it stands (its own keyframe selection, turn frames, witness
frames, PointDiT on) and three substitutions: the walk from AMB3R's poses instead of I3, each cube at AMB3R's metre
instead of the DA3 lock, the chain F0, F4, F5, F6 with F5's anchor tolerance from the tie's seams instead of F2.
42.2 min on the card (AMB3R's pass reused).

- [x] **The keyframes are the original's.** I1's 172 + 2 by coverage = 174, the same 174 frames as the validated
      `bufferStop`; 57 turn frames, 81 interval witnesses, 746 witness frames.
- [x] **Cubes.** 5 cubes of 64, 64, 64, 64, 46 keyframes. Each at AMB3R's metre: 3.0614, 2.8705, 2.4124, 2.2850,
      2.4091; what is left of the fit 4.5, 3.4, 1.8, 4.1, 2.4 cm.
- [x] **The tie.** Seams 2.9, 3.0, 4.1, 2.7 cm (the pipeline's own run on this video: 3.0, 3.2, 4.2, 2.6); cubes moved
      at most 5.1 cm; seams after the solve 3.0 cm median, 5.5 max (3.5 / 5.5).
- [x] **The metre (DA3, once, frame by frame).** x0.9305, MAD 9.9 %, spread 0.641 - 1.909. THE CLOUD IS 7.5 % SMALLER
      THAN THE ORIGINAL'S: the cameras' walk measures 11.96 m against 12.94 m on the same 174 keyframes.
- [x] **F0, F4, F5.** F4: 4,467,947 observations, 761,874 tracks — the pipeline's own totals. F5 anchor tolerance
      1.89 % (seams 1.6, 1.8, 3.5, 2.0 %); held-out R0 0.2896, R1 0.2702, R2 0.2554 px, applied (the pipeline's own
      run: 0.2894 / 0.2699 / 0.2550).
- [x] **F6 with PointDiT.** Vote coverage 59.1 % (kept 58.3 %, contradicted 16.7 %, in place 6.5 %); raw
      60,612,801 → published 3,214,161 points, epoch 2. The validated `bufferStop`: coverage 60.5 %, contradicted
      15.2 %, 3,837,478 points.

- [x] **Deliverable: the cloud in `bufferStop_v5`, for his eye, next to `bufferStop`.** The tie, F4 and F5 read as
      the original's; the vote reads a little lower and the cloud is 7.5 % smaller (the metre). Not verified: whether
      the vote's difference comes from the metre too (the 15 m range is an absolute length).
