# Checklist — version 4: version 3 with the keyframes as the only frames of the session, on a copy of bufferStop

USER 2026-11-21: *"solo los frames que votan los que entran en todo el circuito son los KF, nada mas no entran en
ninguna pasada de nada … hace de cuenta que los unicos frames que tiene la sesion son los kf, nada mas, ni para f6 ni
para relleno de nada"* / *"son 64 con solape de 32, no hay vuelta con eso"* / *"asi es, corregi eso y volvemos a
correr v4"*.

Test session: `server/projects/bufferStop_v4`. Log: `/workspace/tmp/new_sequence/v4/logs/front.log`. Driver:
`v4_front.py` (version 3's substitutions + the turn frames switched off by the pipeline's own key, I1's witness list =
the keyframes, I1's tracker not run). PointDiT off. 10.8 min on the card with AMB3R, the selection's inputs and
DA3's depths of the same video reused.

- [x] **The only frames after the selection are the 104 keyframes.** Turn frames 0, interval witnesses 0, witness
      frames for F4 / F5 0 (the witness list holds the 104 keyframes).
- [x] **Cubes of 64 sharing 32, nothing else in Omega's pass.** 3 cubes: 64, 64, 40 keyframes = 64, 64, 40 Omega
      views (version 3: 110, 99, 64 views).
- [x] **Omega per cube at AMB3R's metre.** 2.9841, 2.6426, 2.0328; what is left of the fit: 37.6 cm in cube 0
      (version 3: 52.3), 3.5 and 2.4 cm in cubes 1 and 2.
- [x] **The tie.** Seams 0<->1 residual 11.8 cm (scale 0.9825), 1<->2 3.5 cm (scale 0.9878) — THE SEAM 0<->1 IS AS
      BAD AS WITH 110 VIEWS: the turn frames were not its cause. 4 closures of 10 revisits (scale 0.953 - 1.131,
      residual 4.5 - 13.0 cm); cubes moved at most 5.2 cm; seams after the solve 4.6 cm median, 8.4 max.
- [x] **The metre (DA3, once).** x0.9449, MAD 7.0 %, spread 0.635 - 1.399.
- [x] **F0, F4, F5.** F4: 18 windows (8 keyframe, 10 loop, 0 witness) — 451,324 observations, 60,166 tracks
      (version 3: 119 windows, 2,914,479 observations, 383,861 tracks). F5 anchor tolerance 6.14 % (seams 9.5 % and
      2.8 %); held-out R0 0.932, R1 0.803, R2 0.751 px, applied — VERSION 3: 0.209 px. With the frames between
      keyframes gone, F4's correspondences are fewer and F5's fit is 3.6 times looser.
- [x] **F6: the factor per keyframe, the vote of the keyframes alone, the cleaner, the publication.** Each keyframe
      hears a median of 97 co-visible keyframes. Vote coverage 57.6 % (kept 49.4 %, contradicted 20.1 %, in place
      11.4 %, under the support 23.5 %) — version 3: 65.8 % (61.8 / 12.2 / 7.6 / 15.1). Raw 35,320,123 → published
      2,117,768 points (version 3: 2,858,186), epoch 2.

- [x] **Deliverable: the cloud in `bufferStop_v4`, for his eye.** By the run's own numbers it is worse than
      version 3 (coverage, contradiction, F5's held-out).
