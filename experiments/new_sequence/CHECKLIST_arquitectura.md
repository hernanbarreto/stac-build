# Checklist — the architecture of 2026-11-21 (morning), and the five tasks

USER 2026-11-21, after the design conversation: *"eso mismo"* to the architecture and the tasks below; *"ok, dale"*.
His order for the night: *"yo me voy a dormir, asi que para que la placa no quede parada, creas una nueva sesion
bufferstop_prueba y me dejas esta nube ahi publicada, al terminar, haces lo mismo con pccr3104 y con pccr2408, en
todos los casos cuando la nube se publico, borra a mano todo el tendal de archivos que hayan quedado antes de empezar
la siguiente reconstruccion porque no quiero que se llene el disco y no pueda entrar esta prueba de pccr2408"*.
("pccr3104" = the scan of 2026-08-31. The sessions: `bufferStop_prueba`, `pccr3108_prueba`, `pccr2408_prueba`.)

A box is ticked only when its check has been shown with its numbers.

## The architecture (his nine steps, as approved)

1. AMB3R sobre todos los frames. Da las poses y la profundidad de AMB3R.
2. Selección. Lee las poses y la profundidad de AMB3R. Entrega los keyframes (solape de 92 a 96 % con el último
   elegido), las revisitas y los testigos de F4, que son los frames que quedan entre keyframes.
3. Cubos de Omega: 64 keyframes compartiendo 32. DA3 usa los mismos.
4. DA3 por cubo, a resolución 504. Da la profundidad DA3 de cada keyframe, para la escala.
5. Omega por cubo. Da las poses y la profundidad de Omega.
6. Atada: costuras, puentes de Omega sobre las revisitas, y grafo. Los 15 m se miden con el tamaño de cada cubo, que
   sale de la profundidad DA3 de ese cubo.
7. Escala: un número, de comparar la profundidad DA3 con la profundidad de Omega ya atada.
8. F0, F4 y F5. F4 usa keyframes, testigos de F4 y revisitas. La tolerancia de F5 sale de las costuras.
9. F6: factor por keyframe, voto entre keyframes, limpiador y publicación.

## What the runs take as given (for him to decide)

- **AMB3R's metre.** The runs read each video's stored AMB3R pass, which is in metres since the original intake (its
  16-frame DA3). The selection's 15 m range needs that metre; on a NEW video this architecture has no source for it.
- **What each keyframe sees of every other** is measured once, with the cube plan, on AMB3R's poses and depth
  (intake/cubes.json). The revisits (the tie's bridges, F4's windows) and the keyframes F6's vote hears are read from
  that one list.
- **A session of one cube** has no seam, so no tolerance for F5's anchors: the run stops naming it.
- **The fork sets a bridge aside** when its scale is outside the seams' bound (the pipeline's rule as it stands):
  46 of bufferStop's 53 bridges.

## The five tasks

- [x] **1. Programar esta arquitectura.** `arq_front.py` (the pipeline's worker with the nine steps' substitutions),
      `arq_select.py` (the band 92 – 96 %), `arq_omega.py` (the fork's cube stage, each cube sized with its own DA3
      window), `arq_session.py`, `arq_clean.py`, `arq_queue.py`, `arq_watch.py`. No file of `server/` or of the fork
      is edited. Shown by task 2: the nine steps ran end to end, 37.2 min on the card.
- [x] **2. Crear `bufferStop_prueba`, correrla y dejar la nube publicada.** Log:
      `/workspace/tmp/new_sequence/arq/logs/bufferStop_prueba.log`. Against the validated `bufferStop` (in brackets):
  - step 1 — AMB3R's pass of this video: 928 frames with a pose, 232 with depth.
  - step 2 — 209 keyframes (174); overlap between consecutive keyframes 0.920 / 0.944 / 0.962 (min / median / max);
    frames between keyframes: median 4, max 25; F4's witnesses: the 719 frames between keyframes; 64 revisits.
  - step 3 — 6 cubes: 64, 64, 64, 64, 64, 49 keyframes, each sharing 32 with the next; nothing else in a cube.
  - step 4 — DA3 per cube at 504: 6 windows, 16.7 GiB at the peak, 13 s a window.
  - steps 5, 6 — the cubes sized with their own DA3 window: 2.91, 2.75, 2.34, 1.77, 1.77, 2.11; seams 2.7, 2.7, 1.6,
    2.7, 2.5 cm (2.6 – 4.2); 53 bridges, 7 closures in the solve; cubes moved at most 6.6 cm; seams after the solve
    1.9 cm median, 2.6 max.
  - step 7 — the scale: x1.0150 over the 209 keyframes (MAD 2.3 %, spread 0.942 – 1.114); the cameras' walk 12.9 m
    (12.94 m).
  - step 8 — F4: 277 windows (17 keyframe, 64 revisit, 196 witness), 5,885,240 observations, 927,301 tracks
    (4,467,947 / 761,874); F5's anchor tolerance 1.65 % (seams 1.40 – 2.54 %); held-out R0 0.3450, R1 0.3183,
    R2 0.2998 px, applied (0.2894 / 0.2699 / 0.2550); cameras moved at most 16.7 cm.
  - step 9 — F6: every keyframe hears a median of 202 keyframes; vote coverage 58.5 % (kept 53.2 %, contradicted
    19.3 %, in place 8.5 %, under the support 23.7 %) (60.5 %, contradicted 15.2 %); raw 72,164,182 → published
    3,406,141 points (3,837,478), epoch 2.
- [x] **3. Borrar a mano los archivos intermedios que hayan quedado.** `arq_clean.py --go` (the queue's log):
      da3_cubes 625 MB, da3_run 198 MB, tracks.npz 113 MB, sky masks 1 MB, the way back to epoch 0 — after the
      worker's own end removed 1,691 MB of pose-only epochs and Omega cube files. The scan holds: Omega's records
      840 MB, frames 207 MB, the octree 162 MB, origins 119 MB, the cloud 92 MB (and its raw copy 92 MB), intake 36 MB.
- [x] **4. Crear `pccr3108_prueba`, correrla, dejar la nube publicada y borrar los intermedios.** Log:
      `/workspace/tmp/new_sequence/arq/logs/pccr3108_prueba.log`; 69.5 min on the card. Against the validated
      pccr 2026-08-31 (in brackets):
  - step 2 — 450 keyframes of 3044 frames (547); overlap between consecutive keyframes 0.920 / 0.944 / 0.960; frames
    between keyframes: median 6, max 58; F4's witnesses: the 2594 frames between keyframes; 45 revisits.
  - step 3 — 14 cubes: thirteen of 64 keyframes and one of 34, each sharing 32 with the next (17 cubes).
  - step 4 — DA3 per cube at 504: 14 windows in 311 s.
  - steps 5, 6 — the cubes sized with their own DA3 window: 1.74 – 3.77; the 13 seams 1.5 – 4.1 cm (1.0 – 3.7);
    30 bridges, 29 closures in the solve (43); cubes moved at most 55.1 cm (61); seams after the solve 1.8 cm
    median, 5.1 max.
  - step 7 — the scale: x0.8633 over the 450 keyframes (MAD 5.4 %, spread 0.735 – 1.267); the cameras' walk 18.0 m.
  - step 8 — F4: 533 windows (37 keyframe, 45 revisit, 451 witness), 26,252,623 observations, 4,179,050 tracks,
    32.4 min; F5's anchor tolerance 1.18 % (seams 0.88 – 1.63 %); held-out R0 0.5263, R1 0.5209, R2 0.5090 px,
    applied (0.534); cameras moved at most 6.2 cm; 7.1 min.
  - step 9 — F6: every keyframe hears a median of 80 keyframes; vote coverage 58.7 % (kept 52.9 %, contradicted
    17.3 %, in place 6.7 %, under the support 23.8 %) (59.9 %, kept 55.3 %); raw 101,980,232 → published
    18,541,200 points (126.5 M → 18.8 M), epoch 2; 6.5 min.
  - the intermediates removed (the queue's log): da3_cubes 1,467 MB, da3_run 425 MB, tracks.npz 476 MB, sky masks,
    after the worker's own end removed 2,332 MB. The scan holds: Omega's records 1,175 MB, the octree 874 MB, origins
    649 MB, the cloud 501 MB (and its raw copy 501 MB), frames 243 MB, intake 76 MB.
- [ ] **5. Crear `pccr2408_prueba`, correrla y dejar la nube publicada.**
