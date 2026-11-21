# Checklist — version 2: new keyframe selection + today's order + a second tie, on a copy of bufferStop

USER 2026-11-21: *"de acuerdo con los 3 puntos finales, y con los 12 pasos de reconstruccion, cheklist y a probar,
todo en placa"*. The twelve steps are the ones he approved, word for word; a box is ticked only when its check has
been shown with its numbers. Everything runs on the card.

Limits of the selection (approved by him: *"si, me cierran"*): overlap with the last chosen keyframe between **90 %**
(more = repeated) and **85 %** (less = a hole). Sources: photogrammetric practice (Pix4D: at least 75 % forward
overlap, 85 % for difficult scenes; Esri: 80 %) and the vote's own rule (5 confirming views: a point stays in view for
1 / (1 − overlap) keyframes — 6 at 83 %, almost 7 at 85 %). The best frame of the band: ONE score, the image's detail
with the light equalised (approved with the three final points). Cubes of 64 sharing 32: VGGT-Long's own default is
60 with 30.

Test session: `server/projects/bufferStop_v2`. Working data: `/workspace/tmp/new_sequence/v2/`.

- [x] **1. AMB3R primero, sobre todos los frames del video, para tener la pose de cada frame.**
  - [x] The pipeline's own call (`reconstruction.amb3r_probe.session_amb3r`), made BEFORE any keyframe is chosen
        (`v2_front.py`): 928 video frames, every one with a pose; 232 reconstructed by its backend (what each saw);
        28 AMB3R keyframes; its map in metres x2.3632 (DA3 over the 16 focal-probe frames, MAD 3.8 %), walk 13.08 m.
        408 s on the card.

- [x] **2. Selección nueva de keyframes sobre ese mapa, del frame 0 al n, comparando cada frame con el último keyframe elegido.**
  - [x] `v2_select.py`, on the card: 104 keyframes of 928 frames (today's rule gives 174 on this video). 61 s.
  - [x] overlap above 90 %: repeated, skipped; between 90 % and 85 %: candidate; at 85 % the best candidate is chosen
        — 100 keyframes chosen as the best of their band, 1 as the best of the last band (the video ended inside it).
  - [x] a jump from above 90 % to under 85 % between two consecutive frames: both are taken — ONE such jump (frames
        642 / 643: the camera turns 2 degrees a frame there; overlap with the last keyframe 0.910 then 0.848).
  - [x] Shown: overlap between consecutive keyframes min 0.852 / median 0.878 / max 0.947 — none under 85 %; the
        two over 90 % are the jump's pair. Video frames between keyframes: min 1 / median 8 / max 40.
  - [x] no brightness gate that rejects (928 / 928 frames usable; bounds open), no 8-frame rule, no coverage fill; the
        turn frames as today, on the windows of the chosen keyframes measured by I1's own tracker: 70 turn frames
        (Omega's context) + 38 interval witnesses (F6 only); 735 witness frames.
  - FOUND AND CORRECTED WHILE WRITING IT: with "a frame sees what its NEAREST reconstructed frame saw" the overlap
    stepped at every change of record (one frame in four is reconstructed) and the walk read four false jumps; a frame
    now sees what its TWO reconstructed neighbours saw, weighted by how near each is.

- [x] **3. Cubos: tandas de 64 keyframes, compartiendo 32 con la siguiente.**
  - [x] The pipeline's planner, unchanged: 3 cubes of 64, 64 and 40 keyframes (keyframes 0-63, 32-95, 64-103), each
        sharing 32 with the next; the last one holds the 40 that are left (not a full 64). Omega views per cube
        (keyframes + turn frames): 110, 99, 64.

- [x] **4. Omega por cubo y candado métrico DA3, como hoy.**
  - [x] The worker's own cube stage: 110 / 99 / 64 views inferred; DA3 lock per cube 3.1827, 2.6142, 2.0635
        (64, 64, 40 anchors).

- [x] **5. Primera atada del pipeline: costuras, cierres con puentes de Omega y grafo Sim(3). Después, escala a metros y orientación, como hoy.**
  - [x] Seams (2): 0<->1 residual 12.1 cm (scale 1.0210), 1<->2 residual 3.6 cm (scale 0.9533). With today's
        selection the seams of this video read 2.6 - 4.2 cm: THE SEAM 0<->1 IS THREE TIMES WORSE — not explained yet.
  - [x] Bridges: 4 of the 10 revisits join two cubes that share no keyframe (all between cubes 0 and 2); 4 closures
        in the solve (scale 0.966 - 1.167, residual 5.4 - 14.8 cm); no scale bound (2 seams: no spread to measure).
  - [x] Graph: cube scales 1.0000, 1.0805, 1.0354; cubes moved at most 10.2 cm; seams after the solve, at their
        surfaces: 11.0 cm and 1.0 cm.
  - [x] Metre: x0.9815 (DA3, 104 anchors, MAD 3.5 %, spread 0.848 - 1.081). Orientation REFUSED by the pipeline's
        step, as on this video before (alignment 0.652 < 0.7: the video is rotated) — the cloud is not levelled.

- [x] **6. F0, como hoy.** (the chain's own step; 0.0 min)

- [x] **7. F2 sobre todo el recorrido, como hoy.**
  - [x] Instrument `da3_windows`; scale per keyframe 0.9353 - 1.0570 along the walk (13.26 m); cameras moved at
        most 27.8 cm. 0.5 min.

- [x] **8. F4, como hoy.**
  - [x] 119 windows (8 keyframe, 10 loop, 101 witness): 2,914,479 observations, 383,861 tracks (76,305 held out),
        keyframes 104 / 104, sigma median 1.11 px. 4.6 min. (Today's selection: 4,467,947 observations, 761,874 tracks.)

- [x] **9. F5, como hoy, sobre toda la sesión.**
  - [x] Held-out median R0 0.2615 px, R1 0.2263 px (taken), R2 0.2083 px (taken, applied) — today's selection on
        this video: 0.2894 / 0.2699 / 0.2550. Camera fx 891.57, fy 889.10, cx 514.71, cy 283.10, k1 +0.0566,
        k2 -0.0571. 1.0 min. The session is at epoch 2.
  - Steps 1 - 9 ran in ONE process, the pipeline's own worker (`workers.map_worker._map_work`) with the substitutions
    listed at the top of `v2_front.py`: 29.2 min in all on the card (AMB3R 6.8, cubes + Omega + tie about 13,
    F0 - F5 6.4).

- [ ] **10. Profundidad de F6: el factor de cada keyframe ajustado a las correspondencias. PointDiT apagado.**
- [ ] **11. Segunda atada, solo con los puentes de Omega entre pasadas.** (before the vote; the seams between
      neighbouring cubes stay as they are, with the tolerance they measured in the first tie — said to him)
- [ ] **12. Voto, limpiador y publicación.**

- [ ] **Deliverable: the cloud of bufferStop in `bufferStop_v2`, for his eye, next to `bufferStop_prueba` (the
      pipeline as it is) and `bufferStop_orden_nuevo` (the first test).**
