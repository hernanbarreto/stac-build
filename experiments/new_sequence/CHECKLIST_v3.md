# Checklist — version 3: version 2's pipeline with what the user took out, on a copy of bufferStop

USER 2026-11-21, after his eye on version 2 ("no quedó bien, hay que corregir las pelotudeces, y los pasos
equivocados"): *"este pipeline, sin i3, sin f2, sin candado por cubo, sin da3 que decida nada mas que cuanto escalo la
nube para pasarla a metros, sin la costura final"*; F5's anchor tolerance from the tie's seams (*"ok, de acuerdo"*);
*"corremos el primero sin pointdit y el segundo rehacemos f6 con pointdit prendido y comparamos ambas nubes"* —
*"una comparacion de ambas nubes como la sabana"*. His word to start: *"si"*.

Test session: `server/projects/bufferStop_v3`. Working data and logs: `/workspace/tmp/new_sequence/v3/`.
A box is ticked only when its check has been shown with its numbers.

- [x] **1. AMB3R primero, sobre todos los frames.**
  - [x] The whole-video pass of this same video reused on its stamp (928 frames with a pose, 232 reconstructed, map in
        metres x2.3632).
- [x] **2. Selección nueva de keyframes (solape 85–90 %).**
  - [x] Reused on the intake's marker: the 104 keyframes of version 2 (overlap between consecutive ones min 0.852 /
        median 0.878 / max 0.947; one real jump). 70 turn frames + 38 interval witnesses.
- [x] **3. Cubos de 64 keyframes compartiendo 32.** 3 cubes: 64, 64 and 40 keyframes.
- [x] **4. Omega por cubo, sin candado DA3. Los 15 m dentro del cubo se miden con el metro de AMB3R.**
  - [x] `v3_omega.py`: each cube scaled to AMB3R's metre (the similarity of its camera centres onto AMB3R's):
        2.8876, 2.6801, 2.0183 (DA3's lock in version 2: 3.1827, 2.6142, 2.0635). What is left of that fit: 52.3 cm
        in cube 0, 3.6 and 2.7 cm in cubes 1 and 2 — CUBE 0'S CAMERA PATH DOES NOT FOLLOW AMB3R'S; not looked into.
- [x] **5. Atada única: costuras, cierres con puentes de Omega y grafo Sim(3).**
  - [x] Seams: 0<->1 residual 10.9 cm (scale 0.9055), 1<->2 residual 3.7 cm (scale 0.9991). Bridges: 4 of 10
        revisits; 4 closures (scale 0.924 - 1.081, residual 6.0 - 14.2 cm). Cube scales 1.0, 0.95, 0.9534; cubes
        moved at most 8.1 cm; seams after the solve 4.9 cm median, 8.9 max.
- [x] **6. Escala a metros con DA3 (lo único que hace DA3) y orientación.**
  - [x] DA3 ran ONCE on the 104 keyframes, frame by frame (plus the focal probe's 16 frames at the start): x0.9928,
        MAD 7.2 %, spread 0.666 - 1.479 (version 2, anchors from I3's windows: x0.9815, MAD 3.5 %, 0.848 - 1.081).
        Orientation REFUSED by the pipeline's step (alignment 0.651 < 0.7), as before: not levelled.
  - THE CLOUD IS 10 % SMALLER THAN VERSION 2'S: the cameras' walk after F5 measures 11.54 m (version 2, same
    keyframes: 12.89 m; AMB3R's walk: 12.91 m). The shape is the same; the metre changed. Which metre is right is
    not known without a real length of the scene; the cause was not verified.
- [x] **7. F0, F4 y F5, con las anclas de F5 a la tolerancia de las costuras. Sin I3 y sin F2.**
  - [x] F5's anchor tolerance 6.12 % (the seams: 9.3 % and 2.9 %; F2 gave 4.0 % in version 2). F4: 2,914,479
        observations, 383,861 tracks (4.4 min). F5 held-out R0 0.2626, R1 0.2269, R2 0.2086 px, applied (version 2:
        0.2083) (0.8 min).
- [x] **8. F6 sin PointDiT: factor por keyframe, voto, limpiador y publicación — nube A.**
  - [x] Vote coverage 65.8 % (kept 61.8 %, contradicted 12.2 %, in place 7.6 %, under the support 15.1 %); raw
        40,338,196 -> published 2,858,186 points, epoch 2 (2.4 min). The whole run: 18.8 min on the card with the
        intake and AMB3R reused. HIS EYE: "es muy parecida a la anterior" / "no es mejor que la original".
- [x] **9. F6 otra vez, con PointDiT prendido — nube B.**
  - [x] The chain's own F6 run again over the same F0, F4, F5: PointDiT tiles 592 of 624 accepted, detail on 41.7 %
        of the valid pixels (299 s); vote coverage 64.8 %; 2,874,453 points, epoch 3. 9.3 min.
- [x] **10. Comparación A contra B pintada como la sábana (verde en 0, rojo desde 2 cm).**
  - [x] `v3_compare.py`, exact nearest neighbour on the card: of cloud A's points 90.6 % lie within 5 mm of cloud B,
        98.0 % within 1 cm, 99.7 % within 2 cm, 0.3 % beyond (red); median 1.7 mm. Of cloud B's: 89.7 %, 97.6 %,
        99.6 %, 0.4 % red; median 1.7 mm.

- [x] **Deliverable: `bufferStop_v3` (cloud B live as epoch 3, cloud A the epoch before it), `bufferStop_v3_pintada_A`
      and `bufferStop_v3_pintada_B` (each cloud painted by its distance to the other). His eye on B and on the painted
      ones: pending.**
