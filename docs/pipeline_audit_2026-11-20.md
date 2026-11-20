# Audit of the reconstruction pipeline, step by step — 2026-11-20

USER 2026-11-20: *"quiero una auditoría correcta y completa de que todos los pasos quedan coherentes, que un paso no
rompe lo que corrigió el anterior, que realmente los pasos agregan valor, coherencia y necesidad absoluta … Cada
cálculo, cada paso que damos en el proceso de reconstrucción debe ser necesario y coherente."*

Session audited: the copy of pccr 2026-08-24 (`pccr2408_prueba`, 13,692 frames, 2,772 keyframes, 170 cubes, 102.8 m of
walk — the longest and hardest session of the project). Everything below is MEASURED on it, read only. The scripts are
in the session's scratchpad (`serio/`), not in the repo.

## 1 · How each step was judged: a real ruler that is in the scene

The raised floor's tiles are all the same size and lie on one grid. Each keyframe draws the joints between tiles with
its own camera and its own depth, so the joints give, with no assumption of ours:

| Ruler | What it reads | Its own precision |
|---|---|---|
| The tile's size in one keyframe's depth | that keyframe's metric scale | 0.2 – 1.1 % per stretch; the two directions of the grid agree (ratio 1.0000, spread 0.83 %) |
| The same joints drawn by two keyframes a step apart | whether the DEPTH and the STEP BETWEEN THE CAMERAS agree (the gap grows with the step: % of the step) | ± 0.1 % |
| The same floor drawn by two DIFFERENT PASSES (over 640 video frames apart) | whether two visits of one place agree (cm between their joints, along the floor) | 690 – 1,770 readings |
| Are the tiles square | the camera model (focal length) | ± 0.15 % |
| The two known defects, on their own pixels | the white cabinet seen from 5.9 m against itself seen from 1 m ("the plate"); the rack seen from its two sides | the pixels placed with the published cameras and factors fall 2.2 cm from the cloud's own points |

The grid's heading reads 71.25° over the whole walk (70.25 – 72.00): no heading drift to speak of.

Units (checked on the cameras, exact): the cubes' stored depth (the Omega records) is in the cubes' own units; epoch 0
= × 0.97158 (scale_align); epoch 4 = × 0.95456 (× F2's one scale 0.98248); F5's cameras = × 0.95392.

## 2 · Every stage under the same ruler

A geometry = camera positions and rotations + a camera model + a depth factor per keyframe. Each numbered row adds
ONE step to the row above it.

| # | Geometry | camera steps vs depth | joints, same moment | joints, DIFFERENT PASSES: median / 90 % (within 2 cm) | tile: median, spread 5–95 % | the plate (within 3 cm of its surface) | rack front–rear | rack side, rear pass vs front pass |
|---|---|---|---|---|---|---|---|---|
| 1 | cubes chained by their seams alone | −2.9 % | 1.2 cm | 3.7 / 7.5 cm (19 %) | 0.629 m, 14.0 % | **54.5 cm (0 %)** | **1.67 m** | **−88.3 cm** |
| 2 | + the revisit closures the run admitted (1,082) | −3.1 % | 1.2 cm | 2.6 / 6.0 cm (35 %) | 0.613 m, 7.7 % | 1.0 cm (78 %) | 0.84 m | −1.2 cm |
| 3 | + every bridge as a closure (1,726) = the cube graph | −3.1 % | 1.2 cm | 3.2 / 6.3 cm (27 %) | 0.615 m, 8.1 % | 0.9 cm (80 %) | 0.79 m | +1.7 cm |
| 4 | + F2, ONE scale for the session | −3.1 % | 1.2 cm | 3.2 / 6.1 cm (28 %) | 0.604 m, 8.1 % | 0.9 cm (81 %) | 0.78 m | +1.6 cm |
| — | *instead of 4: the OLD F2 (DA3's scale along the walk)* | −1.8 % | 1.3 cm | **6.4 / 26.9 cm** (24 %) | 0.607 m, **31.2 %** | 6.5 cm (19 %) | **1.49 m** | **−29.6 cm** |
| 5 | the same cameras, F5's camera model | −1.1 % | 0.8 cm | 2.8 / 8.0 cm (32 %) | 0.588 m, 8.4 % | 3.1 cm (49 %) | 0.83 m | −3.5 cm |
| 6 | + F5's cameras (the depth still at the one scale) | −1.4 % | 0.8 cm | **1.5 / 6.5 cm (58 %)** | 0.588 m, 8.4 % | 2.9 cm (51 %) | 0.88 m | −3.7 cm |
| 7 | + F6's depth factor, one PER KEYFRAME = **published today** | −1.8 % | 1.0 cm | 3.4 / 10.3 cm (33 %) | 0.585 m, 10.9 % | **45.4 cm (0 %)** | **1.02 m** | **+7.6 cm** |
| 7′ | instead of 7: F6's factor, ONE for the session (× 0.9477) | −0.7 % | 0.8 cm | 1.4 / 5.4 cm (60 %) | 0.584 m, 8.4 % | 4.0 cm (36 %) | 0.92 m | −1.8 cm |

Which single depth factor the tiles ask for (F5's cameras and camera model, the same for every keyframe):

| depth × | camera steps vs depth | joints, different passes | the plate | rack front–rear |
|---|---|---|---|---|
| 0.9300 | +1.2 % | 2.9 / 9.4 cm (39 %) | | |
| 0.9400 | +0.1 % | 2.0 / 6.8 cm (50 %) | 3.9 cm (43 %) | 0.96 m |
| 0.9477 (the median F6 fits) | −0.7 % | 1.4 / 5.4 cm (60 %) | 4.0 cm (36 %) | 0.92 m |
| 0.9500 | −1.0 % | 1.4 / 5.6 cm (60 %) | 3.9 cm (37 %) | 0.90 m |
| **0.9546 = the scale the cameras already carry** | −1.4 % | 1.5 / 6.5 cm (58 %) | 2.9 cm (51 %) | 0.88 m |
| 0.9600 | −2.0 % | 2.3 / 7.9 cm (43 %) | 1.5 cm (65 %) | 0.85 m |
| 0.9700 | −3.1 % | 4.4 / 9.3 cm (28 %) | | |

The floor rulers read the floor within 4.5 m of the camera, ALONG the floor; the plate is the far-range check (5.9 m
against 1 m) and reads the distance ACROSS a surface (to the nearest surface of the near pass).

## 3 · Step by step

**Verdicts**: NECESSARY · REDUNDANT (decides again what another step decided, no measurable effect) · HARMFUL AS IT IS ·
NOT JUDGED (these rulers do not see it).

### Intake and structure
| Step | What it decides | Who decides it again later | Measured | Cost | Verdict |
|---|---|---|---|---|---|
| Frames, I0 quality, I1 keyframes | which frames become keyframes (2,756 + 16 by the coverage fill) | — | the study of 2026-11-19: the fixed brightness bound (40) cut a 130-frame hole at a dark turn that bent two cubes | minutes | NECESSARY; the brightness bound is a fixed number that broke a session |
| AMB3R-SLAM, whole video | which keyframes see the same place: the revisit pairs of the bridges, F6's 773,446 witness pairs | — | without the closures it proposes: row 1 | ~1 h | NECESSARY |
| I3, DA3 windows | DA3 depth along the walk (25.8 GB) | — | read by the per-cube lock, scale_align and F2: three estimates of one metre | ~30 min | NECESSARY once, for the metre |
| VGGT-Ω per cube (170 cubes) | the geometry: every keyframe's depth, camera and pose inside its cube | F5 (poses, camera), F6 (depth scale) | its camera steps are 3.1 % short for its own depth (rows 1–4, all along the walk); its own camera model draws the tiles 3.4 % wider than long; its scale drifts slowly along the walk (8 % between the 5th and 95th percentile) | ~1.3 h | NECESSARY |
| DA3 metric lock per cube | each cube's scale | the graph (a scale per cube from seams and closures), then scale_align, then F2 | only cube 0's lock survives the graph, and scale_align re-fits the metre | seconds (at inference) | REDUNDANT as a per-cube decision |
| Seams | each cube against the next | — | the chain itself | ~50 min CPU → 6 s on the card (2026-11-20) | NECESSARY |
| Revisit bridges and the graph | where the walk closes | — | rows 1 → 2: the plate 54.5 → 1.0 cm, the rack 1.67 → 0.84 m and its side −88 → −1 cm, the tile's spread 14.0 → 7.7 % | 129 min of card for 1,726 bridges | NECESSARY — **the step that adds the most; after it both known pieces are right** |
| … the 644 closures the scale bound used to set aside | — | — | rows 2 → 3: 2.6 / 6.0 → 3.2 / 6.3 cm between passes; the plate 1.0 → 0.9 cm; the rack 0.84 → 0.79 m | — | NEUTRAL (the Cauchy loss already judges them) |
| … bridges between cameras over 6 m apart (566) | — | — | 16 % agree with the solution (study of 2026-11-19); the held-out bridges fit the same with or without them | 42 min of card | LITTLE INFORMATION for their cost — his decision |
| Uncertainty maps per frame | — | — | `uncert/` "is read by nothing after the run" (map_worker's own comment); only the session median of uncertainty.json is read | 4.7 min, 4 GB | UNNECESSARY as maps |
| Depth graph | a per-frame depth correction | — | verdict SKIP on the six sessions it ran on: computed, never applied; depth_graph.json has no reader | 6.1 min | UNNECESSARY as it is |
| scale_align | THE session's metre (× 0.9716, DA3) | F2 | — | minutes | NECESSARY (once) |
| Upright orientation | the display frame | the floor stage | — | minutes | NECESSARY |

### The precision chain (11.5 h on this session: F4 343 min, F6 127, F5 110, F2 35, check 4.6)
| Step | What it decides | Who decides it again later | Measured | Cost | Verdict |
|---|---|---|---|---|---|
| F0 camera | the session's camera = Omega's own (fx 363.7, fy 367.0) | F5 | the tiles come out 3.4 % wider than long with it | 20 s | NECESSARY as a start |
| F2, one scale | the metre again (× 0.9825 on scale_align's) | — | rows 3 → 4: not one ruler moves; the tile 0.615 → 0.604 m | 9 – 35 min | REDUNDANT: a second DA3 estimate of the metre scale_align had set |
| (the old F2, scale along the walk) | a scale per stretch (× 0.81 … × 1.25) | — | between passes 3.2 → 6.4 cm (90 %: 6.1 → 26.9), the tile's spread 8 → 31 %, the rack 0.78 → 1.49 m, its side +1.6 → −29.6 cm; DA3 asked × 0.83 where the tiles ask × 0.97 | — | HARMFUL — removed 2026-11-20, rightly |
| F4 tracks | the image tracks F5 and F6 read | — | — | **343 min (51 % of the chain)** | needed by F5 |
| F5 refinement | the camera model (focal +4.6 % / +6.5 %), the poses (≤ 15 cm), the landmarks F6 fits its factor on | — | rows 4 → 6: camera steps vs depth −3.1 → −1.4 %; same-moment joints 1.2 → 0.8 cm; between passes 3.2 → 1.5 cm; tiles 3.4 % off → 1.5 % off the other way; held-out reprojection 0.662 → 0.613 px. AND: the plate 0.9 → 2.9 cm, the rack 0.78 → 0.88 m | 110 min | ADDS VALUE near the camera, costs a little at range — and HARMS at the weak places: next row |
| … F5 at the places with no information | — | F6 follows it | it SHORTENS the camera path: × 0.75 through door 1, × 0.80 over video 3400–3766 (door 2), × 0.87 at the dark turn, × 0.93 – 0.94 on three more stretches | — | HARMFUL THERE — the origin of the displaced copies |
| F6 bend, the depth factor | Omega's depth × a factor fitted on F5's landmarks | — | **as a scale it adds nothing**: the depth at the scale the cameras already carry (row 6) reads 1.5 cm between passes; with one fitted factor 1.4 cm (row 7′); the tiles accept anything from × 0.9477 to × 0.9546 | minutes | REDUNDANT as a scale |
| … decided per keyframe (the record) | a factor per keyframe (window ± 4) | — | rows 6 → 7: between passes 1.5 → 3.4 cm, the plate 2.9 → **45.4 cm**, the rack 0.88 → 1.02 m, its side −3.7 → +7.6 cm, the tile's spread 8.4 → 10.9 %. The factor runs from 0.10 to 2.18 (44 keyframes under 0.80, 72 over 1.05), all "verified" — on landmarks of the same F5; × 0.87 around door 2. Its held-out error is the same for every window (3.01 – 3.04 %): the rule that chooses the window has nothing to choose with | — | **HARMFUL AS IT IS — it un-does what the graph closed. The cause of the plate** |
| F6 PointDiT at the edges | the depth within 1.8° of a discontinuity | the vote | 20.7 M of the 112.4 M published points come from it | 55 min | NOT JUDGED here (the rulers read floors and flat faces) |
| F6 vote | which pixels are published | the cleaner | of 1.02 × 10⁹ valid pixels: kept 49.7 %, in place 8.8 %; out: confidence floor 25.2 %, short of support 22.0 %, contradicted 16.7 %, minority 8.3 %, occlusion veto 5.6 % | 36 min | NECESSARY — but NOT a safety net for a stretch published at the wrong scale: no witness sees through the plate, at any angle (census with the full depth maps) |
| … the 30° upper limit of the triangulation band | which witnesses are heard | — | a number from the literature (the assistant's, 2026-11-08), never measured here; without it ≤ 1 % of the points change | — | an invented number — harmless so far; his decision |
| Cleaner (voxel 5 mm, SOR) | 597.7 M → 112.4 M points | — | — | ~7 min | NECESSARY |
| f6_check | a report (floor / ceiling / camera per cube) | — | read by cloud_metrics → the chat's quality tool; moves no geometry | 4.6 min | REPORT ONLY |

### Written and read by nothing
- `precision/layer_mono_detail.glb`, `precision/layer_mixed_unresolved.glb` — every F6 writes them (2.0 M and 13.5 M
  points); the viewer's layers were removed on 2026-11-19 and no code reads them.
- `precision/refine_residuals.npz` — written by F5, read by nothing in the server.
- `maplong_run/uncert/` — the per-frame uncertainty maps (above).
- `maplong_run/depth_graph.json` — (above).
- `output/omega_probe.json` — a leftover of F3, removed on 2026-11-10.

## 4 · One quantity decided again and again: the scale

per-cube DA3 lock → the graph's cube scales → scale_align × 0.9716 (DA3) → F2 × 0.9825 (DA3 again) → F6's factor on
the depth (median × 0.950 against the cameras' × 0.9546 — and per keyframe from 0.10 to 2.18). Only two of those are
needed: the graph's (the cubes among themselves) and ONE metre for the session.

The tile reads 0.615 m at epoch 0 and 0.604 m after F2 with Omega's camera model, 0.588 m with F5's camera model,
0.585 m as published. **The tile's real size is not known here** — it would say which is the true metre (a standard
raised-floor tile is 600 mm, or 24 in = 609.6 mm).

## 5 · What the audit finds, in order of weight

1. **A later step un-does what the graph closed.** After the cube graph both known pieces are right (the plate at
   0.9 cm, the rack's side at +1.7 cm). F6's per-keyframe factor publishes the plate 45 cm away and the rack's side
   7.6 cm off. The same class of failure as the old F2: a scale decided locally after the walk was closed. Under test
   now: F6 with one factor (row 7′).
2. **F6's factor is not needed as a scale at all**: the depth at the scale the cameras already carry is as good by
   every ruler (row 6 against row 7′). What F6 needs is that one number, which the run already has.
3. **F5 shortens the camera path where the images carry no information** (doors, the dark turn), by 6 – 25 %. To
   correct in F5 once F6's result is seen (USER: "si mejora este f6 corregimos f5").
4. **The old F2 was wrong** (DA3's local metre is off by up to 15 %): twice the gap between passes, the rack at
   1.49 m, tiles from 0.51 to 0.65 m in the original cloud. One scale was the right decision; as it stands F2 is a
   second estimate of the metre and moves nothing.
5. **F4 + F5 cost 7.5 of the chain's 11.5 hours.** They bring: between passes 3.2 → 1.5 cm, the camera model (tiles
   3.4 % → 1.5 % off), same-moment joints 1.2 → 0.8 cm; and they cost the far-range piece 0.9 → 2.9 cm. The cube graph
   alone (row 4) is already the geometry in which both known pieces are right. Whether F5's part is worth its time is
   his decision — these rulers do not see fine detail or edges.
6. **Nothing ties the two sides of the rack**: no valid closure between the cubes that saw its front and those that
   saw its rear (the bridges between opposite faces come out wrong by over 2 m). Its depth reads 0.78 m at the cube
   stage, 0.88 m with F5's cameras, 0.92 m with one factor, 1.02 m published; its real depth is not known here.
7. **Computed and unused**: the two layer files, the per-frame uncertainty maps, the depth graph, F5's residual file.
8. **The scale is decided five times** (section 4).

## 6 · Corrections to what the assistant said earlier the same day

- "With one scale alone the joints between passes fall 7.2 cm apart" and "Omega's camera steps are 6 % short for its
  depth": both carried an error of units (the records' depth was compared without scale_align's factor). Right
  figures: 3.2 cm at the cube stage, 1.5 cm with F5's cameras, and 3.1 %.
- "F6 moves the metre by 3.5 %": wrong for the same reason; F6's factor differs from the cameras' scale by 0.5 – 0.7 %.
- "The 30° limit silences 103 near keyframes that see through the plate": not verified when said, and false — no
  witness sees through it at any angle.
- "The far depth of those cubes is 10 % short": false — the cubes have it right; F6's factor shortened it.

## 7 · Not covered by this audit

The semantic stages (VLM, SAM3, projection), the floor stage, GenRecon; PointDiT's value; each rule of the vote on its
own; any session other than this one. The rulers need a tiled floor: on a session without one the same questions need
another reference.
