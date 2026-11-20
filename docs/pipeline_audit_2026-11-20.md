# Audit of the reconstruction pipeline, calculation by calculation — 2026-11-20

USER 2026-11-20: *"quiero una auditoría correcta y completa de que todos los pasos quedan coherentes, que un paso no
rompe lo que corrigió el anterior, que realmente los pasos agregan valor, coherencia y necesidad absoluta … Cada
cálculo, cada paso que damos en el proceso de reconstrucción debe ser necesario y coherente."* And, on the first
version of this document (which judged whole stages by their measured effect on one session): *"hay que juzgarlo de
dos maneras, conceptualmente y como se aplicó. Si está mal conceptualmente entonces ya está mal, si está bien
conceptualmente pero está mal aplicado, eso es lo más importante. También tenemos bufferstop para analizar, tenemos dos
corridas … esa conclusión debe ser sólida, matemáticamente correcta."*

Sessions: the copy of pccr 2026-08-24 (`pccr2408_prueba`: 2,772 keyframes, 170 cubes, 102.8 m of walk, indoors) and
bufferStop in its two runs (174 keyframes, 5 cubes, 13.4 m, outdoors: `bufferStop` of 2026-11-11 with the old F2, and
`bufferStop_prueba` of 2026-11-20 with one scale). Everything is measured read only; the scripts are in the session's
scratchpad (`serio/`).

STATE OF THIS DOCUMENT: the whole chain is covered — the intake (Part III), the cube stage (IV), the precision chain
F0, F4, F2, F5, F6 with the edge-detail model and the cleaner (V), the publication and what follows it (VI), the verification
layer (VII) and the whole (VIII: what is decided more than once, every calculation by its verdict, the corrections).
The other sessions (pccr 2026-08-31, zaragoza, observatorio, the original pccr 2026-08-24) are read from their own
stored files where a row quotes them. What no ruler of this audit reaches is listed as not judged (VIII · B).

---

## Part I · The criterion (what "coherent" means, in mathematics)

A reconstruction is, for each keyframe k, a pose T_k (rotation R_k, centre C_k), a depth map D_k, and one camera model
K. The published point of pixel u of keyframe k is

    P_k(u) = C_k + D_k(u) · R_k · K⁻¹(u)

**Coherence.** For two keyframes i, j that see the same physical point at pixels u_i, u_j, the error is
ε_ij = P_i(u_i) − P_j(u_j). A reconstruction is coherent when ε_ij ≈ 0 for every such pair — neighbours and revisits.

**The only freedom (the gauge).** Applying ONE similarity S (scale s, rotation, translation) to everything — every
centre C_k → S(C_k), every rotation, every depth D_k → s·D_k — moves the whole cloud by S and leaves every ε_ij at
zero. Nothing else does. So the scale is ONE number for the session.

**Lemma 1 (a scale applied to a part).** Let a step multiply the depth of keyframe k by a_k and move its centre to
C′_k. If the reconstruction was coherent before, then after it, for a point at ranges ρ_i, ρ_j along the unit rays
r_i, r_j:

    ε′_ij = (C′_i − C_i) − (C′_j − C_j) + (a_i − 1)·ρ_i·r_i − (a_j − 1)·ρ_j·r_j

It vanishes for a_i = a_j = a with C′_k = C_0 + a·(C_k − C_0) (the gauge). With the cameras left in place it does
not: the error is |a_i − a_j|·ρ for two views of the same range and direction, |a − 1|·|ρ_i − ρ_j| for one direction
and two ranges, up to |a − 1|·(ρ_i + ρ_j) for opposite views. **A depth factor that is not matched by the cameras
displaces what a keyframe sees in proportion to its distance.**

**Lemma 2 (re-integrating a walk).** A step that sets C′_{k+1} − C′_k = m_k·(C_{k+1} − C_k) with m_k not constant
changes the vector between two cameras i < j of a revisit by Σ_{k=i..j−1} (m_k − m̄)·(C_{k+1} − C_k). That sum is not
zero unless the constraint of the revisit is part of the solve. **A scale that varies along the walk, applied as an
open chain, opens every loop it does not know about.**

**Lemma 3 (one step after another).** Let step A minimise Φ_A and step B, after it, minimise Φ_B over variables Φ_A
depends on. B keeps what A established only if Φ_B contains Φ_A (as terms, or as a prior with A's information), or if
the directions in which Φ_B is flat are directions in which Φ_A is flat too (the gauge). Otherwise B moves, at no
cost to itself, exactly along the directions its own data do not determine — and Φ_A grows. For a bundle adjustment
after a pose graph those directions are: the length of the camera path where the tracks carry no parallax, and the
relative placement of two passes that share no track. They are what the graph's seams and closures fixed.

**The two judgements of every calculation.**
- CONCEPT: what it changes (poses, depth, scale, camera, which points exist), what model it assumes, and whether by
  the lemmas it can keep ε_ij at zero given what the earlier steps fixed. A calculation that changes a scale or a
  position of a PART without the earlier constraints is wrong by construction.
- APPLICATION: whether the code computes what the concept says, with which numbers, and what it measurably does on
  the two sessions.
- NECESSARY: removing it leaves something undetermined or measurably worse. REDUNDANT: another calculation decides
  the same quantity and the result does not change.

---

## Part II · The references

**The floor tiles of pccr** (a physical object of constant size on one grid; each keyframe's own tile size reads its
depth's scale to 0.2 – 1.1 % per stretch; the joints drawn by two keyframes read their agreement along the floor; the
squareness reads the camera model). **The two known defects on their own pixels** (the white cabinet seen from 5.9 m
against itself seen from 1 m — "the plate"; the rack seen from its two sides).

**A ruler that exists in any scene** (needed for bufferStop): one keyframe's pixels carried into another keyframe with
a geometry; ALONG the ray, the carried depth against the other keyframe's own depth (%); ACROSS the ray, the shift
that best matches the carried image to the other keyframe's image (px → cm at the surface). Validated on pccr against
the tiles — the same order between stages for pairs of different passes:

| Geometry (pccr) | tiles: joints of two passes | pair ruler, across the ray | pair ruler, along the ray |
|---|---|---|---|
| the cube graph, Omega's camera | 3.2 cm | 1.73 cm | 0.92 % |
| the same cameras, F5's camera model | 2.8 cm | 0.74 cm | 0.59 % |
| + F5's cameras, depth at the one scale | 1.5 cm | 0.31 cm | 1.26 % |
| + F6's factor per keyframe (published) | 3.4 cm | 1.06 cm | 1.75 % |
| F6's factor, one for the session | 1.4 cm | 0.15 cm | 1.00 % |

Units (checked on the cameras, exact): the cubes' stored depth (the Omega records) is in the cubes' own units; on pccr
epoch 0 = × 0.97158 (scale_align), epoch 4 = × 0.95456 (× F2's 0.98248), F5's cameras = × 0.95392.

---

## Part III · Before the cubes: the intake, calculation by calculation

How the rest of this document was made: four read-only readings of the code (the intake and the cube plan; the cube
stage; F0, F4 and the publication; every test, gate and judge of the chain), each calculation with its file and line;
fifteen of their statements were read again in the code by the assistant, all as reported. The numbers under
"Evidence" are measured on the sessions' own files.

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| The frames: every video frame, the container's rotation applied, long side ≤ 1024 (area resize), JPEG 95 (`intake/frames_manifest.py`) | defines the pixels every later step reads; a uniform resize keeps the camera a pinhole | as coded; the cap is read from the live configuration at upload, not from the job's frozen one (it is recorded in the manifest and F0 checks the frames against it) | pccr's source video is 464 × 832 (0.39 megapixels, no resize); the other three 1024 × 576 | NECESSARY, sound |
| I0, "usable": a frame is dropped when its mean brightness is under 40 or over 220, or over 25 % of its pixels are clipped (`intake/quality.py`) | **WRONG AS A GATE.** Brightness is not information. A dropped frame can never be a keyframe, so a run of them cuts the walk into two pieces that share no view; no later step can restore an overlap that was never reconstructed (a seam needs shared frames) | fixed bounds | pccr 2026-08-24: 144 frames dropped = the two dark runs = the only two gaps over 8 frames (130 and 14); the two cubes across the first were published 36 and 32 cm low. The first pass, which reads every frame, followed the camera through it (0.80 m, 105°). With the gate open (the copy): 13,692 of 13,692 usable, no gap over 8 frames, the turn reads 0.85 m (first pass 0.87). On the other four sessions it dropped nothing | **conceptually wrong; harmful on the one session where it acted, inert on the others.** `config.yaml` still holds 40 / 0.25 — the copy ran with them open in its own frozen configuration |
| I0, sharpness: the rank of each frame's FFT energy among the session's frames → which frame of a band becomes the keyframe | an order inside the session, no threshold | as coded | not judged by a ruler | sound. (`laplacian` and `inter_frame_diff` are computed and read by nothing on this path) |
| The focal probe: 16 frames through DA3, the median K → I1 only (`intake/focal.py`) | I1 turns angles into pixels with it: x % of focal error = x % in I1's angle bars, inside their own ± 25 % band | as coded | against F5's final camera DA3's focal is + 6.0 / + 7.5 % (pccr copy), + 1.9 / + 3.0 (pccr 08-31), − 1.5 / + 3.1 (bufferStop), − 2.8 / + 1.3 (zaragoza), − 3.2 / + 1.1 % (observatorio): as near as Omega's (Part IV) and on both sides | NECESSARY for I1. The same camera is then estimated three more times (the first pass per record, Omega per keyframe, F5) |
| I1, the reading: LK tracks from the window's anchor (forward – backward ≤ 1 px), a rotation fitted to them, "parallax" = the 0.9 quantile of what the rotation does not explain (`intake/parallax.py`) | what a rotation cannot explain is baseline × the scene's depth RELIEF, not baseline. A camera moving sideways by t before a flat surface at distance d shifts the image by f·t/d, which a rotation of angle t/d reproduces up to its second-order term (θ·x², at most θ·tan²(FOV/2)): that motion reads as a rotation | as coded; the rotation is fitted without a robust loss | this bar chose 1819 of 2772 keyframes on the pccr copy (racks at 1 m: strong relief), 281 of 547 on pccr 08-31, 119 of 174 on bufferStop, 39 of 279 on zaragoza, 14 of 195 on observatorio | sound as "enough relief seen from two places"; blind to a walk along a flat scene — which the next row covers |
| I1, the 8-frame rule (`stride_fill`; his, 2026-11-08): never more than 8 video frames without a keyframe | a floor in TIME, independent of what I1 reads: neighbours overlap whatever the scene | as coded; a block with no usable frame gets none (the gate above) | it placed 28 % of the keyframes on the pccr copy, 46 % on pccr 08-31, 25 % on bufferStop, **83 % on zaragoza, 88 % on observatorio** | **NECESSARY** — on two of the five sessions it IS the selection |
| I1, witness frames (parallax ≥ 0.22° from the last one) → F4's witness windows | denser views between keyframes | as coded | 8,577 on the pccr copy; see F4 (Part V) for what the chain in force does with them | see F4 |
| The first pass (AMB3R-SLAM over every frame, the gate above ignored): a pose per frame and sparse depth records; ONE metre from ≤ 16 frames (`reconstruction/amb3r_probe.py`) | an independent first reconstruction used only as a map of WHO SEES WHOM (the fill, the co-visibility, the revisits) — never as geometry. Its drift over a long walk can lay two places on each other or part one place in two | as coded; its metre only places the 15 m range on its own map | the revisits it gives are bridged (Part IV): 98 – 100 % of those bridges agree with the cube graph on pccr 08-31, zaragoza and observatorio, 71 % on bufferStop, 45 – 49 % on pccr 08-24 | NECESSARY: the only source of revisits. About 1 h on pccr 08-24 |
| The coverage fill: one more keyframe where the map has points that no two keyframes see ≥ 2° apart (`reconstruction/coverage_fill.py`) | sound in principle; computed on the sparse records with each view standing in for its nearest record, and its tolerance and stopping bar come from the very samples it judges | as coded | it added **0** keyframes on pccr 08-31, zaragoza and observatorio, 2 of 174 on bufferStop, 16 – 18 of about 2,770 on pccr 08-24 | **REDUNDANT as applied**: the 8-frame rule leaves it nothing to add |
| Turn frames: extra views inside a rotation and between keyframes → each cube's pass (context) and a SECOND pass per cube (witness views for F6's vote) (`intake/turn_frames.py`) | a rotation sweeps the field of view fast; more views keep neighbours overlapping | only I1's own windows are walked: an interval closed by the 8-frame rule gets none | 504 context and 1,544 witness views on the pccr copy | not judged by a ruler. The second pass doubles the inference of every cube for the vote's benefit; its share of the vote's confirmations is not measured |
| I3: DA3 over windows of keyframes → (a) a DA3 depth per keyframe, the metric instrument of every scale step; (b) the windows chained rigidly → the walk's length and chainage (`intake/walk.py`) | (a) an outside metric reference: needed ONCE. (b) feeds only F2's curve along the walk and a metres-per-unit figure of the "metres" cut | as coded | with F2 at one scale and the cut by co-visibility, nothing of (b) decides anything; F2 generates its windows again (25.8 GB, about 30 min of the card on pccr 08-24) to fit a curve that is no longer applied | (a) NECESSARY for one number; (b) **REDUNDANT today** |
| Co-visibility on the first pass's map: c(a→b) = the share of 2000 points of a's record that b's record sees within 25 % in depth, both within 15 m (`reconstruction/cubes.py`) | "do these two see the same place at all", on a coarse map: sound for that | one fixed band (0.25); the session's own measured tolerance is computed beside it (`tol_rel`) and read by nothing | read by the cut, the revisits, F4's loop windows and F6's witness lists | NECESSARY |
| The cut "by co-visibility": a cube takes units while the sum of 1/ℓ stays under 13.67 (ℓ = how many keyframes ahead still share ≥ 30 %) AND it holds ≤ 64 keyframes | a cube = what can be seen together | the budget 13.67 was calibrated on another instrument (DA3 windows); on this map the lengths are far longer | **on the six sessions no cube is closed by the budget**: the largest sum in any cube is 5.1 – 7.3 of 13.67, and the budget alone would let a cube run 84 – 266 units. Every cube is closed by the cap of 64 | the budget is **REDUNDANT as applied**. The cut IS his numbers: 64 keyframes, sharing ½ (¾ in the copy), at least 32 |
| The owner of a keyframe: the cube in which it lies deepest | its record and pose come from the pass where it has the most neighbours on both sides | as coded | — | NECESSARY, sound |
| Revisit pairs: every pair with c > 0 that no F4 keyframe window ties, the best of each neighbourhood of ± 12 keyframes (`precision/tracks.revisit_pairs`) | "c > 0" is the weakest evidence the map can give — one point in common, on a map that drifts — and the pair is then trusted to a bridge | as coded | pccr copy: 429,764 pairs with c > 0 → 1,908 revisits → 1,726 bridges. Their agreement with the cube graph by the distance between the two cameras: 76 – 79 % under 3 m, 66 % at 3 – 4 m, 55 % at 4 – 5 m, 40 % at 5 – 6 m, 16 % at 6 – 8 m, 4 % at 8 – 10 m, 0 % beyond 10 m | NECESSARY; the bar lets through pairs whose bridges are worthless (566 over 6 m: 42 min of the card, no gain on the held-out bridges) |
| `metres_per_unit`, `gravity_frame` (the viewer's boxes), `tol_rel`, `coverage_fill.json`, `turn_frames.json`, the coverage warnings, `frame_quality.json`, `da3_frames.json` | — | written | read by nothing that touches the geometry on this path | UNNECESSARY for the reconstruction |

**The intake in one line**: the keyframes are, in practice, "one every ≤ 8 frames, more where the relief is strong";
the first pass is needed for the revisits alone; one calculation is wrong by concept (the brightness gate), and two
decide nothing on any session (the coverage fill, the co-visibility budget of the cut).

---

## Part IV · The cube stage, calculation by calculation

(`vendor/VGGT-Long/loop_utils/cube_run.py` and what `workers/map_worker.py` does right after it.)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| The cube's pass: one Omega inference over its keyframes and context views; a pose head and a depth head; the points are the depth unprojected with the pose head's own camera | inside a pass the poses, the depth and the camera are three outputs of one network. No step of this stage makes them agree: nothing enforces P_i(u_i) = P_j(u_j) inside a cube | as coded (`vggtomega_adapter._unproject`) | Omega's focal against F5's final one is short on EVERY session: − 4.4 / − 6.1 % (pccr copy), − 4.6 / − 5.4 (pccr 08-31), − 3.2 / − 5.3 (bufferStop), − 2.2 / − 5.6 (zaragoza), − 5.4 / − 7.8 % (observatorio). On pccr the tiles are 3.4 % off square with it and the camera steps 3.1 % short of what the depth implies | NECESSARY (it creates the geometry). Its internal disagreements — a focal 2 – 8 % short, steps 3 % short of the depth — are what F5 exists to remove |
| The DA3 lock: each cube × the median of its anchors' DA3 / Omega near-depth ratio | **A GAUGE CHOICE THE GRAPH REMOVES.** Proof: rescale cube k's units by λ_k. An edge M_ab (cube a → cube b) becomes λ_b·M_ab·λ_a⁻¹ and the unknown C_k becomes C_k·λ_k⁻¹; the edge's error E = C_b⁻¹·C_a·M⁻¹ becomes λ_b·E·λ_b⁻¹, and its residual — the displacement at the centroid, the rotation × the lever, the log scale × the lever, all over σ, the four in cube b's units — does not change. λ_0 of the first cube is one global scale, which scale_align then decides again. So the placed geometry does not depend on the locks | what DOES depend on them: the 15 m range of the lock itself, of the seams and of the bridges, each evaluated in the cube's own locked units | the locks are unreliable: a cube's metre against the first pass runs × 0.45 … × 1.69 on the copy (twice too big at the two doors); between two neighbouring cubes × 0.70 … × 1.52 (the seams' own scales say the same: correlation − 0.92 with the ratio of the two cubes' metres). After the graph the whole session carries the first cube's lock, and scale_align finds it off the session's own DA3 median by − 17.6 % (pccr 08-31), + 6.8 % (observatorio), − 2.8 % (the copy). In a cube locked twice too big the "15 m" range cuts at 7.5 real metres | **REDUNDANT by construction**, and its one real effect breaks his 15 m rule where the lock is wrong. (Its effect on the fits was not measured: the cubes are deleted) |
| The seams: between two cubes that share keyframes, the similarity that maps one cube's points onto the other's AT THE SAME PIXELS (robust Umeyama, ≤ 200,000 points within range) | the same pixel in two passes is the same point: the cleanest correspondence the chain has. Right. "Kept free" (a similarity with its own scale): right, since no cube's scale can be trusted | σ of an edge = the fit's median residual on the points it was fitted to — the two cubes' non-rigid disagreement, used as the edge's weight; the number of shared frames is stored and not used; a seam is never rejected | the copy: 510 seams, σ 0.5 – 17.9 cm (median 2.0); the graph leaves them 1.65 cm apart at the surfaces (median; 90 % under 5.1 cm; 9 over 10 cm, 32 cm at the worst — the first door) | **NECESSARY, sound** |
| The seams' scale bound (1.1 × the 95 % interval of the seams' own scales) | — | computed, logged, flags a closure; since 2026-11-19 it gates nothing | — | report only |
| The bridges → closures: for each revisit pair, ONE joint Omega pass over 9 + 9 keyframes; each side fitted onto its cube; the closure = fit_a ∘ fit_b⁻¹ | right for two windows that truly share content: the joint pass then reconstructs both in one frame. A closure asserts ONE thing — that the pass placed window b right relative to window a — and **its weight does not measure it**: σ = hypot(residual_a, residual_b) says how well each window matches its OWN cube. Two windows with nothing in common are each fitted well and joined arbitrarily | every bridge is a loop edge; the selection is the intake's "c > 0"; a verification by split halves exists in the code (`verify_loop`) and is not called on cubes | the copy: 774 of 1,726 closures agree with the solution (3 σ). By the closure's own σ: 90 % under 3 cm, 72 % at 3 – 6, 41 % at 6 – 10, 18 % at 10 – 20, 5 % beyond. By distance: Part III. The robust loss carries the wrong ones: with the closures the cabinet's far copy goes from 54.5 to 0.9 cm, the rack's side from − 88 to + 1.7 cm | **NECESSARY — the calculation that adds the most on a long walk.** Its weight is mis-specified and its selection lets worthless bridges through (129 min of the card for 1,726; 42 min for the 566 over 6 m) |
| The graph: 7 unknowns per cube, the first cube the gauge; seams squared, closures under a Cauchy loss per component; Levenberg – Marquardt with finite differences (`chunk_graph.py`) | the right problem: a pose graph of similarities | no judge of the result — a leave-one-out judge exists in the code and is not called on cubes. A solve that does not converge falls back to the seams' tree, a geometry with every loop open, with a log line | converged on the three runs read (the copy, bufferStop's two). bufferStop: 24 – 32 closures pull its 4 seams from 0 to 2.7 – 2.8 cm (median) — and after F5 it makes no difference (below) | NECESSARY, sound. The fallback should stop the run instead of publishing a degraded geometry |
| The placement: one similarity per cube | a gauge per cube: cameras and depth of a cube move together (Lemma 1 satisfied) | as coded | — | NECESSARY, sound |
| The two-copy uncertainty (a keyframe's owner copy against one other cube's copy) | a measurement of the cubes' repeatability | one number of it (the session's median) is read later as the noise bar of the floor step and the cloud metrics; the per-frame maps are deleted unread | — | report; one number is used |
| The depth graph: a·z + b per keyframe, from pairs of keyframes, the cameras fixed | a depth correction per keyframe that the cameras do not follow — the family Lemma 1 warns about (F6's factor per keyframe is its twin); gated by held-out pairs | as coded; it holds every keyframe's point map in memory to reach its verdict | **its verdict is SKIP on the six sessions**; 6 min on the copy | **UNNECESSARY as applied** — it has never acted |
| The per-cube clouds (PLY + origins, own confidence bar: the 20th percentile and 10 %) | a preview; not the published cloud | read by the raw preview, by orient's floor offset, by the epoch-0 cloud; rewritten by scale_align and by orient; deleted when the precision core ends | 22 GB on the copy, written three times | not part of the published geometry; kept for the preview and one number |
| The per-keyframe records: depth along the pose's axis from the placed points of the OWNER cube, pose, K, confidence | the depth every later step reads | written ONCE, in the cube world's units, and never rescaled: scale_align and F2 move the cameras and leave the depth that belongs to them | between two steps the stored state is not a coherent reconstruction (Lemma 1): every reader has to repair the units — F2 reads the factor from a marker, F6 re-fits it | NECESSARY; **the scale should reach the depth when it reaches the cameras** |
| scale_align: ONE s = the trimmed median, over the keyframes, of DA3 / Omega on the near, DA3-confident pixels (`reconstruction/scale_align.py`) | the gauge's scale is one number; a robust median over the whole session is the right estimator of one number from a noisy instrument | applied always; no acceptance test; its diagnostics (cross-validation, jackknife) are read by nothing in force | s = 0.972 (the copy), 0.963 (pccr 08-24), 0.824 (pccr 08-31), 1.006 / 1.002 (bufferStop's two runs), 0.997 (zaragoza), 1.068 (observatorio). Its truth: only a known length can say — pccr's tile reads 0.585 m in the published cloud | **NECESSARY — the one metric decision of the chain** |
| orient: "down" = the mean of the cameras' image-down axes; the floor at the 1st percentile of height (`reconstruction/orient.py`) | an ASSUMPTION about how the phone is held, not a measurement of gravity | refused under a mean alignment of 0.7 | the floor step, after the segmentation, still finds 0.44° (pccr 08-24), 2.75° (zaragoza), 3.25° (observatorio) of tilt, and 92.7° on bufferStop (a video stored sideways; mended at extraction for future sessions). A cloud-only run keeps orient's (the copy: 0.83°) | NECESSARY as a first frame; "down" is decided again by the floor step |
| `scale_drift`, `zoom_scale_fix`, `exact_seam_align`, `sigma_seam`, `chunk_graph: sim3`, `loop_closure` (set in the configuration, read by nothing on the cube path); the stages `pose_refine`, `fine_register`, `bundle_adjust`, `dense_fusion` (disabled); `chunk_graph.json` but its poses, `metric_lock.json`, the per-cube (s, R, t) of `chunk_sim3.json`, `camera_poses.ply`, the rigid fit of every seam | — | written or configured | read by nothing | dead configuration and unread outputs |

**bufferStop's two runs at this stage.** The two runs' cube-stage cameras differ by 0.68 cm (median; 1.2 cm at most)
after one similarity — the closures' rule changed between them (24 closures against 32). The old F2 then moved one
run's cameras by up to 9 cm. After F5 the two runs coincide to 0.1 mm. Where the tracks determine the geometry, F5
erases the cube graph's differences too, not only F2's.

**The cube stage in one line**: seams + closures + graph are the chain's skeleton and they are right; the DA3 lock per
cube is a gauge choice the graph removes (redundant, with one bad side effect); the depth graph has never acted; and
the stage hands on a depth in units its cameras no longer have.

---

## Part V · The precision chain, calculation by calculation

### F0 — the session camera (`precision/camera.py`)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| The session's K = the median of Omega's per-keyframe K, on the native frame | a start for F5; one camera for the session | as coded | 2 – 8 % short of F5's result on every session (Part IV); the intake's DA3 focal is as near and not one-sided — two starts exist, and F5 decides either way | NECESSARY as a start |
| The checks: every frame of the size the manifest declares, the records on the declared grid | integrity | they stop the run | — | sound |
| `fx_spread_warning` | — | written | read by nothing | report only |

### F4 — the tracks (`precision/tracks.py`)

On the copy: 4,712 windows — 230 of keyframes, 1,908 of loops, 2,574 of witnesses; 31.85 M tracks seen in at least two
keyframes; F5's solve keeps 12.84 M landmarks. By the per-window times of the original run the three kinds take about
9 %, 60 % and 31 % of F4, which is half of the precision core (5.8 h on pccr 08-24). Counted on the tracks' own file
(`serio/f4_windows_by_kind.py`); "reach 2°" = the angle between the rays of a track's first and last keyframe against
F5's own triangulation bar:

| Windows | tracks in ≥ 2 keyframes | reach 2° | share of what reaches 2° |
|---|---|---|---|
| keyframes (the walk's backbone) | 1.85 M | 1.16 M (62.8 %) | 7.6 % |
| loops — tracks seen in BOTH visits | 0.29 M | 0.29 M (98.9 %) | 1.9 % |
| loops — tracks that stay inside ONE visit | 16.49 M | 13.16 M (79.8 %) | 86.4 % |
| witnesses (two consecutive keyframes) | 13.23 M | 0.62 M (4.7 %) | 4.1 % |

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| Keyframe windows: 24 keyframes, step 12, the tracker started from 2 query frames | feature tracks over a sliding window: the data of a bundle adjustment. Right. A track lives in its window (a point seen in two windows is two landmarks); consecutive windows are tied through the 12 cameras they share | as coded | 1.85 M tracks, 16.0 M observations — 9 % of the keyframe observations handed to F5 | NECESSARY — the walk's backbone, and the SMALLEST part of what F5 receives |
| Loop windows: ± 4 keyframes around each of the two keyframes of a revisit pair, tracked together | the ONLY way a revisit reaches F5. Right in concept — but a tracker follows a point only between views that look alike: not the cabinet from 5.9 m and from 1 m, not a rack from its two sides | every revisit pair gets a window, whatever its two views show | **1,462 of the 1,908 loop windows yield no track seen in both visits.** The tracks that do link two visits: 288,390 (median 27° between their rays). The other 16.49 M stay inside one visit: the same neighbourhoods tracked again — a keyframe sits in about 12 loop-window sides against 2 keyframe windows — and they are 71 % of the keyframe observations handed to F5 and about 86 % of its landmarks | the linking tracks are NECESSARY and few. **Three loop windows in four link nothing**; F4's largest cost goes to tracking the same places again, and F5's adjustment is, in its bulk, those repeats: dense where the room was walked several times, thin where it was walked once |
| Witness windows: the witness frames between two consecutive keyframes, with both | two things come out: tracks seen in the two keyframes, and the witness frames' own observations | the second feed F5's localisation of the witness frames (`witness_poses.txt`), which the chain in force does not use: F6 places its interval witnesses with Omega's relative pose to their anchor keyframe, and the only reader of those poses is the sweep step, dropped from this chain | 13.23 M tracks seen in both keyframes, **of which 4.7 % reach F5's 2°** (median 0.63°) | **REDUNDANT as applied**: about a third of F4's time for 0.62 M usable tracks and for poses that nothing reads |
| An observation is kept on the tracker's own visibility > 0.2 | no forward – backward test and no geometric test; outliers are left to F5's Huber loss, which does not redescend: a CONSISTENT set of false matches (identical racks, identical tiles) would count as evidence | as coded | no harm found; not tested | sound as far as measured |
| The query filter at depth edges (5 %) | a query on a depth edge follows two surfaces | as coded | 1.37 M queries dropped on the copy | sound |
| The held-out split: one track in five, by a hash of its query | held-out tracks share the windows, frames, tracker and cameras of the fit tracks: they measure the cameras' LOCAL reprojection consistency | as coded | — | sound for choosing F5's rung; **not a certificate of the geometry**: a path cut at a door or an opened loop changes no local reprojection (Lemma 3's directions) |
| `obs_uv_sigma` (the tracker's own uncertainty of each observation), `obs_score` (a second copy of the visibility) | a weighted adjustment would be the statistically right use of the first | written | F5 weights every observation the same; the second is a duplicate | information not used; a duplicate |

**F4 in one line**: the tracks are the right data, but as applied about three quarters of the stage's time goes to
loop windows that link no two visits (about 46 %) and to witness windows whose tracks F5 hardly accepts (about 31 %),
while the tracks a long walk needs most — those that join two passes — are 1.9 % of F5's landmarks.

### F2 — the gauge (`precision/gauge.py`)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| Rows: per DA3 window, the gain of DA3's depth over Omega's (Huber, bootstrap σ) | an external metric reference read against the cubes' depth | as coded; "Omega's per-frame depth is written before the global scale is applied, so a row is log(inst/Omega) − log s_applied" | — | sound as a measurement |
| The model x(c): a scale along the walk, piecewise linear, λ by leave-one-window-out | assumes DA3's metre is right LOCALLY, stretch by stretch | — | tiles: DA3 asks × 0.83 on video 1450–2400 where the tiles ask × 0.97; applied, the tile runs from 0.51 to 0.65 m in the original cloud (spread 31 %, 8 % without it) | **wrong assumption**: DA3's local metre is off by up to 15 % |
| The application along the walk: `continuous_transforms`, C′_k = C′_{k−1} + ½(s_k + s_{k−1})·(C_k − C_{k−1}) | **WRONG BY CONSTRUCTION (Lemma 2)**: an open chain, written for a walk without revisits; the cube graph has closed revisits since 2026-11-08 | exactly that loop | pccr: joints of two passes 3.2 → 6.4 cm (90 %: 6.1 → 26.9), the rack 0.78 → 1.49 m, its side +1.6 → −29.6 cm | conceptually wrong — removed 2026-11-20 |
| The application with one scale (`profile: constant`) | a gauge transformation: by Lemma 1 it can break nothing | cameras scaled about the first one; the depth records are NOT rescaled (see F6) | pccr: not one ruler moves; bufferStop: the same | harmless — and REDUNDANT: scale_align already fixed the metre with the same instrument; the second estimate differs from the first by 1.75 % (pccr), 1.0 % (bufferStop), which is the scatter of the estimator, not information |
| What F5 then does with F2's result | — | — | **bufferStop, two runs**: the old F2 moved the cameras up to 9 cm (× 0.946 … × 1.037 along the walk), the test run applied one scale — after F5 the two runs' cameras coincide to **0.1 mm** (0.2 mm at most) up to one scale factor of 1.0039, the same camera model to 4 digits, the published points per keyframe within 0.1 % | where the tracks determine the geometry, **F5 erases whatever F2 did**: F2 is redundant. Where they do not (pccr's long corridor walk) F2's profile survives — and it was wrong |

**F2 in one line**: its only legitimate output is one number, which scale_align has already produced; everything
else it computed was either erased by F5 (bufferStop) or wrong (pccr).

### F5 — the joint refinement (`precision/refine.py`, `precision/ba_torch.py`)

The cost, as coded (`ba_torch._Problem.total_cost`):

    ½ Σ_obs ρ_Huber(‖reprojection‖²)  +  ½ Σ_k ‖(C_k − C_k⁰) / σ_k‖²,     σ_k = σ_rel · sqrt(Σ_{i≤k} d_i²)

over one SE(3) per keyframe, the landmarks, and the camera freedom of the rung. C_k⁰ = the cameras F2 left; d_i the
steps between consecutive cameras; σ_rel = the scatter F2 measured of its DA3 windows (0.0702 on pccr).

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| Reprojection of the tracks under a Huber loss, poses + landmarks free | the classical bundle adjustment: right | as coded (float64, Schur complement, deterministic) | pair ruler, different passes, across the ray: pccr 0.74 → 0.31 cm, bufferStop 2.23 → 0.31 cm; tiles 2.8 → 1.5 cm | **right, and it adds real value on both sessions** |
| The position prior: ONE INDEPENDENT prior per keyframe at its F2 centre, σ_k growing along the walk | **MIS-SPECIFIED (Lemma 3)**. σ_k is the MARGINAL spread of a chain of relative priors (each step known to σ_rel), but the chain's whole content is its correlations: it holds the step i→i+1 to σ_rel·d_i. Independent priors hold that step to sqrt(σ_i² + σ_{i+1}²). And the model is an OPEN chain: it knows nothing of the graph's closures | `prior_sigmas` + `prior_terms`: exactly independent, isotropic | pccr: σ from 0.03 mm (first keyframe) to 15.0 cm (last); a step is 3.6 cm: the prior "holds" it to ± 376 % of its length (the chain it derives from: ± 7 %). bufferStop: 1.6 mm … 8.3 cm, ± 131 % | **the calculation that is wrong in F5.** Nothing holds the length of the path or the relation between passes except the tracks |
| … its consequence where the tracks carry no information | Lemma 3: the solve moves freely along what its data do not determine | — | pccr: the camera path through door 1 (video 2468–2585) is cut to × 0.48 – 0.59, through door 2 (3566–3640) to × 0.56 – 0.74, video 4740–4870 to × 0.40 – 0.52 — stretches with 6,000 – 39,000 tracks across them (tracks there, no parallax: a blank surface against the lens). Tracks linking different passes: 99,017 of 31.9 M (0.31 %) | **right concept, wrong application** — the origin of the displaced copies |
| … σ_rel taken from F2's instrument scatter | a quantity of the scale-along-the-walk model, which no longer exists | still read from gauge.json | — | incoherent with F2 as it is today |
| Rungs R1, R2: the camera's focal, centre and lens freed, one camera for the session | self-calibration; judged on held-out reprojection, which a wrong focal with compensating structure can also satisfy | taken on both sessions | tiles (pccr): with Omega's camera the tiles are 3.4 % wider than long, with F5's 1.5 % the other way — better, overshot; the far-range piece 0.9 → 3.1 cm with F5's camera model on the graph's cameras. bufferStop: fx +3.3 %, fy +5.6 %, identical in both runs | ADDS VALUE; its judge sees reprojection, not metric truth |
| The continuation solve after every rung (the solver's own error, for the judge) | a measurement | 55 % of F5's solve time | — | necessary to the judge as defined; moves no geometry |
| The held-out judge between rungs (cluster bootstrap by keyframe) | his rule | as coded | — | sound |
| Witness frames localised against the fixed landmarks | PnP on fixed structure | — | they vote in F6 | not judged here |
| `refine_residuals.npz` | — | written; F6's input loader (`depth_sweep.load_inputs`) opens it — and stops the run if it is empty — to compute a pixel tolerance that only the sweep step used, and the sweep is not in this chain | F6's own modules never read that tolerance | UNNECESSARY (the loader must stop asking for it first) |

**Does F5 decide the geometry, or its start?** bufferStop answers: two different starts (the old F2's cameras, the
one-scale cameras) → the same solution to 0.1 mm. There the tracks determine everything and the prior is irrelevant.
On pccr the two runs' keyframes differ in size by × 0.90 … × 1.17 along the walk: the start survives. A long walk
through corridors and doors has directions the tracks do not determine (a slowly varying scale; the path's length at
a blank door) — precisely where F5 needs the constraints the cube graph already has, and does not carry them.

**F5 in one line**: right concept, valuable on both sessions; ONE calculation mis-specified — the prior. It should
carry the cube graph (relative constraints between neighbouring cameras with the cubes' own uncertainty, and the
closures), so that the tracks decide wherever they can and the graph holds wherever they cannot.

### F6 — the depth factor ("bend", `precision/depth_on_f5.py`)

k_i(u, v) = c0 + c1·ũ + c2·ṽ multiplies keyframe i's Omega depth; fitted (Huber IRLS) on F5's landmarks of
keyframes i − w … i + w: the rows are z_landmark / z_Omega.

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| That the depth needs a factor at all | scale_align and F2 rescale the cameras; the Omega records stay in the cubes' units. By Lemma 1 the depth must take the SAME factor — which is KNOWN (× 0.95456 on pccr), not something to estimate | F6 does not apply the known factor: it fits c0 against F5's landmarks and so re-derives it (pccr: one fitted factor 0.9483, the cameras' scale 0.9546) | tiles: anything from × 0.9477 to × 0.9546 reads the same (1.4 – 1.5 cm between passes) | the factor is NECESSARY; estimating it is REDUNDANT — and the steps that set the scale should carry it to the depth themselves |
| c0 PER KEYFRAME (window ± w) | legitimate only if Omega's depth has per-keyframe scale errors larger than the landmarks' own uncertainty, AND if F5's local scale can be trusted everywhere; otherwise by Lemma 1 it displaces what the keyframe sees in proportion to distance | as coded | pccr: c0 from 0.10 to 2.18; × 0.87 around door 2 (where F5 cut the path) → the cabinet at 5.9 m published 45 cm short; joints of two passes 1.5 → 3.4 cm; the rack 0.88 → 1.02 m. **bufferStop** (F5 well determined): per keyframe is slightly BETTER than one factor between neighbours (along the ray 0.56 % against 1.05 %, across 0.46 against 0.57 cm) and equal between passes (0.20 cm both) | **it stands or falls with F5.** Sound where F5 is determined (bufferStop: a small gain), harmful where it is not (pccr) |
| The verification of c0 "on held-out rows" | CIRCULAR: the held-out rows are landmarks of the same F5; they cannot detect F5's own error | every keyframe "verified", including c0 = 0.10 and 2.18 on pccr and 0.001 on zaragoza | — | **wrong as a safeguard** |
| The choice of the window w by held-out error | chooses how local the factor is | pccr: 3.01 %, 3.01, 3.01, 3.02, 3.04 % for ± 0, 1, 2, 4, 8 — nothing to choose with; bufferStop: 1.09 → 1.31 % — a real preference for ± 0 | — | sound only where the error depends on w |
| c1, c2 (a tilt of the factor across the image), kept only when significant | corrects a residual of the camera model | kept on few keyframes | inside a keyframe the published factor of points over 4 m is × 0.9987 that of points under 2 m | negligible |
| A keyframe with too few rows keeps k = 1 | then its depth stays in the cubes' units — 4.5 % off the cameras on pccr (Lemma 1) | none on pccr; possible | — | wrong fallback: it should be the session's factor |

Across the sessions, from each one's own report (c0 per keyframe: median, 5–95 %, extremes): bufferStop 0.985,
0.955 – 1.039, 0.936 … 1.057 · observatorio 1.072, 1.044 – 1.090 · zaragoza 0.954, 0.929 – 0.989, **0.001 … 1.223** ·
pccr 2026-08-31 0.777, 0.747 – 0.839 · pccr 2026-08-24 original 0.960, 0.813 – 1.073, **0.103 … 2.311** · its copy
0.950, 0.874 – 1.014, **0.100 … 2.183**. With the OLD F2 a factor per keyframe was the only way F2's scale along the
walk reached the depth: the two belonged together. With one scale, nothing legitimate is left for it to carry except
Omega's own per-keyframe error (about 1 % on bufferStop, 0.3 % on pccr).

**F6's factor in one line**: one factor for the session is the coherent choice today (under test on pccr); a factor
per keyframe becomes legitimate again only once F5 is corrected, and then with a safeguard that is not F5 itself.

### F6 — the vote (`precision/f6_torch.py`, `depth_on_f5.edge_keeping_vote`)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| A pixel is published with ≥ 5 witnesses that confirm its depth (reprojection ≤ 0.301°, depth within τ) | a multi-view test of coherence, pixel by pixel: right as a NOISE filter | as coded | pccr: kept 49.7 % of the valid pixels | NECESSARY |
| … what it cannot see | by Lemma 1 a group of keyframes with a common wrong factor and small steps between them agrees with itself: ε′ ≈ (a − 1) × the step | — | no witness sees through the plate, at any angle (census with the full depth maps): the far pass confirms itself | not a safeguard against a stretch at the wrong scale |
| τ = the 75th percentile of the session's own neighbour disagreement | measured, not fixed | 1.90 % on pccr | — | sound |
| The triangulation band [1.5°, 30°] | the lower end makes a confirmation TEST the depth; the upper end comes from appearance-matching stereo, which this vote is not | 30° is a literature number (the assistant's, 2026-11-08) | without it ≤ 1 % of the points change | the upper limit is unjustified; harmless so far |
| The majority (confirm > see through) and the occlusion veto | free-space reasoning: right | — | 8.3 % and 5.6 % of the valid pixels | sound |
| "In place": a pixel NOT kept whose witnesses' surfaces agree at its own depth is published where it is | meant for pixels short of support | the code takes every pixel not kept — including those the majority or the veto rejected | 8.8 % of the valid pixels | to check: it can re-admit what the two rules above rejected |
| The published depth = the median of the pixel's own depth and the agreeing witnesses' | a small smoothing within τ | — | — | sound |
| Who the witnesses are: the keyframes at ± 3, 6, 12; the turn frames of the cubes' second pass, placed with Omega's relative pose to their anchor; every keyframe the first pass's map says shares anything (c > 0) | every witness is the same network's depth, with the same factor, on the same F5 poses: the vote tests their mutual consistency, not their truth | as coded | — | sound as a noise filter (the limit is two rows above) |

### F6 — the edge-detail model detail (its detail module)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| A monocular detail network on tiles of each keyframe, fitted to the bent depth tile by tile (affine); its high-frequency part replaces Omega's near depth discontinuities, within τ | it cannot move a surface by more than τ (1.9 % on pccr): by construction it neither creates nor mends a displaced copy. Its value is at the edges | as coded | the relaunch of the copy: detail on 15.6 % of the valid pixels; 55 min of the card | not judged by these rulers — his eye |
| The pixels between two surfaces at an edge ("mixed"): sent to the front or the back surface, or removed when neither is supported | a depth map interpolates across an edge: those pixels lie on no surface, and removing them removes floating points | as coded | 24.4 M mixed pixels → 4.6 M to the front, 5.7 M to the back, 14.2 M removed | sound: it is what keeps the edges clean |
| A tile is rejected when its fit's residual is over the 95th percentile of the session's own tiles | a fixed RATE, not a measurement: 5 % of the tiles are rejected whether they are good or bad | as coded (the code calls the bar "self-referential") | 275 of 5,544 tiles (4.96 %), the bar at 13.4 %; 46 more by too little support | harmless (a rejected tile keeps Omega's depth); not a judgement of quality |
| `the detail layer's file (layer_*.glb)`, `the mixed layer's file (layer_*.glb)` | the two viewer layers whose menu entries were removed from the interface on 2026-11-19 | still written at every run; the server keeps the route that serves them (`/api/precision/layer`) | nothing in the interface asks for them | UNNECESSARY (the writing and the route's two entries) |

### F6 — the cleaner (`reconstruction/gpu_cloud_clean.py`)

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| The voxel filter: one point per 5 mm cell — the real point nearest the cell's centre | a DECIMATION, not a fusion: two copies of one surface 1.5 – 3 cm apart both survive, and which keyframe's point stays in a cell is decided by its position in the cell, not by its quality (distance, support) | as coded; 5 mm is his | the first run of the copy: 597.7 M voted points → 117.5 M published | NECESSARY (his number); it sets the cloud's density, it does not join passes |
| The outlier rule: a point whose mean distance to its 8 neighbours is over the cloud's mean + 3 σ, or that has no neighbour within 15 mm | the cloud's own statistic; a noise filter | as coded; σ 3 is his | the relaunch: 114.9 M → 112.4 M (2.2 %; 146 thousand of them isolated) | NECESSARY (his number) |

---

## Part VI · The publication and what follows it

| Calculation | Concept | Application | Evidence | Verdict |
|---|---|---|---|---|
| The publication (`precision/corrected_cloud.py`): the new cloud replaces the old one in one transaction; the epoch's lineage, the provenance column, the cloud's sha256, the octree | integrity of the product | as coded; no geometric acceptance | — | NECESSARY, sound |
| `cleaned_cloud_raw.ply` | a second name for the cloud | written with the same rows — its own reader requires the two files to "hold the same rows"; four other readers take either name | byte-identical to `cleaned_cloud.ply` on bufferStop's test run (checked); on pccr that is a second copy of 112 M points written and hashed | a DUPLICATE |
| "The product is live" (`precision/product.py`): the report exists, the epoch descends from it, the file exists | what makes a resume skip the reconstruction | existence only — the file is not hashed against the report's sha256 | — | sound as a resume rule; it does not detect a changed file |
| The cube check (`f6_check`, `precision/chunk_check.py`): each cube's floor and ceiling departures, from Omega's depth × the factor on the published poses | a check of the BENT depth — not of the published cloud (it runs on what the vote, the edge-detail model and the cleaner have not yet touched) | its result (`chunk_check.json`, with a "to correct" list) is read only by the certification, which is OFF; a missing input there raises AFTER the cloud is already published | 5 min on the copy | report read by nothing in force: **UNNECESSARY as applied** |
| The cloud metrics (`precision/cloud_metrics.py`): the floor's thickness, undulation and tilt on the published cloud | a measurement, "no verdict" | written; the chat's quality tool asks it for four facts under names the file does not use (`thickness_median_m`, `thickness_p90_m`, `tilt_deg` against `thickness_m.median`, `thickness_m.p90`, `plane.tilt_vs_camera_up_deg`): only the undulation reaches the chat | pccr copy: undulation 16.9 cm | report; a naming defect to mend |
| The floor step (`correction/floor.py`; only after a segmentation): ONE rigid motion from the segmented floor under the cameras | a gauge transformation (Lemma 1: it can break nothing); a measurement of "down" where orient assumed | its gates are advisory: a failed gate is declared and the motion applied | tilts of 0.44° (pccr 08-24), 2.75° (zaragoza), 3.25° (observatorio); 92.7° on bufferStop | NECESSARY — the measured "down" |
| The certification | — | OFF since 2026-11-05 (his) | — | not in the chain |

---

## Part VII · The verification layer: who judges each step, and with what

The rule that decides nearly everywhere is `decide_change` (`loop_utils/metric_lock.py`): the median improvement of
paired judges, a bootstrap interval that excludes zero, at least five judges, and an improvement of at least 1.1 × an
error handed in by the caller. As a rule it is sound. Its strength is the independence of the data it is given and
the meaning of that "error" — and both are the caller's.

| What is accepted | Judged with | Independent of what it judges? | When it fails |
|---|---|---|---|
| A frame is usable (I0) | its own brightness against fixed bounds | yes — but brightness is not information | the frame leaves the chain |
| An I1 track | the tracker run backwards (≤ 1 px) | the tracker's own consistency | the track dies |
| The coverage fill's stopping bar | the first pass's map | NO: the tolerance and the bar come from the samples they judge | nothing is rejected |
| The first pass's metre; a cube's lock; scale_align's s | DA3 | a second instrument | **no acceptance test at all** |
| A seam | the same pixels in the two cubes; σ = its own fit's residual | NO for σ | never rejected |
| A closure | one joint pass; σ = its two fits' own residuals; the pair comes from the first pass alone | NO. A split-half verification exists (`verify_loop`) and is not called on cubes | never rejected |
| The graph's solution | — | **no judge.** A leave-one-out judge exists (`judge_leave_one_out`) and is not called on cubes | not converged → the seams' tree |
| The depth graph | held-out pairs of the same depth maps and poses | partly | SKIP |
| F2's instrument | each instrument's own rows, held out | NO: each is judged against itself | — |
| F5's first rung | — | taken with no verdict | — |
| F5's rungs R1, R2 | held-out tracks, triangulated on the very observations they are scored on; the "error" is the solver's own continuation | of the adjustment's landmarks yes; of its cameras, frames and tracker NO. It sees pixels, not positions | the rung is not taken |
| F6's factor per keyframe | landmarks of the same F5 | NO against F5's own error | a wider window; when none verifies, the widest pooled fit is applied all the same |
| F6's τ | the 75th percentile of the session's neighbour disagreement | NO: by construction 75 % of the sampled neighbours agree within it | — |
| A published pixel (the vote) | other views of the same network's depth, with the same factor, on the same poses | against independent noise yes; against a common error NO | the pixel is not published |
| A the edge-detail model tile | the session's own 95th percentile | NO | it keeps Omega's depth |
| A point (the outlier rule) | the cloud's own statistic | NO | removed |
| The cube check; the cloud metrics | the geometry against itself; DA3 anchors already used upstream | partly | advisory: nothing happens |
| The floor's motion | the floor plane against its own bootstrap; advisory gates | NO | declared, applied |

**What this layer is.** Every acceptance in the chain is a test of SELF-CONSISTENCY: of a fit with its own data, of
the network with itself across views, of F5 with its own landmarks. Self-consistency removes noise. It has no power
against an error COMMON to the judge and the judged: a stretch at a wrong scale (the vote confirms it with its own
neighbours — Lemma 1), a camera path cut at a door (the held-out tracks reproject the same), an opened loop (no local
test sees it — Lemma 3). Those are exactly the defects of pccr, and it is why every one of its steps reported itself
as verified. Where the chain HAS stronger tests — the split-half verification of a closure, the leave-one-out judge
of the graph — it does not call them on cubes.

**The references that do not depend on the chain**, in the order in which a session has them:
1. the agreement of different PASSES over the same surface, measured in the images (the pair ruler of Part II: across
   the ray it is a match of two photographs, independent of the depth network) — wherever the walk revisits;
2. a repeated object of constant size (pccr's floor tiles) — when the scene has one;
3. a second instrument (DA3's depth; the first pass's path over a short stretch) — biased, but differently;
4. his known lengths (the calibration).

No acceptance of the chain uses any of them today. The first is what told the stages apart in this audit. Measured
after each step that moves poses or depth (the graph, F5, F6), with the rule "a step may not leave it worse than its
own measured noise", it is the numerical form of *"que un paso no rompa lo que corrigió el anterior"*. (A proposal —
his decision.)

---

## Part VIII · The whole chain

### A · The quantities that are decided more than once (in order; the last one wins)

| Quantity | Decided by, in order | What is needed |
|---|---|---|
| The metre | DA3's own alignment of each window (I3) → the first pass's metre (its range only) → the lock of each cube (170 on the copy) → **scale_align (one number)** → F2 (a second estimate) → F6's fitted factor (per keyframe: 2,772 of them; or one) → his calibration | ONE estimate, carried to the cameras AND to the depth |
| The camera | DA3's probe (for I1) → the first pass, per record → Omega, per keyframe → F0's median → **F5** | a start, and F5 |
| The poses | the first pass (visibility only) → Omega, per cube → **the graph** → (the old F2) → **F5** | the graph and F5 — F5 carrying the graph |
| A keyframe's depth | Omega (the owner cube) → the depth graph (never applied) → F6's factor → the edge-detail model's detail → the vote's median | Omega, the known factor, the vote |
| "Down" | the cube plan's frame (the viewer's boxes) → orient (assumed) → **the floor step (measured)** | one, measured |
| Who sees whom | I1's model of a shared field → the fill's test (the session's tolerance) → the plan's test (a fixed 25 %) → F6's witness lists | one test |
| The keyframes | I1's bars → the 8-frame rule → the fill | the first two |
| The confidence floor | the cubes' own (for the preview) → F6's | F6's |

### B · Every calculation, by its verdict

**Necessary and sound** — the frames · I1's reading with the 8-frame rule · the first pass (for the revisits) · the
co-visibility · the cut by his numbers (64, the overlap, 32) and the owner · the cube's pass · the seams · the closures
and the graph · the placement · the records · scale_align · orient as a first frame · F0 · F4's keyframe windows and
its linking tracks · F5's adjustment and its camera rungs · F6's vote, its edge pixels, the cleaner · the publication ·
the floor step.

**Right in concept, wrong in application** (his *"eso es lo más importante"*) —
1. F5's prior: independent positions where the cube graph should be (Part V).
2. The depth is left in units its cameras no longer have: the scale reaches the cameras, never the records (Part IV);
   F6 has to fit again a factor that is known.
3. The closures: weighted by a residual that does not measure what a closure asserts, selected by "c > 0", never
   verified though the verification exists (Part IV).
4. F4's loop windows: three in four link nothing; the adjustment is, in its bulk, the same places tracked again
   (Part V).
5. The range of 15 m evaluated in each cube's locked units (Part IV).

**Wrong by concept** — the brightness gate of I0 (Part III) · the old F2's open chain (removed) · F6's factor per
keyframe while F5 cannot be trusted everywhere, and its "verification" on F5's own landmarks (Part V) · the depth
graph (the same family; it has never acted).

**Redundant: they decide nothing, or decide again what another calculation decides** — the coverage fill (0 keyframes
on three sessions) · the co-visibility budget of the cut (no cube on six sessions) · the DA3 lock per cube (removed by
the graph) · the seams' scale bound · I3's chain, walk length and chainage · F2, with the DA3 windows it generates
again (35 min on the copy) · the fit of F6's session factor · the depth graph (SKIP on six sessions; 6 min) · F4's
witness windows and F5's localisation of the witness frames (about a third of F4) · the cube check (5 min; read by a
stage that is off) · the 566 bridges over 6 m (42 min).

**Duplicates** — `cleaned_cloud_raw.ply` (the cloud a second time, byte for byte) · `obs_score` of `tracks.npz` (the
visibility a second time).

**Written and used by nothing** — `the detail layer's file (layer_*.glb)`, `the mixed layer's file (layer_*.glb)` (a server route remains, no
menu asks for them) · `refine_residuals.npz` (opened by F6's loader for a value F6 never uses) ·
`witness_poses.txt` (loaded by F6, used by none of its calculations) · `obs_uv_sigma` of `tracks.npz` · the
uncertainty maps · `metric_lock.json`, `chunk_graph.json` but its poses, `depth_graph.json`,
`camera_poses.ply` · `coverage_fill.json`, `turn_frames.json`, `frame_quality.json`, `da3_frames.json`, `tol_rel`,
`laplacian`, `inter_frame_diff` · scale_align's diagnostics · configuration keys no code reads (Part IV).

**Not judged here** (no ruler of this audit reaches them — his eye, or a measurement still to be made) — the turn
frames and the second pass of every cube · the edge-detail model's detail · the upper limit of 30° of the vote · "in place"
re-admitting what the majority or the veto rejected · whether F5's two focal lengths should be one (fx / fy = 0.973 –
0.982 on the three pccr runs, 0.991 – 1.007 on the three 16:9 sessions).

### C · What the sessions say together

1. **The last step that touches a quantity decides it.** The local scale of the walk is set by the seams, then by the
   closures, then (formerly) by F2, then by F5, then per keyframe by F6. On bufferStop F5 erases everything before it:
   two runs whose cube graphs differed by 0.7 cm and whose F2 differed by up to 9 cm end 0.1 mm apart. On pccr F5
   cannot determine everything, so what came before survives where the tracks are silent — and F5 adds its own error
   at the doors.
2. **Two calculations are wrong by construction** (the lemmas, not a measurement): the old F2's open-chain
   re-integration (removed), and a depth factor per keyframe that the cameras do not match (F6, as long as F5 is not
   trustworthy everywhere).
3. **One calculation is right in concept and wrong in application, and it is the most important one: F5's prior.**
   Independent position priors do not carry the cube graph. With relative constraints (neighbours, with the cubes'
   own uncertainty) and the closures inside F5's cost, the tracks decide where they can and the graph holds where
   they cannot.
4. **The chain cannot see its own common errors.** Every acceptance is a self-consistency (Part VII): on pccr every
   step reported itself as verified while the cabinet was published 45 cm from itself.
5. **Much of the machine time goes to calculations that yield little as applied**: in F4 — half of the precision
   core — about three quarters of the time is loop windows that link no two visits and witness windows whose tracks
   F5 hardly accepts.

### D · The corrections, in the order of what they are worth

His decisions; only the first is under way. Whatever changes what he sees goes to ONE session first.

1. F6 with one factor for the session — running on the copy of pccr; his eye decides.
2. F5 carrying the cube graph: relative constraints between neighbouring cameras with the cubes' own uncertainty, and
   the closures, inside F5's cost (*"si mejora este f6 corregimos f5"*).
3. The scale carried to the depth when it is carried to the cameras: the factor handed on, not fitted again.
4. The brightness gate open, as in the copy.
5. The closures verified by what they assert (the split-half test that already exists) instead of trusted on
   "c > 0"; the graph judged (the leave-one-out that already exists), and a solve that does not converge stops the
   run.
6. F4: a loop window only where its two views can be linked; the witness windows out, or their poses used. To be
   measured on one session before anything is decided: the loop windows that link nothing still supply about 86 % of
   F5's landmarks.
7. Out, what decides nothing or is read by nothing (list B). On pccr 08-24's size: F2 35 min, the witness windows
   about 1 h 45 min, the bridges over 6 m 42 min, the depth graph 6 min, the cube check 5 min; the second copy of the
   cloud and the unread files.
8. One acceptance measure that does not depend on the chain, after every step that moves poses or depth (Part VII).

---

## Part IX · Corrections to what the assistant said earlier the same day

- "With one scale alone the joints between passes fall 7.2 cm apart" and "Omega's camera steps are 6 % short for its
  depth": both carried an error of units (the records' depth compared without scale_align's factor). Right: 3.2 cm at
  the cube stage, 1.5 cm with F5's cameras, and 3.1 %.
- "F6 moves the metre by 3.5 %": wrong for the same reason; its fitted factor differs from the cameras' scale by 0.7 %.
- "The 30° limit silences 103 near keyframes that see through the plate": not verified when said, and false.
- "The far depth of those cubes is 10 % short": false — the cubes have it right; F6's factor shortened it.
- "F6's factor per keyframe is conceptually wrong" (said before bufferStop was measured): too strong — it is sound
  where F5 is determined; it stands or falls with F5.
