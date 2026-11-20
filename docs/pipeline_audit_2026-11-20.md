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

STATE OF THIS DOCUMENT: Part I, Part II and the precision chain (F2, F5, F6) are closed. Intake, the cube stage, F0,
F4, PointDiT and the publication are being inventoried calculation by calculation and follow.

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

## Part III · The precision chain, calculation by calculation

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
| `refine_residuals.npz` | — | written | read by nothing | UNNECESSARY |

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

---

## Part IV · What the two sessions say together

1. **The last step that touches a quantity decides it.** The local scale of the walk is set by the seams, then by the
   closures, then (formerly) by F2, then by F5, then per keyframe by F6. On bufferStop F5 erases F2 completely (0.1 mm
   between two runs with different F2). On pccr F5 cannot determine everything, so what came before survives where the
   tracks are silent — and F5 adds its own error at the doors.
2. **Two calculations are wrong by construction** (the lemmas, not a measurement): the old F2's open-chain
   re-integration (removed), and a depth factor per keyframe that the cameras do not match (F6, as long as F5 is not
   trustworthy everywhere).
3. **One calculation is right in concept and wrong in application, and it is the most important one: F5's prior.**
   Independent position priors do not carry the cube graph. With relative constraints (neighbours, with the cubes'
   own uncertainty) and the closures inside F5's cost, the tracks decide where they can and the graph holds where
   they cannot.
4. **Redundant**: F2 (a second estimate of the metre); the fit of F6's factor (the factor is known); F5's residual
   file.

## Part V · Corrections to what the assistant said earlier the same day

- "With one scale alone the joints between passes fall 7.2 cm apart" and "Omega's camera steps are 6 % short for its
  depth": both carried an error of units (the records' depth compared without scale_align's factor). Right: 3.2 cm at
  the cube stage, 1.5 cm with F5's cameras, and 3.1 %.
- "F6 moves the metre by 3.5 %": wrong for the same reason; its fitted factor differs from the cameras' scale by 0.7 %.
- "The 30° limit silences 103 near keyframes that see through the plate": not verified when said, and false.
- "The far depth of those cubes is 10 % short": false — the cubes have it right; F6's factor shortened it.
- "F6's factor per keyframe is conceptually wrong" (said before bufferStop was measured): too strong — it is sound
  where F5 is determined; it stands or falls with F5.
