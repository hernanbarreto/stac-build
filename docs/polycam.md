# Polycam — what is public, and what of it we can use

Researched 2026-09-23 by twelve agents (three sweeps + three adversarial
verifications). **Every claim below is marked with how it was established.**
The verification pass refuted roughly a third of what the first pass reported,
including several of its most attractive findings — those are kept at the bottom
under *Disproved*, so nobody resurrects them.

    [fetched]   the page was opened and says this
    [inferred]  a reasonable reading, not stated by the source
    [refuted]   the first pass claimed it; verification killed it

---

## 0. The direct answer

**Polycam's reconstruction pipeline is closed.** [fetched]
`https://github.com/orgs/PolyCam/repositories` — 30 public repos, zero public
members. No SLAM, no bundle adjuster, no scale solver, no loop closure, no
drift correction, no meshing of their own. What is there: iOS app plumbing
(DeviceKit, firebase-ios-sdk, SCNRecorder, paywall-ios), file-format libraries
(tinygltf, tinyusdz, lunasvg, ifcplusplus, libzip) and mostly-unmodified
mirrors (opencv, open3d, colmap, vcpkg, executorch).

Do not spend more time looking for a hidden reconstruction repo. There is none.

**But the search was still worth it**, for two reasons that have nothing to do
with their source code:

1. Their org points at **LoGeR**, which attacks our drift problem inside the
   network instead of after it.
2. They published, four days before this search, a **floor-plan pipeline post**
   that names six algorithms — and one of them is a fix for a defect we have
   measured and not yet corrected.

---

# PART A — drift, scale and correction

## A1. LoGeR — the one that matters

* Paper: `https://arxiv.org/abs/2603.03269` [fetched]
* Project: `https://loger-project.github.io/` [fetched]
* Code: `https://github.com/Junyi42/LoGeR` (upstream),
  `https://github.com/PolyCam/LoGeR` (fork + a driver `run_loger.py`) [fetched]
* Weights: public on HuggingFace `Junyi42/LoGeR`, **not gated** — direct wget
  URLs in the fork's README (`LoGeR/latest.pt`, `LoGeR_star/latest.pt`) [fetched]
* Authors: Google / Berkeley (Zhang, Herrmann, Hur, Sun, Yang, Cole, Darrell,
  Sun). **Not a Polycam paper** — they forked it.

*Long-Context Geometric Reconstruction with Hybrid Memory.* Two memories:

1. a **parametric test-time-training memory that anchors the global coordinate
   frame and prevents scale drift**, and
2. **non-parametric sliding-window attention** that keeps uncompressed context
   across chunk boundaries.

Trained on 128-frame sequences, generalises to thousands at inference (up to
19,000 frames on a repurposed VBR dataset). **ATE on KITTI reduced by over
74 %** vs prior feed-forward methods.

**Why this is the find.** Our entire drift machinery — visit-drift silhouettes,
`scale_loop_rows`, the DA3 relative-seam ladder — exists to *repair* drift after
the fact. The declared structural ceiling in CLAUDE.md (`n_seams × sigma_seam_log`
≈ 12 % on pccr, which needs ~14 %) is a property of repairing drift one seam at a
time. A continuous learned memory dissolves that ceiling because there is no
ladder.

**Two limits, both verified:**

* It prevents scale **drift** (internal consistency of one gauge across chunks).
  Nothing establishes it emits **metric** scale. **DA3 anchors still needed.**
* The paper **declares revisits and loop closure out of scope** and demonstrates
  on open-loop sequences. Our duplicate-based correction attacks something LoGeR
  leaves open. It would replace half our battery, not all of it.

**Recommended next step:** clone it, run `run_loger.py` on pccr's 216 keyframes,
compare against the Omega chunked baseline we now have twice over (walk 19.1 m,
witness drop 11.49 %).

## A2. VGGT-Long "Map-Long" — SE(3) chunk alignment

`https://github.com/DengKaiCQ/VGGT-Long` [fetched]

Metric MapAnything lets chunks be aligned with **SE(3) instead of Sim(3)**. Our
per-chunk scale ladder has one free scale riser per seam; SE(3) alignment
removes the riser entirely if every chunk is already metric. That is the most
directly actionable attack on the declared 12 % ceiling without changing the
backbone.

*(Caveat: the README does not describe IRLS/Levenberg-Marquardt internals — the
first pass claimed it did. Read the paper, not the README.)* [refuted]

## A3. MapAnything — feed-forward METRIC reconstruction

`https://github.com/facebookresearch/map-anything` (Meta + CMU) [fetched]

Takes depth/poses as **inputs** and emits metric geometry natively. That
subsumes the DA3-anchor → Omega → per-chunk-lock chain, whose whole purpose is
injecting metric scale into a scale-free model.

**Hard caveat:** the "up to 2000 views" figure is *on 140 GB* of GPU memory. We
have one A6000 with 48 GB — about a third. The headline capacity does not
transfer. [refuted as stated]

## A4. VGGT-Align — a scene invariant that must agree across chunks

`https://github.com/WZ-CS/VGGT-Align` [fetched]

States scale drift as *the* critical failure mode. Its mechanism — a scene
invariant measured per chunk that must agree across chunks — is a generalisation
of something already half-present here: **the floor is exactly such an
invariant**, and we already re-level it every epoch (`floor_transform`,
`floor_level`). Worth reading for how they make it a constraint rather than a
post-hoc fix.

## A5. DA3-Streaming chunk-size ablation — measured numbers on chunk sizing

`https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md` [fetched]

The only published measured table on chunk size we found. ATE roughly **doubles**
on both datasets when the chunk drops to 30 frames — below some size the chunk no
longer contains enough parallax to constrain itself, and more seams do not
compensate.

**Caveat that matters to us:** the table **stops at 120 frames**. It says nothing
about our single-chunk regime (216 keyframes) and therefore cannot be cited as
support for the single-pass decision. It supports at most "do not go below ~60".
[refuted as support for single-pass]

## A6. Ray-Aware Pointer Memory — independent confirmation of our own finding

`https://arxiv.org/abs/2605.05749` [fetched] — **no code repository exists**
(the first pass mislabelled it as source). [refuted label]

Their duplicate discriminator is geometric proximity **plus ray-direction
discrepancy** — the same physics as this repo's 2026-09-19 conclusion that
*"the duplicates are separated ALONG THE LINE OF SIGHT"*. An outside group
reached our diagnosis independently. That is the strongest external validation
of the depth-drift model in this document.

## A7. Anchor3R — a pose graph from overlap, not a chain of seams

`https://github.com/polar-explorer/Anchor3R` [fetched]

Transferable idea: build a **relative-pose graph over redundant overlap
constraints** and solve by motion averaging, instead of a chain of pairwise seam
alignments. A ladder with one riser per seam is badly conditioned by
construction; a graph with many redundant constraints is not. This is the
structural answer to the same ceiling A2 attacks from the other side.

## A8. LongStream — gauge decoupling

`https://github.com/3DAgentWorld/LongStream` [fetched]

Anchoring everything to the first frame makes *the world coordinate* and *the
metre* the same variable, so scale error compounds. Our drift-rate model
`E(d) = ε·d` already assumes the start is exact — this is the clean statement of
why that assumption costs us.

## A9. ScaRF-SLAM — the split we already half-made

`https://github.com/ori-drs/ScaRF-SLAM` [fetched]

Let the pose chain come from an instrument that does not hallucinate scale, and
let the feed-forward model do density. That is the published version of
*"the keyframe pose graph is MEASURED, never applied"*.

**Caveat:** their "≈2 cm over 10 m" is **reconstruction error against a LiDAR
ground-truth model, evaluated per chunk**. It is not comparable to our 8.2 cm
closure residual over a 19 m walk. Do not quote them as a bar. [refuted]

## A10. Confidence and uncertainty — problem (4), "did the correction help?"

* **Trust3R** `https://arxiv.org/abs/2605.19539` (ICML 2026) [fetched] —
  evidential **per-point covariance** instead of a scalar. A silhouette closure
  whose points are uncertain *along the ray* and certain *laterally* is exactly
  the depth-drift signature; a covariance would let visit-drift weight closures
  by the direction they are uncertain in, instead of our current tangential
  residual hack.
* **A Calibration Audit of Confidence in Feed-Forward 3D Reconstruction**
  `https://arxiv.org/abs/2608.29705` [fetched] — the quantitative version of our
  own note that *"VGGT confidence is systematically low on legitimate
  flat/textureless surface"*. Confirms: every correction tried is close to right
  on average and still leaves **two-thirds of held-out scenes more than five
  points off in coverage**.
* **ScaleMaster — "Have We Mastered Scale in Deep Monocular Visual SLAM?"**
  `https://arxiv.org/abs/2602.18174` (ICRA 2026) [fetched] — the closest thing
  to an external third-party bar for exactly our pccr failure (~14 % scale drift
  over a 19 m indoor low-texture walk). If the dataset downloads, it is a
  known-answer experiment we do not have.

## A11. Index to chase next

`https://github.com/3D-Vision-World/All-3R-SLAM-in-this-Repo` [fetched] — curated
index of 3R/geometric foundation models. Two entries worth chasing: **LiDAR-VGGT**
(how a metric sensor is fused into VGGT for globally consistent metric mapping —
our DA3 anchors play that role) and the collaborative-SLAM loop-closure entries.

---

# PART B — BIM, floor plans and drawings

## B1. Polycam's floor-plan pipeline — their own engineering post

`https://poly.cam/blog/how-we-turn-raw-spatial-data-into-a-floor-plan-you-can-build-from-inside-polycams-floor-plan-pipeline`
(Polycam Team, 2026-09-19) [fetched]

**Front end: Apple RoomPlan.** *"RoomPlan runs the dots through neural networks
trained on example rooms, and hands back a rough list of walls, doors, windows,
openings, and furniture, with placement and sizes."* Everything after that is
theirs, and **all of it runs on the device.**

Their six post-processing algorithms, in order:

1. **Wall colour detection** — corrects white-balance shift across keyframes,
   using the ceiling as the reference surface.
2. **Straightening walls** — see B2, this is the one for us.
3. **Finding and classifying rooms** — three phases: outline the building edge,
   divide the interior into rooms, then label each room by object detection plus
   geometry validation.
4. **Floor and ceiling detection** — solve height **per room independently**,
   then **reconcile neighbouring rooms to shared heights**; detect ceiling shape
   (flat or vaulted).
5. **Cleaning up objects** — remove object/wall intersections and object/object
   overlaps.
6. **Location and compass heading** — *"we treat the whole thing as one
   optimization: we match the string of GPS points to where the scan thinks you
   were standing, lean on the headings as another signal, and work through the
   jumps and outliers."*

## B2. The one to copy: "straightening walls"

Verbatim: *"LiDAR scans drift. Walls that are dead straight in real life can come
out slightly crooked, because as your phone moves through a space, it's taking
thousands of tiny measurements and constantly tracking its own position … over a
full scan those tiny errors pile up, and a wall that should be ruler-straight
ends up slightly bent."*

Their fix: **find the building's dominant orientation, rotate the whole scan to
line up with it, and snap walls that are close to horizontal or vertical back to
square.**

**Why this is ours to take.** We have this defect, measured and unfixed:
test2 has a **~2° global orientation error** (three horizontal surfaces parallel
within 0.58°, so the scene is internally consistent and globally rotated). We
already have every piece needed: `surface_fit` fits planes with roles,
`contours.py` already snaps polygon directions, and the floor stage already
computes and applies a global rotation. What is missing is exactly their step:
**estimate the dominant orientation of the whole scene from the fitted planes and
rotate once**, then snap near-axis walls.

Note the difference in ambition. They snap walls to square because for a floor
plan the intent *is* a rectilinear building. We must not snap blindly — the
repo's own doctrine is that real slopes survive (`floor.model: plane`). The
transferable part is **measuring the dominant orientation and applying it
globally**, not the snapping.

## B3. Per-room height, reconciled between rooms

Their step 4 is a pattern we do not have: solve floor/ceiling height **per room
independently**, then reconcile neighbours to a shared height. Our floor stage
solves one trend for the whole session. For a multi-room BIM deliverable, per-room
solve + reconciliation is the right shape, and it also explains real level
changes instead of smoothing them away (which our step-demotion already tries to
do session-wide).

## B4. polyform — their shipping data format

`https://github.com/PolyCam/polyform` [fetched]

Documents what a Polycam capture actually contains: ARKit **gravity-aligned VIO
poses**, LiDAR depth as **16-bit millimetre PNG**, ARKit **3-level confidence**
(0/127/255), per-keyframe **blur score** and camera intrinsics, then
`corrected_cameras/` + undistorted `corrected_images/` produced by a global pose
optimisation they themselves call *"aka Loop Closure"*.

Two honest lessons, no algorithm:

* **They do not have our problem #1.** Their scale is metric from ARKit
  LiDAR + VIO from frame one, so their pose optimisation is a bounded refinement,
  not a fight against monocular scale drift. Do not copy their problem statement.
* Their global pose optimisation runs by default only **under ~700 frames** and
  is refused past **1400**. **This is a device budget (iPhone memory), not an
  algorithmic limit** — the README says the corrected directories *"may not exist
  if the session was too large for ON-DEVICE optimization"*. It is **not**
  comparable to our GPU chunk-capacity probe. [refuted as an algorithmic limit]

## B5. IFC

`PolyCam/ifcplusplus` is in the org [fetched] — they carry an IFC toolchain, which
is consistent with a BIM export path. Nothing about how they author the IFC.
Relevant to `docs/BIM_INTEGRATION_PLAN.md` only as "the same library is the
industry default".

---

# Disproved — do not resurrect

Kept because each of these was reported confidently by the first pass and would
have led us somewhere false.

| Claim | What is actually true |
|---|---|
| Polycam "chose" window 32 / overlap 3 / conf 20, corroborating our `conf_percentile: 20` | Those are the **upstream author's defaults**, copied verbatim into the fork's driver. And our own 20 is the **VGGT vendor default**. No independent corroboration exists. |
| `run_loger.py` is "Polycam-authored", reviewed by three staff | Authored by `cdcseacave` from a personal gmail, no stated Polycam affiliation; **one** human approval, not three. He *is* the author of OpenMVS, which is why the driver exports `scene.mvs` — that export is his signature, not a Polycam architectural tell. |
| LoGeR weights may be gated | They are **public on HuggingFace**, direct wget URLs in the README. |
| Polycam has no engineering blog | They do — the floor-plan post of 2026-09-19 (§B1). The first pass checked one marketing URL and generalised. |
| Their ~700/1400 frame limit is where global pose optimisation "stops being affordable" | It is an **on-device memory budget**, explicitly. |
| Their 5 m LiDAR range explains our `depth_trunc` 5 m lesson | Unrelated numbers. Theirs is Apple's sensor range; ours was a TSDF integration default. |
| MME (Cloud_Map_Evaluation) as a ground-truth-free stopping criterion for our correction | Its own README warns MME is **invalid across different scales and after loop-closure optimisation** — precisely our use case. |
| ScaRF-SLAM's 2 cm/10 m as a bar for our 8.2 cm/19 m | Different quantity (per-chunk error vs LiDAR ground truth). |
| MapAnything's 2000 views | Requires 140 GB; we have 48. |
| `PolyCam/RoMaV2` shows they keep a classical dense-matching track | The fork carries three non-upstream commits, one of them from `mark.mccurry@polycam.ai` — so the org is genuinely staffed and did touch it, but a batch pair-processing script is not evidence of a pipeline track. |

---

# What to do with this

1. **Test LoGeR against Omega on pccr.** We have the baseline twice (walk 19.1 m,
   witness drop 11.49 %, floor +1.867 m). Weights are public. This is the only
   item here that could replace a whole stage rather than improve one.
2. **Implement dominant-orientation estimation** (§B2). Small, self-contained,
   fixes a defect we have already measured on test2, and it is a prerequisite for
   any floor-plan or BIM deliverable.
3. **Read A2 + A7 together** before spending anything more on the seam-sigma
   ladder: SE(3) chunk alignment and an overlap pose graph attack the declared
   12 % ceiling from two sides, and both are cheaper than a backbone change.
4. **Trust3R's per-point covariance** (§A10) is the principled replacement for the
   tangential-residual trick in `visit_drift.scale_rows()`.
