<p align="center">
  <img src="docs/assets/stac_banner.png" alt="STAC Build" width="360"/>
</p>

<p align="center">
  <strong>Spatio-Temporal Awareness Core — dimensional control of construction from a phone video</strong>
</p>

<p align="center">
  <em>A metric point cloud measured and corrected frame by frame, a segmented and described scene, and a textured scene mesh
  that copies that cloud — local, deterministic, and traceable to the pixel that produced every number.</em>
</p>

---

## Abstract

STAC Build turns a handheld video into a **metric 3-D reconstruction of a construction site** and the objects in it.

1. **A first pass by AMB3R-SLAM over the whole video** gives every frame a pose and records what every frame it
   reconstructed saw. That is a **coverage map** of the scanned scene, in which every point carries the frame it came
   from. The map is put in **metres** once, by Depth Anything 3 on the session's focal probe.
2. The intake's parallax keyframes are a first proposal, with never more than 8 video frames between two of them. **The
   map completes them**: where the video saw something that no keyframe sees, or that only one sees, the frame that sees
   it becomes a keyframe, until no frame brings more than the measurement's own noise.
3. The keyframes are **cut into cubes by what they see**: a cube takes keyframes along the walk while their
   co-visibility budget lasts and it holds at most 64 of them, neighbouring cubes share half their keyframes, and a cube
   under 32 keyframes merges into its neighbour.
4. A feed-forward multi-view network (VGGT-Ω) reconstructs each cube. Depth Anything 3 makes each cube metric, and **one
   Sim(3) graph** joins the cubes on the frames they share (the seams) and on the revisits of the walk (loop edges, each
   measured by a joint VGGT-Ω pass of its two windows).

**One range for every view.** No keyframe contributes, votes or weighs in any fit with a point farther than
`reconstruction.view_range_m` (15 m) from its own camera — in the coverage fill, the cut, the metric locks, the seams and
closures, the scale, the orientation, the gauge, the bundle adjustment and the final vote.

A **precision core** then measures and corrects, in order:

- the camera;
- a continuous metric gauge along the walk;
- learned native-pixel tracks;
- a bundle adjustment judged on held-out tracks;
- a per-keyframe "bend" of the network's depth onto the adjusted landmarks, then COLMAP's multi-view fusion rule: a
  pixel is published when at least five witnesses confirm its depth from a triangulating angle, more confirm it than see
  through it, and no better-supported surface hides it — or, failing the count, when its witnesses' surfaces agree with
  its own depth (published where it is, never moved).

The published cloud is built from that voted depth. A vision–language model then names everything in the keyframes
and SAM 3.1 segments and tracks it, in rounds, until the model names nothing that is not already segmented. The masks
are projected onto the cloud as object instances, and the floor is levelled on the segmented support surfaces. On
demand, **GenRecon** (a Trellis-class sparse-voxel generative model) builds a textured scene mesh. Presence comes from
the cloud: a voxel is active when it holds enough of the cloud's points. The model shapes the surface inside those
voxels, the keyframes are baked onto it as its real texture, and the mesh is published as 3D Tiles.

Every decision that moves geometry follows **one statistical rule**. A change is applied only when a paired bootstrap on
held-out evidence says it improves, by more than its own measured error, with enough independent judges. Every stage runs
on the GPU, and each GPU port keeps its CPU function as the path of record. The reconstruction and the segmentation
chain run in deterministic mode; the declared exceptions are listed in §19. Every product is stamped with the inputs,
configuration and code that made it.

**Reference runs** (2026-11-09, the user's visual validation of both):

| Session | Keyframes / cubes | Seams after the graph | F5 held-out | F6 coverage | Published points (5 mm) |
|---|---|---|---|---|---|
| zaragoza (a train in a shed) | 279 / 8 | median 3.5 cm, max 11.2 cm (82 revisit closures) | 0.398 px | 42.2 % | 23.3 M |
| pccr 2026-08-31 (a technical room) | 547 / 17 | 1.7 cm median, 4.8 cm max (43 revisit closures) | 0.534 px | 59.9 % | 18.8 M |

---

## Contents

1. [What runs when a reconstruction is launched](#1-what-runs-when-a-reconstruction-is-launched)
2. [Architecture](#2-architecture)
3. [Models](#3-models)
4. [Notation and the decision rule](#4-notation-and-the-decision-rule)
5. [Intake: quality, camera, parallax keyframes](#5-intake-quality-camera-parallax-keyframes)
6. [The first pass: AMB3R-SLAM over the whole video](#6-the-first-pass-amb3r-slam-over-the-whole-video)
7. [The coverage map and the keyframe fill](#7-the-coverage-map-and-the-keyframe-fill)
8. [I3: multi-view metric depth windows and the walk](#8-i3-multi-view-metric-depth-windows-and-the-walk)
9. [Turn frames and interval witnesses](#9-turn-frames-and-interval-witnesses)
10. [Cubes: the views grouped by what they see, and the Ω resolution](#10-cubes-the-views-grouped-by-what-they-see-and-the-ω-resolution)
11. [The VGGT-Ω pass, cube by cube](#11-the-vggt-ω-pass-cube-by-cube)
12. [Metric lock and the cube Sim(3) graph](#12-metric-lock-and-the-cube-sim3-graph)
13. [Global metric scale and the upright frame](#13-global-metric-scale-and-the-upright-frame)
14. [The precision core, F0 → F6](#14-the-precision-core-f0--f6)
15. [The published cloud](#15-the-published-cloud)
16. [Chunk check (report)](#16-chunk-check-report)
17. [Scene understanding: VLM → SAM 3.1 → instances](#17-scene-understanding-vlm--sam-31--instances)
18. [GenRecon scene mesh](#18-genrecon-scene-mesh)
19. [Determinism, devices and reproducibility](#19-determinism-devices-and-reproducibility)
20. [Running it](#20-running-it)
21. [Repository map](#21-repository-map)
22. [The session in the viewer: meshes, the spatial chat, calibration, STAC's commands and voice](#22-the-session-in-the-viewer-meshes-the-spatial-chat-calibration-stacs-commands-and-voice)

---

## 1. What runs when a reconstruction is launched

**Reconstruir** in the UI sends one order per scan to the pipeline manager (`server/pipeline_manager.py`). The manager
runs the jobs one at a time on the card. Each stage is a separate worker process, each model runs in its own conda
environment (§2), and every stage is resumable against stamped products. The default chain is
`pipeline_manager.DEFAULT_STAGE_ORDER`:

| # | Stage | Worker | What it produces |
|---|---|---|---|
| 1 | **Reconstruction** | `workers/map_worker.py` | intake (I0, focal probe, I1) → AMB3R-SLAM over the whole video, put in metres → the coverage fill of the keyframes → turn frames → I3 → the cubes → VGGT-Ω per cube, the bridges of the revisits and the cube Sim(3) graph → metric scale and upright frame → precision core F0–F6 → the published cloud (`output/cleaned_cloud.ply`) and its octree |
| 2 | **Cloud → viewer** | `workers/cloudcompy_worker.py` | the cleaned cloud handed to the viewer (Potree octree) |
| 3 | **Scene analysis** | `workers/vlm_worker.py` | Qwen3-VL reads every keyframe, on a vLLM the stage launches from the job's frozen configuration and stops; the SAM 3 vocabulary (`output/vlm_analysis.json`) |
| 4 | **Segmentation** | `workers/sam3_worker.py` | SAM 3.1 masks on every keyframe, four concepts at a time on one vision backbone, in rounds (VLM → SAM 3) until the VLM names nothing new or the prompt bound is full (`output/seg_masks.npz`, `segmentation.json`, `segmentation_census.json`, `seg_rounds.json`) |
| 5 | **Projection** | `segmentation/pipeline.py` | the masks projected onto the cloud as instances with oriented boxes (`output/segmentation_result.json`, `instance_ids.npy`, `scene_r.db`) and audited against their masks (`mask_audit.json`); SAM 3 projects what it segments, once, after the last round, so this stage runs on its own only for a resume or an order of the projection alone |
| 6 | **Floor** | `workers/floor_worker.py` | the floor levelled on the segmented support surfaces, one rigid motion per session (`pipeline.auto_floor: true`, after every projection) |
| 7 | **Certification** | `workers/certify_worker.py` | switched off (`certify.auto_after_segmentation: false`) |
| 8 | **GenRecon** | `workers/genrecon_worker.py` | the scene mesh with the keyframes baked into its atlas, and its 3D Tiles (`output/tsdf/genrecon_photo/`), only when ordered (`pipeline.auto_genrecon: false`) with the cloud's confidence threshold chosen for that order |

A **Reconstruir** order runs stages 1, 2 and 6 (and 3–5 when *Segment when done* is ticked). The **Autosegment** window
orders the VLM and SAM 3 on the cloud already on disk, with editable prompts; the floor follows the projection. The
**GenRecon** button orders stage 8 alone, and no floor runs with it. An order through the API (§20) can also ask for
the projection alone: the masks on disk are projected on the cloud on disk, nothing is segmented again, and the floor
follows.

**Replace.** **Reconstruir** with *Replace* empties the scan down to its frames, their manifest, its video and its
capture inputs: `intake/`, `output/` (the VLM and SAM 3 products included — masks made on other keyframes are invalid)
and the viewer preferences go. Without *Replace* the job resumes on its stamps.

---

## 2. Architecture

**Backend.** `server/main.py` is a FastAPI + uvicorn service on port 8765 (TLS). It holds the REST API, the viewer
WebSocket, the session store and the pipeline manager.

**Pipeline manager.** One queue of jobs: one job per scan and order, stages in a fixed order, one GPU job at a time. It
has these properties:

- `config.yaml` is read **once, when the order is given**; the job freezes that configuration into
  `output/run_config.yaml`, with per-order stage overrides, and every worker verifies it. A change to `config.yaml`
  reaches only the orders given after it.
- Each stage declares its products. A resumed job (*Replace* off) skips a step whose stamp — inputs, the configuration
  sections it reads, its code — matches, and restarts at the first step whose stamp differs. *Replace* on wipes the scan
  down to its frames and video.
- GPU stages require the **whole card** (`repro.require_exclusive_gpu`): the vLLM service is stopped, and the stop
  verified, before them. A stage that needs the VLM (the scene analysis, a segmentation round) launches **its own**
  vLLM from the job's frozen configuration under an engine lease, and stops it when it ends
  (`semantic/service.job_engine`).
- The chat's vLLM loads **on demand** (`semantic.chat_on_demand: true`). It is not reloaded after a stage, a pipeline
  or the server's start; it comes up when the chat is opened or asked. `false` reloads it whenever the card frees.
- A cancel is cooperative: the running stage gets a bounded time to stop at a safe point, then its process tree is
  killed. Products already on disk (cubes, bridges, finished steps) are resumed by the next order.

**Environments.** The stage workers are processes of the backend's interpreter (`da3`), and a job is refused when the
backend runs on another one. Each model runs in its own environment:

| Environment | What runs in it |
|---|---|
| `da3` | the backend and every stage's worker process, DA3, SAM 3.1, the projection, the floor, F0, F2, F6, scale/orientation, the cleaner, GenRecon's adapter, resume stamps and photo bake |
| `mapanything` | the VGGT-Ω fork, F4, F5 and the sky segmenter |
| `amb3r` | AMB3R-SLAM, the whole-video first pass |
| `semantic` | vLLM serving Qwen3-VL |
| `genrecon` | GenRecon: the generation, the decode, the mesh step and the per-cube bakes |
| `nodejs` | 3D Tiles builder |

**Geometry epochs.** Every correction that moves geometry publishes a **geometry epoch**: a journaled, transactional swap
of the artifacts it changed (F2 and F5 publish pose-only epochs, F6 the cloud, the floor step a rigid motion). Once the
cloud is published, only it is shown: the stored epochs on the live epoch's lineage move to `output/_restart_epochs/`,
out of the viewer's list, and every other stored epoch is deleted (`correction/restart.py`). They are the way back:
when the precision chain must start again from its first step (its code or configuration changed, VGGT-Ω's pass did
not) and no finished step has moved the geometry yet, the runner returns them and selects epoch 0 before F0
(`precision/runner.needs_epoch0`). The transforms of every epoch stay in `output/corrections/` and the ledger.

**UI.** `ui/` is React, TypeScript and Vite, built on Three.js:

- a Potree-format octree viewer streams the cloud; a confidence slider filters it with the same arithmetic used by the
  GenRecon adapter (§18.1);
- mesh layers stream as 3D Tiles — listed unchecked when a session opens and loaded when checked (§22.1);
- panels show instances, oriented boxes and measurement tools, and a chat assistant answers spatial questions with
  deterministic tools (the VLM proposes; tools measure; every figure of an answer is checked against the tools'
  results, §22.2);
- a distance measured on the model can be stored as a calibration measurement of the scan (§22.3);
- the **Instances** tab shows the cubes of the cut, each one switched on and off on its own;
- **View → Background and grid** sets, per session, the background (Default: a charcoal gradient; Horizon: a graduated
  sky over a green ground at $y=0$; Plain: a chosen colour) and the grid (cell 0.1–5 m, a major line every 2/4/5/10
  cells, its colour), with the cell and major spacing written under the scale bar
  (`ui/src/components/viewport/sceneLook.ts`). Colours come from design tokens; every visible string from `ui/src/i18n`.

**Storage.** Each scan lives in `server/projects/<session>/scans/<date>/src_<source>/`:

- `frames/`: the video frames (long side at most 1024 px) and their sha256 manifest;
- `intake/`: quality, parallax, the focal probe, AMB3R's whole-video pass, the coverage fill, the walk, the cubes, and
  their stamps;
- `output/`: the frozen configuration, model passes, epochs, cloud, octree, the mask store and the segmentation, and
  the meshes (`output/tsdf/<mesh>/<mesh>.glb` with its `tiles/`; the raw generated mesh in `output/genrecon/`);
- `user_prefs/<user>.json`: each user's viewer preferences for that session (point size, budget, light, opacity,
  background and grid). Deleting the session, or a *Replace*, deletes them.

GenRecon's intermediate products — the adapter's folder, the sampled latents, the decoded mesh, the cache of baked
cubes — live outside the session, in `vendor/GenRecon/work/`, and are what a resumed mesh order reuses (§18.7).

---

## 3. Models

| Role | Model | Where |
|---|---|---|
| Multi-view feed-forward reconstruction (poses, intrinsics, depth, confidence) | **VGGT-Ω** 1B, `vggt_omega_1b_512.pt`, bf16 | `vendor/VGGT-Long` (fork, branch `gpu-main`) |
| First pass over the whole video: every frame's pose, loops, what each frame saw | **AMB3R-SLAM**: DA3-Small front-end, VGGT-Ω submaps, DBoW loops, offline pose graph | `vendor/AMB3R-SLAM` @ 5418465 + `server/tools/amb3r/amb3r_stac.patch` |
| Metric multi-view depth: focal probe, I3 windows, scale anchors | **Depth Anything 3**, `DA3NESTED-GIANT-LARGE-1.1` | `vendor/depth-anything-3` |
| Place recognition for the chunk layouts' loop candidates (not run on cubes: the revisits are AMB3R's map's, §10.1) | **DINOv2 ViT-B/14 + SALAD** (8448-D descriptor) | fork `LoopModels/` |
| Sky segmentation | `skyseg.onnx` (sha256-pinned), onnxruntime CPU | fork `loop_utils/sky_mask.py` |
| Native-pixel point tracks | **VGGSfM v2** tracker (sha256-pinned) | fork `dependency/vggsfm_tracker` |
| Monocular detail at depth discontinuities | **PointDiT-H/16** with DINOv3 ViT-H+/16 | `vendor/pointdit` |
| Scene understanding (vocabulary, descriptions) | **Qwen3-VL** served by vLLM | `server/semantic/` |
| Promptable segmentation and video tracking | **SAM 3.1** Object Multiplex | `vendor/sam31`, driven by `server/segmentation/` |
| Scene mesh | **GenRecon** (Trellis.2-class sparse-structure + shape/texture SLat flows, fine-tuned, conditioned on DINOv3 ViT-L/16 features of posed images; half-precision shape and texture decoders) | `vendor/GenRecon` @ eaf1468 + `server/tools/genrecon/genrecon_stac.patch` |
| Rasterisation for the photo bake | nvdiffrast | env `da3` |

Two NVIDIA libraries serve the device ports of §19.1 and are not models: **nvJPEG** (the `libnvjpeg` that torchvision
ships, called from `server/jpeg_gpu.py`) and **nvCOMP** (`nvidia-nvcomp-cu12` 6.0.0.12 in env `da3`, the Deflate codec
of the mask store).

---

## 4. Notation and the decision rule

**Notation.**

- Keyframes are indexed $k$; each has a camera-to-world pose $T_k=[R_k\,|\,C_k]$ (OpenCV axes: $x$ right, $y$ down, $z$
  forward) and intrinsics $K$.
- A pixel is $(u,v)$; depth $z$ is measured along the optical axis.
- A ray with unit $z$ component is $\mathbf r(u,v)=K^{-1}[u,v,1]^\top$, so the point at depth $z$ is $X=C_k+z\,R_k\,\mathbf r$.
- $\operatorname{med}$ is the median, $Q_p$ the $p$-quantile, $P_p$ the $p$-th percentile.
- The robust scale is $\mathrm{MAD}$-based: $\hat\sigma=1.4826\,\operatorname{med}|x-\operatorname{med}x|$.

**The range** (`server/view_range.py`, `reconstruction.view_range_m` = $R$ = 15 m; one key for every stage). The distance
of a pixel's point from the camera that measured it is

```math
\delta(u,v,z)=z\,\Big\|\Big(\tfrac{u-c_x}{f_x},\ \tfrac{v-c_y}{f_y},\ 1\Big)\Big\|,
```

and for a 3-D point $X$ and a camera centre $C$ it is $\|X-C\|$. A point with $\delta>R$ is born by no keyframe and weighs
in no fit, pairing, scale, closure, co-visibility or statistic. A missing or non-positive key fails the load.

**Angles, not pixels.** The thresholds that used to be pixels are angles, read in each stage's own pixels through its
focal length: $\text{px}(\theta)=f_x\tan\theta$ (`precision/camera.angle_px`). The same rule then holds on a 464-px and a
1024-px session. The intake reads them with the session focal of §5.2, F4 with the tracker grid's, F6 with F5's.

**The single decision rule** (`decide_change`, fork `loop_utils/metric_lock.py`). Every correction that could move
geometry is judged the same way. Take paired errors on held-out evidence, before ($b_j$) and after ($a_j$), lower is better.
Let $d_j=b_j-a_j$ and $\tilde d=\operatorname{med}_j d_j$.

```math
\text{improves}\iff
\underbrace{\mathrm{CI}^{\,c}_{\text{lo}}\big(\operatorname{med} d\big)>0}_{\text{(a) bootstrap}}\ \wedge\
\underbrace{n_{\text{judges}}\ge\Big\lceil\tfrac{\log(1-c)}{\log 0.5}\Big\rceil}_{\text{(b) }=\,5\text{ at }c=0.95}\ \wedge\
\underbrace{\tilde d\ge f\,\varepsilon}_{\text{(c) beyond its own error}}
```

- $\mathrm{CI}^c$ is the percentile interval of $B=2000$ seeded bootstrap medians, resampling whole clusters when the
  evidence is clustered.
- $c=0.95$.
- $\varepsilon$ is the measured error of the thing being judged.
- $f$ = `correction_graph.graph.improvement_error_factor` = 1.1.

No invented magnitude decides a correction. The bars are the sample's own noise.

---

## 5. Intake: quality, camera, parallax keyframes

**Frames.** The video is extracted with its long side at most 1024 px (`frames_max_long_side`; the one computation is
`intake/frames_manifest.reduced_size`, and F0 checks the frames against the cap the manifest declares). Every frame is
sealed by its sha256 in `frames/manifest.json`.

**Upright.** A video recorded with the phone held upright carries a rotation in its container and is stored
sideways. The decoder's own rotation is left off and the container's rotation is applied to every decoded frame,
explicitly (`frames_manifest.UPRIGHT`: 90° → a clockwise quarter turn, 180°, 270° → a counter-clockwise one; an
exact permutation of the pixels, recorded in the manifest as `orientation_applied`), so the frames on disk — and
everything computed from them — are as the video is watched. The explicit turn is the decoder's own automatic
rotation, bit for bit (measured on a real recording and held by tests on tagged videos); a video with no rotation
is extracted exactly as before. Sessions extracted before 2026-11-14 keep the frames they were computed on.

### 5.1 I0 — frame quality (`server/intake/quality.py`)

Each frame is read as grey $g$.

- **FFT sharpness.** $g'$ is $g$ area-downscaled so that its long side is at most 640 px, and $F=\operatorname{fftshift}(\mathcal F_2 g')$.
  Let $r=\lfloor\min(h',w')/8\rfloor$ and let $C$ be the central $2r\times2r$ square. Then
  ```math
  \mathrm{fft}=\frac{1}{h'w'}\sum_{(u,v)\notin C}|F(u,v)|.
  ```
- **Laplacian variance:** $\operatorname{Var}(\nabla^2 g)$.
- **Luma:** $\bar\ell=\operatorname{mean} g$.
- **Clipped share:** $c=\#\{g\le5\vee g\ge250\}/N$.
- **Inter-frame difference:** $\Delta_t=\operatorname{mean}|g^{(320)}_t-g^{(320)}_{t-1}|$.
- **Sharpness rank:** $\rho_i=\bar r_i/(n-1)$, the average rank of $\mathrm{fft}$ over all frames, ties averaged.

The only rejection is exposure: a frame is dropped when $\bar\ell<40$, when $\bar\ell>220$, or when $c>0.25$.

### 5.2 Session camera (`server/intake/focal.py`)

Sixteen evenly spaced frames go through DA3 at native resolution, rounded up to the 14-px patch. Each predicted grid
intrinsic is mapped to native pixels by $f^n_x=f^g_xW/w_g$ and $c^n_x=(c^g_x+\tfrac12)W/w_g-\tfrac12$, and likewise in $y$.
The session $K$ is the per-parameter median, and the spread $\hat\sigma/\operatorname{med}$ is reported. The probe also
keeps its sixteen depth maps and their grid $K$ (`intake/focal_depth.npz`): they are the metre of AMB3R's map (§6).

### 5.3 I1 — parallax keyframes (`server/intake/parallax.py`)

**Tracking.** From an anchor frame $a$:

- a 32×32 seed grid at ¼ scale is tracked into each following frame by pyramidal Lucas–Kanade (window 21, 3 levels),
  forward and backward;
- the target is first photometrically matched to the anchor: $\tilde g=\operatorname{clip}(\operatorname{rint}((g-\mu_g)\sigma_a/\sigma_g+\mu_a))$;
- a track survives when both directions succeed and its forward–backward error is at most 1 native px;
- a frame with fewer than 40 tracks is *lost*.

**Rotation.** Let $T$ normalise the image (centre it, scale by $2/\sqrt{W^2+H^2}$) and $K_n=TK$. The inter-frame rotation
is the symmetric-transfer least-squares fit of the infinite homography $H=K_nR(\omega)K_n^{-1}$:

```math
\hat\omega=\arg\min_\omega\sum_i\big\|\pi(H\tilde a_i)-\tilde b_i\big\|^2+\big\|\pi(H^{-1}\tilde b_i)-\tilde a_i\big\|^2
```

The solver is trust-region with an analytic Jacobian. It starts from the previous $\omega$, or from the rotation nearest to
$K_n^{-1}H_{\text{RANSAC}}K_n$.

**Readings.** The parallax that rotation cannot explain is

```math
p=Q_{0.9}\Big\{\tfrac12\big(\|\pi(Ha_i)-b_i\|+\|\pi(H^{-1}b_i)-a_i\|\big)\Big\},\qquad r=\big|\operatorname{rem}(\|\hat\omega\|,2\pi)\big|.
```

**Quanta.** The parallax quantum is an angle, $\theta_p$ = `parallax_quantum_deg` = 1.774°, read in the session's pixels
as $q=f_x\tan\theta_p$ (12 px at $f_x\approx387$, where it was calibrated; `intake/parallax.session_view`). The rotation
quantum is

```math
\theta_q=(1-\tau)\,\tfrac14\,\mathrm{FOV}_h,\qquad \mathrm{FOV}_h=2\arctan\frac{W}{2f_x},\qquad \tau=0.3 .
```

**Keyframe rule.** Frames are scanned from the anchor, with the band fraction $b$ = `keyframe_band_frac` = 0.25.

- When $p>(1+b)q$, the keyframe is the sharpest frame whose parallax lies in the band $[(1-b)q,(1+b)q]$.
- Otherwise, when $r>(1+b)\theta_q$, the same rule applies to the rotation band.
- When tracking is lost after a measured baseline ($p\ge q_w$, the witness quantum below), the keyframe is the sharpest
  frame with $p\ge(1-b)\max p$.

The keyframe becomes the next anchor.

**Never more than 8 frames without a keyframe** (`intake/parallax.stride_fill`, AMB3R-SLAM's own completeness rule,
`parallax.max_window_frames` = 8). After the chain: every fixed block of 8 video frames ($\lfloor\text{frame}/8\rfloor$) that
holds no keyframe gets its sharpest usable frame; then any two consecutive keyframes more than 8 frames apart get frames
inserted at even positions until none is. Added keyframes are marked `closed_by: stride`.

**Witness frames.** A parallel chain marks a witness at every keyframe, at every lost frame, and every
$q_w=f_x\tan(0.222°)$ of parallax (`witness_min_parallax_deg`; 1.5 px at the calibration focal).

---

## 6. The first pass: AMB3R-SLAM over the whole video

`server/reconstruction/amb3r_probe.py` runs **AMB3R-SLAM** (`vendor/AMB3R-SLAM`: upstream 5418465 with the STAC commits
recorded as `server/tools/amb3r/amb3r_stac.patch`) right after I1, over
**every frame of the video** (frame stride 1), at the session's Ω grid rounded to the model's patch. Its role is to say
where every frame is and what every frame saw. It never decides the keyframes or the cubes.

**What it runs.**

- A DA3-Small front-end tracks every frame. It promotes a keyframe when the overlap with the last one falls below 0.35,
  or after one backend step.
- The heavy model (VGGT-Ω) builds submaps of 24 views spanning 96 frames, every 32 frames. A submap takes one frame per
  group of $k$, the front-end's most confident, with $k=\operatorname{round}(4\min(\mathrm{fps},10)/10)$, so $k=4$ for any
  video at 10 Hz or more.
- Submaps are joined by Sim(3) edges, and DBoW retrieval proposes loops every 5 frames at least 100 frames apart. The
  verified loops enter an offline pose graph.
- Every frame gets a pose. A frame inside submaps takes the weighted average of their poses, down-weighted at a submap's
  edge. A frame no submap reconstructed takes the Sim(3) interpolation between its two reconstructed neighbours.

**What each reconstructed frame saw.** The light map (`--stac_light_map`) keeps, for every reconstructed frame, one record
placed by its submap's final Sim(3) $(s,R,t)$. With the frame's pose $T=[R_T\,|\,t_T]$ and its points $p$ in the submap:

```math
z=s\,\big(R_T^\top(p-t_T)\big)_z,\qquad c2w=\big[R\,R_T\ \big|\ s\,R\,t_T+t\big].
```

- $z$ is kept on a 4-px grid, and is 0 where the map drops the pixel: the frame's lowest 20 % of confidence, depth edges
  (more than 3 % within 3×3) and sky.
- The pinhole $K$ the points follow is fitted by least squares of $u=f_xx/z+c_x$ and $v=f_yy/z+c_y$ over the valid pixels,
  with the median pixel residual recorded (pccr: 0.29 px).

Every point of this cloud therefore carries the frame it came from. A view of ours, a keyframe or a turn frame, **sees
what its own frame saw** when AMB3R reconstructed it, and otherwise what the nearest reconstructed frame saw, at most half
a backend step away (ties go to the earlier frame). No ray is cast.

**Product.** `intake/amb3r_video.npz` holds every frame's pose, AMB3R's own keyframes (recorded) and the records. It is
stamped by the video (the frames manifest's sha256 inventory), AMB3R's commit, the launcher, the model, the grid and the
strides, and reused while they are the same. AMB3R runs under the pipeline's strict deterministic torch
(`reconstruction/amb3r_launch.py`), on 8 threads over 16 cores, offline.

On pccr the pass took 11 min over 3044 frames and reconstructed 761 of them; on pccr 2026-08-24, 52 min over 13 692
frames (3423 reconstructed, 4 loop closures, 15.4 GB peak).

**The map in metres** (`amb3r_probe.metric_scale` / `make_metric`). AMB3R's heavy model is VGGT-Ω, whose units are
arbitrary, so before anything reads the map it is scaled once by Depth Anything 3 on the focal probe's sixteen frames
(§5.2). For each probe frame $f$ the record nearest to it is projected into $f$'s camera (AMB3R's pose, DA3's $K$ on DA3's
grid), giving AMB3R's depth $z_p$ and DA3's measured depth $D_p$ at the same pixels:

```math
r_f=\operatorname{med}\big\{D_p/z_p:\ z_p\le Q_{0.25}(z)\big\},\qquad
s_{\text{AMB3R}}=\operatorname{med}\big\{r_f:\ r_f\in[P_{10},P_{90}]\big\}\ \text{metres per unit}.
```

A frame needs 50 valid pixels and the scale needs three frames, or the stage fails. The poses' translations, the records'
camera centres and depths are multiplied by $s_{\text{AMB3R}}$, and the npz records the scale and the stamp of the probe it
was measured on (a later probe replaces it, never compounds it). One scale for the whole video is a first
approximation, declared; the spread is logged (pccr 2026-08-31: 2.638 m/unit, MAD 6.3 %). Every downstream reader that
applies the range refuses a map that is not in metres.

---

## 7. The coverage map and the keyframe fill

The intake's keyframes are a first proposal of coverage. **The map decides**: `server/reconstruction/coverage_fill.py`
completes them where the video saw something that no keyframe sees, or that only one keyframe sees.

**The map.** Every valid pixel of every record is a point, $X=c2w_r\big(z\,K_s^{-1}[u,v,1]^\top\big)$. Here $K_s$ is the
record's $K$ with its first two rows divided by the stride, because grid pixel $(j,i)$ is model pixel $(4j,4i)$. A pixel
whose point lies beyond the range, $\delta>R$ (§4), is dropped from its record before anything else: it neither enters
the map nor proves a keyframe sees anything.

**Seeing.** A keyframe $g$, through its own record, sees $X$ when the point falls inside $g$'s image, in front of it, at a
depth consistent with what $g$ saw there:

```math
z_g(X)>10^{-3},\qquad (u,v)=\operatorname{round}\pi_g(X)\in\Omega_g,\qquad D_g(u,v)>0,\qquad
|z_g(X)-D_g(u,v)|\le\mathrm{tol}\cdot D_g(u,v).
```

Occluded, or in front of what $g$ saw, it does not.

**The session's tolerance and noise.** Take the relative disagreement $\rho=D_j/z-1$ of each keyframe's points in the
next keyframe, with $|\rho|<0.25$. Then:

```math
\mathrm{tol}=3\cdot1.4826\operatorname{med}|\rho-\operatorname{med}\rho|,\qquad
\eta=\frac{\#\{|\rho-\operatorname{med}\rho|>\mathrm{tol}\}}{\#\rho}.
```

$\eta$ is the share of samples on one surface that the test itself calls inconsistent: its own miss rate (pccr: tol
2.70 %, $\eta$ = 4.22 %).

**Covered.** A point is **seen** when some keyframe sees it. It is **covered** when a second keyframe sees it from a ray
at least `refine.min_tri_deg` (2°) away from the first one's, $\angle(C_g-X,\,\mathbf d_1)\ge2°$, which is F5's bound
for a landmark. A point not seen, or seen by one view alone, is **weak**. (`coverage_fill.py --min-support n` asks $n$
such views; the session asks one. From 2026-11-07 to 11-08 it asked F6's ten — "ahora necesitamos cobertura" — and
doubled pccr's keyframes, 313 → 686: the cubes, the time and the witnesses' baseline doubled or halved with them, a far
door vanished, a floor duplicated; USER 2026-11-08 "corremos todo así como vos proponés": back to one, and what is far
is covered by F6's witnesses chosen by their baseline, §14.6.)

**The fill.** A candidate is any frame AMB3R reconstructed that I0 found usable and that is not a keyframe's own record.
Its gain is the share of what it saw that it would see first, or add a view to from far enough while it is not covered:

```math
g(c)=\frac{1}{|P_c|}\,\#\big\{X\in P_c:\ \neg\mathrm{seen}(X)\ \vee\ \big(\mathrm{seen}(X)\wedge\neg\mathrm{covered}(X)\wedge
\angle(C_c-X,\mathbf d_1(X))\ge2°\big)\big\}.
```

- While some candidate has $g(c)>\eta$, the one with the largest gain becomes a keyframe. Ties go to the sharper (I0's
  sharpness rank), then to the earlier.
- What it sees enters the map, and the other gains are measured again. A gain can only fall as keyframes are added, so
  the lazy search is exact.
- Every keyframe I1 chose stays. What no frame of the video can cover beyond the noise stays weak and is declared, for
  example a turn on the spot or a corner glimpsed once.

**Products.** The added frames enter `frames/selected_frames.json` with `closed_by: coverage`, and
`frames/witness_frames.json`. The selection keeps I1's own list, so a re-run always starts from I1's keyframes and an
unchanged input is not filled twice. `intake/coverage_fill.json` records the coverage before and after and every frame
added. The fill runs on the card, in a process of its own.

On pccr, with one second view as the bar: I1's 295 keyframes left 0.7 % of the map unseen and 6.4 % seen once; 18 frames
were added (313 keyframes). With the publication's support (10 views ≥ 2°) as the bar, 391 frames were added (686
keyframes) — the run of 2026-11-07/08, judged worse: "heredamos los problemas de antes con menos completitud y mucho más
tiempo de procesamiento" (6 h 28 min, SAM3 alone 3 h). Since the stride rule of §5.3 the intake leaves little to fill:
pccr 2026-08-24's 2747 keyframes left 0.7 % unseen and 18 frames were added; zaragoza's 279, none.

---

## 8. I3: multi-view metric depth windows and the walk

**Window size.** The keyframes are covered by overlapping windows of $N_w$ frames, sized from the card's measured footprint:

```math
N_w=\min\Big(32,\Big\lfloor\frac{T(1-m)-W_{\text{wts}}}{p\,\tau_f}\Big\rfloor\Big)
```

- $T$ is the card's memory and $m=0.15$ the margin.
- $W_{\text{wts}}$ is the weights' footprint.
- $p$ is the memory per token, measured.
- $\tau_f$ is the number of tokens per frame.

Windows overlap by half, and the last one is shifted to end on the final keyframe. DA3 runs at the frames' native
long side rounded up to its 14-px patch, capped at 1024 (`gauge.max_long_side`).

**Joint inference.** DA3-NESTED infers each window jointly. Inside the nested model the metric branch is aligned to the
multi-view branch by least squares:

- $d^m=d^{\text{metric}}\bar f/300$, with $\bar f$ the mean focal length;
- $s_w=\sum_\Omega d^m d/\sum_\Omega d^2$, where $\Omega$ is the set of pixels with confidence above its median, not sky,
  $d^m>0.01$ and $d>10^{-3}$;
- depth and translation are both scaled: $d\leftarrow s_wd$ and $t\leftarrow s_wt$.

**Chaining.** Windows are chained through their shared frames. With $M_f=T^{\text{chain}}_f(T^{\text{win}}_f)^{-1}$, the
rotation is the chordal mean $\bar R=\Pi_{SO(3)}\big(\sum_f R(M_f)\big)$ and the translation is
$\bar t=\overline{c^{\text{chain}}_f-\bar R\,c^{\text{win}}_f}$.

**Walk and anchors.** The **walk** (chainage) is $\sum_k\|C_{k+1}-C_k\|$. Each keyframe's DA3 depth, confidence and $K$
are taken from the window where it is most central. These are the metric anchors read downstream.

**Revisit reference** for loop retrieval:

- the distance bar is the median of the windows' median depths;
- the angle bar is $\cos(\tfrac12\operatorname{med}_w\mathrm{FOV}_h)$;
- each bar's error is its bootstrap standard error.

---

## 9. Turn frames and interval witnesses

`server/intake/turn_frames.py` adds frames **between** consecutive keyframes from the I1 readings. It plays two roles.

**Turn frames, role *context*.** Let $\theta_t=\theta_q/3$, the turn step (`turn_frames.per_rotation_quantum` = 3), and
$b=0.25$. A keyframe interval whose rotation reading is $r_k$ is cut into

```math
n=\max\Big(1,\Big\lceil\frac{r_k}{\theta_t(1+b)}\Big\rceil\Big)
```

steps. When $n\ge2$, one frame is chosen at each target $j\,r_k/n$ for $j=1,\dots,n-1$: the sharpest frame within
$\pm b\cdot r_k/n$ of the target, else the nearest reading.

**Interval frames, role *witness*.** For straight intervals, with `min_per_interval` = 1, one frame is chosen near half
the interval's parallax.

How each role is used:

- **Context frames** are inferred *with* their cube by VGGT-Ω. They give the network the turn it would otherwise see as a
  jump, and their slots are dropped before saving. They also vote in F6.
- **Witness frames** are inferred in a **separate pass after each cube**. They never change the cubes or the poses; they
  only testify in F6's vote.

A keyframe added by the coverage fill lies inside one of I1's windows. The turn frames are cut from I1's readings of that
window, and never on a keyframe.

---

## 10. Cubes: the views grouped by what they see, and the Ω resolution

### 10.1 The cut (`server/reconstruction/cubes.py`)

**The rule** (USER 2026-11-07, night): "los cubos son de 5 m, no los decide nadie … y el solape con los cubos vecinos es
el 50 % de los kf que hay en los cubos"; and one Ω pass holds 64 views at most (USER 2026-11-04: "ese es el límite";
2026-11-08: "corremos todo así como vos proponés", after the cubes with no view cap did not hold together — below).

- **Units.** A unit is a parallax keyframe with the rotation keyframes up to the next parallax keyframe, and the turn
  frames after them. A turn is never cut.
- **Co-visibility length.** For each keyframe $k$ (walk order), $\ell_k\ge1$ is how many of the following keyframes,
  without a gap, still share at least $\tau$ = 0.3 of its view (the larger of $c(a\!\to\!b)$, $c(b\!\to\!a)$ below,
  `cubes.covis_lengths`). A turn in place keeps sharing (long $\ell$), a corridor stops soon (short $\ell$).
- **A cube** (`structure.cut: covis`, version 10). Along the walk, the units in order while the cube's motion budget
  and its keyframe count both hold:

  ```math
  \sum_{k\in\text{cube}}\frac{1}{\ell_k}\le H=13.67,\qquad \#\text{keyframes}\le 64\ (\texttt{structure.chunk\_view\_cap}).
  ```

  $H$ is `reconstruction/chunk_covis.H_LENGTHS_PER_CHUNK`, calibrated on pccr (the largest budget validated good). The
  cap counts keyframes; the turn frames ride along as VGGT-Ω context (a cube's Ω views can exceed 64 by them). A turn in
  place costs almost nothing and never closes a cube (station 2026-11-08: cubes of 5 m that were a turn in place, with
  the camera moving under 1 m, gave seams of 50–76 cm). A cube takes at least two units, so that neighbours always share
  one; a unit alone over the cap is never cut.
- **The overlap.** The next cube starts at the unit where half of this cube's keyframes lie behind it, so neighbouring
  cubes share half their keyframes: the seams of the Sim(3) graph (§12).
- **Small cubes fuse** (`structure.min_chunk_keyframes` = 32). A span with fewer keyframes merges into the adjacent span
  with fewer keyframes (ties: the earlier), smallest offender first, until every span holds 32 or one remains (a seam of
  99.8 cm stood on a joint of 5 shared frames).
- **The owner** of a unit's keyframes is the cube where the unit lies deepest (farthest from both of its ends).

The map the cut reads holds only points within the range (§4). The cut by metres (`cut: metres`: the units' cameras
inside a 5 m box, ≤ 64 Ω views) stays selectable. The cuts before them (k-d camera cells, rays against AMB3R's cloud,
boxes of seen space, groups by what the units see, 5 m with no view cap) are gone; the last, on pccr's 686 keyframes,
gave cubes of 94–358 keyframes whose seams read 5.9 cm median and 15.8 cm max after the graph, a floor twice 6 cm apart
and a table's edge bent 4–5 cm. Today's cuts: pccr 2026-08-31, 17 cubes of 35–64 keyframes; zaragoza, 8 of 55–64;
pccr 2026-08-24, 86 of 35–68.

**A cube's views.** Its Ω views are its keyframes plus the turn frames that lie between two of its keyframes consecutive
in the walk. Its F6 witnesses, the interval frames, follow the same rule.

**What each keyframe sees of every other.** The same measurement with one keyframe on each side, for every two keyframes:

```math
c(a\!\to\!b)=\frac{1}{|S_a|}\,\#\big\{X\in S_a:\ b\text{'s record sees }X\text{ with }|z-d|\le 0.25\,d\big\},
```

with the band of **one surface** (0.25, the band the session's tolerance of §7 is measured in) instead of the tolerance:
two moments of the walk whose geometry disagrees still see the surface each other saw, and those are the ones to tie.
The pairs where one sees **any part** of what the other saw, $\max\big(c(a\!\to\!b),c(b\!\to\!a)\big)>0$, are kept
in the plan (`kf_covisibility`; USER 2026-11-08: "ver lo mismo al 30 % no necesariamente es un número con sustento, ven
lo mismo es ver total o parcialmente el sector" — until then $\tau$ = 0.3). After the cut
nothing tied two moments of the walk that see the same surface from different cubes: F4 tracked along the walk only and
F6 voted among walk neighbours only, so two cubes that disagree left the surface twice (observatorio's wooden wall and
signs, 2026-11-07). These pairs give F4's revisit pairs (§14.3) — the same pairs bridge the cubes in the Sim(3) graph
(§12) — and the co-visible witnesses of F6's vote (§14.6). At
the session's tolerance, pccr's map never saw the walk's end meet its start (AMB3R closed no loop there); on one surface
it does.

**Products.** `intake/cubes.json` and `intake/cubes.png`, stamped by the probe, the inputs and the code. The viewer draws
each cube (**Instances** tab, `reconstruction/cube_boxes.py`) as a box around the cameras of all its views (cubes that
share keyframes overlap), carried into
the live frame by the similarity between the probe's and the published keyframe cameras.

### 10.2 Ω resolution

The resolution is the largest multiple of 16 not above $\min(16\lceil\max(W,H)/16\rceil,\ 1024)$ at which the largest
cube fits the card:

```math
\mathrm{cap}(R)=\Big\lfloor\frac{T(1-m)-4.0}{\kappa\cdot0.086\cdot g_wg_h/(832\cdot464)}\Big\rfloor,\qquad
R^*=\max\{R:\ N^*\le\mathrm{cap}(R)\}
```

- $N^*$ is the largest cube's view count, context or witness views included.
- $\kappa$ is the committed footprint factor of the card table.

An out-of-memory fails the run. It is never silently retried lower.

---

## 11. The VGGT-Ω pass, cube by cube

The fork (`vendor/VGGT-Long`, `loop_utils/cube_run.py`) infers each cube whole. For a cube with keyframes $\mathcal F$ and
context frames $\mathcal C$, the network sees the views $\mathcal F\cup\mathcal C$, sorted, the first being a keyframe. It
returns per view:

- a pose encoding, which gives world-to-camera and $K$;
- a depth $d$;
- a depth confidence $c$.

The world points are $X=R_c\,[(u-c_x)d/f_x,\ (v-c_y)d/f_y,\ d]^\top+t_c$.

**Sky masks.** The sky segmenter's score $o$ is min–max normalised per image, $\hat o=\lfloor255(o-\min o)/(\max o-\min o)\rfloor$,
and the mask is $m=\mathbf 1[\hat o<32]$ (non-sky). The confidence becomes $c\leftarrow c\,m$. The masks of every
keyframe, context and witness frame are precomputed in 12 processes while I3 runs on the card.

**Context records.** Each context view's depth, confidence, $K$ and relative pose to its preceding keyframe are saved,
then every per-view array is cut to the keyframe slots. The **witness pass** infers keyframes ∪ interval frames
separately and saves only the witness slots.

**Confidence floor.** Per cube,

```math
\theta=\max\big(P_{20}(c_{\text{valid}}),\ c_{\min}+0.10\,(c_{\max}-c_{\min})\big),
```

and a pixel stays when $c\ge\theta$ and $c>10^{-5}$. The same 0.10 min–max fraction is the viewer slider's arithmetic.
It also gates the samplers of every pose fit.

---

## 12. Metric lock and the cube Sim(3) graph

**Metric lock.** For each frame, $r_f=\operatorname{med}(d^{\text{DA3}}/d^{\Omega})$ over pixels valid in both, in the
near band $d^\Omega\le Q_{0.25}$ and within the range (DA3's pixel distance $\le R$, §4). A cube is brought to the DA3
metre by its ratio $s^{\text{DA3}}_k=\operatorname{med}_f r_f$, applied in place to its file and recorded there (a resume
never scales twice). A cube without anchors takes the anchored cubes' median, and says so.

**Seams.** Every two cubes that share keyframes are tied by a **seam**: the Sim(3) between their two copies of the shared
frames. The correspondences are exact, the same pixel of the same shared frame in both cubes, up to 200 000 of them, and
only pixels within the range from their camera in both copies. The fit is a weighted Umeyama alignment with Cauchy IRLS
weights (`loop_bridges.robust_sim3`):

```math
w_i=\frac{1}{1+(r_i/c)^2},\qquad c=\max\big(3\cdot1.4826\,\mathrm{MAD}(r),\,10^{-4}\big),\qquad 8\ \text{iterations}.
```

A seam is always fitted with **free scale** (USER 2026-11-09: "ajustalo libre"); a rigid fit stands in only when no Sim(3)
can be fitted at all. Each seam carries:

- $\sigma_e$, the fit's median residual;
- a centroid $c_e$, the median of the overlap points;
- a lever $L_e$, the median camera-to-point distance.

**The session's scale bound** (`cube_run.scale_bound`). Over the seams' log-scales $\ell_e=\ln s_e$, with
$m=\operatorname{med}\ell$ and $\sigma=1.4826\,\operatorname{med}|\ell-m|$, a relative scale $s$ is degenerate when

```math
|\ln s-m|>1.1\cdot z_{0.975}\cdot\sigma ,
```

the session's own 95 % interval widened by the project's factor (no bound below three seams). It is logged for the seams
and judges the closures (zaragoza: $|\ln s+0.0044|\le0.0468$; pccr 2026-08-31: $|\ln s-0.0248|\le0.1523$).

**Revisit closures** (`cube_run.bridges` / `closures`). F4's own revisit pairs (§14.3) over the probe's co-visibility
are handed to the graph (`Model.cube_revisits`). A revisit $(i,j)$ whose two keyframes lie in cubes $a,b$ that share no
frame is **bridged**: one joint VGGT-Ω pass over F4's loop window around each end (`tracks.loop_half_window` = 4
keyframes each side). The metric lock runs first, then every bridge is inferred and **fitted at once** onto its two
locked cubes, on exact correspondences (the same frames in both passes, within the range from the cube's camera):

```math
X^{a}\approx S_a\,X^{\text{bridge}},\qquad X^{b}\approx S_b\,X^{\text{bridge}},\qquad
M_{b\to a}=S_a\,S_b^{-1},\qquad s_{b\to a}=s_a/s_b .
```

Only the two fits are kept (`bridge_fit_*.npy`, a few kB; pccr 2026-08-24 has 1787 bridges, and keeping each 252 MB
prediction filled the disk). A closure enters the graph only when its relative scale passes the session's bound; each
side's own scale is the bridge's raw VGGT-Ω scale against a metric cube and is not judged. The closure's $\sigma$ is
$\sqrt{\sigma_a^2+\sigma_b^2}$, its centroid and lever cube $a$'s. zaragoza: 115 bridges of 134 revisits, 82 closures in
the solve; pccr 2026-08-31: 49 of 75, 43.

**The graph** (fork `loop_utils/chunk_graph.py`). Every cube is a piece of **one** Sim(3) graph.

- **Unknowns.** For $k\ge1$, $C_k=(e^{\lambda_k},\exp[\omega_k]_\times,t_k)$. The gauge is $C_0=I$: cube 0 whole, at its
  metric lock.
- **Residual** of an edge $(a,b,M)$, with $E=(C_b^{-1}\circ C_a)\circ M^{-1}$:

  ```math
  \mathbf r_e=\frac{1}{\sigma_e}\Big[\,E(c_e)-c_e\ ;\ \operatorname{Log}(R_E)\,L_e\ ;\ (\ln s_E)\,L_e\,\Big]\in\mathbb R^7 .
  ```

- **No scale prior** (USER 2026-11-07). The seams alone place the cubes: each is an exact fit of the same pixels in two
  passes, and the cubes must coincide there. The DA3 per-cube scales are recorded, not used. A DA3 prior per cube fought
  the seams wherever the locks disagree (pccr: neighbouring locks 0.876–1.167 apart). At $0.08/\sqrt{n_k}$ (1.1–1.5 %,
  each cube's anchors taken as independent) the solve broke two seams by 16 %, 18 and 24 cm between the copies of the
  same surfaces, while the log read "0.0 cm" because it measured the seam's centroid only. At the locks' measured
  spread ($\sigma_{\text{lock}}=1.4826\,\operatorname{med}_e\lvert\ln s_e\rvert/\sqrt2$ = 8.8 %) it still bent three
  seams by 6–10 cm. F6's 10-view support floor then emptied the keyframes at those transitions: #19–25, #55–65 and
  #187–199 published 0–2 k points each, against a median of 27 k. The metre is `scale_align`'s (every anchor of the
  session); the scale along the walk is F2's continuous gauge (§14.2), a smooth curve of the walked distance that
  cannot put a step at a transition. The locks' spread is still measured and recorded (`locks_spread_log`).
- **Cost.** The seams squared (an exact fit of the same pixels is never down-weighted); the revisit closures under
  Cauchy, so a wrong one is down-weighted, never trusted like a seam,

  ```math
  \min_{\{\omega_k,t_k,\lambda_k\}}\ \tfrac12\Big[\sum_{\text{seams}}r_j^2+\sum_{\text{closures}}\ln\big(1+r_j^2\big)\Big],
  ```

  solved by trust-region reflective least squares (tolerances $10^{-12}$), started from the spanning tree of the seams
  alone, best $\sigma$ first. If the solve does not converge, the spanning tree is applied, declared. The log reports
  every seam at its surfaces, the scale and the rotation read at the lever arm (`edge_surface_error_m`): zaragoza
  3.5 cm median / 11.2 cm max with the cubes moved at most 18.9 cm; pccr 2026-08-31 1.7 / 4.8 cm, moved at most 61.4 cm.

**Not run on cubes, declared.** The SALAD loop candidates are not run: the revisits come from AMB3R's map (§10.1). A
`loop_closures.txt` or revisit reference left by an earlier chunk run is deleted before Ω runs. Neither are the per-chunk
scale ladder and its linear drift run, since a cube has no walk order to drift along.

**Application.**

- Points: $X\leftarrow s_kR_kX+t_k$.
- Depths: $d\leftarrow s_kd$.
- Poses: $[R_kR_c\,|\,s_kR_kc+t_k]$.
- **Ownership.** Each keyframe's points, pose and $K$ come from the cube that owns it. The pixels the owner drops are
  written by one other copy (backfill).

**Depth graph.** Pairwise affine relations $z_g\approx\alpha z_f+\beta$ are fitted at keyframe offsets {1, 2, 3, 5, 8, 12},
with offsets {4, 10} held out. They are solved for per-frame $(a_f,b_f)$ with a mean-zero gauge. The result is applied, as
$p'=c+(p-c)(az+b)/z$, only if it is bounded and the decision rule approves it.

---

## 13. Global metric scale and the upright frame

**Scale.** For each frame, $q_p=z^{\text{DA3}}_p/z^\Omega_p$, over the pixels within the range (DA3's own intrinsics,
§4). The frame's ratio $s_f$ is the median of $q$ over the near band ($z^\Omega\le P_{25}$) intersected with DA3's
top-10 % confidence, with fallbacks. The global scale is

```math
s=\operatorname{med}\{s_f:\ s_f\in[P_{10},P_{90}]\}.
```

It multiplies point coordinates, depths and camera centres. On a cube run, $s$ is also how far the locked and joined cubes
sit from DA3's metre. It is applied and recorded, with no bound (pccr: 0.891).

**Orientation.** Gravity is the normalised mean of the cameras' "down" axes, $\hat g=\operatorname{normalize}\big(\tfrac1N\sum_i\operatorname{normalize}(R_ie_2)\big)$.
It is refused when the mean alignment $\tfrac1N\sum|d_i\cdot\hat g|$ is below 0.7. The frame is rotated by Rodrigues'
formula, $R=I+[v]_\times+[v]_\times^2/(1+c)$ with $v=\hat g\times(0,-1,0)$ and $c=\hat g\cdot(0,-1,0)$. The 1st percentile
of the heights of the points within the range of their chunk's cameras is then set to 0.

---

## 14. The precision core, F0 → F6

The precision core (`server/precision/`) measures and corrects the reconstruction against its own evidence.

**Chain.** With `cloud.source: omega_bent`, the runner executes:

```text
F0 camera → F2 gauge → F4 tracks → F5 refinement → F6 bend → chunk check (report)
```

What only reported and corrected nothing was removed on 2026-11-10 — F3 (§14.4), F5's fourth rung and its temporal
test, the rolling-shutter diagnostic and the prior-depth cross-check: nothing that does not correct geometry costs
machine time.

**Epochs.** F2 and F5 publish pose-only geometry epochs. F6 publishes the new cloud epoch.

**Resume.** Every step is resumed against sha256 stamps of its inputs, the configuration sections it reads and its code
(the code stamp follows each step's imports, so a change to `precision/config.py` restarts every step). When the first
step to run comes before any step that moved the geometry and the session is not in epoch 0, the runner first returns to
VGGT-Ω's epoch 0 from the stored lineage (§2) and runs from F0; a session with no epoch 0 kept fails there, naming it.

### 14.1 F0 — the session camera (`precision/camera.py`)

**Camera model.** One OPENCV camera $\theta=(f_x,f_y,c_x,c_y,k_1,k_2,p_1,p_2)$ in native pixels. The network's grid is
mapped to the native frame exactly, through the vendor's crop, resize and pad:

```math
x_n=(x_g-p_x+\tfrac12)\,s_x-\tfrac12+o_x,\qquad f^n_x=s_xf^g_x,\qquad c^n_x=(c^g_x-p_x+\tfrac12)s_x-\tfrac12+o_x .
```

**Session K.** The session $K$ is the component-wise median of the per-frame VGGT-Ω intrinsics, read from Ω's own record
(`output/maplong_run/intrinsic.txt`; `output/intrinsic.txt` is F6's epoch artifact, the camera of the published cloud).
The focal spread $100\cdot1.4826\operatorname{med}|f_i-\hat f|/|\hat f|$ is reported, with a warning above 2 %.

**Undistortion.** Points are undistorted iteratively (200 iterations, $\epsilon=10^{-12}$). Each one is re-distorted and must
land within $\epsilon+16$ ulp of its observation.

### 14.2 F2 — the continuous metric gauge (`precision/gauge.py`)

F2 gives each keyframe a metric scale $s_k$ that varies **continuously with the distance walked** (chainage $c$).

**Per frame.** The log ratio $r_p=\log z^{\text{DA3}}_p-\log z^\Omega_p$ is located by a Huber estimator:

- start $\mu_0=\operatorname{med}r$;
- scale $\sigma_t=\hat\sigma(r-\mu_t)$;
- weights $w_p=w^0_p\min(1,k\sigma_t/|r_p-\mu_t|)$, with $k=1.345$;
- update $\mu_{t+1}=\sum wr/\sum w$;
- stop when the change falls below Higham's floating-point bound.

**Per window.** Each I3 window gives a row $y_w$, the confidence-weighted median of the frame gains. Its $\sigma_w$ is the
bootstrap standard deviation, and its position $c_w$ is the mean chainage. DA3's pixels beyond the range carry zero
confidence (§4).

**Other instruments.** Each yields rows of the same kind:

- DA3 monocular;
- VIO arc length;
- LiDAR (Stray);
- known dimensions;
- relative visit-drift rows.

**Model.** $x(c)=\log s(c)$ is piecewise linear on knots every $\Delta=2$ m of walk:

```math
\hat x(\lambda)=\arg\min_x\sum_i\Big(\frac{a_i^\top x-y_i}{\sigma_i}\Big)^2+\lambda_{\text{eff}}\,\|D_2x\|^2,\qquad
\lambda_{\text{eff}}=\lambda\,\frac{\sum_i\sigma_i^{-2}}{\max(1,\,c_{\text{reach}}/\Delta)} .
```

- $a_i$ are hat-function weights; a relative row uses the difference of two hats.
- $D_2$ is the second difference.

**Choosing λ.** λ is picked from $\{10^{-3},\dots,10^{3}\}$ by leave-one-window-out. The pick is the largest λ whose
held-out RMS is within 1.1 standard errors of the minimum.

**Choosing the instrument.**

- Each instrument's scatter (the MAD of its leave-one-out residuals) floors its rows' σ.
- The default instrument is the first available one.
- Another instrument replaces it only if the decision rule says it improves the paired held-out error.

**Application.** The scale is $s_k=\exp(h(c_k)^\top\hat x)$. Camera centres are re-integrated along the walk:

```math
C'_0=C_0,\qquad C'_k=C'_{k-1}+\tfrac12(s_k+s_{k-1})(C_k-C_{k-1}),
```

with rotations unchanged and $s_k$ recorded as a depth factor about each camera. F2 refuses to run unless the live epoch
is 0, because it measures against VGGT-Ω's own depth.

### 14.3 F4 — native-pixel tracks (`precision/tracks.py`)

**Tracker.** A learned tracker (VGGSfM v2: feature maps, a coarse predictor with 6 iterations, a fine refinement with a
score) runs on CUDA under deterministic torch.

**Windows.**

- Windows of 24 keyframes with step 12, and 2 query frames each.
- **Revisit windows** (`revisit_pairs`): the keyframes within 4 of both ends of a pair that sees the same surface in
  AMB3R's map (§10.1), queried at both ends, so the tracks cross from one moment of the walk to the other and F5 aligns
  the two moments' poses on landmarks both see. A pair the walk already ties (a keyframe window holds keyframes within
  12, half a window, of both ends) gets none. The pairs that see the most of each other go first; a pair whose two ends
  lie within 12 of an accepted pair's two ends is tied through it and the walk's windows. On the data of 2026-11-07:
  pccr 6 windows, among them its end and its start (1, 309); zaragoza, where every view sees every other, 54.
- Loop windows around SALAD pairs (chunk layouts only) and the re-identified objects' visits.
- Witness windows spanning the witness frames between two keyframes.

**Queries.** A 48-column grid plus up to 512 Shi–Tomasi corners, at least $f_x\tan(1.263°)$ apart on the tracker's grid
(`corner_min_distance_deg`; 6 px at the calibration focal). A query is dropped where VGGT-Ω's depth jumps by more than
5 % within 2×2.

**Observations.**

- An observation is kept where the visibility is above 0.2; a track needs at least 2 frames.
- Coordinates are mapped to native pixels.
- A track is **held out** when the first 8 bytes of SHA-256(seed, query frame, $u$, $v$), read as an integer, fall below
  $0.2\cdot2^{64}$. This is a deterministic 20 % split, independent of order.

### 14.4 F3 — resolution probe (removed)

F3 inferred the central 32 keyframes at 512, 768 and 1024 and reported the projective depth mismatch of each pair. It
decided nothing — the Ω resolution is §10.2's — and was removed from the chain on 2026-11-10, with its step, its module
and its configuration section. The number of this subsection is kept so that the references to §14.5 and §14.6 hold.

### 14.5 F5 — joint refinement (`precision/refine.py`, `precision/ba_torch.py`)

**Unknowns.** World-to-camera $(R_k,t_k)$, the free part of θ, and the landmarks $X_j$. Each landmark starts as a
multi-view DLT of a fit track, accepted only in front of every camera and with a widest ray angle of at least 2°. An
observation on a keyframe whose track, triangulated with the starting poses, lies farther than the range from that
keyframe's camera is dropped before the fit (`refine.within_range`).

**Projection** with the OPENCV lens. With $x=X/Z$ and $r^2=x^2+y^2$:

```math
x_d=x(1+k_1r^2+k_2r^4)+2p_1xy+p_2(r^2+2x^2),\quad y_d=y(1+k_1r^2+k_2r^4)+p_1(r^2+2y^2)+2p_2xy,\quad
u=f_xx_d+c_x,\ v=f_yy_d+c_y .
```

**Objective.** Robust reprojection plus a prior that keeps the cameras near the gauge:

```math
E=\tfrac12\sum_{jk}\rho\big(\|\pi_\theta(R_kX_j+t_k)-u_{jk}\|^2\big)+\tfrac12\sum_k\Big\|\frac{C_k-C^g_k}{\sigma_k}\Big\|^2,
\qquad \rho(s)=\begin{cases}s,&s\le a^2\\ 2a\sqrt s-a^2,&s>a^2\end{cases}
```

- $a=1$ px.
- $C_k=-R_k^\top t_k$, and $C^g_k$ is F2's camera centre.
- $\sigma_k=\sigma_g\big(\sum_{i<k}\|C^g_{i+1}-C^g_i\|^2\big)^{1/2}$: the gauge's measured scatter, growing along the walk.

**Solver.** Levenberg–Marquardt in float64 on the GPU, with deterministic sorted segment sums and no atomics.

- **Perturbation:** $R\leftarrow\exp([\delta\omega]_\times)R$, $t\leftarrow t+\delta t$.
- **Robust weighting:** residuals and Jacobians are scaled by $\sqrt{\rho'}$.
- **Schur complement on the landmark block**, with damping clipped to $[10^{-6},10^{32}]$:

  ```math
  S=H_{cc}-W(H^\lambda_{pp})^{-1}W^\top,\qquad (S+\lambda\operatorname{diag}S)\,\Delta c=-\big(g_c-W(H^\lambda_{pp})^{-1}g_p\big),\qquad
  \Delta p=(H^\lambda_{pp})^{-1}(-g_p-W^\top\Delta c).
  ```

- **Step control:** gain ratio $\varrho=(E-E')/L$. Accept when $\varrho>10^{-3}$, with $\lambda\leftarrow\lambda\max(\tfrac13,1-(2\varrho-1)^3)$; on rejection, $\lambda\leftarrow\nu\lambda$ and $\nu\leftarrow2\nu$.
- **Stop:** $|\Delta E|/E<10^{-6}$ or a negligible step. Anything but convergence fails the stage. A problem that no
  landmark reaches is refused: a bundle adjustment of the priors alone is no solve.

**Massive sessions** (pccr 2026-08-24: 13.0 M landmarks, 101.6 M residual blocks). Two bounds on memory, neither a
decision:

- The track groups are held as arrays only, and the lens undistortion runs in chunks of whole tracks (at most 2 M
  observations). As Python lists the groups took 66 GB per grouping and the batched undistortion's Jacobian 32 GB,
  and F5 went over the pod's 117 GB.
- Above 24 M residual blocks the solver works in **chunks of consecutive landmarks** of at most 8 M observations. A
  landmark is never split, so every landmark block, every Schur pair and every back-substitution stays inside one
  chunk. The camera system is summed chunk by chunk in a fixed order, and the per-observation Jacobians are made again
  per chunk instead of kept. The equations are the same and the solve is deterministic; below the threshold it is the
  one-piece path, bit for bit.

**Rungs.** The camera is freed in rungs:

- R0: θ fixed.
- R1: $+f_x,f_y,c_x,c_y$.
- R2: $+k_1,k_2,p_1,p_2$.

**The judge.** A rung enters only if the decision rule says it improves the **held-out** tracks:

- $h_t$ is the RMS reprojection, in native px, of held-out track $t$ re-triangulated with the rung's poses and camera;
- clusters are each track's first observing keyframe;
- $\varepsilon$ is the solver error, measured by a continuation solve.

**Witness frames.** They are localised afterwards by LO-RANSAC PnP against the chosen rung's landmarks, with θ fixed.
They need at least 12 correspondences, and their inlier RMS must not exceed the 95th bootstrap percentile of the
keyframes' leave-one-view-out error $P_{95}(\rho_k)$ — the session's own bound (pccr 2026-08-31: 2.26 px, R2 applied,
held-out 0.534 px; zaragoza: 42.24 px, R1, 0.398 px).

### 14.6 F6 — VGGT-Ω's depth bent onto F5's landmarks (`precision/depth_on_f5.py`)

F6 keeps VGGT-Ω's dense depth, which is complete and smooth, and corrects it toward F5's sparse metric truth.

**Rows.** Fit and held-out tracks are re-triangulated with F5's poses and camera. Each observation in keyframe $i$ gives a
row $(u,v,z_{\text{lm}})$, kept only when $\delta(u,v,z_{\text{lm}})\le R$ (§4).

**Bend model.** Per keyframe, a planar depth factor over the image:

```math
k_i(u,v)=c_0+c_1\tilde u+c_2\tilde v,\qquad \tilde u=\frac{u-W/2}{W},\ \tilde v=\frac{v-H/2}{H},\qquad z_b=k_i\,z_\Omega .
```

**Fit.** The target is $r=z_{\text{lm}}/z_\Omega$, on the rows of keyframes $i-w,\dots,i+w$. Ten Huber IRLS steps
($k=1.345$) start from least squares:

- residual $e=r-Ac$;
- scale $s=1.4826\operatorname{med}|e|$;
- weights $\omega=\min(1,ks/|e|)$;
- covariance $\Sigma_c=\hat s^2(A^\top\Omega A)^{-1}$, with $\hat s^2=\sum\omega e^2/(n-p)$.

A tilt term $c_m$ is kept only if $|c_m|\ge1.1\,\sigma_{c_m}$. With fewer than 20 rows, $k=1$.

**Window.** The held-out rows are split into halves A and B. On half A the error is
$S(w)=\operatorname{med}|z_\Omega k_w-z_{\text{lm}}|/z_{\text{lm}}$ for $w\in\{0,1,2,4,8\}$. The chosen window is
$w^*=\max\{w:\ S(w)\le S_{\min}+1.1\,\mathrm{se}\}$, with se from a keyframe-cluster bootstrap.

**Verification.** Each keyframe's $c_0$ is verified by the decision rule on half A. Half B reports the final error, in
metres too, and that is the σ floor downstream.

**Validity.** A pixel is valid when $z>0$ and its confidence exceeds $10^{-5}$ (the sky sentinel). It *passes the floor*
when its confidence is at least $\min+0.10(\max-\min)$ of its VGGT-Ω cube.

**PointDiT at the edges only** (`precision/mono_detail.py`). PointDiT-H/16 (16 Euler steps from zeros, 512-px tiles with
¼ overlap) gives an affine-invariant depth. Each tile is aligned to the bent depth:

- the fit $z_b\approx s\,z_{\text{mono}}+b$ uses Huber IRLS;
- the tile's residual is $\hat\sigma\big((z_b-z_{\text{al}})/z_b\big)$;
- a tile is rejected beyond the 95th percentile of those residuals, or below 5 % support.

The detail is transplanted as

```math
z'=\mathrm{LP}_\sigma(z_b)+\big(z_{\text{al}}-\mathrm{LP}_\sigma(z_{\text{al}})\big)
```

(a masked, normalised Gaussian low-pass). It is accepted **only within $f_x\tan(1.807°)$ of a depth discontinuity**
(`detail_zone_deg`, read with F5's focal; 12 px at the calibration focal) and where $|z'-z_b|/z_b\le\tau_0$. Surfaces stay
VGGT-Ω's. Mixed pixels at a discontinuity (a side reach of $f_x\tan(0.452°)$, `side_reach_deg`) are assigned to the front
or back side when $z_{\text{al}}$ agrees with it, and are otherwise left out of the measurement tier.

**The witnesses.** Every candidate pixel of keyframe $f$ (valid, and passing the confidence floor or lying on a depth
edge) is lifted to $X=C_f+z\,\mathbf r(u,v)$ and put to:

- its walk neighbours at offsets $\pm3,\pm6,\pm12$;
- the turn and interval witnesses of its interval;
- the **co-visible witnesses** (`covisible_witnesses`): every keyframe that sees **any part** of what $f$ saw in
  AMB3R's map (§10.1), at any distance along the walk (USER 2026-11-08: "que voten todos … siempre"), listed the widest
  baseline first (pccr 2026-08-31: a median of 96 per keyframe, max 157; zaragoza: 272 of its 279).

**The tolerance** $\tau$ is the 75th percentile of the relative disagreement between interior pixels of neighbours at
offsets $\pm3,\pm6,\pm12$, measured on this session (`bend.tau_quantile`; pccr 1.43 %, zaragoza 0.92 %).

**COLMAP's fusion rule** (`cloud.fusion_rule: colmap`, `precision/f6_torch.two_sided_vote` with `fusion`). For a
witness $g$ with depth map $d_g$, let $z_g$ be the point's depth in $g$ and $(u_g,v_g)$ the pixel it lands on. Carry $g$'s
own surface at that pixel back into $f$, at pixel $(u',v')$, and write $\alpha_g=\angle(C_f-X,\,C_g-X)$ for the
triangulation angle at the point. Then $g$

```math
\textbf{confirms}\iff \|(u',v')-(u,v)\|\le f_x\tan\theta_r\ \wedge\ \Big|\frac{z_g-d_g}{d_g}\Big|\le\tau\ \wedge\
\alpha_g\in[1.5°,\,30°]\ \wedge\ \|X-C_g\|\le R,
```

```math
\textbf{sees through}\iff \frac{z_g-d_g}{d_g}<-\tau\ \wedge\ \alpha_g\in[1.5°,\,30°]\ \wedge\ \|X-C_g\|\le R ,
```

with $\theta_r$ = `fusion_max_reproj_deg` = 0.301° (COLMAP's 2 px at pccr's $f_x$ = 380.3; read with F5's focal) and the
band `fusion_tri_min_deg` / `fusion_tri_max_deg` (1.5° since 2026-11-09, "los 3 grados bajalo … es un número
arbitrario"; 30°). $n_c$ counts the confirming witnesses, $n_t$ those that see through. A witness **occludes** $X$
when its surface, splatted into $f$ by z-buffer and within the range from $g$, lies in front of the point,
$z-s_g>\tau s_g$, and $g$ does not see through it; $n_o$ counts them.

**Kept.** A pixel is published at the median depth of its confirming witnesses when

```math
\delta(u,v,z)\le R\ \wedge\ n_c\ge 5\ \wedge\ n_c>n_t\ \wedge\ \neg\,(n_o>n_c),
```

that is: born within the range, at least `min_support_views` = 5 confirmations (COLMAP's `min_num_pixels`), the
**majority** (`fusion_majority`: more witnesses confirm it than see through it — a floater confirmed by its own two
neighbours but seen through by four leaves), and the **occlusion veto** (`fusion_occlusion_veto`, Merrell et al.,
ICCV 2007: a pixel that more witnesses measure a surface in front of than confirm it is the hidden copy of a surface,
and leaves).

**In place** (`fusion_in_place`, 2026-11-09). A pixel not kept — short of confirmations, in the minority or vetoed —
is still published, **at its own depth**, when its in-range witnesses' surfaces on its ray agree among themselves and
with it: with $S$ the witnesses' splatted depths on $f$'s ray,

```math
\bar s=\operatorname{med}S,\quad A=\{s\in S:\ |s-\bar s|\le\tau\bar s\},\quad |A|\ge5,\quad
\Big|\operatorname{med}A-z\Big|\le\tau z .
```

It is never moved: a pixel whose witnesses agree on another depth stays out. (The repair to the witnesses' median,
2026-11-08, moved pixels and painted their birth colour at the moved depth, and is off: `fusion_repair: false`.) These
are mostly surfaces seen from close by and from one side only — the views that see them lie on the same line of sight,
under the 1.5° band.

**The support** written in the cloud's `confidence` column is $n_c$ for a kept pixel and $|A|$ for one published in
place. The vote's report (`output/precision/depth_on_f5.json`, `vote`) counts every outcome over the valid pixels:

| | pccr 2026-08-31 | zaragoza |
|---|---|---|
| valid pixels | 203.9 M | 146.2 M |
| kept | 55.3 % | 36.4 % |
| in place | 6.7 % | 11.1 % |
| left under the support of 5 | 20.8 % | 35.6 % |
| left by the majority | 6.8 % | 7.5 % |
| left by the occlusion veto | 5.0 % | 8.5 % |
| born beyond the range | 0 % | 8.1 % |
| **coverage** (published of all pixels) | **59.9 %** | **42.2 %** |

The other rules of `cloud.fusion_rule` stay selectable and are not the pipeline: `tau` (the edge-keeping vote of
2026-11-07, with repair and rescue) and `adjust` (the strongest consensus along the ray decides the position).

**Turn and interval witnesses.** These are the context and witness records of §11. Each is carried into the anchor
keyframe's metric frame:

- its depth is scaled by $\phi=\operatorname{med}z_{\Omega,a}/\operatorname{med}z_a$ and bent by the anchor's $k_a$;
- its pose is $c2w^{F5}_a[R_{\text{rel}}\,|\,\phi\,c_{0,a}t_{\text{rel}}]$.

They vote on the keyframes within ±12 of their anchor and never become points.

---

## 15. The published cloud

**Points.** Every keyframe pixel that the vote kept with its support (§14.6) becomes a point:

```math
\mathbf X=c2w^{F5}_f\big(\hat z(u,v)\,K_{F5}^{-1}[u,v,1]^\top\big),
```

coloured from the undistorted frame. Each point records:

- its birth keyframe (`frame_global`);
- its pixel (`pixel_row`, `pixel_col`);
- its support (`confidence`): the witnesses that confirm its depth (at least 5), or, for a pixel published in place,
  the witnesses whose surfaces agree with it (§14.6);
- its source: VGGT-Ω bent, PointDiT detail, band front or band back.

**Cleaner** (`reconstruction/stream_clean.py`, the in-memory `gpu_cloud_clean.py` its path of record), on the GPU:

- A 5 mm world-anchored voxel (`postprocessing.voxel_size`) keeps the **real point nearest each cell centre**, never an
  average.
- **Statistical outlier removal** (`sor_knn` = 8, `sor_sigma` = 3): $\bar d_i$ is the mean distance to the 8 nearest
  neighbours. Points with no neighbour within $\max(3\cdot\text{voxel},\,1\text{ cm})$ = 1.5 cm are dropped and kept out
  of the statistic. The rest are kept when $\bar d_i\le\mu+3\sigma$.

USER 2026-11-09: "el filtro a 1 mm y el SOR aflojado no dejan más geometría, sino que dejan más puntos que no aportan
geometría" — the 1 mm / σ 6 recipe of 2026-11-08 was reverted to 5 mm / σ 3 (zaragoza: 69.5 M raw points → 23.3 M; pccr 2026-08-31:
126.5 M → 18.8 M).

**By pieces.** The cloud is never merged in memory. The chunk PLYs are memory-mapped in place; which points survive is
a byte mask on local scratch (`postprocessing.clean_scratch_dir`); the voxel pass cuts slabs of whole cells and picks
each slab's survivors on the card; the SOR runs slab by slab with a **halo** of whole cells, and a point's result is
taken only when its 8th distance is certified inside the halo (every point outside lies farther), the rest re-run with
a 4× wider halo. The statistic is numpy's own reduction over the live points in the cloud's order. The output is
byte-identical to the in-memory path (`tests/test_stream_clean.py`), with memory independent of the cloud's size.

**Publication.** The cloud is published as a new geometry epoch inside a journaled transaction, then converted to a
Potree octree. The LAS quantum is the smallest power of two $s$ with extent$/s<2^{29}$. PotreeConverter runs on
`postprocessing.potree_threads` (16) threads and rewrites every chunk in a **canonical order** before the hierarchy is
written (`vendor/PotreeConverter`, `stac_canonical`), so the octree is byte-identical for any thread count.

---

## 16. Chunk check (report)

`precision/chunk_check.py` checks, per cube and per keyframe, the floor band, the ceiling band and the camera's
height over the floor. DA3's own camera-to-floor distance serves as an independent confirmation.

**Floor plane.** A seeded RANSAC (400 iterations, 2 cm) gives the floor plane, its normal oriented upward.

**Bands.** Heights are $h=(X-c)\cdot n$. A band is the median of the points within 8 cm of the cube's floor (or of the
5th percentile), and likewise for the ceiling (97th percentile). Each band needs at least 200 points, and its resolution
is $\hat\sigma$.

**Departures.** Departures between cubes are judged by the decision rule, with a two-sample bootstrap and
$\varepsilon$ = the band's resolution.

**Verdicts.**

| Verdict | When |
|---|---|
| ok | the floor does not depart |
| level | the floor departs and the ceiling does not: a real level change |
| depth | the floor and the ceiling depart in opposite directions |
| pose | the floor and the ceiling depart together |
| intra | the floor trend departs within the cube |
| undecided | anything the evidence cannot decide |

The report lists what it would correct. Nothing is applied.

---

## 17. Scene understanding: VLM → SAM 3.1 → instances

> **The VLM proposes, describes and names. It never measures.** Every number comes from deterministic tools over the
> geometry. Every label that enters a product is tagged `vlm_proposed`, `tool_measured` or `human_validated`.

**The chain.** Four stages of the pipeline manager (§1), each a process of its own, one at a time on the card:

```text
VLM round 1 → SAM 3.1 round 1 → [ VLM round k → SAM 3.1 round k ]… → projection (once) → floor
```

- The chain's array work — decoding frames, reading and writing masks, matching them to points, the boxes — runs on
  **one declared device**, `surface_fit.evidence_device` (`cuda`; `cpu` in the tests). Each port keeps its CPU function
  as the path of record, and a test holds the two equal (§19.1).
- SAM 3 and the projection work on the reconstruction's **product epoch**. It is made live before SAM 3 starts, and a
  product epoch that is no longer on disk fails the stage before a frame is touched.
- **Nothing partial is delivered.** A prompt SAM 3 did not run to the end fails the stage and the previous store
  stays; a projection step that cannot run fails the projection and no result is written.

### 17.1 The vocabulary (`server/segmentation/autoprompt/`)

**Model and determinism.** **Qwen3-VL-8B-Instruct** is served by vLLM on `127.0.0.1:8799` with:

- temperature 0, seed 0;
- eager execution, no prefix caching;
- `VLLM_BATCH_INVARIANT=1`.

Re-asked calls are compared byte for byte.

**The stage's own engine** (`semantic/service.job_engine`). The stage takes the engine lease, stops whatever vLLM was
up (a verified stop), requires a card with no other process on it, and launches vLLM from the job's **frozen**
configuration. It verifies the engine's identity (the weights' sha256, the card, the versions), writes it into
`vlm_analysis.json`, and stops the engine when it ends. When the session's analysis is reusable under its stamp, that
identity is computed without a launch and no engine is started. A service that does not come up within its bound, a
call that fails or a configuration error fails the stage: there is no fallback model.

**The calls.** Each keyframe gets one call (`autoprompt.vlm_sampling.all_keyframes`, one view per keyframe). The frame
is decoded on the card with PIL's pixels and PIL's header (§19.1), so the JPEG the model receives is the same bytes
either way. The model returns JSON with:

- the scene type;
- the structural envelope, in a field of its own;
- the objects: category, description, and a shape with form, material and detail.

The rules it is asked to follow: whole objects and never their pieces, one thing per entry, one name per object, the
envelope is never background, and every distinct thing gets its own entry.

**Names to prompts.**

- Names are normalised: lower-cased, articles and counts dropped, singularised.
- Each name is counted by the number of keyframes that named it.
- A merge pass asks short yes/no questions: is it a whole object or a piece? same kind as? a unit of a support surface?
  The links are closed transitively and never form cycles. Only a support surface absorbs its parts and units
  (`fold_only_into_support`), and two names are the same kind only when they share their head noun
  (`same_kind_same_head_only`).
- Structural names go first. The list is bounded at `max_sam3_prompts` = 150.
- Every phrase's fate is recorded in `output/segmentation_census.json`.

**Rounds** (`autoprompt/rounds.py`, `autoprompt.segmentation_rounds: true`). After SAM 3 has run, round $k\ge2$ shows
the VLM every keyframe of round 1 again, with every pixel some masklet already covers painted flat grey (128), and
asks the same question plus one instruction: name only what is not grey.

- **The grey frame** is built on the card. The masks of a frame are inflated there and OR-ed; the frame is decoded
  there; the mask is brought to the frame's grid by OpenCV's own nearest-neighbour samples and painted with one
  `torch.where`. Only the JPEG encode for the VLM stays on the CPU.
- **What is fresh.** A name already prompted (round 1, the store, an earlier round — whether SAM 3 found anything or
  not) or already judged by an earlier merge is not fresh. A name whose head noun names no kind of thing (`object`,
  `item`, `thing`, …) is never a prompt. A name that one keyframe alone proposed is dropped (two witnesses), unless
  it is structural (`structural_always`).
- **The merge.** The fresh names go through the merge pass together with the prompts so far. Each ends as *covered* by
  a prompt already segmented, *merged* into another name, or a *new prompt*. Structural names go first and the total
  never exceeds `max_sam3_prompts`.
- **SAM 3** runs the new prompts over every keyframe and appends their masks to the same store.
- **The end.** The loop stops when a round names nothing new, or when the prompt bound is reached. A round that
  starts with the bound already full is not run at all: no engine is launched, the round is recorded with no new
  prompt (pccr 2026-08-24 reached 150 prompts in round 1).

Every round is recorded in `output/seg_rounds.json` (what each keyframe proposed, the fate of every name). The masks
of all the rounds are projected on the cloud once, after the last one.

### 17.2 Masks (`server/segmentation/sam3_wrapper.py`, `sam3_group.py`, `pipeline.py`)

**Model.** **SAM 3.1 Object Multiplex** (`vendor/sam31`), run in bf16 under strict deterministic torch (TF32 off,
cuDNN deterministic, fixed seeds). The stage stops the semantic service (verified) and refuses a shared card.

**Frames on the card** (`segmentation/sam3_frames_gpu.py`). The model's lazy frame loader is replaced at load time —
no vendor file is edited. Each frame is decoded on the card (PIL's pixels, §19.1), resized to the model's square by
Pillow's own fixed-point bilinear passes, and normalised by a gather from a (3, 256) float16 table that the vendor's
own CPU arithmetic computes once. No float is computed on the card, so the tensor is the vendor's bit for bit; a frame
costs 4.1–4.5 ms instead of 30–36 ms.

**Propagation.** For each text concept and each batch of 250 keyframes (overlap 10), the prompt is added on the first
frame and propagated forward. Every concept runs on every keyframe: the VLM gives the vocabulary only, no frame lists
and no box seeds.

**Concept groups** (`models.segmentation.concept_group` = 4). SAM 3.1 propagates one concept per session, and its
detector used to compute the vision backbone (the 32-block ViT at 1008 px and its three necks) of every 16-frame chunk
again for every concept. Now four concepts advance together over a batch:

- the first runs in the batch's own session, the others in sessions cloned over the same decoded frames;
- their propagations are stepped in turn, frame by frame, so every member processes the same frames in the same turn;
- the backbone of each chunk is computed once and the same tensors are served to every member.

The image features do not depend on the text and the inference draws no random number, so **each concept's masks are
the ones it gets alone**. Measured on pccr 2026-08-31 (250 keyframes, 8 concepts, the card exclusive): 333.8 s one at
a time, 236.6 s in groups of four (×1.41), all 4448 mask and label arrays identical bit for bit. A concept its group
could not complete (the tracker's stop, an error) runs alone on its own call; a group that runs out of memory hands
every member back to run alone.

Two more settings come with the groups:

- `keep_frame_outputs: false` — the vendor's per-frame cache of every object's video-resolution mask (its interactive
  demo's) is not kept, and a session's state is reset as soon as its masks are copied out.
- `planning_copy: shallow` — the vendor deep-copies every object's score on every frame propagated so far, again on
  every frame (22 % of a concept on 120 frames, growing with the frames). The shallow copy copies each frame's dict
  and shares its score tensors: no writer touches a past frame, and the masks are identical.

**Identity.**

- Two ids are the same object when their mask IoUs on the frames they share pass the decision rule's "above a bar" form,
  with a bar of 0.90.
- Across batches, objects are matched greedily by median IoU over the overlap frames.
- Appearance is never used: tracking is the identity.

The pixel counts behind both run on the device (`segmentation/sam3_pairs_torch.py`): a frame's masks are uploaded as
booleans, the intersections are products of 0/1 rows in float32 — every partial sum an integer below $2^{24}$, so
exact — and the IoU is the same float64 division of the same two integers.

**Thresholds** (SAM 3.1): detection 0.3, new detection 0.5, confirmation after 3 consecutive frames.

**A stop is continued, never dropped.** A propagation can stop midway for two reasons: the vendor tracker's own stop,
or CUDA running out of memory (one concept with hundreds of objects). In both cases the frames it completed are kept,
and the rest of the batch is propagated again from (stop − overlap) in a fresh session, linked by the same overlap rule
as any two batches. With no progress the overlap is reduced, down to a seam whose objects get fresh ids. Every
continuation is declared in the census (`identity_decisions`). An out-of-memory stop depends on the card: the same
card with the GPU exclusive stops at the same frame. The only out-of-memory that fails the stage is a session that
cannot complete even one frame. Any other SAM 3 error fails the stage.

**Fallback.** A bare category that confirms nothing is retried with its origins — the VLM's descriptions of it, then
the names merged into it, at most `sam3_fallback_max` = 3 — and the first that finds something is kept under the
category's label.

**The mask store** (`output/seg_masks.npz` + `segmentation.json`).

- **Appended while SAM 3 runs.** Each concept's masks are saved as it finishes, by a writer thread, while the next
  concept runs on the card.
- **Canonical.** At the end the store is rewritten with its members ordered by (object, frame) and a fixed zip entry
  time, with the keyframe list and the reconstruction's id as members: one logical store is one byte sequence.
- **Sealed.** `segmentation.json` carries a stamp of the mask bytes, the prompts, the keyframes, the model and the
  code. A reader refuses a store whose seal names other bytes, or whose swap into `output/` did not complete.
- **Deflated on the card** (`segmentation/store_deflate_gpu.py`). The members are laid out on the device as rows of
  `.npy` bytes, deflated by nvCOMP (raw Deflate streams), their CRC-32 computed there (`crc32_gpu.py`), and **every
  stream is inflated back and held equal to its member before it is written**. Measured on the real stores: pccr
  2026-08-31, 55 079 masks, 81.1 → 19.6 s; pccr 2026-08-24, 290 418 masks, 1280.1 → 93.3 s.
- **Declared.** The streams are nvCOMP's, not zlib's. Every member inflates to the record's bytes, but the file is
  another byte sequence and 2.2× larger (65.5 → 143.8 MB, 349.2 → 771.7 MB). The server's interactive edits keep
  zlib: the server process holds no CUDA context.

**The census** (`output/segmentation_census.json`, version 3) says what the VLM looked at, every phrase and its one
fate, and for every prompt its SAM 3 status and masklets. It is built from `segmentation.json` alone: the masks are not
read back. It is a compared artifact — no clock, no pid, no temporary path; the seconds per prompt go to
`segmentation_timing.json`.

**Cost.** SAM 3 is the slowest stage of the chain: 1 h 43 min on pccr 2026-08-31 (547 keyframes, 87 concepts) and
about 10.5 h on pccr 2026-08-24 (2765 keyframes, 150 concepts, about 4.1 min per concept), where the VLM took
48.5 min. The tracker's control logic is the vendor's Python and the card is about half busy during it.

### 17.3 Masks to cloud (`server/segmentation/pipeline.py`)

The projection is a pure function of its inputs: the cloud, the raw mask store, the keyframe list, the poses, the
session camera, the floor transform, the frozen configuration and the code. Its result carries the stamp of all of them
and the margin of every decision it took (`decisions`).

**Assignment.** Every point was born at one pixel of one keyframe; the exact grid maps of the session camera take that
pixel from the record grid to the mask grid. Its candidates are the objects whose mask contains the pixel, and it takes
the one with the **smallest mask area**, ties going to the lower id:

```math
o^*(p)=\arg\min_{o\,:\,m_{f,o}(u_p,v_p)=1}\ \big(|m_{f,o}|\cdot M+o\big),\qquad f=\text{the keyframe of }p,\quad M=\max o+1 .
```

A point can only be covered by the masks of its own keyframe, so the match walks the cloud **by frame** on the device
(`segmentation/mask_match_torch.py`): all the masks of a keyframe are inflated on the card in one batch, their areas
are pixel counts, and the winner and the runner-up are the two smallest integer keys. Contested points whose two
areas differ by less than 1.1 × the masks' own frame-to-frame area variation are counted and declared; the rule is
unchanged.

**Audit** (`segmentation/mask_filter.py`). The cloud is checked against the masks, and nothing is removed.

- An instance's points are projected into the keyframes where SAM 3 drew it, visit by visit. A visit is a run of
  consecutive mask keyframes (`visit_gap_kf` = 1), and its `max_frames_per_visit` = 8 keyframes showing most of the
  object judge it.
- The mask is dilated by 2 px (`dilate_px`); a point hidden by the cloud's z-buffer (0.15 m) or by the instance's own
  (0.10 m) is not judged.
- A point that lands off its own mask in any view that sees it is marked **out of place** (`out_of_place.npy`, with
  the count of views behind each mark).
- An instance whose mass is off the mask in one visit and on it in another is a `drift_duplicate`; off the mask in
  every visit, `unsupported` (`mask_audit.json`).

The masks of a visit are inflated and dilated in one batch on the device.

**Merges and splits.**

- **Space dedupe.** For instances $i,j$ with 5 cm voxel sets $V_i,V_j$, each keyframe where both have points is a judge:

  ```math
  J_f=\min\Big(\frac{|V_i(f)\cap V_j|}{|V_i(f)|},\ \frac{|V_j(f)\cap V_i|}{|V_j(f)|}\Big),
  ```

  and the pair merges when the "above a bar" rule passes at 0.5 **and the larger of the two is a support surface**
  (`segmentation.merge_support_only: true`, since 2026-11-11). Every other object stays as SAM 3 gave it: a merge by
  shared space sometimes joined two objects far apart, so the same object named twice by the VLM now stays twice, and
  every refused pair is recorded with its shares.
- **Fragments.** Fragments of a support surface that touch (at least 5 adjacent 10 cm voxel pairs) are united.
- **Co-visible split.** Co-visible components separated by free space are split.
- **Attach.** Unlabelled points within 3 cm take the majority label of their 3 nearest labelled points, over 4 rounds.
- **One owner.** A point belongs to one instance or to none; a conflict goes to the smaller instance.
- **Minimum size.** Instances under 1000 points are dropped, except structural ones.

Per-object cleaning is off (`models.segmentation.segment_cleaning.enabled: false`).

**Oriented boxes.** Built on the instance's core component:

- **yaw candidates** come from sequential RANSAC of vertical planes;
- the **footprint area** $A(\theta)$ is computed in each candidate frame;
- the **error** $\sigma_A$ comes from a Poisson bootstrap of the extreme points;
- the tied set is $\{\theta:\ A(\theta)-A_{\min}\le1.1\,\sigma_A(\theta_{\min})\}$, and the candidate with the largest
  support wins;
- the extents are the core's min and max.

The box is computed end to end on the device (`segmentation/obb_gpu.py`). The stable point keys, the stable sample and
the flyer components are the record's, exactly. The hull (gift wrapping), the rotating calipers and the extents are the
same construction in the device's floating point. The RANSAC triplets and the bootstrap's keep / drop flags are
**drawn on the card** by torch's generator, seeded from the configuration and from the point set: deterministic on one
card, but not the numpy record's draws. Measured against the record on pccr 2026-08-31: every object, point, class,
fusion and audit mark is identical, and 421 of 1363 boxes turned by more than 5° (median 1.5°, max 45°; centre p95
8 cm). For about a third of the objects the yaw depends on the RANSAC draw itself; more iterations, until the same
planes come out of every draw, is the open way to remove that.

**Products.** `segmentation_result.json` (with its stamp and decisions), `instance_ids.npy`, `classification.npy`,
`mask_audit.json`, `out_of_place.npy`, the instance store `scene_r.db`, and an octree with the class bytes — built by
the floor's epoch when the floor stage follows. The instance store is written on the pod's local disk and put back in
one copy (`phase_r/instance_store.store_on_local_disk`): sqlite on the network volume wrote a 1.8 GB journal page by
page for over half an hour on pccr 2026-08-24.

**Measured.** The whole projection of pccr 2026-08-31 went from 967 s to 535 s with the device ports (the second
figure taken while another job shared the card), with the same classes, instances, fusion map, audit and marks. The runs of 2026-11-12: zaragoza 408 segments; bufferStop 347,
covering 96.87 % of its 3 837 478 points; observatorio 529, covering 99.80 % of its 23 374 569 points.

**Known.** The audit looks an instance's masks up through the *fused* view of the store (`fusion_map.json`, written by
the previous projection). The SAM 3 stage deletes that map, so SAM 3 → projection is not affected; a projection run
again on its own audits the parts that map had folded as instances without a mask.

### 17.4 The floor (`workers/floor_worker.py`, `correction/floor.py`)

After every projection (`pipeline.auto_floor: true`; also at the end of a **Reconstruir** order), ONE rigid motion
levels the whole session (`correction.floor.model_default: session`, `solve_floor_session`):

- the candidates are the points of the **segmented support surfaces** — instances whose head noun is in
  `autoprompt.support_surfaces` (floor, ground, platform, slab, base, deck, walkway, pavement, andén); without them, the
  points below their own keyframe's camera, declared;
- levels are searched along the support's own normal; **the floor** is the level under the most cameras, and other real
  levels (a platform, a ramp's landing) keep their step;
- the tilt is applied, about the floor's centroid, only when it exceeds 1.1 × its own error; the height — the floor
  under the walk's start taken to $y=0$ — only when it exceeds 1.1 × its own;
- the motion is published as one transactional geometry epoch: the cloud, the poses, the segmentation with its boxes
  recomputed, the octree and the instance store. The epoch it replaced is discarded, so the session keeps one cloud.

The epoch's warp moves every point at once on the device (`correction/warp_torch.py`). A floor already level within
its own error publishes no epoch, and the octree of the live cloud is built here instead. The stage's record
(`floor_session.json`) names the live geometry it left and the segmentation it was found on; it is complete only while
both are still the session's.

A session filmed with the phone rotated comes out of the reconstruction on its side and is set upright here
(bufferStop: 92.68° on its segmented floor, 174 of 174 cameras over it).

---

## 18. GenRecon scene mesh

GenRecon runs only when ordered (`pipeline.auto_genrecon: false`): from the **GenRecon** button of the Instances panel
or an API order (§20), with the cloud's confidence threshold chosen for that order (the reference runs use 0 %: the
published cloud is already filtered by the vote). `workers/genrecon_worker.py` → `server/tools/genrecon/run_cloud.sh`,
mode `cloud`. The worker stops the semantic service and types the keys of the shape, the edge filter, the decode cap
and the devices before anything is launched: a missing key or a wrong type fails naming it. In order:

1. the resume check (§18.7);
2. the adapter: the confidence filter, the levels and the layout (§18.1, §18.2);
3. GenRecon's generation, and the sampled latents saved (§18.3);
4. the joint decode, in one pass or by regions (§18.3);
5. the mesh step: the global hole fill and remesh, the edge filter, the per-cube bakes in parallel (§18.4);
6. the raw mesh, published to `output/genrecon/`;
7. the photo bake, published to `output/tsdf/genrecon_photo/` (§18.5);
8. the 3D Tiles (§18.6).

**The rule of this stage: the cloud is the limit.** Presence (which voxels hold surface) comes from the cloud's points.
The model shapes the surface inside the active voxels and the keyframes are its texture. The mesh's meta says so:
geometry and material `vlm_generated`, base colour `tool_measured`.

**What was measured against the cloud, and what it changed** (2026-11-12, on the meshes of 3.7 m cubes with a solid
ground).

- **Surface where the cloud has none.** 25–72 % of each mesh lay beyond one voxel (11.56 cm) of the cloud: zaragoza
  2233 of 3090 m², pccr 2026-08-24 1376 of 3554, pccr 2026-08-31 259 of 625, observatorio 416 of 1253, bufferStop 23
  of 90. Of that area, 92–99.8 % sat in the voxels of the solid-ground fill (§18.2), which were 37–69 % of the active
  voxels. The fill is off since then (`genrecon.solid_ground: false`); on bufferStop that surface went from 22.6 to
  0.04 m².
- **Holes where the cloud is.** The cloud decides only which voxels are active; the geometry inside a voxel is
  generated. 99.5–99.8 % of the cloud had mesh within one voxel, but only 54–73 % within 3 cm, and from bufferStop's
  keyframes 3–12 % of the cloud's pixels met no mesh at all. The decoder's own mesh shows the same: the hole fill,
  the remesh, the simplification and the tiles add none of it.
- **Smaller cubes.** Only the 512 models exist, so the voxel is the cube / 32 and the surface cell the cube / 512: a
  smaller cube gives the model finer voxels and cells. Against 3.7 m, 2.775 m took the mesh beyond 3 cm of the cloud
  from 31.8 to 23.0 %, the cloud beyond 3 cm of the mesh from 48.2 to 31.0 %, and the holes seen from the keyframes
  from 10.7 to 7.9 %; at 1.85 m the model closed a tube's mouth with a membrane and turned gravel into blocks.

The cube and the points per voxel were then chosen by eye on bufferStop, among 3.7, 3, 2.775, 2.5, 2.31, 1.85 and
0.925 m. The three meshes that were published there, measured the same way:

| Cube · points per voxel | Mesh beyond 3 cm of the cloud | Cloud beyond 3 cm of the mesh | Holes seen from the keyframes |
|---|---|---|---|
| 2.775 m · 6 | 23.4 % | 31.1 % | 7.8 % |
| 3 m · 3 | 24.8 % | 32.7 % | 6.6 % |
| **2.5 m · 3** (in force) | **21.4 %** | **17.3 %** | **5.65 %** |

**The meshes on disk** (runs of 2026-11-13: 2.5 m cubes, 3 points per voxel, confidence 0, the edge filter on). The
numbers are each mesh's own records — `genrecon_photo.meta.json`, `tiles/tiles_report.json`, the layout — and its
stage log.

| Session | Points | Cubes | Latent voxels | Joint decode, card peak (shape / texture) | Remesh cell | Faces | Texels photographed | Tiles |
|---|---|---|---|---|---|---|---|---|
| bufferStop | 3.84 M | 24 | 7 014 | one pass, 4.24 / 4.41 GB | 4.89 mm | 1.96 M | 75.5 % | 35, depth 5, 0.20 GB |
| pccr 2026-08-31 | 18.8 M | 158 | 42 689 | one pass, 17.79 / 18.96 GB | 4.89 mm | 13.28 M | 86.8 % | 235, depth 7, 1.33 GB |
| zaragoza | 23.3 M | 230 | 99 387 | one pass, 24.48 / 26.78 GB | 4.89 mm | 18.94 M | 63.4 % | 333, depth 8, 1.78 GB |
| observatorio | 23.4 M | 251 | 109 523 | one pass, 31.87 / 34.68 GB (run before the cap of §18.3) | 4.89 mm | 19.83 M | 70.1 % | 353, depth 8, 2.08 GB |
| pccr 2026-08-24 | 114.5 M | 358 | 220 739 | 4 regions, 26.64 / 28.91 GB | **10.24 mm** | 33.26 M | 78.3 % | 635, depth 9, 3.66 GB |

**Known, not corrected.**

- The global remesh closes every sheet into a shell (the floor was seen as two sheets on 85–99.8 % of the rays), and
  the simplification to 150 000 faces per cube leaves faceting.
- The global remesh is the one step that still grows with the whole scene. On pccr 2026-08-24 it ran out of the card
  at the decoder's 4.89 mm cells and was retried coarser four times, down to 10.24 mm (§18.4), declared in the log.
- The generation is not identical run to run (§19). Loose floating pieces are not removed in this family.
- The work folders are keyed by the session's id (`vendor/GenRecon/work/<session>…`): a second scan of one session
  replaces the first one's.

### 18.1 The points GenRecon receives

**The frame.** The adapter (`tools/genrecon/adapter.py`) hands the scene over in the session's display frame — the
floor transform applied, so up is up — turned from $y$-up to $z$-up, with the keyframes' certified poses as COLMAP
world-to-camera and the session's OPENCV camera. The transform back to the raw frame is saved for the publication.

**The confidence filter.** The raw `confidence` (the vote's support, §14.6) is normalised by the octree's min and max,
exactly as the viewer's slider does (`PotreeLoader.ts`), and a point stays when

```math
\frac{c-c_{\min}}{c_{\max}-c_{\min}}\ge c^*,\qquad c^*\in[0,1)\ \text{the order's threshold}.
```

**Active voxels.** A voxel of GenRecon's lattice is active when at least `genrecon.min_points_per_voxel` = **3** of the
points that stayed lie in it. The value follows the cube: it was 10 on the 11.56 cm voxels of 3.7 m cubes (with one
point, any noise made mesh), and 3 was chosen against the holes on the 7.81 cm voxels of 2.5 m cubes.

**On the device** (`genrecon.adapter_device: cuda`, `tools/genrecon/layout_gpu.py`). The arithmetic over the cloud's
points — the frame change, the levels of the heights, each point's voxel and the count of every voxel, the objects'
stands — has float64 twins on one device:

- the frame change is written row by row, one rounding per operation, so the same bits on the CPU and on the card;
- the levels and the percentiles are read from one sort with numpy's own arithmetic;
- the voxel keys use a true division on the card too (a multiplication by the inverse moves a point lying on a voxel's
  face to the voxel before it).

Every file the adapter writes is byte-identical to the record's on pccr 2026-08-31 (18 770 254 points) and pccr
2026-08-24 (114 451 841 points), in 21.6 s instead of 28.0 s and 63.5 s instead of 121.2 s.

The keyframes' presence vote (`tools/genrecon/presence.py`: every point put to the other keyframes, kept when its
support reaches `presence_min_views` and no more keyframes see through it) stays in the code and is off
(`genrecon.presence_vote: false`).

### 18.2 Cubes over the whole cloud

`plan_layout` reproduces GenRecon's own chunker over the **exact extent** of the points that stayed, with no percentile
and no subsample. Cube side $L$ = `genrecon.cube_size_m` = 2.5 m, so the voxel is $v=L/32$ = 7.81 cm and the decoder's
surface cell $L/512$ = 4.88 mm.

**Plan ($x$, $y$).** The plan is padded by $\tfrac{3.5}{16}L$ on each side. With $D$ the padded extent and $\delta=L/16$,
take the fewest cubes $n$ whose stride

```math
s_n=\Big\lceil\frac{D-L}{(n-1)\,\delta}\Big\rceil\,\delta
```

is at most $\tfrac34L$, centred on the midpoint.

**Height ($z$), anchored on the floor** (`genrecon_levels.anchored_centers`). A *level* is a 5 cm slice of heights
holding at least 1 % of the points it is counted over. One cube base sits **exactly on the lowest level**, of the
segmented support surfaces or of the whole cloud (the model was trained with the floor at a cube's base). Layers rise
from it by the same stride rule, and continue down only where the cloud reaches lower, so the whole cloud is covered
and every cube stays on one lattice.

**Solid ground — off** (`genrecon.solid_ground: false`; `plan_layout(support_levels=…)`). When on, every column holding
an active voxel in the 5 cm slice of a support level is filled active from under that slice down to the lattice base,
so that no floor has air below it and the model paints only its top face. It was the answer to a double floor, and it
was where the mesh invented surface (above): the mesh now follows the cloud's whole excursion, below $y=0$ too.

**Lattice.** All cubes' $32^3$ grids form **one lattice**, because every stride is a multiple of two voxels. A cube is kept
when its block contains an active cell (summed-area table); each cube's active voxels are exactly the lattice's
(`layout.json`, `active_cells.npy`).

### 18.3 Generation and decode (`vendor/GenRecon`, Trellis.2-class)

**Frames.** For a camera with distortion GenRecon undistorts every frame it reads. With `genrecon.undistort_device:
cuda` the decode and the remap run on the card (`tools/genrecon/remap_gpu.py`): the JPEG decoded with PIL's pixels,
then OpenCV's 8-bit remap as OpenCV computes it — the integer bilinear table, $(\sum w\,p+2^{14})\gg15$, a neighbour
outside the image 0 — for the pixels of the crop alone. The vendor's own `cv2.imwrite` writes the result: 0 of the 916
files differ on bufferStop, observatorio and pccr 2026-08-31. No time is gained (about 3 s either way: the JPEG encode
that stays weighs the most). A frame with an EXIF orientation is refused.

**Conditioning.** Each active voxel of each cube is conditioned on the images that see it:

- the scene images are at most 999 keyframes, evenly spaced along the walk, each non-square frame as two square crops
  along its longer axis, at 512 px;
- DINOv3 ViT-L/16 features of them, encoded 64 images at a time;
- for each view, the patch feature at the voxel centre's projection, taken from the nearest patch;
- an IBRNet-style aggregation over the views:

  ```math
  F=\mu+\sum_n w_n\,\mathrm{MLP}_f\big([F_n,\mu,\sigma^2]\big),\qquad w=\operatorname{softmax}_n\mathrm{MLP}_w(\cdot).
  ```

The projection holds a tensor of voxels × images × features. Its voxel batch is cut so that voxels × images stays at
the working size of a 547-image scene,

```math
b=\max\Big(1,\ \min\Big(2048,\ \Big\lfloor\frac{2048\cdot547}{N_{\text{img}}}\Big\rfloor\Big)\Big),
```

and the aggregation is per voxel, so the numbers are the same: the projection's memory no longer grows with the number
of images.

**Sampling.** The sparse-structure stage is skipped: the active voxels are the coordinates. Two rectified-flow transformers
are sampled jointly over all cubes:

- the **shape structured latent** (32 channels per voxel);
- then the **texture structured latent**, conditioned on the shape.

Each uses 12 Euler steps with $s_k=1-k/12$ and $t_k=\frac{rs_k}{1+(r-1)s_k}$ ($r=3$), and the update $x\leftarrow x-(t_k-t_{k+1})v$.
Classifier-free guidance applies $v=w\,v_c+(1-w)v_\emptyset$ inside the guidance interval:

| Stage | $w$ | Interval | Rescale |
|---|---|---|---|
| Shape | 7.5 | [0.6, 1.0] | 0.5 |
| Texture | 5.0 | [0.6, 0.9] | — |

After every step, the features of voxels shared by overlapping cubes are replaced by their mean over the cubes. That is
what keeps the cubes consistent.

**The sampled latents are saved** before the decode (`tools/genrecon/latents.py`, `latents.pt`): one (features, voxels)
pair per cube for the shape and for the texture, and the cubes' transforms. The image features leave the card at the
same point. A decode that fails, or whose code changes, starts again from them (§18.7).

**Decoding.** The cubes' latents are merged on the lattice into one joint latent (the voxels that overlapping cubes
share are averaged), and decoded at 512 per cube, $L/512$ = 4.88 mm: the surface on a flexible dual grid (one vertex
per fine voxel, quads across the crossed edges), and a PBR attribute volume (base colour, metallic, roughness, alpha).
Both decoders are half-precision checkpoints.

**Decode by regions** (`genrecon.joint_decode_max_voxels` = 100 000, `tools/genrecon/decode_regions.py`). The vendor
decodes the whole scene in one pass, so the card's memory grows with the scene: 4.24 / 4.41 GB (shape / texture) for
bufferStop's 7 014 latent voxels, 17.79 / 18.96 GB for pccr 2026-08-31's 42 689, 31.87 / 34.68 GB for observatorio's
109 523. pccr 2026-08-24 holds 220 739, more than an 80 GB card takes. A scene above the cap is decoded by regions; at
or under it, or with the key at 0, the vendor's single pass runs untouched.

- **Why it can be cut.** Both decoders are local: stacks of 3×3×3 submanifold convolution blocks with per-voxel norms,
  and one subdivision block (a parent → its children) between levels. The structure is checked at run time and any
  other is refused by name. What a decoder writes on a fine voxel therefore depends on the latents within a finite
  radius of its ancestor at the latent level. With $B_\ell$ blocks at level $\ell$ (0 the latent level, $n$ the finest):

  ```math
  r_n=0,\qquad r_{\ell-1}=\Big\lceil\frac{r_\ell+B_\ell+1}{2}\Big\rceil+1,\qquad \rho=r_0+B_0 .
  ```

  For both decoders $B=(4,16,8,4,0)$ and $\rho=18$ latent voxels.
- **The margin** is $\max(\rho_{\text{shape}}+1,\ \rho_{\text{tex}\mid\text{shape}})$ = **19 latent voxels** (1.48 m at
  2.5 m cubes). The $+1$: a quad anchored on a fine voxel reads its neighbours one fine voxel further. The texture
  decoder's subdivisions are given by the shape decoder's, so the shape's radius at each level enters its own.
- **The cut.** A box whose decode — the box and its margin — receives at most the cap is a region. Otherwise it is
  cut across its longest axis at the median of its voxels, each half shrunk to its own voxels. It is a pure function
  of the voxels, the margin and the cap.
- **The merge.** Each region is decoded with its margin. From it are kept the vertices of the fine voxels whose
  ancestor it owns, the faces whose anchor (the per-axis minimum of their corners) it owns, and its texture voxels.
  A face looks its vertices up by voxel, so a quad that straddles two regions joins the two regions' own vertices and
  the mesh has no seam. The parts are put in the row order of the single pass, so the arrays are the same whatever
  the cut.
- **How equal.** In exact arithmetic the result is the single pass's. The decoders run in half precision and a region
  is another batch, so the last bits can differ. A face one of whose corners its owner region did not decode (two
  regions disagreeing on a subdivision bit) is not made, and is counted. Tests hold the logic exact on an integer
  decoder.

pccr 2026-08-24 is cut into 4 regions.

### 18.4 One surface, baked per cube (`inference/chunked_glb.py`)

**One surface.** GenRecon's hole filling (perimeter 1 m) and a narrow-band **dual-contouring remesh** at the decoder's
resolution produce **one** surface for the whole scene:

- every DC vertex is clamped to its own cell;
- vertices are projected back by $v'=v-0.9(v-\mathrm{proj})$, where the projection is the closest point of the surface
  (elsewhere the vertex stays in its cell).

**Out of memory in the remesh.** The retry runs at the largest resolution the remesher accepts that is not above 85 %
of the failed one (`tools/genrecon/remesh_resolution.py`). The remesher builds its grid by doubling a base of at most
32 cells, so it accepts a resolution only when halving it down to 32 never meets an odd number: 3712 → 3072, never
3155. Each retry is declared with its cell size.

**The edge filter** (`genrecon.edge_filter: true`, `tools/genrecon/denoise.py`, step 1.5 of the mesh step). Bilateral
normal filtering (Zheng, Fu, Au & Tai, 2011) of the global mesh: the last step that moves the geometry, after the
remesh and before the partition and the simplification. With $A_i$, $\mathbf c_i$, $\mathbf n_i$ the area, the centre
and the normal of face $i$, and $\mathcal N(i)$ the faces sharing a vertex with it, each of
`denoise_normal_iters` = 20 passes sets

```math
\mathbf n_i\leftarrow\operatorname{normalize}\Big(A_i\mathbf n_i+\sum_{j\in\mathcal N(i)}A_j\,
e^{-\|\mathbf c_i-\mathbf c_j\|^2/2\sigma_s^2}\ e^{-\|\mathbf n_i-\mathbf n_j\|^2/2\sigma_r^2}\ \mathbf n_j\Big),
```

with $\sigma_s$ the mean distance between neighbouring centres and $\sigma_r$ = `denoise_sigma_r` = 0.35. Noise
averages out on a face; across a crease the other face's normals weigh nothing and the crease stays. Then
`denoise_vertex_iters` = 10 passes move each vertex toward the planes of its faces $F(\mathbf v)$:

```math
\mathbf v\leftarrow\mathbf v+\frac{1}{|F(\mathbf v)|}\sum_{f\in F(\mathbf v)}\mathbf n_f\,\big(\mathbf n_f\cdot(\mathbf c_f-\mathbf v)\big).
```

- It runs in float64 under torch's deterministic algorithms.
- The pairs of neighbouring faces are never held all at once: they are made for `denoise_block_faces` = 2 000 000
  source faces at a time, and every face's sum runs over its own neighbours in the same order. The result is the same
  bit for bit whatever the block (a test holds it); the key is memory only.
- Measured on bufferStop's remeshed 7 211 584 faces: normals turned 1.66° at the median (p99 64°), vertices moved
  0.083 mm at the median (p99 2.3 mm, max 8 mm). The card's peak and the time: 13.3 GB / 3.7 s in one block, 4.7 GB /
  3.4 s at 2 000 000 faces, 2.2 GB / 3.6 s at 500 000.
- On that mesh, side-by-side views with and without it showed no visible difference. It stays on for the meshes that
  may need it.

**Partition.** Each face is then assigned to the cube centre nearest its centroid.

**Face budget.** Each cube is simplified to `genrecon.simplify_faces_per_chunk` (150 000) faces with quadric error. When
no per-cube target is asked, a GLB counts its length in 32 bits and the tiler reads it into one 4 GiB buffer, so the
faces are capped by a **total budget shared in proportion**, each cube keeping the same fraction of its own faces:

```math
\alpha=\min\Big(1,\frac{F_{\text{budget}}}{\sum_cF_c}\Big),\qquad F'_c=\max(1000,\lceil\alpha F_c\rceil),\qquad F_{\text{budget}}=64\text{ M}.
```

The atlases are `genrecon.texture_size` = 2048 px per cube. They are reduced, to a power of two, only when the GLB's
estimated size — the faces kept and two atlases per cube — would pass 3.5 GB; the five meshes of §18 kept 2048.

**Per-cube bake.** In parallel processes (`bake_chunks_parallel.py`, `genrecon.bake_workers` = 6). A first pass of the
mesh step writes every cube's mesh, the workers bake them into a cache, and a second pass merges from the cache; both
passes run the global steps, on the same mesh. Each cube is:

- simplified with quadric error;
- UV-unwrapped;
- textured by sampling the attribute volume trilinearly, the volume cropped to the cube's box and 2 voxels around.

The UV unwrap runs on the CPU and is 97–98.5 % of a cube's bake (7.6, 16.4 and 19.1 s on three cubes of 147–149 k
faces). An atlas made on the card would give other UVs, and therefore other textures: it cannot be held identical.

### 18.5 The real texture (`server/tools/genrecon/photo_albedo.py`)

Every texel of GenRecon's own atlas is re-coloured from the keyframes that **see** its surface point $P$. The
keyframes are decoded on the card with PIL's pixels (§19.1).

**Occlusion.** The mesh's own depth $D_k$ is rendered per keyframe with nvdiffrast, in batches of $2^{23}$ triangles, then
min-filtered over 3×3. A self-check against the cloud's z-buffer fixes the image conventions, and the bake fails beyond
25 cm.

**Candidate views.** For each part, every keyframe whose frustum holds a point of the part's box or whose camera stands
inside it. There is no distance bound: a hall's far wall is photographed from 20 m away.

**Visibility and weight.** View $k$ sees $P$ when

```math
Z>0.1,\qquad Z\le D_k+0.03+0.01Z,\qquad |\cos\theta|>0.2,
```

and its weight is $w=\cos^4\theta/\max(Z,0.3)^2$. The best 3 views (`genrecon.bake_views`) are blended:
$c=\sum wc_k/\sum w$, each $c_k$ sampled bilinearly in its keyframe.

**Texels no keyframe sees** keep the generated colour, tone-matched to the photographed texels of their part (an
affine map of mean and deviation per channel, when the part has more than 1000 photographed texels).

**What stays.** Only the base colour changes. The atlas keeps the size it was generated with, and the
metallic-roughness texture, the factors and the UVs stay as generated. Three texels of padding past the triangle edges
keep the old colour from bleeding at the seams.

The share of the covered texels that were photographed is recorded in the mesh's meta: 63–87 % on the five sessions
(§18).

### 18.6 3D Tiles (`tools/glb/tile_mesh.mjs`)

A tileset is to the mesh what the octree is to the cloud: the viewer loads only the tiles on screen, each at the
detail the screen needs.

**Tree.** The mesh's own primitives — one per baked cube, each with its atlas — are the leaves, at full detail. A k-d
tree is built over them: a split at the median centroid along the longest axis.

**Levels.** A tile of height $h$ holds every primitive below it, each simplified by meshoptimizer from its $F$
triangles to $\lfloor F/2^h\rfloor$ with its border locked, so the cut between two cubes never opens. Its textures
are at the source side $\gg\lceil h/2\rceil$. Its geometric error is measured:

```math
\varepsilon=\max\Big(\varepsilon_{\text{simplifier}},\ \max_p\sqrt{\frac{A_p}{A^{uv}_p\,w_p\,h_p}},\ \max_{\text{children}}\varepsilon\Big),
\qquad \varepsilon_{\text{leaf}}=0,
```

the simplifier's own absolute error, the world size of one texel of each primitive at that texture side, and never
less than its children's.

**Never an empty primitive.** The simplifier can leave a small closed piece with no triangle (a speck of 8 triangles
in 1.4 cm, on pccr 2026-08-24). Written as an empty primitive it stopped the encoder. Such a primitive keeps the
geometry, and the error, of the tile below: a tile never loses a primitive. A primitive of the source with no triangle
has nothing to tile and is left out. Both are declared in the log and in `tiles_report.json`.

**Encoding.**

- KTX2 UASTC textures with mipmaps (zstd), so they stay compressed on the card;
- positions quantised to 16 bits, normals to 10, UVs to 14;
- meshopt compression.

The tileset is built in a folder beside the final one and renamed at the end, so a viewer never reads half of it, and
the old tileset leaves before the new mesh is published. The viewer refines to 2 px of screen error.

### 18.7 Resume (`server/tools/genrecon/resume.py`)

**The rule: a run continues from the point of failure, never from a stale product.** The reconstruction's products
carry a stamp of what they are made from:

- **inputs** — the session files the adapter reads: the published cloud, the poses and their frame list, the camera,
  and when they exist the floor transform and the segmentation (the octree's metadata when the order filters by
  confidence);
- **code** — the adapter's files and every source file of the vendor tree, except the two of the mesh step (a fix there
  keeps the products);
- **configuration** — the order's command lines, the configuration values the adapter reads, the keyframes and the
  checkpoints by name and size.

The stamp is computed **before** the adapter runs — an input that changed during the run must not be stamped as what
made the products — and sealed after the reconstruction succeeds. Then an order starts at one of three points:

| The stamp | The run continues |
|---|---|
| the products carry it and are whole | at the **mesh step** (its own folder emptied) |
| the products do not, but the sampled latents carry the generation's stamp — the same without the decode's own code and cap | at the **decode** |
| neither | at the adapter, every difference named |

So a failed mesh step does not repeat the generation, and a failed decode — or a fix of its code, or another cap —
does not repeat the sampling. What follows the mesh step (the publication, the photo bake, the tiles) has no resume
point of its own: a failure there continues at the mesh step. The trial modes that take extra points from the mask
store never resume. `GENRECON_DRY_RESUME=1` says where an order would continue and runs nothing.

The worker names the step that failed and sends the run's last 120 lines to the stage log. The runner echoes its
commands, among them the line that carries the model hub's token. Those lines are kept out of the stage log except
in that tail, so the log of a failed run is to be treated as sensitive.

### 18.8 The shape from the cloud (selectable, off)

`genrecon.shape_from_cloud: false`. The other family tried on 2026-11-12 gives GenRecon the cloud's **shape** and not
only its presence (`tools/genrecon/cloud_shape.py`).

- The cloud is written in GenRecon's own cell representation, 512 cells per cube. Every cell corner takes a signed
  distance to the planes of the points around it, each plane turned toward the camera that saw its point — the side
  where the air is — so along any line the value changes sign once per surface: **one sheet**. An edge is crossed where
  the sign changes, and each voxel's vertex is placed by dual contouring on the planes of its crossings.
- GenRecon's own shape encoder turns the cells into the shape latent, each cube encoded with 64 cells of the cloud
  around it so that overlapping cubes agree.
- `shape_refine_t` > 0 lets the shape flow continue from that latent, noised to $t$:
  $x_t=(1-t)\,x_0+t\,\varepsilon$, then guided by the photos.
- The loose pieces go before the bake (`tools/genrecon/drop_loose.py`): kept are the largest piece, every piece that
  reaches the floor, and — to convergence — every piece whose gap to the kept surface is no more than its own size.
  No length decides.
- The bake keeps the single sheets: no global remesh, and the edge filter on the decoded mesh.

Measured on bufferStop before the bake: the floor is one sheet in 85 % of its 1 cm columns (78 % after GenRecon's
encoder and decoder; 6 % for the closed shell published before), and the holes seen from the keyframes are 3.1 %
(10.6 % for the generated shape of that day). Judged by eye it was not smooth and did not have the quality of the
generated surface, so the generated shape is the product. The code stays, selectable.

---

## 19. Determinism, devices and reproducibility

- **One GPU job at a time.** Every GPU stage checks that it has the whole card, and stops the VLM service first.
- **Deterministic torch** in the reconstruction and in the segmentation chain, AMB3R-SLAM and SAM 3.1 included
  (`reconstruction/amb3r_launch.py`, `sam3_wrapper.load_model`). That means:
  - `use_deterministic_algorithms(True)` and cuDNN deterministic, with no benchmarking and TF32 off;
  - seeded RNGs and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - z-buffers and medians computed by stable sorts and min/max scatters, with no floating-point atomics;
  - the coverage map and the cut counted with integers (cumulative sums over runs), never with float atomics;
  - the bundle adjustment's normal equations accumulated by sorted segment sums.
- **Torch twins.** Each GPU port keeps its CPU function as the path of record — numpy, PIL, OpenCV, zlib or the
  vendor's own lines. Tests hold the twin equal to it: exactly for decisions and integers, and to $10^{-9}$ for geometry.
- **Typed devices.** A port's device is a key of `config.yaml`: `surface_fit.evidence_device` (`cuda | cpu`) for the
  segmentation chain, `genrecon.adapter_device` and `genrecon.undistort_device` (`numpy | cpu | cuda`). `cuda` with no
  visible card fails naming the key: nothing runs somewhere else. The tests run on `cpu`.
- **Stamps and resume.** Every product is stamped with the sha256 of its inputs, its configuration section and its code.
  A resumed job re-runs exactly the steps whose stamp changed. Each job's configuration is frozen and verified by every
  worker.
- **Epochs, ledger and replay.** Every correction is an epoch, published atomically and replayable from the ledger.
- **Statistics.** Every bootstrap and permutation test is seeded. The bars are the measured noise of the sample, never a
  constant someone chose.
- **The octree** is byte-identical for any thread count: PotreeConverter writes every chunk in a canonical order
  before the hierarchy (§15). The cleaner by pieces is byte-identical to the in-memory cleaner.
- **Threads.** The container shows the host's 252 cores while its cgroup grants about 26 CPUs: a torch or BLAS job
  left to its own thread count is throttled to a crawl. Every CPU job sets its own (GenRecon's runner and AMB3R-SLAM
  8, the precision steps a fixed count).
- **Observed on the real sessions.** Re-running zaragoza and pccr 2026-08-31 on 2026-11-09 from a VGGT-Ω pass redone
  under changed code gave the same seams, closures and graph moves to the last printed digit, and on zaragoza the same F5 held-out
  (0.398 434 657 933 px).

### 19.1 The device ports, from the VLM to GenRecon

| Step | Path of record | On the device | How equal | Measured |
|---|---|---|---|---|
| JPEG decode: the VLM's frames, the rounds' frames, SAM 3's loader, GenRecon's frames and texture stack (`jpeg_gpu.py`) | PIL | nvJPEG gives the file's native planes; libjpeg-turbo's chroma upsampling and fixed-point YCbCr → RGB are written in integers on the device | PIL's pixels, bit for bit: 20 279 frames of five sessions, 25 651 937 280 values, 0 differ | 0.215 ms a frame against 2.485 ms |
| SAM 3's frame loader (`sam3_frames_gpu.py`) | the vendor's loader | decode, Pillow's fixed-point bilinear resize, a (3, 256) table for the normalisation | 0 bits differ | 30–36 → 4.1–4.5 ms a frame |
| The rounds' grey painting (`autoprompt/rounds.py`) | PIL + `cv2.resize` nearest | decode, OpenCV's nearest samples as an index table, one `torch.where` | 547 + 2765 frames: 0 images and 0 JPEG payloads differ | 13.8 / 14.7 → 4.75 / 5.35 ms a frame |
| SAM 3's identity IoUs (`sam3_pairs_torch.py`) | numpy, pair by pair | float32 products of 0/1 rows, integer sums below $2^{24}$ | value for value, in order | — |
| Mask store, read (`mask_store_gpu.py`) | zlib, member by member | the file uploaded once, a request's members inflated together by nvCOMP | the same bytes | — |
| Mask store, write (`store_deflate_gpu.py`, `crc32_gpu.py`) | zlib | nvCOMP Deflate, CRC-32 on the card, every stream read back | every member inflates to the record's bytes; **other streams, a file 2.2× larger** | 81.1 → 19.6 s (55 079 masks); 1280.1 → 93.3 s (290 418) |
| Mask → point match and its near-ties (`mask_match_torch.py`) | masklet by masklet | frame by frame, integer keys | identical arrays | — |
| The audit's masks (`mask_filter.py`) | zlib + scipy's dilation | inflated and dilated in one batch per visit | identical marks | of the audit's 368 s on pccr 2026-08-31, 186 s were the CPU dilation and about 96 s the zlib reads |
| Space dedupe, fragments, split, attach, components | numpy | the record's arithmetic on the device | the same decisions | — |
| Oriented boxes (`obb_gpu.py`) | numpy's seeded generator | keys, sample and components exact; hull and calipers in the device's floats; **draws on the card** | deterministic on one card, not the record's boxes (§17.3) | — |
| The epoch warp (`correction/warp_torch.py`) | numpy, keyframe by keyframe | every point at once, the same float64 arithmetic | up to the last bit of a sum | — |
| GenRecon's adapter (`layout_gpu.py`) | numpy | float64 twins | every output file byte-identical | 28.0 → 21.6 s; 121.2 → 63.5 s |
| GenRecon's frame undistortion (`remap_gpu.py`) | `cv2.imread` + `cv2.remap` | decode + OpenCV's integer remap | 0 of 916 files differ | no time gained |
| GenRecon's edge filter (`denoise.py`) | — | float64, deterministic algorithms | the same bits whatever the block | 3.4 s on 7.2 M faces |
| The chat's closest pair of two objects (`phase5_qa/tools_torch.py`) | scipy's k-d tree, one core | exact search down an octree of each set, float64 (§22.2) | the tree's distance to $10^{-12}$ and its pair | floor against wall, 1.06 M × 0.74 M points: not ended in 8 min → 0.09 s; 13.3 → 0.10 s; 5.6 → 1.2 s (first call) |
| The chat's bands, axes, profiles, flatness and volumes (`phase5_qa/tools_torch.py`) | numpy | float64 twins; numpy's median and percentile on sorted values | every tool's answer identical on pccr 2026-08-31 | a volume: 3.3–6.7 s → 0.58 s the first, 0.03 s after |

**The JPEG decoder's scope.** torchvision's own GPU decode is not used: its chroma upsampling and colour conversion are
nvJPEG's, not libjpeg's, and its pixels are not PIL's. The decoder takes sequential Huffman JPEGs of 8 bits, grey or
YCbCr 4:4:4, 4:2:2 or 4:2:0. A progressive, CMYK or truncated file fails naming the file: nothing is decoded somewhere
else in silence.

**Stays on the CPU, declared.**

- SAM 3's tracker logic: the vendor's Python control flow.
- GenRecon's UV unwrap: another atlas would give other textures (§18.4).
- Every JPEG **encode** — the VLM's views, the greyed frames, GenRecon's undistorted frames: another encoder writes
  other bytes — and vLLM's own decode of the image it receives.
- Reading files, sqlite (the instance store is worked on the local disk and copied back once), PotreeConverter, the
  PLY and JSON I/O.
- The mask store's interactive edits by the server, which holds no CUDA context.
- The chat's lookups in the instance store (sqlite) and the Python that words an answer. Its measurements over the
  points run on the card in the chat's own tools process, never in the server (§22.2).

**Declared exceptions to bit-identity.**

- **GenRecon's sampling** is not identical run to run (not investigated). The sampled latents are saved, so a decode
  can be repeated on the same latents.
- **The decode by regions** against the single pass: the last bits of half-precision decoders (§18.3).
- **The oriented boxes** are drawn on the card: deterministic on one card, other boxes than the numpy record's.
- **A mask store written on the card** holds nvCOMP's streams: the same masks, other bytes than zlib's. Written twice
  on the card it is the same bytes.
- **SAM 3's out-of-memory stops** depend on the card's memory: the same card with the GPU exclusive stops at the same
  frame, and every continuation is declared in the census.

---

## 20. Running it

Hardware: one NVIDIA A100 80 GB (developed on an RTX A6000 48 GB), 117 GB RAM and a CPU quota of about 26 cores.
Everything runs locally.

```bash
git clone --recurse-submodules <this repo> stac-build && cd stac-build
bash scripts/setup_vendors.sh                 # pinned vendor clones (vendor/VENDORS.lock.md)
bash scripts/setup_pod_envs.sh                # conda envs: da3, mapanything, semantic, genrecon, nodejs, …
bash server/tools/genrecon/setup_env.sh       # env genrecon + CUDA extensions + the STAC patch
conda run -n da3 pip install nvidia-nvcomp-cu12==6.0.0.12   # the mask store on the card (§17.2)

bash scripts/start.sh                         # backend → https://localhost:8765 (the chat's VLM loads on demand)
cd ui && npm run dev:web                      # the UI (Vite) → http://localhost:5173
```

**Workflow.**

1. Create a session, upload a video, and press **Reconstruir**: the reconstruction, the cloud, then VLM + SAM 3 and the
   floor. **Instances → Cubes** shows how the scene was cut.
2. Inspect the cloud. Set the **confidence** slider to the value that leaves the geometry clean.
3. Order **GenRecon** with its button in the Instances panel, with the confidence threshold for this session (the
   slider's value is pre-filled). **Autosegment**, beside it, runs the VLM and SAM 3 again on the cloud on disk, with
   editable prompts.
4. **View → Background and grid** sets the session's background and grid; every viewer preference is saved per session
   and per user.

**Orders through the API.** The same commands the UI sends (`POST /api/pipeline/run`; scan keys `<date>/<source>` from
`GET /sessions/<id>/scans`; running and queued jobs at `GET /api/pipelines/active`):

```bash
RUN='curl -k -X POST https://localhost:8765/api/pipeline/run -H Content-Type:application/json -d'
# reconstruct (replace false = resume on the stamps), segmenting the scans listed in "segment"
$RUN '{"session_id": "<id>", "scans": ["<date>/<source>"], "replace": false, "segment": ["<date>/<source>"]}'
# the segmentation chain on the cloud on disk; the floor follows
$RUN '{"session_id": "<id>", "scans": ["<date>/<source>"], "autosegment": {"stages": {"vlm": true, "sam3": true}}}'
# the projection alone: the masks on disk on the cloud on disk, nothing segmented again; the floor follows
$RUN '{"session_id": "<id>", "scans": ["<date>/<source>"], "autosegment": {"stages": {"projection": true}}}'
# GenRecon alone, at the viewer's confidence (0 = every point)
$RUN '{"session_id": "<id>", "scans": ["<date>/<source>"], "autosegment": {"stages": {"genrecon": true}, "genrecon_confidence": 0.0}}'
```

An order freezes `config.yaml` as it is at that moment; a GenRecon order that finds its reconstruction's products
under the same stamp continues at the mesh step (§18.7).

All configuration lives in `server/config.yaml`. The loaders are typed, and a missing key fails at load.

---

## 21. Repository map

```text
stac-build/
├─ server/
│   ├─ main.py                 # FastAPI app: REST, WebSockets, the run orders (Reconstruir, Autosegment, GenRecon)
│   ├─ pipeline_manager.py     # stages, frozen config per job, resume probes, the queue, the GPU hand-over, the
│   │                          #   segmentation rounds
│   ├─ view_range.py           # THE RANGE: reconstruction.view_range_m, the one range of every stage
│   ├─ jpeg_gpu.py             # JPEG frames decoded on the card with PIL's pixels (nvJPEG + libjpeg-turbo's integers)
│   ├─ repro.py                # stamps, the deterministic environment, the exclusive-GPU check
│   ├─ potree_converter.py     # the octree: PotreeConverter's launch and its stamp
│   ├─ workers/                # map (reconstruction), cloudcompy, vlm, sam3, floor, genrecon, certify (off)
│   ├─ intake/                 # I0 quality, focal probe, I1 parallax, I3 walk, turn frames, VRAM sizing
│   ├─ reconstruction/         # amb3r_probe (the whole-video pass and its metre), coverage_fill (the keyframes completed
│   │                          #   on the map), cubes + chunk_covis (the cut), cube_boxes (the viewer's cubes), chunk_plan
│   │                          #   (Ω resolution), scale_align, orient, sky precompute, stream_clean + gpu_cloud_clean
│   │                          #   (the cleaner), surface_fit (the evidence device)
│   ├─ precision/              # F0 camera, F2 gauge, F4 tracks, F5 refine + ba_torch, F6 depth_on_f5 + mono_detail +
│   │                          #   f6_torch, corrected cloud, chunk check, epochs
│   ├─ correction/             # epochs (apply/select, ledger, replay), restart (the way back to epoch 0), floor,
│   │                          #   warp_torch (the epoch warp on the device)
│   ├─ segmentation/           # autoprompt/ (the VLM's vocabulary, the merge, rounds), sam3_wrapper + sam3_group +
│   │                          #   sam3_fast + sam3_frames_gpu (SAM 3.1), pipeline (the store, mask → cloud, instances,
│   │                          #   boxes), census, mask_filter (the audit), mask_store_gpu + store_deflate_gpu +
│   │                          #   crc32_gpu (the store on the card), mask_match_torch, sam3_pairs_torch,
│   │                          #   fragments_torch, obb_gpu and the other device twins
│   ├─ semantic/               # vLLM launcher (deterministic serve flags), the engine lease and job_engine, client
│   ├─ phase_r/                # the instance store (scene_r.db)
│   ├─ phase5_qa/              # the spatial chat: orchestrator (one call of the model per turn), answers.py (the
│   │                          #   answer written by the tool layer, es / en / fr), tools.py (the record),
│   │                          #   tools_torch.py (the device), tools_worker.py (the tools process), provenance.py
│   ├─ commands/               # THE DATABASE OF COMMANDS: catalog/*.yaml (the declarations), catalog.py (validated),
│   │                          #   db.py (commands.db: sqlite + FTS5), router.py (what a sentence is offered),
│   │                          #   system_tools.py (conversation, help), export_ui.py (the interface's typed ids)
│   ├─ voice/                  # STAC's voice: serve.py (Whisper + Kokoro, its own process), spoken.py (a written answer
│   │                          #   as it is said), wake.py ("STAC, …"), service.py (its life), api.py (/api/voice)
│   ├─ calibration3d.py  calibration_api.py   # calibration measurements taken on the 3D model, per scan
│   ├─ tools/genrecon/         # adapter + layout_gpu, presence (the vote + layout), genrecon_levels, run_cloud.sh,
│   │                          #   resume + latents, decode_regions, remesh_resolution, denoise (the edge filter),
│   │                          #   remap_gpu, bake_chunks_parallel, photo_albedo, publish, cloud_shape + drop_loose
│   │                          #   (the shape from the cloud, off), measure_mesh, genrecon_stac.patch
│   ├─ tools/amb3r/            # the AMB3R-SLAM driver and amb3r_stac.patch
│   └─ config.yaml
├─ vendor/
│   ├─ VGGT-Long/              # the fork (branch gpu-main): cube_run (Ω per cube, metric lock, seams, revisit bridges
│   │                          #   and closures, ownership), context/witness views, the Sim(3) graph, the decision rule
│   ├─ AMB3R-SLAM/             # upstream 5418465 + tools/amb3r/amb3r_stac.patch: the whole-video pass and its light map
│   ├─ GenRecon/               # upstream eaf1468 + tools/genrecon/genrecon_stac.patch; work/ holds each run's products
│   ├─ depth-anything-3/  pointdit/  sam31/  PotreeConverter/  …
├─ ui/                         # React + TypeScript + Vite; the viewer (viewport/: HUD, palette, sceneLook), panels,
│                              #   GenRecon, Autosegment and Background-and-grid windows; design tokens and i18n
├─ tools/glb/                  # tile_mesh.mjs (3D Tiles)
└─ docs/                       # design notes, A/B results, determinism plan, comparisons
```

**Beyond the reconstruction.** The same backend serves:

- the viewer's measurement tools;
- the spatial assistant, which answers with deterministic tools and replays its measurements as geometry;
- IFC/BIM registration and deviation maps;
- the WebXR view on the phone;
- team workspaces.

See `docs/` for their design.

---

## 22. The session in the viewer: meshes, the spatial chat, calibration, STAC's commands and voice

### 22.1 Opening a session

The cloud opens first. Every mesh of the scan — the GenRecon tiles and the per-instance GLBs — is **listed unchecked
and not downloaded**; checking one loads it (`Viewport.setTsdfVisibility`). A mesh that appears while the session is
open, a GenRecon stage ending, loads at once: it is what the user is waiting for. Before 2026-11-13 the scene meshes
loaded with the session.

### 22.2 The spatial chat (`server/phase5_qa/`)

The assistant — **STAC** — is Qwen3-VL choosing among commands. The model proposes, describes and orchestrates; **it
never measures**: every figure of an answer comes from a deterministic tool over the scan's instance store.

**One call of the model per question.** The model's part of a turn is to pick the command(s) and their arguments
(`tool_choice: required`); the tools run, and the sentence the user reads is **written by the tool layer** from their
results (`answers.py`): the same sentence for the same result, the tool's own figures, in Spanish, English or French,
and no second call of the engine. Only a tool with no wording sends the turn back to the model, with the provenance
check below. The conversation itself (a greeting, thanks) is the one thing the model words: its reply is an argument
of the `small_talk` command, and is refused when it carries a figure with a unit.

**The chat's engine.** The chat starts its own vLLM with the profile `semantic.chat.engine: single_sequence`: one
sequence at a time (no answer depends on a batch), no batch-invariant kernels, decode graphs without compilation, no
prefix cache, a fixed seed. Measured on the A100: the same tokens and log-probabilities bit for bit across request
orders and across launches, 0.8 s a call; a question end to end 0.95–1.16 s. A pipeline stage always launches its own
engine with the batch-invariant profile, and the chat-only keys do not enter that engine's identity. The engine starts
with the backend (`semantic.chat_on_demand: false`), is warmed there (its first call), and leaves the card while a
job runs.

**The language of an answer** is the sentence's own when its words tell (`answers.language_of`), else the
conversation's so far, else the speech recogniser's, else the interface's. A description the VLM wrote for a scan is
kept in Spanish and translated once per language by the model (figures, names and ids unchanged), then kept in the
store.

**The scan it answers about.** The request carries the scan the viewer shows (`scan`); without it the project's
active scan is used (`project_paths.resolve_session`). The store is `output/scene_r.db` of that scan.

**What the model reads.**

- *The session*: project, scan, number of objects, the cached scene description.
- *The whole inventory.* Up to `semantic.chat.list_limit` objects, one line each (label, class, vertical extent). Beyond
  it, every **label** with its count and its largest object: `sign (3, #795)`. An id or a label written in the
  question is resolved by the tool layer, a label standing for that label's largest object ("the floor", "the largest
  wall"); `list_objects(label)` returns a label's ids. A listing never exceeds the context (pccr 2026-08-24's 8,777
  objects were 486 kB of JSON before).
- *The objects the question names.* `label#id`, `#id` and `id N` — as the Instances list writes them — are looked up
  in the store **before** the model reads the question, and the result is appended to it: the id's own label, whether
  it exists, and the ids of that label when it does not. Positions are not written there (read in a prompt they were
  subtracted into distances no tool had measured); `get_object(id)` returns them.
- *The conversation*: its last `semantic.chat.history_turns` turns.

**The provenance check** (`provenance.py`). Let an answer state a figure $v$ written with $d$ decimals and a unit, and
let $T$ be the numbers the tools returned in this turn, the numbers of the tool layer's own text and the numbers the
user wrote. The figure is grounded when

$$\exists\, t \in T,\ k \in K(\text{unit}) :\quad \lvert t\,k - v \rvert \le \tfrac12\,10^{-d},$$

with $K$ the changes of unit of the figure's family (m: 1, $10^{-3}$, $10^{-2}$; cm: 100, 1, 0.1; mm: 1000, 1, 10;
areas, volumes and degrees: 1). Every number of a bracketed vector that holds a decimal is checked the same way, and
every tool the answer cites must have been called in this turn. An answer that fails is sent back **once** with the
failing figures and a tool call required (`tool_choice: required`); what still fails is returned with the answer
(`provenance.ok = false`, the figures listed) and shown under it. Found on pccr 2026-08-31: asked for a distance, the
model answered "1.14 m, measured with get_distance" having called no tool.

**Where the measurements run** (`semantic.chat.tools_device`: `numpy` | `cpu` | `cuda`). The tools that read an
object's points — clearance, parts and bands, extent, plumb, level, height profile, flatness, the volumes — are
`POINT_TOOLS`. With `cuda` they run in the chat's **own process** (`tools_worker.py`), not in the server: a CUDA
context in the backend is a compute process on the card for its whole life, and every GPU step after it refuses a
card that is not free (`repro.require_exclusive_gpu`). The process starts when the chat opens on a scan, keeps the
store and the points it measured on the device between questions, and **ends** — joined, the card free — when a
pipeline job or a server GPU task takes the engine lease, when the question is about another scan, when the scan's
segmentation is written again, and with the backend. While the lease is held it is not started: the chat answers
"busy". A `cuda` with no card fails naming the key.

The points and the boxes are promoted from the store's float32 to **float64** before any arithmetic, on both paths
(before 2026-11-13 a display-frame store was measured in float32: a figure can differ in its last printed digit).

**The closest pair of two objects, exact** (`tools_torch.closest_pair`). Each set is sorted along the Morton curve of
a grid of $2^{10}$ cells per side over its own box, so the cells of every coarser level are runs of the sorted
points; a cell keeps the tight box $[\mathbf{lo}, \mathbf{hi}]$ of its own points. For cell $i$ of one set and $j$ of
the other,

$$\ell_{ij}^2 = \sum_{a \in \{x,y,z\}} \max\bigl(0,\ lo_{i,a} - hi_{j,a},\ lo_{j,a} - hi_{i,a}\bigr)^2 \;\le\; \lVert \mathbf p - \mathbf q \rVert^2 \quad \forall\, \mathbf p \in i,\ \mathbf q \in j,$$

also in floating point, every step being monotone. An upper bound $u^2$ is the squared distance of a **real** pair
(strided samples of the pairs of cells with the smallest $\ell$). A pair of cells is kept while $\ell_{ij}^2 \le u^2$;
the kept pairs are split into their children at the next level and bounded again, until the point pairs left fit the
work bound, and those are measured outright. The pair that realises the minimum is never pruned — its bound cannot
exceed its own distance — so the distance is the k-d tree's. Squared distances are $dx^2 + dy^2 + dz^2$ in that
order. Among pairs at exactly the same distance the first point of the first set wins, then the first of the second;
the tree breaks a tie inside the second set its own way: the pair, never the distance, can differ there.

The other kernels are float64 twins of the numpy code, which stays the path of record: the bands and extremes; the
principal axes from the QR of the centred points and the SVD of its $3\times3$ factor; numpy's median and linear
percentile on sorted values; the stations of a height profile; the box coordinates of a volume, its occupancy voxels
and the first free placement of an item in $i, j, k$ order from a summed-area table.

Measured on pccr 2026-08-31 (18.6 M points in 1,360 instances), the numpy path against the card:

| | numpy, one core | card |
|---|---:|---:|
| Clearance, floor (1,062,834 points) – wall (743,629) | not ended in 8 min | 0.09 s (1.8 mm; the same pair by an independent exact search) |
| Clearance, floor – work surface (59,366) | 13.3 s | 0.10 s |
| Clearance, two objects of 59 k points | 5.6 s | 1.2 s (first call of the process) |
| A 2 m box: objects inside, free volume | 3.3–6.7 s | 0.58 s the first, 0.03 s after |
| Every instance's points read from the store | 17.5 s (1,360 queries) | 0.2 s (one query) |
| Building the prompt, pccr 2026-08-24 (8,777 objects) | 9.9 s | 1.4 s (sqlite, batched) |
| The tools process, start | — | 8.5–9.5 s, once per chat opening |

Every tool compared gave the identical answer on both paths. With the tools on the card a question's time is the
model's: the engine serves 21 tokens/s under the deterministic serve flags (§17.1), about 3 s per call of the loop on
a 9 k-token prompt.

**Tests.** `tests/test_chat_tools_torch.py` (the closest pair against scipy's tree in seven configurations and with
the octree forced; ties; numpy's median, percentile and SVD; the free placement against the record's loops; every
point tool on both paths over a store; the same on the card in strict deterministic mode; the tools process: it
measures, survives a tool's error, ends, is not started beside a job, restarts on a new segmentation) and
`tests/test_chat_assistant.py` (the scan asked, the inventory, the references, the listings, the provenance check and
its retry, the conversation's turns).

### 22.3 Calibration on the 3D model (`server/calibration3d.py`)

A calibration measurement is taken **on the model**, not on the keyframes: with the distance tool the user marks two
points on the cloud or on a mesh and presses the pin of the readout. The window (`ui/src/features/CalibrationDialog.tsx`)
asks for the real length, its uncertainty $\sigma$, its source — tape, laser, total station, or **another scan** of
the project — and its role:

- **gauge**: it calibrates the scale;
- **test**: it measures the error that remains after calibrating.

The server stores it in `output/calibration_3d.json` of the scan. The model's length $m = \lVert \mathbf b - \mathbf a
\rVert$ is computed there from the two stored points, never taken from the client, and carries the geometry epoch it
was picked on: a measurement of another epoch is *stale* — listed, not used, not drawn.

The scan's scale factor is the evaluation battery's estimator over the gauge measurements of the current epoch
(`evaluation.geometry.gauge_scale`):

$$s = \arg\min_s \sum_i w_i\,(s\,m_i - t_i)^2 = \frac{\sum_i w_i\, m_i\, t_i}{\sum_i w_i\, m_i^2}, \qquad w_i = \frac{1}{\sigma_i^2},$$

with $t_i$ the real length (equal weights when every $\sigma$ is 0; a $\sigma$ of 0 among positive ones takes the
smallest positive one). The window shows $s$, the residuals $s\,m_i - t_i$ of the fit and the error on the test
measurements. With **another scan** as the source, the same element is measured on both scans and the other scan's
length is the "real" one: $s$ is then the factor that brings this scan to that scan's scale.

**Planes.** A calibration plane is marked on the model too: three or more picked points (Enter closes it), drawn
filled on the cloud; the server fits the plane and stores its normal, inclination, height, rms and area.

**Global or per axis.** `mode: global` gives the one factor above. `mode: axes` gives one factor per axis from
$t_i^2 = \sum_k (s_k\, d_{ik})^2$ ($d_{ik}$: the measurement's component along axis $k$), solved by least squares for
$s_k^2$ — and is refused unless a gauge measurement runs along each of the three axes; with one or two the calibration
is global and the others verify. **Verification**: every *test* measurement is reported with its calibrated length, its
difference to the real one and whether it lies within its $\sigma$. The scan's state — *uncalibrated*, *calibrated*,
*verified* — is shown on the viewer as the cloud opens.

The stored measurements are drawn on the model in their own colour and stay through "clear measurements".

**Applying it: the real rescale** (`server/correction/rescale.py`). "Apply to the scan" in the window rewrites the
scan's geometry at the calibrated scale, so that everything measured or compared afterwards — against another scan,
a reference human, a mesh of the collection, a BIM model — is at real scale. Only a **global** factor is applied: a
different factor on each axis is not a similarity, and the cameras could not follow it (the scene would no longer
project onto its own images); the per-axis mode stays a diagnosis.

With $s$ the factor, the rescale is a new **geometry epoch**, built by the same transaction as every correction
(staged, sealed, swapped under a journal, the previous epoch kept and selectable):

$$\mathbf x' = s\,\mathbf x \quad\text{(every point of the cloud and of the raw cloud)}, \qquad
\mathbf c' = s\,\mathbf c,\ \ R' = R \quad\text{(every camera)}, \qquad
\mathbf t_f' = s\,\mathbf t_f \quad\text{(the floor transform } \mathbf d = s_f R_f\,\mathbf x + \mathbf t_f\text{)},$$

so the display frame scales about its own origin and the floor stays at height 0. The intrinsics are pixels and do
not change. The objects' boxes are scaled exactly (centre and half extents $\times s$), never refit; the instance
store, the depth sidecar and the octree follow inside the transaction. Nothing is deleted and nothing re-fitted.

What an epoch does not carry is rewritten with it: the evaluation volumes' centres (their sizes are the user's own
metres), the placed objects' positions (a reference human keeps its real size), the meshes — scaled by a transform
at their root (the 3D tiles' `root.transform`, a wrapper node of the GLB), so they are the same mesh at the new
scale — the calibration document (its measurements re-expressed on the new epoch: the calibration that was applied
then reads a factor of 1), and the project's composition (the rescaled scan's transform to the reference composed
with $1/s$; when the reference itself is rescaled, every other scan's with $s$). **Undo** is the inverse scale.

Measured on a copy of bufferStop (3,837,478 points, 174 keyframes): 29 s, $\max \lVert \mathbf x' - s\,\mathbf x
\rVert = 0.24\,\mu\text{m}$ (the cloud's float32), the octree's bounding box $\times s$ on its six bounds.

To bring one scan to another's scale, the same elements are measured on both, stored on the scan to change with
*another scan* as their source, verified, and applied.

Routes: `GET /api/calibration/{session}?scan_key=`, `POST …/measurements`, `PATCH …/measurements/{id}`,
`DELETE …/measurements/{id}`, `POST …/planes`, `PATCH …/planes/{id}`, `DELETE …/planes/{id}`, `PUT …/mode`,
`POST …/apply` (the real rescale, in its own process; `GET …/apply` for its state).
Tests: `tests/test_calibration3d.py`, `tests/test_rescale.py`.

### 22.4 STAC's commands (`server/commands/`)

Everything STAC can be asked to do is a **command**, declared once and kept in a database. A person speaks or writes;
the model picks a command; what it picked says what the sentence was — an **action** (STAC operates the system), a
**query** (STAC measures or reads) or **talk** (conversation).

**The declarations** are `server/commands/catalog/*.yaml`, one file per area. A command states its `kind`
(`action | query | talk`), who executes it (`runs_on: scene | system | ui`), its `status` (`available`: STAC runs it;
`manual`: it exists in the interface and is operated by hand — STAC knows it and says so; `planned`), its arguments,
whether it asks for confirmation, and in **es / en / fr** its name, the ways people ask for it and what STAC says when
it is done (`done`, and `done_off` when what it did was to turn something off). `catalog.py` validates them: a missing
or mistyped field fails the load naming the file, the command and the field.

**The database** `commands.db` (sqlite with an FTS5 index, diacritics folded) is the declarations' deterministic image,
rebuilt whenever their sha256 changes and read from a copy in memory. A search over every command in the three
languages costs 0.1–0.4 ms.

**The executors.**

| `runs_on` | executed by | a command with no executor |
|---|---|---|
| `scene` | `phase5_qa/tools.py` (`SpatialTools`): the measurements and the viewer actions over a segmented scan | fails a test |
| `system` | `commands/system_tools.py`: the conversation, the help | fails at the first turn, naming it |
| `ui` | `ui/src/App.tsx` (and the Assistant panel for the user's name), through `ui/src/commands/registry.ts` | **does not compile** |

`python -m commands.export_ui` writes `ui/src/commands/catalog.generated.ts` — the ids of the interface's commands and
the type of each one's arguments; the app's executors are typed `Executors<UiCommandId>` against it. A test holds the
generated file equal to the catalogue.

**A turn.** `commands/router.py` chooses what a sentence is offered: every tool of the scene while a scan is open, the
commands always offered (conversation, help), opening a session when none is open, and the interface's and system's
commands whose names or ways of asking share words with the sentence — in the three languages at once, a word of four
letters or more also matching the words that share its start (`conectá` finds `conectar`), at most
`semantic.chat.command_candidates` of them. The model then makes its one call. The scene's and the system's commands
run on the server; the interface's come back as `ui_actions`, each with the texts to word its outcome in the
conversation's language, and the interface runs them in order. With no scan open STAC still converses and operates
the application; it is told it cannot measure.

A tool costs about 100 tokens of prompt, so the list offered must stay short however many commands exist. On the
chat's engine a new set of tools costs 0.03–0.2 s the first time. When the catalogue outgrows word search, the next
step is a first call of the model on the list of *areas* alone (0.3 s measured on a short prompt), its commands then
offered — declared in `router.py`, not built.

**To add a command:** declare it in its area's file; for `runs_on: ui` run `python -m commands.export_ui` and write
its executor where the compiler asks; for `scene` / `system` write its handler and, when the tool layer should word
its result, its composer in `answers.py`. Nothing else lists commands.

The database holds **270 commands in 31 areas**. STAC runs 132 of them: 44 of the scene, 2 of the system and 86 of
the interface (the connection, sessions and scans, panels, full screen, language, the viewer's tools, how the cloud
is drawn, meshes, cubes, volumes, placed objects, the brush's settings, physics, calibration, its own voice). The
other 138 commands of the interface — every remaining menu entry, button, toggle and row action — are `manual`:
declared in the three languages and found by how people ask for them; STAC answers that they are done by hand and
under which area. They are what launches GPU work or deletes data (`confirm: true`: a spoken confirmation is not
built yet) and what lives inside a dialog's own state.

Tests: `tests/test_commands_db.py`, `tests/test_commands_turn.py`.

### 22.5 STAC's voice (`server/voice/`)

STAC is spoken to and answers aloud, in Spanish, English and French — the interface's own three languages
(`ui/src/i18n/{es,en,fr}.json`, the same keys in each) — with engines that run locally on the card in their own
process (the `voice` conda env, `scripts/build_voice_env.sh`; the backend never loads a model).

- **Speech → text**: Whisper large-v3-turbo. A voice-activity detector (Silero) is asked first whether there is
  speech at all. The language is decided **among the system's three** from the first decoder step; decoding is
  greedy, so the same audio gives the same text. The recogniser is given a vocabulary: the system's name, the labels
  of the scan on screen, the sessions' and scans' names.
- **Text → speech**: Kokoro-82M, one voice per language (`voice.tts.voices`). Before it is said, a written answer is
  turned into what a person would say (`spoken.py`): `0.0018 m (1.8 mm)` → "0,0018 metros, 1,8 milímetros",
  `sign#797` → "sign 797". Only the form changes: every figure said is the figure written.

Measured on the A100: 7–8 s of speech written in 0.28–0.37 s with the language right; 6.7 s of audio synthesised in
0.12 s; both engines load in 25–35 s and take 2.9 GB of the card. A spoken turn after the user stops speaking is the
transcription plus the model's one call (0.4 s with no scan open, 1.0–1.5 s with one) plus the synthesis.

**In the Assistant panel**: a microphone (press, speak, press again); spoken answers on / off; and **hands-free** (the
ear button): the microphone stays open, the browser cuts each utterance out (its level against the room's own noise)
and the server writes it. Only what **names STAC** is taken — "STAC, open pccr", "STAC Build, …", after a greeting or
at the end of the sentence, as a recogniser may spell it (`voice/wake.py`, deterministic); the rest is dropped by the
interface and nothing of it reaches the model. Saying the name alone makes STAC listen for the next sentence.

**The user's name.** STAC asks it first and speaks to the user by it: it confirms by name a turn in which it only
acted ("Listo, Ana."), starts a conversation's first answer with it, and is told it for its own replies. A name the
user did not say is never stored (the tool layer checks that the name is in the sentence).

**Life.** The voice service starts with the backend (`voice.start_with_backend`), ends — verified — when a job or a GPU
task of the server takes the card (it holds a CUDA context), and starts again when the card frees.

Routes: `GET /api/voice/status`, `POST /api/voice/transcribe` (the recorded audio → text, language, `addressed`,
`command`), `POST /api/voice/speak` ({text, lang} → WAV), `POST /api/voice/stop`. Tests: `tests/test_voice.py`.
