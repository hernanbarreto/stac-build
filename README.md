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
   reconstructed saw. That is a **coverage map** of the scanned scene, in which every point carries the frame it came from.
2. The intake's parallax keyframes are a first proposal. **The map completes them**: where the video saw something that no
   keyframe sees, or that only one sees, the frame that sees it becomes a keyframe, until no frame brings more than the
   measurement's own noise.
3. The views are **grouped into cubes by what they see**: every pass within 5 m of camera displacement, half for the
   cube's own views and half for the views it shares with its neighbours, at most 64.
4. A feed-forward multi-view network (VGGT-Ω) reconstructs each cube. A multi-view monocular-metric depth model (Depth
   Anything 3) makes each cube metric, and **one Sim(3) graph** joins the cubes on the frames they share.

A **precision core** then measures and corrects, in order:

- the camera;
- a continuous metric gauge along the walk;
- learned native-pixel tracks;
- a bundle adjustment judged on held-out tracks;
- a per-keyframe "bend" of the network's depth onto the adjusted landmarks, with an edge-keeping multi-view vote.

The published cloud is built from that voted depth. A vision–language model then names everything in the keyframes,
SAM 3.1 segments and tracks it, and the masks are projected onto the cloud as object instances. On demand, **GenRecon** (a
Trellis-class sparse-voxel generative model) builds a textured scene mesh. Presence comes only from the cloud, decided by
the keyframes' vote. The model only shapes the surface, and the keyframes are baked onto it as its real texture.

Every decision that moves geometry follows **one statistical rule**. A change is applied only when a paired bootstrap on
held-out evidence says it improves, by more than its own measured error, with enough independent judges. Every stage runs
on the GPU in deterministic mode. Every product is stamped with the inputs, configuration and code that made it.

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

---

## 1. What runs when a reconstruction is launched

**Reconstruir** in the UI sends one order per scan to the pipeline manager (`server/pipeline_manager.py`). The manager
runs the jobs one at a time on the card. Each stage is a separate worker process in its own conda environment, and every
stage is resumable against stamped products. The default chain is:

| # | Stage | Worker | What it produces |
|---|---|---|---|
| 1 | **Reconstruction** | `workers/map_worker.py` | intake (I0, I1) → AMB3R-SLAM over the whole video → the coverage fill of the keyframes → turn frames → I3 → the cubes → VGGT-Ω per cube and the cube Sim(3) graph → metric scale and upright frame → precision core F0–F6 → the published cloud (`output/cleaned_cloud.ply`) and its octree |
| 2 | **Cloud → viewer** | `workers/cloudcompy_worker.py` | the cleaned cloud handed to the viewer (Potree octree) |
| 3 | **Scene analysis** | `workers/vlm_worker.py` | Qwen3-VL reads every keyframe; the SAM 3 vocabulary (`output/vlm_analysis.json`) |
| 4 | **Segmentation** | `workers/sam3_worker.py` | SAM 3.1 masks on every keyframe, projected onto the cloud as instances with oriented boxes (`output/segmentation_result.json`, `instance_ids.npy`, `scene_r.db`) |
| 5 | **Floor** | `workers/floor_worker.py` | the floor levelled on the segmented support surfaces, one rigid motion per session (`pipeline.auto_floor`) |

Two more stages exist but are switched off in the chain:

- **GenRecon** (`workers/genrecon_worker.py`) runs from its own **GenRecon** button in the segmentation panel. The user
  chooses the cloud's confidence threshold for that session (`pipeline.auto_genrecon: false`).
- **Certification** stays in the code, switched off (`certify.auto_after_segmentation: false`).

The **Autosegment** button re-runs the VLM and SAM 3 stages on the cloud already on disk, with editable prompts; the
projection and the floor follow.

**Replace and the semantic products.** **Reconstruir** with *Replace* wipes the scan down to its frames, its video and its
capture inputs, except the VLM analysis and the SAM 3 store. Those are kept and are redone only when the keyframes they
were made on are no longer the session's: the frames the VLM's stamp read, and the `keyframes_sha256` SAM 3's store was
sealed with (`pipeline_manager._semantics_keyframes_change`). The coverage fill can add keyframes, and then they are
redone.

---

## 2. Architecture

**Backend.** `server/main.py` is a FastAPI + uvicorn service on port 8765 (TLS). It holds the REST API, the viewer
WebSocket, the session store and the pipeline manager.

**Pipeline manager.** One queue of jobs: one job per scan, stages in a fixed order, one GPU job at a time. It has these
properties:

- A job's configuration is **frozen** into `output/run_config.yaml`, with per-order stage overrides, and verified by every
  worker.
- Each stage declares its products. A resumed job skips a stage whose products and stamps match, and restarts it otherwise.
- GPU stages require the **whole card** (`repro.require_exclusive_gpu`). The vLLM service is stopped before them and
  restarted when the chain needs it.

**Environments.** Each worker runs in its own environment:

| Environment | Workers |
|---|---|
| `da3` | backend, DA3, SAM 3.1, precision core, scale/orientation, photo bake |
| `mapanything` | the VGGT-Ω fork and the sky segmenter |
| `amb3r` | AMB3R-SLAM, the whole-video first pass |
| `semantic` | vLLM serving Qwen3-VL |
| `genrecon` | GenRecon |
| `nodejs` | 3D Tiles builder |

**Geometry epochs.** Every correction that moves geometry publishes a **geometry epoch**: a journaled, transactional swap
of the artifacts it changed. The previous epoch is kept selectable, and a ledger keeps every epoch replayable.

**UI.** `ui/` is React, TypeScript and Vite, built on Three.js:

- a Potree-format octree viewer streams the cloud; a confidence slider filters it with the same arithmetic used by the
  GenRecon adapter (§18.1);
- mesh layers stream as 3D Tiles;
- panels show instances, oriented boxes and measurement tools, and a chat assistant answers spatial questions with
  deterministic tools (the VLM proposes; tools measure);
- the **Instances** tab shows the cubes of the cut, each one switched on and off on its own.

**Storage.** Each scan lives in `server/projects/<session>/scans/<scan>/src_default/`:

- `frames/`: the video frames;
- `intake/`: quality, parallax, AMB3R's whole-video pass, the coverage fill, the walk, the cubes, and their stamps;
- `output/`: model passes, epochs, cloud, octree, segmentation, meshes.

---

## 3. Models

| Role | Model | Where |
|---|---|---|
| Multi-view feed-forward reconstruction (poses, intrinsics, depth, confidence) | **VGGT-Ω** 1B, `vggt_omega_1b_512.pt`, bf16 | `vendor/VGGT-Long` (fork, branch `gpu-main`) |
| First pass over the whole video: every frame's pose, loops, what each frame saw | **AMB3R-SLAM**: DA3-Small front-end, VGGT-Ω submaps, DBoW loops, offline pose graph | `vendor/AMB3R-SLAM` (fork, branch `stac`) |
| Metric multi-view depth: focal probe, I3 windows, scale anchors | **Depth Anything 3**, `DA3NESTED-GIANT-LARGE-1.1` | `vendor/depth-anything-3` |
| Place recognition for the chunk layouts' loop candidates (not run on cubes: the revisits are AMB3R's map's, §10.1) | **DINOv2 ViT-B/14 + SALAD** (8448-D descriptor) | fork `LoopModels/` |
| Sky segmentation | `skyseg.onnx` (sha256-pinned), onnxruntime CPU | fork `loop_utils/sky_mask.py` |
| Native-pixel point tracks | **VGGSfM v2** tracker (sha256-pinned) | fork `dependency/vggsfm_tracker` |
| Monocular detail at depth discontinuities | **PointDiT-H/16** with DINOv3 ViT-H+/16 | `vendor/pointdit` |
| Scene understanding (vocabulary, descriptions) | **Qwen3-VL** served by vLLM | `server/semantic/` |
| Promptable segmentation and video tracking | **SAM 3.1** | `server/segmentation/` |
| Scene mesh | **GenRecon** (Trellis.2-class sparse-structure + shape/texture SLat flows, fine-tuned, conditioned on DINOv3 ViT-L/16 features of posed images) | `vendor/GenRecon` @ eaf1468 + `server/tools/genrecon/genrecon_stac.patch` |
| Rasterisation for the photo bake | nvdiffrast | env `da3` |

---

## 4. Notation and the decision rule

**Notation.**

- Keyframes are indexed $k$; each has a camera-to-world pose $T_k=[R_k\,|\,C_k]$ (OpenCV axes: $x$ right, $y$ down, $z$
  forward) and intrinsics $K$.
- A pixel is $(u,v)$; depth $z$ is measured along the optical axis.
- A ray with unit $z$ component is $\mathbf r(u,v)=K^{-1}[u,v,1]^\top$, so the point at depth $z$ is $X=C_k+z\,R_k\,\mathbf r$.
- $\operatorname{med}$ is the median, $Q_p$ the $p$-quantile, $P_p$ the $p$-th percentile.
- The robust scale is $\mathrm{MAD}$-based: $\hat\sigma=1.4826\,\operatorname{med}|x-\operatorname{med}x|$.

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
The session $K$ is the per-parameter median, and the spread $\hat\sigma/\operatorname{med}$ is reported.

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

**Quanta.** The parallax quantum is $q=12$ px. The rotation quantum is

```math
\theta_q=(1-\tau)\,\tfrac14\,\mathrm{FOV}_h,\qquad \mathrm{FOV}_h=2\arctan\frac{W}{2f_x},\qquad \tau=0.3 .
```

**Keyframe rule.** Frames are scanned from the anchor.

- When $p>1.25q$, the keyframe is the sharpest frame whose parallax lies in the band $[0.75q,1.25q]$.
- Otherwise, when $r>1.25\theta_q$, the same rule applies to the rotation band.
- When tracking is lost after some parallax ($p\ge1.5$ px), the keyframe is the sharpest frame with $p\ge0.75\max p$.

The keyframe becomes the next anchor.

**Witness frames.** A parallel chain marks a witness at every keyframe, at every lost frame, and every 1.5 px of parallax.

---

## 6. The first pass: AMB3R-SLAM over the whole video

`server/reconstruction/amb3r_probe.py` runs **AMB3R-SLAM** (`vendor/AMB3R-SLAM`, branch `stac`) right after I1, over
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

On pccr the pass took 11 min over 3044 frames and reconstructed 761 of them.

---

## 7. The coverage map and the keyframe fill

The intake's keyframes are a first proposal of coverage. **The map decides**: `server/reconstruction/coverage_fill.py`
completes them where the video saw something that no keyframe sees, or that only one keyframe sees.

**The map.** Every valid pixel of every record is a point, $X=c2w_r\big(z\,K_s^{-1}[u,v,1]^\top\big)$. Here $K_s$ is the
record's $K$ with its first two rows divided by the stride, because grid pixel $(j,i)$ is model pixel $(4j,4i)$.

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

**Covered.** A point is **seen** when some keyframe sees it. It is **covered** when a second keyframe sees it from a ray at
least `refine.min_tri_deg` (2°) away from the first one's, $\angle(C_g-X,\,\mathbf d_1)\ge2°$, which is F5's bound for a
landmark. A point not seen, or seen without a second view, is **weak**: one view, or two from the same place, measure
nothing.

**The fill.** A candidate is any frame AMB3R reconstructed that I0 found usable and that is not a keyframe's own record.
Its gain is the share of what it saw that it would see first, or cover as the second view from far enough:

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

On pccr: I1's 295 keyframes left 0.7 % of the map unseen and 6.4 % seen once. 18 frames were added; 0.4 % unseen and 5.8 %
seen once remain, and no frame covers more of them than the noise.

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

Windows overlap by half, and the last one is shifted to end on the final keyframe.

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

A cube is a **group of views that VGGT-Ω reconstructs together**: its own views and the views it shares with its
neighbours. Views go together by **what they see**. A cube's pass never spans more than **5 m** of camera displacement
(`cube_m`): beyond about 5 m one inference of Omega accumulates drift and bends what it infers, and a cube longer than the
others would bend differently from them, so every pass keeps the same reach. Half of it is for the cube's own views and
half for the shared ones: "64 images overlapped 50 % with the neighbours", in space as in views. Two passes over the same
place that see the same go together, whatever moment of the walk they come from. Two cameras side by side that see
different things do not.

**Units.** A unit is a parallax keyframe with the rotation keyframes up to the next parallax keyframe, and the turn frames
after them. A turn is never cut.

**What two units see of each other.** With the records of §6 and the depth test of §7,

```math
c(u\!\to\!v)=\frac{1}{|S_u|}\,\#\big\{X\in S_u:\ \text{some record of }v\text{ sees }X\big\},\qquad
c_{uv}=\min\big(c(u\!\to\!v),\,c(v\!\to\!u)\big),
```

where $S_u$ is a fixed stride of up to 2000 points of what $u$ saw. It is measured for every two units whose cameras lie
within two cubes of each other, and every two units consecutive along the walk.

**Groups.**

1. **Growth by sight.** From the first unit of the walk that no group holds, a group takes, one at a time, the unit that
   sees the most of what it sees ($c\ge\tau=0.3$ with one of its units), until its own views reach half the cap (32).
   Its own cameras always stay within half a cube (2.5 m).
2. **Small groups.** A group left under 32 own views joins another small group first, else the smallest of the others:
   among the groups it sees the same as before those it sees anything of, else its walk neighbour's. The two must stay
   within the cap of 64 and their cameras within half a cube. A group that cannot join stays as it is, and its pass is
   filled with the shared views around it.

**The walk.** Two units consecutive along the walk always share a cube, so the turn frames between them are views of a
cube that holds both and the cubes form one graph.

**The overlap.** Each cube takes the units of other cubes that see what its own see ($c\ge\tau$ with one of them) and
whose cameras fall within its cube, which is its own cameras' box widened to 5 m. The most co-visible go first, up to the
cap. Neighbouring cubes share the strip between them, and a pass with few own views gets its views there. Each unit's
keyframes take their geometry from its own group's cube.

**A cube's views.** Its Ω views are its keyframes plus the turn frames that lie between two of its keyframes consecutive
in the walk. Its F6 witnesses, the interval frames, follow the same rule.

**What each keyframe sees of every other.** The same measurement with one keyframe on each side, for every two keyframes:

```math
c(a\!\to\!b)=\frac{1}{|S_a|}\,\#\big\{X\in S_a:\ b\text{'s record sees }X\text{ with }|z-d|\le 0.25\,d\big\},
```

with the band of **one surface** (0.25, the band the session's tolerance of §7 is measured in) instead of the tolerance:
two moments of the walk whose geometry disagrees still see the surface each other saw, and those are the ones to tie.
The pairs with $\max\big(c(a\!\to\!b),c(b\!\to\!a)\big)\ge\tau$ are kept in the plan (`kf_covisibility`). After the cut
nothing tied two moments of the walk that see the same surface from different cubes: F4 tracked along the walk only and
F6 voted among walk neighbours only, so two cubes that disagree left the surface twice (observatorio's wooden wall and
signs, 2026-11-07). These pairs are F4's revisit windows (§14.3) and the co-visible witnesses of F6's vote (§14.6). At
the session's tolerance, pccr's map never saw the walk's end meet its start (AMB3R closed no loop there); on one surface
it does.

**Products.** `intake/cubes.json` and `intake/cubes.png`, stamped by the probe, the inputs and the code. The viewer draws
each cube (**Instances** tab, `reconstruction/cube_boxes.py`) as a box around the cameras of all its views, its own and the shared ones (cubes that share views overlap), carried into
the live frame by the similarity between the probe's and the published keyframe cameras.

On the dry run of 2026-11-07, every pass within 5 m:

| Session | Cubes | Views per pass | A unit in, on average |
|---|---|---|---|
| zaragoza (185 keyframes, an open hall) | 6 | 63–64 | 1.98 cubes |
| observatorio (76 keyframes, a sparse walk) | 4 | 29–46 | 1.74 cubes |
| pccr (313 keyframes) | 11 | 39–55 | 1.49 cubes |

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
near band $d^\Omega\le Q_{0.25}$. A cube is brought to the DA3 metre by its ratio $s^{\text{DA3}}_k=\operatorname{med}_f r_f$.
A cube without anchors takes the anchored cubes' median, and says so.

**Seams.** Every two cubes that share keyframes are tied by a **seam**: the Sim(3) between their two copies of the shared
frames. The correspondences are exact, the same pixel of the same shared frame in both cubes, up to 200 000 of them. The
fit is a weighted Umeyama alignment with Cauchy IRLS weights:

```math
w_i=\frac{1}{1+(r_i/c)^2},\qquad c=\max\big(3\cdot1.4826\,\mathrm{MAD}(r),\,10^{-4}\big),\qquad 8\ \text{iterations}.
```

Each seam carries:

- $\sigma_e$, the fit's median residual;
- a centroid $c_e$, the median of the overlap points;
- a lever $L_e$, the median camera-to-point distance.

**The graph** (fork `loop_utils/chunk_graph.py`). Every cube is a piece of **one** Sim(3) graph.

- **Unknowns.** For $k\ge1$, $C_k=(e^{\lambda_k},\exp[\omega_k]_\times,t_k)$. The gauge is $C_0=I$: cube 0 whole, at its
  metric lock.
- **Residual** of an edge $(a,b,M)$, with $E=(C_b^{-1}\circ C_a)\circ M^{-1}$:

  ```math
  \mathbf r_e=\frac{1}{\sigma_e}\Big[\,E(c_e)-c_e\ ;\ \operatorname{Log}(R_E)\,L_e\ ;\ (\ln s_E)\,L_e\,\Big]\in\mathbb R^7 .
  ```

- **Scale prior.** Each cube $k\ge1$ has a DA3 scale prior whose width is **measured on the session's own seams**
  (USER 2026-11-07). A seam's scale $s_e$ is how far two neighbouring cubes' DA3 locks disagree, so each lock's error is

  ```math
  \sigma_{\text{lock}}=\frac{1.4826\,\operatorname{med}_e\lvert\ln s_e\rvert}{\sqrt2},\qquad
  r^{\text{DA3}}_k=\frac{\lambda_k-\ln\big(s^{\text{DA3}}_k/s^{\text{lock}}_k\big)}{\max\big(\sigma_{\text{lock}},\,0.08/\sqrt{n_k}\big)} .
  ```

  The width $0.08/\sqrt{n_k}$ alone took a cube's anchors as independent and claimed 1.1–1.5 %, where pccr's seams
  measured neighbouring locks 0.876–1.167 apart ($\sigma_{\text{lock}}=8.8\,\%$; zaragoza 4.7 %, observatorio 5.4 %).
- **Cost.** The seams and the priors **squared**; a loop closure, when there is one, under Cauchy,

  ```math
  \min_{\{\omega_k,t_k,\lambda_k\}}\ \tfrac12\Big[\sum_{\text{seams, priors}}r_j^2+\sum_{\text{loops}}\ln\big(1+r_j^2\big)\Big],
  ```

  solved by trust-region reflective least squares (tolerances $10^{-12}$), started from the spanning tree of the best seams.
  A seam is an exact fit of the same pixels in two passes, never an outlier to drop. With every residual under one
  Cauchy and the narrow prior, pccr's chain of 11 cubes **broke two seams by 16 %**: 18 and 24 cm between the two
  copies of the same surfaces, while the log read "0.0 cm" because it measured the seam's centroid only. Now no seam
  breaks: pccr's worst seam is 10.2 cm at its surfaces. The log reports the disagreement at the surfaces, the scale
  and the rotation read at the lever arm (`edge_surface_error_m`).

**Not run on cubes, declared.** The chain's SALAD loop bridges and geometric revisit closures are not run. A revisit
inside a cube is one Ω pass; a revisit across cubes is tied after the graph, by F4's revisit windows and F6's vote, both
from AMB3R's map (§10.1). A `loop_closures.txt` or revisit reference left by an earlier run is deleted before Ω runs:
pccr's run of 2026-11-06 read 24 SALAD pairs from a chunk run of 11-04, and it was the only session whose F5 saw a
revisit. Neither are the per-chunk scale ladder and its linear drift run, since a cube has no walk order to drift along.

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

**Scale.** For each frame, $q_p=z^{\text{DA3}}_p/z^\Omega_p$. The frame's ratio $s_f$ is the median of $q$ over the near
band ($z^\Omega\le P_{25}$) intersected with DA3's top-10 % confidence, with fallbacks. The global scale is

```math
s=\operatorname{med}\{s_f:\ s_f\in[P_{10},P_{90}]\}.
```

It multiplies point coordinates, depths and camera centres. On a cube run, $s$ is also how far the locked and joined cubes
sit from DA3's metre. It is applied and recorded, with no bound (pccr: 0.891).

**Orientation.** Gravity is the normalised mean of the cameras' "down" axes, $\hat g=\operatorname{normalize}\big(\tfrac1N\sum_i\operatorname{normalize}(R_ie_2)\big)$.
It is refused when the mean alignment $\tfrac1N\sum|d_i\cdot\hat g|$ is below 0.7. The frame is rotated by Rodrigues'
formula, $R=I+[v]_\times+[v]_\times^2/(1+c)$ with $v=\hat g\times(0,-1,0)$ and $c=\hat g\cdot(0,-1,0)$. The 1st percentile
of the heights is then set to 0.

---

## 14. The precision core, F0 → F6

The precision core (`server/precision/`) measures and corrects the reconstruction against its own evidence.

**Chain.** With `cloud.source: omega_bent`, the runner executes:

```text
F0 camera → F2 gauge → F4 tracks → F3 resolution probe (report) → F5 refinement → F6 bend → chunk check (report)
```

**Epochs.** F2 and F5 publish pose-only geometry epochs. F6 publishes the new cloud epoch.

**Resume.** Every step is resumed against sha256 stamps of its inputs, configuration and code.

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
bootstrap standard deviation, and its position $c_w$ is the mean chainage.

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

**Queries.** A 48-column grid plus up to 512 Shi–Tomasi corners. A query is dropped where VGGT-Ω's depth jumps by more
than 5 % within 2×2.

**Observations.**

- An observation is kept where the visibility is above 0.2; a track needs at least 2 frames.
- Coordinates are mapped to native pixels.
- A track is **held out** when the first 8 bytes of SHA-256(seed, query frame, $u$, $v$), read as an integer, fall below
  $0.2\cdot2^{64}$. This is a deterministic 20 % split, independent of order.

### 14.4 F3 — resolution probe (report only)

The central 32 keyframes are inferred at 512, 768 and 1024. Each pair's projective depth mismatch is
$m_{ij}=\operatorname{med}|z_{i\to j}-z_j|/z_j$. The lowest median is reported; the resolution is the user's setting.

### 14.5 F5 — joint refinement (`precision/refine.py`, `precision/ba_torch.py`)

**Unknowns.** World-to-camera $(R_k,t_k)$, the free part of θ, and the landmarks $X_j$. Each landmark starts as a
multi-view DLT of a fit track, accepted only in front of every camera and with a widest ray angle of at least 2°.

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
- **Stop:** $|\Delta E|/E<10^{-6}$ or a negligible step. Anything but convergence fails the stage.

**Rungs.** The camera is freed in rungs:

- R0: θ fixed.
- R1: $+f_x,f_y,c_x,c_y$.
- R2: $+k_1,k_2,p_1,p_2$.
- R3: focal lengths per 60-keyframe block. This rung is tried only if a permutation test shows block structure, and it
  is reported, never applied.

**The judge.** A rung enters only if the decision rule says it improves the **held-out** tracks:

- $h_t$ is the RMS reprojection, in native px, of held-out track $t$ re-triangulated with the rung's poses and camera;
- clusters are each track's first observing keyframe;
- $\varepsilon$ is the solver error, measured by a continuation solve.

**Witness frames.** They are localised afterwards by LO-RANSAC PnP against the chosen rung's landmarks, with θ fixed.
They need at least 12 correspondences, and their inlier RMS must not exceed the 95th bootstrap percentile of the
keyframes' leave-one-view-out error $P_{95}(\rho_k)$.

### 14.6 F6 — VGGT-Ω's depth bent onto F5's landmarks (`precision/depth_on_f5.py`)

F6 keeps VGGT-Ω's dense depth, which is complete and smooth, and corrects it toward F5's sparse metric truth.

**Rows.** Fit and held-out tracks are re-triangulated with F5's poses and camera. Each observation in keyframe $i$ gives a
row $(u,v,z_{\text{lm}})$.

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

(a masked, normalised Gaussian low-pass). It is accepted **only within 12 px of a depth discontinuity** and where
$|z'-z_b|/z_b\le\tau_0$. Surfaces stay VGGT-Ω's. Mixed pixels at a discontinuity are assigned to the front or back side
when $z_{\text{al}}$ agrees with it, and are otherwise left out of the measurement tier.

**The edge-keeping vote.** Every candidate pixel of keyframe $f$ is lifted to $X=C_f+z\,\mathbf r(u,v)$ and put to its six
neighbours, the turn and interval witnesses, and the **co-visible witnesses** (`covisible_witnesses`): the keyframes of
other moments of the walk, more than 12 keyframes away, that see at least $\tau$ of what $f$ saw in AMB3R's map (§10.1),
the most co-visible first, as many as the six neighbours. Two moments that disagree contradict each other instead of
both standing.

The tolerance $\tau$ is the 75th percentile of the relative disagreement between interior pixels of neighbours at offsets
$\pm3,\pm6,\pm12$, measured on this session (≈ 2 %). For a witness $g$ with depth map $d_g$ and the point's depth $z_g$ in
$g$:

```math
e_g=\frac{z_g-d_g}{d_g},\qquad
\textbf{agree}:\ |e_g|\le\tau,\qquad
\textbf{contradict}:\ e_g<-\tau\ \ \text{or}\ \ z-s_g>\tau\,s_g ,
```

where $s_g$ is $g$'s floor-passing surface splatted into $f$, which $f$ should have seen.

- Each agreeing witness gives the depth along $f$'s ray, $\lambda_g=(d_g-a_g)/b_g$, and $z_{\text{med}}=\operatorname{med}(\{z\}\cup\{\lambda_g\})$.
- The *agreeing surface* is $A=\{g:|s_g-\bar s|\le\tau\bar s\}$, with $\bar s=\operatorname{med}_g s_g$.
- When $|A|\ge2$, its depth is $z_{\text{rep}}=\operatorname{med}_A s_g$.

| Outcome | Condition (P = passes the confidence floor, E = edge pixel) | Depth |
|---|---|---|
| keep | P ∧ contradict ≤ agree | $z_{\text{med}}$ |
| repair | P ∧ contradict > agree ∧ $\lvert A\rvert\ge2$ | $z_{\text{rep}}$ |
| admit | ¬P ∧ E ∧ agree ≥ 1 ∧ contradict ≤ agree | $z_{\text{med}}$ |
| **rescue** | valid ∧ ¬P ∧ $\lvert A\rvert\ge2$ ∧ ¬admit, refused when $z<z_{\text{rep}}(1-\tau)$ | $z_{\text{rep}}$ |
| leave | otherwise | — |

**The support** (USER 2026-11-07: "si pedimos piso de 10 vistas … quiero puntos válidos sin ruido"). A vote counts
only when it **tests the depth**: the witness sees the point from at least $\theta_{\min}=$ `refine.min_tri_deg` (2°,
F5's bound for a landmark) away from $f$'s camera, the angle at the point
$\angle\big(C_f-X,\ C_g-X\big)\ge\theta_{\min}$. A view from almost the same place agrees with any depth along the
ray, and a turn or interval frame next to its keyframe repeats it. The support of a published pixel is the number of
such views that agree with the depth it is published at: the agreeing witnesses for keep and admit, the agreeing
surface $A$ for repair and rescue. **A pixel enters the cloud only with a support of at least**
`cloud.min_support_views` (**10**); below it, it leaves. Measured before the rule on the published clouds: per point a
median of 19 (zaragoza) and 28 (observatorio) of its witnesses could test its depth, and 90 % / 97 % of the points had
at least 10 of them; the viewer's confidence slider at 25 % (≥ 8–10 agreeing votes, every vote counted) was what
removed the duplicates and flyers by eye on pccr, observatorio and zaragoza.

**Rescue.** The rescue recovers low-confidence pixels that the surrounding views agree on, which would otherwise be holes.
It is refused when the pixel's own depth is nearer than the agreed surface: the pixel saw something in front, and the
neighbours' surface may lie behind it.

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
- its support (`confidence`): the views that test its depth and agree, at least 10 (§14.6);
- its source: VGGT-Ω bent, PointDiT detail, band front or band back.

**Cleaner** (`reconstruction/gpu_cloud_clean.py`), on the GPU:

- A 5 mm world-anchored voxel keeps the **real point nearest each cell centre**, never an average.
- **Statistical outlier removal:** $\bar d_i$ is the mean distance to the 8 nearest neighbours. Points with no neighbour
  within 1.5 cm are dropped. The rest are kept when $\bar d_i\le\mu+3\sigma$.

**Publication.** The cloud is published as a new geometry epoch inside a journaled transaction, then converted to a
Potree octree. The LAS quantum is the smallest power of two $s$ with extent$/s<2^{29}$.

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

### 17.1 The vocabulary (`server/segmentation/autoprompt/`)

**Model and determinism.** **Qwen3-VL-8B-Instruct** is served by vLLM on `127.0.0.1:8799` with:

- temperature 0, seed 0;
- eager execution, no prefix caching;
- `VLLM_BATCH_INVARIANT=1`.

Re-asked calls are compared byte for byte.

**The calls.** Each keyframe gets one call. The model returns JSON with:

- the scene type;
- the structural envelope;
- the objects: category, description, and a shape with form, material and detail.

The rules it is asked to follow: whole objects only, one name per object, nothing as background.

**Names to prompts.**

- Names are normalised: lower-cased, articles and counts dropped, singularised.
- Each name is counted by the number of keyframes that named it.
- A merge pass asks short yes/no questions: is it a whole object or a piece? same kind as? a unit of a support surface?
  The links are closed transitively and never form cycles.
- Structural names go first. The list is bounded at `max_sam3_prompts` = 150.
- Every phrase's fate is recorded in `output/segmentation_census.json`.

**Rounds.** After SAM 3 has run, the keyframes are shown again with every segmented pixel painted grey, and the model is
asked to name only what is not grey. New names enter as new prompts. The loop stops when a round names nothing new.

### 17.2 Masks (`server/segmentation/sam3_wrapper.py`)

**Model.** **SAM 3.1 Object Multiplex**, run in bf16 under strict deterministic torch.

**Propagation.** For each text concept and each batch of 250 keyframes (overlap 10), the prompt is added on the first
frame and propagated forward.

**Identity.**

- Two ids are the same object when their mask IoUs on the frames they share pass the decision rule's "above a bar" form,
  with a bar of 0.90.
- Across batches, objects are matched greedily by median IoU over the overlap frames.
- Appearance is never used: tracking is the identity.

**Thresholds** (SAM 3.1): detection 0.3, new detection 0.5, confirmation after 3 consecutive frames.

### 17.3 Masks to cloud (`server/segmentation/pipeline.py`)

**Assignment.** Every point was born at one pixel of one keyframe. Its candidates are the objects whose mask contains that
pixel, and it takes the one with the **smallest mask area**, ties going to the lower id.

**Merges and splits.**

- **Space dedupe.** For instances $i,j$ with 5 cm voxel sets $V_i,V_j$, each keyframe where both have points is a judge:

  ```math
  J_f=\min\Big(\frac{|V_i(f)\cap V_j|}{|V_i(f)|},\ \frac{|V_j(f)\cap V_i|}{|V_j(f)|}\Big),
  ```

  and the pair merges when the "above a bar" rule passes at 0.5.
- **Fragments.** Fragments of a support surface that touch (at least 5 adjacent 10 cm voxel pairs) are united.
- **Co-visible split.** Co-visible components separated by free space are split.
- **Attach.** Unlabelled points within 3 cm take the majority label of their 3 nearest labelled points.
- **Minimum size.** Instances under 1000 points are dropped, except structural ones.

**Oriented boxes.** Built on the instance's core component:

- **yaw candidates** come from sequential RANSAC of vertical planes;
- the **footprint area** $A(\theta)$ is computed in each candidate frame;
- the **error** $\sigma_A$ comes from a Poisson bootstrap of the extreme points;
- the tied set is $\{\theta:\ A(\theta)-A_{\min}\le1.1\,\sigma_A(\theta_{\min})\}$, and the candidate with the largest
  support wins;
- the extents are the core's min and max.

**Products.** `segmentation_result.json`, `instance_ids.npy`, `classification.npy`, the instance store `scene_r.db`, and an
octree rebuilt with the class bytes.

---

## 18. GenRecon scene mesh

A **GenRecon** button in the segmentation panel runs `workers/genrecon_worker.py` → `server/tools/genrecon/run_cloud.sh`
with the confidence threshold chosen for that session. In order:

1. the adapter;
2. the keyframes' vote and the layout;
3. GenRecon's generation;
4. the global remesh and the per-chunk bakes in parallel;
5. the raw mesh, published to `output/genrecon/`;
6. the photo bake, published to `output/tsdf/genrecon_photo/`;
7. the 3D Tiles.

**The rule of this stage: the cloud is the limit.** Presence (where there is surface) comes only from the cloud's points.
The model shapes the surface inside the voxels the cloud occupies and never adds a voxel. The keyframes are its texture.

### 18.1 The points GenRecon receives

**1. The viewer's confidence filter.** This is the user's per-session choice, and depends on how the scene was captured.
The raw confidence is normalised by the octree's min and max, exactly as the viewer's slider does:

```math
\text{keep}\iff\frac{c-c_{\min}}{c_{\max}-c_{\min}}\ge c^* .
```

**2. The keyframes' vote** (`server/tools/genrecon/presence.py`). Every point $i$ was born in one keyframe $b(i)$ at one
pixel, so each keyframe's measured depth can be rebuilt **from the cloud itself**. Per keyframe and per 4×4-pixel block
$\beta$, the depth range $[z^{\min}_{k\beta},z^{\max}_{k\beta}]$ is computed over the points born there.

**The session's own tolerance.** On points taken at a fixed stride, in blocks that hold one surface:

```math
\rho=\frac{(z^{\min}+z^{\max})/2}{z_k(X_i)}-1,\qquad \sigma=\hat\sigma(\rho),\qquad \kappa=3\sigma,\qquad
\tau(z)=\max\big(v,\ \kappa z\big),\quad v=L/32 .
```

**Votes.** Every point is projected, with the OPENCV lens, into every other keyframe $k\ne b(i)$, giving its depth $z$
there:

```math
\textbf{AT}:\ z^{\min}-\tau\le z\le z^{\max}+\tau,\qquad
\textbf{THROUGH}:\ z<z^{\min}-\tau\ \ (\text{only surfaces behind it: it floats in air}),
```

and otherwise no vote (hidden, or nothing measured there). With $S_i=1+\#\mathrm{AT}$ and $T_i=\#\mathrm{THROUGH}$,

```math
\text{the point stays}\iff S_i\ge m\ \wedge\ T_i\le S_i,\qquad m=\texttt{presence\_min\_views}=2 .
```

The votes only remove points. On zaragoza, after a 25 % confidence filter:

- σ = 0.84 % of the depth;
- 404 k floating points (0.6 %) removed;
- 7.6 k single-view points removed.

### 18.2 Cubes over the whole cloud

`plan_layout` reproduces GenRecon's own chunker over the **exact extent** of the points that stayed, with no percentile
and no subsample. Cube side $L=2.85$ m, the size the model was trained on, so the voxel is $v=L/32=8.9$ cm.

**Plan ($x$, $y$).** The plan is padded by $\tfrac{3.5}{16}L$ on each side. With $D$ the padded extent and $\delta=L/16$,
take the fewest cubes $n$ whose stride

```math
s_n=\Big\lceil\frac{D-L}{(n-1)\,\delta}\Big\rceil\,\delta
```

is at most $\tfrac34L$, centred on the midpoint.

**Height ($z$).** Cubes are stacked from the lowest point to the highest with the same stride rule.

**Lattice.** All cubes' $32^3$ grids form **one lattice**, with origin at the lowest corner, because every stride is a
multiple of two voxels. As a result:

- a cell is *active* when a point that stayed lies in it;
- a cube is kept when its block contains an active cell (summed-area table);
- each cube's active voxels are exactly the lattice's (`layout.json`, `active_cells.npy`).

### 18.3 Generation (`vendor/GenRecon`, Trellis.2-class)

**Conditioning.** Each active voxel of each cube is conditioned on the images that see it:

- DINOv3 ViT-L/16 features of the undistorted posed keyframes, at 512 px;
- for each view, the patch feature at the voxel centre's projection, taken from the nearest patch;
- an IBRNet-style aggregation over the views:

  ```math
  F=\mu+\sum_n w_n\,\mathrm{MLP}_f\big([F_n,\mu,\sigma^2]\big),\qquad w=\operatorname{softmax}_n\mathrm{MLP}_w(\cdot).
  ```

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

**Decoding.** The latents are decoded at 512 per cube, $L/512=5.6$ mm: the surface on a flexible dual grid, and a PBR
attribute volume (base colour, metallic, roughness, alpha).

### 18.4 One surface, baked per cube (`inference/chunked_glb.py`)

**One surface.** GenRecon's hole filling (perimeter 1 m) and a narrow-band **dual-contouring remesh** at the decoder's
resolution produce **one** surface for the whole scene:

- every DC vertex is clamped to its own cell;
- vertices are projected back by $v'=v-0.9(v-\mathrm{proj})$.

Each face is then assigned to the cube centre nearest its centroid.

**Face budget.** A GLB counts its length in 32 bits, and the tiler reads it into one 4 GiB buffer. So the faces are capped
by a **total budget shared in proportion**, each cube keeping the same fraction of its own faces:

```math
\alpha=\min\Big(1,\frac{F_{\text{budget}}}{\sum_cF_c}\Big),\qquad F'_c=\max(1000,\lceil\alpha F_c\rceil),\qquad F_{\text{budget}}=64\text{ M}.
```

**Per-cube bake.** In parallel processes (`bake_chunks_parallel.py`), each cube is:

- simplified with quadric error;
- UV-unwrapped;
- textured by sampling the attribute volume trilinearly.

### 18.5 The real texture (`server/tools/genrecon/photo_albedo.py`)

Every texel of GenRecon's own atlas is re-coloured from the keyframes that **see** its surface point $P$.

**Occlusion.** The mesh's own depth $D_k$ is rendered per keyframe with nvdiffrast, in batches of $2^{23}$ triangles, then
min-filtered over 3×3. A self-check against the cloud's z-buffer fixes the image conventions, and the bake fails beyond
25 cm.

**Visibility and weight.** View $k$ sees $P$ when

```math
Z>0.1,\qquad Z\le D_k+0.03+0.01Z,\qquad |\cos\theta|>0.2,
```

and its weight is $w=\cos^4\theta/\max(Z,0.3)^2$. The best 3 views are blended: $c=\sum wc_k/\sum w$.

**Segmentation guard.** A view whose pixel shows **another SAM 3 instance** than the texel's does not colour it. The
texel's instance comes from the cloud's 2 cm cell; the pixel's is the nearest cloud point's label in that keyframe.

**Texels no keyframe sees** take the cloud's own colour (the mean of the 2 cm cell). Only where the cloud has no point
either do they keep the generated colour, tone-matched to the photographed texels per part.

**Atlas size.** Each part $p$ wants texels the size of the finest photo pixel on it. With
$\tau_p=d_p/\bar f_x$ ($d_p$ the nearest camera distance) and $A_p$, $A^{uv}_p$ its 3-D and UV areas:

```math
n_p=\frac{\sqrt{A_p/A^{uv}_p}}{\tau_p},\qquad
s=\min\Big(1,\sqrt{\frac{N_{\text{kf}}WH}{\sum_pn_p^2}}\Big),\qquad
T_p=\operatorname{clip}\big(2^{\lfloor\log_2(n_ps)\rfloor},\,512,\,4096\big).
```

The atlases together never hold more texels than the photos have pixels, because no detail exists beyond them. Only the
base colour changes; the metallic-roughness texture, the factors and the UVs stay as generated.

### 18.6 3D Tiles (`tools/glb/tile_mesh.mjs`)

**Tree.** A k-d tree over the GLB's primitives: a split at the median centroid along the longest axis.

**Levels.** A node of height $h$ simplifies its triangles to $\lfloor F/2^h\rfloor$ with meshoptimizer, borders locked. Its
geometric error is the measured simplification error, or the texel footprint if that is larger.

**Encoding.**

- KTX2 UASTC textures with mipmaps;
- quantised positions, normals and UVs;
- meshopt compression.

The viewer refines to 2 px of screen error.

---

## 19. Determinism, devices and reproducibility

- **One GPU job at a time.** Every GPU stage checks that it has the whole card, and stops the VLM service first.
- **Deterministic torch everywhere**, AMB3R-SLAM included (`reconstruction/amb3r_launch.py`). That means:
  - `use_deterministic_algorithms(True)` and cuDNN deterministic, with no benchmarking and TF32 off;
  - seeded RNGs and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - z-buffers and medians computed by stable sorts and min/max scatters, with no floating-point atomics;
  - the coverage map and the cut counted with integers (cumulative sums over runs), never with float atomics;
  - the bundle adjustment's normal equations accumulated by sorted segment sums.
- **Torch twins.** Each GPU port keeps the numpy function as the path of record. Tests hold the torch twin equal to it:
  exactly for decisions, and to $10^{-9}$ for geometry.
- **Stamps and resume.** Every product is stamped with the sha256 of its inputs, its configuration section and its code.
  A resumed job re-runs exactly the steps whose stamp changed. Each job's configuration is frozen and verified by every
  worker.
- **Epochs, ledger and replay.** Every correction is an epoch, published atomically and replayable from the ledger.
- **Statistics.** Every bootstrap and permutation test is seeded. The bars are the measured noise of the sample, never a
  constant someone chose.

---

## 20. Running it

Hardware: one NVIDIA A100 80 GB (developed on an RTX A6000 48 GB), about 30 CPUs, 117 GB RAM. Everything runs locally.

```bash
git clone --recurse-submodules <this repo> stac-build && cd stac-build
bash scripts/setup_vendors.sh                 # pinned vendor clones (vendor/VENDORS.lock.md)
bash scripts/setup_pod_envs.sh                # conda envs: da3, mapanything, semantic, genrecon, nodejs, …
bash server/tools/genrecon/setup_env.sh       # env genrecon + CUDA extensions + the STAC patch

bash scripts/start.sh                         # backend → https://localhost:8765 (starts the VLM service too)
cd ui && npm run dev:web                      # the UI (Vite) → http://localhost:5173
```

**Workflow.**

1. Create a session, upload a video, and press **Reconstruir**: the reconstruction, the cloud, then VLM + SAM 3 and the
   floor. **Instances → Cubes** shows how the scene was cut.
2. Inspect the cloud. Set the **confidence** slider to the value that leaves the geometry clean.
3. Press **GenRecon** in the segmentation panel. The slider's value is pre-filled.

All configuration lives in `server/config.yaml`. The loaders are typed, and a missing key fails at load.

---

## 21. Repository map

```text
stac-build/
├─ server/
│   ├─ main.py                 # FastAPI app: REST, WebSockets, the run orders (Reconstruir, Autosegment, GenRecon)
│   ├─ pipeline_manager.py     # stages, frozen config per job, resume probes, the queue, the GPU hand-over
│   ├─ workers/                # map (reconstruction), cloudcompy, vlm, sam3, floor, genrecon, certify (off)
│   ├─ intake/                 # I0 quality, focal probe, I1 parallax, I3 walk, turn frames, VRAM sizing
│   ├─ reconstruction/         # amb3r_probe (the whole-video pass), coverage_fill (the keyframes completed on the
│   │                          #   map), cubes (the cut), cube_boxes (the viewer's cubes), chunk_plan (Ω resolution),
│   │                          #   scale_align, orient, sky precompute, gpu_cloud_clean, potree_converter
│   ├─ precision/              # F0 camera, F2 gauge, F4 tracks, F3 probe, F5 refine + ba_torch, F6 depth_on_f5 +
│   │                          #   mono_detail + f6_torch, corrected cloud, chunk check, epochs
│   ├─ segmentation/           # autoprompt (VLM), SAM 3 wrapper, mask → cloud, instances, OBBs, store
│   ├─ semantic/               # vLLM launcher and client
│   ├─ tools/genrecon/         # adapter, presence (the vote + layout), run_cloud.sh, bake_chunks_parallel,
│   │                          #   photo_albedo, publish, measure_mesh, genrecon_stac.patch
│   └─ config.yaml
├─ vendor/
│   ├─ VGGT-Long/              # the fork (branch gpu-main): cube_run (Ω per cube, seams, ownership), context/witness
│   │                          #   views, metric lock, the Sim(3) graph, SALAD, the decision rule
│   ├─ AMB3R-SLAM/             # the fork (branch stac): the whole-video pass and its light map
│   ├─ depth-anything-3/  pointdit/  sam31/  GenRecon/  …
├─ ui/                         # React + TypeScript + Vite; the viewer, panels, GenRecon and Autosegment windows
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
