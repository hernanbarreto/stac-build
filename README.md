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

STAC Build turns a handheld video into a **metric 3-D reconstruction of a construction site** and the objects in it. A
feed-forward multi-view network (VGGT-Ω) reconstructs the walk in overlapping chunks planned by co-visibility. The chunks
are made metric by a multi-view monocular-metric depth model (Depth Anything 3) and joined into **one chunk-level Sim(3)
graph**. That graph carries every seam, every retrieved and every geometrically detected loop closure, and a scale prior
per chunk. A **precision core** then measures and corrects, in order:

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
6. [I3: multi-view metric depth windows and the walk](#6-i3-multi-view-metric-depth-windows-and-the-walk)
7. [Turn frames and interval witnesses](#7-turn-frames-and-interval-witnesses)
8. [Chunks planned by co-visibility, and the Ω resolution](#8-chunks-planned-by-co-visibility-and-the-ω-resolution)
9. [The VGGT-Ω pass](#9-the-vggt-ω-pass)
10. [Metric lock of each chunk](#10-metric-lock-of-each-chunk)
11. [Loop closures: retrieval, revisits, bridges](#11-loop-closures-retrieval-revisits-bridges)
12. [One chunk-level Sim(3) graph](#12-one-chunk-level-sim3-graph)
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
| 1 | **Reconstruction** | `workers/map_worker.py` | intake → I3 → chunk plan → VGGT-Ω pass with loop closures and the chunk Sim(3) graph → metric scale and upright frame → precision core F0–F6 → the published cloud (`output/cleaned_cloud.ply`) and its octree |
| 2 | **Cloud → viewer** | `workers/cloudcompy_worker.py` | the cleaned cloud handed to the viewer (Potree octree) |
| 3 | **Scene analysis** | `workers/vlm_worker.py` | Qwen3-VL reads every keyframe; the SAM 3 vocabulary (`output/vlm_analysis.json`) |
| 4 | **Segmentation** | `workers/sam3_worker.py` | SAM 3.1 masks on every keyframe, projected onto the cloud as instances with oriented boxes (`output/segmentation_result.json`, `instance_ids.npy`, `scene_r.db`) |

Two more stages exist but are switched off in the chain:

- **GenRecon** (`workers/genrecon_worker.py`) runs from its own **GenRecon** button in the segmentation panel. The user
  chooses the cloud's confidence threshold for that session (`pipeline.auto_genrecon: false`).
- **Certification** stays in the code, switched off (`certify.auto_after_segmentation: false`).

The **Autosegment** button re-runs the VLM and SAM 3 stages on the cloud already on disk, with editable prompts.

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
  deterministic tools (the VLM proposes; tools measure).

**Storage.** Each scan lives in `server/projects/<session>/scans/<scan>/src_default/`:

- `frames/`: the video frames;
- `intake/`: quality, parallax, walk, plan and their stamps;
- `output/`: model passes, epochs, cloud, octree, segmentation, meshes.

---

## 3. Models

| Role | Model | Where |
|---|---|---|
| Multi-view feed-forward reconstruction (poses, intrinsics, depth, confidence) | **VGGT-Ω** 1B, `vggt_omega_1b_512.pt`, bf16 | `vendor/VGGT-Long` (fork, branch `gpu-main`) |
| Metric multi-view depth: focal probe, I3 windows, scale anchors | **Depth Anything 3**, `DA3NESTED-GIANT-LARGE-1.1` | `vendor/depth-anything-3` |
| Place recognition for loop candidates | **DINOv2 ViT-B/14 + SALAD** (8448-D descriptor) | fork `LoopModels/` |
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

## 6. I3: multi-view metric depth windows and the walk

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

## 7. Turn frames and interval witnesses

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

- **Context frames** are inferred *with* their chunk by VGGT-Ω. They give the network the turn it would otherwise see as a
  jump, and their slots are dropped before saving. They also vote in F6.
- **Witness frames** are inferred in a **separate pass after each chunk**. They never change the chunks or the poses; they
  only testify in F6's vote.

---

## 8. Chunks planned by co-visibility, and the Ω resolution

### 8.1 Co-visibility (`server/reconstruction/chunk_covis.py`)

**Tolerance.** On the I3 depth maps,
$\mathrm{tol}=Q_{0.95}\{|z_a-z_b|/\tfrac12(z_a+z_b)\}$ over about 4000 pixels per frame of the frames shared by
consecutive windows.

**Directed co-visibility.** The share of $i$'s sampled points $P_i$ that $j$ sees at a consistent depth is

```math
c(i\!\to\!j)=\frac{1}{|P_i|}\,\#\big\{X\in P_i:\ \pi_j(X)\in\Omega_j,\ Z_j>0,\ |z_j-Z_j[\pi_j(X)]|\le \mathrm{tol}\,Z_j\big\},
\qquad c_{ij}=\min\big(c(i\!\to\!j),c(j\!\to\!i)\big).
```

**Co-visibility length.** $\ell(a)$ is found by a majority scan from $a$: count the passes ($c_{ab}\ge0.3$) and the
fails, and stop when fails outnumber passes. Then:

- each keyframe costs $\delta=1/\max(\ell,1)$;
- the distance is $D(a,b)=\sum_{[a,b)}\delta$, measured in "co-visibility lengths".

### 8.2 The plan

**One pass** when $D_{\text{tot}}\le H=13.67$ and the views (keyframes, rotation keyframes and their turn frames) number at
most 64.

**Otherwise blocks** $B_0,\dots,B_{m-1}$, with chunk $k=B_k\cup B_{k+1}$ (50 % overlap). The constraints:

- each block holds at least 12 keyframes;
- a block is within budget when $D\le H/2$ and it has at most 32 views. This is `MAX_CHUNK_VIEWS` = 64: half new, half
  shared, as in VGGT-SLAM.

The blocks minimise the cost $1+W_1[\text{over}]+W_2[\text{over}\wedge L>12]$, with $W_1=n+1$ and $W_2=(n+1)^2$, by an
exact dynamic program. Ties are broken lexicographically:

1. minimise the worst seam depth;
2. minimise the worst rotation across a cut, $\theta_i=2\arcsin(\|R_{i+1}-R_i\|_F/2\sqrt2)$, so that cuts avoid turns;
3. minimise the per-block budget;
4. take the leftmost boundary.

$H$ was calibrated on measured sessions. The plan is frozen per session.

### 8.3 Ω resolution

The resolution is the largest multiple of 16 not above $\min(16\lceil\max(W,H)/16\rceil,\ 1024)$ at which the largest
chunk fits the card:

```math
\mathrm{cap}(R)=\Big\lfloor\frac{T(1-m)-4.0}{\kappa\cdot0.086\cdot g_wg_h/(832\cdot464)}\Big\rfloor,\qquad
R^*=\max\{R:\ N^*\le\mathrm{cap}(R)\}
```

- $N^*$ is the largest chunk's view count, context or witness views included.
- $\kappa$ is the committed footprint factor of the card table.

An out-of-memory fails the run. It is never silently retried lower.

---

## 9. The VGGT-Ω pass

For a chunk $[s,e)$ with context frames $\mathcal C$, the network sees the views $[s,e)\cup\mathcal C$, sorted, the first
being a keyframe. It returns per view:

- a pose encoding, which gives world-to-camera and $K$;
- a depth $d$;
- a depth confidence $c$.

The world points are $X=R_c\,[(u-c_x)d/f_x,\ (v-c_y)d/f_y,\ d]^\top+t_c$.

**Sky masks.** The sky segmenter's score $o$ is min–max normalised per image, $\hat o=\lfloor255(o-\min o)/(\max o-\min o)\rfloor$,
and the mask is $m=\mathbf 1[\hat o<32]$ (non-sky). The confidence becomes $c\leftarrow c\,m$. The masks of every
keyframe, context and witness frame are precomputed in 12 processes while I3 runs on the card.

**Context records.** Each context view's depth, confidence, $K$ and relative pose to its preceding keyframe are saved,
then every per-view array is cut to the keyframe slots. The **witness pass** infers keyframes ∪ interval frames separately
and saves only the witness slots.

**Confidence floor.** Per chunk,

```math
\theta=\max\big(P_{20}(c_{\text{valid}}),\ c_{\min}+0.10\,(c_{\max}-c_{\min})\big),
```

and a pixel stays when $c\ge\theta$ and $c>10^{-5}$. The same 0.10 min–max fraction is the viewer slider's arithmetic.
It also gates the samplers of every pose fit.

---

## 10. Metric lock of each chunk

**Anchor ratio.** For each frame, $r_f=\operatorname{med}(d^{\text{DA3}}/d^{\Omega})$ over pixels valid in both, in the
near band $d^\Omega\le Q_{0.25}$. The chunk's ratio is $s^{\text{DA3}}_k=\operatorname{med}_f r_f$.

**Seam ratio.** $r_k=1/\operatorname{med}\big(d^{(k+1)}/d^{(k)}\big)$ over each frame shared by chunks $k$ and $k+1$, then
the median over the shared frames.

**Scale graph.** With $x_k=\ln s_k$,

```math
\min_x\ \sum_k\frac{(x_{k+1}-x_k-\ln r_k)^2}{\sigma_s^2}+\sum_k\frac{(x_k-\ln s^{\text{DA3}}_k)^2}{(\sigma_a/\sqrt{n_k})^2}
```

with $\sigma_s=0.003$, $\sigma_a=0.08$ and $n_k$ anchors per chunk, solved by linear least squares. A zoom chunk, whose
focal differs significantly, loses its anchors.

**Scale drift.** A linear drift of scale inside a chunk, $\ln s(u)=(1-u)x_{k0}+ux_{k1}$, is applied only if the decision
rule approves it on held-out seam frames.

---

## 11. Loop closures: retrieval, revisits, bridges

**Retrieval.** DINOv2 + SALAD descriptors $\hat D$ give the similarity $S=\hat D\hat D^\top$. The candidates are, per row,
the top 5 non-local neighbours (gap at least the larger of 11 keyframes, 1 m of walk, and 20 keyframes) with
$S>\tau_S$, followed by NMS over ±3 keyframes.

The threshold $\tau_S$ is calibrated by Youden's $J=\mathrm{TPR}-\mathrm{FPR}$ against the revisit reference of §6:

- positives are pairs closer than the distance bar and facing within the angle bar;
- pairs within 1.1× either bar's error are undecided;
- the fallback is 0.65.

**Spatial gate.** Only one rule rejects: two frames less than 1 m of walk apart (`visit_drift.min_walk_m`) are one visit.
A narrow corridor (lateral extent under 3 m) with fewer than 4 reciprocal frames is marked ambiguous, and its σ is tripled.

**Geometric revisits** find what retrieval misses. Using the provisional chain, $F_{ij}$ is the share of $i$'s sampled
points inside $j$'s frustum, widened by the drift a walk of length $L$ allows:

- distance $\delta(L)=\max(0.30,0.013L)$ m;
- angle $\theta(L)=\max(2°,0.10L)$.

Pairs from chunks at least 2 apart, at least 1 m of walk apart, with $\min(F_{ij},F_{ji})\ge0.30$, are proposed, keeping
the best 3 per chunk pair.

**Bridges.** Each candidate is measured by one extra VGGT-Ω pass over its two windows: 20 keyframes each plus 4 sharp
in-between frames. The pass is locked to its own anchors, and the exact correspondences are the same keyframe at the same
pixel on both sides.

**Robust Sim(3) fit.** The fit is a weighted Umeyama alignment with Cauchy IRLS weights:

```math
w_i=\frac{1}{1+(r_i/c)^2},\qquad c=\max\big(3\cdot1.4826\,\mathrm{MAD}(r),\,10^{-4}\big),\qquad 8\ \text{iterations}.
```

The bridge composes as $s_{ab}=s_b/s_a$, $R_{ab}=R_bR_a^\top$, $t_{ab}=t_b-s_{ab}R_{ab}t_a$. Its $\sigma$ is the median
split-half held-out residual of the worse side. It carries no scale term: the scale belongs to the graph.

---

## 12. One chunk-level Sim(3) graph

All chunks are pieces of **one** Sim(3) graph (fork `loop_utils/chunk_graph.py`).

**Unknowns.** For $k\ge1$, $C_k=(e^{\lambda_k},\exp[\omega_k]_\times,t_k)$. The gauge is $C_0=I$.

**Seam edges.** A Sim(3) is fitted between chunks $k+1$ and $k$ on all valid shared pixels. A scale outside $[0.6,1.6]$
falls back to a rigid fit. Each edge carries:

- $\sigma_e$, the fit's median residual;
- a centroid $c_e$, the median of the overlap points;
- a lever $L_e$, the median camera-to-point distance.

**Closure edges.** These are the bridges of §11, retrieved and geometric. Every verified closure enters the graph.

**Residual** of an edge $(a,b,M)$, with $E=(C_b^{-1}\circ C_a)\circ M^{-1}$:

```math
\mathbf r_e=\frac{1}{\sigma_e}\Big[\,E(c_e)-c_e\ ;\ \operatorname{Log}(R_E)\,L_e\ ;\ (\ln s_E)\,L_e\,\Big]\in\mathbb R^7 .
```

**Scale prior.** Each chunk $k\ge1$ has a DA3 scale prior:

```math
r^{\text{DA3}}_k=\frac{\lambda_k-\ln\big(s^{\text{DA3}}_k/s^{\text{lock}}_k\big)}{0.08/\sqrt{n_k}} .
```

**Cost.** Cauchy on every scalar component,

```math
\min_{\{\omega_k,t_k,\lambda_k\}}\ \tfrac12\sum_j\ln\big(1+r_j^2\big),
```

solved by trust-region reflective least squares (tolerances $10^{-12}$). The start is the seams-only chain
$C_{k+1}=C_k\circ M_{k+1\to k}$.

**Judge (measured, never deciding).** Each closure $q$ is left out in turn. Its error before is at the seams + priors
solve, and its error after is at the solve without it. The decision rule is evaluated and recorded in
`output/chunk_graph.json`; the closures are applied regardless, as in a SLAM back end.

**Application.**

- Points: $X\leftarrow s_kR_kX+t_k$.
- Depths: $d\leftarrow s_kd$.
- Poses: $[R_kR_c\,|\,s_kR_kc+t_k]$.
- A shared block of frames is split at its midpoint between its two chunks.

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

It multiplies point coordinates, depths and camera centres. On a chunked session the residual $|s-1|$ must stay below
0.10, or the run fails.

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

**Session K.** The session $K$ is the component-wise median of the per-frame VGGT-Ω intrinsics. The focal spread
$100\cdot1.4826\operatorname{med}|f_i-\hat f|/|\hat f|$ is reported, with a warning above 2 %.

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
- Loop windows around every SALAD pair.
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
when its confidence is at least $\min+0.10(\max-\min)$ of its VGGT-Ω chunk.

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
neighbours plus the turn and interval witnesses.

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

**Rescue.** The rescue recovers low-confidence pixels that the surrounding views agree on, which would otherwise be holes.
It is refused when the pixel's own depth is nearer than the agreed surface: the pixel saw something in front, and the
neighbours' surface may lie behind it.

**Turn and interval witnesses.** These are the context and witness records of §9. Each is carried into the anchor
keyframe's metric frame:

- its depth is scaled by $\phi=\operatorname{med}z_{\Omega,a}/\operatorname{med}z_a$ and bent by the anchor's $k_a$;
- its pose is $c2w^{F5}_a[R_{\text{rel}}\,|\,\phi\,c_{0,a}t_{\text{rel}}]$.

They vote on the keyframes within ±12 of their anchor and never become points.

---

## 15. The published cloud

**Points.** Every keyframe pixel that the vote kept becomes a point:

```math
\mathbf X=c2w^{F5}_f\big(\hat z(u,v)\,K_{F5}^{-1}[u,v,1]^\top\big),
```

coloured from the undistorted frame. Each point records:

- its birth keyframe (`frame_global`);
- its pixel (`pixel_row`, `pixel_col`);
- its vote count (`confidence`);
- its source: VGGT-Ω bent, PointDiT detail, band front or band back.

**Cleaner** (`reconstruction/gpu_cloud_clean.py`), on the GPU:

- A 5 mm world-anchored voxel keeps the **real point nearest each cell centre**, never an average.
- **Statistical outlier removal:** $\bar d_i$ is the mean distance to the 8 nearest neighbours. Points with no neighbour
  within 1.5 cm are dropped. The rest are kept when $\bar d_i\le\mu+3\sigma$.

**Publication.** The cloud is published as a new geometry epoch inside a journaled transaction, then converted to a
Potree octree. The LAS quantum is the smallest power of two $s$ with extent$/s<2^{29}$.

---

## 16. Chunk check (report)

`precision/chunk_check.py` checks, per VGGT-Ω chunk and per keyframe, the floor band, the ceiling band and the camera's
height over the floor. DA3's own camera-to-floor distance serves as an independent confirmation.

**Floor plane.** A seeded RANSAC (400 iterations, 2 cm) gives the floor plane, its normal oriented upward.

**Bands.** Heights are $h=(X-c)\cdot n$. A band is the median of the points within 8 cm of the chunk's floor (or of the
5th percentile), and likewise for the ceiling (97th percentile). Each band needs at least 200 points, and its resolution
is $\hat\sigma$.

**Departures.** Departures between chunks are judged by the decision rule, with a two-sample bootstrap and
$\varepsilon$ = the band's resolution.

**Verdicts.**

| Verdict | When |
|---|---|
| ok | the floor does not depart |
| level | the floor departs and the ceiling does not: a real level change |
| depth | the floor and the ceiling depart in opposite directions |
| pose | the floor and the ceiling depart together |
| intra | the floor trend departs within the chunk |
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
- **Deterministic torch everywhere.** That means:
  - `use_deterministic_algorithms(True)` and cuDNN deterministic, with no benchmarking and TF32 off;
  - seeded RNGs and `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
  - z-buffers and medians computed by stable sorts and min/max scatters, with no floating-point atomics;
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

1. Create a session, upload a video, and press **Reconstruir**: the reconstruction, the cloud, then VLM + SAM 3.
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
│   ├─ workers/                # map (reconstruction), cloudcompy, vlm, sam3, genrecon, certify (off)
│   ├─ intake/                 # I0 quality, focal probe, I1 parallax, I3 walk, turn frames, VRAM sizing
│   ├─ reconstruction/         # chunk_covis (plan), chunk_plan (Ω resolution), scale_align, orient, sky precompute,
│   │                          #   loops (spatial gate), gpu_cloud_clean, potree_converter, repro
│   ├─ precision/              # F0 camera, F2 gauge, F4 tracks, F3 probe, F5 refine + ba_torch, F6 depth_on_f5 +
│   │                          #   mono_detail + f6_torch, corrected cloud, chunk check, epochs
│   ├─ segmentation/           # autoprompt (VLM), SAM 3 wrapper, mask → cloud, instances, OBBs, store
│   ├─ semantic/               # vLLM launcher and client
│   ├─ tools/genrecon/         # adapter, presence (the vote + layout), run_cloud.sh, bake_chunks_parallel,
│   │                          #   photo_albedo, publish, measure_mesh, genrecon_stac.patch
│   └─ config.yaml
├─ vendor/
│   ├─ VGGT-Long/              # the fork (branch gpu-main): VGGT-Ω chunks, context/witness views, SALAD, revisits,
│   │                          #   bridges, metric lock, the chunk Sim(3) graph, the decision rule
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
