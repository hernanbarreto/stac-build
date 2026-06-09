# Pipeline Performance Optimization

Plan to speed up the reconstruction pipeline (DA3 → MapAnything/VGGT-Long → loop
closure → cloud merge → TSDF). Ordered by expected return, honest about what is
**confirmed by architecture** vs **needs benchmarking**.

> ⚠️ **Read this first.** The numbers below are *architectural estimates*, not
> measurements on our hardware/models. Per-model speedups are **not** pipeline
> speedups — see Amdahl's law below. **Step 0 (profile) gates everything else.**

---

## Guiding principles

1. **Only neural forward passes are ONNX/TensorRT-able.** The classical parts of
   the pipeline are not: loop closure (Sim3 optimization), SALAD/DBoW retrieval
   logic, CloudComPy filters, point-cloud merge, mesh post-processing. ONNX
   accelerates *inference*, not the whole pipeline.
2. **Amdahl's law caps the total gain.** If inference is 60 % of wall-clock and
   we make it 3× faster: `total = 60/3 + 40 = 60 %` → only **1.65× overall**, not
   3×. Accelerating a stage that isn't the bottleneck is wasted effort.
3. **Baseline matters.** We already run **bf16** (`use_amp: true`). Headline
   "9×/14×" numbers are usually vs FP32-CPU or unoptimized. The marginal gain over
   our bf16 baseline is smaller.
4. **For max GPU speed: ONNX → TensorRT (FP16/INT8)**, not vanilla ONNXRuntime.
5. **INT8 quantization on large transformers is risky** (accuracy drop) — needs
   PTQ calibration or QAT, validated against a metric, not assumed.

---

## Stage-by-stage map

| Stage | Today | Opportunity | Confidence |
|---|---|---|---|
| **DA3** (depth) | GPU bf16 | ONNX → **TensorRT** (FP16/INT8) | high ✅ |
| **MapAnything** (per-chunk 3D) | GPU bf16 | `torch.compile` / quantization — **not ONNX** | medium |
| **Loop closure** | DINOv2 (GPU) + Sim3 (classical) | DINOv2 → TensorRT; Sim3 already small | medium |
| **CloudComPy** (SOR + voxel, ~68 M pts) | **CPU** ❌ | **move to Open3D tensor (CUDA)** | **high — big win** |
| **TSDF integrate** | GPU (VBG CUDA) ✅ | already on GPU | — |
| **TSDF extract** | **CPU** (forced) | blocked by Open3D VBG bug (GPU extract aborts) | — |
| **TSDF rasterization** (`_rasterize_cloud_depth`) | **CPU** (numpy `minimum.at`) | **move z-buffer to torch/cupy (GPU)** | high |
| **TSDF texture bake** | GPU (nvdiffrast) ✅ | already on GPU | — |
| **Potree conversion** | CPU | I/O-bound, little headroom | low |

---

## Per-model: ONNX / quantization feasibility

| Model | ONNX export | Notes |
|---|---|---|
| **DA3** (Depth Anything 3) | ✅ mature | ONNX + TensorRT repos exist; input `[1,3,280,504]` matches our res. Verify our **GIANT** variant (`DA3NESTED-GIANT-LARGE-1.1`) exports — published repos use LARGE/METRIC. |
| **SAM3** | ✅ mature | Exports to 3 ONNX models (image enc + text enc + decoder). TensorRT migrations exist. |
| **DINOv2** | ✅ mature | Standard ViT, widely exported. Used in **both** VGGT patchify and **SALAD loop retrieval** → helps inference *and* loop closure. |
| **MapAnything** (facebook, our per-chunk model) | ⚠️ hard | Variable view count + cross-view attention → dynamic axes; **optional multi-modal inputs** (image + optional DA3 depth/K/poses) → conditional branches ONNX can't express. No mature export. Path = bf16 + `torch.compile` + quantization. |
| **VGGT** (framework lineage) | ⚠️ hard | Same class of difficulty; see "Quantized VGGT" (arXiv 2509.21302). |

### Realistic per-model gains (vs our bf16 baseline)

| Optimization | Realistic gain |
|---|---|
| ONNXRuntime (CUDA) vs PyTorch eager | ~1.2–1.8× |
| **ONNX → TensorRT FP16** | ~2–4× |
| + INT8 (with accuracy validation) | ~1.5–3× extra, **risk of degradation** |

DA3 / SAM3 (clean ViTs): 2–4× feasible. MapAnything: 1.5–2× without serious
quantization work.

---

## Optimization steps (priority order)

### Step 0 — Profile a full run (DO THIS FIRST)
Instrument each stage with wall-clock timing (DA3, MapAnything inference, loop
closure, cloud merge/CloudComPy, TSDF integrate/extract/raster/texture, Potree).
Output seconds-per-stage for one representative scan. **Everything below is
prioritized off this — without it we are guessing.**

### Step 1 — CloudComPy SOR + voxel → Open3D tensor (CUDA)
Replace the CPU CloudComPy SOR + voxel-downsample on ~68 M points with Open3D's
GPU tensor ops (`voxel_down_sample`, `remove_statistical_outlier` on `cuda:0`).
- Expected: large reduction in non-neural time.
- Risk: low–medium. Validate point counts / cloud quality match CloudComPy.
- Note: the noise filter is already off (O(n²)).

### Step 2 — DA3 → ONNX → TensorRT (FP16)
Export DA3 to ONNX, build a TensorRT FP16 engine, swap inference behind a flag.
- Expected: 2–4× on the DA3 stage.
- Risk: medium. Verify the GIANT variant exports; engines are **GPU-specific**.

### Step 3 — TSDF rasterization → GPU
Port `_rasterize_cloud_depth` (numpy `np.minimum.at` z-buffer) to torch/cupy.
- Expected: removes a CPU-bound step in the TSDF stage.
- Risk: low.

### Step 4 — SAM3 → ONNX → TensorRT
Export the 3 SAM3 sub-models; TensorRT FP16 engines.
- Expected: 2–4× on segmentation inference.
- Risk: medium (multi-model wiring).

### Step 5 — MapAnything: `torch.compile` + (later) quantization
No ONNX. Try `torch.compile`, ensure FlashAttention/bf16, fixed `chunk_size=120`
(a fixed view count removes one dynamic-axis obstacle). Quantization later, with
accuracy validation.
- Expected: 1.5–2×. Risk: medium–high (accuracy).

### Step 6 — DINOv2 (loop closure / SALAD) → TensorRT
Helps loop-closure retrieval throughput.
- Risk: low–medium.

---

## Constraints / caveats

- **TensorRT engines are GPU-architecture-specific** — rebuild per GPU.
- **ONNX export can break** on dynamic shapes / custom CUDA ops — each model must
  be verified to export cleanly before assuming a win.
- **TSDF GPU extract is blocked** by an Open3D VBG bug (native abort on large
  grids); extract stays on CPU until upstream-fixed or replaced.
- **Validate quality, not just speed** — every quantized/exported model must be
  checked against the FP32/bf16 output (depth error, mask IoU, mesh fidelity),
  not assumed equivalent.

---

## References

- Depth Anything 3 ONNX / TensorRT: `devin-lai/Depth-Anything-3-Onnx`,
  `yuvraj108c/ComfyUI-Depth-Anything-Tensorrt`
- SAM3 ONNX: `vietanhdev/segment-anything-3-onnx-models`, `samexporter`
- MapAnything: `facebookresearch/map-anything`
- VGGT / Quantized VGGT: arXiv 2503.11651, arXiv 2509.21302
- Open3D tensor point cloud (CUDA): `open3d.t.geometry.PointCloud`
